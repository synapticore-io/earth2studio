# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0

"""
CAMS Analysis & Forecast Workflows

Serve ECMWF/CAMS atmospheric composition data via the Earth2Studio REST API.
Uses run.deterministic with a Persistence model (identity) so the standard
IO pipeline (zarr, progress, results) works identically to built-in workflows.
"""

import json
import logging
from collections import OrderedDict
from typing import Any, Literal

import numpy as np
import zarr
from pydantic import Field

from earth2studio.serve.server.workflow import (
    Workflow,
    WorkflowParameters,
    WorkflowProgress,
    workflow_registry,
)

logger = logging.getLogger(__name__)

EU_SURFACE = ["dust", "pm2p5", "pm10", "so2sfc", "no2sfc", "o3sfc"]
EU_MULTI_LEVEL = [
    "dust", "dust_500m", "dust_1000m", "dust_3000m", "dust_5000m",
    "pm2p5", "pm2p5_500m", "pm2p5_1000m", "pm2p5_3000m", "pm2p5_5000m",
]
GLOBAL_AOD = ["aod550", "duaod550", "omaod550", "bcaod550", "ssaod550", "suaod550"]

PRESETS = {
    "eu_surface": EU_SURFACE,
    "eu_multi_level": EU_MULTI_LEVEL,
    "global_aod": GLOBAL_AOD,
}


def _resolve_variables(
    variables: list[str] | None, preset: str
) -> list[str]:
    if variables:
        return variables
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}', available: {list(PRESETS)}")
    return PRESETS[preset]


# ---------------------------------------------------------------------------
# CAMS Analysis
# ---------------------------------------------------------------------------

class CAMSAnalysisParameters(WorkflowParameters):
    start_time: list[str] = Field(
        default=["2025-06-01T00:00:00"],
        description="Analysis time(s) in ISO 8601 format",
    )
    preset: Literal["eu_surface", "eu_multi_level"] = Field(
        default="eu_surface",
        description="Variable preset (eu_surface or eu_multi_level)",
    )
    variables: list[str] | None = Field(
        default=None,
        description="Explicit variable list (overrides preset if given)",
    )


@workflow_registry.register
class CAMSAnalysisWorkflow(Workflow):
    """CAMS European air quality analysis at 0.1 deg resolution."""

    name = "cams_analysis"
    description = "CAMS EU analysis (dust, PM2.5, SO2, NO2, O3, CO) at 0.1 deg"
    Parameters = CAMSAnalysisParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | CAMSAnalysisParameters
    ) -> CAMSAnalysisParameters:
        return CAMSAnalysisParameters.validate(parameters)

    def run(
        self,
        parameters: dict[str, Any] | CAMSAnalysisParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        parameters = self.validate_parameters(parameters)

        self.update_execution_data(execution_id, WorkflowProgress(
            progress="Importing CAMS components...", current_step=1, total_steps=4,
        ))

        from earth2studio import run as e2run
        from earth2studio.data import CAMS
        from earth2studio.io import ZarrBackend
        from earth2studio.models.px import Persistence

        resolved = _resolve_variables(parameters.variables, parameters.preset)

        self.update_execution_data(execution_id, WorkflowProgress(
            progress="Fetching CAMS grid metadata...", current_step=2, total_steps=4,
        ))

        data = CAMS(cache=True, verbose=True)
        sample = data(parameters.start_time[:1], resolved[:1])
        domain = OrderedDict({
            "lat": sample.coords["lat"].values,
            "lon": sample.coords["lon"].values,
        })

        self.update_execution_data(execution_id, WorkflowProgress(
            progress=f"Running CAMS analysis ({len(resolved)} vars)...",
            current_step=3, total_steps=4,
        ))

        output_dir = self.get_output_path(execution_id)
        io = ZarrBackend(
            file_name=str(output_dir / "results.zarr"),
            chunks={"time": 1, "lead_time": 1},
            backend_kwargs={"overwrite": True},
        )

        model = Persistence(resolved, domain, dt=np.timedelta64(1, "h"))
        e2run.deterministic(parameters.start_time, 0, model, data, io)

        zarr.consolidate_metadata(str(output_dir / "results.zarr"))

        metadata = {
            "start_time": parameters.start_time,
            "preset": parameters.preset,
            "variables": resolved,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.update_execution_data(execution_id, WorkflowProgress(
            progress="Complete!", current_step=4, total_steps=4,
        ))
        self.update_execution_data(execution_id, {
            "metadata": metadata,
        })

        return {"status": "success", "output_path": str(output_dir), "metadata": metadata}


# ---------------------------------------------------------------------------
# CAMS Forecast
# ---------------------------------------------------------------------------

class CAMSForecastParameters(WorkflowParameters):
    start_time: list[str] = Field(
        default=["2025-06-01T00:00:00"],
        description="Forecast initialization time(s) in ISO 8601 format",
    )
    lead_hours: list[int] = Field(
        default=[0, 6, 12, 24, 48],
        description="Lead times in hours",
    )
    preset: Literal["eu_surface", "eu_multi_level", "global_aod"] = Field(
        default="eu_surface",
        description="Variable preset",
    )
    variables: list[str] | None = Field(
        default=None,
        description="Explicit variable list (overrides preset if given)",
    )


@workflow_registry.register
class CAMSForecastWorkflow(Workflow):
    """CAMS forecast — EU surface + global column/AOD."""

    name = "cams_forecast"
    description = "CAMS forecast (EU surface + global AOD/column) via CDS API"
    Parameters = CAMSForecastParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | CAMSForecastParameters
    ) -> CAMSForecastParameters:
        return CAMSForecastParameters.validate(parameters)

    def run(
        self,
        parameters: dict[str, Any] | CAMSForecastParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        parameters = self.validate_parameters(parameters)

        self.update_execution_data(execution_id, WorkflowProgress(
            progress="Importing CAMS forecast components...", current_step=1, total_steps=4,
        ))

        from datetime import timedelta

        from earth2studio import run as e2run
        from earth2studio.data import CAMS_FX
        from earth2studio.io import ZarrBackend
        from earth2studio.models.px import Persistence

        resolved = _resolve_variables(parameters.variables, parameters.preset)

        self.update_execution_data(execution_id, WorkflowProgress(
            progress="Fetching CAMS forecast grid metadata...", current_step=2, total_steps=4,
        ))

        data = CAMS_FX(cache=True, verbose=True)
        sample = data(parameters.start_time[:1], [timedelta(hours=0)], resolved[:1])
        domain = OrderedDict({
            "lat": sample.coords["lat"].values,
            "lon": sample.coords["lon"].values,
        })

        self.update_execution_data(execution_id, WorkflowProgress(
            progress=f"Running CAMS forecast ({len(resolved)} vars, {len(parameters.lead_hours)} lead times)...",
            current_step=3, total_steps=4,
        ))

        output_dir = self.get_output_path(execution_id)
        io = ZarrBackend(
            file_name=str(output_dir / "results.zarr"),
            chunks={"time": 1, "lead_time": 1},
            backend_kwargs={"overwrite": True},
        )

        dt_h = parameters.lead_hours[0] if parameters.lead_hours else 1
        model = Persistence(resolved, domain, dt=np.timedelta64(dt_h, "h"))
        e2run.deterministic(parameters.start_time, 0, model, data, io)

        zarr.consolidate_metadata(str(output_dir / "results.zarr"))

        metadata = {
            "start_time": parameters.start_time,
            "lead_hours": parameters.lead_hours,
            "preset": parameters.preset,
            "variables": resolved,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.update_execution_data(execution_id, WorkflowProgress(
            progress="Complete!", current_step=4, total_steps=4,
        ))
        self.update_execution_data(execution_id, {
            "metadata": metadata,
        })

        return {"status": "success", "output_path": str(output_dir), "metadata": metadata}
