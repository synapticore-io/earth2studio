#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Seasonal Statistics Workflow"""

import json
import logging
from typing import Any, Literal

import zarr
from pydantic import Field

from earth2studio.serve.server.workflow import (
    Workflow,
    WorkflowParameters,
    WorkflowProgress,
    workflow_registry,
)

logger = logging.getLogger(__name__)


class SeasonalStatisticsParameters(WorkflowParameters):
    """Parameters for seasonal statistics workflow."""

    forecast_times: list[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="List of forecast initialization times (ISO format)",
    )
    nsteps: int = Field(
        default=90,
        ge=1,
        le=360,
        description="Number of forecast steps for seasonal predictions",
    )
    nensemble: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of ensemble members",
    )
    model_type: Literal["pangu"] = Field(
        default="pangu",
        description="Prognostic model for seasonal forecasting",
    )
    statistic: Literal["mean", "std", "min", "max"] = Field(
        default="mean",
        description="Statistic to compute across ensemble",
    )


@workflow_registry.register
class SeasonalStatisticsWorkflow(Workflow):
    """Seasonal statistics workflow using Pangu-Weather."""

    name = "seasonal_statistics_workflow"
    description = "Seasonal statistics forecast with Pangu model"
    Parameters = SeasonalStatisticsParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | SeasonalStatisticsParameters
    ) -> SeasonalStatisticsParameters:
        try:
            return SeasonalStatisticsParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | SeasonalStatisticsParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the seasonal statistics workflow."""

        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            self.update_execution_data(execution_id, {"metadata": metadata})

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=5,
            ))

            from earth2studio import run as e2run
            from earth2studio.data import GFS
            from earth2studio.io import ZarrBackend
            from earth2studio.models.px import Pangu
            from earth2studio.perturbation import SphericalGaussian

            self.update_execution_data(execution_id, WorkflowProgress(
                progress=f"Loading {parameters.model_type} model...",
                current_step=2,
                total_steps=5,
            ))

            package = Pangu.load_default_package()
            model = Pangu.load_model(package)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Setting up ensemble perturbation...",
                current_step=3,
                total_steps=5,
            ))

            sg = SphericalGaussian(noise_amplitude=0.1)
            data = GFS()

            output_dir = self.get_output_path(execution_id)
            io = ZarrBackend(
                file_name=str(output_dir / "results.zarr"),
                chunks={"ensemble": 1, "time": 1},
                backend_kwargs={"overwrite": True},
            )

            self.update_execution_data(execution_id, WorkflowProgress(
                progress=f"Running seasonal forecast ({parameters.nensemble} members, {parameters.nsteps} steps)...",
                current_step=4,
                total_steps=5,
            ))

            io = e2run.ensemble(
                parameters.forecast_times,
                parameters.nsteps,
                parameters.nensemble,
                model,
                data,
                io,
                sg,
                batch_size=2,
            )

            zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            forecast_info = {
                "forecast_times": parameters.forecast_times,
                "nsteps": parameters.nsteps,
                "nensemble": parameters.nensemble,
                "model": parameters.model_type,
                "statistic": parameters.statistic,
            }

            metadata_path = output_dir / "forecast_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(forecast_info, f, indent=2)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Complete!",
                current_step=5,
                total_steps=5,
            ))

            self.update_execution_data(execution_id, {
                "metadata": {
                    **metadata,
                    "results_summary": f"Generated {parameters.nensemble}-member seasonal forecast with {parameters.statistic} statistic",
                    "forecast_info": forecast_info,
                }
            })

            return {
                "status": "success",
                "output_path": str(output_dir),
                "forecast_info": forecast_info,
            }

        except Exception as e:
            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Failed!",
                error_message=str(e),
            ))
            raise e
