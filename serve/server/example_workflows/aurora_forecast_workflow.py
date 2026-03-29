#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Aurora Model Forecast Workflow"""

import json
import logging
from typing import Any

import zarr
from pydantic import Field

from earth2studio.serve.server.workflow import (
    Workflow,
    WorkflowParameters,
    WorkflowProgress,
    workflow_registry,
)

logger = logging.getLogger(__name__)


class AuroraForecastParameters(WorkflowParameters):
    """Parameters for Aurora forecast workflow."""

    forecast_times: list[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="List of forecast initialization times (ISO format)",
    )
    num_steps: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of forecast steps",
    )


@workflow_registry.register
class AuroraForecastWorkflow(Workflow):
    """Aurora multi-scale weather forecast workflow."""

    name = "aurora_forecast_workflow"
    description = "Aurora multi-scale weather forecast"
    Parameters = AuroraForecastParameters

    def __init__(self) -> None:
        super().__init__()
        try:
            from earth2studio.models.px import Aurora

            self.package = Aurora.load_default_package()
            self.model = Aurora.load_model(self.package)
        except Exception:
            logger.warning("Aurora model not available, using placeholder")
            self.model = None

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | AuroraForecastParameters
    ) -> AuroraForecastParameters:
        try:
            return AuroraForecastParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | AuroraForecastParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the Aurora forecast workflow."""

        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            self.update_execution_data(execution_id, {"metadata": metadata})

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=5,
            ))

            from datetime import datetime as _dt
            from earth2studio import run as e2run
            from earth2studio.data import GFS
            from earth2studio.io import ZarrBackend

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Loading Aurora model...",
                current_step=2,
                total_steps=5,
            ))

            if self.model is None:
                raise RuntimeError(
                    "Aurora model not available. Ensure earth2studio[aurora] is installed"
                )

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Setting up GFS data source...",
                current_step=3,
                total_steps=5,
            ))

            data = GFS()
            output_dir = self.get_output_path(execution_id)
            io = ZarrBackend(
                file_name=str(output_dir / "results.zarr"),
                chunks={"time": 1, "lead_time": 1},
                backend_kwargs={"overwrite": True},
            )

            self.update_execution_data(execution_id, WorkflowProgress(
                progress=f"Running Aurora forecast ({parameters.num_steps} steps)...",
                current_step=4,
                total_steps=5,
            ))

            start_times = [_dt.fromisoformat(t) for t in parameters.forecast_times]
            io = e2run.deterministic(
                start_times,
                parameters.num_steps,
                self.model,
                data,
                io,
            )

            zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            forecast_info = {
                "forecast_times": parameters.forecast_times,
                "num_steps": parameters.num_steps,
                "model": "Aurora",
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
                    "results_summary": f"Generated Aurora forecast with {parameters.num_steps} lead times",
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
