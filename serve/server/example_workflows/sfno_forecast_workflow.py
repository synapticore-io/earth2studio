# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SFNO Forecast Workflow

5-10 day deterministic forecast using SFNO (Spherical Fourier Neural Operator).
Outputs temperature, precipitation, and wind components at 6-hour intervals.
"""

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


class SFNOForecastParameters(WorkflowParameters):
    """Parameters for SFNO forecast workflow."""

    forecast_times: list[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="List of forecast initialization times (ISO format)",
    )
    num_steps: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Number of forecast steps (each step is 6 hours)",
    )


@workflow_registry.register
class SFNOForecastWorkflow(Workflow):
    """
    SFNO-based deterministic forecast for 5-10 day predictions.

    Generates 6-hourly forecasts of:
    - t2m: 2-meter temperature
    - precip: accumulated precipitation
    - u10m, v10m: 10-meter wind components
    """

    name = "sfno_forecast_workflow"
    description = "5-10 day SFNO deterministic forecast (t2m, precip, u10m, v10m)"
    Parameters = SFNOForecastParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | SFNOForecastParameters
    ) -> SFNOForecastParameters:
        """Validate and convert input parameters."""
        try:
            return SFNOForecastParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | SFNOForecastParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the SFNO forecast workflow."""

        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=5,
            ))

            from earth2studio import run as e2run
            from earth2studio.data import GFS
            from earth2studio.io import ZarrBackend
            from earth2studio.models.px import SFNO

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Loading SFNO model...",
                current_step=2,
                total_steps=5,
            ))

            package = SFNO.load_default_package()
            model = SFNO.load_model(package)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Setting up GFS data source...",
                current_step=3,
                total_steps=5,
            ))

            data = GFS()

            output_dir = self.get_output_path(execution_id)
            io = ZarrBackend(file_name=str(output_dir / "results.zarr"))

            self.update_execution_data(execution_id, WorkflowProgress(
                progress=f"Running SFNO forecast ({parameters.num_steps} steps)...",
                current_step=4,
                total_steps=5,
            ))

            io_result = e2run.deterministic(  # type: ignore[assignment]
                parameters.forecast_times,
                parameters.num_steps,
                model,
                data,
                io,
            )
            io = io_result  # type: ignore[assignment]

            zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            forecast_info = {
                "forecast_times": parameters.forecast_times,
                "num_steps": parameters.num_steps,
                "model": "SFNO",
                "data_source": "GFS",
                "variables": ["t2m", "precip", "u10m", "v10m"],
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
                    "results_summary": f"Generated {parameters.num_steps}-step SFNO forecast for {len(parameters.forecast_times)} time(s)",
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
