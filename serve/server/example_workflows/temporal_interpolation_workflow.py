#!/usr/bin/env python3
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
Temporal Interpolation Workflow

This workflow implements temporal interpolation using SFNO and ModAFNO models
to fill gaps between forecast timesteps.
"""

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalInterpolationWorkflowParameters(WorkflowParameters):
    """Parameters for temporal interpolation workflow"""

    forecast_times: list[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="List of forecast initialization times (ISO format)",
    )
    nsteps: int = Field(
        default=6,
        ge=1,
        le=100,
        description="Number of forecast steps",
    )
    model_type: Literal["sfno", "interp_modafno"] = Field(
        default="sfno",
        description="Prognostic model for forecasting (sfno, interp_modafno)",
    )
    interpolation_steps: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of intermediate interpolation steps between forecast steps",
    )
    data_source: Literal["gfs"] = Field(
        default="gfs",
        description="Data source for initialization",
    )
    output_format: Literal["zarr"] = Field(
        default="zarr",
        description="Output format",
    )
    create_plots: bool = Field(
        default=False,
        description="Whether to create visualization plots",
    )
    plot_variable: Literal["t2m", "msl", "u10m", "v10m", "tcwv", "z500"] = Field(
        default="t2m",
        description="Variable to plot",
    )


@workflow_registry.register
class TemporalInterpolationWorkflow(Workflow):
    """
    Temporal Interpolation workflow using SFNO or ModAFNO.

    This workflow:
    1. Loads a prognostic model for coarse temporal forecasting
    2. Generates intermediate timesteps via interpolation
    3. Saves results with enhanced temporal resolution
    """

    name = "temporal_interpolation_workflow"
    description = (
        "Temporal interpolation workflow using SFNO/ModAFNO with enhanced resolution"
    )
    Parameters = TemporalInterpolationWorkflowParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | TemporalInterpolationWorkflowParameters
    ) -> TemporalInterpolationWorkflowParameters:
        """Validate and convert input parameters"""
        try:
            return TemporalInterpolationWorkflowParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | TemporalInterpolationWorkflowParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run temporal interpolation workflow"""

        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            self.update_execution_data(execution_id, {"metadata": metadata})

            progress = WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            from earth2studio import run
            from earth2studio.data import GFS
            from earth2studio.io import ZarrBackend
            from earth2studio.models.px import InterpModAFNO, SFNO

            progress = WorkflowProgress(
                progress=f"Loading {parameters.model_type} model...",
                current_step=2,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            if parameters.model_type.lower() == "sfno":
                package = SFNO.load_default_package()
                model = SFNO.load_model(package)
            elif parameters.model_type.lower() == "interp_modafno":
                package = InterpModAFNO.load_default_package()
                model = InterpModAFNO.load_model(package)
            else:
                raise ValueError(f"Unsupported model type: {parameters.model_type}")

            progress = WorkflowProgress(
                progress=f"Setting up {parameters.data_source} data source...",
                current_step=3,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            if parameters.data_source.lower() == "gfs":
                data = GFS()
            else:
                raise ValueError(f"Unsupported data source: {parameters.data_source}")

            output_dir = self.get_output_path(execution_id)
            if parameters.output_format.lower() == "zarr":
                io = ZarrBackend(file_name=str(output_dir / "results.zarr"))
            else:
                raise ValueError(f"Unsupported output format: {parameters.output_format}")

            progress = WorkflowProgress(
                progress=f"Running forecast ({parameters.nsteps} steps)...",
                current_step=4,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            io = run.deterministic(
                parameters.forecast_times, parameters.nsteps, model, data, io
            )

            if parameters.output_format.lower() == "zarr":
                zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            forecast_info = {
                "forecast_times": parameters.forecast_times,
                "nsteps": parameters.nsteps,
                "model_type": parameters.model_type,
                "interpolation_steps": parameters.interpolation_steps,
                "data_source": parameters.data_source,
                "output_format": parameters.output_format,
            }

            metadata_path = output_dir / "forecast_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(forecast_info, f, indent=2)

            if parameters.create_plots:
                progress = WorkflowProgress(
                    progress="Creating visualization plots...",
                    current_step=5,
                    total_steps=6,
                )
                self.update_execution_data(execution_id, progress)
                self.create_forecast_plot(io, parameters, execution_id)

            progress = WorkflowProgress(
                progress="Complete!", current_step=6, total_steps=6
            )
            self.update_execution_data(execution_id, progress)

            self.update_execution_data(
                execution_id,
                {
                    "metadata": {
                        **metadata,
                        "results_summary": f"Generated {parameters.nsteps}-step forecast with {parameters.interpolation_steps} interpolation steps",
                        "forecast_info": forecast_info,
                    }
                },
            )

            return {
                "status": "success",
                "output_path": str(output_dir),
                "forecast_info": forecast_info,
            }

        except Exception as e:
            progress = WorkflowProgress(progress="Failed!", error_message=str(e))
            self.update_execution_data(execution_id, progress)
            raise

    def create_forecast_plot(
        self, io: Any, parameters: TemporalInterpolationWorkflowParameters, execution_id: str
    ) -> None:
        """Create forecast visualization plot"""
        try:
            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt

            forecast_time = parameters.forecast_times[0]
            variable = parameters.plot_variable
            step = 0

            plt.close("all")
            projection = ccrs.Robinson()
            _, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(12, 8))

            lon = io["lon"][:]
            lat = io["lat"][:]

            if variable in io:
                data = io[variable][0, step]
            else:
                available_vars = [
                    key for key in io if key not in ["lon", "lat", "time"]
                ]
                if available_vars:
                    variable = available_vars[0]
                    data = io[variable][0, step]
                else:
                    raise ValueError("No data variables found in forecast output")

            im = ax.pcolormesh(
                lon,
                lat,
                data,
                transform=ccrs.PlateCarree(),
                cmap="Spectral_r",
            )

            cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.8)
            cbar.set_label(f"{variable}")

            ax.set_title(f"{forecast_time} - {variable}", fontsize=14)
            ax.coastlines()
            ax.gridlines(alpha=0.5)

            output_dir = self.get_output_path(execution_id)
            plot_path = output_dir / f"forecast_plot_{variable}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception:
            logger.exception("Could not create forecast plot")
            raise
