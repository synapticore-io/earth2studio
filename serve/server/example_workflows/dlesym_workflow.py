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
DLESyM Workflow

This workflow implements the DLESyM (Deep Learning Earth System Model) for
coupled atmosphere-ocean forecasting.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DLESyMWorkflowParameters(WorkflowParameters):
    """Parameters for DLESyM workflow"""

    forecast_times: list[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="List of forecast initialization times (ISO format)",
    )
    nsteps: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Number of forecast steps (6-hour intervals)",
    )
    data_source: str = Field(
        default="era5",
        description="Data source for initialization",
    )
    output_format: str = Field(
        default="zarr",
        description="Output format",
    )
    create_plots: bool = Field(
        default=False,
        description="Whether to create visualization plots",
    )


@workflow_registry.register
class DLESyMWorkflow(Workflow):
    """
    DLESyM (Deep Learning Earth System Model) workflow.

    This workflow:
    1. Loads the coupled DLESyM model (atmosphere + ocean)
    2. Initializes from reanalysis data
    3. Runs coupled forecast
    4. Saves results in Zarr format
    """

    name = "dlesym_workflow"
    description = "DLESyM coupled atmosphere-ocean forecast workflow"
    Parameters = DLESyMWorkflowParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | DLESyMWorkflowParameters
    ) -> DLESyMWorkflowParameters:
        """Validate and convert input parameters"""
        try:
            return DLESyMWorkflowParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | DLESyMWorkflowParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run DLESyM workflow"""

        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            self.update_execution_data(execution_id, {"metadata": metadata})

            progress = WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=5,
            )
            self.update_execution_data(execution_id, progress)

            from earth2studio import run
            from earth2studio.data import ERA5
            from earth2studio.io import ZarrBackend
            from earth2studio.models.px import DLESyM

            progress = WorkflowProgress(
                progress="Loading DLESyM model...",
                current_step=2,
                total_steps=5,
            )
            self.update_execution_data(execution_id, progress)

            package = DLESyM.load_default_package()
            model = DLESyM.load_model(package)

            progress = WorkflowProgress(
                progress=f"Setting up {parameters.data_source} data source...",
                current_step=3,
                total_steps=5,
            )
            self.update_execution_data(execution_id, progress)

            if parameters.data_source.lower() == "era5":
                data = ERA5()
            else:
                raise ValueError(f"Unsupported data source: {parameters.data_source}")

            output_dir = self.get_output_path(execution_id)
            if parameters.output_format.lower() == "zarr":
                io = ZarrBackend(file_name=str(output_dir / "results.zarr"))
            else:
                raise ValueError(f"Unsupported output format: {parameters.output_format}")

            progress = WorkflowProgress(
                progress=f"Running DLESyM forecast ({parameters.nsteps} steps)...",
                current_step=4,
                total_steps=5,
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
                "model_type": "dlesym",
                "data_source": parameters.data_source,
                "output_format": parameters.output_format,
            }

            metadata_path = output_dir / "forecast_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(forecast_info, f, indent=2)

            progress = WorkflowProgress(
                progress="Complete!", current_step=5, total_steps=5
            )
            self.update_execution_data(execution_id, progress)

            self.update_execution_data(
                execution_id,
                {
                    "metadata": {
                        **metadata,
                        "results_summary": f"Generated {parameters.nsteps}-step coupled forecast",
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
