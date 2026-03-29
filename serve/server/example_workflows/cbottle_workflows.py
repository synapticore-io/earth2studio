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
CBottle Workflows

This module implements two CBottle workflows:
1. CBottleGenerationWorkflow - Generates high-resolution climate fields from coarse inputs
2. CBottleSuperResolutionWorkflow - Super-resolves climate fields to higher resolution
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


class CBottleGenerationWorkflowParameters(WorkflowParameters):
    """Parameters for CBottle generation workflow"""

    forecast_times: list[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="List of forecast initialization times (ISO format)",
    )
    nsteps: int = Field(
        default=1,
        ge=1,
        le=12,
        description="Number of forecast steps",
    )
    input_model: Literal["fcn", "dlwp", "sfno"] = Field(
        default="fcn",
        description="Coarse prognostic model for initial field generation",
    )
    data_source: Literal["gfs", "era5"] = Field(
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


class CBottleSuperResolutionWorkflowParameters(WorkflowParameters):
    """Parameters for CBottle super-resolution workflow"""

    forecast_times: list[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="List of forecast initialization times (ISO format)",
    )
    nsteps: int = Field(
        default=1,
        ge=1,
        le=12,
        description="Number of forecast steps",
    )
    input_model: Literal["fcn", "dlwp", "sfno"] = Field(
        default="fcn",
        description="Coarse prognostic model for initial field",
    )
    data_source: Literal["gfs", "era5"] = Field(
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


@workflow_registry.register
class CBottleGenerationWorkflow(Workflow):
    """
    CBottle Generation workflow.

    This workflow:
    1. Runs a coarse prognostic model (FCN, DLWP, SFNO)
    2. Applies CBottle infill diagnostic to generate high-resolution climate fields
    3. Saves results in Zarr format on HEALPix grid
    """

    name = "cbottle_generation_workflow"
    description = "CBottle high-resolution climate field generation workflow"
    Parameters = CBottleGenerationWorkflowParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | CBottleGenerationWorkflowParameters
    ) -> CBottleGenerationWorkflowParameters:
        """Validate and convert input parameters"""
        try:
            return CBottleGenerationWorkflowParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | CBottleGenerationWorkflowParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run CBottle generation workflow"""

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
            from earth2studio.data import ERA5, GFS
            from earth2studio.io import ZarrBackend
            from earth2studio.models.dx import CBottleInfill
            from earth2studio.models.px import DLWP, FCN, SFNO

            progress = WorkflowProgress(
                progress=f"Loading {parameters.input_model} model...",
                current_step=2,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            if parameters.input_model.lower() == "fcn":
                package = FCN.load_default_package()
                prognostic_model = FCN.load_model(package)
            elif parameters.input_model.lower() == "dlwp":
                package = DLWP.load_default_package()
                prognostic_model = DLWP.load_model(package)
            elif parameters.input_model.lower() == "sfno":
                package = SFNO.load_default_package()
                prognostic_model = SFNO.load_model(package)
            else:
                raise ValueError(f"Unsupported input model: {parameters.input_model}")

            progress = WorkflowProgress(
                progress="Loading CBottle generation model...",
                current_step=3,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            package = CBottleInfill.load_default_package()
            diagnostic_model = CBottleInfill.load_model(package)

            progress = WorkflowProgress(
                progress=f"Setting up {parameters.data_source} data source...",
                current_step=4,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            if parameters.data_source.lower() == "gfs":
                data = GFS()
            elif parameters.data_source.lower() == "era5":
                data = ERA5()
            else:
                raise ValueError(f"Unsupported data source: {parameters.data_source}")

            output_dir = self.get_output_path(execution_id)
            if parameters.output_format.lower() == "zarr":
                io = ZarrBackend(file_name=str(output_dir / "results.zarr"))
            else:
                raise ValueError(f"Unsupported output format: {parameters.output_format}")

            progress = WorkflowProgress(
                progress=f"Generating high-resolution fields ({parameters.nsteps} steps)...",
                current_step=5,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            io = run.diagnostic(
                parameters.forecast_times,
                parameters.nsteps,
                prognostic_model,
                diagnostic_model,
                data,
                io,
            )

            if parameters.output_format.lower() == "zarr":
                zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            forecast_info = {
                "forecast_times": parameters.forecast_times,
                "nsteps": parameters.nsteps,
                "input_model": parameters.input_model,
                "diagnostic_model": "cbottle_infill",
                "data_source": parameters.data_source,
                "output_format": parameters.output_format,
            }

            metadata_path = output_dir / "forecast_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(forecast_info, f, indent=2)

            progress = WorkflowProgress(
                progress="Complete!", current_step=6, total_steps=6
            )
            self.update_execution_data(execution_id, progress)

            self.update_execution_data(
                execution_id,
                {
                    "metadata": {
                        **metadata,
                        "results_summary": f"Generated high-resolution climate fields for {parameters.nsteps} timestep(s)",
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


@workflow_registry.register
class CBottleSuperResolutionWorkflow(Workflow):
    """
    CBottle Super-Resolution workflow.

    This workflow:
    1. Runs a coarse prognostic model (FCN, DLWP, SFNO)
    2. Applies CBottle super-resolution diagnostic to upscale climate fields
    3. Saves results in Zarr format with enhanced spatial resolution
    """

    name = "cbottle_super_resolution_workflow"
    description = "CBottle super-resolution workflow for enhanced spatial resolution"
    Parameters = CBottleSuperResolutionWorkflowParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | CBottleSuperResolutionWorkflowParameters
    ) -> CBottleSuperResolutionWorkflowParameters:
        """Validate and convert input parameters"""
        try:
            return CBottleSuperResolutionWorkflowParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | CBottleSuperResolutionWorkflowParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run CBottle super-resolution workflow"""

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
            from earth2studio.data import ERA5, GFS
            from earth2studio.io import ZarrBackend
            from earth2studio.models.dx import CBottleSuperResolution
            from earth2studio.models.px import DLWP, FCN, SFNO

            progress = WorkflowProgress(
                progress=f"Loading {parameters.input_model} model...",
                current_step=2,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            if parameters.input_model.lower() == "fcn":
                package = FCN.load_default_package()
                prognostic_model = FCN.load_model(package)
            elif parameters.input_model.lower() == "dlwp":
                package = DLWP.load_default_package()
                prognostic_model = DLWP.load_model(package)
            elif parameters.input_model.lower() == "sfno":
                package = SFNO.load_default_package()
                prognostic_model = SFNO.load_model(package)
            else:
                raise ValueError(f"Unsupported input model: {parameters.input_model}")

            progress = WorkflowProgress(
                progress="Loading CBottle super-resolution model...",
                current_step=3,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            package = CBottleSuperResolution.load_default_package()
            diagnostic_model = CBottleSuperResolution.load_model(package)

            progress = WorkflowProgress(
                progress=f"Setting up {parameters.data_source} data source...",
                current_step=4,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            if parameters.data_source.lower() == "gfs":
                data = GFS()
            elif parameters.data_source.lower() == "era5":
                data = ERA5()
            else:
                raise ValueError(f"Unsupported data source: {parameters.data_source}")

            output_dir = self.get_output_path(execution_id)
            if parameters.output_format.lower() == "zarr":
                io = ZarrBackend(file_name=str(output_dir / "results.zarr"))
            else:
                raise ValueError(f"Unsupported output format: {parameters.output_format}")

            progress = WorkflowProgress(
                progress=f"Running super-resolution ({parameters.nsteps} steps)...",
                current_step=5,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            io = run.diagnostic(
                parameters.forecast_times,
                parameters.nsteps,
                prognostic_model,
                diagnostic_model,
                data,
                io,
            )

            if parameters.output_format.lower() == "zarr":
                zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            forecast_info = {
                "forecast_times": parameters.forecast_times,
                "nsteps": parameters.nsteps,
                "input_model": parameters.input_model,
                "diagnostic_model": "cbottle_super_resolution",
                "data_source": parameters.data_source,
                "output_format": parameters.output_format,
            }

            metadata_path = output_dir / "forecast_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(forecast_info, f, indent=2)

            progress = WorkflowProgress(
                progress="Complete!", current_step=6, total_steps=6
            )
            self.update_execution_data(execution_id, progress)

            self.update_execution_data(
                execution_id,
                {
                    "metadata": {
                        **metadata,
                        "results_summary": f"Generated super-resolved fields for {parameters.nsteps} timestep(s)",
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
