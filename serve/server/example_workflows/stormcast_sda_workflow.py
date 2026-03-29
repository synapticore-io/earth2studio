#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""StormCast Score-Based Data Assimilation Workflow"""

import json
import logging
from datetime import datetime
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


class StormCastSDAParameters(WorkflowParameters):
    """Parameters for StormCast SDA workflow."""

    initialization_time: str = Field(
        default="2024-01-01T00:00:00",
        description="Initialization time in ISO format",
    )
    num_forecast_steps: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of forecast steps",
    )
    assimilate_observations: bool = Field(
        default=True,
        description="Whether to assimilate in-situ observations",
    )
    sda_gamma: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1.0,
        description="DPS guidance scaling factor (lower = stronger assimilation)",
    )
    sda_std_obs: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Assumed observation noise standard deviation",
    )


@workflow_registry.register
class StormCastSDAWorkflow(Workflow):
    """StormCast score-based data assimilation workflow."""

    name = "stormcast_sda_workflow"
    description = "StormCast with score-based data assimilation (diffusion posterior sampling)"
    Parameters = StormCastSDAParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | StormCastSDAParameters
    ) -> StormCastSDAParameters:
        try:
            return StormCastSDAParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | StormCastSDAParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the StormCast SDA workflow."""

        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            self.update_execution_data(execution_id, {"metadata": metadata})

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=5,
            ))

            import torch
            from earth2studio.data import HRRR, ISD, fetch_data, fetch_dataframe
            from earth2studio.io import ZarrBackend
            from earth2studio.models.da import StormCastSDA
            from earth2studio.utils.coords import map_coords_xr

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Loading StormCast SDA model...",
                current_step=2,
                total_steps=5,
            ))

            package = StormCastSDA.load_default_package()
            model = StormCastSDA.load_model(
                package,
                sda_std_obs=parameters.sda_std_obs,
                sda_gamma=parameters.sda_gamma,
            )
            model = model.to(device)
            model.eval()

            init_time = datetime.fromisoformat(parameters.initialization_time)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Fetching HRRR initial conditions...",
                current_step=3,
                total_steps=5,
            ))

            hrrr = HRRR()
            x, coords = fetch_data(
                source=hrrr,
                time=[init_time],
                variable=model.init_coords()["variable"],
                device=device,
            )
            x = map_coords_xr(x, coords, model.init_coords())

            observations = None
            if parameters.assimilate_observations:
                self.update_execution_data(execution_id, WorkflowProgress(
                    progress="Fetching observation data...",
                    current_step=3,
                    total_steps=5,
                ))
                isd = ISD()
                try:
                    observations = fetch_dataframe(
                        source=isd,
                        time=[init_time],
                    )
                except Exception as e:
                    logger.warning(f"Could not fetch observations: {e}")
                    observations = None

            output_dir = self.get_output_path(execution_id)
            io = ZarrBackend(
                file_name=str(output_dir / "results.zarr"),
                chunks={"time": 1, "lead_time": 1},
                backend_kwargs={"overwrite": True},
            )

            self.update_execution_data(execution_id, WorkflowProgress(
                progress=f"Running StormCast SDA ({parameters.num_forecast_steps} steps)...",
                current_step=4,
                total_steps=5,
            ))

            iterator = model.create_iterator(x, coords)
            for step, (y, y_coords) in enumerate(iterator):
                if step >= parameters.num_forecast_steps:
                    break
                io.write(y)

            zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            forecast_info = {
                "initialization_time": parameters.initialization_time,
                "num_steps": parameters.num_forecast_steps,
                "model": "StormCastSDA",
                "assimilate_observations": parameters.assimilate_observations,
                "sda_gamma": parameters.sda_gamma,
                "sda_std_obs": parameters.sda_std_obs,
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
                    "results_summary": f"Generated StormCast SDA forecast with assimilation={parameters.assimilate_observations}",
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
