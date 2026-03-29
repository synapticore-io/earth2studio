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
Tropical Cyclone Tracking Workflow

Couples SFNO prognostic model with TCTrackerWuDuan diagnostic to track cyclone paths.
Outputs tracking diagnostics (lat, lon, msl, w10m) for identified cyclone centers.
"""

import json
import logging
from typing import Any, Literal

import numpy as np
import torch
import zarr
from pydantic import Field

from earth2studio.serve.server.workflow import (
    Workflow,
    WorkflowParameters,
    WorkflowProgress,
    workflow_registry,
)

logger = logging.getLogger(__name__)


class CycloneTrackingParameters(WorkflowParameters):
    """Parameters for cyclone tracking workflow."""

    start_time: list[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="Forecast initialization time(s) in ISO format",
    )
    num_steps: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Number of forecast steps (each step is 6 hours)",
    )
    region: Literal["global", "atlantic", "pacific", "indian"] = Field(
        default="global",
        description="Geographic region to focus tracking (filtering applied in post-processing)",
    )


@workflow_registry.register
class CycloneTrackingWorkflow(Workflow):
    """
    Tropical cyclone tracking using SFNO + TCTrackerWuDuan.

    Combines:
    - SFNO prognostic model for 6-hourly atmospheric predictions
    - TCTrackerWuDuan diagnostic for identifying and tracking cyclone centers

    Outputs tracking diagnostics:
    - tc_lat, tc_lon: estimated cyclone center coordinates
    - tc_msl: mean sea level pressure at center
    - tc_w10m: 10-meter wind magnitude
    """

    name = "cyclone_tracking_workflow"
    description = "Tropical cyclone tracking with SFNO + TCTrackerWuDuan"
    Parameters = CycloneTrackingParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | CycloneTrackingParameters
    ) -> CycloneTrackingParameters:
        """Validate and convert input parameters."""
        try:
            return CycloneTrackingParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | CycloneTrackingParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the cyclone tracking workflow."""

        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=6,
            ))

            from earth2studio.data import ARCO, fetch_data
            from earth2studio.io import ZarrBackend
            from earth2studio.models.dx import TCTrackerWuDuan
            from earth2studio.models.px import SFNO
            from earth2studio.utils.coords import map_coords
            from earth2studio.utils.time import to_time_array

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Loading SFNO prognostic model...",
                current_step=2,
                total_steps=6,
            ))

            package = SFNO.load_default_package()
            prognostic = SFNO.load_model(package).to(device)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Initializing TC tracker diagnostic...",
                current_step=3,
                total_steps=6,
            ))

            tracker = TCTrackerWuDuan().to(device)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Setting up data source...",
                current_step=4,
                total_steps=6,
            ))

            data = ARCO()

            output_dir = self.get_output_path(execution_id)
            io = ZarrBackend(
                file_name=str(output_dir / "results.zarr"),
                chunks={"time": 1, "lead_time": 1},
                backend_kwargs={"overwrite": True},
            )

            self.update_execution_data(execution_id, WorkflowProgress(
                progress=f"Running SFNO + tracking ({parameters.num_steps} steps)...",
                current_step=5,
                total_steps=6,
            ))

            x, coords = fetch_data(
                source=data,
                time=to_time_array(parameters.start_time),
                variable=prognostic.input_coords()["variable"],
                lead_time=prognostic.input_coords()["lead_time"],
                device=device,
            )

            tracker.reset_path_buffer()
            model = prognostic.create_iterator(x, coords)

            tracking_data_list = []
            time_steps = []

            for step, (x_pred, coords_pred) in enumerate(model):
                x_mapped, coords_mapped = map_coords(x_pred, coords_pred, tracker.input_coords())
                output, output_coords = tracker(x_mapped, coords_mapped)

                output = output[:, 0]  # Remove lead_time dimension
                output_np = output.cpu().numpy()
                tracking_data_list.append(output_np)

                time_steps.append(step)

                if step >= parameters.num_steps:
                    break

            if tracking_data_list:
                tracking_array = np.concatenate(tracking_data_list, axis=1)
                io.write(
                    "tracking_output",
                    torch.from_numpy(tracking_array),
                    output_coords,
                )

            zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            forecast_info = {
                "start_time": parameters.start_time,
                "num_steps": parameters.num_steps,
                "region": parameters.region,
                "model_type": "SFNO",
                "tracker_type": "TCTrackerWuDuan",
                "variables": ["tc_lat", "tc_lon", "tc_msl", "tc_w10m"],
            }

            metadata_path = output_dir / "tracking_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(forecast_info, f, indent=2)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Complete!",
                current_step=6,
                total_steps=6,
            ))

            self.update_execution_data(execution_id, {
                "metadata": {
                    **metadata,
                    "results_summary": f"Generated TC tracks for {parameters.num_steps} forecast steps ({len(parameters.start_time)} initialization(s))",
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
