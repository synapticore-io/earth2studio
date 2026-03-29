#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""StormScope GOES Satellite Nowcasting Workflow"""

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


class StormScopeParameters(WorkflowParameters):
    """Parameters for StormScope workflow."""

    initialization_time: str = Field(
        default="2024-01-01T00:00:00",
        description="Initialization time in ISO format",
    )
    num_forecast_steps: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Number of 10-minute or 60-minute forecast steps",
    )
    model_name: str = Field(
        default="6km_60min_natten_cos_zenith_input_eoe_v2",
        description="StormScope model variant (60min or 10min timestep)",
    )


@workflow_registry.register
class StormScopeWorkflow(Workflow):
    """StormScope satellite-based nowcasting with GOES imagery."""

    name = "stormscope_workflow"
    description = "StormScope GOES satellite nowcasting (10min-1hr resolution)"
    Parameters = StormScopeParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | StormScopeParameters
    ) -> StormScopeParameters:
        try:
            return StormScopeParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | StormScopeParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the StormScope nowcasting workflow."""

        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            self.update_execution_data(execution_id, {"metadata": metadata})

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=6,
            ))

            import torch
            import xarray as xr
            from earth2studio.data import GOES, GFS_FX, fetch_data
            from earth2studio.io import ZarrBackend
            from earth2studio.models.px import StormScopeGOES
            from earth2studio.utils.coords import map_coords

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Loading StormScope GOES model...",
                current_step=2,
                total_steps=6,
            ))

            package = StormScopeGOES.load_default_package()
            model = StormScopeGOES.load_model(
                package,
                model_name=parameters.model_name,
                conditioning_data_source=GFS_FX(),
            )
            model = model.to(device)
            model.eval()

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Setting up GOES and GFS data sources...",
                current_step=3,
                total_steps=6,
            ))

            goes = GOES(satellite="goes19", scan_mode="C")
            output_dir = self.get_output_path(execution_id)
            io = ZarrBackend(
                file_name=str(output_dir / "results.zarr"),
                chunks={"time": 1, "lead_time": 1},
                backend_kwargs={"overwrite": True},
            )

            init_time = datetime.fromisoformat(parameters.initialization_time)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Fetching GOES initialization data...",
                current_step=4,
                total_steps=6,
            ))

            # Get GOES grid coordinates and fetch data
            goes_lat, goes_lon = GOES.grid(satellite="goes19", scan_mode="C")
            x, coords = fetch_data(
                source=goes,
                time=[init_time],
                variable=model.input_coords()["variable"],
                device=device,
            )

            self.update_execution_data(execution_id, WorkflowProgress(
                progress=f"Running StormScope nowcast ({parameters.num_forecast_steps} steps)...",
                current_step=5,
                total_steps=6,
            ))

            # Build interpolators for model grid
            model.build_input_interpolator(goes_lat, goes_lon)
            model.build_conditioning_interpolator(GFS_FX.GFS_LAT, GFS_FX.GFS_LON)

            # Run forecast loop
            y = x
            y_coords = coords
            predictions = []
            lead_times = []

            for step_idx in range(parameters.num_forecast_steps):
                y_pred, y_pred_coords = model(y, y_coords)
                predictions.append(y_pred.detach().cpu())
                lead_times.append(y_pred_coords.get("lead_time", [0])[0])

                # Update sliding window
                y, y_coords = model.next_input(y_pred, y_pred_coords, y, y_coords)

            # Stack predictions and write to IO backend
            if predictions:
                import numpy as np
                pred_stack = torch.stack(predictions).numpy()  # [steps, B, T, L, C, H, W]
                lat = model.latitudes.detach().cpu().numpy()
                lon = model.longitudes.detach().cpu().numpy()

                # Create xarray Dataset with all predictions
                vars_dict = {}
                for ch_idx, var_name in enumerate(model.variables):
                    vars_dict[var_name] = xr.DataArray(
                        pred_stack[:, 0, 0, 0, ch_idx, :, :],  # [steps, H, W]
                        dims=["lead_time", "lat", "lon"],
                        coords={
                            "lead_time": np.arange(len(predictions)),
                            "lat": lat,
                            "lon": lon,
                            "time": init_time,
                        },
                    )

                ds = xr.Dataset(vars_dict)
                io.write(ds)

            zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            forecast_info = {
                "initialization_time": parameters.initialization_time,
                "num_steps": parameters.num_forecast_steps,
                "model": "StormScope",
                "model_name": parameters.model_name,
                "data_source": "GOES",
                "device": str(device),
            }

            metadata_path = output_dir / "forecast_metadata.json"
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
                    "results_summary": f"Generated {parameters.num_forecast_steps}-step StormScope nowcast",
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
