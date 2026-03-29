#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CBottle Super Resolution Workflow"""

import json
import logging
from datetime import datetime
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


class CBottleSuperResolutionParameters(WorkflowParameters):
    """Parameters for CBottle super-resolution workflow."""

    timestamp: str = Field(
        default="2024-01-01T00:00:00",
        description="Target timestamp for super-resolution (ISO format)",
    )
    source: Literal["cbottle", "era5"] = Field(
        default="cbottle",
        description="Data source (cbottle for synthetic, era5 for real data)",
    )
    num_samples: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of super-resolution samples to generate",
    )


@workflow_registry.register
class CBottleSuperResolutionWorkflow(Workflow):
    """CBottle super-resolution workflow for high-res global weather."""

    name = "cbottle_super_resolution_workflow"
    description = "Super-resolution upscaling of global weather data with CBottle"
    Parameters = CBottleSuperResolutionParameters

    def __init__(self) -> None:
        super().__init__()
        try:
            import torch
            from earth2studio.data import CBottle3D, WB2ERA5
            from earth2studio.models.dx import CBottleSR

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.cbottle_package = CBottle3D.load_default_package()
            self.cbottle_ds = CBottle3D.load_model(self.cbottle_package, seed=None)
            self.cbottle_ds = self.cbottle_ds.to(self.device)

            self.sr_package = CBottleSR.load_default_package()
            self.sr_model = CBottleSR.load_model(self.sr_package)
            self.sr_model = self.sr_model.to(self.device)

            self.era5_ds = WB2ERA5()
        except Exception:
            logger.warning("CBottle SR components not available")
            self.cbottle_ds = None
            self.sr_model = None

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | CBottleSuperResolutionParameters
    ) -> CBottleSuperResolutionParameters:
        try:
            return CBottleSuperResolutionParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | CBottleSuperResolutionParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the CBottle super-resolution workflow."""

        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            self.update_execution_data(execution_id, {"metadata": metadata})

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=5,
            ))

            if self.sr_model is None:
                raise RuntimeError("CBottle SR model not available")

            import torch
            from earth2studio.io import ZarrBackend

            timestamp = datetime.fromisoformat(parameters.timestamp)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress=f"Loading {parameters.source} data source...",
                current_step=2,
                total_steps=5,
            ))

            if parameters.source == "cbottle":
                da = self.cbottle_ds(
                    timestamp,
                    self.cbottle_ds.output_coords()["variable"]
                )
            elif parameters.source == "era5":
                da = self.era5_ds(timestamp, self.era5_ds.output_coords()["variable"])
            else:
                raise ValueError(f"Unknown source: {parameters.source}")

            output_dir = self.get_output_path(execution_id)
            io = ZarrBackend(file_name=str(output_dir / "results.zarr"))

            self.update_execution_data(execution_id, WorkflowProgress(
                progress=f"Running super-resolution ({parameters.num_samples} samples)...",
                current_step=3,
                total_steps=5,
            ))

            for sample_idx in range(parameters.num_samples):
                x = torch.from_numpy(da.values).float().unsqueeze(0).to(self.device)

                with torch.no_grad():
                    y, _ = self.sr_model(x, {})

                sr_data = y.squeeze(0).cpu().numpy()
                da.values = sr_data
                io.write(da)

                if (sample_idx + 1) % max(1, parameters.num_samples // 2) == 0:
                    self.update_execution_data(execution_id, WorkflowProgress(
                        progress=f"Completed {sample_idx + 1}/{parameters.num_samples} samples...",
                        current_step=3,
                        total_steps=5,
                    ))

            zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            sr_info = {
                "timestamp": parameters.timestamp,
                "source": parameters.source,
                "num_samples": parameters.num_samples,
                "model": "CBottleSR",
            }

            metadata_path = output_dir / "sr_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(sr_info, f, indent=2)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Complete!",
                current_step=5,
                total_steps=5,
            ))

            self.update_execution_data(execution_id, {
                "metadata": {
                    **metadata,
                    "results_summary": f"Generated {parameters.num_samples} super-resolution samples",
                    "sr_info": sr_info,
                }
            })

            return {
                "status": "success",
                "output_path": str(output_dir),
                "sr_info": sr_info,
            }

        except Exception as e:
            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Failed!",
                error_message=str(e),
            ))
            raise e
