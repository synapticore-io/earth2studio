#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CBottle Data Generation Workflow"""

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


class CBottleGenerationParameters(WorkflowParameters):
    """Parameters for CBottle generation workflow."""

    timestamp: str = Field(
        default="2024-01-01T00:00:00",
        description="Target timestamp for synthetic data generation (ISO format)",
    )
    num_samples: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of synthetic samples to generate",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )


@workflow_registry.register
class CBottleGenerationWorkflow(Workflow):
    """CBottle synthetic weather data generation workflow."""

    name = "cbottle_generation_workflow"
    description = "Generate synthetic global weather data with CBottle diffusion model"
    Parameters = CBottleGenerationParameters

    def __init__(self) -> None:
        super().__init__()
        try:
            import torch
            from earth2studio.data import CBottle3D

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.package = CBottle3D.load_default_package()
            self.data_source = CBottle3D.load_model(self.package, seed=None)
            self.data_source = self.data_source.to(self.device)
        except Exception:
            logger.warning("CBottle data source not available")
            self.data_source = None

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | CBottleGenerationParameters
    ) -> CBottleGenerationParameters:
        try:
            return CBottleGenerationParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | CBottleGenerationParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the CBottle generation workflow."""

        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            self.update_execution_data(execution_id, {"metadata": metadata})

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=4,
            ))

            from earth2studio.io import ZarrBackend

            if self.data_source is None:
                raise RuntimeError("CBottle data source not available")

            timestamp = datetime.fromisoformat(parameters.timestamp)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress=f"Generating {parameters.num_samples} synthetic samples...",
                current_step=2,
                total_steps=4,
            ))

            output_dir = self.get_output_path(execution_id)
            io = ZarrBackend(file_name=str(output_dir / "results.zarr"))

            for sample_idx in range(parameters.num_samples):
                da = self.data_source(timestamp, self.data_source.output_coords()["variable"])
                from earth2studio.utils.coords import map_coords
                import torch
                x = torch.from_numpy(da.values).float().to(self.device)
                coords_dict = {k: v for k, v in da.coords.items()}
                io.write(da)

                if (sample_idx + 1) % max(1, parameters.num_samples // 3) == 0:
                    self.update_execution_data(execution_id, WorkflowProgress(
                        progress=f"Generated {sample_idx + 1}/{parameters.num_samples} samples...",
                        current_step=2,
                        total_steps=4,
                    ))

            zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            generation_info = {
                "timestamp": parameters.timestamp,
                "num_samples": parameters.num_samples,
                "model": "CBottle3D",
            }

            metadata_path = output_dir / "generation_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(generation_info, f, indent=2)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Complete!",
                current_step=4,
                total_steps=4,
            ))

            self.update_execution_data(execution_id, {
                "metadata": {
                    **metadata,
                    "results_summary": f"Generated {parameters.num_samples} synthetic weather samples",
                    "generation_info": generation_info,
                }
            })

            return {
                "status": "success",
                "output_path": str(output_dir),
                "generation_info": generation_info,
            }

        except Exception as e:
            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Failed!",
                error_message=str(e),
            ))
            raise e
