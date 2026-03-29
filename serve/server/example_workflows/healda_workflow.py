#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""HealDA Global Data Assimilation Workflow"""

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


class HealDAParameters(WorkflowParameters):
    """Parameters for HealDA data assimilation workflow."""

    analysis_time: str = Field(
        default="2024-01-01T00:00:00",
        description="Analysis time in ISO format",
    )
    observation_type: Literal["conventional", "satellite", "both"] = Field(
        default="both",
        description="Type of observations to assimilate",
    )
    num_analysis_cycles: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of analysis cycles",
    )


@workflow_registry.register
class HealDAWorkflow(Workflow):
    """HealDA global data assimilation workflow."""

    name = "healda_workflow"
    description = "HealDA global weather analysis from observations"
    Parameters = HealDAParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | HealDAParameters
    ) -> HealDAParameters:
        try:
            return HealDAParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | HealDAParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the HealDA assimilation workflow."""

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
            from earth2studio.data import UFSObsConv, UFSObsSat, fetch_dataframe
            from earth2studio.io import ZarrBackend
            from earth2studio.models.da import HealDA

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Loading HealDA model...",
                current_step=2,
                total_steps=5,
            ))

            package = HealDA.load_default_package()
            model = HealDA.load_model(package)
            model = model.to(device)
            model.eval()

            analysis_time = datetime.fromisoformat(parameters.analysis_time)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Fetching observations...",
                current_step=3,
                total_steps=5,
            ))

            obs_dict = {}

            if parameters.observation_type in ["conventional", "both"]:
                try:
                    obs_conv = UFSObsConv()
                    obs_dict["conventional"] = fetch_dataframe(
                        source=obs_conv,
                        time=[analysis_time],
                    )
                except Exception as e:
                    logger.warning(f"Could not fetch conventional observations: {e}")

            if parameters.observation_type in ["satellite", "both"]:
                try:
                    obs_sat = UFSObsSat()
                    obs_dict["satellite"] = fetch_dataframe(
                        source=obs_sat,
                        time=[analysis_time],
                    )
                except Exception as e:
                    logger.warning(f"Could not fetch satellite observations: {e}")

            output_dir = self.get_output_path(execution_id)
            io = ZarrBackend(
                file_name=str(output_dir / "results.zarr"),
                chunks={"lat": 64, "lon": 64},
                backend_kwargs={"overwrite": True},
            )

            self.update_execution_data(execution_id, WorkflowProgress(
                progress=f"Running HealDA analysis ({parameters.num_analysis_cycles} cycle(s))...",
                current_step=4,
                total_steps=5,
            ))

            for cycle_idx in range(parameters.num_analysis_cycles):
                if obs_dict:
                    obs_conv_df = obs_dict.get("conventional")
                    obs_sat_df = obs_dict.get("satellite")

                    with torch.no_grad():
                        analysis_xa = model(
                            observations_conv=obs_conv_df,
                            observations_sat=obs_sat_df,
                        )

                    if hasattr(analysis_xa, "to_dataset"):
                        io.write(analysis_xa.to_dataset())
                    else:
                        io.write(analysis_xa)
                else:
                    logger.warning("No observations available for assimilation")

            zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            analysis_info = {
                "analysis_time": parameters.analysis_time,
                "observation_type": parameters.observation_type,
                "num_cycles": parameters.num_analysis_cycles,
                "model": "HealDA",
            }

            metadata_path = output_dir / "analysis_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(analysis_info, f, indent=2)

            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Complete!",
                current_step=5,
                total_steps=5,
            ))

            self.update_execution_data(execution_id, {
                "metadata": {
                    **metadata,
                    "results_summary": f"Generated HealDA analysis using {parameters.observation_type} observations",
                    "analysis_info": analysis_info,
                }
            })

            return {
                "status": "success",
                "output_path": str(output_dir),
                "analysis_info": analysis_info,
            }

        except Exception as e:
            self.update_execution_data(execution_id, WorkflowProgress(
                progress="Failed!",
                error_message=str(e),
            ))
            raise e
