# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
"""
Remote conditioning + local StormCast downscaling
===============================================

Hybrid workflow: fetch conditioning fields from the API
(``stormcast_fcn3_workflow`` with ``run_stormcast=False``), then run
``StormCast`` locally with ``earth2studio.run.deterministic``.

This mirrors the SDK script ``serve/client/examples/downscaled_forecast.py`` and needs a
GPU-capable machine, HRRR access, and pretrained StormCast weights.

In this example you will learn:

- Using ``RemoteEarth2WorkflowResult.as_data_source``
- Wiring a remote ``InferenceOutputSource`` into StormCast
- Running local downscaled inference with HRRR initial conditions

.. warning::
   Needs ``earth2studio[stormcast]``, the slim REST client package, and a running API;
   not intended for quick doc-gallery smoke runs without those services.
"""
# /// script
# dependencies = [
#   "earth2studio-client",
#   "earth2studio[stormcast]",
#   "matplotlib>=3.3.0",
#   "python-dotenv",
#   "zarr>=3.1.0",
# ]
#
# [tool.uv.sources]
# earth2studio-client = { path = "../../serve/client" }
# earth2studio = { path = "../.." }
# ///

# %%
# Set Up
# ------
# Remote client for conditioning; local StormCast + HRRR + ``XarrayBackend``.

# %%
import os
from datetime import datetime

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
import torch
from earth2studio_client import RemoteEarth2Workflow

from earth2studio.data import HRRR
from earth2studio.io import XarrayBackend
from earth2studio.models.px import StormCast
from earth2studio.run import deterministic

api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
api_token = os.getenv("EARTH2STUDIO_API_TOKEN", "")
workflow = RemoteEarth2Workflow(
    api_url,
    workflow_name="stormcast_fcn3_workflow",
    token=api_token or None,
)

start_time = datetime(2025, 8, 21, 6)
num_hours = 10
hrrr_x, hrrr_y = 200.0, 200.0
plot_path = "outputs/25_remote_stormcast_t2m.jpg"


def _placeholder_fig(message: str, path: str) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# %%
# Execute
# -------
# Pull conditioning from the server, attach to StormCast, run deterministic rollout on
# ``cuda`` when available.

# %%
try:
    health = workflow.client.health_check()
    print(f"API status: {health.status}")
except Exception as exc:
    print(f"API not available: {exc}")
    _placeholder_fig("Earth2Studio API unreachable — hybrid example skipped", plot_path)
else:
    try:
        conditioning_source = workflow(
            start_time=start_time, num_hours=num_hours, run_stormcast=False
        ).as_data_source()
    except Exception as exc:
        print(f"Remote conditioning failed: {exc}")
        _placeholder_fig(f"Remote conditioning failed: {exc}", plot_path)
    else:
        try:
            stormcast = StormCast.from_pretrained()
            stormcast.conditioning_data_source = conditioning_source
            io = XarrayBackend()
            hrrr_ic = HRRR()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            deterministic([start_time], num_hours, stormcast, hrrr_ic, io, device=device)
        except Exception as exc:
            print(f"Local StormCast failed: {exc}")
            _placeholder_fig(f"Local StormCast failed: {exc}", plot_path)
        else:
            ds = io.root
            t2m = ds["t2m"].sel(hrrr_x=hrrr_x, hrrr_y=hrrr_y, method="nearest").values.ravel()
            time_coord = ds["lead_time"].values.astype("timedelta64[h]")

            plt.close("all")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_coord, t2m, marker="o", linewidth=2, markersize=6)
            ax.set_xlabel("Lead time (h)", fontsize=12)
            ax.set_ylabel("2 m temperature (K)", fontsize=12)
            ax.set_title(
                f"StormCast t2m at HRRR x={hrrr_x}, y={hrrr_y}",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {plot_path}")

workflow.close()
