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
Remote deterministic forecast (REST client)
==========================================

Call a hosted Earth2Studio workflow and plot 2 m temperature at a point — same idea as
``01_deterministic_workflow.py`` but inference runs on the API server.

Requires a running Earth2Studio serve stack and the ``deterministic_earth2_workflow``
workflow (see ``serve/server/example_workflows``). Set ``EARTH2STUDIO_API_URL`` and
optionally ``EARTH2STUDIO_API_TOKEN``.

In this example you will learn:

- How to use ``earth2studio_client.RemoteEarth2Workflow``
- How to retrieve results as an ``xarray`` dataset
- How to plot a lead-time series with Matplotlib

Standalone script copy: ``serve/client/examples/basic_forecast.py``.
"""
# /// script
# dependencies = [
#   "earth2studio-client",
#   "matplotlib>=3.3.0",
#   "numpy",
#   "python-dotenv",
#   "zarr>=3.1.0",
# ]
#
# [tool.uv.sources]
# earth2studio-client = { path = "../../serve/client" }
# ///

# %%
# Set Up
# ------
# Use the high-level client :py:class:`earth2studio_client.RemoteEarth2Workflow`, which
# wraps HTTP submission and polling. Environment variables follow the client SDK README.

# %%
import os
from datetime import datetime

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
import numpy as np
from earth2studio_client import RemoteEarth2Workflow

api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
api_token = os.getenv("EARTH2STUDIO_API_TOKEN", "")
workflow = RemoteEarth2Workflow(
    api_url,
    workflow_name="deterministic_earth2_workflow",
    token=api_token or None,
)

lat, lon = 37.4, -122.0
start_time = datetime(2025, 8, 21, 6)
num_steps = 10
plot_path = "outputs/23_remote_t2m_timeseries.jpg"


def _series_at_lat_lon(ds, var: str, lat0: float, lon0: float):
    """Nearest gridpoint time series (isel avoids non-unique coordinate labels from .sel)."""
    da = ds[var]
    lat_d = ds["lat"].values.astype(np.float64)
    lon_d = ds["lon"].values.astype(np.float64)
    # Handle negative longitudes (datasets may use 0-360)
    lon0_adj = lon0 % 360 if lon_d.min() >= 0 and lon0 < 0 else lon0
    ilat = int(np.argmin(np.abs(lat_d - lat0)))
    ilon = int(np.argmin(np.abs(lon_d - lon0_adj)))
    return da.isel(lat=ilat, lon=ilon).values.ravel()


def _placeholder_fig(message: str, path: str) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# %%
# Execute
# -------
# Health-check the API, submit ``start_time`` and ``num_steps``, then open the workflow
# output as an xarray dataset (Zarr or NetCDF, depending on server storage).

# %%
try:
    health = workflow.client.health_check()
    print(f"API status: {health.status}")
except Exception as exc:
    print(f"API not available: {exc}")
    _placeholder_fig("Earth2Studio API unreachable — start serve and set EARTH2STUDIO_API_URL", plot_path)
else:
    try:
        ds = workflow(start_time=[start_time], num_steps=num_steps).as_dataset()
    except Exception as exc:
        print(f"Forecast failed: {exc}")
        _placeholder_fig(f"Forecast failed: {exc}", plot_path)
    else:
        t2m = _series_at_lat_lon(ds, "t2m", lat, lon)
        time_coord = ds["lead_time"].values / np.timedelta64(1, "h")

        plt.close("all")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_coord, t2m, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Lead time (h)", fontsize=12)
        ax.set_ylabel("2 m temperature (K)", fontsize=12)
        ax.set_title(
            f"Remote t2m at {lat}°, {lon}° — start {start_time} UTC",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {plot_path}")

workflow.close()
