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
Remote precipitation forecast (REST client)
=========================================

Server-side prognostic + diagnostic workflow (e.g. FCN/DLWP + PrecipitationAFNO), same
spirit as ``02_diagnostic_workflow.py`` but all heavy models run remotely.

Uses workflow ``precipitation_forecast`` when that workflow is registered on the server.

In this example you will learn:

- Calling a diagnostic-capable remote workflow by name
- Plotting total precipitation ``tp`` at a point from the returned dataset

Standalone script copy: ``serve/client/examples/diagnostic_analysis.py``.
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
# Same client pattern as the deterministic remote example; only the workflow name and
# plotted variable change.

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
    workflow_name="precipitation_forecast",
    token=api_token or None,
)

lat, lon = 0.0, 110.0  # Tropical region (Indonesia) with monsoon precipitation
start_time = datetime(2025, 8, 21, 6)
num_steps = 10
plot_path = "outputs/24_remote_tp_timeseries.jpg"


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
# Submit the forecast; the server returns fields including ``tp`` (total precipitation).

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
        tp = _series_at_lat_lon(ds, "tp", lat, lon)
        time_coord = ds["lead_time"].values / np.timedelta64(1, "h")

        plt.close("all")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_coord, tp, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Lead time (h)", fontsize=12)
        ax.set_ylabel("Total precipitation (m)", fontsize=12)
        ax.set_title(
            f"Remote tp at {lat}°, {lon}° — start {start_time} UTC",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {plot_path}")

workflow.close()
