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
Precipitation forecast example using Earth2Studio Client SDK.

Runs the server workflow ``precipitation_forecast`` (FCN/DLWP + PrecipitationAFNO),
then plots total precipitation at a point — same idea as the old local-diagnostic
demo, without requiring a client-side ``as_model()`` API.

Sphinx gallery variant: ``examples/client/24_remote_precipitation_forecast.py``.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt

from earth2studio_client import RemoteEarth2Workflow

# /// script
# dependencies = [
#   "earth2studio[serve]>=0.9.0",
#   "matplotlib>=3.3.0",
# ]
# ///


def main(
    plot_file: str = "tp_plot.png",
    lat: float = 37.4,
    lon: float = -122.0,
    start_time: datetime = datetime(2025, 8, 21, 6),
    num_steps: int = 10,
) -> None:
    """Run remote precipitation workflow and save a tp plot.

    Args:
        plot_file: Path to save the precipitation plot (default: 'tp_plot.png')
        lat: Latitude for precipitation extraction (default: 37.4)
        lon: Longitude for precipitation extraction (default: -122.0)
        start_time: Start time for the forecast (default: 2025-08-21 06:00 UTC)
        num_steps: Number of forecast time steps (default: 10)
    """

    api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
    api_token = os.getenv("EARTH2STUDIO_API_TOKEN", "")
    workflow = RemoteEarth2Workflow(
        api_url,
        workflow_name="precipitation_forecast",
        token=api_token or None,
    )

    try:
        health = workflow.client.health_check()
        print(f"✓ API Status: {health.status}")
    except Exception as e:
        print(f"✗ API not available: {e}")
        return

    try:
        ds = workflow(start_time=[start_time], num_steps=num_steps).as_dataset()
    except Exception as e:
        print(f"\n❌ Forecast failed: {e}")
        return

    tp = ds["tp"].sel(lat=lat, lon=lon, method="nearest").values.ravel()
    time_coord = ds["lead_time"].values.astype("timedelta64[h]")

    print(f"   Creating plot: {plot_file}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_coord, tp, marker="o", linewidth=2, markersize=6)
    ax.set_xlabel("Lead Time (h)", fontsize=12)
    ax.set_ylabel("Total Precipitation (m)", fontsize=12)
    ax.set_title(
        f"Precipitation Forecast starting {start_time} at {lat}°N, {lon}°E",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n🎉 Forecast complete!")
    print(f"   Plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
