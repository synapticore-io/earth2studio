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
Aurora 6-Hour Forecast (Remote)
================================

Generate 6-hour deterministic forecasts using Microsoft Aurora on a remote server.

This example demonstrates how to:

- Submit an Aurora forecast request to a remote Earth2Studio server
- Retrieve global forecast results
- Extract and plot mean sea level pressure at multiple lead times
"""
# /// script
# dependencies = [
#   "earth2studio[serve]>=0.9.0",
#   "matplotlib>=3.3.0",
#   "cartopy",
# ]
# ///

import os
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from earth2studio.utils.gallery_reference_time import gallery_reference_datetime_utc
from earth2studio_client import RemoteEarth2Workflow


def main(
    plot_file: str = "aurora_msl_forecast.png",
    start_time: datetime | None = None,
    num_steps: int = 4,
) -> None:
    """Run remote Aurora forecast and plot MSL pressure evolution.

    Args:
        plot_file: Path to save the MSL pressure plot
        start_time: Forecast initialization time (UTC). Defaults to a recent reference
            time (see :py:func:`~earth2studio.utils.gallery_reference_time.gallery_reference_datetime_utc`).
        num_steps: Number of 6-hour forecast steps
    """
    if start_time is None:
        start_time = gallery_reference_datetime_utc(days_back=42, hour=0, minute=0)

    api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
    api_token = os.getenv("EARTH2STUDIO_API_TOKEN", "")
    workflow = RemoteEarth2Workflow(
        api_url,
        workflow_name="aurora_forecast",
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

    print(f"   Creating plot: {plot_file}")
    lead_steps = min(4, num_steps)
    fig, axes = plt.subplots(
        1,
        lead_steps,
        figsize=(16, 4),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    if lead_steps == 1:
        axes = [axes]

    for step, ax in enumerate(axes):
        im = ax.pcolormesh(
            ds["lon"],
            ds["lat"],
            ds["msl"].isel(time=0, lead_time=step) / 100.0,
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=950,
            vmax=1050,
        )
        ax.set_title(f"MSL - Lead time: {6*step}h")
        ax.coastlines()
        ax.gridlines(draw_labels=False)

    fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        fraction=0.046,
        pad=0.05,
        label="MSL Pressure (hPa)",
    )
    fig.suptitle(f"Aurora Forecast: {start_time}", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n🎉 Aurora forecast complete!")
    print(f"   Plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
