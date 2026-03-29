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
StormScope Satellite and Radar Nowcasting (Remote)
===================================================

Run coupled GOES satellite and MRMS radar nowcasting on a remote server.

This example demonstrates how to:

- Submit a StormScope nowcast request with GOES and MRMS data
- Retrieve multi-channel satellite imagery and radar reflectivity
- Visualize radar reflectivity at multiple lead times
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
import numpy as np

from earth2studio.utils.gallery_reference_time import gallery_reference_datetime_utc
from earth2studio_client import RemoteEarth2Workflow


def main(
    plot_file: str = "stormscope_mrms_nowcast.png",
    start_time: datetime | None = None,
    num_steps: int = 6,
) -> None:
    """Run remote StormScope nowcast and visualize radar reflectivity.

    Args:
        plot_file: Path to save the reflectivity nowcast plot
        start_time: Nowcast initialization time (UTC)
        num_steps: Number of forecast steps (typical: 15 min per step)
    """
    if start_time is None:
        start_time = gallery_reference_datetime_utc(days_back=14, hour=20, minute=0)

    api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
    api_token = os.getenv("EARTH2STUDIO_API_TOKEN", "")
    workflow = RemoteEarth2Workflow(
        api_url,
        workflow_name="stormscope_workflow",
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
        print(f"\n❌ Nowcast failed: {e}")
        return

    print(f"   Creating plot: {plot_file}")
    lead_steps = min(4, num_steps)
    fig, axes = plt.subplots(
        1,
        lead_steps,
        figsize=(16, 4),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if lead_steps == 1:
        axes = [axes]

    if "reflectivity" in ds:
        var_name = "reflectivity"
        vmin, vmax = 0, 60
        cmap = "viridis"
        units = "dBZ"
    elif "radar_reflectivity" in ds:
        var_name = "radar_reflectivity"
        vmin, vmax = 0, 60
        cmap = "viridis"
        units = "dBZ"
    else:
        print("⚠ Warning: reflectivity variable not found in output")
        available_vars = [v for v in ds.data_vars if "reflect" in v.lower()]
        if available_vars:
            var_name = available_vars[0]
            vmin, vmax = 0, 60
            cmap = "viridis"
            units = "dBZ"
        else:
            print(f"Available variables: {list(ds.data_vars)}")
            return

    for step, ax in enumerate(axes):
        im = ax.pcolormesh(
            ds["lon"],
            ds["lat"],
            ds[var_name].isel(time=0, lead_time=step),
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Reflectivity - {15*step}min")
        ax.coastlines()
        ax.gridlines(draw_labels=False)

    fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        fraction=0.046,
        pad=0.05,
        label=f"Radar Reflectivity ({units})",
    )
    fig.suptitle(f"StormScope Nowcast: {start_time}", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n🎉 StormScope nowcast complete!")
    print(f"   Plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
