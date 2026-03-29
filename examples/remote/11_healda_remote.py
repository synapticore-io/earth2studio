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
HealDA Global Data Assimilation (Remote)
=========================================

Produce global weather analyses from satellite and in-situ observations on a remote server.

This example demonstrates how to:

- Submit a HealDA data assimilation request with different observation types
- Retrieve global analyses on HEALPix grids
- Compare conventional vs. satellite observation impact
- Visualize geopotential height at different lead times
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
    plot_file: str = "healda_analysis_comparison.png",
    analysis_time: datetime | None = None,
) -> None:
    """Run remote HealDA analyses with different observation types.

    Args:
        plot_file: Path to save the analysis comparison plot
        analysis_time: Analysis time (UTC). Defaults to a recent reference time.
    """
    if analysis_time is None:
        analysis_time = gallery_reference_datetime_utc(days_back=42, hour=0, minute=0)

    api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
    api_token = os.getenv("EARTH2STUDIO_API_TOKEN", "")
    workflow = RemoteEarth2Workflow(
        api_url,
        workflow_name="healda_workflow",
        token=api_token or None,
    )

    try:
        health = workflow.client.health_check()
        print(f"✓ API Status: {health.status}")
    except Exception as e:
        print(f"✗ API not available: {e}")
        return

    print("   Running HealDA with conventional observations...")
    try:
        ds_conv = workflow(
            analysis_time=[analysis_time],
            observation_types=["conventional"],
        ).as_dataset()
    except Exception as e:
        print(f"   ⚠ Conventional analysis failed: {e}")
        ds_conv = None

    print("   Running HealDA with satellite observations...")
    try:
        ds_sat = workflow(
            analysis_time=[analysis_time],
            observation_types=["satellite"],
        ).as_dataset()
    except Exception as e:
        print(f"   ⚠ Satellite analysis failed: {e}")
        ds_sat = None

    print("   Running HealDA with combined observations...")
    try:
        ds_combined = workflow(
            analysis_time=[analysis_time],
            observation_types=["conventional", "satellite"],
        ).as_dataset()
    except Exception as e:
        print(f"   ⚠ Combined analysis failed: {e}")
        ds_combined = None

    if all(ds is None for ds in [ds_conv, ds_sat, ds_combined]):
        print("\n❌ All analyses failed!")
        return

    print(f"   Creating plot: {plot_file}")
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5), subplot_kw={"projection": ccrs.Robinson()}
    )

    datasets = [
        (ds_conv, "Conventional Obs", axes[0]),
        (ds_sat, "Satellite Obs", axes[1]),
        (ds_combined, "Combined Obs", axes[2]),
    ]

    for ds, title, ax in datasets:
        if ds is None:
            ax.text(
                0.5,
                0.5,
                "Analysis Failed",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        if "z500" in ds:
            var = ds["z500"]
            label = "Z500 (m)"
            vmin, vmax = 5000, 6000
        elif "z_500" in ds:
            var = ds["z_500"]
            label = "Z500 (m)"
            vmin, vmax = 5000, 6000
        elif "geopotential_500" in ds:
            var = ds["geopotential_500"] / 9.81
            label = "Z500 (m)"
            vmin, vmax = 5000, 6000
        else:
            available = [v for v in ds.data_vars if "z" in v.lower() or "geo" in v.lower()]
            if available:
                var = ds[available[0]]
                label = f"{available[0]} (m)"
                vmin, vmax = None, None
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No Z500 in output.\nAvailable: {list(ds.data_vars)[:3]}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
                continue

        im = ax.pcolormesh(
            ds["lon"],
            ds["lat"],
            var.isel(time=0),
            transform=ccrs.PlateCarree(),
            cmap="RdYlBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.coastlines()
        ax.gridlines(draw_labels=False, alpha=0.3)
        plt.colorbar(im, ax=ax, label=label, shrink=0.8)

    fig.suptitle(f"HealDA Global Analysis: {analysis_time}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n🎉 HealDA global analysis complete!")
    print(f"   Plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
