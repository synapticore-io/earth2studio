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
Ensemble Forecasting with Downscaling (Remote)
===============================================

Retrieve ensemble downscaling results from a remote Earth2Studio server.

This example demonstrates how to:

- Submit an ensemble downscaling request to a remote server
- Retrieve and post-process multi-member ensemble output
- Visualize ensemble uncertainty with mean and standard deviation
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
    plot_file: str = "ensemble_downscaling_wind.png",
    start_time: datetime | None = None,
    nensemble: int = 2,
) -> None:
    """Run remote ensemble downscaling workflow and visualize uncertainty.

    Args:
        plot_file: Path to save the wind speed uncertainty plot
        start_time: Forecast initialization time (default: recent reference UTC)
        nensemble: Number of ensemble members
    """
    if start_time is None:
        start_time = gallery_reference_datetime_utc(days_back=42, hour=12, minute=0)

    api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
    api_token = os.getenv("EARTH2STUDIO_API_TOKEN", "")
    workflow = RemoteEarth2Workflow(
        api_url,
        workflow_name="ensemble_workflow",
        token=api_token or None,
    )

    try:
        health = workflow.client.health_check()
        print(f"✓ API Status: {health.status}")
    except Exception as e:
        print(f"✗ API not available: {e}")
        return

    try:
        ds = workflow(
            start_time=[start_time],
            nsteps=4,
            nensemble=nensemble,
            noise_amplitude=0.15,
        ).as_dataset()
    except Exception as e:
        print(f"\n❌ Forecast failed: {e}")
        return

    lead_time_steps = 4
    arr = np.sqrt(ds["u10m"] ** 2 + ds["v10m"] ** 2)
    mean_field = arr.mean(dim="ensemble")
    std_field = arr.std(dim="ensemble")

    print(f"   Creating plot: {plot_file}")
    fig, ax = plt.subplots(
        2,
        lead_time_steps,
        figsize=(14, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    for i in range(lead_time_steps):
        p1 = ax[0, i].contourf(
            ds["lon"],
            ds["lat"],
            mean_field.isel(time=0, lead_time=i),
            levels=20,
            vmin=0,
            vmax=40,
            transform=ccrs.PlateCarree(),
            cmap="nipy_spectral",
        )
        ax[0, i].coastlines()
        ax[0, i].set_title(f"Mean (lead_time={6*i}h)")

        p2 = ax[1, i].contourf(
            ds["lon"],
            ds["lat"],
            std_field.isel(time=0, lead_time=i),
            levels=20,
            vmin=0,
            vmax=4,
            transform=ccrs.PlateCarree(),
            cmap="magma",
        )
        ax[1, i].coastlines()
        ax[1, i].set_title(f"Std (lead_time={6*i}h)")

    fig.colorbar(p1, ax=ax[0, :], label="wind speed mean (m/s)", shrink=0.8)
    fig.colorbar(p2, ax=ax[1, :], label="wind speed std (m/s)", shrink=0.8)
    fig.suptitle(
        f"Ensemble Downscaling: {np.datetime_as_string(ds['time'].values[0], unit='h')}",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n🎉 Ensemble downscaling complete!")
    print(f"   Plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
