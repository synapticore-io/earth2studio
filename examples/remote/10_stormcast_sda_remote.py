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
StormCast Score-Based Data Assimilation (Remote)
=================================================

Run convection-allowing forecasts with observation data assimilation on a remote server.

This example demonstrates how to:

- Submit a StormCast SDA forecast with or without assimilation
- Retrieve high-resolution regional forecast output
- Compare assimilated vs. non-assimilated forecasts
- Visualize temperature evolution with and without observations
"""
# /// script
# dependencies = [
#   "earth2studio[serve]>=0.9.0",
#   "matplotlib>=3.3.0",
# ]
# ///

import os
from datetime import datetime

import matplotlib.pyplot as plt

from earth2studio.utils.gallery_reference_time import gallery_reference_datetime_utc
from earth2studio_client import RemoteEarth2Workflow


def main(
    plot_file: str = "stormcast_sda_comparison.png",
    start_time: datetime | None = None,
    num_steps: int = 12,
) -> None:
    """Run remote StormCast SDA forecast with and without assimilation.

    Args:
        plot_file: Path to save the comparison plot
        start_time: Forecast initialization time (UTC)
        num_steps: Number of forecast time steps
    """
    if start_time is None:
        start_time = gallery_reference_datetime_utc(days_back=42, hour=18, minute=0)

    api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
    api_token = os.getenv("EARTH2STUDIO_API_TOKEN", "")
    workflow = RemoteEarth2Workflow(
        api_url,
        workflow_name="stormcast_sda_workflow",
        token=api_token or None,
    )

    try:
        health = workflow.client.health_check()
        print(f"✓ API Status: {health.status}")
    except Exception as e:
        print(f"✗ API not available: {e}")
        return

    print("   Running forecast WITHOUT assimilation...")
    try:
        ds_no_assim = workflow(
            start_time=[start_time],
            num_steps=num_steps,
            assimilate_observations=False,
        ).as_dataset()
    except Exception as e:
        print(f"   ⚠ No assimilation forecast failed: {e}")
        ds_no_assim = None

    print("   Running forecast WITH assimilation...")
    try:
        ds_assim = workflow(
            start_time=[start_time],
            num_steps=num_steps,
            assimilate_observations=True,
        ).as_dataset()
    except Exception as e:
        print(f"   ⚠ Assimilation forecast failed: {e}")
        ds_assim = None

    if ds_no_assim is None and ds_assim is None:
        print("\n❌ Both forecasts failed!")
        return

    print(f"   Creating plot: {plot_file}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if ds_no_assim is not None and "t2m" in ds_no_assim:
        t2m_na = ds_no_assim["t2m"].isel(time=0, lead_time=slice(0, 4)).mean(dim=["hrrr_x", "hrrr_y"])
        axes[0].plot(
            range(len(t2m_na)),
            t2m_na.values,
            marker="o",
            linewidth=2,
            markersize=6,
            label="No Assimilation",
        )
        axes[0].set_xlabel("Lead Time (steps)", fontsize=11)
        axes[0].set_ylabel("2-meter Temperature (K)", fontsize=11)
        axes[0].set_title("StormCast SDA: Without Assimilation", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    if ds_assim is not None and "t2m" in ds_assim:
        t2m_assim = ds_assim["t2m"].isel(time=0, lead_time=slice(0, 4)).mean(dim=["hrrr_x", "hrrr_y"])
        axes[1].plot(
            range(len(t2m_assim)),
            t2m_assim.values,
            marker="s",
            linewidth=2,
            markersize=6,
            color="orange",
            label="With Assimilation",
        )
        axes[1].set_xlabel("Lead Time (steps)", fontsize=11)
        axes[1].set_ylabel("2-meter Temperature (K)", fontsize=11)
        axes[1].set_title("StormCast SDA: With ISD Assimilation", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    fig.suptitle(f"StormCast SDA Comparison: {start_time}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n🎉 StormCast SDA comparison complete!")
    print(f"   Plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
