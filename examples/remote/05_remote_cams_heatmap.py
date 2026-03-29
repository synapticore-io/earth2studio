# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# %%
"""
Remote CAMS air quality heatmap (REST client)
==============================================

Fetch CAMS atmospheric composition data from the remote API and visualize
air quality indicators (PM2.5, dust, O3) as geographic heatmaps over Europe.

Demonstrates multi-panel visualization of European air quality from the
remote CAMS workflow at 0.1° resolution.

In this example you will learn:

- Calling the CAMS analysis workflow with preset variables
- Extracting and visualizing multiple pollutant fields simultaneously
- Creating publication-quality heatmaps with geographic extent
"""
# /// script
# dependencies = [
#   "earth2studio[examples]",
#   "zarr>=3.1.0",
# ]
# ///

import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
import numpy as np

from earth2studio.utils.gallery_reference_time import gallery_reference_datetime_utc
from earth2studio_client import RemoteEarth2Workflow

api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
api_token = os.getenv("EARTH2STUDIO_API_TOKEN", "")
workflow = RemoteEarth2Workflow(
    api_url,
    workflow_name="cams_analysis",
    token=api_token or None,
)

start_time = gallery_reference_datetime_utc(days_back=21, hour=0, minute=0)
preset = "eu_surface"
plot_path = "outputs/05_remote_cams_eu_heatmap.jpg"


def _placeholder_fig(message: str, path: str) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=11,
    )
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


try:
    health = workflow.client.health_check()
    print(f"API status: {health.status}")
except Exception as exc:
    print(f"API not available: {exc}")
    _placeholder_fig(
        "Earth2Studio API unreachable — start serve and set EARTH2STUDIO_API_URL",
        plot_path,
    )
else:
    try:
        ds = workflow(start_time=[start_time.isoformat()], preset=preset).as_dataset()
    except Exception as exc:
        print(f"Forecast failed: {exc}")
        _placeholder_fig(f"Forecast failed: {exc}", plot_path)
    else:
        try:
            # Extract three key pollutants (EU surface preset includes these)
            pm25 = ds["pm2p5"].isel(lead_time=0).values
            dust = ds["dust"].isel(lead_time=0).values
            o3 = ds["o3sfc"].isel(lead_time=0).values

            # Get lat/lon for extent calculation (in grid indices for imshow)
            lat = ds["lat"].values
            lon = ds["lon"].values

            plt.close("all")
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            # PM2.5 heatmap
            im0 = axes[0].imshow(pm25, cmap="YlOrRd", origin="upper", aspect="auto")
            axes[0].set_title("PM2.5 (μg/m³)", fontsize=12, fontweight="bold")
            fig.colorbar(im0, ax=axes[0], label="Concentration")

            # Dust heatmap
            im1 = axes[1].imshow(dust, cmap="YlOrBr", origin="upper", aspect="auto")
            axes[1].set_title("Dust (μg/m³)", fontsize=12, fontweight="bold")
            fig.colorbar(im1, ax=axes[1], label="Concentration")

            # O3 heatmap
            im2 = axes[2].imshow(o3, cmap="BuPu", origin="upper", aspect="auto")
            axes[2].set_title("O₃ (μmol/m³)", fontsize=12, fontweight="bold")
            fig.colorbar(im2, ax=axes[2], label="Concentration")

            # Set common labels
            for ax in axes:
                ax.set_xlabel("Longitude (grid index)")
                ax.set_ylabel("Latitude (grid index)")

            fig.suptitle(
                f"CAMS EU Air Quality Analysis — {start_time.strftime('%Y-%m-%d %H:%M UTC')}",
                fontsize=14,
                fontweight="bold",
                y=1.02,
            )
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {plot_path}")
        except Exception as e:
            _placeholder_fig(f"Heatmap generation failed: {e}", plot_path)

workflow.close()
