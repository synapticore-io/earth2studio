# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# %%
"""
Remote CAMS forecast timeseries (REST client)
==============================================

Fetch multi-day CAMS air quality forecast from the remote API and extract
pollutant timeseries at multiple European cities.

Demonstrates extracting and plotting forecast trajectories at specific
geographic locations from the remote CAMS forecast workflow.

In this example you will learn:

- Calling the CAMS forecast workflow with lead times
- Extracting timeseries at specific lat/lon locations
- Plotting multi-city forecast comparisons
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
    workflow_name="cams_forecast",
    token=api_token or None,
)

start_time = gallery_reference_datetime_utc(days_back=21, hour=0, minute=0)
lead_hours = [0, 6, 12, 18, 24, 30, 36, 42, 48]
preset = "eu_surface"
plot_path = "outputs/06_remote_cams_forecast_timeseries.jpg"

# Three European cities
cities = {
    "Berlin": (52.52, 13.40),
    "Paris": (48.86, 2.35),
    "Warsaw": (52.23, 21.01),
}


def _series_at_lat_lon(ds, var: str, lat0: float, lon0: float):
    """Nearest gridpoint timeseries."""
    da = ds[var]
    lat_d = ds["lat"].values.astype(np.float64)
    lon_d = ds["lon"].values.astype(np.float64)
    # Adjust for dateline (CAMS data may use different convention)
    lon0_adj = lon0 % 360 if lon_d.min() >= 0 and lon0 < 0 else lon0
    ilat = int(np.argmin(np.abs(lat_d - lat0)))
    ilon = int(np.argmin(np.abs(lon_d - lon0_adj)))
    return da.isel(lat=ilat, lon=ilon).values.ravel()


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
        ds = workflow(
            start_time=[start_time.isoformat()],
            lead_hours=lead_hours,
            preset=preset,
        ).as_dataset()
    except Exception as exc:
        print(f"Forecast failed: {exc}")
        _placeholder_fig(f"Forecast failed: {exc}", plot_path)
    else:
        try:
            # Extract timeseries at each city
            lead_time_hours = ds["lead_time"].values / np.timedelta64(1, "h")

            plt.close("all")
            fig, ax = plt.subplots(figsize=(10, 6))

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            markers = ["o", "s", "^"]

            for (city_name, (lat, lon)), color, marker in zip(
                cities.items(), colors, markers
            ):
                pm25 = _series_at_lat_lon(ds, "pm2p5", lat, lon)
                ax.plot(
                    lead_time_hours,
                    pm25,
                    marker=marker,
                    linewidth=2,
                    markersize=6,
                    label=city_name,
                    color=color,
                )

            ax.set_xlabel("Lead time (h)", fontsize=12)
            ax.set_ylabel("PM2.5 (μg/m³)", fontsize=12)
            ax.set_title(
                f"CAMS PM2.5 Forecast at EU Cities — {start_time.strftime('%Y-%m-%d %H:%M UTC')}",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend(fontsize=11, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {plot_path}")
        except Exception as e:
            _placeholder_fig(f"Timeseries generation failed: {e}", plot_path)

workflow.close()
