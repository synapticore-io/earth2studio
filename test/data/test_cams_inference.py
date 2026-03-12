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

"""CAMS data source integration test with visualization.

Run: pytest test/data/test_cams_inference.py -v -s --slow
"""

import datetime
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest

from earth2studio.data.cams import CAMS, CAMS_FX

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "cams")

YESTERDAY = datetime.datetime.now(datetime.UTC).replace(
    hour=0, minute=0, second=0, microsecond=0
) - datetime.timedelta(days=1)


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_cams_eu_analysis_fetch_and_plot():
    """Fetch EU air quality analysis and produce a map for each variable."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    variables = ["dust", "pm2p5", "no2sfc", "o3sfc"]
    date_str = YESTERDAY.strftime("%Y-%m-%d")

    ds = CAMS(cache=True)
    data = ds([YESTERDAY], variables)

    assert data.shape[0] == 1
    assert data.shape[1] == len(variables)
    assert data.shape[2] > 0
    assert data.shape[3] > 0
    assert not np.isnan(data.values).all()

    lat = data.coords["lat"].values
    lon = data.coords["lon"].values

    print(f"\nCAMS EU Analysis shape: {data.shape}")
    print(f"  Grid: {len(lat)} x {len(lon)} (lat x lon)")
    print(f"  Lat range: [{lat.min():.2f}, {lat.max():.2f}]")
    print(f"  Lon range: [{lon.min():.2f}, {lon.max():.2f}]")

    for i, var in enumerate(variables):
        field = data[0, i].values
        print(f"  {var}: min={np.nanmin(field):.4e}, max={np.nanmax(field):.4e}, mean={np.nanmean(field):.4e}")

        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(12, 8)
        )
        im = ax.pcolormesh(
            lon, lat, field, transform=ccrs.PlateCarree(), cmap="YlOrRd", shading="auto"
        )
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        ax.set_title(f"CAMS EU Analysis — {var} — {date_str} 00:00 UTC")
        fig.colorbar(im, ax=ax, shrink=0.7, label=var)
        fig.savefig(os.path.join(OUTPUT_DIR, f"cams_eu_{var}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPlots saved to {OUTPUT_DIR}/")


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_cams_fx_eu_forecast_fetch_and_plot():
    """Fetch EU forecast at two lead times and produce comparison maps."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    variables = ["dust", "pm2p5"]
    date_str = YESTERDAY.strftime("%Y-%m-%d")
    lead_times = [datetime.timedelta(hours=0), datetime.timedelta(hours=24)]

    ds = CAMS_FX(cache=True)
    data = ds([YESTERDAY], lead_times, variables)

    assert data.shape[0] == 1
    assert data.shape[1] == 2
    assert data.shape[2] == 2
    assert not np.isnan(data.values).all()

    lat = data.coords["lat"].values
    lon = data.coords["lon"].values

    print(f"\nCAMS EU Forecast shape: {data.shape}")
    print(f"  Grid: {len(lat)} x {len(lon)} (lat x lon)")

    for vi, var in enumerate(variables):
        fig, axes = plt.subplots(
            1, 2,
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=(20, 8),
        )
        vmin = np.nanmin(data[0, :, vi].values)
        vmax = np.nanmax(data[0, :, vi].values)

        for lt_idx, (ax, lt) in enumerate(zip(axes, lead_times)):
            field = data[0, lt_idx, vi].values
            hours = int(lt.total_seconds() // 3600)
            print(f"  {var} t+{hours}h: min={np.nanmin(field):.4e}, max={np.nanmax(field):.4e}")

            im = ax.pcolormesh(
                lon, lat, field, transform=ccrs.PlateCarree(),
                cmap="YlOrRd", shading="auto", vmin=vmin, vmax=vmax,
            )
            ax.coastlines()
            ax.gridlines(draw_labels=True)
            ax.set_title(f"{var} — t+{hours}h")

        fig.colorbar(im, ax=axes, shrink=0.7, label=var)
        fig.suptitle(f"CAMS EU Forecast — {var} — init {date_str} 00:00 UTC", fontsize=14)
        fig.savefig(os.path.join(OUTPUT_DIR, f"cams_fx_{var}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPlots saved to {OUTPUT_DIR}/")


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_cams_fx_global_forecast_fetch_and_plot():
    """Fetch global atmospheric composition forecast and produce comparison maps."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    variables = ["aod550", "duaod550"]
    date_str = YESTERDAY.strftime("%Y-%m-%d")
    lead_times = [datetime.timedelta(hours=0), datetime.timedelta(hours=24)]

    ds = CAMS_FX(cache=True)
    data = ds([YESTERDAY], lead_times, variables)

    assert data.shape[0] == 1
    assert data.shape[1] == 2
    assert data.shape[2] == 2
    assert not np.isnan(data.values).all()

    lat = data.coords["lat"].values
    lon = data.coords["lon"].values

    print(f"\nCAMS Global Forecast shape: {data.shape}")
    print(f"  Grid: {len(lat)} x {len(lon)} (lat x lon)")

    for vi, var in enumerate(variables):
        fig, axes = plt.subplots(
            1, 2,
            subplot_kw={"projection": ccrs.Robinson()},
            figsize=(20, 8),
        )
        vmin = np.nanmin(data[0, :, vi].values)
        vmax = np.nanmax(data[0, :, vi].values)

        for lt_idx, (ax, lt) in enumerate(zip(axes, lead_times)):
            field = data[0, lt_idx, vi].values
            hours = int(lt.total_seconds() // 3600)
            print(f"  {var} t+{hours}h: min={np.nanmin(field):.4e}, max={np.nanmax(field):.4e}")

            im = ax.pcolormesh(
                lon, lat, field, transform=ccrs.PlateCarree(),
                cmap="inferno", shading="auto", vmin=vmin, vmax=vmax,
            )
            ax.coastlines()
            ax.gridlines()
            ax.set_global()
            ax.set_title(f"{var} — t+{hours}h")

        fig.colorbar(im, ax=axes, shrink=0.5, label=var)
        fig.suptitle(f"CAMS Global Forecast — {var} — init {date_str} 00:00 UTC", fontsize=14)
        fig.savefig(os.path.join(OUTPUT_DIR, f"cams_global_{var}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPlots saved to {OUTPUT_DIR}/")
