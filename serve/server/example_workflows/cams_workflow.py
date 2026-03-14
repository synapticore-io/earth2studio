# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0

"""
CAMS Analysis & Forecast Workflows

Serve ECMWF/CAMS atmospheric composition data (dust, PM2.5, SO₂, O₃, etc.)
via the Earth2Studio REST API.

Two workflows:
- cams_analysis: EU surface analysis (0.1° grid, hourly)
- cams_forecast: EU surface + global column forecasts
"""

from datetime import datetime, timedelta
from typing import Literal

import torch

from earth2studio.data import CAMS, CAMS_FX
from earth2studio.io import IOBackend
from earth2studio.serve.server import Earth2Workflow, workflow_registry

# Common CAMS variable presets
EU_SURFACE = ["dust", "pm2p5", "pm10", "so2sfc", "no2sfc", "o3sfc"]
EU_MULTI_LEVEL = [
    "dust",
    "dust_500m",
    "dust_1000m",
    "dust_3000m",
    "dust_5000m",
    "pm2p5",
    "pm2p5_500m",
    "pm2p5_1000m",
    "pm2p5_3000m",
    "pm2p5_5000m",
]
GLOBAL_AOD = ["aod550", "duaod550", "omaod550", "bcaod550", "ssaod550", "suaod550"]


def _resolve_preset(
    variables: list[str] | None, preset: str
) -> list[str]:
    if variables:
        return variables
    presets = {
        "eu_surface": EU_SURFACE,
        "eu_multi_level": EU_MULTI_LEVEL,
        "global_aod": GLOBAL_AOD,
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}', available: {list(presets)}")
    return presets[preset]


@workflow_registry.register
class CAMSAnalysisWorkflow(Earth2Workflow):
    """CAMS European air quality analysis — hourly surface + multi-level data."""

    name = "cams_analysis"
    description = "CAMS EU analysis (dust, PM2.5, SO₂, NO₂, O₃, CO) at 0.1° resolution"

    def __init__(self) -> None:
        super().__init__()
        self.data = CAMS(cache=True, verbose=True)

    def __call__(
        self,
        io: IOBackend,
        start_time: list[datetime] = [datetime(2025, 6, 1, 0)],
        variables: list[str] | None = None,
        preset: Literal["eu_surface", "eu_multi_level"] = "eu_surface",
    ) -> None:
        """Fetch CAMS EU analysis and write to IO backend.

        Parameters
        ----------
        io : IOBackend
            Output backend (Zarr or NetCDF4), provided by the serve framework.
        start_time : list[datetime]
            UTC timestamps to fetch analysis for (hourly resolution).
        variables : list[str] | None
            Explicit variable list (CAMSLexicon names). Overrides preset.
        preset : str
            Variable preset: "eu_surface" (6 vars) or "eu_multi_level" (10 vars).
        """
        resolved = _resolve_preset(variables, preset)
        da = self.data(start_time, resolved)

        coords = {
            "time": da.coords["time"].values,
            "variable": da.coords["variable"].values,
            "lat": da.coords["lat"].values,
            "lon": da.coords["lon"].values,
        }
        io.add_array(coords, "cams_analysis")
        io.write(torch.from_numpy(da.values), coords, "cams_analysis")


@workflow_registry.register
class CAMSForecastWorkflow(Earth2Workflow):
    """CAMS forecast — EU surface + global column/AOD forecasts."""

    name = "cams_forecast"
    description = "CAMS forecast (EU surface + global AOD/column) via CDS API"

    def __init__(self) -> None:
        super().__init__()
        self.data = CAMS_FX(cache=True, verbose=True)

    def __call__(
        self,
        io: IOBackend,
        start_time: list[datetime] = [datetime(2025, 6, 1, 0)],
        lead_hours: list[int] = [0, 6, 12, 24, 48],
        variables: list[str] | None = None,
        preset: Literal["eu_surface", "eu_multi_level", "global_aod"] = "eu_surface",
    ) -> None:
        """Fetch CAMS forecast and write to IO backend.

        Parameters
        ----------
        io : IOBackend
            Output backend, provided by the serve framework.
        start_time : list[datetime]
            Forecast initialization times (UTC).
        lead_hours : list[int]
            Lead times in hours from initialization.
        variables : list[str] | None
            Explicit variable list. Overrides preset.
        preset : str
            Variable preset: "eu_surface", "eu_multi_level", or "global_aod".
        """
        resolved = _resolve_preset(variables, preset)
        lead_times = [timedelta(hours=h) for h in lead_hours]
        da = self.data(start_time, lead_times, resolved)

        coords = {
            "time": da.coords["time"].values,
            "lead_time": da.coords["lead_time"].values,
            "variable": da.coords["variable"].values,
            "lat": da.coords["lat"].values,
            "lon": da.coords["lon"].values,
        }
        io.add_array(coords, "cams_forecast")
        io.write(torch.from_numpy(da.values), coords, "cams_forecast")
