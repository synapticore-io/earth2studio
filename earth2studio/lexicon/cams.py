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

from collections.abc import Callable

import numpy as np

from .base import LexiconType


class CAMSLexicon(metaclass=LexiconType):
    """Copernicus Atmosphere Monitoring Service Lexicon

    CAMS specified ``<dataset>::<api_variable>::<netcdf_key>::<level>``

    The API variable name (used in the cdsapi request) differs from the NetCDF
    key (used to index the downloaded file). Both are stored in the VOCAB.

    Note
    ----
    Additional resources:
    https://ads.atmosphere.copernicus.eu/datasets/cams-europe-air-quality-forecasts
    https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts
    """

    # Shorthand for dataset names
    _EU = "cams-europe-air-quality-forecasts"
    _GLOBAL = "cams-global-atmospheric-composition-forecasts"

    # Available EU levels: 0(sfc), 50, 100, 250, 500, 750, 1000, 2000, 3000, 5000
    _EU_LEVELS = [0, 50, 100, 250, 500, 750, 1000, 2000, 3000, 5000]

    VOCAB = {
        # ---- CAMS Europe Surface (level 0, 0.1° grid) ----
        # e2s_name: dataset :: api_request_name :: netcdf_key :: level
        "dust": f"{_EU}::dust::dust::0",
        "so2sfc": f"{_EU}::sulphur_dioxide::so2_conc::0",
        "pm2p5": f"{_EU}::particulate_matter_2.5um::pm2p5_conc::0",
        "pm10": f"{_EU}::particulate_matter_10um::pm10_conc::0",
        "no2sfc": f"{_EU}::nitrogen_dioxide::no2_conc::0",
        "o3sfc": f"{_EU}::ozone::o3_conc::0",
        "cosfc": f"{_EU}::carbon_monoxide::co_conc::0",
        "nh3sfc": f"{_EU}::ammonia::nh3_conc::0",
        "nosfc": f"{_EU}::nitrogen_monoxide::no_conc::0",
        # ---- CAMS Europe Multi-Level: ALL pollutants × ALL heights ----
        # Dust (mineral particles)
        "dust_50m": f"{_EU}::dust::dust::50",
        "dust_100m": f"{_EU}::dust::dust::100",
        "dust_250m": f"{_EU}::dust::dust::250",
        "dust_500m": f"{_EU}::dust::dust::500",
        "dust_750m": f"{_EU}::dust::dust::750",
        "dust_1000m": f"{_EU}::dust::dust::1000",
        "dust_2000m": f"{_EU}::dust::dust::2000",
        "dust_3000m": f"{_EU}::dust::dust::3000",
        "dust_5000m": f"{_EU}::dust::dust::5000",
        # PM2.5 (fine particles)
        "pm2p5_50m": f"{_EU}::particulate_matter_2.5um::pm2p5_conc::50",
        "pm2p5_100m": f"{_EU}::particulate_matter_2.5um::pm2p5_conc::100",
        "pm2p5_250m": f"{_EU}::particulate_matter_2.5um::pm2p5_conc::250",
        "pm2p5_500m": f"{_EU}::particulate_matter_2.5um::pm2p5_conc::500",
        "pm2p5_750m": f"{_EU}::particulate_matter_2.5um::pm2p5_conc::750",
        "pm2p5_1000m": f"{_EU}::particulate_matter_2.5um::pm2p5_conc::1000",
        "pm2p5_2000m": f"{_EU}::particulate_matter_2.5um::pm2p5_conc::2000",
        "pm2p5_3000m": f"{_EU}::particulate_matter_2.5um::pm2p5_conc::3000",
        "pm2p5_5000m": f"{_EU}::particulate_matter_2.5um::pm2p5_conc::5000",
        # PM10 (coarse particles)
        "pm10_50m": f"{_EU}::particulate_matter_10um::pm10_conc::50",
        "pm10_250m": f"{_EU}::particulate_matter_10um::pm10_conc::250",
        "pm10_500m": f"{_EU}::particulate_matter_10um::pm10_conc::500",
        "pm10_1000m": f"{_EU}::particulate_matter_10um::pm10_conc::1000",
        "pm10_3000m": f"{_EU}::particulate_matter_10um::pm10_conc::3000",
        "pm10_5000m": f"{_EU}::particulate_matter_10um::pm10_conc::5000",
        # SO₂ (acid gas)
        "so2_50m": f"{_EU}::sulphur_dioxide::so2_conc::50",
        "so2_100m": f"{_EU}::sulphur_dioxide::so2_conc::100",
        "so2_250m": f"{_EU}::sulphur_dioxide::so2_conc::250",
        "so2_500m": f"{_EU}::sulphur_dioxide::so2_conc::500",
        "so2_1000m": f"{_EU}::sulphur_dioxide::so2_conc::1000",
        "so2_2000m": f"{_EU}::sulphur_dioxide::so2_conc::2000",
        "so2_3000m": f"{_EU}::sulphur_dioxide::so2_conc::3000",
        "so2_5000m": f"{_EU}::sulphur_dioxide::so2_conc::5000",
        # NO₂ (traffic/industry)
        "no2_50m": f"{_EU}::nitrogen_dioxide::no2_conc::50",
        "no2_100m": f"{_EU}::nitrogen_dioxide::no2_conc::100",
        "no2_250m": f"{_EU}::nitrogen_dioxide::no2_conc::250",
        "no2_500m": f"{_EU}::nitrogen_dioxide::no2_conc::500",
        "no2_1000m": f"{_EU}::nitrogen_dioxide::no2_conc::1000",
        "no2_3000m": f"{_EU}::nitrogen_dioxide::no2_conc::3000",
        "no2_5000m": f"{_EU}::nitrogen_dioxide::no2_conc::5000",
        # O₃ (ozone)
        "o3_250m": f"{_EU}::ozone::o3_conc::250",
        "o3_500m": f"{_EU}::ozone::o3_conc::500",
        "o3_1000m": f"{_EU}::ozone::o3_conc::1000",
        "o3_3000m": f"{_EU}::ozone::o3_conc::3000",
        "o3_5000m": f"{_EU}::ozone::o3_conc::5000",
        # CO (carbon monoxide)
        "co_250m": f"{_EU}::carbon_monoxide::co_conc::250",
        "co_1000m": f"{_EU}::carbon_monoxide::co_conc::1000",
        "co_5000m": f"{_EU}::carbon_monoxide::co_conc::5000",
        # ---- CAMS Global (column/AOD, 0.4° grid) ----
        "aod550": f"{_GLOBAL}::total_aerosol_optical_depth_550nm::aod550::",
        "duaod550": f"{_GLOBAL}::dust_aerosol_optical_depth_550nm::duaod550::",
        "omaod550": f"{_GLOBAL}::organic_matter_aerosol_optical_depth_550nm::omaod550::",
        "bcaod550": f"{_GLOBAL}::black_carbon_aerosol_optical_depth_550nm::bcaod550::",
        "ssaod550": f"{_GLOBAL}::sea_salt_aerosol_optical_depth_550nm::ssaod550::",
        "suaod550": f"{_GLOBAL}::sulphate_aerosol_optical_depth_550nm::suaod550::",
        "tcco": f"{_GLOBAL}::total_column_carbon_monoxide::tcco::",
        "tcno2": f"{_GLOBAL}::total_column_nitrogen_dioxide::tcno2::",
        "tco3": f"{_GLOBAL}::total_column_ozone::tco3::",
        "tcso2": f"{_GLOBAL}::total_column_sulphur_dioxide::tcso2::",
        "gtco3": f"{_GLOBAL}::gems_total_column_ozone::gtco3::",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in CAMS vocabulary."""
        cams_key = cls.VOCAB[val]

        def mod(x: np.array) -> np.array:
            """Modify name (if necessary)."""
            return x

        return cams_key, mod
