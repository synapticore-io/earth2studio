---
name: earth2studio
description: Build AI weather and climate inference pipelines with NVIDIA Earth2Studio. Use this skill whenever the user mentions Earth2Studio, earth2studio, weather AI models, climate forecasting with AI, FourCastNet, Atlas, CorrDiff, StormScope, SFNO, DLESyM, Pangu weather, GraphCast, Aurora, CAMS data source, atmospheric composition forecasting, air quality modeling, prognostic models, diagnostic models, ensemble weather forecasting, or any task involving AI-driven weather/climate simulation. Also trigger when the user wants to fetch weather data from ERA5, GFS, HRRR, IFS, CAMS, or ARCO for AI model inference, or when working with the synapticore-io/earth2studio fork. Trigger on phrases like "weather model", "climate simulation", "forecast pipeline", "atmospheric data", "dust forecasting", "air quality forecast", or "ensemble prediction".
---

# Earth2Studio — AI Weather & Climate Inference Toolkit

NVIDIA Earth2Studio is an open-source Python framework for building, deploying, and exploring AI weather/climate inference pipelines. It provides modular components (data sources, prognostic models, diagnostic models, perturbation methods, IO backends, statistics) that compose into flexible workflows.

**Repository:** https://github.com/NVIDIA/earth2studio
**Fork (with CAMS integration):** https://github.com/synapticore-io/earth2studio
**Docs:** https://nvidia.github.io/earth2studio

## Architecture Overview

Earth2Studio's core design: **Workflow = glue binding modular components.**

```
DataSource → [fetch_data] → Tensor + CoordSystem → PrognosticModel → DiagnosticModel → IOBackend
                                                         ↑
                                                   Perturbation (ensembles)
                                                         ↓
                                                    Statistics
```

Components communicate via `(x: torch.Tensor, coords: CoordSystem)` tuples. CoordSystem is an `OrderedDict[str, np.ndarray]` tracking dimensions (time, variable, lat, lon, etc.).

Components follow **Python protocols** (not inheritance) — implement the interface, plug into any workflow.

## Installation

```bash
# Recommended: uv
uv init --python=3.12
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git"
# For the synapticore fork with CAMS multi-level support:
# uv add "earth2studio @ git+https://github.com/synapticore-io/earth2studio.git"

# Models need extras:
uv add earth2studio --extra atlas        # Atlas (15-day medium range)
uv add earth2studio --extra fcn          # FourCastNet
uv add earth2studio --extra corrdiff     # CorrDiff (downscaling)
uv add earth2studio --extra data         # All data source deps (cdsapi, cfgrib, etc.)
uv add earth2studio --extra serve        # REST API server + client
```

**Hardware:** Most models require GPU with ≥40 GB VRAM (L40S, A6000, H100). Some lighter models work on consumer GPUs.

## Core Components

### 1. Data Sources (`earth2studio.data`)

Two interfaces:
- **DataSource**: `__call__(time, variable) → xr.DataArray`
- **ForecastSource**: `__call__(time, lead_time, variable) → xr.DataArray`

| Source | Dataset | Resolution | Type | Credentials |
|--------|---------|------------|------|-------------|
| `GFS` / `GFS_FX` | NOAA GFS | 0.25° global | Analysis / Forecast | Free |
| `HRRR` / `HRRR_FX` | NOAA HRRR | 3 km CONUS | Analysis / Forecast | Free |
| `CDS` | ERA5 via CDS API | 0.25° global | Reanalysis | CDS API Key |
| `ARCO` | ERA5 cloud-optimized | 0.25° global | Reanalysis (fast) | Free |
| `IFS` / `IFS_FX` | ECMWF IFS | 0.1° global | Analysis / Forecast | Free (OpenData) |
| `CAMS` | CAMS EU Air Quality | 0.1° Europe | Analysis | CDS API Key |
| `CAMS_FX` | CAMS EU + Global | 0.1°/0.4° | Forecast | CDS API Key |
| `GOES` | GOES satellite | varies | Observation | Free |
| `MRMS` | NOAA Radar | 1 km CONUS | Observation | Free |
| `ISD` | Surface stations | Point obs | Observation | Free |

**Generic data fetch pattern:**

```python
from earth2studio.data import GFS, fetch_data
from earth2studio.models.px import FCN

model = FCN.load_model(FCN.load_default_package())
ds = GFS()

# fetch_data returns (tensor, coords) ready for model input
x, coords = fetch_data(
    source=ds,
    time=["2025-01-01T00:00"],
    variable=model.input_coords()["variable"],
    lead_time=model.input_coords().get("lead_time"),
    device="cuda:0",
)
```

**CAMS Data Source (synapticore-io fork):**

```python
from earth2studio.data import CAMS, CAMS_FX
from datetime import datetime, timedelta, timezone

yesterday = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)

# EU surface analysis
ds = CAMS()
data = ds(yesterday, ["dust", "pm2p5", "so2sfc"])

# Forecast with lead times
fx = CAMS_FX()
data = fx(yesterday, [timedelta(hours=0), timedelta(hours=24)], ["dust", "pm2p5"])

# Global AOD (column-integrated, 0.4° grid)
data = CAMS_FX()(yesterday, [timedelta(0)], ["duaod550", "aod550"])
```

CAMS Lexicon — EU Surface (0.1°): `dust`, `so2sfc`, `pm2p5`, `pm10`, `no2sfc`, `o3sfc`, `cosfc`, `nh3sfc`, `nosfc`. Global Column/AOD (0.4°): `aod550`, `duaod550`, `omaod550`, `bcaod550`, `ssaod550`, `suaod550`, `tcco`, `tcno2`, `tco3`, `tcso2`, `gtco3`.

### 2. Prognostic Models (`earth2studio.models.px`)

Auto-regressive time integration models:

```python
from earth2studio.models.px import FCN

pkg = FCN.load_default_package()
model = FCN.load_model(pkg)

# Single step
x_out, coords_out = model(x, coords)

# Time-series (iterator — step 0 = input, step 1+ = forecasts)
for step, (x_step, coords_step) in enumerate(model.create_iterator(x, coords)):
    io.write(x_step, coords_step, "fields")
```

| Model | Architecture | Range | Extra |
|-------|-------------|-------|-------|
| `Atlas` | Latent diffusion transformer | 15 days | `atlas` |
| `FCN` / `FCN3` | FourCastNet (AFNO) | 10 days | `fcn` / `fcn3` |
| `SFNO` | Spherical FNO | 10+ days | `sfno` |
| `DLWP` | Deep Learning Weather Pred | 10 days | `dlwp` |
| `DLESyM` | Deep Learning Earth System | S2S (weeks) | `dlesym` |
| `Pangu` | Pangu-Weather (ONNX) | 7 days | `pangu` |
| `FuXi` | FuXi (ONNX) | 15 days | `fuxi` |
| `GraphCast` | Graph neural network | 10 days | `graphcast` |
| `Aurora` | Microsoft Aurora | 10 days | `aurora` |
| `AIFS` / `AIFSENS` | ECMWF AIFS | 10 days | `aifs` |
| `StormCast` | Regional storm model | Hours | `stormcast` |
| `StormScope` | Satellite/radar nowcast | Hours | `stormscope` |

### 3. Diagnostic Models (`earth2studio.models.dx`)

| Model | Purpose | Extra |
|-------|---------|-------|
| `CorrDiff` / `CorrDiffTaiwan` | Downscaling (25km→3km) | `corrdiff` |
| `PrecipitationAFNO` | Total precipitation | `precip-afno` |
| `SolarRadiationAFNO` | Solar irradiance | `solarradiation-afno` |
| `WindGustAFNO` | Wind gusts | `windgust-afno` |
| `ClimateNet` | Tropical cyclone detection | `climatenet` |
| `TCTrackerWuDuan` / `TCTrackerVitart` | Cyclone tracking | `cyclone` |
| `CBottleInfill` / `CBottleSR` | Data infilling / super-res | `cbottle` |
| `DerivedWS`, `DerivedRH`, `DerivedVPD` | Derived physics variables | (built-in) |

### 4. Perturbation Methods (`earth2studio.perturbation`)

```python
from earth2studio.perturbation import (
    SphericalGaussian,       # Isotropic noise on sphere
    CorrelatedSphericalGaussian,  # Matern covariance
    BredVector,              # Classical breeding vectors
    HemisphericCentredBredVector, # Regional variant
    Gaussian,                # Simple white noise
    LaggedEnsemble,          # Time-lagged (no noise)
    Zero,                    # Identity (for stochastic models)
)
```

### 5. IO Backends (`earth2studio.io`)

```python
from earth2studio.io import ZarrBackend, NetCDF4Backend, KVBackend, XarrayBackend

io = ZarrBackend()  # Recommended default (best datetime support)
io.add_array(total_coords, "fields")
io.write(x, coords, "fields")
ds = io["fields"]  # Access as xarray
```

**Important:** `io.write()` expects `torch.Tensor`, not numpy arrays. Data sources return `xr.DataArray` — wrap with `torch.from_numpy(da.values)` when writing raw data source output to IO.

### 6. Statistics (`earth2studio.statistics`)

In-pipeline evaluation: `mean`, `std`, `variance`, `lat_weight`, `rmse`, `mae`, `crps`, `acc`, `brier_score`, `fss`, `rank_histogram`.

## Built-in Workflows (`earth2studio.run`)

Three workflow functions that compose components:

```python
from earth2studio.run import deterministic, ensemble, diagnostic

# Deterministic forecast
io = deterministic(time, nsteps, prognostic, data, io)

# Ensemble forecast with perturbation
io = ensemble(time, nsteps, nensemble, prognostic, data, io, perturbation, batch_size=None)

# Prognostic + diagnostic chain (e.g. weather → precipitation)
io = diagnostic(time, nsteps, prognostic, diagnostic_model, data, io)
```

All three accept optional `output_coords` (CoordSystem) for subsetting output variables and `device` (torch.device).

## Serve: REST API & Docker

### Server Architecture

The `earth2studio-serve` Docker container bundles all services internally:
- **Redis** (in-container queue backend)
- **Uvicorn** (FastAPI, 4 workers, port 8000)
- **RQ Workers** (inference, result_zip, object_storage, finalize_metadata)
- **Cleanup Daemon** (TTL-based result expiry)

```bash
cd serve && docker compose up -d    # Start
curl http://localhost:8000/health    # Health check
curl http://localhost:8000/v1/workflows  # List workflows
curl http://localhost:8000/docs      # OpenAPI docs
```

### Writing Serve Workflows (`Earth2Workflow`)

Subclass `Earth2Workflow` for auto-REST-API generation. `__init__` runs once per worker (load models here). `__call__` parameters become the JSON API schema via Pydantic.

```python
from earth2studio.serve.server import Earth2Workflow, workflow_registry
from earth2studio.io import IOBackend
from earth2studio.models.px import FCN
from earth2studio.data import GFS
from earth2studio import run
from datetime import datetime

@workflow_registry.register
class MyWorkflow(Earth2Workflow):
    name = "my_workflow"
    description = "My custom forecast"

    def __init__(self):
        super().__init__()
        self.model = FCN.load_model(FCN.load_default_package())
        self.data = GFS()

    def __call__(
        self,
        io: IOBackend,  # Required — server provides the IO backend
        start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
        num_steps: int = 10,
    ) -> None:
        run.deterministic(start_time, num_steps, self.model, self.data, io)
```

**Rules:**
- `__call__` must have `io: IOBackend` parameter (server-provided)
- Other parameters must have type hints and be JSON-serializable (Pydantic-compatible)
- `datetime` and `timedelta` types are supported (including nested in lists)
- Place workflow files in `WORKFLOW_DIR` directory for auto-discovery
- Use `@workflow_registry.register` decorator

**REST API usage:**
```bash
# Submit
curl -X POST http://localhost:8000/v1/infer/my_workflow \
  -H "Content-Type: application/json" \
  -d '{"parameters": {"start_time": ["2024-01-01T00:00:00"], "num_steps": 10}}'

# Check status
curl http://localhost:8000/v1/infer/my_workflow/{execution_id}/status

# Get results
curl http://localhost:8000/v1/infer/my_workflow/{execution_id}/results
```

### Python Client SDK (`earth2studio-client` / `earth2studio_client`)

Two client interfaces for accessing the REST API from Python:

#### RemoteEarth2Workflow (high-level, recommended)

Seamless integration with Earth2Studio and xarray. Lazy Zarr download.

```python
from earth2studio_client import RemoteEarth2Workflow

workflow = RemoteEarth2Workflow(
    "http://localhost:8000",
    workflow_name="deterministic_earth2_workflow",
)

# Submit and get result as xarray Dataset (lazy — only downloads accessed data)
result = workflow(start_time=[datetime(2025, 8, 21, 6)], num_steps=10)
ds = result.as_dataset()
t2m = ds["t2m"].sel(lat=37.4, lon=-122.0, method="nearest")

# Optional: as_data_source requires `earth2studio` installed
# data_source = result.as_data_source()
# t2m = data_source(datetime(2025, 8, 22, 6), "t2m")
```

**Key patterns:**

```python
# Resume previous execution (save execution_id, access later)
result = RemoteEarth2WorkflowResult(workflow, "exec_1766159252_5f779460")
ds = result.as_dataset()  # blocks until results are available
```

#### Earth2StudioClient (low-level)

Direct API access and file management:

```python
from earth2studio_client import Earth2StudioClient, InferenceRequest

client = Earth2StudioClient("http://localhost:8000", workflow_name="my_workflow")
client.health_check()

# Synchronous (submit + wait)
request = InferenceRequest(parameters={"start_time": [datetime(2025, 8, 21, 6)]})
result = client.run_inference_sync(request)

# Download individual files
for f in result.output_files:
    content = client.download_result(result, f.path)

# Async (submit, poll, retrieve)
response = client.submit_inference_request(request)
status = client.get_request_status(response.execution_id)
result = client.wait_for_completion(response.execution_id, poll_interval=5.0)
```

**Authentication:** Both clients accept `token="your-api-token"` for Bearer auth.

## Lexicon System

Unified naming scheme (ECMWF param-db based). Format: `dataset::api_variable::netcdf_key::level`.

Key weather vars: `t2m`, `u10m`/`v10m`, `z500`, `tcwv`, `sp`, `msl`, `tp`.

CAMS vars: `dust`, `so2sfc`, `pm2p5`, `pm10`, `no2sfc`, `aod550`, `duaod550`.

## Environment Configuration

```bash
export EARTH2STUDIO_CACHE=~/.cache/earth2studio
export EARTH2STUDIO_API_URL=http://localhost:8000  # Client SDK default
```

CDS API credentials (`~/.cdsapirc`):
```
url: https://cds.climate.copernicus.eu/api
key: YOUR_UID:YOUR_API_KEY
```

## Developer Notes

- **Package manager:** uv
- **Testing:** `pytest test/ -v --no-testmon -k "not slow"`
- **Linting:** `ruff check earth2studio/` / `ruff format earth2studio/`
- **Code style:** Google-style docstrings, SPDX headers on all files
