# CLAUDE.md

## Repository

Fork von [NVIDIA/earth2studio](https://github.com/NVIDIA/earth2studio). Upstream-Sync via `git fetch upstream && git merge upstream/main`.

Eigene Erweiterungen: CAMS DataSource/Lexicon (`earth2studio/data/cams.py`, `earth2studio/lexicon/cams.py`), CAMS Serve-Workflows (`serve/server/example_workflows/cams_workflow.py`).

## Commands

```bash
# Dev-Install (alle Extras)
uv pip install -e ".[all]"

# Nur bestimmte Modelle
uv pip install -e ".[data,serve,fcn3,stormcast]"

# Tests
pytest test/                       # Alle Tests
pytest test/ -m "not slow"         # Ohne Slow-Tests
pytest test/data/test_cams.py      # Spezifisch

# Lint
ruff check earth2studio/
ruff format earth2studio/

# Serve (Docker)
cd serve && docker compose up -d   # API auf Port 8000
docker compose logs -f             # Logs
```

## Serve-Architektur

Der Container `earth2studio-serve` bündelt alles intern:
- **Redis** (in-container, nicht als separater Service)
- **Uvicorn** (4 Workers, Port 8000)
- **RQ Workers** (inference, result_zip, object_storage, finalize_metadata)
- **Cleanup Daemon**

Startup-Reihenfolge: Redis → API + Workers → Health-Check → Warmup.

Config: `serve/server/conf/config.yaml`. Custom Workflows via `WORKFLOW_DIR` env var.

## Key Gotchas

- **CAMS/CDS Credentials**: `~/.cdsapirc` muss existieren (URL + Key von [CDS](https://cds.climate.copernicus.eu/)). Im Container via Volume-Mount.
- **Line Endings**: `.gitattributes` erzwingt LF für Shell-Scripts. Windows-Checkouts mit CRLF brechen den Container (`cannot execute`).
- **Zarr IO erwartet Tensors**: `io.write()` braucht `torch.Tensor`, nicht numpy arrays. Data Sources liefern `xr.DataArray` → `torch.from_numpy(da.values)`.
- **Erststart langsam**: ~3-5 min weil 4 Uvicorn Workers + RQ Worker parallel alle ML-Module importieren (PyTorch, CUDA, earth2studio models).
- **PyTorch SHMEM**: Container braucht `ipc: host` und `ulimits` (memlock, stack), sonst PyTorch-Fehler.
- **SFNO/CuPy CUDA 13**: Container (nvcr.io/nvidia/pytorch:26.01) hat CUDA 13.1, CuPy sucht `libnvrtc.so.12`. SFNO-basierte Workflows (Cyclone Tracking) funktionieren nicht bis CuPy aktualisiert wird.
- **FCN3/StormCast OOM**: `stormcast_fcn3_workflow` braucht mehr VRAM als eine einzelne Consumer-GPU bietet. Braucht ≥40 GB (A6000/H100).
- **Cyclone Tracker Variablen**: TCTrackerWuDuan braucht `u850, v850, u10m, v10m, msl` (721 lat) — nur SFNO liefert das. FCN hat 720 lat, DLWP fehlen Wind-Vars.
- **fetch_data → map_coords**: `fetch_data()` liefert GFS-Koordinaten (721 lat), aber FCN erwartet 720. Bei custom Workflows nach `fetch_data` immer `map_coords(x, coords, model.input_coords())` aufrufen.

## Deployed Serve-Workflows

| Workflow | Modell | Getestet |
|----------|--------|----------|
| `cams_analysis` | CAMS EU (0.1°, 6 Vars) | ja |
| `cams_forecast` | CAMS_FX EU+Global | ja |
| `ensemble_forecast` | FCN + SphericalGaussian | ja |
| `precipitation_forecast` | FCN + PrecipitationAFNO | ja |
| `corrdiff_taiwan` | CorrDiffTaiwan (3km) | ja |
| `deterministic_earth2_workflow` | FCN/DLWP | ja |
| `deterministic_fcn_workflow` | FCN | ja |
| `deterministic_workflow` | FCN/DLWP + Plot | ja |
| `diagnostic_workflow` | FCN/DLWP + PrecipAFNO + Plot | ja |
| `stormcast_fcn3_workflow` | FCN3 + StormCast | OOM |
| `example_user_workflow` | Warmup/Template | ja |

## Client-SDK

```python
from earth2studio_client import RemoteEarth2Workflow
workflow = RemoteEarth2Workflow("http://localhost:8000", workflow_name="cams_analysis")
ds = workflow(start_time=[datetime(2025, 6, 1)], preset="eu_surface").as_dataset()
```

## Extras → Modelle

| Extra | Modell/Feature |
|-------|---------------|
| `data` | CDS, CAMS, ARCO, GFS, HRRR Data Sources |
| `serve` | REST API + RQ Workers |
| `fcn3` | FourCastNet v3 |
| `stormcast` | StormCast |
| `dlwp` | DLWP-CS |
| `sfno` | SFNO |
| `pangu` | Pangu-Weather |
| `atlas` | Atlas |
| `corrdiff` | CorrDiff |
| `cbottle` | CBottle |
| `all` | Alles |
