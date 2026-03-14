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
