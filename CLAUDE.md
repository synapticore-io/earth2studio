# CLAUDE.md

## Repository

Fork von [NVIDIA/earth2studio](https://github.com/NVIDIA/earth2studio). Upstream-Sync via `git fetch upstream && git merge upstream/main`.

**Fork-Strategie:** Upstream-Files NICHT modifizieren. Eigene Additions in separaten Files halten. Nur `__init__.py` Imports und `pyproject.toml` Extras minimal patchen.

Eigene Erweiterungen:
- CAMS DataSource/Lexicon (`earth2studio/data/cams.py`, `earth2studio/lexicon/cams.py`)
- Serve Workflows (`serve/server/example_workflows/`)
- Client SDK (`serve/client/earth2studio_client/`)
- Windows Build Scripts (`scripts/`)

## Project Rules

Coding Rules in `.cursor/rules/`:

| Rule file | Topic |
|---|---|
| `e2s-002-api-documentation.mdc` | Docstrings and public API docs |
| `e2s-004-data-sources.mdc` | Implementing `DataSource` classes |
| `e2s-008-lexicon-usage.mdc` | Variable lexicons and coordinate conventions |
| `e2s-009-prognostic-models.mdc` | Implementing prognostic models |
| `e2s-011-examples.mdc` | Writing gallery examples |
| `e2s-013-assimilation-models.mdc` | Implementing data assimilation models |

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

## Custom Commands

| Command | Action |
|---|---|
| `/format` | Auto-format code |
| `/lint` | Run all linters (ruff, mypy) |
| `/test` | Run tests for a specific tox environment |
| `/docs` | Build docs |

## Serve-Architektur

Der Container `earth2studio-serve` buendelt alles intern:
- **Redis** (in-container, nicht als separater Service)
- **Uvicorn** (4 Workers, Port 8000)
- **RQ Workers** (inference, result_zip, object_storage, finalize_metadata)
- **Cleanup Daemon**

Config: `serve/server/conf/config.yaml`. Custom Workflows via `WORKFLOW_DIR` env var.

## Key Gotchas

- **CAMS/CDS Credentials**: `~/.cdsapirc` muss existieren (URL + Key von CDS). Im Container via Volume-Mount.
- **Line Endings**: `.gitattributes` erzwingt LF fuer Shell-Scripts. Windows-Checkouts mit CRLF brechen den Container.
- **Zarr IO erwartet Tensors**: `io.write()` braucht `torch.Tensor`, nicht numpy arrays.
- **PyTorch SHMEM**: Container braucht `ipc: host` und `ulimits` (memlock, stack).
- **fetch_data -> map_coords**: `fetch_data()` liefert GFS-Koordinaten (721 lat), aber FCN erwartet 720. Bei custom Workflows nach `fetch_data` immer `map_coords(x, coords, model.input_coords())` aufrufen.

## Client-SDK

```python
from earth2studio_client import RemoteEarth2Workflow
workflow = RemoteEarth2Workflow("http://localhost:8000", workflow_name="cams_analysis")
ds = workflow(start_time=[datetime(2025, 6, 1)], preset="eu_surface").as_dataset()
```
