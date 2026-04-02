# Fork Differences: synapticore-io/earth2studio

Technical reference of differences from [NVIDIA/earth2studio](https://github.com/NVIDIA/earth2studio).

Based on upstream `0.14.0a0`.

---

## Upstream Contributions

The following were developed in this fork and merged upstream:

| PR | Feature | Status |
|----|---------|--------|
| [#790](https://github.com/NVIDIA/earth2studio/pull/790) | CAMS Global FX DataSource + CAMSLexicon | **Merged** (2026-04-02) |

---

## 1. Serve Workflows

Custom inference workflows in `serve/server/example_workflows/`:

| Workflow | Description |
|----------|-------------|
| `cams_workflow.py` | CAMS atmospheric composition analysis |
| `corrdiff_workflow.py` | CorrDiff downscaling |
| `cyclone_tracking_workflow.py` | Cyclone tracking with SFNO |
| `precipitation_workflow.py` | Ensemble precipitation forecast |
| `sfno_forecast_workflow.py` | SFNO deterministic forecast |
| `stormcast_sda_workflow.py` | StormCast sequential data assimilation |
| `stormscope_workflow.py` | StormScope nowcasting |
| `temporal_interpolation_workflow.py` | Temporal interpolation between forecast steps |
| + others | Aurora, CBottle, DLESyM, HealDA, ensemble, diagnostic |

---

## 2. Client SDK

Extended client SDK in `serve/client/` — adds `earth2studio_client` package with:

- `RemoteEarth2Workflow` for programmatic workflow execution
- Zarr chunk download support for large results
- Examples for forecast, diagnostic, and downscaling use cases

Registered as uv workspace member.

---

## 3. Docker / Deployment

- `serve/docker-compose.yml` — single-container stack (Redis, Uvicorn, RQ Workers in-container)
- `serve/Dockerfile` — minor changes: `pip` → `uv`, apt cache cleanup

---

## 4. Remote Examples

Client-side examples in `examples/remote/` demonstrating workflow execution against the serve API.

---

## 5. Windows Build Scripts

Scripts in `scripts/` for building GRIB dependencies on Windows:

| Script | Purpose |
|--------|---------|
| `build-eccodes-windows.ps1` | Build ecCodes from source |
| `build-pygrib-wheels.ps1` | Build pygrib wheel |
| `setup-windows-grib.ps1` | One-shot GRIB setup |

---

## 6. What Has NOT Changed

- Prognostic Models (`earth2studio/models/px/`)
- Diagnostic Models (`earth2studio/models/dx/`)
- IO Backends (`earth2studio/io/`)
- Perturbation Methods (`earth2studio/perturbation/`)
- Statistics (`earth2studio/statistics/`)
- Built-in Workflows (`earth2studio/run/`)
- Existing DataSources and Lexicon entries
- Tests (`test/`)

---

## 7. Sync Strategy

- **Upstream remote:** `upstream` → `https://github.com/NVIDIA/earth2studio.git`
- **Merge strategy:** `git fetch upstream main && git merge upstream/main`, manual conflict resolution
- **Fork-Regel:** Upstream-Files nicht modifizieren. Eigene Additions in separaten Files. Nur `__init__.py` Imports und `pyproject.toml` Extras minimal patchen.

---

**Last Updated:** 2026-04-02
