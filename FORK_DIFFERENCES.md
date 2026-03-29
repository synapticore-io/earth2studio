# Fork Differences: synapticore-io/earth2studio

Technical reference of differences from [NVIDIA/earth2studio](https://github.com/NVIDIA/earth2studio).

---

## 1. DataSources

### CAMS (`earth2studio/data/cams.py`) — **new**

Two new DataSource classes for Copernicus Atmosphere Monitoring Service data via CDS API:

| Class | Type | Grid | Coverage | Variables |
|-------|------|------|----------|-----------|
| `CAMS` | `DataSource` (Analysis) | 0.1° | Europe | dust, SO₂, PM2.5, PM10, NO₂, O₃, CO, NH₃, NO |
| `CAMS_FX` | `ForecastSource` (Forecast) | 0.1° EU / 0.4° Global | EU + Global | EU surface + Global AOD/Column |

Registered in `earth2studio/data/__init__.py`.

### Upstream DataSources

No changes to existing DataSources (GFS, HRRR, CDS, ARCO, IFS, etc.).

---

## 2. Lexicon

### CAMSLexicon (`earth2studio/lexicon/cams.py`) — **new**

Variable mapping following Earth2Studio convention (`<dataset>::<api_variable>::<netcdf_key>::<level>`).

**EU Surface (level 0, 0.1° grid):** `dust`, `so2sfc`, `pm2p5`, `pm10`, `no2sfc`, `o3sfc`, `cosfc`, `nh3sfc`, `nosfc`

**EU Multi-Level (50m–5000m):** all pollutants × 10 height levels (0, 50, 100, 250, 500, 750, 1000, 2000, 3000, 5000m). Examples: `dust_500m`, `pm2p5_1000m`, `so2_3000m`, `no2_250m`, `o3_5000m`, `co_1000m`

**Global Column/AOD (0.4° grid):** `aod550`, `duaod550`, `omaod550`, `bcaod550`, `ssaod550`, `suaod550`, `tcco`, `tcno2`, `tco3`, `tcso2`, `gtco3`

Registered in `earth2studio/lexicon/__init__.py`.

### Upstream Lexicon

No changes to existing Lexicon entries.

---

## 3. Export Tooling

### `scripts/export_exr_sequence.py` — **new**

CAMS → EXR/PNG image sequence exporter for DCC and realtime engines (e.g. Unreal Engine, Blender, TouchDesigner).

- Packs atmospheric variables into RGBA channels (R=Dust, G=PM2.5, B=SO₂, A=Dust@5km)
- Optional: 3D volume export — multiple species × height levels as separate layers
- Output: one frame per hour, normalized 0–1 float32 (EXR, 16-bit PNG fallback, NumPy)
- Grid metadata (`grid_meta.npz`) for georeferenced post-processing

---

## 4. Docker

Uses upstream's `serve/Dockerfile` (NVIDIA PyTorch base image with GPU support). Fork changes:

- `pip` → `uv` migration (no redundant pip upgrade, no curl uv install)
- `apt` cache cleanup in same RUN layer
- CAMS dependencies included via `[data]` extra in `pyproject.toml`

---

## 5. Fork Documentation

Files that only exist in this fork:

| File | Content |
|------|---------|
| `ROADMAP.md` | Fork vision, sync strategy, phase plan |
| `FORK_GUIDE.md` | Maintenance guide (upstream sync, branching, releases) |
| `FORK_DIFFERENCES.md` | This document |
| `LOCAL_DEPLOYMENT.md` | Offline/air-gapped deployment guide |
| `GPU_OPTIMIZATION.md` | Consumer GPU tips |
| `KNOWN_ISSUES.md` | Known problems + workarounds |

---

## 6. Versioning & Sync

- **Schema:** upstream tracking — `v0.12.0-fork.1`, `v0.12.0-fork.2`, etc.
- **Sync frequency:** monthly or on upstream major releases
- **Merge strategy:** full merge from `upstream/main`, manual conflict resolution in fork-specific modules
- **Current status:** based on upstream `v0.12.0`

---

## 7. What Has NOT Changed

- Prognostic Models (`earth2studio/models/px/`)
- Diagnostic Models (`earth2studio/models/dx/`)
- IO Backends (`earth2studio/io/`)
- Perturbation Methods (`earth2studio/perturbation/`)
- Statistics (`earth2studio/statistics/`)
- Built-in Workflows (`earth2studio/run/`)
- Existing DataSources and Lexicon entries
- CI/CD Workflows (`.github/workflows/` — unchanged from upstream)
- Tests (`test/`) — unchanged from upstream, CAMS tests pending

---

**Last Updated:** 2026-03-12
