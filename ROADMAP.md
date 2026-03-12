# Fork Roadmap

**Repository:** synapticore-io/earth2studio
**Fork of:** [NVIDIA/earth2studio](https://github.com/NVIDIA/earth2studio)
**Last Updated:** 2026-03-11

---

## Mission

Extend Earth2Studio into a focused toolkit for **CAMS/atmospheric composition** and **data-driven weather/climate visualization**.

Specifically:
1. **CAMS/Atmospheric Composition** — integrate ECMWF/CAMS data (dust, aerosols, air quality) as first-class DataSources
2. **Reproducible Workflows** — stable pipelines for ECMWF/CAMS & other reanalysis/forecast data
3. **Close to upstream** — keep core in sync with NVIDIA/earth2studio, fork-specific features (DataSources, Lexicon, Docker/CI) developed separately and well documented

### Key Differentiators

1. **CAMS Integration** — `CAMS` / `CAMS_FX` DataSources for EU surface (0.1°) and global column/AOD data (0.4°)
2. **Extended Lexicon** — atmospheric composition variables (dust, SO₂, PM2.5, PM10, NO₂, O₃, CO, NH₃, AOD550, etc.)
3. **Visualization Pipelines** — data-driven workflows for weather/climate visualization (export to NumPy, Zarr, EXR for external tools)

---

## Upstream Sync Strategy

- **Sync frequency:** monthly (or on important upstream releases)
- **Merge strategy:** full merge from `upstream/main`, manual conflict resolution in fork-specific modules
- **Versioning:** upstream tracking (`v0.12.0-fork.1`, `v0.12.0-fork.2`, etc.)

### Current Status

- **Upstream version:** v0.12.0
- **Last sync:** 2026-01 (fork creation)
- **Next planned sync:** on upstream v0.13.0

---

## Completed

- [x] CAMS DataSource (`earth2studio/data/cams.py`) — `CAMS` (EU surface analysis) and `CAMS_FX` (EU + Global forecasts)
- [x] CAMS Lexicon (`earth2studio/lexicon/cams.py`) — variable mapping for EU surface and global column variables
- [x] Image sequence exporter (`scripts/export_exr_sequence.py`) — CAMS → EXR/PNG for DCC/realtime engines
- [x] Docker pipeline (`docker/`) — reproducible CAMS export container
- [x] Fork documentation (FORK_GUIDE.md, FORK_DIFFERENCES.md, LOCAL_DEPLOYMENT.md, GPU_OPTIMIZATION.md, KNOWN_ISSUES.md)

---

## Short-Term (next 3 months)

### Phase 1: Foundation

- [ ] Enable GitHub Issues
- [ ] Set repository topics (`weather-forecasting`, `cams`, `atmospheric-composition`, `deep-learning`, `pytorch`)
- [ ] Update repository description
- [ ] Branch protection for `main`
- [ ] Set up upstream remote + first manual sync
- [ ] Create issue labels (`upstream-tracking`, `fork-specific`, `cams`, `datasource`, `lexicon`)
- [ ] CI: set up `upstream-sync-check.yml` workflow (weekly check)

### Phase 2: Harden CAMS Integration

- [ ] Write CAMS DataSource tests (unit + integration)
- [ ] Validate CAMS Lexicon — check all variables against CDS API catalog
- [ ] Error handling for CDS API rate limits and timeouts
- [ ] CAMS example notebook: EU dust + PM2.5 visualization
- [ ] CAMS example notebook: global AOD550 forecast
- [ ] Documentation: CAMS setup guide (CDS API key, `.cdsapirc`, variable reference)

### Phase 3: Evaluate Additional Data Sources

- [ ] Evaluate CAMS Global Reanalysis (EAC4) as DataSource
- [ ] Evaluate CAMS Global Forecast (higher resolution)
- [ ] Add ERA5 atmospheric composition variables to Lexicon
- [ ] Check Copernicus ADS (Atmosphere Data Store) API migration (CDS → ADS transition)

---

## Medium-Term (3–6 months)

### Visualization Pipelines

- [ ] Export utilities: Zarr → GeoTIFF, NetCDF → EXR for external rendering tools
- [ ] Standardized colormaps for atmospheric composition variables
- [ ] Multi-scale pipeline example: CAMS EU surface + global AOD comparison
- [ ] Time-series visualization for forecast validation

### Workflow Stabilization

- [ ] Docker setup for reproducible CAMS pipelines
- [ ] CI/CD: automated tests against live CDS API (scheduled, not on every push)
- [ ] Cache strategy for CDS data (local cache, size limits)
- [ ] Reproducibility: pinned dependencies, lockfile, deterministic outputs

### Upstream Contributions

- [ ] Prepare CAMS DataSource as PR for NVIDIA/earth2studio (once stable)
- [ ] Contribute bug fixes from fork back to upstream

---

## Long-Term (6–12+ months)

### Extended Atmospheric Composition Features

- [ ] Combine CAMS + prognostic models (e.g. dust forecast + FCN weather overlay)
- [ ] Diagnostic models for Air Quality Index (AQI) calculation
- [ ] Ensemble workflows for atmospheric composition uncertainty

### Ecosystem Integration

- [ ] Integration with DCC/realtime engines (via EXR/NumPy export)
- [ ] DuckDB/Parquet export for analytical workflows
- [ ] REST API for CAMS queries (optional, if demand exists)

---

## Fork-Specific Files

| File | Status | Description |
|------|--------|-------------|
| `earth2studio/data/cams.py` | Implemented | CAMS + CAMS_FX DataSources |
| `earth2studio/lexicon/cams.py` | Implemented | CAMS variable mapping |
| `scripts/export_exr_sequence.py` | Implemented | Image sequence exporter |
| `docker/` | Implemented | Export pipeline container |
| `ROADMAP.md` | Current | This document |
| `FORK_GUIDE.md` | Present | Fork maintenance guide |
| `FORK_DIFFERENCES.md` | Present | Detailed differences from upstream |

---

**Next review:** 2026-04-11
