# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0
#
# Earth2Studio CAMS → EXR/PNG Image Sequence Exporter
# for DCC and realtime engines (Unreal Engine, Blender, TouchDesigner, etc.)
#
# Packs atmospheric variables into RGBA channels:
#   R = Dust Surface [µg/m³]       (CAMS EU, Level 0)
#   G = PM2.5 Surface [µg/m³]      (CAMS EU, Level 0)
#   B = SO₂ Surface [µg/m³]        (CAMS EU, Level 0)
#   A = Dust 5000m [µg/m³]         (CAMS EU, Level 5000 — proxy for column/height transport)
#
# Optional: Full 3D volume export with all 10 vertical levels.
# All from the same CAMS EU dataset (0.1° grid) — no resampling needed.
#
# Output: one frame per hour, normalized 0-1 float32 (EXR, 16-bit PNG fallback).
#
# Usage:
#   uv run python scripts/export_exr_sequence.py
#   uv run python scripts/export_exr_sequence.py --days 3 --start 2025-06-01

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from earth2studio.data.cams import CAMS


# ── Normalization ranges (µg/m³ for all EU surface + height variables) ──
RANGES = {
    "dust": 600.0,
    "pm2p5": 100.0,
    "so2sfc": 80.0,
    "dust_5000m": 600.0,
}

# 3D Volume mode: all particle species at multiple heights
VOLUME_SPECIES = {
    "dust": {
        "levels": ["dust", "dust_50m", "dust_250m", "dust_500m",
                   "dust_1000m", "dust_2000m", "dust_3000m", "dust_5000m"],
        "range": 600.0,
    },
    "pm2p5": {
        "levels": ["pm2p5", "pm2p5_50m", "pm2p5_250m", "pm2p5_500m",
                   "pm2p5_1000m", "pm2p5_2000m", "pm2p5_3000m", "pm2p5_5000m"],
        "range": 100.0,
    },
    "so2": {
        "levels": ["so2sfc", "so2_50m", "so2_250m", "so2_500m",
                   "so2_1000m", "so2_2000m", "so2_3000m", "so2_5000m"],
        "range": 80.0,
    },
    "no2": {
        "levels": ["no2sfc", "no2_50m", "no2_250m", "no2_500m",
                   "no2_1000m", "no2_3000m", "no2_5000m"],
        "range": 80.0,
    },
    "o3": {
        "levels": ["o3sfc", "o3_250m", "o3_500m",
                   "o3_1000m", "o3_3000m", "o3_5000m"],
        "range": 200.0,
    },
}

# Height in meters for each level index
LEVEL_HEIGHTS_M = [0, 50, 250, 500, 1000, 2000, 3000, 5000]

OUTPUT_DIR = Path(os.environ.get("E2S_OUTPUT_DIR", "outputs/exr_sequence"))


def normalize(data: np.ndarray, vmax: float) -> np.ndarray:
    """Clamp and normalize to 0-1 range."""
    return np.clip(data / vmax, 0.0, 1.0).astype(np.float32)


def save_exr(path: Path, rgba: np.ndarray) -> None:
    """Save float32 RGBA array as EXR. Falls back to 16-bit PNG."""
    try:
        import OpenEXR
        import Imath

        h, w, c = rgba.shape
        header = OpenEXR.Header(w, h)
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header["channels"] = {
            "R": half_chan,
            "G": half_chan,
            "B": half_chan,
            "A": half_chan,
        }
        out = OpenEXR.OutputFile(str(path.with_suffix(".exr")), header)
        out.writePixels(
            {
                "R": rgba[:, :, 0].tobytes(),
                "G": rgba[:, :, 1].tobytes(),
                "B": rgba[:, :, 2].tobytes(),
                "A": rgba[:, :, 3].tobytes(),
            }
        )
        out.close()
        return

    except ImportError:
        pass

    # Fallback: 16-bit PNG via imageio
    try:
        import imageio.v3 as iio

        png16 = (rgba * 65535).clip(0, 65535).astype(np.uint16)
        iio.imwrite(str(path.with_suffix(".png")), png16)
        return

    except ImportError:
        pass

    # Last resort: raw numpy
    np.save(str(path.with_suffix(".npy")), rgba)


def export_sequence(start_date: datetime, days: int, volume_3d: bool = False) -> None:
    """Export CAMS data as image sequence.

    Parameters
    ----------
    start_date : datetime
        UTC start date
    days : int
        Number of days to export
    volume_3d : bool
        If True, export all particle species at all height levels.
        Each species gets its own subfolder with one image per level per timestep.
        If False, export RGBA (R=Dust, G=PM2.5, B=SO₂, A=Dust@5000m).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds_eu = CAMS(cache=True)
    total_hours = days * 24

    if volume_3d:
        species_names = list(VOLUME_SPECIES.keys())
        # Collect all unique variable names across all species
        all_vars = []
        for spec in VOLUME_SPECIES.values():
            all_vars.extend(spec["levels"])
        all_vars = list(dict.fromkeys(all_vars))  # deduplicate preserving order

        mode_label = f"3D VOLUME — {len(species_names)} species × {len(LEVEL_HEIGHTS_M)} heights"
        print(f"\n{'='*60}")
        print(f"  Earth2Studio CAMS → Image Sequence Exporter")
        print(f"  Start:    {start_date.strftime('%Y-%m-%d')}")
        print(f"  Duration: {days} days ({total_hours} frames)")
        print(f"  Mode:     {mode_label}")
        print(f"  Species:  {', '.join(species_names)}")
        print(f"  Levels:   {LEVEL_HEIGHTS_M}")
        print(f"  Source:   CAMS EU 0.1° (all from same dataset)")
        print(f"  Output:   {OUTPUT_DIR.resolve()}")
        print(f"{'='*60}\n")

        # Create subdirs per species
        for name in species_names:
            (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

        lat_eu = None

        for hour_idx in range(total_hours):
            t = start_date + timedelta(hours=hour_idx)
            print(f"  [{hour_idx+1:4d}/{total_hours}] {t.strftime('%Y-%m-%d %H:%M')} UTC ...", end=" ")

            try:
                data = ds_eu([t], all_vars)
            except Exception as e:
                print(f"SKIP ({e})")
                continue

            if lat_eu is None:
                lat_eu = data.coords["lat"].values
                lon_eu = data.coords["lon"].values
                h, w = len(lat_eu), len(lon_eu)
                # Write comprehensive metadata
                meta = {
                    "lat": lat_eu,
                    "lon": lon_eu,
                    "lat_range": [lat_eu.min(), lat_eu.max()],
                    "lon_range": [lon_eu.min(), lon_eu.max()],
                    "level_heights_m": np.array(LEVEL_HEIGHTS_M),
                    "species": np.array(species_names),
                    "mode": "multi_species_3d",
                }
                for name, spec in VOLUME_SPECIES.items():
                    meta[f"range_{name}"] = spec["range"]
                np.savez(OUTPUT_DIR / "grid_meta.npz", **meta)
                print(f"\n  Grid: {h}×{w}, {len(species_names)} species × {len(LEVEL_HEIGHTS_M)} levels")
                print(f"  [{hour_idx+1:4d}/{total_hours}] {t.strftime('%Y-%m-%d %H:%M')} UTC ...", end=" ")

            h, w = len(lat_eu), len(lon_eu)
            stats = []

            for name, spec in VOLUME_SPECIES.items():
                for li, var in enumerate(spec["levels"]):
                    layer = normalize(
                        data.sel(variable=var).values.squeeze(),
                        spec["range"],
                    )
                    rgba = np.stack([layer, layer, layer, np.ones_like(layer)], axis=-1)
                    save_exr(OUTPUT_DIR / name / f"{name}_{hour_idx:04d}_L{li:02d}", rgba)
                peak = max(
                    data.sel(variable=spec["levels"][0]).values.max(),
                    data.sel(variable=spec["levels"][-1]).values.max(),
                )
                stats.append(f"{name}:{peak:.0f}")

            print(f"OK  [{' '.join(stats)}]")

        _write_import_readme(volume_3d=True)

    else:
        eu_vars = ["dust", "pm2p5", "so2sfc", "dust_5000m"]
        mode_label = "RGBA (R=Dust G=PM2.5 B=SO₂ A=Dust@5km)"

        print(f"\n{'='*60}")
        print(f"  Earth2Studio CAMS → Image Sequence Exporter")
        print(f"  Start:    {start_date.strftime('%Y-%m-%d')}")
        print(f"  Duration: {days} days ({total_hours} frames)")
        print(f"  Mode:     {mode_label}")
        print(f"  Source:   CAMS EU 0.1° (all levels from same dataset)")
        print(f"  Output:   {OUTPUT_DIR.resolve()}")
        print(f"{'='*60}\n")

        lat_eu = None

        for hour_idx in range(total_hours):
            t = start_date + timedelta(hours=hour_idx)
            print(f"  [{hour_idx+1:4d}/{total_hours}] {t.strftime('%Y-%m-%d %H:%M')} UTC ...", end=" ")

            try:
                eu_data = ds_eu([t], eu_vars)
            except Exception as e:
                print(f"SKIP ({e})")
                continue

            if lat_eu is None:
                lat_eu = eu_data.coords["lat"].values
                lon_eu = eu_data.coords["lon"].values
                h, w = len(lat_eu), len(lon_eu)
                np.savez(
                    OUTPUT_DIR / "grid_meta.npz",
                    lat=lat_eu, lon=lon_eu,
                    lat_range=[lat_eu.min(), lat_eu.max()],
                    lon_range=[lon_eu.min(), lon_eu.max()],
                    variables=eu_vars,
                    mode="rgba",
                    ranges=list(RANGES.values()),
                )
                print(f"\n  Grid: {h}×{w}")
                print(f"  [{hour_idx+1:4d}/{total_hours}] {t.strftime('%Y-%m-%d %H:%M')} UTC ...", end=" ")

            h, w = len(lat_eu), len(lon_eu)

            r = normalize(eu_data.sel(variable="dust").values.squeeze(), RANGES["dust"])
            g = normalize(eu_data.sel(variable="pm2p5").values.squeeze(), RANGES["pm2p5"])
            b = normalize(eu_data.sel(variable="so2sfc").values.squeeze(), RANGES["so2sfc"])
            a = normalize(eu_data.sel(variable="dust_5000m").values.squeeze(), RANGES["dust_5000m"])

            rgba = np.stack([r, g, b, a], axis=-1)
            save_exr(OUTPUT_DIR / f"atmos_{hour_idx:04d}", rgba)

            stats = f"D:{r.max():.2f} PM:{g.max():.2f} SO₂:{b.max():.2f} D5k:{a.max():.2f}"
            print(f"OK  [{stats}]")

        _write_import_readme(volume_3d=False)

    print(f"\n{'='*60}")
    print(f"  DONE — {total_hours} frames exported")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}\n")


def _write_import_readme(volume_3d: bool = False) -> None:
    """Write import instructions alongside the exported sequence."""
    readme = OUTPUT_DIR / "IMPORT.md"

    if volume_3d:
        channel_table = """\
## Multi-Species 3D Volume Export

Each particle species gets its own subfolder with height layers per timestep.

| Species | Folder | Levels |
|---------|--------|--------|
| Dust | `dust/` | 8 layers (0m–5000m) |
| PM2.5 | `pm2p5/` | 8 layers (0m–5000m) |
| SO₂ | `so2/` | 8 layers (0m–5000m) |
| NO₂ | `no2/` | 7 layers (0m–5000m) |
| O₃ | `o3/` | 6 layers (0m–5000m) |

### Height Layers
| Layer | Height |
|-------|--------|
| L00 | Surface (0m) |
| L01 | 50m |
| L02 | 250m |
| L03 | 500m |
| L04 | 1,000m |
| L05 | 2,000m |
| L06 | 3,000m |
| L07 | 5,000m |

Files: `{species}/{species}_XXXX_L00.exr` through `L07`"""
    else:
        channel_table = """\
## Channel Mapping (RGBA Mode)
| Channel | Variable | Unit | Normalization |
|---------|----------|------|---------------|
| R | Dust (surface) | µg/m³ | 0-600 → 0-1 |
| G | PM2.5 (surface) | µg/m³ | 0-100 → 0-1 |
| B | SO₂ (surface) | µg/m³ | 0-80 → 0-1 |
| A | Dust at 5000m | µg/m³ | 0-600 → 0-1 |"""

    readme.write_text(
        f"""\
# CAMS Image Sequence — Import Guide

All data from CAMS European Air Quality (0.1° grid, same dataset).
No resampling between different grids needed.

{channel_table}

## Grid Info
- Resolution: 0.1° × 0.1° (~10km) over Europe
- Source: CAMS EU 9-model ensemble + EEA station data
- Vertical levels available: 0m, 50m, 100m, 250m, 500m, 750m, 1km, 2km, 3km, 5km
- See `grid_meta.npz` for exact coordinates

## Denormalization
```
Dust_ugm3 = R * 600.0
PM25_ugm3 = G * 100.0
SO2_ugm3  = B * 80.0
Dust5km   = A * 600.0
```
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CAMS atmospheric data as EXR/PNG image sequence"
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days to export (default: 7)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: 2 days ago)",
    )
    parser.add_argument(
        "--volume-3d",
        action="store_true",
        help="Export all species at multiple height levels (3D volume)",
    )
    args = parser.parse_args()

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=2)

    export_sequence(start, args.days, volume_3d=args.volume_3d)


if __name__ == "__main__":
    main()
