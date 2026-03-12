# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0
#
# Download and verify ONNX weather models from Earth2Studio.
# Outputs standalone .onnx files ready for any ONNX runtime.
#
# Usage:
#   uv run python scripts/download_onnx_models.py
#   uv run python scripts/download_onnx_models.py --model pangu --inspect
#   uv run python scripts/download_onnx_models.py --model all --output-dir ./onnx_models

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


def download_pangu(output_dir: Path, inspect: bool = False) -> list[Path]:
    """Download Pangu Weather ONNX checkpoints.

    3 models: 3h, 6h, 24h step sizes.
    Input:  pressure [5, 13, 721, 1440] + surface [4, 721, 1440] = 69 variables
    Output: same shape, next timestep.

    Variables (69 total):
      Pressure levels (13): 1000,925,850,700,600,500,400,300,250,200,150,100,50 hPa
      Per level (5): geopotential(z), humidity(q), temperature(t), wind_u(u), wind_v(v)
      Surface (4): mean_sea_level_pressure(msl), 10m_wind_u(u10m), 10m_wind_v(v10m), 2m_temperature(t2m)

    Grid: 0.25° global, 721×1440 (lat×lon), lat: 90°N → 90°S
    """
    from earth2studio.models.auto import Package

    print("\n  Downloading Pangu Weather ONNX models...")
    pkg = Package(
        "hf://NickGeneva/earth_ai/pangu",
        cache_options={
            "cache_storage": Package.default_cache("pangu"),
            "same_names": True,
        },
    )

    files = {
        "pangu_weather_3.onnx": "3-hour step",
        "pangu_weather_6.onnx": "6-hour step",
        "pangu_weather_24.onnx": "24-hour step",
    }

    outputs = []
    for fname, desc in files.items():
        print(f"    Resolving {fname} ({desc})...", end=" ", flush=True)
        src = pkg.resolve(fname)
        dst = output_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"OK  [{size_mb:.0f} MB]")
        outputs.append(dst)

    if inspect:
        _inspect_onnx(outputs[0], "Pangu 3h")

    return outputs


def download_fuxi(output_dir: Path, inspect: bool = False) -> list[Path]:
    """Download FuXi ONNX checkpoints.

    Input/Output: similar to Pangu, 0.25° global grid.
    """
    from earth2studio.models.auto import Package

    print("\n  Downloading FuXi ONNX models...")
    pkg = Package(
        "hf://NickGeneva/earth_ai/fuxi",
        cache_options={
            "cache_storage": Package.default_cache("fuxi"),
            "same_names": True,
        },
    )

    outputs = []
    for fname in ["fuxi.onnx"]:
        print(f"    Resolving {fname}...", end=" ", flush=True)
        src = pkg.resolve(fname)
        dst = output_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"OK  [{size_mb:.0f} MB]")
        outputs.append(dst)

    if inspect:
        _inspect_onnx(outputs[0], "FuXi")

    return outputs


def download_fengwu(output_dir: Path, inspect: bool = False) -> list[Path]:
    """Download FengWu ONNX checkpoint.

    Input/Output: similar to Pangu, 0.25° global grid.
    """
    from earth2studio.models.auto import Package

    print("\n  Downloading FengWu ONNX models...")
    pkg = Package(
        "hf://NickGeneva/earth_ai/fengwu",
        cache_options={
            "cache_storage": Package.default_cache("fengwu"),
            "same_names": True,
        },
    )

    outputs = []
    for fname in ["fengwu.onnx"]:
        print(f"    Resolving {fname}...", end=" ", flush=True)
        src = pkg.resolve(fname)
        dst = output_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"OK  [{size_mb:.0f} MB]")
        outputs.append(dst)

    if inspect:
        _inspect_onnx(outputs[0], "FengWu")

    return outputs


def _inspect_onnx(path: Path, name: str) -> None:
    """Print ONNX model input/output specs."""
    try:
        import onnx
    except ImportError:
        print("    (onnx package not installed — skipping inspection)")
        return

    print(f"\n  ─── {name} Model Inspection ───")
    model = onnx.load(str(path), load_external_data=False)

    print(f"  IR version:   {model.ir_version}")
    print(f"  Producer:     {model.producer_name}")
    print(f"  Opset:        {[f'{o.domain or \"ai.onnx\"}:{o.version}' for o in model.opset_import]}")

    print(f"\n  Inputs ({len(model.graph.input)}):")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"    {inp.name:25s} {dtype:8s} {shape}")

    print(f"\n  Outputs ({len(model.graph.output)}):")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        print(f"    {out.name:25s} {dtype:8s} {shape}")

    # Count ops
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    top_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top operators ({len(model.graph.node)} total):")
    for op, count in top_ops:
        print(f"    {op:25s} ×{count}")

    del model


def write_model_card(output_dir: Path, models: list[str]) -> None:
    """Write MODEL_CARD.md with specs for each downloaded model."""
    card = output_dir / "MODEL_CARD.md"
    card.write_text(
        f"""\
# ONNX Weather Models

Downloaded from Earth2Studio / HuggingFace model hub.

## Models Present
{chr(10).join(f'- {m}' for m in models)}

## Common Specs
- Grid: 0.25° global equirectangular (721 lat × 1440 lon)
- Lat: 90°N → 90°S (inclusive), Lon: 0° → 359.75°E
- Precision: float32
- Total variables: 69

## Variable Layout

### Pressure fields: [5, 13, 721, 1440]
5 variables × 13 pressure levels:
- z  (geopotential)      at 1000,925,850,700,600,500,400,300,250,200,150,100,50 hPa
- q  (specific humidity)  at same levels
- t  (temperature)        at same levels
- u  (u-wind component)   at same levels
- v  (v-wind component)   at same levels

### Surface fields: [4, 721, 1440]
- msl  (mean sea level pressure, Pa)
- u10m (10m u-wind, m/s)
- v10m (10m v-wind, m/s)
- t2m  (2m temperature, K)

## ONNX Input/Output Binding

### Pangu Weather
| Name | Direction | Shape | Description |
|------|-----------|-------|-------------|
| input | in | [5, 13, 721, 1440] | Pressure fields |
| input_surface | in | [4, 721, 1440] | Surface fields |
| output | out | [5, 13, 721, 1440] | Predicted pressure |
| output_surface | out | [4, 721, 1440] | Predicted surface |

### Autoregressive Loop
```
state_t = initial_conditions  (from ERA5, GFS, or CAMS)
for step in range(N):
    state_t+1 = model(state_t)
    render(state_t+1)
    state_t = state_t+1
```

## Wind Field for Particle Advection
For driving particle systems, extract u/v wind components:
- Surface wind: u10m (index 66), v10m (index 67) from output_surface
- Wind at 850hPa: u[2] and v[2] from pressure output (typical boundary layer)
- Wind at 500hPa: u[5] and v[5] (mid-troposphere, dust transport level)
- Wind at 250hPa: u[8] and v[8] (jet stream level)

## License
Check individual model licenses before commercial use:
- Pangu: Apache 2.0 (via ECMWF)
- FuXi: Check original repository
- FengWu: Check original repository
""",
        encoding="utf-8",
    )
    print(f"\n  Model card written: {card}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ONNX weather models from Earth2Studio"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pangu",
        choices=["pangu", "fuxi", "fengwu", "all"],
        help="Which model to download (default: pangu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/onnx_models",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print model input/output specs after download",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Earth2Studio ONNX Model Downloader")
    print(f"  Output: {output_dir.resolve()}")
    print(f"{'='*60}")

    downloaded = []

    if args.model in ("pangu", "all"):
        download_pangu(output_dir, args.inspect)
        downloaded.append("pangu")

    if args.model in ("fuxi", "all"):
        download_fuxi(output_dir, args.inspect)
        downloaded.append("fuxi")

    if args.model in ("fengwu", "all"):
        download_fengwu(output_dir, args.inspect)
        downloaded.append("fengwu")

    write_model_card(output_dir, downloaded)

    # List everything
    print(f"\n  Files in {output_dir}:")
    total_size = 0
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        total_size += size_mb
        print(f"    {f.name:35s} {size_mb:8.1f} MB")
    print(f"    {'─'*44}")
    print(f"    {'TOTAL':35s} {total_size:8.1f} MB")

    print(f"\n{'='*60}")
    print(f"  DONE — ONNX models ready in {output_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
