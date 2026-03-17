# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0
#
# Functional test for pygrib: read a GRIB file, verify shape and values.
# Run after setup-windows-grib.ps1 (eccodes.dll must be in pygrib/ or PATH).
#
# Usage: uv run python scripts/test_pygrib_functional.py <path.grib2>
#
# Provide a GRIB2 file path to verify pygrib reads correctly. ECCODES samples
# (if present) or download from NOMADS subsetter for a small test file.

import os
import sys
from pathlib import Path

import numpy as np
import pygrib


def test_file(path: Path) -> None:
    grbs = pygrib.open(str(path))
    g = grbs[1]
    vals = g.values
    grbs.close()
    assert vals.shape[0] > 0 and vals.shape[1] > 0, f"Unexpected shape {vals.shape}"
    assert not np.isnan(vals).all(), "All NaN"
    assert np.isfinite(vals).any(), "No finite values"
    print(f"OK: {path.name} shape={vals.shape} min={vals.min():.2f} max={vals.max():.2f}")


def main() -> int:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1
        test_file(path)
        return 0

    # Try ECCODES samples
    ecc = os.environ.get("ECCODES_DIR", "")
    if not ecc:
        ecc = os.path.join(os.environ.get("LOCALAPPDATA", ""), "eccodes")
    samples = Path(ecc) / "share" / "eccodes" / "samples"
    if samples.exists():
        for f in sorted(samples.iterdir()):
            if f.suffix in (".grib2", ".grb2", ".grib1", ".grb1"):
                test_file(f)
                return 0

    print("Usage: python test_pygrib_functional.py <path.grib2>", file=sys.stderr)
    print("Provide a GRIB file to verify pygrib reads correctly.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
