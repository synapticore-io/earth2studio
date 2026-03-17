# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0
#
# Patch a pygrib wheel to bundle eccodes.dll and definitions for Windows.
# Usage: python patch-pygrib-wheel.py <wheel.whl> <eccodes_dir>

import base64
import csv
import hashlib
import io
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def _compute_record(work_dir: Path, record_path: Path) -> None:
    """Regenerate the RECORD file with correct sha256 hashes and sizes."""
    record_rel = record_path.relative_to(work_dir)
    rows = []
    for root, _, files in os.walk(work_dir):
        for f in files:
            p = Path(root) / f
            rel = p.relative_to(work_dir)
            if rel == record_rel:
                continue
            data = p.read_bytes()
            digest = hashlib.sha256(data).digest()
            b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            rows.append((str(rel).replace("\\", "/"), f"sha256={b64}", str(len(data))))
    rows.append((str(record_rel).replace("\\", "/"), "", ""))
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(rows)
    record_path.write_text(buf.getvalue(), encoding="utf-8")


def patch_wheel(wheel_path: Path, eccodes_dir: Path) -> None:
    """Add eccodes.dll and definitions to wheel, patch __init__.py for runtime."""
    eccodes_dll = eccodes_dir / "bin" / "eccodes.dll"
    definitions_src = eccodes_dir / "share" / "eccodes" / "definitions"
    if not eccodes_dll.exists():
        raise FileNotFoundError(f"eccodes.dll not found: {eccodes_dll}")
    if not definitions_src.exists():
        raise FileNotFoundError(f"definitions not found: {definitions_src}")

    work_dir = Path(tempfile.gettempdir()) / f"_wheel_patch_{wheel_path.stem}"
    work_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(wheel_path, "r") as z:
            z.extractall(work_dir)

        # Find pygrib package dir (root pygrib/ or .data/platlib/pygrib/)
        pygrib_inits = list(work_dir.rglob("__init__.py"))
        pygrib_dir = None
        for p in pygrib_inits:
            if p.parent.name == "pygrib":
                pygrib_dir = p.parent
                break
        if not pygrib_dir or not pygrib_dir.is_dir():
            raise RuntimeError("pygrib package not found in wheel")

        # Copy eccodes.dll next to the .pyd
        shutil.copy2(eccodes_dll, pygrib_dir)

        # Copy definitions
        def_dest = pygrib_dir / "share" / "eccodes" / "definitions"
        def_dest.mkdir(parents=True, exist_ok=True)
        for item in definitions_src.iterdir():
            dst = def_dest / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

        # Patch __init__.py to set ECCODES_DEFINITION_PATH and DLL search path
        init_py = pygrib_dir / "__init__.py"
        init_content = init_py.read_text(encoding="utf-8")
        bootstrap = '''# Bootstrap: set ECCODES paths for bundled wheel (earth2studio)
import os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_def_path = _os.path.join(_here, "share", "eccodes", "definitions")
if _os.path.exists(_def_path):
    _os.environ.setdefault("ECCODES_DEFINITION_PATH", _def_path)
if _os.name == "nt":
    _os.add_dll_directory(_here) if hasattr(_os, "add_dll_directory") else None
'''
        if "ECCODES_DEFINITION_PATH" not in init_content:
            init_py.write_text(bootstrap + init_content, encoding="utf-8")

        # Regenerate RECORD with correct hashes for all files
        dist_info_dirs = list(work_dir.glob("*.dist-info"))
        if dist_info_dirs:
            record_path = dist_info_dirs[0] / "RECORD"
            _compute_record(work_dir, record_path)

        # Repack wheel
        wheel_path.unlink()
        with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(work_dir):
                for f in files:
                    p = Path(root) / f
                    arcname = p.relative_to(work_dir)
                    z.write(p, arcname)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: patch-pygrib-wheel.py <wheel.whl> <eccodes_dir>", file=sys.stderr)
        sys.exit(1)
    wheel_path = Path(sys.argv[1])
    eccodes_dir = Path(sys.argv[2])
    if not wheel_path.exists():
        print(f"Wheel not found: {wheel_path}", file=sys.stderr)
        sys.exit(1)
    if not eccodes_dir.is_dir():
        print(f"ECCODES dir not found: {eccodes_dir}", file=sys.stderr)
        sys.exit(1)
    patch_wheel(wheel_path, eccodes_dir)
    print(f"Patched: {wheel_path}")
