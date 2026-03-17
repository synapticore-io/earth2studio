# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0
#
# Set ECCODES_DIR (if ECCODES was built via build-eccodes-windows.ps1) and run uv sync
# so that pygrib builds against the self-built ECCODES.
#
# Usage:
#   # After building ECCODES once:
#   .\scripts\setup-windows-grib.ps1
#
#   # Or with explicit prefix:
#   .\scripts\setup-windows-grib.ps1 -EccodesPrefix "C:\eccodes"

param(
    [string]$EccodesPrefix = $env:ECCODES_DIR
)

$ErrorActionPreference = "Stop"

if (-not $EccodesPrefix) {
    $defaultPrefix = "$env:LOCALAPPDATA\eccodes"
    if (Test-Path "$defaultPrefix\include\eccodes.h") {
        $EccodesPrefix = $defaultPrefix
        Write-Host "Using ECCODES at: $EccodesPrefix"
    } else {
        Write-Error "ECCODES_DIR not set and no ECCODES found at $defaultPrefix. Run scripts\build-eccodes-windows.ps1 first."
    }
}

if (-not (Test-Path "$EccodesPrefix\include\eccodes.h")) {
    Write-Error "ECCODES not found at $EccodesPrefix (include\eccodes.h missing). Run scripts\build-eccodes-windows.ps1 first."
}

$env:ECCODES_DIR = $EccodesPrefix
Write-Host "ECCODES_DIR=$env:ECCODES_DIR"
Write-Host "Running uv sync..."
& uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done. You can now use: uv run python -c `"from earth2studio.data import GFS; print('OK')`""
