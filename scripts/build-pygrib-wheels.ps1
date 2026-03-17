# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0
#
# Build pygrib wheels for Windows with bundled ECCODES.
# Requires: ECCODES built (or run build-eccodes-windows.ps1 first), uv, Python 3.11/3.12/3.13.
#
# Usage:
#   .\scripts\build-pygrib-wheels.ps1
#   .\scripts\build-pygrib-wheels.ps1 -EccodesPrefix "C:\eccodes" -PythonVersions "3.11","3.12"
#
# Output: dist/pygrib-*.whl (patched with eccodes.dll and definitions)

param(
    [string]$EccodesPrefix = $env:ECCODES_DIR,
    [string[]]$PythonVersions = @("3.11", "3.12", "3.13"),
    [string]$OutputDir = "dist",
    [switch]$BuildEccodes
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if (-not $EccodesPrefix) {
    $defaultPrefix = "$env:LOCALAPPDATA\eccodes"
    if (Test-Path "$defaultPrefix\include\eccodes.h") {
        $EccodesPrefix = $defaultPrefix
    } else {
        Write-Error "ECCODES_DIR not set and no ECCODES at $defaultPrefix. Run build-eccodes-windows.ps1 or set -EccodesPrefix."
    }
}

if (-not (Test-Path "$EccodesPrefix\include\eccodes.h")) {
    Write-Error "ECCODES not found at $EccodesPrefix. Run build-eccodes-windows.ps1 first."
}

if ($BuildEccodes) {
    Write-Host "Building ECCODES..."
    & "$RepoRoot\scripts\build-eccodes-windows.ps1" -InstallPrefix $EccodesPrefix
}

$env:ECCODES_DIR = $EccodesPrefix
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

foreach ($py in $PythonVersions) {
    Write-Host "Building pygrib wheel for Python $py..."
    $pyPath = $null
    $pyOut = & py -$py -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $pyOut -and (Test-Path ($pyOut.Trim()))) {
        $pyPath = $pyOut.Trim()
    }
    if (-not $pyPath) {
        $venvPy = "$RepoRoot\.venv\Scripts\python.exe"
        if ((Test-Path $venvPy) -and ($py -eq "3.12" -or $py -eq "3.11")) {
            $pyPath = $venvPy
        }
    }
    if (-not $pyPath) {
        Write-Warning "Python $py not found, skipping."
        continue
    }

    Push-Location $RepoRoot
    try {
        & $pyPath -m pip wheel pygrib --no-deps -w $OutputDir 2>&1
        if ($LASTEXITCODE -ne 0) { throw "pip wheel failed for Python $py" }
    } finally {
        Pop-Location
    }

    $cp = $py.Replace('.', '')
    $wheels = Get-ChildItem $OutputDir -Filter "pygrib-*-cp${cp}-*-win_amd64.whl"
    foreach ($whl in $wheels) {
        Write-Host "Patching $($whl.Name)..."
        & $pyPath "$RepoRoot\scripts\patch-pygrib-wheel.py" $whl.FullName $EccodesPrefix
    }
}

Write-Host ""
Write-Host "Wheels built in: $OutputDir"
Get-ChildItem $OutputDir -Filter "pygrib-*.whl" | ForEach-Object { Write-Host "  $($_.Name)" }
