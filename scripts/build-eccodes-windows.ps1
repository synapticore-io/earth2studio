# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0
#
# Build ECCODES on Windows from source so that pygrib can be built (eccodes.h).
# Requires: CMake, Visual Studio with C++ workload, Git for Windows (bash for ecbuild symlink step).
#
# Usage:
#   .\scripts\build-eccodes-windows.ps1
#   .\scripts\build-eccodes-windows.ps1 -InstallPrefix "C:\eccodes"
#
# After running, set ECCODES_DIR to InstallPrefix before uv sync / pip install pygrib:
#   $env:ECCODES_DIR = "C:\eccodes"
#   uv sync

param(
    [string]$InstallPrefix = "$env:LOCALAPPDATA\eccodes",
    [string]$EccodesVersion = "2.45.0",
    [string]$BuildDir = "$env:LOCALAPPDATA\eccodes-build"
)

$ErrorActionPreference = "Stop"
$SourceZip = "$env:TEMP\eccodes-$EccodesVersion.tar.gz"
$SourceDir = "$env:TEMP\eccodes-$EccodesVersion"
$BuildSubDir = "$BuildDir\build"

Write-Host "ECCODES Windows self-build"
Write-Host "  Version:       $EccodesVersion"
Write-Host "  Install prefix: $InstallPrefix"
Write-Host "  Build dir:     $BuildDir"
Write-Host ""

# Check for CMake
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Error "CMake not found. Install CMake and add it to PATH (e.g. from https://cmake.org/download/ or via Visual Studio)."
}

# ecbuild needs bash for Windows symlink replacement; add Git for Windows to PATH
$gitPaths = @("C:\Program Files\Git\usr\bin", "C:\Program Files\Git\bin")
foreach ($p in $gitPaths) {
    if (Test-Path $p) {
        $env:PATH = "$p;$env:PATH"
        break
    }
}
$bash = Get-Command bash -ErrorAction SilentlyContinue
if (-not $bash) {
    Write-Error "bash not found. Install Git for Windows (https://git-scm.com/download/win) - ecbuild needs it for symlink replacement."
}

# Download source if not present
if (-not (Test-Path "$SourceDir\CMakeLists.txt")) {
    $url = "https://github.com/ecmwf/eccodes/archive/refs/tags/$EccodesVersion.tar.gz"
    Write-Host "Downloading ECCODES $EccodesVersion from GitHub..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $SourceZip -UseBasicParsing
    } catch {
        Write-Error "Download failed: $_. Exception: $($_.Exception.Message)"
    }
    Write-Host "Extracting..."
    tar --force-local -xzf $SourceZip -C $env:TEMP
    $extracted = Get-ChildItem "$env:TEMP" -Filter "eccodes-*" -Directory | Where-Object { $_.Name -match "eccodes-\d" } | Select-Object -First 1
    if ($extracted -and $extracted.FullName -ne (Resolve-Path $SourceDir -ErrorAction SilentlyContinue).Path) {
        if (Test-Path $SourceDir) { Remove-Item $SourceDir -Recurse -Force }
        Rename-Item $extracted.FullName $SourceDir
    }
}

# Configure and build
New-Item -ItemType Directory -Force -Path $BuildSubDir | Out-Null
Push-Location $BuildSubDir

try {
    Write-Host "Configuring ECCODES with CMake..."
    & cmake -G "Visual Studio 17 2022" -A x64 `
        -DCMAKE_INSTALL_PREFIX="$InstallPrefix" `
        -DBUILD_SHARED_LIBS=ON `
        -DENABLE_FORTRAN=OFF `
        -DENABLE_AEC=OFF `
        -DENABLE_PKGCONFIG=OFF `
        "$SourceDir"
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }

    Write-Host "Building (Release)..."
    & cmake --build . --config Release
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed." }

    Write-Host "Installing to $InstallPrefix..."
    New-Item -ItemType Directory -Force -Path $InstallPrefix | Out-Null
    & cmake --install . --config Release
    if ($LASTEXITCODE -ne 0) { throw "CMake install failed." }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "ECCODES installed to: $InstallPrefix"
Write-Host "Set ECCODES_DIR and run uv sync:"
Write-Host "  `$env:ECCODES_DIR = `"$InstallPrefix`""
Write-Host "  uv sync"
Write-Host ""
Write-Host "To make ECCODES_DIR permanent for this user, run:"
Write-Host "  [Environment]::SetEnvironmentVariable('ECCODES_DIR', '$InstallPrefix', 'User')"
