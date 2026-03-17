# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0
#
# Download a small GRIB2 sample for functional testing.
# Usage: .\scripts\download_grib_sample.ps1

$out = "$PSScriptRoot\..\test\fixtures\sample.grib2"
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null
$url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260311/00/atmos/gfs.t00z.pgrb2.1p00.f000"
Write-Host "Downloading GRIB2 sample (~40MB)..."
Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
Write-Host "Saved: $out"
