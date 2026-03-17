# Windows: ECCODES and GRIB self-build

Earth2Studio depends on **pygrib**, which requires the **ECCODES** C library. On Windows there are no official ECCODES wheels, so you must either use pre-built wheels from our releases or build ECCODES from source. This document describes both options (no Conda, no WSL).

## Option A: Pre-built wheels (recommended)

When we publish a release, we attach pygrib Windows wheels with bundled ECCODES. Download the wheel matching your Python version from the [Releases](https://github.com/synapticore-io/earth2studio/releases) page and install:

```powershell
pip install pygrib-2.1.8-cp311-cp311-win_amd64.whl  # adjust version and cp311 for your Python
```

Then install the rest of earth2studio as usual (`uv sync` or `pip install -e .`).

## Option B: Self-build from source

## Prerequisites

- **CMake** (3.12 or newer), e.g. from [cmake.org](https://cmake.org/download/) or via Visual Studio Installer.
- **Visual Studio 2022** with the **"Desktop development with C++"** workload (MSVC compiler, Windows SDK).
- **Git for Windows** ([git-scm.com](https://git-scm.com/download/win)) — ecbuild needs `bash` for symlink replacement on Windows.
- **PowerShell** (Windows 10/11).
- **tar** (included on Windows 10 1903+; or use 7-Zip and extract the GitHub source manually if `tar` is missing).

## Step 1: Build ECCODES from source

From the repository root, run:

```powershell
.\scripts\build-eccodes-windows.ps1
```

This downloads the ECCODES source from GitHub, configures with CMake, builds with Visual Studio, and installs to `%LOCALAPPDATA%\eccodes` by default. The build typically takes 10–15 minutes.

To install to a custom prefix (e.g. `C:\eccodes`):

```powershell
.\scripts\build-eccodes-windows.ps1 -InstallPrefix "C:\eccodes"
```

## Step 2: Set ECCODES_DIR and install the project

Before running `uv sync`, the build of **pygrib** must see the ECCODES installation. Set the environment variable to the install prefix from Step 1:

```powershell
$env:ECCODES_DIR = "$env:LOCALAPPDATA\eccodes"
uv sync
```

Or use the helper script (uses the default prefix if ECCODES was built there):

```powershell
.\scripts\setup-windows-grib.ps1
```

To make `ECCODES_DIR` permanent for your user account (so every new terminal has it):

```powershell
[Environment]::SetEnvironmentVariable('ECCODES_DIR', "$env:LOCALAPPDATA\eccodes", 'User')
```

Then open a new terminal and run `uv sync`.

## Step 3: Verify (validation)

Run:

```powershell
uv run python -c "from earth2studio.data import GFS; print('OK')"
```

If this runs without import or build errors, the full stack including GRIB is working. This is the recommended validation step after completing the self-build.

## Optional: ECCODES definition path

If at runtime you see an error about `boot.def` or ECCODES definition files not found, set:

```powershell
$env:ECCODES_DEFINITION_PATH = "$env:ECCODES_DIR\share\eccodes\definitions"
```

(Adjust if your install prefix is different; the build script installs definitions under `share\eccodes\definitions`.)

## Troubleshooting

- **CMake not found:** Add CMake to PATH or open a **Developer PowerShell for VS 2022** and run the script from there.
- **"eccodes.h not found" when building pygrib:** Ensure `ECCODES_DIR` is set to the same prefix used in Step 1 and that `ECCODES_DIR\include\eccodes.h` exists.
- **Visual Studio generator:** The script uses `Visual Studio 17 2022`. If you have a different version, edit `scripts\build-eccodes-windows.ps1` and change the `-G` argument (e.g. `Visual Studio 16 2019` and `-A x64`).
- **"eccodes.pc.tmp" CMake error:** Fixed by `-DENABLE_PKGCONFIG=OFF` in the build script (pygrib does not need pkg-config).
