# Local and Offline Deployment Guide

This guide provides comprehensive instructions for deploying and running Earth2Studio in local, offline, or air-gapped environments where internet access is limited or unavailable.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Phase 1: Downloading Resources](#phase-1-downloading-resources)
- [Phase 2: Setting Up Offline Environment](#phase-2-setting-up-offline-environment)
- [Phase 3: Running Models Locally](#phase-3-running-models-locally)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Overview

Earth2Studio can be fully operational in offline environments by pre-downloading:
1. Python package and dependencies
2. Model weights and checkpoints
3. Initial condition data (or pre-staged data files)
4. Supporting data (coordinates, constants, etc.)

**Estimated Storage Requirements:**
- Base package: ~50 MB
- All dependencies: ~5-10 GB
- Model weights (per model): 100 MB - 10 GB
- Data cache: Varies (10-100+ GB depending on use case)

## Prerequisites

### Online System (for downloading)

- Python 3.11+ with pip or uv
- Git
- Internet connection
- Sufficient disk space (50+ GB recommended)
- NVIDIA GPU (optional, for GPU-enabled downloads)

### Offline System (target deployment)

- Python 3.11+ (can be bundled)
- NVIDIA GPU with CUDA support (for most models)
- Matching system architecture and Python version to download system

## Phase 1: Downloading Resources

### Step 1: Clone the Repository

On a system with internet access:

```bash
# Clone Earth2Studio repository
git clone https://github.com/NVIDIA/earth2studio.git
cd earth2studio

# Optional: checkout specific version
git checkout v0.12.0
```

### Step 2: Download Python Dependencies

#### Option A: Using pip download

```bash
# Create download directory
mkdir -p offline-packages

# Download Earth2Studio and all dependencies
pip download earth2studio -d offline-packages/

# Download specific model dependencies (example: AIFS)
pip download earth2studio[aifs] -d offline-packages/ --no-build-isolation

# Download other optional dependencies as needed
pip download earth2studio[graphcast] -d offline-packages/
pip download earth2studio[data] -d offline-packages/
```

#### Option B: Using uv (recommended for better dependency resolution)

```bash
# Initialize uv project
mkdir earth2studio-offline && cd earth2studio-offline
uv init --python=3.12

# Add Earth2Studio with desired extras
uv add "earth2studio[aifs,graphcast,data] @ git+https://github.com/synapticore-io/earth2studio.git@v0.12.0"

# Export lock file for reproducible builds
uv export --frozen > requirements.txt

# Download all packages
mkdir packages
uv pip download -r requirements.txt -d packages/
```

### Step 3: Download Model Weights

Earth2Studio models are typically stored on NGC, HuggingFace, or AWS S3. You need to pre-download these.

#### Method 1: Programmatic Download

Create a download script `download_models.py`:

```python
#!/usr/bin/env python
import os
from pathlib import Path

# Set cache directory
cache_dir = Path("./earth2studio_cache")
os.environ["EARTH2STUDIO_CACHE"] = str(cache_dir)
cache_dir.mkdir(exist_ok=True)

# Download models you need
from earth2studio.models.px import FCN3, AIFS, GraphCastOperational

models_to_download = [
    ("FCN3", FCN3),
    ("AIFS", AIFS),
    ("GraphCast", GraphCastOperational),
]

for name, model_class in models_to_download:
    try:
        print(f"Downloading {name}...")
        package = model_class.load_default_package()
        model = model_class.load_model(package)
        print(f"✓ {name} downloaded successfully")
        del model  # Free memory
    except Exception as e:
        print(f"✗ Failed to download {name}: {e}")

print(f"\nAll models cached in: {cache_dir}")
```

Run the script:
```bash
python download_models.py
```

#### Method 2: Manual Download from NGC

For models hosted on NGC (FourCastNet, Atlas, etc.):

```bash
# Install NGC CLI
wget -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip
unzip ngccli_linux.zip
chmod u+x ngc-cli/ngc

# Download specific model (example)
./ngc-cli/ngc registry model download-version "nvidia/modulus/modulus_fourcastnet_v0:0.1"

# Move to cache structure
mkdir -p earth2studio_cache/checkpoints/
mv fourcastnet_* earth2studio_cache/checkpoints/
```

#### Method 3: Direct Download URLs

Some models can be downloaded directly:

```bash
# Example: Download GraphCast from Google
mkdir -p earth2studio_cache/checkpoints/graphcast
cd earth2studio_cache/checkpoints/graphcast

# Download operational model (example URL structure)
# Note: Actual URLs may vary, check model documentation
wget https://storage.googleapis.com/graphcast/params/...
```

### Step 4: Download Initial Condition Data (Optional)

If you need to pre-stage weather data:

#### Option A: Download GFS/HRRR Data

```bash
# Create data download script
mkdir -p local_data/gfs

# Download specific GFS forecast
# Example for 2025-01-01 00Z
DATE="20250101"
HOUR="00"
URL="https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.${DATE}/${HOUR}/atmos/"

# Download required files
for VAR in "pgrb2.0p25.f000" "pgrb2.0p25.f006"; do
    wget "${URL}gfs.t${HOUR}z.${VAR}" -P local_data/gfs/
done
```

#### Option B: Create Synthetic Data

For testing without real data:

```python
# create_test_data.py
import numpy as np
import xarray as xr
from datetime import datetime

# Create synthetic atmospheric data
lats = np.linspace(-90, 90, 721)
lons = np.linspace(0, 359.75, 1440)
time = [datetime(2025, 1, 1, 0, 0)]

data_vars = {
    'u10m': (['time', 'lat', 'lon'], np.random.randn(1, 721, 1440) * 5),
    'v10m': (['time', 'lat', 'lon'], np.random.randn(1, 721, 1440) * 5),
    't2m': (['time', 'lat', 'lon'], np.random.randn(1, 721, 1440) * 10 + 273.15),
    'msl': (['time', 'lat', 'lon'], np.random.randn(1, 721, 1440) * 1000 + 101325),
}

ds = xr.Dataset(
    data_vars,
    coords={'time': time, 'lat': lats, 'lon': lons}
)

# Save to netCDF
ds.to_netcdf('local_data/synthetic_ic.nc')
print("Synthetic initial conditions created")
```

### Step 5: Package Everything

Create a complete offline package:

```bash
# Create archive structure
mkdir -p earth2studio-offline-bundle
cd earth2studio-offline-bundle

# Copy repository
cp -r ../earth2studio ./

# Copy packages
cp -r ../offline-packages ./

# Copy model cache
cp -r ../earth2studio_cache ./cache

# Copy any data
cp -r ../local_data ./data

# Create installation script
cat > install.sh << 'EOF'
#!/bin/bash
set -e

echo "Earth2Studio Offline Installation"
echo "=================================="

# Install Python packages
echo "Installing Python packages..."
pip install --no-index --find-links=./offline-packages earth2studio

# Set up cache directory
echo "Setting up model cache..."
mkdir -p ~/.cache/earth2studio
cp -r ./cache/* ~/.cache/earth2studio/

echo "Installation complete!"
echo "Set EARTH2STUDIO_CACHE=~/.cache/earth2studio"
EOF

chmod +x install.sh

# Create archive
cd ..
tar -czf earth2studio-offline-bundle.tar.gz earth2studio-offline-bundle/
```

## Phase 2: Setting Up Offline Environment

### Transfer to Offline System

Transfer the bundle to your offline system:
```bash
# Via physical media, network transfer, etc.
scp earth2studio-offline-bundle.tar.gz user@offline-system:/path/to/destination/
```

### Install on Offline System

```bash
# Extract bundle
tar -xzf earth2studio-offline-bundle.tar.gz
cd earth2studio-offline-bundle

# Run installation
./install.sh

# Or manual installation:
pip install --no-index --find-links=./offline-packages earth2studio

# Set environment variables
export EARTH2STUDIO_CACHE=/path/to/bundle/cache
export EARTH2STUDIO_DISABLE_MSC=1  # Disable remote data access
```

### Verify Installation

```bash
python << EOF
import earth2studio
print(f"Earth2Studio version: {earth2studio.__version__}")

# Test model loading
from earth2studio.models.px import FCN3
print("Testing model import: Success")

# Verify cache
import os
cache = os.environ.get('EARTH2STUDIO_CACHE')
print(f"Cache directory: {cache}")
EOF
```

## Phase 3: Running Models Locally

### Using Local Data Files

#### Example 1: Inference with Pre-downloaded Data

```python
from earth2studio.models.px import FCN3
from earth2studio.data import XarrayDataSource
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic as run
import xarray as xr

# Load local data file
local_data = xr.open_dataset('/path/to/local_data/synthetic_ic.nc')

# Create data source from local file
data_source = XarrayDataSource(local_data)

# Load model from local cache
model = FCN3.load_model(FCN3.load_default_package())

# Run inference
io = ZarrBackend("outputs/forecast.zarr")
run(["2025-01-01T00:00:00"], 10, model, data_source, io)
```

#### Example 2: Using Cached Local Files

```python
from earth2studio.models.px import AIFS
from earth2studio.data import DataArrayFile
from earth2studio.io import NetCDF4Backend
from earth2studio.run import deterministic as run

# Point to local GRIB/NetCDF files
data = DataArrayFile(file_path="./local_data/gfs/*.grib2")

# Run with local data
model = AIFS.load_model(AIFS.load_default_package())
io = NetCDF4Backend("outputs/aifs_forecast.nc")
run(["2025-01-01T00:00:00"], 8, model, data, io)
```

### Custom Local Data Source

For maximum flexibility, create a custom data source:

```python
from earth2studio.data import DataSource
from datetime import datetime
import numpy as np
import torch

class LocalDataSource(DataSource):
    """Custom data source that reads from local directory"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def __call__(
        self,
        time: datetime | list[datetime],
        variable: str | list[str],
    ) -> torch.Tensor:
        """Load data from local files"""
        # Implement your data loading logic here
        # This is a simplified example
        
        if isinstance(time, datetime):
            time = [time]
        if isinstance(variable, str):
            variable = [variable]
        
        # Load from your local storage format
        data = np.load(f"{self.data_dir}/{time[0].strftime('%Y%m%d')}.npy")
        
        return torch.from_numpy(data)

# Use custom source
local_source = LocalDataSource("/path/to/data")
```

### Running Without Internet Access

Ensure these environment variables are set:

```bash
# Disable remote data access
export EARTH2STUDIO_DISABLE_MSC=1

# Point to local cache
export EARTH2STUDIO_CACHE=/path/to/local/cache

# Disable package download timeout checks (optional)
export EARTH2STUDIO_PACKAGE_TIMEOUT=0

# For NGC models, disable auth attempts
unset NGC_CLI_API_KEY

# For HuggingFace, work offline
export HF_HUB_OFFLINE=1
```

## Troubleshooting

### Model Fails to Load

**Problem**: Model cannot find checkpoint files

**Solution**:
```bash
# Verify cache structure
ls -R $EARTH2STUDIO_CACHE

# Ensure checkpoints are in correct location
# Expected structure:
# cache/
#   checkpoints/
#     fcn3/
#     aifs/
#     graphcast/

# Set explicit cache path
export EARTH2STUDIO_CACHE=/absolute/path/to/cache
```

### Missing Dependencies

**Problem**: Import errors for model-specific packages

**Solution**:
```bash
# Verify all packages installed
pip list | grep -i torch
pip list | grep -i earth2

# Reinstall from offline cache
pip install --no-index --find-links=./offline-packages earth2studio[aifs]
```

### Data Loading Errors

**Problem**: Cannot fetch remote data

**Solution**:
```python
# Always use local data sources
from earth2studio.data import XarrayDataSource
import xarray as xr

# Load from local file
ds = xr.open_dataset('/path/to/local/data.nc')
data = XarrayDataSource(ds)
```

### CUDA/GPU Issues

**Problem**: GPU not detected in offline environment

**Solution**:
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If needed, reinstall PyTorch with correct CUDA version offline
pip install --no-index --find-links=./offline-packages torch torchvision
```

## Advanced Topics

### Creating a Docker Image for Offline Use

```dockerfile
# Dockerfile for offline Earth2Studio
FROM nvcr.io/nvidia/pytorch:25.12-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libeccodes-tools libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy offline packages
COPY offline-packages /tmp/packages
COPY cache /workspace/cache

# Install Earth2Studio offline
RUN pip install --no-index --find-links=/tmp/packages earth2studio

# Set environment
ENV EARTH2STUDIO_CACHE=/workspace/cache
ENV EARTH2STUDIO_DISABLE_MSC=1

WORKDIR /workspace
```

Build and save:
```bash
# Build image
docker build -t earth2studio-offline:latest .

# Save image to file
docker save earth2studio-offline:latest | gzip > earth2studio-offline.tar.gz

# Transfer and load on offline system
docker load < earth2studio-offline.tar.gz
```

### Automating Updates

Create an update script for periodic syncs:

```bash
#!/bin/bash
# update_offline_bundle.sh

ONLINE_DIR="/path/to/online/system"
BUNDLE_DIR="earth2studio-offline-bundle"
DATE=$(date +%Y%m%d)

cd $ONLINE_DIR

# Update repository
git pull

# Download latest dependencies
pip download earth2studio[aifs,graphcast,data] -d $BUNDLE_DIR/offline-packages/

# Download new models if needed
python download_models.py

# Create new bundle
tar -czf earth2studio-offline-$DATE.tar.gz $BUNDLE_DIR/

echo "New bundle created: earth2studio-offline-$DATE.tar.gz"
```

### Optimizing for Limited Storage

If storage is constrained:

1. **Selective Model Download**: Only download models you'll use
2. **Data Compression**: Use compressed data formats (Zarr with compression)
3. **On-demand Loading**: Don't pre-download all data, stage minimal initial conditions
4. **Model Quantization**: Use FP16 or INT8 models where available

```python
# Example: Load model in FP16 to save memory
model = AIFS.load_model(AIFS.load_default_package())
model = model.half()  # Convert to FP16
```

## Best Practices

1. **Version Control**: Tag your offline bundles with date and version
2. **Testing**: Test bundle on online system before deploying offline
3. **Documentation**: Keep list of included models and dependencies
4. **Security**: Scan packages for vulnerabilities before deployment
5. **Updates**: Schedule regular updates when possible
6. **Validation**: Include test scripts in bundle to verify installation

## Resources

- [Earth2Studio Documentation](https://nvidia.github.io/earth2studio/)
- [NGC Model Registry](https://catalog.ngc.nvidia.com/)
- [PyTorch Offline Installation](https://pytorch.org/get-started/locally/)
- [pip Offline Installation](https://pip.pypa.io/en/stable/user_guide/#installing-from-local-packages)

## Appendix: Complete Example Script

```python
#!/usr/bin/env python
"""
Complete offline inference example
"""
import os
from pathlib import Path
from earth2studio.models.px import FCN3
from earth2studio.io import ZarrBackend
import torch
import numpy as np

def setup_offline_env():
    """Configure environment for offline operation"""
    cache_dir = Path("/path/to/cache")
    os.environ["EARTH2STUDIO_CACHE"] = str(cache_dir)
    os.environ["EARTH2STUDIO_DISABLE_MSC"] = "1"
    
def create_synthetic_initial_conditions():
    """Create synthetic IC for testing"""
    # 721x1440 grid at 0.25 degree resolution
    batch, time, lat, lon = 1, 1, 721, 1440
    num_vars = 20  # Number of atmospheric variables
    
    # Random initial conditions (replace with real data)
    ic = torch.randn(batch, time, num_vars, lat, lon)
    return ic

def run_offline_forecast():
    """Run complete forecast offline"""
    setup_offline_env()
    
    # Load model from cache
    print("Loading model from local cache...")
    model = FCN3.load_model(FCN3.load_default_package())
    
    # Create initial conditions
    print("Preparing initial conditions...")
    ic = create_synthetic_initial_conditions()
    
    # Run forecast
    print("Running forecast...")
    io = ZarrBackend("offline_forecast.zarr")
    
    # Manual forecast loop (alternative to run())
    x = ic
    for step in range(10):
        x = model(x)
        # Save to IO backend
        # io.write(x, step)
        print(f"Step {step+1}/10 complete")
    
    print("Forecast complete! Output saved to offline_forecast.zarr")

if __name__ == "__main__":
    run_offline_forecast()
```

This script demonstrates a complete offline workflow that can run without any network access once properly configured.
