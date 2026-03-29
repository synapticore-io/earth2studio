# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# %%
"""
Custom Data Source via Remote Workflow
========================================

Using a remote workflow with custom data source implementations.

Rather than implementing a custom data source client-side, this example
demonstrates how to leverage a pre-configured remote workflow that uses
custom data sources (e.g., ARCO with derived fields like relative humidity).

In this example you will learn:

- Submitting requests to workflows with custom data sources
- Using historical data (e.g., ERA5) via remote inference
- Chaining custom data sources with forecast models
"""
# /// script
# dependencies = [
#   "earth2studio[client] @ git+https://github.com/synapticore-io/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Remote Custom Data Source Workflow
# -----------------------------------
# A remote server can be configured with custom data sources that:
# - Fetch data from ERA5/ARCO with calculated fields (e.g., relative humidity)
# - Integrate multiple data sources (GFS for near-term, historical for backtesting)
# - Cache results to avoid repeated downloads

# %%
import os
from datetime import datetime

os.makedirs("outputs", exist_ok=True)

from earth2studio_client import RemoteEarth2Workflow

# %%
# Initialize Remote Workflow
# ---------------------------
# This workflow uses a custom data source backend (e.g., ARCO with RH calculation).

# %%
workflow = RemoteEarth2Workflow(
    base_url="http://localhost:8000",
    workflow_name="deterministic_fcn_workflow",
)

# %%
# Submit Historical Forecast
# ---------------------------
# Fixed ERA5/ARCO lesson date (reproducible). For a recent analysis day, set
# ``EARTH2STUDIO_EXAMPLE_ANCHOR_DATE`` (see :py:func:`~earth2studio.utils.gallery_reference_time.gallery_reference_datetime_utc`).

# %%
result = workflow(
    start_time=[datetime(1993, 4, 5)],  # ERA5 snapshot — override via env if needed
    lead_time=4,  # Short lead time for testing
)

# %%
# Retrieve Results
# ----------------

# %%
print("Waiting for remote forecast with custom data source...")
ds = result.as_dataset()

print("\nForecast Dataset:")
print(ds)

# %%
# Post Processing
# ---------------
# Compare results: plot a forecast variable affected by the custom data source.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

forecast_date = "1993-04-05"
variable = "tcwv"  # Total column water vapor

if variable not in ds.data_vars:
    # Fallback to first available variable
    variable = list(ds.data_vars)[0]
    print(f"Variable {variable} not found; using {variable}")

plt.close("all")
fig, ax = plt.subplots(2, 2, figsize=(6, 4))

lead_times = [0, 1, 2, 3]
for idx, lt in enumerate(lead_times):
    row, col = divmod(idx, 2)
    if variable in ds.data_vars and lt < ds.dims.get("lead_time", 0):
        data = ds[variable].isel(lead_time=lt).values
        im = ax[row, col].imshow(
            data,
            vmin=0,
            vmax=80,
            cmap="magma",
        )
        ax[row, col].set_title(f"Lead time: {lt}h")

plt.suptitle(f"{variable} - {forecast_date} (Remote + Custom Data Source)")
plt.savefig("outputs/03_custom_datasource_remote.jpg", bbox_inches="tight")
print(f"\nPlot saved to outputs/03_custom_datasource_remote.jpg")

# %%
# Advanced: Use Result as Data Source
# ------------------------------------
# Convert the remote result into an Earth2Studio InferenceOutputSource
# for further chaining or processing.

# %%
# Uncomment to use result as data source for downstream workflows:
# ds_source = result.as_data_source()
# # Now can chain this with other models or diagnostics locally

# %%
# Clean up
workflow.close()
