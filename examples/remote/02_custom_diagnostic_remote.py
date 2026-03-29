# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# %%
"""
Custom Diagnostic via Remote Workflow
=======================================

Running a custom diagnostic model on a remote Earth2Studio server.

This example demonstrates how to execute a pre-deployed diagnostic workflow
that includes custom diagnostic models (e.g., unit conversions, derived metrics).

In this example you will learn:

- Submitting requests to a remote diagnostic workflow
- Accessing custom-computed variables in the results
- Using RemoteEarth2Workflow with ensemble or multi-model setups
"""
# /// script
# dependencies = [
#   "earth2studio[examples]",
#   "cartopy",
# ]
# ///

# %%
# Remote Diagnostic Workflow
# ---------------------------
# Diagnostic models (like temperature conversion K→C, derived indices, etc.)
# are often deployed server-side to compute post-processed variables
# automatically as part of the inference pipeline.

# %%
import os

os.makedirs("outputs", exist_ok=True)

from earth2studio.utils.gallery_reference_time import gallery_reference_datetime_utc
from earth2studio_client import RemoteEarth2Workflow

_ref = gallery_reference_datetime_utc(days_back=42)

# %%
# Initialize the remote diagnostic workflow.
# This workflow runs a prognostic model (e.g., DLWP) and applies custom
# diagnostics (e.g., K→Celsius conversion).

# %%
workflow = RemoteEarth2Workflow(
    base_url="http://localhost:8000",
    workflow_name="diagnostic_workflow",
)

# %%
# Submit Request
# --------------
# Submit a forecast request with your desired time range.

# %%
result = workflow(
    start_time=[_ref],
    lead_time=20,  # 20 time-steps (hours)
)

# %%
# Fetch Results
# -------------

# %%
print("Waiting for diagnostic forecast to complete...")
ds = result.as_dataset()

print("\nForecast Dataset with Diagnostics:")
print(ds)
print("\nAvailable variables:", list(ds.data_vars))

# %%
# Post Processing
# ---------------
# Plot the custom diagnostic variable (e.g., t2m_c if converted to Celsius).

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

forecast_date = _ref.strftime("%Y-%m-%d")

# Try to plot a diagnostic variable if available
diagnostic_vars = ["t2m_c", "t2m", "tp"]
variable = None
for var in diagnostic_vars:
    if var in ds.data_vars:
        variable = var
        break

if variable is None:
    variable = list(ds.data_vars)[0]
    print(f"Using available variable: {variable}")
else:
    print(f"Plotting diagnostic variable: {variable}")

plt.close("all")
fig, ax = plt.subplots(
    1,
    5,
    figsize=(12, 4),
    subplot_kw={"projection": ccrs.Orthographic()},
    constrained_layout=True,
)

if "lead_time" in ds.dims:
    max_steps = min(5, ds.dims["lead_time"])
    step = max(1, ds.dims["lead_time"] // max_steps)

    for i, t in enumerate(range(0, min(20, ds.dims["lead_time"]), step)):
        if i >= 5:
            break
        if variable in ds.data_vars:
            data = ds[variable].isel(lead_time=t).values
            ax[i].imshow(
                data,
                transform=ccrs.PlateCarree(),
                vmin=data.min(),
                vmax=data.max(),
            )
            ax[i].set_title(f"Lead: {t}h")
            ax[i].coastlines()
            ax[i].gridlines()

plt.suptitle(f"{variable} - {forecast_date} (Remote Diagnostic Forecast)")
plt.savefig("outputs/02_custom_diagnostic_remote.jpg")
print(f"\nPlot saved to outputs/02_custom_diagnostic_remote.jpg")

# %%
# Clean up
workflow.close()
