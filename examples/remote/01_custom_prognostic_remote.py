# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# %%
"""
Custom Prognostic via Remote Workflow
=======================================

Running a custom prognostic model on a remote Earth2Studio server.

This example demonstrates how to execute a pre-deployed deterministic workflow
on a remote server using the Earth2Studio client SDK.

In this example you will learn:

- Setting up the RemoteEarth2Workflow client
- Submitting inference requests with custom parameters
- Retrieving and working with remote forecast results
"""
# /// script
# dependencies = [
#   "earth2studio[examples]",
#   "matplotlib",
# ]
# ///

# %%
# Remote Workflow Setup
# ---------------------
# Rather than implementing a custom prognostic locally, this example shows how to
# leverage a remote server that already has a custom model deployed. The server
# handles all model execution, while the client focuses on submitting requests and
# retrieving results.

# %%
import os

os.makedirs("outputs", exist_ok=True)

from earth2studio.utils.gallery_reference_time import gallery_reference_datetime_utc
from earth2studio_client import RemoteEarth2Workflow

_ref = gallery_reference_datetime_utc(days_back=42)

# %%
# Create a RemoteEarth2Workflow client pointing to your server instance.
# The workflow_name parameter selects which pre-deployed workflow to use.

# %%
workflow = RemoteEarth2Workflow(
    base_url="http://localhost:8000",
    workflow_name="deterministic_earth2_workflow",
)

# %%
# Submit Inference Request
# ------------------------
# Execute the workflow by calling the workflow object with desired parameters.
# These parameters depend on the specific workflow deployment.

# %%
result = workflow(
    start_time=[_ref],
    lead_time=24,  # 24 hours forecast
)

# %%
# Retrieve Results
# ----------------
# The result object is lazy—it doesn't block until you access data.
# Call ``as_dataset()`` to download and parse the results.

# %%
print("Waiting for inference to complete...")
ds = result.as_dataset()

print("\nForecast Dataset:")
print(ds)

# %%
# Post Processing
# ---------------
# Now you can work with the results just like a local Zarr/NetCDF dataset.

# %%
import matplotlib.pyplot as plt

forecast_date = _ref.strftime("%Y-%m-%d")
variable = "u10m"

plt.close("all")
fig, ax = plt.subplots(2, 2, figsize=(6, 4))

# Plot u10m at different lead times
lead_times = [0, 6, 12, 18]
for idx, lt in enumerate(lead_times):
    row, col = divmod(idx, 2)
    if variable in ds.data_vars and lt < ds.dims["lead_time"]:
        ax[row, col].imshow(ds[variable].isel(lead_time=lt).values, vmin=-20, vmax=20)
        ax[row, col].set_title(f"Lead time: {lt}hrs")

plt.suptitle(f"{variable} - {forecast_date} (Remote Forecast)")
plt.savefig("outputs/01_custom_prognostic_remote.jpg", bbox_inches="tight")
print(f"\nPlot saved to outputs/01_custom_prognostic_remote.jpg")

# %%
# Clean up
workflow.close()
