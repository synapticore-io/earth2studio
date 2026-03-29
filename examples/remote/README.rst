Remote Workflow Examples
=========================

These examples demonstrate how to use the Earth2Studio client SDK to submit
inference requests to a remote Earth2Studio server and retrieve results.

Instead of running models locally, these workflows assume that:

- A running Earth2Studio server (via Docker or manual deployment)
- Pre-configured workflows with custom models, diagnostics, and data sources
- Results retrieved via HTTP (Zarr/NetCDF downloads or S3 storage)

Prerequisites
-------------

1. **Server Running**: Ensure an Earth2Studio server is running on ``http://localhost:8000``
   (adjust base_url in examples if different)

   .. code-block:: bash

       cd serve && docker compose up -d

2. **Client SDK** and plotting extras for the gallery-style scripts:

   .. code-block:: bash

       uv pip install "earth2studio[examples]"

Examples
--------

01_custom_prognostic_remote.py
   Submit a forecast request to a remote deterministic workflow.
   Demonstrates:
   - Creating a ``RemoteEarth2Workflow`` client
   - Submitting inference requests with parameters
   - Retrieving and visualizing remote forecast results

02_custom_diagnostic_remote.py
   Use a pre-deployed diagnostic workflow with custom-computed variables.
   Demonstrates:
   - Submitting requests to diagnostic workflows
   - Accessing custom diagnostic variables in results
   - Handling workflows with multi-step inference

03_custom_datasource_remote.py
   Leverage custom data source implementations on the server.
   Demonstrates:
   - Using workflows with ARCO/ERA5 custom data sources
   - Historical inference via remote server
   - Converting results to ``InferenceOutputSource`` for local chaining

04_remote_deterministic_forecast.py
   Point time series of 2 m temperature (deterministic hosted workflow), with
   graceful placeholders when the API is offline.

05_remote_cams_heatmap.py
   CAMS analysis workflow: multi-panel PM2.5 / dust / O₃ over Europe.

06_remote_cams_forecast.py
   CAMS forecast workflow: PM2.5 trajectories at several European cities.

Key Concepts
------------

**RemoteEarth2Workflow**
   High-level client that submits requests and manages the result lifecycle.
   Workflows are lazy—execution blocks only when you call ``.as_dataset()``.

**InferenceRequest**
   Encapsulates workflow parameters. Submitted via the underlying
   ``Earth2StudioClient``.

**StorageType (SERVER or S3)**
   Results can be stored on the server (HTTP download) or in S3
   (signed URL download). Client handles both transparently.

**InferenceOutputSource**
   Returned by ``.as_data_source()``, allows chaining remote results
   as inputs to local workflows.

Server Configuration
--------------------

Deploy a server with custom workflows:

.. code-block:: yaml

   # serve/server/conf/config.yaml
   workflows:
     custom_workflow:
       path: path/to/workflow.py
       class: CustomWorkflow

Place custom workflows in ``serve/server/example_workflows/`` or specify
``WORKFLOW_DIR`` at runtime.

Error Handling
--------------

Use the client's exception types for graceful error handling:

.. code-block:: python

   from earth2studio_client.exceptions import (
       Earth2StudioAPIError,
       InferenceRequestNotFoundError,
       RequestTimeoutError,
   )

   try:
       result = workflow(start_time=[...])
       ds = result.as_dataset()
   except RequestTimeoutError as e:
       print(f"Inference timed out: {e}")
   except Earth2StudioAPIError as e:
       print(f"Server error: {e}")
