# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import aiohttp
import xarray as xr

from . import fsspec_utils
from .client import Earth2StudioClient
from .exceptions import Earth2StudioAPIError
from .models import InferenceRequest, InferenceRequestResults, StorageType


class RemoteEarth2Workflow:
    """High-level client for executing a remote Earth2Studio workflow."""

    def __init__(
        self,
        base_url: str,
        workflow_name: str,
        xr_args: dict[str, Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.base_url = base_url
        self.workflow_name = workflow_name
        self.client = Earth2StudioClient(base_url=base_url, workflow_name=workflow_name, **client_kwargs)
        self.xr_args = xr_args.copy() if xr_args else {}

    def __call__(self, **kwargs: Any) -> RemoteEarth2WorkflowResult:
        request = InferenceRequest(parameters=kwargs.copy())
        response = self.client.submit_inference_request(request)
        return RemoteEarth2WorkflowResult(self, response.execution_id)


@dataclass
class RemoteEarth2WorkflowResult:
    """Lazy result handle for a remote inference request."""

    workflow: RemoteEarth2Workflow
    execution_id: str
    _result: InferenceRequestResults | None = None

    def _get_result(self) -> InferenceRequestResults:
        if self._result is None:
            self._result = self.workflow.client.wait_for_completion(self.execution_id)
        return self._result

    def as_dataset(self) -> xr.Dataset:
        result = self._get_result()
        result_paths = result.result_paths()
        if not result_paths:
            raise Earth2StudioAPIError("The request did not return any outputs.")

        # Prefer first output as the primary dataset
        result_path = result_paths[0]

        if result_path.endswith(".zarr"):
            return _open_zarr_dataset(self.workflow, result, result_path)

        if result_path.endswith(".nc"):
            result_data = self.workflow.client.download_result(result, result_path)
            return xr.open_dataset(result_data, engine="netcdf4", **self.workflow.xr_args)

        raise ValueError(
            f"Unsupported result file format: {result_path!r}. Only .zarr and .nc are supported."
        )

    def as_data_source(self) -> Any:
        """Return the results as an Earth2Studio InferenceOutputSource.

        Note: Requires the `earth2studio` package to be installed.
        """
        from earth2studio.data import InferenceOutputSource  # type: ignore[import-untyped]

        return InferenceOutputSource(self.as_dataset())


def _open_zarr_dataset(
    workflow: RemoteEarth2Workflow,
    result: InferenceRequestResults,
    result_path: str,
) -> xr.Dataset:
    if result.storage_type == StorageType.S3:
        # Strip out the execution-id prefix from the path (first component)
        zarr_path = "/".join(result_path.split("/")[1:])
        mapper = fsspec_utils.get_mapper(result, zarr_path)
        return xr.open_zarr(mapper, consolidated=True, **workflow.xr_args)

    if result.storage_type == StorageType.SERVER:
        result_url = urljoin(
            workflow.base_url + "/",
            (workflow.client.result_root_path(result) + result_path).lstrip("/"),
        )

        xr_kwargs = dict(workflow.xr_args)
        storage_options = dict(xr_kwargs.pop("storage_options", {}))

        # Forward auth token for private endpoints
        if workflow.client.token:
            headers = dict(storage_options.get("headers", {}))
            headers["Authorization"] = f"Bearer {workflow.client.token}"
            storage_options["headers"] = headers

        # Ensure large zarr reads don't time out immediately
        zarr_timeout = max(300.0, workflow.client.timeout)
        client_kwargs = dict(storage_options.get("client_kwargs", {}))
        if "timeout" not in client_kwargs:
            client_kwargs["timeout"] = aiohttp.ClientTimeout(total=zarr_timeout)
        storage_options["client_kwargs"] = client_kwargs

        return xr.open_zarr(
            result_url,
            consolidated=True,
            storage_options=storage_options or None,
            **xr_kwargs,
        )

    raise ValueError(f"Unsupported storage type: {result.storage_type}")

