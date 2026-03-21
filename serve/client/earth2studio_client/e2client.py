# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import xarray as xr

from . import fsspec_utils
from .client import Earth2StudioClient
from .exceptions import Earth2StudioAPIError
from .models import InferenceRequest, InferenceRequestResults, StorageType

logger = logging.getLogger(__name__)


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

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.client.close()


@dataclass
class RemoteEarth2WorkflowResult:
    """Lazy result handle for a remote inference request."""

    workflow: RemoteEarth2Workflow
    execution_id: str
    _result: InferenceRequestResults | None = None
    _closed: bool = False
    _tmpdir: str | None = field(default=None, repr=False)

    def close(self) -> None:
        """Close the underlying HTTP session.

        Automatically called after ``as_dataset()`` or ``as_data_source()`` completes.
        Can be called manually if the result is discarded without accessing data.
        """
        if not self._closed:
            self._closed = True
            self.workflow.client.close()

    def _get_result(self) -> InferenceRequestResults:
        if self._result is None:
            self._result = self.workflow.client.wait_for_completion(self.execution_id)
            self._closed = True
            self.workflow.client.close()
        return self._result

    def as_dataset(self) -> xr.Dataset:
        result = self._get_result()
        result_paths = result.result_paths()
        if not result_paths:
            raise Earth2StudioAPIError("The request did not return any outputs.")

        # Prefer first output as the primary dataset
        result_path = result_paths[0]

        if result_path.endswith(".zarr"):
            return _open_zarr_dataset(self, result, result_path)

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


def _download_zarr_to_local(
    workflow_result: RemoteEarth2WorkflowResult,
    result: InferenceRequestResults,
    result_path: str,
    max_workers: int = 8,
) -> str:
    """Download a remote Zarr store to a local temp directory via synchronous HTTP.

    Works around a bug in zarr 3.x / xarray where async HTTP chunk reads
    return fill values instead of actual data for some chunks.
    """
    client = workflow_result.workflow.client
    zarr_prefix = result_path
    if not zarr_prefix.endswith("/"):
        zarr_prefix += "/"

    zarr_files = [f for f in result.output_files if zarr_prefix in f.path or f.path == result_path]

    tmpdir = tempfile.mkdtemp(prefix="e2s_zarr_")
    workflow_result._tmpdir = tmpdir
    zarr_dir = os.path.join(tmpdir, "results.zarr")

    root_url = urljoin(
        workflow_result.workflow.base_url + "/",
        client.result_root_path(result).lstrip("/"),
    )

    def _download_file(file_info: Any) -> None:
        rel = file_info.path
        prefix = f"{result.request_id}/"
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
        # Strip results.zarr/ prefix to get internal zarr path
        zarr_store_prefix = "results.zarr/"
        if zarr_store_prefix in rel:
            internal_path = rel[rel.index(zarr_store_prefix) + len(zarr_store_prefix):]
        else:
            internal_path = rel

        local_path = os.path.join(zarr_dir, internal_path.replace("/", os.sep))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        chunk_url = f"{root_url}{rel}"
        import requests
        headers = {}
        if client.token:
            headers["Authorization"] = f"Bearer {client.token}"
        resp = requests.get(chunk_url, headers=headers, timeout=client.timeout)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)

    logger.info("Downloading %d zarr files to %s", len(zarr_files), zarr_dir)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_file, f): f for f in zarr_files}
        for future in as_completed(futures):
            future.result()  # raises on error

    return zarr_dir


def _open_zarr_dataset(
    workflow_result: RemoteEarth2WorkflowResult,
    result: InferenceRequestResults,
    result_path: str,
) -> xr.Dataset:
    if result.storage_type == StorageType.S3:
        # Strip out the execution-id prefix from the path (first component)
        zarr_path = "/".join(result_path.split("/")[1:])
        mapper = fsspec_utils.get_mapper(result, zarr_path)
        return xr.open_zarr(mapper, consolidated=True, zarr_format=3, **workflow_result.workflow.xr_args)

    if result.storage_type == StorageType.SERVER:
        zarr_dir = _download_zarr_to_local(workflow_result, result, result_path)
        return xr.open_zarr(zarr_dir, consolidated=True, zarr_format=3, **workflow_result.workflow.xr_args)

    raise ValueError(f"Unsupported storage type: {result.storage_type}")
