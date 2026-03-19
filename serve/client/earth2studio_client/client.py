# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import time
from typing import Any, cast
from urllib.parse import urljoin

import requests  # type: ignore[import-untyped]
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from requests.packages.urllib3.util.retry import Retry  # type: ignore[import-untyped]

from .exceptions import APIConnectionError as ClientConnectionError
from .exceptions import (
    BadRequestError,
    Earth2StudioAPIError,
    InferenceRequestNotFoundError,
    InternalServerError,
    RequestTimeoutError,
)
from .models import (
    HealthStatus,
    InferenceRequest,
    InferenceRequestResponse,
    InferenceRequestResults,
    InferenceRequestStatus,
    RequestStatus,
    StorageType,
)


class Earth2StudioClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        workflow_name: str = "deterministic_earth2_workflow",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.3,
        token: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.workflow_name = workflow_name
        self.timeout = timeout
        self.token = token

        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: dict | None = None,
        params: dict | None = None,
        return_response: bool = False,
        stream: bool = False,
        timeout: float | None = None,
    ) -> dict | requests.Response:
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        timeout = self.timeout if timeout is None else timeout
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
                timeout=timeout,
                stream=stream,
            )
        except requests.exceptions.Timeout:
            raise RequestTimeoutError(f"Request to {url} timed out after {timeout} seconds")
        except requests.exceptions.ConnectionError as e:
            raise ClientConnectionError(f"Failed to connect to {url}: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise Earth2StudioAPIError(f"Request failed: {str(e)}")

        if response.status_code == 400:
            error_data = self._parse_error_response(response)
            raise BadRequestError(error_data.get("error", "Bad request"), details=error_data.get("details"))
        if response.status_code == 404:
            error_data = self._parse_error_response(response)
            raise InferenceRequestNotFoundError(
                error_data.get("error", "Not found"), details=error_data.get("details")
            )
        if response.status_code == 500:
            error_data = self._parse_error_response(response)
            raise InternalServerError(
                error_data.get("error", "Internal server error"), details=error_data.get("details")
            )
        if not response.ok:
            error_data = self._parse_error_response(response)
            raise Earth2StudioAPIError(
                error_data.get("error", f"HTTP {response.status_code} error"),
                status_code=response.status_code,
                details=error_data.get("details"),
            )

        if return_response:
            return response

        try:
            return response.json()
        except json.JSONDecodeError:
            raise Earth2StudioAPIError("Invalid JSON response from server")

    def _parse_error_response(self, response: requests.Response) -> dict:
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"error": response.text or f"HTTP {response.status_code} error"}

    def health_check(self) -> HealthStatus:
        response_data = self._make_request("GET", "/health")
        return HealthStatus.from_dict(cast(dict[str, Any], response_data))

    def submit_inference_request(self, request: InferenceRequest) -> InferenceRequestResponse:
        response_data = self._make_request("POST", f"/v1/infer/{self.workflow_name}", json_data=request.to_dict())
        return InferenceRequestResponse.from_dict(cast(dict[str, Any], response_data))

    def get_request_status(self, request_id: str) -> InferenceRequestStatus:
        response_data = self._make_request("GET", f"/v1/infer/{self.workflow_name}/{request_id}/status")
        return InferenceRequestStatus.from_dict(cast(dict[str, Any], response_data))

    def get_request_results(self, request_id: str, timeout: float | None = None) -> InferenceRequestResults:
        response = cast(
            requests.Response,
            self._make_request(
                method="GET",
                endpoint=f"/v1/infer/{self.workflow_name}/{request_id}/results",
                return_response=True,
                timeout=timeout,
            ),
        )

        if response.status_code == 202:
            try:
                error_data = response.json()
                raise Earth2StudioAPIError(
                    error_data.get("message", "Request is still processing"),
                    status_code=202,
                    details=error_data.get("status"),
                )
            except json.JSONDecodeError:
                raise Earth2StudioAPIError("Request is still processing", status_code=202)

        response_data = response.json()
        return InferenceRequestResults.from_dict(response_data)

    def wait_for_completion(
        self, request_id: str, poll_interval: float = 5.0, timeout: float | None = None
    ) -> InferenceRequestResults:
        start_time = time.time()
        while True:
            status = self.get_request_status(request_id)
            if status.status == RequestStatus.COMPLETED:
                return self.get_request_results(request_id, timeout)
            if status.status == RequestStatus.FAILED:
                raise Earth2StudioAPIError(f"Inference request {request_id} failed: {status.error_message}")
            if status.status == RequestStatus.CANCELLED:
                raise Earth2StudioAPIError("Inference request was cancelled")

            if timeout is not None and (time.time() - start_time) > timeout:
                raise RequestTimeoutError(f"Request {request_id} did not complete within {timeout} seconds")
            time.sleep(poll_interval)

    def run_inference_sync(
        self, request: InferenceRequest, poll_interval: float = 5.0, timeout: float | None = None
    ) -> InferenceRequestResults:
        response = self.submit_inference_request(request)
        return self.wait_for_completion(response.execution_id, poll_interval, timeout)

    def result_root_path(self, result: InferenceRequestResults) -> str:
        return f"/v1/infer/{self.workflow_name}/{result.request_id}/results/"

    def download_result(self, result: InferenceRequestResults, path: str, timeout: float | None = None) -> io.BytesIO:
        if result.storage_type == StorageType.S3:
            if not result.signed_url:
                raise Earth2StudioAPIError("S3 storage type requires a signed URL")

            from .fsspec_utils import create_cloudfront_mapper

            mapper = create_cloudfront_mapper(result.signed_url, zarr_path="")
            parts = path.split("/")
            if len(parts) < 2:
                raise Earth2StudioAPIError(
                    f"Expected S3 result path to include an execution-id prefix, got: {path!r}"
                )
            path = "/".join(parts[1:])
            content = mapper.fs.cat_file(path)
            return io.BytesIO(content)

        response = cast(
            requests.Response,
            self._make_request(
                method="GET",
                endpoint=f"{self.result_root_path(result)}{path}",
                return_response=True,
                stream=True,
                timeout=timeout,
            ),
        )
        return io.BytesIO(response.content)

    def close(self) -> None:
        self.session.close()  # type: ignore[no-untyped-call]

    def __enter__(self) -> "Earth2StudioClient":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        self.close()

