# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np


class StorageType(str, Enum):
    SERVER = "server"
    S3 = "s3"


class RequestStatus(Enum):
    ACCEPTED = "accepted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING_RESULTS = "pending_results"


@dataclass
class InferenceRequest:
    parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        parameters = json.loads(json.dumps(self.parameters, default=InferenceRequest.json_serial))
        return {"parameters": parameters}

    @staticmethod
    def json_serial(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return list(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.datetime64):
            return str(obj)
        if isinstance(obj, np.timedelta64):
            return obj / np.timedelta64(1, "s")
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class InferenceRequestResponse:
    execution_id: str
    status: RequestStatus
    message: str
    timestamp: datetime

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceRequestResponse":
        return cls(
            execution_id=data["execution_id"],
            status=RequestStatus(data["status"]),
            message=data["message"],
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
        )


@dataclass
class ProgressInfo:
    progress: str
    current_step: int
    total_steps: int


@dataclass
class InferenceRequestStatus:
    execution_id: str
    status: RequestStatus
    progress: ProgressInfo | None
    error_message: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceRequestStatus":
        progress_data = data.get("progress")
        progress = (
            ProgressInfo(
                progress=progress_data["progress"],
                current_step=progress_data["current_step"],
                total_steps=progress_data["total_steps"],
            )
            if progress_data is not None
            else None
        )
        return cls(
            execution_id=data["execution_id"],
            status=RequestStatus(data["status"]),
            progress=progress,
            error_message=data.get("error_message"),
        )


@dataclass
class OutputFile:
    path: str
    size: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputFile":
        return cls(path=data["path"], size=data["size"])


@dataclass
class InferenceRequestResults:
    request_id: str
    status: RequestStatus
    output_files: list[OutputFile]
    completion_time: datetime | None
    execution_time_seconds: float | None = None
    storage_type: StorageType = StorageType.SERVER
    signed_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceRequestResults":
        output_files = [OutputFile.from_dict(file_data) for file_data in data["output_files"]]
        completion_timestamp = data.get("completion_time")
        completion_time = (
            datetime.fromisoformat(completion_timestamp.replace("Z", "+00:00"))
            if completion_timestamp is not None
            else None
        )
        return cls(
            request_id=data["request_id"],
            status=RequestStatus(data["status"]),
            output_files=output_files,
            completion_time=completion_time,
            execution_time_seconds=data.get("execution_time_seconds"),
            storage_type=StorageType(data.get("storage_type", StorageType.SERVER.value)),
            signed_url=data.get("signed_url"),
        )

    def result_paths(self) -> list[str]:
        zarr_paths = {f.path for f in self.output_files if (".zarr/" in f.path) or f.path.endswith(".zarr")}
        zarr_paths_sorted = sorted({path[: path.find(".zarr") + len(".zarr")] for path in zarr_paths})
        netcdf_paths = [f.path for f in self.output_files if f.path.endswith(".nc")]
        return zarr_paths_sorted + netcdf_paths


@dataclass
class HealthStatus:
    status: str
    timestamp: datetime

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HealthStatus":
        return cls(
            status=data["status"],
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
        )

