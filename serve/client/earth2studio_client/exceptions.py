# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class Earth2StudioAPIError(Exception):
    """Base exception for Earth2Studio API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details

    def __str__(self) -> str:
        error_msg = f"Earth2Studio API Error: {self.message}"
        if self.status_code:
            error_msg += f" (HTTP {self.status_code})"
        if self.details:
            error_msg += f" - {self.details}"
        return error_msg


class BadRequestError(Earth2StudioAPIError):
    def __init__(self, message: str, details: str | None = None):
        super().__init__(message, status_code=400, details=details)


class InferenceRequestNotFoundError(Earth2StudioAPIError):
    def __init__(self, message: str, details: str | None = None):
        super().__init__(message, status_code=404, details=details)


class InternalServerError(Earth2StudioAPIError):
    def __init__(self, message: str, details: str | None = None):
        super().__init__(message, status_code=500, details=details)


class RequestTimeoutError(Earth2StudioAPIError):
    def __init__(self, message: str = "Request timed out"):
        super().__init__(message)


class APIConnectionError(Earth2StudioAPIError):
    def __init__(self, message: str = "Failed to connect to Earth2Studio API"):
        super().__init__(message)

