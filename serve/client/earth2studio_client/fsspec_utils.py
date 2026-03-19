# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NoReturn
from urllib.parse import parse_qs, urlencode, urlparse

import fsspec

from .models import InferenceRequestResults, StorageType


class SignedURLFileSystem(fsspec.AbstractFileSystem):
    def __init__(self, base_fs: Any, query_params: dict[str, str], base_url: str) -> None:
        super().__init__()
        self._fs = base_fs
        self._query_params = query_params
        self._base_url = base_url
        self._query_string = urlencode(query_params, safe="~")

    def _make_signed_path(self, path: str) -> str:
        if path.startswith("http"):
            full_url = path
        else:
            clean_path = path.lstrip("/")
            full_url = f"{self._base_url}/{clean_path}" if clean_path else self._base_url
        separator = "&" if "?" in full_url else "?"
        return f"{full_url}{separator}{self._query_string}"

    def _handle_403(self, e: BaseException, path: str) -> "NoReturn":
        error_str = str(e).lower()
        if "403" in str(e) or "forbidden" in error_str:
            raise FileNotFoundError(f"File not found: {path}") from None
        raise e

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        try:
            return self._fs._open(self._make_signed_path(path), mode=mode, **kwargs)
        except Exception as e:
            self._handle_403(e, path)

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any) -> Any:
        try:
            return self._fs.cat_file(self._make_signed_path(path), start=start, end=end, **kwargs)
        except Exception as e:
            self._handle_403(e, path)

    def _cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any) -> Any:
        return self.cat_file(path, start=start, end=end, **kwargs)

    def info(self, path: str, **kwargs: Any) -> Any:
        try:
            return self._fs.info(self._make_signed_path(path), **kwargs)
        except Exception as e:
            self._handle_403(e, path)

    def exists(self, path: str, **kwargs: Any) -> bool:
        try:
            return self._fs.exists(self._make_signed_path(path), **kwargs)
        except Exception as e:
            try:
                self._handle_403(e, path)
            except FileNotFoundError:
                return False


def create_cloudfront_mapper(signed_url: str, zarr_path: str = "") -> Any:
    parsed = urlparse(signed_url)
    query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    base_path = parsed.path.rstrip("/*").rstrip("*")
    base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}/{zarr_path}" if zarr_path else f"{parsed.scheme}://{parsed.netloc}{base_path}"
    fs = fsspec.filesystem("https")
    signed_fs = SignedURLFileSystem(fs, query_params, base_url)
    return fsspec.mapping.FSMap(root="", fs=signed_fs, check=False, create=False)


def get_mapper(request_result: InferenceRequestResults, zarr_path: str = "") -> Any | None:
    if request_result.storage_type == StorageType.S3:
        if request_result.signed_url is None:
            raise ValueError("S3 storage type requires a signed URL")
        return create_cloudfront_mapper(request_result.signed_url, zarr_path)
    if request_result.storage_type == StorageType.SERVER:
        return None
    raise ValueError(f"Unsupported storage type: {request_result.storage_type}")

