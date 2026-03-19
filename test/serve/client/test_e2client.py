#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from earth2studio_client.e2client import RemoteEarth2Workflow, RemoteEarth2WorkflowResult
from earth2studio_client.models import InferenceRequestResponse, InferenceRequestResults, RequestStatus


class TestRemoteEarth2WorkflowInitialization:
    def test_initialization_copies_xr_args(self) -> None:
        with patch("earth2studio_client.e2client.Earth2StudioClient"):
            xr_args = {"chunks": {"time": 1}}
            workflow = RemoteEarth2Workflow(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
                xr_args=xr_args,
            )
            assert workflow.xr_args == xr_args
            assert workflow.xr_args is not xr_args

    def test_initialization_passes_client_kwargs(self) -> None:
        with patch("earth2studio_client.e2client.Earth2StudioClient") as mock_client_class:
            RemoteEarth2Workflow(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
                timeout=60.0,
                max_retries=5,
            )
            mock_client_class.assert_called_once_with(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
                timeout=60.0,
                max_retries=5,
            )


class TestRemoteEarth2WorkflowCall:
    def test_call_submits_request(self) -> None:
        with patch("earth2studio_client.e2client.Earth2StudioClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = InferenceRequestResponse(
                execution_id="exec_123",
                status=RequestStatus.ACCEPTED,
                message="Request accepted",
                timestamp=datetime.now(),
            )
            mock_client.submit_inference_request.return_value = mock_response

            workflow = RemoteEarth2Workflow(base_url="http://localhost:8000", workflow_name="test_workflow")
            result = workflow(foo="bar", nsteps=20)

            assert isinstance(result, RemoteEarth2WorkflowResult)
            assert result.execution_id == "exec_123"

            call_args = mock_client.submit_inference_request.call_args
            request = call_args[0][0]
            assert request.parameters["foo"] == "bar"
            assert request.parameters["nsteps"] == 20


class TestRemoteEarth2WorkflowResult:
    def test_result_caching(self) -> None:
        with patch("earth2studio_client.e2client.Earth2StudioClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_inference_result = InferenceRequestResults(
                request_id="exec_123",
                status=RequestStatus.COMPLETED,
                output_files=[],
                completion_time=datetime.now(),
            )
            mock_client.wait_for_completion.return_value = mock_inference_result

            workflow = RemoteEarth2Workflow(base_url="http://localhost:8000", workflow_name="test_workflow")
            result = RemoteEarth2WorkflowResult(workflow, "exec_123")

            r1 = result._get_result()
            r2 = result._get_result()
            assert r1 is r2
            mock_client.wait_for_completion.assert_called_once()


@pytest.mark.parametrize("attr", ["to", "device", "as_model"])
def test_removed_legacy_surface(attr: str) -> None:
    # The wheel client intentionally doesn't expose torch/model integration.
    assert not hasattr(RemoteEarth2Workflow, attr)

