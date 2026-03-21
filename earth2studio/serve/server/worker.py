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

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import redis  # type: ignore[import-untyped]
except ImportError:
    OptionalDependencyFailure("serve")
    redis = None
    redis_client = None

from earth2studio.serve.server.config import get_config, get_config_manager
from earth2studio.serve.server.utils import (
    get_inference_request_output_path_key,
    queue_next_stage,
)
from earth2studio.serve.server.workflow import (
    Workflow,
    WorkflowStatus,
    workflow_registry,
)

config_manager = get_config_manager()
config = get_config()

if redis:
    redis_client = redis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        password=config.redis.password,
        decode_responses=config.redis.decode_responses,
        socket_connect_timeout=config.redis.socket_connect_timeout,
        socket_timeout=config.redis.socket_timeout,
    )

# Configure logging
config_manager.setup_logging()
logger = logging.getLogger(__name__)

# Path configuration from config
DEFAULT_OUTPUT_DIR = Path(config.paths.default_output_dir)
RESULTS_ZIP_DIR = Path(config.paths.results_zip_dir)

# Model registry for caching loaded models
model_registry: dict[str, Any] = {}

# Register custom workflows in the worker process
try:
    from earth2studio.serve.server.workflow import register_all_workflows

    register_all_workflows(redis_client)
    logger.info("Custom workflows registered successfully in worker process")
except ImportError:
    logger.warning(
        "Workflow registration module not found in worker, skipping custom workflow registration"
    )
except Exception as e:
    logger.error(f"Failed to register custom workflows in worker: {e}")
    # Don't raise - worker can still handle other tasks


def _finalize_inline(
    workflow_class: type[Workflow],
    workflow_name: str,
    execution_id: str,
    output_path: Path,
    execution_time_seconds: float,
    log: logging.LoggerAdapter,
) -> None:
    """Finalize workflow results directly without queuing through RQ pipeline.

    Builds the file manifest, writes metadata JSON, and sets status to COMPLETED
    in a single step. Used when neither result_zip nor object_storage is enabled.
    """
    from earth2studio.serve.server.cpu_worker import (
        ResultMetadata,
        build_file_manifest,
    )

    # Build file manifest
    file_manifest = build_file_manifest(output_path)
    log.info(f"Built manifest: {len(file_manifest)} files")

    # Get execution data for metadata and patch in execution time
    execution_data = workflow_class._get_execution_data(
        redis_client, workflow_name, execution_id
    )
    execution_data.execution_time_seconds = execution_time_seconds

    # Build and write metadata
    now = datetime.now(timezone.utc)
    metadata = ResultMetadata.from_workflow_result(
        workflow_result=execution_data,
        request_id=f"{workflow_name}:{execution_id}",
        file_manifest=file_manifest,
        zip_created_at=now.isoformat().replace("+00:00", "Z"),
    )
    metadata_path = RESULTS_ZIP_DIR / f"metadata_{workflow_name}:{execution_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    import json as json_mod

    with open(metadata_path, "w") as f:
        json_mod.dump(metadata.to_dict(), f, indent=2)

    # Expose output dir for GET .../results/{filepath} (same key as zip pipeline)
    request_id = f"{workflow_name}:{execution_id}"
    output_path_key = get_inference_request_output_path_key(request_id)
    redis_client.setex(output_path_key, 86400, str(output_path))

    # Set COMPLETED
    updates: dict[str, Any] = {
        "status": WorkflowStatus.COMPLETED,
        "end_time": now.isoformat(),
        "execution_time_seconds": execution_time_seconds,
    }
    workflow_class._update_execution_data(
        redis_client, workflow_name, execution_id, updates
    )
    log.info(f"Fast-path finalize complete for {workflow_name}:{execution_id}")


def get_output_path(
    io_config: dict[str, Any] | None,
    timestamp: str,
    workflow_type: str,
    request_id: str,
) -> Path:
    """
    Generate output path for workflow results.

    Parameters
    ----------
    io_config : dict or None
        I/O configuration (used for backend_type); if None, backend_type defaults to "zarr".
    timestamp : str
        Timestamp string used in the output directory name.
    workflow_type : str
        Type of workflow (e.g. workflow name).
    request_id : str
        Request identifier.

    Returns
    -------
    Path
        Path to the forecast output file (e.g. forecast.zarr).
    """
    # Create timestamp-based subdirectory
    timestamp_str = timestamp.replace(":", "").replace("Z", "").replace("+", "")
    output_dir = DEFAULT_OUTPUT_DIR / f"{workflow_type}_{timestamp_str}_{request_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Default file name
    backend_type = io_config.get("backend_type", "zarr") if io_config else "zarr"
    return output_dir / f"forecast.{backend_type}"


@check_optional_dependencies()
def run_custom_workflow(
    workflow_name: str, execution_id: str, parameters: dict[str, Any]
) -> Any:
    """
    RQ worker function to run a custom workflow.

    Executes the workflow, updates execution status in Redis, and queues the
    next pipeline stage (e.g. result_zip or object_storage) as configured.

    Parameters
    ----------
    workflow_name : str
        Name of the registered workflow to run.
    execution_id : str
        Unique execution identifier.
    parameters : dict
        Workflow input parameters (validated by the workflow's parameter type).

    Returns
    -------
    Any
        Result returned by the workflow's run() method.

    Raises
    ------
    ValueError
        If the workflow is not found or cannot be instantiated.
    RuntimeError
        If queuing the next pipeline stage fails.
    """
    # Create logger adapter with execution_id for automatic log prefixing
    log = logging.LoggerAdapter(logger, {"execution_id": execution_id})

    log.info(f"Starting custom workflow {workflow_name}")

    # Get workflow class from registry
    workflow_class = workflow_registry.get_workflow_class(workflow_name)
    if not workflow_class:
        raise ValueError(f"Custom workflow '{workflow_name}' not found in registry")

    # Create workflow instance for execution
    custom_workflow = workflow_registry.get(workflow_name, redis_client=redis_client)
    if custom_workflow is None:
        raise ValueError(f"Custom workflow '{workflow_name}' could not be instantiated")

    try:
        start_timestamp = time.time()
        start_time = datetime.now(timezone.utc).isoformat()
        updates: dict[str, Any] = {
            "status": WorkflowStatus.RUNNING,
            "start_time": start_time,
        }
        workflow_class._update_execution_data(
            redis_client, workflow_name, execution_id, updates
        )
        log.info(f"Executing workflow {workflow_name} with execution ID {execution_id}")

        result = custom_workflow.run(parameters, execution_id)

        # Record execution time
        execution_time_seconds = time.time() - start_timestamp
        output_path = custom_workflow.get_output_path(execution_id)

        # Fast path: finalize directly when no zip and no object storage needed
        if not config.paths.result_zip_enabled and not config.object_storage.enabled:
            log.info(
                f"Fast-path finalize for {workflow_name}:{execution_id} "
                f"(no zip, no object storage)"
            )
            _finalize_inline(
                workflow_class, workflow_name, execution_id,
                output_path, execution_time_seconds, log,
            )
            return result

        # Standard path: queue through pipeline stages
        updates = {
            "status": WorkflowStatus.PENDING_RESULTS,
            "execution_time_seconds": execution_time_seconds,
        }
        workflow_class._update_execution_data(
            redis_client, workflow_name, execution_id, updates
        )
        log.info(f"Workflow {workflow_name} execution {execution_id} pending results")

        job_id = queue_next_stage(
            redis_client=redis_client,
            current_stage="inference",
            workflow_name=workflow_name,
            execution_id=execution_id,
            output_path_str=str(output_path),
            results_zip_dir_str=str(RESULTS_ZIP_DIR),
        )
        if not job_id:
            error_msg = f"Failed to queue next pipeline stage for {workflow_name}:{execution_id}"
            log.error(error_msg)
            try:
                updates = {
                    "status": WorkflowStatus.FAILED,
                    "end_time": datetime.now(timezone.utc).isoformat(),
                    "error_message": error_msg,
                }
                workflow_class._update_execution_data(
                    redis_client, workflow_name, execution_id, updates
                )
            except Exception:
                log.exception("Failed to update workflow status after queue failure")
            raise RuntimeError(error_msg)

        logger.info(
            f"Queued next stage for {workflow_name}:{execution_id} with RQ job ID: {job_id}"
        )
        return result

    except Exception as e:
        log.exception(
            f"Custom workflow {workflow_name} execution {execution_id} failed"
        )
        # Update workflow status to failed if possible
        try:
            updates = {
                "status": WorkflowStatus.FAILED,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error_message": str(e),
            }
            workflow_class._update_execution_data(
                redis_client, workflow_name, execution_id, updates
            )
            log.info(
                f"Workflow {workflow_name} execution {execution_id} failed with error: {str(e)}"
            )
        except Exception:
            log.exception("Failed to update workflow status after failure")
        raise e  # Re-raise for RQ to handle
