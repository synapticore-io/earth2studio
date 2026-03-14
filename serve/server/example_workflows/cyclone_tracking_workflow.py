# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0

"""
Cyclone Tracking Workflow

Runs SFNO global forecast coupled with TCTrackerWuDuan to detect and track
tropical cyclones. Returns track paths (lat, lon, intensity per step).
"""

import json
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch

from earth2studio.data import GFS, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.dx import TCTrackerWuDuan
from earth2studio.models.px import SFNO
from earth2studio.serve.server import Earth2Workflow, workflow_registry
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array


@workflow_registry.register
class CycloneTrackingWorkflow(Earth2Workflow):
    """SFNO forecast + TCTrackerWuDuan cyclone tracking."""

    name = "cyclone_tracking"
    description = "Tropical cyclone tracking (SFNO + TCTrackerWuDuan)"

    def __init__(self):
        super().__init__()
        self.prognostic = SFNO.load_model(SFNO.load_default_package())
        self.tracker = TCTrackerWuDuan()
        self.data = GFS()

    def __call__(
        self,
        io: IOBackend,
        start_time: list[datetime] = [datetime(2024, 9, 1, 0)],
        num_steps: int = 16,
    ) -> None:
        """Run cyclone tracking forecast.

        Parameters
        ----------
        io : IOBackend
            Output backend, provided by the serve framework.
        start_time : list[datetime]
            Forecast initialization time (UTC).
        num_steps : int
            Number of forecast steps (each ~6h for SFNO).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prognostic = self.prognostic.to(device)
        tracker = self.tracker.to(device)
        tracker.reset_path_buffer()

        time = to_time_array(start_time)
        x, coords = fetch_data(
            source=self.data,
            time=time,
            variable=prognostic.input_coords()["variable"],
            lead_time=prognostic.input_coords().get("lead_time"),
            device=device,
        )

        # Run prognostic + tracker loop
        model_iter = prognostic.create_iterator(x, coords)
        for step, (x_step, coords_step) in enumerate(model_iter):
            x_track, coords_track = map_coords(
                x_step, coords_step, tracker.input_coords()
            )
            output, output_coords = tracker(x_track, coords_track)
            if step >= num_steps:
                break

        # output shape: [batch, lead_time, path, step, variable]
        # variable order: lat, lon, ...
        tracks = output.cpu()

        # Write full forecast fields from last step to IO
        total_coords = OrderedDict(
            {
                k: v
                for k, v in coords_step.items()
                if k != "batch" and v.shape != (0,)
            }
        )
        io.add_array(total_coords, "forecast")
        io.write(x_step, coords_step, "forecast")

        # Write tracks as a separate array
        track_np = tracks.numpy()
        n_paths = track_np.shape[1]
        n_track_steps = track_np.shape[2]
        n_vars = track_np.shape[3]

        track_coords = OrderedDict(
            {
                "path": np.arange(n_paths),
                "track_step": np.arange(n_track_steps),
                "track_variable": np.array(
                    output_coords["variable"][:n_vars]
                    if "variable" in output_coords
                    else [f"v{i}" for i in range(n_vars)]
                ),
            }
        )
        io.add_array(track_coords, "tracks")
        io.write(
            torch.from_numpy(track_np[0]),  # remove batch dim
            track_coords,
            "tracks",
        )
