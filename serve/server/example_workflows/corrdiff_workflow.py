# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0

"""
CorrDiff Downscaling Workflow

Generative super-resolution over Taiwan using the CorrDiff diffusion model.
Downscales 0.25° global GFS data to ~3km regional resolution.
"""

from collections import OrderedDict
from datetime import datetime

import torch

from earth2studio.data import GFS, prep_data_array
from earth2studio.io import IOBackend
from earth2studio.models.dx import CorrDiffTaiwan
from earth2studio.serve.server import Earth2Workflow, workflow_registry
from earth2studio.utils.coords import map_coords, split_coords
from earth2studio.utils.time import to_time_array


@workflow_registry.register
class CorrDiffWorkflow(Earth2Workflow):
    """CorrDiff generative downscaling for Taiwan (~3km)."""

    name = "corrdiff_taiwan"
    description = "CorrDiff generative downscaling (0.25° → 3km, Taiwan region)"

    def __init__(self):
        super().__init__()
        self.corrdiff = CorrDiffTaiwan.load_model(
            CorrDiffTaiwan.load_default_package()
        )
        self.data = GFS()

    def __call__(
        self,
        io: IOBackend,
        start_time: list[datetime] = [datetime(2023, 10, 4, 18)],
        number_of_samples: int = 1,
    ) -> None:
        """Run CorrDiff downscaling.

        Parameters
        ----------
        io : IOBackend
            Output backend, provided by the serve framework.
        start_time : list[datetime]
            Times to downscale (UTC). Default is Typhoon Koinu case.
        number_of_samples : int
            Number of diffusion samples per timestep.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        corrdiff = self.corrdiff.to(device)
        corrdiff.number_of_samples = number_of_samples

        time = to_time_array(start_time)
        x, coords = prep_data_array(
            self.data(time, corrdiff.input_coords()["variable"]), device=device
        )
        x, coords = map_coords(x, coords, corrdiff.input_coords())

        output_coords = corrdiff.output_coords(corrdiff.input_coords())
        total_coords = OrderedDict(
            {
                "time": coords["time"],
                "sample": output_coords["sample"],
                "lat": output_coords["lat"],
                "lon": output_coords["lon"],
            }
        )
        io.add_array(total_coords, output_coords["variable"])

        x, coords = corrdiff(x, coords)
        io.write(*split_coords(x, coords))
