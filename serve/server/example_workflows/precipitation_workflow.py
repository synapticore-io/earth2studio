# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0

"""
Precipitation Forecast Workflow

Couples a prognostic model (FCN/DLWP) with the PrecipitationAFNO diagnostic
to produce total precipitation forecasts via the Earth2Studio REST API.
"""

from datetime import datetime
from typing import Literal

from earth2studio import run
from earth2studio.data import GFS
from earth2studio.io import IOBackend
from earth2studio.models.dx import PrecipitationAFNO
from earth2studio.models.px import DLWP, FCN
from earth2studio.serve.server import Earth2Workflow, workflow_registry


@workflow_registry.register
class PrecipitationWorkflow(Earth2Workflow):
    """Precipitation forecast using prognostic + PrecipitationAFNO diagnostic."""

    name = "precipitation_forecast"
    description = "Precipitation forecast (FCN/DLWP + PrecipitationAFNO diagnostic)"

    def __init__(self, model_type: Literal["fcn", "dlwp"] = "fcn"):
        super().__init__()
        if model_type == "fcn":
            self.model = FCN.load_model(FCN.load_default_package())
        else:
            self.model = DLWP.load_model(DLWP.load_default_package())
        self.diagnostic = PrecipitationAFNO.load_model(
            PrecipitationAFNO.load_default_package()
        )
        self.data = GFS()

    def __call__(
        self,
        io: IOBackend,
        start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
        num_steps: int = 8,
    ) -> None:
        """Run precipitation diagnostic forecast.

        Parameters
        ----------
        io : IOBackend
            Output backend, provided by the serve framework.
        start_time : list[datetime]
            Forecast initialization times (UTC).
        num_steps : int
            Number of forecast steps (each ~6h for FCN/DLWP).
        """
        run.diagnostic(
            start_time,
            num_steps,
            self.model,
            self.diagnostic,
            self.data,
            io,
        )
