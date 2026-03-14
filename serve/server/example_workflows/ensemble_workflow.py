# SPDX-FileCopyrightText: Copyright (c) 2026 Synapticore.
# SPDX-License-Identifier: Apache-2.0

"""
Ensemble Forecast Workflow

Runs an ensemble forecast via the Earth2Studio REST API.
Perturbs initial conditions with SphericalGaussian noise,
produces multiple ensemble members with spread information.
"""

from datetime import datetime
from typing import Literal

from earth2studio import run
from earth2studio.data import GFS
from earth2studio.io import IOBackend
from earth2studio.models.px import DLWP, FCN
from earth2studio.perturbation import SphericalGaussian
from earth2studio.serve.server import Earth2Workflow, workflow_registry


@workflow_registry.register
class EnsembleWorkflow(Earth2Workflow):
    """Ensemble forecast with SphericalGaussian perturbation."""

    name = "ensemble_forecast"
    description = "Ensemble weather forecast (FCN/DLWP) with uncertainty spread"

    def __init__(self, model_type: Literal["fcn", "dlwp"] = "fcn"):
        super().__init__()
        if model_type == "fcn":
            self.model = FCN.load_model(FCN.load_default_package())
        else:
            self.model = DLWP.load_model(DLWP.load_default_package())
        self.data = GFS()

    def __call__(
        self,
        io: IOBackend,
        start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
        num_steps: int = 10,
        num_ensemble: int = 8,
        noise_amplitude: float = 0.15,
    ) -> None:
        """Run ensemble forecast.

        Parameters
        ----------
        io : IOBackend
            Output backend, provided by the serve framework.
        start_time : list[datetime]
            Forecast initialization times (UTC).
        num_steps : int
            Number of forecast steps (each ~6h for FCN/DLWP).
        num_ensemble : int
            Number of ensemble members.
        noise_amplitude : float
            SphericalGaussian perturbation amplitude.
        """
        perturbation = SphericalGaussian(noise_amplitude=noise_amplitude)
        run.ensemble(
            start_time,
            num_steps,
            num_ensemble,
            self.model,
            self.data,
            io,
            perturbation,
        )
