from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .params import ControlParameters
from .logging_utils import setup_logger

logger = setup_logger(__name__)


def probe_envelope(t: float, cp: ControlParameters) -> float:
    return float(np.exp(-((t - cp.t0_probe) ** 2) / (2 * cp.tau_probe**2)))


def control_envelope(t: float, cp: ControlParameters) -> float:
    return float(np.exp(-((t - cp.t0_control) ** 2) / (2 * cp.tau_control**2)))


@dataclass
class PulseNoise:
    amplitude: Dict[str, float]
    phase: Dict[str, float]

    @classmethod
    def sample(cls, cp: ControlParameters, rng: Optional[np.random.Generator] = None) -> "PulseNoise":
        rng = rng or np.random.default_rng()
        eps = rng.normal(scale=cp.sigma_eps, size=3)
        phi = rng.normal(scale=cp.sigma_phi, size=3)
        fields = ["sigma_plus", "sigma_minus", "pi"]
        amplitude = {f: 1.0 + eps[i] for i, f in enumerate(fields)}
        phase = {f: float(phi[i]) for i, f in enumerate(fields)}
        return cls(amplitude=amplitude, phase=phase)


def _apply_noise(value: complex, noise: Optional[PulseNoise], field: str) -> complex:
    if noise is None:
        return value
    return value * noise.amplitude.get(field, 1.0) * np.exp(1j * noise.phase.get(field, 0.0))


def Omega_sigma_plus(
    t: float,
    cp: ControlParameters,
    alpha: complex,
    beta: complex,
    phi: float,
    noise: Optional[PulseNoise] = None,
) -> complex:
    value = alpha * cp.Omega_sigma_plus_max * probe_envelope(t, cp)
    if not np.isfinite(value):
        logger.warning("Non-finite Omega_sigma_plus at t=%s", t)
    return _apply_noise(value, noise, "sigma_plus")


def Omega_sigma_minus(
    t: float,
    cp: ControlParameters,
    alpha: complex,
    beta: complex,
    phi: float,
    noise: Optional[PulseNoise] = None,
) -> complex:
    value = beta * np.exp(1j * phi) * cp.Omega_sigma_minus_max * probe_envelope(t, cp)
    if not np.isfinite(value):
        logger.warning("Non-finite Omega_sigma_minus at t=%s", t)
    return _apply_noise(value, noise, "sigma_minus")


def Omega_pi(
    t: float,
    cp: ControlParameters,
    noise: Optional[PulseNoise] = None,
) -> complex:
    value = cp.Omega_pi_max * control_envelope(t, cp)
    if not np.isfinite(value):
        logger.warning("Non-finite Omega_pi at t=%s", t)
    return _apply_noise(value, noise, "pi")
