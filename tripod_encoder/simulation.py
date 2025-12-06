from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from qutip import Options, Qobj, ket2dm, mesolve

from .logging_utils import setup_logger
from .lindblad import build_collapse_operators
from .operators import (
    Pg_0,
    Pg_m1,
    Pg_p1,
    Pe,
    basis_g0,
    basis_gm1,
    basis_gp1,
    sigma_e_gm1,
    sigma_e_g0,
    sigma_e_gp1,
)
from .params import ControlParameters, PhysicalParameters, SimulationParameters
from .pulses import Omega_pi, Omega_sigma_minus, Omega_sigma_plus, PulseNoise

logger = setup_logger(__name__)


def logical_state(alpha: complex, beta: complex, phi: float) -> Qobj:
    state = alpha * basis_gm1() + beta * np.exp(1j * phi) * basis_gp1()
    return state.unit()


def _detuning_value(detuning, t: float) -> float:
    return float(detuning(t)) if callable(detuning) else float(detuning)


def _hamiltonian_list(
    physical_params: PhysicalParameters,
    control_params: ControlParameters,
    alpha: complex,
    beta: complex,
    phi: float,
    mode: str,
    noise: Optional[PulseNoise],
):
    H_list = []
    H_static = physical_params.Delta * Pe

    proj_det_list = [
        (Pg_m1, physical_params.delta_minus1),
        (Pg_0, physical_params.delta_0),
        (Pg_p1, physical_params.delta_plus1),
    ]

    for projector, detuning in proj_det_list:
        if callable(detuning):
            logger.info("Adding time-dependent detuning for projector %s", projector)
            H_list.append([projector, lambda t, args, det=detuning: _detuning_value(det, t)])
        else:
            H_static = H_static + float(detuning) * projector

    H_list.insert(0, H_static)

    def Omega_plus_t(t, args):
        if mode == "projection":
            return 0.0
        return Omega_sigma_plus(t, control_params, alpha, beta, phi, noise)

    def Omega_minus_t(t, args):
        if mode == "projection":
            return 0.0
        return Omega_sigma_minus(t, control_params, alpha, beta, phi, noise)

    def Omega_pi_t(t, args):
        if mode == "projection":
            return 0.0
        return Omega_pi(t, control_params, noise)

    H_sigma_plus = -0.5 * (sigma_e_gm1 + sigma_e_gm1.dag())
    H_sigma_minus = -0.5 * (sigma_e_gp1 + sigma_e_gp1.dag())
    H_pi = -0.5 * (sigma_e_g0 + sigma_e_g0.dag())

    H_list.extend(
        [
            [H_sigma_plus, Omega_plus_t],
            [H_sigma_minus, Omega_minus_t],
            [H_pi, Omega_pi_t],
        ]
    )
    logger.info("Hamiltonian list constructed with %s terms", len(H_list))
    return H_list


def run_tripod_sequence(
    alpha: complex,
    beta: complex,
    phi: float,
    mode: str,
    physical_params: PhysicalParameters,
    control_params: ControlParameters,
    sim_params: SimulationParameters,
    enable_channels: Optional[Dict[str, bool]] = None,
    noise: Optional[PulseNoise] = None,
):
    try:
        logger.info("Starting tripod sequence")
        logger.info("Mode: %s", mode)
        logger.info("alpha=%s, beta=%s, phi=%s", alpha, beta, phi)

        if enable_channels is None:
            enable_channels = {
                "spontaneous_emission": True,
                "dephasing": True,
                "ground_relaxation": True,
                "raman": True,
            }

        noise = noise or (
            PulseNoise.sample(control_params)
            if (control_params.sigma_eps or control_params.sigma_phi)
            else None
        )

        H = _hamiltonian_list(
            physical_params=physical_params,
            control_params=control_params,
            alpha=alpha,
            beta=beta,
            phi=phi,
            mode=mode,
            noise=noise,
        )

        c_ops = build_collapse_operators(physical_params, enable_channels)

        if mode == "projection":
            rho0 = ket2dm(logical_state(alpha, beta, phi))
        elif mode == "stirap":
            rho0 = ket2dm(basis_g0())
        else:
            raise ValueError("mode must be 'projection' or 'stirap'")

        tlist = sim_params.tlist
        logger.info(
            "Time grid: %s steps from %0.2e to %0.2e", len(tlist), tlist[0], tlist[-1]
        )
        if len(tlist) < 10:
            logger.warning("tlist is extremely short — check n_steps")

        opt_kwargs = {}
        if sim_params.atol is not None:
            opt_kwargs["atol"] = sim_params.atol
        if sim_params.rtol is not None:
            opt_kwargs["rtol"] = sim_params.rtol
        options = Options(**opt_kwargs) if opt_kwargs else None

        logger.info("Calling mesolve")
        result = mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=None, options=options)
        logger.info("mesolve completed successfully")

        return {
            "result": result,
            "tlist": tlist,
            "noise": noise,
            "collapse_ops": c_ops,
        }
    except Exception:
        logger.exception("run_tripod_sequence FAILED")
        raise
