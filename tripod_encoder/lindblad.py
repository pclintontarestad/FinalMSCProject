from __future__ import annotations

from typing import Dict, List

import numpy as np
from qutip import Qobj

from .logging_utils import setup_logger
from .operators import basis_e, basis_gm1, basis_g0, basis_gp1
from .params import PhysicalParameters

logger = setup_logger(__name__)


GROUND_KETS = {"g-1": basis_gm1(), "g0": basis_g0(), "g+1": basis_gp1()}
EXCITED_BRA = basis_e().dag()


def spontaneous_emission_ops(params: PhysicalParameters) -> List[Qobj]:
    ops: List[Qobj] = []
    for label, ket in GROUND_KETS.items():
        gamma = params.Gamma_branch.get(label, 0.0)
        if gamma > 0:
            ops.append(np.sqrt(gamma) * ket * EXCITED_BRA)
    logger.info("Spontaneous emission operators: %s", len(ops))
    return ops


def dephasing_ops(params: PhysicalParameters) -> List[Qobj]:
    ops: List[Qobj] = []
    for (i, j), rate in params.gamma_dephasing_pairs.items():
        if rate <= 0:
            continue
        ket_i = GROUND_KETS.get(i)
        ket_j = GROUND_KETS.get(j)
        if ket_i is None or ket_j is None:
            continue
        proj_i = ket_i * ket_i.dag()
        proj_j = ket_j * ket_j.dag()
        ops.append(np.sqrt(rate / 2.0) * (proj_i - proj_j))
    logger.info("Dephasing operators: %s", len(ops))
    return ops


def ground_relaxation_ops(params: PhysicalParameters) -> List[Qobj]:
    ops: List[Qobj] = []
    if params.Gamma1 is None or params.Gamma1 <= 0:
        return ops
    p_target_dist = params.p_th or {"g-1": 0.0, "g0": 0.0, "g+1": 0.0}
    for i, ket_i in GROUND_KETS.items():
        for j, ket_j in GROUND_KETS.items():
            if i == j:
                continue
            p_target = p_target_dist.get(j, 0.0)
            if p_target <= 0:
                continue
            ops.append(np.sqrt(params.Gamma1 * p_target) * ket_j * ket_i.dag())
    logger.info("Ground relaxation operators: %s", len(ops))
    return ops


def raman_ops(params: PhysicalParameters) -> List[Qobj]:
    ops: List[Qobj] = []
    for (i, j), rate in params.gamma_raman.items():
        if rate <= 0:
            continue
        ket_i = GROUND_KETS.get(i)
        ket_j = GROUND_KETS.get(j)
        if ket_i is None or ket_j is None:
            continue
        ops.append(np.sqrt(rate) * ket_j * ket_i.dag())
    logger.info("Raman scattering operators: %s", len(ops))
    return ops


def build_collapse_operators(
    params: PhysicalParameters,
    enable_channels: Dict[str, bool],
) -> List[Qobj]:
    ops: List[Qobj] = []
    if enable_channels.get("spontaneous_emission", True):
        ops.extend(spontaneous_emission_ops(params))
    if enable_channels.get("dephasing", True):
        ops.extend(dephasing_ops(params))
    if enable_channels.get("ground_relaxation", True):
        ops.extend(ground_relaxation_ops(params))
    if enable_channels.get("raman", True):
        ops.extend(raman_ops(params))
    if not ops:
        logger.warning("No collapse operators enabled")
    else:
        logger.info("Total collapse operators: %s", len(ops))
    for L in ops:
        if L.shape != (4, 4):
            logger.error("Collapse operator has wrong dimension: %s", L.shape)
    return ops
