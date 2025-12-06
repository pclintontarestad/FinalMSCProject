from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from qutip import Qobj, fidelity

from .logging_utils import setup_logger
from .operators import Pg_m1, Pg_p1, basis_gm1, basis_gp1

logger = setup_logger(__name__)

P_LOGICAL = Pg_m1 + Pg_p1


def logical_projection(rho: Qobj) -> Tuple[Qobj, float]:
    rho_logical = P_LOGICAL * rho * P_LOGICAL
    trace_logical = float(rho_logical.tr())
    logger.info("Logical trace: %s", trace_logical)
    return rho_logical, trace_logical


def logical_fidelity(rho: Qobj, target: Qobj) -> Dict[str, float]:
    rho_logical, trace_logical = logical_projection(rho)
    target_dm = target if target.isherm else target * target.dag()
    full = fidelity(rho, target_dm)
    if trace_logical > 0:
        rho_logical_norm = rho_logical / trace_logical
        logical = fidelity(rho_logical_norm, target_dm)
        logical_purity_val = float((rho_logical_norm * rho_logical_norm).tr())
    else:
        logical = 0.0
        logical_purity_val = 0.0
    logger.info(
        "Fidelity diagnostics -> full: %s, logical: %s, trace: %s, purity: %s",
        full,
        logical,
        trace_logical,
        logical_purity_val,
    )
    return {
        "full": full,
        "logical": logical,
        "trace_logical": trace_logical,
        "logical_purity": logical_purity_val,
    }


def logical_purity(rho: Qobj) -> float:
    rho_logical, trace_logical = logical_projection(rho)
    if trace_logical <= 0:
        return 0.0
    rho_logical_norm = rho_logical / trace_logical
    return float((rho_logical_norm * rho_logical_norm).tr())


def canonical_logical_states() -> Dict[str, Qobj]:
    ket_0 = basis_gm1()
    ket_1 = basis_gp1()
    ket_plus = (ket_0 + ket_1).unit()
    ket_plus_y = (ket_0 + 1j * ket_1).unit()
    return {
        "0": ket_0,
        "1": ket_1,
        "0_L": ket_0,
        "1_L": ket_1,
        "+": ket_plus,
        "+y": ket_plus_y,
        "+_y": ket_plus_y,
    }


def build_gram_matrix(states: Dict[str, Qobj]) -> Tuple[np.ndarray, List[str]]:
    labels = list(states.keys())
    n = len(labels)
    G = np.zeros((n, n), dtype=complex)
    projected: Dict[str, Qobj] = {}
    for label, rho in states.items():
        rho_logical, trace_logical = logical_projection(rho)
        projected[label] = rho_logical / trace_logical if trace_logical > 0 else rho_logical
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            G[i, j] = (projected[li] * projected[lj]).tr()
    return G, labels


def gram_singular_values(G: np.ndarray) -> np.ndarray:
    return np.linalg.svd(G, compute_uv=False)


def encoder_gram_from_outputs(outputs: Dict[str, Qobj]) -> Tuple[np.ndarray, np.ndarray]:
    G, labels = build_gram_matrix(outputs)
    eigs = np.linalg.eigvals(G)
    logger.info("Encoder Gram matrix labels: %s", labels)
    logger.info("Encoder Gram eigenvalues: %s", eigs)
    return G, eigs
