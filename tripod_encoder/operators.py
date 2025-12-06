from __future__ import annotations

from typing import Dict

from qutip import Qobj, basis

BASIS_INDEX: Dict[str, int] = {"e": 0, "g-1": 1, "g0": 2, "g+1": 3}


def basis_e() -> Qobj:
    return basis(4, BASIS_INDEX["e"])


def basis_gm1() -> Qobj:
    return basis(4, BASIS_INDEX["g-1"])


def basis_g0() -> Qobj:
    return basis(4, BASIS_INDEX["g0"])


def basis_gp1() -> Qobj:
    return basis(4, BASIS_INDEX["g+1"])


Pe = basis_e() * basis_e().dag()
Pg_m1 = basis_gm1() * basis_gm1().dag()
Pg_0 = basis_g0() * basis_g0().dag()
Pg_p1 = basis_gp1() * basis_gp1().dag()


sigma_e_gm1 = basis_e() * basis_gm1().dag()
sigma_e_g0 = basis_e() * basis_g0().dag()
sigma_e_gp1 = basis_e() * basis_gp1().dag()


SIGMA_E_G: Dict[str, Qobj] = {
    "g-1": sigma_e_gm1,
    "g0": sigma_e_g0,
    "g+1": sigma_e_gp1,
}


def sigma_e_g(label: str) -> Qobj:
    if label not in SIGMA_E_G:
        raise ValueError(f"Unknown ground-state label: {label}")
    return SIGMA_E_G[label]
