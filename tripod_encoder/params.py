from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union

NumberLike = Union[float, int]
TimeFunction = Callable[[float], NumberLike]


def _default_detuning() -> Dict[str, NumberLike]:
    return {"g-1": 0.0, "g0": 0.0, "g+1": 0.0}


def _default_branching() -> Dict[str, NumberLike]:
    return {"g-1": 0.0, "g0": 0.0, "g+1": 0.0}


def _default_ground_pop() -> Dict[str, NumberLike]:
    return {"g-1": 1 / 3, "g0": 1 / 3, "g+1": 1 / 3}


def _default_dict() -> Dict:  # type: ignore[override]
    return {}


@dataclass
class PhysicalParameters:
    """Physical parameters for the tripod system."""

    Gamma_e: float = 0.0
    Gamma_branch: Dict[str, float] = field(default_factory=_default_branching)
    Delta: float = 0.0
    delta_minus1: Union[NumberLike, TimeFunction] = 0.0
    delta_0: Union[NumberLike, TimeFunction] = 0.0
    delta_plus1: Union[NumberLike, TimeFunction] = 0.0
    gamma_dephasing_pairs: Dict[tuple, float] = field(default_factory=_default_dict)
    Gamma1: float = 0.0
    p_th: Dict[str, float] = field(default_factory=_default_ground_pop)
    gamma_raman: Dict[tuple, float] = field(default_factory=_default_dict)


@dataclass
class ControlParameters:
    """Control fields and pulse-shape parameters."""

    Omega_sigma_plus_max: float = 0.0
    Omega_sigma_minus_max: float = 0.0
    Omega_pi_max: float = 0.0

    t0_probe: float = 0.0
    tau_probe: float = 1.0
    t0_control: float = 0.0
    tau_control: float = 1.0

    sigma_eps: float = 0.0
    sigma_phi: float = 0.0


@dataclass
class SimulationParameters:
    """Numerical solver settings."""

    t_start: float = 0.0
    t_end: float = 1.0
    n_steps: int = 100
    atol: Optional[float] = None
    rtol: Optional[float] = None
    method: str = "mesolve"

    @property
    def tlist(self):
        import numpy as np

        return np.linspace(self.t_start, self.t_end, self.n_steps)
