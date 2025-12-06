"""Tripod logical encoder simulator package."""

from .params import PhysicalParameters, ControlParameters, SimulationParameters
from .simulation import run_tripod_sequence
from .diagnostics import (
    logical_projection,
    logical_fidelity,
    logical_purity,
    build_gram_matrix,
    canonical_logical_states,
    encoder_gram_from_outputs,
)

__all__ = [
    "PhysicalParameters",
    "ControlParameters",
    "SimulationParameters",
    "run_tripod_sequence",
    "logical_projection",
    "logical_fidelity",
    "logical_purity",
    "build_gram_matrix",
    "canonical_logical_states",
    "encoder_gram_from_outputs",
]
