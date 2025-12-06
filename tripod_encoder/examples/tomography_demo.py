"""Construct a Gram matrix from projection-mode simulations of canonical inputs."""

import numpy as np

from tripod_encoder.diagnostics import build_gram_matrix, canonical_logical_states, gram_singular_values
from tripod_encoder.params import ControlParameters, PhysicalParameters, SimulationParameters
from tripod_encoder.simulation import run_tripod_sequence


def simulate_inputs():
    physical = PhysicalParameters(Gamma_branch={"g-1": 0.0, "g0": 0.0, "g+1": 0.0})
    control = ControlParameters()
    sim = SimulationParameters(t_start=0.0, t_end=1.0, n_steps=50)

    logical_inputs = {
        "0": (1.0, 0.0, 0.0),
        "1": (0.0, 1.0, 0.0),
        "+": (1 / np.sqrt(2), 1 / np.sqrt(2), 0.0),
        "+y": (1 / np.sqrt(2), 1 / np.sqrt(2), np.pi / 2),
    }

    outputs = {}
    for label, (alpha, beta, phi) in logical_inputs.items():
        result = run_tripod_sequence(
            alpha=alpha,
            beta=beta,
            phi=phi,
            mode="projection",
            physical_params=physical,
            control_params=control,
            sim_params=sim,
            enable_channels={},
        )
        outputs[label] = result["result"].states[-1]
    return outputs


if __name__ == "__main__":
    outputs = simulate_inputs()
    G, labels = build_gram_matrix(outputs)
    svals = gram_singular_values(G)
    print("Labels:", labels)
    print("Gram matrix:\n", G)
    print("Singular values:", svals)
