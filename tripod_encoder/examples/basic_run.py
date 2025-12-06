"""Minimal example running a projection-mode simulation."""

from tripod_encoder.params import ControlParameters, PhysicalParameters, SimulationParameters
from tripod_encoder.simulation import run_tripod_sequence
from tripod_encoder.diagnostics import logical_fidelity, logical_purity, canonical_logical_states


if __name__ == "__main__":
    physical = PhysicalParameters(Gamma_branch={"g-1": 0.0, "g0": 0.0, "g+1": 0.0})
    control = ControlParameters()
    sim = SimulationParameters(t_start=0.0, t_end=1.0, n_steps=100)

    states = canonical_logical_states()
    alpha, beta, phi = 1.0, 0.0, 0.0  # |0_L>
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
    rho_final = result["result"].states[-1]

    fidelity_full = logical_fidelity(rho_final, states["0"])
    purity = logical_purity(rho_final)

    print("Logical fidelity:", fidelity_full)
    print("Logical purity:", purity)
