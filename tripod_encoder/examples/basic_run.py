import numpy as np
import matplotlib.pyplot as plt

from tripod_encoder.params import PhysicalParameters, ControlParameters, SimulationParameters
from tripod_encoder.simulation import run_tripod_sequence, logical_state
from tripod_encoder.diagnostics import (
    logical_fidelity,
    encoder_gram_from_outputs,
)
from tripod_encoder.pulses import Omega_sigma_plus, Omega_sigma_minus, Omega_pi
from tripod_encoder.logging_utils import setup_logger
from tripod_encoder.operators import Pe

logger = setup_logger("basic_run")


def excited_state_population(states):
    return [float((Pe * rho).tr()) for rho in states]


def run_single_sequence(alpha, beta, phi, phys, ctrl, sim):
    logger.info("Running sequence alpha=%s beta=%s phi=%s", alpha, beta, phi)
    return run_tripod_sequence(
        alpha=alpha,
        beta=beta,
        phi=phi,
        mode="projection",
        physical_params=phys,
        control_params=ctrl,
        sim_params=sim,
        enable_channels={},
    )


def main():
    phys = PhysicalParameters(
        Gamma_e=2 * np.pi * 6.0666e6,
        Gamma_branch={"g-1": 1 / 3, "g0": 1 / 3, "g+1": 1 / 3},
        Delta=0.0,
        delta_minus1=0.0,
        delta_0=0.0,
        delta_plus1=0.0,
        gamma_dephasing_pairs={},
        Gamma1=None,
        p_th=None,
        gamma_raman={},
    )

    ctrl = ControlParameters(
        Omega_sigma_plus_max=2 * np.pi * 1e6,
        Omega_sigma_minus_max=2 * np.pi * 1e6,
        Omega_pi_max=2 * np.pi * 8e6,
        t0_probe=1.0e-6,
        tau_probe=0.25e-6,
        t0_control=0.8e-6,
        tau_control=0.6e-6,
        sigma_eps=0.0,
        sigma_phi=0.0,
    )

    sim = SimulationParameters(
        t_start=0.0,
        t_end=4.0e-6,
        n_steps=2000,
        atol=1e-9,
        rtol=1e-7,
    )

    alpha, beta, phi = 1 / np.sqrt(2), 1 / np.sqrt(2), 0.0
    target = logical_state(alpha, beta, phi)

    res = run_single_sequence(alpha, beta, phi, phys, ctrl, sim)
    states = res["result"].states
    tlist = res["tlist"]

    rho_final = states[-1]
    diag = logical_fidelity(rho_final, target)

    print("\n=== FIDELITY DIAGNOSTICS ===")
    for k, v in diag.items():
        print(f"{k:20s}: {v}")

    inputs = {
        "0_L": (1.0, 0.0, 0.0),
        "1_L": (0.0, 1.0, 0.0),
        "+": (1 / np.sqrt(2), 1 / np.sqrt(2), 0.0),
        "+_y": (1 / np.sqrt(2), 1 / np.sqrt(2), np.pi / 2),
    }

    outputs = {}
    for label, (a, b, ph) in inputs.items():
        seq = run_single_sequence(a, b, ph, phys, ctrl, sim)
        outputs[label] = seq["result"].states[-1]

    gram, eigs = encoder_gram_from_outputs(outputs)
    print("\n=== GRAM MATRIX ===")
    print(gram)
    print("Singular / eigenvalues:", eigs)

    excited = excited_state_population(states)

    Omega_p = [Omega_sigma_plus(t, ctrl, alpha, beta, phi) for t in tlist]
    Omega_m = [Omega_sigma_minus(t, ctrl, alpha, beta, phi) for t in tlist]
    Omega_control = [Omega_pi(t, ctrl) for t in tlist]

    plt.figure(figsize=(10, 6))
    plt.plot(tlist * 1e6, np.abs(Omega_p), label="|Ωσ+|")
    plt.plot(tlist * 1e6, np.abs(Omega_m), label="|Ωσ-|")
    plt.plot(tlist * 1e6, np.abs(Omega_control), label="|Ωπ|")
    plt.xlabel("Time (µs)")
    plt.ylabel("Rabi amplitude")
    plt.title("Control & Probe Pulses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(tlist * 1e6, excited, label="Excited-state population")
    plt.xlabel("Time (µs)")
    plt.ylabel("Population")
    plt.title("Excited-state population vs time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
