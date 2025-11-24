"""
Few-mode tripod atom + dual-photon simulation helpers.

This module lifts the content of the exploratory notebook into a reusable
Python module so that future changes can be reviewed and tested incrementally.
All physical parameters follow the Rb-87 D2 configuration used in the
TQM-SIM-V2 style simulations.  The utilities provide pulse definitions,
Hamiltonian construction, encoder calculation, SU(3) diagnostics, and simple
scan helpers.  Plotting routines remain available for quick checks.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # non-interactive backend keeps scripts from blocking on plt.show
import matplotlib.pyplot as plt
import numpy as np
from qutip import (
    Options,
    Qobj,
    basis,
    destroy,
    expect,
    ket2dm,
    mesolve,
    num,
    qeye,
    tensor,
)

EPS = np.finfo(float).eps
ERROR_LOG_PATH = Path(__file__).with_name("tripod_error.log")
FIGURES_DIR = Path("thesis_figures")
RUN_OUTPUT_PATH = Path(__file__).with_name("tripod_run_outputs.txt")


def configure_error_logging(log_path: Path = ERROR_LOG_PATH) -> None:
    """Set up file-backed logging and mirror messages to stdout."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _log_unhandled_exceptions(exc_type, exc_value, exc_traceback) -> None:
    """Capture uncaught exceptions in the error log before exiting."""

    logging.critical(
        "Unhandled exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )
    print(f"Unhandled exception logged to {ERROR_LOG_PATH}", flush=True)


configure_error_logging()
sys.excepthook = _log_unhandled_exceptions


def _ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(name: str) -> Path:
    """Save the active matplotlib figure to the thesis figures directory."""

    _ensure_figures_dir()
    output_path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logging.info("Figure saved to %s", output_path)
    return output_path


def write_run_outputs(lines: list[str], path: Path = RUN_OUTPUT_PATH) -> Path:
    """Persist human-readable run outputs (e.g., efficiencies, fidelities)."""

    output = "\n".join(lines).rstrip() + "\n"
    path.write_text(output, encoding="utf-8")
    logging.info("Run outputs written to %s", path)
    return path

# ------------------------------------------------------------------
# 1. Physical parameters (Rb-87 D2, TQM-SIM-V2 style)
# ------------------------------------------------------------------

# Atomic data
Gamma_e = 2 * np.pi * 6.07e6  # excited-state linewidth (Hz)
Delta_phys = 0.0  # on-resonance EIT baseline

# Control/probe Rabi frequencies (Hz)
Omega_C_phys = 2 * np.pi * 3e6  # control Rabi frequency
Omega_P_phys = 2 * np.pi * 1.0e6  # probe scale (used to set coupling g)

# Pulse width and total interaction time (s)
sigma_phys = 0.5e-6  # Gaussian sigma
T_total_phys = 5.0e-6  # write window

# Ground two-photon detunings (Hz) – start with zero
delta_m_phys = np.array([0.0, 0.0, 0.0])  # [delta_-1, delta_0, delta_+1]


@dataclass
class SweepParameters:
    """Container for parameters we may sweep without altering defaults."""

    clebsch_sigma_plus: float = 1.0
    clebsch_sigma_minus: float = 1.0
    delta_one_photon: float = Delta_phys
    delta_two_photon: np.ndarray = field(
        default_factory=lambda: delta_m_phys.copy()
    )
    zeeman_shifts: np.ndarray = field(default_factory=lambda: np.zeros(3))
    adiabaticity: float = Omega_C_phys * sigma_phys  # pulse area proxy
    omega_control: float = Omega_C_phys
    omega_probe: float = Omega_P_phys

    def dimensionless_one_photon(self) -> float:
        return self.delta_one_photon / Gamma_e

    def dimensionless_two_photon(self) -> np.ndarray:
        return (self.delta_two_photon + self.zeeman_shifts) / Gamma_e

    def dimensionless_omega_control(self) -> float:
        return self.omega_control / Gamma_e

    def dimensionless_omega_probe(self) -> float:
        return self.omega_probe / Gamma_e


# Default sweep parameters mirror the original fixed configuration
sweep_params = SweepParameters()

# Dimensionless units: t_sim = t_phys * Gamma_e; Omega_sim = Omega_phys / Gamma_e
Gamma = 1.0
Delta = sweep_params.dimensionless_one_photon()
Omega_c0 = sweep_params.dimensionless_omega_control()  # peak control Rabi frequency in simulation units

# Single-photon couplings in units of Gamma_e
g_plus = sweep_params.clebsch_sigma_plus * sweep_params.dimensionless_omega_probe()
g_minus = sweep_params.clebsch_sigma_minus * sweep_params.dimensionless_omega_probe()

# Time grid (dimensionless)
T_START = 0.0
T_END = T_total_phys * Gamma_e
N_STEPS = 1001
tlist = np.linspace(T_START, T_END, N_STEPS)  # evenly spaced for mesolve

# Settling window keeps a weak control field after write to let bright
# components decay before the gate.
SETTLE_DURATION_PHYS = 0.2e-6
T_SETTLE_END = T_END + SETTLE_DURATION_PHYS * Gamma_e
N_STEPS_SETTLE = max(11, int(np.round((T_SETTLE_END - T_END) / (T_END / (N_STEPS - 1)))) + 1)
tlist_settle = np.linspace(T_END, T_SETTLE_END, N_STEPS_SETTLE)

# Gate and read windows reuse the same duration as the write phase so that we can
# propagate through write → gate → read with equal time steps for each segment.
T_GATE_START = T_SETTLE_END
T_GATE_END = T_GATE_START + T_END
T_READ_START = T_GATE_END
T_READ_END = T_READ_START + T_END

tlist_gate = np.linspace(T_GATE_START, T_GATE_END, N_STEPS)
tlist_read = np.linspace(T_READ_START, T_READ_END, N_STEPS)

# Pulse centres (dimensionless; counter-intuitive sequence)
t_c_phys = 2.0e-6  # control pulse centre
t_p_phys = 3.0e-6  # effective probe time-centre (for diagnostics)
t_c = t_c_phys * Gamma_e
t_p = t_p_phys * Gamma_e
tau_c = sigma_phys * Gamma_e
tau_p = sigma_phys * Gamma_e  # identical widths simplify symmetry checks

delta_m = sweep_params.dimensionless_two_photon()


# ------------------------------------------------------------------
# 2. Pulse envelopes and mixing angle
# ------------------------------------------------------------------

def Omega_c(t: float) -> complex:
    """Control field envelope in units of Gamma_e with adiabatic boundaries."""

    envelope = np.exp(-((t - t_c) ** 2) / (2 * tau_c**2))
    turn_on = 1 - np.exp(-(t / (5 * tau_c)))
    turn_off = 1 - np.exp(-((T_END - t) / (5 * tau_c)))
    return Omega_c0 * envelope * turn_on * turn_off


def Omega_p_eff(t: float) -> complex:
    """
    Effective probe scale for plotting/mixing angle.

    The temporal envelope is not used directly in the few-mode Hamiltonian,
    but we keep it for diagnostics to stay consistent with the DSP picture.
    Retrieval mirrors this envelope even though g± encode its effect.
    """

    return (Omega_P_phys / Gamma_e) * np.exp(-((t - t_p) ** 2) / (2 * tau_p**2))


def Omega_c_gate(_t: float) -> complex:
    """Control field is off during the gate phase."""

    return 0.0


def Omega_c_settle(t: float) -> complex:
    """Weak control continuation to let bright components settle after write."""

    return 0.2 * Omega_c(t)


def Omega_c_read(t: float) -> complex:
    """
    Time-reversed control pulse for the read phase to emulate retrieval.

    The read window runs from T_READ_START to T_READ_END; mapping t →
    (T_READ_END - t) mirrors the write pulse so that the counter-intuitive
    sequence is replayed in reverse.
    """

    mirrored_t = T_READ_END - t
    return Omega_c(mirrored_t)


def Omega_p_reverse(t: float) -> complex:
    """
    Time-reversed probe envelope for retrieval consistency.

    Note: Ω_p is implicit in the few-mode coupling strengths g±; we mirror the
    diagnostic envelope for consistency with DSP intuition even though the
    Hamiltonian does not take Omega_p_func explicitly.
    """

    mirrored_t = T_READ_END - t
    return Omega_p_eff(mirrored_t)


def g_pulsed(t: float, g0: float) -> float:
    """Time-dependent probe coupling scaled by the probe envelope."""

    envelope = np.abs(Omega_p_eff(t))
    max_env = np.abs(Omega_p_eff(t_p))
    if max_env < np.sqrt(EPS):
        return 0.0
    return g0 * (envelope / max_env)


def g_plus_t(t: float) -> float:
    """Convenience wrapper: pulsed σ+ coupling at time ``t``."""

    return g_pulsed(t, g_plus)


def g_minus_t(t: float) -> float:
    """Convenience wrapper: pulsed σ− coupling at time ``t``."""

    return g_pulsed(t, g_minus)


def evaluate_coupling(
    coupling: float | complex | Callable[[float], complex], t: float
) -> complex:
    """Resolve a coupling that may be a scalar or callable."""

    if callable(coupling):
        return coupling(t)
    return coupling


def mixing_angle(t: float) -> float:
    """Dark-state mixing angle theta(t) = arctan(|Omega_p| / |Omega_c|)."""

    Oc = Omega_c(t)
    Op = Omega_p_eff(t)
    denom = np.sqrt(np.abs(Oc) ** 2 + np.abs(Op) ** 2)
    if denom < np.sqrt(EPS):
        return 0.0
    return np.arctan(np.abs(Op) / np.abs(Oc))


def mixing_angle_stats(times: np.ndarray = tlist) -> dict:
    """Summarise the mixing angle over a time grid."""

    theta_vals = np.array([mixing_angle(t) for t in times])
    return {
        "theta_start": float(theta_vals[0]),
        "theta_end": float(theta_vals[-1]),
        "theta_min": float(theta_vals.min()),
        "theta_max": float(theta_vals.max()),
    }


def dark_coupling_residuals(
    times: np.ndarray,
    Omega_c_func=Omega_c,
    g_plus_runtime=g_plus_t,
    g_minus_runtime=g_minus_t,
) -> np.ndarray:
    """
    Compute |<e|H|D>| / ||coupling|| across time by nulling the coupling vector.

    The bright coupling vector to |e> is proportional to (g_minus, Omega_c(t), g_plus);
    the nullspace of this 1x3 row gives two dark vectors. Any non-zero overlap of the
    coupling with those dark vectors flags imperfect darkness.
    """

    residuals: list[float] = []
    for t in times:
        g_minus_val = evaluate_coupling(g_minus_runtime, t)
        g_plus_val = evaluate_coupling(g_plus_runtime, t)
        coupling_vec = np.array([g_minus_val, Omega_c_func(t), g_plus_val], dtype=complex)
        norm = np.linalg.norm(coupling_vec)
        if norm < np.sqrt(EPS):
            residuals.append(0.0)
            continue

        _, _, vt = np.linalg.svd(coupling_vec.reshape(1, 3))
        nullspace = vt[1:].T
        max_residual = 0.0
        for col in nullspace.T:
            col_norm = np.linalg.norm(col)
            if col_norm < np.sqrt(EPS):
                continue
            d_vec = col / col_norm
            max_residual = max(max_residual, abs(np.vdot(coupling_vec, d_vec)) / norm)
        residuals.append(max_residual)

    return np.array(residuals)


def plot_pulses_and_mixing() -> None:
    """Plot control and probe envelopes together with the mixing angle."""

    Oc_vals = np.array([Omega_c(t) for t in tlist])
    Op_vals = np.array([Omega_p_eff(t) for t in tlist])
    theta_vals = np.array([mixing_angle(t) for t in tlist])

    t_us = tlist / Gamma_e * 1e6  # microseconds for plotting

    plt.figure(figsize=(8, 5))
    plt.plot(t_us, np.abs(Oc_vals), label=r"$|\Omega_c(t)|$")
    plt.plot(t_us, np.abs(Op_vals), label=r"$|\Omega_p(t)|$ (effective)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Rabi frequency (in units of Γ_e)")
    plt.title("Control and Effective Probe Pulses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("pulses.pdf")

    plt.figure(figsize=(8, 5))
    plt.plot(t_us, theta_vals)
    plt.xlabel("Time (µs)")
    plt.ylabel(r"$\Theta(t)$")
    plt.title("Mixing Angle vs Time")
    plt.grid(True, alpha=0.3)
    save_current_figure("mixing_angle.pdf")


def plot_full_timeline_pulses() -> None:
    """Show pulse envelopes across write → settle → gate → read."""

    segments = [
        (tlist, Omega_c, Omega_p_eff, "write"),
        (tlist_settle, Omega_c_settle, lambda _t: 0.0, "settle"),
        (tlist_gate, Omega_c_gate, lambda _t: 0.0, "gate"),
        (tlist_read, Omega_c_read, Omega_p_reverse, "read"),
    ]

    plt.figure(figsize=(9, 5))
    for times, oc_func, op_func, label in segments:
        t_us = times / Gamma_e * 1e6
        oc_vals = [abs(oc_func(t)) for t in times]
        op_vals = [abs(op_func(t)) for t in times]
        plt.plot(t_us, oc_vals, label=f"|Ωc| ({label})")
        plt.plot(t_us, op_vals, linestyle=":", label=f"|Ωp| ({label})")

    plt.xlabel("Time (µs)")
    plt.ylabel("Rabi frequency (Γ_e units)")
    plt.title("Full-timeline control/probe envelopes")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize="small")
    save_current_figure("pulses_full_timeline.pdf")


def plot_dark_residuals(residuals: np.ndarray) -> None:
    """Plot the coupling nullspace residual across the write window."""

    t_us = tlist / Gamma_e * 1e6
    plt.figure(figsize=(8, 5))
    plt.plot(t_us, residuals)
    plt.xlabel("Time (µs)")
    plt.ylabel(r"$|\langle e|H|D\rangle| / ||g,\Omega||$")
    plt.title("Dark-coupling residual vs time")
    plt.grid(True, alpha=0.3)
    save_current_figure("dark_coupling_residual.pdf")


# ------------------------------------------------------------------
# 3. Hilbert space: atom + σ+ photon + σ- photon
# ------------------------------------------------------------------

# Atom basis: |e>, |g_-1>, |g_0>, |g_+1>
ket_e = basis(4, 0)
ket_gm1 = basis(4, 1)
ket_g0 = basis(4, 2)
ket_gp1 = basis(4, 3)

P_e = ket_e * ket_e.dag()
P_gm1 = ket_gm1 * ket_gm1.dag()
P_g0 = ket_g0 * ket_g0.dag()
P_gp1 = ket_gp1 * ket_gp1.dag()

# Photon Fock dims
N_ph = 2  # support 0,1 photons

# Photon mode operators in full space
a_plus = tensor(qeye(4), destroy(N_ph), qeye(N_ph))
a_minus = tensor(qeye(4), qeye(N_ph), destroy(N_ph))

# Photon number operators in full space
N_plus_full = tensor(qeye(4), num(N_ph), qeye(N_ph))
N_minus_full = tensor(qeye(4), qeye(N_ph), num(N_ph))

# Projectors lifted to atom+photon space
P_e_full = tensor(P_e, qeye(N_ph), qeye(N_ph))
P_gm1_full = tensor(P_gm1, qeye(N_ph), qeye(N_ph))
P_g0_full = tensor(P_g0, qeye(N_ph), qeye(N_ph))
P_gp1_full = tensor(P_gp1, qeye(N_ph), qeye(N_ph))

# Atomic raising/lowering operators (in full space)
sig_egm1 = tensor(ket_e * ket_gm1.dag(), qeye(N_ph), qeye(N_ph))
sig_eg0 = tensor(ket_e * ket_g0.dag(), qeye(N_ph), qeye(N_ph))
sig_egp1 = tensor(ket_e * ket_gp1.dag(), qeye(N_ph), qeye(N_ph))


# ------------------------------------------------------------------
# 4. Time-dependent Hamiltonian for the few-mode tripod
# ------------------------------------------------------------------

def H_tripod_t(t: float, args: dict) -> Qobj:
    """
    Full Hamiltonian H(t)/ħ in dimensionless units (frequencies scaled by Γ_e).

    Includes:
      - excited state detuning -Delta |e><e|
      - ground two-photon detunings delta_m
      - couplings:
          g_minus a_minus |e><g_-1| + h.c.  (σ− photon: Δm = -1)
          g_plus  a_plus  |e><g_+1| + h.c.  (σ+ photon: Δm = +1)
          Omega_c(t) |e><g_0| + h.c.
    """

    delta = args.get("delta_m", delta_m)  # allow runtime detuning sweeps
    g_plus_runtime = args.get("g_plus", g_plus_t)
    g_minus_runtime = args.get("g_minus", g_minus_t)
    omega_c_func = args.get("Omega_c_func", Omega_c)

    # Diagonal detunings: excited-state offset plus optional ground offsets
    H = (-Delta) * P_e_full
    H += delta[0] * P_gm1_full
    H += delta[1] * P_g0_full
    H += delta[2] * P_gp1_full

    # σ- coupling: |g_{-1};1_-> ↔ |e;0_-> (Δm = -1)
    gminus_t = evaluate_coupling(g_minus_runtime, t)
    H += gminus_t * (a_minus * sig_egm1.dag() + a_minus.dag() * sig_egm1)

    # σ+ coupling: |g_{+1};1_+> ↔ |e;0_+> (Δm = +1)
    gplus_t = evaluate_coupling(g_plus_runtime, t)
    H += gplus_t * (a_plus * sig_egp1.dag() + a_plus.dag() * sig_egp1)

    # Control field coupling (classical) |g_0> ↔ |e>
    Oc = omega_c_func(t)
    H += Oc * sig_eg0 + np.conj(Oc) * sig_eg0.dag()

    return H


# ------------------------------------------------------------------
# 5. Initial states: photonic σ+, σ− and atomic |g0>
# ------------------------------------------------------------------

# Photon Fock basis: |n_+>, |n_->
ket_0_plus = basis(N_ph, 0)
ket_1_plus = basis(N_ph, 1)
ket_0_minus = basis(N_ph, 0)
ket_1_minus = basis(N_ph, 1)

# Single-photon circular polarisations
ket_sigma_plus = tensor(ket_1_plus, ket_0_minus)  # one σ+ photon
ket_sigma_minus = tensor(ket_0_plus, ket_1_minus)  # one σ- photon

# Atomic initial state |g0>
psi_atom0 = ket_g0

# Full initial states
psi_in_sigma_plus = tensor(psi_atom0, ket_1_plus, ket_0_minus)
psi_in_sigma_minus = tensor(psi_atom0, ket_0_plus, ket_1_minus)

# Convenient vacuum references for leakage checks and amplitude extraction
ket_vacuum_photons = tensor(ket_0_plus, ket_0_minus)
ket_gm1_vacuum = tensor(ket_gm1, ket_0_plus, ket_0_minus)
ket_g0_vacuum = tensor(ket_g0, ket_0_plus, ket_0_minus)
ket_gp1_vacuum = tensor(ket_gp1, ket_0_plus, ket_0_minus)
P_vacuum_photons = tensor(qeye(4), ket_0_plus * ket_0_plus.dag(), ket_0_minus * ket_0_minus.dag())


# ------------------------------------------------------------------
# Collapse operators (optional noise model)
# ------------------------------------------------------------------

def build_collapse_operators(gamma_decay: float = 0.0, gamma_dephase: float = 0.0):
    """
    Construct collapse operators for spontaneous emission and pure dephasing.

    The rates are provided in simulation (dimensionless) units. Passing zeros
    preserves the default coherent evolution so existing outputs remain
    unchanged while allowing dissipation to be enabled for sweeps.
    """

    c_ops = []

    if gamma_decay > 0:
        decay_rate = np.sqrt(gamma_decay)
        c_ops.extend(
            [
                decay_rate * sig_egm1.dag(),
                decay_rate * sig_eg0.dag(),
                decay_rate * sig_egp1.dag(),
            ]
        )

    if gamma_dephase > 0:
        c_ops.append(np.sqrt(gamma_dephase) * P_e_full)

    return c_ops


DEFAULT_C_OPS = build_collapse_operators()


# ------------------------------------------------------------------
# Sweep helpers
# ------------------------------------------------------------------

def scaled_control_envelope(scale: float):
    """Return a scaled copy of the write-pulse envelope."""

    return lambda t: scale * Omega_c(t)


def build_runtime_args_from_params(params: SweepParameters) -> dict:
    """Translate a SweepParameters instance into mesolve args overrides."""

    g_plus_dimless = params.clebsch_sigma_plus * params.dimensionless_omega_probe()
    g_minus_dimless = params.clebsch_sigma_minus * params.dimensionless_omega_probe()
    delta_override = params.dimensionless_two_photon()

    omega_c_scale = 1.0
    if Omega_c0 > 0:
        omega_c_scale = params.dimensionless_omega_control() / Omega_c0

    return {
        "delta_m": delta_override,
        "g_plus": lambda t: g_pulsed(t, g_plus_dimless),
        "g_minus": lambda t: g_pulsed(t, g_minus_dimless),
        "Omega_c_func": scaled_control_envelope(omega_c_scale),
    }


# ------------------------------------------------------------------
# 6. Encoder Construction (physical + renormalised)
# ------------------------------------------------------------------

def evolve_and_extract(
    alpha_plus: complex,
    alpha_minus: complex,
    do_log: bool = False,
    c_ops=None,
    h_args: dict | None = None,
    return_diagnostics: bool = False,
):
    """
    Evolve a single tripod atom + 2 photonic modes for a given input
    photonic polarisation (alpha_plus, alpha_minus) and return the
    3-component ground-state vector c = (c_{-1}, c_0, c_{+1}).

    This uses the full few-mode Hamiltonian and projects onto the
    atomic ground manifold at t = T_end.
    """

    norm = np.sqrt(abs(alpha_plus) ** 2 + abs(alpha_minus) ** 2)  # guard against degenerate input
    if norm < 1e-12:
        raise ValueError("Input photonic state has zero norm.")
    ap = alpha_plus / norm
    am = alpha_minus / norm

    # Initial state: |1_photon_in_polarisation> ⊗ |g_0>
    psi_in = ap * psi_in_sigma_plus + am * psi_in_sigma_minus

    runtime_c_ops = DEFAULT_C_OPS if c_ops is None else c_ops
    args = {
        "delta_m": delta_m,
        "Omega_c_func": Omega_c,
        "g_plus": g_plus_t,
        "g_minus": g_minus_t,
    }
    if h_args:
        args.update(h_args)

    res = mesolve(
        H_tripod_t,
        psi_in,
        tlist,
        runtime_c_ops,
        e_ops=[],
        args=args,
    )

    psi_T = res.states[-1]  # final state after the write window

    # Track residual photons to flag incomplete absorption during encoding
    remaining_plus = expect(N_plus_full, psi_T)
    remaining_minus = expect(N_minus_full, psi_T)
    if do_log:
        print(
            f"Residual photons after write: n_plus={remaining_plus:.3e}, n_minus={remaining_minus:.3e}"
        )

    if do_log:
        rho_full = ket2dm(psi_T)
        pop_e_final = (P_e_full * rho_full).tr().real
        print(f"Final excited-state population: {pop_e_final:.3e}")

    # Project amplitudes onto vacuum-photon ground states to avoid mixing with leftover light
    amp_gm1 = ket_gm1_vacuum.overlap(psi_T)
    amp_g0 = ket_g0_vacuum.overlap(psi_T)
    amp_gp1 = ket_gp1_vacuum.overlap(psi_T)

    coeffs = np.array([amp_gm1, amp_g0, amp_gp1], dtype=complex)
    if return_diagnostics:
        return coeffs, {
            "residual_n_plus": remaining_plus,
            "residual_n_minus": remaining_minus,
            "pop_e_final": (P_e_full * ket2dm(psi_T)).tr().real,
        }

    return coeffs


def build_encoder():
    """
    Construct both the physical encoder (E_phys) and a renormalised
    isometric encoder (E_iso) based on the few-mode tripod model.

    Returns:
        E_phys, E_iso, eta, G_phys, G_iso, eigvals_G_phys, eigvals_G_iso, encoder_metrics
    """

    print("Building physically faithful encoder E_phys (few-mode tripod model)...")

    c_sigma_plus, diag_plus = evolve_and_extract(1.0, 0.0, do_log=True, return_diagnostics=True)  # σ+
    c_sigma_minus, diag_minus = evolve_and_extract(0.0, 1.0, do_log=True, return_diagnostics=True)  # σ-

    E_phys = np.column_stack((c_sigma_plus, c_sigma_minus))  # ground-level amplitudes for each photon basis state

    G_phys = E_phys.conj().T @ E_phys  # overlap/efficiency matrix
    eigvals_G_phys = np.linalg.eigvalsh(G_phys)

    eta_plus = np.linalg.norm(c_sigma_plus) ** 2
    eta_minus = np.linalg.norm(c_sigma_minus) ** 2
    eta = 0.5 * (eta_plus + eta_minus)  # average storage efficiency over σ± inputs

    if eta > 1e-12:
        scale = np.sqrt(eta)
        E_iso = E_phys / scale
        G_iso = E_iso.conj().T @ E_iso
        eigvals_G_iso = np.linalg.eigvalsh(G_iso)
    else:
        E_iso = np.zeros_like(E_phys)
        G_iso = np.zeros((2, 2), dtype=complex)
        eigvals_G_iso = np.array([0.0, 0.0])

    singular_values = np.linalg.svd(E_phys, compute_uv=False)
    cond_number = singular_values[0] / singular_values[-1] if singular_values[-1] > EPS else np.inf
    renorm_gap = np.abs(singular_values - np.sqrt(eta))
    encoder_metrics = {
        "eta_plus": eta_plus,
        "eta_minus": eta_minus,
        "residual_n_plus": diag_plus["residual_n_plus"],
        "residual_n_minus": diag_minus["residual_n_minus"],
        "pop_e_plus": diag_plus["pop_e_final"],
        "pop_e_minus": diag_minus["pop_e_final"],
        "orthogonality_cplus_cminus": np.vdot(c_sigma_plus, c_sigma_minus),
        "condition_number": cond_number,
        "singular_values": singular_values,
        "renorm_gap": renorm_gap,
    }

    print("\nEncoder matrix E_phys (rows: g-1, g0, g+1; cols: sigma+, sigma-):")
    print(E_phys)

    print("\nGram matrix G_phys = E_phys^† E_phys:")
    print(G_phys)
    print("\nEigenvalues of G_phys (ideal is [η, η] for symmetric encoder):")
    print(eigvals_G_phys)
    print(f"\nAverage storage efficiency eta = Tr(G_phys)/2 = {eta:.4e}")

    print("\nRenormalised isometric encoder E_iso = E_phys / sqrt(eta):")
    print(E_iso)

    print("\nG_iso = E_iso^† E_iso (should be close to identity):")
    print(G_iso)
    print("\nEigenvalues of G_iso (ideal is [1, 1]):")
    print(eigvals_G_iso)

    return E_phys, E_iso, eta, G_phys, G_iso, eigvals_G_phys, eigvals_G_iso, encoder_metrics


def build_decoder(E_phys: np.ndarray, pseudo_inverse: bool = True) -> np.ndarray:
    """
    Construct a decoder that maps stored ground-state amplitudes back to the
    photonic basis. When `pseudo_inverse` is True, a Moore–Penrose inverse is
    used to remain well defined even if the encoder is not perfectly isometric;
    otherwise the simple Hermitian transpose is returned.
    """

    if pseudo_inverse:
        return np.linalg.pinv(E_phys)
    return E_phys.conj().T


def extract_ground_trajectory(states) -> np.ndarray:
    """Project a list of states onto the photon-vacuum ground components."""

    coeffs = np.zeros((len(states), 3), dtype=complex)
    for idx, state in enumerate(states):
        coeffs[idx, 0] = ket_gm1_vacuum.overlap(state)
        coeffs[idx, 1] = ket_g0_vacuum.overlap(state)
        coeffs[idx, 2] = ket_gp1_vacuum.overlap(state)
    return coeffs


def encoder_validation_diagnostics(collector: list[str] | None = None) -> dict:
    """Track encoder coefficients, overlaps, and Gram spectra across the write window."""

    res_plus = run_trajectory(1.0, 0.0)
    res_minus = run_trajectory(0.0, 1.0)

    coeffs_plus = extract_ground_trajectory(res_plus.states)
    coeffs_minus = extract_ground_trajectory(res_minus.states)

    overlaps = np.array([np.vdot(cp, cm) for cp, cm in zip(coeffs_plus, coeffs_minus)])
    gram_eigs = np.zeros((len(tlist), 2))
    for idx, (cp, cm) in enumerate(zip(coeffs_plus, coeffs_minus)):
        E_t = np.column_stack((cp, cm))
        gram_eigs[idx] = np.linalg.eigvalsh(E_t.conj().T @ E_t)

    t_us = tlist / Gamma_e * 1e6

    labels = [r"$|c_{-1}|^2$", r"$|c_0|^2$", r"$|c_{+1}|^2$"]
    plt.figure(figsize=(10, 6))
    for i, lbl in enumerate(labels):
        plt.plot(t_us, np.abs(coeffs_plus[:, i]) ** 2, label=f"σ+ {lbl}")
    for i, lbl in enumerate(labels):
        plt.plot(t_us, np.abs(coeffs_minus[:, i]) ** 2, linestyle=":", label=f"σ− {lbl}")
    plt.xlabel("Time (µs)")
    plt.ylabel("Ground-manifold population")
    plt.title("Encoder coefficient magnitudes across write window")
    plt.legend(ncol=2, fontsize="small")
    plt.grid(True, alpha=0.3)
    save_current_figure("encoder_coefficients.pdf")

    plt.figure(figsize=(8, 5))
    plt.plot(t_us, np.abs(overlaps))
    plt.xlabel("Time (µs)")
    plt.ylabel(r"$|\langle c_{+} | c_{-} \rangle|$")
    plt.title("Encoder image overlap vs time")
    plt.grid(True, alpha=0.3)
    save_current_figure("encoder_overlap.pdf")

    plt.figure(figsize=(8, 5))
    plt.plot(t_us, gram_eigs[:, 0], label=r"$\lambda_1(G)$")
    plt.plot(t_us, gram_eigs[:, 1], label=r"$\lambda_2(G)$")
    plt.xlabel("Time (µs)")
    plt.ylabel("Eigenvalues of $G(t)$")
    plt.title("Gram spectrum during absorption")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("encoder_gram_trajectory.pdf")

    metrics = {
        "overlap_final": overlaps[-1],
        "gram_final": gram_eigs[-1],
        "sigma_plus_final": np.abs(coeffs_plus[-1]) ** 2,
        "sigma_minus_final": np.abs(coeffs_minus[-1]) ** 2,
    }

    if collector is not None:
        collector.append(
            "Encoder coefficient norms (final) σ+: "
            + ", ".join(f"{val:.3e}" for val in metrics["sigma_plus_final"])
        )
        collector.append(
            "Encoder coefficient norms (final) σ-: "
            + ", ".join(f"{val:.3e}" for val in metrics["sigma_minus_final"])
        )
        collector.append(f"Encoder overlap at T_end: {metrics['overlap_final']:.3e}")
        collector.append(
            "Gram eigenvalues at T_end: " + ", ".join(f"{val:.3e}" for val in metrics["gram_final"])
        )

    return metrics


# ------------------------------------------------------------------
# 7. EIT-like diagnostics: populations vs time
# ------------------------------------------------------------------

def run_trajectory(
    alpha_plus: complex = 1.0,
    alpha_minus: complex = 0.0,
    c_ops=None,
    h_args: dict | None = None,
):
    """
    Evolve the few-mode tripod for a given input photonic polarisation.
    Returns the mesolve result object so we can inspect populations, etc.
    """

    norm = np.sqrt(abs(alpha_plus) ** 2 + abs(alpha_minus) ** 2)  # ensure unit-norm inputs
    if norm < 1e-12:
        raise ValueError("Input photonic state has zero norm.")
    ap = alpha_plus / norm
    am = alpha_minus / norm

    psi_in = ap * psi_in_sigma_plus + am * psi_in_sigma_minus

    runtime_c_ops = DEFAULT_C_OPS if c_ops is None else c_ops
    args = {
        "delta_m": delta_m,
        "Omega_c_func": Omega_c,
        "g_plus": g_plus_t,
        "g_minus": g_minus_t,
    }
    if h_args:
        args.update(h_args)

    return mesolve(
        H_tripod_t,
        psi_in,
        tlist,
        runtime_c_ops,
        e_ops=[],
        args=args,
    )


def plot_eit_and_populations(
    alpha_plus: complex = 1.0, alpha_minus: complex = 0.0, collector: list[str] | None = None
) -> dict:
    """Plot atomic populations and photon number diagnostics and return summary stats."""

    res = run_trajectory(alpha_plus, alpha_minus)

    pop_e = expect(P_e_full, res.states)
    pop_gm1 = expect(P_gm1_full, res.states)
    pop_g0 = expect(P_g0_full, res.states)
    pop_gp1 = expect(P_gp1_full, res.states)

    n_plus = expect(N_plus_full, res.states)
    n_minus = expect(N_minus_full, res.states)
    pop_ground = pop_gm1 + pop_g0 + pop_gp1

    t_us = tlist / Gamma_e * 1e6

    plt.figure(figsize=(8, 5))
    plt.plot(t_us, pop_e, label=r"$P_e$")
    plt.plot(t_us, pop_gm1, label=r"$P_{g_{-1}}$")
    plt.plot(t_us, pop_g0, label=r"$P_{g_0}$")
    plt.plot(t_us, pop_gp1, label=r"$P_{g_{+1}}$")
    plt.xlabel("Time (µs)")
    plt.ylabel("Population")
    plt.title("Atomic level populations vs time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("populations.pdf")

    plt.figure(figsize=(8, 5))
    plt.plot(t_us, n_plus, label=r"$\langle n_{+}\rangle$")
    plt.plot(t_us, n_minus, label=r"$\langle n_{-}\rangle$")
    plt.xlabel("Time (µs)")
    plt.ylabel("Photon number (expectation)")
    plt.title("Photon occupations vs time (few-mode model)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("photon_numbers.pdf")

    plt.figure(figsize=(8, 5))
    plt.plot(t_us, pop_e)
    plt.xlabel("Time (µs)")
    plt.ylabel(r"$P_e(t)$")
    plt.title("Excited-state population (EIT-like transparency diagnostic)")
    plt.grid(True, alpha=0.3)
    save_current_figure("excited_state_population.pdf")

    metrics = {
        "max_ground_population": float(np.max(pop_ground)),
        "max_excited_population": float(np.max(pop_e)),
        "max_photon_expectation": float(np.max(n_plus + n_minus)),
    }
    if collector is not None:
        collector.append(
            "Max populations during write: "
            f"ground={metrics['max_ground_population']:.3e}, "
            f"excited={metrics['max_excited_population']:.3e}, "
            f"photons={metrics['max_photon_expectation']:.3e}"
        )

    return metrics


def plot_bright_population(alpha_plus: complex = 1.0, alpha_minus: complex = 0.0, collector: list[str] | None = None) -> dict:
    """Plot bright-projector population during the write window."""

    res = run_trajectory(alpha_plus, alpha_minus)
    bright_pop = bright_population_trajectory(res.states, tlist)
    t_us = tlist / Gamma_e * 1e6

    plt.figure(figsize=(8, 5))
    plt.plot(t_us, bright_pop)
    plt.xlabel("Time (µs)")
    plt.ylabel(r"$\langle B|\rho|B\rangle$")
    plt.title("Bright-state population during write")
    plt.grid(True, alpha=0.3)
    save_current_figure("bright_population_write.pdf")

    metrics = {
        "bright_max": float(np.max(bright_pop)),
        "bright_final": float(bright_pop[-1]),
    }

    if collector is not None:
        collector.append(
            f"Bright population (write): max={metrics['bright_max']:.3e}, final={metrics['bright_final']:.3e}"
        )

    return metrics


# ------------------------------------------------------------------
# 8. Three-period EIT cycle: write, gate, read
# ------------------------------------------------------------------


def run_gate_phase(psi_in: Qobj, c_ops=None, h_args: dict | None = None):
    """Propagate through the gate window with lasers off to track leakage."""

    runtime_c_ops = DEFAULT_C_OPS if c_ops is None else c_ops
    args = {
        "delta_m": delta_m,
        "Omega_c_func": Omega_c_gate,
        "g_plus": lambda _t: 0.0,
        "g_minus": lambda _t: 0.0,
    }
    if h_args:
        args.update(h_args)

    return mesolve(
        H_tripod_t,
        psi_in,
        tlist_gate,
        runtime_c_ops,
        e_ops=[],
        args=args,
    )


def run_settle_phase(psi_in: Qobj, c_ops=None, h_args: dict | None = None):
    """Allow bright components to settle with a weak control tail."""

    runtime_c_ops = DEFAULT_C_OPS if c_ops is None else c_ops
    args = {
        "delta_m": delta_m,
        "Omega_c_func": Omega_c_settle,
        "g_plus": g_plus_t,
        "g_minus": g_minus_t,
    }
    if h_args:
        args.update(h_args)

    return mesolve(
        H_tripod_t,
        psi_in,
        tlist_settle,
        runtime_c_ops,
        e_ops=[],
        args=args,
    )


def run_read_phase(psi_in: Qobj, c_ops=None, h_args: dict | None = None):
    """Propagate through the read window with the control pulse reversed."""

    runtime_c_ops = DEFAULT_C_OPS if c_ops is None else c_ops
    args = {
        "delta_m": delta_m,
        "Omega_c_func": Omega_c_read,
        "g_plus": g_plus_t,
        "g_minus": g_minus_t,
    }
    if h_args:
        args.update(h_args)

    return mesolve(
        H_tripod_t,
        psi_in,
        tlist_read,
        runtime_c_ops,
        e_ops=[],
        args=args,
    )


def run_three_phase_sequence(
    alpha_plus: complex = 1.0,
    alpha_minus: complex = 0.0,
    c_ops=None,
    h_args: dict | None = None,
    gate_h_args: dict | None = None,
    read_h_args: dict | None = None,
):
    """
    Execute write → gate → read in sequence.

    The write phase reuses the existing EIT trajectory; the gate phase holds
    the system with lasers off; the read phase mirrors the write pulse.
    Returns the three mesolve result objects so that diagnostics can still be
    performed at the end of the write window while also exposing the final
    state for fidelity checks.
    """

    res_write = run_trajectory(alpha_plus, alpha_minus, c_ops=c_ops, h_args=h_args)
    psi_after_write = res_write.states[-1]

    res_settle = run_settle_phase(psi_after_write, c_ops=c_ops, h_args=h_args)
    psi_after_settle = res_settle.states[-1]

    res_gate = run_gate_phase(psi_after_settle, c_ops=c_ops, h_args=gate_h_args)
    psi_after_gate = res_gate.states[-1]

    res_read = run_read_phase(psi_after_gate, c_ops=c_ops, h_args=read_h_args)

    return res_write, res_settle, res_gate, res_read


def extract_photonic_density(psi: Qobj) -> Qobj:
    """Trace out the atom to obtain the reduced photonic density matrix."""

    rho_full = ket2dm(psi)
    return rho_full.ptrace([1, 2])


def compute_full_cycle_fidelity(
    alpha_plus: complex = 1.0,
    alpha_minus: complex = 0.0,
    c_ops=None,
    h_args: dict | None = None,
    gate_h_args: dict | None = None,
    read_h_args: dict | None = None,
):
    """
    Calculate the photonic retrieval fidelity after write → gate → read.

    The atomic subsystem is traced out so that we compare the retrieved photon
    state against the target single-photon polarisation, regardless of any
    residual atomic population.
    """

    norm = np.sqrt(abs(alpha_plus) ** 2 + abs(alpha_minus) ** 2)
    if norm < 1e-12:
        raise ValueError("Input photonic state has zero norm.")

    ap = alpha_plus / norm
    am = alpha_minus / norm
    # Target photonic density operator for fidelity evaluation
    rho_target_photonic = ket2dm(ap * ket_sigma_plus + am * ket_sigma_minus)

    _, _, _, res_read = run_three_phase_sequence(
        alpha_plus,
        alpha_minus,
        c_ops=c_ops,
        h_args=h_args,
        gate_h_args=gate_h_args,
        read_h_args=read_h_args,
    )
    psi_out = res_read.states[-1]

    rho_photonic_out = extract_photonic_density(psi_out)

    fidelity = (rho_photonic_out * rho_target_photonic).tr().real
    return fidelity, psi_out


def compute_full_cycle_metrics(
    E_iso: np.ndarray,
    alpha_plus: complex = 1.0,
    alpha_minus: complex = 0.0,
    c_ops=None,
    h_args: dict | None = None,
    gate_h_args: dict | None = None,
    read_h_args: dict | None = None,
):
    """Aggregate diagnostics across write → settle → gate → read."""

    res_write, res_settle, res_gate, res_read = run_three_phase_sequence(
        alpha_plus,
        alpha_minus,
        c_ops=c_ops,
        h_args=h_args,
        gate_h_args=gate_h_args,
        read_h_args=read_h_args,
    )

    psi_out = res_read.states[-1]
    n_plus_final = expect(N_plus_full, psi_out)
    n_minus_final = expect(N_minus_full, psi_out)
    p_return = n_plus_final + n_minus_final
    p_ground_final = expect(P_gm1_full + P_g0_full + P_gp1_full, psi_out)

    # Track bright population using the instantaneous polariton projector.
    bright_proj_write = bright_projector_full(tlist[-1], Omega_c, g_minus_t, g_plus_t)
    bright_proj_read = bright_projector_full(tlist_read[0], Omega_c_read, g_minus_t, g_plus_t)
    bright_at_read_start = expect(bright_proj_read, res_read.states[0])
    bright_at_write_end = expect(bright_proj_write, res_write.states[-1])

    return {
        "p_return": p_return,
        "p_ground_final": p_ground_final,
        "bright_at_read_start": bright_at_read_start,
        "bright_at_write_end": bright_at_write_end,
        "norms": [state.norm() for state in res_write.states + res_settle.states + res_gate.states + res_read.states],
    }


def plot_readout_photon_dynamics(
    alpha_plus: complex = 1.0,
    alpha_minus: complex = 0.0,
    collector: list[str] | None = None,
):
    """Plot photon re-emission and bright population during the read window."""

    _, _, _, res_read = run_three_phase_sequence(alpha_plus, alpha_minus)
    t_us_read = tlist_read / Gamma_e * 1e6

    n_plus = expect(N_plus_full, res_read.states)
    n_minus = expect(N_minus_full, res_read.states)
    bright_pop_read = bright_population_trajectory(res_read.states, tlist_read, omega_c_func=Omega_c_read)

    plt.figure(figsize=(8, 5))
    plt.plot(t_us_read, n_plus, label=r"$\langle n_{+}\rangle$")
    plt.plot(t_us_read, n_minus, label=r"$\langle n_{-}\rangle$")
    plt.xlabel("Time (µs)")
    plt.ylabel("Photon number (expectation)")
    plt.title("Photon re-emission during read")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("photon_numbers_read.pdf")

    plt.figure(figsize=(8, 5))
    plt.plot(t_us_read, bright_pop_read)
    plt.xlabel("Time (µs)")
    plt.ylabel(r"$\langle B|\rho|B\rangle$")
    plt.title("Bright-state population during read")
    plt.grid(True, alpha=0.3)
    save_current_figure("bright_population_read.pdf")

    metrics = {
        "max_photon_read": float(np.max(n_plus + n_minus)),
        "final_photon_read": float(n_plus[-1] + n_minus[-1]),
        "bright_read_max": float(np.max(bright_pop_read)),
        "bright_read_final": float(bright_pop_read[-1]),
    }

    if collector is not None:
        collector.append(
            "Readout photon return: "
            f"max={metrics['max_photon_read']:.3e}, final={metrics['final_photon_read']:.3e}"
        )
        collector.append(
            "Bright population (read): "
            f"max={metrics['bright_read_max']:.3e}, final={metrics['bright_read_final']:.3e}"
        )

    return metrics


def report_retrieval_tomography(collector: list[str] | None = None) -> list[tuple[str, float]]:
    """Print and return retrieval fidelities for a small tomography set of inputs."""

    test_states = [
        ("sigma+", (1.0, 0.0)),
        ("sigma-", (0.0, 1.0)),
        ("+", (1 / np.sqrt(2), 1 / np.sqrt(2))),
        ("i", (1 / np.sqrt(2), 1j / np.sqrt(2))),
    ]

    results: list[tuple[str, float]] = []
    for label, (ap, am) in test_states:
        fidelity, _ = compute_full_cycle_fidelity(alpha_plus=ap, alpha_minus=am)
        results.append((label, fidelity))
        line = f"Retrieval fidelity for {label:6s}: {fidelity:.6f}"
        print(line)
        if collector is not None:
            collector.append(line)

    return results


def plot_retrieval_fidelity_landscape(
    n_theta: int = 6,
    n_phi: int = 13,
    collector: list[str] | None = None,
):
    """Sample retrieval fidelity over a small grid of input polarisations."""

    theta_vals = np.linspace(0, np.pi / 2, n_theta)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi)
    fidelity_grid = np.zeros((n_theta, n_phi))

    for i, theta in enumerate(theta_vals):
        for j, phi in enumerate(phi_vals):
            ap = np.cos(theta)
            am = np.exp(1j * phi) * np.sin(theta)
            fidelity, _ = compute_full_cycle_fidelity(alpha_plus=ap, alpha_minus=am)
            fidelity_grid[i, j] = fidelity

    plt.figure(figsize=(8, 5))
    extent = [phi_vals[0], phi_vals[-1], theta_vals[0], theta_vals[-1]]
    im = plt.imshow(
        fidelity_grid,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    plt.colorbar(im, label="Retrieval fidelity")
    plt.xlabel("Relative phase φ (rad)")
    plt.ylabel("Mixing angle θ (rad)")
    plt.title("Retrieval fidelity landscape")
    save_current_figure("retrieval_fidelity_landscape.pdf")

    if collector is not None:
        collector.append(
            "Fidelity landscape stats: "
            f"min={np.min(fidelity_grid):.3e}, mean={np.mean(fidelity_grid):.3e}, max={np.max(fidelity_grid):.3e}"
        )

    return fidelity_grid


def run_parameter_sweep(
    param_sets,
    alpha_plus: complex = 1.0,
    alpha_minus: complex = 0.0,
    c_ops=None,
):
    """
    Evaluate full-cycle fidelity across a collection of SweepParameters.

    Returns a list of dicts with the parameters and resulting fidelity so that
    callers can log or plot sweep outcomes. Defaults keep legacy behaviour.
    """

    results = []
    for params in param_sets:
        overrides = build_runtime_args_from_params(params)
        gate_overrides = {"delta_m": overrides["delta_m"], "Omega_c_func": Omega_c_gate}
        read_overrides = {**overrides, "Omega_c_func": Omega_c_read}

        fidelity, _ = compute_full_cycle_fidelity(
            alpha_plus,
            alpha_minus,
            c_ops=c_ops,
            h_args=overrides,
            gate_h_args=gate_overrides,
            read_h_args=read_overrides,
        )
        results.append({"params": params, "fidelity": fidelity})

    return results


def check_unitarity(res) -> None:
    """
    Warn if mesolve evolution drifts away from unit norm.

    This is useful when experimenting with modified retrieval Hamiltonians or
    extended gate windows; it leaves the evolution unchanged and only reports
    if norms stray outside a tight tolerance.
    """

    norms = [state.norm() for state in res.states]
    if not norms:
        return

    min_norm, max_norm = min(norms), max(norms)
    if abs(min_norm - 1.0) > 1e-6 or abs(max_norm - 1.0) > 1e-6:
        print(
            f"WARNING: Non-unitary drift detected (min={min_norm:.6f}, max={max_norm:.6f})"
        )


def plot_full_cycle_diagnostics(alpha_plus: complex = 1.0, alpha_minus: complex = 0.0) -> None:
    """
    Visualise photon absorption/emission and ground-state population across write → gate → read.

    This concatenates all three phases so any discontinuities at window boundaries become
    visible. The write/gate/read regions are shaded for clarity; evolution itself is unchanged.
    """

    res_write, res_settle, res_gate, res_read = run_three_phase_sequence(alpha_plus, alpha_minus)

    all_states = res_write.states + res_settle.states + res_gate.states + res_read.states
    all_times = np.concatenate((tlist, tlist_settle, tlist_gate, tlist_read))

    n_plus = [expect(N_plus_full, s) for s in all_states]
    n_minus = [expect(N_minus_full, s) for s in all_states]
    pop_ground = [expect(P_gm1_full + P_g0_full + P_gp1_full, s) for s in all_states]

    t_us = all_times / Gamma_e * 1e6

    plt.figure(figsize=(12, 5))
    plt.axvspan(T_START / Gamma_e * 1e6, T_END / Gamma_e * 1e6, alpha=0.15, color="tab:blue", label="Write")
    plt.axvspan(T_END / Gamma_e * 1e6, T_SETTLE_END / Gamma_e * 1e6, alpha=0.15, color="tab:orange", label="Settle")
    plt.axvspan(T_GATE_START / Gamma_e * 1e6, T_GATE_END / Gamma_e * 1e6, alpha=0.15, color="tab:gray", label="Gate")
    plt.axvspan(T_READ_START / Gamma_e * 1e6, T_READ_END / Gamma_e * 1e6, alpha=0.15, color="tab:green", label="Read")

    plt.plot(t_us, n_plus, "r-", label="⟨n₊⟩")
    plt.plot(t_us, n_minus, "m-", label="⟨n₋⟩")
    plt.plot(t_us, pop_ground, "k-", label="P_ground")

    plt.xlabel("Time (µs)")
    plt.ylabel("Population / photon number")
    plt.title("Full-cycle photon and ground-state diagnostics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("full_cycle_diagnostics.pdf")


# ------------------------------------------------------------------
# 9. SU(3) utility functions (BARE BASIS)
# ------------------------------------------------------------------

def su3_generators_bare_3x3():
    """Return the eight Gell-Mann matrices acting on the bare ground manifold."""

    λ1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    λ2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
    λ3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
    λ4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
    λ5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
    λ6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    λ7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
    λ8 = (1 / np.sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex)
    return [λ1, λ2, λ3, λ4, λ5, λ6, λ7, λ8]


# ------------------------------------------------------------------
# 10. SU(3) Basis Adaptation: Dark Space & Leakage
# ------------------------------------------------------------------

def dark_bright_unitary_from_couplings(
    omega_c_val: complex,
    g_minus_val: complex,
    g_plus_val: complex,
) -> np.ndarray:
    """Return the dark/dark/bright basis aligned to the instantaneous coupling vector.

    The coupling vector to the excited state is ``c = [g_minus, Omega_c, g_plus]``.
    An SVD of this 1×3 row yields a right-singular matrix whose first column is
    the bright direction (parallel to ``c``) and remaining columns span the null
    space (the dark manifold). We reorder to ``[|D0>, |D1>, |B>]``.
    """

    coupling_row = np.array([[g_minus_val, omega_c_val, g_plus_val]], dtype=complex)

    u, s, vh = np.linalg.svd(coupling_row, full_matrices=True)
    V = vh.conj().T  # right singular vectors

    # Guard against a vanishing coupling vector: fall back to identity.
    if np.allclose(s, 0.0, atol=EPS):
        return np.eye(3, dtype=complex)

    bright = V[:, 0]
    dark1 = V[:, 1]
    dark2 = V[:, 2]

    Q = np.column_stack((dark1, dark2, bright))

    # Sanity: enforce orthonormality and warn if the bright vector drifts from
    # orthogonality due to numerical issues.
    bright_vec = Q[:, 2]
    dark_block = Q[:, :2]
    orth_error = np.linalg.norm(dark_block.conj().T @ bright_vec)
    if orth_error > 1e3 * np.sqrt(EPS):
        logging.warning("Dark/bright basis lost orthogonality: |<D|B>| = %.2e", orth_error)

    return Q  # Columns: |D0>, |D1>, |B> derived from instantaneous couplings


def get_dark_basis_unitary(E_iso: np.ndarray) -> np.ndarray:
    """Preserve compatibility: use encoder SVD when an explicit basis is requested."""

    u, s, _ = np.linalg.svd(E_iso, full_matrices=True)

    if s.size < 3 or s[-1] < 1e3 * EPS:
        logging.warning(
            "Encoder SVD is ill-conditioned (s_min=%.2e); falling back to identity basis",
            s[-1] if s.size else float("nan"),
        )
        return np.eye(3, dtype=complex)

    return u[:, :3]


def transform_and_classify_su3(U_atom: np.ndarray):
    """
    Rotate the bare Gell-Mann matrices into the dark-adapted basis and classify
    logical vs leakage generators.
    """

    bare_gen = su3_generators_bare_3x3()
    rotated_gen = []
    hermiticity_errors: list[float] = []
    trace_errors: list[float] = []

    leakage_indices = []
    logical_indices = []

    print("\n--- SU(3) Generator Classification (Dark-Adapted Basis) ---")

    for i, lam in enumerate(bare_gen):
        lam_prime = U_atom.conj().T @ lam @ U_atom  # rotate generator into dark basis

        herm_check = np.linalg.norm(lam_prime - lam_prime.conj().T)  # numerical Hermiticity guard
        hermiticity_errors.append(herm_check)
        if herm_check > 1e-10:
            print(
                f"WARNING: lambda'_{i + 1} lost Hermiticity! Norm(H - H^dag) = {herm_check:.2e}"
            )

        trace_err = abs(np.trace(lam_prime))
        trace_errors.append(trace_err)

        rotated_gen.append(lam_prime)

        block_A = lam_prime[0:2, 0:2]
        block_b = lam_prime[0:2, 2]

        norm_A = np.linalg.norm(block_A)
        norm_b = np.linalg.norm(block_b)

        scale = np.linalg.norm(lam_prime)
        tol = 1e-6 * scale if scale > 0 else 1e-6

        is_leakage = norm_b > tol
        is_logical = (not is_leakage) and (norm_A > tol)

        idx = i + 1  # 1-based index for display

        if is_leakage:
            leakage_indices.append(i)
            print(f"lambda'_{idx}: Leakage (Off-block norm = {norm_b:.4f})")
        elif is_logical:
            logical_indices.append(i)
            print(f"lambda'_{idx}: Logical (Block A norm = {norm_A:.4f})")
        else:
            print(f"lambda'_{idx}: Auxiliary/Bright (Acts primarily on |B>)")

    return rotated_gen, leakage_indices, logical_indices, hermiticity_errors, trace_errors


def get_full_space_operators(su3_atomic_matrices):
    """Lift 3x3 atomic matrices (ground manifold) to full atom+photon space."""

    ops_full = []
    for op3 in su3_atomic_matrices:
        op4 = np.zeros((4, 4), dtype=complex)
        op4[1:, 1:] = op3
        q_op = tensor(Qobj(op4), qeye(N_ph), qeye(N_ph))
        ops_full.append(q_op)
    return ops_full


def bright_projector_full(
    t: float,
    omega_c_func=Omega_c,
    g_minus_val=g_minus_t,
    g_plus_val=g_plus_t,
) -> Qobj:
    """
    Projector onto the instantaneous bright polariton combining atom and photons.

    Uses the coupling vector components to the excited state as amplitudes for the
    bright superposition |B> ∝ Ω_c|g_0;0> + g_−|g_-1;1_-> + g_+|g_+1;1_+>.
    """

    Oc = omega_c_func(t)
    gminus_t = evaluate_coupling(g_minus_val, t)
    gplus_t = evaluate_coupling(g_plus_val, t)

    ket_components = [
        (Oc, tensor(ket_g0, ket_0_plus, ket_0_minus)),
        (gminus_t, tensor(ket_gm1, ket_0_plus, ket_1_minus)),
        (gplus_t, tensor(ket_gp1, ket_1_plus, ket_0_minus)),
    ]

    ket_B = sum(coeff * state for coeff, state in ket_components)
    norm = ket_B.norm()
    if norm < np.sqrt(EPS):
        return tensor(qeye(4), qeye(N_ph), qeye(N_ph)) * 0.0

    ket_B = ket_B / norm
    return ket_B * ket_B.dag()


def bright_population_trajectory(states, times, omega_c_func=Omega_c, g_minus_val=g_minus_t, g_plus_val=g_plus_t):
    """Return bright-state population vs time using instantaneous couplings."""

    bright_pop = np.zeros(len(times))
    for idx, (t, state) in enumerate(zip(times, states)):
        bright_proj = bright_projector_full(t, omega_c_func, g_minus_val, g_plus_val)
        bright_pop[idx] = expect(bright_proj, state)
    return bright_pop


def su3_leakage_diagnostics(
    E_iso: np.ndarray,
    alpha_plus: complex = 1.0,
    alpha_minus: complex = 0.0,
    omega_c_func=Omega_c,
    g_minus_val: complex | None = None,
    g_plus_val: complex | None = None,
) -> dict:
    """Compute SU(3) expectations and bright-projector leakage with a time-dependent basis.

    The dark/bright basis is rebuilt at every time from the instantaneous coupling
    vector ``[g_minus, Omega_c(t), g_plus]`` so the leakage metric tracks the
    actual bright population rather than a static encoder-derived complement.
    """

    g_minus_use = g_minus_t if g_minus_val is None else g_minus_val
    g_plus_use = g_plus_t if g_plus_val is None else g_plus_val

    # Classify generators using the instantaneous basis at t0 for consistent
    # labelling, but expectations are evaluated with a time-dependent basis.
    U_init = dark_bright_unitary_from_couplings(
        omega_c_func(tlist[0]),
        evaluate_coupling(g_minus_use, tlist[0]),
        evaluate_coupling(g_plus_use, tlist[0]),
    )
    _, leak_idxs, log_idxs, _, _ = transform_and_classify_su3(U_init)

    res = run_trajectory(alpha_plus, alpha_minus)
    pop_e = expect(P_e_full, res.states)
    t_us = tlist / Gamma_e * 1e6

    n_steps = len(tlist)
    exps = [np.zeros(n_steps, dtype=complex) for _ in range(8)]
    herm_errors = np.zeros(8)
    trace_errors = np.zeros(8)
    bright_pop = np.zeros(n_steps)

    bare_gen = su3_generators_bare_3x3()

    for idx, (t, state) in enumerate(zip(tlist, res.states)):
        U_t = dark_bright_unitary_from_couplings(
            omega_c_func(t), evaluate_coupling(g_minus_use, t), evaluate_coupling(g_plus_use, t)
        )
        bright_proj = bright_projector_full(t, omega_c_func, g_minus_use, g_plus_use)
        bright_pop[idx] = expect(bright_proj, state)

        for j, lam in enumerate(bare_gen):
            lam_prime = U_t.conj().T @ lam @ U_t
            herm_errors[j] = max(herm_errors[j], np.linalg.norm(lam_prime - lam_prime.conj().T))
            trace_errors[j] = max(trace_errors[j], abs(np.trace(lam_prime)))

            op4 = np.zeros((4, 4), dtype=complex)
            op4[1:, 1:] = lam_prime
            op_full = tensor(Qobj(op4), qeye(N_ph), qeye(N_ph))
            exps[j][idx] = expect(op_full, state)

    r_leak_raw = np.array(bright_pop)
    r_leak = r_leak_raw  # no normalisation to preserve physical scale

    if np.std(pop_e) > 0 and np.std(r_leak_raw) > 0:
        leak_corr = float(np.corrcoef(r_leak_raw, pop_e)[0, 1])
    else:
        leak_corr = float("nan")

    return {
        "U_atom": U_init,
        "exps": exps,
        "pop_e": pop_e,
        "t_us": t_us,
        "r_leak": r_leak,
        "r_leak_raw": r_leak_raw,
        "bright_pop": bright_pop,
        "leak_idxs": leak_idxs,
        "log_idxs": log_idxs,
        "hermiticity_errors": herm_errors.tolist(),
        "trace_errors": trace_errors.tolist(),
        "leak_corr_with_pe": leak_corr,
    }


def plot_su3_expectations(
    E_iso: np.ndarray,
    alpha_plus: complex = 1.0,
    alpha_minus: complex = 0.0,
    collector: list[str] | None = None,
) -> dict:
    """
    Plot expectation values of SU(3) generators transformed into the dark-adapted
    basis determined by the isometric encoder E_iso, plus the basis-agnostic leakage radius.
    Returns the diagnostics for logging.
    """

    diag = su3_leakage_diagnostics(E_iso, alpha_plus=alpha_plus, alpha_minus=alpha_minus)
    exps = diag["exps"]
    leak_idxs = diag["leak_idxs"]
    log_idxs = diag["log_idxs"]
    t_us = diag["t_us"]
    r_leak = diag["r_leak"]

    labels = [f"$\\lambda'_{i + 1}$" for i in range(8)]

    plt.figure(figsize=(10, 6))

    for i in log_idxs:
        plt.plot(t_us, exps[i], linestyle="-", linewidth=2, label=f"{labels[i]} (Logical)")

    for i in leak_idxs:
        plt.plot(
            t_us,
            exps[i],
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"{labels[i]} (Leakage)",
        )

    plt.xlabel("Time (µs)")
    plt.ylabel("Expectation value (Dark Basis)")
    plt.title("SU(3) Expectations in Physical Dark Basis")
    plt.legend(ncol=2, fontsize="small", loc="upper right")
    plt.grid(True, alpha=0.3)
    save_current_figure("su3_expectations.pdf")

    plt.figure(figsize=(8, 5))
    plt.plot(t_us, r_leak, "k-", linewidth=2)
    plt.xlabel("Time (µs)")
    plt.ylabel(r"Basis-Adapted Leakage Radius $r_{leak}(t)$")
    plt.title("Leakage out of Physical Dark Manifold")
    plt.grid(True, alpha=0.3)
    save_current_figure("leakage_radius.pdf")

    if collector is not None:
        collector.append(f"Hermiticity errors (max): {max(diag['hermiticity_errors']):.3e}")
        collector.append(f"Trace errors (max): {max(diag['trace_errors']):.3e}")
        collector.append(f"Leakage radius max (raw): {np.max(diag['r_leak_raw']):.3e}")
        collector.append(f"Leakage vs P_e correlation: {diag['leak_corr_with_pe']:.3e}")

    return diag


# ------------------------------------------------------------------
# 11. Spectral Scan: EIT Transparency Window
# ------------------------------------------------------------------

def scan_transparency_window(scan_range_mhz: float = 15.0, n_points: int = 41):
    """
    Scan the two-photon detuning of the probe field relative to the control field
    and measure total scattering.
    """

    print(f"\nScanning EIT transparency window (+/- {scan_range_mhz} MHz)...")

    range_dimless = (2 * np.pi * scan_range_mhz * 1e6) / Gamma_e  # convert MHz span to simulation units
    deltas = np.linspace(-range_dimless, range_dimless, n_points)

    integrated_Pe = []

    psi_in = psi_in_sigma_plus  # pure sigma+ input

    for d in deltas:
        current_delta_m = np.array([d, 0.0, d])

        res = mesolve(
            H_tripod_t,
            psi_in,
            tlist,
            c_ops=[],
            e_ops=[P_e_full],
            args={
                "delta_m": current_delta_m,
                "Omega_c_func": Omega_c,
                "g_plus": g_plus_t,
                "g_minus": g_minus_t,
            },
            options=Options(store_states=False),
        )

        total_scat = np.trapz(res.expect[0], tlist)  # area under excited-state population
        integrated_Pe.append(total_scat)

    integrated_Pe = np.array(integrated_Pe)
    deltas_mhz = deltas * Gamma_e / (2 * np.pi * 1e6)

    plt.figure(figsize=(8, 5))
    plt.plot(deltas_mhz, integrated_Pe, "o-", markersize=4, label="Integrated $P_e$")

    Omega_c_mhz = Omega_C_phys / (2 * np.pi * 1e6)
    plt.axvline(x=-Omega_c_mhz, color="r", linestyle="--", alpha=0.5, label=r"$\pm \Omega_c$")
    plt.axvline(x=Omega_c_mhz, color="r", linestyle="--", alpha=0.5)
    plt.axvline(x=0, color="k", linestyle=":", alpha=0.8)

    plt.xlabel("Two-photon detuning $\\delta$ (MHz)")
    plt.ylabel("Total Scattering (arb. units)")
    plt.title(f"EIT Transparency Window\n(Control $\\Omega_c \\approx$ {Omega_c_mhz:.1f} MHz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("transparency_window.pdf")


def run_EIT_validation():
    """Standalone EIT checks comparing control on/off and a window scan."""

    print("\nRunning EIT validation scan...")

    res_on = run_trajectory(alpha_plus=1.0, alpha_minus=0.0)
    res_off = run_trajectory(
        alpha_plus=1.0,
        alpha_minus=0.0,
        h_args={"Omega_c_func": lambda _t: 0.0},
    )

    Pe_on = expect(P_e_full, res_on.states)
    Pe_off = expect(P_e_full, res_off.states)
    t_us = tlist / Gamma_e * 1e6

    plt.figure(figsize=(8, 5))
    plt.plot(t_us, Pe_on, label="Control ON")
    plt.plot(t_us, Pe_off, label="Control OFF")
    plt.title("EIT Transparency Check: Excited-State Population")
    plt.xlabel("Time (µs)")
    plt.ylabel(r"$P_e(t)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure("EIT_validation_Pe.pdf")

    scan_transparency_window()


if __name__ == "__main__":
    try:
        _ensure_figures_dir()
        print(f"Saving figures to {FIGURES_DIR.resolve()}")

        run_outputs: list[str] = []

        angle_stats = mixing_angle_stats()
        dark_residuals = dark_coupling_residuals(tlist)
        run_outputs.append(
            "Mixing angle stats: "
            f"theta_start={angle_stats['theta_start']:.3e}, "
            f"theta_end={angle_stats['theta_end']:.3e}, "
            f"theta_min={angle_stats['theta_min']:.3e}, "
            f"theta_max={angle_stats['theta_max']:.3e}"
        )
        run_outputs.append(
            f"Dark-coupling residuals: max={dark_residuals.max():.3e}, mean={dark_residuals.mean():.3e}"
        )

        print("Plotting pulses and mixing angle...")
        plot_pulses_and_mixing()
        plot_full_timeline_pulses()
        plot_dark_residuals(dark_residuals)

        E_phys, E_iso, eta, G_phys, G_iso, eig_G_phys, eig_G_iso, encoder_metrics = build_encoder()
        run_outputs.append(f"Average storage efficiency eta = {eta:.6f}")
        run_outputs.append(f"Eigenvalues of G_phys: {eig_G_phys}")
        run_outputs.append(
            f"Residual photons after write: n_plus={encoder_metrics['residual_n_plus']:.3e}, "
            f"n_minus={encoder_metrics['residual_n_minus']:.3e}"
        )
        run_outputs.append(
            f"Encoder orthogonality <c+|c->: {encoder_metrics['orthogonality_cplus_cminus']:.3e}"
        )
        run_outputs.append(f"Condition number of E_phys: {encoder_metrics['condition_number']:.3e}")
        run_outputs.append(f"Singular values of E_phys: {encoder_metrics['singular_values']}")
        run_outputs.append(f"Renormalisation gaps: {encoder_metrics['renorm_gap']}")
        run_outputs.append(
            f"Final P_e after write (sigma+, sigma-): {encoder_metrics['pop_e_plus']:.3e}, {encoder_metrics['pop_e_minus']:.3e}"
        )
        encoder_validation_diagnostics(collector=run_outputs)

        print("\nRunning EIT-style diagnostics for input σ+ ...")
        plot_eit_and_populations(alpha_plus=1.0, alpha_minus=0.0, collector=run_outputs)
        plot_bright_population(alpha_plus=1.0, alpha_minus=0.0, collector=run_outputs)

        print("\nPlotting Adaptive SU(3) Expectations and Leakage Radius...")
        su3_diag = plot_su3_expectations(E_iso, alpha_plus=1.0, alpha_minus=0.0, collector=run_outputs)

        run_EIT_validation()

        fidelity, _ = compute_full_cycle_fidelity(alpha_plus=1.0, alpha_minus=0.0)
        print(f"\nFull write → gate → read fidelity for σ+: {fidelity:.6f}")
        run_outputs.append(f"Full-cycle fidelity for sigma+: {fidelity:.6f}")

        cycle_metrics = compute_full_cycle_metrics(E_iso, alpha_plus=1.0, alpha_minus=0.0)
        run_outputs.append(
            f"Photon return probability (n_plus+n_minus) after read: {cycle_metrics['p_return']:.6f}"
        )
        run_outputs.append(
            f"Ground population after read: {cycle_metrics['p_ground_final']:.6f}"
        )
        run_outputs.append(
            f"Bright projection at read start: {cycle_metrics['bright_at_read_start']:.3e}; "
            f"at write end: {cycle_metrics['bright_at_write_end']:.3e}"
        )
        run_outputs.append(
            f"Norm drift across full cycle: min={min(cycle_metrics['norms']):.6e}, max={max(cycle_metrics['norms']):.6e}"
        )

        plot_readout_photon_dynamics(alpha_plus=1.0, alpha_minus=0.0, collector=run_outputs)

        print("\nTomography validation (retrieval fidelities):")
        report_retrieval_tomography(collector=run_outputs)
        plot_retrieval_fidelity_landscape(collector=run_outputs)
        write_run_outputs(run_outputs)
        print("Figure generation complete. Inspect thesis_figures/ for outputs.")
    except Exception:
        logging.exception("Fatal error during tripod simulation run")
        print(
            f"Fatal error encountered. Details logged to {ERROR_LOG_PATH}",
            flush=True,
        )
        raise
