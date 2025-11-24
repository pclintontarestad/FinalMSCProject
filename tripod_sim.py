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
Delta_phys = 2 * np.pi * 100e6  # one-photon detuning (Hz), "moderate detuning"

# Control/probe Rabi frequencies (Hz)
Omega_C_phys = 2 * np.pi * 5e6  # control Rabi frequency
Omega_P_phys = 2 * np.pi * 0.3e6  # "typical" probe scale (used to set coupling g)

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
    g_plus_runtime: float = g_plus,
    g_minus_runtime: float = g_minus,
) -> np.ndarray:
    """
    Compute |<e|H|D>| / ||coupling|| across time by nulling the coupling vector.

    The bright coupling vector to |e> is proportional to (g_minus, Omega_c(t), g_plus);
    the nullspace of this 1x3 row gives two dark vectors. Any non-zero overlap of the
    coupling with those dark vectors flags imperfect darkness.
    """

    residuals: list[float] = []
    for t in times:
        coupling_vec = np.array([g_minus_runtime, Omega_c_func(t), g_plus_runtime], dtype=complex)
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
    g_plus_runtime = args.get("g_plus", g_plus)
    g_minus_runtime = args.get("g_minus", g_minus)
    omega_c_func = args.get("Omega_c_func", Omega_c)

    # Diagonal detunings: excited-state offset plus optional ground offsets
    H = (-Delta) * P_e_full
    H += delta[0] * P_gm1_full
    H += delta[1] * P_g0_full
    H += delta[2] * P_gp1_full

    # σ- coupling: |g_{-1};1_-> ↔ |e;0_-> (Δm = -1)
    H += g_minus_runtime * (a_minus * sig_egm1.dag() + a_minus.dag() * sig_egm1)

    # σ+ coupling: |g_{+1};1_+> ↔ |e;0_+> (Δm = +1)
    H += g_plus_runtime * (a_plus * sig_egp1.dag() + a_plus.dag() * sig_egp1)

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
        "g_plus": g_plus_dimless,
        "g_minus": g_minus_dimless,
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
        "g_plus": g_plus,
        "g_minus": g_minus,
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


# ------------------------------------------------------------------
# 7. EIT-like diagnostics: populations vs time
# ------------------------------------------------------------------

# Photon number operators in full space
N_plus_full = tensor(qeye(4), num(N_ph), qeye(N_ph))
N_minus_full = tensor(qeye(4), qeye(N_ph), num(N_ph))


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
        "g_plus": g_plus,
        "g_minus": g_minus,
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


def plot_eit_and_populations(alpha_plus: complex = 1.0, alpha_minus: complex = 0.0) -> None:
    """Plot atomic populations and photon number diagnostics."""

    res = run_trajectory(alpha_plus, alpha_minus)

    pop_e = expect(P_e_full, res.states)
    pop_gm1 = expect(P_gm1_full, res.states)
    pop_g0 = expect(P_g0_full, res.states)
    pop_gp1 = expect(P_gp1_full, res.states)

    n_plus = expect(N_plus_full, res.states)
    n_minus = expect(N_minus_full, res.states)

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


# ------------------------------------------------------------------
# 8. Three-period EIT cycle: write, gate, read
# ------------------------------------------------------------------

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


def run_gate_phase(psi_in: Qobj, c_ops=None, h_args: dict | None = None):
    """Propagate through the gate window with lasers off to track leakage."""

    runtime_c_ops = DEFAULT_C_OPS if c_ops is None else c_ops
    args = {
        "delta_m": delta_m,
        "Omega_c_func": Omega_c_gate,
        "g_plus": 0.0,
        "g_minus": 0.0,
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
        "g_plus": g_plus,
        "g_minus": g_minus,
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
        "g_plus": g_plus,
        "g_minus": g_minus,
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
        read_h_args={"Omega_p_func": Omega_p_reverse} | (read_h_args or {}),
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
        read_h_args={"Omega_p_func": Omega_p_reverse} | (read_h_args or {}),
    )

    psi_out = res_read.states[-1]
    n_plus_final = expect(N_plus_full, psi_out)
    n_minus_final = expect(N_minus_full, psi_out)
    p_return = n_plus_final + n_minus_final
    p_ground_final = expect(P_gm1_full + P_g0_full + P_gp1_full, psi_out)

    U_atom = get_dark_basis_unitary(E_iso)
    bright_proj = bright_projector_full(U_atom)
    bright_at_read_start = expect(bright_proj, res_read.states[0])
    bright_at_write_end = expect(bright_proj, res_write.states[-1])

    return {
        "p_return": p_return,
        "p_ground_final": p_ground_final,
        "bright_at_read_start": bright_at_read_start,
        "bright_at_write_end": bright_at_write_end,
        "norms": [state.norm() for state in res_write.states + res_settle.states + res_gate.states + res_read.states],
    }


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
        read_overrides = {**overrides, "Omega_c_func": Omega_c_read, "Omega_p_func": Omega_p_reverse}

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

def get_dark_basis_unitary(E_iso: np.ndarray) -> np.ndarray:
    """
    Construct the transformation U that maps the standard SU(3) basis to the
    Dark/Bright basis using an SVD of the *isometric* encoder.

    Using SVD guarantees that the third column is the orthogonal complement to
    the two-dimensional dark manifold (i.e., the bright direction), avoiding the
    arbitrary third column produced by QR. This keeps the leakage metric tied to
    the physically correct bright axis rather than a numerical artefact.
    """

    u, _, _ = np.linalg.svd(E_iso, full_matrices=True)
    Q = u[:, :3]

    # Sanity: enforce orthonormality and warn if the bright vector drifts from
    # orthogonality due to numerical issues.
    bright = Q[:, 2]
    dark_block = Q[:, :2]
    orth_error = np.linalg.norm(dark_block.conj().T @ bright)
    if orth_error > 1e3 * np.sqrt(EPS):
        logging.warning("Dark/bright basis lost orthogonality: |<D|B>| = %.2e", orth_error)

    return Q  # Columns: |D0>, |D1>, |B> derived from SVD


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


def bright_projector_full(U_atom: np.ndarray) -> Qobj:
    """Projector onto the bright state embedded in atom ⊗ photon identity."""

    bright = U_atom[:, 2]
    proj3 = np.outer(bright, bright.conj())
    op4 = np.zeros((4, 4), dtype=complex)
    op4[1:, 1:] = proj3
    return tensor(Qobj(op4), qeye(N_ph), qeye(N_ph))


def su3_leakage_diagnostics(
    E_iso: np.ndarray, alpha_plus: complex = 1.0, alpha_minus: complex = 0.0
) -> dict:
    """
    Compute SU(3) expectations, leakage radius, and basis health metrics without plotting.
    """

    U_atom = get_dark_basis_unitary(E_iso)
    rotated_matrices, leak_idxs, log_idxs, herm_errors, trace_errors = transform_and_classify_su3(U_atom)
    ops_full = get_full_space_operators(rotated_matrices)

    res = run_trajectory(alpha_plus, alpha_minus)

    exps = [expect(op, res.states) for op in ops_full]
    pop_e = expect(P_e_full, res.states)
    t_us = tlist / Gamma_e * 1e6

    leakage_sq_sum = np.zeros_like(t_us)
    for idx in leak_idxs:
        leakage_sq_sum += np.abs(exps[idx]) ** 2
    r_leak = np.sqrt(leakage_sq_sum)
    if np.max(r_leak) > 0:
        r_leak = r_leak / np.max(r_leak)

    if np.std(pop_e) > 0 and np.std(r_leak) > 0:
        leak_corr = float(np.corrcoef(r_leak**2, pop_e)[0, 1])
    else:
        leak_corr = float("nan")

    return {
        "U_atom": U_atom,
        "exps": exps,
        "pop_e": pop_e,
        "t_us": t_us,
        "r_leak": r_leak,
        "leak_idxs": leak_idxs,
        "log_idxs": log_idxs,
        "hermiticity_errors": herm_errors,
        "trace_errors": trace_errors,
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
            args={"delta_m": current_delta_m},
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

        print("\nRunning EIT-style diagnostics for input σ+ ...")
        plot_eit_and_populations(alpha_plus=1.0, alpha_minus=0.0)

        print("\nPlotting Adaptive SU(3) Expectations and Leakage Radius...")
        su3_diag = plot_su3_expectations(E_iso, alpha_plus=1.0, alpha_minus=0.0, collector=run_outputs)

        run_outputs.append(
            f"Leakage radius max (normalised): {np.max(su3_diag['r_leak']):.3e}"
        )
        run_outputs.append(
            f"Leakage vs P_e correlation: {su3_diag['leak_corr_with_pe']:.3e}"
        )

        scan_transparency_window(scan_range_mhz=20.0, n_points=51)

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

        print("\nTomography validation (retrieval fidelities):")
        report_retrieval_tomography(collector=run_outputs)
        write_run_outputs(run_outputs)
        print("Figure generation complete. Inspect thesis_figures/ for outputs.")
    except Exception:
        logging.exception("Fatal error during tripod simulation run")
        print(
            f"Fatal error encountered. Details logged to {ERROR_LOG_PATH}",
            flush=True,
        )
        raise
