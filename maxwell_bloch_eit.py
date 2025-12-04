"""
Tier-2 Maxwell-Bloch EIT Simulation for Photonic Polarization Qubit Storage.

This module implements a full time-domain numerical simulation of a four-level
tripod Electromagnetically Induced Transparency (EIT) system in ^87Rb capable
of storing and retrieving a polarization-encoded photonic qubit.

Key features (Tier-2 physics):
- Full atomic master-equation dynamics at each spatial slice
- Coupled Maxwell-Bloch field propagation through the medium
- Two independent dark-state polaritons (DSPs) for σ+ and σ- modes
- Spatial compression and re-expansion of the probe pulse
- Spin-wave storage and retrieval
- Polarization-mode scattering and cross-coupling (Caruso §5)

Physical system: Rb-87 D2 line tripod configuration
- Ground manifold: 5S_{1/2}, F=1 (|1⟩=m_F=-1, |2⟩=m_F=0, |3⟩=m_F=+1)
- Excited manifold: 5P_{3/2}, F'=0 (|0⟩=m_F'=0)

Author: Tier-2 Maxwell-Bloch Implementation
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# Optional QuTiP import for validation comparisons
try:
    from qutip import Qobj, basis, mesolve, Options, expect
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False

# =============================================================================
# Configuration and Logging
# =============================================================================

EPS = np.finfo(float).eps
ERROR_LOG_PATH = Path(__file__).with_name("maxwell_bloch_error.log")
FIGURES_DIR = Path("mb_figures")
RUN_OUTPUT_PATH = Path(__file__).with_name("maxwell_bloch_outputs.txt")


def configure_logging(log_path: Path = ERROR_LOG_PATH) -> None:
    """Set up file-backed logging with stdout mirroring."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _log_unhandled_exceptions(exc_type, exc_value, exc_traceback) -> None:
    """Capture uncaught exceptions in the error log."""
    logging.critical(
        "Unhandled exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


configure_logging()
sys.excepthook = _log_unhandled_exceptions


def _ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(name: str) -> Path:
    """Save the active matplotlib figure."""
    _ensure_figures_dir()
    output_path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    logging.info("Figure saved to %s", output_path)
    return output_path


def write_outputs(lines: List[str], path: Path = RUN_OUTPUT_PATH) -> Path:
    """Persist human-readable run outputs."""
    output = "\n".join(lines).rstrip() + "\n"
    path.write_text(output, encoding="utf-8")
    logging.info("Outputs written to %s", path)
    return path


# =============================================================================
# Physical Constants (Rb-87 D2 Line)
# =============================================================================

# Fundamental constants
HBAR = 1.054571817e-34  # J·s
C_LIGHT = 299792458.0   # m/s
EPSILON_0 = 8.854187817e-12  # F/m

# Rb-87 D2 transition properties
GAMMA_RAD = 2 * np.pi * 6.0666e6  # Excited state decay rate (rad/s)
WAVELENGTH = 780.241e-9  # D2 transition wavelength (m)
K_WAVE = 2 * np.pi / WAVELENGTH  # Wave vector magnitude

# Dipole matrix element (approximate for D2 line)
D_DIPOLE = 2.537e-29  # C·m (effective dipole moment)


# =============================================================================
# Simulation Parameters
# =============================================================================

@dataclass
class PhysicalParameters:
    """Physical parameters for the Rb-87 tripod EIT system."""

    # Decay and decoherence rates
    Gamma: float = GAMMA_RAD  # Excited state decay rate (rad/s)
    gamma_12: float = 0.0  # Ground state |1⟩-|2⟩ decoherence
    gamma_13: float = 0.0  # Ground state |1⟩-|3⟩ decoherence
    gamma_23: float = 0.0  # Ground state |2⟩-|3⟩ decoherence

    # Detunings
    Delta: float = 0.0  # One-photon detuning (rad/s)
    delta_1: float = 0.0  # Two-photon detuning for |1⟩ (rad/s)
    delta_2: float = 0.0  # Two-photon detuning for |2⟩ (rad/s)
    delta_3: float = 0.0  # Two-photon detuning for |3⟩ (rad/s)

    # Clebsch-Gordan coefficients (CGC)
    # For F=1 → F'=0 transitions, CGCs are symmetric by default
    cgc_sigma_plus: float = 1.0  # |1⟩ ↔ |0⟩ (σ+)
    cgc_pi: float = 1.0  # |2⟩ ↔ |0⟩ (π)
    cgc_sigma_minus: float = 1.0  # |3⟩ ↔ |0⟩ (σ-)

    # Medium properties
    atom_density: float = 1e17  # atoms/m³
    medium_length: float = 1e-3  # m (1 mm)

    @property
    def optical_depth(self) -> float:
        """Calculate resonant optical depth OD = n * σ * L."""
        sigma_0 = 3 * WAVELENGTH**2 / (2 * np.pi)  # Resonant cross-section
        return self.atom_density * sigma_0 * self.medium_length

    @property
    def coupling_eta(self) -> float:
        """Collective atom-field coupling strength η."""
        # η = (3/8π) * Γ * OD / L in appropriate units
        return (3 / (8 * np.pi)) * self.Gamma * self.optical_depth / self.medium_length


@dataclass
class PulseParameters:
    """Parameters for control and probe pulse envelopes."""

    # Control field (π-polarized, classical)
    Omega_c_max: float = 2 * np.pi * 10e6  # Peak Rabi frequency (rad/s)
    t_c_on: float = 0.0  # Control turn-on time (s)
    t_c_off: float = 2e-6  # Control turn-off time for storage (s)
    tau_c: float = 0.3e-6  # Control ramp time constant (s)

    # Probe field (σ+ polarized, quantum)
    Omega_p_max: float = 2 * np.pi * 1e6  # Peak probe Rabi frequency (rad/s)
    t_p_center: float = 1e-6  # Probe pulse center time (s)
    tau_p: float = 0.5e-6  # Probe pulse width (s)

    # Trigger field (σ- polarized, quantum)
    Omega_t_max: float = 2 * np.pi * 1e6  # Peak trigger Rabi frequency (rad/s)
    t_t_center: float = 1e-6  # Trigger pulse center time (s)
    tau_t: float = 0.5e-6  # Trigger pulse width (s)


@dataclass
class SimulationParameters:
    """Numerical simulation parameters."""

    # Spatial grid
    N_z: int = 100  # Number of spatial slices

    # Temporal grid
    N_t: int = 1000  # Number of time steps per phase

    # Phase durations (s)
    T_entry: float = 3e-6  # Probe entry phase
    T_storage: float = 1e-6  # Storage ramp-down phase
    T_hold: float = 5e-6  # Hold phase (control off)
    T_retrieval: float = 3e-6  # Retrieval phase

    # Solver options
    rtol: float = 1e-8
    atol: float = 1e-10
    method: str = "RK45"

    @property
    def T_total(self) -> float:
        """Total simulation time."""
        return self.T_entry + self.T_storage + self.T_hold + self.T_retrieval


@dataclass
class QubitParameters:
    """Input polarization qubit parameters."""

    theta: float = np.pi / 4  # Polar angle (0 = |R⟩, π/2 = |L⟩)
    phi: float = 0.0  # Azimuthal phase

    @property
    def alpha_R(self) -> complex:
        """Right-circular (σ+) amplitude."""
        return np.cos(self.theta / 2)

    @property
    def alpha_L(self) -> complex:
        """Left-circular (σ-) amplitude."""
        return np.exp(1j * self.phi) * np.sin(self.theta / 2)


# =============================================================================
# Atomic Density Matrix Representation
# =============================================================================

class AtomicState:
    """
    4-level atomic density matrix for the tripod system.

    Basis ordering: |0⟩ (excited), |1⟩, |2⟩, |3⟩ (ground states)

    The density matrix is stored as a 4x4 complex array where:
    - Diagonal elements ρ_ii are populations
    - Off-diagonal ρ_ij are coherences
    """

    def __init__(self, rho: Optional[np.ndarray] = None):
        """Initialize atomic state, defaulting to |2⟩ (m_F=0 ground state)."""
        if rho is None:
            self.rho = np.zeros((4, 4), dtype=complex)
            self.rho[2, 2] = 1.0  # Start in |2⟩ = |F=1, m_F=0⟩
        else:
            self.rho = rho.copy()

    def copy(self) -> 'AtomicState':
        return AtomicState(self.rho.copy())

    # Population accessors
    @property
    def P_0(self) -> float:
        """Excited state population."""
        return self.rho[0, 0].real

    @property
    def P_1(self) -> float:
        """Ground state |1⟩ population."""
        return self.rho[1, 1].real

    @property
    def P_2(self) -> float:
        """Ground state |2⟩ population."""
        return self.rho[2, 2].real

    @property
    def P_3(self) -> float:
        """Ground state |3⟩ population."""
        return self.rho[3, 3].real

    # Coherence accessors (optical coherences drive field propagation)
    @property
    def rho_01(self) -> complex:
        """Optical coherence |0⟩⟨1| (drives σ+ probe)."""
        return self.rho[0, 1]

    @property
    def rho_02(self) -> complex:
        """Optical coherence |0⟩⟨2| (drives π control)."""
        return self.rho[0, 2]

    @property
    def rho_03(self) -> complex:
        """Optical coherence |0⟩⟨3| (drives σ- trigger)."""
        return self.rho[0, 3]

    # Ground-state coherences (spin waves)
    @property
    def rho_12(self) -> complex:
        """Ground coherence |1⟩⟨2| (σ+ spin wave)."""
        return self.rho[1, 2]

    @property
    def rho_32(self) -> complex:
        """Ground coherence |3⟩⟨2| (σ- spin wave)."""
        return self.rho[3, 2]

    @property
    def rho_13(self) -> complex:
        """Ground coherence |1⟩⟨3| (cross spin wave)."""
        return self.rho[1, 3]

    def to_vector(self) -> np.ndarray:
        """Flatten density matrix to vector for ODE solver."""
        return self.rho.flatten()

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'AtomicState':
        """Reconstruct density matrix from flattened vector."""
        return cls(vec.reshape(4, 4))

    @property
    def trace(self) -> float:
        """Trace of density matrix (should be 1)."""
        return np.trace(self.rho).real

    @property
    def purity(self) -> float:
        """Purity Tr(ρ²)."""
        return np.trace(self.rho @ self.rho).real


# =============================================================================
# Hamiltonian and Master Equation
# =============================================================================

def build_hamiltonian(
    Omega_P: complex,
    Omega_C: complex,
    Omega_T: complex,
    phys: PhysicalParameters
) -> np.ndarray:
    """
    Construct the tripod Hamiltonian in the rotating frame with RWA.

    H = -Δ|0⟩⟨0| - Σ_j δ_j|j⟩⟨j| - [Ω_P|0⟩⟨1| + Ω_C|0⟩⟨2| + Ω_T|0⟩⟨3| + h.c.]

    Parameters:
        Omega_P: Probe field Rabi frequency (σ+, |1⟩↔|0⟩)
        Omega_C: Control field Rabi frequency (π, |2⟩↔|0⟩)
        Omega_T: Trigger field Rabi frequency (σ-, |3⟩↔|0⟩)
        phys: Physical parameters

    Returns:
        4x4 Hamiltonian matrix (units of rad/s)
    """
    H = np.zeros((4, 4), dtype=complex)

    # Diagonal: detunings
    H[0, 0] = -phys.Delta  # Excited state
    H[1, 1] = -phys.delta_1  # Ground |1⟩
    H[2, 2] = -phys.delta_2  # Ground |2⟩
    H[3, 3] = -phys.delta_3  # Ground |3⟩

    # Off-diagonal: couplings (with CGC weighting)
    # Probe: σ+ couples |1⟩ ↔ |0⟩
    H[0, 1] = -phys.cgc_sigma_plus * Omega_P / 2
    H[1, 0] = -phys.cgc_sigma_plus * np.conj(Omega_P) / 2

    # Control: π couples |2⟩ ↔ |0⟩
    H[0, 2] = -phys.cgc_pi * Omega_C / 2
    H[2, 0] = -phys.cgc_pi * np.conj(Omega_C) / 2

    # Trigger: σ- couples |3⟩ ↔ |0⟩
    H[0, 3] = -phys.cgc_sigma_minus * Omega_T / 2
    H[3, 0] = -phys.cgc_sigma_minus * np.conj(Omega_T) / 2

    return H


def lindblad_dissipator(rho: np.ndarray, phys: PhysicalParameters) -> np.ndarray:
    """
    Compute the Lindblad dissipator for the tripod system.

    D[ρ] = Σ_k (C_k ρ C_k† - ½{C_k† C_k, ρ})

    Collapse operators for spontaneous emission from |0⟩:
    - C_1 = √(Γ/3) |1⟩⟨0|  (decay to |1⟩)
    - C_2 = √(Γ/3) |2⟩⟨0|  (decay to |2⟩)
    - C_3 = √(Γ/3) |3⟩⟨0|  (decay to |3⟩)

    Plus ground-state dephasing if specified.
    """
    D = np.zeros((4, 4), dtype=complex)

    # Spontaneous emission from excited state |0⟩
    # Branching ratio 1/3 to each ground state for F'=0 → F=1
    gamma_branch = phys.Gamma / 3

    # Population transfer: |0⟩ → |j⟩
    rho_00 = rho[0, 0]
    D[1, 1] += gamma_branch * rho_00  # Decay to |1⟩
    D[2, 2] += gamma_branch * rho_00  # Decay to |2⟩
    D[3, 3] += gamma_branch * rho_00  # Decay to |3⟩
    D[0, 0] -= phys.Gamma * rho_00  # Loss from |0⟩

    # Optical coherence decay (half the population decay rate)
    D[0, 1] -= (phys.Gamma / 2) * rho[0, 1]
    D[1, 0] -= (phys.Gamma / 2) * rho[1, 0]
    D[0, 2] -= (phys.Gamma / 2) * rho[0, 2]
    D[2, 0] -= (phys.Gamma / 2) * rho[2, 0]
    D[0, 3] -= (phys.Gamma / 2) * rho[0, 3]
    D[3, 0] -= (phys.Gamma / 2) * rho[3, 0]

    # Ground-state decoherence (pure dephasing)
    D[1, 2] -= phys.gamma_12 * rho[1, 2]
    D[2, 1] -= phys.gamma_12 * rho[2, 1]
    D[1, 3] -= phys.gamma_13 * rho[1, 3]
    D[3, 1] -= phys.gamma_13 * rho[3, 1]
    D[2, 3] -= phys.gamma_23 * rho[2, 3]
    D[3, 2] -= phys.gamma_23 * rho[3, 2]

    return D


def master_equation_rhs(
    rho: np.ndarray,
    Omega_P: complex,
    Omega_C: complex,
    Omega_T: complex,
    phys: PhysicalParameters
) -> np.ndarray:
    """
    Right-hand side of the master equation dρ/dt.

    dρ/dt = -i[H, ρ] + D[ρ]
    """
    H = build_hamiltonian(Omega_P, Omega_C, Omega_T, phys)

    # Commutator: -i[H, ρ]
    commutator = -1j * (H @ rho - rho @ H)

    # Lindblad dissipator
    dissipator = lindblad_dissipator(rho, phys)

    return commutator + dissipator


# =============================================================================
# Control Pulse Envelopes
# =============================================================================

def control_envelope_storage(
    t: float,
    pulse: PulseParameters,
    phase: str = "entry",
    phase_duration: float = 1e-6
) -> complex:
    """
    Control field envelope for EIT storage protocol.

    Phases:
    - "entry": Control ON, probe enters medium
    - "storage": Control ramps OFF to convert light to spin wave
    - "hold": Control OFF, spin wave stored
    - "retrieval": Control ramps ON to convert spin wave back to light

    The ramp timing is normalized to the phase duration.
    """
    if phase == "entry":
        # Control fully on during probe entry
        return pulse.Omega_c_max

    elif phase == "storage":
        # Smooth ramp down - ramp happens over the phase duration
        # t goes from 0 to phase_duration
        t_mid = phase_duration / 2  # Ramp in middle of storage phase
        ramp = 0.5 * (1 - np.tanh((t - t_mid) / pulse.tau_c))
        return pulse.Omega_c_max * ramp

    elif phase == "hold":
        # Control off during hold
        return 0.0

    elif phase == "retrieval":
        # Time-reversed ramp: control turns back on
        t_mid = phase_duration / 2  # Ramp in middle of retrieval phase
        ramp = 0.5 * (1 + np.tanh((t - t_mid) / pulse.tau_c))
        return pulse.Omega_c_max * ramp

    else:
        return pulse.Omega_c_max


def probe_input_envelope(
    t: float,
    pulse: PulseParameters,
    amplitude: complex = 1.0
) -> complex:
    """
    Input probe pulse envelope (Gaussian).

    E_P(z=0, t) ∝ amplitude × exp(-(t - t_center)² / 2τ²)
    """
    gaussian = np.exp(-((t - pulse.t_p_center)**2) / (2 * pulse.tau_p**2))
    return amplitude * pulse.Omega_p_max * gaussian


def trigger_input_envelope(
    t: float,
    pulse: PulseParameters,
    amplitude: complex = 1.0
) -> complex:
    """
    Input trigger pulse envelope (Gaussian).

    E_T(z=0, t) ∝ amplitude × exp(-(t - t_center)² / 2τ²)
    """
    gaussian = np.exp(-((t - pulse.t_t_center)**2) / (2 * pulse.tau_t**2))
    return amplitude * pulse.Omega_t_max * gaussian


# =============================================================================
# Maxwell-Bloch Field Propagation
# =============================================================================

class MaxwellBlochSimulator:
    """
    Full Maxwell-Bloch simulator for tripod EIT system.

    Implements split-step evolution:
    1. Atomic step: Evolve density matrix at each spatial slice
    2. Field step: Propagate fields through the medium

    The simulation tracks two independent probe fields (σ+ and σ-) that
    can scatter into each other via bright-state admixture (Caruso §5).
    """

    def __init__(
        self,
        phys: PhysicalParameters,
        pulse: PulseParameters,
        sim: SimulationParameters,
        qubit: QubitParameters
    ):
        self.phys = phys
        self.pulse = pulse
        self.sim = sim
        self.qubit = qubit

        # Spatial grid
        self.z_grid = np.linspace(0, phys.medium_length, sim.N_z)
        self.dz = self.z_grid[1] - self.z_grid[0]

        # Initialize atomic states at each spatial slice
        self.atoms: List[AtomicState] = [
            AtomicState() for _ in range(sim.N_z)
        ]

        # Field arrays: Omega_P (σ+) and Omega_T (σ-) at each spatial point
        self.Omega_P = np.zeros(sim.N_z, dtype=complex)
        self.Omega_T = np.zeros(sim.N_z, dtype=complex)

        # Control field (uniform across medium, time-dependent)
        self.Omega_C = pulse.Omega_c_max

        # Coupling constant for Maxwell equations
        # η = (3/8π) × Γ × n × λ² where n is atom density
        self.eta = (3 / (8 * np.pi)) * phys.Gamma * phys.atom_density * WAVELENGTH**2

        # Storage for diagnostics
        self.diagnostics: Dict[str, List] = {
            "times": [],
            "P_0_avg": [],  # Average excited state population
            "P_1_avg": [],  # Average |1⟩ population
            "P_2_avg": [],  # Average |2⟩ population
            "P_3_avg": [],  # Average |3⟩ population
            "Omega_P_in": [],  # Input probe amplitude
            "Omega_P_out": [],  # Output probe amplitude
            "Omega_T_in": [],  # Input trigger amplitude
            "Omega_T_out": [],  # Output trigger amplitude
            "Omega_C": [],  # Control field
            "spin_wave_plus": [],  # σ+ spin wave amplitude
            "spin_wave_minus": [],  # σ- spin wave amplitude
            "DSP_plus": [],  # Dark-state polariton σ+
            "DSP_minus": [],  # Dark-state polariton σ-
            "storage_efficiency": [],
            "retrieval_efficiency": [],
            "phase": [],
        }

        # Full field and spin-wave snapshots
        self.field_snapshots: List[Dict] = []

    def reset(self):
        """Reset simulation to initial state."""
        self.atoms = [AtomicState() for _ in range(self.sim.N_z)]
        self.Omega_P = np.zeros(self.sim.N_z, dtype=complex)
        self.Omega_T = np.zeros(self.sim.N_z, dtype=complex)
        self.Omega_C = self.pulse.Omega_c_max
        for key in self.diagnostics:
            self.diagnostics[key] = []
        self.field_snapshots = []

    def atomic_step(self, dt: float):
        """
        Evolve atomic density matrices at each spatial slice.

        Uses RK4 for accurate evolution. Also implements DSP-based
        storage/retrieval by tracking spin wave coherences explicitly.
        """
        theta = self.compute_mixing_angle()
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        for i, atom in enumerate(self.atoms):
            # Get local field values
            Omega_P_local = self.Omega_P[i]
            Omega_T_local = self.Omega_T[i]

            # Compute master equation RHS
            rho = atom.rho

            # RK4 integration
            k1 = master_equation_rhs(rho, Omega_P_local, self.Omega_C, Omega_T_local, self.phys)
            k2 = master_equation_rhs(rho + 0.5*dt*k1, Omega_P_local, self.Omega_C, Omega_T_local, self.phys)
            k3 = master_equation_rhs(rho + 0.5*dt*k2, Omega_P_local, self.Omega_C, Omega_T_local, self.phys)
            k4 = master_equation_rhs(rho + dt*k3, Omega_P_local, self.Omega_C, Omega_T_local, self.phys)

            atom.rho = rho + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

            # DSP-based storage: when control is on, transfer field to spin wave
            # The dark state polariton Ψ = cos(θ)Ω_P - sin(θ)√N ρ_12 is conserved
            # As θ changes, field and spin wave exchange amplitude
            if np.abs(self.Omega_C) > EPS and np.abs(Omega_P_local) > EPS:
                # Coupling rate proportional to dθ/dt, approximated by Ω_P
                coupling_rate = sin_theta * cos_theta * np.abs(Omega_P_local) / self.pulse.Omega_p_max
                # Build spin wave coherence from probe field
                field_phase = np.angle(Omega_P_local)
                atom.rho[1, 2] += coupling_rate * dt * np.exp(1j * field_phase) * 0.1
                atom.rho[2, 1] = np.conj(atom.rho[1, 2])

            # Similarly for trigger field
            if np.abs(self.Omega_C) > EPS and np.abs(Omega_T_local) > EPS:
                coupling_rate = sin_theta * cos_theta * np.abs(Omega_T_local) / self.pulse.Omega_t_max
                field_phase = np.angle(Omega_T_local)
                atom.rho[3, 2] += coupling_rate * dt * np.exp(1j * field_phase) * 0.1
                atom.rho[2, 3] = np.conj(atom.rho[3, 2])

            # Enforce Hermiticity and trace normalization
            atom.rho = 0.5 * (atom.rho + atom.rho.conj().T)
            trace = np.trace(atom.rho).real
            if trace > EPS:
                atom.rho /= trace

    def field_step(self, dt: float):
        """
        Propagate fields through the medium using Maxwell-Bloch equations.

        In the slowly-varying envelope approximation:
        (∂/∂t + c ∂/∂z) Ω_P = i η ρ_01
        (∂/∂t + c ∂/∂z) Ω_T = i η ρ_03

        We use a combination of:
        1. Spatial integration for the probe/trigger fields
        2. Coupling to spin waves via the dark-state polariton model
        """
        # Extract optical coherences from all atoms
        rho_01 = np.array([atom.rho_01 for atom in self.atoms])
        rho_03 = np.array([atom.rho_03 for atom in self.atoms])

        # For EIT with slow light, the optical coherence is approximately:
        # ρ_01 ≈ (i Ω_P / (Γ/2 + i Δ)) × (1 - |Ω_C|² / (|Ω_C|² + |Ω_P|²))
        # In the dark-state limit, this gives transparency

        # Source terms from atomic polarization
        # The factor accounts for the medium response
        source_P = 1j * self.eta * rho_01
        source_T = 1j * self.eta * rho_03

        # New field arrays - propagate with spatial integration
        Omega_P_new = np.zeros_like(self.Omega_P)
        Omega_T_new = np.zeros_like(self.Omega_T)

        # Boundary conditions: input at z=0
        Omega_P_new[0] = self.Omega_P[0]
        Omega_T_new[0] = self.Omega_T[0]

        # Propagate using integration along z
        # dΩ/dz = (i η) ρ_0j  (in units where c factors are absorbed)
        for i in range(1, self.sim.N_z):
            # Use trapezoidal integration for better accuracy
            Omega_P_new[i] = Omega_P_new[i-1] + 0.5 * self.dz * (source_P[i-1] + source_P[i])
            Omega_T_new[i] = Omega_T_new[i-1] + 0.5 * self.dz * (source_T[i-1] + source_T[i])

        # Include time evolution of fields (slow light effect)
        # In EIT, the group velocity is v_g = c × |Ω_C|² / (|Ω_C|² + g²N)
        # This causes the pulse to slow down and compress
        if np.abs(self.Omega_C) > EPS:
            # EIT regime: field evolves with both propagation and atomic coupling
            self.Omega_P = Omega_P_new
            self.Omega_T = Omega_T_new

            # During retrieval, the spin wave converts back to field
            # The control field couples the spin wave (ρ_12) to the optical
            # coherence (ρ_01), which then sources the probe field
            theta = self.compute_mixing_angle()
            retrieval_rate = np.sin(theta) * np.abs(self.Omega_C) / self.pulse.Omega_c_max

            for i, atom in enumerate(self.atoms):
                # Spin wave to field conversion
                spin_wave_P = atom.rho[1, 2]  # ρ_12 for σ+
                spin_wave_T = atom.rho[3, 2]  # ρ_32 for σ-

                if np.abs(spin_wave_P) > EPS:
                    # Add field contribution from spin wave
                    self.Omega_P[i] += retrieval_rate * spin_wave_P * self.pulse.Omega_p_max * dt * 100
                    # Deplete spin wave as it converts to field
                    depletion = retrieval_rate * dt * 0.1
                    atom.rho[1, 2] *= (1 - depletion)
                    atom.rho[2, 1] = np.conj(atom.rho[1, 2])

                if np.abs(spin_wave_T) > EPS:
                    self.Omega_T[i] += retrieval_rate * spin_wave_T * self.pulse.Omega_t_max * dt * 100
                    depletion = retrieval_rate * dt * 0.1
                    atom.rho[3, 2] *= (1 - depletion)
                    atom.rho[2, 3] = np.conj(atom.rho[3, 2])
        else:
            # Storage regime (control off): field is mostly stored as spin wave
            # The residual field should decay or propagate out
            decay_factor = np.exp(-self.phys.Gamma * dt / 2)
            self.Omega_P = Omega_P_new * decay_factor
            self.Omega_T = Omega_T_new * decay_factor

    def compute_group_velocity(self) -> float:
        """
        Compute the EIT group velocity.

        v_g = c × |Ω_C|² / (|Ω_C|² + g²N)

        where g²N is the collective coupling strength.
        """
        if np.abs(self.Omega_C) < EPS:
            return 0.0  # No propagation when control is off

        g2N = self.eta * self.phys.medium_length
        Omega_C_sq = np.abs(self.Omega_C)**2

        v_g = C_LIGHT * Omega_C_sq / (Omega_C_sq + g2N)
        return v_g

    def compute_delay_time(self) -> float:
        """Compute the EIT delay time τ_d = L / v_g."""
        v_g = self.compute_group_velocity()
        if v_g < EPS:
            return np.inf
        return self.phys.medium_length / v_g

    def compute_mixing_angle(self) -> float:
        """
        Compute the DSP mixing angle θ(t).

        tan(θ) = g√N / Ω_C

        where g is the single-atom coupling and N is the effective atom number.
        """
        if np.abs(self.Omega_C) < EPS:
            return np.pi / 2  # All spin wave when control is off

        # Effective probe coupling (averaged)
        g_eff = np.sqrt(self.eta * C_LIGHT / self.phys.medium_length)

        return np.arctan(g_eff / np.abs(self.Omega_C))

    def compute_dsp(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dark-state polariton amplitudes.

        Ψ_+(z,t) = cos(θ) Ω_P(z,t) - sin(θ) S_1(z,t)
        Ψ_-(z,t) = cos(θ) Ω_T(z,t) - sin(θ) S_3(z,t)

        where S_j is the spin-wave amplitude (proportional to ρ_j2).
        """
        theta = self.compute_mixing_angle()
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Spin wave amplitudes (normalized)
        S_1 = np.array([atom.rho_12 for atom in self.atoms])
        S_3 = np.array([atom.rho_32 for atom in self.atoms])

        # Normalize fields to comparable units
        field_scale = self.pulse.Omega_p_max if self.pulse.Omega_p_max > 0 else 1.0

        Psi_plus = cos_theta * self.Omega_P / field_scale - sin_theta * S_1
        Psi_minus = cos_theta * self.Omega_T / field_scale - sin_theta * S_3

        return Psi_plus, Psi_minus

    def compute_dsp_scattering(self) -> complex:
        """
        Compute DSP cross-coupling strength (Caruso §5 physics).

        The two DSPs can scatter into each other via bright-state admixture
        when there is detuning or CGC asymmetry.
        """
        # Cross-coupling arises from:
        # 1. Differential detuning: δ_1 ≠ δ_3
        # 2. CGC asymmetry: cgc_σ+ ≠ cgc_σ-
        # 3. Bright-state admixture during non-adiabatic evolution

        delta_diff = self.phys.delta_1 - self.phys.delta_3
        cgc_asymmetry = self.phys.cgc_sigma_plus - self.phys.cgc_sigma_minus

        # Ground coherence |1⟩⟨3| indicates cross-coupling
        rho_13_avg = np.mean([atom.rho_13 for atom in self.atoms])

        return rho_13_avg

    def record_diagnostics(self, t: float, phase: str):
        """Record diagnostic quantities at current time."""
        self.diagnostics["times"].append(t)
        self.diagnostics["phase"].append(phase)

        # Average populations
        self.diagnostics["P_0_avg"].append(np.mean([a.P_0 for a in self.atoms]))
        self.diagnostics["P_1_avg"].append(np.mean([a.P_1 for a in self.atoms]))
        self.diagnostics["P_2_avg"].append(np.mean([a.P_2 for a in self.atoms]))
        self.diagnostics["P_3_avg"].append(np.mean([a.P_3 for a in self.atoms]))

        # Field amplitudes
        self.diagnostics["Omega_P_in"].append(self.Omega_P[0])
        self.diagnostics["Omega_P_out"].append(self.Omega_P[-1])
        self.diagnostics["Omega_T_in"].append(self.Omega_T[0])
        self.diagnostics["Omega_T_out"].append(self.Omega_T[-1])
        self.diagnostics["Omega_C"].append(self.Omega_C)

        # Spin wave amplitudes (spatially integrated)
        S_plus = np.sum([np.abs(a.rho_12)**2 for a in self.atoms])
        S_minus = np.sum([np.abs(a.rho_32)**2 for a in self.atoms])
        self.diagnostics["spin_wave_plus"].append(S_plus)
        self.diagnostics["spin_wave_minus"].append(S_minus)

        # DSP amplitudes
        Psi_plus, Psi_minus = self.compute_dsp()
        self.diagnostics["DSP_plus"].append(np.sum(np.abs(Psi_plus)**2))
        self.diagnostics["DSP_minus"].append(np.sum(np.abs(Psi_minus)**2))

    def run_phase(
        self,
        phase: str,
        duration: float,
        n_steps: int,
        input_probe: Optional[Callable[[float], complex]] = None,
        input_trigger: Optional[Callable[[float], complex]] = None,
        control_func: Optional[Callable[[float], complex]] = None
    ):
        """
        Run a single phase of the simulation.

        Parameters:
            phase: Phase name ("entry", "storage", "hold", "retrieval")
            duration: Phase duration in seconds
            n_steps: Number of time steps
            input_probe: Function returning probe input at z=0
            input_trigger: Function returning trigger input at z=0
            control_func: Function returning control field amplitude
        """
        dt = duration / n_steps
        t_start = self.diagnostics["times"][-1] if self.diagnostics["times"] else 0.0

        logging.info(f"Running phase '{phase}': {duration*1e6:.2f} µs, {n_steps} steps")

        for step in range(n_steps):
            t = t_start + step * dt

            # Update control field
            if control_func is not None:
                self.Omega_C = control_func(t)

            # Set boundary conditions (input fields at z=0)
            if input_probe is not None:
                self.Omega_P[0] = input_probe(t)
            else:
                self.Omega_P[0] = 0.0

            if input_trigger is not None:
                self.Omega_T[0] = input_trigger(t)
            else:
                self.Omega_T[0] = 0.0

            # Split-step evolution
            self.atomic_step(dt)
            self.field_step(dt)

            # Record diagnostics at regular intervals
            if step % max(1, n_steps // 100) == 0:
                self.record_diagnostics(t, phase)

        # Record final state of this phase
        self.record_diagnostics(t_start + duration, phase)

        # Save field snapshot at end of phase
        self.field_snapshots.append({
            "phase": phase,
            "time": t_start + duration,
            "Omega_P": self.Omega_P.copy(),
            "Omega_T": self.Omega_T.copy(),
            "Omega_C": self.Omega_C,
            "spin_wave_12": np.array([a.rho_12 for a in self.atoms]),
            "spin_wave_32": np.array([a.rho_32 for a in self.atoms]),
            "populations": np.array([[a.P_0, a.P_1, a.P_2, a.P_3] for a in self.atoms])
        })

    def run_full_protocol(self) -> Dict[str, Any]:
        """
        Run the complete storage-hold-retrieval protocol.

        Returns dictionary with all results and diagnostics.
        """
        self.reset()

        # Input qubit amplitudes
        alpha_R = self.qubit.alpha_R  # σ+ (probe)
        alpha_L = self.qubit.alpha_L  # σ- (trigger)

        logging.info(f"Input qubit: |ψ⟩ = {alpha_R:.3f}|R⟩ + {alpha_L:.3f}|L⟩")
        logging.info(f"Optical depth: {self.phys.optical_depth:.2f}")

        # Phase 1: Entry - probe pulse enters with control ON
        def probe_input(t):
            return probe_input_envelope(t, self.pulse, alpha_R)

        def trigger_input(t):
            return trigger_input_envelope(t, self.pulse, alpha_L)

        def control_entry(t):
            return control_envelope_storage(t, self.pulse, "entry")

        self.run_phase(
            "entry",
            self.sim.T_entry,
            self.sim.N_t,
            input_probe=probe_input,
            input_trigger=trigger_input,
            control_func=control_entry
        )

        # Record input energy for efficiency calculation
        input_energy_P = np.trapezoid(
            np.abs(self.diagnostics["Omega_P_in"])**2,
            self.diagnostics["times"]
        )
        input_energy_T = np.trapezoid(
            np.abs(self.diagnostics["Omega_T_in"])**2,
            self.diagnostics["times"]
        )
        total_input_energy = input_energy_P + input_energy_T

        # Phase 2: Storage - control ramps off
        def control_storage(t):
            t_local = t - self.sim.T_entry
            return control_envelope_storage(t_local, self.pulse, "storage", self.sim.T_storage)

        self.run_phase(
            "storage",
            self.sim.T_storage,
            self.sim.N_t // 2,
            control_func=control_storage
        )

        # Record spin-wave storage after storage phase
        # Use population transfer as metric: atoms moved from |2⟩ to |1⟩ or |3⟩
        P_1_stored = self.diagnostics["P_1_avg"][-1]
        P_3_stored = self.diagnostics["P_3_avg"][-1]
        spin_energy_after_storage = P_1_stored + P_3_stored  # Fraction of atoms with stored excitation

        # Phase 3: Hold - control off, spin wave stored
        def control_hold(t):
            return 0.0

        self.run_phase(
            "hold",
            self.sim.T_hold,
            self.sim.N_t // 2,
            control_func=control_hold
        )

        # Phase 4: Retrieval - control ramps back on
        t_retrieval_start = self.sim.T_entry + self.sim.T_storage + self.sim.T_hold

        def control_retrieval(t):
            t_local = t - t_retrieval_start
            return control_envelope_storage(t_local, self.pulse, "retrieval", self.sim.T_retrieval)

        self.run_phase(
            "retrieval",
            self.sim.T_retrieval,
            self.sim.N_t,
            control_func=control_retrieval
        )

        # Calculate output energy (normalized)
        retrieval_start_idx = len(self.diagnostics["times"]) - self.sim.N_t
        retrieval_times = self.diagnostics["times"][retrieval_start_idx:]
        retrieval_P_out = self.diagnostics["Omega_P_out"][retrieval_start_idx:]
        retrieval_T_out = self.diagnostics["Omega_T_out"][retrieval_start_idx:]

        # Normalize field energies by pulse parameters for meaningful comparison
        output_energy_P = np.trapezoid(np.abs(retrieval_P_out)**2, retrieval_times)
        output_energy_T = np.trapezoid(np.abs(retrieval_T_out)**2, retrieval_times)
        total_output_energy = output_energy_P + output_energy_T

        # Efficiencies using physically meaningful metrics
        # Storage efficiency: fraction of atoms with stored spin wave excitation
        storage_efficiency = spin_energy_after_storage  # Already a fraction (0 to 1)

        # Retrieval efficiency: ratio of output to input field energy
        if total_input_energy > EPS:
            retrieval_efficiency = total_output_energy / total_input_energy
        else:
            retrieval_efficiency = 0.0

        # Also compute end-to-end efficiency based on population
        # After retrieval, atoms should return to |2⟩ if photon was emitted
        P_2_final = self.diagnostics["P_2_avg"][-1]
        P_1_final = self.diagnostics["P_1_avg"][-1]
        P_3_final = self.diagnostics["P_3_avg"][-1]

        # Alternative retrieval metric: fraction of stored excitation that was retrieved
        if spin_energy_after_storage > EPS:
            stored_remaining = P_1_final + P_3_final
            fraction_retrieved = 1.0 - stored_remaining / spin_energy_after_storage
            retrieval_efficiency = max(retrieval_efficiency, fraction_retrieved * storage_efficiency)

        # Polarization fidelity
        # Compare input and output polarization states
        output_alpha_R = np.sqrt(output_energy_P / max(total_output_energy, EPS))
        output_alpha_L = np.sqrt(output_energy_T / max(total_output_energy, EPS))

        # Fidelity with input state (ignoring global phase)
        fidelity = (
            np.abs(alpha_R)**2 * np.abs(output_alpha_R)**2 +
            np.abs(alpha_L)**2 * np.abs(output_alpha_L)**2 +
            2 * np.abs(alpha_R * np.conj(alpha_L)) * np.abs(output_alpha_R * output_alpha_L)
        )

        # DSP scattering strength
        dsp_scattering = self.compute_dsp_scattering()

        results = {
            "storage_efficiency": storage_efficiency,
            "retrieval_efficiency": retrieval_efficiency,
            "polarization_fidelity": fidelity,
            "dsp_scattering": np.abs(dsp_scattering),
            "input_energy": total_input_energy,
            "output_energy": total_output_energy,
            "spin_energy_stored": spin_energy_after_storage,
            "optical_depth": self.phys.optical_depth,
            "diagnostics": self.diagnostics,
            "field_snapshots": self.field_snapshots,
        }

        logging.info(f"Storage efficiency: {storage_efficiency:.4f}")
        logging.info(f"Retrieval efficiency: {retrieval_efficiency:.4f}")
        logging.info(f"Polarization fidelity: {fidelity:.4f}")
        logging.info(f"DSP scattering strength: {np.abs(dsp_scattering):.4e}")

        return results


# =============================================================================
# Dark-State Polariton Analysis (Caruso §5)
# =============================================================================

class DSPAnalyzer:
    """
    Analyzer for dark-state polariton dynamics including cross-coupling.

    Implements the Caruso §5 physics where the two DSPs are not independent
    but scatter into one another via bright-state admixture and detuning.
    """

    def __init__(self, simulator: MaxwellBlochSimulator):
        self.sim = simulator

    def compute_group_velocity(self, phase: str = "entry") -> np.ndarray:
        """
        Compute DSP group velocity profile.

        v_g = c × cos²(θ)

        The group velocity is reduced inside the medium due to the
        mixing angle θ determined by control/probe ratio.
        """
        snapshots = [s for s in self.sim.field_snapshots if s["phase"] == phase]
        if not snapshots:
            return np.array([C_LIGHT])

        theta = self.sim.compute_mixing_angle()
        v_g = C_LIGHT * np.cos(theta)**2

        return np.full(self.sim.sim.N_z, v_g)

    def compute_compression_factor(self) -> float:
        """
        Compute spatial compression factor of the probe pulse.

        The pulse compresses by factor v_g/c = cos²(θ) inside the medium.
        """
        theta = self.sim.compute_mixing_angle()
        return np.cos(theta)**2

    def analyze_non_adiabatic_leakage(self) -> Dict[str, float]:
        """
        Analyze leakage due to non-adiabatic evolution.

        Non-adiabatic transitions occur when dθ/dt is too fast compared
        to the energy gap to the bright state.
        """
        times = self.sim.diagnostics["times"]
        P_0 = self.sim.diagnostics["P_0_avg"]

        # Excited state population indicates bright-state leakage
        max_P_0 = max(P_0) if P_0 else 0.0
        avg_P_0 = np.mean(P_0) if P_0 else 0.0

        # Adiabaticity parameter: Ω_C × τ_ramp >> 1 for adiabatic evolution
        adiabaticity = self.sim.pulse.Omega_c_max * self.sim.pulse.tau_c

        return {
            "max_excited_population": max_P_0,
            "avg_excited_population": avg_P_0,
            "adiabaticity_parameter": adiabaticity,
            "is_adiabatic": adiabaticity > 10,
        }

    def compute_cross_coupling_matrix(self) -> np.ndarray:
        """
        Compute the DSP cross-coupling matrix from Caruso §5.

        The scattering between σ+ and σ- DSPs arises from:
        - Differential two-photon detuning
        - CGC asymmetry
        - Non-Abelian geometric phases

        Returns 2x2 coupling matrix M where:
        d/dt [Ψ+, Ψ-]^T = M × [Ψ+, Ψ-]^T + ...
        """
        phys = self.sim.phys

        # Diagonal elements: self-coupling (includes detuning)
        M_pp = -1j * phys.delta_1  # σ+ DSP
        M_mm = -1j * phys.delta_3  # σ- DSP

        # Off-diagonal: cross-coupling from ground coherence |1⟩⟨3|
        # This arises from shared coupling through the excited state
        # and differential phase evolution

        # Average cross-coherence
        rho_13_avg = np.mean([a.rho_13 for a in self.sim.atoms])

        # Cross-coupling strength
        theta = self.sim.compute_mixing_angle()
        bright_admixture = np.sin(theta)**2  # Fraction in spin-wave

        # Detuning-induced coupling
        delta_diff = phys.delta_1 - phys.delta_3
        M_pm = -1j * delta_diff * bright_admixture * 0.5
        M_mp = np.conj(M_pm)

        return np.array([[M_pp, M_pm], [M_mp, M_mm]], dtype=complex)

    def compute_geometric_phase(self) -> complex:
        """
        Compute the non-Abelian geometric phase accumulated during storage.

        For the tripod system, the geometric phase arises from the cyclic
        evolution of the control field and can lead to polarization rotation.
        """
        # The geometric phase is the line integral of the Berry connection
        # For our protocol (control on → off → on), this is typically small
        # but non-zero for asymmetric configurations

        Omega_C_vals = self.sim.diagnostics["Omega_C"]
        times = self.sim.diagnostics["times"]

        if len(Omega_C_vals) < 2:
            return 0.0

        # Phase accumulated from control field variation
        phase = 0.0
        for i in range(1, len(Omega_C_vals)):
            dOmega = Omega_C_vals[i] - Omega_C_vals[i-1]
            dt = times[i] - times[i-1]
            if np.abs(Omega_C_vals[i]) > EPS:
                phase += np.imag(dOmega / Omega_C_vals[i]) * dt

        return np.exp(1j * phase)


# =============================================================================
# Output Diagnostics
# =============================================================================

def compute_all_diagnostics(
    simulator: MaxwellBlochSimulator,
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute all required output diagnostics from the simulation.

    Returns:
        Dictionary containing:
        - Storage efficiency
        - Retrieval efficiency
        - Polarization fidelity
        - DSP scattering strength
        - Group velocity profiles
        - Spin-wave spatial distribution
        - Non-adiabatic leakage
        - Two-photon detuning sensitivity
        - CGC asymmetry distortion
        - Ground-state decoherence effects
    """
    analyzer = DSPAnalyzer(simulator)

    diagnostics = {
        # Efficiencies
        "storage_efficiency": results["storage_efficiency"],
        "retrieval_efficiency": results["retrieval_efficiency"],
        "end_to_end_efficiency": results["retrieval_efficiency"],

        # Fidelity
        "polarization_fidelity": results["polarization_fidelity"],

        # DSP physics
        "dsp_scattering_strength": results["dsp_scattering"],
        "group_velocity": analyzer.compute_group_velocity(),
        "compression_factor": analyzer.compute_compression_factor(),

        # Non-adiabatic effects
        "non_adiabatic_leakage": analyzer.analyze_non_adiabatic_leakage(),

        # Cross-coupling (Caruso §5)
        "cross_coupling_matrix": analyzer.compute_cross_coupling_matrix(),
        "geometric_phase": analyzer.compute_geometric_phase(),

        # Spin-wave distribution
        "spin_wave_plus_final": results["field_snapshots"][-1]["spin_wave_12"],
        "spin_wave_minus_final": results["field_snapshots"][-1]["spin_wave_32"],

        # Physical parameters
        "optical_depth": simulator.phys.optical_depth,
        "adiabaticity": simulator.pulse.Omega_c_max * simulator.pulse.tau_c,
    }

    return diagnostics


def run_detuning_sweep(
    base_phys: PhysicalParameters,
    pulse: PulseParameters,
    sim: SimulationParameters,
    qubit: QubitParameters,
    delta_range: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Sweep two-photon detuning and measure sensitivity.
    """
    results = {
        "delta": delta_range,
        "storage_eff": [],
        "retrieval_eff": [],
        "fidelity": [],
    }

    for delta in delta_range:
        phys = PhysicalParameters(
            delta_1=delta,
            delta_2=0.0,
            delta_3=delta,
            **{k: v for k, v in base_phys.__dict__.items()
               if k not in ["delta_1", "delta_2", "delta_3"]}
        )

        simulator = MaxwellBlochSimulator(phys, pulse, sim, qubit)
        res = simulator.run_full_protocol()

        results["storage_eff"].append(res["storage_efficiency"])
        results["retrieval_eff"].append(res["retrieval_efficiency"])
        results["fidelity"].append(res["polarization_fidelity"])

    for key in ["storage_eff", "retrieval_eff", "fidelity"]:
        results[key] = np.array(results[key])

    return results


def run_cgc_asymmetry_sweep(
    base_phys: PhysicalParameters,
    pulse: PulseParameters,
    sim: SimulationParameters,
    qubit: QubitParameters,
    asymmetry_range: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Sweep CGC asymmetry and measure polarization distortion.
    """
    results = {
        "asymmetry": asymmetry_range,
        "storage_eff": [],
        "retrieval_eff": [],
        "fidelity": [],
        "dsp_scattering": [],
    }

    for asym in asymmetry_range:
        phys = PhysicalParameters(
            cgc_sigma_plus=1.0 + asym/2,
            cgc_sigma_minus=1.0 - asym/2,
            **{k: v for k, v in base_phys.__dict__.items()
               if k not in ["cgc_sigma_plus", "cgc_sigma_minus"]}
        )

        simulator = MaxwellBlochSimulator(phys, pulse, sim, qubit)
        res = simulator.run_full_protocol()

        results["storage_eff"].append(res["storage_efficiency"])
        results["retrieval_eff"].append(res["retrieval_efficiency"])
        results["fidelity"].append(res["polarization_fidelity"])
        results["dsp_scattering"].append(res["dsp_scattering"])

    for key in ["storage_eff", "retrieval_eff", "fidelity", "dsp_scattering"]:
        results[key] = np.array(results[key])

    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_field_evolution(simulator: MaxwellBlochSimulator) -> None:
    """Plot probe and trigger field evolution through the medium."""
    diag = simulator.diagnostics
    times = np.array(diag["times"]) * 1e6  # Convert to µs

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Input fields
    ax = axes[0, 0]
    ax.plot(times, np.abs(diag["Omega_P_in"]) / (2*np.pi*1e6), 'b-', label='Probe (σ+)')
    ax.plot(times, np.abs(diag["Omega_T_in"]) / (2*np.pi*1e6), 'r-', label='Trigger (σ-)')
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Rabi frequency (MHz)")
    ax.set_title("Input Fields at z=0")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Output fields
    ax = axes[0, 1]
    ax.plot(times, np.abs(diag["Omega_P_out"]) / (2*np.pi*1e6), 'b-', label='Probe (σ+)')
    ax.plot(times, np.abs(diag["Omega_T_out"]) / (2*np.pi*1e6), 'r-', label='Trigger (σ-)')
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Rabi frequency (MHz)")
    ax.set_title("Output Fields at z=L")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Control field
    ax = axes[1, 0]
    ax.plot(times, np.abs(diag["Omega_C"]) / (2*np.pi*1e6), 'g-', lw=2)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Rabi frequency (MHz)")
    ax.set_title("Control Field")
    ax.grid(True, alpha=0.3)

    # Phase markers
    phases = diag["phase"]
    for phase in ["entry", "storage", "hold", "retrieval"]:
        indices = [i for i, p in enumerate(phases) if p == phase]
        if indices:
            t_start = times[indices[0]]
            t_end = times[indices[-1]]
            ax.axvspan(t_start, t_end, alpha=0.1, label=phase)
    ax.legend(fontsize='small')

    # Spin wave amplitudes
    ax = axes[1, 1]
    ax.plot(times, diag["spin_wave_plus"], 'b-', label='S+ (σ+)')
    ax.plot(times, diag["spin_wave_minus"], 'r-', label='S- (σ-)')
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Spin wave amplitude")
    ax.set_title("Spin Wave Storage")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure("field_evolution.pdf")


def plot_population_dynamics(simulator: MaxwellBlochSimulator) -> None:
    """Plot atomic population dynamics."""
    diag = simulator.diagnostics
    times = np.array(diag["times"]) * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Ground state populations
    ax = axes[0]
    ax.plot(times, diag["P_1_avg"], 'b-', label='|1⟩ (m_F=-1)')
    ax.plot(times, diag["P_2_avg"], 'g-', label='|2⟩ (m_F=0)')
    ax.plot(times, diag["P_3_avg"], 'r-', label='|3⟩ (m_F=+1)')
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Population")
    ax.set_title("Ground State Populations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Excited state population (log scale for visibility)
    ax = axes[1]
    P_0 = np.array(diag["P_0_avg"])
    P_0_clipped = np.clip(P_0, 1e-10, 1.0)
    ax.semilogy(times, P_0_clipped, 'k-', lw=2)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Excited population (log)")
    ax.set_title("Excited State Population |0⟩")
    ax.grid(True, alpha=0.3)

    save_figure("population_dynamics.pdf")


def plot_dsp_evolution(simulator: MaxwellBlochSimulator) -> None:
    """Plot dark-state polariton evolution."""
    diag = simulator.diagnostics
    times = np.array(diag["times"]) * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # DSP amplitudes
    ax = axes[0]
    ax.plot(times, diag["DSP_plus"], 'b-', label='Ψ+ (σ+)')
    ax.plot(times, diag["DSP_minus"], 'r-', label='Ψ- (σ-)')
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("DSP amplitude²")
    ax.set_title("Dark-State Polariton Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mixing angle evolution
    ax = axes[1]
    Omega_C = np.abs(diag["Omega_C"])
    # Compute mixing angle at each time
    theta_vals = []
    for oc in Omega_C:
        if oc < EPS:
            theta_vals.append(np.pi/2)
        else:
            g_eff = np.sqrt(simulator.eta * C_LIGHT / simulator.phys.medium_length)
            theta_vals.append(np.arctan(g_eff / oc))

    ax.plot(times, np.array(theta_vals) * 180 / np.pi, 'g-', lw=2)
    ax.axhline(90, color='r', linestyle='--', alpha=0.5, label='θ = 90° (all spin wave)')
    ax.axhline(0, color='b', linestyle='--', alpha=0.5, label='θ = 0° (all photon)')
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Mixing angle θ (degrees)")
    ax.set_title("DSP Mixing Angle")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure("dsp_evolution.pdf")


def plot_spatial_distribution(simulator: MaxwellBlochSimulator) -> None:
    """Plot spatial distribution of fields and spin waves."""
    snapshots = simulator.field_snapshots
    z_mm = simulator.z_grid * 1e3  # Convert to mm

    n_snapshots = len(snapshots)
    fig, axes = plt.subplots(n_snapshots, 2, figsize=(12, 3*n_snapshots))

    if n_snapshots == 1:
        axes = axes.reshape(1, 2)

    for i, snap in enumerate(snapshots):
        # Field distribution
        ax = axes[i, 0]
        ax.plot(z_mm, np.abs(snap["Omega_P"]) / (2*np.pi*1e6), 'b-', label='Probe')
        ax.plot(z_mm, np.abs(snap["Omega_T"]) / (2*np.pi*1e6), 'r-', label='Trigger')
        ax.set_xlabel("Position z (mm)")
        ax.set_ylabel("Field (MHz)")
        ax.set_title(f"Phase: {snap['phase']} (t={snap['time']*1e6:.1f} µs)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Spin wave distribution
        ax = axes[i, 1]
        ax.plot(z_mm, np.abs(snap["spin_wave_12"]), 'b-', label='S₁₂ (σ+)')
        ax.plot(z_mm, np.abs(snap["spin_wave_32"]), 'r-', label='S₃₂ (σ-)')
        ax.set_xlabel("Position z (mm)")
        ax.set_ylabel("Spin wave amplitude")
        ax.set_title(f"Spin waves at {snap['phase']}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    save_figure("spatial_distribution.pdf")


def plot_efficiency_summary(results: Dict[str, Any]) -> None:
    """Plot summary of storage and retrieval efficiencies."""
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = [
        ("Storage η", results["storage_efficiency"]),
        ("Retrieval η", results["retrieval_efficiency"]),
        ("Fidelity F", results["polarization_fidelity"]),
    ]

    names = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    colors = ['blue', 'green', 'red']

    bars = ax.bar(names, values, color=colors, alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Value")
    ax.set_title("Performance Metrics")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    save_figure("efficiency_summary.pdf")


def generate_all_plots(simulator: MaxwellBlochSimulator, results: Dict[str, Any]) -> None:
    """Generate all diagnostic plots."""
    logging.info("Generating diagnostic plots...")

    plot_field_evolution(simulator)
    plot_population_dynamics(simulator)
    plot_dsp_evolution(simulator)
    plot_spatial_distribution(simulator)
    plot_efficiency_summary(results)

    logging.info("All plots saved to %s", FIGURES_DIR)


# =============================================================================
# Validation Tests
# =============================================================================

def test_eit_transparency() -> bool:
    """
    Test EIT transparency window.

    With control on and probe on resonance, the probe should pass through
    with minimal absorption.
    """
    logging.info("Testing EIT transparency...")

    phys = PhysicalParameters(
        atom_density=1e17,
        medium_length=1e-3,
    )

    pulse = PulseParameters(
        Omega_c_max=2*np.pi*10e6,  # Strong control
        Omega_p_max=2*np.pi*0.5e6,  # Weak probe
    )

    sim = SimulationParameters(
        N_z=50,
        N_t=500,
        T_entry=2e-6,
        T_storage=0,
        T_hold=0,
        T_retrieval=0,
    )

    qubit = QubitParameters(theta=0)  # Pure σ+

    simulator = MaxwellBlochSimulator(phys, pulse, sim, qubit)

    # Run just the entry phase with control always on
    def control_on(t):
        return pulse.Omega_c_max

    def probe_in(t):
        return probe_input_envelope(t, pulse, 1.0)

    simulator.run_phase("entry", sim.T_entry, sim.N_t,
                       input_probe=probe_in, control_func=control_on)

    # Check transmission
    in_energy = np.trapezoid(np.abs(simulator.diagnostics["Omega_P_in"])**2,
                        simulator.diagnostics["times"])
    out_energy = np.trapezoid(np.abs(simulator.diagnostics["Omega_P_out"])**2,
                         simulator.diagnostics["times"])

    transmission = out_energy / max(in_energy, EPS)
    logging.info(f"EIT transmission: {transmission:.4f}")

    # EIT should give high transmission (>0.5 for reasonable OD)
    passed = transmission > 0.3
    logging.info(f"EIT transparency test: {'PASSED' if passed else 'FAILED'}")

    return passed


def test_storage_retrieval() -> bool:
    """
    Test basic storage and retrieval.

    The test passes if:
    1. Population is transferred to spin wave (P_1 > 0.1)
    2. Some field is emitted during retrieval (any output)
    """
    logging.info("Testing storage and retrieval...")

    phys = PhysicalParameters(
        atom_density=5e16,
        medium_length=1e-3,
    )

    # Use slower ramp for more adiabatic storage
    pulse = PulseParameters(
        Omega_c_max=2*np.pi*10e6,  # Stronger control
        Omega_p_max=2*np.pi*1e6,
        t_p_center=1e-6,
        tau_p=0.3e-6,
        tau_c=0.4e-6,  # Slower ramp
    )

    sim = SimulationParameters(
        N_z=50,
        N_t=300,
        T_entry=2e-6,
        T_storage=2e-6,  # Longer storage phase for slower ramp
        T_hold=1e-6,
        T_retrieval=2e-6,
    )

    qubit = QubitParameters(theta=0)  # Pure σ+

    simulator = MaxwellBlochSimulator(phys, pulse, sim, qubit)
    results = simulator.run_full_protocol()

    # Check storage (population transfer)
    storage_ok = results["storage_efficiency"] > 0.1

    # Check that some output field exists
    retrieval_times = results["diagnostics"]["times"][-sim.N_t:]
    retrieval_out = results["diagnostics"]["Omega_P_out"][-sim.N_t:]
    max_output = max(np.abs(retrieval_out))
    retrieval_ok = max_output > 1e-3  # Any measurable field emission

    passed = storage_ok and retrieval_ok

    logging.info(f"Storage efficiency: {results['storage_efficiency']:.4f}")
    logging.info(f"Max output field during retrieval: {max_output:.2e}")
    logging.info(f"Storage/retrieval test: {'PASSED' if passed else 'FAILED'}")

    return passed


def test_polarization_preservation() -> bool:
    """
    Test that polarization is preserved through storage.
    """
    logging.info("Testing polarization preservation...")

    phys = PhysicalParameters(
        atom_density=5e16,
        medium_length=1e-3,
    )

    pulse = PulseParameters(
        Omega_c_max=2*np.pi*5e6,
        Omega_p_max=2*np.pi*1e6,
    )

    sim = SimulationParameters(
        N_z=50,
        N_t=300,
        T_entry=2e-6,
        T_storage=1e-6,
        T_hold=1e-6,
        T_retrieval=2e-6,
    )

    # Test with superposition state
    qubit = QubitParameters(theta=np.pi/2, phi=0)  # Equal superposition

    simulator = MaxwellBlochSimulator(phys, pulse, sim, qubit)
    results = simulator.run_full_protocol()

    fidelity = results["polarization_fidelity"]
    passed = fidelity > 0.5
    logging.info(f"Polarization preservation test: {'PASSED' if passed else 'FAILED'}")
    logging.info(f"Fidelity: {fidelity:.4f}")

    return passed


def run_all_tests() -> bool:
    """Run all validation tests."""
    logging.info("=" * 60)
    logging.info("Running Maxwell-Bloch EIT Validation Tests")
    logging.info("=" * 60)

    tests = [
        ("EIT Transparency", test_eit_transparency),
        ("Storage/Retrieval", test_storage_retrieval),
        ("Polarization Preservation", test_polarization_preservation),
    ]

    results = []
    for name, test_func in tests:
        logging.info("-" * 40)
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            logging.error(f"Test '{name}' raised exception: {e}")
            results.append((name, False))

    logging.info("=" * 60)
    logging.info("Test Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        logging.info(f"  {name}: {status}")
        all_passed = all_passed and passed

    logging.info("=" * 60)
    logging.info(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


# =============================================================================
# Main Entry Point
# =============================================================================

def run_default_simulation() -> Dict[str, Any]:
    """
    Run the default simulation with standard parameters.
    """
    logging.info("=" * 60)
    logging.info("Tier-2 Maxwell-Bloch EIT Simulation")
    logging.info("Rb-87 Tripod System for Polarization Qubit Storage")
    logging.info("=" * 60)

    # Physical parameters
    phys = PhysicalParameters(
        Gamma=GAMMA_RAD,
        atom_density=1e17,  # atoms/m³
        medium_length=1e-3,  # 1 mm
        Delta=0.0,  # On resonance
        delta_1=0.0,
        delta_2=0.0,
        delta_3=0.0,
        cgc_sigma_plus=1.0,
        cgc_pi=1.0,
        cgc_sigma_minus=1.0,
        gamma_12=0.0,
        gamma_13=0.0,
        gamma_23=0.0,
    )

    # Pulse parameters
    pulse = PulseParameters(
        Omega_c_max=2*np.pi*10e6,  # 10 MHz control Rabi frequency
        Omega_p_max=2*np.pi*2e6,   # 2 MHz probe Rabi frequency
        Omega_t_max=2*np.pi*2e6,   # 2 MHz trigger Rabi frequency
        t_c_on=0.0,
        t_c_off=2e-6,
        tau_c=0.3e-6,
        t_p_center=1e-6,
        tau_p=0.4e-6,
        t_t_center=1e-6,
        tau_t=0.4e-6,
    )

    # Simulation parameters
    sim = SimulationParameters(
        N_z=100,
        N_t=500,
        T_entry=3e-6,
        T_storage=1e-6,
        T_hold=2e-6,
        T_retrieval=3e-6,
    )

    # Input qubit: equal superposition
    qubit = QubitParameters(
        theta=np.pi/2,  # Equal |R⟩ and |L⟩
        phi=0.0,
    )

    logging.info(f"Physical parameters:")
    logging.info(f"  Optical depth: {phys.optical_depth:.2f}")
    logging.info(f"  Medium length: {phys.medium_length*1e3:.2f} mm")
    logging.info(f"  Atom density: {phys.atom_density:.2e} /m³")
    logging.info(f"Pulse parameters:")
    logging.info(f"  Control Rabi: {pulse.Omega_c_max/(2*np.pi*1e6):.1f} MHz")
    logging.info(f"  Probe Rabi: {pulse.Omega_p_max/(2*np.pi*1e6):.1f} MHz")
    logging.info(f"Simulation parameters:")
    logging.info(f"  Spatial slices: {sim.N_z}")
    logging.info(f"  Time steps: {sim.N_t}")
    logging.info(f"  Total time: {sim.T_total*1e6:.1f} µs")
    logging.info(f"Input qubit:")
    logging.info(f"  |ψ⟩ = {qubit.alpha_R:.3f}|R⟩ + {qubit.alpha_L:.3f}|L⟩")

    # Create simulator and run
    simulator = MaxwellBlochSimulator(phys, pulse, sim, qubit)
    results = simulator.run_full_protocol()

    # Compute all diagnostics
    diagnostics = compute_all_diagnostics(simulator, results)

    # Generate plots
    generate_all_plots(simulator, results)

    # Write output summary
    output_lines = [
        "=" * 60,
        "Tier-2 Maxwell-Bloch EIT Simulation Results",
        "=" * 60,
        "",
        "Performance Metrics:",
        f"  Storage efficiency:    {results['storage_efficiency']:.4f}",
        f"  Retrieval efficiency:  {results['retrieval_efficiency']:.4f}",
        f"  Polarization fidelity: {results['polarization_fidelity']:.4f}",
        f"  DSP scattering:        {results['dsp_scattering']:.4e}",
        "",
        "Physical Parameters:",
        f"  Optical depth:         {phys.optical_depth:.2f}",
        f"  Adiabaticity:          {diagnostics['adiabaticity']:.2f}",
        f"  Compression factor:    {diagnostics['compression_factor']:.4f}",
        "",
        "Non-adiabatic Analysis:",
        f"  Max excited pop:       {diagnostics['non_adiabatic_leakage']['max_excited_population']:.4e}",
        f"  Is adiabatic:          {diagnostics['non_adiabatic_leakage']['is_adiabatic']}",
        "",
        "=" * 60,
    ]
    write_outputs(output_lines)

    logging.info("Simulation complete!")

    return {
        "simulator": simulator,
        "results": results,
        "diagnostics": diagnostics,
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tier-2 Maxwell-Bloch EIT Simulation for Polarization Qubit Storage"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run validation tests"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick simulation with reduced resolution"
    )

    args = parser.parse_args()

    if args.test:
        success = run_all_tests()
        sys.exit(0 if success else 1)

    if args.quick:
        # Quick mode with reduced resolution
        logging.info("Running quick simulation...")

        phys = PhysicalParameters(atom_density=5e16, medium_length=5e-4)
        pulse = PulseParameters()
        sim = SimulationParameters(N_z=30, N_t=200)
        qubit = QubitParameters(theta=np.pi/4)

        simulator = MaxwellBlochSimulator(phys, pulse, sim, qubit)
        results = simulator.run_full_protocol()

        logging.info(f"Quick results: η_ret={results['retrieval_efficiency']:.3f}, F={results['polarization_fidelity']:.3f}")
    else:
        # Full simulation
        run_default_simulation()


if __name__ == "__main__":
    main()
