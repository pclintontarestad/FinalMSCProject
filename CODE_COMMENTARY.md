# Tripod simulation code commentary

This note summarizes the current `tripod_sim.py` implementation without altering
any of the logic. It is intended to help future edits stay aligned with the
existing structure and numerical assumptions.

## Overall structure

`tripod_sim.py` keeps the simulation in a single module, split into numbered
sections that mirror the original notebook ordering. Physical constants are
defined up front using Rb-87 D2 parameters with time and frequency normalized to
`Gamma_e`. Subsequent helpers build pulses, Hilbert-space operators, and
analysis routines.

### Four-layer navigation for new contributors

1. **Physical + numerical setup** (sections 1–3, 7, 8, 11): dimensional Rb-87
   inputs, pulse shapes, time grids, and the baseline EIT validation scans.
2. **Encoder construction** (section 6): builds the 3×2 encoder matrices
   (`E_phys`, `E_iso`) and reports efficiencies/conditioning.
3. **Three-phase memory cycle** (section 8): write → settle → gate → read with
   fidelity/tomography and photon return metrics.
4. **SU(3)/leakage diagnostics** (sections 9–10): time-dependent dark/bright
   basis, bright-population leakage, non-adiabatic dark–bright coherence, and
   encoder non-isometry (`iso_gap`).

### Key parameter choices

- Detuning and Rabi frequencies are set in dimensional units and immediately
  scaled to the dimensionless simulation frame (`Gamma_e`-scaled). The time grid
  uses 1001 points across a 5 µs write window, matching the `TQM-SIM-V2`
  conventions. 【F:tripod_sim.py†L23-L59】
- Gaussian pulse centers place the control earlier than the probe (`t_c = 2 µs`,
  `t_p = 3 µs`) to implement the counter-intuitive sequence. Both use the same
  width (`sigma_phys = 0.5 µs`). 【F:tripod_sim.py†L52-L59】

### Pulse and mixing-angle helpers

`Omega_c`, `Omega_p_eff`, and `mixing_angle` provide analytic envelopes and a
simple diagnostic for adiabaticity. The plotting helper converts the normalized
`Gamma_e` time base back to microseconds for readability. 【F:tripod_sim.py†L64-L116】

### Hilbert-space construction

The module fixes a four-level atomic basis (`|e>, |g_-1>, |g_0>, |g_+1>`) and
uses two single-photon modes limited to `N_ph = 2` (0 or 1 photon). All atomic
projectors and photon creation/annihilation operators are promoted to the full
tensor space early, so later routines can assemble Hamiltonians without
recomputing identities. 【F:tripod_sim.py†L124-L165】

### Time-dependent Hamiltonian

`H_tripod_t` builds the tripod Hamiltonian with time-dependent control-field
coupling, symmetric single-photon couplings (`g_plus == g_minus` from the shared
probe scale), and optional two-photon detunings passed via `args`. The function
returns a `qutip.Qobj` ready for `mesolve`, with the excited-state detuning
`-Delta` applied directly to `|e⟩`. 【F:tripod_sim.py†L173-L209】

### State preparation

Convenience states for single-`σ+` and single-`σ−` photons are defined, along
with `psi_in_sigma_plus` and `psi_in_sigma_minus` that tensor the photonic states
with the atomic `|g_0⟩` reference. These are the canonical inputs for encoder and
trajectory routines. 【F:tripod_sim.py†L217-L246】

### Encoder construction

`evolve_and_extract` evolves a normalized input polarization through the full
Hamiltonian, then projects the final state onto the three ground levels to build
amplitudes `(c_-1, c_0, c_+1)`. `build_encoder` assembles these into the
physical encoder `E_phys`, computes the Gram matrix/efficiency, and produces a
renormalized isometric encoder `E_iso`. Logged diagnostics include the Gram
matrix eigenvalues to indicate symmetry/efficiency. 【F:tripod_sim.py†L254-L356】

### Diagnostics and plots

- `run_trajectory` and `plot_eit_and_populations` compute populations and photon
  numbers versus time for arbitrary input polarizations using `mesolve`. Photon
  number operators (`num`) are prebuilt for efficiency. 【F:tripod_sim.py†L373-L447】
- `plot_su3_leakage_dashboard` is the thesis-facing SU(3) view: it plots the
  physical bright population, non-adiabatic dark–bright coherence, and the
  instantaneous encoder non-isometry metric `||G(t)-η(t)I||`, all using a
  time-dependent dark/bright basis built from the instantaneous couplings. This
  collapses to near-zero traces in the symmetric adiabatic case and lights up
  when Clebsch/Zeeman asymmetries or non-adiabatic ramps are introduced.
  【F:tripod_sim.py†L1739-L1833】
- `plot_su3_expectations` remains available for debugging generator-level
  expectations in the adapted basis; it uses the same diagnostics engine as the
  dashboard and logs Hermiticity/trace checks for the rotated λ_i. 【F:tripod_sim.py†L1835-L1895】
- `scan_transparency_window` sweeps two-photon detunings symmetrically about
  zero, integrates excited-state population as a proxy for scattering, and marks
  the control Rabi frequency as the expected transparency edge. 【F:tripod_sim.py†L644-L696】

### Entry point

The `__main__` block runs every major diagnostic in sequence: pulse plots,
encoder build, EIT-style population plots, SU(3) expectations, and the
transparency scan. This is helpful for interactive exploration but may be heavy
for automated tests; importing individual functions remains the lightweight
path. 【F:tripod_sim.py†L704-L716】

## Potential follow-ups

- Parameterization: the hard-coded Gaussian widths, centers, and detunings could
  be exposed as arguments or configuration to simplify scenario sweeps.
- Performance: `mesolve` is used everywhere without collapse operators; swapping
  in `sesolve` when dissipation is absent could reduce runtime.
- Testing: the module currently lacks automated assertions beyond successful
  execution; lightweight checks on encoder normalization and leakage radius
  bounds would help guard against regressions.
