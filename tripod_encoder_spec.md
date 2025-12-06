# Tripod Encoder Simulator Specification

## 0. High-Level Design & Goals
- Simulate a single-atom four-level tripod system with fixed basis ordering: |e>, |g_{-1}>, |g_0>, |g_{+1}>.
- Drive with three optical fields (σ⁺, π, σ⁻) with time-dependent Rabi frequencies.
- Propagate the full density matrix with coherent Hamiltonian and configurable Lindblad channels.
- Support two modes: projection (initialise directly in logical subspace) and STIRAP (explicit pulses from |g_0>).
- Provide diagnostics for logical projection, fidelities, purity, trace loss, encoder map, and Gram/SVD analysis.

## 1. Core Conventions
- Basis index mapping: 0→|e>, 1→|g_{-1}>, 2→|g_0>, 3→|g_{+1}>.
- Logical photonic qubit α|σ⁺> + βe^{iφ}|σ⁻> maps to atomic α|g_{-1}> + βe^{iφ}|g_{+1}>.
- Canonical inputs for tomography: |0_L>=|g_{-1}>, |1_L>=|g_{+1}>, |+>=(|g_{-1}>+|g_{+1}>)/√2, |+_y>=(|g_{-1}>+i|g_{+1}>)/√2.

## 2. Recommended Package Layout
```
tripod_encoder/
    __init__.py
    params.py
    operators.py
    pulses.py
    lindblad.py
    simulation.py
    diagnostics.py
    examples/
        basic_run.py
        tomography_demo.py
```

## 3. Parameter Containers (params.py)
### PhysicalParameters
- `Gamma_e`: total spontaneous emission rate from |e>.
- `Gamma_branch`: dict mapping 'g-1','g0','g+1' → branching fractions or absolute rates.
- `delta_minus1(t)`, `delta_0(t)`, `delta_plus1(t)`: detunings, constants or callables.
- `Delta`: single-photon detuning of |e>.
- `gamma_dephasing_pairs`: dict of pairwise dephasing rates `{('g-1','g0'): val, ...}`.
- `Gamma1`: ground-manifold T1 rate (optional).
- `p_th`: thermal target populations `{state: prob}`.
- `gamma_raman`: dict of Raman scattering rates between pairs.

### ControlParameters
- Max Rabi amplitudes: `Omega_sigma_plus_max`, `Omega_sigma_minus_max`, `Omega_pi_max`.
- Pulse timings/widths: `t0_probe`, `tau_probe`, `t0_control`, `tau_control`.
- Noise: amplitude/phase noise variances per field (`sigma_eps`, `sigma_phi`).

### SimulationParameters
- `t_start`, `t_end`, `n_steps` define `tlist`.
- Solver options: `atol`, `rtol`, `method` for `qutip.mesolve`.

## 4. Operators & Basis (operators.py)
- Basis helpers: `basis_e()`, `basis_gm1()`, `basis_g0()`, `basis_gp1()`.
- Projectors: `Pe`, `Pg_m1`, `Pg_0`, `Pg_p1`.
- Transition operators: `sigma_e_gm1`, `sigma_e_g0`, `sigma_e_gp1` and adjoints. Generic helper `sigma_e_g(label)`.

## 5. Pulses & Rabi Functions (pulses.py)
- Gaussian probe envelope: `probe_envelope(t, cp)` with center `t0_probe`, width `tau_probe`.
- Gaussian control envelope: `control_envelope(t, cp)` with `t0_control`, `tau_control`.
- Logical-to-field mapping:
  - `Omega_sigma_plus(t, cp, alpha, beta, phi) = alpha * Omega_sigma_plus_max * probe_envelope`.
  - `Omega_sigma_minus(t, cp, alpha, beta, phi) = beta * e^{iφ} * Omega_sigma_minus_max * probe_envelope`.
  - `Omega_pi(t, cp) = Omega_pi_max * control_envelope`.
- Optional per-shot noise multipliers `(1+ε) e^{iφ_noise}` sampled once per run.

## 6. Lindblad Operators (lindblad.py)
- Spontaneous emission: `sqrt(Gamma_m) |g_m><e|` for each ground state, toggled via flag.
- Pairwise dephasing: `sqrt(gamma_ij/2) (|i><i| - |j><j|)` for listed pairs.
- Ground T1 relaxation toward `p_th`: `sqrt(Gamma1 * p_th[j]) |j><i|` for ordered pairs i≠j.
- Raman scattering: `sqrt(gamma_raman[(m,n)]) |n><m|` for specified pairs.

## 7. Hamiltonian Construction (simulation.py)
- Static detuning part: `H_det = Delta Pe + delta_-1 Pg_m1 + delta_0 Pg_0 + delta_+1 Pg_p1`.
- Interaction terms (common prefactor -1/2):
  - `H_sigma_plus = -0.5 (sigma_e_gm1 + h.c.)`
  - `H_pi = -0.5 (sigma_e_g0 + h.c.)`
  - `H_sigma_minus = -0.5 (sigma_e_gp1 + h.c.)`
- Time-dependent Hamiltonian list: `[H_det, [H_sigma_plus, Omega_sigma_plus_t], [H_sigma_minus, Omega_sigma_minus_t], [H_pi, Omega_pi_t]]` with callable Ω(t).
- Detunings may be constants or time-dependent via supplied callables or pre-sampled arrays.

## 8. Initialisation Modes
### Projection mode
- Build logical pure state `alpha|g_{-1}> + beta e^{iφ}|g_{+1}>`, set `rho0=ket2dm`.
- Set σ±/π couplings to zero to model storage already completed; focus on idle/gate noise.

### STIRAP mode
- Start in |g_0> (or chosen ground state) with explicit σ± probes and π control pulses enabling adiabatic transfer.

## 9. Simulation Driver (simulation.py)
Implement `run_tripod_sequence(alpha, beta, phi, mode, physical_params, control_params, sim_params, enable_channels)` to:
1. Build operators and Hamiltonian pieces.
2. Assemble `c_ops` from `lindblad.py` using flags in `enable_channels`.
3. Initialise `rho0` per mode.
4. Create `tlist = linspace(t_start, t_end, n_steps)`.
5. Call `qutip.mesolve` with `H`, `rho0`, `tlist`, `c_ops`, and any `args` for Ω(t).
6. Return `res.states`, `tlist`, and metadata.

## 10. Diagnostics (diagnostics.py)
- Logical projector: `P_logical = Pg_m1 + Pg_p1`; obtain `rho_logical = P_logical rho P_logical`, `trace_logical`, and normalized `rho_logical_norm`.
- Fidelity: use full 4-level target or logical-only target; compute both raw and logical-subspace fidelities.
- Purity: compute from `rho_logical_norm`; report trace in logical subspace separately.
- Encoder/Gram analysis:
  - Generate canonical inputs `[|0_L>, |1_L>, |+>, |+_y>]`.
  - For noiseless runs, extract 2×2 encoder matrix from output kets and compute `G = E†E` and its SVD.
  - For mixed outputs, build Gram matrix `G_ij = Tr(rho_i rho_j)` (logical subspace) and eigendecompose to separate unitary distortion vs loss.
- Parameter sweep helpers for plotting fidelity/trace/Gram eigenvalues vs detuning or pulse area (plateau vs cliff behaviour).

## 11. Validation Scenarios
1. **Dark-state sanity (no noise, symmetric, δ=0):** STIRAP from |g_0> → |g_{±1}> with low excited-state population.
2. **Projection identity:** initialise logical state, set H≈0 and tiny noise → Gram ≈ identity and singular values ≈ 1.
3. **Asymmetric distortion (no dissipation):** unequal σ± amplitudes → unitary rotation (trace/purity ≈1, Gram eigenvalues ≈1).
4. **Detuning/noise destruction:** add two-photon detuning or dephasing → reduced trace/purity and subunit Gram eigenvalues with fidelity cliffs.
