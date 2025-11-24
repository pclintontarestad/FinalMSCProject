# FinalMSCProject

This repository now contains a reusable Python module, `tripod_sim.py`, that
captures the few-mode tripod atom and dual-photon simulation utilities that were
previously held in a notebook. The module retains the original numerical setup
(Rb-87 D2, TQM-SIM-V2 style) and exposes functions for plotting control/probe
pulses, building the encoder, running EIT-style diagnostics, evaluating SU(3)
expectations, and scanning the transparency window.

## Getting started

1. Install the Python dependencies (consider using a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. Run the module directly to reproduce the exploratory plots and console
   summaries. Plots are written to `thesis_figures/` as PDFs so the script can
   run headlessly without blocking on `plt.show()`:

   ```bash
   python tripod_sim.py
   ```

   The main section now produces a validation bundle that includes:

   - Full-timeline control/probe envelopes and mixing angles
   - Write-window populations, photon absorption, bright-population traces, and
     dark-coupling residuals
   - Encoder trajectory plots (coefficient magnitudes, overlaps, and Gram
     eigenvalues vs time)
   - SU(3)/bright-leakage diagnostics and transparency-window scan
   - Read-window photon re-emission and a small retrieval-fidelity landscape

   A concise summary of the run (storage efficiency, leakage metrics,
   tomography fidelities, etc.) is also written to `tripod_run_outputs.txt` in
   the repository root for later inspection.

## Working incrementally

The functions in `tripod_sim.py` are written to be imported individually so that
future changes can be focused and reviewable. For example, you can reuse
`build_encoder` in a separate script or notebook without triggering the plotting
heavyweight main routine.
