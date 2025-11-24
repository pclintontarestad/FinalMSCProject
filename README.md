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
   summaries:

   ```bash
   python tripod_sim.py
   ```

   The main section will plot pulse envelopes, construct the encoder, show
   EIT-style diagnostics, plot SU(3) expectations/leakage, and scan the
   transparency window.

## Working incrementally

The functions in `tripod_sim.py` are written to be imported individually so that
future changes can be focused and reviewable. For example, you can reuse
`build_encoder` in a separate script or notebook without triggering the plotting
heavyweight main routine.
