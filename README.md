# Photometry Pipeline (GUI + CLI)

Photometry Pipeline is a GUI-first wrapper and analysis stack for long-session fiber photometry data (RWD/NPM). It runs tonic + phasic processing, writes reproducible run artifacts, and provides a results workspace for review and retuning.

## Current Scope (What Exists Now)

- GUI-first setup, validation, and run launching
- Tonic and phasic analysis outputs
- Post-run review and retuning tools
- Results workspace for inspecting plots, dayplots, and summaries

## Requirements

- Recommended for student beta: Python 3.10 or 3.11
- Use a fresh virtual environment and install from the requirements files below.
- Windows is the primary tested environment

## Installation

From repo root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

GUI install (recommended for students):

```bash
pip install -r requirements_gui.txt
```

CLI-only install:

```bash
pip install -r requirements.txt
```

Notes:
- `requirements_gui.txt` includes `requirements.txt` plus PySide6.
- Core runtime dependencies now include: numpy, scipy, pandas, pyyaml, tqdm, pyarrow, h5py, matplotlib, Pillow, scikit-learn.

## Launch

GUI (recommended):

```bash
python -m gui.app
```

CLI full runner:

```bash
python tools/run_full_pipeline_deliverables.py --input <INPUT_DIR> --out-base <OUTPUT_BASE_DIR> --config <CONFIG_YAML> --format <auto|rwd|npm>
```

Single-pass CLI (advanced/legacy usage):

```bash
python analyze_photometry.py --input <INPUT_DIR> --out <OUTPUT_DIR> --config <CONFIG_YAML> --format <auto|rwd|npm> --mode <phasic|tonic>
```

## Student Quickstart

1. Install dependencies (GUI path above).
2. Launch `python -m gui.app` from repo root.
3. In the GUI, set:
   - Input directory
   - Output base directory
   - Config source (default `config/qc_universal_config.yaml` or custom YAML)
4. Click Validate, then Run.
5. After completion, use the Results workspace tabs and (if needed) Correction Retune.

## Input Expectations (High Level)

- RWD: session folders containing `fluorescence.csv`
- NPM: CSV files with LED-state structured rows
- Format can be auto-detected or forced (`rwd`/`npm`)

## Key Output Structure (Runner)

Each run directory (under `--out-base`) includes artifacts such as:

- `status.json`
- `MANIFEST.json`
- `run_report.json`
- `config_effective.yaml`
- `_analysis/phasic_out/...`
- `_analysis/tonic_out/...`
- region-level plots/tables surfaced in the results workspace

## Known Beta Constraints

- Correction retune/downstream retune require a successful completed run with phasic cache artifacts.
- If custom config source is enabled, the path must point to a valid YAML before running.
- Verification/legacy scripts under `tools/verification` and `tools/legacy` are not the primary student workflow.
