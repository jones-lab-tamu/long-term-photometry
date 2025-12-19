# Photometry Pipeline V1

A robust, lab-default, long-term fiber photometry analysis pipeline in Python.

## Overview

This pipeline implements a strict **two-pass streaming architecture** to process chunked photometry recordings (RWD or NPM formats). It is designed for scalability and scientific reproducibility, enforcing:

1.  **Unified Internal Representation**: All inputs are converted to a uniform time grid (default 40Hz).
2.  **Strict Separation of Concerns**: Artifact correction (dynamic regression) is decoupled from baseline normalization (session-level F0).
3.  **Two-Pass Logic**:
    *   **Pass 1**: Computes session-wide baseline statistics (F0) without loading the entire session into memory.
    *   **Pass 2**: Streams data again to apply artifact correction, normalization, and feature extraction.

## Analytical Contract (Strict Mode and Signal Semantics)

*   **Strict mode guarantees**:
    *   timestamps are strictly increasing (per channel where relevant)
    *   resampling onto a fixed grid of length round(chunk_duration_sec * target_fs_hz)
    *   no extrapolation beyond available data (strict start/end coverage enforced)
    *   NPM UV and SIG are validated independently (per-channel checks)
*   **Strict mode does NOT guarantee**:
    *   no tonic–phasic decomposition (not implemented yet)
    *   no Bayesian or probabilistic artifact correction (not implemented yet)
    *   no event-triggered analysis framework (not a current goal)
*   **Signal definitions**:
    *   `uv_raw`: raw isosbestic channel on the canonical grid
    *   `sig_raw`: raw calcium-dependent channel on the canonical grid
    *   `uv_fit`: estimated artifact component derived from uv_filt fit to sig_filt (artifact estimate only)
    *   `delta_f`: sig_raw - uv_fit (artifact-corrected numerator only, not normalized)
    *   `F0`: a separate baseline reference used ONLY for normalization (placeholder, policy defined later)
*   **Tonic policy statement**:
    *   "Slow tonic structure is preserved by preprocessing, but explicit tonic–phasic decomposition is not yet implemented."

**Strict Mode Note**: Strict NPM now enforces per-channel monotonicity pre-alignment and computes t0 from earliest valid timestamps, preventing silent misalignment when CSV rows are unsorted.

## Installation

Requires Python 3.9+.

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

**Note**: `pyarrow` is required for Parquet output support.

## Usage

### 3. Run Analysis
Run the pipeline on an input folder containing CSV chunks.
**Note**: You must provide a strict configuration file (see `config.py` defaults or `tests/strict_config.yaml`).

```bash
python analyze_photometry.py --input data/session_01 --out output/session_01 --config config.yaml
```

**Arguments:**
*   `--input`: Path to folder containing chunked CSV files.
*   `--out`: Output directory.
*   `--format`: `rwd` or `npm`.
*   `--config`: Path to YAML config (Required).
*   `--overwrite`: Force overwrite of existing output.

### Output Structure

```
output_folder/
├── run_metadata.json       # Session params, F0 values, Global Fit coeffs
├── config_used.yaml        # Copy of effective configuration
├── traces/                 # Per-chunk CSVs with trace data
│   ├── chunk_0000.csv
│   └── ...
├── features/               # Extracted features
│   ├── features.parquet    # Efficient binary format
│   └── features.csv        # Summary CSV
└── qc/
    └── qc_summary.json     # Quality control stats (failed chunks, ROIs)
```

## Synthetic Data Generator

A strict validation tool included in `tests/generate_synthetic_session.py` generates data with known ground-truth properties (GCaMP events, shared artifacts, drift).

```bash
python tests/generate_synthetic_session.py --format rwd --out tests/syn_data --n_chunks 5 --chunk_duration_sec 600 --fs_hz 40 --seed 42
```

Use this to verify pipeline correctness (artifact rejection, event recovery) under controlled conditions.

## License

Internal Research Use.
