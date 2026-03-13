# Photometry Pipeline V1

A strict, chunk-streaming fiber photometry analysis pipeline for long-term recordings. It processes raw recordings into artifact-corrected, normalized signals and summary features, producing structured output artifacts (`run_report.json`, `events.ndjson`, traces, features, QC, and visualizations) with full auditability. The pipeline fails loudly on data or configuration violations rather than silently producing invalid results.

## Installation

Requires Python 3.10+ and pip.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

Dependencies (from `requirements.txt`): numpy, scipy, pandas, pyyaml, tqdm, pyarrow. Visualization (`viz/`) additionally requires matplotlib (install separately if not already present).

## Input Data Formats

### RWD Format

Each session is a folder containing a `fluorescence.csv` file. Sessions are discovered by recursive search for these folders, then sorted in natural order.

Required CSV columns:
- A time column (configured via `rwd_time_col`, default `"Time(s)"`)
- Paired UV (isosbestic) and signal (calcium) columns per ROI, matched by suffix (configured via `uv_suffix` and `sig_suffix`, for example `"G"` and `"R"`, or `"-410"` and `"-470"`)

### NPM Format

Multi-row format with interleaved LED states. Uses explicit timestamp and LED columns defined in configuration (`npm_time_axis`, `npm_frame_col`, `npm_system_ts_col`, `npm_computer_ts_col`, `npm_led_col`, `npm_region_prefix`, `npm_region_suffix`). Supports strict and permissive handling of partial chunks via `allow_partial_final_chunk`.

## High-Level Architecture (Two-Pass System)

### Pass 1: Session-Wide Baseline Estimation

Estimates a stable session-wide baseline fluorescence (F0) for each ROI using deterministic reservoir sampling (seeded via `config.seed`). A configured percentile (`baseline_percentile`) of this reservoir is used as F0. If F0 is NaN, infinite, or below `f0_min_value`, the ROI is flagged as invalid and recorded in QC artifacts.

### Pass 2: Artifact Correction, Normalization, and Feature Extraction

For each chunk:

1. The chunk is loaded, validated, and resampled onto a uniform grid at `target_fs_hz`.
2. Raw signals are low-pass filtered (cutoff `lowpass_hz`) if enabled.
3. Dynamic isosbestic regression is performed using a sliding window (`window_sec`, `step_sec`).
4. Artifact-corrected delta F and dF/F are computed using the Pass 1 baseline.
5. Per-chunk, per-ROI features are extracted (mean, median, std, MAD, peak_count, AUC).
6. Traces, features, QC, and visualizations are written to disk.

If regression fails for a given ROI or window, outputs are set to NaN and recorded in QC logs.

## Command Line Usage

### `analyze_photometry.py` (Primary CLI)

```bash
python analyze_photometry.py \
  --input <INPUT_DIR> \
  --out <OUTPUT_DIR> \
  --config config.yaml \
  --format rwd
```

**Required arguments:**

| Flag | Description |
|------|-------------|
| `--input` | Folder containing session data (RWD folders or NPM CSVs) |
| `--out` | Output directory |
| `--config` | Path to YAML configuration file |

**Optional arguments:**

| Flag | Description |
|------|-------------|
| `--format {auto,rwd,npm}` | Force input format (default: `auto`) |
| `--mode {phasic,tonic}` | Analysis mode: `phasic` (dynamic sliding-window fit) or `tonic` (global fit). Default: `phasic` |
| `--recursive` | Search input folder recursively |
| `--file-glob` / `--glob` | Glob pattern for CSV files (default: `*.csv`) |
| `--overwrite` | Overwrite output directory if it exists |
| `--include-rois` | Comma-separated list of ROIs to process exclusively |
| `--exclude-rois` | Comma-separated list of ROIs to exclude |
| `--traces-only` | Produce traces and QC only, skip feature extraction (`features.csv`) |
| `--event-signal {dff,delta_f}` | Signal used for peak detection features (default from config: `dff`) |
| `--representative-session-index INT` | Force a specific session (0-based) for representative plots. Fails closed if out of range |
| `--preview-first-n INT` | Preview mode: process only the first N discovered sessions (after discovery and sort) |
| `--events-path PATH` | Path to events.ndjson file; the pipeline appends audit events when provided |

**Example commands:**

Standard full run:
```bash
python analyze_photometry.py \
  --input "C:/Data/MyExperiment" \
  --out "C:/Results/run_001" \
  --config config.yaml \
  --format rwd --mode phasic --recursive
```

Traces-only run (no feature extraction):
```bash
python analyze_photometry.py \
  --input "C:/Data/MyExperiment" \
  --out "C:/Results/traces_only" \
  --config config.yaml \
  --format rwd --traces-only
```

Preview run with representative session and event signal override:
```bash
python analyze_photometry.py \
  --input "C:/Data/MyExperiment" \
  --out "C:/Results/preview_run" \
  --config config.yaml \
  --format rwd \
  --preview-first-n 3 \
  --representative-session-index 0 \
  --event-signal delta_f \
  --events-path "C:/Results/preview_run/events.ndjson"
```

### `tools/run_full_pipeline_deliverables.py` (Runner)

Orchestrates both tonic and phasic analysis passes, creates a structured run directory under `--out-base`, and writes `status.json`, `MANIFEST.json`, and `events.ndjson`. All CLI flags from `analyze_photometry.py` are threaded through to the subprocess invocations.

```bash
python tools/run_full_pipeline_deliverables.py \
  --input <INPUT_DIR> \
  --out-base <BASE_DIR> \
  --config config.yaml \
  --format rwd
```

**Key arguments (beyond those shared with `analyze_photometry.py`):**

| Flag | Description |
|------|-------------|
| `--out-base` | Base directory; the runner creates `<out-base>/<run-id>/` |
| `--out` | Legacy mode: explicit run directory path (mutually exclusive with `--out-base`) |
| `--run-id` | Optional run ID for `--out-base` mode |
| `--validate-only` | Validate inputs and exit without running analysis |
| `--sessions-per-hour INT` | Force sessions per hour |
| `--session-duration-s FLOAT` | Recording duration in seconds per chunk |
| `--events {auto,PATH}` | Events NDJSON path, or `auto` (default: `run_dir/events.ndjson`) |
| `--cancel-flag {auto,PATH}` | Cancel flag path for cooperative cancellation |

**Example: Full runner invocation:**
```bash
python tools/run_full_pipeline_deliverables.py \
  --input "C:/Data/MyExperiment" \
  --out-base "C:/Results" \
  --config config.yaml \
  --format rwd \
  --preview-first-n 5
```

**Example: Validate-only:**
```bash
python tools/run_full_pipeline_deliverables.py \
  --input "C:/Data/MyExperiment" \
  --out "C:/Results/validate_check" \
  --config config.yaml \
  --format rwd \
  --validate-only
```



## Supported vs. Legacy / Verification Tools

The supported workflow for this repository is the GUI-deliverables path centered on:
- `tools/run_full_pipeline_deliverables.py`
- `analyze_photometry.py`
- `photometry_pipeline/pipeline.py`
- HDF5 cache-backed plotting and deliverables

Some scripts under `tools/` are retained as verification-only utilities or deprecated legacy utilities. These non-GUI scripts are not part of the supported artifact contract and are not maintained as part of the live GUI runtime path.

Legacy scripts may still reference obsolete outputs such as `traces/`, `viz/`, or older standalone workflows. Verification-only scripts may still be useful for diagnostics, smoke tests, and migration validation, but they are not part of the live GUI runtime path.

## Configuration (`config.yaml`)

All numerical behavior is controlled via configuration. Unknown keys cause a hard error.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_fs_hz` | `40.0` | Output sampling rate (Hz) |
| `chunk_duration_sec` | `600.0` | Duration of each chunk after resampling (seconds) |
| `lowpass_hz` | `1.0` | Low-pass filter cutoff frequency |
| `filter_order` | `3` | Butterworth filter order |
| `window_sec` | `60.0` | Sliding window length for dynamic regression |
| `step_sec` | `10.0` | Step size for sliding window |
| `min_valid_windows` | `5` | Minimum valid regression windows required |
| `baseline_method` | `uv_raw_percentile_session` | F0 estimation method |
| `baseline_percentile` | `10.0` | Percentile used to compute F0 |
| `f0_min_value` | `1e-9` | Minimum allowed F0 before ROI is flagged invalid |
| `peak_threshold_method` | `mean_std` | Peak detection threshold: `mean_std`, `percentile`, `median_mad`, or `absolute` |
| `peak_threshold_k` | `2.0` | Standard deviations above mean (for `mean_std`) |
| `peak_threshold_percentile` | `95.0` | Percentile (for `percentile` method) |
| `peak_threshold_abs` | `0.0` | Absolute threshold (must be >0 when method is `absolute`) |
| `peak_min_distance_sec` | `0.5` | Minimum distance between peaks (seconds) |
| `event_signal` | `dff` | Signal for peak detection: `dff` or `delta_f` |
| `representative_session_index` | `None` | Force representative session (0-based index) |
| `preview_first_n` | `None` | Preview mode: process only first N sessions (must be >0) |
| `allow_partial_final_chunk` | `False` | Strict (False) vs permissive (True) chunk handling |
| `rwd_time_col` | `Time(s)` | Time column name in RWD CSVs |
| `uv_suffix` | `-410` | Suffix identifying UV (isosbestic) channels |
| `sig_suffix` | `-470` | Suffix identifying signal (calcium) channels |

## Outputs

After a successful run, the output directory contains:

### `run_report.json`

Machine-readable run report with configuration snapshot, derived settings, and analytical contract. The `run_context` object includes:

| Field | Description |
|-------|-------------|
| `run_type` | `"full"` or `"preview"` |
| `preview` | Preview metadata object (or `null`). Contains `selector`, `first_n`, `n_total_discovered`, `n_sessions_resolved` |
| `traces_only` | Whether traces-only mode was active |
| `features_extracted` | `null` (normal run, deferred) or `false` (traces-only) |
| `event_signal` | `"dff"` or `"delta_f"` |
| `representative_session_index` | 0-based index of the representative session |
| `representative_session_id` | Session folder name of the representative session |
| `n_sessions_resolved` | Number of sessions in the resolved list |
| `user_provided_representative_session_index` | Whether the user explicitly set the representative index |

### `events.ndjson`

Append-only newline-delimited JSON audit log. Key event types (by `stage:type`):

| Event | When | Payload |
|-------|------|---------|
| `engine:context` | Before pipeline.run | `run_type`, `preview`, `traces_only`, `event_signal`, `representative_session_index` |
| `inputs:preview` | After session discovery (if preview active) | `selector`, `first_n`, `n_total_discovered`, `n_sessions_resolved` |
| `inputs:representative_session` | After representative resolution | `representative_session_index`, `representative_session_id`, `user_provided`, `resolved_session_ids_preview` |
| `inputs:roi_selection` | After pipeline.run | `discovered_rois`, `selected_rois`, `include_rois`, `exclude_rois` |

### `status.json` (runner only)

Written by `run_full_pipeline_deliverables.py`. Contains `run_type`, `preview`, `phase`, `status`, timing, input/output paths, and error lists.

### `traces/`

One CSV per chunk with columns: `time_sec`, `{ROI}_uv_raw`, `{ROI}_sig_raw`, `{ROI}_uv_fit`, `{ROI}_deltaF`, `{ROI}_dff`.

### `features/features.csv`

Per-chunk, per-ROI summary: `chunk_id`, `source_file`, `roi`, `mean`, `median`, `std`, `mad`, `peak_count`, `auc`. Skipped in `--traces-only` mode.

### `qc/qc_summary.json`

Quality control summary: failed chunks, invalid baseline ROIs, chunk failure fraction.

### `run_metadata.json`

Configuration values, baseline method, F0 values, regression parameters, invalid baseline ROIs.

### `viz/`

Diagnostic plots:
- **Plot A** (`plot_A_raw_traces_{ROI}.png`): Single-session raw UV and signal traces
- **Plot B** (`plot_B_continuous_{ROI}.png`): Continuous multi-day overlay
- **Plot C** (`plot_C_stacked_{ROI}.png`): Stacked session-aligned delta F
- **Plot D** (`plot_D_correction_impact_{ROI}.png`): Correction impact panel (raw, fit, corrected)

Representative session plots (A, D) are generated for the session selected by `representative_session_index`. Aggregate plots (B, C) cover all sessions.

## Safety and Failure Philosophy

The pipeline is intentionally conservative. It will fail or warn if:

- Timestamps are non-monotonic or non-finite
- Baselines are invalid (NaN, infinite, or below `f0_min_value`)
- Regression windows are ill-posed
- `representative_session_index` is out of range (raises `ValueError` before analysis)
- `preview_first_n` is not a positive integer (raises `ValueError` at config load)
- `event_signal` is not `dff` or `delta_f` (raises `ValueError` at config load)

When `representative_session_index` is explicitly provided and the session fails during plotting, a `RuntimeError` is raised (fail-closed). When no index is provided, the pipeline defaults to the first loadable session and logs warnings on failure.

## Tests

| Test File | Purpose | Command |
|-----------|---------|---------|
| `tests/test_event_signal_selection.py` | Verifies `dff` vs `delta_f` routing, absolute threshold method, traces-only semantics, and audit stamping | `pytest tests/test_event_signal_selection.py -v` |
| `tests/test_representative_session_index.py` | Verifies fail-closed bounds, selection changes output, default first-loadable fallback, audit event schema | `pytest tests/test_representative_session_index.py -v` |
| `tests/test_preview_mode_first_n.py` | Verifies preview hard wall (excluded sessions produce no artifacts), run_report and events stamping, config validation | `pytest tests/test_preview_mode_first_n.py -v` |

Run all three:
```bash
pytest tests/test_event_signal_selection.py tests/test_representative_session_index.py tests/test_preview_mode_first_n.py -v
```

## Roadmap

- GUI presets and YAML template generation
- Additional preview selectors beyond `first_n` (for example, random sampling or session-name pattern matching)
- NPM format end-to-end integration tests
