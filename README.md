# Photometry Pipeline V1

A strict, chunk-streaming fiber photometry analysis pipeline for long-term recordings, designed to fail loudly on data or configuration violations rather than silently produce invalid results.

## What Does This Do?

This software processes raw fiber photometry recordings into artifact-corrected, normalized signals (ΔF/F) and summary features. It is built specifically for long recordings that cannot be loaded into memory at once and for datasets where timestamp irregularities, drift, or partial corruption are common.

Core problems addressed:

1. **Motion and Drift Artifacts**  
   Long recordings exhibit drift and motion artifacts. This pipeline removes them using isosbestic correction, fitting a control channel to the signal channel and subtracting the estimated artifact.

2. **Scalability for Long Recordings**  
   Recordings spanning hours to days are processed as independent chunks. Data are streamed from disk and never fully loaded into RAM.

3. **Strict Correctness Guarantees**  
   

The pipeline enforces explicit contracts on timestamps, baselines, and numerical validity. If these contracts are violated, the pipeline raises errors or records warnings rather than fabricating data.

## High-Level Architecture (Two-Pass System)

The analysis proceeds in two distinct passes over the input data.

### Pass 1: Session-Wide Baseline Estimation

**Goal**  
Estimate a stable session-wide baseline fluorescence (F0) for each ROI.

**Method**  
Each chunk is read sequentially. For each ROI, values are added to a deterministic reservoir sampler (seeded). A configured percentile of this reservoir is used as F0.

**Why This Exists**  
ΔF/F normalization requires a global baseline, but long recordings cannot be held in memory. Reservoir sampling provides a memory-bounded estimate.

**Failure Handling**  
If F0 is NaN, infinite, or below a minimum threshold, the ROI is flagged as invalid. Invalid baselines are recorded in run metadata and QC artifacts.

### Pass 2: Artifact Correction, Normalization, and Feature Extraction

For each chunk:

1. The chunk is loaded and validated.
2. Timestamps are resampled onto a uniform grid at `target_fs_hz` for `chunk_duration_sec`.
3. Raw signals are low-pass filtered if enabled.
4. **Dynamic isosbestic regression** is performed using a sliding window:
   - Regression is computed on filtered signals.
   - The fitted artifact is applied to raw signals.
5. Artifact-corrected ΔF is computed.
6. ΔF/F is computed using the Pass 1 baseline.
7. Per-chunk, per-ROI features are extracted.
8. Traces, features, QC information, and visualizations are written to disk.

If regression fails for a given ROI or window, outputs for that ROI are set to NaN and recorded in QC logs.

## Supported Input Formats

Input data must be provided as CSV files representing sequential chunks.

### RWD Format
- Columns include a time column (for example `Time(s)`) and paired UV and signal columns per ROI.
- UV and signal channels are matched by suffix.

### NPM Format
- Multi-row format with interleaved LED states.
- Uses explicit timestamp and LED columns defined in the configuration.
- Supports strict and permissive handling of partial chunks.

## Running the Pipeline

### Installation
Requires Python 3.9 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Command Line Usage

```bash
python analyze_photometry.py \
  --input "C:/Path/To/Data" \
  --out "C:/Path/To/Output" \
  --config config.yaml
```

**Arguments:**

*   `--input`: Folder containing chunk CSV files.
*   `--out`: Output directory.
*   `--config`: Path to YAML configuration file.
*   `--format`: auto, rwd, or npm (optional).
*   `--recursive`: Search input folder recursively (optional).
*   `--glob`: File glob pattern (optional). Alias: `--file-glob`.

## Configuration (`config.yaml`)
All numerical behavior is controlled via configuration.

**Key parameters:**

*   `target_fs_hz`: Output sampling rate used for resampling chunks.
*   `chunk_duration_sec`: Duration of each chunk after resampling.
*   `lowpass_hz`: Low-pass filter cutoff frequency. If set to 0 or ≥ Nyquist, filtering is disabled and a warning is recorded.
*   `baseline_method`: Method for F0 estimation, for example percentile of raw UV or global-fit UV.
*   `baseline_percentile`: Percentile used to compute F0.
*   `f0_min_value`: Minimum allowed F0 before an ROI is flagged invalid.
*   `window_sec`: Sliding window length for dynamic regression.
*   `peak_threshold_method`: "mean_std" or "percentile".
*   `peak_threshold_k`: Number of standard deviations above the mean when using "mean_std".
*   `peak_threshold_percentile`: Percentile used when "percentile" mode is selected.
*   `allow_partial_final_chunk`: Controls strict versus permissive handling of incomplete coverage.

## Strict vs Permissive Behavior

### Strict Mode (`allow_partial_final_chunk = False`)
*   Timestamps must be strictly increasing and finite.
*   Full temporal coverage is required.
*   Missing coverage or invalid timestamps cause hard failure.

### Permissive Mode (`allow_partial_final_chunk = True`)
*   Timestamps must still be strictly increasing and finite.
*   Partial coverage is allowed.
*   Samples outside measured time support are filled with NaN, never extrapolated.

## Outputs
After a successful run, the output directory contains:

*   **`traces/`**: One CSV per chunk with columns:
    *   `time_sec`
    *   `{ROI}_uv_raw`
    *   `{ROI}_sig_raw`
    *   `{ROI}_uv_fit`
    *   `{ROI}_deltaF`
    *   `{ROI}_dff`

*   **`features/`**:
    *   `features.csv`: Per-chunk, per-ROI summary including mean, median, standard deviation, MAD, peak_count, AUC.

*   **`qc/qc_summary.json`**:
    *   Quality control summary including failed chunks, invalid baseline ROIs, baseline invalid ROI counts, chunk failure fraction.

*   **`run_metadata.json`**:
    *   Machine-readable record of configuration values, baseline method, F0 values, invalid baseline ROIs.

*   **`run_report.json`**:
    *   Human-readable report describing derived settings, strictness assumptions, warnings such as Nyquist violations or invalid baselines.

*   **`viz/`**:
    *   Automatically generated diagnostic plots for raw signals, correction impact, and continuous multi-chunk traces.

## Safety and Failure Philosophy
This pipeline is intentionally conservative.

It will fail or warn if:
*   timestamps are non-monotonic or non-finite
*   baselines are invalid
*   regression windows are ill-posed
*   percentile thresholds cannot be computed safely

If outputs are produced, they satisfy the declared analytical contract.

### Tonic/Phasic Analysis Tool

To run the pipeline and generate paper-ready plots separating Tonic (baseline trend) and Phasic (fast event) components:

```bash
python tools/run_cli_and_make_tonic_phasic_plots.py \
  --input "C:/Path/To/Data" \
  --out "C:/Path/To/Output" \
  --config config.yaml \
  --tonic-percentile 5.0 --phasic-highpass-hz 0.01
```

**Outputs:**
- Generates `tonic_{ROI}.png` and `phasic_{ROI}.png` in `{output_dir}/paper_plots`.
- Saves reproducible parameters in `plot_params_{ROI}.json`.
- Runs the standard pipeline first, then post-processes the traces.
