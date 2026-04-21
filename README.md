# Long-Term Photometry Analysis App

## What This Software Is
This repository provides a GUI-first workflow for long-term fiber photometry analysis, review, and reproducible output generation.

Core workflow:
- ingest session data (`rwd`, `npm`, or strict `custom_tabular`)
- run phasic and/or tonic analysis
- generate run artifacts (reports, caches, plots, tables)
- load completed runs for inspection
- run bounded post-run retuning (downstream and correction-sensitive)

The GUI is the primary user surface (`python -m gui.app`). Runner scripts and CLI tools provide the execution backend and support advanced or scripted workflows.

## Key Capabilities
- Validate-before-run workflow in the GUI, with consistency checks to ensure runs reflect the validated configuration.
- Full analysis runs and reduced preparation runs for post-run retuning workflows.
- Completed-run loading for inspection, review, and bounded post-run refinement.
- Separate downstream and correction-sensitive retuning workflows.
- Multiple dynamic fitting strategies for phasic correction, including rolling, global, robust, and adaptive modes.
- Configurable event-detection polarity for positive, negative, or bidirectional signal excursions.
- Optional bleach correction modes for traces with pronounced monotonic decay.
- Reproducible run outputs with explicit provenance, configuration snapshots, and retune records.
- Strict support for exported tabular data through a contract-based `custom_tabular` import mode.
- Optional export and rerender tools for users who want to inspect or restyle generated figures outside the default outputs.

## What This Software Is Not
- Not an acquisition system or vendor control interface.
- Not a universal file reader for all proprietary photometry formats.
- Not a general-purpose time-series analysis framework for arbitrary continuous signals.
- Not a heuristic CSV importer that tries to infer structure from messy files.

## Installation
From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

For the GUI workflow (recommended):

```bash
pip install -r requirements_gui.txt
```

For CLI/backend-only use:

```bash
pip install -r requirements.txt
```

Notes:

- requirements_gui.txt installs the core analysis dependencies plus the GUI dependencies.
- requirements.txt installs the analysis/backend dependencies only.
- The GUI is the primary user workflow; the CLI is mainly for backend execution and advanced scripted use.

## Running the App

### GUI (recommended)
Launch the GUI from the repository root:

```bash
python -m gui.app
```

### CLI (advanced)
For advanced or scripted workflows, the main orchestration entry point is:
```bash
python tools/run_full_pipeline_deliverables.py ^
  --input <INPUT_DIR> ^
  --out <RUN_DIR> ^
  --config <CONFIG_YAML> ^
  --format <auto|rwd|npm|custom_tabular> ^
  --mode <both|phasic|tonic> ^
  --run-type <full|tuning_prep> ^
  --sessions-per-hour <INT>
```

Notes:
- `--sessions-per-hour` is usually needed for packaged runs, because the output layout depends on explicit session scheduling.
- `--validate-only` checks inputs and configuration without running the analysis.
- `--discover` reports what the software finds in the input path, then exits without creating run artifacts.

## Supported Input Formats
The software currently supports four input-format modes:
- `auto`
- `rwd`
- `npm`
- `custom_tabular`

### `auto`
`auto` detection is conservative. It currently detects `rwd` and `npm` patterns only. It does not detect `custom_tabular`; use `--format custom_tabular` when importing strict tabular data.

### `rwd`
- For data recorded using RWD Life Science's photometry system.
- Expects RWD session folders containing `fluorescence.csv`.
- Loader validates time/channel structure and applies strict grid resampling.

### `npm`
- For data recorded using Neurophotometrics' photometry system.
- Supports CSV-based NPM inputs with explicit LED/time/channel expectations.
- UV/SIG support windows are validated before strict interpolation.

### `custom_tabular (strict)`
This mode is intended for already-exported tabular data rather than native proprietary acquisition files.

Contract:
- one CSV file = one session
- one exact time column
- paired ROI columns for isosbestic and signal channels
- each ROI base must have both channels in the same file
- time must be numeric, finite, and strictly increasing
- channel values must be numeric and finite
- no heuristic guessing
- no sidecar mapping wizard

Default naming contract:
- time column: `time_sec`
- isosbestic suffix: `_iso`
- signal suffix: `_sig`

Minimal example header:

```csv
time_sec,Region0_iso,Region0_sig,Region1_iso,Region1_sig
```

If you need different names, the contract can be adjusted through configuration using:
- `custom_tabular_time_col`
- `custom_tabular_uv_suffix`
- `custom_tabular_sig_suffix`

## Core Workflows

### Validate Before Run
Use the GUI's **Validate Only** step to check inputs and configuration before launching analysis. The GUI tracks whether the active configuration still matches the last successful validation and will require revalidation if key settings have changed.

### Full Run
A full run executes the selected tonic and/or phasic analysis workflow and writes the standard run artifacts, including internal analysis outputs and per-ROI summaries, plots, and tables.

### Tuning Prep Run
A `tuning_prep` run is a reduced run mode intended for post-run refinement workflows. It skips nonessential production outputs while preserving the artifacts needed for completed-run loading, downstream retune, and correction retune. Because these workflows depend on phasic artifacts, tonic-only mode is not supported for `tuning_prep`.

### Load Completed Run
The GUI can reopen successful completed runs for later inspection and post-run refinement. This requires the normal success/provenance artifacts produced by a finished run.

### Downstream Retune
Downstream retune recomputes event-facing outputs from cached phasic results without rerunning the full analysis. This is useful when adjusting event-detection-side settings after a run has already completed. Retune outputs are written to a dedicated subdirectory under the original run.

### Correction Retune
Correction retune revisits correction-sensitive settings and recomputes the relevant cached correction outputs for a selected ROI without rerunning the full dataset from scratch. Correction-retune outputs are written to a dedicated subdirectory under the original run.

## Important Options
- **Dynamic fit mode**: selects how the phasic correction reference is estimated, including rolling, global, robust, and adaptive strategies.
- **Bleach correction mode**: optional correction-stage detrending for traces with pronounced monotonic decay (`none`, `single_exponential`, `double_exponential`).
- **Signal excursion polarity**: controls whether event detection looks for positive, negative, or bidirectional signal excursions.
- **Event signal**: controls whether events are detected from `dff` or `delta_f`.
- **Timeline anchor mode**: controls how sessions are aligned in time (`civil`, `elapsed`, or `fixed_daily_anchor`).
- **Baseline subtraction before fit**: optional rolling-mode preprocessing step to reduce slow baseline influence on local fit estimation.

## Typical Output Layout
A typical run directory includes:

```text
<run_dir>/
  status.json
  MANIFEST.json
  run_report.json
  config_effective.yaml          # GUI-generated runner config
  gui_run_spec.json              # GUI intent record (GUI launches)
  command_invoked.txt            # GUI launches
  _analysis/
    phasic_out/
      config_used.yaml
      run_report.json
      phasic_trace_cache.h5
      features/features.csv
      qc/qc_summary.json
    tonic_out/
      config_used.yaml
      run_report.json
      tonic_trace_cache.h5
  <ROI_NAME>/
    summary/
    day_plots/
    tables/
```

Retune workflows write their results into dedicated subdirectories under the original run directory. These typically include:
- `retune_config_effective.yaml`
- `retune_request.json`
- `retune_result.json`
- retuned CSV/PNG/H5 outputs (path-specific to downstream vs correction retune)

## Troubleshooting
- **Tabular CSVs are not detected by `auto` format selection.** Use `--format custom_tabular` and make sure the file satisfies the strict tabular import contract.
- **A run is blocked after a previous validation.** The active settings no longer match the last successful validation. Re-run **Validate Only** in the GUI before launching the run.
- **Retune is unavailable in the GUI.** Load a successful completed run first and make sure the required phasic cache artifacts are present.
- **A retune command rejects one or more keys.** Downstream retune and correction retune support different override sets, so use the retune workflow that matches the setting you are trying to change.
- **dF/F dayplot rerender is unavailable.** Rerendering requires a valid completed-run context, the relevant phasic cache/config artifacts, ROI selection, and session scheduling metadata.

## Repository Orientation
For readers who want a quick high-level map of the codebase:

- `gui/`: primary desktop UI, including run setup, completed-run loading, retune controls, and display tools.
- `tools/run_full_pipeline_deliverables.py`: main orchestration entry point for validation, discovery, full runs, and `tuning_prep` runs.
- `analyze_photometry.py`: core single-pass analysis entry point.
- `photometry_pipeline/config.py`: config schema and validation rules.
- `photometry_pipeline/io/adapters.py`: format-specific ingestion and strict import contracts.
- `photometry_pipeline/discovery.py`: dataset/session discovery logic.
- `photometry_pipeline/pipeline.py`: core analysis flow, reporting, and cache generation.
- `photometry_pipeline/tuning/`: downstream and correction retune backends.
- `tools/plot_*`: plot-generation and display-artifact utilities.