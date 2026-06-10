# GUI Quickstart with Bundled Synthetic Data

New user? Start with this page and the bundled dataset in `examples/data/synthetic_photometry_basic/`.

This tutorial is for a first-time user who wants to open the GUI, run a small example dataset, inspect outputs, and understand the major workflow choices.

The bundled dataset is synthetic RWD-style data. It is for workflow demonstration and software testing, not biological validation.

## 1. Install and launch

Install dependencies from the repository root. See the main `README.md` installation section for environment setup.

Launch the GUI:

```powershell
python -m gui.app
```

If that entry point is unavailable in your environment, use:

```powershell
python gui/main.py
```

## 2. Load the bundled dataset

Use these paths in the GUI:

1. Input Directory: `examples/data/synthetic_photometry_basic`
2. Output Directory: `tutorial_outputs/synthetic_photometry_basic_run`
3. Config: `examples/data/synthetic_photometry_basic/tutorial_config.yaml`
4. Format: `rwd`
5. Mode: `both`, if the GUI exposes tonic/phasic mode selection
6. Sessions per hour: `2`, if that control is visible

The input folder already contains data. You do not need to run the synthetic generator for this quickstart.

## 3. Keep default correction settings

For the first run, use the config provided with the dataset.

Key defaults in this tutorial config:
- dynamic fit mode: `robust_global_event_reject`
- negative-slope constraint: default package behavior is unconstrained unless changed in the GUI/config
- bleach correction: not enabled in this tutorial config
- event threshold: `mean_std` with modest tutorial settings

The optional `Prevent negative slopes` / nonnegative slope constraint can be useful when the fitted reference channel inverts polarity. Treat it as an explicit analysis intervention and report it if used. Do not enable it by default just because a warning appears; inspect correction-quality plots first.

## 4. Validate before running

Click the GUI validation control, usually labeled `Validate` or `Validate Only`.

Expected result for the bundled tutorial dataset:

```text
VALIDATE-ONLY: OK
```

Validation checks that the input files, selected format, timing structure, ROI/channel pairing, and configuration are internally consistent. A validation failure means the run should not be interpreted until the path, format, config, or acquisition settings are corrected.

## 5. Run the analysis

Click `Run Pipeline`.

On a typical laptop this small dataset should finish in under one minute. Outputs are written under:

```text
tutorial_outputs/synthetic_photometry_basic_run
```

The run should produce:
- `status.json`
- `MANIFEST.json`
- `run_report.json`
- `events.ndjson`
- `_analysis/phasic_out/`
- `_analysis/tonic_out/`
- one output folder per detected ROI/channel; this bundled dataset produces `CH1/` and `CH2/`

## 6. Open Results

Click `Open Results...` after the run completes. If starting fresh, use `Open Results...` and select the completed output folder.

Inspect these outputs first:
- run status / verification summary
- `Summary` plots for tonic overview and phasic time-series summaries
- `Day Plots` for signal/reference, dynamic fit, corrected dF/F, and stacked views
- `Tables` for event-rate, AUC, and trace-summary CSV outputs

Correction-quality plots are important. They show whether the isosbestic/reference fit is plausible before interpreting event tables.

## 7. Export and reuse outputs

The GUI writes ordinary files that can be inspected outside the app.

Common locations:
- figures: `<run_dir>/<ROI>/summary/` and `<run_dir>/<ROI>/day_plots/`
- tables: `<run_dir>/<ROI>/tables/`
- phasic features: `<run_dir>/_analysis/phasic_out/features/features.csv`
- analysis configs: `<run_dir>/_analysis/phasic_out/config_used.yaml` and `<run_dir>/_analysis/tonic_out/config_used.yaml`
- GUI launch provenance, when launched from the GUI: `config_effective.yaml`, `gui_run_spec.json`, and `command_invoked.txt`
- cache/provenance: `<run_dir>/_analysis/*/`, `events.ndjson`, `status.json`, `MANIFEST.json`, and `run_report.json`

Downstream statistics, group comparisons, and publication-specific modeling should be performed outside this GUI unless a separate repository workflow explicitly implements them.

## 8. Optional: continuous recordings

Continuous recordings use `acquisition_mode=continuous` and are processed in fixed elapsed-time windows. Supported continuous paths currently include RWD and strict `custom_tabular`; NPM/interleaved continuous mode is not currently implemented.

For long recordings, the GUI may show summary plots, tables, and downsampled full-trace overview plots instead of rendering every raw point at once. Use summary plots to locate time ranges of interest, then inspect detailed per-window outputs and cached traces as needed.

See `docs/continuous_recordings.md` and `docs/synthetic_demo_datasets.md` for commands that generate continuous examples.

## 9. Optional: batch mode

Batch mode treats each immediate subfolder of a selected batch input root as one independent dataset. It applies one shared configuration to each dataset and writes one normal completed-run output per dataset plus batch manifests.

Batch mode does not perform group statistics, averaging, or simultaneous multi-recording visualization.

See `docs/batch_processing.md`.

## 10. Troubleshooting

- App cannot find input files: confirm Input Directory is `examples/data/synthetic_photometry_basic`, not one of the individual session subfolders.
- Wrong format selected: use `rwd` for this bundled quickstart dataset.
- Column mapping wrong: for RWD, the tutorial config expects `TimeStamp`, `-410`, and `-470` channel naming.
- Validation fails: re-check Input Directory, Config, Format, and Sessions per hour.
- Only part of a continuous trace is visible: this is expected for long recordings; use continuous summary and overview outputs.
- Correction fit looks wrong: inspect signal/reference and dynamic-fit plots before changing event thresholds.
- Large events distort fit: try robust/event-gated dynamic fit settings or correction retuning on representative traces.
- Negative slope warning appears: inspect correction plots; nonnegative slope constraint is optional and should be reported if used.
- Need logs/status: check `status.json`, `events.ndjson` when enabled, `MANIFEST.json`, and `run_report.json` in the run directory.
- Batch row failed: open `batch_manifest.csv` / `batch_manifest.json` in the batch output root and inspect the failed row output folder.
