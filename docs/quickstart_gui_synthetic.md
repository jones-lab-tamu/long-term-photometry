# Guided Quickstart with Synthetic CSV Data

New user? Start with the app's generated Guided CSV demo.

This tutorial is for a first-time user who wants to generate an example recording,
complete Guided setup, run it, and inspect Results.

The generated files are synthetic workflow-demonstration data, not real
biological data and not biological validation.

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

## 2. Generate and select the demo

1. Choose `Tools -> Generate Guided Demo Dataset`.
2. Select a destination folder.
3. Wait for `long_term_photometry_guided_demo` to be created.
4. Open Guided Workflow and start a new analysis.
5. On Select data, choose the generated recording folder.
6. Follow its short `README.md` and the ordinary Guided prompts.

The demo contains 48 CSV files, one per session, at 20 Hz with two mapped ROIs.
It does not require a custom config or any Full Control setup.

## 3. Complete ordinary Guided choices

Confirm filename order, select `ElapsedSeconds` in seconds, and map:

- ROI1: `ROI1_Signal` with `ROI1_Reference`
- ROI2: `ROI2_Signal` with `ROI2_Reference`

Use intermittent acquisition, 2 sessions/hour, and 600-second sessions. The
README also provides one fixed-daily-anchor example for illustrative Day Plots.
Inspect correction and Feature Detection previews for both ROIs before running.

## 4. Validate before running

Use Guided `Check my setup`.

Validation checks that the input files, selected format, timing structure, ROI/channel pairing, and configuration are internally consistent. A validation failure means the run should not be interpreted until the path, format, config, or acquisition settings are corrected.

## 5. Run the analysis

Use the Guided Run step and select an output destination when prompted.

The run should produce:
- `status.json`
- `MANIFEST.json`
- `run_report.json`
- `events.ndjson`
- `_analysis/phasic_out/`
- `_analysis/tonic_out/`
- one output folder for each generated ROI: `ROI1/` and `ROI2/`

## 6. Open Results

Click `Open Results...` after the run completes. If starting fresh, use `Open Results...` and select the completed output folder.

Inspect these outputs first:
- run status / verification summary
- `Summary` plots for tonic overview and phasic time-series summaries
- `Day Plots` for signal/reference, dynamic fit, corrected dF/F, and stacked views
- `Tables` for event-rate, AUC, and trace-summary CSV outputs

Correction-quality plots are important. They show whether the isosbestic/reference fit is plausible before interpreting event tables.
Correction inspection outputs also include per-chunk dynamic-fit QC metrics under the phasic analysis `qc/` folder.

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

- App cannot find input files: select the generated
  `long_term_photometry_guided_demo` folder, not its parent.
- Wrong format selected: choose CSV files (one file per session), or use automatic detection.
- Column mapping wrong: use the signal/reference names listed above and in the generated README.
- Validation fails: re-check Input Directory, Config, Format, and Sessions per hour.
- Only part of a continuous trace is visible: this is expected for long recordings; use continuous summary and overview outputs.
- Correction fit looks wrong: inspect signal/reference and dynamic-fit plots before changing event thresholds.
- Large events distort fit: try robust/event-gated dynamic fit settings or correction retuning on representative traces.
- Negative slope warning appears: inspect correction plots; the nonnegative reference-coupling diagnostic is optional, advanced, and should be reported if used.
- Need logs/status: check `status.json`, `events.ndjson` when enabled, `MANIFEST.json`, and `run_report.json` in the run directory.
- Batch row failed: open `batch_manifest.csv` / `batch_manifest.json` in the batch output root and inspect the failed row output folder.
