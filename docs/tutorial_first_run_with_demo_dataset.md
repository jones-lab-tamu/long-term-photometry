# Tutorial: First Run with a Demo Dataset

## A) Purpose of this tutorial

This tutorial uses a synthetic but realistic RWD demo dataset to teach the workflow end to end.
- The dataset includes mild real-world irregularities.
- It is designed to remain usable for a first run.
- The goal is to learn the software workflow, not to make a biological conclusion.

## B) Generate the demo dataset

Run this from the repo root:

```powershell
python tools/synth_photometry_dataset.py --out example_data/demo_realistic_rwd --format rwd --config tests/test_config.yaml --preset biological_shared_nuisance --total-days 2 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 4 --start-iso 2025-01-03T11:22:00 --start-iso-random-offset-min 20 80 --session-drop-prob 0.04 --edge-truncate-samples-max 0 --timestamp-jitter-ms-std 1.2 --near-threshold-end-coverage true --seed 2026
```

## C) Inspect `generation_manifest.yaml`

After generation, open:
- `example_data/demo_realistic_rwd/generation_manifest.yaml`

Focus on these fields first:
1. `seed`
2. `start_time`
3. `sessions_requested`, `sessions_generated`, `sessions_dropped`
4. `sessions_truncated`
5. `timestamp_jitter`
6. `near_threshold_coverage`

Use the manifest as the source of truth for exactly which irregularities were injected in this run.

## D) Launch the GUI

```powershell
python gui/main.py
```

## E) Configure the run in the GUI

1. Set Input Directory to `example_data/demo_realistic_rwd`.
2. Set Output Directory to `tutorial_outputs/demo_realistic_rwd_run`.
3. Set Config to `tests/test_config.yaml`.
4. Set Format to `rwd` (or `auto`).
5. Confirm sessions per hour is `2` if that control is visible/applicable.

## F) Validate

1. Click `Validate`.
2. Validation should complete without fatal errors.
3. Expected result for this tutorial command: `VALIDATE-ONLY: OK`.
4. Mild irregularities may appear in logs/metadata, but this dataset is intended to remain usable.
5. If validation fails, re-check input path, config path, format, and sessions-per-hour.

## G) Run the pipeline

1. Click `Run Pipeline`.
2. The run should proceed through validation, tonic analysis, phasic analysis, and plotting.
3. Outputs are written to the selected output directory.

## H) Open and inspect completed outputs

Open the completed run and inspect:
1. Verification tab: quick overview of run status and generated deliverables.
2. Tonic outputs: slow baseline trend summaries over time.
3. Phasic signal/isosbestic views: raw signal and reference-channel behavior for artifact context.
4. Dynamic fit: how isosbestic-based correction was fit over each session.
5. Corrected phasic dF/F: event-scale signal after correction.
6. Stacked phasic traces: day/session-aligned visual comparison across windows.
7. Phasic summaries: tabular and plotted event-rate/AUC summaries for quick tuning feedback.

## I) Try correction retuning

Correction retuning is for evaluating correction/fitting assumptions.
- Tune on representative traces, not every segment.
- Save/apply settings programmatically for consistency.
- Once selected, settings should be applied consistently across the relevant dataset or reused through a saved configuration.
- The preview trace is for inspection only, not manual trace-by-trace correction.

Practical GUI flow:
1. Open the tuning workspace in the main GUI window.
2. Select the completed run directory and choose an ROI.
3. Set correction-related retune parameters in the retune controls panel.
4. Run a retune preview on representative sessions.
5. Review retuned diagnostics and exported artifacts.

## J) Try downstream event reanalysis

Downstream event reanalysis tunes event detection after correction.
- It does not recompute upstream correction.
- Use it after inspecting corrected phasic traces.
- Tune on representative examples, then apply consistently.

Practical GUI flow:
1. Open downstream/event retune controls in the tuning workspace.
2. Select event-threshold and event-feature settings.
3. Run reanalysis and compare summaries/plots.
4. Save the chosen settings.

## K) Save or reuse a custom configuration

Saving configs helps standardize analysis across datasets.
- Save the effective config used in accepted runs.
- Archive/report config files with results for reproducibility.

## L) Locate exported outputs

Look in your run output directory (for example `tutorial_outputs/demo_realistic_rwd_run`) for:
1. corrected traces
2. event tables/summaries
3. plots
4. run reports
5. configs/provenance files

Synthetic input provenance is in:
- `example_data/demo_realistic_rwd/generation_manifest.yaml`

Exact filenames may vary by run profile. Use the run output directory and run report as the index.

## M) Troubleshooting

- GUI cannot find input directory:
  Check the path exists and points to `example_data/demo_realistic_rwd`.
- Wrong format selected:
  Use `rwd` or `auto` for this tutorial dataset.
- Validation fails:
  Re-check config path, format, and sessions-per-hour.
- Output directory is not writable:
  Choose a user-writable path (for example under `tutorial_outputs/`).
- Sessions-per-hour mismatch:
  Use `2` to match the generation command.
- Fixed-anchor alignment looks wrong:
  Confirm timeline anchor settings and clock value in the run options.
- No events detected or too many events detected:
  Use downstream reanalysis to adjust event thresholds consistently.
- Correction fit looks poor:
  Use correction retune on representative traces, save settings, and rerun.

## N) Interpretation warning

This synthetic dataset is for learning the software workflow and testing analysis behavior. It should not be interpreted as biological data.
