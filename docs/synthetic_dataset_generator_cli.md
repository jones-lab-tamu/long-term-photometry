# Synthetic Dataset Generator CLI

Synthetic datasets in this repository are for workflow demonstration, software testing, and regression coverage. They are not biological validation datasets.

## 1. What the generator is for

`tools/synth_photometry_dataset.py` creates deterministic synthetic photometry-like datasets for tutorials, GUI smoke checks, and regression tests. It can emit RWD-style, NPM-style, and supported `custom_tabular` synthetic outputs depending on acquisition mode and format.

## 2. GUI presets versus full CLI control

Normal GUI users should start with `Tools -> Generate Synthetic Demo Dataset`. The GUI exposes only two curated presets:
- Fast quickstart demo: copies the committed bundled dataset.
- Long-duration intermittent demo: generates a fixed 48 h RWD-style dataset with 10 min sessions, 2 sessions/hour, 10 Hz sampling, and 2 ROIs.

Developers and power users can use the CLI generator directly when they need parameter control.

## 3. Minimal small RWD demo command

```powershell
python tools/synth_photometry_dataset.py --out scratch/small_rwd_demo --format rwd --config examples/data/synthetic_photometry_basic/tutorial_config.yaml --preset biological_shared_nuisance --total-days 0.05 --recording-duration-min 2 --recordings-per-hour 2 --fs-hz 10 --n-rois 2 --start-iso 2025-01-03T11:22:00 --seed 2026 --phasic-min-events-per-chunk 3 --artifact-enable-motion --artifact-motion-min-per-day 1 --artifact-motion-rate-per-day 20
```

## 4. Long-duration intermittent demo command

Recommended one-command path:

```powershell
python examples/generate_long_duration_demo.py
```

This wrapper writes the matching `tutorial_config.yaml`, runs the curated long-duration generator command, and prints the input folder, config path, and recommended GUI settings.

Advanced equivalent command:

```powershell
python tools/synth_photometry_dataset.py --out example_data/synthetic_long_duration_demo --format rwd --config example_data/synthetic_long_duration_demo/tutorial_config.yaml --preset biological_shared_nuisance --total-days 2 --recording-duration-min 10 --recordings-per-hour 2 --acquisition-mode intermittent --fs-hz 10 --n-rois 2 --start-iso 2025-01-03T11:22:00 --seed 2026 --phasic-min-events-per-chunk 3 --artifact-enable-motion --artifact-motion-min-per-day 1 --artifact-motion-rate-per-day 20
```

The raw generator command assumes that the config file passed to `--config` already exists and is valid. If you want automatic config writing, use `examples/generate_long_duration_demo.py`.

## 5. Continuous demo command

Continuous generation is currently supported for RWD and strict `custom_tabular`. NPM/interleaved continuous generation is intentionally unsupported.

```powershell
python tools/synth_photometry_dataset.py --out example_data/demo_continuous_custom_tabular --format custom_tabular --config tests/test_config.yaml --acquisition-mode continuous --preset continuous_realistic --continuous-duration-hours 0.67 --fs-hz 10 --n-rois 2 --start-iso 2025-01-01T13:37:11 --seed 2026
```

## 6. NPM-style demo command

```powershell
python tools/synth_photometry_dataset.py --out scratch/npm_demo --format npm --config tests/test_config.yaml --preset biological_shared_nuisance --total-days 0.05 --recording-duration-min 10 --recordings-per-hour 2 --fs-hz 10 --n-rois 2 --start-iso 2025-01-03T11:22:00 --seed 2026
```

## 7. custom_tabular demo command

Intermittent `custom_tabular` synthetic generation is not currently supported. Use RWD/NPM for intermittent demos or continuous `custom_tabular` for continuous-mode demos.

```powershell
python tools/synth_photometry_dataset.py --out scratch/continuous_custom_tabular_demo --format custom_tabular --config tests/test_config.yaml --acquisition-mode continuous --preset continuous_realistic --continuous-duration-hours 0.67 --fs-hz 10 --n-rois 2 --start-iso 2025-01-01T13:37:11 --seed 2026
```

## 8. Parameter groups

Key parameter groups are:
- duration/session structure: `--total-days`, `--recording-duration-min`, `--recordings-per-hour`, `--acquisition-mode`, `--continuous-duration-hours`
- sampling rate/ROI count: `--fs-hz`, `--n-rois`
- phasic events: `--phasic-*` options
- drift/bleach/shared nuisance: artifact drift, bleach, and shared wobble options
- motion artifacts: `--artifact-enable-motion` and motion-rate/amplitude/timing options
- timing irregularities/dropped sessions: `--session-drop-prob`, timestamp jitter, edge truncation, and near-threshold coverage options
- output format: `--format rwd`, `--format npm`, or supported `--format custom_tabular`
- random seed: `--seed`

Use `python tools/synth_photometry_dataset.py --help` for the full flag list.

## 9. Troubleshooting

- Output folder exists: GUI presets ask before replacing. CLI users should choose a fresh folder or delete the old output intentionally.
- Config missing: the generator requires a valid YAML config. The GUI long-duration preset and `examples/generate_long_duration_demo.py` write one automatically.
- Validation fails: confirm `--format`, `--sessions-per-hour`, and config channel naming match the generated output.
- Dataset too large/slow: reduce `--total-days`, `--fs-hz`, or `--n-rois` for testing.
- Synthetic data are not biological validation: use them to learn and test the software workflow, not to validate sensor biology or experimental conclusions.
