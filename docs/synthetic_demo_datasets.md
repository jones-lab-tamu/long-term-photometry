# Synthetic Demo and Regression Datasets

This project uses `tools/synth_photometry_dataset.py` for both regression fixtures and tutorial datasets.

For a first GUI walkthrough, users do not need to run the generator. A small
bundled RWD-style dataset is committed at `examples/data/synthetic_photometry_basic/`
and documented in `docs/quickstart_gui_synthetic.md`.

## Dataset classes
- clean regression fixtures: deterministic happy-path checks
- realism-stress regression fixtures: biologically plausible data with timing/session irregularities
- tutorial/demo datasets: user-facing examples with mild irregularities suitable for guided walkthroughs

All commands below write `generation_manifest.yaml` by default. The manifest records generator settings, random seed, session structure, injected anomalies, timing summaries, and channel support summaries.

## Shared notes
- Run commands from repo root.
- Use `tests/test_config.yaml` unless a different config is explicitly required.
- For wrapper runs, pass `--sessions-per-hour` to match generation.
- For strict tabular conversion datasets, use `docs/custom_tabular_conversion_guide.md` and `examples/custom_tabular/`.

## A) Clean RWD fixture
```powershell
python tools/synth_photometry_dataset.py --out tests/out_fixture_clean_rwd --format rwd --config tests/test_config.yaml --preset biological_shared_nuisance --total-days 1 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 3 --start-iso 2025-01-01T00:00:00 --seed 101
```

Expected validation behavior:
- should pass validation/tonic/phasic/full deliverables

## B) RWD realism-stress fixture
```powershell
python tools/synth_photometry_dataset.py --out tests/out_fixture_stress_rwd --format rwd --config tests/test_config.yaml --preset realism_stress --total-days 1 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 3 --start-iso 2025-01-01T00:00:00 --start-iso-random-offset-min 30 210 --seed 102
```

Expected validation behavior:
- should remain pipeline-usable while documenting injected timing/session irregularities
- exact injected anomalies are seed-dependent and are recorded in `generation_manifest.yaml`

## C) Clean NPM fixture
```powershell
python tools/synth_photometry_dataset.py --out tests/out_fixture_clean_npm --format npm --config tests/test_config.yaml --preset biological_shared_nuisance --total-days 1 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 3 --start-iso 2025-01-01T00:00:00 --seed 103
```

Expected behavior:
- should pass validation and the currently supported NPM analysis workflow

## D) NPM realism-stress fixture
```powershell
python tools/synth_photometry_dataset.py --out tests/out_fixture_stress_npm --format npm --config tests/test_config.yaml --preset realism_stress --total-days 1 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 3 --start-iso 2025-01-01T00:00:00 --start-iso-random-offset-min 25 180 --seed 104
```

Expected stress behavior:
- should exercise UV/SIG asymmetry, timestamp offset/jitter, support mismatch, and optional startup rows
- should remain usable for supported NPM regression workflow coverage
- exact injected anomalies are seed-dependent and are recorded in `generation_manifest.yaml`

## E) Near-threshold end-coverage fixture
```powershell
python tools/synth_photometry_dataset.py --out tests/out_fixture_near_threshold --format rwd --config tests/test_config.yaml --preset biological_shared_nuisance --total-days 1 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 2 --start-iso 2025-01-01T00:00:00 --near-threshold-end-coverage true --edge-truncate-samples-max 2 --seed 105
```

Goal:
- exercise strict session admission edge cases around end-coverage mismatch tolerances

## F) Non-midnight fixed-anchor fixture
```powershell
python tools/synth_photometry_dataset.py --out tests/out_fixture_non_midnight --format rwd --config tests/test_config.yaml --preset biological_shared_nuisance --total-days 1 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 2 --start-iso 2025-01-01T13:37:11 --start-iso-random-offset-min 17 17 --seed 106
```

Recommended validation command:
```powershell
python tools/run_full_pipeline_deliverables.py --input tests/out_fixture_non_midnight --out tests/out_fixture_non_midnight_validate --config tests/test_config.yaml --format rwd --validate-only --sessions-per-hour 2 --timeline-anchor-mode fixed_daily_anchor --fixed-daily-anchor-clock 06:00:00
```

## G) Dropped/truncated session fixture
```powershell
python tools/synth_photometry_dataset.py --out tests/out_fixture_dropped_truncated --format rwd --config tests/test_config.yaml --preset biological_shared_nuisance --total-days 1 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 2 --start-iso 2025-01-01T00:00:00 --session-drop-prob 0.5 --edge-truncate-samples-max 4 --seed 107
```

Goal:
- exercise manifest/session rejection and sparse/truncated dayplot occupancy paths

## Documentation-ready tutorial dataset command

The tutorial/demo dataset is intended for documentation and GUI walkthroughs. It should be realistic enough to expose users to non-midnight starts, mild missing/truncated sessions, tonic structure, phasic events, and shared nuisance/artifact signals, but not so pathological that first-time users cannot complete the tutorial.

```powershell
python tools/synth_photometry_dataset.py --out example_data/demo_realistic_rwd --format rwd --config tests/test_config.yaml --preset biological_shared_nuisance --total-days 2 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 4 --start-iso 2025-01-03T11:22:00 --start-iso-random-offset-min 20 80 --session-drop-prob 0.04 --edge-truncate-samples-max 0 --timestamp-jitter-ms-std 1.2 --near-threshold-end-coverage true --seed 2026
```

Characteristics:
- non-midnight start
- mild drop/truncation and timestamp irregularity
- tonic + phasic content
- shared nuisance/artifact structure
- multi-ROI sessions
- auditable anomaly manifest

## Continuous synthetic demo datasets

Continuous synthetic datasets use the same signal realism components as the intermittent generator but emit one continuous acquisition source. CI examples are intentionally short; multi-day demo commands use the same code path.

Short custom_tabular CI/regression command:
```powershell
python tools/synth_photometry_dataset.py --out tests/out_continuous_custom_tabular_ci --format custom_tabular --config tests/test_config.yaml --acquisition-mode continuous --preset continuous_realistic --continuous-duration-hours 0.67 --fs-hz 10 --n-rois 2 --start-iso 2025-01-01T13:37:11 --seed 401
```

Short RWD CI/regression command:
```powershell
python tools/synth_photometry_dataset.py --out tests/out_continuous_rwd_ci --format rwd --config tests/test_config.yaml --acquisition-mode continuous --preset continuous_realistic --continuous-duration-hours 0.67 --fs-hz 10 --n-rois 2 --start-iso 2025-01-01T13:37:11 --seed 402
```

Multi-day custom_tabular demo command:
```powershell
python tools/synth_photometry_dataset.py --out example_data/demo_continuous_custom_tabular --format custom_tabular --config tests/test_config.yaml --acquisition-mode continuous --preset continuous_realistic --total-days 3 --fs-hz 10 --n-rois 4 --start-iso 2025-01-03T11:22:00 --seed 2026
```

Multi-day RWD demo command:
```powershell
python tools/synth_photometry_dataset.py --out example_data/demo_continuous_rwd --format rwd --config tests/test_config.yaml --acquisition-mode continuous --preset continuous_realistic --total-days 3 --fs-hz 10 --n-rois 4 --start-iso 2025-01-03T11:22:00 --seed 2027
```

Continuous dataset characteristics:
- `custom_tabular` output is one top-level `continuous_recording.csv`
- RWD output is one timestamped folder containing one `fluorescence.csv`
- `generation_manifest.yaml` records duration, sample count, ROI count, channel columns, event counts, motion artifacts, timestamp jitter, and expected 600 s continuous windows
- NPM/interleaved continuous synthetic generation is intentionally unsupported
- `continuous_realistic` is pipeline-usable synthetic data, not a malformed-file stress fixture

## Manual end-to-end continuous workflow smoke test

Use this sequence when manually checking the generated continuous workflow outside pytest. It creates a short generated custom_tabular continuous dataset, runs validation, then runs the full tonic+phasic workflow with 600 s windows.

```powershell
python tools/synth_photometry_dataset.py --out example_data/continuous_custom_tabular_smoke --format custom_tabular --config tests/test_config.yaml --acquisition-mode continuous --preset continuous_realistic --continuous-duration-hours 0.67 --fs-hz 10 --n-rois 2 --start-iso 2025-01-01T13:37:11 --seed 2026
```

```powershell
python tools/run_full_pipeline_deliverables.py --input example_data/continuous_custom_tabular_smoke --out example_data/continuous_custom_tabular_validate --config tests/test_config.yaml --format custom_tabular --acquisition-mode continuous --continuous-window-sec 600 --continuous-step-sec 600 --validate-only --overwrite
```

Expected validation behavior:
- `VALIDATE-ONLY: OK`
- planned windows should match `generation_manifest.yaml` `continuous_windows.expected_continuous_window_count`

```powershell
python tools/run_full_pipeline_deliverables.py --input example_data/continuous_custom_tabular_smoke --out example_data/continuous_custom_tabular_full --config tests/test_config.yaml --format custom_tabular --mode both --acquisition-mode continuous --continuous-window-sec 600 --continuous-step-sec 600 --overwrite
```

Expected full-run behavior:
- `status.json` and `MANIFEST.json` report success
- `_analysis/phasic_out/phasic_trace_cache.h5` and `_analysis/tonic_out/tonic_trace_cache.h5` are produced
- each ROI has continuous phasic and tonic summary tables under `<roi>/tables/`
- each ROI has elapsed-time phasic and tonic summary plots under `<roi>/summary/`
- each ROI has full cached-trace overview plots under `<roi>/summary/`: `continuous_phasic_dff_trace_overview.png` and `continuous_tonic_trace_overview.png`
- completed-run loading should expose the ROI `Summary` and `Tables` folders

## What these fixtures do not test

These fixtures do not currently simulate:
- malformed CSV rows
- clock resets
- random file loss during a session
- hard-corruption cases
- true multiday vendor hardware clock drift

These failure classes should be covered by separate failure-mode fixtures or sanitized real-data regression tests.
