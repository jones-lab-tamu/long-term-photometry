# Vendor RWD Regression Fixtures

This page is developer-facing, RWD-specific synthetic regression documentation for CLI/GUI parity checks.

Fixture classes on this page:
- clean fixtures: deterministic happy-path baselines
- realism-stress fixtures: pipeline-usable datasets with intentionally irregular timing/session structure

Both fixture types write `generation_manifest.yaml`.
For the full cross-format synthetic dataset guide, including NPM fixtures and tutorial/demo datasets, see `docs/synthetic_demo_datasets.md`.

## 1) Clean RWD fixture
From repo root:

```powershell
python tools/synth_photometry_dataset.py --out tests/out_rwd_clean --format rwd --config tests/test_config.yaml --preset biological_shared_nuisance --total-days 2 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 3 --start-iso 2025-01-01T00:00:00 --seed 42
```

Expected behavior:
- discovery, validation, and the full analysis workflow should complete successfully
- `sessions_dropped.count` should be `0` for this command and seed
- `generation_manifest.yaml` should show no intentional timing/session anomalies beyond the clean baseline settings

## 2) RWD realism-stress fixture
From repo root:

```powershell
python tools/synth_photometry_dataset.py --out tests/out_rwd_realism_stress --format rwd --config tests/test_config.yaml --preset realism_stress --total-days 2 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 3 --start-iso 2025-01-01T00:00:00 --start-iso-random-offset-min 45 210 --seed 42
```

Expected behavior:
- should pass validation and complete the full deliverables workflow
- injected irregularities are intended to exercise timing/session edge paths without producing catastrophic corruption
- `generation_manifest.yaml` records any injected jitter, truncation, dropped sessions, and near-threshold coverage manipulations
- this fixture exercises timing/session code paths that clean fixtures do not
- this fixture is not intended to simulate malformed files or hard-corruption cases

## 3) Validate and run wrapper

Validate-only preflight:

```powershell
python tools/run_full_pipeline_deliverables.py --input tests/out_rwd_realism_stress --out tests/out_rwd_realism_stress_validate --config tests/test_config.yaml --format rwd --validate-only --sessions-per-hour 2
```

Full run:

```powershell
python tools/run_full_pipeline_deliverables.py --input tests/out_rwd_realism_stress --out tests/out_rwd_realism_stress_run --config tests/test_config.yaml --format rwd --mode both --sessions-per-hour 2 --overwrite
```

## 4) Fixed-anchor non-midnight validation example

```powershell
python tools/run_full_pipeline_deliverables.py --input tests/out_rwd_realism_stress --out tests/out_rwd_fixed_anchor_validate --config tests/test_config.yaml --format rwd --validate-only --sessions-per-hour 2 --timeline-anchor-mode fixed_daily_anchor --fixed-daily-anchor-clock 06:00:00
```

## 5) GUI procedure
1. Launch GUI from repo root: `python gui/main.py`
2. Set Input Directory to `tests/out_rwd_clean` or `tests/out_rwd_realism_stress`
3. Set Output Directory to a writable run folder, for example `tests/out_rwd_clean_gui_validation` or `tests/out_rwd_realism_stress_gui_validation`
4. Set Config to `tests/test_config.yaml`
5. Set Format to `rwd` (or `auto`)
6. Run `Validate`, then `Run Pipeline`

Expected GUI outcome:
- GUI execution should produce completed run artifacts comparable to the CLI full-run wrapper, including run reports, analysis outputs, and deliverable plots/tables for the selected profile.

## 6) Fixture intent
- `tests/out_rwd_clean`: deterministic baseline for happy-path regression
- `tests/out_rwd_realism_stress`: regression coverage for real-world timing/session irregularities
