# Vendor-Faithful Synthetic RWD Regression Fixture

This fixture is the reproducible baseline for CLI/GUI parity checks on the RWD path.

## 1) Generate vendor-faithful synthetic RWD
From repo root:

```powershell
python tools/synth_photometry_dataset.py --out tests/out_legacy_rwd --format rwd --config tests/test_config.yaml --preset biological_shared_nuisance --total-days 5 --recording-duration-min 10 --recordings-per-hour 2 --n-rois 5 --start-iso 2025-01-01T00:00:00 --seed 42
```

## 2) CLI validation command
From repo root:

```powershell
python tools/run_full_pipeline_deliverables.py --input tests/out_legacy_rwd --out tests/out_legacy_rwd_cli_validation --config tests/test_config.yaml --format rwd --mode both --events auto --cancel-flag auto
```

## 3) Minimal GUI validation procedure
1. Launch GUI from repo root (`python gui/main.py`).
2. Set Input Directory to `tests/out_legacy_rwd`.
3. Set Output Directory to a writable folder (for example `tests/out_legacy_rwd_gui_validation`).
4. Set Config to `tests/test_config.yaml`.
5. Set Format to `rwd` (or `auto`; both should resolve correctly now).
6. Click `Validate`, then `Run Pipeline`.

## 4) Expected passing behavior
- Discovery succeeds and resolves RWD sessions/ROIs.
- Validate completes without strict end-coverage mismatch errors.
- Run Pipeline completes and writes standard run artifacts.

## 5) Fixture location and regeneration
- Fixture root: `tests/out_legacy_rwd`
- Regenerate by re-running command in section (1).

## 6) Intended use
This fixture is a regression asset for CLI/GUI parity on vendor-faithful RWD structure.
