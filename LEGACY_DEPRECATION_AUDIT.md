# Legacy / Deprecation Audit

Date: 2026-03-13

## Scope

This audit is anchored to the current GUI contract:

- GUI calls only `tools/run_full_pipeline_deliverables.py`.
- Deliverables path is cache-based (`*_trace_cache.h5`) + manifest packaging.

Primary evidence:

- `gui/README.md`
- `gui/process_runner.py`
- `tools/run_full_pipeline_deliverables.py`

## Active (Do Not Deprecate)

These are on the live GUI-deliverables execution path:

- `tools/run_full_pipeline_deliverables.py`
- `analyze_photometry.py`
- `photometry_pipeline/pipeline.py`
- `photometry_pipeline/io/hdf5_cache.py`
- `photometry_pipeline/io/hdf5_cache_reader.py`
- `tools/plot_tonic_48h.py`
- `tools/plot_phasic_correction_impact.py`
- `tools/plot_phasic_time_series_summary.py`
- `tools/plot_phasic_dayplot_bundle.py`
- `photometry_pipeline/viz/phasic_data_prep.py` (used by dayplot bundle)

## Deprecate Candidates (High Confidence)

These are not in the GUI path and have no meaningful runtime callers in the current deliverables flow.

1. `tools/DEPRECATED_run_cli_and_make_tonic_phasic_plots.py`
   - Already explicitly named `DEPRECATED_`.
2. `tools/plot_multi_roi_tonic_verification.py`
   - No in-repo callers found.
3. `tools/plot_tonic_verification_panel.py`
   - No in-repo callers found.
4. `tools/plot_verification_panel.old.py`
   - Explicit `.old` legacy file.
5. `tools/run_tonic_dff_demo.py`
   - Standalone demo chain; no runtime callers.
6. `tools/plot_tonic_dff_panel.py`
   - Only called by `tools/run_tonic_dff_demo.py`.
7. `tools/plot_verification_panel.py`
   - Standalone panel generator; not called by GUI flow.
8. `tools/verify_paper_alignment.py`
   - Enforces legacy `traces/` and `viz/` expectations; not GUI-deliverables contract aligned.

## Verification-Only (Non-GUI Runtime)

These are useful diagnostics, but not part of live GUI deliverables. They should be marked as verification utilities (or moved under `tools/legacy/verification/` if desired).

1. `tools/plot_phasic_qc_grid.py`
   - Called by `tools/run_task_c_smoke.py`, `tools/run_biological_synth_verification.py`, migration tests.
2. `tools/plot_session_grid.py`
   - Not called by wrapper; covered by migration tests only.
3. `tools/plot_phasic_stacked.py`
   - Verification usage + migration test coverage.
4. `tools/plot_phasic_intermediate_chain.py`
   - Verification-chain tests only.
5. `tools/plot_phasic_stacked_day_smoothed.py`
   - Not used by wrapper; only mentioned in helper docs/tests.
6. `tools/run_biological_synth_verification.py`
   - Standalone verification orchestration.
7. `tools/run_task_c_smoke.py`
   - Standalone smoke orchestration.
8. `tools/plot_raw_stitched.py`
   - Used by synthetic/verification scripts and tests, not GUI deliverables.

## Module-Level Legacy Candidate

1. `photometry_pipeline/viz/plots.py`
   - Legacy trace/viz plot API.
   - Referenced by `tests/test_reporting_and_viz.py` but not by the deliverables wrapper path.
2. `photometry_pipeline/pipeline.py` imports `from .viz import plots` in two locations without active use in current run path.
   - Candidate cleanup after confirming no hidden side effects are required.

## Docs / Contract Drift To Address

1. `README.md` still documents `traces/` and `viz/` outputs as first-class outputs.
2. Several script headers mention `traces/chunk_*.csv` even where implementation is now cache-backed.

These should be relabeled as `legacy` or updated to prevent accidental reactivation of old contracts.

## Recommended Marking Strategy

Phase 1 (non-breaking, immediate):

1. Add a standard header tag to candidate scripts:
   - `# LEGACY: Not used by GUI deliverables path`
   - `# STATUS: verification-only` or `# STATUS: deprecated`
2. Update docstrings/usage text to indicate legacy status.
3. Add a short section in `README.md`:
   - "Legacy / Verification Tools (not used by GUI)"

Phase 2 (structural cleanup):

1. Move deprecated scripts to `tools/legacy/`.
2. Move verification-only scripts to `tools/verification/`.
3. Keep compatibility shims (thin wrappers) for one transition cycle if external users may call old paths.

## Suggested Next Cleanup Pass

1. Mark + relabel all "High Confidence" candidates now.
2. Mark "Verification-Only" scripts as non-GUI and keep them runnable.
3. Remove or gate legacy `viz/plots.py` imports in `pipeline.py` if no hidden dependence is confirmed.
