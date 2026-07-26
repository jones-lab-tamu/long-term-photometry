# Known Baseline Test Failures

A listed failure does not block an unrelated patch only when the same exact test node fails with the same documented failure signature. It must still be reported.

A changed traceback, changed assertion, additional failure, or newly failing neighboring test must be treated as a possible regression. “Known baseline failure” does not mean the behavior is correct. Do not broadly ignore an entire test file merely because one or more nodes from that file appear here.

## Guided GUI suite expectation

`tests/test_gui_guided_workflow.py` and `tests/test_gui_guided_feature_detection_preview.py` are expected to complete with **zero ordinary failures**. Deferred defects are marked individually with `pytest.mark.xfail(strict=True)` and carry a `GUIDED-DEFERRED-nn` identifier in their reason. Any ordinary failure in these suites must be investigated as a regression, not tolerated as a baseline.

Because the marks are strict, a deferred defect that starts passing reports `XPASS(strict)` and fails the run, so a stale quarantine cannot survive silently.

## A. Deferred Guided product defects (strict xfail)

### GUIDED-DEFERRED-01 — the draft plan records the wrong output destination

- Affected nodes:
  - `tests/test_gui_guided_workflow.py::test_guided_new_analysis_draft_plan_distinguishes_select_output_from_run_output`
  - `tests/test_gui_guided_workflow.py::test_guided_new_analysis_applied_output_parent_is_real_draft_state`
- Scientist-facing defect: the Select data output folder is reported as an applied run-output policy. Before a run output parent is chosen, the page says one is still required while the plan already claims `applied`; after the scientist applies one and is told “Output parent folder is configured”, the plan still records the Select data folder. The plan a scientist reviews is not the destination they chose.
- Why deferred: `_build_guided_new_analysis_draft_plan` derives the output policy from the Select data field instead of the authoritative `_guided_new_analysis_output_policy_*` state that the status label already uses. Correcting it changes the output-destination contract for every Guided run and needs its own patch with execution-readiness and plan-identity coverage.
- Expected resolution area: the output-policy block of `_build_guided_new_analysis_draft_plan` in `gui/main_window.py`.

## C. Known execution-wiring test or fixture failures

These failures were reproduced individually, so they were not caused by combined-test contamination. They remain pending focused investigation.

- `tests/test_guided_gui_run_execution_wiring.py::test_real_gui_path_press_run_after_authorization[Robust Global Event-Reject Fit]`
- `tests/test_guided_gui_run_execution_wiring.py::test_real_gui_path_press_run_after_authorization[Adaptive Event-Gated Fit]`
- `tests/test_guided_gui_run_execution_wiring.py::test_real_gui_path_press_run_after_authorization[Global Linear Regression]`
  - Baseline signature for all three nodes: the test expected the output base not to exist, but it already existed.

## D. Deprioritized Full Control continuous failure

- `tests/test_continuous_mode_gui_production_workflow.py::test_gui_equivalent_continuous_full_run_outputs_are_viewer_visible`
  - Baseline signature: continuous outputs were generated, but `RunReportViewer.load_report()` rejected the completed result.
  - Deferred and deprioritized because Full Control continuous is not a current product priority.

## E. Additional known long-running or failing missing-session nodes

- `tests/test_guided_missing_session_authorization.py::test_guided_incomplete_final_exclusion_real_signal_only_lifecycle`
  - Baseline signature: `phasic_review_model is None`.
- `tests/test_guided_missing_session_authorization.py::test_guided_missing_session_real_gui_rerun_lifecycle`
  - Baseline signature: exceeds the finite five-minute test limit.

During CR1-0 verification, the complete `tests/test_guided_missing_session_authorization.py` file exceeded both five- and ten-minute finite limits, while the modified verification-boundary node passed.

## Resolved during baseline hygiene

- `tests/test_guided_npm_gui_natural_path.py::test_natural_path_npm_reaches_shared_completion_and_results_handoff`
  - This node previously failed at the pre-CR1-0 baseline. Diagnosis showed that the production completion check was correct: a stale synthetic worker fixture stamped the wrong NPM output time basis. The fixture now consumes the accepted normalized-description time basis, and the complete test file passes.

### Guided GUI suites (11 of the former 13 baseline failures)

Each was classified from its actual failure rather than quarantined.

Broken tests, repaired:

- `test_guided_diagnostics_guidance_new_analysis_failed_cache` — a stray copy of another test's `monkeypatch.setattr(...)` call had been appended after its last assertion, referencing two names that do not exist in its scope. The fragment was deleted.
- `test_guided_confirm_strategy_is_real_planning_ui_and_run_stays_skipped_in_open_results_mode`, `test_guided_confirm_strategy_never_auto_selects_from_loaded_or_generated_evidence`, `test_guided_confirm_strategy_explicit_mark_is_ui_state_only` — completed-run loading moved onto a worker thread, but these three asserted immediately after clicking Open Results. They now use the existing `_wait_for_guided_results_open` helper, as their passing neighbours already did.
- `test_full_control_report_viewer_unaffected_by_guided_review_viewer` — `_make_preview_completed_run` wrote only `time_sec`/`sig_raw`/`uv_raw`, so reviewing the run was correctly refused as missing canonical data. The fixture now also writes `dff` and `fit_ref`.
- `test_preview_uses_custom_roi_settings_when_customized`, `test_preview_uses_effective_settings_for_custom_roi_not_sparse_override`, `test_preview_inactive_absolute_field_does_not_block` — the shared evidence mock seeded an empty `locked_evidence_candidates`, so the real per-ROI currency check correctly marked the mocked Signal-Only F0 choice stale. The mock now builds the candidate the way production does, which also let a `_refresh_guided_draft_run_plan_preview` bypass be removed.

Obsolete expectations, updated to the accepted requirement:

- `test_select_data_page_avoids_developer_facing_wording` and `test_guided_ambiguous_or_unsupported_timing_does_not_overwrite_values` — both asserted wording removed in `69c5240` (“repair guided rwd and npm setup-to-run workflow”). They now assert the current scientist-facing phrasing; the forbidden-terms coverage is unchanged.
- `test_guided_new_analysis_cleared_feature_event_profile_still_blocks_validation` → `..._returns_to_usable_defaults` — it required a cleared profile to block validation, contradicting the accepted policy in `is_saved_feature_event_profile_current` (“Loaded valid Defaults are already consumed”) and the backend's own `test_loaded_default_feature_profile_is_current_without_explicit_apply`. It now asserts that clearing returns to usable Defaults, keeping its unique GUI-to-backend handoff coverage.
