# Known Baseline Test Failures

The active failures below were independently reproduced at the exact pre-CR1-0 baseline commit `e076296961b40e17a811aada188e1ed0632648c8` and remain current known baseline failures. A listed failure does not block an unrelated patch only when the same exact test node fails with the same documented failure signature. It must still be reported.

A changed traceback, changed assertion, additional failure, or newly failing neighboring test must be treated as a possible regression. “Known baseline failure” does not mean the behavior is correct. Do not broadly ignore an entire test file merely because one or more nodes from that file appear here.

## A. Deferred broad Guided workflow failures

These failures were not investigated during CR1-0. They may be stale tests, fixture problems, or real product defects.

- `tests/test_gui_guided_workflow.py::test_guided_confirm_strategy_is_real_planning_ui_and_run_stays_skipped_in_open_results_mode`
  - Baseline signature: ROI combo was empty instead of `["CH1", "CH2"]`.
- `tests/test_gui_guided_workflow.py::test_guided_confirm_strategy_never_auto_selects_from_loaded_or_generated_evidence`
  - Baseline signature: Signal-Only diagnostic remained “not generated.”
- `tests/test_gui_guided_workflow.py::test_guided_confirm_strategy_explicit_mark_is_ui_state_only`
  - Baseline signature: confirmation button was disabled.
- `tests/test_gui_guided_workflow.py::test_guided_new_analysis_cleared_feature_event_profile_still_blocks_validation`
  - Baseline signature: materialization succeeded instead of failing.
- `tests/test_gui_guided_workflow.py::test_full_control_report_viewer_unaffected_by_guided_review_viewer`
  - Baseline signature: completed Results load returned false.
  - Lower priority: this concerns Full Control behavior, which is not a current product priority.
- `tests/test_gui_guided_workflow.py::test_guided_ambiguous_or_unsupported_timing_does_not_overwrite_values`
  - Baseline signature: timing guidance text mismatch.
- `tests/test_gui_guided_workflow.py::test_guided_new_analysis_draft_plan_distinguishes_select_output_from_run_output`
  - Baseline signature: output policy was applied instead of missing.
- `tests/test_gui_guided_workflow.py::test_guided_new_analysis_applied_output_parent_is_real_draft_state`
  - Baseline signature: plan retained the Select Data output path.
- `tests/test_gui_guided_workflow.py::test_select_data_page_avoids_developer_facing_wording`
  - Baseline signature: expected automatic-format wording was absent.

## B. Clearly broken test

- `tests/test_gui_guided_workflow.py::test_guided_diagnostics_guidance_new_analysis_failed_cache`
  - Baseline signature: `NameError` because `monkeypatch` is undefined.
  - This test should eventually be repaired. Supplying the fixture alone is insufficient because the test also references the undefined local name `successful_computation`, so it is left unchanged pending a focused repair.

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
