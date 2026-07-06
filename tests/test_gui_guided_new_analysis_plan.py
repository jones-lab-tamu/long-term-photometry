"""GUI tests for the new_analysis Guided draft plan summary."""

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDatasetContractSnapshot,
    GuidedNewAnalysisDraftPlan,
    GuidedPlanCorrectionChoice,
    evaluate_guided_new_analysis_execution_subset_readiness,
    evaluate_new_analysis_plan_readiness,
)
from tests.test_gui_guided_workflow import (
    _configure_guided_raw_cache_setup,
    _generate_ready_guided_correction_preview,
    _write_minimal_guided_cache_outputs,
    _FakeDiagnosticCacheRunner,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def _complete_new_analysis_plan_for_gui(**overrides):
    plan = GuidedNewAnalysisDraftPlan(
        input_source_path="C:/raw/input",
        resolved_input_source_path="C:/raw/input",
        input_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        acquisition_structure_status="ready",
        discovered_roi_ids=["CH1"],
        included_roi_ids=["CH1"],
        cache_id="cache-1",
        cache_root_path="C:/cache",
        artifact_record_path="C:/cache/guided_diagnostic_cache_artifact.json",
        provenance_path="C:/cache/guided_diagnostic_cache_provenance.json",
        source_setup_signature="setup-1",
        diagnostic_scope_signature="scope-1",
        build_request_signature="build-1",
        stale_or_current="current",
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="CH1",
                selected_strategy="global_linear_regression",
                source_type="diagnostic_cache",
                diagnostic_cache_id="cache-1",
                diagnostic_cache_root="C:/cache",
                source_setup_signature="setup-1",
                diagnostic_scope_signature="scope-1",
                build_request_signature="build-1",
                current_or_stale="current",
                explicit_user_mark=True,
            )
        ],
        feature_event_profile_status="applied",
        feature_event_profile_id="feature-profile-1",
        feature_event_values={"event_signal": "dff"},
        feature_event_explicitly_applied=True,
        output_policy_status="applied",
        output_policy_path="C:/planned/output",
        output_policy_explicitly_applied=True,
    )
    for key, value in overrides.items():
        setattr(plan, key, value)
    return plan


def test_local_preview_choices_satisfy_strategy_readiness_without_full_evidence():
    evidence = {
        "evidence_source_type": "local_correction_preview",
        "preview_only": True,
        "production_analysis": False,
        "preview_id": "local-preview-1",
        "roi": "CH1",
        "selected_segment_label": "session-2",
        "message": (
            "Confirmed from local correction preview. Final analysis will "
            "recompute correction using the full selected recordings."
        ),
    }
    plan = _complete_new_analysis_plan_for_gui(
        cache_id=None,
        cache_root_path=None,
        artifact_record_path=None,
        provenance_path=None,
        source_setup_signature=None,
        diagnostic_scope_signature=None,
        build_request_signature=None,
        stale_or_current="missing",
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="CH1",
                selected_strategy="global_linear_regression",
                source_type="local_correction_preview",
                evidence_chunk=2,
                evidence_summary=evidence["message"],
                current_or_stale="current",
                explicit_user_mark=True,
                evidence_reference=evidence,
            )
        ],
    )

    readiness = evaluate_new_analysis_plan_readiness(plan)
    blocking = {
        issue.category
        for issue in readiness.blocking_issues
    }
    assert "missing_diagnostic_cache" not in blocking
    assert "stale_strategy_choice" not in blocking
    assert "missing_strategy_choice_for_included_roi" not in blocking
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    subset_blocking = {
        issue.category for issue in subset.blocking_issues
    }
    assert "missing_strategy_choice_for_execution_subset" not in subset_blocking
    assert "non_explicit_strategy_choice" not in subset_blocking


def test_stale_local_preview_choice_cannot_replace_full_evidence_readiness():
    evidence = {
        "evidence_source_type": "local_correction_preview",
        "preview_only": True,
        "production_analysis": False,
        "preview_id": "local-preview-stale",
        "roi": "CH1",
    }
    plan = _complete_new_analysis_plan_for_gui(
        cache_id=None,
        cache_root_path=None,
        artifact_record_path=None,
        provenance_path=None,
        source_setup_signature=None,
        diagnostic_scope_signature=None,
        build_request_signature=None,
        stale_or_current="missing",
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="CH1",
                selected_strategy="global_linear_regression",
                source_type="local_correction_preview",
                current_or_stale="stale",
                explicit_user_mark=True,
                evidence_reference=evidence,
            )
        ],
    )

    blocking = {
        issue.category
        for issue in evaluate_new_analysis_plan_readiness(
            plan
        ).blocking_issues
    }
    assert "missing_diagnostic_cache" in blocking
    assert "stale_strategy_choice" in blocking


def _configure_complete_guided_new_analysis_draft(
    window,
    tmp_path,
    monkeypatch,
    *,
    acquisition_mode="continuous",
    signal_only_f0=False,
    strategy_by_roi=None,
    write_rwd_file=False,
    session_duration=None,
):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    input_dir, _output_dir = _configure_guided_raw_cache_setup(
        window, tmp_path, monkeypatch
    )
    if write_rwd_file:
        session = input_dir / "2026_07_02-12_00_00"
        session.mkdir()
        rows = ["Time(s),CH1-410,CH1-470"]
        rows.extend(
            f"{index / 20.0:.2f},1.0,2.0"
            for index in range(12_000)
        )
        (session / "fluorescence.csv").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
    acquisition_idx = window._guided_acquisition_mode_combo.findData(acquisition_mode)
    if acquisition_idx >= 0:
        window._guided_acquisition_mode_combo.setCurrentIndex(acquisition_idx)
    if acquisition_mode == "intermittent":
        window._guided_sessions_per_hour_edit.setText("6")
        window._guided_session_duration_edit.setText(
            str(session_duration if session_duration is not None else 120)
        )

    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)
    _generate_ready_guided_correction_preview(window)

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    for index, roi in enumerate(("CH1", "CH2", "CH3")):
        window._guided_confirm_roi_combo.setCurrentIndex(window._guided_confirm_roi_combo.findData(roi))
        window._guided_confirm_chunk_combo.setCurrentIndex(0)
        strategy_text = "Signal-Only F0" if signal_only_f0 and index == 0 else "Global Linear Regression"
        if strategy_by_roi and roi in strategy_by_roi:
            strategy_text = strategy_by_roi[roi]
        strategy_index = window._guided_confirm_strategy_combo.findText(strategy_text)
        if strategy_index < 0:
            strategy_index = window._guided_confirm_strategy_combo.findData(strategy_text)
        assert strategy_index >= 0
        window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
        window._guided_confirm_ack_cb.setChecked(True)
        window._guided_confirm_mark_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_feature_event_apply_btn.click()

    output_parent = tmp_path / "planned_outputs"
    output_parent.mkdir()
    output_target = output_parent / "future_run_outputs"
    window._guided_output_path_edit.setText(str(output_target))
    window._guided_output_apply_btn.click()
    return output_parent, output_target


def test_new_analysis_draft_plan_displays_summary_fields(window, tmp_path, monkeypatch):
    # Enter new_analysis mode
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    assert window._guided_workflow_mode == "new_analysis"

    # Configure setup
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)    # Go to Draft Plan step
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    summary_text = window._guided_draft_run_plan_preview_label.text()
    assert "Status: new_analysis draft plan" in summary_text
    assert "Format: rwd" in summary_text
    assert "Acquisition mode: intermittent" in summary_text
    assert "ROI counts:" in summary_text
    assert "Diagnostic cache: missing" in summary_text
    assert "Correction strategy coverage:" in summary_text
    assert "Feature/event profile status: default_initialized" in summary_text
    assert "Output policy status: missing" in summary_text
    assert "Draft plan completeness: incomplete for future RunSpec handoff" in summary_text
    assert "Execution: unavailable, Final Guided Run/RunSpec is not implemented in this stage." in summary_text
    assert "This draft plan is not executable yet. Final Run is not implemented in this stage." in summary_text


def test_new_analysis_readiness_rendering_separates_planning_complete_from_execution_unavailable(window):
    plan = _complete_new_analysis_plan_for_gui()
    readiness = evaluate_new_analysis_plan_readiness(plan)

    summary = window._guided_new_analysis_draft_plan_summary_text(plan, readiness)
    readiness_summary = window._guided_new_analysis_readiness_summary_text(plan, readiness)
    window._refresh_guided_new_analysis_draft_plan_checklist(plan, readiness)
    checklist = window._guided_draft_run_plan_checklist_label.text()

    assert "Draft plan completeness: complete for future RunSpec handoff" in summary
    assert "Execution: unavailable, Final Guided Run/RunSpec is not implemented in this stage." in summary
    assert "Draft plan completeness: complete for future RunSpec handoff" in readiness_summary
    assert "Execution: unavailable, Final Guided Run/RunSpec is not implemented in this stage." in readiness_summary
    assert "Execution availability: unavailable" in checklist
    assert "Draft plan complete for handoff: true" in checklist
    assert "Execution available: false" in checklist


def test_new_analysis_readiness_rendering_shows_stale_feature_and_output_reasons(window):
    plan = _complete_new_analysis_plan_for_gui(
        feature_event_profile_status="stale",
        feature_event_stale_reasons=["baseline changed"],
        output_policy_status="stale",
        output_policy_stale_reasons=["target appeared"],
    )
    readiness = evaluate_new_analysis_plan_readiness(plan)

    summary = window._guided_new_analysis_draft_plan_summary_text(plan, readiness)
    readiness_summary = window._guided_new_analysis_readiness_summary_text(plan, readiness)
    window._refresh_guided_new_analysis_draft_plan_checklist(plan, readiness)
    checklist = window._guided_draft_run_plan_checklist_label.text()

    assert readiness.plan_complete_for_handoff is False
    assert "Feature/event profile stale reasons: baseline changed" in summary
    assert "Output policy stale reasons: target appeared" in summary
    assert "Feature/event settings (stale)" in readiness_summary
    assert "Output destination (stale)" in readiness_summary
    assert "Feature/event settings: fail - Feature/event profile is stale: baseline changed" in checklist
    assert "Output destination: fail - Output policy is stale: target appeared" in checklist


def _snapshot_files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


def test_new_analysis_dataset_contract_default_state_is_missing(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    plan = window._build_guided_new_analysis_draft_plan()

    assert plan.dataset_contract_snapshot.status == "missing"
    assert plan.dataset_contract_snapshot.current_applied is False
    assert "Stored dataset contract snapshot: missing" in window._guided_dataset_contract_status_label.text()
    assert "Dataset contract snapshot status: missing" in window._guided_draft_run_plan_preview_label.text()


def test_new_analysis_dataset_contract_apply_valid_rwd_snapshot_without_writes(
    window,
    tmp_path,
    monkeypatch,
):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    input_dir, _output_dir = _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    before = _snapshot_files(tmp_path)
    validation_revision_before = window._guided_backend_validation_revision

    window._guided_dataset_contract_apply_btn.click()
    plan = window._build_guided_new_analysis_draft_plan()
    snapshot = plan.dataset_contract_snapshot

    assert snapshot.status == "applied"
    assert snapshot.explicitly_applied is True
    assert snapshot.current_applied is True
    assert snapshot.source_identity.input_source_path == str(input_dir)
    assert snapshot.source_identity.input_format == "rwd"
    assert snapshot.source_identity.acquisition_mode == "intermittent"
    assert snapshot.source_identity.sessions_per_hour == 6
    assert snapshot.source_identity.session_duration_sec == 120.0
    assert snapshot.source_identity.exclude_incomplete_final_rwd_chunk is False
    assert snapshot.contract_values["rwd_time_col"] == "Time(s)"
    assert snapshot.contract_values["uv_suffix"] == "-410"
    assert snapshot.contract_values["sig_suffix"] == "-470"
    assert "explicit_guided_apply" in snapshot.provenance
    assert window._guided_backend_validation_revision == (
        validation_revision_before + 1
    )
    assert window._guided_backend_validation_outcome_revision is None
    assert "Dataset contract current_applied: true" in window._guided_draft_run_plan_preview_label.text()
    assert _snapshot_files(tmp_path) == before


def test_new_analysis_applied_rwd_dataset_contract_reaches_backend_materialization(
    window, tmp_path, monkeypatch
):
    from photometry_pipeline.guided_backend_validation_materialization import (
        GuidedBackendValidationMaterializationSuccess,
        materialize_guided_backend_validation_facts,
    )
    from tests.test_guided_backend_validation_materialization import (
        _valid_parser_contract,
    )

    _configure_complete_guided_new_analysis_draft(
        window,
        tmp_path,
        monkeypatch,
        acquisition_mode="intermittent",
        write_rwd_file=True,
        session_duration=600,
    )
    monkeypatch.setattr(
        window,
        "_infer_dataset_contract_overrides",
        MainWindow._infer_dataset_contract_overrides.__get__(
            window, MainWindow
        ),
    )
    window._discovery_cache["resolved_format"] = " RWD "
    window._guided_dataset_contract_apply_btn.click()
    plan = window._build_guided_new_analysis_draft_plan()

    assert plan.dataset_contract_snapshot.current_applied is True
    assert plan.dataset_contract_snapshot.explicitly_applied is True
    assert plan.dataset_contract_snapshot.contract_values[
        "session_duration_sec"
    ] == 600.0
    assert plan.dataset_contract_snapshot.contract_values[
        "rwd_time_col"
    ] == "Time(s)"
    assert plan.dataset_contract_snapshot.contract_values["uv_suffix"] == "-410"
    assert plan.dataset_contract_snapshot.contract_values["sig_suffix"] == "-470"
    result = materialize_guided_backend_validation_facts(
        plan,
        parser_contract=_valid_parser_contract(),
    )
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)


def test_run_page_validation_uses_current_applied_dataset_contract(
    window, tmp_path, monkeypatch
):
    _configure_complete_guided_new_analysis_draft(
        window,
        tmp_path,
        monkeypatch,
        acquisition_mode="intermittent",
        write_rwd_file=True,
        session_duration=600,
    )
    monkeypatch.setattr(
        window,
        "_infer_dataset_contract_overrides",
        MainWindow._infer_dataset_contract_overrides.__get__(
            window, MainWindow
        ),
    )
    window._discovery_cache["resolved_format"] = " RWD "
    window._guided_dataset_contract_apply_btn.click()
    preview_plan = window._build_guided_new_analysis_draft_plan()
    preview_snapshot = preview_plan.dataset_contract_snapshot
    assert preview_snapshot.current_applied is True
    assert preview_snapshot.input_format == "rwd"
    assert preview_snapshot.resolved_input_format == "rwd"
    assert preview_snapshot.acquisition_mode == "intermittent"

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Run")
    )
    context = window._capture_guided_backend_validation_context()
    validation_snapshot = context.draft.dataset_contract_snapshot
    assert validation_snapshot is preview_snapshot
    assert validation_snapshot.contract_values["session_duration_sec"] == 600.0
    assert validation_snapshot.contract_values["rwd_time_col"] == "Time(s)"
    assert validation_snapshot.contract_values["uv_suffix"] == "-410"
    assert validation_snapshot.contract_values["sig_suffix"] == "-470"

    window._guided_backend_validate_btn.click()

    issue_codes = {
        issue.detail_code
        for issue in window._guided_backend_validation_outcome.blocking_issues
    }
    assert "dataset_snapshot_missing_or_invalid" not in issue_codes
    assert "contract_snapshot_missing_or_invalid" not in issue_codes


def test_run_page_revalidation_rebuilds_draft_after_dataset_contract_apply(
    window, tmp_path, monkeypatch
):
    _configure_complete_guided_new_analysis_draft(
        window,
        tmp_path,
        monkeypatch,
        acquisition_mode="intermittent",
        write_rwd_file=True,
        session_duration=600,
    )
    monkeypatch.setattr(
        window,
        "_infer_dataset_contract_overrides",
        MainWindow._infer_dataset_contract_overrides.__get__(
            window, MainWindow
        ),
    )
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Run")
    )

    window._guided_backend_validate_btn.click()
    first_codes = {
        issue.detail_code
        for issue in window._guided_backend_validation_outcome.blocking_issues
    }
    assert "dataset_snapshot_missing_or_invalid" in first_codes
    first_revision = window._guided_backend_validation_outcome_revision

    window._guided_dataset_contract_apply_btn.click()
    assert window._guided_backend_validation_outcome_revision is None
    assert window._guided_backend_validation_revision > first_revision
    window._guided_backend_validate_btn.click()

    second_codes = {
        issue.detail_code
        for issue in window._guided_backend_validation_outcome.blocking_issues
    }
    assert "dataset_snapshot_missing_or_invalid" not in second_codes
    assert "contract_snapshot_missing_or_invalid" not in second_codes
    assert window._guided_backend_validation_outcome_revision == (
        window._guided_backend_validation_revision
    )


def test_run_page_unanimous_robust_marks_drive_dynamic_fit_contract(
    window, tmp_path, monkeypatch
):
    import photometry_pipeline.guided_execution_request_builder as request_builder

    robust_by_roi = {
        roi: "Robust Global Event-Reject Fit"
        for roi in ("CH1", "CH2", "CH3")
    }
    _configure_complete_guided_new_analysis_draft(
        window,
        tmp_path,
        monkeypatch,
        acquisition_mode="intermittent",
        strategy_by_roi=robust_by_roi,
        write_rwd_file=True,
        session_duration=600,
    )
    monkeypatch.setattr(
        window,
        "_infer_dataset_contract_overrides",
        MainWindow._infer_dataset_contract_overrides.__get__(
            window, MainWindow
        ),
    )
    window._guided_dataset_contract_apply_btn.click()
    plan = window._build_guided_new_analysis_draft_plan()

    assert {
        choice.selected_strategy
        for choice in plan.per_roi_correction_strategy_choices
        if choice.roi_id in plan.included_roi_ids
        and choice.current_or_stale == "current"
    } == {"robust_global_event_reject"}
    assert plan.dynamic_fit_parameter_contract.dynamic_fit_mode == (
        "robust_global_event_reject"
    )
    assert plan.dynamic_fit_parameter_contract.provenance[
        "dynamic_fit_mode"
    ] == "unanimous current explicit included-ROI strategy marks"

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Run")
    )
    monkeypatch.setattr(
        request_builder,
        "resolve_application_build_identity",
        lambda **_kwargs: SimpleNamespace(build_identity=None),
    )
    window._guided_backend_validate_btn.click()
    issue_codes = {
        issue.detail_code
        for issue in window._guided_backend_validation_outcome.blocking_issues
    }
    assert "dynamic_fit_mode_mismatch" not in issue_codes
    assert "dataset_snapshot_missing_or_invalid" not in issue_codes
    assert "correction_preview_missing_or_stale" not in issue_codes
    assert "feature_event_profile_missing" not in issue_codes
    assert "output_policy_missing" not in issue_codes
    assert window._guided_backend_validation_outcome.status == (
        "validator_accepted"
    )
    assert window._guided_run_authorization_result is None
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_run_readiness_label.text() == (
        "Guided validation succeeded, but Guided Run execution is unavailable "
        "in this build."
    )
    assert "Validate the setup first" not in (
        window._guided_run_readiness_label.text()
    )


def test_new_analysis_dataset_contract_missing_duration_or_semantics_cannot_apply(
    window, tmp_path, monkeypatch
):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.clear()
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )

    window._guided_dataset_contract_apply_btn.click()

    snapshot = window._build_guided_new_analysis_draft_plan().dataset_contract_snapshot
    assert snapshot.current_applied is False
    assert "intermittent session duration is missing or invalid" in (
        window._guided_dataset_contract_status_label.text()
    )

    window._guided_session_duration_edit.setText("600")
    monkeypatch.setattr(window, "_infer_dataset_contract_overrides", lambda _fmt: {})
    window._guided_dataset_contract_apply_btn.click()

    snapshot = window._build_guided_new_analysis_draft_plan().dataset_contract_snapshot
    assert snapshot.current_applied is False
    assert "required RWD dataset semantics are unresolved" in (
        window._guided_dataset_contract_status_label.text()
    )


def test_dataset_contract_resolved_format_mismatch_cannot_reach_run_validation(
    window, tmp_path, monkeypatch
):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("600")
    window._discovery_cache["resolved_format"] = "npm"

    window._guided_dataset_contract_apply_btn.click()

    snapshot = window._build_guided_new_analysis_draft_plan().dataset_contract_snapshot
    assert snapshot.current_applied is False
    assert "resolved input format does not match the selected format" in (
        window._guided_dataset_contract_status_label.text()
    )


def test_new_analysis_dataset_contract_invalid_candidate_cannot_apply(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_format_combo.setCurrentText("npm")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_dataset_contract_apply_btn.click()
    plan = window._build_guided_new_analysis_draft_plan()
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)

    assert plan.dataset_contract_snapshot.current_applied is False
    assert "Dataset contract was not applied" in window._guided_dataset_contract_status_label.text()
    assert "NPM channel mapping is not represented" in window._guided_dataset_contract_candidate_label.text()
    assert any(issue.category == "missing_npm_channel_mapping" for issue in subset.blocking_issues)


def test_new_analysis_dataset_contract_npm_continuous_remains_unsupported(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_format_combo.setCurrentText("npm")
    idx = window._guided_acquisition_mode_combo.findData("continuous")
    window._guided_acquisition_mode_combo.setCurrentIndex(idx)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_dataset_contract_apply_btn.click()
    plan = window._build_guided_new_analysis_draft_plan()
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)

    assert plan.dataset_contract_snapshot.current_applied is False
    assert "unsupported_npm_continuous" in window._guided_dataset_contract_candidate_label.text()
    assert any(issue.category == "unsupported_npm_continuous" for issue in subset.blocking_issues)


def test_new_analysis_dataset_contract_clear_preserves_other_draft_state(window, tmp_path, monkeypatch):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)
    window._guided_dataset_contract_apply_btn.click()
    assert window._build_guided_new_analysis_draft_plan().dataset_contract_snapshot.current_applied is True
    before_cache = window._guided_diagnostic_cache_record
    before_choices = dict(window._guided_strategy_choices)
    before_feature_status = window._guided_new_analysis_feature_event_profile_status
    before_output_status = window._guided_new_analysis_output_policy_status

    window._guided_dataset_contract_clear_btn.click()
    plan = window._build_guided_new_analysis_draft_plan()

    assert plan.dataset_contract_snapshot.status == "missing"
    assert plan.dataset_contract_snapshot.current_applied is False
    assert window._guided_diagnostic_cache_record is before_cache
    assert window._guided_strategy_choices == before_choices
    assert window._guided_new_analysis_feature_event_profile_status == before_feature_status
    assert window._guided_new_analysis_output_policy_status == before_output_status


def test_new_analysis_dataset_contract_marks_stale_on_setup_change(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_dataset_contract_apply_btn.click()
    assert window._build_guided_new_analysis_draft_plan().dataset_contract_snapshot.current_applied is True

    window._guided_session_duration_edit.setText("180")
    window._refresh_guided_draft_run_plan_preview()
    snapshot = window._build_guided_new_analysis_draft_plan().dataset_contract_snapshot

    assert snapshot.status == "stale"
    assert snapshot.explicitly_applied is True
    assert snapshot.current_applied is False
    assert any(
        "session_duration_sec changed" in reason
        for reason in snapshot.stale_reasons
    )
    assert "Dataset contract stale reasons:" in window._guided_draft_run_plan_preview_label.text()


def test_new_analysis_dataset_contract_applied_satisfies_rwd_execution_subset_dataset_blocker(
    window,
    tmp_path,
    monkeypatch,
):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_dataset_contract_apply_btn.click()

    plan = window._build_guided_new_analysis_draft_plan()
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    fields = {field.field_name: field for field in subset.field_classifications}

    assert fields["dataset_contract_snapshot"].status == "present"
    assert fields["dataset_contract_overrides"].status == "present"
    assert not any(issue.category == "missing_rwd_dataset_contract" for issue in subset.blocking_issues)
    assert subset.execution_available is False


def test_new_analysis_dataset_contract_apply_clear_stale_do_not_call_execution_helpers(
    window,
    tmp_path,
    monkeypatch,
):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    called = []

    def fail_helper(name):
        def _fail(*_args, **_kwargs):
            called.append(name)
            raise AssertionError(f"{name} should not be called")
        return _fail

    monkeypatch.setattr(window, "_build_run_spec", fail_helper("_build_run_spec"))
    monkeypatch.setattr(window, "_build_argv", fail_helper("_build_argv"))
    monkeypatch.setattr(window, "_on_validate", fail_helper("_on_validate"))
    monkeypatch.setattr(window, "_on_run", fail_helper("_on_run"))

    window._guided_dataset_contract_apply_btn.click()
    window._guided_input_dir_edit.setText(str(tmp_path / "changed_input"))
    window._refresh_guided_draft_run_plan_preview()
    window._guided_dataset_contract_clear_btn.click()

    assert called == []


def test_new_analysis_dataset_contract_apply_clear_stale_do_not_create_files(
    window,
    tmp_path,
    monkeypatch,
):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    before = _snapshot_files(tmp_path)

    window._guided_dataset_contract_apply_btn.click()
    changed_input = tmp_path / "changed_raw_input"
    changed_input.mkdir()
    before_after_manual_dir = _snapshot_files(tmp_path)
    window._guided_input_dir_edit.setText(str(changed_input))
    window._refresh_guided_draft_run_plan_preview()
    window._guided_dataset_contract_clear_btn.click()

    assert _snapshot_files(tmp_path) == before_after_manual_dir
    assert changed_input.exists()
    assert before_after_manual_dir != before


def test_new_analysis_run_preview_displays_missing_dataset_contract_snapshot(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Dataset contract snapshot:" in preview_text
    assert "stored status: missing" in preview_text
    assert "current_applied: false" in preview_text
    assert "execution consumption: not enabled in this stage" in preview_text
    assert "Execution: unavailable" in preview_text


def test_new_analysis_run_preview_displays_applied_dataset_contract_consumed_by_readiness(
    window,
    tmp_path,
    monkeypatch,
):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_dataset_contract_apply_btn.click()

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Dataset contract snapshot:" in preview_text
    assert "stored status: applied" in preview_text
    assert "current_applied: true" in preview_text
    assert "explicitly_applied: true" in preview_text
    assert "input_format: rwd" in preview_text
    assert "acquisition_mode: intermittent" in preview_text
    assert "validation issues: none" in preview_text
    assert "stale reasons: none" in preview_text
    assert "execution consumption: enabled for first-subset readiness classification" in preview_text
    assert "missing_rwd_dataset_contract" not in preview_text
    assert "execution_available: false" in preview_text
    assert "ready to run" not in preview_text.lower()


def test_new_analysis_run_preview_displays_stale_dataset_contract_snapshot(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_dataset_contract_apply_btn.click()

    changed_input = tmp_path / "changed_raw_input"
    changed_input.mkdir()
    window._guided_input_dir_edit.setText(str(changed_input))
    window._refresh_guided_draft_run_plan_preview()
    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Dataset contract snapshot:" in preview_text
    assert "stored status: stale" in preview_text
    assert "current_applied: false" in preview_text
    assert "explicitly_applied: true" in preview_text
    assert "stale reasons:" in preview_text
    assert "input_source_path changed" in preview_text
    assert "execution consumption: not enabled in this stage" in preview_text


def test_new_analysis_run_preview_displays_represented_unsupported_dataset_contract_snapshot(window):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_new_analysis_dataset_contract_snapshot = GuidedNewAnalysisDatasetContractSnapshot(
        status="unsupported",
        input_format="npm",
        resolved_input_format="npm",
        acquisition_mode="continuous",
        validation_issues=("unsupported_npm_continuous",),
    )
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._refresh_guided_draft_run_plan_preview()
    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Dataset contract snapshot:" in preview_text
    assert "stored status: unsupported" in preview_text
    assert "current_applied: false" in preview_text
    assert "validation issues: unsupported_npm_continuous" in preview_text
    assert "execution consumption: not enabled in this stage" in preview_text
    assert "unsupported_npm_continuous" in preview_text


def test_new_analysis_run_preview_keeps_existing_sections_with_dataset_contract(window, tmp_path, monkeypatch):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)
    window._guided_dataset_contract_apply_btn.click()
    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Preview schema version:" in preview_text
    assert "Plan schema version:" in preview_text
    assert "Source/input:" in preview_text
    assert "Acquisition:" in preview_text
    assert "Execution intent:" in preview_text
    assert "Dataset contract snapshot:" in preview_text
    assert "Included ROIs:" in preview_text
    assert "Correction strategies:" in preview_text
    assert "Feature/event:" in preview_text
    assert "Feature/event consumption:" in preview_text
    assert "Output policy status:" in preview_text
    assert "Output creation policy:" in preview_text
    assert "Diagnostic cache:" in preview_text
    assert "First execution subset:" in preview_text
    assert "Guided execution-spec preview:" in preview_text
    assert "backend_mapping_status: preview_only_not_mapped_to_RunSpec" in preview_text
    assert "dynamic_fit_parameter_contract:" in preview_text
    assert "backend_config_mapping_status: label_and_parameters_ready_for_future_mapping" in preview_text
    assert "output: no directories or files created" in preview_text
    assert "Execution: unavailable" in preview_text
    assert "No files or directories were created." in preview_text
    assert "This preview is read-only and non-executing." in preview_text


def test_new_analysis_run_preview_displays_execution_intent_and_output_creation_policy(
    window,
    tmp_path,
    monkeypatch,
):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Execution intent:" in preview_text
    assert "timeline_anchor_mode: civil" in preview_text
    assert "fixed_daily_anchor_clock: none" in preview_text
    assert "execution_mode: phasic" in preview_text
    assert "run_profile: full" in preview_text
    assert "execution consumption: enabled for first-subset readiness classification" in preview_text
    assert "Feature/event consumption:" in preview_text
    assert "  execution_mode: phasic" in preview_text
    assert "  run_profile: full" in preview_text
    assert "  traces_only: false" in preview_text
    assert "  feature_event_profile_required: true" in preview_text
    assert "  feature_event_profile_current_applied: true" in preview_text
    assert "  feature_event_values_consumed: true" in preview_text
    assert "  feature_extraction_in_scope: true" in preview_text
    assert "  feature_dependent_phasic_summaries_in_scope: true" in preview_text
    assert "  tonic_outputs_in_scope: false" in preview_text
    assert "  full_both_mode_outputs_in_scope: false" in preview_text
    assert "feature_event_effective_values:" in preview_text
    assert "backend_config_mapping_status: effective_values_ready_for_future_mapping" in preview_text
    assert "unresolved_fields: none" in preview_text
    assert "Output creation policy:" in preview_text
    assert "path_role: output_base" in preview_text
    assert "creation_timing: future_execution_start_only" in preview_text
    assert "run_directory_strategy: derive_unique_run_id_under_output_base" in preview_text
    assert "overwrite: false" in preview_text
    assert "precreate_during_preview: false" in preview_text
    assert "config_write_timing: future_execution_or_validation_only" in preview_text
    assert "gui_preflight_writes_enabled: false" in preview_text
    assert "ready to run" not in preview_text.lower()
    assert "execution-ready" not in preview_text.lower()
    assert "runnable" not in preview_text.lower()
    assert "RunSpec generated" not in preview_text
    assert "config generated" not in preview_text
    assert "output folder created" not in preview_text


def test_new_analysis_run_preview_feature_event_consumption_requires_applied_profile(
    window,
    tmp_path,
    monkeypatch,
):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)
    window._guided_new_analysis_feature_event_profile_status = "default_initialized"
    window._guided_new_analysis_feature_event_profile_errors = []
    window._guided_new_analysis_feature_event_profile_stale_reasons = []
    window._guided_new_analysis_feature_event_profile_explicitly_applied = False
    window._refresh_guided_draft_run_plan_preview()

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "feature_event_profile_not_applied" in preview_text
    assert "Feature/event consumption:" in preview_text
    assert "  execution_mode: phasic" in preview_text
    assert "  run_profile: full" in preview_text
    assert "  traces_only: false" in preview_text
    assert "  feature_event_profile_required: true" in preview_text
    assert "  feature_event_profile_current_applied: false" in preview_text
    assert "  feature_event_values_consumed: false" in preview_text
    assert "  feature_extraction_in_scope: true" in preview_text
    assert "  feature_dependent_phasic_summaries_in_scope: true" in preview_text
    assert "  execution consumption: not enabled until feature/event profile is applied and current" in preview_text
    assert "Execution unavailable" in preview_text
    assert "ready to run" not in preview_text.lower()
    assert "execution-ready" not in preview_text.lower()
    assert "runnable" not in preview_text.lower()


@pytest.mark.parametrize(
    ("status", "issues_attr", "issues", "expected_category"),
    [
        ("invalid", "_guided_new_analysis_feature_event_profile_errors", ["bad threshold"], "invalid_feature_event_profile"),
        ("stale", "_guided_new_analysis_feature_event_profile_stale_reasons", ["baseline changed"], "stale_feature_event_profile"),
    ],
)
def test_new_analysis_run_preview_feature_event_invalid_or_stale_blocks_consumption(
    window,
    tmp_path,
    monkeypatch,
    status,
    issues_attr,
    issues,
    expected_category,
):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)
    window._guided_new_analysis_feature_event_profile_status = status
    setattr(window, issues_attr, issues)
    window._refresh_guided_draft_run_plan_preview()

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert expected_category in preview_text
    assert "Feature/event consumption:" in preview_text
    assert "  feature_event_profile_required: true" in preview_text
    assert "  feature_event_profile_current_applied: false" in preview_text
    assert "  feature_event_values_consumed: false" in preview_text
    assert "  execution consumption: not enabled until feature/event profile is applied and current" in preview_text
    assert "Execution unavailable" in preview_text


def test_new_analysis_draft_plan_reports_choices_as_current_after_build_and_mark(window, tmp_path, monkeypatch):
    # Enter new_analysis mode and configure
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)

    # Build cache
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)
    _generate_ready_guided_correction_preview(window)

    # Confirm strategy step: select ROI and mark
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    
    # Check that we can select and mark CH1
    window._guided_confirm_roi_combo.setCurrentIndex(window._guided_confirm_roi_combo.findData("CH1"))
    window._guided_confirm_chunk_combo.setCurrentIndex(0)
    window._guided_confirm_strategy_combo.setCurrentIndex(window._guided_confirm_strategy_combo.findText("Global Linear Regression"))
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    # Verify choices are recorded as current in Draft Plan summary
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    summary_text = window._guided_draft_run_plan_preview_label.text()
    
    assert "Diagnostic cache: current" in summary_text
    assert "1/3 ROIs covered" in summary_text


def test_new_analysis_draft_plan_marks_stale_when_roi_selection_changes(window, tmp_path, monkeypatch):
    # Enter mode, configure, build cache, mark strategy
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)

    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    _write_minimal_guided_cache_outputs(Path(fake_runner.run_dir))
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)
    _generate_ready_guided_correction_preview(window)

    # Confirm CH1
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._guided_confirm_roi_combo.setCurrentIndex(window._guided_confirm_roi_combo.findData("CH1"))
    window._guided_confirm_chunk_combo.setCurrentIndex(0)
    window._guided_confirm_strategy_combo.setCurrentIndex(window._guided_confirm_strategy_combo.findText("Global Linear Regression"))
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    # Switch back to Select data or ROI discovery and change ROI inclusion selection
    # Uncheck CH2
    for i in range(window._guided_roi_list.count()):
        item = window._guided_roi_list.item(i)
        if item.text() == "CH2":
            item.setCheckState(Qt.Unchecked)

    # Go to Draft Plan step and check that summary/cache is marked stale
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    summary_text = window._guided_draft_run_plan_preview_label.text()

    assert "Diagnostic cache: stale" in summary_text
    assert "stale_diagnostic_cache" in summary_text


def test_new_analysis_draft_plan_no_runspec_or_outputs_written(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)

    # Records initial files in temp directories
    initial_files = list(tmp_path.glob("**/*"))

    # Refresh plan summary
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._refresh_guided_draft_run_plan_preview()

    # Assert no new files/directories were created under output base
    current_files = list(tmp_path.glob("**/*"))
    assert len(current_files) == len(initial_files)

    # Assert no RunSpec was generated
    assert not hasattr(window, "_generated_run_spec") or window._generated_run_spec is None
    assert window._current_run_dir == ""


def test_new_analysis_run_preview_panel_renders_complete_plan(window, tmp_path, monkeypatch):
    _parent, output_target = _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Non-executing preview" in preview_text
    assert "Preview schema version: guided_new_analysis_run_preview.v1" in preview_text
    assert "Plan schema version: guided_new_analysis_plan.v1" in preview_text
    assert "Source/input:" in preview_text
    assert "Included ROIs: 3 (CH1, CH2, CH3)" in preview_text
    assert "Output destination:" in preview_text
    assert output_target.name in preview_text
    assert "Execution unavailable" in preview_text
    assert "Final Guided Run/RunSpec is not implemented in this stage." in preview_text
    assert "First execution subset:" in preview_text
    assert "subset: global_dynamic_fit_only.v1" in preview_text
    assert "first_subset_executable: false" in preview_text
    assert "allowed_dynamic_fit_strategy: global_linear_regression" in preview_text
    assert "execution_available: false" in preview_text
    assert "No files or directories were created." in preview_text
    assert not output_target.exists()


def test_new_analysis_run_preview_shows_missing_execution_subset_fields(window, tmp_path, monkeypatch):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Draft plan completeness: complete for future RunSpec handoff" in preview_text
    assert "status: not executable under global_dynamic_fit_only.v1" in preview_text
    assert "Execution-subset blockers:" in preview_text
    assert "missing_rwd_dataset_contract" in preview_text
    assert "missing_timeline_anchor_mode" not in preview_text
    assert "missing_execution_mode" not in preview_text
    assert "missing_run_profile" not in preview_text
    assert "missing_output_creation_policy" not in preview_text
    assert "timeline_anchor_mode: fixed_default=civil" in preview_text
    assert "mode: fixed_default=phasic" in preview_text
    assert "run_profile: fixed_default=full" in preview_text
    assert "output_creation_policy: present" in preview_text
    assert "traces_only: fixed_default=False" in preview_text
    assert "preview_first_n: fixed_default" in preview_text
    assert "dataset_contract_overrides: required_missing, blocks subset" in preview_text


def test_new_analysis_run_preview_applied_rwd_dataset_contract_satisfies_dataset_blocker(
    window,
    tmp_path,
    monkeypatch,
):
    _configure_complete_guided_new_analysis_draft(
        window,
        tmp_path,
        monkeypatch,
        acquisition_mode="intermittent",
    )

    window._guided_dataset_contract_apply_btn.click()
    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Dataset contract snapshot:" in preview_text
    assert "Execution intent:" in preview_text
    assert "timeline_anchor_mode: civil" in preview_text
    assert "execution_mode: phasic" in preview_text
    assert "run_profile: full" in preview_text
    assert "execution consumption: enabled for first-subset readiness classification" in preview_text
    assert "stored status: applied" in preview_text
    assert "current_applied: true" in preview_text
    assert "missing_rwd_dataset_contract" not in preview_text
    assert "dataset_contract_overrides: present" in preview_text
    assert "missing_timeline_anchor_mode" not in preview_text
    assert "missing_execution_mode" not in preview_text
    assert "missing_run_profile" not in preview_text
    assert "missing_output_creation_policy" not in preview_text
    assert "first_subset_executable: true" in preview_text
    assert "Feature/event consumption:" in preview_text
    assert "  feature_event_profile_current_applied: true" in preview_text
    assert "  feature_event_values_consumed: true" in preview_text
    assert "  feature_extraction_in_scope: true" in preview_text
    assert "  feature_dependent_phasic_summaries_in_scope: true" in preview_text
    assert "status: complete for future execution-spec preview; actual execution remains unavailable" in preview_text
    assert "Guided execution-spec preview:" in preview_text
    assert "spec_preview_available: true" in preview_text
    assert "first_subset_executable: true" in preview_text
    assert "backend_mapping_status: preview_only_not_mapped_to_RunSpec" in preview_text
    assert "dynamic_fit_parameter_contract:" in preview_text
    assert "dynamic_fit_mode: global_linear_regression" in preview_text
    assert "selected_strategy: global_linear_regression" in preview_text
    assert "active_parameter_set: global_linear_regression" in preview_text
    assert "backend_config_mapping_status: label_and_parameters_ready_for_future_mapping" in preview_text
    assert "feature_event_effective_values:" in preview_text
    assert "backend_config_mapping_status: effective_values_ready_for_future_mapping" in preview_text
    assert "unresolved_fields: none" in preview_text
    assert "rwd_dataset_normalization:" in preview_text
    assert "backend_config_mapping_status: rwd_dataset_contract_ready_for_future_mapping" in preview_text
    assert "missing_required_fields: none" in preview_text
    assert "inconsistent_fields: none" in preview_text
    assert "output_safety_ownership:" in preview_text
    assert "backend_config_mapping_status: output_base_ready_for_runner_owned_future_mapping" in preview_text
    assert "future_output_owner: runner" in preview_text
    assert "future_run_dir: unresolved_until_execution_start" in preview_text
    assert "blockers: none" in preview_text
    assert "first_subset_mapping_preview:" in preview_text
    assert "mapping_preview_available: true" in preview_text
    assert "scope: rwd_intermittent_phasic_full_dynamic_fit" in preview_text
    assert "future_cli_target: out_base_concept_only" in preview_text
    assert "config_generated: false" in preview_text
    assert "argv_generated: false" in preview_text
    assert "guided_runner_request_preview:" in preview_text
    assert "runner_request_preview_available: true" in preview_text
    assert "future_runner_owner: runner" in preview_text
    assert "config_payload_generated: false" in preview_text
    assert "validation_run: false" in preview_text
    assert "execution_run: false" in preview_text
    assert "output: no directories or files created" in preview_text
    assert "missing_required_rwd_contract_field" not in preview_text
    assert "unresolved_rwd_dataset_contract_normalization" not in preview_text
    assert "execution_available: false" in preview_text
    assert "ready to run" not in preview_text.lower()
    assert "ready for execution" not in preview_text.lower()
    assert "execution-ready" not in preview_text.lower()
    assert "runnable" not in preview_text.lower()
    assert "RunSpec generated" not in preview_text
    assert "config generated" not in preview_text
    assert "output folder created" not in preview_text


def test_new_analysis_run_preview_stale_dataset_contract_keeps_dataset_blocker(
    window,
    tmp_path,
    monkeypatch,
):
    _configure_complete_guided_new_analysis_draft(
        window,
        tmp_path,
        monkeypatch,
        acquisition_mode="intermittent",
    )
    window._guided_dataset_contract_apply_btn.click()

    changed_input = tmp_path / "changed_raw_input"
    changed_input.mkdir()
    window._guided_input_dir_edit.setText(str(changed_input))
    window._refresh_guided_draft_run_plan_preview()
    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Dataset contract snapshot:" in preview_text
    assert "stored status: stale" in preview_text
    assert "current_applied: false" in preview_text
    assert "stale reasons:" in preview_text
    assert "stale_dataset_contract_snapshot" in preview_text
    assert "execution_available: false" in preview_text
    assert "ready to run" not in preview_text.lower()


def test_new_analysis_run_preview_panel_shows_incomplete_plan_unresolved_items(window):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Non-executing preview" in preview_text
    assert "Draft plan completeness: incomplete for future RunSpec handoff" in preview_text
    assert "Run preview unresolved items:" in preview_text
    assert "missing_diagnostic_cache" in preview_text
    assert "missing_output_policy" in preview_text
    assert "First execution subset:" in preview_text
    assert "incomplete_planning_readiness" in preview_text
    assert "Execution unavailable" in preview_text


def test_new_analysis_run_preview_complete_plan_keeps_execution_unavailable(window, tmp_path, monkeypatch):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Draft plan completeness: complete for future RunSpec handoff" in preview_text
    assert "per_roi_correction_execution_contract_unresolved" not in preview_text
    assert "global collapse false" in preview_text
    assert "missing_rwd_dataset_contract" in preview_text
    assert "missing_timeline_anchor_mode" not in preview_text
    assert "missing_execution_mode" not in preview_text
    assert "missing_run_profile" not in preview_text
    assert "missing_output_creation_policy" not in preview_text
    assert "Execution unavailable" in preview_text
    assert "Guided execution-spec preview:" in preview_text
    assert "spec_preview_available: false" in preview_text
    assert "first_subset_executable: false" in preview_text
    assert "dynamic_fit_parameter_contract:" in preview_text
    assert "ready to run" not in preview_text.lower()
    assert "ready for execution" not in preview_text.lower()
    assert "execution-ready" not in preview_text.lower()
    assert "runnable" not in preview_text.lower()
    assert "RunSpec generated" not in preview_text
    assert "config generated" not in preview_text
    assert "output folder created" not in preview_text


def test_new_analysis_run_preview_signal_only_f0_unresolved_routing(window, tmp_path, monkeypatch):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch, signal_only_f0=True)

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "CH1: signal_only_f0" in preview_text
    assert "Execution unavailable" in preview_text


def test_new_analysis_run_preview_mixed_per_roi_strategies_subset_blocked_not_planning_blocked(
    window, tmp_path, monkeypatch
):
    _configure_complete_guided_new_analysis_draft(
        window,
        tmp_path,
        monkeypatch,
        strategy_by_roi={
            "CH1": "Global Linear Regression",
            "CH2": "robust_global_event_reject",
            "CH3": "Global Linear Regression",
        },
    )

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Draft plan completeness: complete" in preview_text
    assert "mixed_dynamic_fit_modes_execution_not_enabled" in preview_text
    assert "status: not executable under global_dynamic_fit_only.v1" in preview_text
    assert "Execution unavailable" in preview_text
    assert "ready to run" not in preview_text.lower()


def test_new_analysis_run_preview_rendering_does_not_create_output_files(window, tmp_path, monkeypatch):
    output_parent, output_target = _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)
    before_files = sorted(str(path.relative_to(output_parent)) for path in output_parent.rglob("*"))

    window._refresh_guided_draft_run_plan_preview()

    after_files = sorted(str(path.relative_to(output_parent)) for path in output_parent.rglob("*"))
    assert after_files == before_files
    assert not output_target.exists()
    assert "No files or directories were created." in window._guided_new_analysis_run_preview_label.text()


def test_new_analysis_run_preview_rendering_does_not_mutate_completed_run_state(
    window, tmp_path, monkeypatch
):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)
    window._current_run_dir = str(tmp_path / "completed_run_sentinel")
    window._guided_draft_output_policy_by_run = {"completed": "policy"}
    window._guided_draft_feature_event_profiles_by_run = {"completed": [{"profile_id": "existing"}]}

    before_current_run = window._current_run_dir
    before_output_policies = dict(window._guided_draft_output_policy_by_run)
    before_feature_profiles = {
        key: list(value) for key, value in window._guided_draft_feature_event_profiles_by_run.items()
    }

    window._refresh_guided_draft_run_plan_preview()

    assert window._current_run_dir == before_current_run
    assert window._guided_draft_output_policy_by_run == before_output_policies
    assert window._guided_draft_feature_event_profiles_by_run == before_feature_profiles


def test_new_analysis_run_preview_rendering_does_not_call_execution_helpers(
    window, tmp_path, monkeypatch
):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)
    calls = []

    def forbidden(name):
        def _inner(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not be called by non-executing preview rendering")
        return _inner

    monkeypatch.setattr(window, "_build_run_spec", forbidden("_build_run_spec"))
    monkeypatch.setattr(window, "_build_argv", forbidden("_build_argv"))
    monkeypatch.setattr(window, "_on_validate", forbidden("_on_validate"))
    monkeypatch.setattr(window, "_on_run", forbidden("_on_run"))

    window._refresh_guided_draft_run_plan_preview()

    assert calls == []
    assert "Non-executing preview" in window._guided_new_analysis_run_preview_label.text()


def test_new_analysis_feature_event_profile_gui_flows(window, tmp_path, monkeypatch):
    # Case F & K: Initialize from Full Control active baseline, not bare Config()
    custom_cfg_path = tmp_path / "custom_config.yaml"
    custom_cfg_content = """
event_signal: delta_f
signal_excursion_polarity: negative
peak_threshold_method: absolute
peak_threshold_k: 4.5
peak_threshold_percentile: 92.5
peak_threshold_abs: 0.123
peak_min_distance_sec: 12.0
peak_min_prominence_k: 2.5
peak_min_width_sec: 1.5
peak_pre_filter: lowpass
event_auc_baseline: median
"""
    custom_cfg_path.write_text(custom_cfg_content, encoding="utf-8")
    
    # Configure custom baseline YAML in Full Control
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(custom_cfg_path))
    
    # Enter new_analysis mode and configure setup
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    
    # Go to Draft Plan step to initialize feature/event editor
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    
    # Verify the defaults are initialized from active baseline custom config (Case F and Case K drift test)
    assert window._guided_feature_event_polarity_combo.currentText() == "negative"
    assert window._guided_feature_event_peak_method_combo.currentText() == "absolute"
    assert window._guided_feature_event_peak_k_edit.text() == "4.5"
    assert window._guided_feature_event_peak_pct_edit.text() == "92.5"
    assert window._guided_feature_event_peak_abs_edit.text() == "0.123"
    assert window._guided_feature_event_peak_distance_edit.text() == "12.0"
    assert window._guided_feature_event_peak_prominence_edit.text() == "2.5"
    assert window._guided_feature_event_peak_width_edit.text() == "1.5"
    assert window._guided_feature_event_pre_filter_combo.currentText() == "lowpass"
    assert window._guided_feature_event_auc_baseline_combo.currentText() == "median"
    
    # Verify profile status is default_initialized, which does not count as configured/passed (blocking issue)
    assert window._guided_new_analysis_feature_event_profile_status == "default_initialized"
    summary_text = window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event profile status: default_initialized" in summary_text
    checklist_text = window._guided_draft_run_plan_checklist_label.text()
    assert "Feature/event settings: fail" in checklist_text
    
    # Case G: GUI Apply valid feature/event settings in new_analysis
    window._guided_feature_event_apply_btn.click()
    assert window._guided_new_analysis_feature_event_profile_status == "applied"
    
    # Checklist should now pass for Feature/event settings
    summary_text_applied = window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event profile status: applied" in summary_text_applied
    checklist_text_applied = window._guided_draft_run_plan_checklist_label.text()
    assert "Feature/event settings: pass" in checklist_text_applied
    
    # Case H: GUI Invalid Apply is local and non-executing
    # Set invalid value
    window._guided_feature_event_peak_abs_edit.setText("invalid_non_numeric")
    # Record current files to verify no outputs are written
    initial_files = list(tmp_path.glob("**/*"))
    window._guided_feature_event_apply_btn.click()
    
    assert window._guided_new_analysis_feature_event_profile_status == "invalid"
    # Verify status label contains error message
    assert "invalid" in window._guided_feature_event_status_label.text().lower() or "must be a number" in window._guided_feature_event_status_label.text().lower()
    # Verify no files were written
    current_files = list(tmp_path.glob("**/*"))
    assert len(current_files) == len(initial_files)
    # The applied profile itself is NOT updated/overwritten with invalid values (still has previous applied values or defaults)
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.123
    
    # Case I: GUI Baseline change marks applied profile stale
    # Restore valid value in widget first
    window._guided_feature_event_peak_abs_edit.setText("0.123")
    window._guided_feature_event_apply_btn.click()
    assert window._guided_new_analysis_feature_event_profile_status == "applied"
    
    # Create another custom config representing baseline drift
    another_cfg_path = tmp_path / "another_config.yaml"
    another_cfg_path.write_text("event_signal: delta_f\nsignal_excursion_polarity: positive\npeak_pre_filter: none", encoding="utf-8")
    
    # Change active baseline config source path in Full Control
    window._config_path.setText(str(another_cfg_path))
    # Trigger sync manually to simulate UI panel refresh/step control sync
    window._sync_guided_feature_event_editor_to_current_run()
    
    assert window._guided_new_analysis_feature_event_profile_status == "stale"
    assert "active baseline config source path changed" in window._guided_new_analysis_feature_event_profile_stale_reasons
    # Values are preserved, not overwritten with defaults
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.123
    
    # Case J: GUI Completed-run feature/event behavior still works (separate state)
    # Change workflow mode to open_results
    window._set_guided_workflow_mode("open_results")
    assert window._guided_new_analysis_feature_event_profile_status == "stale"  # new_analysis state remains stale
    
    # Sync in open_results mode should reset or load for current run (which is empty, so should reset to defaults)
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    assert window._guided_feature_event_status_label.text() == "No draft feature/event profile applied."


def test_new_analysis_feature_event_forced_refresh_and_clear_rules(window, tmp_path, monkeypatch):
    custom_cfg_path = tmp_path / "custom_config.yaml"
    custom_cfg_content = """
event_signal: delta_f
signal_excursion_polarity: negative
peak_threshold_method: absolute
peak_threshold_abs: 0.123
"""
    custom_cfg_path.write_text(custom_cfg_content, encoding="utf-8")
    
    # Configure custom config in Full Control
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(custom_cfg_path))
    
    # Enter new_analysis mode and configure setup
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    
    # Go to Draft Plan step to initialize feature/event editor defaults (default_initialized)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    assert window._guided_new_analysis_feature_event_profile_status == "default_initialized"
    
    # 1. Test: forced refresh (force=True) on default_initialized when widgets are NOT edited should reload defaults
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    assert window._guided_new_analysis_feature_event_profile_status == "default_initialized"
    
    # 2. Test: forced refresh (force=True) on default_initialized when widgets ARE edited must NOT overwrite unapplied edits
    # Edit the widget value
    window._guided_feature_event_peak_abs_edit.setText("0.999")
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    # The widget text must still be the user's unapplied edit "0.999", not reset to defaults
    assert window._guided_feature_event_peak_abs_edit.text() == "0.999"
    assert window._guided_new_analysis_feature_event_profile_status == "default_initialized"
    
    # Apply to move status to "applied"
    window._guided_feature_event_apply_btn.click()
    assert window._guided_new_analysis_feature_event_profile_status == "applied"
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.999
    
    # 3. Test: forced refresh (force=True) on "applied" status must NOT discard stored values
    # Let's type something else in widgets
    window._guided_feature_event_peak_abs_edit.setText("0.111")
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    # The stored values must be preserved, and the editor display must refresh from stored applied state ("0.999", not "0.111")
    assert window._guided_new_analysis_feature_event_profile_status == "applied"
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.999
    assert window._guided_feature_event_peak_abs_edit.text() == "0.999"
    
    # 4. Test: forced refresh (force=True) on "stale" status must NOT discard stored values
    # Force baseline config change to mark stale
    another_cfg_path = tmp_path / "another_config.yaml"
    another_cfg_path.write_text("event_signal: dff\nsignal_excursion_polarity: positive", encoding="utf-8")
    window._config_path.setText(str(another_cfg_path))
    
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    assert window._guided_new_analysis_feature_event_profile_status == "stale"
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.999
    assert window._guided_feature_event_peak_abs_edit.text() == "0.999"
    
    # 5. Test: forced refresh (force=True) on "invalid" status must NOT discard stored values
    # Restore config path so we can apply again
    window._config_path.setText(str(custom_cfg_path))
    window._guided_feature_event_apply_btn.click() # Re-applies successfully
    assert window._guided_new_analysis_feature_event_profile_status == "applied"
    
    # Apply invalid value to trigger "invalid" status
    window._guided_feature_event_peak_abs_edit.setText("invalid_non_numeric")
    window._guided_feature_event_apply_btn.click()
    assert window._guided_new_analysis_feature_event_profile_status == "invalid"
    
    # Run force=True sync on invalid status
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    assert window._guided_new_analysis_feature_event_profile_status == "invalid"
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.999 # preserves last valid applied value
    
    # 6. Test: Clear/Reset reloads defaults from the active Full Control baseline config and sets status to default_initialized
    window._guided_feature_event_clear_btn.click()
    assert window._guided_new_analysis_feature_event_profile_status == "default_initialized"
    assert window._guided_feature_event_peak_abs_edit.text() == "0.123" # loaded defaults from active baseline custom_cfg_path
    
    # 7. Test: Summary display shows baseline source, status, and details when present
    window._guided_feature_event_apply_btn.click()
    summary_text = window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event profile status: applied" in summary_text
    assert "Feature/event profile baseline source:" in summary_text
    assert "custom_config.yaml" in summary_text
    assert "Feature/event profile baseline status: custom_config" in summary_text


def test_new_analysis_output_policy_typed_path_is_not_applied(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._refresh_guided_draft_run_plan_preview()

    assert window._guided_new_analysis_output_policy_status == "missing"
    summary_text = window._guided_draft_run_plan_preview_label.text()
    checklist_text = window._guided_draft_run_plan_checklist_label.text()
    assert "Output policy status: missing" in summary_text
    assert "Output destination: fail" in checklist_text
    assert not target.exists()


def test_new_analysis_output_policy_apply_valid_path_stores_state_without_creating_directory(
    window, tmp_path, monkeypatch
):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    before_files = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "applied"
    assert window._guided_new_analysis_output_policy_path == str(target.resolve())
    assert window._guided_new_analysis_output_policy_explicitly_applied is True
    assert "No directories or files were created" in window._guided_output_status_label.text()
    assert "Output policy status: applied" in window._guided_draft_run_plan_preview_label.text()
    assert target.name in window._guided_draft_run_plan_preview_label.text()
    assert "Output destination: pass" in window._guided_draft_run_plan_checklist_label.text()
    assert not target.exists()
    after_files = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert after_files == before_files


def test_new_analysis_output_policy_invalid_paths_are_rejected_without_creating_outputs(
    window, tmp_path, monkeypatch
):
    input_dir, _output_dir = _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._set_guided_workflow_mode("new_analysis")
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    valid_target = parent / "future_run_outputs"
    existing_target = parent / "existing_target"
    existing_target.mkdir()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(existing_target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "already exists" in window._guided_output_status_label.text()

    window._guided_output_path_edit.setText(str(input_dir))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "must not be the same as the source path" in window._guided_output_status_label.text()

    window._guided_output_path_edit.setText(str(Path(input_dir) / "inside_input_outputs"))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "source/input folder" in window._guided_output_status_label.text()
    assert not (Path(input_dir) / "inside_input_outputs").exists()

    containing_target = tmp_path / "container_for_input"
    nested_source = containing_target / "raw_input"
    nested_source.mkdir(parents=True)
    nested_output = containing_target / "future_outputs"
    window._guided_output_path_edit.setText(str(nested_output))
    window._guided_input_dir_edit.setText(str(nested_source))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "Source/input folder must not be inside the output" in window._guided_output_status_label.text()
    assert not nested_output.exists()
    window._guided_input_dir_edit.setText(str(input_dir))

    window._guided_output_path_edit.setText(str(valid_target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "applied"
    assert not valid_target.exists()

    previous_path = window._guided_new_analysis_output_policy_path
    window._guided_output_path_edit.setText(str(existing_target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert window._guided_new_analysis_output_policy_path == previous_path


def test_new_analysis_output_policy_rejects_diagnostic_cache_overlap(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)

    target = cache_path / "future_run_outputs"
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "protected output/source root" in window._guided_output_status_label.text()
    assert not target.exists()

    preview_target = cache_path / "_guided_workflow" / "previews" / "future_outputs"
    window._guided_output_path_edit.setText(str(preview_target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "protected output/source root" in window._guided_output_status_label.text()
    assert not preview_target.exists()

    signal_target = cache_path / "_guided_workflow" / "signal_only_f0_diagnostics" / "future_outputs"
    window._guided_output_path_edit.setText(str(signal_target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "protected output/source root" in window._guided_output_status_label.text()
    assert not signal_target.exists()


def test_new_analysis_output_policy_marks_stale_when_context_changes(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    input_dir, _output_dir = _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "applied"

    new_input = tmp_path / "new_raw_input"
    new_input.mkdir()
    window._guided_input_dir_edit.setText(str(new_input))
    window._refresh_guided_draft_run_plan_preview()

    assert window._guided_new_analysis_output_policy_status == "stale"
    assert "input source path changed" in window._guided_new_analysis_output_policy_stale_reasons
    assert "Output policy status: stale" in window._guided_draft_run_plan_preview_label.text()
    assert "Output destination: fail" in window._guided_draft_run_plan_checklist_label.text()
    assert not target.exists()

    window._guided_input_dir_edit.setText(str(input_dir))
    window._refresh_guided_draft_run_plan_preview()

    assert window._guided_new_analysis_output_policy_status == "stale"
    assert "Output policy status: stale" in window._guided_draft_run_plan_preview_label.text()

    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "applied"
    assert window._guided_new_analysis_output_policy_stale_reasons == []
    assert not target.exists()


def test_new_analysis_output_policy_apply_requires_valid_source_context(window, tmp_path):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_input_dir_edit.setText("")
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "Raw input/source path is required" in window._guided_output_status_label.text()
    assert window._guided_new_analysis_output_policy_explicitly_applied is False
    assert not target.exists()

    missing_source = tmp_path / "missing_raw_source"
    window._guided_input_dir_edit.setText(str(missing_source))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "does not exist or is not a directory" in window._guided_output_status_label.text()
    assert window._guided_new_analysis_output_policy_explicitly_applied is False
    assert not target.exists()


def test_new_analysis_output_policy_marks_stale_when_target_appears(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "applied"

    target.mkdir()
    window._refresh_guided_draft_run_plan_preview()

    assert window._guided_new_analysis_output_policy_status == "stale"
    assert any("already exists" in reason for reason in window._guided_new_analysis_output_policy_stale_reasons)
    assert "Output destination: fail" in window._guided_draft_run_plan_checklist_label.text()


def test_new_analysis_output_policy_clear_removes_state(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "applied"

    window._guided_output_clear_btn.click()

    assert window._guided_new_analysis_output_policy_status == "missing"
    assert window._guided_new_analysis_output_policy_path is None
    assert window._guided_output_path_edit.text() == ""
    assert "Output policy status: missing" in window._guided_draft_run_plan_preview_label.text()
    assert not target.exists()


def test_guided_new_analysis_compilation_bindings_4J11i(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)

    # 1. Output base path mapping
    window._guided_output_dir_edit.setText("C:/guided_test_output")
    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.output_base_path == "C:/guided_test_output"

    # 2. Rejects completed-run-scoped _guided_strategy_choices mapping
    window._guided_correction_intent = None
    window._guided_strategy_choices = {
        ("cache_key", "ROI0"): {"strategy": "robust_global_event_reject", "source_type": "diagnostic_cache"}
    }
    plan_with_choices = window._build_guided_new_analysis_draft_plan()
    assert plan_with_choices.global_correction_strategy is None
    assert plan_with_choices.dynamic_fit_mode is None

    # 3. Dynamic fit mode mapping from _guided_correction_intent
    window._guided_correction_intent = "Global Linear Regression"
    plan_with_intent = window._build_guided_new_analysis_draft_plan()
    assert plan_with_intent.global_correction_strategy == "dynamic_fit"
    assert plan_with_intent.dynamic_fit_mode == "global_linear_regression"

    # 4. Signal-only F0 mapping from _guided_correction_intent
    from gui.main_window import GUIDED_SIGNAL_ONLY_F0_CARD
    window._guided_correction_intent = GUIDED_SIGNAL_ONLY_F0_CARD
    plan_signal_only = window._build_guided_new_analysis_draft_plan()
    assert plan_signal_only.global_correction_strategy == "signal_only_f0"
    assert plan_signal_only.dynamic_fit_mode is None


def test_guided_new_analysis_preview_request_checks_4J11m(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)

    # Base state: default layout refresh without base path or strategy
    window._guided_output_dir_edit.setText("")
    window._guided_correction_intent = None
    window._refresh_guided_draft_run_plan_preview()
    text = window._guided_draft_run_plan_preview_label.text()

    # 1. Preview includes local check section
    assert "Local setup verification (in-memory only):" in text

    # 2. Missing output_base_path appears as a local blocking issue
    assert "Blocking local setup issues:" in text
    assert "[missing_output_base]" in text

    # 3. Missing strategy appears as a local blocking issue
    assert "[unsupported_correction_strategy]" in text

    # 4. Signal-Only F0 intent is supported, but still needs per-ROI choices.
    from gui.main_window import GUIDED_SIGNAL_ONLY_F0_CARD
    window._guided_correction_intent = GUIDED_SIGNAL_ONLY_F0_CARD
    window._refresh_guided_draft_run_plan_preview()
    signal_only_text = window._guided_draft_run_plan_preview_label.text()
    assert "[unsupported_correction_strategy]" not in signal_only_text
    assert "[missing_strategy_choice_for_included_roi]" in signal_only_text

    # 5. dynamic_fit with allowed mode passes local checks once output path is valid
    window._guided_correction_intent = "Global Linear Regression"
    window._guided_output_dir_edit.setText(str(tmp_path / "valid_output"))
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")
    window._refresh_guided_draft_run_plan_preview()
    text_passed = window._guided_draft_run_plan_preview_label.text()
    assert "Draft plan local checks: Passed" in text_passed
    assert "Draft request fingerprint:" in text_passed

    # 6. Preview wording does not include unsafe terms
    for term in ["Backend validation passed", "Ready to run", "Plan validated", "Preflight complete"]:
        assert term not in text_passed

    # 7. No Run button or Full Control run state is enabled by local checks
    window._new_run_btn.setEnabled(False)
    window._refresh_guided_draft_run_plan_preview()
    assert not window._new_run_btn.isEnabled()

    # 8. No files or directories are created by preview refresh
    output_dir = tmp_path / "valid_output"
    assert not output_dir.exists()
