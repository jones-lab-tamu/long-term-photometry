from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QPushButton

from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
    GuidedBackendValidationWorkflowIssue,
    GuidedBackendValidationWorkflowOutcome,
)
import photometry_pipeline.guided_backend_validation_workflow as workflow
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _refused_outcome(*, stale: bool = False):
    return GuidedBackendValidationWorkflowOutcome(
        status="internal_error",
        accepted_for_backend_validation=False,
        run_authorization=False,
        request_identity=None,
        validation_result=None,
        compile_result=None,
        materialization_result=None,
        blocking_issues=(
            GuidedBackendValidationWorkflowIssue(
                stage="workflow",
                category="workflow_internal_error",
                section="workflow",
                message="failed",
            ),
        ),
        user_summary="failed",
        stale=stale,
    )


def test_context_adapter_captures_only_backend_neutral_inputs(
    window,
    monkeypatch,
):
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=r"C:\source",
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    monkeypatch.setattr(
        window,
        "_build_guided_new_analysis_draft_plan",
        lambda: draft,
    )
    monkeypatch.setattr(
        window,
        "_current_guided_completed_run_dir",
        lambda: r"C:\completed",
    )
    monkeypatch.setattr(
        window,
        "_on_validate",
        lambda: pytest.fail("Full Control Validate must not be called"),
    )
    monkeypatch.setattr(
        window,
        "_on_run",
        lambda: pytest.fail("Full Control Run must not be called"),
    )
    before_input = window._input_dir.text()
    before_output = window._output_dir.text()

    context = window._capture_guided_backend_validation_context()

    assert isinstance(context, GuidedBackendValidationGuiContext)
    assert context.draft is draft
    assert context.parser_contract is (
        window._guided_backend_validation_parser_contract
    )
    # 4J16k18: extended to also recognize "TimeStamp" as a real-world RWD
    # time-column name, alongside the original "Time(s)".
    assert context.parser_contract.time_column_candidates == (
        "Time(s)",
        "TimeStamp",
    )
    assert context.validator_contract is window._guided_backend_validator_contract
    assert context.validator_contract.validator_capability_version
    assert context.additional_protected_roots == (
        ("completed_run", r"C:\completed"),
    )
    assert context.revision == window._guided_backend_validation_revision
    assert window._input_dir.text() == before_input
    assert window._output_dir.text() == before_output


def _configure_minimal_npm_contract_state(window):
    window._guided_input_dir_edit.setText(r"C:\npm-source")
    window._guided_format_combo.setCurrentText("npm")
    window._discovery_cache = {"resolved_format": "npm"}
    window._guided_acquisition_mode_combo.setCurrentText("intermittent")
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")


def test_npm_dataset_contract_candidate_does_not_call_legacy_inference(
    window,
    monkeypatch,
):
    _configure_minimal_npm_contract_state(window)

    def fail_legacy_inference(_format):
        raise AssertionError("legacy GUI NPM inference must not be called")

    monkeypatch.setattr(
        window,
        "_infer_dataset_contract_overrides",
        fail_legacy_inference,
    )

    candidate = window._guided_new_analysis_dataset_contract_candidate()

    assert candidate.status == "inferred"
    assert candidate.format_specific["dataset_semantics_source"] == "configured"
    assert candidate.format_specific[
        "dataset_semantics_inferred_from_selected_input"
    ] is False
    assert candidate.contract_values["npm_led_col"] == (
        window._active_baseline_config().npm_led_col
    )


def test_npm_dataset_panel_uses_scientist_facing_settings_language(
    window,
):
    _configure_minimal_npm_contract_state(window)
    window._refresh_guided_dataset_contract_panel()

    visible_text = "\n".join(
        (
            window._guided_dataset_contract_status_label.text(),
            window._guided_dataset_contract_candidate_label.text(),
            window._guided_dataset_contract_stored_label.text(),
        )
    ).lower()
    for term in (
        "adapter",
        "backend",
        "contract",
        "materialization",
        "normalized",
        "parser policy",
        "provenance",
        "authorization",
        "identity",
        "digest",
        "schema",
        "canonical",
    ):
        assert term not in visible_text
    assert (
        "npm settings: the selected recording will be checked against the current "
        "npm import settings during setup check."
        in visible_text
    )


def test_npm_context_builds_parser_from_applied_settings_without_legacy_inference(
    window,
    monkeypatch,
):
    _configure_minimal_npm_contract_state(window)
    monkeypatch.setattr(
        window,
        "_infer_dataset_contract_overrides",
        lambda _format: pytest.fail("legacy GUI NPM inference was called"),
    )
    candidate = window._guided_new_analysis_dataset_contract_candidate()
    snapshot = replace(candidate, status="applied", explicitly_applied=True)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=r"C:\npm-source",
        input_format="npm",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        dataset_contract_snapshot=snapshot,
    )
    monkeypatch.setattr(
        window,
        "_build_guided_new_analysis_draft_plan",
        lambda: draft,
    )

    context = window._capture_guided_backend_validation_context()

    assert context.parser_contract.npm_led_col == (
        window._active_baseline_config().npm_led_col
    )
    assert context.parser_contract.npm_region_prefix == (
        window._active_baseline_config().npm_region_prefix
    )
    assert context.parser_contract.target_fs_hz == (
        window._active_baseline_config().target_fs_hz
    )


def test_applied_npm_parser_setting_change_marks_dataset_contract_stale(
    window,
    monkeypatch,
):
    _configure_minimal_npm_contract_state(window)
    candidate = window._guided_new_analysis_dataset_contract_candidate()
    window._guided_new_analysis_dataset_contract_snapshot = replace(
        candidate,
        status="applied",
        explicitly_applied=True,
    )
    baseline = window._active_baseline_config()
    monkeypatch.setattr(
        window,
        "_active_baseline_config",
        lambda: replace(baseline, npm_led_col="ChangedLedState"),
    )

    window._refresh_guided_new_analysis_dataset_contract_staleness()

    snapshot = window._guided_new_analysis_dataset_contract_snapshot
    assert snapshot.status == "stale"
    assert "NPM import settings changed" in snapshot.stale_reasons


def test_context_adapter_uses_no_additional_root_without_completed_run(
    window,
    monkeypatch,
):
    monkeypatch.setattr(
        window,
        "_build_guided_new_analysis_draft_plan",
        GuidedNewAnalysisDraftPlan,
    )
    monkeypatch.setattr(
        window,
        "_current_guided_completed_run_dir",
        lambda: "",
    )
    context = window._capture_guided_backend_validation_context()
    assert context.additional_protected_roots == ()


def test_context_model_is_frozen(window, monkeypatch):
    monkeypatch.setattr(
        window,
        "_build_guided_new_analysis_draft_plan",
        GuidedNewAnalysisDraftPlan,
    )
    monkeypatch.setattr(
        window,
        "_current_guided_completed_run_dir",
        lambda: "",
    )
    context = window._capture_guided_backend_validation_context()
    with pytest.raises(FrozenInstanceError):
        context.revision = 10  # type: ignore[misc]


def test_invalidation_increments_revision_and_marks_outcome_stale(window):
    window._guided_backend_validation_revision = 7
    window._guided_backend_validation_outcome = _refused_outcome()
    window._guided_backend_validation_outcome_revision = 7

    window._invalidate_guided_backend_validation("ROI changed")

    assert window._guided_backend_validation_revision == 8
    assert window._guided_backend_validation_outcome.stale is True
    assert window._guided_backend_validation_outcome_revision is None
    assert window._guided_backend_validation_stale_reason == "ROI changed"
    assert window._is_guided_backend_validation_outcome_current() is False


def test_outcome_currentness_requires_matching_revision_and_not_stale(window):
    window._guided_backend_validation_revision = 3
    window._guided_backend_validation_outcome = _refused_outcome()
    window._guided_backend_validation_outcome_revision = 3
    assert window._is_guided_backend_validation_outcome_current() is True

    window._guided_backend_validation_outcome_revision = 2
    assert window._is_guided_backend_validation_outcome_current() is False

    window._guided_backend_validation_outcome_revision = 3
    window._guided_backend_validation_outcome = _refused_outcome(stale=True)
    assert window._is_guided_backend_validation_outcome_current() is False

    window._guided_backend_validation_outcome = None
    assert window._is_guided_backend_validation_outcome_current() is False


def test_source_widget_change_invalidates_backend_validation(window):
    before = window._guided_backend_validation_revision
    window._guided_input_dir_edit.setText(r"C:\changed-source")
    assert window._guided_backend_validation_revision > before


def test_dataset_clear_handler_invalidates_backend_validation(window):
    before = window._guided_backend_validation_revision
    window._on_guided_clear_dataset_contract()
    assert window._guided_backend_validation_revision > before


def _accepted_outcome():
    return GuidedBackendValidationWorkflowOutcome(
        status="validator_accepted",
        accepted_for_backend_validation=True,
        run_authorization=False,
        request_identity="a" * 64,
        validation_result=SimpleNamespace(accepted=True),
        compile_result=SimpleNamespace(),
        materialization_result=SimpleNamespace(),
        blocking_issues=(),
        user_summary="accepted",
    )


def _failure_outcome(status: str):
    return GuidedBackendValidationWorkflowOutcome(
        status=status,
        accepted_for_backend_validation=False,
        run_authorization=False,
        request_identity=None,
        validation_result=None,
        compile_result=None,
        materialization_result=None,
        blocking_issues=(
            GuidedBackendValidationWorkflowIssue(
                stage="test",
                category="test_category",
                section="test_section",
                message="test message",
                detail_code="test_detail",
            ),
        ),
        user_summary="failed",
    )


def test_guided_validate_and_disabled_run_widgets_exist(window):
    assert window._guided_backend_validate_btn.text() == (
        "Check my setup"
    )
    assert window._guided_backend_validation_status_label is not None
    assert window._guided_backend_validation_details_label is not None
    assert window._guided_run_btn.text() == "Run Guided Analysis"
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_run_readiness_label is not None
    assert not hasattr(window, "_guided_validation_artifact_link")


def test_npm_accepted_setup_check_uses_approved_message_and_keeps_run_disabled(
    window,
):
    outcome = replace(
        _accepted_outcome(),
        compile_result=SimpleNamespace(
            request=SimpleNamespace(
                source=SimpleNamespace(source_format="npm"),
            )
        ),
    )
    window._guided_backend_validation_outcome = outcome
    window._guided_backend_validation_outcome_revision = (
        window._guided_backend_validation_revision
    )
    window._refresh_guided_backend_validation_display()

    expected = (
        "This NPM recording setup was checked successfully. Running NPM analyses "
        "is not available yet."
    )
    assert window._guided_backend_validation_status_label.text() == expected
    assert window._guided_run_readiness_label.text() == expected
    assert window._guided_run_btn.isEnabled() is False


def test_npm_failure_details_hide_internal_section_names(window):
    outcome = _failure_outcome("materialization_failed")
    issue = replace(
        outcome.blocking_issues[0],
        section="normalized_recording",
        message=(
            "The app could not determine how to read this NPM recording from "
            "the current settings. The ROI columns are not consistent across "
            "all selected NPM sessions."
        ),
        detail_code="npm_roi_inventory_mismatch",
    )
    window._guided_backend_validation_outcome = replace(
        outcome,
        blocking_issues=(issue,),
    )
    window._guided_backend_validation_outcome_revision = (
        window._guided_backend_validation_revision
    )
    window._refresh_guided_backend_validation_display()

    details = window._guided_backend_validation_details_label.text().lower()
    assert "normalized" not in details
    assert "category:" not in details
    assert "section:" not in details
    assert "roi columns are not consistent" in details
    assert "rerun setup check" in details


def test_validate_button_calls_workflow_through_module_namespace(
    window,
    monkeypatch,
):
    draft = GuidedNewAnalysisDraftPlan()
    context = GuidedBackendValidationGuiContext(
        draft=draft,
        parser_contract=window._guided_backend_validation_parser_contract,
        additional_protected_roots=(),
        validator_contract=window._guided_backend_validator_contract,
        revision=window._guided_backend_validation_revision,
    )
    captured = {}
    monkeypatch.setattr(
        window,
        "_capture_guided_backend_validation_context",
        lambda: context,
    )

    def validate(draft_arg, **kwargs):
        captured["draft"] = draft_arg
        captured.update(kwargs)
        return _accepted_outcome()

    monkeypatch.setattr(
        workflow,
        "validate_current_guided_draft_for_backend",
        validate,
    )
    for name in ("_on_validate", "_on_run", "_build_run_spec", "_build_argv"):
        monkeypatch.setattr(
            window,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"{_name} must not be called"
            ),
        )

    window._guided_backend_validate_btn.click()

    assert captured["draft"] is draft
    assert captured["parser_contract"] is context.parser_contract
    assert captured["validator_contract"] is context.validator_contract
    assert window._guided_backend_validation_outcome.status == (
        "validator_accepted"
    )
    assert window._guided_backend_validation_outcome_revision == (
        context.revision
    )
    assert "passed for the current Guided setup" in (
        window._guided_backend_validation_status_label.text()
    )
    assert "does not authorize or start a run" in (
        window._guided_backend_validation_status_label.text()
    )
    assert window._guided_backend_validation_outcome.run_authorization is False


@pytest.mark.parametrize(
    "status,expected",
    [
        ("materialization_failed", "setup is incomplete or stale"),
        ("compile_failed", "could not be processed"),
        ("validator_refused", "found a problem with the current Guided setup"),
        ("internal_error", "could not complete safely"),
    ],
)
def test_refused_and_error_outcome_display(
    window,
    status: str,
    expected: str,
):
    window._guided_backend_validation_outcome = _failure_outcome(status)
    window._guided_backend_validation_outcome_revision = (
        window._guided_backend_validation_revision
    )
    window._refresh_guided_backend_validation_display()
    assert expected in window._guided_backend_validation_status_label.text()
    details = window._guided_backend_validation_details_label.text()
    assert "Message: test message" in details
    assert "Category:" not in details
    assert "Section:" not in details
    assert "Detail code:" not in details
    assert "Guided Run is not available for this configuration yet" in details


def test_stale_accepted_outcome_never_displays_current_acceptance(window):
    window._guided_backend_validation_outcome = _accepted_outcome()
    window._guided_backend_validation_outcome_revision = (
        window._guided_backend_validation_revision
    )
    window._invalidate_guided_backend_validation("ROI changed")
    status = window._guided_backend_validation_status_label.text()
    assert "out of date because the setup changed" in status
    assert "passed for the current Guided setup" not in status


def test_context_capture_exception_is_safe(window, monkeypatch):
    monkeypatch.setattr(
        window,
        "_capture_guided_backend_validation_context",
        lambda: (_ for _ in ()).throw(RuntimeError("sensitive traceback")),
    )
    monkeypatch.setattr(
        window,
        "_on_validate",
        lambda: pytest.fail("Full Control Validate must not be called"),
    )
    window._guided_backend_validate_btn.click()
    assert window._guided_backend_validation_outcome.status == "internal_error"
    text = window._guided_backend_validation_status_label.text()
    assert "could not complete safely" in text
    assert "traceback" not in text
    assert window._guided_backend_validation_outcome.run_authorization is False


def test_validate_click_calls_no_write_run_or_allocation_api(
    window,
    monkeypatch,
):
    context = GuidedBackendValidationGuiContext(
        draft=GuidedNewAnalysisDraftPlan(),
        parser_contract=window._guided_backend_validation_parser_contract,
        additional_protected_roots=(),
        validator_contract=window._guided_backend_validator_contract,
        revision=window._guided_backend_validation_revision,
    )
    monkeypatch.setattr(
        window,
        "_capture_guided_backend_validation_context",
        lambda: context,
    )
    monkeypatch.setattr(
        workflow,
        "validate_current_guided_draft_for_backend",
        lambda *_args, **_kwargs: _accepted_outcome(),
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("write/run/allocation API is prohibited")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(Path, "touch", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    monkeypatch.setattr(window, "_on_validate", fail)
    monkeypatch.setattr(window, "_on_run", fail)
    monkeypatch.setattr(window, "_build_run_spec", fail)
    monkeypatch.setattr(window, "_build_argv", fail)
    monkeypatch.setattr(window._runner, "start", fail)

    window._guided_backend_validate_btn.click()
    assert window._guided_backend_validation_outcome.status == (
        "validator_accepted"
    )


def test_revision_change_during_workflow_stores_stale_outcome(
    window,
    monkeypatch,
):
    revision = window._guided_backend_validation_revision
    context = GuidedBackendValidationGuiContext(
        draft=GuidedNewAnalysisDraftPlan(),
        parser_contract=window._guided_backend_validation_parser_contract,
        additional_protected_roots=(),
        validator_contract=window._guided_backend_validator_contract,
        revision=revision,
    )
    monkeypatch.setattr(
        window,
        "_capture_guided_backend_validation_context",
        lambda: context,
    )

    def validate(*_args, **_kwargs):
        window._guided_backend_validation_revision += 1
        return _accepted_outcome()

    monkeypatch.setattr(
        workflow,
        "validate_current_guided_draft_for_backend",
        validate,
    )
    window._guided_backend_validate_btn.click()
    assert window._guided_backend_validation_outcome.stale is True
    assert "out of date because the setup changed" in (
        window._guided_backend_validation_status_label.text()
    )


def test_run_step_avoids_developer_facing_wording(window, qapp):
    """4J16k26: Run step must read as a plain setup check, not as a backend
    validation request against an unspecified 'supported subset'."""
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Run")
    )
    window._refresh_guided_backend_validation_display()
    qapp.processEvents()

    run_widget = window._guided_workflow_stack.widget(
        list(GUIDED_WORKFLOW_STEPS).index("Run")
    )
    texts = []
    for widget in run_widget.findChildren(QGroupBox):
        texts.append(widget.title())
    for widget in run_widget.findChildren(QLabel):
        texts.append(widget.text())
    for widget in run_widget.findChildren(QPushButton):
        texts.append(widget.text())
    visible_text = " ".join(t for t in texts if t)
    lowered = visible_text.lower()

    forbidden = (
        "backend",
        "guided request",
        "first execution subset",
        "supported subset",
    )
    found = [term for term in forbidden if term in lowered]
    assert found == []

    assert "Check setup" in visible_text
    assert "Check my setup" in visible_text
    assert window._guided_backend_validate_btn.text() == "Check my setup"
