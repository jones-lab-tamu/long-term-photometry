from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
    GuidedBackendValidationWorkflowIssue,
    GuidedBackendValidationWorkflowOutcome,
)
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
    assert context.parser_contract.time_column_candidates == ("Time(s)",)
    assert context.validator_contract is window._guided_backend_validator_contract
    assert context.validator_contract.validator_capability_version
    assert context.additional_protected_roots == (
        ("completed_run", r"C:\completed"),
    )
    assert context.revision == window._guided_backend_validation_revision
    assert window._input_dir.text() == before_input
    assert window._output_dir.text() == before_output


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


def test_no_guided_validate_button_or_panel_was_added(window):
    assert not hasattr(window, "_guided_validate_btn")
    assert not hasattr(window, "_guided_backend_validation_panel")
