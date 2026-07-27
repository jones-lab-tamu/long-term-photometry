"""CR1-0 production capability gate for Guided acquisition modes."""

from __future__ import annotations

import inspect

import pytest
from PySide6.QtWidgets import QApplication, QFormLayout

from gui.main_window import MainWindow
from photometry_pipeline import guided_capabilities
from photometry_pipeline.guided_backend_validation_request import (
    GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION,
    GUIDED_BACKEND_VALIDATION_SCOPE,
    GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION,
    GuidedBackendValidationCompileFailure,
    GuidedBackendValidationMaterializedFacts,
    GuidedBackendValidatorContract,
    compile_guided_backend_validation_request,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan


pytestmark = pytest.mark.usefixtures("no_real_modals")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _combo_values(combo) -> tuple[str, ...]:
    return tuple(str(combo.itemData(index)) for index in range(combo.count()))


def _timeline_row_visibility(window, field_name: str):
    field = getattr(window, field_name)
    form = window._guided_timeline_group.layout()
    row, _role = form.getWidgetPosition(field)
    label_item = form.itemAt(row, QFormLayout.LabelRole)
    label = label_item.widget() if label_item is not None else None
    assert label is not None
    return label.text(), label.isVisible(), field.isVisible()


def test_guided_capability_contract_is_narrow_immutable_and_not_environment_driven(
    monkeypatch,
):
    monkeypatch.setenv("GUIDED_CONTINUOUS_ENABLED", "1")

    # CR1-E3 added continuous to the production contract, at the point where
    # Guided can genuinely select, check, prepare, execute, complete, and
    # open a continuous-RWD recording end to end. The contract is still a
    # closed, immutable list -- not an environment switch.
    assert guided_capabilities.GUIDED_PRODUCTION_ACQUISITION_MODES == (
        "intermittent",
        "continuous",
    )
    assert isinstance(
        guided_capabilities.GUIDED_PRODUCTION_ACQUISITION_MODES,
        tuple,
    )
    assert guided_capabilities.is_guided_production_acquisition_mode(
        "intermittent"
    )
    assert guided_capabilities.is_guided_production_acquisition_mode(
        "continuous"
    )
    assert not guided_capabilities.is_guided_production_acquisition_mode(
        "unsupported_mode"
    )
    source = inspect.getsource(guided_capabilities)
    assert "getenv" not in source
    assert "environ" not in source


def test_guided_selector_uses_capability_contract_and_full_control_stays_separate(
    window,
):
    from gui.main_window import GUIDED_STRUCTURE_CHOICE_AUTO

    # CR1-F1-B: the selector offers "Detect automatically" ahead of the
    # production acquisition modes, and defaults to it. Auto-detect is a
    # Select-data choice, not an acquisition mode, so the capability contract
    # itself still lists only real modes.
    expected = (
        GUIDED_STRUCTURE_CHOICE_AUTO,
    ) + guided_capabilities.GUIDED_PRODUCTION_ACQUISITION_MODES

    assert _combo_values(window._guided_acquisition_mode_combo) == expected
    assert window._guided_acquisition_mode_combo.currentData() == (
        GUIDED_STRUCTURE_CHOICE_AUTO
    )
    assert window._guided_acquisition_mode_combo.findData("continuous") >= 0
    assert window._acquisition_mode_combo.findData("continuous") >= 0

    # Guided and Full Control remain separate selections: changing Full
    # Control does not change what Guided is planning.
    full_continuous = window._acquisition_mode_combo.findData("continuous")
    window._acquisition_mode_combo.setCurrentIndex(full_continuous)

    assert window._selected_acquisition_mode() == "continuous"
    assert _combo_values(window._guided_acquisition_mode_combo) == expected
    assert window._guided_selected_acquisition_mode() == (
        GUIDED_STRUCTURE_CHOICE_AUTO
    )
    # Nothing has been discovered, so the structure in force is still the
    # historical default and Guided is unaffected by Full Control.
    assert window._guided_effective_acquisition_mode() == "intermittent"
    assert window._guided_setup_summary_state()["acquisition_mode"] == (
        "intermittent"
    )


def test_guided_timeline_controls_use_ordered_default_and_do_not_touch_full_control(
    window,
):
    assert _combo_values(window._guided_timeline_mode_combo) == (
        "fixed_daily_anchor",
        "civil",
        "elapsed",
    )
    assert window._guided_timeline_mode_combo.currentData() == (
        "fixed_daily_anchor"
    )
    assert window._guided_fixed_daily_anchor_clock_edit.text() == "07:00"
    assert window._guided_timeline_help_label.text() == (
        "Fixed daily anchor places each day relative to the selected "
        "circadian-day start.\n"
        "Civil clock uses actual clock time with midnight as the day boundary.\n"
        "Elapsed starts the plot at the first recording."
    )

    full_control_mode = window._timeline_anchor_mode_combo.currentData()
    window._guided_timeline_mode_combo.setCurrentIndex(
        window._guided_timeline_mode_combo.findData("elapsed")
    )
    window._guided_fixed_daily_anchor_clock_edit.setText("06:00")
    window._on_guided_start_setup_new_analysis()

    assert window._guided_timeline_mode_combo.currentData() == (
        "fixed_daily_anchor"
    )
    assert window._guided_fixed_daily_anchor_clock_edit.text() == "07:00"
    assert window._timeline_anchor_mode_combo.currentData() == full_control_mode


def test_guided_timeline_hides_complete_conditional_form_rows(window, tmp_path):
    window.show()
    window._on_guided_start_setup_new_analysis()
    window._on_guided_stepper_row_changed(
        window._guided_step_index("Recording structure")
    )
    window._guided_format_combo.setCurrentText("rwd")
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("continuous")
    )

    assert _timeline_row_visibility(
        window, "_guided_timeline_mode_combo"
    ) == ("Time display:", True, True)
    assert _timeline_row_visibility(
        window, "_guided_fixed_daily_anchor_clock_edit"
    ) == ("Start of plotted day:", True, True)
    assert _timeline_row_visibility(
        window, "_guided_recording_start_clock_edit"
    ) == ("Clock time at recording start:", True, True)

    window._guided_timeline_mode_combo.setCurrentIndex(
        window._guided_timeline_mode_combo.findData("civil")
    )
    assert _timeline_row_visibility(
        window, "_guided_fixed_daily_anchor_clock_edit"
    ) == ("Start of plotted day:", False, False)
    assert _timeline_row_visibility(
        window, "_guided_recording_start_clock_edit"
    ) == ("Clock time at recording start:", True, True)

    window._guided_timeline_mode_combo.setCurrentIndex(
        window._guided_timeline_mode_combo.findData("elapsed")
    )
    assert _timeline_row_visibility(
        window, "_guided_fixed_daily_anchor_clock_edit"
    ) == ("Start of plotted day:", False, False)
    assert _timeline_row_visibility(
        window, "_guided_recording_start_clock_edit"
    ) == ("Clock time at recording start:", False, False)

    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )
    session = tmp_path / "2026_06_30-12_00_00" / "Fluorescence.csv"
    window._discovery_cache = {
        "resolved_format": "rwd",
        "sessions": [{"path": str(session)}],
    }
    window._sync_guided_recording_visibility()
    window._guided_timeline_mode_combo.setCurrentIndex(
        window._guided_timeline_mode_combo.findData("fixed_daily_anchor")
    )

    assert _timeline_row_visibility(
        window, "_guided_fixed_daily_anchor_clock_edit"
    ) == ("Start of plotted day:", True, True)
    assert _timeline_row_visibility(
        window, "_guided_recording_start_clock_edit"
    ) == ("Clock time at recording start:", False, False)

    window._guided_timeline_mode_combo.setCurrentIndex(
        window._guided_timeline_mode_combo.findData("civil")
    )
    assert _timeline_row_visibility(
        window, "_guided_fixed_daily_anchor_clock_edit"
    ) == ("Start of plotted day:", False, False)
    assert _timeline_row_visibility(
        window, "_guided_recording_start_clock_edit"
    ) == ("Clock time at recording start:", False, False)


def test_guided_continuous_selection_produces_a_continuous_draft(window):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText("rwd")
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("continuous")
    )
    window._guided_continuous_window_sec_spin.setValue(600.0)
    window._guided_recording_start_clock_edit.setText("11:00")

    assert window._guided_selected_acquisition_mode() == "continuous"
    draft = window._build_guided_new_analysis_draft_plan()
    assert draft.acquisition_mode == "continuous"
    assert draft.input_format == "rwd"
    assert draft.continuous_window_sec == 600.0
    assert draft.continuous_step_sec == 600.0
    assert draft.execution_intent.timeline_anchor_mode == "fixed_daily_anchor"
    assert draft.execution_intent.fixed_daily_anchor_clock == "07:00"
    assert draft.execution_intent.recording_start_clock == "11:00"
    assert draft.execution_intent.recording_start_clock_source == "user_confirmed"
    # The session-timing questions do not apply to one long recording.
    assert window._guided_sessions_per_hour_edit.isHidden() is True
    assert window._guided_session_duration_edit.isHidden() is True
    assert window._guided_continuous_window_sec_spin.isHidden() is False
    assert window._guided_recording_structure_readiness() == (
        True,
        "Recording structure is ready.",
    )


def test_guided_continuous_timeline_requires_start_clock_and_reviews_mapping(
    window,
):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText("rwd")
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("continuous")
    )

    assert window._guided_fixed_daily_anchor_clock_edit.isHidden() is False
    assert window._guided_recording_start_clock_edit.isHidden() is False
    assert window._guided_recording_structure_readiness() == (
        False,
        "Enter the clock time when this recording began.",
    )

    window._guided_recording_start_clock_edit.setText("11:00")
    assert window._guided_recording_structure_readiness() == (
        True,
        "Recording structure is ready.",
    )
    fixed_plan = window._build_guided_new_analysis_draft_plan()
    assert window._guided_timeline_review_lines(fixed_plan) == [
        "Time display: Fixed daily anchor",
        "Start of plotted day: 07:00",
        "Clock time at recording start: 11:00",
        "First data will appear 4 hours after the plotted day begins.",
    ]

    window._guided_timeline_mode_combo.setCurrentIndex(
        window._guided_timeline_mode_combo.findData("civil")
    )
    civil_plan = window._build_guided_new_analysis_draft_plan()
    assert window._guided_fixed_daily_anchor_clock_edit.isHidden()
    assert window._guided_recording_start_clock_edit.isHidden() is False
    assert window._guided_timeline_review_lines(civil_plan) == [
        "Time display: Civil clock",
        "Clock time at recording start: 11:00",
        "Days begin at midnight.",
    ]

    window._guided_timeline_mode_combo.setCurrentIndex(
        window._guided_timeline_mode_combo.findData("elapsed")
    )
    elapsed_plan = window._build_guided_new_analysis_draft_plan()
    assert window._guided_recording_start_clock_edit.isHidden()
    assert elapsed_plan.execution_intent.recording_start_clock is None
    assert elapsed_plan.execution_intent.recording_start_clock_source == (
        "not_applicable"
    )
    assert window._guided_timeline_review_lines(elapsed_plan) == [
        "Time display: Elapsed from recording start",
        "The first recorded sample will appear at time 0.",
    ]


def test_guided_intermittent_rwd_uses_validated_session_clock_without_editable_start(
    window, tmp_path
):
    session = tmp_path / "2026_06_30-12_00_00" / "Fluorescence.csv"
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText("rwd")
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )
    window._discovery_cache = {
        "resolved_format": "rwd",
        "sessions": [{"path": str(session)}],
    }
    window._sync_guided_recording_visibility()

    assert window._guided_recording_start_clock_edit.isHidden()
    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock"] == "12:00"
    assert values["recording_start_clock_source"] == "validated_metadata"


@pytest.mark.parametrize("source_format", ["npm", "custom_tabular"])
def test_guided_continuous_is_refused_for_non_rwd_input(window, source_format):
    """The complete continuous production path exists only for RWD. The
    older chunked custom_tabular continuous-output workflow must not be
    reachable through this selection."""
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText(source_format)
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("continuous")
    )

    ready, reason = window._guided_recording_structure_readiness()

    assert ready is False
    assert "RWD" in reason
    assert window._maybe_start_guided_continuous_rwd_recording_check() is False
    assert window._maybe_start_guided_continuous_rwd_preparation() is False
    assert window._guided_continuous_rwd_live_draft() is None


def test_unsupported_guided_widget_state_fails_closed(window):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")

    window._guided_acquisition_mode_combo.addItem(
        "Injected unsupported mode",
        "episodic_unsupported",
    )
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("episodic_unsupported")
    )

    ready, reason = window._guided_recording_structure_readiness()
    draft = window._build_guided_new_analysis_draft_plan()

    assert ready is False
    assert reason == "Select an acquisition mode to continue."
    assert draft.acquisition_mode == "episodic_unsupported"
    assert draft.acquisition_structure_status == "unknown"


@pytest.mark.parametrize("source_format", ["rwd", "npm"])
def test_guided_intermittent_setup_remains_available(window, source_format):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText(source_format)
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )
    # No discovery facts are installed in this focused structure test, so use
    # the explicit elapsed-only option instead of claiming a clock start.
    window._guided_timeline_mode_combo.setCurrentIndex(
        window._guided_timeline_mode_combo.findData("elapsed")
    )
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")

    assert window._guided_selected_acquisition_mode() == "intermittent"
    assert window._guided_sessions_per_hour_edit.isHidden() is False
    assert window._guided_session_duration_edit.isHidden() is False
    assert window._guided_recording_structure_readiness() == (
        True,
        "Recording structure is ready.",
    )


def test_supported_intermittent_sync_does_not_spuriously_invalidate_plan(window):
    from gui.main_window import GUIDED_STRUCTURE_CHOICE_AUTO

    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )
    assert window._guided_selected_acquisition_mode() == "intermittent"
    revision = window._guided_backend_validation_revision

    window._sync_guided_setup_from_full()

    assert window._guided_selected_acquisition_mode() == "intermittent"
    assert window._guided_backend_validation_revision == revision
    assert GUIDED_STRUCTURE_CHOICE_AUTO != "intermittent"


def test_direct_continuous_request_remains_refused_by_backend_compiler():
    result = compile_guided_backend_validation_request(
        GuidedNewAnalysisDraftPlan(
            input_format="rwd",
            acquisition_mode="continuous",
        ),
        facts=GuidedBackendValidationMaterializedFacts(
            complete_for_compilation=True,
        ),
        validator_contract=GuidedBackendValidatorContract(
            validation_scope=GUIDED_BACKEND_VALIDATION_SCOPE,
            validation_contract_version=(
                GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION
            ),
            validator_capability_version="cr1_0_test.v1",
            supported_subset_rule_version=(
                GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION
            ),
        ),
    )

    assert isinstance(result, GuidedBackendValidationCompileFailure)
    assert result.blocking_issues[0].category == "unsupported_acquisition_mode"
