"""CR1-0 production capability gate for Guided acquisition modes."""

from __future__ import annotations

import inspect

import pytest
from PySide6.QtWidgets import QApplication

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


def test_guided_capability_contract_is_narrow_immutable_and_not_environment_driven(
    monkeypatch,
):
    monkeypatch.setenv("GUIDED_CONTINUOUS_ENABLED", "1")

    assert guided_capabilities.GUIDED_PRODUCTION_ACQUISITION_MODES == (
        "intermittent",
    )
    assert isinstance(
        guided_capabilities.GUIDED_PRODUCTION_ACQUISITION_MODES,
        tuple,
    )
    assert guided_capabilities.is_guided_production_acquisition_mode(
        "intermittent"
    )
    assert not guided_capabilities.is_guided_production_acquisition_mode(
        "continuous"
    )
    source = inspect.getsource(guided_capabilities)
    assert "getenv" not in source
    assert "environ" not in source


def test_guided_selector_uses_capability_contract_and_full_control_stays_separate(
    window,
):
    expected = guided_capabilities.GUIDED_PRODUCTION_ACQUISITION_MODES

    assert _combo_values(window._guided_acquisition_mode_combo) == expected
    assert window._guided_acquisition_mode_combo.currentData() == "intermittent"
    assert window._guided_acquisition_mode_combo.findData("continuous") == -1
    assert window._acquisition_mode_combo.findData("continuous") >= 0

    full_continuous = window._acquisition_mode_combo.findData("continuous")
    window._acquisition_mode_combo.setCurrentIndex(full_continuous)

    assert window._selected_acquisition_mode() == "continuous"
    assert _combo_values(window._guided_acquisition_mode_combo) == expected
    assert window._guided_selected_acquisition_mode() == "intermittent"
    assert window._guided_setup_summary_state()["acquisition_mode"] == (
        "intermittent"
    )


def test_unsupported_guided_widget_state_fails_closed(window):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")

    window._guided_acquisition_mode_combo.addItem(
        "Injected unsupported mode",
        "continuous",
    )
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("continuous")
    )

    ready, reason = window._guided_recording_structure_readiness()
    draft = window._build_guided_new_analysis_draft_plan()

    assert ready is False
    assert reason == "Select an acquisition mode to continue."
    assert draft.acquisition_mode == "continuous"
    assert draft.acquisition_structure_status == "unknown"


@pytest.mark.parametrize("source_format", ["rwd", "npm"])
def test_guided_intermittent_setup_remains_available(window, source_format):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText(source_format)
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
    assert window._guided_selected_acquisition_mode() == "intermittent"
    revision = window._guided_backend_validation_revision

    window._sync_guided_setup_from_full()

    assert window._guided_selected_acquisition_mode() == "intermittent"
    assert window._guided_backend_validation_revision == revision


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
