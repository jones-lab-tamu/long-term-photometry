from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    build_guided_continuous_rwd_review_binding,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisTonicSettingsContract,
)
from tests.test_guided_continuous_rwd_review_binding import _authorities, _draft


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


@pytest.fixture
def state_case(window, tmp_path, monkeypatch):
    source, recording, evaluation = _authorities(tmp_path)
    draft = _draft(source_folder=source.parent)
    current = {"draft": draft}
    monkeypatch.setattr(
        window,
        "_build_guided_new_analysis_draft_plan",
        lambda: current["draft"],
    )
    binding = build_guided_continuous_rwd_review_binding(
        draft,
        recording=recording,
        continuity_evaluation=evaluation,
        current_source_path=source,
    )
    return current, binding


def test_initial_state_keeps_scientific_authority_only_in_optional_binding(window):
    assert window._guided_continuous_rwd_review_binding is None
    continuous_fields = {
        name
        for name in vars(window)
        if name.startswith("_guided_continuous_rwd_")
    }
    assert "_guided_continuous_rwd_review_binding" in continuous_fields
    assert window._guided_continuous_rwd_check_thread is None
    assert window._guided_continuous_rwd_check_worker is None
    assert window._guided_continuous_rwd_check_active_token is None
    assert not any("pending" in name for name in continuous_fields)
    assert not any(
        name != "_guided_continuous_rwd_review_binding"
        and ("recording" in name or "evaluation" in name)
        for name in continuous_fields
    )


def test_valid_binding_installs_and_invalid_object_is_refused(state_case, window):
    _, binding = state_case
    revision = window._guided_backend_validation_revision
    window._set_guided_continuous_rwd_review_binding(binding)
    assert window._guided_continuous_rwd_review_binding == binding
    assert window._guided_backend_validation_revision == revision + 1
    with pytest.raises(TypeError, match="GuidedContinuousRwdReviewBinding"):
        window._set_guided_continuous_rwd_review_binding(object())


def test_clear_is_idempotent(state_case, window):
    _, binding = state_case
    window._set_guided_continuous_rwd_review_binding(binding)
    window._clear_guided_continuous_rwd_review_binding()
    revision = window._guided_backend_validation_revision
    window._clear_guided_continuous_rwd_review_binding()
    assert window._guided_continuous_rwd_review_binding is None
    assert window._guided_backend_validation_revision == revision


@pytest.mark.parametrize(
    "changes",
    [
        {"global_correction_strategy": "robust_global_event_reject"},
        {
            "tonic_settings_contract": GuidedNewAnalysisTonicSettingsContract(
                tonic_timeline_mode="gap_free_elapsed_time"
            )
        },
        {"feature_event_values": {"peak_threshold_k": 9.0}},
        {"output_base_path": "C:/different-output"},
    ],
)
def test_settings_change_rebuilds_with_same_scientific_objects(
    state_case, window, changes
):
    current, binding = state_case
    window._set_guided_continuous_rwd_review_binding(binding)
    current["draft"] = replace(current["draft"], **changes)
    window._invalidate_guided_backend_validation("ordinary draft setting changed")
    rebuilt = window._guided_continuous_rwd_review_binding
    assert rebuilt is not None
    assert rebuilt.draft_plan_identity != binding.draft_plan_identity
    assert rebuilt.recording is binding.recording
    assert rebuilt.continuity_evaluation is binding.continuity_evaluation
    assert rebuilt.current_source_path == binding.current_source_path


@pytest.mark.parametrize(
    "changes",
    [
        {"input_source_path": "C:/different-source"},
        {"resolved_input_source_path": "C:/different-source"},
        {"input_format": "npm"},
        {"acquisition_mode": "intermittent"},
        {"discovered_roi_ids": ["CH1"]},
        {"included_roi_ids": ["CH1"]},
        {"excluded_roi_ids": ["CH10"]},
    ],
)
def test_scientific_authority_change_clears_binding(state_case, window, changes):
    current, binding = state_case
    window._set_guided_continuous_rwd_review_binding(binding)
    current["draft"] = replace(current["draft"], **changes)
    window._invalidate_guided_backend_validation("scientific authority changed")
    assert window._guided_continuous_rwd_review_binding is None


def test_malformed_current_draft_clears_safely(state_case, window, monkeypatch):
    _, binding = state_case
    window._set_guided_continuous_rwd_review_binding(binding)
    monkeypatch.setattr(
        window,
        "_build_guided_new_analysis_draft_plan",
        lambda: (_ for _ in ()).throw(ValueError("malformed")),
    )
    window._refresh_guided_continuous_rwd_review_binding_for_current_draft()
    assert window._guided_continuous_rwd_review_binding is None


def test_current_binding_yields_only_plain_review_facts(state_case, window):
    current, binding = state_case
    window._set_guided_continuous_rwd_review_binding(binding)
    facts = window._guided_continuous_rwd_review_facts(current["draft"])
    assert facts == {
        "recording_type": "Continuous RWD",
        "duration": "10 minutes",
        "included_roi_ids": ("CH1", "CH2"),
        "timestamp_continuity": "Passed",
        "current_source_location": binding.current_source_path,
    }
    rendered = repr(facts)
    assert binding.recording.recording_identity not in rendered
    assert binding.recording.source.sha256 not in rendered


def test_duration_formatter_uses_plain_days_hours_minutes_and_seconds(window):
    assert window._format_guided_continuous_duration(347_346) == (
        "4 days, 29 minutes, 6 seconds"
    )
    assert window._format_guided_continuous_duration(3_661) == (
        "1 hour, 1 minute, 1 second"
    )


def test_stale_or_absent_binding_suppresses_continuous_review_facts(
    state_case, window
):
    current, binding = state_case
    assert window._guided_continuous_rwd_review_facts(current["draft"]) is None
    window._set_guided_continuous_rwd_review_binding(binding)
    stale = replace(current["draft"], output_base_path="C:/stale")
    assert window._guided_continuous_rwd_review_facts(stale) is None


def test_passive_review_appends_facts_only_for_current_continuous_binding(
    state_case, window, monkeypatch
):
    current, binding = state_case
    window._set_guided_continuous_rwd_review_binding(binding)
    monkeypatch.setattr(
        window,
        "_guided_new_analysis_draft_plan_summary_text",
        lambda *_args: "ordinary intermittent summary sentinel",
    )
    window._guided_workflow_mode = "new_analysis"
    window._refresh_guided_draft_run_plan_preview()
    text = window._guided_draft_run_plan_preview_label.text()
    assert "ordinary intermittent summary sentinel" in text
    assert "Recording type: Continuous RWD" in text
    assert "Timestamp continuity: Passed" in text

    window._clear_guided_continuous_rwd_review_binding()
    window._refresh_guided_draft_run_plan_preview()
    assert window._guided_draft_run_plan_preview_label.text() == (
        "ordinary intermittent summary sentinel"
    )


def test_install_rebuild_and_review_do_not_call_inspector_or_evaluator(
    state_case, window, monkeypatch
):
    import photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation as b2
    import photometry_pipeline.io.rwd_continuous_source as cr1a

    def forbidden(*_args, **_kwargs):
        raise AssertionError("recording checks must not run in B3b1")

    monkeypatch.setattr(cr1a, "inspect_continuous_rwd_acquisition_folder", forbidden)
    monkeypatch.setattr(b2, "evaluate_continuous_rwd_timestamp_continuity", forbidden)
    current, binding = state_case
    window._set_guided_continuous_rwd_review_binding(binding)
    current["draft"] = replace(current["draft"], output_base_path="C:/changed")
    window._invalidate_guided_backend_validation("settings changed")
    assert window._guided_continuous_rwd_review_facts(current["draft"]) is not None


def test_install_grants_no_validation_run_or_execution_authority(state_case, window):
    _, binding = state_case
    window._set_guided_continuous_rwd_review_binding(binding)
    assert window._guided_backend_validation_outcome is None
    assert window._guided_validated_plan_identity is None
    assert window._guided_startup_authority is None
    assert window._guided_execution_payload_result is None
    assert window._guided_run_readiness.ready is False
    assert window._guided_run_readiness.status == "no_validation"
    assert window._guided_startup_transaction_request is None
    assert window._guided_review_go_to_run_btn.isEnabled() is False


def test_methods_have_no_persistence_worker_or_recording_check_surface():
    methods = (
        MainWindow._set_guided_continuous_rwd_review_binding,
        MainWindow._clear_guided_continuous_rwd_review_binding,
        MainWindow._refresh_guided_continuous_rwd_review_binding_for_current_draft,
        MainWindow._guided_continuous_rwd_review_facts,
    )
    source = "\n".join(inspect.getsource(method) for method in methods).lower()
    for forbidden in (
        "qsettings",
        "json",
        "serializ",
        "thread",
        "worker",
        "progress",
        "inspect_continuous_rwd_acquisition_folder",
        "evaluate_continuous_rwd_timestamp_continuity",
        "hashlib",
        "open(",
        "read_",
        "write_",
    ):
        assert forbidden not in source
