"""CR1-E3: the live Guided continuous-RWD run workflow.

Drives the real ``MainWindow`` boundaries a scientist actually touches --
the acquisition selection, the navigation actions that start the accepted
recording check and authority preparation, the Run button, and the completed
-run handoff -- rather than calling the private execution helpers directly
(CR1-E2 already covers those).

The recording used here is the same small synthetic continuous-RWD source
the accepted CR1-D/E backend tests build, so preparation exercises the real
accepted builders instead of stand-in objects.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import threading

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
from gui.main_window import (
    MainWindow,
    _GuidedContinuousRwdExecutionRequest,
    _GuidedContinuousRwdPreparedRun,
)
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    evaluate_continuous_rwd_timestamp_continuity,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    build_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    build_guided_continuous_rwd_review_binding,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedNewAnalysisExecutionIntent,
)
from photometry_pipeline.guided_plan_identity import (
    compute_guided_new_analysis_draft_plan_identity,
)

from tests.test_guided_continuous_rwd_correction_pass_persistence import (
    _choices,
    _values,
)


pytestmark = pytest.mark.usefixtures("no_real_modals")

MAIN_THREAD_IDENT = threading.get_ident()


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance._guided_continuous_rwd_prepare_closing = True
    thread = getattr(instance, "_guided_continuous_rwd_prepare_thread", None)
    try:
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(10_000)
    except RuntimeError:
        # The QThread already finished and was deleted by its own
        # deleteLater; only a stale Python wrapper remains. Nothing to wait
        # for, and this must not fail the test that just passed.
        pass
    instance.close()
    instance.deleteLater()


# ---------------------------------------------------------------------------
# One accepted continuous-RWD plan, built the way the backend tests build it
# ---------------------------------------------------------------------------


def _write_source(folder):
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "Fluorescence.csv"
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(6001):
        time, control, signal = _values(np.array([index], dtype=float))
        lines.append(
            f"{time[0]:.1f},{control[0,0]:.12f},{signal[0,0]:.12f},"
            f"{control[0,1]:.12f},{signal[0,1]:.12f}\n"
        )
    source.write_text("".join(lines), encoding="utf-8", newline="")
    return source


def _draft_for(
    folder,
    source,
    *,
    feature_values=None,
    feature_status="default_initialized",
    included=("ROI1", "ROI2"),
    strategies=None,
    output_base=None,
    window_sec=90.0,
):
    from photometry_pipeline.guided_new_analysis_plan import (
        canonical_feature_event_backend_defaults,
    )

    strategies = strategies or {
        "ROI1": "global_linear_regression",
        "ROI2": "signal_only_f0",
    }
    # A live Guided plan whose Feature Detection step is ready carries the
    # complete confirmed field set (the loaded Default profile), not an empty
    # dict -- see MainWindow._build_guided_new_analysis_draft_plan and
    # _guided_feature_detection_readiness. Mirror that here so preparation
    # exercises the real confirmed-settings gate.
    confirmed_values = canonical_feature_event_backend_defaults()
    confirmed_values.update(dict(feature_values or {}))
    included = tuple(included)
    excluded = tuple(
        roi for roi in ("ROI1", "ROI2") if roi not in included
    )
    kwargs = dict(
        input_source_path=str(folder),
        resolved_input_source_path=str(folder),
        input_format="rwd",
        acquisition_mode="continuous",
        continuous_window_sec=window_sec,
        continuous_step_sec=window_sec,
        discovered_roi_ids=["ROI1", "ROI2"],
        included_roi_ids=list(included),
        excluded_roi_ids=list(excluded),
        output_base_path=str(output_base or (folder / "output")),
        global_correction_strategy=next(iter(strategies.values())),
        per_roi_correction_strategy_choices=_choices(
            {roi: strategies[roi] for roi in included}
        ),
        feature_event_profile_id="default",
        feature_event_values=confirmed_values,
        feature_event_profile_status=feature_status,
        feature_event_explicitly_applied=(feature_status == "applied"),
        # One Guided analysis, both outputs -- exactly what the production
        # draft builder sets unconditionally.
        execution_intent=GuidedNewAnalysisExecutionIntent(
            recording_start_clock="11:00",
            recording_start_clock_source="user_entered",
            execution_mode="both",
        ),
    )
    return GuidedNewAnalysisDraftPlan(**kwargs)


def _binding_for(draft, source, *, included=("ROI1", "ROI2")):
    from photometry_pipeline.io.rwd_continuous_source import (
        inspect_continuous_rwd_acquisition_folder,
    )

    inspection = inspect_continuous_rwd_acquisition_folder(source.parent)
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=tuple(included)
    )
    continuity = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=source
    )
    return build_guided_continuous_rwd_review_binding(
        draft,
        recording=recording,
        continuity_evaluation=continuity,
        current_source_path=source,
    )


@pytest.fixture(scope="module")
def source_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_e3_source") / "recording"
    return folder, _write_source(folder)


def _install_plan(window, monkeypatch, draft, source, *, included=("ROI1", "ROI2")):
    """Put one accepted continuous plan into the live window.

    The draft builder is redirected the same way the accepted CR1-D3b review
    -state tests redirect it, so every readiness/preparation/Run decision
    still runs against a real ``GuidedNewAnalysisDraftPlan`` through the real
    production code paths.
    """
    state = {"draft": draft}
    monkeypatch.setattr(
        window,
        "_build_guided_new_analysis_draft_plan",
        lambda: state["draft"],
    )
    window._set_guided_continuous_rwd_review_binding(
        _binding_for(draft, source, included=included)
    )
    # Installing a binding refreshes it against the current draft (existing
    # accepted behavior), so the retained object is the authority.
    return state, window._guided_continuous_rwd_review_binding


def _pump(qapp, predicate, *, timeout_ms=180_000):
    from PySide6.QtCore import QDeadlineTimer

    deadline = QDeadlineTimer(timeout_ms)
    while not predicate() and not deadline.hasExpired():
        qapp.processEvents()
    return predicate()


def _prepare(window, qapp):
    assert window._maybe_start_guided_continuous_rwd_preparation() is True
    assert _pump(
        qapp,
        lambda: not window._guided_continuous_rwd_preparation_active(),
    )
    # Let the thread's finished/cleanup signals drain.
    _pump(
        qapp,
        lambda: getattr(window, "_guided_continuous_rwd_prepare_thread", None)
        is None,
        timeout_ms=20_000,
    )


# ---------------------------------------------------------------------------
# Preparation builds every accepted authority, bound to one accepted plan
# ---------------------------------------------------------------------------


def test_preparation_builds_every_accepted_authority(
    window, qapp, monkeypatch, source_folder
):
    folder, source = source_folder
    draft = _draft_for(folder, source)
    _state, binding = _install_plan(window, monkeypatch, draft, source)

    _prepare(window, qapp)

    prepared = window._guided_continuous_rwd_prepared_run
    assert isinstance(prepared, _GuidedContinuousRwdPreparedRun)
    request = prepared.request
    assert isinstance(request, _GuidedContinuousRwdExecutionRequest)

    from photometry_pipeline.config import Config
    from photometry_pipeline.guided_continuous_rwd_block_plan import (
        GuidedContinuousRwdBlockPlan,
    )
    from photometry_pipeline.guided_continuous_rwd_correction_segments import (
        GuidedContinuousRwdCorrectionSegmentPlan,
        GuidedContinuousRwdDynamicF0Authority,
    )
    from photometry_pipeline.guided_continuous_rwd_target_grid import (
        GuidedContinuousRwdTargetGridDescription,
    )
    from photometry_pipeline.guided_execution_payloads import (
        GuidedExecutionStartupMappingContract,
        build_guided_execution_startup_mapping_contract,
    )

    assert isinstance(request.target_grid, GuidedContinuousRwdTargetGridDescription)
    assert isinstance(request.block_plan, GuidedContinuousRwdBlockPlan)
    assert isinstance(request.segment_plan, GuidedContinuousRwdCorrectionSegmentPlan)
    assert isinstance(
        request.dynamic_f0_authority, GuidedContinuousRwdDynamicF0Authority
    )
    assert isinstance(
        request.startup_mapping_contract, GuidedExecutionStartupMappingContract
    )
    assert request.startup_mapping_contract == (
        build_guided_execution_startup_mapping_contract()
    )
    assert isinstance(request.config, Config)
    assert request.cancellation_requested is None

    # The accepted plan objects are retained, not rebuilt copies.
    assert request.review_binding is binding
    assert request.accepted_draft is draft
    assert request.output_base == draft.output_base_path
    assert prepared.output_base == draft.output_base_path

    # Prepared state, review binding, and live draft all name one plan.
    live_identity = compute_guided_new_analysis_draft_plan_identity(draft)
    assert prepared.plan_identity == live_identity
    assert binding.draft_plan_identity == live_identity


def test_preparation_runs_off_the_gui_thread(
    window, qapp, monkeypatch, source_folder
):
    """The expensive dynamic-F0 authority must be built by the preparation
    worker, not synchronously inside the navigation or Run callback."""
    folder, source = source_folder
    draft = _draft_for(folder, source)
    _install_plan(window, monkeypatch, draft, source)

    module = importlib.import_module(
        "photometry_pipeline.guided_continuous_rwd_correction_segments"
    )
    real = module.prepare_guided_continuous_rwd_dynamic_f0_authority
    seen: dict[str, object] = {}

    def recording_wrapper(*args, **kwargs):
        seen["ident"] = threading.get_ident()
        # While the worker is inside the expensive builder, the GUI must
        # already be showing the preparing state with Run disabled.
        seen["status"] = window._guided_continuous_rwd_status_message
        seen["run_enabled"] = window._guided_run_btn.isEnabled()
        seen["active"] = window._guided_continuous_rwd_preparation_active()
        return real(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "prepare_guided_continuous_rwd_dynamic_f0_authority",
        recording_wrapper,
    )

    _prepare(window, qapp)

    assert seen["ident"] != MAIN_THREAD_IDENT
    assert seen["active"] is True
    assert seen["run_enabled"] is False
    assert "Preparing continuous analysis" in str(seen["status"])
    assert window._guided_continuous_rwd_prepared_run is not None


# ---------------------------------------------------------------------------
# The live workflow starts the accepted recording check and preparation
# ---------------------------------------------------------------------------


class _FakeThread(main_window_module.QObject):
    started = main_window_module.Signal()
    finished = main_window_module.Signal()
    instances: list = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.__class__.instances.append(self)

    def start(self):
        self.running = True
        self.started.emit()

    def quit(self, *_args):
        pass

    def isRunning(self):
        return self.running

    def deleteLater(self, *_args):
        pass


class _FakeCheckWorker(main_window_module.QObject):
    stage_changed = main_window_module.Signal(str)
    succeeded = main_window_module.Signal(object)
    failed = main_window_module.Signal(object)
    cancelled = main_window_module.Signal()
    instances: list = []

    def __init__(self, request):
        super().__init__()
        self.request = request
        self.__class__.instances.append(self)

    def moveToThread(self, _thread):
        pass

    def run(self):
        pass

    def request_cancel(self):
        pass

    def deleteLater(self, *_args):
        pass


@pytest.fixture
def fake_check_runtime(monkeypatch):
    _FakeThread.instances = []
    _FakeCheckWorker.instances = []
    monkeypatch.setattr(main_window_module, "QThread", _FakeThread)
    monkeypatch.setattr(
        main_window_module,
        "_GuidedContinuousRwdRecordingCheckWorker",
        _FakeCheckWorker,
    )
    return _FakeCheckWorker


def _select_continuous_rwd_setup(window, folder):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(folder / "output"))
    window._guided_format_combo.setCurrentText("rwd")
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("continuous")
    )
    window._guided_continuous_window_sec_spin.setValue(90.0)
    window._guided_recording_start_clock_edit.setText("11:00")


def test_continue_from_recording_structure_starts_the_recording_check(
    window, monkeypatch, tmp_path, fake_check_runtime
):
    folder = tmp_path / "trigger"
    source = _write_source(folder)
    draft = _draft_for(folder, source)
    monkeypatch.setattr(
        window, "_build_guided_new_analysis_draft_plan", lambda: draft
    )
    _select_continuous_rwd_setup(window, folder)
    assert window._guided_recording_structure_ready_to_continue() is True

    window._on_guided_continue_to_correction_approach()

    assert len(fake_check_runtime.instances) == 1
    request = fake_check_runtime.instances[0].request
    assert tuple(request.included_roi_ids) == ("ROI1", "ROI2")
    assert str(request.selected_acquisition_folder) == str(folder)
    assert "Inspecting recording" in (
        window._guided_continuous_rwd_check_status_label.text()
    )


def test_recording_check_is_not_started_for_intermittent(
    window, tmp_path, fake_check_runtime
):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_input_dir_edit.setText(str(tmp_path))
    window._guided_output_dir_edit.setText(str(tmp_path / "out"))
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")

    window._on_guided_continue_to_correction_approach()

    assert fake_check_runtime.instances == []


def test_recording_check_is_not_repeated_for_an_already_checked_plan(
    window, monkeypatch, tmp_path, fake_check_runtime
):
    folder = tmp_path / "no_repeat"
    source = _write_source(folder)
    draft = _draft_for(folder, source)
    # The recording structure is chosen first (CR1-F1-A moved that choice into
    # Select data). Choosing it *after* installing a binding would correctly
    # invalidate that binding, which is a different behavior -- covered by
    # tests/test_gui_guided_recording_structure_before_discovery.py.
    _select_continuous_rwd_setup(window, folder)
    _install_plan(window, monkeypatch, draft, source)

    window._on_guided_continue_to_correction_approach()
    window._on_guided_continue_to_correction_approach()

    assert fake_check_runtime.instances == []


def test_reaching_the_run_step_is_passive(
    window, monkeypatch, tmp_path
):
    folder = tmp_path / "run_step_trigger"
    source = _write_source(folder)
    draft = _draft_for(folder, source)
    _install_plan(window, monkeypatch, draft, source)

    started: list[bool] = []
    monkeypatch.setattr(
        window,
        "_maybe_start_guided_continuous_rwd_preparation",
        lambda: started.append(True) or True,
    )
    window._reach_guided_step("Run")
    window._guided_workflow_stepper.setCurrentRow(
        window._guided_step_index("Run")
    )

    assert started == []


# ---------------------------------------------------------------------------
# Readiness transitions
# ---------------------------------------------------------------------------


def test_run_readiness_transitions(window, qapp, monkeypatch, tmp_path):
    """One Guided analysis, producing both tonic and phasic outputs -- the
    only plan the live continuous workflow accepts."""
    folder = tmp_path / "readiness"
    source = _write_source(folder)
    draft = _draft_for(folder, source)
    assert draft.execution_intent.execution_mode == "both"
    _install_plan(window, monkeypatch, draft, source)

    # Before preparation.
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False

    assert window._maybe_start_guided_continuous_rwd_preparation() is True
    # During preparation.
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False
    assert "Preparing continuous analysis" in (
        window._guided_backend_validation_status_label.text()
    )
    assert window._guided_run_readiness_label.text() == (
        "Check your Guided setup before running."
    )
    assert window._guided_continuous_rwd_cancel_btn.isHidden() is False

    assert _pump(
        qapp, lambda: not window._guided_continuous_rwd_preparation_active()
    )
    _pump(
        qapp,
        lambda: getattr(window, "_guided_continuous_rwd_prepare_thread", None)
        is None,
        timeout_ms=20_000,
    )

    # Ready.
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is True
    assert window._guided_backend_validation_status_label.text() == (
        "Setup check passed. Ready to run."
    )
    assert window._guided_run_readiness_label.text() == (
        "The setup is ready. Start Guided Analysis when you are ready."
    )

    # During execution.
    window._guided_continuous_rwd_execution_active = True
    window._set_guided_continuous_rwd_status(
        "Running continuous analysis…", analysis=True
    )
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_run_readiness_label.text() == (
        "Running continuous analysis…"
    )
    window._guided_continuous_rwd_execution_active = False


def test_confirmed_feature_settings_are_always_required(
    window, qapp, monkeypatch, tmp_path
):
    """Every live Guided run includes phasic analysis, so confirmed Feature
    Detection settings are required unconditionally."""
    folder = tmp_path / "unconfirmed"
    source = _write_source(folder)
    draft = _draft_for(folder, source, feature_status="missing")
    _install_plan(window, monkeypatch, draft, source)

    assert window._maybe_start_guided_continuous_rwd_preparation() is False
    ready, message = window._guided_continuous_rwd_run_readiness()
    assert ready is False
    assert message == "Confirm the Feature Detection settings before running."
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False


# ---------------------------------------------------------------------------
# Only the accepted Guided plan may run: anything else fails closed
#
# The lower-level continuous backend and E2's selector still support
# correction-only, tonic-only, and phasic-only execution (covered by
# tests/test_guided_continuous_rwd_execution_worker.py). None of those is a
# live Guided workflow, so a live draft carrying one is malformed or stale --
# never a scientist choice -- and must be refused before anything runs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injected_mode", ["tonic", "phasic", "traces_only", "", "unexpected"]
)
def test_injected_non_guided_mode_is_refused_before_anything_runs(
    window, qapp, monkeypatch, tmp_path, injected_mode
):
    folder = tmp_path / f"injected_{injected_mode or 'empty'}"
    source = _write_source(folder)
    draft = _draft_for(folder, source)
    state, _binding = _install_plan(window, monkeypatch, draft, source)
    calls, _seen = _guard_backends(monkeypatch)

    injected = dataclasses.replace(
        draft,
        execution_intent=GuidedNewAnalysisExecutionIntent(
            execution_mode=injected_mode
        ),
    )
    state["draft"] = injected
    # Give the injected plan a review binding of its own, so the analysis-mode
    # gate is the only thing left that can refuse -- not the plan-identity
    # backstop, which is covered separately.
    window._set_guided_continuous_rwd_review_binding(
        _binding_for(injected, source)
    )
    assert window._guided_continuous_rwd_accepted_plan() is not None

    supported, refusal = window._guided_continuous_rwd_analysis_supported(
        state["draft"]
    )
    assert supported is False
    assert refusal == (
        "This Guided analysis is not configured correctly. Return to Setup "
        "and review the analysis plan."
    )
    # Refused before preparation, so no authority and no Config are built.
    assert window._maybe_start_guided_continuous_rwd_preparation() is False
    assert window._guided_continuous_rwd_prepared_run is None
    assert window._guided_continuous_rwd_preparation_active() is False

    # Refused at readiness, so Run is never offered.
    ready, message = window._guided_continuous_rwd_run_readiness()
    assert ready is False
    assert message == refusal
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False

    # Refused at the Run action, so no worker starts and no backend is called.
    window._on_guided_run_clicked_backend_guarded()
    assert window._guided_continuous_rwd_execution_active is False
    assert getattr(window, "_guided_run_execution_worker", None) is None
    assert calls == {}


def test_an_already_prepared_run_cannot_execute_after_the_mode_is_corrupted(
    window, qapp, monkeypatch, tmp_path
):
    """Backstop: even a successfully prepared request must not execute once
    the live plan no longer matches the accepted Guided contract."""
    state, draft, _binding, _folder, _source = _prepared_window(
        window, qapp, monkeypatch, tmp_path, "corrupted_after_prepare"
    )
    prepared = window._guided_continuous_rwd_prepared_run
    assert window._guided_continuous_rwd_run_readiness()[0] is True
    calls, _seen = _guard_backends(monkeypatch)

    state["draft"] = dataclasses.replace(
        draft,
        execution_intent=GuidedNewAnalysisExecutionIntent(
            execution_mode="phasic"
        ),
    )
    # Reinstate the prepared request without invalidating, so the gate itself
    # is what refuses -- not an invalidation hook.
    window._guided_continuous_rwd_prepared_run = prepared

    assert window._guided_continuous_rwd_run_readiness()[0] is False
    window._on_guided_run_clicked_backend_guarded()
    assert window._guided_continuous_rwd_execution_active is False
    assert calls == {}


def test_missing_output_destination_blocks_preparation(
    window, monkeypatch, tmp_path
):
    folder = tmp_path / "no_output"
    source = _write_source(folder)
    draft = dataclasses.replace(
        _draft_for(folder, source),
        output_base_path=None,
        output_policy_path=None,
        output_policy_status="missing",
        output_policy_explicitly_applied=False,
    )
    _install_plan(window, monkeypatch, draft, source)

    assert window._maybe_start_guided_continuous_rwd_preparation() is False
    ready, message = window._guided_continuous_rwd_run_readiness()
    assert ready is False
    assert message == "Choose where to save results before running."


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def _prepared_window(window, qapp, monkeypatch, tmp_path, name, **draft_kwargs):
    folder = tmp_path / name
    source = _write_source(folder)
    draft = _draft_for(folder, source, **draft_kwargs)
    state, binding = _install_plan(window, monkeypatch, draft, source)
    _prepare(window, qapp)
    assert window._guided_continuous_rwd_prepared_run is not None
    return state, draft, binding, folder, source


@pytest.mark.parametrize(
    "change",
    [
        "roi_selection",
        "correction_choice",
        "feature_settings",
        "output_destination",
        "window_length",
    ],
)
def test_relevant_changes_invalidate_the_prepared_run(
    window, qapp, monkeypatch, tmp_path, change
):
    state, draft, _binding, folder, _source = _prepared_window(
        window, qapp, monkeypatch, tmp_path, f"inval_{change}"
    )

    if change == "roi_selection":
        updated = dataclasses.replace(
            draft,
            included_roi_ids=["ROI1"],
            excluded_roi_ids=["ROI2"],
            per_roi_correction_strategy_choices=_choices(
                {"ROI1": "global_linear_regression"}
            ),
        )
    elif change == "correction_choice":
        updated = dataclasses.replace(
            draft,
            per_roi_correction_strategy_choices=_choices(
                {"ROI1": "signal_only_f0", "ROI2": "signal_only_f0"}
            ),
        )
    elif change == "feature_settings":
        updated = dataclasses.replace(
            draft,
            feature_event_values={"peak_threshold_percentile": 77.5},
        )
    elif change == "output_destination":
        updated = dataclasses.replace(
            draft, output_base_path=str(folder / "elsewhere")
        )
    else:
        updated = dataclasses.replace(
            draft, continuous_window_sec=45.0, continuous_step_sec=45.0
        )
    state["draft"] = updated

    window._invalidate_guided_backend_validation(f"{change} changed")

    assert window._guided_continuous_rwd_prepared_run is None
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False


def test_a_new_plan_stops_offering_the_previous_completed_run(
    window, qapp, monkeypatch, tmp_path
):
    _prepared_window(window, qapp, monkeypatch, tmp_path, "handoff_invalidation")
    window._guided_continuous_rwd_completed_run_dir = str(tmp_path / "old_run")
    window._refresh_guided_review_handoff_display()
    assert window._guided_load_completed_run_for_review_btn.isHidden() is False

    window._invalidate_guided_backend_validation("a new analysis started")

    assert window._guided_continuous_rwd_completed_run_dir is None
    assert window._guided_load_completed_run_for_review_btn.isHidden() is True


def test_prepared_run_is_refused_for_a_different_plan_identity(
    window, qapp, monkeypatch, tmp_path
):
    """The plan-identity check is the backstop: even with the prepared run
    still installed and no invalidation hook having fired, a live plan that
    differs from the prepared one must not be runnable.

    The difference used here is an ordinary user-editable choice (the output
    destination), so this covers the identity mechanism itself rather than any
    analysis-family change.
    """
    state, draft, _binding, folder, _source = _prepared_window(
        window, qapp, monkeypatch, tmp_path, "identity_backstop"
    )
    prepared = window._guided_continuous_rwd_prepared_run
    assert window._guided_continuous_rwd_run_readiness()[0] is True

    state["draft"] = dataclasses.replace(
        draft, output_base_path=str(folder / "somewhere_else")
    )
    # Deliberately do NOT invalidate: prove readiness fails closed on its own.
    window._guided_continuous_rwd_prepared_run = prepared

    ready, _message = window._guided_continuous_rwd_run_readiness()
    assert ready is False
    window._on_guided_run_clicked_backend_guarded()
    assert window._guided_continuous_rwd_execution_active is False


def test_stale_preparation_completion_is_discarded(
    window, monkeypatch, tmp_path
):
    folder = tmp_path / "stale"
    source = _write_source(folder)
    draft = _draft_for(folder, source)
    state, _binding = _install_plan(window, monkeypatch, draft, source)

    assert window._maybe_start_guided_continuous_rwd_preparation() is True
    token = window._guided_continuous_rwd_prepare_active_token
    worker = window._guided_continuous_rwd_prepare_worker

    # The scientist edits the plan while preparation is still running.
    state["draft"] = dataclasses.replace(
        draft, output_base_path=str(folder / "moved_output")
    )
    window._invalidate_guided_backend_validation("plan changed mid-preparation")

    # The worker's eventual success is refused.
    window._on_guided_continuous_rwd_preparation_succeeded(
        token,
        worker,
        _GuidedContinuousRwdPreparedRun(
            request=object(),
            plan_identity=compute_guided_new_analysis_draft_plan_identity(draft),
            output_base=str(folder / "output"),
        ),
    )
    assert window._guided_continuous_rwd_prepared_run is None
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False


def test_stale_completion_for_the_same_token_but_changed_plan_is_discarded(
    window, qapp, monkeypatch, tmp_path
):
    """No invalidation hook fires here at all: the completion handler itself
    re-derives the live plan identity and refuses a mismatch."""
    folder = tmp_path / "stale_no_hook"
    source = _write_source(folder)
    draft = _draft_for(folder, source)
    state, _binding = _install_plan(window, monkeypatch, draft, source)

    assert window._maybe_start_guided_continuous_rwd_preparation() is True
    token = window._guided_continuous_rwd_prepare_active_token
    worker = window._guided_continuous_rwd_prepare_worker
    state["draft"] = dataclasses.replace(draft, output_base_path=str(folder / "b"))

    window._on_guided_continuous_rwd_preparation_succeeded(
        token,
        worker,
        _GuidedContinuousRwdPreparedRun(
            request=object(),
            plan_identity=compute_guided_new_analysis_draft_plan_identity(draft),
            output_base=str(folder / "output"),
        ),
    )
    assert window._guided_continuous_rwd_prepared_run is None
    assert "setup changed" in window._guided_continuous_rwd_status_message


# ---------------------------------------------------------------------------
# The live Run action reaches exactly one accepted backend
# ---------------------------------------------------------------------------


def _guard_backends(monkeypatch):
    calls: dict[str, int] = {}
    seen: dict[str, object] = {}
    for key, module_path, attr in (
        (
            "correction",
            "photometry_pipeline.guided_continuous_rwd_correction_run",
            "execute_guided_continuous_rwd_correction_run",
        ),
        (
            "tonic",
            "photometry_pipeline.guided_continuous_rwd_tonic_run",
            "execute_guided_continuous_rwd_tonic_run",
        ),
        (
            "phasic",
            "photometry_pipeline.guided_continuous_rwd_phasic_run",
            "execute_guided_continuous_rwd_phasic_run",
        ),
        (
            "combined",
            "photometry_pipeline.guided_continuous_rwd_combined_run",
            "execute_guided_continuous_rwd_combined_run",
        ),
    ):
        module = importlib.import_module(module_path)

        def stub(*args, _key=key, **kwargs):
            calls[_key] = calls.get(_key, 0) + 1
            seen["args"] = args
            seen["kwargs"] = kwargs
            return _FakeRunResult(run_dir="")

        monkeypatch.setattr(module, attr, stub)
    return calls, seen


@dataclasses.dataclass
class _FakeRunResult:
    run_dir: str


def _press_run_and_wait(window, qapp):
    window._on_guided_run_clicked_backend_guarded()
    assert _pump(
        qapp,
        lambda: window._guided_continuous_rwd_execution_active is False,
        timeout_ms=60_000,
    )
    _pump(
        qapp,
        lambda: getattr(window, "_guided_run_execution_thread", None) is None,
        timeout_ms=20_000,
    )


def test_live_run_calls_exactly_one_accepted_backend(
    window, qapp, monkeypatch, tmp_path
):
    """The accepted live Guided plan produces both tonic and phasic outputs
    from one run, so exactly one backend -- the combined one -- is ever
    reached from the live GUI."""
    _state, draft, _binding, _folder, _source = _prepared_window(
        window, qapp, monkeypatch, tmp_path, "route_live"
    )
    assert draft.execution_intent.execution_mode == "both"
    prepared = window._guided_continuous_rwd_prepared_run
    assert prepared.request.accepted_draft.execution_intent.execution_mode == (
        "both"
    )
    calls, seen = _guard_backends(monkeypatch)

    constructed: list[object] = []
    real_worker = main_window_module._GuidedRunExecutionWorker

    class RecordingWorker(real_worker):
        def __init__(self, request, runner, continuous_execution=None):
            super().__init__(request, runner, continuous_execution)
            constructed.append(continuous_execution)

    monkeypatch.setattr(
        main_window_module, "_GuidedRunExecutionWorker", RecordingWorker
    )

    _press_run_and_wait(window, qapp)

    assert calls == {"combined": 1}
    # The live Run action reached the CR1-E2 worker branch with the prepared
    # request, not an intermittent startup request.
    assert len(constructed) == 1
    passed = constructed[0]
    assert isinstance(passed, _GuidedContinuousRwdExecutionRequest)
    assert passed.review_binding is prepared.request.review_binding
    assert passed.target_grid is prepared.request.target_grid
    assert passed.block_plan is prepared.request.block_plan
    assert passed.segment_plan is prepared.request.segment_plan
    assert passed.dynamic_f0_authority is prepared.request.dynamic_f0_authority
    assert passed.accepted_draft is prepared.request.accepted_draft
    assert (
        passed.startup_mapping_contract
        is prepared.request.startup_mapping_contract
    )
    assert passed.config is prepared.request.config
    assert passed.output_base == prepared.request.output_base
    # Only the cooperative cancellation callable is attached at Run press.
    assert callable(passed.cancellation_requested)

    # The backend received those same objects, unchanged.
    assert seen["args"][0] is prepared.request.review_binding
    assert seen["args"][1] is prepared.request.target_grid
    assert seen["kwargs"]["config"] is prepared.request.config


def test_live_progress_attaches_after_run_allocation_and_tracks_stages(
    window, qapp, monkeypatch, tmp_path
):
    """The real worker boundary attaches the existing live-status machinery
    only after the backend has created its authoritative run directory."""
    _prepared_window(window, qapp, monkeypatch, tmp_path, "live_progress")
    module = importlib.import_module(
        "photometry_pipeline.guided_continuous_rwd_combined_run"
    )
    completion_contract = importlib.import_module(
        "photometry_pipeline.run_completion_contract"
    )
    terminal_success = completion_contract.TERMINAL_SUCCESS_CURRENT
    markers: list[str] = []
    handler_thread_ids: list[int] = []
    worker_signal_thread_ids: list[int] = []
    gui_thread_id = threading.get_ident()
    monkeypatch.setattr(
        completion_contract,
        "classify_run_terminal_state",
        lambda _run_dir: (
            markers.append("classifier")
            or type("Classification", (), {"state": terminal_success})()
        ),
    )

    run_dir = tmp_path / "live_progress_run"
    run_id = "continuous-live-progress"
    release = threading.Event()

    real_start_status = window._start_guided_run_live_status

    def record_start_status(*args, **kwargs):
        markers.append("run_start_callback_delivered")
        result = real_start_status(*args, **kwargs)
        markers.append("follower_attached")
        return result

    monkeypatch.setattr(
        window, "_start_guided_run_live_status", record_start_status
    )

    real_success_handler = (
        window._on_guided_continuous_rwd_execution_succeeded
    )

    def record_success_handler(worker, result):
        markers.append("gui_success_handler_entered")
        handler_thread_ids.append(threading.get_ident())
        try:
            return real_success_handler(worker, result)
        finally:
            markers.append("gui_success_handler_finished")

    monkeypatch.setattr(
        window,
        "_on_guided_continuous_rwd_execution_succeeded",
        record_success_handler,
    )

    real_worker = main_window_module._GuidedRunExecutionWorker

    class RecordingWorker(real_worker):
        def __init__(self, request, runner, continuous_execution=None):
            super().__init__(request, runner, continuous_execution)

            def record_success(_result):
                markers.append("worker_succeeded_emitted")
                worker_signal_thread_ids.append(threading.get_ident())

            self.succeeded.connect(record_success)

    monkeypatch.setattr(
        main_window_module, "_GuidedRunExecutionWorker", RecordingWorker
    )

    def write_status(phase, status="running"):
        run_dir.joinpath("status.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "phase": phase,
                    "status": status,
                }
            ),
            encoding="utf-8",
        )

    def running_backend(*_args, **kwargs):
        markers.append("run_directory_allocated")
        run_dir.mkdir()
        write_status("initializing")
        kwargs["run_started_callback"](str(run_dir), run_id)
        release.wait(30.0)
        write_status("final", status="success")
        markers.append("terminal_success_written")
        markers.append("backend_returned")
        return _FakeRunResult(run_dir=str(run_dir))

    monkeypatch.setattr(module, "execute_guided_continuous_rwd_combined_run", running_backend)
    window._on_guided_run_clicked_backend_guarded()
    try:
        attached = _pump(
            qapp,
            lambda: getattr(window, "_guided_run_status_follower", None)
            is not None,
            timeout_ms=20_000,
        )
        assert attached
        follower = window._guided_run_status_follower
        elapsed_timer = window._guided_run_elapsed_timer
        assert follower is not None and follower.is_active
        assert elapsed_timer is not None and elapsed_timer.isActive()
        assert window._guided_run_live_status_group.isHidden() is False

        window._on_guided_continuous_rwd_run_started(
            str(tmp_path / "stale_run"), "stale-run"
        )
        assert window._guided_continuous_rwd_live_run_dir == str(run_dir.resolve())
        assert window._guided_run_status_follower is follower

        expected = (
            ("preparing_recording", "Currently: Preparing recording"),
            ("correcting_signals", "Currently: Correcting signals"),
            ("analyzing_tonic_signal", "Currently: Analyzing tonic signal"),
            ("detecting_features", "Currently: Detecting features"),
            ("building_summaries", "Currently: Building summaries"),
            ("saving_results", "Currently: Saving results"),
        )
        for phase, label in expected:
            write_status(phase)
            follower._poll()
            assert window._guided_run_live_phase_label.text() == label
            assert window._guided_run_elapsed_timer is elapsed_timer
            assert elapsed_timer.isActive()

        current_phase = window._guided_run_live_phase_label.text()
        window._on_guided_run_live_status_received(
            {"run_id": "stale-run", "phase": "detecting_features"}
        )
        assert window._guided_run_live_phase_label.text() == current_phase

        visible = " ".join(
            (
                window._guided_run_live_status_label.text(),
                window._guided_run_live_phase_label.text(),
                window._guided_run_live_elapsed_label.text(),
            )
        ).lower()
        assert "%" not in visible
        assert "eta" not in visible
    finally:
        release.set()

    assert _pump(
        qapp,
        lambda: window._guided_continuous_rwd_execution_active is False,
        timeout_ms=20_000,
    )
    assert window._guided_run_status_follower is None
    assert window._guided_run_elapsed_timer is None
    assert window._guided_run_live_status_group.isHidden()
    assert window._guided_continuous_rwd_completed_run_dir == str(run_dir)
    assert handler_thread_ids == [gui_thread_id]
    assert worker_signal_thread_ids
    assert worker_signal_thread_ids[0] != gui_thread_id
    assert markers.index("run_directory_allocated") < markers.index(
        "run_start_callback_delivered"
    )
    assert markers.index("follower_attached") < markers.index(
        "terminal_success_written"
    )
    assert markers.index("terminal_success_written") < markers.index(
        "backend_returned"
    )
    assert markers.index("backend_returned") < markers.index(
        "worker_succeeded_emitted"
    )
    assert markers.index("worker_succeeded_emitted") < markers.index(
        "gui_success_handler_entered"
    )
    assert markers.index("gui_success_handler_entered") < markers.index(
        "classifier"
    )
    assert markers.index("classifier") < markers.index(
        "gui_success_handler_finished"
    )
    assert _pump(
        qapp,
        lambda: getattr(window, "_guided_run_execution_thread", None) is None,
        timeout_ms=20_000,
    )
    assert getattr(window, "_guided_run_execution_worker", None) is None
    assert getattr(window, "_guided_run_execution_thread", None) is None

    opened: list[str] = []
    monkeypatch.setattr(
        window,
        "_start_guided_completed_review_load",
        lambda candidate: opened.append(candidate),
    )
    window._on_guided_load_completed_run_for_review_clicked()
    assert opened == [str(run_dir)]


def test_live_progress_stops_after_forced_failure(
    window, qapp, monkeypatch, tmp_path
):
    _prepared_window(window, qapp, monkeypatch, tmp_path, "live_progress_failure")
    module = importlib.import_module(
        "photometry_pipeline.guided_continuous_rwd_combined_run"
    )
    run_dir = tmp_path / "live_progress_failure_run"
    run_id = "continuous-live-failure"
    release = threading.Event()
    gui_thread_id = threading.get_ident()
    failure_thread_ids: list[int] = []
    real_failure_handler = window._on_guided_continuous_rwd_execution_failed

    def record_failure_handler(worker, message):
        failure_thread_ids.append(threading.get_ident())
        return real_failure_handler(worker, message)

    monkeypatch.setattr(
        window,
        "_on_guided_continuous_rwd_execution_failed",
        record_failure_handler,
    )

    def failing_backend(*_args, **kwargs):
        run_dir.mkdir()
        run_dir.joinpath("status.json").write_text(
            json.dumps({"run_id": run_id, "phase": "initializing", "status": "running"}),
            encoding="utf-8",
        )
        kwargs["run_started_callback"](str(run_dir), run_id)
        release.wait(30.0)
        raise RuntimeError("forced backend failure")

    monkeypatch.setattr(module, "execute_guided_continuous_rwd_combined_run", failing_backend)
    window._on_guided_run_clicked_backend_guarded()
    try:
        assert _pump(
            qapp,
            lambda: getattr(window, "_guided_run_status_follower", None)
            is not None,
            timeout_ms=20_000,
        )
        assert window._guided_run_elapsed_timer is not None
    finally:
        release.set()

    assert _pump(
        qapp,
        lambda: window._guided_continuous_rwd_execution_active is False,
        timeout_ms=20_000,
    )
    assert window._guided_run_status_follower is None
    assert window._guided_run_elapsed_timer is None
    assert window._guided_run_live_status_group.isHidden()
    assert window._guided_continuous_rwd_completed_run_dir is None
    assert window._guided_continuous_rwd_status_message == (
        "Continuous analysis could not be completed."
    )
    assert failure_thread_ids == [gui_thread_id]
    assert _pump(
        qapp,
        lambda: getattr(window, "_guided_run_execution_thread", None) is None,
        timeout_ms=20_000,
    )
    assert getattr(window, "_guided_run_execution_worker", None) is None
    assert getattr(window, "_guided_run_execution_thread", None) is None


def test_stale_continuous_terminal_worker_is_ignored(
    window, qapp, monkeypatch, tmp_path
):
    current_worker = object()
    stale_worker = object()
    run_dir = tmp_path / "current_run"
    run_dir.mkdir()
    run_dir.joinpath("status.json").write_text(
        json.dumps(
            {
                "run_id": "current-run",
                "phase": "running",
                "status": "running",
            }
        ),
        encoding="utf-8",
    )
    window._guided_run_execution_worker = current_worker
    window._guided_continuous_rwd_execution_active = True
    window._guided_continuous_rwd_live_run_dir = str(run_dir.resolve())
    window._start_guided_run_live_status(
        str(run_dir), run_identity="current-run"
    )
    follower = window._guided_run_status_follower
    elapsed_timer = window._guided_run_elapsed_timer

    window._on_guided_continuous_rwd_execution_succeeded(
        stale_worker, _FakeRunResult(run_dir=str(run_dir))
    )
    window._on_guided_continuous_rwd_execution_failed(
        stale_worker, "stale failure"
    )

    assert window._guided_continuous_rwd_execution_active is True
    assert window._guided_run_status_follower is follower
    assert window._guided_run_elapsed_timer is elapsed_timer
    assert window._guided_continuous_rwd_completed_run_dir is None

    completion_contract = importlib.import_module(
        "photometry_pipeline.run_completion_contract"
    )
    monkeypatch.setattr(
        completion_contract,
        "classify_run_terminal_state",
        lambda _run_dir: type(
            "Classification", (), {"state": completion_contract.TERMINAL_SUCCESS_CURRENT}
        )(),
    )
    window._on_guided_continuous_rwd_execution_succeeded(
        current_worker, _FakeRunResult(run_dir=str(run_dir))
    )
    assert window._guided_continuous_rwd_execution_active is False
    assert window._guided_run_status_follower is None
    assert window._guided_run_elapsed_timer is None
    assert window._guided_continuous_rwd_completed_run_dir == str(run_dir)


def test_continuous_terminal_run_directory_guard_preserves_current_run(
    window, qapp, monkeypatch, tmp_path
):
    current_worker = object()
    current_dir = tmp_path / "current_run"
    stale_dir = tmp_path / "stale_run"
    current_dir.mkdir()
    stale_dir.mkdir()
    current_dir.joinpath("status.json").write_text(
        json.dumps(
            {
                "run_id": "current-run",
                "phase": "running",
                "status": "running",
            }
        ),
        encoding="utf-8",
    )
    window._guided_run_execution_worker = current_worker
    window._guided_continuous_rwd_execution_active = True
    window._guided_continuous_rwd_live_run_dir = str(current_dir.resolve())
    window._start_guided_run_live_status(
        str(current_dir), run_identity="current-run"
    )
    follower = window._guided_run_status_follower
    elapsed_timer = window._guided_run_elapsed_timer

    window._on_guided_continuous_rwd_execution_succeeded(
        current_worker, _FakeRunResult(run_dir=str(stale_dir))
    )

    assert window._guided_continuous_rwd_execution_active is True
    assert window._guided_run_status_follower is follower
    assert window._guided_run_elapsed_timer is elapsed_timer
    assert window._guided_continuous_rwd_completed_run_dir is None

    completion_contract = importlib.import_module(
        "photometry_pipeline.run_completion_contract"
    )
    monkeypatch.setattr(
        completion_contract,
        "classify_run_terminal_state",
        lambda _run_dir: type(
            "Classification", (), {"state": completion_contract.TERMINAL_SUCCESS_CURRENT}
        )(),
    )
    window._on_guided_continuous_rwd_execution_succeeded(
        current_worker, _FakeRunResult(run_dir=str(current_dir))
    )
    assert window._guided_continuous_rwd_execution_active is False
    assert window._guided_run_status_follower is None
    assert window._guided_run_elapsed_timer is None
    assert window._guided_continuous_rwd_completed_run_dir == str(current_dir)


def test_run_press_rebuilds_no_authority(
    window, qapp, monkeypatch, tmp_path
):
    """Hard requirement: pressing Run must consume the prepared authorities
    and build nothing."""
    _prepared_window(window, qapp, monkeypatch, tmp_path, "no_rebuild")
    _calls, _seen = _guard_backends(monkeypatch)

    def forbidden(name):
        def _raise(*_args, **_kwargs):
            raise AssertionError(f"{name} was rebuilt at Run press")

        return _raise

    for module_path, attr in (
        (
            "photometry_pipeline.guided_continuous_rwd_target_grid",
            "build_guided_continuous_rwd_target_grid",
        ),
        (
            "photometry_pipeline.guided_continuous_rwd_block_plan",
            "build_guided_continuous_rwd_block_plan",
        ),
        (
            "photometry_pipeline.guided_continuous_rwd_correction_segments",
            "build_guided_continuous_rwd_correction_segment_plan",
        ),
        (
            "photometry_pipeline.guided_continuous_rwd_correction_segments",
            "prepare_guided_continuous_rwd_dynamic_f0_authority",
        ),
        (
            "photometry_pipeline.guided_continuous_rwd_run_config",
            "build_guided_continuous_rwd_run_config",
        ),
    ):
        monkeypatch.setattr(
            importlib.import_module(module_path), attr, forbidden(attr)
        )

    _press_run_and_wait(window, qapp)

    assert window._guided_continuous_rwd_execution_active is False


def test_duplicate_run_press_starts_only_one_execution(
    window, qapp, monkeypatch, tmp_path
):
    _prepared_window(window, qapp, monkeypatch, tmp_path, "duplicate_run")
    calls, _seen = _guard_backends(monkeypatch)

    window._guided_continuous_rwd_execution_active = True
    window._on_guided_run_clicked_backend_guarded()
    assert calls == {}
    window._guided_continuous_rwd_execution_active = False


# ---------------------------------------------------------------------------
# Config composition: the scientist's confirmed settings reach the backend
# ---------------------------------------------------------------------------


def test_confirmed_nondefault_feature_settings_reach_the_backend(
    window, qapp, monkeypatch, tmp_path
):
    nondefault = {
        "peak_threshold_method": "percentile",
        "peak_threshold_percentile": 91.5,
        "peak_min_distance_sec": 2.75,
        "event_signal": "delta_f",
        "signal_excursion_polarity": "negative",
    }
    folder = tmp_path / "confirmed_features"
    source = _write_source(folder)
    draft = _draft_for(
        folder,
        source,
        feature_values=nondefault,
        feature_status="applied",
    )
    _install_plan(window, monkeypatch, draft, source)
    _prepare(window, qapp)

    prepared = window._guided_continuous_rwd_prepared_run
    assert prepared is not None
    config = prepared.request.config
    for name, value in nondefault.items():
        assert getattr(config, name) == value, name

    calls, seen = _guard_backends(monkeypatch)
    _press_run_and_wait(window, qapp)
    assert calls == {"combined": 1}
    delivered = seen["kwargs"]["config"]
    assert delivered is config
    for name, value in nondefault.items():
        assert getattr(delivered, name) == value, name


def test_widget_edits_after_acceptance_cannot_change_the_prepared_config(
    window, qapp, monkeypatch, tmp_path
):
    folder = tmp_path / "frozen_config"
    source = _write_source(folder)
    draft = _draft_for(
        folder,
        source,
        feature_values={"peak_threshold_percentile": 88.0},
        feature_status="applied",
    )
    state, _binding = _install_plan(window, monkeypatch, draft, source)
    _prepare(window, qapp)
    prepared = window._guided_continuous_rwd_prepared_run
    assert prepared.request.config.peak_threshold_percentile == 88.0

    # A later edit does not reach into the already-prepared Config; it
    # invalidates the prepared run instead.
    state["draft"] = dataclasses.replace(
        draft, feature_event_values={"peak_threshold_percentile": 12.0}
    )
    assert prepared.request.config.peak_threshold_percentile == 88.0
    assert window._guided_continuous_rwd_run_readiness()[0] is False


def test_unconfirmed_feature_settings_fail_config_composition():
    from photometry_pipeline.guided_continuous_rwd_run_config import (
        GuidedContinuousRwdRunConfigError,
        build_guided_continuous_rwd_run_config,
    )
    from photometry_pipeline.guided_execution_payloads import (
        build_guided_execution_startup_mapping_contract,
    )

    draft = GuidedNewAnalysisDraftPlan(
        input_format="rwd",
        acquisition_mode="continuous",
        feature_event_profile_status="missing",
    )
    with pytest.raises(GuidedContinuousRwdRunConfigError) as excinfo:
        build_guided_continuous_rwd_run_config(
            draft, build_guided_execution_startup_mapping_contract()
        )
    assert "Feature Detection" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_preparation_cancellation_never_enables_run(
    window, qapp, monkeypatch, tmp_path
):
    folder = tmp_path / "cancel_prepare"
    source = _write_source(folder)
    draft = _draft_for(folder, source)
    _install_plan(window, monkeypatch, draft, source)

    assert window._maybe_start_guided_continuous_rwd_preparation() is True
    assert window._request_guided_continuous_rwd_cancellation() is True

    assert _pump(
        qapp, lambda: not window._guided_continuous_rwd_preparation_active()
    )
    assert _pump(
        qapp,
        lambda: getattr(window, "_guided_continuous_rwd_prepare_thread", None)
        is None,
        timeout_ms=20_000,
    )
    assert window._guided_continuous_rwd_prepared_run is None
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False


def test_execution_cancellation_is_reported_as_cancelled_not_failed(
    window, qapp, monkeypatch, tmp_path
):
    _prepared_window(window, qapp, monkeypatch, tmp_path, "cancel_run")

    from photometry_pipeline.guided_continuous_rwd_correction_pass import (
        GuidedContinuousRwdCorrectionPassError,
    )

    module = importlib.import_module(
        "photometry_pipeline.guided_continuous_rwd_combined_run"
    )
    observed: dict[str, object] = {}

    def cancelling_backend(*args, **kwargs):
        observed["cancellation_requested"] = kwargs.get("cancellation_requested")
        # Exactly the (type, category) pair the accepted backends already
        # classify as cancellation -- no new worker-side category.
        raise GuidedContinuousRwdCorrectionPassError(
            "segment_correction_pass_interrupted",
            "The analysis was cancelled.",
        )

    monkeypatch.setattr(
        module, "execute_guided_continuous_rwd_combined_run", cancelling_backend
    )

    _press_run_and_wait(window, qapp)

    # The cooperative callable reached the accepted backend contract.
    assert callable(observed["cancellation_requested"])
    assert window._guided_continuous_rwd_status_message == (
        "Continuous analysis was cancelled."
    )
    assert window._guided_continuous_rwd_completed_run_dir is None
    assert window._guided_run_btn.isEnabled() is False


def test_closing_after_preparation_finished_does_not_raise(
    window, qapp, monkeypatch, tmp_path
):
    """A finished preparation thread is deleted by its own deleteLater, which
    can leave a stale wrapper behind. Closing the window must not raise on
    it -- otherwise the scientist cannot close the application."""
    from PySide6.QtGui import QCloseEvent

    _prepared_window(window, qapp, monkeypatch, tmp_path, "close_after_prepare")

    class _DeletedThread:
        def isRunning(self):
            raise RuntimeError(
                "Internal C++ object (PySide6.QtCore.QThread) already deleted."
            )

    window._guided_continuous_rwd_prepare_thread = _DeletedThread()

    assert window._guided_continuous_rwd_preparation_thread_is_running() is False
    assert window._guided_continuous_rwd_prepare_thread is None

    window._guided_continuous_rwd_prepare_thread = _DeletedThread()
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is True


def test_cancel_control_is_visible_only_while_work_is_active(
    window, qapp, monkeypatch, tmp_path
):
    _prepared_window(window, qapp, monkeypatch, tmp_path, "cancel_visibility")
    window._refresh_guided_run_readiness_display()
    assert window._guided_continuous_rwd_cancel_btn.isHidden() is True

    window._guided_continuous_rwd_execution_active = True
    window._refresh_guided_run_readiness_display()
    assert window._guided_continuous_rwd_cancel_btn.isHidden() is False
    window._guided_continuous_rwd_execution_active = False


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


def test_preparation_failure_keeps_run_disabled_and_shows_plain_text(
    window, qapp, monkeypatch, tmp_path
):
    folder = tmp_path / "prep_failure"
    source = _write_source(folder)
    draft = _draft_for(folder, source)
    state, _binding = _install_plan(window, monkeypatch, draft, source)

    module = importlib.import_module(
        "photometry_pipeline.guided_continuous_rwd_target_grid"
    )
    monkeypatch.setattr(
        module,
        "build_guided_continuous_rwd_target_grid",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("deep internal detail")),
    )

    _prepare(window, qapp)

    assert window._guided_continuous_rwd_prepared_run is None
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False
    message = window._guided_continuous_rwd_status_message
    assert message.startswith("Continuous analysis could not be prepared.")
    assert "Traceback" not in message
    assert "RuntimeError" not in message
    assert "deep internal detail" not in message

    # The same plan is not re-attempted, but a corrected plan may prepare again.
    assert window._maybe_start_guided_continuous_rwd_preparation() is False
    monkeypatch.undo()
    state["draft"] = draft
    monkeypatch.setattr(
        window, "_build_guided_new_analysis_draft_plan", lambda: state["draft"]
    )
    window._invalidate_guided_backend_validation("scientist corrected setup")
    window._set_guided_continuous_rwd_review_binding(
        _binding_for(state["draft"], source)
    )
    _prepare(window, qapp)
    assert window._guided_continuous_rwd_prepared_run is not None


def test_preparation_failure_shows_an_accepted_scientist_facing_reason(
    window, qapp, monkeypatch, tmp_path
):
    """A reason the accepted builders wrote for a scientist is shown; an
    unexpected internal error is not."""
    folder = tmp_path / "prep_reason"
    source = _write_source(folder)
    draft = _draft_for(folder, source)
    _install_plan(window, monkeypatch, draft, source)

    from photometry_pipeline.guided_continuous_rwd_run_config import (
        GuidedContinuousRwdRunConfigError,
    )

    config_module = importlib.import_module(
        "photometry_pipeline.guided_continuous_rwd_run_config"
    )

    def refusing(*_args, **_kwargs):
        raise GuidedContinuousRwdRunConfigError(
            "The saved Feature Detection settings are not ready for this "
            "analysis."
        )

    monkeypatch.setattr(
        config_module, "build_guided_continuous_rwd_run_config", refusing
    )

    _prepare(window, qapp)

    assert window._guided_continuous_rwd_prepared_run is None
    message = window._guided_continuous_rwd_status_message
    assert message.startswith("Continuous analysis could not be prepared.")
    assert "Feature Detection" in message
    assert "reason:" not in message
    assert "internal:" not in message
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False


def test_execution_failure_has_no_success_or_results_handoff(
    window, qapp, monkeypatch, tmp_path
):
    _prepared_window(window, qapp, monkeypatch, tmp_path, "run_failure")
    calls, _seen = _guard_backends(monkeypatch)
    module = importlib.import_module(
        "photometry_pipeline.guided_continuous_rwd_combined_run"
    )

    def failing(*_args, **_kwargs):
        calls["combined"] = calls.get("combined", 0) + 1
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(
        module, "execute_guided_continuous_rwd_combined_run", failing
    )

    _press_run_and_wait(window, qapp)

    assert calls == {"combined": 1}
    assert window._guided_continuous_rwd_status_message == (
        "Continuous analysis could not be completed."
    )
    assert window._guided_continuous_rwd_completed_run_dir is None
    assert window._guided_load_completed_run_for_review_btn.isHidden() is True
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_continuous_rwd_execution_active is False
    assert window._guided_backend_execution_active is False


def test_result_that_does_not_classify_as_success_is_not_completed(
    window, qapp, monkeypatch, tmp_path
):
    """A returned object is not proof of a completed analysis."""
    _prepared_window(window, qapp, monkeypatch, tmp_path, "unclassified")
    _calls, _seen = _guard_backends(monkeypatch)

    _press_run_and_wait(window, qapp)

    assert window._guided_continuous_rwd_status_message == (
        "Continuous analysis could not be completed."
    )
    assert window._guided_continuous_rwd_completed_run_dir is None
    assert window._guided_load_completed_run_for_review_btn.isHidden() is True


# ---------------------------------------------------------------------------
# Success: completion classification and the continuous Results handoff
# ---------------------------------------------------------------------------


@pytest.mark.extended
def test_live_workflow_produces_a_completed_run_that_opens_in_results(
    window, qapp, monkeypatch, tmp_path
):
    """One small real combined analysis, driven through the live GUI:
    preparation -> Run action -> CR1-E2 worker -> accepted combined backend
    -> completion classification -> continuous Results."""
    from photometry_pipeline.completed_continuous_rwd_review import (
        load_continuous_run_overview,
    )
    from photometry_pipeline.run_completion_contract import (
        TERMINAL_SUCCESS_CURRENT,
        classify_run_terminal_state,
    )

    folder = tmp_path / "e2e"
    source = _write_source(folder)
    draft = _draft_for(
        folder,
        source,
        feature_status="applied",
        feature_values={
            "peak_threshold_method": "percentile",
            "peak_threshold_percentile": 50.0,
            "peak_min_distance_sec": 1.0,
        },
    )
    _install_plan(window, monkeypatch, draft, source)
    _prepare(window, qapp)

    backend_calls: dict[str, int] = {}
    module = importlib.import_module(
        "photometry_pipeline.guided_continuous_rwd_combined_run"
    )
    real_backend = module.execute_guided_continuous_rwd_combined_run

    def counted(*args, **kwargs):
        backend_calls["combined"] = backend_calls.get("combined", 0) + 1
        return real_backend(*args, **kwargs)

    monkeypatch.setattr(
        module, "execute_guided_continuous_rwd_combined_run", counted
    )

    traversals: dict[str, int] = {}
    pass_module = importlib.import_module(
        "photometry_pipeline.guided_continuous_rwd_correction_pass"
    )
    real_traversal = pass_module.iterate_guided_continuous_rwd_corrected_segments

    def counted_traversal(*args, **kwargs):
        traversals["count"] = traversals.get("count", 0) + 1
        return real_traversal(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "iterate_guided_continuous_rwd_corrected_segments",
        counted_traversal,
    )

    _press_run_and_wait(window, qapp)

    assert backend_calls == {"combined": 1}
    assert traversals == {"count": 1}
    run_dir = window._guided_continuous_rwd_completed_run_dir
    assert run_dir, window._guided_continuous_rwd_status_message
    # One completion message shared with the intermittent path, so the Run
    # page cannot say the analysis both finished and is still running.
    assert window._guided_continuous_rwd_status_message == (
        "Guided analysis completed successfully."
    )

    classification = classify_run_terminal_state(run_dir)
    assert classification.state == TERMINAL_SUCCESS_CURRENT

    overview = load_continuous_run_overview(run_dir)
    assert tuple(overview.included_roi_ids) == ("ROI1", "ROI2")
    assert overview.acquisition_mode == "continuous"
    assert overview.correction_completed is True
    assert overview.tonic_analysis is True
    assert overview.phasic_analysis is True

    # The completed run is offered for review and opens through the accepted
    # continuous Results branch.
    assert window._guided_load_completed_run_for_review_btn.isHidden() is False
    detection_module = importlib.import_module(
        "photometry_pipeline.guided_continuous_rwd_phasic_detection"
    )
    redetections: dict[str, int] = {}
    real_detect = detection_module.detect_guided_continuous_rwd_phasic_features

    def counted_detect(*args, **kwargs):
        redetections["count"] = redetections.get("count", 0) + 1
        return real_detect(*args, **kwargs)

    monkeypatch.setattr(
        detection_module,
        "detect_guided_continuous_rwd_phasic_features",
        counted_detect,
    )

    window._on_guided_load_completed_run_for_review_clicked()
    assert _pump(
        qapp,
        lambda: getattr(window, "_guided_completed_review_loading", False)
        is False,
        timeout_ms=120_000,
    )
    assert window._guided_run_readiness_label.text() == (
        "Completed run loaded for review."
    )
    assert redetections == {}

    # The completed run uses the current manifest-backed native Results
    # workspace. The former legacy continuous tab widget is intentionally not
    # used for this package.
    viewer = window._guided_report_viewer
    assert viewer._native_continuous_mode is True
    assert viewer._native_continuous_artifact_index is not None
    assert [
        viewer._tabs.tabText(i) for i in range(viewer._tabs.count())
    ] == ["Verification", "Tonic", "Phasic Summary"]


# ---------------------------------------------------------------------------
# The Guided execution contract: one analysis, both outputs, no choice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_format,acquisition_mode",
    [
        ("rwd", "intermittent"),
        ("rwd", "continuous"),
        ("npm", "intermittent"),
        ("custom_tabular", "intermittent"),
    ],
)
def test_production_guided_draft_is_always_both(
    window, input_format, acquisition_mode
):
    """The real draft builder -- not a fixture -- always requests both tonic
    and phasic outputs, for every format and acquisition mode Guided offers."""
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText(input_format)
    index = window._guided_acquisition_mode_combo.findData(acquisition_mode)
    assert index >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(index)

    draft = window._build_guided_new_analysis_draft_plan()

    assert draft.execution_intent.execution_mode == "both"


def test_guided_offers_no_analysis_family_control(window):
    """No Guided widget lets a scientist ask for tonic only or phasic only."""
    from PySide6.QtWidgets import QCheckBox, QComboBox, QRadioButton

    guided_tab = window._guided_workflow_tab
    offending: list[str] = []
    for widget_type in (QComboBox, QRadioButton, QCheckBox):
        for widget in guided_tab.findChildren(widget_type):
            texts = [str(widget.objectName()), str(widget.toolTip())]
            if isinstance(widget, QComboBox):
                texts.extend(
                    str(widget.itemText(i)) for i in range(widget.count())
                )
                texts.extend(
                    str(widget.itemData(i)) for i in range(widget.count())
                )
            else:
                texts.append(str(widget.text()))
            blob = " ".join(texts).lower()
            # "tonic output mode"/"tonic timeline" are presentation settings
            # for the tonic output every run already produces, not a choice of
            # whether to produce it.
            if "tonic" in blob and "phasic" in blob:
                offending.append(str(widget.objectName()))
    assert offending == []

    # Full Control's Mode combo is the only analysis-family control, and it
    # lives outside the Guided tab.
    assert window._mode_combo not in guided_tab.findChildren(QComboBox)


def test_full_control_mode_choice_cannot_change_the_guided_draft(window):
    window._set_guided_workflow_mode("new_analysis")
    before = window._build_guided_new_analysis_draft_plan()
    assert before.execution_intent.execution_mode == "both"

    for mode in ("tonic", "phasic", "both"):
        window._mode_combo.setCurrentText(mode)
        assert window._mode_combo.currentText() == mode
        after = window._build_guided_new_analysis_draft_plan()
        assert after.execution_intent.execution_mode == "both"


# ---------------------------------------------------------------------------
# The intermittent path is untouched
# ---------------------------------------------------------------------------


def test_intermittent_run_never_reaches_the_continuous_path(window):
    # CR1-F1-B made "Detect automatically" the default choice; with nothing
    # discovered it resolves to repeated sessions, which is not the continuous
    # path.
    assert window._guided_effective_acquisition_mode() == "intermittent"
    assert window._guided_continuous_rwd_live_draft() is None
    assert window._refresh_guided_continuous_rwd_run_readiness_display() is False
    # The intermittent readiness evaluation still owns the Run affordance.
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness is not None
