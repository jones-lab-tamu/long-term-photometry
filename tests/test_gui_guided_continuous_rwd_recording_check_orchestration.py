from __future__ import annotations

import dataclasses
import inspect
import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import gui.main_window as subject
from gui.main_window import (
    MainWindow,
    _GuidedContinuousRwdRecordingCheckFailure,
    _GuidedContinuousRwdRecordingCheckRequest,
    _GuidedContinuousRwdRecordingCheckSuccess,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisTonicSettingsContract,
)
from photometry_pipeline.guided_plan_identity import (
    compute_guided_new_analysis_draft_plan_identity,
)
from tests.test_guided_continuous_rwd_review_binding import _authorities, _draft


class _FakeThread(QObject):
    started = Signal()
    finished = Signal()
    instances = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.quit_calls = 0
        self.__class__.instances.append(self)

    def start(self):
        self.running = True
        self.started.emit()

    def quit(self, *_args):
        self.quit_calls += 1

    def isRunning(self):
        return self.running

    def finish(self):
        self.running = False
        self.finished.emit()

    def deleteLater(self, *_args):
        pass


class _FakeWorker(QObject):
    stage_changed = Signal(str)
    succeeded = Signal(object)
    failed = Signal(object)
    cancelled = Signal()
    instances = []

    def __init__(self, request):
        super().__init__()
        self.request = request
        self.moved_to = None
        self.run_calls = 0
        self.cancel_calls = 0
        self.__class__.instances.append(self)

    def moveToThread(self, thread):
        self.moved_to = thread

    def run(self):
        self.run_calls += 1

    def request_cancel(self):
        self.cancel_calls += 1

    def deleteLater(self, *_args):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    thread = getattr(instance, "_guided_continuous_rwd_check_thread", None)
    if isinstance(thread, _FakeThread):
        thread.finish()
    instance.close()
    instance.deleteLater()


@pytest.fixture
def fake_runtime(monkeypatch):
    _FakeThread.instances = []
    _FakeWorker.instances = []
    monkeypatch.setattr(subject, "QThread", _FakeThread)
    monkeypatch.setattr(subject, "_GuidedContinuousRwdRecordingCheckWorker", _FakeWorker)
    return _FakeThread, _FakeWorker


@pytest.fixture
def check_case(window, tmp_path, monkeypatch, fake_runtime):
    source, recording, evaluation = _authorities(tmp_path)
    current = {"draft": _draft(source_folder=source.parent)}
    monkeypatch.setattr(
        window, "_build_guided_new_analysis_draft_plan", lambda: current["draft"]
    )
    assert window._start_guided_continuous_rwd_recording_check() is True
    worker = _FakeWorker.instances[-1]
    thread = _FakeThread.instances[-1]
    success = _GuidedContinuousRwdRecordingCheckSuccess(
        request=worker.request,
        recording=recording,
        continuity_evaluation=evaluation,
        current_source_path=str(source),
    )
    return current, source, recording, evaluation, worker, thread, success


def test_initial_ownership_is_empty_and_contains_no_pending_or_scientific_duplicates(window):
    assert window._guided_continuous_rwd_check_thread is None
    assert window._guided_continuous_rwd_check_worker is None
    assert window._guided_continuous_rwd_check_active_token is None
    names = {name for name in vars(window) if name.startswith("_guided_continuous_rwd_")}
    assert not any("pending" in name for name in names)
    assert not any(
        name != "_guided_continuous_rwd_review_binding"
        and ("recording" in name or "evaluation" in name)
        for name in names
    )


def test_start_captures_minimal_request_snapshot_and_launches_one_worker(check_case):
    current, _, _, _, worker, thread, _ = check_case
    draft = current["draft"]
    request = worker.request
    assert isinstance(request, _GuidedContinuousRwdRecordingCheckRequest)
    assert request.selected_acquisition_folder == draft.resolved_input_source_path
    assert request.included_roi_ids == tuple(draft.included_roi_ids)
    assert set(dataclasses.asdict(request)) == {
        "selected_acquisition_folder",
        "included_roi_ids",
    }
    assert worker.parent() is None
    owner_snapshot = thread.parent()._guided_continuous_rwd_check_snapshot
    assert owner_snapshot.input_source_path == draft.input_source_path
    assert owner_snapshot.resolved_input_source_path == draft.resolved_input_source_path
    assert owner_snapshot.input_format == draft.input_format
    assert owner_snapshot.acquisition_mode == draft.acquisition_mode
    assert owner_snapshot.discovered_roi_ids == tuple(draft.discovered_roi_ids)
    assert owner_snapshot.included_roi_ids == tuple(draft.included_roi_ids)
    assert owner_snapshot.excluded_roi_ids == tuple(draft.excluded_roi_ids)
    assert worker.moved_to is thread and worker.run_calls == 1
    assert thread.parent()._guided_continuous_rwd_check_thread is thread


@pytest.mark.parametrize(
    "draft_change",
    [
        {"input_format": "npm"},
        {"acquisition_mode": "intermittent"},
        {"input_source_path": "", "resolved_input_source_path": ""},
    ],
)
def test_malformed_or_noncontinuous_draft_refuses_start(
    window, tmp_path, monkeypatch, fake_runtime, draft_change
):
    source, _, _ = _authorities(tmp_path)
    draft = dataclasses.replace(_draft(source_folder=source.parent), **draft_change)
    monkeypatch.setattr(window, "_build_guided_new_analysis_draft_plan", lambda: draft)
    assert window._start_guided_continuous_rwd_recording_check() is False
    assert _FakeWorker.instances == []
    assert "cannot be checked" in window._guided_continuous_rwd_check_status_label.text()


def test_progress_maps_only_known_stages_and_is_indeterminate(check_case):
    _, _, _, _, worker, _, _ = check_case
    progress = worker.moved_to.parent()._guided_continuous_rwd_check_progress
    label = worker.moved_to.parent()._guided_continuous_rwd_check_status_label
    assert progress.minimum() == progress.maximum() == 0
    assert progress.isTextVisible() is False
    worker.stage_changed.emit("checking_timestamp_continuity")
    assert label.text() == "Checking timestamp continuity…"
    worker.stage_changed.emit("secret_internal_stage")
    assert label.text() == "Checking recording…"
    assert "secret_internal_stage" not in label.text()


def test_success_uses_real_b3a_and_installs_only_through_setter(
    check_case, monkeypatch
):
    current, _, recording, evaluation, worker, thread, success = check_case
    window = thread.parent()
    installed = []
    real_setter = window._set_guided_continuous_rwd_review_binding

    def retaining_setter(binding):
        installed.append(binding)
        real_setter(binding)

    monkeypatch.setattr(window, "_set_guided_continuous_rwd_review_binding", retaining_setter)
    worker.succeeded.emit(success)
    assert len(installed) == 1
    binding = installed[0]
    assert binding.recording is recording
    assert binding.continuity_evaluation is evaluation
    assert binding.current_source_path == success.current_source_path
    assert binding.draft_plan_identity == compute_guided_new_analysis_draft_plan_identity(
        current["draft"]
    )
    assert window._guided_continuous_rwd_review_binding == binding
    assert window._guided_continuous_rwd_check_active_token is None
    assert window._guided_continuous_rwd_check_progress.isHidden()
    assert window._guided_continuous_rwd_check_status_label.text() == (
        "Recording check completed."
    )
    assert thread.quit_calls == 1


def test_b3a_receives_exact_worker_objects_and_source(check_case, monkeypatch):
    import photometry_pipeline.guided_continuous_rwd_review_binding as b3a

    current, _, recording, evaluation, worker, _, success = check_case
    calls = []
    real_builder = b3a.build_guided_continuous_rwd_review_binding

    def capture(draft, **kwargs):
        calls.append((draft, kwargs))
        return real_builder(draft, **kwargs)

    monkeypatch.setattr(b3a, "build_guided_continuous_rwd_review_binding", capture)
    worker.succeeded.emit(success)
    expected = {
        "recording": recording,
        "continuity_evaluation": evaluation,
        "current_source_path": success.current_source_path,
    }
    assert calls[0] == (current["draft"], expected)
    assert calls[1] == (current["draft"], expected)


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
def test_settings_only_change_builds_binding_for_current_draft(check_case, changes):
    current, _, _, _, worker, thread, success = check_case
    current["draft"] = dataclasses.replace(current["draft"], **changes)
    expected_identity = compute_guided_new_analysis_draft_plan_identity(current["draft"])
    worker.succeeded.emit(success)
    assert thread.parent()._guided_continuous_rwd_review_binding.draft_plan_identity == (
        expected_identity
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"input_source_path": "C:/different-source"},
        {"resolved_input_source_path": "C:/different-source"},
        {"input_format": "npm"},
        {"acquisition_mode": "intermittent"},
    ],
)
def test_source_format_or_mode_change_rejects_success(check_case, changes):
    current, _, _, _, worker, thread, success = check_case
    current["draft"] = dataclasses.replace(current["draft"], **changes)
    worker.succeeded.emit(success)
    window = thread.parent()
    assert window._guided_continuous_rwd_review_binding is None
    assert window._guided_continuous_rwd_check_status_label.text() == (
        "Setup changed while the recording was checked. Check the recording again."
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"discovered_roi_ids": ["CH1"]},
        {"included_roi_ids": ["CH1"]},
        {"excluded_roi_ids": ["CH10"]},
    ],
)
def test_roi_change_is_refused_by_real_b3a(check_case, changes):
    current, _, _, _, worker, thread, success = check_case
    current["draft"] = dataclasses.replace(current["draft"], **changes)
    worker.succeeded.emit(success)
    window = thread.parent()
    assert window._guided_continuous_rwd_review_binding is None
    text = window._guided_continuous_rwd_check_status_label.text()
    assert text == "Setup changed while the recording was checked. Check the recording again."
    assert "ROI" not in text and "identity" not in text


def test_malformed_current_draft_rejects_success_without_raw_error(
    check_case, monkeypatch
):
    _, _, _, _, worker, thread, success = check_case
    window = thread.parent()
    monkeypatch.setattr(
        window,
        "_build_guided_new_analysis_draft_plan",
        lambda: (_ for _ in ()).throw(ValueError("secret draft payload")),
    )
    worker.succeeded.emit(success)
    assert window._guided_continuous_rwd_review_binding is None
    assert "secret" not in window._guided_continuous_rwd_check_status_label.text()


def test_superseded_callbacks_cannot_install_or_overwrite_status(check_case):
    _, _, _, _, worker, thread, success = check_case
    window = thread.parent()
    old_token = window._guided_continuous_rwd_check_active_token
    window._cancel_guided_continuous_rwd_recording_check()
    expected = window._guided_continuous_rwd_check_status_label.text()
    worker.stage_changed.emit("checking_timestamp_continuity")
    worker.failed.emit(
        _GuidedContinuousRwdRecordingCheckFailure("inspection", "x", "late failure")
    )
    worker.cancelled.emit()
    worker.succeeded.emit(success)
    assert window._guided_continuous_rwd_check_active_token != old_token
    assert window._guided_continuous_rwd_review_binding is None
    assert window._guided_continuous_rwd_check_status_label.text() == expected


def test_second_start_directly_cancels_without_overlap_or_pending_then_can_restart(
    check_case
):
    _, _, _, _, worker, thread, _ = check_case
    window = thread.parent()
    assert window._start_guided_continuous_rwd_recording_check() is False
    assert worker.cancel_calls == 1
    assert len(_FakeWorker.instances) == 1
    assert not any(
        name.startswith("_guided_continuous_rwd_check_pending")
        for name in vars(window)
    )
    thread.finish()
    assert window._guided_continuous_rwd_check_worker is None
    assert window._start_guided_continuous_rwd_recording_check() is True
    assert len(_FakeWorker.instances) == 2


def test_cancel_is_direct_idempotent_and_keeps_existing_binding(check_case):
    current, source, recording, evaluation, worker, thread, _ = check_case
    window = thread.parent()
    from photometry_pipeline.guided_continuous_rwd_review_binding import (
        build_guided_continuous_rwd_review_binding,
    )

    binding = build_guided_continuous_rwd_review_binding(
        current["draft"],
        recording=recording,
        continuity_evaluation=evaluation,
        current_source_path=source,
    )
    window._set_guided_continuous_rwd_review_binding(binding)
    retained = window._guided_continuous_rwd_review_binding
    window._cancel_guided_continuous_rwd_recording_check()
    window._cancel_guided_continuous_rwd_recording_check()
    assert worker.cancel_calls == 1
    assert window._guided_continuous_rwd_review_binding is retained
    assert window._guided_continuous_rwd_check_progress.isHidden()
    assert window._guided_continuous_rwd_check_status_label.text() == (
        "Recording check cancelled."
    )


@pytest.mark.parametrize("terminal", ["failure", "cancelled"])
def test_replacement_failure_or_cancellation_retains_current_binding(
    check_case, terminal
):
    current, source, recording, evaluation, worker, thread, _ = check_case
    window = thread.parent()
    from photometry_pipeline.guided_continuous_rwd_review_binding import (
        build_guided_continuous_rwd_review_binding,
    )

    binding = build_guided_continuous_rwd_review_binding(
        current["draft"],
        recording=recording,
        continuity_evaluation=evaluation,
        current_source_path=source,
    )
    window._set_guided_continuous_rwd_review_binding(binding)
    retained = window._guided_continuous_rwd_review_binding
    if terminal == "failure":
        worker.failed.emit(
            _GuidedContinuousRwdRecordingCheckFailure(
                "inspection", "source_missing", "Select the recording again."
            )
        )
        assert window._guided_continuous_rwd_check_status_label.text() == (
            "Select the recording again."
        )
    else:
        worker.cancelled.emit()
    assert window._guided_continuous_rwd_review_binding is retained


def test_malformed_failure_uses_generic_bounded_status(check_case):
    _, _, _, _, worker, thread, _ = check_case
    worker.failed.emit(SimpleNamespace(scientist_summary="secret raw exception"))
    text = thread.parent()._guided_continuous_rwd_check_status_label.text()
    assert text == "The recording could not be checked."
    assert "secret" not in text


@pytest.mark.parametrize("terminal", ["success", "failure", "cancelled"])
def test_each_terminal_quits_and_matching_finish_cleans_ownership(check_case, terminal):
    _, _, _, _, worker, thread, success = check_case
    window = thread.parent()
    if terminal == "success":
        worker.succeeded.emit(success)
    elif terminal == "failure":
        worker.failed.emit(
            _GuidedContinuousRwdRecordingCheckFailure("inspection", "x", "failed")
        )
    else:
        worker.cancelled.emit()
    assert thread.quit_calls == 1
    thread.finish()
    assert window._guided_continuous_rwd_check_thread is None
    assert window._guided_continuous_rwd_check_worker is None
    assert window._guided_continuous_rwd_check_progress.isHidden()


def test_old_cleanup_cannot_clear_newer_owned_worker(check_case):
    _, _, _, _, old_worker, old_thread, _ = check_case
    window = old_thread.parent()
    window._cancel_guided_continuous_rwd_recording_check()
    old_thread.finish()
    assert window._start_guided_continuous_rwd_recording_check() is True
    new_worker = window._guided_continuous_rwd_check_worker
    new_thread = window._guided_continuous_rwd_check_thread
    window._cleanup_guided_continuous_rwd_recording_check(
        1, old_worker, old_thread
    )
    assert window._guided_continuous_rwd_check_worker is new_worker
    assert window._guided_continuous_rwd_check_thread is new_thread


def test_unexpected_thread_finish_is_bounded_failure(check_case):
    _, _, _, _, _, thread, _ = check_case
    window = thread.parent()
    thread.finish()
    assert window._guided_continuous_rwd_check_worker is None
    assert window._guided_continuous_rwd_check_status_label.text() == (
        "The recording could not be checked."
    )


def test_close_deferral_cleanup_allows_continued_use_and_rejects_late_success(
    check_case,
):
    _, _, _, _, worker, thread, success = check_case
    window = thread.parent()
    token = window._guided_continuous_rwd_check_active_token
    event = SimpleNamespace(ignored=False, ignore=lambda: setattr(event, "ignored", True))
    window.closeEvent(event)
    assert event.ignored is True
    assert worker.cancel_calls == 1 and thread.quit_calls == 1
    assert window._guided_continuous_rwd_check_closing is True
    assert window._guided_continuous_rwd_check_active_token != token
    worker.succeeded.emit(success)
    assert window._guided_continuous_rwd_review_binding is None
    thread.finish()
    assert window._guided_continuous_rwd_check_closing is False
    assert window._start_guided_continuous_rwd_recording_check() is True
    assert len(_FakeWorker.instances) == 2
    assert "terminate" not in inspect.getsource(MainWindow.closeEvent)


def test_backend_close_deferral_does_not_poison_recording_check_start(
    window, tmp_path, monkeypatch, fake_runtime
):
    source, _, _ = _authorities(tmp_path)
    draft = _draft(source_folder=source.parent)
    monkeypatch.setattr(window, "_build_guided_new_analysis_draft_plan", lambda: draft)
    monkeypatch.setattr(subject.QMessageBox, "information", lambda *_args: None)
    window._guided_backend_execution_active = True
    event = SimpleNamespace(ignored=False, ignore=lambda: setattr(event, "ignored", True))

    window.closeEvent(event)

    assert event.ignored is True
    assert window._guided_continuous_rwd_check_closing is False
    window._guided_backend_execution_active = False
    assert window._start_guided_continuous_rwd_recording_check() is True


def test_accepted_close_rejects_callbacks_from_cleaned_recording_check(check_case):
    _, _, _, _, worker, thread, success = check_case
    window = thread.parent()
    deferred = QCloseEvent()
    window.closeEvent(deferred)
    assert deferred.isAccepted() is False
    thread.finish()
    assert window._guided_continuous_rwd_check_closing is False

    accepted = QCloseEvent()
    window.closeEvent(accepted)
    assert accepted.isAccepted() is True
    status = window._guided_continuous_rwd_check_status_label.text()
    worker.succeeded.emit(success)
    assert window._guided_continuous_rwd_review_binding is None
    assert window._guided_continuous_rwd_check_status_label.text() == status


def test_real_worker_scientific_entry_runs_off_gui_thread(
    window, tmp_path, monkeypatch
):
    import photometry_pipeline.io.rwd_continuous_source as cr1a

    source, _, _ = _authorities(tmp_path)
    draft = _draft(source_folder=source.parent)
    monkeypatch.setattr(window, "_build_guided_new_analysis_draft_plan", lambda: draft)
    worker_threads = []

    def refuse(*_args, **_kwargs):
        worker_threads.append(threading.get_ident())
        return SimpleNamespace(
            status="failed",
            outcome_category="fluorescence_csv_missing",
            scientist_summary="Select the recording again.",
        )

    monkeypatch.setattr(cr1a, "inspect_continuous_rwd_acquisition_folder", refuse)
    gui_thread = threading.get_ident()
    assert window._start_guided_continuous_rwd_recording_check() is True
    for _ in range(200):
        QApplication.processEvents()
        if window._guided_continuous_rwd_check_thread is None:
            break
        QTest.qWait(5)
    assert worker_threads and worker_threads[0] != gui_thread
    assert window._guided_continuous_rwd_check_thread is None


def test_success_keeps_validation_run_and_visible_capability_closed(check_case):
    _, _, _, _, worker, thread, success = check_case
    window = thread.parent()
    worker.succeeded.emit(success)
    assert window._guided_backend_validation_outcome is None
    assert window._guided_validated_plan_identity is None
    assert window._guided_startup_authority is None
    assert window._guided_execution_payload_result is None
    assert window._guided_startup_transaction_request is None
    assert window._guided_run_readiness.status == "no_validation"
    assert window._guided_review_go_to_run_btn.isEnabled() is False
    visible_modes = {
        button.text().strip().lower()
        for button in window.findChildren(subject.QPushButton)
        if button.isVisible()
    }
    assert "continuous" not in visible_modes
    assert not any("check recording" in text for text in visible_modes)


def test_orchestration_has_no_persistence_execution_or_scientific_processing_surface():
    methods = (
        MainWindow._start_guided_continuous_rwd_recording_check,
        MainWindow._cancel_guided_continuous_rwd_recording_check,
        MainWindow._on_guided_continuous_rwd_check_succeeded,
        MainWindow._cleanup_guided_continuous_rwd_recording_check,
    )
    lowered = "\n".join(inspect.getsource(method) for method in methods).lower()
    for forbidden in (
        "qsettings",
        "json",
        "serializ",
        "cache",
        "target_grid",
        "interpolat",
        "backend_validation_workflow",
        "materializ",
        "authorization",
        "startup_authority",
        "execute_guided",
        "completion",
        "inspect_continuous_rwd_acquisition_folder",
        "evaluate_continuous_rwd_timestamp_continuity",
    ):
        assert forbidden not in lowered
