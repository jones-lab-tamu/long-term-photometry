from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

import gui.main_window as main_window_module
import photometry_pipeline.guided_npm_run_launch_builder as npm_builder_module
import photometry_pipeline.guided_npm_worker_launch as npm_launch_module
import photometry_pipeline.guided_npm_worker_reconciliation as npm_reconciliation_module
from gui.main_window import MainWindow, _GuidedNpmRunWorker
from photometry_pipeline.guided_npm_run_launch_builder import (
    GuidedNpmRunLaunchBuildResult,
)
from photometry_pipeline.guided_npm_worker_launch import (
    GuidedNpmLaunchedWorkerRuntime,
    GuidedNpmPostLaunchRuntime,
    GuidedNpmWorkerLaunchCancelled,
    GuidedNpmWorkerLaunchFailure,
    GuidedNpmWorkerLaunchIssue,
)
from tests.test_npm_user_language import _accepted_npm_outcome


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance._guided_backend_execution_active = False
    thread = getattr(instance, "_guided_npm_run_worker_thread", None)
    if thread is not None and thread.isRunning():
        thread.quit()
        thread.wait(2000)
    instance.close()
    instance.deleteLater()


def _pump_until(qapp, condition, *, timeout_s: float = 5.0) -> None:
    """Process GUI-thread events until `condition()` is true.

    Guided NPM Run now builds, launches, and reconciles entirely on a
    worker thread; results reach the GUI thread via queued signals, which
    are only delivered when the GUI event loop is pumped.
    """
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met before timeout")
        qapp.processEvents()


def _set_npm_ready(window, *, revision: int = 4):
    window._guided_format_combo.setCurrentText("npm")
    outcome = _accepted_npm_outcome()
    window._guided_backend_validation_revision = revision
    window._guided_backend_validation_outcome = outcome
    window._guided_backend_validation_outcome_revision = revision
    window._guided_backend_execution_result = None
    window._guided_backend_execution_active = False
    window._refresh_guided_run_readiness_display()
    return outcome


def _fake_build_ok(**_kwargs) -> GuidedNpmRunLaunchBuildResult:
    return GuidedNpmRunLaunchBuildResult(
        status="built",
        ok=True,
        prelaunch_claim=SimpleNamespace(run_directory_path=r"C:\fake\npm-run"),
        application_build_identity=SimpleNamespace(),
        blocking_issues=(),
        current_gui_revision=4,
    )


def _fake_build_failed(status: str = "production_mapping_failed") -> GuidedNpmRunLaunchBuildResult:
    return GuidedNpmRunLaunchBuildResult(
        status=status,
        ok=False,
        prelaunch_claim=None,
        application_build_identity=None,
        blocking_issues=(),
        current_gui_revision=4,
    )


def _fake_handle(pid: int = 4242, wait_result: int = 0):
    return SimpleNamespace(pid=pid, wait=lambda timeout=None: wait_result)


def _fake_launched_runtime():
    return GuidedNpmLaunchedWorkerRuntime(
        prelaunch_claim=SimpleNamespace(run_directory_path=r"C:\fake\npm-run"),
        launch_invocation=SimpleNamespace(),
        launch_context=SimpleNamespace(),
        execution_start_receipt=SimpleNamespace(),
        process_handle=_fake_handle(),
    )


def _fake_post_launch_runtime():
    return GuidedNpmPostLaunchRuntime(
        prelaunch_claim=SimpleNamespace(run_directory_path=r"C:\fake\npm-run"),
        launch_invocation=SimpleNamespace(),
        launch_context=SimpleNamespace(),
        process_handle=_fake_handle(),
        process_id=4242,
        launch_failure=SimpleNamespace(blocking_issues=()),
    )


def _fake_launch_failure(message: str = "setup issue", category: str = "launch_internal_error"):
    return GuidedNpmWorkerLaunchFailure(
        blocking_issues=(
            GuidedNpmWorkerLaunchIssue(
                category=category,
                section="launch",
                message=message,
                detail_code="test_detail",
            ),
        )
    )


def _fake_launch_cancelled():
    return GuidedNpmWorkerLaunchCancelled(
        blocking_issues=(
            GuidedNpmWorkerLaunchIssue(
                category="launch_cancelled",
                section="launch",
                message="cancelled",
                detail_code="test_detail",
            ),
        )
    )


def _install_success_path(monkeypatch):
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fake_build_ok,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: _fake_launched_runtime(),
    )


# ---------------------------------------------------------------------------
# Enablement (section 37)
# ---------------------------------------------------------------------------


def test_validated_supported_npm_enables_run(window):
    _set_npm_ready(window)
    assert window._guided_run_btn.isEnabled() is True
    assert window._guided_npm_run_readiness.status == "ready"


def test_pre_validation_npm_does_not_enable_run(window):
    window._guided_format_combo.setCurrentText("npm")
    window._guided_backend_validation_outcome = None
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_npm_run_readiness.status == "no_validation"


def test_validation_failure_does_not_enable_run(window):
    from tests.test_guided_npm_production_mapping import _unsafe_replace

    outcome = _set_npm_ready(window)
    refused = _unsafe_replace(
        outcome, status="validator_refused", accepted_for_backend_validation=False
    )
    window._guided_backend_validation_outcome = refused
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_npm_run_readiness.status == "validation_not_accepted"


def test_stale_validation_disables_run(window):
    _set_npm_ready(window)
    assert window._guided_run_btn.isEnabled() is True
    window._guided_backend_validation_revision += 1
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_npm_run_readiness.status == "validation_stale"


def test_active_run_disables_run(window):
    _set_npm_ready(window)
    window._guided_backend_execution_active = True
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_npm_run_readiness.status == "run_active"


def test_completed_run_keeps_run_disabled(window):
    _set_npm_ready(window)
    window._guided_backend_execution_result = object()
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_npm_run_readiness.status == "result_pending"


def test_custom_tabular_does_not_use_npm_readiness_path(window):
    window._guided_format_combo.setCurrentText("custom_tabular")
    window._refresh_guided_run_readiness_display()
    # custom_tabular is routed through the unmodified RWD evaluator, not
    # the new NPM predicate at all.
    assert window._guided_npm_run_readiness is None or (
        window._guided_run_target_is_npm() is False
    )
    assert window._guided_run_target_is_npm() is False


def test_rwd_format_still_uses_unmodified_rwd_evaluator(window):
    window._guided_format_combo.setCurrentText("rwd")
    assert window._guided_run_target_is_npm() is False


# ---------------------------------------------------------------------------
# Launch tests (section 38) -- build/launch/reconcile now run on a worker
# thread; assertions immediately after `.click()` only see the cheap,
# synchronous GUI-thread prechecks (preparing state), and later assertions
# must pump the event loop for queued cross-thread signals.
# ---------------------------------------------------------------------------


def test_one_click_launches_exactly_once_and_enters_running_state(
    window, monkeypatch, qapp
):
    _set_npm_ready(window)
    launch_calls = []

    def fake_launch(claim, **kwargs):
        launch_calls.append((claim, kwargs))
        return _fake_launched_runtime()

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fake_build_ok,
    )
    monkeypatch.setattr(
        npm_launch_module, "launch_guided_npm_worker_runtime", fake_launch
    )
    release = threading.Event()

    def slow_reconcile(_runtime):
        assert release.wait(timeout=5)
        raise RuntimeError("stop before completion for this test")

    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        slow_reconcile,
    )

    window._guided_run_btn.click()

    assert window._guided_backend_execution_active is True
    assert window._guided_run_btn.isEnabled() is False
    assert "preparing" in window._guided_run_readiness_label.text().lower()

    _pump_until(qapp, lambda: window._guided_npm_launch_runtime is not None)
    assert len(launch_calls) == 1
    assert "running" in window._guided_run_readiness_label.text().lower()

    release.set()
    _pump_until(qapp, lambda: window._guided_npm_run_worker_thread is None)
    assert len(launch_calls) == 1


def test_double_click_launches_once(window, monkeypatch, qapp):
    _set_npm_ready(window)
    launch_calls = []

    def fake_launch(claim, **kwargs):
        launch_calls.append(claim)
        return _fake_launched_runtime()

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fake_build_ok,
    )
    monkeypatch.setattr(
        npm_launch_module, "launch_guided_npm_worker_runtime", fake_launch
    )
    release = threading.Event()

    def slow_reconcile(_runtime):
        assert release.wait(timeout=5)
        raise RuntimeError("stop before completion for this test")

    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        slow_reconcile,
    )

    window._guided_run_btn.click()
    assert window._guided_run_btn.isEnabled() is False
    window._guided_run_btn.click()
    window._on_guided_run_clicked_backend_guarded()

    release.set()
    _pump_until(qapp, lambda: window._guided_npm_run_worker_thread is None)
    assert len(launch_calls) == 1


def test_pre_process_launch_failure_never_enters_running_state(
    window, monkeypatch, qapp
):
    _set_npm_ready(window)
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fake_build_ok,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: _fake_launch_failure(),
    )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)

    assert window._guided_npm_launch_runtime is None
    assert "could not be started" in window._guided_run_readiness_label.text()
    assert window._guided_run_btn.isEnabled() is False
    assert (
        window._guided_run_execution_details_label.text()
        == "The application could not start the analysis."
    )
    assert "setup issue" not in window._guided_run_execution_details_label.text()


def test_launch_cancellation_shows_neutral_not_started(window, monkeypatch, qapp):
    _set_npm_ready(window)
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fake_build_ok,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: _fake_launch_cancelled(),
    )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)

    assert "not started" in window._guided_run_readiness_label.text().lower()
    assert "fail" not in window._guided_run_readiness_label.text().lower()


def test_post_launch_runtime_enters_unconfirmed_active_path(
    window, monkeypatch, qapp
):
    _set_npm_ready(window)
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fake_build_ok,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: _fake_post_launch_runtime(),
    )
    post_launch_calls = []

    def fake_post_launch_reconcile(runtime):
        post_launch_calls.append(runtime)
        return SimpleNamespace(final_outcome="post_launch_evidence_failed")

    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_post_launch_runtime",
        fake_post_launch_reconcile,
    )

    window._guided_run_btn.click()
    assert window._guided_backend_execution_active is True

    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)
    assert len(post_launch_calls) == 1
    assert (
        "could not confirm"
        in window._guided_run_readiness_label.text().lower()
    )


def test_stale_authority_blocks_launch(window, monkeypatch):
    _set_npm_ready(window)
    build_calls = []
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        lambda **kwargs: build_calls.append(kwargs) or _fake_build_ok(),
    )
    # Simulate the setup changing after the readiness check but before the
    # click handler captures a fresh context, by bumping the revision.
    # This check happens on the GUI thread before any worker starts, so no
    # pump is needed here.
    window._guided_backend_validation_revision += 1

    window._guided_run_btn.click()

    assert build_calls == []
    assert window._guided_backend_execution_active is False


# ---------------------------------------------------------------------------
# Preserved launch-attempt messages (B2-E1 narrow follow-up, section 13)
# ---------------------------------------------------------------------------


def test_stale_context_message_survives_the_handler_returning(window, monkeypatch):
    _set_npm_ready(window)
    # Force the stale-context branch: bump the revision after the initial
    # readiness recheck already passed, so the freshly-captured context no
    # longer matches. This check stays on the GUI thread, so no pump is
    # needed.
    monkeypatch.setattr(
        window,
        "_capture_guided_backend_validation_context",
        lambda: SimpleNamespace(
            revision=window._guided_backend_validation_revision + 1
        ),
    )

    window._guided_run_btn.click()

    final_text = window._guided_run_readiness_label.text()
    assert "changed after it was checked" in final_text
    assert "ready to run" not in final_text.lower()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_backend_execution_active is False


def test_builder_failure_message_survives_the_handler_returning(
    window, monkeypatch, qapp
):
    _set_npm_ready(window)
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        lambda **kwargs: _fake_build_failed(),
    )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)

    final_text = window._guided_run_readiness_label.text()
    assert "could not be started" in final_text
    assert "ready to run" not in final_text.lower()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_backend_execution_active is False


def test_unexpected_launch_outcome_message_survives_the_handler_returning(
    window, monkeypatch, qapp
):
    _set_npm_ready(window)
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fake_build_ok,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: object(),
    )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)

    final_text = window._guided_run_readiness_label.text().lower()
    assert "could not confirm" in final_text
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_backend_execution_active is False


# ---------------------------------------------------------------------------
# Threading (sections 39 / narrow-follow-up sections 19-20)
# ---------------------------------------------------------------------------


def test_run_worker_does_not_reference_main_window():
    import inspect

    init_params = list(
        inspect.signature(_GuidedNpmRunWorker.__init__).parameters
    )
    assert init_params == [
        "self",
        "validation_context",
        "validation_outcome",
        "current_gui_revision",
    ]

    run_source = inspect.getsource(_GuidedNpmRunWorker.run)
    assert "build_guided_npm_worker_prelaunch_claim_from_validation(" in run_source
    assert "launch_guided_npm_worker_runtime(" in run_source
    assert "reconcile_guided_npm_worker_runtime(" in run_source
    assert "reconcile_guided_npm_post_launch_runtime(" in run_source
    prohibited = (
        "MainWindow",
        "self._window",
        "self.window",
        "_guided_run_btn",
        "_guided_run_readiness_label",
        "_guided_backend_execution_active",
        "_capture_guided_backend_validation_context",
    )
    for term in prohibited:
        assert term not in run_source, f"{term!r} leaked into worker.run source"


def test_builder_runs_off_gui_thread(window, monkeypatch, qapp):
    _set_npm_ready(window)
    gui_thread_id = threading.get_ident()
    captured = {}

    def recording_build(**kwargs):
        captured["build_thread_id"] = threading.get_ident()
        return _fake_build_ok(**kwargs)

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        recording_build,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: _fake_launched_runtime(),
    )
    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        lambda runtime: SimpleNamespace(
            final_outcome="verified_completed", run_directory_path="x"
        ),
    )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)

    assert "build_thread_id" in captured
    assert captured["build_thread_id"] != gui_thread_id


def test_launch_and_reconciliation_run_on_same_worker_thread_off_gui(
    window, monkeypatch, qapp
):
    _set_npm_ready(window)
    gui_thread_id = threading.get_ident()
    captured = {}

    def recording_launch(claim, **kwargs):
        captured["launch_thread_id"] = threading.get_ident()
        return _fake_launched_runtime()

    def recording_reconcile(runtime):
        captured["reconcile_thread_id"] = threading.get_ident()
        return SimpleNamespace(
            final_outcome="verified_completed", run_directory_path="x"
        )

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fake_build_ok,
    )
    monkeypatch.setattr(
        npm_launch_module, "launch_guided_npm_worker_runtime", recording_launch
    )
    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        recording_reconcile,
    )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)

    assert captured["launch_thread_id"] != gui_thread_id
    assert captured["reconcile_thread_id"] != gui_thread_id
    assert captured["launch_thread_id"] == captured["reconcile_thread_id"]


def test_gui_result_handlers_run_on_gui_thread(window, monkeypatch, qapp):
    _set_npm_ready(window)
    gui_thread_id = threading.get_ident()
    _install_success_path(monkeypatch)
    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        lambda runtime: SimpleNamespace(
            final_outcome="verified_completed", run_directory_path="x"
        ),
    )
    captured = {}
    original_finish = window._finish_guided_npm_run_with_result

    def recording_finish(result):
        captured["thread_id"] = threading.get_ident()
        return original_finish(result)

    monkeypatch.setattr(window, "_finish_guided_npm_run_with_result", recording_finish)

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)

    assert captured["thread_id"] == gui_thread_id


def test_blocked_builder_keeps_gui_responsive_and_shows_preparing(
    window, monkeypatch, qapp
):
    _set_npm_ready(window)
    validate_btn = window._guided_backend_validate_btn
    entered = threading.Event()
    release = threading.Event()

    def blocking_build(**kwargs):
        entered.set()
        assert release.wait(timeout=5), "release was never set"
        return _fake_build_ok(**kwargs)

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        blocking_build,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: _fake_launched_runtime(),
    )
    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        lambda runtime: SimpleNamespace(
            final_outcome="verified_completed", run_directory_path="x"
        ),
    )

    window._guided_run_btn.click()

    # Control returned to the test immediately; the worker thread is
    # blocked inside the builder.
    assert entered.wait(timeout=5), "the builder never started"
    assert window._guided_run_btn.isEnabled() is False
    assert validate_btn.isEnabled() is False
    assert (
        "preparing your npm analysis"
        in window._guided_run_readiness_label.text().lower()
    )

    # Qt events can still be processed while the builder blocks -- this
    # would hang if preparation still ran on the GUI thread.
    qapp.processEvents()

    release.set()
    _pump_until(qapp, lambda: window._guided_npm_run_worker_thread is None)


def test_preparing_to_running_transition(window, monkeypatch, qapp):
    _set_npm_ready(window)
    validate_btn = window._guided_backend_validate_btn
    _install_success_path(monkeypatch)
    release = threading.Event()

    def slow_reconcile(_runtime):
        assert release.wait(timeout=5)
        return SimpleNamespace(
            final_outcome="verified_completed", run_directory_path=r"C:\fake\npm-run"
        )

    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        slow_reconcile,
    )

    window._guided_run_btn.click()
    assert "preparing" in window._guided_run_readiness_label.text().lower()

    _pump_until(qapp, lambda: window._guided_npm_launch_runtime is not None)
    assert (
        window._guided_run_readiness_label.text()
        == "Your NPM analysis is running."
    )
    assert window._guided_backend_execution_active is True
    assert window._guided_run_btn.isEnabled() is False
    assert validate_btn.isEnabled() is False
    assert window._guided_backend_execution_result is None

    release.set()
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)
    assert (
        window._guided_run_readiness_label.text()
        == "Your NPM analysis finished successfully."
    )


def _wrap_counting_finish(window, monkeypatch):
    calls = []
    original = window._finish_guided_npm_run_with_result

    def counting_finish(result):
        calls.append(result)
        return original(result)

    monkeypatch.setattr(window, "_finish_guided_npm_run_with_result", counting_finish)
    return calls


_SUCCESS_RESULT = SimpleNamespace(
    final_outcome="verified_completed", run_directory_path=r"C:\fake\npm-run"
)
_UNEXPECTED_FAILURE = SimpleNamespace(stage="unexpected_error")


@pytest.mark.parametrize(
    "first_callback,first_arg,second_callback,second_arg,expected_first_text",
    [
        (
            "succeeded",
            _SUCCESS_RESULT,
            "succeeded",
            _SUCCESS_RESULT,
            "finished successfully",
        ),
        (
            "failed",
            _UNEXPECTED_FAILURE,
            "failed",
            _UNEXPECTED_FAILURE,
            "could not confirm",
        ),
        (
            "succeeded",
            _SUCCESS_RESULT,
            "failed",
            _UNEXPECTED_FAILURE,
            "finished successfully",
        ),
        (
            "failed",
            _UNEXPECTED_FAILURE,
            "succeeded",
            _SUCCESS_RESULT,
            "could not confirm",
        ),
    ],
    ids=[
        "success_then_success",
        "failure_then_failure",
        "success_then_failure",
        "failure_then_success",
    ],
)
def test_only_the_first_terminal_callback_is_applied(
    window,
    monkeypatch,
    first_callback,
    first_arg,
    second_callback,
    second_arg,
    expected_first_text,
):
    _set_npm_ready(window)
    window._guided_backend_execution_active = True
    calls = _wrap_counting_finish(window, monkeypatch)

    getattr(window, f"_on_guided_npm_run_worker_{first_callback}")(first_arg)
    assert len(calls) == 1
    assert window._guided_backend_execution_active is False
    first_text = window._guided_run_readiness_label.text().lower()
    assert expected_first_text in first_text
    stored_result = window._guided_backend_execution_result

    # A duplicate or late-queued signal for the same run must be ignored
    # completely: no second finish call, no change to visible text, stored
    # result, active state, or button state.
    getattr(window, f"_on_guided_npm_run_worker_{second_callback}")(second_arg)

    assert len(calls) == 1
    assert window._guided_backend_execution_result is stored_result
    assert window._guided_run_readiness_label.text().lower() == first_text
    assert window._guided_backend_execution_active is False


def test_launched_signal_is_not_a_terminal_callback(window, monkeypatch):
    """The launched signal must never consume the terminal-callback guard:
    it leaves `_guided_backend_execution_active` untouched, and a terminal
    callback delivered afterward must still be the one that applies."""
    _set_npm_ready(window)
    calls = _wrap_counting_finish(window, monkeypatch)
    window._guided_backend_execution_active = True

    window._on_guided_npm_run_worker_launched(_fake_launched_runtime(), False)
    assert window._guided_backend_execution_active is True
    assert calls == []

    window._on_guided_npm_run_worker_succeeded(_SUCCESS_RESULT)
    assert len(calls) == 1
    assert window._guided_backend_execution_active is False


def test_launched_signal_alone_never_shows_success(window):
    window._guided_backend_execution_active = True
    window._on_guided_npm_run_worker_launched(_fake_launched_runtime(), False)
    label_text = window._guided_run_readiness_label.text().lower()
    assert "finished successfully" not in label_text
    assert window._guided_backend_execution_result is None


# ---------------------------------------------------------------------------
# Result mapping (section 40) via the GUI finish handler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "final_outcome,expected_fragment",
    [
        ("verified_completed", "finished successfully"),
        ("verified_failed_before_consumed_authority", "did not finish"),
        ("verified_failed_after_consumed_authority", "did not finish"),
        ("verified_failed_during_output_finalization", "did not finish"),
        ("process_failed_without_terminal_evidence", "did not finish"),
        ("authority_refused", "did not finish"),
        ("terminal_receipt_publication_failed", "could not confirm"),
        ("process_exited_zero_without_terminal_evidence", "could not confirm"),
        ("terminal_evidence_invalid", "could not confirm"),
        ("consumed_authority_evidence_invalid", "could not confirm"),
        ("process_identity_mismatch", "could not confirm"),
        ("completed_output_integrity_failed", "could not confirm"),
        ("indeterminate", "could not confirm"),
        ("post_launch_evidence_failed", "could not confirm"),
        ("some_unrecognized_future_outcome", "could not confirm"),
    ],
)
def test_every_reconciliation_outcome_maps_to_exactly_one_category(
    window, final_outcome, expected_fragment
):
    result = SimpleNamespace(
        final_outcome=final_outcome, run_directory_path=r"C:\fake\npm-run"
    )
    window._finish_guided_npm_run_with_result(result)
    assert (
        expected_fragment in window._guided_run_readiness_label.text().lower()
    )


def test_only_verified_completed_shows_output_directory(window):
    success = SimpleNamespace(
        final_outcome="verified_completed", run_directory_path=r"C:\fake\npm-run"
    )
    window._finish_guided_npm_run_with_result(success)
    assert r"C:\fake\npm-run" in window._guided_run_execution_details_label.text()

    failure = SimpleNamespace(final_outcome="authority_refused", run_directory_path=r"C:\fake\npm-run")
    window._finish_guided_npm_run_with_result(failure)
    assert r"C:\fake\npm-run" not in window._guided_run_execution_details_label.text()


# ---------------------------------------------------------------------------
# Open-results-folder affordance and post-completion cleanup (B2-E2A)
# ---------------------------------------------------------------------------


def test_verified_success_exposes_open_output_button_bound_to_exact_path(
    window, tmp_path
):
    run_dir = str(tmp_path)  # a real, existing directory
    window._finish_guided_npm_run_with_result(
        SimpleNamespace(final_outcome="verified_completed", run_directory_path=run_dir)
    )
    assert window._guided_npm_open_output_btn.isHidden() is False
    assert window._guided_npm_open_output_btn.isEnabled() is True
    assert window._guided_npm_completed_output_dir == run_dir
    assert run_dir in window._guided_run_execution_details_label.text()


def test_open_output_button_targets_exact_stored_path(window, tmp_path, monkeypatch):
    run_dir = str(tmp_path)
    opened = []
    monkeypatch.setattr(main_window_module, "_open_folder", lambda p: opened.append(p))
    window._finish_guided_npm_run_with_result(
        SimpleNamespace(final_outcome="verified_completed", run_directory_path=run_dir)
    )
    window._on_guided_npm_open_output_folder_clicked()
    assert opened == [run_dir]


@pytest.mark.parametrize(
    "final_outcome",
    [
        "authority_refused",
        "verified_failed_before_consumed_authority",
        "indeterminate",
        "post_launch_evidence_failed",
        "terminal_evidence_invalid",
    ],
)
def test_open_output_button_hidden_for_failure_and_unconfirmed(
    window, tmp_path, final_outcome
):
    window._finish_guided_npm_run_with_result(
        SimpleNamespace(final_outcome=final_outcome, run_directory_path=str(tmp_path))
    )
    assert window._guided_npm_open_output_btn.isHidden() is True
    assert window._guided_npm_completed_output_dir is None


def test_open_output_button_hidden_for_unexpected_error(window):
    from photometry_pipeline.guided_npm_run_result_presentation import (
        GuidedNpmRunUnexpectedError,
    )

    window._finish_guided_npm_run_with_result(GuidedNpmRunUnexpectedError("boom"))
    assert window._guided_npm_open_output_btn.isHidden() is True
    assert window._guided_npm_completed_output_dir is None


def test_editing_setup_after_success_clears_output_line_and_open_button(
    window, tmp_path
):
    _set_npm_ready(window)
    run_dir = str(tmp_path)
    window._finish_guided_npm_run_with_result(
        SimpleNamespace(final_outcome="verified_completed", run_directory_path=run_dir)
    )
    assert run_dir in window._guided_run_execution_details_label.text()
    assert window._guided_npm_open_output_btn.isHidden() is False

    # Any setup-affecting change clears the completed result; the stale
    # output-folder line and its button must go with it so they can never
    # be confused with the newly edited draft setup.
    window._invalidate_guided_backend_validation("user changed a setting")

    assert window._guided_backend_execution_result is None
    assert window._guided_run_execution_details_label.text() == ""
    assert window._guided_npm_open_output_btn.isHidden() is True
    assert window._guided_npm_completed_output_dir is None
    assert (
        "changed after it was checked"
        in window._guided_run_readiness_label.text()
    )


def test_open_output_missing_folder_is_handled_gracefully(window, monkeypatch):
    opened = []
    monkeypatch.setattr(main_window_module, "_open_folder", lambda p: opened.append(p))
    window._finish_guided_npm_run_with_result(
        SimpleNamespace(
            final_outcome="verified_completed",
            run_directory_path=r"C:\definitely\does\not\exist_zzz",
        )
    )
    window._on_guided_npm_open_output_folder_clicked()
    # No crash, no open attempt on a non-existent path, and a plain message.
    assert opened == []
    text = window._guided_run_execution_details_label.text().lower()
    assert "could not be found" in text
    for term in ("artifact", "receipt", "backend", "json", "worker", "exception"):
        assert term not in text


def test_open_output_button_label_is_scientist_facing(window):
    label_text = window._guided_npm_open_output_btn.text().lower()
    for term in ("artifact", "receipt", "backend", "json", "worker", "run directory"):
        assert term not in label_text
    assert "results" in label_text


def test_new_run_start_hides_stale_open_output_button(window, monkeypatch, qapp):
    # A prior success left the button visible; starting a fresh run must
    # hide it during preparation before any new result exists.
    window._finish_guided_npm_run_with_result(
        SimpleNamespace(final_outcome="verified_completed", run_directory_path=r"C:\prev\run")
    )
    assert window._guided_npm_open_output_btn.isHidden() is False

    _set_npm_ready(window)
    release = threading.Event()

    def blocking_build(**kwargs):
        assert release.wait(timeout=5)
        return _fake_build_ok(**kwargs)

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        blocking_build,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: _fake_launched_runtime(),
    )
    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        lambda runtime: SimpleNamespace(
            final_outcome="indeterminate", run_directory_path="x"
        ),
    )

    window._guided_run_btn.click()
    assert window._guided_npm_open_output_btn.isHidden() is True
    assert window._guided_npm_completed_output_dir is None
    release.set()
    _pump_until(qapp, lambda: window._guided_npm_run_worker_thread is None)


# ---------------------------------------------------------------------------
# Stale NPM handoff cleared on format switch (B2-E2A narrow follow-up)
# ---------------------------------------------------------------------------


def _complete_npm_success(window, run_dir):
    _set_npm_ready(window)
    window._finish_guided_npm_run_with_result(
        SimpleNamespace(final_outcome="verified_completed", run_directory_path=run_dir)
    )


@pytest.mark.parametrize("other_format", ["rwd", "custom_tabular"])
def test_switching_format_clears_stale_npm_completion_handoff(
    window, tmp_path, other_format
):
    run_dir = str(tmp_path)
    _complete_npm_success(window, run_dir)
    assert window._guided_npm_open_output_btn.isHidden() is False
    assert window._guided_npm_completed_output_dir == run_dir
    assert run_dir in window._guided_run_execution_details_label.text()

    window._guided_format_combo.setCurrentText(other_format)
    window._refresh_guided_run_readiness_display()

    assert window._guided_npm_open_output_btn.isHidden() is True
    assert window._guided_npm_open_output_btn.isEnabled() is False
    assert window._guided_npm_completed_output_dir is None
    assert run_dir not in window._guided_run_execution_details_label.text()


def test_switching_to_rwd_preserves_legitimate_rwd_readiness_text(window, tmp_path):
    from photometry_pipeline.guided_run_readiness import _SUMMARIES

    _complete_npm_success(window, str(tmp_path))
    window._guided_format_combo.setCurrentText("rwd")
    window._refresh_guided_run_readiness_display()

    text = window._guided_run_readiness_label.text()
    # The readiness label must carry the RWD evaluator's own legitimate
    # message -- not be blanked, and not still show the NPM success text.
    assert text != "Your NPM analysis finished successfully."
    assert text in set(_SUMMARIES.values())
    assert window._guided_run_readiness_label.text() != ""


def test_active_npm_success_handoff_survives_readiness_refresh(window, tmp_path):
    # While the format stays NPM and the verified result is current, a
    # readiness refresh must not strip the handoff (guards against
    # over-broad cleanup).
    run_dir = str(tmp_path)
    _complete_npm_success(window, run_dir)
    window._refresh_guided_run_readiness_display()

    assert window._guided_npm_open_output_btn.isHidden() is False
    assert window._guided_npm_completed_output_dir == run_dir
    assert run_dir in window._guided_run_execution_details_label.text()


# ---------------------------------------------------------------------------
# Background failure/cancellation integration tests (sections 22-25)
# ---------------------------------------------------------------------------


def test_background_builder_failure_never_freezes_gui_and_cleans_up(
    window, monkeypatch, qapp
):
    _set_npm_ready(window)
    validate_btn = window._guided_backend_validate_btn
    entered = threading.Event()
    release = threading.Event()

    def blocking_then_fail_build(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return _fake_build_failed()

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        blocking_then_fail_build,
    )

    window._guided_run_btn.click()
    assert entered.wait(timeout=5)
    # The GUI does not freeze while the builder blocks.
    qapp.processEvents()
    release.set()

    _pump_until(qapp, lambda: window._guided_npm_run_worker_thread is None)

    assert window._guided_npm_launch_runtime is None
    assert "could not be started" in window._guided_run_readiness_label.text()
    assert validate_btn.isEnabled() is True
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_backend_execution_active is False
    assert window._guided_npm_run_worker is None


def test_background_launch_failure_not_synchronous_in_click_handler(
    window, monkeypatch, qapp
):
    _set_npm_ready(window)
    validate_btn = window._guided_backend_validate_btn
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fake_build_ok,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: _fake_launch_failure("raw backend detail"),
    )

    window._guided_run_btn.click()
    # Even a fast fake launch failure cannot have been handled yet: `.click()`
    # only starts the QThread, it does not pump the event loop, and the
    # cross-thread queued signal has not been delivered -- proving launch
    # failure handling no longer happens synchronously inside the click
    # handler.
    assert "preparing" in window._guided_run_readiness_label.text().lower()

    _pump_until(qapp, lambda: window._guided_npm_run_worker_thread is None)

    assert window._guided_npm_launch_runtime is None
    assert "could not be started" in window._guided_run_readiness_label.text()
    assert (
        "raw backend detail"
        not in window._guided_run_execution_details_label.text()
    )
    assert validate_btn.isEnabled() is True
    assert window._guided_run_btn.isEnabled() is False


def test_background_cancellation_shows_neutral_not_started_and_cleans_up(
    window, monkeypatch, qapp
):
    _set_npm_ready(window)
    validate_btn = window._guided_backend_validate_btn
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fake_build_ok,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: _fake_launch_cancelled(),
    )
    reconcile_calls = []
    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        lambda runtime: reconcile_calls.append(runtime),
    )

    window._guided_run_btn.click()
    assert "preparing" in window._guided_run_readiness_label.text().lower()

    _pump_until(qapp, lambda: window._guided_npm_run_worker_thread is None)

    assert "not started" in window._guided_run_readiness_label.text().lower()
    assert "fail" not in window._guided_run_readiness_label.text().lower()
    assert reconcile_calls == []
    assert window._guided_npm_launch_runtime is None
    assert window._guided_backend_execution_active is False
    assert validate_btn.isEnabled() is True
    assert window._guided_npm_run_worker is None


def test_reconciliation_exception_shows_unconfirmed_and_stays_stable(
    window, monkeypatch, qapp
):
    _set_npm_ready(window)
    _install_success_path(monkeypatch)

    def raising_reconcile(_runtime):
        raise RuntimeError("simulated unexpected reconciliation failure")

    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        raising_reconcile,
    )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_npm_run_worker_thread is None)

    assert (
        "could not confirm"
        in window._guided_run_readiness_label.text().lower()
    )
    assert window._guided_backend_execution_result is not None
    assert window._guided_run_btn.isEnabled() is False
    # The retained runtime remains available for diagnostics during and
    # after result handling -- it is never cleared on a reconciliation
    # exception.
    assert window._guided_npm_launch_runtime is not None
    # No crash, no second launch possible while result is pending.
    window._guided_run_btn.click()
    assert window._guided_backend_execution_active is False


def test_reconciliation_publication_failure_shows_unconfirmed(window, monkeypatch, qapp):
    """A publication failure surfaces to `reconcile_guided_npm_worker_runtime`
    as an exception (see guided_npm_worker_reconciliation._publish_once); the
    GUI must map it to "could not confirm", never to success."""
    _set_npm_ready(window)
    _install_success_path(monkeypatch)

    from photometry_pipeline.guided_npm_worker_reconciliation import (
        GuidedNpmReconciliationPublicationError,
    )

    def publication_failing_reconcile(_runtime):
        raise GuidedNpmReconciliationPublicationError("reconciliation_publication_failed")

    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        publication_failing_reconcile,
    )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)

    assert (
        "could not confirm"
        in window._guided_run_readiness_label.text().lower()
    )
    assert "finished successfully" not in window._guided_run_readiness_label.text().lower()


# ---------------------------------------------------------------------------
# No-false-running tests (section 27)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "configure",
    [
        "builder_fails",
        "builder_raises",
        "launch_fails",
        "launch_cancelled",
        "launch_unexpected",
    ],
)
def test_running_text_never_shown_for_non_launch_outcomes(
    window, monkeypatch, qapp, configure
):
    _set_npm_ready(window)
    if configure == "builder_fails":
        monkeypatch.setattr(
            npm_builder_module,
            "build_guided_npm_worker_prelaunch_claim_from_validation",
            lambda **kwargs: _fake_build_failed(),
        )
    elif configure == "builder_raises":
        def raising_build(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            npm_builder_module,
            "build_guided_npm_worker_prelaunch_claim_from_validation",
            raising_build,
        )
    else:
        monkeypatch.setattr(
            npm_builder_module,
            "build_guided_npm_worker_prelaunch_claim_from_validation",
            _fake_build_ok,
        )
        if configure == "launch_fails":
            monkeypatch.setattr(
                npm_launch_module,
                "launch_guided_npm_worker_runtime",
                lambda claim, **kwargs: _fake_launch_failure(),
            )
        elif configure == "launch_cancelled":
            monkeypatch.setattr(
                npm_launch_module,
                "launch_guided_npm_worker_runtime",
                lambda claim, **kwargs: _fake_launch_cancelled(),
            )
        elif configure == "launch_unexpected":
            monkeypatch.setattr(
                npm_launch_module,
                "launch_guided_npm_worker_runtime",
                lambda claim, **kwargs: object(),
            )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_npm_run_worker_thread is None)

    assert (
        "your npm analysis is running"
        not in window._guided_run_readiness_label.text().lower()
    )
    assert window._guided_npm_launch_runtime is None


# ---------------------------------------------------------------------------
# Backend-language leakage (B2-E1 narrow follow-up, sections 17-18)
# ---------------------------------------------------------------------------


_LEAKAGE_TERMS = (
    "worker",
    "receipt",
    "artifact",
    "execution request",
    "backend",
    "json",
)


def test_raw_launch_failure_message_never_leaks_to_gui_text(window, monkeypatch, qapp):
    _set_npm_ready(window)
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fake_build_ok,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: _fake_launch_failure(
            "worker receipt artifact execution request backend "
            "JSON guided_npm_worker_request.json failed"
        ),
    )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)

    label_text = window._guided_run_readiness_label.text().lower()
    details_text = window._guided_run_execution_details_label.text().lower()
    for term in _LEAKAGE_TERMS:
        assert term not in label_text, f"{term!r} leaked into: {label_text!r}"
        assert term not in details_text, f"{term!r} leaked into: {details_text!r}"
    assert "could not be started" in label_text
    assert details_text  # a safe, non-empty controlled detail is shown


def test_raw_builder_issue_message_never_leaks_to_gui_text(window, monkeypatch, qapp):
    _set_npm_ready(window)
    from photometry_pipeline.guided_npm_run_launch_builder import (
        GuidedNpmRunLaunchBuildIssue,
    )

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        lambda **kwargs: GuidedNpmRunLaunchBuildResult(
            status="worker_request_materialization_failed",
            ok=False,
            prelaunch_claim=None,
            application_build_identity=None,
            blocking_issues=(
                GuidedNpmRunLaunchBuildIssue(
                    category="worker_request_materialization_failed",
                    section="worker_request_materialization",
                    message=(
                        "worker receipt artifact execution request backend "
                        "JSON guided_npm_worker_request.json failed"
                    ),
                ),
            ),
            current_gui_revision=4,
        ),
    )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)

    label_text = window._guided_run_readiness_label.text().lower()
    details_text = window._guided_run_execution_details_label.text().lower()
    for term in _LEAKAGE_TERMS:
        assert term not in label_text, f"{term!r} leaked into: {label_text!r}"
        assert term not in details_text, f"{term!r} leaked into: {details_text!r}"
    assert "could not be started" in label_text
    assert details_text


# ---------------------------------------------------------------------------
# Scientist-facing text audit (section 44)
# ---------------------------------------------------------------------------


_FORBIDDEN_TERMS = (
    "artifact",
    "receipt",
    "authority",
    "reconciliation",
    "worker",
    "subprocess",
    "exit code",
    "backend",
    "manifest",
    "json",
    "runtime",
    "process identity",
    "startup payload",
    "execution request",
)


@pytest.mark.parametrize(
    "final_outcome",
    [
        "verified_completed",
        "authority_refused",
        "indeterminate",
        "post_launch_evidence_failed",
    ],
)
def test_npm_run_result_text_is_scientist_facing(window, final_outcome):
    result = SimpleNamespace(
        final_outcome=final_outcome, run_directory_path=r"C:\fake\npm-run"
    )
    window._finish_guided_npm_run_with_result(result)
    label_text = window._guided_run_readiness_label.text().lower()
    details_text = window._guided_run_execution_details_label.text().lower()
    for term in _FORBIDDEN_TERMS:
        assert term not in label_text, f"{term!r} leaked into: {label_text!r}"
        assert term not in details_text, f"{term!r} leaked into: {details_text!r}"


def test_npm_readiness_text_is_scientist_facing():
    from photometry_pipeline.guided_npm_run_readiness import _SUMMARIES

    for summary in _SUMMARIES.values():
        lowered = summary.lower()
        for term in _FORBIDDEN_TERMS:
            assert term not in lowered, f"{term!r} leaked into: {summary!r}"


def test_preparing_text_is_scientist_facing():
    prohibited = _FORBIDDEN_TERMS + ("thread", "signal")
    lowered = "Preparing your NPM analysis…".lower()
    for term in prohibited:
        assert term not in lowered


# ---------------------------------------------------------------------------
# Close/navigation (section 45 / narrow-follow-up section 29)
# ---------------------------------------------------------------------------


def test_close_event_refused_while_npm_run_active(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *args, **kwargs: shown.append(args)),
    )
    window._guided_backend_execution_active = True
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert shown


def test_close_event_allowed_when_npm_run_not_active(window):
    window._guided_backend_execution_active = False
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is True


def test_close_event_refused_while_builder_still_preparing(window, monkeypatch, qapp):
    """`_guided_backend_execution_active` becomes True as soon as Run is
    pressed (during preparation), before any process exists -- the existing
    close guard must therefore already block closing while the builder is
    still running on the worker thread."""
    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *args, **kwargs: shown.append(args)),
    )
    _set_npm_ready(window)
    entered = threading.Event()
    release = threading.Event()

    def blocking_build(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return _fake_build_ok(**kwargs)

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        blocking_build,
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: _fake_launched_runtime(),
    )
    monkeypatch.setattr(
        npm_reconciliation_module,
        "reconcile_guided_npm_worker_runtime",
        lambda runtime: SimpleNamespace(
            final_outcome="verified_completed", run_directory_path="x"
        ),
    )

    window._guided_run_btn.click()
    assert entered.wait(timeout=5)

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert shown

    release.set()
    _pump_until(qapp, lambda: window._guided_npm_run_worker_thread is None)


# ---------------------------------------------------------------------------
# Real-path integration test (section 41 / narrow-follow-up section 26)
# ---------------------------------------------------------------------------


def test_real_gui_path_reaches_verified_completed_and_shows_success(
    window, tmp_path, monkeypatch, qapp
):
    """Exercise the actual GUI/controller state machine end to end.

    The prelaunch claim and the "launched runtime" are built once, up
    front, via the exact same real backend construction chain the
    committed backend test suite already uses (`_transaction` /
    `run_guided_npm_worker_to_terminal_receipt`), producing REAL, durable
    terminal and consumed-authority receipts on disk for a real (in this
    process) executed child. Only the OS process-creation step is
    substituted with a deterministic fake handle carrying this test
    process's own real `os.getpid()` -- a real, separately-spawned OS
    process can never reach `verified_completed` in this environment
    because its freshly-resolved build identity can never match the
    synthetic fixture identity used to build the claim (a pre-existing,
    documented environmental limitation, not a shortcut of convenience).

    `reconcile_guided_npm_worker_runtime` itself is NOT mocked: it reads
    the real receipts from disk and must independently conclude
    `verified_completed` for this test to pass. The full sequence now
    exercises: Run click -> preparing state -> worker-thread build/launch
    seam -> launched signal -> running state -> real (unmocked)
    reconciliation -> verified completion -> exact output path.
    """
    import os

    import photometry_pipeline.guided_npm_worker_entry as entry_module
    from tests.test_guided_npm_worker_acknowledgement import _transaction

    claim, worker, runtime_obj, invocation, context, start, evidence, _ = _transaction(
        tmp_path, pid=os.getpid()
    )
    code, terminal = entry_module.run_guided_npm_worker_to_terminal_receipt(
        worker, launch_context=context
    )
    assert code == entry_module.GUIDED_NPM_WORKER_ENTRY_SUCCESS

    fake_handle = SimpleNamespace(pid=os.getpid(), wait=lambda timeout=None: 0)
    real_runtime = GuidedNpmLaunchedWorkerRuntime(
        claim, invocation, context, start, fake_handle
    )

    _set_npm_ready(window)
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        lambda **kwargs: GuidedNpmRunLaunchBuildResult(
            status="built",
            ok=True,
            prelaunch_claim=claim,
            application_build_identity=claim.application_build_identity,
            blocking_issues=(),
            current_gui_revision=4,
        ),
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: real_runtime,
    )
    # reconcile_guided_npm_worker_runtime is intentionally left un-mocked.

    window._guided_run_btn.click()
    assert window._guided_backend_execution_active is True
    assert "preparing" in window._guided_run_readiness_label.text().lower()

    # Reconciliation here reads already-written real receipts and completes
    # near-instantly, so the "launched" and "succeeded" signals may both be
    # delivered within the same event-loop pump -- the transient "running"
    # text is proven separately (with a deliberately blocked reconciler) by
    # `test_preparing_to_running_transition`. This test's job is the full
    # real, unmocked path to `verified_completed`.
    _pump_until(qapp, lambda: window._guided_npm_launch_runtime is not None)

    _pump_until(qapp, lambda: window._guided_backend_execution_active is False)

    result = window._guided_backend_execution_result
    assert result.final_outcome == "verified_completed"
    assert (
        window._guided_run_readiness_label.text()
        == "Your NPM analysis finished successfully."
    )
    assert result.run_directory_path in (
        window._guided_run_execution_details_label.text()
    )
    assert window._guided_run_btn.isEnabled() is False
