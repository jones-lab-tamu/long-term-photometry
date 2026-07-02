from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

import photometry_pipeline.guided_backend_execution as backend_execution
import photometry_pipeline.guided_startup_orchestration as orchestration
from gui.main_window import MainWindow
from tests.test_gui_guided_backend_validation_context import _accepted_outcome
from tests.test_guided_startup_transaction import startup_request


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _visible_text(window):
    return " ".join(
        (
            window._guided_run_btn.text(),
            window._guided_run_btn.toolTip(),
            window._guided_run_readiness_label.text(),
        )
    )


def test_guided_run_affordance_is_visible_disabled_and_guided_scoped(window):
    button = window._guided_run_btn
    assert button.text() == "Run Guided Analysis"
    assert button.isHidden() is False
    assert button.isEnabled() is False
    assert window._guided_workflow_tab.isAncestorOf(button)
    assert not window._full_control_tab.isAncestorOf(button)


def test_no_validation_message_asks_for_validation(window):
    window._guided_backend_validation_outcome = None
    window._guided_backend_validation_outcome_revision = None
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "no_validation"
    assert "Validate the Guided setup" in window._guided_run_readiness_label.text()
    assert window._guided_run_btn.isEnabled() is False


def test_stale_validation_message_asks_for_revalidation(window):
    window._guided_backend_validation_revision = 4
    window._guided_backend_validation_outcome = replace(
        _accepted_outcome(), stale=True
    )
    window._guided_backend_validation_outcome_revision = 4
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "validation_stale"
    assert "Validate again" in window._guided_run_readiness_label.text()
    assert window._guided_run_btn.isEnabled() is False


def test_ready_hidden_remains_disabled(window, startup_request):
    window._guided_backend_validation_revision = (
        startup_request.current_guided_revision
    )
    window._guided_backend_validation_outcome = _accepted_outcome()
    window._guided_backend_validation_outcome_revision = (
        startup_request.current_guided_revision
    )
    window._guided_run_authorization_result = (
        startup_request.authorization_result
    )
    window._guided_execution_payload_result = startup_request.payload_result
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "ready_hidden"
    assert window._guided_run_readiness.ready is True
    assert window._guided_run_btn.isEnabled() is False
    assert "running from Guided Mode is not enabled" in (
        window._guided_run_readiness_label.text()
    )


def test_rendering_and_disabled_click_call_no_execution(
    window, monkeypatch
):
    def fail(*_args, **_kwargs):
        raise AssertionError("execution must not be called")

    monkeypatch.setattr(backend_execution, "execute_guided_backend_run", fail)
    monkeypatch.setattr(orchestration, "run_guided_startup_to_wrapper", fail)
    window._refresh_guided_run_readiness_display()
    window._guided_run_btn.click()
    assert window._guided_run_btn.isEnabled() is False


def test_rendering_writes_no_files_or_directories(window, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("write must not occur")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False


def test_visible_text_excludes_internal_terms(window):
    window._refresh_guided_run_readiness_display()
    text = _visible_text(window).lower()
    prohibited = (
        "manifest",
        "preallocated",
        "command_invoked",
        "wrapper claim",
        "startup transaction",
        "hash",
        "--guided",
        "config_effective.yaml",
        "runner_request",
        "startup_transaction_unavailable",
        "guided_candidate_manifest",
        "guided_startup",
        "wrapper_claim",
    )
    assert not any(term in text for term in prohibited)


def test_full_control_run_control_is_unchanged_by_readiness_refresh(window):
    full_control_run = window._run_btn
    before = (
        full_control_run.text(),
        full_control_run.isEnabled(),
        full_control_run.toolTip(),
    )
    window._refresh_guided_run_readiness_display()
    after = (
        full_control_run.text(),
        full_control_run.isEnabled(),
        full_control_run.toolTip(),
    )
    assert after == before
