from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

import photometry_pipeline.guided_backend_execution as backend_execution
import photometry_pipeline.guided_startup_allocation as allocation
import photometry_pipeline.guided_startup_materialization as materialization
import photometry_pipeline.guided_startup_orchestration as orchestration
import photometry_pipeline.pipeline as pipeline
import tools.run_full_pipeline_deliverables as wrapper
from gui.main_window import MainWindow
from gui.run_report_parser import classify_completed_run_candidate
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


def _set_ready(window, request):
    from photometry_pipeline.guided_plan_identity import (
        compute_guided_new_analysis_draft_plan_identity,
    )

    window._guided_backend_validation_revision = request.current_guided_revision
    window._guided_backend_validation_outcome = _accepted_outcome()
    window._guided_backend_validation_outcome_revision = (
        request.current_guided_revision
    )
    window._guided_startup_authority = request.startup_authority
    window._guided_execution_payload_result = request.payload_result
    # See tests/test_guided_gui_run_execution_wiring.py::_set_ready -- the
    # authoritative identity check requires the freshly-rebuilt draft's
    # canonical identity to match what was "validated".
    window._guided_validated_plan_identity = (
        compute_guided_new_analysis_draft_plan_identity(
            window._build_guided_new_analysis_draft_plan()
        )
    )
    window._refresh_guided_run_readiness_display()


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


def test_missing_authorization_keeps_button_disabled(window, startup_request):
    _set_ready(window, startup_request)
    window._guided_startup_authority = None
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "authorization_missing"
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_run_readiness_label.text() == (
        "Guided validation succeeded, but Guided Run execution is unavailable "
        "in this build."
    )
    assert "Validate the setup first" not in (
        window._guided_run_readiness_label.text()
    )


def test_missing_payload_keeps_button_disabled(window, startup_request):
    _set_ready(window, startup_request)
    window._guided_execution_payload_result = None
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "payload_missing"
    assert window._guided_run_btn.isEnabled() is False


def test_payload_not_ready_keeps_button_disabled(window, startup_request):
    _set_ready(window, startup_request)
    window._guided_execution_payload_result = replace(
        startup_request.payload_result, ok=False
    )
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "payload_not_ready"
    assert window._guided_run_btn.isEnabled() is False


def test_ready_hidden_enables_guarded_button(window, startup_request):
    _set_ready(window, startup_request)
    assert window._guided_run_readiness.status == "ready_hidden"
    assert window._guided_run_readiness.ready is True
    assert window._guided_run_btn.isEnabled() is True
    assert window._guided_run_btn.toolTip() == "Guided Run is ready to start."
    assert window._guided_run_readiness_label.text() == (
        "Guided Run is ready to start."
    )


def test_enabled_click_without_retained_request_calls_no_execution_path(
    window, startup_request, monkeypatch
):
    def fail(*_args, **_kwargs):
        raise AssertionError("execution must not be called")

    monkeypatch.setattr(backend_execution, "execute_guided_backend_run", fail)
    monkeypatch.setattr(orchestration, "run_guided_startup_to_wrapper", fail)
    monkeypatch.setattr(allocation, "allocate_guided_startup_directory", fail)
    monkeypatch.setattr(
        materialization, "materialize_guided_startup_artifacts", fail
    )
    monkeypatch.setattr(wrapper, "main", fail)
    monkeypatch.setattr(wrapper.subprocess, "run", fail)
    monkeypatch.setattr(pipeline, "Pipeline", fail)
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_run_readiness_label.text() == (
        "Guided Run could not start because the validated setup is no longer "
        "current."
    )


def test_enabled_click_writes_nothing_and_creates_no_completed_run(
    window, startup_request, tmp_path, monkeypatch
):
    _set_ready(window, startup_request)
    before = tuple(tmp_path.iterdir())

    def fail(*_args, **_kwargs):
        raise AssertionError("write must not occur")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    window._guided_run_btn.click()
    assert tuple(tmp_path.iterdir()) == before
    assert classify_completed_run_candidate(str(tmp_path))[0] is False


def test_visible_text_excludes_internal_terms(window, startup_request):
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
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
        "backend adapter",
        "orchestration",
        "subprocess",
    )
    assert not any(term in text for term in prohibited)


def test_full_control_run_control_is_unchanged_by_refresh_and_click(
    window, startup_request
):
    full_control_run = window._run_btn
    before = (
        full_control_run.text(),
        full_control_run.isEnabled(),
        full_control_run.toolTip(),
    )
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    after = (
        full_control_run.text(),
        full_control_run.isEnabled(),
        full_control_run.toolTip(),
    )
    assert after == before
