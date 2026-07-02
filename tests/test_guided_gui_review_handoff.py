from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow
from tests.test_guided_gui_run_execution_wiring import _result


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _completed_candidate(tmp_path: Path) -> Path:
    run_dir = tmp_path / "completed"
    (run_dir / "Region0" / "summary").mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {"schema_version": 1, "phase": "final", "status": "success"}
        ),
        encoding="utf-8",
    )
    return run_dir


def _set_result(window, result) -> None:
    window._guided_backend_execution_result = result
    window._refresh_guided_review_handoff_display()


def test_handoff_control_is_hidden_before_run(window):
    button = window._guided_load_completed_run_for_review_btn
    assert button.text() == "Load completed run for review"
    assert button.isVisible() is False
    assert button.isEnabled() is False
    assert window._guided_workflow_tab.isAncestorOf(button)
    assert not window._full_control_tab.isAncestorOf(button)


@pytest.mark.parametrize(
    "status",
    (
        "refused_before_startup",
        "startup_allocation_failed",
        "wrapper_running",
        "wrapper_failed",
    ),
)
def test_noncompleted_results_do_not_enable_handoff(window, status):
    result = _result(status, "Guided Run could not continue.")
    _set_result(window, result)
    assert window._guided_load_completed_run_for_review_btn.isVisible() is False
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is False


def test_completed_candidate_enables_handoff_without_auto_loading(
    window, tmp_path, monkeypatch
):
    run_dir = _completed_candidate(tmp_path)
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    monkeypatch.setattr(
        window,
        "_open_completed_results_dir",
        lambda *_args, **_kwargs: pytest.fail("Review auto-loaded"),
    )
    _set_result(window, completed)
    button = window._guided_load_completed_run_for_review_btn
    assert button.isHidden() is False
    assert button.isEnabled() is True
    assert completed.completed_run_claim is False


def test_accepted_candidate_is_classified_then_opened_once(
    window, tmp_path, monkeypatch
):
    run_dir = _completed_candidate(tmp_path)
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    classifications = []
    original_classifier = window._is_openable_completed_results_dir

    def classify(path):
        classifications.append(path)
        return original_classifier(path)

    opened = []
    monkeypatch.setattr(
        window, "_is_openable_completed_results_dir", classify
    )
    monkeypatch.setattr(
        window,
        "_open_completed_results_dir",
        lambda path: opened.append(path) or True,
    )
    _set_result(window, completed)
    window._guided_load_completed_run_for_review_btn.click()
    assert classifications == [str(run_dir)]
    assert opened == [str(run_dir)]
    assert window._guided_workflow_stepper.currentRow() == (
        list(GUIDED_WORKFLOW_STEPS).index("Review")
    )
    assert window._guided_run_readiness_label.text() == (
        "Completed run loaded for review."
    )


def test_incomplete_candidate_is_rejected_without_review_transition(
    window, tmp_path, monkeypatch
):
    run_dir = tmp_path / "startup-stop"
    (run_dir / "Region0" / "summary").mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {"schema_version": 1, "phase": "running", "status": "running"}
        ),
        encoding="utf-8",
    )
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    before = window._guided_workflow_stepper.currentRow()
    monkeypatch.setattr(
        window,
        "_open_completed_results_dir",
        lambda *_args, **_kwargs: pytest.fail("Rejected run was opened"),
    )
    _set_result(window, completed)
    window._guided_load_completed_run_for_review_btn.click()
    assert window._guided_workflow_stepper.currentRow() == before
    assert window._guided_run_readiness_label.text() == (
        "The completed run could not be loaded for review. "
        "The output folder may be incomplete."
    )


def test_backend_status_without_candidate_path_refuses(window):
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        completed_run_candidate_path=None,
    )
    _set_result(window, completed)
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is False
    window._on_guided_load_completed_run_for_review_clicked()
    assert window._guided_run_readiness_label.text() == (
        "Guided Run did not provide a completed-run candidate."
    )


def test_invalidation_clears_and_hides_handoff(window, tmp_path):
    run_dir = _completed_candidate(tmp_path)
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    _set_result(window, completed)
    assert window._guided_load_completed_run_for_review_btn.isEnabled()
    window._invalidate_guided_backend_validation("input changed")
    assert window._guided_backend_execution_result is None
    assert window._guided_load_completed_run_for_review_btn.isVisible() is False
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is False


def test_handoff_text_is_safe_and_full_control_unchanged(
    window, tmp_path, monkeypatch
):
    run_dir = _completed_candidate(tmp_path)
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    full_control_before = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    monkeypatch.setattr(
        window, "_open_completed_results_dir", lambda _path: True
    )
    _set_result(window, completed)
    window._guided_load_completed_run_for_review_btn.click()
    full_control_after = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    assert full_control_after == full_control_before
    visible = " ".join(
        (
            window._guided_load_completed_run_for_review_btn.text(),
            window._guided_load_completed_run_for_review_btn.toolTip(),
            window._guided_run_readiness_label.text(),
        )
    ).lower()
    assert str(run_dir).lower() not in visible
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
        "raw command",
    )
    assert not any(term in visible for term in prohibited)
