import json
import os
import sys

import pytest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gui.main_window import MainWindow
from gui.process_runner import RunnerState


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def _write_validate_artifacts(run_dir: str, *, status: str = "success") -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "gui_run_spec.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": "C:/input",
                "format": "rwd",
                "sessions_per_hour": 2,
                "session_duration_s": 1800.0,
                "smooth_window_s": 1.0,
                "config_source_path": "C:/cfg/base.yaml",
                "config_overrides": {},
                "sig_iso_render_mode": None,
                "dff_render_mode": None,
                "stacked_render_mode": None,
                "mode": "both",
                "run_profile": "full",
                "traces_only": False,
                "preview_first_n": None,
                "representative_session_index": None,
                "include_roi_ids": None,
                "exclude_roi_ids": None,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    with open(os.path.join(run_dir, "config_effective.yaml"), "w", encoding="utf-8") as f:
        f.write("lowpass_hz: 0.5\n")
    with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "phase": "final", "status": status}, f)


def test_validate_success_auto_cleans_run_dir(window, tmp_path):
    run_dir = tmp_path / "validate_success_run"
    _write_validate_artifacts(str(run_dir), status="success")

    window._current_run_dir = str(run_dir)
    window._is_validate_only = True
    window._validation_passed = False
    window._did_finalize_run_ui = False
    window._runner._state = RunnerState.SUCCESS
    window._runner.final_status_code = "success"

    window._on_run_finished_failsafe(0)

    assert window._validation_passed is True
    assert isinstance(window._validated_run_signature, str)
    assert window._validated_run_signature
    assert not run_dir.exists()
    assert window._current_run_dir == ""
    assert "auto-cleaned" in window._log_view.toPlainText()


def test_validate_failure_retains_debug_run_dir(window, tmp_path):
    run_dir = tmp_path / "validate_failure_run"
    _write_validate_artifacts(str(run_dir), status="error")

    window._current_run_dir = str(run_dir)
    window._is_validate_only = True
    window._validation_passed = False
    window._did_finalize_run_ui = False
    window._runner._state = RunnerState.FAILED
    window._runner.final_status_code = "error"
    window._runner.final_errors = ["bad config"]

    window._on_run_finished_failsafe(1)

    assert window._validation_passed is False
    assert run_dir.exists()
    assert window._pending_validate_cleanup_dir == ""
    assert "Run failed during execution." in window._log_view.toPlainText()


def test_full_run_artifacts_remain_durable(window, tmp_path):
    run_dir = tmp_path / "full_run_success"
    _write_validate_artifacts(str(run_dir), status="success")

    window._current_run_dir = str(run_dir)
    window._is_validate_only = False
    window._validation_passed = False
    window._did_finalize_run_ui = False
    window._runner._state = RunnerState.SUCCESS
    window._runner.final_status_code = "success"

    window._on_run_finished_failsafe(0)

    assert run_dir.exists()
    assert window._pending_validate_cleanup_dir == ""
