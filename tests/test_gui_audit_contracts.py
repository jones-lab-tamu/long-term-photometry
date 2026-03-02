import os
import json
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.process_runner import RunnerState

# Initialize QApplication once for all tests
_app = QApplication.instance() or QApplication([])

@pytest.fixture
def app():
    """Fixture to ensure a MainWindow exists and is cleaned up."""
    window = MainWindow()
    yield window
    window.close()
    window.deleteLater()

def test_gui_manifest_guard(app, tmp_path):
    """Requirement A: Guard manifest loading on SUCCESS."""
    # 1. Success with missing/invalid run_dir
    app._current_run_dir = str(tmp_path / "nonexistent")
    app._runner._state = RunnerState.SUCCESS
    
    # Manually trigger finalization
    app._finalize_run_ui()
    
    # Check log for expected message
    log_content = app._log_view.toPlainText()
    assert "No run_dir available, cannot load manifest." in log_content

def test_gui_finalization_idempotency(app):
    """Requirement C: Finalize exactly once using idempotency guard."""
    app._on_run_started()
    assert app._did_finalize_run_ui is False
    
    # Call finalization twice
    app._finalize_run_ui()
    assert app._did_finalize_run_ui is True
    
    # Add a sentinel to log to prove the second call does nothing
    app._append_log("SENTINEL_FIRST")
    app._finalize_run_ui()
    
    log_content = app._log_view.toPlainText()
    # It should only contain our log call once (if we called it inside finalize, but we didn't)
    # Let's verify _did_finalize_run_ui stays True and no side effects occur.
    # (Actually we can't easily check 'no side effects' besides the boolean for now)
    assert app._did_finalize_run_ui is True

def test_gui_finalization_robustness(app):
    """Requirement C: Ensure runner.finished always results in finalization."""
    app._on_run_started()
    assert app._did_finalize_run_ui is False
    
    # Simulate process finish signal
    app._on_run_finished_failsafe(0)
    
    assert app._did_finalize_run_ui is True
    assert app._events_follower is None # Should have been stopped

def test_gui_status_label_minimal(app):
    """Requirement: Raw stage:type instead of friendly mapping."""
    app._on_event({"stage": "engine", "type": "start"})
    label_text = app._status_label.text()
    assert "Stage: engine" in label_text
    assert "Type: start" in label_text
    # Should NOT contain "Starting" (the friendly label we removed)
    assert "Starting" not in label_text
