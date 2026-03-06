"""
Tests for Final Status Refresh (Fix stale top-status field).
Verifies that _refresh_status_from_disk_final correctly updates MainWindow fields.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import MagicMock
from gui.main_window import MainWindow

@pytest.fixture
def mock_window():
    # We mock out things that would require a real QApplication or heavy initialization
    with tempfile.TemporaryDirectory() as tmp_dir:
        window = MagicMock(spec=MainWindow)
        # Set up real attributes we want to test
        window._current_run_dir = tmp_dir
        window._last_status_phase = "running"
        window._last_status_state = "running"
        window._last_status_duration = "0.0s"
        window._last_status_errors = []
        window._last_status_msg = "some warning"
        
        # Bind the real method logic to the mock instance for testing
        # (This is a trick to test strictly the logic of the method on a mock data-container)
        window._refresh_status_from_disk_final = MainWindow._refresh_status_from_disk_final.__get__(window, MainWindow)
        
        yield window

def test_refresh_status_from_disk_final_success(mock_window):
    # 1. Create a terminal status.json
    status_data = {
        "phase": "final",
        "status": "success",
        "duration_sec": 85.7,
        "errors": []
    }
    status_path = os.path.join(mock_window._current_run_dir, "status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status_data, f)
        
    # 2. Call the helper
    mock_window._refresh_status_from_disk_final()
    
    # 3. Assertions
    assert mock_window._last_status_phase == "final"
    assert mock_window._last_status_state == "success"
    assert mock_window._last_status_duration == "85.7s"
    assert mock_window._last_status_msg == ""

def test_refresh_status_from_disk_preserves_on_missing(mock_window):
    # Ensure no status.json exists
    status_path = os.path.join(mock_window._current_run_dir, "status.json")
    if os.path.exists(status_path):
        os.remove(status_path)
        
    # Call the helper
    mock_window._refresh_status_from_disk_final()
    
    # Assertions: values should remain unchanged (stale but not erased)
    assert mock_window._last_status_phase == "running"
    assert mock_window._last_status_state == "running"
    assert mock_window._last_status_duration == "0.0s"

def test_refresh_status_from_disk_handles_invalid_json(mock_window):
    # Create invalid JSON
    status_path = os.path.join(mock_window._current_run_dir, "status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        f.write("{ invalid json")
        
    # Call the helper
    mock_window._refresh_status_from_disk_final()
    
    # Assertions: values should remain unchanged, no crash
    assert mock_window._last_status_phase == "running"
    assert mock_window._last_status_state == "running"
    assert mock_window._last_status_duration == "0.0s"
