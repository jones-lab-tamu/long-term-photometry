"""
Unit tests for Step 8 Artifact Semantics:
- StatusFollower (last-known-good parse preservation)
- CANCEL.REQUESTED file path selection
- run_report.json parsing & quick-link resolution
"""

import os
import json
import tempfile
import pytest

from gui.process_runner import PipelineRunner, RunnerState
from gui.status_follower import StatusFollower
from gui.run_report_parser import (
    parse_run_report,
    get_preview_mode,
    get_summary_fields,
    resolve_quick_links
)


# =====================================================================
# StatusFollower Tests
# =====================================================================

def test_status_follower_last_known_good():
    """
    Simulate partial writes and missing keys in status.json to ensure
    StatusFollower retains its last good fields and does not crash.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        status_path = os.path.join(tmpdir, "status.json")
        follower = StatusFollower(status_path, poll_ms=10)
        
        # 1. Start with valid data
        with open(status_path, "w") as f:
            json.dump({"phase": "init", "status": "running"}, f)
        follower._poll()
        assert follower._last_good_status["phase"] == "init"
        
        # 2. Corrupt the file (partial write simulation)
        with open(status_path, "w") as f:
            f.write('{"phase": "corrupt", ') # Invalid JSON
        follower._poll()
        # Should retain previous valid state
        assert follower._last_good_status["phase"] == "init"
        
        # 3. Valid JSON but missing keys
        with open(status_path, "w") as f:
            json.dump({"status": "success"}, f) # Missing 'phase'
        follower._poll()
        # Should merge: retain 'phase': 'init', but update 'status'
        assert follower._last_good_status["phase"] == "init"
        assert follower._last_good_status["status"] == "success"
        
        # Note: terminal_reached is tested separately or via state inspection
        # For this test, verifying merger is sufficient.


def test_status_follower_terminal_stops_polling():
    """Verify that reaching a terminal state stops the internal timer."""
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        status_path = os.path.join(tmpdir, "status.json")
        follower = StatusFollower(status_path, poll_ms=10)
        
        with patch.object(follower, 'stop') as mock_stop:
            # Write terminal status
            with open(status_path, "w") as f:
                json.dump({"status": "success"}, f)
                
            follower._poll()
            mock_stop.assert_called_once()


def test_status_follower_warning_emitted_once():
    """Verify StatusFollower emits status_warning(str) once per unique status and NOT parse_error."""
    from unittest.mock import MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        status_path = os.path.join(tmpdir, "status.json")
        follower = StatusFollower(status_path, poll_ms=10)
        
        warn_handler = MagicMock()
        err_handler = MagicMock()
        follower.status_warning.connect(warn_handler)
        follower.parse_error.connect(err_handler)
        
        # 1. Unknown status
        with open(status_path, "w") as f:
            json.dump({"status": "aliens_invading"}, f)
        follower._poll()
        warn_handler.assert_called_with("unknown status: aliens_invading")
        assert warn_handler.call_count == 1
        assert err_handler.call_count == 0  # Should NOT be a parse error
        
        # 2. Same status again - should NOT emit again
        follower._poll()
        assert warn_handler.call_count == 1
        
        # 3. Different unknown status - should emit
        with open(status_path, "w") as f:
            json.dump({"status": "coffee_empty"}, f)
        follower._poll()
        assert warn_handler.call_count == 2
        warn_handler.assert_called_with("unknown status: coffee_empty")


def test_status_follower_parse_error_signal():
    """Verify StatusFollower emits parse_error(str) on failures."""
    from unittest.mock import MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        status_path = os.path.join(tmpdir, "status.json")
        follower = StatusFollower(status_path, poll_ms=10)
        
        mock_handler = MagicMock()
        follower.parse_error.connect(mock_handler)
        
        # Case 1: File missing
        follower._poll()
        mock_handler.assert_called_with("status.json missing")
        
        # Case 2: Invalid JSON
        with open(status_path, "w") as f:
            f.write("{ invalid }")
        follower._poll()
        assert mock_handler.call_count == 2
        assert "read/parse error" in mock_handler.call_args[0][0]


# =====================================================================
# Cancel Semantics Tests
# =====================================================================

def test_runner_writes_exact_cancel_file():
    """Verify that canceling produces exactly CANCEL.REQUESTED in run_dir."""
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = PipelineRunner()
        runner.set_run_dir(tmpdir)
        
        # We just need it to think it is running to test the cancel artifact creation
        with patch.object(runner, 'is_running', return_value=True):
            from PySide6.QtCore import QProcess
            runner._process = QProcess()
            with patch.object(runner._process, 'terminate'), \
                 patch.object(runner._process, 'kill'), \
                 patch.object(runner._process, 'waitForFinished', return_value=True):
                runner.cancel()
        
        cancel_file = os.path.join(tmpdir, "CANCEL.REQUESTED")
        assert os.path.isfile(cancel_file)
        assert open(cancel_file, "r").read().strip() == "cancelled by gui"


def test_runner_state_truthful_even_with_report():
    """Verify that runner keeps truthful state (FAIL_CLOSED/FAILED) even if run_report.json exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = PipelineRunner()
        runner.set_run_dir(tmpdir)
        
        # Create a report file
        open(os.path.join(tmpdir, "run_report.json"), "w").write("{}")
        
        # Mocking exit code 1 without status.json -> FAIL_CLOSED logic
        from PySide6.QtCore import QProcess
        runner._process = QProcess()
        
        # Exercise the completion hook
        runner._on_finished(1, QProcess.NormalExit)
        
        assert runner.state == RunnerState.FAIL_CLOSED # Truthful: no status.json
        assert runner.has_run_report is True # Flag is set for UI benefit


# =====================================================================
# Run Report Tests
# =====================================================================

def test_run_report_invalid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "run_report.json")
        with open(report_path, "w") as f:
            f.write("{ bad json ]")
            
        data, err = parse_run_report(report_path)
        assert err is not None
        assert "Parse error" in err
        assert not data

def test_get_preview_mode():
    assert get_preview_mode({"run_context": {"run_type": "preview"}}) is True
    assert get_preview_mode({"run_context": {"run_type": "full"}}) is False
    assert get_preview_mode({}) is False
    assert get_preview_mode({"run_context": "Not a dict"}) is False

def test_get_summary_fields():
    report = {
        "run_context": {
            "run_type": "full",
            "event_signal": "dff"
        },
        "configuration": {
            "lowpass_hz": 2.5,
            "baseline_method": "median"
        }
    }
    fields = dict(get_summary_fields(report))
    assert fields["run_type"] == "full"
    assert fields["event_signal"] == "dff"
    assert fields["lowpass_hz"] == "2.5"
    assert fields["baseline_method"] == "median"

def test_resolve_quick_links_isolation():
    """Verify isolation: rejection messages must include the resolved absolute path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        real_tmpdir = os.path.realpath(tmpdir)
        os.makedirs(os.path.join(real_tmpdir, "traces"))
        
        report = {
            "artifacts": {
                "sneaky": "../outside.txt",
                "normal": "traces/file.csv",
                "traversal2": "traces/../../secret.txt"
            }
        }
        open(os.path.join(real_tmpdir, "traces", "file.csv"), "w").close()
        
        links = resolve_quick_links(real_tmpdir, report)
        
        def find_status(label):
            for l, p, s in links:
                if label in l: return s
            return None
            
        # Traversal Rejection
        assert find_status("Artifact: sneaky") == "missing/invalid (directory traversal rejected)"
        assert find_status("Artifact: traversal2") == "missing/invalid (directory traversal rejected)"
        assert find_status("Artifact: normal") == "ok"

        # Absolute path outside
        bad_abs = "/etc/passwd" if os.name != 'nt' else "C:/Windows/System32/cmd.exe"
        resolved_bad = os.path.realpath(bad_abs)
        abs_report = {"artifacts": {"evil": bad_abs}}
        links_abs = resolve_quick_links(real_tmpdir, abs_report)
        evil_status = [s for l, p, s in links_abs if "evil" in l][0]
        
        # Must include resolved path for debugging
        assert "outside run_dir" in evil_status
        assert resolved_bad in evil_status

        # Absolute path INSIDE
        inside_abs = os.path.join(real_tmpdir, "traces", "file.csv")
        inside_report = {"artifacts": {"legit": inside_abs}}
        links_inside = resolve_quick_links(real_tmpdir, inside_report)
        legit_status = [s for l, p, s in links_inside if "legit" in l][0]
        assert legit_status == "ok"
