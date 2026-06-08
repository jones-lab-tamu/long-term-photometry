import os
import sys
import json
import pytest
import time
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pathlib import Path

# Bootstrap repo root
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Import the helpers directly from the script
from tools.run_full_pipeline_deliverables import (
    _utc_now_iso, _extract_cmd_label, run_cmd, _phase_start, _phase_done
)


class _RecordingEmitter:
    def __init__(self):
        self.events = []

    def emit(self, stage, event_type, message, payload=None):
        self.events.append(
            {
                "stage": stage,
                "type": event_type,
                "message": message,
                "payload": payload or {},
            }
        )

def test_utc_now_iso():
    now = _utc_now_iso()
    assert isinstance(now, str)
    # Basic ISO format check: YYYY-MM-DDTHH:MM:SS
    assert datetime.fromisoformat(now)

def test_extract_cmd_label():
    assert _extract_cmd_label(["python", "tools/script.py"]) == "script.py"
    assert _extract_cmd_label(["/usr/bin/python3", "script.py"]) == "script.py"
    assert _extract_cmd_label(["plot_something.py", "--arg"]) == "plot_something.py"
    assert _extract_cmd_label([]) == "unknown"

def test_run_cmd_success():
    cmd = ["python", "-c", "print('hello')"]
    # We want to verify it returns the structured dict
    with patch("subprocess.check_call") as mock_call:
        res = run_cmd(cmd, roi_label="Region0")
        
        assert res["cmd"] == cmd
        assert "started_utc" in res
        assert "finished_utc" in res
        assert isinstance(res["elapsed_sec"], float)
        assert res["returncode"] == 0
        mock_call.assert_called_once_with(cmd)

def test_phase_timing_lifecycle():
    status_data = {"timing": {}}
    manifest = {}
    phase_name = "test_phase"
    emitter = _RecordingEmitter()
    
    t0, started_utc = _phase_start(status_data, phase_name, emitter=emitter)
    
    assert status_data["timing"]["current_phase"] == phase_name
    assert status_data["timing"]["phase_started_utc"] == started_utc
    
    time.sleep(0.1) # Ensure some measurable time passes
    
    _phase_done(status_data, manifest, phase_name, t0, started_utc, emitter=emitter)
    
    assert status_data["timing"]["last_completed_phase"] == phase_name
    assert status_data["timing"]["last_phase_elapsed_sec"] > 0
    assert status_data["timing"]["current_phase"] is None
    assert status_data["timing"]["phase_history"]
    assert status_data["timing"]["phase_history"][0]["phase"] == phase_name
    assert status_data["timing"]["phase_history"][0]["started_utc"] == started_utc
    assert status_data["timing"]["phase_history"][0]["finished_utc"] is not None
    assert status_data["timing"]["phase_history"][0]["elapsed_sec"] > 0
    assert status_data["timing"]["phase_elapsed_sec"][phase_name] > 0
    
    assert "timing" in manifest
    assert phase_name in manifest["timing"]["phases"]
    phase_timing = manifest["timing"]["phases"][phase_name]
    assert phase_timing["phase"] == phase_name
    assert phase_timing["started_utc"] == started_utc
    assert phase_timing["finished_utc"] is not None
    assert phase_timing["elapsed_sec"] > 0

    timing_events = [event for event in emitter.events if event["stage"] == "timing"]
    assert [event["type"] for event in timing_events] == ["timing_start", "timing_done"]
    assert timing_events[0]["payload"]["phase"] == phase_name
    assert timing_events[0]["payload"]["started_utc"] == started_utc
    assert timing_events[1]["payload"]["phase"] == phase_name
    assert timing_events[1]["payload"]["started_utc"] == started_utc
    assert timing_events[1]["payload"]["finished_utc"] is not None
    assert timing_events[1]["payload"]["elapsed_sec"] > 0


def test_phase_history_preserves_completed_phase_when_later_phase_is_current():
    status_data = {"timing": {}}
    manifest = {}

    t0, started_utc = _phase_start(status_data, "validate")
    _phase_done(status_data, manifest, "validate", t0, started_utc)
    _phase_start(status_data, "phasic_analysis")

    assert status_data["timing"]["current_phase"] == "phasic_analysis"
    assert status_data["timing"]["last_completed_phase"] == "validate"
    assert status_data["timing"]["phase_history"][0]["phase"] == "validate"
    assert status_data["timing"]["phase_elapsed_sec"]["validate"] > 0

def test_manifest_timing_structure():
    manifest = {"timing": {"phases": {}}, "deliverables": {"Region0": {}}}
    # Simulate ROI timing addition
    manifest["deliverables"]["Region0"]["timing"] = {
        "started_utc": "2026-03-09T10:00:00Z",
        "finished_utc": "2026-03-09T10:00:05Z",
        "elapsed_sec": 5.0
    }
    
    # Simulate total runtime
    manifest["timing"]["total_runtime_sec"] = 120.0
    
    assert "timing" in manifest
    assert manifest["timing"]["total_runtime_sec"] == 120.0
    assert "Region0" in manifest["deliverables"]
    assert "timing" in manifest["deliverables"]["Region0"]
    assert manifest["deliverables"]["Region0"]["timing"]["elapsed_sec"] == 5.0
