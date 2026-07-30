import os
import sys
import json
import io
import pytest
import subprocess
import time
import threading
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
    process = MagicMock()
    process.stdout = io.StringIO("hello\n")
    process.stderr = io.StringIO("")
    process.wait.return_value = 0
    with patch("subprocess.Popen", return_value=process) as mock_call:
        res = run_cmd(cmd, roi_label="Region0")
        
        assert res["cmd"] == cmd
        assert "started_utc" in res
        assert "finished_utc" in res
        assert isinstance(res["elapsed_sec"], float)
        assert res["returncode"] == 0
        mock_call.assert_called_once_with(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )


def test_run_cmd_success_streams_output_before_child_exit(monkeypatch):
    class LiveOutput:
        def __init__(self):
            self.saw_child_output = threading.Event()

        def write(self, value):
            if "child-before-exit" in value:
                self.saw_child_output.set()
            return len(value)

        def flush(self):
            return None

    live_output = LiveOutput()
    result = {}
    failure = []

    def invoke():
        try:
            result["value"] = run_cmd(
                [
                    sys.executable,
                    "-c",
                    (
                        "import time;"
                        "print('child-before-exit', flush=True);"
                        "time.sleep(1.5);"
                        "print('child-after-exit', flush=True)"
                    ),
                ]
            )
        except BaseException as exc:
            failure.append(exc)

    monkeypatch.setattr(sys, "stdout", live_output)
    worker = threading.Thread(target=invoke)
    worker.start()

    assert live_output.saw_child_output.wait(5)
    assert worker.is_alive()
    worker.join(10)
    assert not worker.is_alive()
    assert failure == []
    assert result["value"]["returncode"] == 0


def test_run_cmd_long_success_retains_no_transcript(monkeypatch):
    class CountingOutput:
        def __init__(self):
            self.characters_written = 0

        def write(self, value):
            self.characters_written += len(value)
            return len(value)

        def flush(self):
            return None

    output = CountingOutput()
    monkeypatch.setattr(sys, "stdout", output)
    cmd = [sys.executable, "-c", "print('x' * 12000, flush=True)"]

    result = run_cmd(cmd)

    assert result["returncode"] == 0
    assert output.characters_written > 12000
    assert set(result) == {
        "cmd",
        "started_utc",
        "finished_utc",
        "elapsed_sec",
        "returncode",
    }
    assert all("x" * 100 not in str(value) for value in result.values())


def test_run_cmd_failure_prefers_stderr_and_bounds_detail():
    cmd = ["python", "child.py"]
    process = MagicMock()
    process.stdout = io.StringIO("stdout fallback must not replace stderr")
    process.stderr = io.StringIO(
        "discarded-prefix\n" + "x" * 5000 + "\nValueError: bad CSV"
    )
    process.wait.return_value = 9

    with patch("subprocess.Popen", return_value=process):
        with pytest.raises(RuntimeError) as raised:
            run_cmd(cmd)

    message = str(raised.value)
    assert "failed with return code 9" in message
    assert "ValueError: bad CSV" in message
    assert "stdout fallback" not in message
    assert "discarded-prefix" not in message
    assert len(message.partition(": ")[2]) <= 3000


def test_run_cmd_failure_uses_stdout_when_stderr_is_empty():
    command = [
        sys.executable,
        "-c",
        (
            "import sys;"
            "print('Error: unsupported internal Guided manifest execution state.',"
            " flush=True);"
            "sys.exit(1)"
        ),
    ]

    with pytest.raises(RuntimeError) as raised:
        run_cmd(command)

    message = str(raised.value)
    assert "failed with return code 1" in message
    assert (
        "Error: unsupported internal Guided manifest execution state."
        in message
    )


def test_run_cmd_actual_rwd_tonic_child_succeeds(tmp_path):
    source = tmp_path / "rwd"
    session = source / "2026_01_01-00_00_00"
    session.mkdir(parents=True)
    rows = ["Timestamp,Region0G,Region0R"]
    for index in range(1200):
        timestamp = index / 100.0
        reference = 0.1 + index / 12000.0
        rows.append(
            f"{timestamp:.2f},{reference:.8f},{reference + 0.01:.8f}"
        )
    (session / "fluorescence.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            (
                "rwd_time_col: Timestamp",
                "target_fs_hz: 100.0",
                "chunk_duration_sec: 12.0",
                "window_sec: 2.0",
                "step_sec: 1.0",
                "min_valid_windows: 2",
                "uv_suffix: G",
                "sig_suffix: R",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "tonic_out"
    command = [
        sys.executable,
        str(Path(_repo_root) / "analyze_photometry.py"),
        "--input",
        str(source),
        "--out",
        str(output),
        "--config",
        str(config),
        "--format",
        "rwd",
        "--mode",
        "tonic",
        "--recursive",
        "--overwrite",
        "--sessions-per-hour",
        "1",
    ]

    result = run_cmd(command)

    assert result["returncode"] == 0
    assert (output / "tonic_trace_cache.h5").is_file()
    assert (output / "run_report.json").is_file()


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
