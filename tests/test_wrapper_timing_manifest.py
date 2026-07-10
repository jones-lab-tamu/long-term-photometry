import os
import sys
import json
import pytest
import h5py
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Bootstrap repo root
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tools.run_full_pipeline_deliverables import main
from tests.terminal_run_fixtures import (
    BASE_CONFIG_PATH,
    seed_wrapper_analysis_outputs,
    seed_wrapper_deliverables,
)

@pytest.fixture
def mock_run_dir(tmp_path):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    return run_dir

def test_wrapper_timing_manifest_full_run_mocked(mock_run_dir, tmp_path):
    """
    Test main() with heavy mocking to verify manifest timing structure 
    on the success path.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_file = tmp_path / "config.yaml"
    config_file.write_text(BASE_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    # Seed the artifacts the mocked analysis subprocesses would really produce.
    # The wrapper now refuses to finalize a run whose mandatory outputs are absent,
    # so a mocked run must leave behind a real terminal artifact set, and a real
    # discoverable input so the run-wide freeze succeeds.
    seed_wrapper_analysis_outputs(mock_run_dir, input_dir)
    # The plot/table subprocesses are mocked too, so stand in for the per-ROI
    # deliverables a real full run would have written.
    seed_wrapper_deliverables(mock_run_dir, ["Region0"])

    args = [
        "tools/run_full_pipeline_deliverables.py",
        "--input", str(input_dir),
        "--out", str(mock_run_dir),
        "--config", str(config_file),
        "--format", "rwd",
        "--mode", "both",
        "--sessions-per-hour", "1",
        "--overwrite"
    ]
    
    # Mock discovery results
    mock_discovery = {
        "sessions": [{"id": "chunk_0000"}],
        "rois": ["Region0"]
    }

    with patch("sys.argv", args), \
         patch("subprocess.check_call") as mock_run, \
         patch("photometry_pipeline.discovery.discover_inputs", return_value=mock_discovery), \
         patch("tools.run_full_pipeline_deliverables.validate_inputs"), \
         patch("tools.run_full_pipeline_deliverables._cleanup_run_outputs_in_place"):
        
        # Suppress EventEmitter real files if needed, but here we let it write to tmp
        try:
            main()
        except SystemExit as e:
            if e.code != 0:
                print(f"\nWRAPPER FAILED WITH CODE: {e.code}")
                # The exception info might still be available in sys.exc_info
                import traceback
                traceback.print_exc()
            assert e.code == 0
        except Exception as ex:
            print(f"\nUNCAUGHT EXCEPTION IN MAIN: {ex}")
            import traceback
            traceback.print_exc()
            raise ex
            
        # 1. Verify Manifest
        manifest_path = mock_run_dir / "MANIFEST.json"
        assert manifest_path.exists()
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        assert "timing" in manifest
        assert "total_runtime_sec" in manifest["timing"]
        assert manifest["timing"]["total_runtime_sec"] >= 0
        
        phases = manifest["timing"]["phases"]
        # Required phases
        expected_phases = [
            "validate", "tonic_analysis", "phasic_analysis", 
            "session_compute", "plots_total", "manifest_write", 
            "finalize_artifacts"
        ]
        for p in expected_phases:
            assert p in phases, f"Phase {p} missing from manifest timing"
            assert "elapsed_sec" in phases[p]
            
        # 2. Verify ROI timing
        assert "Region0" in manifest["deliverables"]
        assert "timing" in manifest["deliverables"]["Region0"]
        roi_timing = manifest["deliverables"]["Region0"]["timing"]
        assert "started_utc" in roi_timing
        assert "finished_utc" in roi_timing
        assert "elapsed_sec" in roi_timing
        
        # 3. Verify status.json
        status_path = mock_run_dir / "status.json"
        assert status_path.exists()
        with open(status_path, "r") as f:
            status = json.load(f)
            
        assert "timing" in status
        assert status["timing"]["last_completed_phase"] == "finalize_artifacts"
        assert "last_phase_elapsed_sec" in status["timing"]
        assert "phase_history" in status["timing"]
        assert "phase_elapsed_sec" in status["timing"]
        history_phases = [record["phase"] for record in status["timing"]["phase_history"]]
        for p in expected_phases:
            assert p in history_phases
            assert p in status["timing"]["phase_elapsed_sec"]
            assert status["timing"]["phase_elapsed_sec"][p] >= 0

        events_path = mock_run_dir / "events.ndjson"
        assert events_path.exists()
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        timing_events = [event for event in events if event.get("stage") == "timing"]
        assert any(event.get("type") == "timing_start" for event in timing_events)
        assert any(event.get("type") == "timing_done" for event in timing_events)
