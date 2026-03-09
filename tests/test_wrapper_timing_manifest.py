import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Bootstrap repo root
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tools.run_full_pipeline_deliverables import main

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
    config_file.write_text("sessions_per_hour: 1")
    
    # Mock data for session compute
    phasic_out = mock_run_dir / "_analysis" / "phasic_out"
    traces_dir = phasic_out / "traces"
    traces_dir.mkdir(parents=True)
    (traces_dir / "chunk_0000.csv").write_text("time_sec,Region0_deltaF\n0.0,0.0\n3600.0,1.0")
    
    # Mock phasic out so has_features is true
    (phasic_out / "features").mkdir(parents=True)
    (phasic_out / "features" / "features.csv").write_text("roi,chunk_id\nRegion0,0")
    
    # Also need tonic_out for plots
    tonic_out = mock_run_dir / "_analysis" / "tonic_out"
    tonic_out.mkdir(parents=True)
    
    args = [
        "tools/run_full_pipeline_deliverables.py",
        "--input", str(input_dir),
        "--out", str(mock_run_dir),
        "--config", str(config_file),
        "--format", "rwd",
        "--mode", "both",
        "--overwrite"
    ]
    
    # Mock Config
    mock_cfg = MagicMock()
    mock_cfg.sessions_per_hour = 1
    mock_cfg.event_signal = "dff"
    mock_cfg.representative_session_index = 0
    mock_cfg.preview_first_n = None
    mock_cfg.smooth_window_s = 1.0
    
    # Mock discovery results
    mock_discovery = {
        "sessions": [{"id": "chunk_0000"}],
        "rois": ["Region0"]
    }
    
    with patch("sys.argv", args), \
         patch("subprocess.check_call") as mock_run, \
         patch("photometry_pipeline.config.Config.from_yaml", return_value=mock_cfg), \
         patch("photometry_pipeline.discovery.discover_inputs", return_value=mock_discovery), \
         patch("tools.run_full_pipeline_deliverables.validate_inputs"), \
         patch("tools.run_full_pipeline_deliverables._cleanup_run_outputs_in_place"), \
         patch("tools.run_full_pipeline_deliverables._ensure_root_run_report", return_value=True):
        
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
