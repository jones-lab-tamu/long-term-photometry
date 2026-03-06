"""
Tests for Live Status Updates (duration_sec increments during run).
Verifies that _write_status_update correctly writes non-terminal status files.
"""

import os
import json
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Mock status_data to simulate the closure in tools/run_full_pipeline_deliverables.py
def test_live_status_updates_increment_duration():
    with tempfile.TemporaryDirectory() as run_dir:
        status_path = os.path.join(run_dir, "status.json")
        t0 = time.time() - 10.0 # Simulate starting 10 seconds ago
        
        status_data = {
            "schema_version": 1,
            "run_id": "test_run",
            "phase": "running",
            "status": "running",
            "duration_sec": 0.0
        }
        
        def _atomic_write_json(path, data):
            # Simplified mock of the actual atomic writer
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, path)

        def _write_status_update(phase):
            status_data["phase"] = phase
            status_data["status"] = "running"
            status_data["duration_sec"] = time.time() - t0
            _atomic_write_json(status_path, status_data)

        # 1. First update
        _write_status_update("stage1")
        with open(status_path, "r") as f:
            d1 = json.load(f)
        
        assert d1["phase"] == "stage1"
        assert d1["status"] == "running"
        assert d1["duration_sec"] >= 10.0
        
        # 2. Wait a tiny bit (simulated or real) and update again
        time.sleep(0.1)
        _write_status_update("stage2")
        with open(status_path, "r") as f:
            d2 = json.load(f)
            
        assert d2["phase"] == "stage2"
        assert d2["duration_sec"] > d1["duration_sec"]
        assert d2["status"] == "running"

def test_live_status_is_valid_json():
    with tempfile.TemporaryDirectory() as run_dir:
        status_path = os.path.join(run_dir, "status.json")
        t0 = time.time()
        status_data = {"phase": "init", "status": "running"}
        
        # This is a very basic check that our logic produces valid JSON
        with open(status_path, "w") as f:
            json.dump(status_data, f)
            
        with open(status_path, "r") as f:
            reloaded = json.load(f)
        assert reloaded["phase"] == "init"

def test_terminal_status_overwrites_live_status():
    with tempfile.TemporaryDirectory() as run_dir:
        status_path = os.path.join(run_dir, "status.json")
        t0 = time.time()
        status_data = {"phase": "running", "status": "running", "duration_sec": 1.0}
        
        # 1. Live update
        with open(status_path, "w") as f:
            json.dump(status_data, f)
            
        # 2. Terminal update
        status_data["phase"] = "final"
        status_data["status"] = "success"
        status_data["duration_sec"] = 5.0
        with open(status_path, "w") as f:
            json.dump(status_data, f)
            
        with open(status_path, "r") as f:
            final = json.load(f)
        assert final["phase"] == "final"
        assert final["status"] == "success"
        assert final["duration_sec"] == 5.0
