"""
Tests for validate-only status.json emission contract.
Ensures that the runner writes a terminal status.json even in validate-only mode.
"""

import os
import json
import subprocess
import tempfile
import pytest
import sys

def test_cli_validate_only_success():
    """Verify that a successful validate-only run produces status.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        run_dir = os.path.join(tmpdir, "run")
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("lowpass_hz: 2.0\n")
        
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", input_dir,
            "--out", run_dir,
            "--config", config_path,
            "--format", "auto",
            "--validate-only"
        ]
        subprocess.check_call(cmd)
        
        status_path = os.path.join(run_dir, "status.json")
        assert os.path.exists(status_path), "status.json should be created in validate-only mode"
        
        with open(status_path, "r") as f:
            data = json.load(f)
            
        assert data["status"] == "success"
        assert data["phase"] == "final"
        assert data["run_type"] == "validate_only"

def test_cli_validate_only_failure():
    """Verify that a failed validation produces status.json with error status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Intentionally missing input directory
        input_dir = os.path.join(tmpdir, "NON_EXISTENT_DIR")
        run_dir = os.path.join(tmpdir, "run_fail")
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("lowpass_hz: 2.0\n")
        
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", input_dir,
            "--out", run_dir,
            "--config", config_path,
            "--format", "auto",
            "--validate-only"
        ]
        
        # Expect exit 1 due to validation error
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1
        
        status_path = os.path.join(run_dir, "status.json")
        assert os.path.exists(status_path), "status.json should be created even on validation failure"
        
        with open(status_path, "r") as f:
            data = json.load(f)
            
        assert data["status"] == "error"
        assert data["phase"] == "final"
        assert len(data["errors"]) > 0
        assert "not a directory" in data["errors"][0].lower()
