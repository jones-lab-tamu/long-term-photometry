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


def test_cli_validate_only_continuous_records_planned_settings():
    """Validate-only in continuous mode should record settings and report gating note."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        run_dir = os.path.join(tmpdir, "run_continuous")
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                "lowpass_hz: 2.0\n"
                "acquisition_mode: continuous\n"
                "continuous_window_sec: 600.0\n"
                "continuous_step_sec: 600.0\n"
                "allow_partial_final_window: false\n"
            )

        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", input_dir,
            "--out", run_dir,
            "--config", config_path,
            "--format", "auto",
            "--validate-only",
            "--acquisition-mode", "continuous",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, res.stderr
        assert "VALIDATE-ONLY: OK" in res.stdout
        assert "Continuous acquisition mode is recognized" in res.stdout

        status_path = os.path.join(run_dir, "status.json")
        assert os.path.exists(status_path)
        with open(status_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["status"] == "success"
        assert data["phase"] == "final"
        assert data["acquisition_mode"] == "continuous"
        assert float(data["continuous_window_sec"]) == 600.0
        assert float(data["continuous_step_sec"]) == 600.0
        assert data["allow_partial_final_window"] is False


def test_cli_full_run_continuous_is_gated_before_analysis():
    """Full continuous runs must fail early before analysis output trees are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        run_dir = os.path.join(tmpdir, "run_continuous_full")
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                "lowpass_hz: 2.0\n"
                "acquisition_mode: continuous\n"
                "continuous_window_sec: 600.0\n"
                "continuous_step_sec: 600.0\n"
                "allow_partial_final_window: false\n"
            )

        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", input_dir,
            "--out", run_dir,
            "--config", config_path,
            "--format", "auto",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 1
        combined = f"{res.stdout}\n{res.stderr}"
        assert (
            "Continuous acquisition mode is recognized, but continuous windowed ingestion is not yet implemented in this build."
            in combined
        )

        status_path = os.path.join(run_dir, "status.json")
        assert os.path.exists(status_path)
        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)
        assert status["status"] == "error"
        assert status["phase"] == "final"
        assert status["acquisition_mode"] == "continuous"
        assert not os.path.exists(os.path.join(run_dir, "_analysis"))
