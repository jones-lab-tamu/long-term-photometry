"""
Tests for Fix B1v4: Validate->Run Consistency Policy.
Verifies signature computation and consistency checking (isolated directories).
"""

import os
import json
import tempfile
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gui.validate_run_policy import (
    compute_run_signature,
    is_validation_current
)

def _write_artifacts(run_dir, spec_dict, config_content):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "gui_run_spec.json"), "w") as f:
        json.dump(spec_dict, f, sort_keys=True)
    with open(os.path.join(run_dir, "config_effective.yaml"), "w") as f:
        f.write(config_content)

def test_compute_run_signature_stable():
    """Verify that identical intent produces identical signature despite metadata changes."""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        # Same intent
        spec_shared = {"input_dir": "data", "format": "rwd"}
        config = "lowpass_hz: 2.0\n"
        
        # Metadata differs
        spec1 = spec_shared.copy()
        spec1["timestamp_local"] = "2023-01-01T12:00:00"
        spec1["run_dir"] = d1
        
        spec2 = spec_shared.copy()
        spec2["timestamp_local"] = "2023-01-02T12:00:00"
        spec2["run_dir"] = d2
        
        _write_artifacts(d1, spec1, config)
        _write_artifacts(d2, spec2, config)
        
        sig1 = compute_run_signature(d1)
        sig2 = compute_run_signature(d2)
        
        assert sig1 == sig2
        assert len(sig1) == 64

def test_compute_run_signature_detects_intent_changes():
    """Verify changes in intent fields are detected."""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        config = "lowpass_hz: 2.0\n"
        
        _write_artifacts(d1, {"input_dir": "data", "format": "rwd"}, config)
        _write_artifacts(d2, {"input_dir": "data", "format": "npm"}, config)
        
        sig1 = compute_run_signature(d1)
        sig2 = compute_run_signature(d2)
        assert sig1 != sig2

def test_is_validation_current():
    """Verify the consistency check logic."""
    sig_v = "abcd"
    assert is_validation_current(sig_v, "abcd") is True
    assert is_validation_current(sig_v, "efgh") is False
    assert is_validation_current(None, "abcd") is False

def test_validate_then_run_isolation_simulation():
    """
    Simulate the new policy: 
    Validate in dir V, then Run in dir R.
    Verify they are isolated but signature matches.
    """
    with tempfile.TemporaryDirectory() as out_base:
        # 1. Validate Only
        v_dir = os.path.join(out_base, "run_v")
        v_spec = {"input_dir": "data", "format": "rwd", "timestamp_local": "T1"}
        v_config = "lowpass_hz: 1.0\n"
        _write_artifacts(v_dir, v_spec, v_config)
        v_sig = compute_run_signature(v_dir)
        
        # 2. Run
        r_dir = os.path.join(out_base, "run_r")
        r_spec = {"input_dir": "data", "format": "rwd", "timestamp_local": "T2"}
        r_config = "lowpass_hz: 1.0\n"
        _write_artifacts(r_dir, r_spec, r_config)
        r_sig = compute_run_signature(r_dir)
        
        # Consistency Check
        assert is_validation_current(v_sig, r_sig) is True
        
        # Isolation Check
        assert v_dir != r_dir
        assert os.path.exists(v_dir)
        assert os.path.exists(r_dir)
        
        # Verify both directories preserve their own artifacts
        with open(os.path.join(v_dir, "gui_run_spec.json")) as f:
            assert json.load(f)["timestamp_local"] == "T1"
        with open(os.path.join(r_dir, "gui_run_spec.json")) as f:
            assert json.load(f)["timestamp_local"] == "T2"
