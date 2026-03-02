import pytest
import os
import json
import tempfile
from photometry_pipeline.config import Config
from photometry_pipeline.core.reporting import generate_run_report

def test_baseline_semantics_generation():
    """
    Verifies that run_report.json contains correct, method-specific baseline semantics.
    """
    # Case 1: uv_raw_percentile_session
    cfg1 = Config()
    cfg1.baseline_method = 'uv_raw_percentile_session'
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_run_report(cfg1, tmpdir)
        with open(os.path.join(tmpdir, "run_report.json"), 'r') as f:
            report = json.load(f)
            
        semantics = report['analytical_contract']['baseline_semantics']
        assert semantics['method'] == 'uv_raw_percentile_session'
        assert semantics['f0_source'] == 'uv_raw'
        assert semantics['f0_units'] == 'uv-scale'
        assert semantics['dff_formula'] == '100 * (sig_raw - uv_fit) / F0'

    # Case 2: uv_globalfit_percentile_session
    cfg2 = Config()
    cfg2.baseline_method = 'uv_globalfit_percentile_session'
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_run_report(cfg2, tmpdir)
        with open(os.path.join(tmpdir, "run_report.json"), 'r') as f:
            report = json.load(f)
            
        semantics = report['analytical_contract']['baseline_semantics']
        assert semantics['method'] == 'uv_globalfit_percentile_session'
        assert semantics['f0_source'] == 'uv_est'
        assert semantics['f0_units'] == 'signal-scale'
