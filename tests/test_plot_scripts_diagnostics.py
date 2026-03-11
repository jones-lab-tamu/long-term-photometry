import os
import subprocess
import tempfile
import pandas as pd
import numpy as np

def test_plot_phasic_time_series_summary_no_datetime_warning(tmp_path):
    """
    Ensures that when features.csv has non-datetime source_file strings (e.g., 'chunk_0.csv'),
    the pandas pd.to_datetime generic inference warning is not triggered.
    """
    # 1. Setup analysis output structure
    analysis_out = tmp_path / "analysis_out"
    feat_dir = analysis_out / "features"
    feat_dir.mkdir(parents=True)
    
    # Create minimal features.csv
    df = pd.DataFrame({
        'chunk_id': [0, 1],
        'roi': ['ROI1', 'ROI1'],
        'peak_count': [5, 10],
        'auc': [1.5, 2.5],
        'source_file': ['chunk_0.csv', 'chunk_1.csv']
    })
    df.to_csv(feat_dir / "features.csv", index=False)
    
    # We must also satisfy the fallback time axis (Chunk ID fallback)
    # The script should run successfully.
    script_path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'plot_phasic_time_series_summary.py')
    
    # Run the script, capturing stderr
    cmd = [
        "python",
        os.path.abspath(script_path),
        "--analysis-out", str(analysis_out)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Assert successful execution
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    # The crucial assertion: Stderr should NOT contain pandas datetime warnings
    # like "UserWarning: Could not infer format" or DateParseWarning. 
    # Just checking for "warning" or "Warning" alongside "datetime" or "dateutil"
    lower_err = result.stderr.lower()
    assert "dateutil" not in lower_err
    assert "to_datetime" not in lower_err
    assert "could not infer format" not in lower_err
    
    # Ensure it says it's using the fallback
    assert "Fallback" in result.stdout or "Chunk ID fallback" in result.stdout

def test_plot_tonic_48h_timing_output(tmp_path):
    """
    Verifies that the new script-local timing lines are printed securely during typical execution.
    """
    analysis_out = tmp_path / "analysis_out"
    traces_dir = analysis_out / "traces"
    traces_dir.mkdir(parents=True)
    
    # Create a minimal chunk file
    df = pd.DataFrame({
        'time_sec': np.linspace(0, 10, 100),
        'ROI1_sig_raw': np.random.randn(100),
        'ROI1_uv_raw': np.random.randn(100),
        'ROI1_deltaF': np.random.randn(100)
    })
    df.to_csv(traces_dir / "chunk_0.csv", index=False)
    
    script_path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'plot_tonic_48h.py')
    
    cmd = [
        "python",
        os.path.abspath(script_path),
        "--analysis-out", str(analysis_out)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    out = result.stdout
    
    # Check for required timing checkpoints
    assert "PLOT_TIMING START script=plot_tonic_48h.py" in out
    assert "step=discovery" in out
    assert "step=csv_read" in out
    assert "step=assembly" in out
    assert "step=plotting" in out
    assert "step=figure_save" in out
    assert "PLOT_TIMING DONE script=plot_tonic_48h.py total_sec=" in out

def test_plot_phasic_dayplot_bundle_timing_output(tmp_path):
    """
    Verifies that the refactored phasic bundle prints granular timing lines for its internal stages.
    """
    analysis_out = tmp_path / "analysis_out"
    traces_dir = analysis_out / "traces"
    features_dir = analysis_out / "features"
    out_dir = tmp_path / "plots_out"
    config_path = analysis_out / "config_used.yaml"
    
    traces_dir.mkdir(parents=True)
    features_dir.mkdir()
    out_dir.mkdir()
    
    with open(config_path, "w") as f:
        f.write("target_fs_hz: 10.0\n")
        f.write("peak_threshold_method: 'absolute'\n")
        f.write("peak_threshold_abs: 0.5\n")
        f.write("peak_min_distance_sec: 1.0\n")
        
    # Minimal chunk matching discovery expectations
    df = pd.DataFrame({
        'time_sec': np.linspace(0, 10, 100),
        'Region1_sig_raw': np.random.randn(100),
        'Region1_uv_raw': np.random.randn(100),
        'Region1_dff': np.zeros(100)
    })
    df.to_csv(traces_dir / "chunk_0.csv", index=False)
    
    # minimal features.csv
    pd.DataFrame([{
        'chunk_id': 0, 'roi': 'Region1', 'peak_count': 0, 'auc': 0.0
    }]).to_csv(features_dir / "features.csv", index=False)
    
    script_path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'plot_phasic_dayplot_bundle.py')
    
    cmd = [
        "python", os.path.abspath(script_path),
        "--analysis-out", str(analysis_out),
        "--roi", "Region1",
        "--output-dir", str(out_dir),
        "--sessions-per-hour", "1",
        "--session-duration-s", "10.0"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    out = result.stdout
    assert "PLOT_TIMING START script=plot_phasic_dayplot_bundle.py" in out
    assert "step=discovery" in out
    assert "step=csv_read" in out
    assert "step=verification" in out
    assert "step=cache_build" in out
    assert "step=global_limits" in out
    assert "step=plotting" in out
    assert "step=figure_save" in out
    assert "PLOT_TIMING DONE script=plot_phasic_dayplot_bundle.py total_sec=" in out
