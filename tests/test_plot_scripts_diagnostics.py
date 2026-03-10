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
