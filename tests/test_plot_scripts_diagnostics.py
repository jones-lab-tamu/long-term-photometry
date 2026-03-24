import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

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
    import h5py
    analysis_out = tmp_path / "analysis_out"
    tonic_out_dir = analysis_out / "tonic_out"
    tonic_out_dir.mkdir(parents=True)
    
    # Create a minimal h5 cache
    cache_path = tonic_out_dir / "tonic_trace_cache.h5"
    with h5py.File(cache_path, 'w') as f:
        meta = f.create_group('meta')
        meta.attrs['mode'] = 'tonic'
        meta.attrs['schema_version'] = '1.0'
        dt_str = h5py.string_dtype(encoding='utf-8')
        meta.create_dataset('rois', data=np.array(['ROI1'], dtype=object), dtype=dt_str)
        meta.create_dataset('chunk_ids', data=np.array([0], dtype=int))
        meta.create_dataset('source_files', data=np.array(['f0.csv'], dtype=object), dtype=dt_str)
        
        grp_c = f.create_group('roi/ROI1/chunk_0')
        t = np.linspace(0, 10, 100)
        grp_c.create_dataset('time_sec', data=t)
        grp_c.create_dataset('sig_raw', data=np.random.randn(100))
        grp_c.create_dataset('uv_raw', data=np.random.randn(100))
        grp_c.create_dataset('deltaF', data=np.random.randn(100))
    
    script_path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'plot_tonic_48h.py')
    
    cmd = [
        "python",
        os.path.abspath(script_path),
        "--analysis-out", str(tonic_out_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0, f"Script failed: {result.stderr}\n\n{result.stdout}"
    
    out = result.stdout
    
    # Check for required timing checkpoints
    assert "PLOT_TIMING START script=plot_tonic_48h.py" in out
    assert "step=discovery" in out
    assert "step=cache_read" in out
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
    
    # Phasic trace cache expectations
    import h5py
    cache_path = analysis_out / 'phasic_trace_cache.h5'
    with h5py.File(cache_path, 'w') as f:
        meta = f.create_group('meta')
        meta.attrs['mode'] = 'phasic'
        meta.attrs['schema_version'] = '1.0'
        
        dt_str = h5py.string_dtype(encoding='utf-8')
        meta.create_dataset('rois', data=np.array(['Region1'], dtype=object), dtype=dt_str)
        meta.create_dataset('chunk_ids', data=np.array([0], dtype=int))
        
        c0 = f.create_group('roi/Region1/chunk_0')
        c0.create_dataset('time_sec', data=np.linspace(0, 10, 100))
        c0.create_dataset('sig_raw', data=np.random.randn(100))
        c0.create_dataset('uv_raw', data=np.random.randn(100))
        c0.create_dataset('dff', data=np.zeros(100))
    
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
    assert "step=cache_read" in out
    assert "step=verification" in out
    assert "step=cache_build" in out
    assert "step=global_limits" in out
    assert "step=plotting" in out
    assert "step=figure_save" in out
    assert "PLOT_TIMING DONE script=plot_phasic_dayplot_bundle.py total_sec=" in out


def test_plot_phasic_time_series_summary_prefers_datetime_from_full_source_path(tmp_path):
    """
    Real RWD features often store source_file ending in fluorescence.csv, with the
    timestamp carried by the parent folder. Ensure datetime-mode is still used.
    """
    analysis_out = tmp_path / "analysis_out"
    feat_dir = analysis_out / "features"
    feat_dir.mkdir(parents=True)

    rows = []
    t0 = datetime(2026, 3, 10, 0, 0, 0)
    for cid in range(96):
        dt = t0 + timedelta(minutes=30 * cid)
        rows.append(
            {
                "chunk_id": cid,
                "roi": "ROI1",
                "peak_count": 3,
                "auc": 1.25,
                "source_file": f"C:/vendor_rwd/{dt.strftime('%Y_%m_%d-%H_%M_%S')}/fluorescence.csv",
            }
        )
    pd.DataFrame(rows).to_csv(feat_dir / "features.csv", index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rate_csv = out_dir / "rate.csv"
    out_auc_csv = out_dir / "auc.csv"
    out_rate_png = out_dir / "rate.png"
    out_auc_png = out_dir / "auc.png"

    script_path = os.path.join(os.path.dirname(__file__), "..", "tools", "plot_phasic_time_series_summary.py")
    cmd = [
        "python",
        os.path.abspath(script_path),
        "--analysis-out",
        str(analysis_out),
        "--sessions-per-hour",
        "5",
        "--out-rate-png",
        str(out_rate_png),
        "--out-auc-png",
        str(out_auc_png),
        "--out-rate-csv",
        str(out_rate_csv),
        "--out-auc-csv",
        str(out_auc_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}\n\n{result.stdout}"
    assert "Datetime-derived session timeline" in result.stdout
    assert "Timeline anchor: civil-clock" in result.stdout

    df_rate = pd.read_csv(out_rate_csv)
    assert float(df_rate["time_hours"].max()) > 47.0
    assert float(df_rate["time_hours"].max()) == pytest.approx(47.5, rel=0.0, abs=1e-6)
    assert int(df_rate["day"].max()) == 1
    assert (df_rate["timeline_anchor_mode"] == "civil").all()
    assert (df_rate["timeline_anchor_label"] == "civil-clock").all()
    assert df_rate["fixed_daily_anchor_clock"].isna().all()


def test_plot_phasic_time_series_summary_fixed_daily_anchor_mode(tmp_path):
    analysis_out = tmp_path / "analysis_out"
    feat_dir = analysis_out / "features"
    feat_dir.mkdir(parents=True)
    rows = [
        {
            "chunk_id": 0,
            "roi": "ROI1",
            "peak_count": 1,
            "auc": 0.1,
            "source_file": "C:/vendor/2026_03_10-11_33_05/fluorescence.csv",
        },
        {
            "chunk_id": 1,
            "roi": "ROI1",
            "peak_count": 1,
            "auc": 0.2,
            "source_file": "C:/vendor/2026_03_10-12_03_05/fluorescence.csv",
        },
    ]
    pd.DataFrame(rows).to_csv(feat_dir / "features.csv", index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rate_csv = out_dir / "rate.csv"
    out_auc_csv = out_dir / "auc.csv"
    out_rate_png = out_dir / "rate.png"
    out_auc_png = out_dir / "auc.png"
    script_path = os.path.join(os.path.dirname(__file__), "..", "tools", "plot_phasic_time_series_summary.py")
    cmd = [
        "python",
        os.path.abspath(script_path),
        "--analysis-out",
        str(analysis_out),
        "--sessions-per-hour",
        "2",
        "--timeline-anchor-mode",
        "fixed_daily_anchor",
        "--fixed-daily-anchor-clock",
        "07:00",
        "--out-rate-png",
        str(out_rate_png),
        "--out-auc-png",
        str(out_auc_png),
        "--out-rate-csv",
        str(out_rate_csv),
        "--out-auc-csv",
        str(out_auc_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}\n\n{result.stdout}"
    assert "Datetime-derived session timeline" in result.stdout
    assert "Timeline anchor: fixed-daily-anchor@07:00:00" in result.stdout

    df_rate = pd.read_csv(out_rate_csv).sort_values("time_hours").reset_index(drop=True)
    # 11:33 with anchor 07:00 => hour 4, right slot.
    assert int(df_rate.loc[0, "hour"]) == 4
    assert int(df_rate.loc[0, "session_in_hour"]) == 1
    assert float(df_rate.loc[0, "time_hours"]) == pytest.approx(4 + (33 * 60 + 5) / 3600.0, rel=0.0, abs=1e-6)
    # 12:03 with anchor 07:00 => hour 5, left slot.
    assert int(df_rate.loc[1, "hour"]) == 5
    assert int(df_rate.loc[1, "session_in_hour"]) == 0
    assert float(df_rate.loc[1, "time_hours"]) == pytest.approx(5 + (3 * 60 + 5) / 3600.0, rel=0.0, abs=1e-6)
    assert (df_rate["timeline_anchor_mode"] == "fixed_daily_anchor").all()
    assert (df_rate["timeline_anchor_label"] == "fixed-daily-anchor@07:00:00").all()
    assert (df_rate["fixed_daily_anchor_clock"] == "07:00:00").all()
    assert (df_rate["time_axis_semantics"] == "Anchored time (hours from daily anchor 07:00:00)").all()


def test_plot_phasic_time_series_summary_elapsed_mode_uses_elapsed_hours(tmp_path):
    analysis_out = tmp_path / "analysis_out"
    feat_dir = analysis_out / "features"
    feat_dir.mkdir(parents=True)
    rows = [
        {
            "chunk_id": 0,
            "roi": "ROI1",
            "peak_count": 1,
            "auc": 0.1,
            "source_file": "C:/vendor/2026_03_10-11_33_05/fluorescence.csv",
        },
        {
            "chunk_id": 1,
            "roi": "ROI1",
            "peak_count": 1,
            "auc": 0.2,
            "source_file": "C:/vendor/2026_03_10-12_03_05/fluorescence.csv",
        },
        {
            "chunk_id": 2,
            "roi": "ROI1",
            "peak_count": 1,
            "auc": 0.3,
            "source_file": "C:/vendor/2026_03_10-12_33_05/fluorescence.csv",
        },
    ]
    pd.DataFrame(rows).to_csv(feat_dir / "features.csv", index=False)

    out_dir = tmp_path / "out_elapsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rate_csv = out_dir / "rate.csv"
    out_auc_csv = out_dir / "auc.csv"
    out_rate_png = out_dir / "rate.png"
    out_auc_png = out_dir / "auc.png"
    script_path = os.path.join(os.path.dirname(__file__), "..", "tools", "plot_phasic_time_series_summary.py")
    cmd = [
        "python",
        os.path.abspath(script_path),
        "--analysis-out",
        str(analysis_out),
        "--sessions-per-hour",
        "2",
        "--timeline-anchor-mode",
        "elapsed",
        "--out-rate-png",
        str(out_rate_png),
        "--out-auc-png",
        str(out_auc_png),
        "--out-rate-csv",
        str(out_rate_csv),
        "--out-auc-csv",
        str(out_auc_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}\n\n{result.stdout}"
    assert "Timeline anchor: elapsed-from-first-session" in result.stdout
    assert "Summary x-axis: Elapsed time (hours from first session)" in result.stdout

    df_rate = pd.read_csv(out_rate_csv).sort_values("time_hours").reset_index(drop=True)
    assert float(df_rate.loc[0, "time_hours"]) == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert float(df_rate.loc[1, "time_hours"]) == pytest.approx(0.5, rel=0.0, abs=1e-6)
    assert float(df_rate.loc[2, "time_hours"]) == pytest.approx(1.0, rel=0.0, abs=1e-6)
    assert (df_rate["time_axis_semantics"] == "Elapsed time (hours from first session)").all()


def test_plot_phasic_time_series_summary_civil_mode_uses_civil_hour_axis(tmp_path):
    analysis_out = tmp_path / "analysis_out"
    feat_dir = analysis_out / "features"
    feat_dir.mkdir(parents=True)
    rows = [
        {
            "chunk_id": 0,
            "roi": "ROI1",
            "peak_count": 1,
            "auc": 0.1,
            "source_file": "C:/vendor/2026_03_10-11_33_05/fluorescence.csv",
        },
        {
            "chunk_id": 1,
            "roi": "ROI1",
            "peak_count": 1,
            "auc": 0.2,
            "source_file": "C:/vendor/2026_03_10-12_03_05/fluorescence.csv",
        },
        {
            "chunk_id": 2,
            "roi": "ROI1",
            "peak_count": 1,
            "auc": 0.3,
            "source_file": "C:/vendor/2026_03_10-12_33_05/fluorescence.csv",
        },
    ]
    pd.DataFrame(rows).to_csv(feat_dir / "features.csv", index=False)

    out_dir = tmp_path / "out_civil"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rate_csv = out_dir / "rate.csv"
    out_auc_csv = out_dir / "auc.csv"
    out_rate_png = out_dir / "rate.png"
    out_auc_png = out_dir / "auc.png"
    script_path = os.path.join(os.path.dirname(__file__), "..", "tools", "plot_phasic_time_series_summary.py")
    cmd = [
        "python",
        os.path.abspath(script_path),
        "--analysis-out",
        str(analysis_out),
        "--sessions-per-hour",
        "2",
        "--timeline-anchor-mode",
        "civil",
        "--out-rate-png",
        str(out_rate_png),
        "--out-auc-png",
        str(out_auc_png),
        "--out-rate-csv",
        str(out_rate_csv),
        "--out-auc-csv",
        str(out_auc_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}\n\n{result.stdout}"
    assert "Timeline anchor: civil-clock" in result.stdout
    assert "Summary x-axis: Civil-clock time (hours from day-0 midnight)" in result.stdout

    df_rate = pd.read_csv(out_rate_csv).sort_values("time_hours").reset_index(drop=True)
    assert float(df_rate.loc[0, "time_hours"]) == pytest.approx(11 + (33 * 60 + 5) / 3600.0, rel=0.0, abs=1e-6)
    assert float(df_rate.loc[1, "time_hours"]) == pytest.approx(12 + (3 * 60 + 5) / 3600.0, rel=0.0, abs=1e-6)
    assert float(df_rate.loc[2, "time_hours"]) == pytest.approx(12 + (33 * 60 + 5) / 3600.0, rel=0.0, abs=1e-6)
    # Guard against silently keeping elapsed-only semantics for civil mode.
    assert not np.allclose(df_rate["time_hours"].to_numpy(dtype=float), np.array([0.0, 0.5, 1.0]))
    assert (df_rate["time_axis_semantics"] == "Civil-clock time (hours from day-0 midnight)").all()
