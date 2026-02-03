
import argparse
import sys
import os
import glob
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

# Import pipeline components
# Ensure current dir is in sys.path
sys.path.append(os.getcwd())
from photometry_pipeline.config import Config
from photometry_pipeline.pipeline import Pipeline

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(args):
    """Executes the standard pipeline via CLI arguments."""
    print(">>> Running Pipeline...")
    config = Config.from_yaml(args.config)
    pipeline = Pipeline(config)
    pipeline.run(
        input_dir=args.input, 
        output_dir=args.out, 
        force_format=args.format, 
        recursive=args.recursive, 
        glob_pattern=args.file_glob
    )
    return config

def load_concatenated_traces(traces_dir, config, roi_name=None):
    """
    Loads all chunk_*.csv files from traces_dir, concatenates them,
    and returns a DataFrame with time and the selected ROI's data.
    """
    csv_files = sorted(glob.glob(os.path.join(traces_dir, "chunk_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No chunk traces found in {traces_dir}")
    
    dfs = []
    chunk_dur = config.chunk_duration_sec
    
    for f in csv_files:
        df = pd.read_csv(f)
        
        # Reconstruct absolute time if known pipeline reset occurs
        try:
            basename = os.path.basename(f)
            idx_str = basename.replace('chunk_', '').replace('.csv', '')
            idx = int(idx_str)
            offset = idx * chunk_dur
            
            if not df.empty and 'time_sec' in df.columns:
                t_start = df['time_sec'].iloc[0]
                # If trace starts near 0 but expected start > 0
                if t_start < (offset - chunk_dur * 0.1):
                    df['time_sec'] += offset
        except Exception:
            pass
            
        dfs.append(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Sort and deduplicate by time to ensure monotonicity
    if 'time_sec' in full_df.columns:
        full_df = full_df.sort_values('time_sec').drop_duplicates(subset=['time_sec']).reset_index(drop=True)
        # Note: traces usually have columns like time_sec, Region0_dff, etc. 
        # Deduplicating on time_sec is usually sufficient per file-set as they share columns.
        # full_df = full_df.drop_duplicates(subset=['time_sec']) 

    if not roi_name:
        # Look for columns ending in _dff or _deltaF
        # Typical format: Region0_dff
        candidates = [c.replace('_dff', '').replace('_deltaF', '') 
                      for c in full_df.columns 
                      if c.endswith('_dff') or c.endswith('_deltaF')]
        candidates = sorted(list(set(candidates)))
        
        if not candidates:
            raise ValueError("Could not auto-detect any ROIs with dff or deltaF in traces.")
            
        roi_name = candidates[0]
        print(f"Auto-selected ROI: {roi_name}")
        
    # Prefer dff, fallback to deltaF
    dff_col = f"{roi_name}_dff"
    deltaf_col = f"{roi_name}_deltaF"
    
    if dff_col in full_df.columns:
        signal = full_df[dff_col]
        metric = "dff"
    elif deltaf_col in full_df.columns:
        signal = full_df[deltaf_col]
        metric = "deltaF"
    else:
        raise KeyError(f"Traces contain neither {dff_col} nor {deltaf_col}.")
        
    time_col = 'time_sec' # Standard pipeline output
    if time_col not in full_df.columns:
        # Fallback if names changed, but they shouldn't have
        raise KeyError("time_sec column missing from trace outputs")
        
    return full_df[time_col], signal, roi_name, metric

def validate_trace_arrays(time_series, signal_series, roi, metric):
    """Performs strict sanity checks on the loaded traces."""
    # Convert to numeric, coercing errors
    t = pd.to_numeric(time_series, errors='coerce').to_numpy()
    s = pd.to_numeric(signal_series, errors='coerce').to_numpy()
    
    n = len(t)
    n_sig = len(s)
    
    # 1. Length checks
    if n == 0 or n_sig == 0:
        print(f"TRACE_INVALID roi={roi} metric={metric} n={n} empty_arrays=True")
        sys.exit(1)
        
    if n != n_sig:
        print(f"TRACE_INVALID roi={roi} metric={metric} n_time={n} n_sig={n_sig} length_mismatch=True")
        sys.exit(1)
        
    # 2. Finite checks
    mask_fin = np.isfinite(t) & np.isfinite(s)
    n_fin = np.sum(mask_fin)
    frac_fin_t = np.isfinite(t).sum() / n
    frac_fin_s = np.isfinite(s).sum() / n
    
    # 3. Stats on finite
    if n_fin > 0:
        s_fin = s[mask_fin]
        min_s, max_s = np.min(s_fin), np.max(s_fin)
    else:
        min_s, max_s = np.nan, np.nan
        
    # Diagnostic Line
    print(f"TRACE_CHECK roi={roi} metric={metric} n={n} n_finite={n_fin} frac_t={frac_fin_t:.2f} frac_s={frac_fin_s:.2f} min_s={min_s:.2f} max_s={max_s:.2f}")
    
    # Hard Fail Conditions
    if n_fin < 100:
        print("FAIL: Too few finite samples (<100).")
        sys.exit(1)
        
    if frac_fin_t < 0.95 or frac_fin_s < 0.95:
        print("FAIL: Data contains excessive NaNs (>5%).")
        sys.exit(1)
        
    # 4. Monotonicity check (on finite timestamps)
    t_fin = t[mask_fin]
    if np.any(np.diff(t_fin) <= 0):
        print("FAIL: Timestamps not strictly increasing.")
        sys.exit(1)
        
    return t, s

def compute_tonic_phasic(time, signal, fs, percentile, phasic_hp_hz):
    """
    Computes tonic (slow) and phasic (fast) components.
    
    Tonic: Rolling percentile over large window.
    Phasic: Signal - Tonic, optionally high-passed.
    """
    # 1. Define Window
    min_window_sec = 600.0 # 10 minutes default preference
    period_window_sec = 10.0 / phasic_hp_hz if phasic_hp_hz > 1e-9 else 600.0
    
    # Initial target window
    target_window_sec = max(min_window_sec, period_window_sec)
    target_window_samples = int(target_window_sec * fs)
    
    # Clamp window to data length (robustness for short recordings)
    n_finite = np.isfinite(signal).sum()
    max_allowed = max(51, int(n_finite // 2))
    
    # Force odd
    if max_allowed % 2 == 0: max_allowed -= 1
    
    window_samples = min(target_window_samples, max_allowed)
    # Ensure at least minimal window
    if window_samples < 51:
        print(f"FAIL: Available data ({n_finite} samples) too short for tonic analysis.")
        sys.exit(1)
        
    if window_samples % 2 == 0: window_samples += 1
    
    real_window_sec = window_samples / fs
    print(f"Computing Tonic: TargetWindow={target_window_sec:.1f}s ActualWindow={real_window_sec:.1f}s ({window_samples} samples), Percentile={percentile}")
    
    # 2. Compute Tonic (Rolling Percentile)
    # Convert to Series for rolling
    sig_series = pd.Series(signal)
    
    min_periods = max(10, window_samples // 5)
    
    tonic_series = sig_series.rolling(
        window=window_samples, 
        center=True, 
        min_periods=min_periods
    ).quantile(percentile / 100.0)
    
    # Fill ends if needed (propagate)
    tonic_series = tonic_series.bfill().ffill()
    
    tonic_arr = tonic_series.to_numpy()
    
    # Check tonic validity
    if np.sum(np.isfinite(tonic_arr)) == 0:
        print("FAIL: tonic trend is all NaN, check trace content or rolling window/min_periods.")
        sys.exit(1)
    
    # 3. Compute Phasic
    phasic_raw = signal - tonic_arr
    
    # 4. Filter Phasic (Highpass) to remove residual drift
    if phasic_hp_hz > 0:
        # Handle NaNs before filtering if any remain? 
        # sosfiltfilt can handle some, but ideally we interpolate or just accept edge artifacts.
        # For simplicity in this tool, we'll strip NaNs for filtering or just run it (scipy >= 1.4 handles NaNs badly usually)
        # We will mask NaNs -> 0 for filtering? No, linearly interpolate is better.
        # Minimal approach: fillna with 0 for filtering only
        mask = np.isnan(phasic_raw)
        phasic_filled = phasic_raw.copy()
        phasic_filled[mask] = 0.0 # Zero pad for filtering
        
        sos = scipy.signal.butter(1, phasic_hp_hz, 'highpass', fs=fs, output='sos')
        phasic_filtered = scipy.signal.sosfiltfilt(sos, phasic_filled)
        
        # Restore NaNs
        phasic_filtered[mask] = np.nan
    else:
        phasic_filtered = phasic_raw
        
    return tonic_arr, phasic_filtered, real_window_sec

def make_plots(time_arr, signal, tonic, phasic, out_dir, roi, metric, params):
    os.makedirs(out_dir, exist_ok=True)
    
    # Sanity check for plotting
    mask_tonic_plot = np.isfinite(time_arr) & np.isfinite(signal) & np.isfinite(tonic)
    if np.sum(mask_tonic_plot) < 100:
        print(f"SKIP: Too few points to plot Tonic for {roi}")
        return

    t_hours = time_arr / 3600.0
    
    # TONIC PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(t_hours[mask_tonic_plot], signal[mask_tonic_plot], color='lightgray', label='Corrected Trace', alpha=0.7)
    plt.plot(t_hours[mask_tonic_plot], tonic[mask_tonic_plot], color='blue', label='Tonic Trend', linewidth=2)
    plt.title(f"Tonic Component: {roi} (24h)\n(p{params['tonic_percentile']}, window {params['tonic_window_min']:.1f}m)")
    plt.xlabel("Time (hours)")
    plt.ylabel(f"Signal ({metric})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"tonic_{roi}.png"))
    plt.close()
    
    # PHASIC PLOT
    mask_phasic_plot = np.isfinite(time_arr) & np.isfinite(phasic)
    if np.sum(mask_phasic_plot) < 100:
        print(f"SKIP: Too few points to plot Phasic for {roi}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(t_hours[mask_phasic_plot], phasic[mask_phasic_plot], color='black', linewidth=1)
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)
    plt.title(f"Phasic Component: {roi}\n(HP > {params['phasic_hp_hz']} Hz)")
    plt.xlabel("Time (hours)")
    plt.ylabel(f"Phasic ({metric})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"phasic_{roi}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run Pipeline & Generate Tonic/Phasic Plots")
    
    # Pipeline Args
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--format', default='auto')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--file-glob', default='*.csv')
    
    # Plotting Args
    parser.add_argument('--roi', help="ROI to plot (default: first found)")
    parser.add_argument('--segment-minutes', type=int, default=60, help="Unused logic anchor")
    parser.add_argument('--tonic-percentile', type=float, default=2.0)
    parser.add_argument('--phasic-highpass-hz', type=float, default=0.01)
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Execute Pipeline
    proc_config = run_pipeline(args)
    
    # Process Outputs
    traces_dir = os.path.join(args.out, 'traces')
    time_series, signal_series, roi_name, metric_name = load_concatenated_traces(traces_dir, proc_config, args.roi)
    
    # Validate
    validate_trace_arrays(time_series, signal_series, roi_name, metric_name)
    
    # Calculate components
    fs = proc_config.target_fs_hz
    tonic, phasic, win_sec = compute_tonic_phasic(
        time_series, signal_series, fs, 
        args.tonic_percentile, args.phasic_highpass_hz
    )
    
    # Output
    plots_dir = os.path.join(args.out, 'paper_plots')
    
    params = {
        'roi': roi_name,
        'fs_hz': fs,
        'signal_metric': metric_name,
        'tonic_percentile': args.tonic_percentile,
        'tonic_window_sec': win_sec,
        'tonic_window_min': win_sec / 60.0,
        'phasic_hp_hz': args.phasic_highpass_hz,
        'input_glob': args.file_glob
    }
    
    make_plots(time_series, signal_series, tonic, phasic, plots_dir, roi_name, metric_name, params)
    
    # Save Params
    with open(os.path.join(plots_dir, f"plot_params_{roi_name}.json"), 'w') as f:
        json.dump(params, f, indent=2)
        
    print(f"Done. Outputs in {plots_dir}")

if __name__ == "__main__":
    main()
