
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure we can import from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from photometry_pipeline.core.tonic_dff import (
    compute_global_iso_fit, 
    compute_global_iso_fit_robust,
    apply_global_fit, 
    compute_session_tonic_df_from_global
)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze and Visualize Tonic dFF from Pipeline Outputs (Global Fit)")
    parser.add_argument('--input-dir', required=True, help="Pipeline output directory (containing 'traces' subdir)")
    parser.add_argument('--output-dir', required=True, help="Where to save tonic analysis outputs")
    parser.add_argument('--roi', default='Region0', help="ROI to analyze (default Region0)")
    parser.add_argument('--representative-idx', type=int, default=None, help="Explicit index for representative plot (default middle)")
    return parser.parse_args()

def load_all_traces(trace_files, roi):
    """
    Loads raw data from all trace files.
    trace_files: list of pathlib.Path objects
    Returns list of dicts with keys: 't', 'uv', 'sig', 'fpath' (Path object)
    """
    loaded_sessions = []
    
    for i, fpath in enumerate(trace_files):
        try:
            df = pd.read_csv(fpath)
            uv_col = f"{roi}_uv_raw"
            sig_col = f"{roi}_sig_raw"
            
            if uv_col not in df.columns or sig_col not in df.columns:
                print(f"Warning: Missing columns for {roi} in {fpath.name}. Skipping.")
                continue
                
            session_data = {
                't': df['time_sec'].values,
                'uv': df[uv_col].values,
                'sig': df[sig_col].values,
                'fpath': fpath,
                'idx': i
            }
            loaded_sessions.append(session_data)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, KeyError, ValueError) as e:
            print(f"Error loading {fpath}: {e}")
            
    return loaded_sessions

def analyze_dataset_global(sessions, roi, rep_idx_arg=None):
    """
    Performs 2-Pass Analysis:
    1. Concatenate all sessions -> Compute Global Fit
    2. Apply Global Fit to each session -> Compute dFF & Scalar
    
    Returns: (summary_df, representative_data, slope, intercept, n_fit)
    """
    if not sessions:
        return pd.DataFrame(), None, np.nan, np.nan, 0

    # Pass 1: Global Fit (Robust)
    all_uv = np.concatenate([s['uv'] for s in sessions])
    all_sig = np.concatenate([s['sig'] for s in sessions])
    
    # Calculate fit sample count (finite)
    mask = np.isfinite(all_uv) & np.isfinite(all_sig)
    n_fit = np.sum(mask)
    
    print(f"Computing Robust Global Fit (N={n_fit})...")
    slope, intercept, ok, n_used = compute_global_iso_fit_robust(all_uv, all_sig)
    
    if not ok:
        print("Global fit failed (insufficient data or variance).")
        return pd.DataFrame(), None, np.nan, np.nan, n_used
    
    print(f"Global Fit OK: Slope={slope:.4f}, Intercept={intercept:.4f}, Used={n_used}")
        
    # Pass 2: Per-Session dFF
    summary_data = []
    
    for s in sessions:
        idx = s['idx']
        t = s['t']
        uv = s['uv']
        sig = s['sig']
        
        # Apply Fit (Raw)
        iso_fit = apply_global_fit(uv, slope, intercept)
        res = compute_session_tonic_df_from_global(sig, uv, iso_fit, percentile=2.0)
        
        scalar = res['tonic_scalar']
        
        # Record Summary
        summary_data.append({
            'session_idx': idx,
            'time_start': t[0] if len(t)>0 else np.nan,
            'tonic_scalar': scalar,
            'global_slope': slope,
            'global_intercept': intercept
        })
        
    # Representative Selection Logic
    rep_data = None
    rep_session = None
    
    if rep_idx_arg is not None:
        # User requested specific original index
        found = [s for s in sessions if s['idx'] == rep_idx_arg]
        if found:
            rep_session = found[0]
        else:
            print(f"Warning: Requested representative session index {rep_idx_arg} not in loaded sessions. Falling back to middle.")
    
    if rep_session is None:
        # Default: Middle by position
        mid_pos = len(sessions) // 2
        rep_session = sessions[mid_pos]
        
    # Construct rep_data
    if rep_session:
        uv = rep_session['uv']
        sig = rep_session['sig']
        # Apply Fit (Raw)
        iso_fit = apply_global_fit(uv, slope, intercept)
        res = compute_session_tonic_df_from_global(sig, uv, iso_fit, percentile=2.0)
        rep_data = {
            'idx': rep_session['idx'],
            't': rep_session['t'],
            'uv': uv,
            'sig': sig,
            'iso_fit': iso_fit,
            'df': res['df'],
            'df_valid': res['valid_mask'],
            'scalar': res['tonic_scalar']
        }
            
    return pd.DataFrame(summary_data), rep_data, slope, intercept, n_fit

def plot_representative_session(rep_data, roi, out_dir):
    if rep_data is None:
        return

    idx = rep_data['idx']
    t = rep_data['t']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Raw Signals
    ax = axes[0]
    ax.plot(t, rep_data['sig'], color='green', label='Signal (470)', alpha=0.8)
    ax.plot(t, rep_data['uv'], color='purple', label='Isosbestic (410)', alpha=0.8)
    ax.set_title(f"Session {idx}: Raw Signals")
    ax.set_ylabel("Fluorescence (AU)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. Global Fit vs Signal
    ax = axes[1]
    ax.plot(t, rep_data['sig'], color='green', label='Signal', alpha=0.6)
    ax.plot(t, rep_data['iso_fit'], color='k', linestyle='--', label='Global Iso Fit', alpha=0.8)
    ax.set_title(f"Session {idx}: Signal vs Global Fit")
    ax.set_ylabel("Fluorescence (AU)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 3. Tonic df
    ax = axes[2]
    df_plot = rep_data['df'].copy()
    
    ax.plot(t, df_plot, color='blue', label='Tonic df (AU)')
    
    scalar = rep_data['scalar']
    if np.isfinite(scalar):
         ax.axhline(scalar, color='red', linestyle=':', linewidth=2, label=f'2nd %ile ({scalar:.2f} AU)')
         
    ax.set_title(f"Session {idx}: Tonic df (Global Fit)")
    ax.set_ylabel("df (AU)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = Path(out_dir) / f"{roi}_session_{idx}_tonic_dff.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved representative plot to {path}")

def plot_summary(df, roi, out_dir):
    if df.empty:
        return
        
    valid = df.dropna(subset=['tonic_scalar'])
    if valid.empty:
        return
        
    plt.figure(figsize=(12, 5))
    
    t_hr = valid['time_start'] / 3600.0
    y = valid['tonic_scalar']
    plt.plot(t_hr, y, marker='o', linestyle='-', markersize=4, alpha=0.7, color='black')
    plt.xlabel("Time (Hours)")
    plt.ylabel("Tonic df Baseline (AU)")
    plt.title(f"{roi}: Tonic df Summary (Global Fit)")
    plt.grid(True, alpha=0.3)
    
    path = Path(out_dir) / f"{roi}_tonic_scalar_summary.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved summary plot to {path}")

def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    traces_dir = input_dir / 'traces'
    if not traces_dir.is_dir():
        if input_dir.name == 'traces':
             traces_dir = input_dir
        else:
             print(f"Error: Could not find 'traces' subdirectory in {input_dir}")
             return
             
    # Discover CSVs
    files = sorted(list(traces_dir.glob("*.csv")))
    n_found = len(files)
    if not files:
        print(f"No CSV trace files found in {traces_dir}")
        return
        
    # Analyze
    viz_dir = output_dir / 'tonic_dff_viz' / args.roi
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load
    sessions = load_all_traces(files, args.roi)
    n_loaded = len(sessions)
    
    # 2. Global Analysis and Diagnostics
    df, rep_data, slope, intercept, n_fit = analyze_dataset_global(sessions, args.roi, args.representative_idx)
    
    if df.empty:
        print("Analysis produced no data.")
        return

    # Diagnostics
    print(f"Diagnostics:")
    print(f"  Files Found: {n_found}")
    print(f"  Sessions Loaded: {n_loaded}")
    print(f"  Global Fit Slope: {slope:.4f}")
    print(f"  Global Fit Intercept: {intercept:.4f}")
    print(f"  Global Fit Samples: {n_fit}")

    # Save CSV
    csv_path = viz_dir / f"{args.roi}_tonic_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved stats to {csv_path}")
    
    # Visuals
    plot_representative_session(rep_data, args.roi, viz_dir)
    plot_summary(df, args.roi, viz_dir)
    
    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        print(f"FAILED: {e}")
