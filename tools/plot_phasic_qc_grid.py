#!/usr/bin/env python3
"""
Phasic QC Grid Generator (Signal-Locked)
========================================

Generates interpretable stacked grid figures from Phasic analysis outputs.
Strictly ensures the plotted trace and annotated peak counts match the 
actual analysis by performing an internal verification step using the
pipeline's core feature extraction logic.

Requirements:
- <analysis-out>/config_used.yaml (created by pipeline)
- <analysis-out>/features/features.csv
- <analysis-out>/traces/chunk_*.csv

Usage:
    python tools/plot_phasic_qc_grid.py --analysis-out <DIR> [--signal <COL>]

Verification:
    The tool internally constructs a Chunk and calls extract_features.
    It fails if the computed peak count differs from features.csv.
"""

import os
import sys
import argparse
import glob
import re
import math
import logging
import yaml
import json
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Ensure repo root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Core Imports
from photometry_pipeline.config import Config
from photometry_pipeline.core.feature_extraction import extract_features
from photometry_pipeline.core.types import Chunk

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Phasic QC Grid Plots (Signal-Locked)")
    parser.add_argument('--analysis-out', required=True, help="Path to analysis output directory")
    parser.add_argument('--roi', default=None, help="Specific ROI to plot (default: first found)")
    parser.add_argument('--signal', default='auto', help="Trace column to plot/verify (default: auto detects *_dff)")
    parser.add_argument('--output-dir', default=None, help="Override output directory")
    parser.add_argument('--dpi', type=int, default=150, help="Output DPI")
    parser.add_argument('--sessions-per-hour', type=int, default=None, help="Force grid columns")
    parser.add_argument('--mode', choices=['dff', 'raw'], default='dff', help="Plot mode: dff (default) or raw (sig+iso)")
    
    # Y-Limit Tuning
    parser.add_argument('--dff-y-percentile-low', type=float, default=0.5, help="Global Y-min percentile for DFF")
    parser.add_argument('--dff-y-percentile-high', type=float, default=99.9, help="Global Y-max percentile for DFF")
    parser.add_argument('--dff-y-pad-frac', type=float, default=0.10, help="Y-axis padding fraction")
    
    return parser.parse_args()

def load_config_obj(out_dir):
    path = os.path.join(out_dir, "config_used.yaml")
    if not os.path.exists(path):
        print(f"CRITICAL: config_used.yaml not found in {out_dir}. Pipeline must complete successfully first.")
        sys.exit(1)
    try:
        return Config.from_yaml(path)
    except Exception as e:
        print(f"CRITICAL: Failed to load config from {path}: {e}")
        sys.exit(1)

def infer_datetime_from_string(s):
    if not isinstance(s, str): return None
    patterns = [
        r'(\d{4})[-_](\d{2})[-_](\d{2})[-_](\d{2})[_:](\d{2})[_:](\d{2})',
        r'(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})',
        r'(\d{4})[-_](\d{2})[-_](\d{2})\s+(\d{2})[:](\d{2})[:](\d{2})'
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            try:
                parts = list(map(int, m.groups()))
                return datetime(*parts)
            except ValueError:
                continue
    return None

def determine_signal_column(trace_path, roi, requested='auto'):
    """
    Identifies the correct column for PEAK_INPUT_SIGNAL.
    """
    df = pd.read_csv(trace_path, nrows=1)
    cols = df.columns.tolist()
    
    if requested != 'auto':
        if requested in cols:
            return requested
        else:
            print(f"CRITICAL: Requested signal '{requested}' not found in {trace_path}")
            sys.exit(1)
    
    cand = f"{roi}_dff"
    if cand in cols:
        return cand
    
    match = [c for c in cols if roi in c and c.endswith('_dff')]
    if match:
        return match[0]
        
    print(f"CRITICAL: Could not auto-detect signal for ROI {roi} in {trace_path}. Available: {cols}")
    sys.exit(1)

def infer_fs(time_arr, config, context=""):
    """
    Infers sampling rate from time array.
    """
    if len(time_arr) < 2:
        fallback = getattr(config, 'sampling_rate_hz_fallback', config.target_fs_hz)
        logging.warning(f"{context}: Time array too short to infer fs. Using fallback {fallback}")
        return fallback

    dt = np.median(np.diff(time_arr))
    if dt <= 0 or not np.isfinite(dt):
        fallback = getattr(config, 'sampling_rate_hz_fallback', config.target_fs_hz)
        logging.warning(f"{context}: Invalid dt ({dt}). Using fallback {fallback}")
        return fallback
        
    return 1.0 / dt

def get_local_peak_indices(trace_arr, fs, config):
    """
    Mirrors feature_extraction detection logic STRICTLY to return indices for plotting.
    Note: verify_peak_count_strict uses the actual pipeline to verify the COUNT,
    but we still need this to know WHERE to plot points (since pipelines returns stats only).
    """
    is_valid = np.isfinite(trace_arr)
    clean_trace = trace_arr[is_valid]
    
    if len(clean_trace) == 0:
        return np.array([], dtype=int)
        
    # Threshold Calculation
    method = config.peak_threshold_method
    thresh = np.inf
    
    # Calculate stats on global clean trace
    if method == 'mean_std':
        mu = np.mean(clean_trace)
        sigma = np.std(clean_trace)
        thresh = mu + config.peak_threshold_k * sigma
        sigma_robust = sigma # For prominence if needed (fallback)
        
    elif method == 'percentile':
        thresh = np.percentile(clean_trace, config.peak_threshold_percentile)
        sigma_robust = 0 # Not used for prominence usually unless mixed
        
    elif method == 'median_mad':
        median = np.median(clean_trace)
        # MAD STRICT: median(abs(x - median))
        mad = np.median(np.abs(clean_trace - median))
        
        # Consistent with feature_extraction.py: sigma_robust = 1.4826 * mad
        sigma_robust = 1.4826 * mad
        
        if sigma_robust == 0:
             if config.peak_threshold_k == 0:
                 thresh = median
             else:
                 thresh = float('inf') 
        else:
            thresh = median + config.peak_threshold_k * sigma_robust

    # Constraints
    dist_samples = max(1, int(config.peak_min_distance_sec * fs))
    

    # Segmented Detection
    padded = np.concatenate(([False], is_valid, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    all_peaks = []
    for s, e in zip(starts, ends):
        seg_trace = trace_arr[s:e]
        
        seg_for_peaks = seg_trace
        
        p_kwargs = {'height': thresh, 'distance': dist_samples}
        
        p_inds, _ = find_peaks(seg_for_peaks, **p_kwargs)
        all_peaks.append(p_inds + s)
        
    if all_peaks:
        return np.concatenate(all_peaks)
    else:
        return np.array([], dtype=int)

def verify_peak_count_strict(trace_arr, time_arr, fs, config, expected_count, roi, cid, src_file):
    """
    Constructs a Chunk and calls the ACTUAL pipeline logic to verify peak count.
    Also checks if local plotting logic yields the same indices.
    """
    if pd.isna(expected_count):
        print(f"CRITICAL: Expected count is NaN for Chunk {cid}. Verification failed (pipeline did not produce count).")
        sys.exit(1)

    # 1. Pipeline Logic Verification
    dff_in = trace_arr.reshape(-1, 1)
    
    # Dummy raw
    raw = np.zeros_like(dff_in)
    
    chunk = Chunk(
        chunk_id=cid,
        source_file=src_file,
        format='rwd', # Placeholder
        time_sec=time_arr,
        dff=dff_in,
        uv_raw=raw,
        sig_raw=raw,
        fs_hz=fs,
        channel_names=[roi]
    )
    
    # Call Pipeline Feature Extraction
    df_feat = extract_features(chunk, config)
    
    if df_feat.empty:
        print(f"CRITICAL: extract_features returned empty for Chunk {cid}")
        sys.exit(1)
        
    pipeline_count = df_feat.iloc[0]['peak_count']
    
    if pipeline_count != expected_count:
        print(f"CRITICAL: Verification Failed for Chunk {cid}, ROI {roi}")
        print(f"  Expected (CSV): {expected_count}")
        print(f"  Computed (Pipeline): {pipeline_count}")
        safe_filter = getattr(config, 'peak_pre_filter', 'none')
        print(f"  Config: method={config.peak_threshold_method}, k={config.peak_threshold_k}, filter={safe_filter}")
        sys.exit(1)

    # 2. Local Plotting Indices Verification
    # We need to know where to plot them. We run the local helper.
    local_peaks = get_local_peak_indices(trace_arr, fs, config)
    local_count = len(local_peaks)
    
    if local_count != pipeline_count:
        print(f"CRITICAL: Plotting Logic Mismatch for Chunk {cid}, ROI {roi}")
        print(f"  Pipeline Found: {pipeline_count}")
        print(f"  Plotter Found (Local): {local_count}")
        print("  The local re-implementation of peak detection in this script is out of sync with feature_extraction.py")
        sys.exit(1)
        
    return local_peaks

def main():
    print("Running tools/plot_phasic_qc_grid.py (FIXED)")
    args = parse_args()
    
    # 1. Load Config Object (Strict)
    config = load_config_obj(args.analysis_out)
    
    feats_path = os.path.join(args.analysis_out, 'features', 'features.csv')
    if not os.path.exists(feats_path):
        print("CRITICAL: features.csv missing.")
        sys.exit(1)
    df_feat = pd.read_csv(feats_path)
    
    traces_dir = os.path.join(args.analysis_out, 'traces')
    trace_files = sorted(glob.glob(os.path.join(traces_dir, 'chunk_*.csv')))
    if not trace_files:
        print("CRITICAL: No trace files found.")
        sys.exit(1)
        
    # 2. Select ROI
    first_trace = pd.read_csv(trace_files[0], nrows=1)
    dff_cols = [c for c in first_trace.columns if '_dff' in c]
    available_rois = [c.replace('_dff', '') for c in dff_cols]
    
    if args.roi:
        if args.roi not in available_rois:
            if args.roi not in df_feat['roi'].unique():
                print(f"CRITICAL: ROI '{args.roi}' not found.")
                sys.exit(1)
        plot_roi = args.roi
    else:
        if not available_rois:
            print("CRITICAL: No _dff columns found. Cannot auto-select ROI.")
            sys.exit(1)
        plot_roi = sorted(available_rois)[0]
        print(f"Auto-selected ROI: {plot_roi}")

    # 3. Build Grid Mapping
    grid_rows = []
    feat_map = {}
    for _, row in df_feat.iterrows():
        feat_map[(row['chunk_id'], row['roi'])] = row
        
    for tpath in trace_files:
        fname = os.path.basename(tpath)
        m = re.search(r'chunk_(\d+)\.csv', fname)
        if not m: continue
        cid = int(m.group(1))
        
        dt = None
        if (cid, plot_roi) in feat_map:
            src = feat_map[(cid, plot_roi)].get('source_file', '')
            dt = infer_datetime_from_string(src)
        else:
            src = tpath # Fallback

        grid_rows.append({
            'chunk_id': cid,
            'trace_path': tpath,
            'datetime': dt,
            'source_file': src
        })
        
    df_grid = pd.DataFrame(grid_rows)
    
    # 4. Infer Layout
    mapped = df_grid['datetime'].notnull()
    pct_mapped = mapped.mean() * 100
    sessions_ph = args.sessions_per_hour
    
    if pct_mapped > 90 and sessions_ph is None:
        t0 = df_grid['datetime'].min()
        day_start = t0.replace(hour=0, minute=0, second=0, microsecond=0)
        df_grid['elapsed'] = (df_grid['datetime'] - day_start).dt.total_seconds()
        df_grid['day_idx'] = (df_grid['elapsed'] // 86400).astype(int)
        df_grid['hour_idx'] = ((df_grid['elapsed'] % 86400) // 3600).astype(int)
        df_grid = df_grid.sort_values(['day_idx', 'hour_idx', 'datetime'])
        df_grid['hour_rank'] = df_grid.groupby(['day_idx', 'hour_idx']).cumcount()
        modes = df_grid.groupby(['day_idx', 'hour_idx']).size()
        sessions_ph = int(modes.mode()[0]) if not modes.empty else 1
    else:
        if sessions_ph is None:
            n_chunks = len(df_grid)
            n_days_est = max(1, math.ceil(n_chunks / 48)) 
            sph_est = max(1, round(n_chunks / (24 * n_days_est)))
            sessions_ph = sph_est
            print(f"Fallback: Inferred {sessions_ph} sessions/hour from count.")
            
        df_grid = df_grid.sort_values('chunk_id')
        df_grid['day_idx'] = df_grid.index // (24 * (sessions_ph or 1))
        df_grid['hour_idx'] = (df_grid.index // (sessions_ph or 1)) % 24
        df_grid['hour_rank'] = df_grid.index % (sessions_ph or 1)
        
    sessions_ph = int(max(1, sessions_ph))
    print(f"Layout: {sessions_ph} columns.")

    # 5. Iteration & Signal Loading & Verification
    all_traces = []
    
    print("Verifying peaks and loading traces...")
    
    plot_data = [] 
    
    for _, row in df_grid.iterrows():
        cid = row['chunk_id']
        tpath = row['trace_path']
        
        try:
            tdf = pd.read_csv(tpath)
            
            # X axis & FS Inference
            x = None
            if 'time_sec' in tdf.columns:
                x = tdf['time_sec'].values
            elif getattr(config, 'rwd_time_col', '') in tdf.columns:
                x = tdf[config.rwd_time_col].values
            elif 'Time(s)' in tdf.columns:
                x = tdf['Time(s)'].values
            
            # Infer FS
            if x is not None and len(x) > 1:
                fs = infer_fs(x, config, context=f"Chunk {cid}")
            else:
                fs = config.target_fs_hz
                if x is None:
                    x = np.arange(len(tdf)) / fs
            
            x = x - x[0] # Normalize
            
            if args.mode == 'dff':
                col = determine_signal_column(tpath, plot_roi, args.signal)
                y = tdf[col].values
                uv = None
                exp_count = np.nan
                
                feat_row = feat_map.get((cid, plot_roi))
                if feat_row is not None:
                    exp_count = feat_row['peak_count']
                
                # Strict Verification using Pipeline
                indices = verify_peak_count_strict(
                    y, x, fs, config, 
                    exp_count, plot_roi, cid, row.get('source_file', tpath)
                )
                
            elif args.mode == 'raw':
                col_sig = f"{plot_roi}_sig_raw"
                col_uv = f"{plot_roi}_uv_raw"
                if col_sig not in tdf.columns or col_uv not in tdf.columns:
                    print(f"Skipping {cid}: Raw columns missing")
                    continue
                y = tdf[col_sig].values
                uv = tdf[col_uv].values
                indices = []
                exp_count = np.nan
            
            all_traces.append(y)
            if uv is not None:
                all_traces.append(uv)
            
            plot_data.append({
                'day': row['day_idx'],
                'hour': row['hour_idx'],
                'col': row['hour_rank'],
                'x': x,
                'y': y,
                'uv': uv,
                'peak_indices': indices,
                'count': exp_count,
                'chunk_id': cid 
            })
            
        except Exception as e:
            print(f"Error processing chunk {cid}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # 6. Global Y Limits (Percentile-based)
    if not all_traces:
        print("No trace data loaded.")
        sys.exit(1)
        
    flat_y = np.concatenate([t[np.isfinite(t)] for t in all_traces])
    if len(flat_y) == 0:
        global_ymin, global_ymax = -1, 1
    else:
        p_lo = args.dff_y_percentile_low
        p_hi = args.dff_y_percentile_high
        global_ymin, global_ymax = np.percentile(flat_y, [p_lo, p_hi])
        
        yrange = global_ymax - global_ymin
        pad = args.dff_y_pad_frac * yrange
        if pad == 0: pad = 0.1
        global_ymin -= pad
        global_ymax += pad
        
    print(f"Global (DFF-based) Y-Limits: [{global_ymin:.3f}, {global_ymax:.3f}]")

    # 7. Plotting
    output_dir = args.output_dir or os.path.join(args.analysis_out, 'phasic_qc')
    os.makedirs(output_dir, exist_ok=True)
    
    unique_days = sorted(df_grid['day_idx'].unique())
    cols = sessions_ph
    rows = 24 
    figsize_width = 4*sessions_ph + 2
    figsize_height = 24

    for d in unique_days:
        day_items = [p for p in plot_data if p['day'] == d]
        n_plots = len(day_items)
        if n_plots == 0: continue
            
        # Limits
        if args.mode == 'dff':
             day_ymin, day_ymax = global_ymin, global_ymax
        elif args.mode == 'raw':
             # Per-day limits for raw
             day_values = []
             for p in day_items:
                 if p['y'] is not None: day_values.append(p['y'][np.isfinite(p['y'])])
                 if p['uv'] is not None: day_values.append(p['uv'][np.isfinite(p['uv'])])
             
             if not day_values:
                 day_ymin, day_ymax = -1, 1
             else:
                 all_day_v = np.concatenate(day_values)
                 if len(all_day_v) == 0:
                     day_ymin, day_ymax = -1, 1
                 else:
                     raw_min, raw_max = np.min(all_day_v), np.max(all_day_v)
                     pad = 0.05 * (raw_max - raw_min)
                     day_ymin = raw_min - pad
                     day_ymax = raw_max + pad

        fig, axes = plt.subplots(nrows=rows, ncols=cols, 
                                 figsize=(figsize_width, figsize_height),
                                 sharex=True, sharey=False)

        if rows == 1 and cols == 1: axes = np.array([[axes]])
        elif rows == 1 or cols == 1: axes = axes.reshape(rows, cols)
            
        fig.suptitle(f"Phasic QC - Day {d} - ROI {plot_roi} - Mode: {args.mode.upper()}", fontsize=16)
        
        # Calculate visualization eps based on DAY limits
        y_span = day_ymax - day_ymin
        eps = 0.01 * y_span if y_span > 0 else 1e-6
        
        for i, p in enumerate(day_items):
            h, c = p['hour'], p['col']
            if c >= cols: continue
            ax = axes[h, c]
            
            # Subplot Chunk Label (Restored)
            ax.set_title(f"Chunk {p['chunk_id']}", fontsize=6, pad=2)
            
            # Plot Trace
            if args.mode == 'dff':
                ax.plot(p['x'], p['y'], 'k', lw=0.8)
                ax.set_ylim(day_ymin, day_ymax)
                
                # Clipping Aware Peak Plotting (uses GLOBAL/DAY limits from plotting)
                p_idxs = p['peak_indices']
                if len(p_idxs) > 0:
                    px = p['x'][p_idxs]
                    py_true = p['y'][p_idxs]
                    
                    # Clip Y
                    py_plot = np.clip(py_true, day_ymin + eps, day_ymax - eps)
                    
                    # Identify clipped
                    mask_hi = py_true > (day_ymax - eps)
                    mask_lo = py_true < (day_ymin + eps)
                    mask_ok = ~(mask_hi | mask_lo)
                    
                    # Plot Normal
                    if np.any(mask_ok):
                        ax.scatter(px[mask_ok], py_plot[mask_ok], s=10, c='red', alpha=0.6, zorder=3)
                        
                    # Plot Clipped High
                    if np.any(mask_hi):
                        ax.scatter(px[mask_hi], py_plot[mask_hi], s=12, marker='^', c='red', alpha=0.8, zorder=4)
                        
                    # Plot Clipped Low
                    if np.any(mask_lo):
                        ax.scatter(px[mask_lo], py_plot[mask_lo], s=12, marker='v', c='red', alpha=0.8, zorder=4)
                        
                    n_clipped = np.sum(mask_hi) + np.sum(mask_lo)
                else:
                    n_clipped = 0
                
                # Annotation
                val = p['count']
                if pd.isna(val):
                    txt = "peaks=NaN"
                    color = 'red'
                else:
                    txt = f"peaks={int(val)}"
                    if n_clipped > 0:
                        txt += f"\n({n_clipped} clipped)"
                    color = 'blue'
                    
                ax.text(0.95, 0.9, txt, transform=ax.transAxes, 
                        ha='right', va='top', fontsize=8, color=color, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                # Label only on leftmost column?
                if c == 0:
                    ax.set_ylabel(f"H{h:02d}", rotation=0, labelpad=15, va='center', fontweight='bold')
                
                # Remove generic ylabel to avoid clutter
                # ax.set_ylabel("dFF") 

            elif args.mode == 'raw':
                ax.plot(p['x'], p['y'], color='green', lw=0.8, label='Sig')
                if p['uv'] is not None:
                    ax.plot(p['x'], p['uv'], color='purple', lw=0.8, alpha=0.7, label='Iso')
                ax.set_ylim(day_ymin, day_ymax)
                if c == 0:
                    ax.set_ylabel(f"H{h:02d}", rotation=0, labelpad=15, va='center', fontweight='bold')

        # Cleanup Empty
        for r in range(rows):
            for c in range(cols):
                if not any(pi['hour'] == r and pi['col'] == c for pi in day_items):
                    axes[r, c].axis('off')
                    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Output Naming Handling
        if args.mode == 'dff':
            out_name = f"day_{d:03d}.png"
        else: # raw
            out_name = f"day_{d:03d}_raw.png"
            
        out_path = os.path.join(output_dir, out_name)
        plt.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {out_path}")
        
        # Conditional Copy for Raw Day 0
        if args.mode == 'raw' and d == 0:
            copy_path = os.path.join(output_dir, "fig_phasic_raw_qc_grid.png")
            if os.path.exists(out_path):
                shutil.copy2(out_path, copy_path)
                print(f"Copied to {copy_path}")
            else:
                raise RuntimeError(f"Cannot copy Day 0 plot: {out_path} not generated")

if __name__ == "__main__":
    main()
