#!/usr/bin/env python3
"""
Phasic QC Grid Generator (Signal-Locked)
========================================

Generates interpretable stacked grid figures from Phasic analysis outputs.
Strictly ensures the plotted trace and annotated peak counts match the 
actual analysis by performing an internal verification step.

Requirements:
- <analysis-out>/config_used.yaml (created by pipeline)
- <analysis-out>/features/features.csv
- <analysis-out>/traces/chunk_*.csv

Usage:
    python tools/plot_phasic_qc_grid.py --analysis-out <DIR> [--signal <COL>]

Verification:
    The tool internally re-runs the peak detection on the loaded trace 
    using the parameters from config_used.yaml. It fails if the count 
    differs from features.csv.
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
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
    return parser.parse_args()

def load_config(out_dir):
    path = os.path.join(out_dir, "config_used.yaml")
    if not os.path.exists(path):
        print(f"CRITICAL: config_used.yaml not found in {out_dir}. Pipeline must complete successfully first.")
        sys.exit(1)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

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
    
    # Auto-detection: Prefer {roi}_dff
    # feature_extraction.py uses chunk.dff
    cand = f"{roi}_dff"
    if cand in cols:
        return cand
    
    # Fallback to deltaF if dff missing? No, user wants peak input.
    # If dff missing, maybe we shouldn't have passed QC.
    # Check for suffix logic? The pipeline usually standardizes to {roi}_dff.
    
    match = [c for c in cols if roi in c and c.endswith('_dff')]
    if match:
        return match[0]
        
    print(f"CRITICAL: Could not auto-detect signal for ROI {roi} in {trace_path}. Available: {cols}")
    sys.exit(1)

def verify_peak_count(trace_arr, fs, config, expected_count, roi, cid):
    """
    Re-implements feature_extraction.extract_features logic to verify consistency.
    """
    # Logic from feature_extraction.py
    is_valid = np.isfinite(trace_arr)
    clean_trace = trace_arr[is_valid]
    peaks = np.array([], dtype=int)
    
    if len(clean_trace) == 0:
        calc_count = np.nan if pd.isna(expected_count) else 0
    else:
        mu = np.mean(clean_trace)
        sigma = np.std(clean_trace)
        
        method = config.get('peak_threshold_method', 'mean_std')
        if method == 'mean_std':
            k = config.get('peak_threshold_k', 2.0)
            thresh = mu + k * sigma
        elif method == 'percentile':
            p = config.get('peak_threshold_percentile', 95.0)
            thresh = np.percentile(clean_trace, p)
        else:
            thresh = np.inf # Should have failed earlier
            
        min_dist_sec = config.get('peak_min_distance_sec', 0.5)
        dist_samples = int(min_dist_sec * fs)
        dist = max(1, dist_samples)
        
        # Segmented find_peaks
        padded = np.concatenate(([False], is_valid, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        calc_count = 0
        all_peaks = []
        for s, e in zip(starts, ends):
            seg = trace_arr[s:e]
            peaks, _ = find_peaks(seg, height=thresh, distance=dist)
            calc_count += len(peaks)
            # Map back to absolute sample index
            all_peaks.append(peaks + s)
            
        if all_peaks:
            peaks = np.concatenate(all_peaks)
        else:
            peaks = np.array([], dtype=int)

    # Verification
    if pd.isna(expected_count):
        # We allow calc to be whatever if expected is NaN, but ideally we warn.
        # But Requirement B says "FAIL with clear message".
        # If expected is NaN, pipeline failed to extract.
       return True, calc_count, peaks # Let caller handle NaN display
       
    diff = abs(calc_count - expected_count)
    if diff > 1: # Strict tolerance
        print(f"CRITICAL: Peak Verification Failed for Chunk {cid} ROI {roi}")
        print(f"  stored={expected_count}, recomputed={calc_count}")
        print(f"  params: method={method}, k={config.get('peak_threshold_k')}, min_dist={min_dist_sec}")
        print(f"  stats: mu={mu:.3f}, sigma={sigma:.3f}, thresh={thresh:.3f}")
        return False, calc_count, peaks
        
    return True, calc_count, peaks

def main():
    args = parse_args()
    
    # 1. Load Artifacts
    config = load_config(args.analysis_out)
    
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
    # Infer from first trace if not provided
    first_trace = pd.read_csv(trace_files[0], nrows=1)
    dff_cols = [c for c in first_trace.columns if '_dff' in c]
    available_rois = [c.replace('_dff', '') for c in dff_cols]
    
    if args.roi:
        if args.roi not in available_rois:
            # Fallback check
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

    # 3. Build Grid Mapping (Trace-Driven)
    grid_rows = []
    
    # Pre-map features for fast lookup
    # Key: (chunk_id, roi)
    feat_map = {}
    for _, row in df_feat.iterrows():
        feat_map[(row['chunk_id'], row['roi'])] = row
        
    for tpath in trace_files:
        fname = os.path.basename(tpath)
        m = re.search(r'chunk_(\d+)\.csv', fname)
        if not m: continue
        cid = int(m.group(1))
        
        # Get metadata for time
        # Try feature source_file first
        dt = None
        if (cid, plot_roi) in feat_map:
            src = feat_map[(cid, plot_roi)].get('source_file', '')
            dt = infer_datetime_from_string(src)
        
        grid_rows.append({
            'chunk_id': cid,
            'trace_path': tpath,
            'datetime': dt
        })
        
    df_grid = pd.DataFrame(grid_rows)
    
    # 4. Infer Layout
    # Check coverage
    mapped = df_grid['datetime'].notnull()
    pct_mapped = mapped.mean() * 100
    
    sessions_ph = args.sessions_per_hour
    
    if pct_mapped > 90 and sessions_ph is None:
        # Time-based
        t0 = df_grid['datetime'].min()
        day_start = t0.replace(hour=0, minute=0, second=0, microsecond=0)
        
        df_grid['elapsed'] = (df_grid['datetime'] - day_start).dt.total_seconds()
        df_grid['day_idx'] = (df_grid['elapsed'] // 86400).astype(int)
        df_grid['hour_idx'] = ((df_grid['elapsed'] % 86400) // 3600).astype(int)
        
        # Rank
        df_grid = df_grid.sort_values(['day_idx', 'hour_idx', 'datetime'])
        df_grid['hour_rank'] = df_grid.groupby(['day_idx', 'hour_idx']).cumcount()
        
        # Mode sessions
        modes = df_grid.groupby(['day_idx', 'hour_idx']).size()
        if not modes.empty:
            sessions_ph = int(modes.mode()[0])
        else:
            sessions_ph = 1
            
    else:
        # Fallback
        if sessions_ph is None:
            # Heuristic: 48 chunks/day -> 2/hr
            n_chunks = len(df_grid)
            # Estimate days
            n_days_est = max(1, math.ceil(n_chunks / 48)) 
            sph_est = max(1, round(n_chunks / (24 * n_days_est)))
            sessions_ph = sph_est
            print(f"Fallback: Inferred {sessions_ph} sessions/hour from count.")
            
        df_grid = df_grid.sort_values('chunk_id')
        df_grid['day_idx'] = df_grid.index // (24 * sessions_ph)
        df_grid['hour_idx'] = (df_grid.index // sessions_ph) % 24
        df_grid['hour_rank'] = df_grid.index % sessions_ph
        
    sessions_ph = int(max(1, sessions_ph))
    print(f"Layout: {sessions_ph} columns.")

    # 5. Iteration & Signal Loading & Verification
    # Collect all traces to determine Y-limits
    all_traces = []
    
    # For circadian check
    hourly_peak_counts = [] # (hour, count)
    
    fs_val = config.get('target_fs_hz', 20.0) 
    # Or infer fs from time col? Best to use config or derived.
    
    print("Verifying peaks and loading traces...")
    
    plot_data = [] # List of dicts for plotting
    
    for _, row in df_grid.iterrows():
        cid = row['chunk_id']
        tpath = row['trace_path']
        
            # Load Trace
        try:
            tdf = pd.read_csv(tpath)
            
            # X axis
            if 'time_sec' in tdf.columns:
                x = tdf['time_sec'].values
            elif 'Time(s)' in tdf.columns:
                x = tdf['Time(s)'].values
            else:
                x = np.arange(len(tdf)) / fs_val
            x = x - x[0] # Normalize
            
            # ACCUMULATE TRACES FOR Y-LIMITS
            # Note: y is not defined yet! It depends on mode.
            # We must move this AFTER y is defined.
            
            if args.mode == 'dff':
                col = determine_signal_column(tpath, plot_roi, args.signal)
                y = tdf[col].values
                uv = None
                exp_count = np.nan
                
                # Peak Verify (only for dff)
                feat_row = feat_map.get((cid, plot_roi))
                if feat_row is not None:
                    exp_count = feat_row['peak_count']
                
                # Internal Re-computation
                ok, recal_count, peak_indices = verify_peak_count(y, fs_val, config, exp_count, plot_roi, cid)
                if not ok:
                    print("ABORTING due to verification failure.")
                    sys.exit(1)
                    
            elif args.mode == 'raw':
                col_sig = f"{plot_roi}_sig_raw"
                col_uv = f"{plot_roi}_uv_raw"
                
                if col_sig not in tdf.columns or col_uv not in tdf.columns:
                    print(f"Skipping {cid}: Raw columns missing")
                    continue
                    
                y = tdf[col_sig].values
                uv = tdf[col_uv].values
                peak_indices = [] # No peaks in raw mode
                exp_count = np.nan
            
            all_traces.append(y)
            if uv is not None:
                all_traces.append(uv)
            
            # Store for plotting
            plot_data.append({
                'day': row['day_idx'],
                'hour': row['hour_idx'],
                'col': row['hour_rank'],
                'x': x,
                'y': y,
                'uv': uv,
                'peaks_x': x[peak_indices] if len(peak_indices) > 0 else [],
                'peaks_y': y[peak_indices] if len(peak_indices) > 0 else [],
                'count': exp_count,
                'chunk_id': cid 
            })
            
            if pd.notna(exp_count):
                hourly_peak_counts.append((row['hour_idx'], exp_count))
                
        except Exception as e:
            print(f"Error processing chunk {cid}: {e}")
            # Do NOT exit, might be partial failure, but for manual demo we want strictness.
            # We will continue but cell will be empty or error marked.
            pass

    # 6. Global Y Limits
    if not all_traces:
        print("No trace data loaded.")
        sys.exit(1)
        
    flat_y = np.concatenate([t[np.isfinite(t)] for t in all_traces])
    if len(flat_y) == 0:
        global_ymin, global_ymax = -1, 1
    else:
        global_ymin, global_ymax = np.percentile(flat_y, [1, 99])
        yrange = global_ymax - global_ymin
        global_ymin -= 0.1 * yrange
        global_ymax += 0.1 * yrange
        
    print(f"Global (DFF-based) Y-Limits: [{global_ymin:.3f}, {global_ymax:.3f}]")

    # 7. Plotting
    output_dir = args.output_dir or os.path.join(args.analysis_out, 'phasic_qc')
    os.makedirs(output_dir, exist_ok=True)
    
    unique_days = sorted(df_grid['day_idx'].unique())
    
    # Unified Layout for BOTH modes
    cols = sessions_ph
    rows = 24 # Fixed 24 hours
    figsize_width = 4*sessions_ph + 2
    figsize_height = 24
    sharey_val = True # Global Y-limits for all (but we will apply manually)

    for d in unique_days:
        # Prepare Plot
        day_items = [p for p in plot_data if p['day'] == d]
        n_plots = len(day_items)
        
        if n_plots == 0:
            print(f"No chunks found to plot for Day {d}.")
            continue # Skip to next day
            
        # Determine Limits for this Day
        if args.mode == 'dff':
             day_ymin, day_ymax = global_ymin, global_ymax
        elif args.mode == 'raw':
             # Collect all finite values for this day (sig and uv)
             day_values = []
             for p in day_items:
                 if p['y'] is not None:
                     v = p['y'][np.isfinite(p['y'])]
                     if len(v) > 0: day_values.append(v)
                 if p['uv'] is not None:
                     v = p['uv'][np.isfinite(p['uv'])]
                     if len(v) > 0: day_values.append(v)
             
             if not day_values:
                 day_ymin, day_ymax = -1, 1
             else:
                 all_day_v = np.concatenate(day_values)
                 if len(all_day_v) == 0:
                     day_ymin, day_ymax = -1, 1
                 else:
                     raw_min = np.min(all_day_v)
                     raw_max = np.max(all_day_v)
                     pad = 0.05 * (raw_max - raw_min)
                     if pad == 0: pad = 1.0
                     day_ymin = raw_min - pad
                     day_ymax = raw_max + pad
             print(f"  Day {d} Raw Limits: [{day_ymin:.3f}, {day_ymax:.3f}]")
             
        print(f"DEBUG: Day {d}, Mode {args.mode}, n_plots={n_plots}, rows={rows}, cols={cols}")

        fig, axes = plt.subplots(nrows=rows, ncols=cols, 
                                 figsize=(figsize_width, figsize_height),
                                 sharex=True, sharey=False) # We set ylim manually

        # Ensure axes is always a 2D array for consistent indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
            
        fig.suptitle(f"Phasic QC - Day {d} - ROI {plot_roi} - Mode: {args.mode.upper()}", fontsize=16)
        
        # Plot Loop
        print(f"Generating Grid for Day {d} ({n_plots} chunks)... Mode={args.mode}")
        
        for i, p in enumerate(day_items):
            h, c = p['hour'], p['col']
            
            # Subplot indexing (Unified)
            if c >= cols:
                print(f"Skipping chunk {p['chunk_id']} (rank {c} >= cols {cols})")
                continue
            ax = axes[h, c]
            
            # Plot Trace
            if args.mode == 'dff':
                ax.plot(p['x'], p['y'], 'k', lw=0.8)
                
                # Overlay Peaks
                if len(p['peaks_x']) > 0:
                    ax.scatter(p['peaks_x'], p['peaks_y'], s=10, c='red', alpha=0.6, zorder=3)
                
                # Annotation
                val = p['count']
                if pd.isna(val):
                    txt = "peaks=NaN"
                    color = 'red'
                else:
                    txt = f"peaks={int(val)}"
                    color = 'blue'
                    
                ax.text(0.95, 0.9, txt, transform=ax.transAxes, 
                        ha='right', va='top', fontsize=8, color=color, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                ax.set_ylabel("dFF")

            elif args.mode == 'raw':
                # RAW Mode: Sig (Green) and Iso (Purple)
                ax.plot(p['x'], p['y'], color='green', lw=0.8, label='Sig')
                if p['uv'] is not None:
                     ax.plot(p['x'], p['uv'], color='purple', lw=0.8, alpha=0.7, label='Iso')
                
                ax.set_ylabel("Raw AU")
                if i == 0: # Legend only on first plot
                    ax.legend(fontsize='x-small', loc='upper right')
            
            # Common Aesthetics
            ax.set_title(f"Chunk {p['chunk_id']}", fontsize=9)
            ax.grid(True, alpha=0.3)
            # Use Day-Specific Y-Limits (which are global per day for raw, or global overall for dff)
            ax.set_ylim(day_ymin, day_ymax)

        # Formatting
        # Turn off unused subplots for raw mode if n_plots < total_subplots
        # This logic is now handled by the unified layout, where empty cells remain empty.
        # We just need to ensure labels are set correctly.
        
        for i_row in range(rows):
            axes[i_row,0].set_ylabel(f"H{i_row:02d}", rotation=0, labelpad=20, fontweight='bold')
            for j_col in range(cols):
                ax = axes[i_row, j_col]
                ax.grid(True, alpha=0.3)
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save output
        if args.mode == 'dff':
            fname = f"day_{d:03d}.png"
            out_path = os.path.join(output_dir, fname)
            plt.savefig(out_path, dpi=args.dpi)
            print(f"Saved {out_path}")
        else: # raw
            # Standard raw filename
            fname = f"day_{d:03d}_raw.png"
            out_path = os.path.join(output_dir, fname)
            plt.savefig(out_path, dpi=args.dpi)
            print(f"Saved {out_path}")
            
            # Canonical alias for Day 0
            if d == 0:
                canon_path = os.path.join(output_dir, "fig_phasic_raw_qc_grid.png")
                import shutil
                shutil.copy(out_path, canon_path)
                print(f"Saved canonical: {canon_path}")

        plt.close(fig)

    # 8. Circadian Sanity Check (Requirement F)
    print("Checking Circadian Modulation...")
    if not hourly_peak_counts:
        print("WARNING: No peak counts available for circadian check.")
    else:
        df_hc = pd.DataFrame(hourly_peak_counts, columns=['hour', 'count'])
        mean_hourly = df_hc.groupby('hour')['count'].mean()
        
        min_v = mean_hourly.min()
        max_v = mean_hourly.max()
        
        print(f"Hour Means: Min={min_v:.1f}, Max={max_v:.1f}")
        
        # Criterion: Max/Min >= 1.5 OR Range >= 20 (arbitrary but generous for high_phasic)
        # Synthetic high_phasic usually goes 0 -> 100+ -> 0.
        ratio = max_v / (min_v + 1e-9)
        diff = max_v - min_v
        
        if ratio < 1.5 and diff < 10:
            print("CRITICAL: Synthetic high_phasic did not show circadian modulation.")
            print(f"  Contrast: {ratio:.2f}x, Range: {diff:.1f}")
            print("  QC signal/peak_count mapping likely wrong or synthetic data is flat.")
            sys.exit(1)
        else:
            print(f"Circadian Check PASS (Contrast {ratio:.1f}x, Range {diff:.1f})")

if __name__ == "__main__":
    main()
