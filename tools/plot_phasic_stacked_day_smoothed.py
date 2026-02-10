#!/usr/bin/env python3
"""
Phasic Stacked Plotter (Smoothed, Per-Day)
==========================================

Generates "Display-Only" smoothed stacked phasic trace plots for each day.

Usage:
    python tools/plot_phasic_stacked_day_smoothed.py --analysis-out <DIR> --roi <ROI> --out-dir <DIR>
"""

import os
import sys
import argparse
import glob
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from datetime import datetime

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-out', required=True)
    parser.add_argument('--roi', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--sessions-per-hour', type=int, default=None)
    parser.add_argument('--smooth-window-s', type=float, default=1.0)
    parser.add_argument('--dpi', type=int, default=150)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Features
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

    # 2. Build Grid/Group Mapping
    feat_map = {}
    for _, row in df_feat.iterrows():
        feat_map[(row['chunk_id'], row['roi'])] = row
        
    grid_rows = []
    for tpath in trace_files:
        fname = os.path.basename(tpath)
        m = re.search(r'chunk_(\d+)\.csv', fname)
        if not m: continue
        cid = int(m.group(1))
        
        # Determine Source/Time
        source = tpath
        dt = None
        if (cid, args.roi) in feat_map:
            source = feat_map[(cid, args.roi)].get('source_file', tpath)
            dt = infer_datetime_from_string(source)
            
        grid_rows.append({
            'chunk_id': cid,
            'trace_path': tpath,
            'datetime': dt,
            'source': source
        })

    df_grid = pd.DataFrame(grid_rows)
    
    # Inferred Grouping Logic
    mapped = df_grid['datetime'].notnull()
    pct_mapped = mapped.mean() * 100
    sessions_ph = args.sessions_per_hour
    
    # Logic to match other tools perfectly
    if pct_mapped > 90 and sessions_ph is None:
        # Datetime mode
        t0 = df_grid['datetime'].min()
        day_start = t0.replace(hour=0, minute=0, second=0, microsecond=0)
        df_grid['elapsed'] = (df_grid['datetime'] - day_start).dt.total_seconds()
        df_grid['day_idx'] = (df_grid['elapsed'] // 86400).astype(int)
    else:
        # Fallback loop
        if sessions_ph is None:
             # Basic default inference if not provided
             n_chunks = len(df_grid)
             n_days_est = max(1, math.ceil(n_chunks / 48)) 
             sph_est = max(1, round(n_chunks / (24 * n_days_est)))
             sessions_ph = sph_est
        
        df_grid = df_grid.sort_values('chunk_id')
        df_grid['day_idx'] = df_grid.index // (24 * (sessions_ph or 1))

    # 3. Process Each Day
    os.makedirs(args.out_dir, exist_ok=True)
    unique_days = sorted(df_grid['day_idx'].unique())

    print(f"Processing {len(unique_days)} days for ROI {args.roi}...")
    
    # Detect dFF Column
    df0 = pd.read_csv(trace_files[0], nrows=1)
    col_dff = f"{args.roi}_dff"
    
    # If not found, look for alternates
    if col_dff not in df0.columns:
        cand = [c for c in df0.columns if args.roi in c and '_dff' in c]
        if cand: 
            col_dff = cand[0]
        else:
             print(f"CRITICAL: DFF column for ROI {args.roi} not found.")
             sys.exit(1)

    for d in unique_days:
        day_chunks = df_grid[df_grid['day_idx'] == d].sort_values('chunk_id')
        if day_chunks.empty: continue
        
        traces = []
        
        for _, row in day_chunks.iterrows():
            try:
                df = pd.read_csv(row['trace_path'])
                y = df[col_dff].values.astype(float)
                t = df['time_sec'].values.astype(float)
                
                # Check for NaNs
                mask = np.isfinite(y)
                y = y[mask]
                t = t[mask]
                
                if len(y) < 2: continue
                
                t = t - t[0]
                
                # Infer fs
                dt = np.median(np.diff(t))
                fs = 1.0 / dt if dt > 0 else 1.0
                
                # Smooth
                w_s = args.smooth_window_s
                w_samples = int(round(fs * w_s))
                if w_samples < 1: w_samples = 1
                
                # Use standard uniform filter
                y_smooth = uniform_filter1d(y, size=w_samples)
                
                # No re-centering unless requested (user said "Do NOT re-center ... Prefer no centering")
                # But stacked plots need offset. 
                # We will just offset by index.
                
                traces.append((t, y_smooth))
                
            except Exception as e:
                print(f"Warning: Failed to load chunk {row['chunk_id']}: {e}")
                continue
                
        if not traces:
            continue
            
        fig, ax = plt.subplots(figsize=(6, len(traces)*0.3 + 2))
        
        # Calculate Offset Step
        ranges = [np.ptp(tr[1]) for tr in traces]
        avg_rng = np.median(ranges) if ranges else 1.0
        step = max(0.1, avg_rng * 0.8)
        
        for i, (t, y) in enumerate(traces):
            # Invert: First session (i=0) at TOP (highest y), Last session at BOTTOM
            offset = (len(traces) - 1 - i) * step
            ax.plot(t, y + offset, 'k', lw=0.5)
            
        ax.set_yticks([])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Sessions ({len(traces)})")
        ax.set_title(f"Day {d} Stacked (Smoothed {args.smooth_window_s}s) - {args.roi}")
        
        out_name = f"phasic_stacked_day_{d:03d}.png"
        out_path = os.path.join(args.out_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
