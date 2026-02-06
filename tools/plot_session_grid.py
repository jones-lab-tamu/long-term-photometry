#!/usr/bin/env python3
"""
Session QC Grid & Boundary Audit
================================

Step 3 of Verification Protocol.
1. Strictly audits every chunk for time monotonicity and valid duration.
2. Plots Raw (Sig/UV) traces in a 24xN grid.

Inputs:
- <analysis-out>/traces/chunk_*.csv

Outputs:
- <analysis-out>/session_qc/day_{N}_raw_iso_<roi>.png
"""

import os
import sys
import glob
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import layout logic from plot_phasic_qc_grid? 
# Better to duplicate minimal logic to stay self-contained or import if allowed.
# I'll implement self-contained simple mapping to ensure independence.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-out', required=True)
    parser.add_argument('--roi', default=None)
    parser.add_argument('--sessions-per-hour', type=int, default=None)
    return parser.parse_args()

def check_monotonicity(time_arr):
    """Returns True if strictly monotonic increasing."""
    return np.all(np.diff(time_arr) > 0)

def check_continuity(time_arr, expected_dt):
    """Returns True if no jumps > 2 * dt."""
    diffs = np.diff(time_arr)
    # Allow some jitter, but gaps > 2x are bad
    return np.all(diffs < (2.0 * expected_dt))

def main():
    args = parse_args()
    traces_dir = os.path.join(args.analysis_out, 'traces')
    files = sorted(glob.glob(os.path.join(traces_dir, 'chunk_*.csv')))
    
    if not files:
        print("CRITICAL: No traces found.")
        sys.exit(1)

    # 1. Audit
    print("Auditing session boundaries...")
    valid_files = []
    
    # Auto-detect ROI
    df0 = pd.read_csv(files[0], nrows=1)
    sig_cols = [c for c in df0.columns if '_sig_raw' in c]
    rois = [c.replace('_sig_raw','') for c in sig_cols]
    if args.roi: 
        roi = args.roi
    else:
        roi = rois[0]
        print(f"Auto-selected ROI: {roi}")
        
    for f in files:
        try:
            df = pd.read_csv(f)
            t = df['time_sec'].values
            
            # Monotonicity
            if not check_monotonicity(t):
                print(f"CRITICAL: Non-monotonic time in {f}")
                sys.exit(1)
                
            # Duration
            duration = t[-1] - t[0]
            # 10 min = 600s. Tolerate small padding diffs? 
            # Request says [590, 610]
            if not (590 <= duration <= 610):
                print(f"CRITICAL: Invalid duration {duration:.2f}s in {f}")
                sys.exit(1)
                
            # Continuity
            # Infer dt from median diff
            dt = np.median(np.diff(t))
            if not check_continuity(t, dt):
                print(f"CRITICAL: Discontinuity detected in {f}")
                sys.exit(1)
                
            valid_files.append((f, df))
            
        except Exception as e:
            print(f"CRITICAL: Read error {f}: {e}")
            sys.exit(1)
            
    print(f"Audit PASS: {len(valid_files)} chunks verified.")
    
    # 2. Grid Plot
    # Simplest layout: sequential unless datetimes present in text
    # Step 3 requirement: "builds the day/hour/session grid"
    # We will iterate and fill.
    
    sph = args.sessions_per_hour
    if sph is None:
        # Heuristic: 2 days, 96 chunks -> 2/hr
        n_chunks = len(valid_files)
        sph = max(1, round(n_chunks / 48.0))
        print(f"Inferred {sph} sessions/hour")
        
    out_dir = os.path.join(args.analysis_out, 'session_qc')
    os.makedirs(out_dir, exist_ok=True)
    
    # Group by day
    chunks_per_day = 24 * sph
    
    # Process all chunks, store plot data
    plot_items = []
    
    for idx, (fpath, df) in enumerate(valid_files):
        day = idx // chunks_per_day
        hour = (idx // sph) % 24
        col = idx % sph
        
        plot_items.append({
            'day': day, 
            'hour': hour, 
            'col': col,
            't': df['time_sec'].values - df['time_sec'].values[0],
            'sig': df[f"{roi}_sig_raw"].values,
            'uv': df[f"{roi}_uv_raw"].values
        })
        
    unique_days = sorted(list(set(p['day'] for p in plot_items)))
    
    for d in unique_days:
        fig, axes = plt.subplots(nrows=24, ncols=sph, 
                                 figsize=(4*sph + 2, 24),
                                 sharex=True)
        if sph == 1: axes = axes.reshape(-1, 1)
        
        fig.suptitle(f"Day {d} Raw/Iso - {roi}", fontsize=16)
        
        day_items = [p for p in plot_items if p['day'] == d]
        
        for p in day_items:
            ax = axes[p['hour'], p['col']]
            ax.plot(p['t'], p['sig'], 'g', lw=0.5, label='Sig')
            ax.plot(p['t'], p['uv'], 'm', lw=0.5, label='Iso')
            
            if p['t'].max() > 550:
                ax.set_xlim(0, 600)
                
        # Labels
        for i in range(24):
            axes[i,0].set_ylabel(f"H{i:02d}", rotation=0, labelpad=20)
            
        out_path = os.path.join(out_dir, f"day_{d:03d}_raw_iso_{roi}.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(out_path)
        print(f"Saved {out_path}")
        plt.close(fig)

if __name__ == '__main__':
    main()
