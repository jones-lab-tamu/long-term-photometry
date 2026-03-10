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
import time

# Ensure repo root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from photometry_pipeline.viz.phasic_data_prep import (
    discover_chunks, build_feature_map, compute_day_layout,
    infer_datetime_from_string,
)

# infer_datetime_from_string is now imported from photometry_pipeline.viz.phasic_data_prep

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
    t_start = time.perf_counter()
    print("PLOT_TIMING START script=plot_phasic_stacked_day_smoothed.py", flush=True)
    
    args = parse_args()
    
    # 1. Shared data preparation (chunk discovery, feature map, layout)
    traces_dir = os.path.join(args.analysis_out, 'traces')
    feats_path = os.path.join(args.analysis_out, 'features', 'features.csv')
    
    try:
        chunk_entries = discover_chunks(traces_dir)
    except RuntimeError as e:
        print(f"CRITICAL: {e}")
        sys.exit(1)
    
    feat_map = build_feature_map(feats_path, roi=args.roi)
    pds = compute_day_layout(chunk_entries, feat_map, args.roi, args.sessions_per_hour)
    
    # 2. Process Each Day
    os.makedirs(args.out_dir, exist_ok=True)
    unique_days = sorted(pds.chunks_by_day.keys())

    print(f"Processing {len(unique_days)} days for ROI {args.roi}...")
    
    # Detect dFF Column
    df0 = pd.read_csv(chunk_entries[0][1], nrows=1)
    col_dff = f"{args.roi}_dff"
    
    # If not found, look for alternates
    if col_dff not in df0.columns:
        cand = [c for c in df0.columns if args.roi in c and '_dff' in c]
        if cand: 
            col_dff = cand[0]
        else:
             print(f"CRITICAL: DFF column for ROI {args.roi} not found.")
             sys.exit(1)

    print(f"PLOT_TIMING STEP script=plot_phasic_stacked_day_smoothed.py step=discovery elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # Pre-process traces
    prepared_traces = {}
    for cr in pds.chunks:
        try:
            df = pd.read_csv(cr.trace_path)
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
            
            prepared_traces[cr.chunk_id] = (t, y_smooth)
            
        except Exception as e:
            print(f"Warning: Failed to load chunk {cr.chunk_id}: {e}")
            continue

    print(f"PLOT_TIMING STEP script=plot_phasic_stacked_day_smoothed.py step=data_loading elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
    print(f"PLOT_TIMING STEP script=plot_phasic_stacked_day_smoothed.py step=data_preparation elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    for d in unique_days:
        day_chunks = sorted(pds.chunks_by_day.get(d, []), key=lambda c: c.chunk_id)
        if not day_chunks: continue
        
        traces = []
        for cr in day_chunks:
            cid = cr.chunk_id
            if cid in prepared_traces:
                traces.append(prepared_traces[cid])
                
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
        print(f"PLOT_TIMING STEP script=plot_phasic_stacked_day_smoothed.py step=plotting day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
        plt.savefig(out_path, dpi=args.dpi)
        print(f"PLOT_TIMING STEP script=plot_phasic_stacked_day_smoothed.py step=figure_save day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"PLOT_TIMING DONE script=plot_phasic_stacked_day_smoothed.py total_sec={time.perf_counter() - t_start:.3f}", flush=True)

if __name__ == '__main__':
    main()
