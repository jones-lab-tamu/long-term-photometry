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
import time

# Ensure repo root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from photometry_pipeline.viz.phasic_data_prep import (
    compute_day_layout,
)
from photometry_pipeline.io.hdf5_cache_reader import (
    open_phasic_cache, resolve_cache_roi, list_cache_chunk_ids, list_cache_source_files, load_cache_chunk_fields
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-out', required=True)
    parser.add_argument('--roi', default=None)
    parser.add_argument('--sessions-per-hour', type=int, default=None)
    parser.add_argument('--session-duration-s', type=float, default=None, help="Expected session duration in seconds. If provided, used for validation.")
    parser.add_argument('--output-dir', default=None, help="Override output directory")
    parser.add_argument('--output-pattern', default=None, help="E.g. phasic_sig_iso_day_{d:03d}.png")
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
    t_start = time.perf_counter()
    print("PLOT_TIMING START script=plot_session_grid.py", flush=True)
    
    args = parse_args()
    
    # Locate Cache
    cache_path = os.path.join(args.analysis_out, 'phasic_trace_cache.h5')
    if not os.path.exists(cache_path):
        print(f"CRITICAL: Phasic cache not found: {cache_path}")
        sys.exit(1)
        
    cache = open_phasic_cache(cache_path)
    
    # Discover Chunks via Cache Metadata (IDs and original source files for layout inference)
    cids = list_cache_chunk_ids(cache)
    sfs = list_cache_source_files(cache)
    
    if not cids:
        print("CRITICAL: No chunks found in cache.")
        sys.exit(1)
        
    # Build entries for layout engine (second element is source_file for datetime inference)
    chunk_entries = list(zip(cids, sfs))

    # Resolve ROI via Cache
    try:
        roi = resolve_cache_roi(cache, args.roi)
    except RuntimeError as e:
        print(f"CRITICAL: {e}")
        sys.exit(1)
    if not args.roi:
        print(f"Auto-selected ROI: {roi}")

    pds = compute_day_layout(chunk_entries, None, roi, args.sessions_per_hour)
    sph = pds.sessions_per_hour

    # 1. Audit
    print(f"PLOT_TIMING STEP script=plot_session_grid.py step=discovery elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
    print("Auditing session boundaries via cache...")
    plot_items = []
    
    for cr in pds.chunks:
        cid = cr.chunk_id
        try:
            # Load required fields from cache
            t, sig, uv = load_cache_chunk_fields(cache, roi, cid, ['time_sec', 'sig_raw', 'uv_raw'])
            
            # Monotonicity
            if not check_monotonicity(t):
                print(f"CRITICAL: Non-monotonic time in cache for chunk {cid}")
                sys.exit(1)
                
            # Duration
            duration = t[-1] - t[0]
            
            # Validation Logic
            if args.session_duration_s is not None:
                 # Validated against provided duration with tolerance
                 expected = args.session_duration_s
                 tol = max(2.0, 0.005 * expected)
                 if abs(duration - expected) > tol:
                      print(f"CRITICAL: Duration mismatch in cache for chunk {cid}. Expected ~{expected:.2f}s, got {duration:.2f}s (Diff > {tol:.2f}s)")
                      sys.exit(1)
            else:
                 # Standard check (10 min = 600s). Request says [590, 610]
                 if not (590 <= duration <= 610):
                     print(f"CRITICAL: Invalid duration {duration:.2f}s in cache for chunk {cid} (Expected ~600s)")
                     sys.exit(1)
                
            # Continuity
            # Infer dt from median diff
            dt = np.median(np.diff(t))
            if not check_continuity(t, dt):
                print(f"CRITICAL: Discontinuity detected in cache for chunk {cid}")
                sys.exit(1)
                
            # Store plot data
            plot_items.append({
                'day': cr.day_idx, 
                'hour': cr.hour_idx, 
                'col': cr.hour_rank,
                't': t - t[0],
                'sig': sig,
                'uv': uv
            })
            
        except Exception as e:
            print(f"CRITICAL: Read error from cache for chunk {cid}: {e}")
            sys.exit(1)
            
    # Done with cache
    cache.close()
    
    print(f"Audit PASS: {len(plot_items)} chunks verified.")
    print(f"PLOT_TIMING STEP script=plot_session_grid.py step=data_loading elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
    
    # 2. Grid Plot
    # Simplest layout: sequential unless datetimes present in text
    # Step 3 requirement: "builds the day/hour/session grid"
    # We will iterate and fill.
    
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(args.analysis_out, 'session_qc')
    os.makedirs(out_dir, exist_ok=True)
    
    unique_days = sorted(pds.chunks_by_day.keys())
    print(f"PLOT_TIMING STEP script=plot_session_grid.py step=data_preparation elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
    
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
            
        if args.output_pattern:
            out_name = args.output_pattern.format(d=d, roi=roi)
        else:
            out_name = f"day_{d:03d}_raw_iso_{roi}.png"
            
        out_path = os.path.join(out_dir, out_name)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        print(f"PLOT_TIMING STEP script=plot_session_grid.py step=plotting day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
        plt.savefig(out_path)
        print(f"PLOT_TIMING STEP script=plot_session_grid.py step=figure_save day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
        print(f"Saved {out_path}")
        plt.close(fig)

    print(f"PLOT_TIMING DONE script=plot_session_grid.py total_sec={time.perf_counter() - t_start:.3f}", flush=True)

if __name__ == '__main__':
    main()
