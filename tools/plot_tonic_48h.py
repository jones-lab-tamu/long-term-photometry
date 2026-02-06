#!/usr/bin/env python3
"""
Tonic QC Overview Plotter
=========================

Stitches all chunks for a given ROI to visualize the 48h structure.
Serves as the "Ground Truth Anchor" to verify synthetic generator output.

Inputs:
- <analysis-out>/traces/chunk_*.csv

Outputs:
- <analysis-out>/tonic_qc/tonic_48h_overview_<roi>.png
"""

import os
import sys
import glob
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-out', required=True)
    parser.add_argument('--roi', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    traces_dir = os.path.join(args.analysis_out, 'traces')
    files = sorted(glob.glob(os.path.join(traces_dir, 'chunk_*.csv')))
    
    if not files:
        print("CRITICAL: No trace files found.")
        sys.exit(1)
        
    # Auto-detect ROI
    df0 = pd.read_csv(files[0], nrows=1)
    sig_cols = [c for c in df0.columns if '_sig_raw' in c]
    available_rois = [c.replace('_sig_raw','') for c in sig_cols]
    
    if args.roi:
        roi = args.roi
    else:
        roi = available_rois[0]
        print(f"Auto-selected ROI: {roi}")

    # Columns
    col_sig = f"{roi}_sig_raw"
    col_uv = f"{roi}_uv_raw"
    col_dff = f"{roi}_dff" # Tonic dFF
    
    # Load and Stitch
    all_chunks = []
    
    print("Loading and stitching traces...")
    for f in files:
        df = pd.read_csv(f)
        if col_sig not in df.columns: continue
        
        # We need absolute time if possible, or just concat
        # Pipeline enforces time_sec is relative to chunk usually? 
        # Actually pipeline puts "time_sec" in trace. 
        # But if we want 48h, we need real timestamps or just strict concat if contiguous.
        # The synth data is generated in chunks but they represent a sequence.
        # We'll rely on sequential stitching for visualization if absolute time is tricky.
        
        all_chunks.append(df)
        
    full_df = pd.concat(all_chunks, ignore_index=True)
    
    # Create fake continuous time if needed, or use existing if it stitches well
    # Synth data likely restarts time_sec per chunk? 
    # Let's check first chunk dt
    dt = full_df['time_sec'].iloc[1] - full_df['time_sec'].iloc[0]
    full_df['continuous_time'] = np.arange(len(full_df)) * dt
    
    # Plot
    out_dir = os.path.join(args.analysis_out, 'tonic_qc')
    os.makedirs(out_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 1. Raw
    t = full_df['continuous_time'] / 3600.0 # Hours
    ax1.plot(t, full_df[col_sig], label='Sig', color='green', lw=0.5)
    ax1.plot(t, full_df[col_uv], label='Iso', color='purple', lw=0.5, alpha=0.8)
    ax1.set_ylabel("Raw Signal")
    ax1.set_title(f"48h Overview - {roi} - Raw Inputs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Tonic dFF
    if col_dff in full_df.columns:
        ax2.plot(t, full_df[col_dff], label='Tonic dFF', color='black', lw=0.8)
        ax2.set_ylabel("dFF %")
        ax2.set_title("Tonic Output")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No Tonic dFF data", ha='center')
        
    ax2.set_xlabel("Time (Hours)")
    
    out_path = os.path.join(out_dir, f"tonic_48h_overview_{roi}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    
if __name__ == '__main__':
    main()
