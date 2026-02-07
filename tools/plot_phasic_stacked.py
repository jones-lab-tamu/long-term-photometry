#!/usr/bin/env python3
"""
Phasic Stacked Plotter
======================

Plots stacked phasic traces from analysis output.
Similar to the legacy plot_C_stacked style.

Inputs:
- <analysis-out>/traces/chunk_*.csv (or phasic_intermediates)

Outputs:
- <analysis-out>/phasic_qc/plot_C_stacked_<roi>.png
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-out', required=True)
    parser.add_argument('--roi', default='Region0')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Locate traces
    # Phasic analysis outputs intermediates with dFF usually in 'phasic_intermediates'
    # or in 'traces'. We prefer 'phasic_intermediates' for the most "phasic-y" signal
    # but 'traces' is the standard output.
    # The user said: "Read chunk traces from <analysis-out>/traces/chunk_*.csv"
    # So we stick to that first.
    
    traces_dir = os.path.join(args.analysis_out, 'traces')
    files = sorted(glob.glob(os.path.join(traces_dir, 'chunk_*.csv')))
    
    if not files:
        print(f"CRITICAL: No trace files found in {traces_dir}")
        sys.exit(1)
        
    print(f"Found {len(files)} chunks.")
    
    # Identify Column (Strict DFF)
    df0 = pd.read_csv(files[0], nrows=5)
    col_dff = f"{args.roi}_dff"
    
    if col_dff not in df0.columns:
        print(f"CRITICAL: Column {col_dff} not found in traces.")
        print(f"Stacked Phasic Plot requires strict '{col_dff}' to ensure phasic signal (not tonic).")
        sys.exit(1)
        
    # Plotting
    fig, ax = plt.subplots(figsize=(10, len(files)*0.2 + 2))
    
    offset_step = 2.0 # Visual spacing
    # We might need to auto-scale this if dFF is huge/small
    # But usually dFF is % or similar. 2.0 is usually fine for dFF% (0-5 range typically)
    # If dFF is fractional (0.0 - 0.05), we need smaller step.
    # Let's check magnitude of first chunk
    y0 = df0[col_dff].values
    rng = np.nanmax(y0) - np.nanmin(y0)
    if rng < 0.5:
        offset_step = 0.2
    
    print(f"Plotting {col_dff} with vertical offset step {offset_step}...")
    
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f)
            y = df[col_dff].values
            t = df['time_sec'].values
            
            # Median center
            y = y - np.nanmedian(y)
            
            # Offset (stack upwards)
            y_plot = y + i * offset_step
            
            ax.plot(t, y_plot, lw=0.5, color='black', alpha=0.8)
            
        except Exception as e:
            print(f"Warning: Error plotting {f}: {e}")
            
    ax.set_xlabel("Time within Chunk (s)")
    ax.set_yticks([]) # Hide y-axis as it's arbitrary stacked
    ax.set_ylabel(f"Sessions (Chunk 0 (bottom) to {len(files)-1})")
    ax.set_title(f"Stacked Phasic Traces - {args.roi}")
    
    # Save
    qc_dir = os.path.join(args.analysis_out, 'phasic_qc')
    os.makedirs(qc_dir, exist_ok=True)
    out_path = os.path.join(qc_dir, f"plot_C_stacked_{args.roi}.png")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
