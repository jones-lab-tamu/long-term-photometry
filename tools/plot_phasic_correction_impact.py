#!/usr/bin/env python3
"""
Phasic Correction Impact Plotter
================================

Generates a 3-panel figure showing the impact of artifact correction for a specific diagnostic chunk.
Panel 1: Raw Signal vs Raw Isosbestic
Panel 2: Raw Signal vs Dynamic Iso Fit
Panel 3: Final dFF

Usage:
    python tools/plot_phasic_correction_impact.py --analysis-out <DIR> --roi <ROI> --chunk-id <ID> --out <FILE>
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-out', required=True)
    parser.add_argument('--roi', required=True)
    parser.add_argument('--chunk-id', required=True, type=int)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dpi', type=int, default=150)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Construct path to intermediate CSV
    # Expected: <analysis-out>/phasic_intermediates/chunk_<ID>_<ROI>.csv
    # Note: ID is 04d formatted in file generation usually?
    # Let's check glob or try formatted.
    
    inter_dir = os.path.join(args.analysis_out, 'phasic_intermediates')
    if not os.path.exists(inter_dir):
        print(f"CRITICAL: Intermediate directory not found: {inter_dir}")
        sys.exit(1)
        
    # Try finding the file
    candidate_pattern = f"chunk_{args.chunk_id:04d}_{args.roi}.csv"
    csv_path = os.path.join(inter_dir, candidate_pattern)
    
    if not os.path.exists(csv_path):
        # Fallback for legacy naming if any?
        # Try unpadded?
        cand2 = f"chunk_{args.chunk_id}_{args.roi}.csv"
        path2 = os.path.join(inter_dir, cand2)
        if os.path.exists(path2):
            csv_path = path2
        else:
            print(f"CRITICAL: Intermediate CSV not found at {csv_path}")
            sys.exit(1)
            
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"CRITICAL: Failed to read CSV: {e}")
        sys.exit(1)
        
    # Columns expected: time_sec, sig_raw, iso_raw, fit_ref, dff
    required = ['time_sec', 'sig_raw', 'iso_raw', 'fit_ref', 'dff']
    for c in required:
        if c not in df.columns:
            print(f"CRITICAL: Missing column {c} in {csv_path}")
            sys.exit(1)
            
    t = df['time_sec'].values
    sig = df['sig_raw'].values
    iso = df['iso_raw'].values
    fit = df['fit_ref'].values
    dff = df['dff'].values
    
    # Normalize time
    t = t - t[0]
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # 1. Raw vs Iso
    ax1.plot(t, sig, 'g', label='Signal (470nm)', lw=0.8)
    ax1.plot(t, iso, 'm', label='Iso (415nm)', lw=0.8, alpha=0.7)
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Raw Output (V)")
    ax1.set_title(f"Correction Impact - ROI {args.roi} - Chunk {args.chunk_id} - Raw Inputs")
    ax1.grid(True, alpha=0.3)
    
    # 2. Raw vs Fit
    ax2.plot(t, sig, 'g', label='Signal', lw=0.8)
    ax2.plot(t, fit, 'k', label='Iso Fit (Scaled)', lw=0.8, linestyle='--')
    ax2.legend(loc='upper right')
    ax2.set_ylabel("Raw Output (V)")
    ax2.set_title("Dynamic Reference Fitting (Lasso/ElasticNet)")
    ax2.grid(True, alpha=0.3)
    
    # 3. Final dFF
    ax3.plot(t, dff, 'b', label='dFF (Phasic)', lw=0.8)
    ax3.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax3.legend(loc='upper right')
    ax3.set_ylabel("dFF")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Final Corrected Signal")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.out}")

if __name__ == '__main__':
    main()
