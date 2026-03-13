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
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        
    from photometry_pipeline.io.hdf5_cache_reader import open_phasic_cache, load_cache_chunk_fields
    
    # Construct path to phasic cache
    cache_path = os.path.join(args.analysis_out, 'phasic_trace_cache.h5')
    if not os.path.exists(cache_path):
        print(f"CRITICAL: Phasic cache not found: {cache_path}")
        sys.exit(1)
        
    try:
        with open_phasic_cache(cache_path) as f:
            fields = ['time_sec', 'sig_raw', 'uv_raw', 'fit_ref', 'dff']
            t, sig, iso, fit, dff = load_cache_chunk_fields(f, args.roi, args.chunk_id, fields)
    except Exception as e:
        print(f"CRITICAL: Failed to read cache for ROI {args.roi} Chunk {args.chunk_id}: {e}")
        sys.exit(1)
    
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
