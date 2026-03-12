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
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from photometry_pipeline.io.hdf5_cache_reader import (
    open_phasic_cache, resolve_cache_roi, list_cache_chunk_ids, load_cache_chunk_fields
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-out', required=True)
    parser.add_argument('--roi', default='Region0')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Locate Cache
    cache_path = os.path.join(args.analysis_out, 'phasic_trace_cache.h5')
    if not os.path.exists(cache_path):
        print(f"CRITICAL: Phasic cache not found: {cache_path}")
        sys.exit(1)
        
    cache = open_phasic_cache(cache_path)
    try:
        # Resolve ROI
        roi = resolve_cache_roi(cache, args.roi)
        
        # Discover Chunks
        cids = list_cache_chunk_ids(cache)
        if not cids:
            print(f"CRITICAL: No chunks found in cache: {cache_path}")
            sys.exit(1)
            
        print(f"Found {len(cids)} chunks in cache.")
        
        # Load and verify first chunk for magnitude/scaling check
        t0, dff0 = load_cache_chunk_fields(cache, roi, cids[0], ['time_sec', 'dff'])
        
        # Visual spacing
        offset_step = 2.0 
        rng = np.nanmax(dff0) - np.nanmin(dff0)
        if rng < 0.5:
            offset_step = 0.2
        
        print(f"Plotting ROI {roi} with vertical offset step {offset_step}...")
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, len(cids)*0.2 + 2))
        
        for i, cid in enumerate(cids):
            try:
                # Load from cache
                t, y = load_cache_chunk_fields(cache, roi, cid, ['time_sec', 'dff'])
                
                # Median center
                y = y - np.nanmedian(y)
                
                # Offset (stack upwards)
                y_plot = y + i * offset_step
                
                ax.plot(t, y_plot, lw=0.5, color='black', alpha=0.8)
                
            except Exception as e:
                print(f"Warning: Error plotting chunk {cid}: {e}")
                
        ax.set_xlabel("Time within Chunk (s)")
        ax.set_yticks([]) # Hide y-axis as it's arbitrary stacked
        ax.set_ylabel(f"Sessions (Chunk 0 (bottom) to {len(cids)-1})")
        ax.set_title(f"Stacked Phasic Traces - {roi}")
        
        # Save
        qc_dir = os.path.join(args.analysis_out, 'phasic_qc')
        os.makedirs(qc_dir, exist_ok=True)
        out_path = os.path.join(qc_dir, f"plot_C_stacked_{roi}.png")
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        
        print(f"Saved {out_path}")
    finally:
        cache.close()

if __name__ == '__main__':
    main()
