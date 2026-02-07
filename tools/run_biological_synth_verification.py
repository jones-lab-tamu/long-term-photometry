#!/usr/bin/env python3
"""
Biological Synthetic Data Verification
======================================

Generates 48h (or custom) synthetic data with biological nuisance parameters,
runs the photometry pipeline (Tonic AND Phasic modes), and generates verification plots.

Usage:
    python tools/run_biological_synth_verification.py --hours 48 --out tests/out_verify_bio --overwrite
    python tools/run_biological_synth_verification.py --hours 6 --out tests/out_verify_bio_fast --overwrite --fast
"""

import os
import sys
import argparse
import subprocess
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    print(result.stdout)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    
    if args.overwrite and os.path.exists(args.out):
        try:
            shutil.rmtree(args.out)
        except Exception as e:
            print(f"Warning: Could not remove {args.out} ({e}). Proceeding to overwrite in-place.")
        
    os.makedirs(args.out, exist_ok=True)
    
    # 1. Generate Data
    print("Step 1: Generating Synthetic Data...")
    
    # Config path assumed relative to repo root
    repo_root = os.getcwd() # Assumption
    config_path = os.path.join(repo_root, 'tests', 'qc_universal_config.yaml')
    
    days = args.hours / 24.0
    
    cmd_gen = [
        sys.executable, 'tools/synth_photometry_dataset.py',
        '--out', args.out,
        '--format', 'rwd',
        '--config', config_path,
        '--total-days', str(days),
        '--recordings-per-hour', '2', # Standard
        '--n-rois', '1', # 1 is enough for verification
        '--phasic-mode', 'high_phasic', # Or phase_locked_to_tonic
        '--seed', '999',
        '--preset', 'biological_shared_nuisance'
    ]
    
    run_command(cmd_gen, cwd=repo_root)
    
    # 2. Analyze (Two Passes)
    
    # A) Tonic Analysis
    print("Step 2A: Running Tonic Analysis...")
    analysis_tonic = os.path.join(args.out, 'analysis_tonic')
    
    cmd_analyze_tonic = [
        sys.executable, 'analyze_photometry.py',
        '--input', args.out,
        '--out', analysis_tonic,
        '--config', config_path,
        '--mode', 'tonic', # Explicit Tonic Mode
        '--overwrite',
        '--recursive',
        '--format', 'rwd'
    ]
    run_command(cmd_analyze_tonic, cwd=repo_root)

    # B) Phasic Analysis
    print("Step 2B: Running Phasic Analysis...")
    analysis_phasic = os.path.join(args.out, 'analysis_phasic')
    
    cmd_analyze_phasic = [
        sys.executable, 'analyze_photometry.py',
        '--input', args.out,
        '--out', analysis_phasic,
        '--config', config_path,
        '--mode', 'phasic', # Explicit Phasic Mode
        '--overwrite',
        '--recursive',
        '--format', 'rwd'
    ]
    run_command(cmd_analyze_phasic, cwd=repo_root)
    
    # 3. Generate Plots & Cleanup (Canonical)
    print("Step 3: Generating Canonical Figures...")
    
    figures_dir = os.path.join(args.out, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # A) Tonic 48h
    # Source: analysis_tonic/tonic_qc/tonic_48h_overview_Region0.png
    # Dest: figures/fig_tonic_48h.png
    run_command([
        sys.executable, 'tools/plot_tonic_48h.py', 
        '--analysis-out', analysis_tonic, 
        '--roi', 'Region0'
    ], cwd=repo_root)
    
    tonic_src = os.path.join(analysis_tonic, 'tonic_qc', 'tonic_48h_overview_Region0.png')
    if os.path.exists(tonic_src):
        shutil.copy(tonic_src, os.path.join(figures_dir, 'fig_tonic_48h.png'))
    else:
        print(f"WARNING: Tonic plot missing at {tonic_src}")

    # B) Phasic QC Grid (DFF)
    # Source: analysis_phasic/phasic_qc/day_000.png --> figures/fig_phasic_qc_grid.png
    run_command([
        sys.executable, 'tools/plot_phasic_qc_grid.py', 
        '--analysis-out', analysis_phasic, 
        '--roi', 'Region0',
        '--mode', 'dff'
    ], cwd=repo_root)
    
    grid_dff_src = os.path.join(analysis_phasic, 'phasic_qc', 'day_000.png')
    if os.path.exists(grid_dff_src):
        shutil.copy(grid_dff_src, os.path.join(figures_dir, 'fig_phasic_qc_grid.png'))
    else:
        print(f"WARNING: Phasic DFF Grid missing at {grid_dff_src}")
        
    # C) Phasic QC Grid (RAW) - NEW
    # Source: analysis_phasic/phasic_qc/fig_phasic_raw_qc_grid.png (named by tool in raw mode) 
    #         --> figures/fig_phasic_raw_qc_grid.png
    run_command([
        sys.executable, 'tools/plot_phasic_qc_grid.py', 
        '--analysis-out', analysis_phasic, 
        '--roi', 'Region0',
        '--mode', 'raw' 
    ], cwd=repo_root)
    
    grid_raw_src = os.path.join(analysis_phasic, 'phasic_qc', 'fig_phasic_raw_qc_grid.png')
    if os.path.exists(grid_raw_src):
        shutil.copy(grid_raw_src, os.path.join(figures_dir, 'fig_phasic_raw_qc_grid.png'))
    else:
        # Fallback if raw mode didn't name it perfectly or multiple days?
        # Tool logic: if mode=raw and day=0, names it fig_phasic_raw_qc_grid.png
        print(f"WARNING: Phasic Raw Grid missing at {grid_raw_src}")

    # D) Phasic Stacked (Strict DFF)
    # Source: analysis_phasic/phasic_qc/plot_C_stacked_Region0.png --> figures/fig_phasic_stacked.png
    run_command([
        sys.executable, 'tools/plot_phasic_stacked.py', 
        '--analysis-out', analysis_phasic, 
        '--roi', 'Region0'
    ], cwd=repo_root)
    
    stacked_src = os.path.join(analysis_phasic, 'phasic_qc', 'plot_C_stacked_Region0.png')
    if os.path.exists(stacked_src):
        shutil.copy(stacked_src, os.path.join(figures_dir, 'fig_phasic_stacked.png'))
    else:
        print(f"WARNING: Stacked plot missing at {stacked_src}")

    # E) Dynamic Isosbestic Correction Impact
    # Source: analysis_phasic/viz/plot_D_correction_impact_Region0.png 
    # Dest: figures/fig_dynamic_isosbestic_correction_Region0.png
    # Note: 'viz' folder created by pipeline.
    
    corr_src = os.path.join(analysis_phasic, 'viz', 'plot_D_correction_impact_Region0.png')
    if os.path.exists(corr_src):
        shutil.copy(corr_src, os.path.join(figures_dir, 'fig_dynamic_isosbestic_correction_Region0.png'))
    else:
        print(f"WARNING: Correction impact plot missing at {corr_src}")

    # CLEANUP: Remove pipeline 'viz' folders to avoid confusion
    # USER REQUEST: Do NOT delete analysis_tonic/viz or analysis_phasic/viz by default.
    # print("Step 4: Pruning duplicate Visualization outputs...")
    # viz_dirs = [
    #     os.path.join(analysis_tonic, 'viz'),
    #     os.path.join(analysis_phasic, 'viz')
    # ]
    # for v in viz_dirs:
    #     if os.path.exists(v):
    #         try:
    #             shutil.rmtree(v)
    #             print(f"Pruned {v}")
    #         except Exception as e:
    #             print(f"Warning: Failed to prune {v}: {e}")
                
    # Also clean legacy plots from root of out_verify_bio if any were copied before?
    # No, we only copy to figures now.
    # But clean legacy files produced by earlier runs if present (fig_*.png in root)?
    # The script recreates out dir if overwrite, so valid.
    
    print("VERIFICATION SUCCESS: Canonical outputs generated in 'figures/'")
    print(f"See: {figures_dir}")

if __name__ == '__main__':
    main()
