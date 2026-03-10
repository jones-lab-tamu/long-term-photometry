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
import numpy as np
import matplotlib.pyplot as plt
import time

# Only import pandas for the one-time header sniff
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-out', required=True)
    parser.add_argument('--roi', default=None)
    parser.add_argument('--out', default=None, help="Explicit output path for the PNG")
    return parser.parse_args()


# ======================================================================
# Stage 1: Discovery & Column Resolution
# ======================================================================

def discover_trace_files(traces_dir):
    files = sorted(glob.glob(os.path.join(traces_dir, 'chunk_*.csv')))
    if not files:
        print("CRITICAL: No trace files found.")
        sys.exit(1)
    return files


def resolve_roi_and_columns(first_file, requested_roi):
    """Read the header of the first chunk once and resolve
    the ROI name plus the exact column names needed."""
    df0 = pd.read_csv(first_file, nrows=0)  # header only, no data rows
    all_cols = df0.columns.tolist()

    sig_cols = [c for c in all_cols if c.endswith('_sig_raw')]
    available_rois = [c.replace('_sig_raw', '') for c in sig_cols]

    if requested_roi:
        roi = requested_roi
    else:
        if not available_rois:
            print("CRITICAL: No _sig_raw columns found. Cannot auto-select ROI.")
            sys.exit(1)
        roi = available_rois[0]
        print(f"Auto-selected ROI: {roi}")

    col_sig = f"{roi}_sig_raw"
    col_uv = f"{roi}_uv_raw"
    col_deltaF = f"{roi}_deltaF"

    # Validate the required columns exist in the header
    for col in (col_sig, col_uv, col_deltaF):
        if col not in all_cols:
            print(f"CRITICAL: Required column {col} not found in {first_file}")
            sys.exit(1)

    # Determine which time column is present
    if 'time_sec' in all_cols:
        col_time = 'time_sec'
    elif 'Time(s)' in all_cols:
        col_time = 'Time(s)'
    else:
        col_time = None  # will synthesize later

    usecols = [c for c in [col_time, col_sig, col_uv, col_deltaF] if c is not None]

    return roi, col_time, col_sig, col_uv, col_deltaF, usecols


# ======================================================================
# Stage 2: Trace Reading
# ======================================================================

def read_chunks(files, col_time, col_sig, col_uv, col_deltaF, usecols):
    """Load only required columns from each chunk CSV, accumulate
    as lists of numpy arrays. Does NOT concatenate."""
    list_sig = []
    list_uv = []
    list_deltaF = []

    for f in files:
        df = pd.read_csv(f, usecols=usecols)
        if col_sig not in df.columns:
            continue
        list_sig.append(df[col_sig].values)
        list_uv.append(df[col_uv].values)
        list_deltaF.append(df[col_deltaF].values)

    if not list_sig:
        print("CRITICAL: No chunks contained the required signal columns.")
        sys.exit(1)

    return list_sig, list_uv, list_deltaF


def assemble_arrays(list_sig, list_uv, list_deltaF, files, col_time, usecols):
    """Concatenate accumulated lists and build continuous time axis."""
    sig_raw = np.concatenate(list_sig)
    uv_raw = np.concatenate(list_uv)
    deltaf_val = np.concatenate(list_deltaF)

    # Infer dt from the first chunk's time column, if present
    if col_time is not None:
        first_chunk = pd.read_csv(files[0], usecols=[col_time], nrows=2)
        t_vals = first_chunk[col_time].values
        dt = t_vals[1] - t_vals[0] if len(t_vals) >= 2 else 1.0
    else:
        dt = 1.0

    total_pts = len(sig_raw)
    continuous_time = np.arange(total_pts) * dt

    return continuous_time, sig_raw, uv_raw, deltaf_val


# ======================================================================
# Stage 3: Plotting & Save
# ======================================================================

def plot_tonic_overview(t_h, sig_plot, uv_plot, deltaf_plot, roi, out_path, t_start):
    """Render the two-panel tonic overview and save."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 1. Raw
    ax1.plot(t_h, sig_plot, label='Sig', color='green', lw=0.5)
    ax1.plot(t_h, uv_plot, label='Iso', color='purple', lw=0.5, alpha=0.8)
    ax1.set_ylabel("Raw Signal")
    ax1.set_title(f"48h Overview - {roi} - Raw Inputs")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Tonic Output (deltaF)
    ax2.plot(t_h, deltaf_plot, label='Tonic (deltaF)', color='black', lw=0.8)
    ax2.set_ylabel("deltaF")
    ax2.set_title(f"Tonic Output (Preserved Slow Dynamics) - {roi}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    ax2.set_xlabel("Time (Hours)")

    print(f"PLOT_TIMING STEP script=plot_tonic_48h.py step=plotting elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"PLOT_TIMING STEP script=plot_tonic_48h.py step=figure_save elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
    print(f"Saved {out_path}")
    plt.close(fig)


# ======================================================================
# Main Driver
# ======================================================================

def main():
    t_start = time.perf_counter()
    print("PLOT_TIMING START script=plot_tonic_48h.py", flush=True)
    args = parse_args()

    traces_dir = os.path.join(args.analysis_out, 'traces')

    # --- Stage 1: Discovery ---
    files = discover_trace_files(traces_dir)
    roi, col_time, col_sig, col_uv, col_deltaF, usecols = resolve_roi_and_columns(
        files[0], args.roi
    )
    print(f"PLOT_TIMING STEP script=plot_tonic_48h.py step=discovery elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # --- Stage 2a: CSV Reading ---
    list_sig, list_uv, list_deltaF = read_chunks(
        files, col_time, col_sig, col_uv, col_deltaF, usecols
    )
    print(f"PLOT_TIMING STEP script=plot_tonic_48h.py step=csv_read elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # --- Stage 2b: Assembly & Decimation ---
    continuous_time, sig_raw, uv_raw, deltaf_val = assemble_arrays(
        list_sig, list_uv, list_deltaF, files, col_time, usecols
    )

    t_h = continuous_time / 3600.0
    n_pts = len(t_h)
    decimate = max(1, n_pts // 100000)

    t_plot = t_h[::decimate]
    sig_plot = sig_raw[::decimate]
    uv_plot = uv_raw[::decimate]
    deltaf_plot = deltaf_val[::decimate]

    print(f"PLOT_TIMING STEP script=plot_tonic_48h.py step=assembly elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # --- Stage 3: Output path resolution ---
    if args.out:
        out_path = args.out
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.join(args.analysis_out, 'tonic_qc')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"tonic_48h_overview_{roi}.png")

    # --- Stage 4: Plotting & Save ---
    plot_tonic_overview(t_plot, sig_plot, uv_plot, deltaf_plot, roi, out_path, t_start)

    print(f"PLOT_TIMING DONE script=plot_tonic_48h.py total_sec={time.perf_counter() - t_start:.3f}", flush=True)


if __name__ == '__main__':
    main()
