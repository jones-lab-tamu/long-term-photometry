#!/usr/bin/env python3
"""
Phasic Day-Plot Bundle Driver
=============================

Unified driver for Region-level phasic day plots (Part 2).
Replaces the three separate subprocess calls with a single process that:
1. Discovers and lays out chunk files once.
2. Loads chunk CSVs into memory EXACTLY ONCE per ROI.
3. Generates the three daily plot families from the cached data:
   - phasic_dFF_day_{d:03d}.png
   - phasic_sig_iso_day_{d:03d}.png
   - phasic_stacked_day_{d:03d}.png

This drastically reduces I/O by preventing 3x redundant reads of the heavy trace CSVs.
"""

import os
import sys
import argparse
import logging
import math
import time
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Ensure repo root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Core Imports
from photometry_pipeline.config import Config
from photometry_pipeline.viz.phasic_data_prep import (
    discover_chunks, build_feature_map, resolve_roi, compute_day_layout
)

# We need the strict peak verification logic from the qc grid script
# We can just import it from there to avoid duplication, OR recreate it here if preferred.
# For isolation, it's safer to reproduce the peak verification here or extract it to a helper.
# Since we must NOT modify the old scripts and we want minimal new helpers, we'll
# recreate the minimal necessary `verify_peak_count_strict` and `get_local_peak_indices` 
# logic here.
from photometry_pipeline.core.feature_extraction import extract_features
from photometry_pipeline.core.types import Chunk
from scipy.signal import find_peaks

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Phasic Day-Plot Generator")
    parser.add_argument('--analysis-out', required=True, help="Path to analysis output directory")
    parser.add_argument('--roi', required=True, help="Specific ROI to plot")
    parser.add_argument('--output-dir', required=True, help="Output directory for the day plots")
    parser.add_argument('--sessions-per-hour', type=int, required=True, help="Grid columns")
    
    # Optional / Tuning
    parser.add_argument('--session-duration-s', type=float, default=None, help="Expected session duration in seconds")
    parser.add_argument('--smooth-window-s', type=float, default=1.0, help="Smoothing window for stacked plots")
    parser.add_argument('--dpi', type=int, default=150, help="Output DPI")
    parser.add_argument('--signal', default='auto', help="Trace column for dFF (default: auto detects *_dff)")
    
    # Enable/Disable Families (default: all generated)
    parser.add_argument('--write-dff-grid', action='store_true', default=True, help="(default true)")
    parser.add_argument('--no-write-dff-grid', dest='write_dff_grid', action='store_false')
    
    parser.add_argument('--write-sig-iso-grid', action='store_true', default=True, help="(default true)")
    parser.add_argument('--no-write-sig-iso-grid', dest='write_sig_iso_grid', action='store_false')
    
    parser.add_argument('--write-stacked', action='store_true', default=True, help="(default true)")
    parser.add_argument('--no-write-stacked', dest='write_stacked', action='store_false')
    
    # Limits for dFF Grid
    parser.add_argument('--dff-y-percentile-low', type=float, default=0.5)
    parser.add_argument('--dff-y-percentile-high', type=float, default=99.9)
    parser.add_argument('--dff-y-pad-frac', type=float, default=0.10)
    
    return parser.parse_args()


# ======================================================================
# Verification & Audit Helpers
# ======================================================================

def load_config_obj(out_dir):
    path = os.path.join(out_dir, "config_used.yaml")
    if not os.path.exists(path):
        print(f"CRITICAL: config_used.yaml not found in {out_dir}.")
        sys.exit(1)
    return Config.from_yaml(path)

def determine_signal_column(cols, roi, requested='auto'):
    if requested != 'auto':
        if requested in cols: return requested
        print(f"CRITICAL: Requested signal '{requested}' not found in columns")
        sys.exit(1)
    cand = f"{roi}_dff"
    if cand in cols: return cand
    match = [c for c in cols if roi in c and c.endswith('_dff')]
    if match: return match[0]
    print(f"CRITICAL: Could not auto-detect signal for ROI {roi}")
    sys.exit(1)

def check_monotonicity(time_arr):
    return np.all(np.diff(time_arr) > 0)

def check_continuity(time_arr, expected_dt):
    diffs = np.diff(time_arr)
    return np.all(diffs < (2.0 * expected_dt))

def get_local_peak_indices(trace_arr, fs, config):
    is_valid = np.isfinite(trace_arr)
    clean_trace = trace_arr[is_valid]
    if len(clean_trace) == 0: return np.array([], dtype=int)
    
    method = config.peak_threshold_method
    if method == 'absolute':
        thresh = getattr(config, 'peak_threshold_abs', 0.0)
    elif method == 'mean_std':
        mu, sigma = np.mean(clean_trace), np.std(clean_trace)
        thresh = mu + config.peak_threshold_k * sigma
    elif method == 'percentile':
        thresh = np.percentile(clean_trace, config.peak_threshold_percentile)
    elif method == 'median_mad':
        median = np.median(clean_trace)
        mad = np.median(np.abs(clean_trace - median))
        sigma_robust = 1.4826 * mad
        if sigma_robust == 0:
             thresh = median if config.peak_threshold_k == 0 else float('inf') 
        else:
             thresh = median + config.peak_threshold_k * sigma_robust

    dist_samples = max(1, int(config.peak_min_distance_sec * fs))
    padded = np.concatenate(([False], is_valid, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    all_peaks = []
    for s, e in zip(starts, ends):
        seg_trace = trace_arr[s:e]
        p_inds, _ = find_peaks(seg_trace, height=thresh, distance=dist_samples)
        all_peaks.append(p_inds + s)
        
    return np.concatenate(all_peaks) if all_peaks else np.array([], dtype=int)

def verify_peak_count_strict(trace_arr, time_arr, fs, config, expected_count, roi, cid, src_file):
    if pd.isna(expected_count):
        print(f"CRITICAL: Expected count is NaN for Chunk {cid}.")
        sys.exit(1)

    dff_in = trace_arr.reshape(-1, 1)
    raw = np.zeros_like(dff_in)
    chunk = Chunk(
        chunk_id=cid, source_file=src_file, format='rwd', time_sec=time_arr,
        dff=dff_in, uv_raw=raw, sig_raw=raw, fs_hz=fs, channel_names=[roi]
    )
    df_feat = extract_features(chunk, config)
    if df_feat.empty:
        print(f"CRITICAL: extract_features returned empty for Chunk {cid}")
        sys.exit(1)
        
    pipeline_count = df_feat.iloc[0]['peak_count']
    if pipeline_count != expected_count:
        print(f"CRITICAL: Verification Failed for Chunk {cid}, ROI {roi} (expected {expected_count}, got {pipeline_count})")
        sys.exit(1)

    local_peaks = get_local_peak_indices(trace_arr, fs, config)
    if len(local_peaks) != pipeline_count:
        print(f"CRITICAL: Plotting Logic Mismatch for Chunk {cid}, ROI {roi} ({pipeline_count} vs {len(local_peaks)})")
        sys.exit(1)
        
    return local_peaks

def infer_fs(time_arr, config, context=""):
    if len(time_arr) < 2: return getattr(config, 'sampling_rate_hz_fallback', config.target_fs_hz)
    dt = np.median(np.diff(time_arr))
    if dt <= 0 or not np.isfinite(dt): return getattr(config, 'sampling_rate_hz_fallback', config.target_fs_hz)
    return 1.0 / dt


# ======================================================================
# Main Driver
# ======================================================================

def main():
    t_start = time.perf_counter()
    print("PLOT_TIMING START script=plot_phasic_dayplot_bundle.py", flush=True)
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = load_config_obj(args.analysis_out)
    feats_path = os.path.join(args.analysis_out, 'features', 'features.csv')
    traces_dir = os.path.join(args.analysis_out, 'traces')
    
    needs_dff_trace = args.write_dff_grid or args.write_stacked
    needs_peak_verification = args.write_dff_grid
    
    try:
        chunk_entries = discover_chunks(traces_dir)
    except RuntimeError as e:
        print(f"CRITICAL: {e}")
        sys.exit(1)
        
    # 1. Read first chunk header exactly once
    first_trace = chunk_entries[0][1]
    df0 = pd.read_csv(first_trace, nrows=1)
    chunk0_cols = df0.columns.tolist()

    # 2. Resolve ROI
    plot_roi = args.roi
    suffix = '_dff' if needs_dff_trace else '_sig_raw'
    
    # If explicit roi is passed, validate it. Otherwise detect.
    if not plot_roi:
        cands = [c.replace(suffix, '') for c in chunk0_cols if suffix in c]
        if not cands:
             print(f"CRITICAL: Could not auto-detect ROI using suffix {suffix} in {first_trace}")
             sys.exit(1)
        plot_roi = sorted(cands)[0]
    else:
        if f"{plot_roi}{suffix}" not in chunk0_cols:
             print(f"CRITICAL: Requested ROI {plot_roi} lacks expected column {plot_roi}{suffix}")
             sys.exit(1)
             
    print(f"Plots using ROI: {plot_roi}")
        
    # 3. Features Map (Conditional)
    if needs_peak_verification:
        if not os.path.exists(feats_path):
             print(f"CRITICAL: features.csv not found but dFF outputs enabled.")
             sys.exit(1)
        feat_map = build_feature_map(feats_path, roi=plot_roi)
    else:
        feat_map = {}
        
    pds = compute_day_layout(chunk_entries, feat_map, plot_roi, args.sessions_per_hour)
    sph = pds.sessions_per_hour
    
    # 4. Identify signals
    col_dff = determine_signal_column(chunk0_cols, plot_roi, args.signal) if needs_dff_trace else None
    col_sig = f"{plot_roi}_sig_raw"
    col_uv = f"{plot_roi}_uv_raw"

    # 5. Extract target usecols once
    time_cands = ['time_sec', getattr(config, 'rwd_time_col', ''), 'Time(s)']
    col_time = next((c for c in time_cands if c in chunk0_cols), None)

    usecols = []
    if col_time: usecols.append(col_time)
    
    if args.write_sig_iso_grid:
        usecols.append(col_sig)
        usecols.append(col_uv)
        
    if needs_dff_trace and col_dff:
        usecols.append(col_dff)
        
    usecols = list(dict.fromkeys(usecols))

    print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=discovery elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # ------------------------------------------------------------------
    # 1. Single-Pass CSV Reading
    # ------------------------------------------------------------------
    raw_chunks = []
    
    for cr in pds.chunks:
        try:
            tdf = pd.read_csv(cr.trace_path, usecols=usecols)
            rec = {
                'cr': cr,
                'x': tdf[col_time].values if col_time and col_time in tdf.columns else None,
                'y_sig': tdf[col_sig].values if col_sig and col_sig in tdf.columns else None,
                'y_uv': tdf[col_uv].values if col_uv and col_uv in tdf.columns else None,
                'y_dff': tdf[col_dff].values if col_dff and col_dff in tdf.columns else None,
                'N': len(tdf)
            }
            raw_chunks.append(rec)
        except Exception as e:
            print(f"CRITICAL: Error reading chunk {cr.chunk_id}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=csv_read elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # ------------------------------------------------------------------
    # 2. Verification
    # ------------------------------------------------------------------
    for rec in raw_chunks:
        cr = rec['cr']
        x = rec['x']
        
        fs = infer_fs(x, config, context=f"Chunk {cr.chunk_id}") if x is not None else config.target_fs_hz
        if x is None:
            x = np.arange(rec['N']) / fs
            rec['x'] = x
            
        # Monotonicity & Continuity & Duration Audits (from session grid)
        if not check_monotonicity(x):
            print(f"CRITICAL: Non-monotonic time in {cr.trace_path}")
            sys.exit(1)
        
        duration = x[-1] - x[0]
        if args.session_duration_s is not None:
            expected = args.session_duration_s
            tol = max(2.0, 0.005 * expected)
            if abs(duration - expected) > tol:
                print(f"CRITICAL: Duration mismatch. Expected ~{expected:.2f}s, got {duration:.2f}s")
                sys.exit(1)
        else:
            if not (590 <= duration <= 610): # Strict fallback
                print(f"CRITICAL: Invalid duration {duration:.2f}s (Expected ~600s)")
                sys.exit(1)

        dt_median = np.median(np.diff(x))
        if not check_continuity(x, dt_median):
            print(f"CRITICAL: Discontinuity detected in {cr.trace_path}")
            sys.exit(1)
            
        # Verify peaks
        if needs_peak_verification:
            feat_row = feat_map.get((cr.chunk_id, plot_roi))
            exp_count = feat_row['peak_count'] if feat_row is not None else np.nan
                
            peak_indices = verify_peak_count_strict(
                rec['y_dff'], x, fs, config, exp_count, plot_roi, cr.chunk_id, cr.source_file
            )
            rec['peak_indices'] = peak_indices
            rec['exp_count'] = exp_count

    print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=verification elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # ------------------------------------------------------------------
    # 3. Cache Build
    # ------------------------------------------------------------------
    cached_data = []
    cached_by_day = {}
    global_dff_values = []
    
    for rec in raw_chunks:
        cr = rec['cr']
        x = rec['x']
        t_norm = x - x[0]
        
        c_rec = {
            'day': cr.day_idx,
            'hour': cr.hour_idx,
            'col': cr.hour_rank,
            'chunk_id': cr.chunk_id,
            't': t_norm
        }
        if args.write_sig_iso_grid:
            c_rec['sig'] = rec['y_sig']
            c_rec['uv'] = rec['y_uv']
        if needs_dff_trace:
            c_rec['dff'] = rec['y_dff']
        if needs_peak_verification:
            c_rec['peak_indices'] = rec.get('peak_indices', np.array([], dtype=int))
            c_rec['count'] = rec.get('exp_count', np.nan)
            
        cached_data.append(c_rec)
        cached_by_day.setdefault(cr.day_idx, []).append(c_rec)
        
        if needs_peak_verification:
            y_dff = rec['y_dff']
            global_dff_values.append(y_dff[np.isfinite(y_dff)])

    print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=cache_build elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # ------------------------------------------------------------------
    # 4. Global Limits
    # ------------------------------------------------------------------
    if global_dff_values:
        flat_y = np.concatenate(global_dff_values)
        if len(flat_y) == 0:
            global_ymin, global_ymax = -1, 1
        else:
            p_lo, p_hi = args.dff_y_percentile_low, args.dff_y_percentile_high
            global_ymin, global_ymax = np.percentile(flat_y, [p_lo, p_hi])
            yrange = global_ymax - global_ymin
            pad = args.dff_y_pad_frac * yrange
            if pad == 0: pad = 0.1
            global_ymin -= pad
            global_ymax += pad
    else:
        global_ymin, global_ymax = -1, 1

    unique_days = sorted(pds.chunks_by_day.keys())
    print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=global_limits elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
    
    
    # ------------------------------------------------------------------
    # 2. Render Family 1: dFF Grid
    # ------------------------------------------------------------------
    if args.write_dff_grid:
        for d in unique_days:
            day_items = cached_by_day.get(d, [])
            if not day_items: continue
                
            fig, axes = plt.subplots(nrows=24, ncols=sph, figsize=(4*sph + 2, 24), sharex=True)
            if sph == 1: axes = axes.reshape(-1, 1)
                
            fig.suptitle(f"Phasic QC - Day {d} - ROI {plot_roi} - Mode: DFF", fontsize=16)
            y_span = global_ymax - global_ymin
            eps = 0.01 * y_span if y_span > 0 else 1e-6
            
            for p in day_items:
                h, c = p['hour'], p['col']
                if c >= sph: continue
                ax = axes[h, c]
                ax.set_title(f"Chunk {p['chunk_id']}", fontsize=6, pad=2)
                
                ax.plot(p['t'], p['dff'], 'k', lw=0.8)
                ax.set_ylim(global_ymin, global_ymax)
                
                # Peak Overlays (Clipped vs unclipped)
                p_idxs = p['peak_indices']
                n_clipped = 0
                if len(p_idxs) > 0:
                    px = p['t'][p_idxs]
                    py_true = p['dff'][p_idxs]
                    py_plot = np.clip(py_true, global_ymin + eps, global_ymax - eps)
                    
                    mask_hi = py_true > (global_ymax - eps)
                    mask_lo = py_true < (global_ymin + eps)
                    mask_ok = ~(mask_hi | mask_lo)
                    
                    if np.any(mask_ok): ax.scatter(px[mask_ok], py_plot[mask_ok], s=10, c='red', alpha=0.6, zorder=3)
                    if np.any(mask_hi): ax.scatter(px[mask_hi], py_plot[mask_hi], s=12, marker='^', c='red', alpha=0.8, zorder=4)
                    if np.any(mask_lo): ax.scatter(px[mask_lo], py_plot[mask_lo], s=12, marker='v', c='red', alpha=0.8, zorder=4)
                    n_clipped = np.sum(mask_hi) + np.sum(mask_lo)
                    
                # Annotation
                val = p['count']
                txt = "peaks=NaN" if pd.isna(val) else f"peaks={int(val)}"
                if n_clipped > 0: txt += f"\n({n_clipped} clipped)"
                color = 'red' if pd.isna(val) else 'blue'
                ax.text(0.95, 0.9, txt, transform=ax.transAxes, ha='right', va='top', fontsize=8, color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                if c == 0: ax.set_ylabel(f"H{h:02d}", rotation=0, labelpad=15, va='center', fontweight='bold')
                    
            # Cleanup Empty
            for r in range(24):
                for c in range(sph):
                    if not any(pi['hour'] == r and pi['col'] == c for pi in day_items): axes[r, c].axis('off')
                        
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_path = os.path.join(args.output_dir, f"phasic_dFF_day_{d:03d}.png")
            print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=plotting family=dff day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
            plt.savefig(out_path, dpi=args.dpi)
            print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=figure_save family=dff day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
            plt.close(fig)
            

    # ------------------------------------------------------------------
    # 3. Render Family 2: Sig/Iso Grid
    # ------------------------------------------------------------------
    if args.write_sig_iso_grid:
        for d in unique_days:
            day_items = cached_by_day.get(d, [])
            if not day_items: continue
                
            fig, axes = plt.subplots(nrows=24, ncols=sph, figsize=(4*sph + 2, 24), sharex=True)
            if sph == 1: axes = axes.reshape(-1, 1)
            fig.suptitle(f"Day {d} Raw/Iso - {plot_roi}", fontsize=16)
            
            for p in day_items:
                ax = axes[p['hour'], p['col']]
                ax.plot(p['t'], p['sig'], 'g', lw=0.5, label='Sig')
                ax.plot(p['t'], p['uv'], 'm', lw=0.5, label='Iso')
                if p['t'].max() > 550: ax.set_xlim(0, 600)
                
            for i in range(24): axes[i,0].set_ylabel(f"H{i:02d}", rotation=0, labelpad=20)
                
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            out_path = os.path.join(args.output_dir, f"phasic_sig_iso_day_{d:03d}.png")
            print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=plotting family=sig_iso day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
            plt.savefig(out_path, dpi=args.dpi)
            print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=figure_save family=sig_iso day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
            plt.close(fig)
            
            
    # ------------------------------------------------------------------
    # 4. Render Family 3: Stacked Smoothed
    # ------------------------------------------------------------------
    if args.write_stacked:
        # Pre-smooth the dFF for all chunks
        smoothed_data = {}
        for c in cached_data:
            mask = np.isfinite(c['dff'])
            y = c['dff'][mask]
            t = c['t'][mask]
            if len(y) < 2: continue
            
            dt = np.median(np.diff(t))
            fs = 1.0 / dt if dt > 0 else 1.0
            w_samples = max(1, int(round(fs * args.smooth_window_s)))
            y_smooth = uniform_filter1d(y, size=w_samples)
            smoothed_data[c['chunk_id']] = (t, y_smooth)
            
        for d in unique_days:
            # Sort chronologically
            day_items = sorted(cached_by_day.get(d, []), key=lambda x: x['chunk_id'])
            traces = [smoothed_data[c['chunk_id']] for c in day_items if c['chunk_id'] in smoothed_data]
            if not traces: continue
                
            fig, ax = plt.subplots(figsize=(6, len(traces)*0.3 + 2))
            
            ranges = [np.ptp(tr[1]) for tr in traces]
            avg_rng = np.median(ranges) if ranges else 1.0
            step = max(0.1, avg_rng * 0.8)
            
            for i, (t, y) in enumerate(traces):
                offset = (len(traces) - 1 - i) * step
                ax.plot(t, y + offset, 'k', lw=0.5)
                
            ax.set_yticks([])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"Sessions ({len(traces)})")
            ax.set_title(f"Day {d} Stacked (Smoothed {args.smooth_window_s}s) - {plot_roi}")
            
            plt.tight_layout()
            out_path = os.path.join(args.output_dir, f"phasic_stacked_day_{d:03d}.png")
            print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=plotting family=stacked day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
            plt.savefig(out_path, dpi=args.dpi)
            print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=figure_save family=stacked day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
            plt.close(fig)

    print(f"PLOT_TIMING DONE script=plot_phasic_dayplot_bundle.py total_sec={time.perf_counter() - t_start:.3f}", flush=True)

if __name__ == '__main__':
    main()
