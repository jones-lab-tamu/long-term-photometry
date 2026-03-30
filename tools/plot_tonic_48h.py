#!/usr/bin/env python3
"""
Tonic QC Overview Plotter
=========================

Stitches all chunks for a given ROI from the HDF5 cache to visualize long-session
tonic structure. Serves as the "Ground Truth Anchor" to verify synthetic generator output.

Inputs:
- <analysis-out>/tonic_out/tonic_trace_cache.h5

Outputs:
- <analysis-out>/tonic_qc/tonic_48h_overview_<roi>.png (legacy filename retained)
"""

import os
import sys

# Ensure photometry_pipeline is in the path when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

from photometry_pipeline.core.tonic_output import (
    TONIC_OUTPUT_MODE_PRESERVE_RAW,
    apply_tonic_output_mode_to_session,
    normalize_tonic_output_mode,
    tonic_output_mode_label,
)
from photometry_pipeline.core.tonic_timeline import (
    TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
    TONIC_TIMELINE_MODE_REAL_ELAPSED,
    build_tonic_chunk_time_axis,
    normalize_tonic_timeline_mode,
    remap_gapfree_axis_to_elapsed_span,
    tonic_timeline_axis_label,
    tonic_timeline_mode_label,
)
from photometry_pipeline.io.hdf5_cache_reader import (
    open_tonic_cache,
    resolve_cache_roi,
    iter_cache_chunks_for_roi
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot tonic overview from HDF5 cache chunks.")
    parser.add_argument('--analysis-out', required=True, help="Path to _analysis directory")
    parser.add_argument('--roi', help="Specific ROI to plot (e.g., Region0). Auto-selected if omitted.")
    parser.add_argument('--out', help="Explicit output file path. Overrides default.")
    parser.add_argument('--input', help="Original input directory for rigorous schedule recovery")
    parser.add_argument('--format', help="Format string for robust file recovery", default='auto')
    parser.add_argument('--sessions-per-hour', type=float, help="Duty cycle parameter to compute offline gap strides")
    parser.add_argument(
        '--tonic-output-mode',
        default=TONIC_OUTPUT_MODE_PRESERVE_RAW,
        help=(
            "Tonic output mode. "
            "Supported: preserve_raw_session_shape, "
            "flatten_session_bleach_preserve_session_baseline"
        ),
    )
    parser.add_argument(
        '--tonic-timeline-mode',
        default=TONIC_TIMELINE_MODE_REAL_ELAPSED,
        help=(
            "Tonic timeline mode. "
            "Supported: real_elapsed_time, gap_free_elapsed_time "
            "(legacy alias compressed_recording_time also accepted)"
        ),
    )
    return parser.parse_args()


# ======================================================================
# Stage 1: Assembly
# ======================================================================

def assemble_arrays(cache, roi, args=None):
    """Iterate chunks from the cache and build a discontinuous, accurate absolute time axis."""
    if args is None:
        class _Args:
            input = None
            format = 'auto'
            sessions_per_hour = None
            tonic_output_mode = TONIC_OUTPUT_MODE_PRESERVE_RAW
            tonic_timeline_mode = TONIC_TIMELINE_MODE_REAL_ELAPSED
            include_visual_separators = False
        args = _Args()
    include_visual_separators = bool(getattr(args, "include_visual_separators", True))

    list_time = []
    list_sig = []
    list_uv = []
    list_deltaF = []
    mode = normalize_tonic_output_mode(getattr(args, 'tonic_output_mode', None))
    timeline_mode = normalize_tonic_timeline_mode(getattr(args, 'tonic_timeline_mode', None))
    fallback_count = 0

    prev_chunk_end_sec = None
    prev_dt_sec = None
    prev_real_end_sec = None
    prev_real_dt_sec = None
    real_elapsed_start_sec = None
    real_elapsed_end_sec = None
    stride_s = (3600.0 / args.sessions_per_hour) if args.sessions_per_hour else None
    
    cids = list(cache['meta']['chunk_ids'][:]) if 'meta' in cache and 'chunk_ids' in cache['meta'] else []
    source_files = []
    if "meta" in cache and "source_files" in cache["meta"]:
        source_files = [f.decode('utf-8') if isinstance(f, bytes) else f for f in cache['meta']['source_files'][:]]

    from photometry_pipeline.utils.timeline import map_cached_sources_to_schedule_positions
    actual_positions = map_cached_sources_to_schedule_positions(
        args.input, args.format, source_files, cids
    ) if args.input else cids

    required_fields = ['time_sec', 'sig_raw', 'uv_raw', 'deltaF']

    for i, (t, s, u, d) in enumerate(iter_cache_chunks_for_roi(cache, roi, required_fields)):
        # Reconstruct exactly matching positional chronology
        cid = cids[i] if i < len(cids) else i
        actual_schedule_idx = actual_positions[i] if (args.input and i < len(actual_positions)) else None
        t_abs_real, timeline_state_real = build_tonic_chunk_time_axis(
            time_sec_local=t,
            timeline_mode_raw=TONIC_TIMELINE_MODE_REAL_ELAPSED,
            chunk_sequence_index=cid,
            actual_schedule_index=actual_schedule_idx,
            stride_sec=stride_s,
            prev_chunk_end_sec=prev_real_end_sec,
            prev_dt_sec=prev_real_dt_sec,
        )
        prev_real_end_sec = timeline_state_real["prev_chunk_end_sec"]
        prev_real_dt_sec = timeline_state_real["prev_dt_sec"]
        if real_elapsed_start_sec is None:
            real_elapsed_start_sec = float(t_abs_real[0])
        real_elapsed_end_sec = float(t_abs_real[-1])

        if timeline_mode == TONIC_TIMELINE_MODE_REAL_ELAPSED:
            t_abs = t_abs_real
            sep_dt_sec = prev_real_dt_sec
        else:
            t_abs, timeline_state = build_tonic_chunk_time_axis(
                time_sec_local=t,
                timeline_mode_raw=TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
                chunk_sequence_index=cid,
                actual_schedule_index=actual_schedule_idx,
                stride_sec=stride_s,
                prev_chunk_end_sec=prev_chunk_end_sec,
                prev_dt_sec=prev_dt_sec,
            )
            prev_chunk_end_sec = timeline_state["prev_chunk_end_sec"]
            prev_dt_sec = timeline_state["prev_dt_sec"]
            sep_dt_sec = prev_dt_sec

        s_out, u_out, d_out, mode_meta = apply_tonic_output_mode_to_session(
            time_sec=t,
            sig_raw=s,
            uv_raw=u,
            deltaf_raw=d,
            mode_raw=mode,
        )
        fallback_count += int(mode_meta.get("fallback_count", 0))

        list_time.append(t_abs)
        list_sig.append(s_out)
        list_uv.append(u_out)
        list_deltaF.append(d_out)

        if include_visual_separators:
            # Inject visual trace separator gap for discontinuous timeline plotting rendering
            list_time.append(np.array([t_abs[-1] + sep_dt_sec]))
            list_sig.append(np.array([np.nan]))
            list_uv.append(np.array([np.nan]))
            list_deltaF.append(np.array([np.nan]))

    if not list_sig:
        print("CRITICAL: No valid chunks found for ROI.")
        sys.exit(1)

    continuous_time = np.concatenate(list_time)
    sig_raw = np.concatenate(list_sig)
    uv_raw = np.concatenate(list_uv)
    deltaf_val = np.concatenate(list_deltaF)
    if include_visual_separators:
        continuous_time = continuous_time[:-1]  # drop trailing separator
        sig_raw = sig_raw[:-1]
        uv_raw = uv_raw[:-1]
        deltaf_val = deltaf_val[:-1]

    if (
        timeline_mode != TONIC_TIMELINE_MODE_REAL_ELAPSED
        and real_elapsed_start_sec is not None
        and real_elapsed_end_sec is not None
    ):
        continuous_time = remap_gapfree_axis_to_elapsed_span(
            continuous_time,
            elapsed_start_sec=real_elapsed_start_sec,
            elapsed_end_sec=real_elapsed_end_sec,
        )

    if fallback_count > 0:
        print(
            f"WARNING: tonic output mode '{mode}' used per-session fallback "
            f"{fallback_count} time(s).",
            flush=True,
        )

    return continuous_time, sig_raw, uv_raw, deltaf_val


# ======================================================================
# Stage 2: Plotting & Save
# ======================================================================

def build_overview_prefix_from_time_hours(time_hours: np.ndarray) -> str:
    finite_time = np.asarray(time_hours, dtype=float)
    finite_time = finite_time[np.isfinite(finite_time)]
    if finite_time.size >= 2:
        duration_h = float(np.max(finite_time) - np.min(finite_time))
        duration_txt = f"{duration_h:.1f}".rstrip("0").rstrip(".")
        return f"{duration_txt} h Overview"
    return "Overview"


def plot_tonic_overview(
    t_h,
    sig_plot,
    uv_plot,
    deltaf_plot,
    roi,
    out_path,
    t_start,
    mode_label,
    timeline_mode_label,
    timeline_axis_label,
):
    """Render the two-panel tonic overview and save."""
    overview_prefix = build_overview_prefix_from_time_hours(t_h)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 1. Raw
    ax1.plot(t_h, sig_plot, label='Sig', color='green', lw=0.5)
    ax1.plot(t_h, uv_plot, label='Iso', color='purple', lw=0.5, alpha=0.8)
    ax1.set_ylabel("Raw Signal")
    ax1.set_title(f"{overview_prefix} - {roi} - Sig/Iso ({mode_label}; {timeline_mode_label})")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Tonic Output (deltaF)
    ax2.plot(t_h, deltaf_plot, label='Tonic (deltaF)', color='black', lw=0.8)
    ax2.set_ylabel("deltaF")
    ax2.set_title(f"Tonic Output (deltaF; {mode_label}; {timeline_mode_label}) - {roi}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    ax2.set_xlabel(timeline_axis_label)

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
    mode = normalize_tonic_output_mode(args.tonic_output_mode)
    mode_label = tonic_output_mode_label(mode)
    timeline_mode = normalize_tonic_timeline_mode(args.tonic_timeline_mode)
    timeline_mode_name = tonic_timeline_mode_label(timeline_mode)
    timeline_axis = tonic_timeline_axis_label(timeline_mode)

    cache_path = os.path.join(args.analysis_out, 'tonic_trace_cache.h5')

    # --- Stage 1: Discovery ---
    cache = open_tonic_cache(cache_path)
    roi = resolve_cache_roi(cache, args.roi)
    
    print(f"PLOT_TIMING STEP script=plot_tonic_48h.py step=discovery elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # --- Stage 2a: Cache Reading & Assembly ---
    continuous_time, sig_raw, uv_raw, deltaf_val = assemble_arrays(cache, roi, args)
    cache.close()
    
    print(f"PLOT_TIMING STEP script=plot_tonic_48h.py step=cache_read elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # --- Stage 2b: Decimation ---
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
    plot_tonic_overview(
        t_plot,
        sig_plot,
        uv_plot,
        deltaf_plot,
        roi,
        out_path,
        t_start,
        mode_label,
        timeline_mode_name,
        timeline_axis,
    )

    print(f"PLOT_TIMING DONE script=plot_tonic_48h.py total_sec={time.perf_counter() - t_start:.3f}", flush=True)


if __name__ == '__main__':
    main()
