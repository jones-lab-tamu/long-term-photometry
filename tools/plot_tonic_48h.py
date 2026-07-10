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
import math

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
from photometry_pipeline.viz.phasic_data_prep import (
    build_authoritative_plot_sessions,
)


TONIC_OVERVIEW_TARGET_DISPLAY_POINTS = 30000


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
    parser.add_argument(
        '--export-display-series-csv',
        action='store_true',
        help=(
            "Advanced export: write long-format CSV of tonic_overview displayed "
            "series (post-display decimation)."
        ),
    )
    parser.add_argument(
        '--source-run-profile',
        default='unknown',
        help="Run profile label to stamp into display-series export metadata.",
    )
    return parser.parse_args()


# ======================================================================
# Stage 1: Assembly
# ======================================================================

def assemble_arrays(cache, roi, args=None, *, return_missing_metadata=False):
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
    authoritative_sessions = build_authoritative_plot_sessions(
        args.analysis_out, cids, source_files
    ) if getattr(args, "analysis_out", None) else None
    if authoritative_sessions is not None:
        cache_to_session = {
            int(item["cache_chunk_id"]): int(item["session_index"])
            for item in authoritative_sessions
            if item.get("cache_chunk_id") is not None
        }
        actual_positions = [cache_to_session.get(int(cid), int(cid)) for cid in cids]
        missing_sessions = [
            item for item in authoritative_sessions if item.get("status") != "valid"
        ]
        # The authoritative index is also the timing contract.  When the CLI
        # does not receive ``--sessions-per-hour``, derive a regular stride
        # from validated expected starts so a missing middle session cannot be
        # compressed into the next cache contribution.
        if stride_s is None:
            starts = sorted(
                item.get("expected_start_time")
                for item in authoritative_sessions
                if item.get("expected_start_time") is not None
            )
            deltas = [
                float((b - a).total_seconds())
                for a, b in zip(starts, starts[1:])
                if float((b - a).total_seconds()) > 0
            ]
            if deltas:
                stride_s = float(np.median(deltas))
        starts_with_index = [
            (item.get("expected_start_time"), int(item.get("session_index", 0)))
            for item in authoritative_sessions
            if item.get("expected_start_time") is not None
        ]
        if starts_with_index and stride_s is not None:
            # Use the first expected session as the stable origin.  The hidden
            # metadata below lets marker placement retain the full chronology
            # without changing the public session record schema.
            origin = min(
                float((start - starts_with_index[0][0]).total_seconds())
                - float(index) * float(stride_s)
                for start, index in starts_with_index
            )
            for item in missing_sessions:
                item["_timeline_origin_sec"] = origin
                item["_timeline_stride_sec"] = float(stride_s)
                item["_timeline_first_start_time"] = starts_with_index[0][0]
    else:
        actual_positions = map_cached_sources_to_schedule_positions(
            args.input, args.format, source_files, cids
        ) if args.input else cids
        missing_sessions = []

    required_fields = ['time_sec', 'sig_raw', 'uv_raw', 'deltaF']

    for i, (t, s, u, d) in enumerate(iter_cache_chunks_for_roi(cache, roi, required_fields)):
        # Reconstruct exactly matching positional chronology
        cid = cids[i] if i < len(cids) else i
        actual_schedule_idx = (
            actual_positions[i]
            if authoritative_sessions is not None and i < len(actual_positions)
            else (actual_positions[i] if args.input and i < len(actual_positions) else None)
        )
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

    if return_missing_metadata:
        return continuous_time, sig_raw, uv_raw, deltaf_val, missing_sessions
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


def compute_tonic_overview_display_decimation(
    n_points: int,
    target_points: int = TONIC_OVERVIEW_TARGET_DISPLAY_POINTS,
) -> int:
    """Return stride decimation for tonic overview display series."""
    n_points = int(n_points)
    target_points = max(1, int(target_points))
    if n_points <= target_points:
        return 1
    return max(1, int(math.ceil(float(n_points) / float(target_points))))


def _build_missing_intervals(missing_sessions, *, sessions_per_hour=None):
    """Convert authoritative missing sessions to real elapsed plot intervals."""
    if not missing_sessions:
        return []
    starts = [
        item.get("expected_start_time")
        for item in missing_sessions
        if item.get("expected_start_time") is not None
    ]
    # Current authorized missing sessions always have validated timestamps.  A
    # conservative index-based fallback keeps legacy/unit callers deterministic.
    base = min(starts) if starts else None
    stride_hours = 1.0 / float(sessions_per_hour) if sessions_per_hour else 1.0
    intervals = []
    for item in missing_sessions:
        start = item.get("expected_start_time")
        origin = item.get("_timeline_origin_sec")
        first_start = item.get("_timeline_first_start_time")
        if first_start is not None and start is not None:
            x_hours = float((start - first_start).total_seconds()) / 3600.0
        elif origin is not None and start is not None:
            x_hours = float((start - base).total_seconds()) / 3600.0 - float(origin) / 3600.0
        elif base is not None and start is not None:
            x_hours = float((start - base).total_seconds()) / 3600.0
        else:
            x_hours = float(item.get("session_index", 0)) * stride_hours
        duration = item.get("expected_duration_sec")
        try:
            width_hours = float(duration) / 3600.0 if duration is not None else stride_hours
        except (TypeError, ValueError):
            width_hours = stride_hours
        intervals.append({
            "x_hours": x_hours + 0.5 * width_hours,
            "width_hours": max(width_hours, 1e-6),
            "session_index": int(item.get("session_index", 0)),
            "status": str(item.get("status", "missing_corrupted")),
            "missing_reason": str(item.get("missing_reason", "")),
        })
    return intervals


def _annotate_missing_intervals(ax, intervals):
    if not intervals:
        return
    y_top = ax.get_ylim()[1]
    for item in intervals:
        x = float(item["x_hours"])
        width = float(item["width_hours"])
        status = str(item.get("status", "missing_corrupted"))
        label = (
            "Final incomplete session excluded"
            if status == "authorized_final_exclusion"
            else "Missing/corrupted session"
        )
        ax.axvspan(
            x - 0.5 * width,
            x + 0.5 * width,
            color="#e9a06d",
            alpha=0.28,
            zorder=0,
        )
        ax.text(
            x,
            y_top,
            f"Session {int(item['session_index']) + 1}\n{label}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color="#963014",
            clip_on=True,
        )


def _crosses_missing_interval(x0: float, x1: float, intervals) -> bool:
    """Return whether an adjacent display pair crosses an approved gap."""
    if not (np.isfinite(x0) and np.isfinite(x1)):
        return False
    lo, hi = sorted((float(x0), float(x1)))
    for item in intervals or []:
        center = float(item["x_hours"])
        width = max(float(item["width_hours"]), 1e-12)
        left = center - 0.5 * width
        right = center + 0.5 * width
        if lo < right and hi > left:
            return True
    return False


def _split_display_trace_at_missing_intervals(
    t_plot,
    y_plot,
    missing_intervals=None,
):
    """Split a plotting series wherever it crosses an approved missing gap.

    This is deliberately plotting-only.  It preserves the supplied elapsed
    timestamps and never writes separators into the cache or scientific table.
    Existing non-finite samples (including visual separators from assembly) also
    terminate a segment.  The returned segments can therefore be passed to
    separate ``Axes.plot`` calls without allowing decimation to reconnect the
    samples on opposite sides of a gap.
    """
    t_arr = np.asarray(t_plot, dtype=np.float64)
    y_arr = np.asarray(y_plot, dtype=np.float64)
    if t_arr.shape != y_arr.shape:
        raise ValueError("Tonic display time and value arrays must have equal shape")
    if t_arr.ndim != 1:
        raise ValueError("Tonic display time and value arrays must be one-dimensional")

    # Preserve the historical single Line2D object for clean recordings.  Any
    # existing plotting-only NaN separators remain embedded in that object;
    # approved missing intervals are handled explicitly below.
    if not missing_intervals:
        return [(t_arr, y_arr)] if np.any(np.isfinite(t_arr) & np.isfinite(y_arr)) else []

    segments = []
    start = None
    previous_finite_index = None

    def _finish(end_index):
        nonlocal start
        if start is not None and end_index > start:
            segments.append((t_arr[start:end_index], y_arr[start:end_index]))
        start = None

    for index, (x_value, y_value) in enumerate(zip(t_arr, y_arr)):
        if not (np.isfinite(x_value) and np.isfinite(y_value)):
            _finish(index)
            previous_finite_index = None
            continue
        if start is None:
            start = index
        elif previous_finite_index is not None and _crosses_missing_interval(
            t_arr[previous_finite_index], x_value, missing_intervals
        ):
            _finish(index)
            start = index
        previous_finite_index = index

    _finish(len(t_arr))
    return segments


def _plot_tonic_series(ax, t_plot, y_plot, *, label, color, linewidth, missing_intervals=None, alpha=1.0):
    """Plot one tonic display series as gap-safe Line2D segments."""
    lines = []
    for segment_index, (t_segment, y_segment) in enumerate(
        _split_display_trace_at_missing_intervals(
            t_plot, y_plot, missing_intervals
        )
    ):
        lines.extend(
            ax.plot(
                t_segment,
                y_segment,
                label=label if segment_index == 0 else "_nolegend_",
                color=color,
                lw=linewidth,
                alpha=alpha,
            )
        )
    return lines


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
    missing_intervals=None,
):
    """Render the two-panel tonic overview and save."""
    overview_prefix = build_overview_prefix_from_time_hours(t_h)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 1. Raw
    _plot_tonic_series(
        ax1,
        t_h,
        sig_plot,
        label="Sig",
        color="green",
        linewidth=0.5,
        missing_intervals=missing_intervals,
    )
    _plot_tonic_series(
        ax1,
        t_h,
        uv_plot,
        label="Iso",
        color="purple",
        linewidth=0.5,
        alpha=0.8,
        missing_intervals=missing_intervals,
    )
    _annotate_missing_intervals(ax1, missing_intervals or [])
    ax1.set_ylabel("Raw Signal")
    ax1.set_title(f"{overview_prefix} - {roi} - Sig/Iso ({mode_label}; {timeline_mode_label})")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Tonic Output (deltaF)
    _plot_tonic_series(
        ax2,
        t_h,
        deltaf_plot,
        label="Tonic (deltaF)",
        color="black",
        linewidth=0.8,
        missing_intervals=missing_intervals,
    )
    _annotate_missing_intervals(ax2, missing_intervals or [])
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


def _write_tonic_display_series_csv(
    *,
    out_path: str,
    roi: str,
    source_run_profile: str,
    t_plot: np.ndarray,
    sig_plot: np.ndarray,
    uv_plot: np.ndarray,
    deltaf_plot: np.ndarray,
    decimate: int,
    missing_intervals=None,
):
    csv_path = os.path.splitext(out_path)[0] + "_display_series.csv"
    source_artifact = os.path.basename(out_path)
    display_downsampled = bool(int(decimate) > 1)
    if display_downsampled:
        downsample_rule = f"stride decimation from assembled tonic series: every {int(decimate)} points"
    else:
        downsample_rule = "none (display series equals assembled tonic series)"

    rows = []
    series_map = {
        "sig_raw_display": np.asarray(sig_plot, dtype=np.float64),
        "iso_raw_display": np.asarray(uv_plot, dtype=np.float64),
        "tonic_deltaf_display": np.asarray(deltaf_plot, dtype=np.float64),
    }
    t_arr = np.asarray(t_plot, dtype=np.float64)
    for trace_kind, y_arr in series_map.items():
        segments = _split_display_trace_at_missing_intervals(
            t_arr, y_arr, missing_intervals
        )
        for segment_id, (t_segment, y_segment) in enumerate(segments):
            for x, y in zip(t_segment, y_segment):
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                rows.append(
                    {
                        "roi": str(roi),
                        "plot_type": "tonic_overview",
                        "source_run_profile": str(source_run_profile),
                        "source_artifact": str(source_artifact),
                        "trace_kind": str(trace_kind),
                        "display_point_role": "sample",
                        "segment_id": int(segment_id),
                        "x": float(x),
                        "y": float(y),
                        "display_series_export": True,
                        "display_downsampled": display_downsampled,
                        "display_downsample_rule": downsample_rule,
                    }
                )
    for item in missing_intervals or []:
        rows.append(
            {
                "roi": str(roi),
                "plot_type": "tonic_overview",
                "source_run_profile": str(source_run_profile),
                "source_artifact": str(source_artifact),
                "trace_kind": "missing_session_marker",
                "display_point_role": "missing_interval",
                "segment_id": -1,
                "x": float(item["x_hours"]),
                "y": np.nan,
                "display_series_export": True,
                "display_downsampled": bool(int(decimate) > 1),
                "display_downsample_rule": downsample_rule,
                "session_index": int(item["session_index"]),
                "session_status": str(item["status"]),
                "missing_reason": str(item.get("missing_reason", "")),
            }
        )
    if not rows:
        return
    cols = [
        "roi",
        "plot_type",
        "source_run_profile",
        "source_artifact",
        "trace_kind",
        "display_point_role",
        "segment_id",
        "x",
        "y",
        "display_series_export",
        "display_downsampled",
        "display_downsample_rule",
        "session_index",
        "session_status",
        "missing_reason",
    ]
    import pandas as pd
    pd.DataFrame(rows, columns=cols).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


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
    if not os.path.isfile(cache_path):
        print(f"CRITICAL: Cache file not found: {cache_path}")
        raise SystemExit(1)

    # --- Stage 1: Discovery ---
    try:
        cache = open_tonic_cache(cache_path)
        roi = resolve_cache_roi(cache, args.roi)
    except Exception as exc:
        message = str(exc)
        if args.roi and "not found in cache" in message:
            print(f"CRITICAL: {message}")
        elif "Missing dataset" in message:
            print(f"CRITICAL: {message}")
        else:
            print(f"CRITICAL: {message}")
        raise SystemExit(1)
    
    print(f"PLOT_TIMING STEP script=plot_tonic_48h.py step=discovery elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # --- Stage 2a: Cache Reading & Assembly ---
    try:
        continuous_time, sig_raw, uv_raw, deltaf_val, missing_sessions = assemble_arrays(
            cache, roi, args, return_missing_metadata=True
        )
    except Exception as exc:
        cache.close()
        print(f"CRITICAL: {exc}")
        raise SystemExit(1)
    cache.close()

    if missing_sessions and timeline_mode != TONIC_TIMELINE_MODE_REAL_ELAPSED:
        raise RuntimeError(
            "Approved missing sessions require the real elapsed-time tonic view; "
            "gap-free elapsed time would compress the missing interval."
        )

    missing_intervals = _build_missing_intervals(
        missing_sessions,
        sessions_per_hour=args.sessions_per_hour,
    )
    
    print(f"PLOT_TIMING STEP script=plot_tonic_48h.py step=cache_read elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # --- Stage 2b: Decimation ---
    t_h = continuous_time / 3600.0
    n_pts = len(t_h)
    decimate = compute_tonic_overview_display_decimation(n_pts)

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
        missing_intervals=missing_intervals,
    )
    if args.export_display_series_csv:
        _write_tonic_display_series_csv(
            out_path=out_path,
            roi=roi,
            source_run_profile=args.source_run_profile,
            t_plot=t_plot,
            sig_plot=sig_plot,
            uv_plot=uv_plot,
            deltaf_plot=deltaf_plot,
            decimate=decimate,
            missing_intervals=missing_intervals,
        )

    print(f"PLOT_TIMING DONE script=plot_tonic_48h.py total_sec={time.perf_counter() - t_start:.3f}", flush=True)


if __name__ == '__main__':
    main()
