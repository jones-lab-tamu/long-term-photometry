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
from PIL import Image, ImageDraw, ImageFont

# Ensure repo root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Core Imports
from photometry_pipeline.config import Config
from photometry_pipeline.viz.phasic_data_prep import (
    discover_chunks, build_feature_map, resolve_roi, compute_day_layout
)
from photometry_pipeline.io.hdf5_cache_reader import (
    open_phasic_cache, resolve_cache_roi, load_cache_chunk_fields,
    list_cache_chunk_ids, list_cache_source_files
)

from photometry_pipeline.core.feature_extraction import get_peak_indices_for_trace

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


PNG_SAVE_KWARGS = {
    # Keep PNG artifact format unchanged while reducing encoder CPU cost.
    "pil_kwargs": {"compress_level": 1},
}

def parse_args():
    default_sig_iso_mode = os.getenv("PHOTOMETRY_SIGISO_RENDER_MODE", "qc").strip().lower()
    if default_sig_iso_mode not in {"qc", "full"}:
        default_sig_iso_mode = "qc"
    default_dff_mode = os.getenv("PHOTOMETRY_DFF_RENDER_MODE", "qc").strip().lower()
    if default_dff_mode not in {"qc", "full"}:
        default_dff_mode = "qc"
    default_stacked_mode = os.getenv("PHOTOMETRY_STACKED_RENDER_MODE", "qc").strip().lower()
    if default_stacked_mode not in {"qc", "full"}:
        default_stacked_mode = "qc"

    parser = argparse.ArgumentParser(description="Unified Phasic Day-Plot Generator")
    parser.add_argument('--analysis-out', required=True, help="Path to analysis output directory")
    parser.add_argument('--roi', required=True, help="Specific ROI to plot")
    parser.add_argument('--output-dir', required=True, help="Output directory for the day plots")
    parser.add_argument('--sessions-per-hour', type=int, required=True, help="Grid columns")
    
    # Optional / Tuning
    parser.add_argument('--session-duration-s', type=float, default=None, help="Expected session duration in seconds")
    parser.add_argument('--smooth-window-s', type=float, default=1.0, help="Smoothing window for stacked plots")
    parser.add_argument('--dpi', type=int, default=120, help="Output DPI")
    parser.add_argument('--signal', default='auto', help="Trace column for dFF (default: auto detects *_dff)")
    
    # Enable/Disable Families (default: all generated)
    parser.add_argument('--write-dff-grid', action='store_true', default=True, help="(default true)")
    parser.add_argument('--no-write-dff-grid', dest='write_dff_grid', action='store_false')
    parser.add_argument(
        '--dff-render-mode',
        choices=['qc', 'full'],
        default=default_dff_mode,
        help="dFF renderer mode: 'qc' (default fast lightweight renderer) or 'full' (higher-fidelity Matplotlib renderer)"
    )
    
    parser.add_argument('--write-sig-iso-grid', action='store_true', default=True, help="(default true)")
    parser.add_argument('--no-write-sig-iso-grid', dest='write_sig_iso_grid', action='store_false')
    parser.add_argument(
        '--sig-iso-render-mode',
        choices=['qc', 'full'],
        default=default_sig_iso_mode,
        help="Sig/iso renderer mode: 'qc' (default fast lightweight renderer) or 'full' (higher-fidelity Matplotlib renderer)"
    )
    
    parser.add_argument('--write-stacked', action='store_true', default=True, help="(default true)")
    parser.add_argument('--no-write-stacked', dest='write_stacked', action='store_false')
    parser.add_argument(
        '--stacked-render-mode',
        choices=['qc', 'full'],
        default=default_stacked_mode,
        help="Stacked renderer mode: 'qc' (default lightweight renderer) or 'full' (higher-fidelity Matplotlib renderer)"
    )
    
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

def verify_peak_count_strict(detection_trace, time_arr, fs, config, expected_count, roi, cid, src_file):
    if pd.isna(expected_count):
        print(f"CRITICAL: Expected count is NaN for Chunk {cid}.")
        sys.exit(1)
    if detection_trace is None:
        print(
            f"CRITICAL: Missing detection trace for Chunk {cid}, ROI {roi}; "
            f"required by event_signal='{getattr(config, 'event_signal', 'dff')}'."
        )
        sys.exit(1)

    # Source of truth: phasic analysis output features.csv peak_count.
    expected_i = int(expected_count)

    # Plot-time indices must come from the same detector logic used by analysis.
    local_peaks = get_peak_indices_for_trace(detection_trace, fs, config)
    if len(local_peaks) != expected_i:
        print(f"CRITICAL: Plotting Logic Mismatch for Chunk {cid}, ROI {roi} ({expected_i} vs {len(local_peaks)})")
        sys.exit(1)
        
    return local_peaks

def infer_fs(time_arr, config, context=""):
    if len(time_arr) < 2: return getattr(config, 'sampling_rate_hz_fallback', config.target_fs_hz)
    dt = np.median(np.diff(time_arr))
    if dt <= 0 or not np.isfinite(dt): return getattr(config, 'sampling_rate_hz_fallback', config.target_fs_hz)
    return 1.0 / dt


def build_day_slot_maps(cached_by_day, sph):
    day_slots = {}
    for day, items in cached_by_day.items():
        slot_map = {}
        for p in items:
            c = p['col']
            if c >= sph:
                continue
            slot_map[(p['hour'], c)] = p
        day_slots[day] = slot_map
    return day_slots


def init_grid_figure(sph, top=0.95):
    fig, axes = plt.subplots(nrows=24, ncols=sph, figsize=(4 * sph + 2, 24), sharex=True)
    if sph == 1:
        axes = axes.reshape(-1, 1)
    # Pre-set static spacing once; avoids per-day tight_layout solve on a dense grid.
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.02, top=top, hspace=0.28, wspace=0.18)
    return fig, axes


def save_png_fast(fig, out_path, dpi):
    fig.savefig(out_path, dpi=dpi, **PNG_SAVE_KWARGS)


def _sig_iso_tile_layout(sph: int, dpi: int):
    panel_w_in = 4.0
    panel_h_in = 0.9
    tile_w = max(240, int(round(panel_w_in * dpi)))
    tile_h = max(54, int(round(panel_h_in * dpi)))
    return {
        "tile_w": tile_w,
        "tile_h": tile_h,
        "left_label_w": max(48, int(round(0.45 * dpi))),
        "right_pad": max(16, int(round(0.15 * dpi))),
        "top_title_h": max(42, int(round(0.35 * dpi))),
        "bottom_pad": max(14, int(round(0.12 * dpi))),
        "col_gap": max(10, int(round(0.12 * dpi))),
        "row_gap": max(6, int(round(0.06 * dpi))),
        "sph": sph,
        "dpi": dpi,
    }


def _get_font(size):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _trace_domain(panel_t, xlim_600):
    t_finite = np.isfinite(panel_t)
    if xlim_600:
        return 0.0, 600.0
    if not np.any(t_finite):
        return 0.0, 1.0
    x0 = float(np.min(panel_t[t_finite]))
    x1 = float(np.max(panel_t[t_finite]))
    if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
        return 0.0, 1.0
    return x0, x1


def _yrange_from_panel(panel, x0, x1):
    t = panel['t']
    sig = panel['sig']
    uv = panel['uv']
    base_mask = np.isfinite(t) & (t >= x0) & (t <= x1)
    sig_vals = sig[base_mask & np.isfinite(sig)]
    uv_vals = uv[base_mask & np.isfinite(uv)]
    if sig_vals.size == 0 and uv_vals.size == 0:
        return -1.0, 1.0
    vals = np.concatenate([sig_vals, uv_vals])
    y0 = float(np.min(vals))
    y1 = float(np.max(vals))
    if not np.isfinite(y0) or not np.isfinite(y1):
        return -1.0, 1.0
    if y1 <= y0:
        delta = 1.0 if y0 == 0.0 else max(1e-3, 0.1 * abs(y0))
        return y0 - delta, y1 + delta
    pad = 0.05 * (y1 - y0)
    return y0 - pad, y1 + pad


def _sig_iso_panel_ranges_with_day_min_span(slot_map):
    local_ranges = {}
    local_spans = []

    for slot, panel in slot_map.items():
        x0, x1 = _trace_domain(panel['t'], panel.get('xlim_600', False))
        y0, y1 = _yrange_from_panel(panel, x0, x1)
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
            y0, y1 = -1.0, 1.0
        span = y1 - y0
        local_ranges[slot] = (float(y0), float(y1))
        if np.isfinite(span) and span > 0:
            local_spans.append(float(span))

    if not local_ranges:
        return {}

    if local_spans:
        day_min_span = float(np.median(np.asarray(local_spans, dtype=np.float64)))
    else:
        day_min_span = 2.0
    if not np.isfinite(day_min_span) or day_min_span <= 0:
        day_min_span = 2.0

    final_ranges = {}
    for slot, (y0, y1) in local_ranges.items():
        span = y1 - y0
        if not np.isfinite(span) or span <= 0:
            center = 0.5 * (y0 + y1) if np.isfinite(y0) and np.isfinite(y1) else 0.0
            half = 0.5 * day_min_span
            final_ranges[slot] = (center - half, center + half)
            continue

        if span < day_min_span:
            center = 0.5 * (y0 + y1)
            half = 0.5 * day_min_span
            final_ranges[slot] = (center - half, center + half)
        else:
            final_ranges[slot] = (y0, y1)

    return final_ranges


def _paint_trace_minmax(tile_arr, x_idx, y_idx, plot_x0, plot_y0, plot_w, plot_h, color, stroke=1):
    if x_idx.size == 0:
        return
    y_min = np.full(plot_w, plot_h, dtype=np.int32)
    y_max = np.full(plot_w, -1, dtype=np.int32)
    np.minimum.at(y_min, x_idx, y_idx)
    np.maximum.at(y_max, x_idx, y_idx)
    cols = np.where(y_max >= 0)[0]
    if cols.size == 0:
        return

    half = max(0, stroke // 2)
    for col in cols:
        y0 = int(y_min[col])
        y1 = int(y_max[col])
        if y1 < y0:
            y0, y1 = y1, y0
        yy0 = max(0, min(plot_h - 1, y0))
        yy1 = max(0, min(plot_h - 1, y1))
        px = plot_x0 + int(col)
        for dx in range(-half, half + 1):
            px2 = px + dx
            if px2 < plot_x0 or px2 >= plot_x0 + plot_w:
                continue
            tile_arr[plot_y0 + yy0:plot_y0 + yy1 + 1, px2, :] = color


def _render_sig_iso_panel_tile_lightweight(panel, layout, title_font, panel_y_range=None):
    tile_h = layout["tile_h"]
    tile_w = layout["tile_w"]
    arr = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)

    pad_x = max(8, int(0.02 * tile_w))
    title_h = max(14, int(0.18 * tile_h))
    plot_x0 = pad_x
    plot_x1 = tile_w - pad_x - 1
    plot_y0 = title_h + 2
    plot_y1 = tile_h - max(6, int(0.08 * tile_h)) - 1
    plot_w = max(2, plot_x1 - plot_x0 + 1)
    plot_h = max(2, plot_y1 - plot_y0 + 1)

    # Plot frame
    arr[plot_y0, plot_x0:plot_x1 + 1, :] = (220, 220, 220)
    arr[plot_y1, plot_x0:plot_x1 + 1, :] = (220, 220, 220)
    arr[plot_y0:plot_y1 + 1, plot_x0, :] = (220, 220, 220)
    arr[plot_y0:plot_y1 + 1, plot_x1, :] = (220, 220, 220)

    t = panel['t']
    x0, x1 = _trace_domain(t, panel.get('xlim_600', False))
    if panel_y_range is None:
        y0, y1 = _yrange_from_panel(panel, x0, x1)
    else:
        y0, y1 = panel_y_range
    x_span = x1 - x0
    y_span = y1 - y0
    if x_span <= 0 or not np.isfinite(x_span):
        x_span = 1.0
    if y_span <= 0 or not np.isfinite(y_span):
        y_span = 1.0

    domain_mask = np.isfinite(t) & (t >= x0) & (t <= x1)
    x_float = ((t[domain_mask] - x0) / x_span) * (plot_w - 1)
    x_idx = np.rint(x_float).astype(np.int32)
    x_idx = np.clip(x_idx, 0, plot_w - 1)

    sig = panel['sig'][domain_mask]
    sig_mask = np.isfinite(sig)
    if np.any(sig_mask):
        y_float = ((sig[sig_mask] - y0) / y_span) * (plot_h - 1)
        y_idx = (plot_h - 1) - np.rint(y_float).astype(np.int32)
        y_idx = np.clip(y_idx, 0, plot_h - 1)
        _paint_trace_minmax(
            arr, x_idx[sig_mask], y_idx, plot_x0, plot_y0, plot_w, plot_h,
            color=(0, 140, 0), stroke=1
        )

    uv = panel['uv'][domain_mask]
    uv_mask = np.isfinite(uv)
    if np.any(uv_mask):
        y_float = ((uv[uv_mask] - y0) / y_span) * (plot_h - 1)
        y_idx = (plot_h - 1) - np.rint(y_float).astype(np.int32)
        y_idx = np.clip(y_idx, 0, plot_h - 1)
        _paint_trace_minmax(
            arr, x_idx[uv_mask], y_idx, plot_x0, plot_y0, plot_w, plot_h,
            color=(180, 0, 180), stroke=1
        )

    tile = Image.fromarray(arr, mode='RGB')
    draw = ImageDraw.Draw(tile)
    draw.text((plot_x0, max(1, int(0.03 * tile_h))), f"Chunk {panel['chunk_id']}", fill='black', font=title_font)
    return tile


def _build_blank_sig_iso_tile(layout):
    tile_h = layout["tile_h"]
    tile_w = layout["tile_w"]
    arr = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
    pad_x = max(8, int(0.02 * tile_w))
    title_h = max(14, int(0.18 * tile_h))
    plot_x0 = pad_x
    plot_x1 = tile_w - pad_x - 1
    plot_y0 = title_h + 2
    plot_y1 = tile_h - max(6, int(0.08 * tile_h)) - 1
    arr[plot_y0, plot_x0:plot_x1 + 1, :] = (235, 235, 235)
    arr[plot_y1, plot_x0:plot_x1 + 1, :] = (235, 235, 235)
    arr[plot_y0:plot_y1 + 1, plot_x0, :] = (235, 235, 235)
    arr[plot_y0:plot_y1 + 1, plot_x1, :] = (235, 235, 235)
    return Image.fromarray(arr, mode='RGB')


def _compose_sig_iso_day_tile_canvas(day, plot_roi, sph, slot_map, layout, panel_y_ranges=None):
    tile_w = layout["tile_w"]
    tile_h = layout["tile_h"]
    col_gap = layout["col_gap"]
    row_gap = layout["row_gap"]
    left_label_w = layout["left_label_w"]
    top_title_h = layout["top_title_h"]
    right_pad = layout["right_pad"]
    bottom_pad = layout["bottom_pad"]

    canvas_w = left_label_w + (sph * tile_w) + ((sph - 1) * col_gap) + right_pad
    canvas_h = top_title_h + (24 * tile_h) + (23 * row_gap) + bottom_pad
    day_canvas = Image.new('RGB', (canvas_w, canvas_h), color='white')

    blank_tile = _build_blank_sig_iso_tile(layout)
    draw = ImageDraw.Draw(day_canvas)
    title_font = _get_font(max(13, int(round(0.11 * layout["dpi"]))))
    label_font = _get_font(max(11, int(round(0.09 * layout["dpi"]))))
    chunk_font = _get_font(max(10, int(round(0.08 * layout["dpi"]))))
    title_txt = f"Day {day} Raw/Iso - {plot_roi}"
    draw.text((canvas_w // 2, max(6, top_title_h // 4)), title_txt, fill='black', anchor='ma', font=title_font)

    for h in range(24):
        y = top_title_h + h * (tile_h + row_gap)
        # Keep left-column hour labels visible as part of contract semantics.
        draw.text((max(4, left_label_w // 2), y + (tile_h // 2)), f"H{h:02d}", fill='black', anchor='mm', font=label_font)
        for c in range(sph):
            x = left_label_w + c * (tile_w + col_gap)
            panel = slot_map.get((h, c))
            if panel is None:
                day_canvas.paste(blank_tile, (x, y))
            else:
                day_canvas.paste(
                    _render_sig_iso_panel_tile_lightweight(
                        panel, layout, chunk_font,
                        panel_y_range=None if panel_y_ranges is None else panel_y_ranges.get((h, c))
                    ),
                    (x, y)
                )

    return day_canvas


def _dff_tile_layout(sph: int, dpi: int):
    panel_w_in = 4.0
    panel_h_in = 0.95
    tile_w = max(240, int(round(panel_w_in * dpi)))
    tile_h = max(58, int(round(panel_h_in * dpi)))
    return {
        "tile_w": tile_w,
        "tile_h": tile_h,
        "left_label_w": max(48, int(round(0.45 * dpi))),
        "right_pad": max(16, int(round(0.15 * dpi))),
        "top_title_h": max(42, int(round(0.35 * dpi))),
        "bottom_pad": max(14, int(round(0.12 * dpi))),
        "col_gap": max(10, int(round(0.12 * dpi))),
        "row_gap": max(6, int(round(0.06 * dpi))),
        "sph": sph,
        "dpi": dpi,
    }


def _build_blank_dff_tile(layout):
    tile_h = layout["tile_h"]
    tile_w = layout["tile_w"]
    arr = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
    pad_x = max(8, int(0.02 * tile_w))
    title_h = max(14, int(0.16 * tile_h))
    plot_x0 = pad_x
    plot_x1 = tile_w - pad_x - 1
    plot_y0 = title_h + 2
    plot_y1 = tile_h - max(6, int(0.08 * tile_h)) - 1
    arr[plot_y0, plot_x0:plot_x1 + 1, :] = (235, 235, 235)
    arr[plot_y1, plot_x0:plot_x1 + 1, :] = (235, 235, 235)
    arr[plot_y0:plot_y1 + 1, plot_x0, :] = (235, 235, 235)
    arr[plot_y0:plot_y1 + 1, plot_x1, :] = (235, 235, 235)
    return Image.fromarray(arr, mode='RGB')


def _render_dff_panel_tile_lightweight(panel, layout, title_font, global_ymin, global_ymax):
    trace_sec = 0.0
    marker_sec = 0.0
    title_text_sec = 0.0

    tile_h = layout["tile_h"]
    tile_w = layout["tile_w"]
    arr = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)

    pad_x = max(8, int(0.02 * tile_w))
    title_h = max(14, int(0.16 * tile_h))
    plot_x0 = pad_x
    plot_x1 = tile_w - pad_x - 1
    plot_y0 = title_h + 2
    plot_y1 = tile_h - max(6, int(0.08 * tile_h)) - 1
    plot_w = max(2, plot_x1 - plot_x0 + 1)
    plot_h = max(2, plot_y1 - plot_y0 + 1)

    arr[plot_y0, plot_x0:plot_x1 + 1, :] = (220, 220, 220)
    arr[plot_y1, plot_x0:plot_x1 + 1, :] = (220, 220, 220)
    arr[plot_y0:plot_y1 + 1, plot_x0, :] = (220, 220, 220)
    arr[plot_y0:plot_y1 + 1, plot_x1, :] = (220, 220, 220)

    t = panel['t']
    y = panel['dff']
    x0, x1 = _trace_domain(t, False)

    y0 = float(global_ymin)
    y1 = float(global_ymax)
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
        y0, y1 = -1.0, 1.0

    x_span = x1 - x0
    y_span = y1 - y0
    if x_span <= 0 or not np.isfinite(x_span):
        x_span = 1.0
    if y_span <= 0 or not np.isfinite(y_span):
        y_span = 1.0

    trace_t0 = time.perf_counter()
    domain_mask = np.isfinite(t) & np.isfinite(y) & (t >= x0) & (t <= x1)
    if np.any(domain_mask):
        x_float = ((t[domain_mask] - x0) / x_span) * (plot_w - 1)
        x_idx = np.rint(x_float).astype(np.int32)
        x_idx = np.clip(x_idx, 0, plot_w - 1)

        y_float = ((y[domain_mask] - y0) / y_span) * (plot_h - 1)
        y_idx = (plot_h - 1) - np.rint(y_float).astype(np.int32)
        y_idx = np.clip(y_idx, 0, plot_h - 1)

        _paint_trace_minmax(
            arr, x_idx, y_idx, plot_x0, plot_y0, plot_w, plot_h,
            color=(0, 0, 0), stroke=1
        )
    trace_sec = time.perf_counter() - trace_t0

    tile = Image.fromarray(arr, mode='RGB')
    draw = ImageDraw.Draw(tile)
    title_t0 = time.perf_counter()
    draw.text((plot_x0, max(1, int(0.02 * tile_h))), f"Chunk {panel['chunk_id']}", fill='black', font=title_font)
    title_text_sec = time.perf_counter() - title_t0

    p_idxs = panel.get('peak_indices', np.array([], dtype=int))
    y_eps = 0.01 * y_span if y_span > 0 else 1e-6

    marker_t0 = time.perf_counter()
    for idx in p_idxs:
        if idx < 0 or idx >= len(t):
            continue
        px_t = t[idx]
        py_true = y[idx]
        if not np.isfinite(px_t) or not np.isfinite(py_true) or px_t < x0 or px_t > x1:
            continue

        x_float = ((px_t - x0) / x_span) * (plot_w - 1)
        px = int(round(plot_x0 + np.clip(x_float, 0, plot_w - 1)))

        if py_true > (y1 - y_eps):
            top_y = plot_y0 + 1
            draw.polygon([(px, top_y), (px - 3, top_y + 6), (px + 3, top_y + 6)], fill=(220, 0, 0))
        elif py_true < (y0 + y_eps):
            bot_y = plot_y1 - 1
            draw.polygon([(px, bot_y), (px - 3, bot_y - 6), (px + 3, bot_y - 6)], fill=(220, 0, 0))
        else:
            y_float = ((py_true - y0) / y_span) * (plot_h - 1)
            py = int(round(plot_y0 + (plot_h - 1) - np.clip(y_float, 0, plot_h - 1)))
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(220, 0, 0))
    marker_sec = time.perf_counter() - marker_t0

    return tile, {
        "trace_sec": trace_sec,
        "marker_sec": marker_sec,
        "title_text_sec": title_text_sec,
    }


def _compose_dff_day_tile_canvas_lightweight(day, plot_roi, sph, slot_map, layout, global_ymin, global_ymax):
    tile_w = layout["tile_w"]
    tile_h = layout["tile_h"]
    col_gap = layout["col_gap"]
    row_gap = layout["row_gap"]
    left_label_w = layout["left_label_w"]
    top_title_h = layout["top_title_h"]
    right_pad = layout["right_pad"]
    bottom_pad = layout["bottom_pad"]

    canvas_w = left_label_w + (sph * tile_w) + ((sph - 1) * col_gap) + right_pad
    canvas_h = top_title_h + (24 * tile_h) + (23 * row_gap) + bottom_pad
    day_canvas = Image.new('RGB', (canvas_w, canvas_h), color='white')

    blank_tile = _build_blank_dff_tile(layout)
    draw = ImageDraw.Draw(day_canvas)
    title_font = _get_font(max(13, int(round(0.11 * layout["dpi"]))))
    label_font = _get_font(max(11, int(round(0.09 * layout["dpi"]))))
    chunk_font = _get_font(max(10, int(round(0.08 * layout["dpi"]))))
    title_txt = f"Phasic QC - Day {day} - ROI {plot_roi} - Mode: DFF"
    draw.text((canvas_w // 2, max(6, top_title_h // 4)), title_txt, fill='black', anchor='ma', font=title_font)
    stats = {
        "trace_sec": 0.0,
        "marker_sec": 0.0,
        "title_text_sec": 0.0,
        "paste_sec": 0.0,
        "panels": 0,
    }

    for h in range(24):
        y = top_title_h + h * (tile_h + row_gap)
        draw.text((max(4, left_label_w // 2), y + (tile_h // 2)), f"H{h:02d}", fill='black', anchor='mm', font=label_font)
        for c in range(sph):
            x = left_label_w + c * (tile_w + col_gap)
            panel = slot_map.get((h, c))
            if panel is None:
                day_canvas.paste(blank_tile, (x, y))
            else:
                panel_tile, panel_stats = _render_dff_panel_tile_lightweight(
                    panel, layout, chunk_font, global_ymin, global_ymax
                )
                paste_t0 = time.perf_counter()
                day_canvas.paste(
                    panel_tile,
                    (x, y)
                )
                stats["paste_sec"] += time.perf_counter() - paste_t0
                stats["trace_sec"] += panel_stats["trace_sec"]
                stats["marker_sec"] += panel_stats["marker_sec"]
                stats["title_text_sec"] += panel_stats["title_text_sec"]
                stats["panels"] += 1

    return day_canvas, stats


def _render_stacked_day_canvas_lightweight(day, plot_roi, traces, smooth_window_s, dpi):
    n_traces = len(traces)
    fig_w_in = 6.0
    fig_h_in = max(2.0, (n_traces * 0.285) + 1.6)
    canvas_w = max(640, int(round(fig_w_in * dpi)))
    canvas_h = max(320, int(round(fig_h_in * dpi)))

    left_pad = max(44, int(0.075 * canvas_w))
    right_pad = max(12, int(0.02 * canvas_w))
    top_pad = max(28, int(0.03 * canvas_h))
    bottom_pad = max(34, int(0.055 * canvas_h))
    plot_x0 = left_pad
    plot_x1 = canvas_w - right_pad - 1
    plot_y0 = top_pad
    plot_y1 = canvas_h - bottom_pad - 1
    plot_w = max(2, plot_x1 - plot_x0 + 1)
    plot_h = max(2, plot_y1 - plot_y0 + 1)

    ranges = [np.ptp(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 0.0 for _, y in traces]
    avg_rng = float(np.median(ranges)) if ranges else 1.0
    if not np.isfinite(avg_rng) or avg_rng <= 0:
        avg_rng = 1.0
    step = max(0.1, avg_rng * 0.8)

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for i, (t, y) in enumerate(traces):
        mask = np.isfinite(t) & np.isfinite(y)
        if not np.any(mask):
            continue
        offset = (n_traces - 1 - i) * step
        tv = t[mask]
        yv = y[mask] + offset
        xmins.append(float(np.min(tv)))
        xmaxs.append(float(np.max(tv)))
        ymins.append(float(np.min(yv)))
        ymaxs.append(float(np.max(yv)))

    if xmins and xmaxs:
        x0 = min(xmins)
        x1 = max(xmaxs)
    else:
        x0, x1 = 0.0, 1.0
    if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
        x0, x1 = 0.0, 1.0

    if ymins and ymaxs:
        y0 = min(ymins)
        y1 = max(ymaxs)
    else:
        y0, y1 = -1.0, 1.0
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
        y0, y1 = -1.0, 1.0
    y_span = y1 - y0
    y_pad = 0.05 * y_span
    if not np.isfinite(y_pad) or y_pad <= 0:
        y_pad = 0.1
    y0 -= y_pad
    y1 += y_pad

    x_span = x1 - x0
    y_span = y1 - y0
    if x_span <= 0 or not np.isfinite(x_span):
        x_span = 1.0
    if y_span <= 0 or not np.isfinite(y_span):
        y_span = 1.0

    arr = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    arr[plot_y0, plot_x0:plot_x1 + 1, :] = (205, 205, 205)
    arr[plot_y1, plot_x0:plot_x1 + 1, :] = (205, 205, 205)
    arr[plot_y0:plot_y1 + 1, plot_x0, :] = (205, 205, 205)
    arr[plot_y0:plot_y1 + 1, plot_x1, :] = (205, 205, 205)

    for i, (t, y) in enumerate(traces):
        mask = np.isfinite(t) & np.isfinite(y)
        if not np.any(mask):
            continue
        offset = (n_traces - 1 - i) * step
        tv = t[mask]
        yv = y[mask] + offset
        x_float = ((tv - x0) / x_span) * (plot_w - 1)
        x_idx = np.rint(x_float).astype(np.int32)
        x_idx = np.clip(x_idx, 0, plot_w - 1)
        y_float = ((yv - y0) / y_span) * (plot_h - 1)
        y_idx = (plot_h - 1) - np.rint(y_float).astype(np.int32)
        y_idx = np.clip(y_idx, 0, plot_h - 1)
        _paint_trace_minmax(
            arr, x_idx, y_idx, plot_x0, plot_y0, plot_w, plot_h,
            color=(0, 0, 0), stroke=1
        )

    img = Image.fromarray(arr, mode='RGB')
    draw = ImageDraw.Draw(img)
    title_font = _get_font(max(12, int(round(0.11 * dpi))))
    label_font = _get_font(max(10, int(round(0.09 * dpi))))
    tick_font = _get_font(max(9, int(round(0.08 * dpi))))
    draw.text(
        (canvas_w // 2, max(10, top_pad // 2)),
        f"Day {day} Stacked (Smoothed {smooth_window_s}s) - {plot_roi}",
        fill='black',
        anchor='ma',
        font=title_font
    )

    # Add simple x ticks to better match the full renderer's axis readability.
    tick_values = np.linspace(x0, x1, 7)
    for tv in tick_values:
        tx = int(round(plot_x0 + np.clip(((tv - x0) / x_span) * (plot_w - 1), 0, plot_w - 1)))
        draw.line([(tx, plot_y1 + 1), (tx, plot_y1 + 5)], fill=(80, 80, 80), width=1)
        draw.text((tx, plot_y1 + 7), f"{int(round(tv))}", fill=(70, 70, 70), anchor='ma', font=tick_font)

    draw.text((plot_x0 + (plot_w // 2), canvas_h - max(12, bottom_pad // 3)), "Time (s)", fill='black', anchor='ma', font=label_font)

    # Draw rotated y-axis label for closer visual alignment with full Matplotlib output.
    y_label = f"Sessions ({n_traces})"
    tmp = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.textbbox((0, 0), y_label, font=label_font)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    label_img = Image.new('RGBA', (tw + 2, th + 2), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((1, 1), y_label, fill=(0, 0, 0, 255), font=label_font)
    label_rot = label_img.rotate(90, expand=True)
    label_x = max(2, int(0.012 * canvas_w))
    label_y = max(2, (plot_y0 + plot_y1) // 2 - label_rot.height // 2)
    img.paste(label_rot, (label_x, label_y), label_rot)

    return img


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
    
    needs_dff_trace = args.write_dff_grid or args.write_stacked
    needs_peak_verification = args.write_dff_grid
    
    # 1. Open the HDF5 Cache (Mandatory source for discovery and data)
    cache_path = os.path.join(args.analysis_out, 'phasic_trace_cache.h5')
    if not os.path.exists(cache_path):
        print(f"CRITICAL: Phasic cache not found: {cache_path}")
        sys.exit(1)
        
    cache = open_phasic_cache(cache_path)
    
    # 2. Discover Chunks via Cache Metadata (No longer dependent on traces/ CSV folder)
    cids = list_cache_chunk_ids(cache)
    
    # Robustly handle caches missing meta/source_files (common in synthetic tests/legacy caches)
    meta = cache.get('meta')
    if meta and 'source_files' in meta:
        sfs = list_cache_source_files(cache)
    else:
        # Fallback: synthesize names if missing, allowing test suites to pass
        # without requiring full production metadata datasets.
        sfs = [f"chunk_{cid}.csv" for cid in cids]
    
    if not cids:
        print("CRITICAL: No chunks found in cache.")
        sys.exit(1)
        
    if len(cids) != len(sfs):
        print(f"CRITICAL: Cache metadata mismatch: {len(cids)} IDs vs {len(sfs)} source files.")
        sys.exit(1)
        
    # Build discovery entries for layout engine. 
    # The layout engine uses the 2nd element (source_file) for datetime inference.
    chunk_entries = list(zip(cids, sfs))
    
    # 3. Resolve ROI via cache
    plot_roi = resolve_cache_roi(cache, args.roi)
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
    
    # 4. Identify signals to pull from cache
    # Explicitly enforce mode-minimal field loading contract
    if args.write_sig_iso_grid and needs_dff_trace:
        # Full mode
        fields_to_load = ['time_sec', 'sig_raw', 'uv_raw', 'dff']
    elif args.write_sig_iso_grid and not needs_dff_trace:
        # Sig/iso only mode
        fields_to_load = ['time_sec', 'sig_raw', 'uv_raw']
    elif not args.write_sig_iso_grid and needs_dff_trace:
        # Stacked-only or dFF-grid-only mode
        fields_to_load = ['time_sec', 'dff']
    else:
        # Fallback minimal
        fields_to_load = ['time_sec']

    detection_field = None
    if needs_peak_verification:
        signal_cfg = getattr(config, 'event_signal', 'dff')
        detection_field = 'dff' if signal_cfg == 'dff' else 'delta_f'
        if detection_field not in fields_to_load:
            fields_to_load.append(detection_field)
        
    print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=discovery elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # ------------------------------------------------------------------
    # 1. Single-Pass Loading (Now from Cache, no longer CSV)
    # ------------------------------------------------------------------
    raw_chunks = []
    
    for cr in pds.chunks:
        try:
            # Shared reader returns tuple matching the order of fields_to_load
            arrays = load_cache_chunk_fields(cache, plot_roi, cr.chunk_id, fields_to_load)
            
            # Map back to exactly what downstream expects
            arr_map = dict(zip(fields_to_load, arrays))
            
            rec = {
                'cr': cr,
                'x': arr_map.get('time_sec'),
                'y_sig': arr_map.get('sig_raw'),
                'y_uv': arr_map.get('uv_raw'),
                'y_dff': arr_map.get('dff'),
                'y_detect': arr_map.get(detection_field) if detection_field else None,
                # N must be derived from the actual length of an array we got
                'N': len(arrays[0]) if arrays else 0
            }
            raw_chunks.append(rec)
        except Exception as e:
            print(f"CRITICAL: Error reading chunk {cr.chunk_id} from cache: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    # Close cache explicitly since we are done loading.
    cache.close()
            
    print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=cache_read elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    # ------------------------------------------------------------------
    # 2. Verification
    # ------------------------------------------------------------------
    for rec in raw_chunks:
        cr = rec['cr']
        x = rec['x']
        
        fs_plot = infer_fs(x, config, context=f"Chunk {cr.chunk_id}") if x is not None else float(getattr(config, 'target_fs_hz', 0.0))
        if x is None:
            if not np.isfinite(fs_plot) or fs_plot <= 0:
                fs_plot = float(getattr(config, 'sampling_rate_hz_fallback', 1.0))
            x = np.arange(rec['N']) / fs_plot
            rec['x'] = x

        # Strict peak-count verification must mirror analysis-time feature extraction
        # semantics. Analysis uses the authoritative config sampling rate, so do not
        # prefer floating-point inferred plotting fs for this invariant check.
        fs_cfg = float(getattr(config, 'target_fs_hz', np.nan))
        if np.isfinite(fs_cfg) and fs_cfg > 0:
            fs_verify = fs_cfg
        else:
            fs_verify = fs_plot
            
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
                rec['y_detect'], x, fs_verify, config, exp_count, plot_roi, cr.chunk_id, cr.source_file
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
            c_rec['xlim_600'] = bool(np.max(t_norm) > 550)
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
    day_slot_maps = build_day_slot_maps(cached_by_day, sph)
    print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=global_limits elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
    
    
    # ------------------------------------------------------------------
    # 2. Render Family 1: dFF Grid
    # ------------------------------------------------------------------
    if args.write_dff_grid:
        if args.dff_render_mode == 'full':
            fig_dff, axes_dff = init_grid_figure(sph, top=0.95)
            y_span = global_ymax - global_ymin
            eps = 0.01 * y_span if y_span > 0 else 1e-6
            for d in unique_days:
                slot_map = day_slot_maps.get(d, {})
                if not slot_map:
                    continue

                fig_dff.suptitle(f"Phasic QC - Day {d} - ROI {plot_roi} - Mode: DFF", fontsize=16)

                for h in range(24):
                    for c in range(sph):
                        ax = axes_dff[h, c]
                        p = slot_map.get((h, c))
                        ax.cla()
                        if p is None:
                            ax.axis('off')
                            continue
                        ax.axis('on')
                        ax.set_ylim(global_ymin, global_ymax)
                        if c == 0:
                            ax.set_ylabel(f"H{h:02d}", rotation=0, labelpad=15, va='center', fontweight='bold')
                        ax.set_title(f"Chunk {p['chunk_id']}", fontsize=6, pad=2)
                        ax.plot(p['t'], p['dff'], 'k', lw=0.8)

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

                            if np.any(mask_ok):
                                ax.scatter(px[mask_ok], py_plot[mask_ok], s=10, c='red', alpha=0.6, zorder=3)
                            if np.any(mask_hi):
                                ax.scatter(px[mask_hi], py_plot[mask_hi], s=12, marker='^', c='red', alpha=0.8, zorder=4)
                            if np.any(mask_lo):
                                ax.scatter(px[mask_lo], py_plot[mask_lo], s=12, marker='v', c='red', alpha=0.8, zorder=4)
                            n_clipped = np.sum(mask_hi) + np.sum(mask_lo)

                        # Annotation
                        val = p['count']
                        txt = "peaks=NaN" if pd.isna(val) else f"peaks={int(val)}"
                        if n_clipped > 0:
                            txt += f"\n({n_clipped} clipped)"
                        color = 'red' if pd.isna(val) else 'blue'
                        ax.text(
                            0.95, 0.9, txt, transform=ax.transAxes, ha='right', va='top', fontsize=8, color=color,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                        )

                out_path = os.path.join(args.output_dir, f"phasic_dFF_day_{d:03d}.png")
                print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=plotting family=dff_full day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
                save_png_fast(fig_dff, out_path, args.dpi)
                print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=figure_save family=dff_full day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
            plt.close(fig_dff)
        else:
            dff_qc_layout = _dff_tile_layout(sph, args.dpi)
            for d in unique_days:
                slot_map = day_slot_maps.get(d, {})
                if not slot_map:
                    continue

                compose_t0 = time.perf_counter()
                proto_canvas, proto_stats = _compose_dff_day_tile_canvas_lightweight(
                    day=d,
                    plot_roi=plot_roi,
                    sph=sph,
                    slot_map=slot_map,
                    layout=dff_qc_layout,
                    global_ymin=global_ymin,
                    global_ymax=global_ymax,
                )
                compose_sec = time.perf_counter() - compose_t0
                out_path = os.path.join(args.output_dir, f"phasic_dFF_day_{d:03d}.png")
                print(
                    f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py "
                    f"step=plotting family=dff_qc day={d} elapsed_sec={time.perf_counter() - t_start:.3f}",
                    flush=True
                )
                save_t0 = time.perf_counter()
                proto_canvas.save(out_path, compress_level=1)
                save_sec = time.perf_counter() - save_t0
                print(
                    f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py "
                    f"step=figure_save family=dff_qc day={d} elapsed_sec={time.perf_counter() - t_start:.3f}",
                    flush=True
                )
                print(
                    f"PLOT_TIMING DETAIL script=plot_phasic_dayplot_bundle.py "
                    f"family=dff_qc day={d} "
                    f"trace_sec={proto_stats['trace_sec']:.4f} marker_sec={proto_stats['marker_sec']:.4f} "
                    f"title_text_sec={proto_stats['title_text_sec']:.4f} "
                    f"paste_sec={proto_stats['paste_sec']:.4f} compose_sec={compose_sec:.4f} save_sec={save_sec:.4f} "
                    f"panels={proto_stats['panels']}",
                    flush=True
                )

    # ------------------------------------------------------------------
    # 3. Render Family 2: Sig/Iso Grid
    # ------------------------------------------------------------------
    if args.write_sig_iso_grid:
        if args.sig_iso_render_mode == 'full':
            fig_sig, axes_sig = init_grid_figure(sph, top=0.97)
            for d in unique_days:
                slot_map = day_slot_maps.get(d, {})
                if not slot_map:
                    continue
                panel_y_ranges = _sig_iso_panel_ranges_with_day_min_span(slot_map)

                fig_sig.suptitle(f"Day {d} Raw/Iso - {plot_roi}", fontsize=16)

                for h in range(24):
                    for c in range(sph):
                        ax = axes_sig[h, c]
                        p = slot_map.get((h, c))
                        ax.cla()
                        if p is None:
                            ax.axis('off')
                            continue
                        ax.axis('on')
                        panel_y = panel_y_ranges.get((h, c))
                        if panel_y is None:
                            x0, x1 = _trace_domain(p['t'], p.get('xlim_600', False))
                            panel_y = _yrange_from_panel(p, x0, x1)
                        ax.set_ylim(panel_y[0], panel_y[1])
                        ax.plot(p['t'], p['sig'], 'g', lw=0.5, label='Sig')
                        ax.plot(p['t'], p['uv'], 'm', lw=0.5, label='Iso')
                        if p.get('xlim_600', False):
                            ax.set_xlim(0, 600)
                        if c == 0:
                            ax.set_ylabel(f"H{h:02d}", rotation=0, labelpad=20)

                out_path = os.path.join(args.output_dir, f"phasic_sig_iso_day_{d:03d}.png")
                print(
                    f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py "
                    f"step=plotting family=sig_iso_full day={d} elapsed_sec={time.perf_counter() - t_start:.3f}",
                    flush=True
                )
                save_png_fast(fig_sig, out_path, args.dpi)
                print(
                    f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py "
                    f"step=figure_save family=sig_iso_full day={d} elapsed_sec={time.perf_counter() - t_start:.3f}",
                    flush=True
                )
            plt.close(fig_sig)
        else:
            qc_layout = _sig_iso_tile_layout(sph, args.dpi)
            for d in unique_days:
                slot_map = day_slot_maps.get(d, {})
                if not slot_map:
                    continue
                panel_y_ranges = _sig_iso_panel_ranges_with_day_min_span(slot_map)

                day_canvas = _compose_sig_iso_day_tile_canvas(
                    day=d,
                    plot_roi=plot_roi,
                    sph=sph,
                    slot_map=slot_map,
                    layout=qc_layout,
                    panel_y_ranges=panel_y_ranges,
                )
                out_path = os.path.join(args.output_dir, f"phasic_sig_iso_day_{d:03d}.png")
                print(
                    f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py "
                    f"step=plotting family=sig_iso_qc day={d} elapsed_sec={time.perf_counter() - t_start:.3f}",
                    flush=True
                )
                day_canvas.save(out_path, compress_level=1)
                print(
                    f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py "
                    f"step=figure_save family=sig_iso_qc day={d} elapsed_sec={time.perf_counter() - t_start:.3f}",
                    flush=True
                )

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

        if args.stacked_render_mode == 'full':
            for d in unique_days:
                # Sort chronologically
                day_items = sorted(cached_by_day.get(d, []), key=lambda x: x['chunk_id'])
                traces = [smoothed_data[c['chunk_id']] for c in day_items if c['chunk_id'] in smoothed_data]
                if not traces:
                    continue

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
                print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=plotting family=stacked_full day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
                plt.savefig(out_path, dpi=args.dpi)
                print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=figure_save family=stacked_full day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
                plt.close(fig)
        else:
            for d in unique_days:
                day_items = sorted(cached_by_day.get(d, []), key=lambda x: x['chunk_id'])
                traces = [smoothed_data[c['chunk_id']] for c in day_items if c['chunk_id'] in smoothed_data]
                if not traces:
                    continue

                out_path = os.path.join(args.output_dir, f"phasic_stacked_day_{d:03d}.png")
                print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=plotting family=stacked_qc day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)
                canvas = _render_stacked_day_canvas_lightweight(
                    day=d,
                    plot_roi=plot_roi,
                    traces=traces,
                    smooth_window_s=args.smooth_window_s,
                    dpi=args.dpi,
                )
                canvas.save(out_path, compress_level=1)
                print(f"PLOT_TIMING STEP script=plot_phasic_dayplot_bundle.py step=figure_save family=stacked_qc day={d} elapsed_sec={time.perf_counter() - t_start:.3f}", flush=True)

    print(f"PLOT_TIMING DONE script=plot_phasic_dayplot_bundle.py total_sec={time.perf_counter() - t_start:.3f}", flush=True)

if __name__ == '__main__':
    main()
