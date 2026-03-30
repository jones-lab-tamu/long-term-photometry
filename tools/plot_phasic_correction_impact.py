#!/usr/bin/env python3
"""
Phasic Correction Impact Plotter
================================

Generates a 4-panel figure showing the impact of artifact correction for a specific diagnostic chunk.
Panel 1: Raw Signal vs Raw Isosbestic (absolute)
Panel 2: Baseline-centered Raw Signal vs Raw Isosbestic (common gain)
Panel 3: Raw Signal vs Selected Dynamic Iso Fit
Panel 4: Final dFF

Usage:
    python tools/plot_phasic_correction_impact.py --analysis-out <DIR> --roi <ROI> --chunk-id <ID> --out <FILE>
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaml


_DYNAMIC_FIT_MODE_ALIAS = {
    "rolling_local_regression": "rolling_filtered_to_raw",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-out', required=True)
    parser.add_argument('--roi', required=True)
    parser.add_argument('--chunk-id', required=True, type=int)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dpi', type=int, default=150)
    return parser.parse_args()


def _normalize_dynamic_fit_mode(mode_raw: str) -> str:
    mode = str(mode_raw or "").strip()
    if not mode:
        return "rolling_filtered_to_raw"
    return _DYNAMIC_FIT_MODE_ALIAS.get(mode, mode)


def _is_rolling_mode(mode_raw: str) -> bool:
    mode = _normalize_dynamic_fit_mode(mode_raw)
    return mode in {"rolling_filtered_to_raw", "rolling_filtered_to_filtered"}


def _resolve_dynamic_fit_settings(analysis_out: str) -> tuple[str, bool]:
    """
    Resolve fit settings from config_used.yaml in analysis_out.
    Fallback is rolling_filtered_to_raw with baseline-subtract disabled.
    """
    cfg_path = os.path.join(analysis_out, "config_used.yaml")
    if not os.path.exists(cfg_path):
        return "rolling_filtered_to_raw", False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        mode = _normalize_dynamic_fit_mode(data.get("dynamic_fit_mode", "rolling_filtered_to_raw"))
        if mode not in {
            "rolling_filtered_to_raw",
            "rolling_filtered_to_filtered",
            "global_linear_regression",
            "robust_global_event_reject",
            "adaptive_event_gated_regression",
        }:
            mode = "rolling_filtered_to_raw"
        baseline_subtract = bool(data.get("baseline_subtract_before_fit", False))
        return mode, baseline_subtract
    except Exception:
        pass
    return "rolling_filtered_to_raw", False


def _dynamic_fit_mode_label(mode_raw: str) -> str:
    mode = _normalize_dynamic_fit_mode(mode_raw)
    if mode == "adaptive_event_gated_regression":
        return "Adaptive event-gated regression"
    if mode == "robust_global_event_reject":
        return "Robust global fit + event rejection"
    if mode == "global_linear_regression":
        return "Global linear regression"
    if mode == "rolling_filtered_to_filtered":
        return "Rolling regression (filtered→filtered)"
    return "Rolling regression (filtered→raw)"


def _dynamic_fit_honesty_suffix(mode_raw: str, baseline_subtract_before_fit: bool) -> str:
    mode = _normalize_dynamic_fit_mode(mode_raw)
    if _is_rolling_mode(mode):
        return (
            "baseline subtract before fit: on"
            if bool(baseline_subtract_before_fit)
            else "baseline subtract before fit: off"
        )
    return "baseline subtract before fit: inactive"


def build_correction_impact_figure(
    t,
    sig,
    iso,
    fit,
    dff,
    roi,
    chunk_id,
    dynamic_fit_mode: str = "rolling_filtered_to_raw",
    baseline_subtract_before_fit: bool = False,
):
    from photometry_pipeline.viz.display_prep import prepare_centered_common_gain

    sig_centered, iso_centered = prepare_centered_common_gain(sig, iso)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # 1. Raw absolute panel (preserves original offset relationship)
    ax1.plot(t, sig, 'g', label='Signal (470nm)', lw=0.8)
    ax1.plot(t, iso, 'm', label='Iso (415nm)', lw=0.8, alpha=0.7)
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Raw Output (V)")
    ax1.set_title(
        f"Correction Impact - ROI {roi} - Chunk {chunk_id} - Raw Inputs (Absolute)"
    )
    ax1.grid(True, alpha=0.3)

    # 2. Centered common-gain readability panel (no amplitude equalization)
    ax2.plot(t, sig_centered, 'g', label='Signal centered (470nm)', lw=0.8)
    ax2.plot(t, iso_centered, 'm', label='Iso centered (415nm)', lw=0.8, alpha=0.7)
    ax2.legend(loc='upper right')
    ax2.set_ylabel("Centered (V)")
    ax2.set_title("Centered Raw Inputs (Common Gain; median-centered, no per-trace scaling)")
    ax2.grid(True, alpha=0.3)

    # 3. Raw vs Fit (unchanged semantics)
    ax3.plot(t, sig, 'g', label='Signal', lw=0.8)
    ax3.plot(t, fit, 'k', label='Iso Fit (Scaled)', lw=0.8, linestyle='--')
    ax3.legend(loc='upper right')
    ax3.set_ylabel("Raw Output (V)")
    ax3.set_title(
        "Dynamic Reference Fitting "
        f"({_dynamic_fit_mode_label(dynamic_fit_mode)}; "
        f"{_dynamic_fit_honesty_suffix(dynamic_fit_mode, baseline_subtract_before_fit)})"
    )
    ax3.grid(True, alpha=0.3)

    # 4. Final dFF (unchanged semantics)
    ax4.plot(t, dff, 'b', label='dFF (Phasic)', lw=0.8)
    ax4.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax4.legend(loc='upper right')
    ax4.set_ylabel("dFF")
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Final Corrected Signal")
    ax4.grid(True, alpha=0.3)
    return fig, (ax1, ax2, ax3, ax4)


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
    
    dynamic_fit_mode, baseline_subtract_before_fit = _resolve_dynamic_fit_settings(args.analysis_out)

    fig, _axes = build_correction_impact_figure(
        t=t,
        sig=sig,
        iso=iso,
        fit=fit,
        dff=dff,
        roi=args.roi,
        chunk_id=args.chunk_id,
        dynamic_fit_mode=dynamic_fit_mode,
        baseline_subtract_before_fit=baseline_subtract_before_fit,
    )
    plt.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.out}")

if __name__ == '__main__':
    main()
