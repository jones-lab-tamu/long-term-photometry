#!/usr/bin/env python3
"""
Phasic Correction Impact Plotter
================================

Generates a 4-panel figure showing correction stages for a specific diagnostic chunk.
Panel 1: Original signal/isosbestic + bleach-fit overlays
Panel 2: Bleach-corrected signal/isosbestic
Panel 3: Dynamic-fit view in bleach-corrected frame
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


def _normalize_bleach_correction_mode(mode_raw: str) -> str:
    mode = str(mode_raw or "").strip().lower()
    if mode in {"none", "single_exponential", "double_exponential"}:
        return mode
    return "none"


def _resolve_dynamic_fit_settings(analysis_out: str) -> tuple[str, bool, str]:
    """
    Resolve fit settings from config_used.yaml in analysis_out.
    Fallback is rolling_filtered_to_raw with baseline-subtract disabled.
    """
    cfg_path = os.path.join(analysis_out, "config_used.yaml")
    if not os.path.exists(cfg_path):
        return "rolling_filtered_to_raw", False, "none"
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
        bleach_mode = _normalize_bleach_correction_mode(
            data.get("bleach_correction_mode", "none")
        )
        return mode, baseline_subtract, bleach_mode
    except Exception:
        pass
    return "rolling_filtered_to_raw", False, "none"


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


def _bleach_mode_label(mode_raw: str) -> str:
    mode = _normalize_bleach_correction_mode(mode_raw)
    if mode == "double_exponential":
        return "double exponential"
    if mode == "single_exponential":
        return "single exponential"
    return "off"


def _reconstruct_bleach_series_from_attrs(
    trace: np.ndarray,
    time_s: np.ndarray,
    attrs: dict,
    *,
    prefix: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        fit_succeeded = bool(attrs.get(f"{prefix}_fit_succeeded", False))
    except Exception:
        fit_succeeded = False
    if not fit_succeeded:
        return None, None

    t = np.asarray(time_s, dtype=float).reshape(-1)
    raw = np.asarray(trace, dtype=float).reshape(-1)
    fit_model = str(attrs.get(f"{prefix}_fit_model", "")).strip().lower()
    if not fit_model:
        fit_model = (
            "double_exponential"
            if (f"{prefix}_tau_fast_sec" in attrs or f"{prefix}_tau_slow_sec" in attrs)
            else "single_exponential"
        )

    if fit_model == "double_exponential":
        try:
            tau_fast = float(attrs.get(f"{prefix}_tau_fast_sec"))
            amp_fast = float(attrs.get(f"{prefix}_amplitude_fast"))
            tau_slow = float(attrs.get(f"{prefix}_tau_slow_sec"))
            amp_slow = float(attrs.get(f"{prefix}_amplitude_slow"))
            offset = float(attrs.get(f"{prefix}_offset"))
        except Exception:
            return None, None
        if (
            (not np.isfinite(tau_fast))
            or (not np.isfinite(tau_slow))
            or tau_fast <= 0.0
            or tau_slow <= 0.0
            or (not np.isfinite(amp_fast))
            or (not np.isfinite(amp_slow))
            or (not np.isfinite(offset))
        ):
            return None, None
        if tau_fast > tau_slow:
            tau_fast, tau_slow = tau_slow, tau_fast
            amp_fast, amp_slow = amp_slow, amp_fast
        decay = amp_fast * np.exp(-t / tau_fast) + amp_slow * np.exp(-t / tau_slow)
    else:
        try:
            tau_sec = float(attrs.get(f"{prefix}_tau_sec"))
            amplitude = float(attrs.get(f"{prefix}_amplitude"))
            offset = float(attrs.get(f"{prefix}_offset"))
        except Exception:
            return None, None
        if not (np.isfinite(tau_sec) and tau_sec > 0.0 and np.isfinite(amplitude) and np.isfinite(offset)):
            return None, None
        decay = amplitude * np.exp(-t / tau_sec)
    fit_trace = decay + offset
    corrected = raw - decay
    return fit_trace, corrected


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
    bleach_correction_mode: str = "none",
    sig_bleach_fit: np.ndarray | None = None,
    sig_bleach_corrected: np.ndarray | None = None,
    iso_bleach_fit: np.ndarray | None = None,
    iso_bleach_corrected: np.ndarray | None = None,
):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # 1) Original traces + bleach fits (same original frame)
    ax1.plot(t, sig, 'g', label='Signal (470nm)', lw=0.8)
    ax1.plot(t, iso, 'm', label='Iso (415nm)', lw=0.8, alpha=0.7)
    if sig_bleach_fit is not None:
        ax1.plot(
            t,
            np.asarray(sig_bleach_fit, dtype=float),
            color="#1f77b4",
            lw=0.8,
            linestyle=":",
            label="Signal bleach fit",
        )
    if iso_bleach_fit is not None:
        ax1.plot(
            t,
            np.asarray(iso_bleach_fit, dtype=float),
            color="#7d3c98",
            lw=0.8,
            linestyle=":",
            label="Iso bleach fit",
        )
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Raw Output (V)")
    ax1.set_title(
        f"Stage 1 - Original Inputs + Bleach Fits | ROI {roi} | Chunk {chunk_id}"
    )
    ax1.grid(True, alpha=0.3)

    # 2) Bleach-corrected traces (engine input frame when bleach is enabled)
    sig_engine = (
        np.asarray(sig_bleach_corrected, dtype=float)
        if sig_bleach_corrected is not None
        else np.asarray(sig, dtype=float)
    )
    iso_engine = (
        np.asarray(iso_bleach_corrected, dtype=float)
        if iso_bleach_corrected is not None
        else np.asarray(iso, dtype=float)
    )
    ax2.plot(t, sig_engine, 'g', label='Signal (bleach-corrected)', lw=0.8)
    ax2.plot(t, iso_engine, 'm', label='Iso (bleach-corrected)', lw=0.8, alpha=0.7)
    ax2.legend(loc='upper right')
    ax2.set_ylabel("Bleach-corrected (V)")
    ax2.set_title("Stage 2 - Bleach-corrected Inputs")
    if _normalize_bleach_correction_mode(bleach_correction_mode) == "none":
        ax2.text(
            0.01,
            0.99,
            "Bleach correction disabled: traces equal original inputs",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "0.7"},
        )
    ax2.grid(True, alpha=0.3)

    # 3) Dynamic fit view in engine frame (no mixed-frame overlays)
    fit_engine = np.asarray(fit, dtype=float)
    if sig_bleach_corrected is not None:
        # fit is stored reconciled to original frame; subtract removed signal decay
        # so this panel remains in the same frame dynamic fit actually used.
        sig_decay_removed = np.asarray(sig, dtype=float) - np.asarray(sig_bleach_corrected, dtype=float)
        fit_engine = fit_engine - sig_decay_removed
    ax3.plot(t, sig_engine, 'g', label='Signal (fit input frame)', lw=0.8)
    ax3.plot(t, fit_engine, 'k', label='Dynamic fit reference (same frame)', lw=0.8, linestyle='--')
    ax3.set_ylabel("Raw Output (V)")
    ax3.set_title(
        "Stage 3 - Dynamic Reference Fitting "
        f"({_dynamic_fit_mode_label(dynamic_fit_mode)}; "
        f"{_dynamic_fit_honesty_suffix(dynamic_fit_mode, baseline_subtract_before_fit)}; "
        f"bleach correction: {_bleach_mode_label(bleach_correction_mode)})"
    )
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 4) Final dFF output stage
    ax4.plot(t, dff, 'b', label='dFF (Phasic)', lw=0.8)
    ax4.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax4.legend(loc='upper right')
    ax4.set_ylabel("dFF")
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Stage 4 - Final Corrected dF/F")
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
    
    dynamic_fit_mode, baseline_subtract_before_fit, bleach_correction_mode = _resolve_dynamic_fit_settings(args.analysis_out)
    sig_bleach_fit = None
    sig_bleach_corrected = None
    iso_bleach_fit = None
    iso_bleach_corrected = None
    if bleach_correction_mode != "none":
        try:
            with open_phasic_cache(cache_path) as f:
                grp = f[f"roi/{args.roi}/chunk_{args.chunk_id}"]
                attrs = dict(grp.attrs.items())
                sig_bleach_fit, sig_bleach_corrected = _reconstruct_bleach_series_from_attrs(
                    sig,
                    t,
                    attrs,
                    prefix="bleach_signal",
                )
                iso_bleach_fit, iso_bleach_corrected = _reconstruct_bleach_series_from_attrs(
                    iso,
                    t,
                    attrs,
                    prefix="bleach_iso",
                )
        except Exception:
            sig_bleach_fit = None
            sig_bleach_corrected = None
            iso_bleach_fit = None
            iso_bleach_corrected = None

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
        bleach_correction_mode=bleach_correction_mode,
        sig_bleach_fit=sig_bleach_fit,
        sig_bleach_corrected=sig_bleach_corrected,
        iso_bleach_fit=iso_bleach_fit,
        iso_bleach_corrected=iso_bleach_corrected,
    )
    plt.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.out}")

if __name__ == '__main__':
    main()
