"""Standalone dynamic-fit slope constraint example.

Run from the repository root:

    python examples/standalone_dynamic_fit_slope_constraint.py

This imports examples/dynamic_fitting_standalone.py, a single-file standalone
utility that can be copied into another folder without the photometry_pipeline
package or slope_qc.py.

The package file photometry_pipeline/core/dynamic_fitting.py is package-oriented
and imports slope_qc.py. For a single-file standalone copy, use
examples/dynamic_fitting_standalone.py.

The nonnegative slope constraint is an advanced diagnostic. It tests
whether the fitted reference remains supported when UV/reference-to-signal
coupling is required to be nonnegative. Collapsed, flat, or fallback fits
indicate unsupported negative or mixed-sign coupling, not a general correction
improvement.
"""

from __future__ import annotations

import numpy as np

from dynamic_fitting_standalone import (
    fit_adaptive_event_gated_regression,
)


def make_demo_trace() -> tuple[np.ndarray, np.ndarray, float]:
    fs = 40.0
    t = np.arange(2400, dtype=float) / fs
    uv = 3.0 + 0.25 * np.sin(2.0 * np.pi * 0.08 * t)
    sig = 1.2 * uv + 0.7

    # Broad event-like region where local anti-correlation can drive negative
    # adaptive slopes in an unconstrained fit.
    event = (t >= 22.0) & (t <= 36.0)
    uv[event] -= 0.7 * np.sin(np.pi * (t[event] - 22.0) / 14.0)
    sig[event] += 1.3 * np.sin(np.pi * (t[event] - 22.0) / 14.0)
    return sig, uv, fs


def run_fit(sig: np.ndarray, uv: np.ndarray, fs: float, *, slope_constraint: str) -> dict:
    return fit_adaptive_event_gated_regression(
        signal_raw=sig,
        iso_raw=uv,
        signal_fit_input=sig,
        iso_fit_input=uv,
        sample_rate_hz=fs,
        residual_z_thresh=50.0,
        local_var_window_sec=None,
        local_var_ratio_thresh=None,
        smooth_window_sec=1.0,
        min_trust_fraction=0.1,
        freeze_interp_method="linear_hold",
        signal_excursion_polarity="both",
        slope_constraint=slope_constraint,
        min_slope=0.0,
    )


def main() -> None:
    sig, uv, fs = make_demo_trace()

    unconstrained = run_fit(sig, uv, fs, slope_constraint="unconstrained")
    constrained = run_fit(sig, uv, fs, slope_constraint="nonnegative")

    print("Unconstrained negative slope fraction:")
    print(unconstrained["slope_summary"]["slope_negative_fraction"])
    print("Constrained negative slope fraction:")
    print(constrained["slope_summary"]["slope_negative_fraction"])
    print("Constraint applied:")
    print(constrained["slope_constraint_summary"]["slope_constraint_applied"])
    print("Clamped slope fraction (final check):")
    print(constrained["slope_constraint_summary"]["slope_clamped_fraction"])
    print("Unconstrained diagnostic negative slope fraction:")
    print(constrained["unconstrained_slope_summary"]["slope_negative_fraction"])
    print("Final constrained diagnostic negative slope fraction:")
    print(constrained["slope_summary"]["slope_negative_fraction"])
    print("Negative slope support windows/samples:")
    print(constrained["slope_constraint_summary"]["n_negative_slope_support_windows"])
    print("Negative slope support fraction:")
    print(constrained["slope_constraint_summary"]["negative_slope_support_fraction"])
    print("Valid nonnegative support windows/samples:")
    print(constrained["slope_constraint_summary"]["n_valid_nonnegative_support_windows"])
    print("Valid nonnegative support fraction:")
    print(constrained["slope_constraint_summary"]["valid_nonnegative_support_fraction"])
    print("Fallback used:")
    print(constrained["slope_constraint_summary"]["fallback_used"])
    print("Fallback reason:")
    print(constrained["slope_constraint_summary"]["fallback_reason"])

    # Useful arrays for plotting or further inspection:
    coef_slope = constrained["coef_slope"]
    coef_slope_unconstrained = constrained["coef_slope_unconstrained"]
    fit_reference = constrained["iso_fit_signal_units"]
    delta_f = sig - fit_reference
    print("Array shapes:", coef_slope.shape, coef_slope_unconstrained.shape, delta_f.shape)


if __name__ == "__main__":
    main()
