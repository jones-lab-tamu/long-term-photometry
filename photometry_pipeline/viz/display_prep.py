"""
Shared display-prep helpers for phasic plotting surfaces.

This module only contains display transforms. It does not alter analysis
values, cache contents, or detection behavior.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _validate_1d_trace(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got shape {arr.shape}.")
    return arr


def _robust_center(trace: np.ndarray, center_method: str) -> float:
    method = str(center_method or "median").strip().lower()
    finite = trace[np.isfinite(trace)]
    if finite.size == 0:
        raise ValueError("Cannot center trace with no finite values.")
    if method != "median":
        raise ValueError(f"Unsupported center_method '{center_method}'. Only 'median' is supported.")
    return float(np.median(finite))


def prepare_centered_common_gain(
    sig: np.ndarray,
    iso: np.ndarray,
    *,
    center_method: str = "median",
    return_centers: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Prepare raw signal/isosbestic traces for centered common-gain display.

    Semantics:
    - Center each trace independently using a robust center (default: median).
    - Do not normalize amplitudes per-trace.
    - Do not z-score, baseline-divide, or variance-match.

    Parameters
    ----------
    sig, iso:
        1D arrays of raw signal/isobestic values.
    center_method:
        Robust center method. Only ``median`` is currently supported.
    return_centers:
        When True, also return the two computed centers.

    Returns
    -------
    centered_sig, centered_iso [, sig_center, iso_center]
    """
    sig_arr = _validate_1d_trace(sig, "sig")
    iso_arr = _validate_1d_trace(iso, "iso")

    if sig_arr.shape != iso_arr.shape:
        raise ValueError(
            f"sig and iso must have the same shape; got {sig_arr.shape} and {iso_arr.shape}."
        )

    sig_center = _robust_center(sig_arr, center_method)
    iso_center = _robust_center(iso_arr, center_method)

    centered_sig = sig_arr - sig_center
    centered_iso = iso_arr - iso_center

    if return_centers:
        return centered_sig, centered_iso, sig_center, iso_center
    return centered_sig, centered_iso

