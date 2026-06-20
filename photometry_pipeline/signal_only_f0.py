"""Standalone Signal-Only F0 helpers for already-extracted table data.

This module is intentionally independent of the GUI, Guided Workflow, RunSpec,
raw vendor import, ROI discovery, and full pipeline execution. It reuses the
existing core Signal-Only F0 candidate implementation and applies the accepted
production formula:

    (signal_raw_for_dff - signal_only_f0_uncapped_for_dff)
    / signal_only_f0_uncapped_for_dff

The helper is intended for already-extracted CSV/table data. Parameter choices
should be recorded with preliminary analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from photometry_pipeline.core.signal_only_f0_candidate import (
    DEFAULTS as SIGNAL_ONLY_F0_DEFAULTS,
    compute_signal_only_f0_candidate,
)


@dataclass(frozen=True)
class SignalOnlyF0Result:
    signal: np.ndarray
    signal_only_f0: np.ndarray
    dff: np.ndarray
    parameters: dict[str, object]
    warnings: tuple[str, ...] = ()


def _as_1d_float_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    return arr


def _config_from_parameters(
    *,
    baseline_window_sec: float | None,
    percentile: float | None,
    smoothing_window_sec: float | None,
) -> tuple[dict[str, object], dict[str, str]]:
    cfg = dict(SIGNAL_ONLY_F0_DEFAULTS)
    sources = {
        "percentile": "core_default",
        "baseline_window_sec": "core_default",
        "smoothing_window_sec": "core_default",
    }
    if percentile is not None:
        quantile = float(percentile)
        if quantile > 1.0:
            quantile = quantile / 100.0
        if not np.isfinite(quantile) or quantile < 0.0 or quantile > 0.5:
            raise ValueError("percentile must correspond to a lower quantile between 0 and 50")
        cfg["signal_only_f0_low_quantile"] = quantile
        sources["percentile"] = "user_override"
    if baseline_window_sec is not None:
        cfg["signal_only_f0_window_sec"] = baseline_window_sec
        sources["baseline_window_sec"] = "user_override"
    if smoothing_window_sec is not None:
        cfg["signal_only_f0_smoothing_window_sec"] = smoothing_window_sec
        sources["smoothing_window_sec"] = "user_override"
    return cfg, sources


def _time_from_sampling_rate(n: int, sampling_rate_hz: float | None) -> np.ndarray | None:
    if sampling_rate_hz is None:
        return None
    rate = float(sampling_rate_hz)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sampling_rate_hz must be positive when provided")
    return np.arange(int(n), dtype=float) / rate


def compute_signal_only_f0_dff(
    signal,
    *,
    sampling_rate_hz: float | None = None,
    time=None,
    baseline_window_sec: float | None = None,
    percentile: float | None = None,
    smoothing_window_sec: float | None = None,
    min_f0: float | None = None,
    preserve_negative: bool = True,
) -> SignalOnlyF0Result:
    """Compute Signal-Only F0 dF/F from an already-extracted signal array.

    Negative dF/F is preserved. Huge/prolonged peaks are not clipped or
    normalized away. `min_f0` is a validation threshold only; it does not floor
    or otherwise alter the denominator.
    """
    if preserve_negative is not True:
        raise ValueError("preserve_negative=False is not supported; Signal-Only F0 preserves negative dF/F")

    signal_arr = _as_1d_float_array(signal, name="signal")
    if np.any(~np.isfinite(signal_arr)):
        raise ValueError("signal contains non-finite values; clean or segment the signal before Signal-Only F0")
    warnings: list[str] = []

    if time is not None:
        time_arr = _as_1d_float_array(time, name="time")
        if time_arr.shape != signal_arr.shape:
            raise ValueError("time must have the same length as signal")
        if sampling_rate_hz is not None:
            warnings.append("time was provided; sampling_rate_hz is ignored")
    else:
        time_arr = _time_from_sampling_rate(signal_arr.size, sampling_rate_hz)

    cfg, parameter_sources = _config_from_parameters(
        baseline_window_sec=baseline_window_sec,
        percentile=percentile,
        smoothing_window_sec=smoothing_window_sec,
    )
    diagnostics = compute_signal_only_f0_candidate(
        signal_arr,
        time_arr,
        config=cfg,
        return_uncapped_candidate=True,
    )
    denominator = diagnostics.get("signal_only_f0_candidate_uncapped")
    if denominator is None:
        warning = str(diagnostics.get("signal_only_f0_warning") or "signal_only_f0_uncapped_unavailable")
        raise ValueError(f"Signal-Only F0 denominator unavailable: {warning}")

    f0_arr = np.asarray(denominator, dtype=float).reshape(-1)
    if f0_arr.shape != signal_arr.shape:
        raise ValueError(
            f"Signal-Only F0 denominator length mismatch: signal has {signal_arr.size}, "
            f"denominator has {f0_arr.size}"
        )
    if not np.all(np.isfinite(f0_arr)):
        raise ValueError("Signal-Only F0 denominator contains non-finite values")
    if np.any(f0_arr <= 0):
        raise ValueError("Signal-Only F0 denominator contains non-positive values")
    if min_f0 is not None:
        threshold = float(min_f0)
        if not np.isfinite(threshold):
            raise ValueError("min_f0 must be finite when provided")
        if np.any(f0_arr < threshold):
            raise ValueError("Signal-Only F0 denominator is below min_f0")

    dff = (signal_arr - f0_arr) / f0_arr
    parameters: dict[str, object] = {
        "formula": "(signal - signal_only_f0_uncapped_for_dff) / signal_only_f0_uncapped_for_dff",
        "denominator_source": "signal_only_f0_candidate_uncapped",
        "negative_dff_preserved": True,
        "percentile": percentile,
        "low_quantile": float(cfg["signal_only_f0_low_quantile"]),
        "baseline_window_sec": baseline_window_sec,
        "smoothing_window_sec": smoothing_window_sec,
        "effective_baseline_window_sec": cfg["signal_only_f0_window_sec"],
        "effective_smoothing_window_sec": cfg["signal_only_f0_smoothing_window_sec"],
        "parameter_sources": parameter_sources,
        "sampling_rate_hz": sampling_rate_hz,
        "min_f0": min_f0,
        "core_status": diagnostics.get("signal_only_f0_status"),
        "core_viability": diagnostics.get("signal_only_f0_candidate_viability"),
        "core_confidence": diagnostics.get("signal_only_f0_candidate_confidence"),
        "core_flags": tuple(diagnostics.get("signal_only_f0_flags") or ()),
        "f0_capped_for_dff": False,
        "dff_clipped": False,
    }
    return SignalOnlyF0Result(
        signal=signal_arr.copy(),
        signal_only_f0=f0_arr.copy(),
        dff=np.asarray(dff, dtype=float),
        parameters=parameters,
        warnings=tuple(warnings),
    )


def compute_signal_only_f0_dff_from_csv(
    input_csv,
    *,
    signal_column: str,
    time_column: str | None = None,
    output_csv=None,
    sampling_rate_hz: float | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read an extracted CSV, append Signal-Only F0 columns, and optionally write it.

    Original columns are preserved. The input CSV is never overwritten unless the
    same path is explicitly passed as `output_csv`, which is rejected by default.
    """
    input_path = Path(input_csv)
    df = pd.read_csv(input_path)
    if signal_column not in df.columns:
        raise ValueError(f"signal column not found: {signal_column}")
    if time_column is not None and time_column not in df.columns:
        raise ValueError(f"time column not found: {time_column}")

    time_values = df[time_column].to_numpy(dtype=float) if time_column is not None else None
    result = compute_signal_only_f0_dff(
        df[signal_column].to_numpy(dtype=float),
        time=time_values,
        sampling_rate_hz=sampling_rate_hz,
        **kwargs,
    )
    out = df.copy()
    out["signal_only_f0"] = result.signal_only_f0
    out["signal_only_dff"] = result.dff

    if output_csv is not None:
        output_path = Path(output_csv)
        if output_path.resolve() == input_path.resolve():
            raise ValueError("output_csv must be distinct from input_csv")
        out.to_csv(output_path, index=False)
    return out
