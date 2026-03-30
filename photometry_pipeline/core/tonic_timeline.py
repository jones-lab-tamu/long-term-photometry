from __future__ import annotations

from typing import Any

import numpy as np

TONIC_TIMELINE_MODE_REAL_ELAPSED = "real_elapsed_time"
TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED = "gap_free_elapsed_time"
# Backward-compatible alias name used in prior patch/configs.
TONIC_TIMELINE_MODE_COMPRESSED = TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED

_TONIC_TIMELINE_MODE_ALIASES = {
    "real_elapsed_time": TONIC_TIMELINE_MODE_REAL_ELAPSED,
    "elapsed": TONIC_TIMELINE_MODE_REAL_ELAPSED,
    "real": TONIC_TIMELINE_MODE_REAL_ELAPSED,
    "gap_free_elapsed_time": TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
    "gap_free": TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
    "compressed_recording_time": TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
    "compressed": TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
}

_TONIC_TIMELINE_MODE_LABELS = {
    TONIC_TIMELINE_MODE_REAL_ELAPSED: "Real elapsed time",
    TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED: "Gap-free elapsed time",
}

_TONIC_TIMELINE_AXIS_LABELS = {
    TONIC_TIMELINE_MODE_REAL_ELAPSED: "Time (hours)",
    TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED: "Gap-free elapsed time (hours)",
}


def normalize_tonic_timeline_mode(mode_raw: str | None) -> str:
    if mode_raw is None:
        return TONIC_TIMELINE_MODE_REAL_ELAPSED
    key = str(mode_raw).strip()
    if not key:
        return TONIC_TIMELINE_MODE_REAL_ELAPSED
    if key in _TONIC_TIMELINE_MODE_ALIASES:
        return _TONIC_TIMELINE_MODE_ALIASES[key]
    raise ValueError(
        f"Unknown tonic_timeline_mode={mode_raw!r}. "
        "Allowed: {'real_elapsed_time', 'gap_free_elapsed_time'} "
        "(legacy alias 'compressed_recording_time' is also accepted)."
    )


def tonic_timeline_mode_label(mode_raw: str | None) -> str:
    mode = normalize_tonic_timeline_mode(mode_raw)
    return _TONIC_TIMELINE_MODE_LABELS.get(mode, mode)


def tonic_timeline_axis_label(mode_raw: str | None) -> str:
    mode = normalize_tonic_timeline_mode(mode_raw)
    return _TONIC_TIMELINE_AXIS_LABELS.get(mode, "Time (hours)")


def _infer_dt(time_sec: np.ndarray, fallback_dt_sec: float | None) -> float:
    finite = np.asarray(time_sec, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size >= 2:
        diffs = np.diff(finite)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size > 0:
            return float(np.median(diffs))
    if fallback_dt_sec is not None and np.isfinite(fallback_dt_sec) and fallback_dt_sec > 0:
        return float(fallback_dt_sec)
    return 0.0


def build_tonic_chunk_time_axis(
    *,
    time_sec_local: np.ndarray,
    timeline_mode_raw: str | None,
    chunk_sequence_index: int,
    actual_schedule_index: int | None,
    stride_sec: float | None,
    prev_chunk_end_sec: float | None,
    prev_dt_sec: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build per-chunk absolute/compressed timeline in seconds.

    Returns:
      (time_sec_axis, state_dict)
      state_dict keys:
        - prev_chunk_end_sec
        - prev_dt_sec
        - effective_mode
    """
    mode = normalize_tonic_timeline_mode(timeline_mode_raw)
    t_local = np.asarray(time_sec_local, dtype=float)
    if t_local.ndim != 1 or t_local.size == 0:
        return t_local.copy(), {
            "prev_chunk_end_sec": prev_chunk_end_sec,
            "prev_dt_sec": prev_dt_sec,
            "effective_mode": mode,
        }

    dt_sec = _infer_dt(t_local, prev_dt_sec)
    t_zero = t_local - float(t_local[0])

    if mode == TONIC_TIMELINE_MODE_REAL_ELAPSED:
        if stride_sec is not None and np.isfinite(stride_sec) and stride_sec > 0:
            if actual_schedule_index is not None:
                t_axis = (float(actual_schedule_index) * float(stride_sec)) + t_local
            else:
                t_axis = (float(chunk_sequence_index) * float(stride_sec)) + t_local
        else:
            start = 0.0 if prev_chunk_end_sec is None else float(prev_chunk_end_sec) + dt_sec
            t_axis = start + t_zero
    else:
        start = 0.0 if prev_chunk_end_sec is None else float(prev_chunk_end_sec) + dt_sec
        t_axis = start + t_zero

    state = {
        "prev_chunk_end_sec": float(t_axis[-1]),
        "prev_dt_sec": float(dt_sec),
        "effective_mode": mode,
    }
    return t_axis, state


def remap_gapfree_axis_to_elapsed_span(
    time_sec_gapfree: np.ndarray,
    *,
    elapsed_start_sec: float,
    elapsed_end_sec: float,
) -> np.ndarray:
    """
    Remap a gap-free monotonic axis onto the full real elapsed span.

    This removes OFF gaps visually while preserving full elapsed start/end extent.
    """
    out = np.asarray(time_sec_gapfree, dtype=float).copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out

    src = out[finite]
    src_min = float(np.min(src))
    src_max = float(np.max(src))
    src_span = src_max - src_min
    elapsed_span = float(elapsed_end_sec) - float(elapsed_start_sec)

    if src_span <= 0 or not np.isfinite(src_span):
        out[finite] = float(elapsed_start_sec)
        return out
    if elapsed_span < 0 or not np.isfinite(elapsed_span):
        return out

    scale = elapsed_span / src_span
    out[finite] = float(elapsed_start_sec) + (src - src_min) * scale
    return out
