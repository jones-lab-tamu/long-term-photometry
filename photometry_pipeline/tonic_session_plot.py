"""Per-ROI figure for the session-level repeated-session tonic result.

Two vertically stacked panels sharing one elapsed-recording-time axis:

* top -- the raw cached channels (``sig_raw`` / ``uv_raw``) across the recording,
  as observational context only. No correction, fitting, or bleach flattening is
  applied here, and the old black residual trace is never drawn.
* bottom -- the saved session-level tonic values from
  ``tonic_session_summary.csv``, one point per session, in that ROI's own tonic
  method and units.

No tonic mathematics happens in this module: it reads the already-saved summary
and the already-cached raw channels and draws them.

Elapsed time comes from the same authoritative session index the summary itself
was built from (``expected_start_time`` per session slot), so recorded and
missing sessions keep their true positions and real gaps keep their real width.
Sessions are never placed back-to-back, and the obsolete daily-anchor timeline is
not used.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence

TONIC_SESSION_PLOT_FILENAME = "tonic_session_summary.png"

METHOD_GLOBAL_ISOSBESTIC = "global_isosbestic_tonic_dff"
METHOD_SIGNAL_ONLY = "signal_only_bleach_corrected_tonic_f"

#: Scientist-facing name of each tonic method.
TONIC_METHOD_LABELS = {
    METHOD_GLOBAL_ISOSBESTIC: "Global-isosbestic ΔF/F₀",
    METHOD_SIGNAL_ONLY: "Tonic F, signal-only bleach corrected",
}

# Keep the compact figure title stable while the Results note identifies the
# primary method more precisely.
TONIC_PLOT_TITLE_LABELS = {
    METHOD_GLOBAL_ISOSBESTIC: "Tonic ΔF/F₀",
    METHOD_SIGNAL_ONLY: "Tonic F, signal-only bleach corrected",
}

#: Y-axis label of each tonic method, carrying its real units.
TONIC_METHOD_Y_LABELS = {
    METHOD_GLOBAL_ISOSBESTIC: "Tonic ΔF/F₀ (fraction)",
    METHOD_SIGNAL_ONLY: "Tonic F (raw fluorescence, AU)",
}

TONIC_FALLBACK_NOTE = (
    "The global isosbestic fit was unusable for this ROI. Tonic is shown as "
    "signal-only, bleach-corrected fluorescence."
)

#: Same bounded display budget the previous tonic overview used for the raw
#: channels, so a multi-day recording stays renderable.
RAW_OVERVIEW_TARGET_POINTS = 30000

ELAPSED_AXIS_LABEL = "Elapsed recording time (hours)"
SESSION_INDEX_AXIS_LABEL = "Session index"

RAW_SIGNAL_LABEL = "Raw signal"
RAW_ISOSBESTIC_LABEL = "Raw isosbestic"

STATUS_VALID = "valid"


class TonicSessionPlotError(RuntimeError):
    """The saved session-level tonic result could not be rendered."""


def _summary_path(run_dir: str, summary_path: Optional[str]) -> str:
    if summary_path:
        return str(summary_path)
    from photometry_pipeline.tonic_session_summary import (
        TONIC_SESSION_SUMMARY_FILENAME,
    )

    return os.path.join(str(run_dir), TONIC_SESSION_SUMMARY_FILENAME)


def _tonic_out_dir(run_dir: str, tonic_out_dir: Optional[str]) -> str:
    if tonic_out_dir:
        return str(tonic_out_dir)
    return os.path.join(str(run_dir), "_analysis", "tonic_out")


def read_tonic_session_summary(run_dir: str, *, summary_path: Optional[str] = None):
    """Return the saved summary table, or ``None`` when the run has none."""
    import pandas as pd

    path = _summary_path(run_dir, summary_path)
    if not os.path.isfile(path):
        return None
    frame = pd.read_csv(path)
    if "roi" not in frame.columns or "tonic_value" not in frame.columns:
        raise TonicSessionPlotError(
            f"The session-level tonic summary is missing required columns: {path}"
        )
    return frame


def tonic_method_by_roi(run_dir: str, *, summary_path: Optional[str] = None) -> dict[str, dict[str, str]]:
    """One record per ROI describing the tonic method that run actually used.

    Returns ``{}`` when the run predates the session-level tonic summary, so a
    caller can simply fall back to its previous behavior with no migration code.
    """
    frame = read_tonic_session_summary(run_dir, summary_path=summary_path)
    if frame is None or frame.empty:
        return {}
    out: dict[str, dict[str, str]] = {}
    for roi, group in frame.groupby("roi", sort=True):
        reasons = [
            str(value)
            for value in group.get("fallback_reason", [])
            if str(value) and str(value).lower() != "nan"
        ]
        out[str(roi)] = {
            "tonic_method": str(group["tonic_method"].iloc[0]),
            "units": str(group["units"].iloc[0]),
            "fallback_reason": reasons[0] if reasons else "",
        }
    return out


def tonic_method_label(method: str) -> str:
    """Scientist-facing name of one tonic method."""
    return TONIC_METHOD_LABELS.get(str(method), "")


def _valid_runs(statuses: Sequence[str]) -> list[list[int]]:
    """Contiguous runs of plotted sessions; any other status breaks the line."""
    runs: list[list[int]] = []
    current: list[int] = []
    for position, status in enumerate(statuses):
        if str(status) == STATUS_VALID:
            current.append(position)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def session_elapsed_seconds(records: Sequence[dict]) -> dict[int, float]:
    """Elapsed seconds per authoritative session slot, or ``{}`` if untimed.

    ``expected_start_time`` is the authoritative placement the previous tonic
    overview also used, and it is present for recorded *and* approved-missing
    slots, so real gaps keep their real width. A slot without its own timestamp
    is interpolated only between its nearest timed neighbors, or extrapolated
    from the nearest usable local cadence at one end of the recording. An
    unsafe slot is left unplaced.
    """
    starts: dict[int, Any] = {}
    for record in records:
        index = record.get("session_index")
        start = record.get("expected_start_time")
        if index is None or start is None:
            continue
        starts[int(index)] = start
    if not starts:
        return {}

    ordered = sorted(starts)
    origin_index = ordered[0]
    origin = starts[origin_index]
    elapsed = {
        index: float((starts[index] - origin).total_seconds()) for index in ordered
    }

    session_indices = sorted(
        {
            int(record["session_index"])
            for record in records
            if record.get("session_index") is not None
        }
    )
    untimed = [
        index
        for index in session_indices
        if index not in elapsed
    ]
    if untimed:
        def _local_stride(left: int, right: int) -> Optional[float]:
            if right <= left:
                return None
            duration = elapsed[right] - elapsed[left]
            if not math.isfinite(duration) or duration <= 0.0:
                return None
            stride = duration / float(right - left)
            return stride if math.isfinite(stride) and stride > 0.0 else None

        def _safe_inference(index: int, candidate: float) -> bool:
            """Accept only a placement that stays strictly monotonic."""
            earlier = [value for position, value in elapsed.items() if position < index]
            later = [value for position, value in elapsed.items() if position > index]
            return (
                math.isfinite(candidate)
                and all(candidate > value for value in earlier)
                and all(candidate < value for value in later)
            )

        # Each interior run is bounded by its own nearest authoritative times.
        # This deliberately avoids allowing a distant recording gap to set the
        # cadence of an otherwise local interpolation.
        for left, right in zip(ordered, ordered[1:]):
            stride = _local_stride(left, right)
            if stride is None:
                continue
            for index in untimed:
                if not left < index < right:
                    continue
                candidate = elapsed[left] + float(index - left) * stride
                if elapsed[left] < candidate < elapsed[right] and _safe_inference(
                    index, candidate
                ):
                    elapsed[index] = candidate

        # Leading slots use the first usable timed interval to their right.
        leading = [index for index in untimed if index < ordered[0]]
        if leading:
            stride = None
            for left, right in zip(ordered, ordered[1:]):
                stride = _local_stride(left, right)
                if stride is not None:
                    break
            if stride is not None:
                for index in leading:
                    candidate = elapsed[ordered[0]] - float(ordered[0] - index) * stride
                    if candidate < elapsed[ordered[0]] and _safe_inference(index, candidate):
                        elapsed[index] = candidate

        # Trailing slots use the last usable timed interval to their left.
        trailing = [index for index in untimed if index > ordered[-1]]
        if trailing:
            stride = None
            for left, right in reversed(list(zip(ordered, ordered[1:]))):
                stride = _local_stride(left, right)
                if stride is not None:
                    break
            if stride is not None:
                for index in trailing:
                    candidate = elapsed[ordered[-1]] + float(index - ordered[-1]) * stride
                    if candidate > elapsed[ordered[-1]] and _safe_inference(index, candidate):
                        elapsed[index] = candidate

    floor = min(elapsed.values())
    return {index: value - floor for index, value in elapsed.items()}


def _raw_decimation(total_points: int) -> int:
    if total_points <= RAW_OVERVIEW_TARGET_POINTS:
        return 1
    return max(1, int(math.ceil(float(total_points) / float(RAW_OVERVIEW_TARGET_POINTS))))


def _load_raw_sessions(cache, roi: str, records: Sequence[dict], elapsed_sec: dict[int, float]):
    """Per-session raw traces placed on the elapsed axis, one entry per session."""
    from photometry_pipeline.io.hdf5_cache_reader import load_cache_chunk_fields

    import numpy as np

    segments = []
    total = 0
    for record in records:
        chunk_id = record.get("cache_chunk_id")
        index = record.get("session_index")
        if chunk_id is None or index is None:
            continue
        time_sec, sig_raw, uv_raw = load_cache_chunk_fields(
            cache, roi, int(chunk_id), ["time_sec", "sig_raw", "uv_raw"]
        )
        local = np.asarray(time_sec, dtype=float).reshape(-1)
        offset = elapsed_sec.get(int(index))
        if offset is None:
            # With partial authoritative timing, an unplaced slot must remain
            # absent. Only a recording with no timing at all uses session order.
            if elapsed_sec:
                continue
            # In session-index mode the raw samples share the tonic points'
            # coordinates. Keep each trace inside its own slot instead of
            # converting its local recording seconds to elapsed hours.
            finite_local = local[np.isfinite(local)]
            if finite_local.size < 2:
                hours = np.full(local.shape, float(index), dtype=float)
            else:
                duration = float(finite_local[-1] - finite_local[0])
                if not math.isfinite(duration) or duration <= 0.0:
                    hours = np.full(local.shape, float(index), dtype=float)
                else:
                    normalized = np.full(local.shape, np.nan, dtype=float)
                    finite_mask = np.isfinite(local)
                    normalized[finite_mask] = (
                        local[finite_mask] - finite_local[0]
                    ) / duration
                    normalized[finite_mask] = np.clip(
                        normalized[finite_mask], 0.0, 1.0
                    )
                    hours = float(index) + normalized * 0.8
        else:
            hours = (offset + local) / 3600.0
        segments.append(
            {
                "hours": hours,
                "sig": np.asarray(sig_raw, dtype=float).reshape(-1),
                "uv": np.asarray(uv_raw, dtype=float).reshape(-1),
            }
        )
        total += int(local.size)
    return segments, total


def generate_tonic_session_plots(
    run_dir: str,
    *,
    summary_path: Optional[str] = None,
    tonic_out_dir: Optional[str] = None,
    rois: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Write one two-panel tonic figure per ROI beside that ROI's summaries."""
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    from matplotlib import pyplot as plt

    from photometry_pipeline.io.hdf5_cache_reader import (
        list_cache_rois,
        open_tonic_cache,
    )

    # The same authoritative session records the saved summary was built from,
    # so the figure's timeline cannot drift from the result it displays.
    from photometry_pipeline.tonic_session_summary import _authoritative_sessions

    frame = read_tonic_session_summary(run_dir, summary_path=summary_path)
    if frame is None:
        raise TonicSessionPlotError(
            "The session-level tonic summary required for the Tonic view is missing."
        )

    analysis_dir = _tonic_out_dir(run_dir, tonic_out_dir)
    cache_path = os.path.join(analysis_dir, "tonic_trace_cache.h5")
    if not os.path.isfile(cache_path):
        raise TonicSessionPlotError(
            f"The tonic trace cache required for the raw overview is missing: {cache_path}"
        )

    selected = [str(roi) for roi in rois] if rois else sorted({str(v) for v in frame["roi"]})
    results: list[dict[str, Any]] = []

    cache = open_tonic_cache(cache_path)
    try:
        available = list_cache_rois(cache)
        records = _authoritative_sessions(analysis_dir, cache)
        elapsed_sec = session_elapsed_seconds(records)
        x_label = ELAPSED_AXIS_LABEL if elapsed_sec else SESSION_INDEX_AXIS_LABEL

        for roi in selected:
            group = frame[frame["roi"].astype(str) == roi]
            if group.empty or roi not in available:
                continue
            group = group.sort_values("session_index", kind="stable")

            method = str(group["tonic_method"].iloc[0])
            units = str(group["units"].iloc[0])
            title_label = TONIC_PLOT_TITLE_LABELS.get(method, "Tonic")
            y_label = TONIC_METHOD_Y_LABELS.get(method, "Tonic value")

            statuses = [str(value) for value in group["status"]]
            values = [float(value) for value in group["tonic_value"]]
            indices = [int(value) for value in group["session_index"]]
            if elapsed_sec:
                hours = [elapsed_sec.get(index, float("nan")) / 3600.0 for index in indices]
            else:
                hours = [float(index) for index in indices]

            segments, total_points = _load_raw_sessions(cache, roi, records, elapsed_sec)
            stride = _raw_decimation(total_points)

            summary_dir = os.path.join(str(run_dir), roi, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            out_path = os.path.join(summary_dir, TONIC_SESSION_PLOT_FILENAME)

            figure, (raw_axis, tonic_axis) = plt.subplots(
                2, 1, figsize=(11, 7.2), sharex=True
            )
            try:
                # -- top panel: raw channels, one line run per session so real
                #    recording gaps stay empty instead of being bridged.
                for position, segment in enumerate(segments):
                    raw_axis.plot(
                        segment["hours"][::stride],
                        segment["sig"][::stride],
                        color="#2e7d32",
                        linewidth=0.6,
                        label=RAW_SIGNAL_LABEL if position == 0 else "_nolegend_",
                    )
                    raw_axis.plot(
                        segment["hours"][::stride],
                        segment["uv"][::stride],
                        color="#6a1b9a",
                        linewidth=0.6,
                        alpha=0.85,
                        label=RAW_ISOSBESTIC_LABEL if position == 0 else "_nolegend_",
                    )
                raw_axis.set_ylabel("Raw fluorescence (AU)")
                raw_axis.set_title(f"{roi} — signal and isosbestic overview")
                raw_axis.grid(True, alpha=0.3)
                if segments:
                    raw_axis.legend(loc="upper right", fontsize="small")

                # -- bottom panel: the saved session-level tonic result.
                plotted_statuses = [
                    STATUS_VALID
                    if status == STATUS_VALID and np.isfinite(hours[position])
                    else "unplaced"
                    for position, status in enumerate(statuses)
                ]
                for run in _valid_runs(plotted_statuses):
                    if len(run) > 1:
                        tonic_axis.plot(
                            [hours[i] for i in run],
                            [values[i] for i in run],
                            color="#1f4e79",
                            linewidth=1.2,
                            zorder=2,
                        )
                plotted = [
                    i
                    for i, status in enumerate(plotted_statuses)
                    if status == STATUS_VALID
                ]
                tonic_axis.plot(
                    [hours[i] for i in plotted],
                    [values[i] for i in plotted],
                    linestyle="none",
                    marker="o",
                    markersize=3.5,
                    color="#1f4e79",
                    zorder=3,
                )
                if method == METHOD_GLOBAL_ISOSBESTIC:
                    tonic_axis.axhline(
                        0.0, color="#999999", linewidth=0.8, linestyle=":", zorder=1
                    )
                tonic_axis.set_title(f"{roi} — {title_label}")
                tonic_axis.set_xlabel(x_label)
                tonic_axis.set_ylabel(y_label)
                tonic_axis.grid(True, alpha=0.3)

                figure.tight_layout()
                figure.savefig(out_path, dpi=110)
            finally:
                plt.close(figure)

            results.append(
                {
                    "roi": roi,
                    "output_path": out_path,
                    "relative_path": f"{roi}/summary/{TONIC_SESSION_PLOT_FILENAME}",
                    "tonic_method": method,
                    "units": units,
                    "title": f"{roi} — {title_label}",
                    "raw_title": f"{roi} — signal and isosbestic overview",
                    "y_label": y_label,
                    "x_label": x_label,
                    "n_plotted": len(plotted),
                    "n_sessions": len(statuses),
                    "n_raw_sessions": len(segments),
                    "raw_decimation": stride,
                    "elapsed_hours": list(hours),
                }
            )
    finally:
        cache.close()

    return results
