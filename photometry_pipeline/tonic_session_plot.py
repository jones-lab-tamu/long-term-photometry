"""Per-ROI plot of the session-level repeated-session tonic result.

Renders exactly what ``tonic_session_summary.csv`` already contains: one point
per authoritative session, on the real recording timeline, using that ROI's own
single tonic method and units. No tonic mathematics happens here -- this module
only reads the saved summary and draws it.

Sessions that did not produce a value (missing, insufficient_samples,
no_finite_samples, invalid_denominator, tonic_unavailable) are left as visible
gaps: they are never plotted and the connecting line is broken across them, so
no interpolation can imply data that does not exist.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional, Sequence

TONIC_SESSION_PLOT_FILENAME = "tonic_session_summary.png"

METHOD_GLOBAL_ISOSBESTIC = "global_isosbestic_tonic_dff"
METHOD_SIGNAL_ONLY = "signal_only_bleach_corrected_tonic_f"

#: Scientist-facing name of each tonic method.
TONIC_METHOD_LABELS = {
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

STATUS_VALID = "valid"


class TonicSessionPlotError(RuntimeError):
    """The saved session-level tonic summary could not be rendered."""


def _summary_path(run_dir: str, summary_path: Optional[str]) -> str:
    if summary_path:
        return str(summary_path)
    from photometry_pipeline.tonic_session_summary import (
        TONIC_SESSION_SUMMARY_FILENAME,
    )

    return os.path.join(str(run_dir), TONIC_SESSION_SUMMARY_FILENAME)


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


def _time_axis(start_texts: Sequence[str]) -> tuple[list[float], str]:
    """Elapsed hours from the first session start, or session index."""
    parsed = []
    for text in start_texts:
        value = str(text or "").strip()
        if not value or value.lower() == "nan":
            parsed = []
            break
        try:
            parsed.append(datetime.fromisoformat(value))
        except ValueError:
            parsed = []
            break
    if parsed:
        origin = min(parsed)
        return (
            [float((moment - origin).total_seconds()) / 3600.0 for moment in parsed],
            "Time (hours from first session)",
        )
    return [], ""


def generate_tonic_session_plots(
    run_dir: str,
    *,
    summary_path: Optional[str] = None,
    rois: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Write one tonic session plot per ROI beside that ROI's other summaries."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    frame = read_tonic_session_summary(run_dir, summary_path=summary_path)
    if frame is None:
        raise TonicSessionPlotError(
            "The session-level tonic summary required for the Tonic view is missing."
        )

    selected = [str(roi) for roi in rois] if rois else sorted({str(v) for v in frame["roi"]})
    results: list[dict[str, Any]] = []

    for roi in selected:
        group = frame[frame["roi"].astype(str) == roi]
        if group.empty:
            continue
        group = group.sort_values("session_index", kind="stable")

        method = str(group["tonic_method"].iloc[0])
        units = str(group["units"].iloc[0])
        title_label = TONIC_METHOD_LABELS.get(method, "Tonic")
        y_label = TONIC_METHOD_Y_LABELS.get(method, "Tonic value")

        statuses = [str(value) for value in group["status"]]
        values = [float(value) for value in group["tonic_value"]]
        indices = [int(value) for value in group["session_index"]]
        hours, x_label = _time_axis(list(group.get("session_start_time", [])))
        if not hours:
            hours = [float(index) for index in indices]
            x_label = "Session index"

        summary_dir = os.path.join(str(run_dir), roi, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        out_path = os.path.join(summary_dir, TONIC_SESSION_PLOT_FILENAME)

        figure, axis = plt.subplots(figsize=(11, 4.2))
        try:
            for run in _valid_runs(statuses):
                if len(run) > 1:
                    axis.plot(
                        [hours[i] for i in run],
                        [values[i] for i in run],
                        color="#1f4e79",
                        linewidth=1.2,
                        zorder=2,
                    )
            plotted = [i for i, status in enumerate(statuses) if status == STATUS_VALID]
            axis.plot(
                [hours[i] for i in plotted],
                [values[i] for i in plotted],
                linestyle="none",
                marker="o",
                markersize=3.5,
                color="#1f4e79",
                zorder=3,
            )
            axis.set_title(f"{roi} — {title_label}")
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.grid(True, alpha=0.3)
            if method == METHOD_GLOBAL_ISOSBESTIC:
                axis.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":", zorder=1)
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
                "y_label": y_label,
                "x_label": x_label,
                "n_plotted": len(plotted),
                "n_sessions": len(statuses),
            }
        )

    return results
