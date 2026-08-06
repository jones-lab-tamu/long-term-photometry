"""Saved Results artifacts for native Guided continuous runs.

This module is deliberately a publication adapter.  It reads the accepted
continuous caches, finalized window tables, and persisted provenance; it does
not run correction, detection, event counting, or AUC calculation.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from photometry_pipeline.continuous_outputs import (
    CONTINUOUS_TRACE_OVERVIEW_MAX_POINTS,
    _allocate_trace_points,
    _require_continuous_attrs,
    _select_finite_positions_for_chunk,
)
from photometry_pipeline.guided_timeline import (
    map_elapsed_coordinate,
    parse_guided_clock,
    timeline_mode_label,
)
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    load_cache_chunk_attrs,
    load_cache_chunk_fields,
    open_phasic_cache,
    open_tonic_cache,
)


PHASIC_CORRECTION_IMPACT_FILENAME = "phasic_correction_impact.png"
PHASIC_AUC_FILENAME = "phasic_auc_timeseries.png"
PHASIC_RATE_FILENAME = "phasic_peak_rate_timeseries.png"
TONIC_OVERVIEW_FILENAME = "tonic_overview.png"
PHASIC_SUMMARY_FILENAME = "continuous_phasic_window_summary.csv"
TONIC_SUMMARY_FILENAME = "continuous_tonic_window_summary.csv"
_TONIC_SUMMARY_REQUIRED_COLUMNS = frozenset(
    {
        "roi",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "window_duration_sec",
        "elapsed_hour_start",
        "elapsed_hour_mid",
        "tonic_mean",
        "tonic_median",
        "tonic_min",
        "tonic_max",
        "tonic_p05",
        "tonic_p95",
        "tonic_n_finite",
        "tonic_nan_fraction",
        "tonic_value",
        "tonic_status",
        "tonic_method",
        "units",
        "tonic_percentile",
        "tonic_fallback",
        "fallback_reason",
        "is_partial_final_window",
        "continuous_window_sec",
        "continuous_step_sec",
        "acquisition_mode",
    }
)

_NATIVE_TONIC_METHOD_LABELS = {
    "Global-isosbestic ΔF/F₀": "Global-isosbestic ΔF/F₀",
    "Signal-only bleach corrected": "Signal-only bleach corrected",
}
_NATIVE_TONIC_UNITS_LABELS = {
    "fractional ΔF/F₀": "fractional ΔF/F₀",
    "raw fluorescence AU": "raw fluorescence AU",
}

CONTINUOUS_ARTIFACT_MAX_POINTS = CONTINUOUS_TRACE_OVERVIEW_MAX_POINTS

CONTINUOUS_DAY_PLOT_EXPLANATION = (
    "Continuous Day Plots sample two 10-minute windows from each plotted hour: "
    "minutes 00–10 and 30–40. Blank positions indicate that the complete "
    "requested interval was unavailable. These display windows do not define "
    "acquisition sessions or analysis windows."
)
_CONTINUOUS_DAY_PLOT_COLUMN_LABELS = ("00–10 min", "30–40 min")
_CONTINUOUS_DAY_PLOT_SAMPLE_SEC = 600.0
_CONTINUOUS_DAY_PLOT_SLOT_SEC = 1800.0
_CONTINUOUS_DAY_PLOT_DPI = 120
_CONTINUOUS_DAY_PLOT_SMOOTH_SEC = 1.0
_CONTINUOUS_DAY_PLOT_FAMILIES = (
    (
        "sampled_signal_reference",
        "Phasic Sig/Iso",
        "phasic_sig_iso_day_{day:03d}.png",
    ),
    (
        "sampled_correction_reference",
        "Correction Reference",
        "phasic_correction_reference_day_{day:03d}.png",
    ),
    (
        "sampled_phasic_dff",
        "Phasic dFF",
        "phasic_dFF_day_{day:03d}.png",
    ),
    (
        "sampled_stacked",
        "Phasic Stacked",
        "phasic_stacked_day_{day:03d}.png",
    ),
)
# Display-only copies live beside, never inside, the canonical day_plots root
# and are never registered as manifest artifacts.
CONTINUOUS_MARKER_ON_VARIANT_RELATIVE_DIR = os.path.join(
    "day_plots", "rerendered_display_variants", "dff_peak_markers_on"
)


class GuidedContinuousSavedArtifactError(RuntimeError):
    """The accepted continuous results cannot produce a truthful artifact."""


def _relative_path(*parts: str) -> str:
    return "/".join(str(part).replace("\\", "/").strip("/") for part in parts)


def _require_file(path: str, description: str) -> None:
    if not os.path.isfile(path):
        raise GuidedContinuousSavedArtifactError(
            f"Required {description} is missing: {path}"
        )
    if os.path.getsize(path) <= 0:
        raise GuidedContinuousSavedArtifactError(
            f"Required {description} is empty: {path}"
        )


def _validate_image(path: str, description: str) -> tuple[int, int]:
    _require_file(path, description)
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        raise GuidedContinuousSavedArtifactError(
            f"Required {description} is not a decodable image: {path} ({exc})"
        ) from exc
    if int(width) <= 0 or int(height) <= 0:
        raise GuidedContinuousSavedArtifactError(
            f"Required {description} has invalid image dimensions: {path}"
        )
    return int(width), int(height)


def continuous_plot_coordinates(
    elapsed_seconds: Any, timeline_contract: dict[str, Any]
) -> np.ndarray:
    """Map saved elapsed coordinates to one monotonic plotting axis.

    Elapsed placement stays in elapsed seconds.  Civil and fixed placement
    use the accepted Guided timeline mapper and then unwrap day/hour positions
    into one monotonic coordinate.  The returned coordinate is never
    compressed to the first plotted point, so a leading fixed/civil blank is
    preserved by the caller's x-axis lower bound of zero.
    """
    if not isinstance(timeline_contract, dict):
        raise GuidedContinuousSavedArtifactError(
            "The accepted timeline provenance is missing or unreadable."
        )
    mode = str(timeline_contract.get("timeline_mode", "")).strip().lower()
    fixed_clock = timeline_contract.get("fixed_daily_anchor_clock")
    start_clock = timeline_contract.get("recording_start_clock")
    values = np.asarray(elapsed_seconds, dtype=float).reshape(-1)
    if values.size and not np.all(np.isfinite(values)):
        raise GuidedContinuousSavedArtifactError(
            "Continuous plot coordinates contain a non-finite elapsed time."
        )
    if mode == "elapsed":
        coordinates = values.copy()
    elif mode in {"civil", "fixed_daily_anchor"}:
        mapped = [
            map_elapsed_coordinate(
                float(value),
                timeline_anchor_mode=mode,
                fixed_daily_anchor_clock=fixed_clock,
                recording_start_clock=start_clock,
            )
            for value in values
        ]
        coordinates = np.asarray(
            [float(day) * 86400.0 + float(within_day) for day, within_day in mapped],
            dtype=float,
        )
    else:
        raise GuidedContinuousSavedArtifactError(
            f"Unsupported accepted timeline mode for continuous plots: {mode!r}"
        )
    if coordinates.size and np.any(np.diff(coordinates) < -1e-8):
        raise GuidedContinuousSavedArtifactError(
            "Continuous plot coordinates are not monotonic after accepted timeline mapping."
        )
    return coordinates


def _timeline_axis_label(timeline_contract: dict[str, Any]) -> str:
    mode = str(timeline_contract.get("timeline_mode", "")).strip().lower()
    if mode == "elapsed":
        return "Elapsed time (hours from recording start)"
    if mode == "fixed_daily_anchor":
        return (
            "Fixed daily anchor time (hours from "
            f"{timeline_contract.get('fixed_daily_anchor_clock')})"
        )
    if mode == "civil":
        return "Civil-clock time (hours from day-0 midnight)"
    raise GuidedContinuousSavedArtifactError(
        f"Unsupported accepted timeline mode for plot labels: {mode!r}"
    )


def _load_phasic_auc_provenance(phasic_out_dir: str) -> dict[str, Any]:
    report_path = os.path.join(phasic_out_dir, "run_report.json")
    _require_file(report_path, "phasic analysis run report")
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except Exception as exc:
        raise GuidedContinuousSavedArtifactError(
            f"Phasic analysis run report is not readable: {report_path} ({exc})"
        ) from exc
    provenance = report.get("continuous_phasic_auc")
    if not isinstance(provenance, dict):
        raise GuidedContinuousSavedArtifactError(
            "The phasic analysis run report does not declare continuous AUC provenance."
        )
    return provenance


def _auc_settings_for_roi(provenance: dict[str, Any], roi: str) -> dict[str, Any]:
    effective = provenance.get("effective_settings_by_roi")
    if effective is not None:
        if not isinstance(effective, dict) or roi not in effective:
            raise GuidedContinuousSavedArtifactError(
                f"Continuous AUC provenance has no effective settings for ROI {roi!r}."
            )
        settings = effective[roi]
    else:
        settings = provenance.get("global_defaults")
        if settings is None:
            settings = {
                key: provenance.get(key)
                for key in (
                    "event_signal",
                    "auc_baseline_mode",
                    "polarity",
                    "prefilter",
                    "prefilter_parameters",
                    "signal_units",
                    "auc_units",
                )
            }
    if not isinstance(settings, dict):
        raise GuidedContinuousSavedArtifactError(
            f"Continuous AUC provenance for ROI {roi!r} is unreadable."
        )
    units = str(settings.get("auc_units", "")).strip()
    if not units:
        raise GuidedContinuousSavedArtifactError(
            f"Continuous AUC provenance has no units for ROI {roi!r}."
        )
    return dict(settings)


def build_window_plot_data(
    summary: pd.DataFrame,
    *,
    timeline_contract: dict[str, Any],
    value_column: str,
) -> dict[str, np.ndarray]:
    """Build plot arrays without dropping missing scientific values."""
    required = {"window_midpoint_sec", value_column}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise GuidedContinuousSavedArtifactError(
            f"Continuous summary is missing plot columns: {missing}"
        )
    elapsed = pd.to_numeric(summary["window_midpoint_sec"], errors="coerce").to_numpy(
        dtype=float
    )
    values = pd.to_numeric(summary[value_column], errors="coerce").to_numpy(dtype=float)
    valid_x = np.isfinite(elapsed)
    if not np.any(valid_x):
        raise GuidedContinuousSavedArtifactError(
            f"Continuous summary has no finite window coordinates for {value_column}."
        )
    elapsed = elapsed[valid_x]
    values = values[valid_x]
    x_sec = continuous_plot_coordinates(elapsed, timeline_contract)
    order = np.argsort(x_sec, kind="mergesort")
    x_sec = x_sec[order]
    values = values[order]
    if not np.any(np.isfinite(values)):
        raise GuidedContinuousSavedArtifactError(
            f"Continuous summary has no finite values for required plot {value_column}."
        )
    return {
        "elapsed_sec": elapsed[order],
        "x_sec": x_sec,
        "x_hours": x_sec / 3600.0,
        "values": values,
    }


def _write_window_plot(
    *,
    data: dict[str, np.ndarray],
    title: str,
    ylabel: str,
    timeline_contract: dict[str, Any],
    out_path: str,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        data["x_hours"],
        data["values"],
        marker="o",
        linewidth=1.2,
        markersize=3.5,
    )
    ax.set_xlabel(_timeline_axis_label(timeline_contract))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    right = max(1.0, float(np.nanmax(data["x_hours"])))
    ax.set_xlim(left=0.0, right=right)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    dimensions = _validate_image(out_path, title)
    return {
        "n_points_plotted": int(data["values"].size),
        "max_plotted_points": int(data["values"].size),
        "image_dimensions": list(dimensions),
    }


def _read_roi_summary(run_dir: str, roi: str, filename: str) -> pd.DataFrame:
    path = os.path.join(run_dir, roi, "tables", filename)
    _require_file(path, f"{roi} {filename} table")
    try:
        summary = pd.read_csv(path)
    except Exception as exc:
        raise GuidedContinuousSavedArtifactError(
            f"Continuous summary table is not readable: {path} ({exc})"
        ) from exc
    observed = set(summary.get("roi", pd.Series(dtype=str)).astype(str))
    if observed != {str(roi)}:
        raise GuidedContinuousSavedArtifactError(
            f"Continuous summary table has the wrong ROI identity: {path} "
            f"(expected {roi!r}, got {sorted(observed)!r})"
        )
    if filename == TONIC_SUMMARY_FILENAME:
        missing = sorted(_TONIC_SUMMARY_REQUIRED_COLUMNS - set(summary.columns))
        if missing:
            raise GuidedContinuousSavedArtifactError(
                f"Tonic summary table has the wrong schema: {path}; missing {missing}"
            )
        modes = set(summary["acquisition_mode"].astype(str).str.lower())
        if modes != {"continuous"}:
            raise GuidedContinuousSavedArtifactError(
                f"Tonic summary table is not bound to continuous acquisition: {path}"
            )
    return summary.sort_values(["window_index"], kind="mergesort").reset_index(drop=True)


def _artifact_record(
    *,
    run_dir: str,
    roi: str,
    family: str,
    analysis_family: str,
    filename: str,
    **metadata: Any,
) -> dict[str, Any]:
    relative_path = _relative_path(roi, "summary", filename)
    path = os.path.join(run_dir, roi, "summary", filename)
    _validate_image(path, f"{roi} {filename}")
    return {
        "relative_path": relative_path,
        "roi": str(roi),
        "family": str(family),
        "analysis_family": str(analysis_family),
        "artifact_type": "image",
        **metadata,
    }


def _extract_continuous_day_plot_panels(
    *,
    run_dir: str,
    roi: str,
    timeline_contract: dict[str, Any],
) -> dict[str, Any]:
    """Extract fixed display windows from the saved continuous trace caches.

    This is deliberately a display-only adapter.  It creates no analysis
    windows, session records, feature rows, event records, or persisted
    metadata.  Each trace family is read with its own saved timestamp array so
    display alignment never depends on cache chunk indexes being shared.
    """
    from photometry_pipeline.completed_run_review import (
        resolve_persisted_cache_strategy,
    )
    from photometry_pipeline.viz.phasic_data_prep import compute_day_layout

    mode = str(timeline_contract.get("timeline_mode") or "").strip().lower()
    if mode not in {"elapsed", "civil", "fixed_daily_anchor"}:
        raise GuidedContinuousSavedArtifactError(
            f"Continuous Day Plots received an unsupported timeline mode: {mode!r}."
        )

    def _load_series(cache: Any, fields: list[str], family: str) -> dict[str, np.ndarray]:
        records: list[tuple[float, int, dict[str, np.ndarray]]] = []
        for chunk_id in list_cache_chunk_ids(cache):
            cid = int(chunk_id)
            attrs = load_cache_chunk_attrs(cache, roi, cid)
            try:
                _require_continuous_attrs(attrs, roi=roi, chunk_id=cid)
                arrays = load_cache_chunk_fields(cache, roi, cid, fields)
            except Exception as exc:
                raise GuidedContinuousSavedArtifactError(
                    f"Cannot read saved continuous {family} trace for ROI {roi!r}, "
                    f"window {cid}: {exc}"
                ) from exc

            local_time = np.asarray(arrays[0], dtype=float).reshape(-1)
            if local_time.size == 0:
                continue
            if not np.all(np.isfinite(local_time)) or (
                local_time.size > 1 and np.any(np.diff(local_time) <= 0.0)
            ):
                raise GuidedContinuousSavedArtifactError(
                    f"Saved continuous {family} timestamps are not strictly "
                    f"monotonic for ROI {roi!r}, window {cid}."
                )
            trace_arrays: dict[str, np.ndarray] = {}
            for field, value in zip(fields[1:], arrays[1:]):
                trace = np.asarray(value, dtype=float).reshape(-1)
                if trace.shape != local_time.shape:
                    raise GuidedContinuousSavedArtifactError(
                        f"Saved continuous {family} trace shape does not match "
                        f"timestamps for ROI {roi!r}, window {cid}, field {field!r}."
                    )
                trace_arrays[field] = trace
            records.append(
                (
                    float(attrs["window_start_sec"]),
                    cid,
                    {
                        "time_sec": float(attrs["window_start_sec"]) + local_time,
                        **trace_arrays,
                    },
                )
            )

        records.sort(key=lambda item: (item[0], item[1]))
        if not records:
            return {
                "time_sec": np.asarray([], dtype=float),
                **{field: np.asarray([], dtype=float) for field in fields[1:]},
            }
        names = ["time_sec", *fields[1:]]
        merged = {
            name: np.concatenate([record[2][name] for record in records])
            for name in names
        }
        merged_time = merged["time_sec"]
        if merged_time.size > 1 and np.any(np.diff(merged_time) <= 0.0):
            raise GuidedContinuousSavedArtifactError(
                f"Saved continuous {family} timestamps are not globally monotonic "
                f"for ROI {roi!r}."
            )
        return merged

    corrected_cache_path = os.path.join(
        run_dir, "continuous_corrected_trace_cache.h5"
    )
    phasic_cache_path = os.path.join(
        run_dir, "_analysis", "phasic_out", "phasic_trace_cache.h5"
    )
    _require_file(corrected_cache_path, f"{roi} corrected trace cache")
    _require_file(phasic_cache_path, f"{roi} phasic trace cache")

    with open_phasic_cache(corrected_cache_path) as corrected_cache:
        corrected_chunk_ids = list_cache_chunk_ids(corrected_cache)
        if not corrected_chunk_ids:
            raise GuidedContinuousSavedArtifactError(
                f"The corrected cache contains no continuous windows for ROI {roi!r}."
            )
        try:
            correction = resolve_persisted_cache_strategy(
                corrected_cache,
                roi,
                corrected_chunk_ids,
                strict_current=False,
            )
        except Exception as exc:
            raise GuidedContinuousSavedArtifactError(
                f"Cannot resolve the persisted correction reference for ROI {roi!r}: {exc}"
            ) from exc
        reference_field = str(correction.get("field") or "").strip()
        if reference_field not in {"fit_ref", "signal_only_f0_baseline"}:
            raise GuidedContinuousSavedArtifactError(
                f"The persisted correction reference for ROI {roi!r} is unsupported."
            )
        corrected = _load_series(
            corrected_cache,
            ["time_sec", "sig_raw", "uv_raw", reference_field],
            "corrected",
        )

    with open_phasic_cache(phasic_cache_path) as phasic_cache:
        phasic = _load_series(phasic_cache, ["time_sec", "dff"], "phasic dF/F")

    time_arrays = [
        values["time_sec"]
        for values in (corrected, phasic)
        if values["time_sec"].size
    ]
    if not time_arrays:
        raise GuidedContinuousSavedArtifactError(
            f"Saved continuous caches contain no timestamps for ROI {roi!r}."
        )
    recording_end_sec = max(float(values[-1]) for values in time_arrays)

    candidates: list[dict[str, Any]] = []
    if mode == "elapsed":
        candidate_start = 0.0
        display_base = datetime(2000, 1, 1)
        while candidate_start <= recording_end_sec + 1e-9:
            candidates.append(
                {
                    "candidate_id": len(candidates),
                    "start_sec": float(candidate_start),
                    "end_sec": float(candidate_start + _CONTINUOUS_DAY_PLOT_SAMPLE_SEC),
                    "display_dt": display_base + timedelta(seconds=candidate_start),
                }
            )
            candidate_start += _CONTINUOUS_DAY_PLOT_SLOT_SEC
    else:
        recording_start_sec, _ = parse_guided_clock(
            timeline_contract.get("recording_start_clock"),
            field_name="Clock time at recording start",
        )
        recording_start_dt = datetime(2000, 1, 1) + timedelta(
            seconds=recording_start_sec
        )
        candidate_dt = recording_start_dt.replace(
            minute=0, second=0, microsecond=0
        )
        while (candidate_dt - recording_start_dt).total_seconds() <= recording_end_sec + 1e-9:
            start_sec = float((candidate_dt - recording_start_dt).total_seconds())
            candidates.append(
                {
                    "candidate_id": len(candidates),
                    "start_sec": start_sec,
                    "end_sec": start_sec + _CONTINUOUS_DAY_PLOT_SAMPLE_SEC,
                    "display_dt": candidate_dt,
                }
            )
            candidate_dt += timedelta(seconds=_CONTINUOUS_DAY_PLOT_SLOT_SEC)

    if not candidates:
        raise GuidedContinuousSavedArtifactError(
            f"Saved continuous caches contain no plottable display hours for ROI {roi!r}."
        )

    layout = compute_day_layout(
        [
            (
                int(candidate["candidate_id"]),
                "display://"
                + candidate["display_dt"].strftime("%Y_%m_%d-%H_%M_%S")
                + f"/slot_{int(candidate['candidate_id'])}.csv",
            )
            for candidate in candidates
        ],
        None,
        roi,
        sessions_per_hour=2,
        timeline_anchor_mode=mode,
        fixed_daily_anchor_clock=(
            timeline_contract.get("fixed_daily_anchor_clock")
            if mode == "fixed_daily_anchor"
            else None
        ),
    )
    candidate_by_id = {
        int(candidate["candidate_id"]): candidate for candidate in candidates
    }

    def _extract_family(
        values: dict[str, np.ndarray], fields: tuple[str, ...]
    ) -> dict[int, list[dict[str, Any]]]:
        timestamps = values["time_sec"]
        if timestamps.size > 1:
            sample_period = float(np.median(np.diff(timestamps)))
        else:
            sample_period = float("nan")
        if not np.isfinite(sample_period) or sample_period <= 0.0:
            return {}
        coverage_tolerance = max(sample_period * 1.5, 1e-6)
        gap_limit = sample_period * 1.5
        by_day: dict[int, list[dict[str, Any]]] = {}
        for chunk in layout.chunks:
            candidate = candidate_by_id[int(chunk.chunk_id)]
            start = float(candidate["start_sec"])
            end = float(candidate["end_sec"])
            left = int(np.searchsorted(timestamps, start, side="left"))
            right = int(np.searchsorted(timestamps, end, side="left"))
            selected_time = timestamps[left:right]
            if (
                selected_time.size < 2
                or selected_time[0] > start + coverage_tolerance
                or selected_time[-1] < end - coverage_tolerance
                or (
                    selected_time.size > 1
                    and np.any(np.diff(selected_time) > gap_limit)
                )
            ):
                continue
            panel = {
                "day": int(chunk.day_idx),
                "hour": int(chunk.hour_idx),
                "col": int(chunk.hour_rank),
                "chunk_id": int(candidate["candidate_id"]),
                "is_missing": False,
                "display_label": _CONTINUOUS_DAY_PLOT_COLUMN_LABELS[
                    int(chunk.hour_rank)
                ],
                "t": selected_time - start,
                # This display window's own recording-global bounds. Renderers
                # ignore these keys; they exist so a display-only consumer can
                # place already-saved recording-global coordinates (such as the
                # saved continuous event times) onto this panel without
                # recomputing the candidate grid.
                "panel_start_sec": start,
                "panel_end_sec": end,
                "xlim_600": bool(float(selected_time[-1] - start) > 550.0),
            }
            for field in fields:
                panel[field] = values[field][left:right]
            by_day.setdefault(int(chunk.day_idx), []).append(panel)
        return by_day

    corrected_signal = _extract_family(corrected, ("sig_raw", "uv_raw"))
    corrected_reference = _extract_family(corrected, ("sig_raw", reference_field))
    phasic_dff = _extract_family(phasic, ("dff",))

    return {
        "layout": layout,
        "signal_reference": corrected_signal,
        "correction_reference": corrected_reference,
        "phasic_dff": phasic_dff,
        "correction_strategy_family": str(correction.get("strategy_family") or ""),
        "correction_strategy_label": str(correction.get("label") or ""),
        "reference_field": reference_field,
        "correction_reference_label": (
            "Correction Reference (signal-only F0 baseline)"
            if str(correction.get("strategy_family") or "") == "signal_only_f0"
            else "Correction Reference (fitted)"
        ),
        "timeline_anchor_label": _timeline_anchor_label_for_day_plot(
            mode, timeline_contract.get("fixed_daily_anchor_clock")
        ),
    }


def _continuous_dff_display_limits(
    dff_panels: dict[int, list[dict[str, Any]]]
) -> tuple[float, float]:
    """Resolve the shared dF/F day-plot y-limits for one ROI.

    Single source of truth for the continuous dF/F display scale so the
    canonical marker-free plot and any display-only copy of it are drawn on
    exactly the same axis.
    """
    finite_dff = [
        panel["dff"][np.isfinite(panel["dff"])]
        for panels in dff_panels.values()
        for panel in panels
        if np.any(np.isfinite(panel["dff"]))
    ]
    if not finite_dff:
        return -1.0, 1.0
    values = np.concatenate(finite_dff)
    global_ymin, global_ymax = np.percentile(values, [0.5, 99.9])
    pad = 0.10 * (global_ymax - global_ymin)
    if not np.isfinite(pad) or pad == 0.0:
        pad = 0.1
    return float(global_ymin - pad), float(global_ymax + pad)


def map_continuous_event_times_to_panel_indices(
    panel: dict[str, Any], event_times_sec: Any
) -> np.ndarray:
    """Map saved recording-global event seconds onto one display panel.

    The panel's own recording-global axis is ``panel_start_sec + panel["t"]``,
    which is the same ``window_start_sec + local_time`` axis the saved
    continuous event table was written against, so no clock conversion is
    involved.

    Events are admitted on the half-open interval
    ``[panel_start_sec, panel_end_sec)``: an event exactly at the panel end
    belongs to the next display window and is excluded here rather than being
    drawn twice. Admitted events are resolved to the nearest sample on the
    panel's own axis, so every returned index is in range and the renderer is
    never relied on to silently discard an out-of-range value.

    This is arithmetic over already-saved coordinates. It never runs the
    detector and never opens a trace cache.
    """
    panel_t = np.asarray(panel.get("t"), dtype=float).reshape(-1)
    times = np.asarray(event_times_sec, dtype=float).reshape(-1)
    if panel_t.size == 0 or times.size == 0:
        return np.asarray([], dtype=int)

    start = float(panel["panel_start_sec"])
    end = float(panel["panel_end_sec"])
    finite = times[np.isfinite(times)]
    admitted = finite[(finite >= start) & (finite < end)]
    if admitted.size == 0:
        return np.asarray([], dtype=int)
    if panel_t.size == 1:
        return np.asarray([0], dtype=int)

    global_axis = start + panel_t
    right = np.clip(np.searchsorted(global_axis, admitted, side="left"), 1, panel_t.size - 1)
    left = right - 1
    nearest = np.where(
        np.abs(admitted - global_axis[left]) <= np.abs(global_axis[right] - admitted),
        left,
        right,
    )
    return np.unique(nearest).astype(int)


def load_continuous_marker_event_times(run_dir: str, roi: str) -> np.ndarray:
    """Recording-global event seconds for one ROI from the saved event table.

    Reads ``_analysis/phasic_out/features/continuous_phasic_events.csv``
    through the existing completed-run event loader, which is the sole
    authority for a continuous run's events and never reruns detection.
    """
    from photometry_pipeline.completed_continuous_rwd_review import (
        load_continuous_phasic_events,
    )

    events = load_continuous_phasic_events(run_dir, roi_id=str(roi))
    if events.empty:
        return np.asarray([], dtype=float)
    return events["global_time_sec"].to_numpy(dtype=float)


def build_continuous_marker_on_dff_dayplots(
    run_dir: str,
    *,
    roi: str,
    timeline_contract: dict[str, Any],
    event_times_sec: Any,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Write display-only dF/F day-plot copies that show detected peaks.

    Reuses the existing continuous panel extractor and the existing dF/F tile
    renderer; it neither reruns correction or detection nor publishes a
    manifest artifact. Output goes to an isolated variant directory and the
    canonical marker-free day plots are never touched.
    """
    from tools.plot_phasic_dayplot_bundle import (
        _compose_dff_day_tile_canvas_lightweight,
        _dff_tile_layout,
        build_day_slot_maps,
    )

    canonical_dir = os.path.realpath(os.path.join(run_dir, str(roi), "day_plots"))
    target_dir = os.path.realpath(
        output_dir
        if output_dir
        else os.path.join(run_dir, str(roi), CONTINUOUS_MARKER_ON_VARIANT_RELATIVE_DIR)
    )
    if target_dir == canonical_dir:
        raise GuidedContinuousSavedArtifactError(
            "Marker-on day-plot copies cannot be written into the canonical "
            "day_plots folder."
        )

    extracted = _extract_continuous_day_plot_panels(
        run_dir=run_dir,
        roi=str(roi),
        timeline_contract=timeline_contract,
    )
    dff_panels = extracted["phasic_dff"]
    if not any(dff_panels.values()):
        raise GuidedContinuousSavedArtifactError(
            f"The saved continuous results contain no displayable dF/F day-plot "
            f"windows for ROI {str(roi)!r}."
        )

    global_ymin, global_ymax = _continuous_dff_display_limits(dff_panels)
    marked_panels: dict[int, list[dict[str, Any]]] = {}
    marker_counts_by_day: dict[int, int] = {}
    for day, panels in dff_panels.items():
        marked = []
        for panel in panels:
            indices = map_continuous_event_times_to_panel_indices(
                panel, event_times_sec
            )
            marker_counts_by_day[int(day)] = marker_counts_by_day.get(
                int(day), 0
            ) + int(indices.size)
            marked.append({**panel, "peak_indices": indices})
        marked_panels[int(day)] = marked

    sph = 2
    layout = _dff_tile_layout(sph, _CONTINUOUS_DAY_PLOT_DPI)
    slot_maps = build_day_slot_maps(marked_panels, sph)
    os.makedirs(target_dir, exist_ok=True)

    written: list[str] = []
    for day in sorted(extracted["layout"].chunks_by_day):
        slot_map = slot_maps.get(int(day), {})
        if not slot_map:
            continue
        image, _stats = _compose_dff_day_tile_canvas_lightweight(
            day=day,
            plot_roi=str(roi),
            sph=sph,
            slot_map=slot_map,
            layout=layout,
            global_ymin=global_ymin,
            global_ymax=global_ymax,
            show_peak_markers=True,
            timeline_anchor_label=extracted["timeline_anchor_label"],
            title_override=(
                f"Sampled Phasic dF/F with detected peaks - Day {day} - ROI {roi}"
            ),
            column_labels=_CONTINUOUS_DAY_PLOT_COLUMN_LABELS,
        )
        path = os.path.join(target_dir, f"phasic_dFF_day_{int(day):03d}.png")
        image.save(path, compress_level=1)
        written.append(path)

    if not written:
        raise GuidedContinuousSavedArtifactError(
            f"No displayable dF/F day plots could be produced for ROI {str(roi)!r}."
        )

    return {
        "roi": str(roi),
        "output_dir": target_dir,
        "paths": written,
        "marker_counts_by_day": marker_counts_by_day,
        "total_markers": int(sum(marker_counts_by_day.values())),
    }


def _timeline_anchor_label_for_day_plot(mode: str, fixed_clock: Any) -> str:
    mode = str(mode or "").strip().lower()
    if mode == "elapsed":
        return "elapsed-from-recording-start"
    if mode == "fixed_daily_anchor":
        clock = str(fixed_clock or "unset")
        if clock.count(":") == 1:
            clock += ":00"
        return f"fixed-daily-anchor@{clock}"
    return "civil-clock"


def _day_plot_artifact_record(
    *,
    run_dir: str,
    roi: str,
    family: str,
    label: str,
    filename: str,
    day: int,
    timeline_contract: dict[str, Any],
    **metadata: Any,
) -> dict[str, Any]:
    relative_path = _relative_path(roi, "day_plots", filename)
    path = os.path.join(run_dir, roi, "day_plots", filename)
    dimensions = _validate_image(path, f"{roi} {filename}")
    return {
        "relative_path": relative_path,
        "roi": str(roi),
        "family": str(family),
        "analysis_family": "phasic",
        "artifact_type": "image",
        "label": str(label),
        "day_index": int(day),
        "timeline_mode": str(timeline_contract.get("timeline_mode") or ""),
        "image_dimensions": list(dimensions),
        **metadata,
    }


def _publish_continuous_day_plots(
    *,
    run_dir: str,
    roi: str,
    timeline_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Render one manifest-backed sampled Day Plot package for one ROI."""
    from tools.plot_phasic_dayplot_bundle import (
        _build_stacked_slot_traces,
        _compose_dff_day_tile_canvas_lightweight,
        _compose_dynamic_fit_day_tile_canvas,
        _compose_sig_iso_day_tile_canvas,
        _dynamic_fit_panel_ranges_with_day_min_span,
        _dynamic_fit_tile_layout,
        _prepare_sig_iso_centered_panel,
        _render_stacked_day_canvas_lightweight,
        _sig_iso_panel_ranges_with_day_min_span,
        _sig_iso_tile_layout,
        _dff_tile_layout,
        build_day_slot_maps,
        uniform_filter1d,
    )

    extracted = _extract_continuous_day_plot_panels(
        run_dir=run_dir,
        roi=roi,
        timeline_contract=timeline_contract,
    )
    layout = extracted["layout"]
    sph = 2
    days = sorted(layout.chunks_by_day)
    anchor_label = extracted["timeline_anchor_label"]
    column_labels = _CONTINUOUS_DAY_PLOT_COLUMN_LABELS
    output_dir = os.path.join(run_dir, roi, "day_plots")
    os.makedirs(output_dir, exist_ok=True)

    signal_panels: dict[int, list[dict[str, Any]]] = {}
    for day, panels in extracted["signal_reference"].items():
        signal_panels[day] = []
        for panel in panels:
            sig, uv = _prepare_sig_iso_centered_panel(
                panel["sig_raw"], panel["uv_raw"]
            )
            signal_panels[day].append({**panel, "sig": sig, "uv": uv})

    reference_panels: dict[int, list[dict[str, Any]]] = {}
    correction_label = extracted["correction_reference_label"]
    for day, panels in extracted["correction_reference"].items():
        reference_panels[day] = [
            {
                **panel,
                "sig_fit": panel["sig_raw"],
                "fit_ref": panel[extracted["reference_field"]],
                "reference_label": extracted["correction_strategy_label"],
            }
            for panel in panels
        ]

    dff_panels = extracted["phasic_dff"]
    global_ymin, global_ymax = _continuous_dff_display_limits(dff_panels)

    smoothed_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for panels in dff_panels.values():
        for panel in panels:
            mask = np.isfinite(panel["dff"])
            y = panel["dff"][mask]
            t = panel["t"][mask]
            if y.size < 2:
                continue
            dt = float(np.median(np.diff(t)))
            fs = 1.0 / dt if np.isfinite(dt) and dt > 0.0 else 1.0
            width = max(1, int(round(fs * _CONTINUOUS_DAY_PLOT_SMOOTH_SEC)))
            smoothed_data[panel["chunk_id"]] = (
                t,
                uniform_filter1d(y, size=width),
            )

    families = {
        "sampled_signal_reference": signal_panels,
        "sampled_correction_reference": reference_panels,
        "sampled_phasic_dff": dff_panels,
    }
    artifacts: list[dict[str, Any]] = []
    for family, label, filename_template in _CONTINUOUS_DAY_PLOT_FAMILIES:
        if family == "sampled_stacked":
            family_panels = dff_panels
        else:
            family_panels = families[family]
        for day in days:
            slot_map = build_day_slot_maps(family_panels, sph).get(day, {})
            filename = filename_template.format(day=int(day))
            path = os.path.join(output_dir, filename)
            if family == "sampled_signal_reference":
                image = _compose_sig_iso_day_tile_canvas(
                    day=day,
                    plot_roi=roi,
                    sph=sph,
                    slot_map=slot_map,
                    layout=_sig_iso_tile_layout(sph, _CONTINUOUS_DAY_PLOT_DPI),
                    panel_y_ranges=_sig_iso_panel_ranges_with_day_min_span(slot_map),
                    timeline_anchor_label=anchor_label,
                    title_override=f"Day {day} Sampled Signal + Reference - {roi}",
                    column_labels=column_labels,
                )
            elif family == "sampled_correction_reference":
                image = _compose_dynamic_fit_day_tile_canvas(
                    day=day,
                    plot_roi=roi,
                    sph=sph,
                    slot_map=slot_map,
                    layout=_dynamic_fit_tile_layout(sph, _CONTINUOUS_DAY_PLOT_DPI),
                    panel_y_ranges=_dynamic_fit_panel_ranges_with_day_min_span(slot_map),
                    timeline_anchor_label=anchor_label,
                    title_override=f"Day {day} {correction_label} - {roi}",
                    column_labels=column_labels,
                )
            elif family == "sampled_phasic_dff":
                image, _stats = _compose_dff_day_tile_canvas_lightweight(
                    day=day,
                    plot_roi=roi,
                    sph=sph,
                    slot_map=slot_map,
                    layout=_dff_tile_layout(sph, _CONTINUOUS_DAY_PLOT_DPI),
                    global_ymin=global_ymin,
                    global_ymax=global_ymax,
                    show_peak_markers=False,
                    timeline_anchor_label=anchor_label,
                    title_override=f"Sampled Phasic dF/F - Day {day} - ROI {roi}",
                    column_labels=column_labels,
                )
            else:
                dff_slot_map = build_day_slot_maps(dff_panels, sph).get(day, {})
                slot_traces = _build_stacked_slot_traces(
                    dff_slot_map, smoothed_data, sph
                )
                image = _render_stacked_day_canvas_lightweight(
                    day=day,
                    plot_roi=roi,
                    slot_traces=slot_traces,
                    smooth_window_s=_CONTINUOUS_DAY_PLOT_SMOOTH_SEC,
                    dpi=_CONTINUOUS_DAY_PLOT_DPI,
                    timeline_anchor_label=anchor_label,
                    slot_map=dff_slot_map,
                    sph=sph,
                    title_override=(
                        f"Sampled Stacked Phasic dF/F - Day {day} - ROI {roi}"
                    ),
                    column_labels=column_labels,
                )
            image.save(path, compress_level=1)
            artifacts.append(
                _day_plot_artifact_record(
                    run_dir=run_dir,
                    roi=roi,
                    family=family,
                    label=label,
                    filename=filename,
                    day=day,
                    timeline_contract=timeline_contract,
                    display_explanation=CONTINUOUS_DAY_PLOT_EXPLANATION,
                    sampled_column_labels=list(column_labels),
                    sampled_interval_sec=_CONTINUOUS_DAY_PLOT_SAMPLE_SEC,
                    correction_strategy_family=(
                        extracted["correction_strategy_family"]
                        if family == "sampled_correction_reference"
                        else None
                    ),
                    correction_strategy_label=(
                        extracted["correction_strategy_label"]
                        if family == "sampled_correction_reference"
                        else None
                    ),
                    correction_reference_label=(
                        correction_label
                        if family == "sampled_correction_reference"
                        else None
                    ),
                    stacked_smoothing_sec=(
                        _CONTINUOUS_DAY_PLOT_SMOOTH_SEC
                        if family == "sampled_stacked"
                        else None
                    ),
                    order=80
                    + (next(
                        index
                        for index, family_spec in enumerate(_CONTINUOUS_DAY_PLOT_FAMILIES)
                        if family_spec[0] == family
                    ) * 1000)
                    + int(day),
                )
            )
    return artifacts
def _select_correction_window(cache: Any, roi: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for chunk_id in list_cache_chunk_ids(cache):
        cid = int(chunk_id)
        attrs = load_cache_chunk_attrs(cache, roi, cid)
        _require_continuous_attrs(attrs, roi=roi, chunk_id=cid)
        start = float(attrs["window_start_sec"])
        end = float(attrs["window_end_sec"])
        duration = float(attrs["window_duration_sec"])
        records.append(
            {
                "chunk_id": cid,
                "window_index": int(attrs["window_index"]),
                "window_start_sec": start,
                "window_end_sec": end,
                "window_duration_sec": duration,
                "is_partial_final_window": bool(
                    attrs.get("is_partial_final_window", False)
                ),
            }
        )
    if not records:
        raise GuidedContinuousSavedArtifactError(
            f"The phasic cache contains no continuous windows for ROI {roi!r}."
        )
    center = (
        min(record["window_start_sec"] for record in records)
        + max(record["window_end_sec"] for record in records)
    ) / 2.0
    complete = [
        record for record in records if not record["is_partial_final_window"]
    ]
    if complete:
        chosen = min(
            complete,
            key=lambda record: (
                abs(
                    (record["window_start_sec"] + record["window_end_sec"]) / 2.0
                    - center
                ),
                record["window_index"],
                record["chunk_id"],
            ),
        )
        rule = "complete_window_nearest_temporal_center_tie_lowest_window_index"
    else:
        chosen = min(
            records,
            key=lambda record: (
                -record["window_duration_sec"],
                record["window_index"],
                record["chunk_id"],
            ),
        )
        rule = "fallback_longest_window_tie_lowest_window_index"
    selected = dict(chosen)
    selected["selection_rule"] = rule
    selected["recording_temporal_center_sec"] = float(center)
    return selected


def _publish_correction_impact(
    *,
    run_dir: str,
    roi: str,
    timeline_contract: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    from photometry_pipeline.completed_run_review import (
        resolve_persisted_cache_strategy,
    )
    from tools.plot_phasic_correction_impact import (
        _dynamic_fit_mode_label,
        _reconstruct_bleach_series_from_attrs,
        _resolve_dynamic_fit_settings,
        build_correction_impact_figure,
    )

    phasic_out_dir = os.path.join(run_dir, "_analysis", "phasic_out")
    window_cache_path = os.path.join(phasic_out_dir, "phasic_trace_cache.h5")
    correction_cache_path = os.path.join(run_dir, "continuous_corrected_trace_cache.h5")
    _require_file(window_cache_path, f"{roi} phasic trace cache")
    _require_file(correction_cache_path, f"{roi} corrected trace cache")
    output_path = os.path.join(run_dir, roi, "summary", PHASIC_CORRECTION_IMPACT_FILENAME)
    cache_passes = 2
    with open_phasic_cache(window_cache_path) as window_cache:
        selection = _select_correction_window(window_cache, roi)
    with open_phasic_cache(correction_cache_path) as cache:
        chunk_ids = list_cache_chunk_ids(cache)
        consumed_families = set()
        for chunk_id in chunk_ids:
            chunk_attrs = load_cache_chunk_attrs(cache, roi, int(chunk_id))
            family_value = str(
                chunk_attrs.get("correction_strategy_family", "")
            ).strip()
            if not family_value:
                raise GuidedContinuousSavedArtifactError(
                    f"The accepted corrected cache has no per-ROI correction strategy "
                    f"for ROI {roi!r}, window {int(chunk_id)}."
                )
            consumed_families.add(family_value)
        if len(consumed_families) != 1:
            raise GuidedContinuousSavedArtifactError(
                f"The accepted corrected cache changes correction strategy across "
                f"windows for ROI {roi!r}: {sorted(consumed_families)!r}."
            )
        strategy = resolve_persisted_cache_strategy(
            cache,
            roi,
            chunk_ids,
            strict_current=False,
        )
        if str(strategy.get("strategy_family", "")).strip() != next(
            iter(consumed_families)
        ):
            raise GuidedContinuousSavedArtifactError(
                f"Persisted correction strategy resolution disagrees with the "
                f"accepted corrected cache for ROI {roi!r}."
            )
        reference_field = str(strategy.get("field", "")).strip()
        if not reference_field:
            raise GuidedContinuousSavedArtifactError(
                f"The phasic cache has no persisted correction reference for ROI {roi!r}."
            )
        fields = ["time_sec", "sig_raw", "uv_raw", reference_field, "dff"]
        try:
            t, sig, iso, reference, dff = load_cache_chunk_fields(
                cache, roi, int(selection["chunk_id"]), fields
            )
            attrs = load_cache_chunk_attrs(cache, roi, int(selection["chunk_id"]))
        except Exception as exc:
            raise GuidedContinuousSavedArtifactError(
                f"Cannot read the selected persisted correction window for ROI {roi!r}: {exc}"
            ) from exc

    arrays = [np.asarray(value, dtype=float).reshape(-1) for value in (t, sig, iso, reference, dff)]
    if not arrays[0].size or any(array.shape != arrays[0].shape for array in arrays):
        raise GuidedContinuousSavedArtifactError(
            f"Selected correction-impact traces for ROI {roi!r} are empty or have mismatched lengths."
        )
    if not np.any(np.isfinite(arrays[0])):
        raise GuidedContinuousSavedArtifactError(
            f"Selected correction-impact traces for ROI {roi!r} contain no finite time values."
        )
    t = arrays[0] - arrays[0][0]
    dynamic_mode, baseline_subtract, bleach_mode = _resolve_dynamic_fit_settings(
        phasic_out_dir
    )
    family = str(strategy.get("strategy_family", "")).strip()
    if family == "signal_only_f0":
        strategy_label = "Signal-Only F0"
    elif family == "dynamic_fit":
        dynamic_mode = str(
            attrs.get("correction_dynamic_fit_mode")
            or attrs.get("dynamic_fit_mode_resolved")
            or dynamic_mode
        )
        strategy_label = _dynamic_fit_mode_label(dynamic_mode)
    else:
        raise GuidedContinuousSavedArtifactError(
            f"Unsupported persisted correction strategy for ROI {roi!r}: {family!r}"
        )

    signal_only_qc = {
        key: value
        for key, value in attrs.items()
        if str(key).startswith("signal_only_f0_production_")
        or str(key)
        in {
            "signal_only_f0_warning",
            "signal_only_f0_note",
            "signal_only_f0_candidate_viability",
            "signal_only_f0_candidate_support",
            "signal_only_f0_support",
            "signal_only_f0_candidate_confidence",
            "signal_only_f0_confidence",
        }
    }
    sig_bleach_fit = sig_bleach_corrected = None
    iso_bleach_fit = iso_bleach_corrected = None
    if str(bleach_mode).strip().lower() != "none":
        sig_bleach_fit, sig_bleach_corrected = _reconstruct_bleach_series_from_attrs(
            arrays[1], t, attrs, prefix="bleach_signal"
        )
        iso_bleach_fit, iso_bleach_corrected = _reconstruct_bleach_series_from_attrs(
            arrays[2], t, attrs, prefix="bleach_iso"
        )

    import matplotlib.pyplot as plt

    fig, _axes = build_correction_impact_figure(
        t=t,
        sig=arrays[1],
        iso=arrays[2],
        fit=arrays[3],
        dff=arrays[4],
        roi=roi,
        chunk_id=int(selection["window_index"]),
        dynamic_fit_mode=dynamic_mode,
        baseline_subtract_before_fit=baseline_subtract,
        bleach_correction_mode=bleach_mode,
        sig_bleach_fit=sig_bleach_fit,
        sig_bleach_corrected=sig_bleach_corrected,
        iso_bleach_fit=iso_bleach_fit,
        iso_bleach_corrected=iso_bleach_corrected,
        strategy_family=family,
        correction_reference_label="Correction Reference",
        strategy_label=strategy_label,
        signal_only_qc=signal_only_qc,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    dimensions = _validate_image(output_path, f"{roi} correction-impact figure")
    selection_public = {
        "roi": str(roi),
        "window_index": int(selection["window_index"]),
        "elapsed_start_sec": float(selection["window_start_sec"]),
        "elapsed_end_sec": float(selection["window_end_sec"]),
        "rule": str(selection["selection_rule"]),
    }
    artifact = _artifact_record(
        run_dir=run_dir,
        roi=roi,
        family="phasic_correction_impact",
        analysis_family="phasic",
        filename=PHASIC_CORRECTION_IMPACT_FILENAME,
        correction_strategy_family=family,
        correction_strategy_label=strategy_label,
        representative_window=selection_public,
        image_dimensions=list(dimensions),
    )
    return artifact, cache_passes


def _sample_tonic_trace(
    cache: Any,
    roi: str,
    *,
    max_points: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    trace_names = ("raw_signal", "raw_reference", "tonic_signal")
    chunk_records: list[dict[str, Any]] = []
    samples_seen = 0
    for chunk_id in list_cache_chunk_ids(cache):
        cid = int(chunk_id)
        attrs = load_cache_chunk_attrs(cache, roi, cid)
        _require_continuous_attrs(attrs, roi=roi, chunk_id=cid)
        local_t, raw_signal, raw_reference, tonic_signal = load_cache_chunk_fields(
            cache, roi, cid, ["time_sec", "sig_raw", "uv_raw", "deltaF"]
        )
        local_t = np.asarray(local_t, dtype=float).reshape(-1)
        raw_signal = np.asarray(raw_signal, dtype=float).reshape(-1)
        raw_reference = np.asarray(raw_reference, dtype=float).reshape(-1)
        tonic_signal = np.asarray(tonic_signal, dtype=float).reshape(-1)
        if any(
            trace.shape != local_t.shape
            for trace in (raw_signal, raw_reference, tonic_signal)
        ):
            raise GuidedContinuousSavedArtifactError(
                f"Tonic trace shape mismatch for ROI {roi!r} chunk {cid}."
            )
        finite = np.isfinite(local_t) & np.isfinite(tonic_signal)
        finite_idx = np.flatnonzero(finite)
        start = float(attrs["window_start_sec"])
        elapsed = start + local_t[finite_idx]
        chunk_records.append(
            {
                "chunk_id": cid,
                "window_index": int(attrs["window_index"]),
                "window_start_sec": start,
                "window_end_sec": float(attrs["window_end_sec"]),
                "n_samples": int(tonic_signal.size),
                "n_finite": int(finite_idx.size),
                "elapsed_start": float(elapsed[0]) if elapsed.size else float("nan"),
                "elapsed_end": float(elapsed[-1]) if elapsed.size else float("nan"),
                "fs_hz": float(attrs.get("fs_hz", float("nan"))),
            }
        )
        samples_seen += int(tonic_signal.size)
    chunk_records.sort(
        key=lambda record: (
            float(record["window_start_sec"]),
            int(record["window_index"]),
            int(record["chunk_id"]),
        )
    )
    if not any(int(record["n_finite"]) for record in chunk_records):
        raise GuidedContinuousSavedArtifactError(
            f"Tonic cache contains no finite overview samples for ROI {roi!r}."
        )
    allocations = _allocate_trace_points(chunk_records, max_points=max_points)
    finite_records = [record for record in chunk_records if record["n_finite"] > 0]
    first_finite = int(finite_records[0]["chunk_id"])
    last_finite = int(finite_records[-1]["chunk_id"])
    x_parts: list[np.ndarray] = []
    trace_parts: dict[str, list[np.ndarray]] = {name: [] for name in trace_names}
    previous_record: dict[str, Any] | None = None
    for record in chunk_records:
        n_select = int(allocations.get(int(record["chunk_id"]), 0))
        if n_select <= 0:
            # Keep the last chunk that actually contains finite samples as the
            # gap boundary.  An empty/missing chunk must not erase that
            # boundary or the next finite chunk would be joined across it.
            if int(record["n_finite"]) > 0:
                previous_record = record
            continue
        local_t, raw_signal, raw_reference, tonic_signal = load_cache_chunk_fields(
            cache,
            roi,
            int(record["chunk_id"]),
            ["time_sec", "sig_raw", "uv_raw", "deltaF"],
        )
        local_t = np.asarray(local_t, dtype=float).reshape(-1)
        traces = {
            "raw_signal": np.asarray(raw_signal, dtype=float).reshape(-1),
            "raw_reference": np.asarray(raw_reference, dtype=float).reshape(-1),
            "tonic_signal": np.asarray(tonic_signal, dtype=float).reshape(-1),
        }
        if any(trace.shape != local_t.shape for trace in traces.values()):
            raise GuidedContinuousSavedArtifactError(
                f"Tonic trace shape mismatch for ROI {roi!r} chunk {int(record['chunk_id'])}."
            )
        finite = np.isfinite(local_t) & np.isfinite(traces["tonic_signal"])
        finite_idx = np.flatnonzero(finite)
        selected_positions = _select_finite_positions_for_chunk(
            finite_idx.size,
            n_select,
            force_first=int(record["chunk_id"]) == first_finite,
            force_last=int(record["chunk_id"]) == last_finite,
        )
        selected = finite_idx[selected_positions]
        x_selected = float(record["window_start_sec"]) + local_t[selected]
        selected_traces = {
            name: trace[selected] for name, trace in traces.items()
        }
        if selected.size > 1:
            x_with_gaps = [x_selected[0]]
            trace_with_gaps = {
                name: [selected_trace[0]]
                for name, selected_trace in selected_traces.items()
            }
            for selected_idx, (previous_idx, current_idx, current_x) in enumerate(
                zip(selected[:-1], selected[1:], x_selected[1:])
            ):
                if np.any(~finite[int(previous_idx) + 1 : int(current_idx)]):
                    x_with_gaps.append(float(current_x))
                    for values in trace_with_gaps.values():
                        values.append(float("nan"))
                x_with_gaps.append(float(current_x))
                for name, selected_trace in selected_traces.items():
                    trace_with_gaps[name].append(selected_trace[selected_idx + 1])
            x_selected = np.asarray(x_with_gaps, dtype=float)
            selected_traces = {
                name: np.asarray(values, dtype=float)
                for name, values in trace_with_gaps.items()
            }
        if previous_record is not None:
            previous_end = float(previous_record["elapsed_end"])
            current_start = float(record["elapsed_start"])
            previous_dt = (
                1.0 / float(previous_record["fs_hz"])
                if np.isfinite(previous_record["fs_hz"]) and previous_record["fs_hz"] > 0
                else 0.0
            )
            current_dt = (
                1.0 / float(record["fs_hz"])
                if np.isfinite(record["fs_hz"]) and record["fs_hz"] > 0
                else 0.0
            )
            threshold = max(previous_dt, current_dt, 1e-9) * 1.5
            if np.isfinite(previous_end) and np.isfinite(current_start) and current_start - previous_end > threshold:
                x_parts.append(np.asarray([x_selected[0]], dtype=float))
                for name in trace_names:
                    trace_parts[name].append(np.asarray([np.nan], dtype=float))
        x_parts.append(x_selected)
        for name in trace_names:
            trace_parts[name].append(selected_traces[name])
        previous_record = record

    if not x_parts:
        raise GuidedContinuousSavedArtifactError(
            f"Tonic cache produced no bounded overview samples for ROI {roi!r}."
        )
    elapsed = np.concatenate(x_parts)
    traces = {
        name: np.concatenate(trace_parts[name]) for name in trace_names
    }
    order = np.argsort(elapsed, kind="mergesort")
    elapsed = elapsed[order]
    traces = {name: trace[order] for name, trace in traces.items()}
    if elapsed.size > max_points:
        marker_indices = np.flatnonzero(~np.isfinite(traces["tonic_signal"]))
        finite_indices = np.flatnonzero(np.isfinite(traces["tonic_signal"]))
        budget = max(1, int(max_points) - int(marker_indices.size))
        selected_finite = _select_finite_positions_for_chunk(
            finite_indices.size,
            budget,
            force_first=True,
            force_last=True,
        )
        keep = np.sort(
            np.concatenate((marker_indices, finite_indices[selected_finite]))
        )
        if keep.size > max_points:
            keep = keep[:max_points]
        elapsed = elapsed[keep]
        traces = {name: trace[keep] for name, trace in traces.items()}
    details = {
        "n_chunks": int(len(chunk_records)),
        "n_samples_seen": int(samples_seen),
        "n_finite_samples": int(sum(record["n_finite"] for record in chunk_records)),
        "n_points_plotted": int(elapsed.size),
        "max_plot_points": int(max_points),
        "target_points": int(max_points),
        "method": "deterministic proportional finite-sample selection with endpoint preservation",
        "cache_passes": 2,
        "contains_gap_markers": bool(
            np.any(~np.isfinite(traces["tonic_signal"]))
        ),
    }
    return elapsed, traces, details


def _publish_tonic_overview(
    *,
    run_dir: str,
    roi: str,
    timeline_contract: dict[str, Any],
    tonic_summary: pd.DataFrame,
    max_plot_points: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_path = os.path.join(run_dir, "_analysis", "tonic_out", "tonic_trace_cache.h5")
    _require_file(cache_path, f"{roi} tonic trace cache")
    output_path = os.path.join(run_dir, roi, "summary", TONIC_OVERVIEW_FILENAME)
    with open_tonic_cache(cache_path) as cache:
        elapsed, traces, details = _sample_tonic_trace(
            cache, roi, max_points=max_plot_points
        )
    x_sec = continuous_plot_coordinates(elapsed, timeline_contract)
    summary = tonic_summary.copy()
    summary["window_midpoint_sec"] = (
        pd.to_numeric(summary["window_start_sec"], errors="coerce")
        + pd.to_numeric(summary["window_end_sec"], errors="coerce")
    ) / 2.0
    tonic_data = build_window_plot_data(
        summary,
        timeline_contract=timeline_contract,
        value_column="tonic_value",
    )
    methods = {
        str(value).strip()
        for value in summary["tonic_method"].dropna().astype(str)
        if str(value).strip()
    }
    units = {
        str(value).strip()
        for value in summary["units"].dropna().astype(str)
        if str(value).strip()
    }
    if len(methods) != 1 or len(units) != 1:
        raise GuidedContinuousSavedArtifactError(
            f"Native tonic summary has mixed or missing method/units for ROI {roi!r}: "
            f"methods={sorted(methods)!r}, units={sorted(units)!r}."
        )
    method = next(iter(methods))
    unit = next(iter(units))
    method_label = _NATIVE_TONIC_METHOD_LABELS.get(method, method)
    units_label = _NATIVE_TONIC_UNITS_LABELS.get(unit, unit)
    fallback_values = {
        str(value).strip().lower()
        for value in summary["tonic_fallback"].dropna().astype(str)
    }
    if fallback_values not in ({"true"}, {"false"}):
        raise GuidedContinuousSavedArtifactError(
            f"Native tonic summary has mixed fallback status for ROI {roi!r}: "
            f"{sorted(fallback_values)!r}."
        )
    fallback = fallback_values == {"true"}
    fallback_reasons = {
        str(value).strip()
        for value in summary["fallback_reason"].dropna().astype(str)
        if str(value).strip()
    }
    data = {
        "elapsed_sec": elapsed,
        "x_sec": x_sec,
        "x_hours": x_sec / 3600.0,
        **traces,
    }
    import matplotlib.pyplot as plt

    fig, (raw_ax, tonic_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    raw_ax.plot(
        data["x_hours"],
        data["raw_signal"],
        linewidth=0.7,
        color="green",
        label="Raw signal",
    )
    raw_ax.plot(
        data["x_hours"],
        data["raw_reference"],
        linewidth=0.7,
        color="purple",
        label="Raw reference",
    )
    raw_ax.set_ylabel("Raw signal")
    raw_ax.set_title(f"{roi} Raw signal and reference")
    raw_ax.grid(True, alpha=0.3)
    raw_ax.legend(loc="best")

    tonic_ax.plot(
        tonic_data["x_hours"],
        tonic_data["values"],
        linewidth=1.0,
        color="black",
        marker="o",
        markersize=3.5,
        label=f"{method_label} (P2 per window)",
    )
    tonic_ax.set_xlabel(_timeline_axis_label(timeline_contract))
    tonic_ax.set_ylabel(units_label)
    tonic_ax.set_title(f"{roi} {method_label}")
    x_max = max(
        float(np.nanmax(data["x_hours"])),
        float(np.nanmax(tonic_data["x_hours"])),
    )
    tonic_ax.set_xlim(left=0.0, right=max(1.0, x_max))
    tonic_ax.grid(True, alpha=0.3)
    tonic_ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    dimensions = _validate_image(output_path, f"{roi} tonic overview")
    details.update(
        {
            "image_dimensions": list(dimensions),
            "timeline_mode": str(timeline_contract.get("timeline_mode")),
            "trace_labels": [
                "Raw signal",
                "Raw reference",
                f"{method_label} (P2 per window)",
            ],
            "tonic_method": method,
            "tonic_method_label": method_label,
            "tonic_units": unit,
            "tonic_units_label": units_label,
            "tonic_fallback": fallback,
            "tonic_fallback_reason": "; ".join(sorted(fallback_reasons)),
            "tonic_summary_percentile": float(
                pd.to_numeric(
                    summary["tonic_percentile"], errors="coerce"
                ).dropna().iloc[0]
            ),
            "tonic_summary_points": int(tonic_data["values"].size),
        }
    )
    artifact = _artifact_record(
        run_dir=run_dir,
        roi=roi,
        family="tonic_overview",
        analysis_family="tonic",
        filename=TONIC_OVERVIEW_FILENAME,
        overview_sampling=details,
        image_dimensions=list(dimensions),
        tonic_method=method,
        tonic_method_label=method_label,
        tonic_units=unit,
        tonic_units_label=units_label,
        tonic_fallback=fallback,
        tonic_fallback_reason="; ".join(sorted(fallback_reasons)),
    )
    return artifact, details


def publish_guided_continuous_saved_artifacts(
    run_dir: str,
    *,
    included_roi_ids: tuple[str, ...] | list[str],
    timeline_contract: dict[str, Any],
    window_timing: dict[str, Any],
    phasic_analysis: bool,
    tonic_analysis: bool,
    max_plot_points: int = CONTINUOUS_ARTIFACT_MAX_POINTS,
    include_day_plots: bool = False,
) -> dict[str, Any]:
    """Publish and validate native Guided continuous saved artifacts."""
    started = time.perf_counter()
    run_dir = os.path.abspath(run_dir)
    rois = tuple(str(roi) for roi in included_roi_ids)
    if not rois:
        raise GuidedContinuousSavedArtifactError(
            "Saved continuous Results publication requires at least one ROI."
        )
    if not phasic_analysis and not tonic_analysis:
        raise GuidedContinuousSavedArtifactError(
            "Saved continuous Results publication requires phasic or tonic analysis."
        )
    if int(max_plot_points) < 2:
        raise GuidedContinuousSavedArtifactError("max_plot_points must be at least 2.")
    if not isinstance(window_timing, dict):
        raise GuidedContinuousSavedArtifactError(
            "Accepted continuous window timing provenance is missing."
        )
    for key in ("window_length_sec", "window_step_sec"):
        try:
            value = float(window_timing[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise GuidedContinuousSavedArtifactError(
                f"Accepted continuous window timing is missing {key}."
            ) from exc
        if not np.isfinite(value) or value <= 0:
            raise GuidedContinuousSavedArtifactError(
                f"Accepted continuous window timing has invalid {key}: {value!r}."
            )

    artifacts: list[dict[str, Any]] = []
    performance = {
        "max_plotted_points": 0,
        "cache_passes": 0,
    }
    correction_selection: dict[str, Any] = {}

    if phasic_analysis:
        phasic_out_dir = os.path.join(run_dir, "_analysis", "phasic_out")
        auc_provenance = _load_phasic_auc_provenance(phasic_out_dir)
        for roi in rois:
            summary = _read_roi_summary(run_dir, roi, PHASIC_SUMMARY_FILENAME)
            settings = _auc_settings_for_roi(auc_provenance, roi)
            rate_data = build_window_plot_data(
                summary,
                timeline_contract=timeline_contract,
                value_column="event_rate_per_min",
            )
            auc_data = build_window_plot_data(
                summary,
                timeline_contract=timeline_contract,
                value_column="phasic_signal_auc",
            )
            rate_path = os.path.join(run_dir, roi, "summary", PHASIC_RATE_FILENAME)
            auc_path = os.path.join(run_dir, roi, "summary", PHASIC_AUC_FILENAME)
            rate_details = _write_window_plot(
                data=rate_data,
                title="Peak rate per analysis window",
                ylabel="Peak rate per analysis window (events/min)",
                timeline_contract=timeline_contract,
                out_path=rate_path,
            )
            auc_details = _write_window_plot(
                data=auc_data,
                title="Phasic signal AUC per analysis window",
                ylabel=(
                    "Phasic signal AUC per analysis window "
                    f"({settings['auc_units']})"
                ),
                timeline_contract=timeline_contract,
                out_path=auc_path,
            )
            artifacts.append(
                _artifact_record(
                    run_dir=run_dir,
                    roi=roi,
                    family="phasic_timeseries",
                    analysis_family="phasic",
                    filename=PHASIC_RATE_FILENAME,
                    plot_metric="event_rate_per_min",
                    timeline_mode=timeline_contract.get("timeline_mode"),
                    n_points_plotted=rate_details["n_points_plotted"],
                    image_dimensions=rate_details["image_dimensions"],
                )
            )
            artifacts.append(
                _artifact_record(
                    run_dir=run_dir,
                    roi=roi,
                    family="phasic_timeseries",
                    analysis_family="phasic",
                    filename=PHASIC_AUC_FILENAME,
                    plot_metric="phasic_signal_auc",
                    auc_units=settings["auc_units"],
                    timeline_mode=timeline_contract.get("timeline_mode"),
                    n_points_plotted=auc_details["n_points_plotted"],
                    image_dimensions=auc_details["image_dimensions"],
                )
            )
            correction_artifact, cache_passes = _publish_correction_impact(
                run_dir=run_dir,
                roi=roi,
                timeline_contract=timeline_contract,
            )
            artifacts.append(correction_artifact)
            correction_selection[roi] = correction_artifact["representative_window"]
            performance["cache_passes"] += int(cache_passes)
            performance["max_plotted_points"] = max(
                int(performance["max_plotted_points"]),
                int(rate_details["max_plotted_points"]),
                int(auc_details["max_plotted_points"]),
            )

    tonic_sampling_by_roi: dict[str, Any] = {}
    if tonic_analysis:
        for roi in rois:
            tonic_summary = _read_roi_summary(run_dir, roi, TONIC_SUMMARY_FILENAME)
            for timing_column, timing_key in (
                ("continuous_window_sec", "window_length_sec"),
                ("continuous_step_sec", "window_step_sec"),
            ):
                observed_timing = pd.to_numeric(
                    tonic_summary[timing_column], errors="coerce"
                ).to_numpy(dtype=float)
                if (
                    observed_timing.size == 0
                    or not np.all(np.isfinite(observed_timing))
                    or not np.allclose(
                        observed_timing,
                        float(window_timing[timing_key]),
                        rtol=0.0,
                        atol=1e-9,
                    )
                ):
                    raise GuidedContinuousSavedArtifactError(
                        f"Tonic summary timing column {timing_column!r} for ROI {roi!r} "
                        "does not match the accepted window timing."
                    )
            tonic_artifact, details = _publish_tonic_overview(
                run_dir=run_dir,
                roi=roi,
                timeline_contract=timeline_contract,
                tonic_summary=tonic_summary,
                max_plot_points=int(max_plot_points),
            )
            artifacts.append(tonic_artifact)
            tonic_sampling_by_roi[roi] = details
            performance["cache_passes"] += int(details.get("cache_passes", 0))
            performance["max_plotted_points"] = max(
                int(performance["max_plotted_points"]),
                int(details.get("n_points_plotted", 0)),
            )

    day_plot_artifacts: list[dict[str, Any]] = []
    if include_day_plots:
        if not (phasic_analysis and tonic_analysis):
            raise GuidedContinuousSavedArtifactError(
                "Sampled continuous Day Plots are supported only for the combined "
                "Guided continuous workflow."
            )
        for roi in rois:
            day_plot_artifacts.extend(
                _publish_continuous_day_plots(
                    run_dir=run_dir,
                    roi=roi,
                    timeline_contract=timeline_contract,
                )
            )
        artifacts.extend(day_plot_artifacts)

    expected_artifact_count = len(rois) * (
        (3 if phasic_analysis else 0) + (1 if tonic_analysis else 0)
    ) + len(day_plot_artifacts)
    if len(artifacts) != expected_artifact_count:
        raise GuidedContinuousSavedArtifactError(
            "Saved continuous Results publication produced an incomplete artifact set: "
            f"expected={expected_artifact_count}, actual={len(artifacts)}."
        )

    performance["artifact_elapsed_sec"] = float(time.perf_counter() - started)
    return {
        "artifacts": artifacts,
        "window_timing": {
            "window_length_sec": float(window_timing["window_length_sec"]),
            "window_step_sec": float(window_timing["window_step_sec"]),
            "window_length_source": str(window_timing.get("window_length_source", "")),
            "window_step_source": str(window_timing.get("window_step_source", "")),
        },
        "timeline": dict(timeline_contract),
        "correction_impact_selection_by_roi": correction_selection,
        "tonic_overview_sampling_by_roi": tonic_sampling_by_roi,
        "tonic_method_by_roi": {
            roi: details.get("tonic_method_label", "")
            for roi, details in tonic_sampling_by_roi.items()
        },
        "tonic_units_by_roi": {
            roi: details.get("tonic_units_label", "")
            for roi, details in tonic_sampling_by_roi.items()
        },
        "tonic_fallback_by_roi": {
            roi: bool(details.get("tonic_fallback", False))
            for roi, details in tonic_sampling_by_roi.items()
        },
        "tonic_fallback_reason_by_roi": {
            roi: str(details.get("tonic_fallback_reason", ""))
            for roi, details in tonic_sampling_by_roi.items()
        },
        "performance": performance,
        "timeline_label": timeline_mode_label(str(timeline_contract.get("timeline_mode", ""))),
    }
