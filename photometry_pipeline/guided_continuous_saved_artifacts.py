"""Saved Results artifacts for native Guided continuous runs.

This module is deliberately a publication adapter.  It reads the accepted
continuous caches, finalized window tables, and persisted provenance; it does
not run correction, detection, event counting, or AUC calculation.
"""

from __future__ import annotations

import json
import os
import time
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
        "is_partial_final_window",
        "continuous_window_sec",
        "continuous_step_sec",
        "acquisition_mode",
    }
)

CONTINUOUS_ARTIFACT_MAX_POINTS = CONTINUOUS_TRACE_OVERVIEW_MAX_POINTS


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
        data["x_hours"],
        data["tonic_signal"],
        linewidth=1.0,
        color="black",
        label="Tonic signal (deltaF)",
    )
    tonic_ax.set_xlabel(_timeline_axis_label(timeline_contract))
    tonic_ax.set_ylabel("Tonic signal (deltaF)")
    tonic_ax.set_title(f"{roi} Tonic overview")
    tonic_ax.set_xlim(left=0.0, right=max(1.0, float(np.nanmax(data["x_hours"]))))
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
            "trace_labels": ["Raw signal", "Raw reference", "Tonic signal (deltaF)"],
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
) -> dict[str, Any]:
    """Publish and validate the required native Guided continuous images."""
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
                max_plot_points=int(max_plot_points),
            )
            artifacts.append(tonic_artifact)
            tonic_sampling_by_roi[roi] = details
            performance["cache_passes"] += int(details.get("cache_passes", 0))
            performance["max_plotted_points"] = max(
                int(performance["max_plotted_points"]),
                int(details.get("n_points_plotted", 0)),
            )

    expected_artifact_count = len(rois) * (
        (3 if phasic_analysis else 0) + (1 if tonic_analysis else 0)
    )
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
        "performance": performance,
        "timeline_label": timeline_mode_label(str(timeline_contract.get("timeline_mode", ""))),
    }
