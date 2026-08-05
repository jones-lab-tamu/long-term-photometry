"""Native Guided continuous tonic publication.

The correction pass still supplies the accepted continuous raw-channel cache
used by the native workflow, but tonic values are computed independently from
those raw signal/reference channels. Each ROI receives one recording-wide
robust isosbestic fit and one method for the entire recording. The existing
saved continuous output windows are then summarized with the settled P2 tonic
rule; an invalid primary fit uses the recording-level signal-only bleach
fallback. The selected phasic correction strategy is never tonic authority.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from photometry_pipeline.config import Config
from photometry_pipeline.continuous_outputs import _window_metadata_row
from photometry_pipeline.core.tonic_dff import (
    apply_global_fit,
    compute_global_iso_fit_robust,
)
from photometry_pipeline.core.tonic_output import _fit_exponential_bleach_trend
from photometry_pipeline.core.reporting import generate_run_report
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.guided_continuous_rwd_block_plan import (
    GuidedContinuousRwdBlockPlan,
)
from photometry_pipeline.guided_continuous_rwd_correction_pass import (
    GuidedContinuousRwdCorrectionPassCompletion,
    GuidedContinuousRwdCorrectionPassError,
    GuidedContinuousRwdCorrectionPassTraversal,
    iterate_guided_continuous_rwd_corrected_segments,
)
from photometry_pipeline.guided_continuous_rwd_correction_pass_persistence import (
    persist_guided_continuous_rwd_correction_pass,
)
from photometry_pipeline.guided_continuous_rwd_correction_run import (
    CORRECTED_CACHE_RELATIVE_PATH,
    _allocate_run_directory,
    _is_cancelled_traversal,
    _notify_continuous_run_started,
    _per_roi_provenance,
    _validate_persisted_cache,
    _write_continuous_progress_status,
    _write_running_status,
    _write_terminal_failure_status,
)
from photometry_pipeline.guided_continuous_rwd_correction_segments import (
    GuidedContinuousRwdCorrectionSegmentPlan,
    GuidedContinuousRwdDynamicF0Authority,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    GuidedContinuousRwdReviewBinding,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    GuidedContinuousRwdTargetGridDescription,
)
from photometry_pipeline.guided_execution_payloads import (
    GuidedExecutionStartupMappingContract,
)
from photometry_pipeline.guided_continuous_saved_artifacts import (
    publish_guided_continuous_saved_artifacts,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from photometry_pipeline.guided_timeline import (
    accepted_continuous_window_timing,
    timeline_provenance_from_intent,
)
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    list_cache_rois,
    load_cache_chunk_attrs,
    load_cache_chunk_fields,
    open_phasic_cache,
    open_tonic_cache,
)
from photometry_pipeline.run_completion_contract import (
    COMPLETION_KEY,
    FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY,
    MANIFEST_FILENAME,
    PROFILE_CONTINUOUS,
    REPORT_COMPLETION_KEY,
    RUN_REPORT_FILENAME,
    STATUS_FILENAME,
    build_continuous_window_index,
    build_manifest_completion_block,
    build_report_completion_block,
    build_status_completion_block,
    classify_run_terminal_state,
    normalize_run_mode,
    sha256_file,
)
from photometry_pipeline.guided_continuous_rwd_correction_run import _write_json

_RUN_ID_PREFIX = "continuous_rwd_tonic_run"
_RUN_PROFILE = "guided_continuous_rwd_tonic"
_RUN_TYPE = "tonic_only"
_TOOL_NAME = "photometry_pipeline.guided_continuous_rwd_tonic_run"
TONIC_ANALYSIS_RELATIVE_DIR = os.path.join("_analysis", "tonic_out")
TONIC_CACHE_FILENAME = "tonic_trace_cache.h5"

TONIC_PERCENTILE = 2.0
MIN_FITTED_ISOSBESTIC = 1e-9
GLOBAL_FIT_MAX_POINTS = 200_000

TONIC_METHOD_GLOBAL_ISOSBESTIC = "Global-isosbestic ΔF/F₀"
TONIC_METHOD_SIGNAL_ONLY = "Signal-only bleach corrected"
TONIC_UNITS_FRACTIONAL = "fractional ΔF/F₀"
TONIC_UNITS_RAW_AU = "raw fluorescence AU"

TONIC_STATUS_VALID = "valid"
TONIC_STATUS_NO_FINITE = "no_finite_samples"
TONIC_STATUS_INVALID_DENOMINATOR = "invalid_denominator"
TONIC_STATUS_UNAVAILABLE = "tonic_unavailable"

TONIC_SUMMARY_COLUMNS = [
    "roi",
    "source_file",
    "chunk_id",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "window_duration_sec",
    "elapsed_hour_start",
    "elapsed_hour_mid",
    "tonic_value",
    "tonic_status",
    "tonic_method",
    "units",
    "tonic_percentile",
    "tonic_fallback",
    "tonic_mean",
    "tonic_median",
    "tonic_min",
    "tonic_max",
    "tonic_p05",
    "tonic_p95",
    "tonic_n_finite",
    "tonic_nan_fraction",
    "global_slope",
    "global_intercept",
    "global_fit_n_used",
    "fallback_reason",
    "is_partial_final_window",
    "original_file_duration_sec",
    "continuous_window_sec",
    "continuous_step_sec",
    "acquisition_mode",
]


class GuidedContinuousRwdTonicRunError(RuntimeError):
    """A narrow refusal while executing or publishing one continuous-RWD
    tonic run. Errors raised directly by C4c/D1 propagate unchanged; this
    exception covers only run-directory-level and tonic-cache-production
    concerns."""


@dataclass(frozen=True)
class GuidedContinuousRwdTonicRunResult:
    """What a caller needs to know about one completed continuous-RWD tonic run."""

    run_dir: str
    run_id: str
    corrected_cache_path: str
    tonic_cache_path: str
    completion: GuidedContinuousRwdCorrectionPassCompletion
    terminal_state: str
    tonic_summary_paths: dict[str, str]
    tonic_summary_row_counts: dict[str, int]


def _build_run_mode(included_roi_ids: tuple[str, ...]) -> dict:
    return normalize_run_mode(
        run_profile=_RUN_PROFILE,
        run_type=_RUN_TYPE,
        acquisition_mode="continuous",
        traces_only=False,
        phasic_analysis=False,
        tonic_analysis=True,
        feature_extraction_ran=False,
        deliverable_profile=PROFILE_CONTINUOUS,
        expected_rois=list(included_roi_ids),
        continuous_outputs_ran=True,
        chunked_input_processing=False,
        shared_input_manifest=False,
    )


def _write_tonic_trace_cache(
    *,
    corrected_cache_path: str,
    tonic_cache_path: str,
    included_roi_ids: tuple[str, ...],
    config: Config,
    window_timing: dict | None = None,
) -> dict[str, Any]:
    """Compute and publish the native continuous tonic result.

    Raw signal/reference arrays are read from the accepted continuous cache.
    The corrected ``delta_f`` arrays are deliberately not read: the native
    tonic result is computed once per ROI from the full recording and then
    written through the existing tonic-cache plumbing. The cache's historical
    ``deltaF`` field carries the authoritative per-sample tonic trace so older
    cache readers still have a bounded trace to display; the summary table and
    saved Results image carry its explicit method and units.
    """
    method_by_roi: dict[str, dict[str, Any]] = {}
    for roi_id in included_roi_ids:
        method_by_roi[roi_id] = _prepare_native_roi_tonic_method(
            roi_id,
            _native_cache_roi_window_factory(
                corrected_cache_path,
                included_roi_ids=included_roi_ids,
                window_timing=window_timing,
                roi=roi_id,
            ),
        )

    writer = Hdf5TraceCacheWriter(tonic_cache_path, "tonic", config)
    rows_by_roi: dict[str, list[dict[str, Any]]] = {
        roi_id: [] for roi_id in included_roi_ids
    }
    chunk_ids: list[int] = []
    try:
        for chunk_id, windows_by_roi in _load_native_tonic_windows(
            corrected_cache_path,
            included_roi_ids=included_roi_ids,
            window_timing=window_timing,
        ):
            sig_cols = []
            uv_cols = []
            tonic_cols = []
            time_sec = None
            fs_hz = None
            source_file = ""
            window_meta: dict[str, Any] = {}
            tonic_trace: np.ndarray | None = None
            for roi_id in included_roi_ids:
                window = windows_by_roi[roi_id]
                if time_sec is None:
                    time_sec = np.asarray(window["time_sec"], dtype=np.float64)
                    fs_hz = float(window["fs_hz"])
                    source_file = str(window["meta"].get("source_file", ""))
                    meta = window["meta"]
                    window_meta = {
                        "acquisition_mode": "continuous",
                        "window_index": float(meta["window_index"]),
                        "window_start_sec": float(meta["window_start_sec"]),
                        "window_end_sec": float(meta["window_end_sec"]),
                        "window_duration_sec": float(meta["window_duration_sec"]),
                        "is_partial_final_window": bool(
                            meta.get("is_partial_final_window", False)
                        ),
                    }
                    for key in ("original_file_duration_sec",):
                        value = meta.get(key)
                        if value is not None and np.isfinite(float(value)):
                            window_meta[key] = float(value)
                    if window_timing is not None:
                        window_meta.update(
                            {
                                "continuous_window_sec": float(
                                    window_timing["window_length_sec"]
                                ),
                                "continuous_step_sec": float(
                                    window_timing["window_step_sec"]
                                ),
                            }
                        )
                sig_cols.append(np.asarray(window["sig"], dtype=np.float64))
                uv_cols.append(np.asarray(window["uv"], dtype=np.float64))
                row, tonic_trace = _build_native_tonic_window_result(
                    roi_id,
                    window,
                    method_by_roi[roi_id],
                )
                rows_by_roi[roi_id].append(row)
                tonic_cols.append(tonic_trace)

            chunk = Chunk(
                chunk_id=int(chunk_id),
                source_file=source_file,
                format="rwd",
                time_sec=time_sec,
                uv_raw=np.column_stack(uv_cols),
                sig_raw=np.column_stack(sig_cols),
                delta_f=np.column_stack(tonic_cols),
                fs_hz=float(fs_hz),
                channel_names=list(included_roi_ids),
                metadata=window_meta,
            )
            writer.add_chunk(chunk, chunk_id=int(chunk_id), source_file=source_file)
            chunk_ids.append(int(chunk_id))
            # Drop the completed window's raw and tonic arrays before asking
            # the cache generator for the next output window.
            del chunk, windows_by_roi, sig_cols, uv_cols, tonic_cols
            del time_sec, fs_hz, source_file, window_meta, tonic_trace
        if not chunk_ids:
            raise GuidedContinuousRwdTonicRunError(
                "The accepted continuous cache contains no tonic output windows."
            )
        for roi_id in included_roi_ids:
            if not any(
                row["tonic_status"] == TONIC_STATUS_VALID
                for row in rows_by_roi[roi_id]
            ):
                raise GuidedContinuousRwdTonicRunError(
                    f"Native tonic produced no valid output for ROI {roi_id!r}."
                )
        writer.finalize()
    except Exception:
        writer.abort()
        raise

    results_by_roi = {
        roi_id: {
            "rows": rows_by_roi[roi_id],
            "tonic_method": method_by_roi[roi_id]["tonic_method"],
            "units": method_by_roi[roi_id]["units"],
            "tonic_fallback": bool(method_by_roi[roi_id]["tonic_fallback"]),
            "fallback_reason": method_by_roi[roi_id]["fallback_reason"],
        }
        for roi_id in included_roi_ids
    }
    return {
        "by_roi": results_by_roi,
        "method_by_roi": {
            roi_id: result["tonic_method"]
            for roi_id, result in results_by_roi.items()
        },
        "units_by_roi": {
            roi_id: result["units"]
            for roi_id, result in results_by_roi.items()
        },
        "fallback_by_roi": {
            roi_id: bool(result["tonic_fallback"])
            for roi_id, result in results_by_roi.items()
        },
        "fallback_reason_by_roi": {
            roi_id: str(result["fallback_reason"])
            for roi_id, result in results_by_roi.items()
        },
    }


def _load_native_tonic_windows(
    corrected_cache_path: str,
    *,
    included_roi_ids: tuple[str, ...],
    window_timing: dict[str, Any] | None,
) -> Iterator[tuple[int, dict[str, dict[str, Any]]]]:
    """Yield one existing persisted continuous output window at a time."""
    source_cache = open_phasic_cache(corrected_cache_path)
    try:
        chunk_ids = [int(chunk_id) for chunk_id in list_cache_chunk_ids(source_cache)]
        if not chunk_ids:
            raise GuidedContinuousRwdTonicRunError(
                "The accepted continuous cache contains no tonic output windows."
            )
        for chunk_id in chunk_ids:
            windows_by_roi: dict[str, dict[str, Any]] = {}
            canonical_time: np.ndarray | None = None
            for roi_id in included_roi_ids:
                attrs = load_cache_chunk_attrs(source_cache, roi_id, chunk_id)
                time_sec, sig_raw, uv_raw = load_cache_chunk_fields(
                    source_cache,
                    roi_id,
                    chunk_id,
                    ["time_sec", "sig_raw", "uv_raw"],
                )
                time_sec = np.asarray(time_sec, dtype=np.float64).reshape(-1)
                sig_raw = np.asarray(sig_raw, dtype=np.float64).reshape(-1)
                uv_raw = np.asarray(uv_raw, dtype=np.float64).reshape(-1)
                if (
                    time_sec.size == 0
                    or sig_raw.shape != time_sec.shape
                    or uv_raw.shape != time_sec.shape
                ):
                    raise GuidedContinuousRwdTonicRunError(
                        f"Native tonic raw window shape mismatch for ROI {roi_id!r}, "
                        f"chunk {chunk_id}."
                    )
                if canonical_time is None:
                    canonical_time = time_sec
                elif not np.array_equal(canonical_time, time_sec):
                    raise GuidedContinuousRwdTonicRunError(
                        f"Native tonic raw window timestamps disagree across ROIs "
                        f"for chunk {chunk_id}."
                    )
                meta = _window_metadata_row(
                    attrs, roi=roi_id, chunk_id=chunk_id
                )
                if window_timing is not None:
                    if not np.isfinite(float(meta["continuous_window_sec"])):
                        meta["continuous_window_sec"] = float(
                            window_timing["window_length_sec"]
                        )
                    if not np.isfinite(float(meta["continuous_step_sec"])):
                        meta["continuous_step_sec"] = float(
                            window_timing["window_step_sec"]
                        )
                windows_by_roi[roi_id] = {
                    "chunk_id": chunk_id,
                    "time_sec": time_sec,
                    "sig": sig_raw,
                    "uv": uv_raw,
                    "fs_hz": float(attrs["fs_hz"]),
                    "meta": meta,
                }
            yield chunk_id, windows_by_roi
            del windows_by_roi
    finally:
        source_cache.close()


def _native_cache_roi_window_factory(
    corrected_cache_path: str,
    *,
    included_roi_ids: tuple[str, ...],
    window_timing: dict[str, Any] | None,
    roi: str,
) -> Callable[[], Iterator[dict[str, Any]]]:
    """Return a fresh bounded cache traversal for one ROI."""
    def factory() -> Iterator[dict[str, Any]]:
        for _chunk_id, windows_by_roi in _load_native_tonic_windows(
            corrected_cache_path,
            included_roi_ids=included_roi_ids,
            window_timing=window_timing,
        ):
            yield windows_by_roi[roi]

    return factory


def _tonic_distribution_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "tonic_mean": np.nan,
            "tonic_median": np.nan,
            "tonic_min": np.nan,
            "tonic_max": np.nan,
            "tonic_p05": np.nan,
            "tonic_p95": np.nan,
        }
    return {
        "tonic_mean": float(np.mean(finite)),
        "tonic_median": float(np.median(finite)),
        "tonic_min": float(np.min(finite)),
        "tonic_max": float(np.max(finite)),
        "tonic_p05": float(np.percentile(finite, 5.0)),
        "tonic_p95": float(np.percentile(finite, 95.0)),
    }


def _tonic_row(
    roi: str,
    window: dict[str, Any],
    *,
    method: str,
    units: str,
    fallback: bool,
    fallback_reason: str,
) -> dict[str, Any]:
    row = dict(window["meta"])
    row.update(
        {
            "roi": str(roi),
            "tonic_value": np.nan,
            "tonic_status": TONIC_STATUS_UNAVAILABLE,
            "tonic_method": method,
            "units": units,
            "tonic_percentile": TONIC_PERCENTILE,
            "tonic_fallback": bool(fallback),
            "tonic_n_finite": 0,
            "tonic_nan_fraction": np.nan,
            "global_slope": np.nan,
            "global_intercept": np.nan,
            "global_fit_n_used": np.nan,
            "fallback_reason": fallback_reason,
        }
    )
    row.update(_tonic_distribution_stats(np.asarray([], dtype=float)))
    return row


def _global_fit_sample_positions(n_pairs: int) -> np.ndarray:
    """Match the bounded deterministic sampling in the existing robust fit."""
    n_pairs = int(n_pairs)
    if n_pairs <= GLOBAL_FIT_MAX_POINTS:
        return np.arange(max(0, n_pairs), dtype=np.int64)
    step = n_pairs / float(GLOBAL_FIT_MAX_POINTS)
    positions = np.arange(0, n_pairs, step).astype(np.int64)
    positions = np.unique(positions)
    return positions[positions < n_pairs]


def _count_finite_pairs(
    window_factory: Callable[[], Iterable[dict[str, Any]]],
) -> int:
    total = 0
    for window in window_factory():
        sig = np.asarray(window["sig"], dtype=float)
        uv = np.asarray(window["uv"], dtype=float)
        total += int(np.count_nonzero(np.isfinite(sig) & np.isfinite(uv)))
        del sig, uv, window
    return total


def _collect_global_fit_sample(
    window_factory: Callable[[], Iterable[dict[str, Any]]],
    selected_positions: np.ndarray,
    total_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect only the bounded paired sample consumed by the robust fit."""
    sample_size = int(selected_positions.size)
    sample_sig = np.empty(sample_size, dtype=np.float64)
    sample_uv = np.empty(sample_size, dtype=np.float64)
    fill = 0
    pair_offset = 0
    for window in window_factory():
        sig = np.asarray(window["sig"], dtype=float)
        uv = np.asarray(window["uv"], dtype=float)
        finite_indices = np.flatnonzero(np.isfinite(sig) & np.isfinite(uv))
        n_finite = int(finite_indices.size)
        if n_finite == 0:
            del finite_indices, sig, uv, window
            continue
        if sample_size == total_pairs:
            sample_sig[fill : fill + n_finite] = sig[finite_indices]
            sample_uv[fill : fill + n_finite] = uv[finite_indices]
            fill += n_finite
            pair_offset += n_finite
            del finite_indices, sig, uv, window
            continue

        global_positions = pair_offset + np.arange(n_finite, dtype=np.int64)
        destinations = np.searchsorted(
            selected_positions, global_positions, side="left"
        )
        in_range = destinations < sample_size
        matches = np.zeros(n_finite, dtype=bool)
        matches[in_range] = (
            selected_positions[destinations[in_range]]
            == global_positions[in_range]
        )
        if np.any(matches):
            selected_destinations = destinations[matches]
            sample_sig[selected_destinations] = sig[finite_indices[matches]]
            sample_uv[selected_destinations] = uv[finite_indices[matches]]
            fill += int(np.count_nonzero(matches))
        pair_offset += n_finite
        del finite_indices, global_positions, destinations, matches, sig, uv, window

    if pair_offset != int(total_pairs) or fill != sample_size:
        raise GuidedContinuousRwdTonicRunError(
            "Native tonic bounded global-fit sampling lost paired samples."
        )
    return sample_uv, sample_sig


def _fit_global_from_window_factory(
    window_factory: Callable[[], Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    """Fit the existing robust model from a bounded deterministic sample."""
    total_pairs = _count_finite_pairs(window_factory)
    selected_positions = _global_fit_sample_positions(total_pairs)
    if selected_positions.size == 0:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "fit_ok": False,
            "n_used": 0,
            "n_pairs": total_pairs,
        }
    sample_uv, sample_sig = _collect_global_fit_sample(
        window_factory, selected_positions, total_pairs
    )
    try:
        slope, intercept, fit_ok, n_used = compute_global_iso_fit_robust(
            sample_uv,
            sample_sig,
            max_points=GLOBAL_FIT_MAX_POINTS,
        )
    except Exception:
        slope, intercept, fit_ok, n_used = 0.0, 0.0, False, 0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "fit_ok": bool(fit_ok),
        "n_used": int(n_used),
        "n_pairs": total_pairs,
    }


def _has_valid_primary_output(
    window_factory: Callable[[], Iterable[dict[str, Any]]],
    *,
    slope: float,
    intercept: float,
) -> bool:
    for window in window_factory():
        sig = np.asarray(window["sig"], dtype=float)
        uv = np.asarray(window["uv"], dtype=float)
        fitted = apply_global_fit(uv, slope, intercept)
        usable = (
            np.isfinite(sig)
            & np.isfinite(uv)
            & np.isfinite(fitted)
            & (fitted > MIN_FITTED_ISOSBESTIC)
        )
        if np.any(usable):
            dff = (sig[usable] - fitted[usable]) / fitted[usable]
            if np.any(np.isfinite(dff)):
                return True
        del fitted, usable, sig, uv, window
    return False


def _collect_fallback_window_values(
    window_factory: Callable[[], Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Keep only the scalar raw floors and midpoint times needed by fallback."""
    values: list[dict[str, Any]] = []
    for window in window_factory():
        signal = np.asarray(window["sig"], dtype=float)
        finite_signal = signal[np.isfinite(signal)]
        values.append(
            {
                "chunk_id": int(window["chunk_id"]),
                "midpoint_sec": (
                    float(window["meta"]["window_start_sec"])
                    + float(window["meta"]["window_end_sec"])
                )
                / 2.0,
                "raw_signal_floor": (
                    float(np.percentile(finite_signal, TONIC_PERCENTILE))
                    if finite_signal.size
                    else np.nan
                ),
            }
        )
        del finite_signal, signal, window
    return values


def _fit_recording_fallback(
    roi: str,
    values: list[dict[str, Any]],
    *,
    fallback_reason: str,
) -> dict[str, Any]:
    usable_positions = [
        index
        for index, value in enumerate(values)
        if np.isfinite(float(value["raw_signal_floor"]))
    ]
    if len(usable_positions) < 3:
        raise GuidedContinuousRwdTonicRunError(
            f"Native tonic signal-only fallback failed for ROI {roi!r}: "
            f"only {len(usable_positions)} valid output windows "
            f"after {fallback_reason}."
        )
    times = np.asarray(
        [values[index]["midpoint_sec"] for index in usable_positions],
        dtype=float,
    )
    baselines = np.asarray(
        [values[index]["raw_signal_floor"] for index in usable_positions],
        dtype=float,
    )
    bleach, bleach_meta = _fit_exponential_bleach_trend(times, baselines)
    if bleach_meta.get("fallback", False):
        raise GuidedContinuousRwdTonicRunError(
            f"Native tonic signal-only fallback failed for ROI {roi!r}: "
            f"bleach fit failed ({bleach_meta.get('reason', 'unknown')}) "
            f"after {fallback_reason}."
        )
    anchor = float(bleach_meta.get("anchor", np.nan))
    if not np.isfinite(anchor) or not np.all(np.isfinite(bleach)):
        raise GuidedContinuousRwdTonicRunError(
            f"Native tonic signal-only fallback failed for ROI {roi!r}: "
            "the recording-level bleach trend is not finite."
        )
    corrected_by_chunk = {
        int(values[position]["chunk_id"]): float(
            baselines[offset] - bleach[offset] + anchor
        )
        for offset, position in enumerate(usable_positions)
    }
    return {
        "slope": np.nan,
        "intercept": np.nan,
        "n_used": np.nan,
        "tonic_method": TONIC_METHOD_SIGNAL_ONLY,
        "units": TONIC_UNITS_RAW_AU,
        "tonic_fallback": True,
        "fallback_reason": fallback_reason,
        "fallback_value_by_chunk": corrected_by_chunk,
    }


def _prepare_native_roi_tonic_method(
    roi: str,
    window_factory: Callable[[], Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    fit = _fit_global_from_window_factory(window_factory)
    slope = float(fit["slope"])
    intercept = float(fit["intercept"])
    if (
        fit["fit_ok"]
        and np.isfinite(slope)
        and slope > 0.0
        and _has_valid_primary_output(
            window_factory, slope=slope, intercept=intercept
        )
    ):
        return {
            "slope": slope,
            "intercept": intercept,
            "n_used": int(fit["n_used"]),
            "tonic_method": TONIC_METHOD_GLOBAL_ISOSBESTIC,
            "units": TONIC_UNITS_FRACTIONAL,
            "tonic_fallback": False,
            "fallback_reason": "",
            "fallback_value_by_chunk": {},
        }

    if not fit["fit_ok"]:
        fallback_reason = "global_fit_failed"
    elif not np.isfinite(slope) or slope <= 0.0:
        fallback_reason = "nonpositive_global_slope"
    else:
        fallback_reason = "invalid_global_denominator"
    fallback_values = _collect_fallback_window_values(window_factory)
    return _fit_recording_fallback(
        roi,
        fallback_values,
        fallback_reason=fallback_reason,
    )


def _build_native_tonic_window_result(
    roi: str,
    window: dict[str, Any],
    method: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """Calculate one window's scalar row and bounded cache trace."""
    signal = np.asarray(window["sig"], dtype=float)
    reference = np.asarray(window["uv"], dtype=float)
    row = _tonic_row(
        roi,
        window,
        method=method["tonic_method"],
        units=method["units"],
        fallback=bool(method["tonic_fallback"]),
        fallback_reason=str(method["fallback_reason"]),
    )
    if not method["tonic_fallback"]:
        fitted = apply_global_fit(
            reference,
            float(method["slope"]),
            float(method["intercept"]),
        )
        usable = (
            np.isfinite(signal)
            & np.isfinite(reference)
            & np.isfinite(fitted)
            & (fitted > MIN_FITTED_ISOSBESTIC)
        )
        tonic_trace = np.full(signal.shape, np.nan, dtype=float)
        tonic_trace[usable] = (signal[usable] - fitted[usable]) / fitted[usable]
        finite_tonic = tonic_trace[np.isfinite(tonic_trace)]
        row["global_slope"] = float(method["slope"])
        row["global_intercept"] = float(method["intercept"])
        row["global_fit_n_used"] = int(method["n_used"])
        row["tonic_n_finite"] = int(finite_tonic.size)
        row["tonic_nan_fraction"] = (
            1.0 - finite_tonic.size / float(signal.size)
            if signal.size
            else np.nan
        )
        if not np.any(np.isfinite(signal) & np.isfinite(reference)):
            row["tonic_status"] = TONIC_STATUS_NO_FINITE
        elif finite_tonic.size == 0:
            row["tonic_status"] = TONIC_STATUS_INVALID_DENOMINATOR
        else:
            row["tonic_status"] = TONIC_STATUS_VALID
            row["tonic_value"] = float(
                np.percentile(finite_tonic, TONIC_PERCENTILE)
            )
            row.update(_tonic_distribution_stats(finite_tonic))
        return row, tonic_trace

    finite_signal = signal[np.isfinite(signal)]
    finite_count = int(finite_signal.size)
    row["tonic_n_finite"] = finite_count
    row["tonic_nan_fraction"] = (
        1.0 - finite_count / float(signal.size) if signal.size else np.nan
    )
    row["tonic_status"] = (
        TONIC_STATUS_VALID if finite_count else TONIC_STATUS_NO_FINITE
    )
    corrected = method["fallback_value_by_chunk"].get(int(window["chunk_id"]))
    tonic_trace = np.full(signal.shape, np.nan, dtype=float)
    if corrected is not None and np.isfinite(float(corrected)):
        row["tonic_status"] = TONIC_STATUS_VALID
        row["tonic_value"] = float(corrected)
        row.update(_tonic_distribution_stats(np.full(finite_count, corrected)))
        tonic_trace[np.isfinite(signal)] = float(corrected)
    elif row["tonic_status"] == TONIC_STATUS_VALID:
        row["tonic_status"] = TONIC_STATUS_UNAVAILABLE
        row["fallback_reason"] = (
            f"{method['fallback_reason']};bleach_fit_failed:nonfinite_trend"
        )
    return row, tonic_trace


def _compute_native_roi_tonic_result(
    roi: str, windows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute one ROI from a bounded window sequence for deterministic tests."""
    if not windows:
        raise GuidedContinuousRwdTonicRunError(
            f"Native tonic has no output windows for ROI {roi!r}."
        )
    method = _prepare_native_roi_tonic_method(roi, lambda: iter(windows))
    rows = [
        _build_native_tonic_window_result(roi, window, method)[0]
        for window in windows
    ]
    if not any(row["tonic_status"] == TONIC_STATUS_VALID for row in rows):
        raise GuidedContinuousRwdTonicRunError(
            f"Native tonic produced no valid output for ROI {roi!r}."
        )
    return {
        "rows": rows,
        "tonic_method": method["tonic_method"],
        "units": method["units"],
        "tonic_fallback": bool(method["tonic_fallback"]),
        "fallback_reason": method["fallback_reason"],
    }


def _validate_tonic_cache(
    tonic_cache_path: str,
    *,
    included_roi_ids: tuple[str, ...],
    completion: GuidedContinuousRwdCorrectionPassCompletion,
) -> None:
    """Reopen the just-written tonic cache through the existing reader and
    confirm it faithfully represents the one continuous recording: canonical
    ROI order, every corrected storage chunk present, no duplication."""
    if not os.path.isfile(tonic_cache_path):
        raise GuidedContinuousRwdTonicRunError(
            "The tonic trace cache is missing after tonic-cache production claimed success."
        )
    if os.path.isfile(tonic_cache_path + ".tmp"):
        raise GuidedContinuousRwdTonicRunError(
            "A .tmp tonic-cache artifact remains after finalize."
        )
    cache = open_tonic_cache(tonic_cache_path)
    try:
        rois = list_cache_rois(cache)
        if rois != list(included_roi_ids):
            raise GuidedContinuousRwdTonicRunError(
                "The tonic cache's ROI set/order does not match the accepted "
                f"review binding: cache={rois!r}, expected={list(included_roi_ids)!r}."
            )
        chunk_ids = list_cache_chunk_ids(cache)
        if len(chunk_ids) != completion.corrected_segment_count:
            raise GuidedContinuousRwdTonicRunError(
                "The tonic cache's chunk count does not match the C4c "
                f"completion: cache={len(chunk_ids)}, "
                f"expected={completion.corrected_segment_count}."
            )
        if sorted(chunk_ids) != list(range(completion.corrected_segment_count)):
            raise GuidedContinuousRwdTonicRunError(
                "The tonic cache's chunk identities are not a contiguous "
                "0-based range."
            )
    finally:
        cache.close()


def _generate_tonic_summary(
    run_dir: str,
    tonic_out_dir: str,
    included_roi_ids: tuple[str, ...],
    tonic_result: dict[str, Any],
) -> tuple[dict[str, str], dict[str, int]]:
    """Write the native recording-wide tonic result into existing table paths.

    Returns ``(relative_paths_by_roi, row_counts_by_roi)``. Raises
    ``GuidedContinuousRwdTonicRunError`` if any ROI is missing, mixed-method,
    or has no output rows.
    """
    cache_path = os.path.join(tonic_out_dir, TONIC_CACHE_FILENAME)
    if not os.path.isfile(cache_path):
        raise GuidedContinuousRwdTonicRunError(
            f"The native tonic cache is missing before summary publication: {cache_path}"
        )
    relative_paths = {
        roi_id: f"{roi_id}/tables/continuous_tonic_window_summary.csv"
        for roi_id in included_roi_ids
    }
    row_counts_by_roi: dict[str, int] = {}
    for roi_id, relative_path in relative_paths.items():
        roi_result = (tonic_result.get("by_roi") or {}).get(roi_id)
        if not isinstance(roi_result, dict):
            raise GuidedContinuousRwdTonicRunError(
                f"Native tonic did not produce a result for ROI {roi_id!r}."
            )
        rows = list(roi_result.get("rows") or [])
        if not rows:
            raise GuidedContinuousRwdTonicRunError(
                f"Native tonic produced no output rows for ROI {roi_id!r}."
            )
        methods = {str(row.get("tonic_method", "")) for row in rows}
        units = {str(row.get("units", "")) for row in rows}
        if len(methods) != 1 or len(units) != 1:
            raise GuidedContinuousRwdTonicRunError(
                f"Native tonic selected mixed methods or units for ROI {roi_id!r}: "
                f"methods={sorted(methods)!r}, units={sorted(units)!r}."
            )
        output_path = os.path.join(run_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        frame = pd.DataFrame(rows, columns=TONIC_SUMMARY_COLUMNS)
        frame = frame.sort_values(
            ["window_index", "chunk_id"], kind="stable"
        ).reset_index(drop=True)
        frame.to_csv(output_path, index=False)
        row_counts_by_roi[roi_id] = int(len(frame))
    return relative_paths, row_counts_by_roi


def execute_guided_continuous_rwd_tonic_run(
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    block_plan: GuidedContinuousRwdBlockPlan,
    segment_plan: GuidedContinuousRwdCorrectionSegmentPlan,
    dynamic_f0_authority: GuidedContinuousRwdDynamicF0Authority,
    *,
    accepted_draft: GuidedNewAnalysisDraftPlan,
    startup_mapping_contract: GuidedExecutionStartupMappingContract,
    output_base: str,
    config: Config,
    cancellation_requested: Callable[[], bool] | None = None,
    run_started_callback: Callable[[str, str], None] | None = None,
) -> GuidedContinuousRwdTonicRunResult:
    """Produce one coherent Native Guided continuous-RWD tonic run.

    Tonic is computed from one recording-wide raw signal/reference method per
    ROI, summarized over the accepted continuous output windows, and published
    through the existing tonic trace-cache writer/reader.

    Accepts exactly the same accepted continuous authorities as
    :func:`photometry_pipeline.guided_continuous_rwd_correction_run.
    execute_guided_continuous_rwd_correction_run`, plus the same
    ``output_base``/``config``/``cancellation_requested``. Publication order:
    allocate the run directory -> write a running status -> build the C4c
    traversal and persist it through D1 -> cross-check the finalized
    correction cache against the accepted authorities and the C4c completion
    -> compute and publish the recording-wide tonic result into a genuine
    ``_analysis/tonic_out/tonic_trace_cache.h5`` via the existing tonic-mode
    writer -> write the existing ``_analysis/tonic_out/{run_report.json,
    config_used.yaml}`` pair via the existing production report writer ->
    write the existing continuous tonic window summary from that result ->
    write the run-level ``run_report.json`` -> build and write
    ``MANIFEST.json`` (with the tonic continuous-window index) -> write the
    final success ``status.json`` -> run the existing completed-run
    validator as the last gate. Any failure at any step writes a terminal
    ``error``/``cancelled`` status instead and re-raises -- no run directory
    this function touches can be left claiming success after a failure or
    cancellation.

    Phasic/feature analysis is not run and is explicitly recorded as such in
    the run mode and run report.
    """
    included_roi_ids = tuple(review_binding.recording.roi.included_roi_ids)
    run_mode = _build_run_mode(included_roi_ids)
    timeline_contract = timeline_provenance_from_intent(
        accepted_draft.execution_intent
    )
    window_timing = accepted_continuous_window_timing(accepted_draft)
    run_id, run_dir = _allocate_run_directory(output_base)
    _write_running_status(
        run_dir, run_id=run_id, run_mode=run_mode, phase="initializing"
    )
    _notify_continuous_run_started(run_started_callback, run_dir, run_id)

    cache_path = os.path.join(run_dir, CORRECTED_CACHE_RELATIVE_PATH)
    tonic_out_dir = os.path.join(run_dir, TONIC_ANALYSIS_RELATIVE_DIR)
    tonic_cache_path = os.path.join(tonic_out_dir, TONIC_CACHE_FILENAME)
    traversal: GuidedContinuousRwdCorrectionPassTraversal | None = None
    try:
        _write_continuous_progress_status(
            run_dir,
            run_id=run_id,
            run_mode=run_mode,
            phase="preparing_recording",
        )
        _write_continuous_progress_status(
            run_dir,
            run_id=run_id,
            run_mode=run_mode,
            phase="correcting_signals",
        )
        traversal = iterate_guided_continuous_rwd_corrected_segments(
            review_binding,
            target_grid,
            block_plan,
            segment_plan,
            dynamic_f0_authority,
            accepted_draft=accepted_draft,
            startup_mapping_contract=startup_mapping_contract,
            cancellation_requested=cancellation_requested,
        )
        completion = persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=review_binding,
            target_grid=target_grid,
            output_path=cache_path,
            config=config,
        )
        _validate_persisted_cache(
            cache_path,
            review_binding=review_binding,
            target_grid=target_grid,
            completion=completion,
        )

        os.makedirs(tonic_out_dir, exist_ok=True)
        _write_continuous_progress_status(
            run_dir,
            run_id=run_id,
            run_mode=run_mode,
            phase="analyzing_tonic_signal",
        )
        tonic_result = _write_tonic_trace_cache(
            corrected_cache_path=cache_path,
            tonic_cache_path=tonic_cache_path,
            included_roi_ids=included_roi_ids,
            config=config,
            window_timing=window_timing,
        )
        _validate_tonic_cache(
            tonic_cache_path, included_roi_ids=included_roi_ids, completion=completion
        )
        _write_continuous_progress_status(
            run_dir,
            run_id=run_id,
            run_mode=run_mode,
            phase="building_summaries",
        )
        generate_run_report(config, tonic_out_dir, traces_only=False)

        tonic_paths, tonic_row_counts = _generate_tonic_summary(
            run_dir, tonic_out_dir, included_roi_ids, tonic_result
        )
        _write_continuous_progress_status(
            run_dir,
            run_id=run_id,
            run_mode=run_mode,
            phase="saving_results",
        )
        saved_artifacts = publish_guided_continuous_saved_artifacts(
            run_dir,
            included_roi_ids=included_roi_ids,
            timeline_contract=timeline_contract,
            window_timing=window_timing,
            phasic_analysis=False,
            tonic_analysis=True,
        )
        provenance = _per_roi_provenance(cache_path, included_roi_ids, first_chunk_id=0)

        report = {
            "schema_name": "guided_continuous_rwd_tonic_run_report",
            "schema_version": "v1",
            "run_context": {"run_id": run_id, "stage": "continuous_correction_and_tonic"},
            "summary": {
                "narrative": (
                    "Continuous recording correction completed. Tonic (slow, "
                    "sustained-signal) analysis completed for this recording, "
                    f"producing a per-window tonic summary for each of the "
                    f"{len(included_roi_ids)} region(s) of interest. Phasic "
                    "(event) analysis has not been run for this recording."
                ),
            },
            "source": {
                "acquisition_mode": "continuous",
                "canonical_source_path": review_binding.recording.source.fluorescence_path_canonical,
                "source_content_identity": review_binding.recording.source.source_content_identity,
                "recording_identity": review_binding.recording.recording_identity,
            },
            "timeline": timeline_contract,
            "target_grid": {
                "target_grid_identity": target_grid.target_grid_identity,
                "target_sample_count": target_grid.target_sample_count,
            },
            "included_roi_ids": list(included_roi_ids),
            "per_roi_correction": provenance,
            "corrected_cache": {
                "relative_path": CORRECTED_CACHE_RELATIVE_PATH,
                "corrected_segment_count": completion.corrected_segment_count,
            },
            "tonic_analysis": {
                "trace_cache_relative_path": f"{TONIC_ANALYSIS_RELATIVE_DIR}/{TONIC_CACHE_FILENAME}".replace(
                    "\\", "/"
                ),
                "output_relative_paths": tonic_paths,
                "window_row_counts": tonic_row_counts,
                "tonic_method_by_roi": tonic_result["method_by_roi"],
                "tonic_units_by_roi": tonic_result["units_by_roi"],
                "tonic_fallback_by_roi": tonic_result["fallback_by_roi"],
                "tonic_fallback_reason_by_roi": tonic_result[
                    "fallback_reason_by_roi"
                ],
            },
            "saved_artifacts": saved_artifacts,
            "continuous_correction_pass_completion_identity": completion.completion_identity,
        }
        report[REPORT_COMPLETION_KEY] = build_report_completion_block(run_id=run_id)
        _write_json(os.path.join(run_dir, RUN_REPORT_FILENAME), report)

        continuous_index = build_continuous_window_index(
            run_dir,
            run_mode=run_mode,
            row_counts_by_family={
                FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY: dict(tonic_row_counts),
            },
            saved_artifacts=saved_artifacts["artifacts"],
            window_timing=saved_artifacts["window_timing"],
        )
        finalized_utc = datetime.now(timezone.utc).isoformat()
        manifest = {
            "tool": _TOOL_NAME,
            "run_id": run_id,
            "run_profile": run_mode["run_profile"],
            "run_type": run_mode["run_type"],
            "timeline": timeline_contract,
            COMPLETION_KEY: build_manifest_completion_block(
                run_dir,
                run_id=run_id,
                run_mode=run_mode,
                finalized_utc=finalized_utc,
                optional_artifacts=[CORRECTED_CACHE_RELATIVE_PATH],
                continuous_index=continuous_index,
            ),
        }
        manifest_path = os.path.join(run_dir, MANIFEST_FILENAME)
        _write_json(manifest_path, manifest)

        status = {
            "schema_version": 1,
            "run_id": run_id,
            "run_profile": run_mode["run_profile"],
            "run_type": run_mode["run_type"],
            "acquisition_mode": run_mode["acquisition_mode"],
            "traces_only": run_mode["traces_only"],
            "phase": "final",
            "status": "success",
            "errors": [],
            "warnings": [],
            COMPLETION_KEY: build_status_completion_block(
                run_id=run_id, manifest_sha256=sha256_file(manifest_path)
            ),
        }
        _write_json(os.path.join(run_dir, STATUS_FILENAME), status)

        classification = classify_run_terminal_state(run_dir)
        if not classification.is_success:
            raise GuidedContinuousRwdTonicRunError(
                "The existing completed-run validator refused this run: "
                f"{classification.reason}"
            )
    except Exception as exc:
        cancelled = (
            isinstance(exc, GuidedContinuousRwdCorrectionPassError)
            and exc.category == "segment_correction_pass_interrupted"
        ) or _is_cancelled_traversal(traversal)
        _write_terminal_failure_status(
            run_dir, run_id=run_id, run_mode=run_mode, cancelled=cancelled, message=str(exc)
        )
        raise

    return GuidedContinuousRwdTonicRunResult(
        run_dir=run_dir,
        run_id=run_id,
        corrected_cache_path=cache_path,
        tonic_cache_path=tonic_cache_path,
        completion=completion,
        terminal_state=classification.state,
        tonic_summary_paths=tonic_paths,
        tonic_summary_row_counts=tonic_row_counts,
    )
