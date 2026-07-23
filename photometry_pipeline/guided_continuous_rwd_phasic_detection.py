"""CR1-D3b-A: run the existing phasic detector on one continuous corrected
recording with one-recording (not per-storage-chunk) semantics.

Scientific contract (traced from ``photometry_pipeline/core/feature_extraction.py``
and its production caller in ``pipeline.py``):

* ``get_peak_indices_for_trace(trace, fs_hz, config, ...)`` is the
  authoritative per-trace peak/event index detector, shared by analysis and
  plotting. It operates purely on **sample positions within whatever array is
  passed to it** -- it never receives a time array. Its threshold
  (``compute_detection_threshold_bounds``), prominence
  (``_resolve_prominence_requirement``), and any pre-filter
  (``apply_peak_prefilter``) are each computed **once, from the finite values
  of the array actually passed in** -- there is no per-call state carried
  from a previous invocation.
* ``extract_features(chunk, config)`` calls this same detector on
  ``chunk``'s own event-signal column and returns one row (per ROI) of
  chunk-scoped summary statistics (``mean, median, std, mad, peak_count,
  auc``); it has no concept of "session" or "storage chunk" beyond whatever
  array boundary the caller happened to construct the ``Chunk`` with.

Because both the threshold and ``scipy.signal.find_peaks``'s own minimum-
distance/edge behavior are computed **from whatever array is passed in**,
running the detector independently on each 600-second D1 storage chunk
silently produces a *different* result than running it once on the whole
recording: a different (chunk-local) threshold, a different (chunk-local)
prominence floor, lost minimum-distance suppression across the chunk
boundary, and a real edge-exclusion at every chunk boundary instead of only
at the true start/end of the recording (``scipy.signal.find_peaks`` cannot
report a peak at the very first or last sample of whatever array it is
given). This is empirically demonstrated in
``tests/test_guided_continuous_rwd_phasic_detection.py``.

This module never re-implements or modifies that mathematics. It assembles
one continuous, gap-free ROI trace (and its true global time axis) from an
accepted, finalized D1 corrected cache, then invokes
``core.feature_extraction.extract_features``/``get_peak_indices_for_trace``
**exactly once per ROI** on that one unbroken array -- reproducing "the
existing detector run once on one unbroken trace" exactly, at real-recording
scale, one ROI's arrays in memory at a time.

Scope: this establishes the detection kernel only. It does not write
``features.csv``, ``feature_event_provenance.json``, any run-level manifest/
status/report artifact, or a continuous phasic summary table -- publication
is CR1-D3b-B's responsibility. It does not connect to the GUI, worker,
Results, or Guided Run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from photometry_pipeline.config import Config
from photometry_pipeline.core.feature_extraction import (
    extract_features,
    get_peak_indices_for_trace,
)
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.feature_event_provenance import (
    compute_feature_config_digest,
    feature_fields_from_config,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    GuidedContinuousRwdReviewBinding,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    GuidedContinuousRwdTargetGridDescription,
)
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    list_cache_rois,
    load_cache_chunk_attrs,
    load_cache_chunk_fields,
    open_phasic_cache,
)

EXECUTION_STRATEGY_ROI_AT_A_TIME = "roi_at_a_time"

# Tight tolerances for the cache's own float64 time/rate representation --
# matching the sample-alignment tolerance already established elsewhere in
# this codebase for equivalent checks (e.g.
# run_completion_contract._correction_completion_error_for_analysis's
# ``abs(samples_from_origin - nearest_sample) > 1e-6`` sample-origin check).
_TIME_TOLERANCE_SEC = 1e-6
_FS_HZ_TOLERANCE_HZ = 1e-9


class GuidedContinuousRwdPhasicDetectionError(RuntimeError):
    """A narrow refusal while reconstructing one continuous ROI trace or
    running the existing phasic detector against it."""


@dataclass(frozen=True)
class GuidedContinuousRwdPhasicRoiDetection:
    """One ROI's one-recording phasic detection result."""

    roi_id: str
    mean: float
    median: float
    std: float
    mad: float
    event_count: int
    auc: float
    peak_global_times_sec: np.ndarray
    peak_polarities: np.ndarray

    def __post_init__(self) -> None:
        self.peak_global_times_sec.flags.writeable = False
        self.peak_polarities.flags.writeable = False


@dataclass(frozen=True)
class GuidedContinuousRwdPhasicDetectionResult:
    """One accepted continuous recording's one-recording phasic detection,
    for every canonical included ROI, entirely in memory."""

    recording_identity: str
    target_grid_identity: str
    completion_identity: str
    sampling_rate_hz: float
    included_roi_ids: tuple[str, ...]
    target_sample_count: int
    global_time_start_sec: float
    global_time_end_sec: float
    execution_strategy: str
    detector_parameter_identity: str
    per_roi: dict[str, GuidedContinuousRwdPhasicRoiDetection]


def _check_cancellation(callback: Callable[[], bool] | None) -> None:
    if callback is None:
        return
    if not callable(callback):
        raise GuidedContinuousRwdPhasicDetectionError(
            "cancellation_requested must be callable or None."
        )
    if callback():
        raise GuidedContinuousRwdPhasicDetectionError(
            "Continuous phasic detection was cancelled."
        )


def _validate_cache_authorities(
    cache,
    *,
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
) -> tuple[tuple[str, ...], tuple[int, ...], str]:
    """Validate the finalized D1 cache against the accepted authorities.

    There is no live C4c completion object at this call site (the cache may
    be an accepted, retained artifact from an earlier run), so this checks
    the cache's own self-recorded completion metadata against
    ``review_binding``/``target_grid`` directly, rather than against a
    freshly constructed completion authority
    (:func:`photometry_pipeline.guided_continuous_rwd_correction_run.
    _validate_persisted_cache` does the latter, for the fresh-persist case).

    Returns ``(included_roi_ids, chunk_ids, completion_identity)``.
    """
    meta = cache["meta"]
    if str(meta.attrs.get("mode", "")) != "phasic":
        raise GuidedContinuousRwdPhasicDetectionError(
            "The corrected cache is not the expected phasic-mode D1 schema."
        )
    if str(meta.attrs.get("continuous_acquisition_mode", "")) != "continuous":
        raise GuidedContinuousRwdPhasicDetectionError(
            "The corrected cache does not declare continuous acquisition."
        )
    expected_recording_identity = review_binding.recording.recording_identity
    if str(meta.attrs.get("continuous_recording_identity", "")) != expected_recording_identity:
        raise GuidedContinuousRwdPhasicDetectionError(
            "The corrected cache's recording identity does not match the "
            "accepted review binding."
        )
    if str(meta.attrs.get("continuous_target_grid_identity", "")) != target_grid.target_grid_identity:
        raise GuidedContinuousRwdPhasicDetectionError(
            "The corrected cache's target-grid identity does not match the "
            "accepted target grid."
        )
    if int(meta.attrs.get("continuous_target_sample_count", -1)) != target_grid.target_sample_count:
        raise GuidedContinuousRwdPhasicDetectionError(
            "The corrected cache's target sample count does not match the "
            "accepted target grid."
        )
    completion_identity = str(meta.attrs.get("continuous_completion_identity", ""))
    if not completion_identity:
        raise GuidedContinuousRwdPhasicDetectionError(
            "The corrected cache does not record a completion identity."
        )

    included_roi_ids = tuple(review_binding.recording.roi.included_roi_ids)
    rois = tuple(list_cache_rois(cache))
    if rois != included_roi_ids:
        raise GuidedContinuousRwdPhasicDetectionError(
            "The corrected cache's ROI set/order does not match the accepted "
            f"review binding: cache={rois!r}, expected={included_roi_ids!r}."
        )
    chunk_ids = tuple(int(c) for c in list_cache_chunk_ids(cache))
    corrected_segment_count = int(meta.attrs.get("continuous_corrected_segment_count", -1))
    if len(chunk_ids) != corrected_segment_count:
        raise GuidedContinuousRwdPhasicDetectionError(
            "The corrected cache's chunk count does not match its own "
            "recorded corrected-segment count."
        )
    if chunk_ids != tuple(range(len(chunk_ids))):
        raise GuidedContinuousRwdPhasicDetectionError(
            "The corrected cache's chunk identities are not a contiguous "
            "0-based range."
        )
    return included_roi_ids, chunk_ids, completion_identity


def _assert_uniform_cadence(time_sec: np.ndarray, fs_hz: float, *, context: str) -> None:
    """The one small shared check behind both the per-chunk local-time grid
    validation and the final concatenated-global-grid validation: every
    consecutive sample is separated by exactly one canonical sampling
    interval, within the cache's float64 tolerance."""
    if time_sec.size < 2:
        return
    diffs = np.diff(time_sec)
    if not np.all(np.abs(diffs - (1.0 / fs_hz)) <= _TIME_TOLERANCE_SEC):
        raise GuidedContinuousRwdPhasicDetectionError(
            f"{context}: sample spacing is not uniformly 1/fs_hz "
            f"(fs_hz={fs_hz!r}, max_deviation_sec="
            f"{float(np.max(np.abs(diffs - (1.0 / fs_hz))))!r})."
        )


def _reconstruct_roi_trace(
    cache,
    roi_id: str,
    chunk_ids: tuple[int, ...],
    *,
    event_signal_field: str,
    target_sample_count: int,
    expected_fs_hz: float,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Concatenate one ROI's event-signal values and true global time axis
    across every storage chunk, validating contiguity and cadence as it goes.

    All cadence arithmetic uses the cache's own stored ``fs_hz`` (the exact
    value every chunk's ``time_sec`` was actually written against), not
    ``expected_fs_hz`` recomputed from ``target_grid.cadence_fraction``:
    ``target_grid.cadence_fraction`` is a rational reconstruction that need
    not be bit-identical in float64 to the cache's own value, and the
    existing detector's sample-count arithmetic (``int(...)`` truncation in
    ``core.feature_extraction._resolve_width_samples``/minimum-distance
    resolution) is sensitive at the ULP level. ``expected_fs_hz`` is used
    only to confirm the cache does not silently disagree with the accepted
    target grid.

    Returns ``(event_values, global_time_sec, fs_hz, source_file)``.
    """
    event_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    source_file: str | None = None
    cache_fs_hz: float | None = None
    expected_next_window_start: float | None = None

    for chunk_id in chunk_ids:
        attrs = load_cache_chunk_attrs(cache, roi_id, chunk_id)
        if int(attrs["window_index"]) != chunk_id:
            raise GuidedContinuousRwdPhasicDetectionError(
                f"Chunk {chunk_id} window_index {attrs['window_index']!r} does "
                "not match its own chunk identity."
            )
        chunk_fs_hz = float(attrs["fs_hz"])
        if cache_fs_hz is None:
            if abs(chunk_fs_hz - expected_fs_hz) > _FS_HZ_TOLERANCE_HZ:
                raise GuidedContinuousRwdPhasicDetectionError(
                    f"Chunk {chunk_id} sampling rate {chunk_fs_hz!r} does not "
                    f"match the accepted target-grid cadence {expected_fs_hz!r}."
                )
            cache_fs_hz = chunk_fs_hz
        elif chunk_fs_hz != cache_fs_hz:
            raise GuidedContinuousRwdPhasicDetectionError(
                f"Chunk {chunk_id} sampling rate {chunk_fs_hz!r} does not match "
                f"the recording's established stored rate {cache_fs_hz!r}."
            )
        chunk_source_file = str(attrs.get("source_file", ""))
        if source_file is None:
            source_file = chunk_source_file
        elif chunk_source_file != source_file:
            raise GuidedContinuousRwdPhasicDetectionError(
                f"Chunk {chunk_id} stores a different source_file "
                f"({chunk_source_file!r}) than the recording's established "
                f"value ({source_file!r})."
            )

        window_start = float(attrs["window_start_sec"])
        window_end = float(attrs["window_end_sec"])
        if expected_next_window_start is None:
            if abs(window_start) > _TIME_TOLERANCE_SEC:
                raise GuidedContinuousRwdPhasicDetectionError(
                    "The first storage chunk does not start at the accepted "
                    f"recording origin (window_start_sec={window_start!r})."
                )
        else:
            gap = window_start - expected_next_window_start
            if abs(gap - (1.0 / cache_fs_hz)) > _TIME_TOLERANCE_SEC:
                raise GuidedContinuousRwdPhasicDetectionError(
                    f"Chunk {chunk_id} does not begin exactly one sampling "
                    "interval after the previous chunk's last sample "
                    f"(observed_gap_sec={gap!r})."
                )

        (local_time, event_values) = load_cache_chunk_fields(
            cache, roi_id, chunk_id, ["time_sec", event_signal_field]
        )
        local_time = np.asarray(local_time, dtype=np.float64)
        event_values = np.asarray(event_values, dtype=np.float64)
        if local_time.ndim != 1 or event_values.ndim != 1:
            raise GuidedContinuousRwdPhasicDetectionError(
                f"Chunk {chunk_id} time_sec/{event_signal_field} must be one-dimensional."
            )
        if local_time.size == 0 or event_values.size == 0:
            raise GuidedContinuousRwdPhasicDetectionError(
                f"Chunk {chunk_id} time_sec/{event_signal_field} must not be empty."
            )
        if local_time.shape[0] != event_values.shape[0]:
            raise GuidedContinuousRwdPhasicDetectionError(
                f"Chunk {chunk_id} time_sec length ({local_time.shape[0]}) does "
                f"not match {event_signal_field} length ({event_values.shape[0]})."
            )
        if not np.all(np.isfinite(local_time)):
            raise GuidedContinuousRwdPhasicDetectionError(
                f"Chunk {chunk_id} local time_sec contains non-finite values."
            )
        if abs(float(local_time[0])) > _TIME_TOLERANCE_SEC:
            raise GuidedContinuousRwdPhasicDetectionError(
                f"Chunk {chunk_id} local time_sec does not begin at zero "
                f"(local_time[0]={local_time[0]!r})."
            )
        _assert_uniform_cadence(
            local_time, cache_fs_hz, context=f"Chunk {chunk_id} local time_sec"
        )

        global_time = window_start + local_time
        if not np.isclose(global_time[0], window_start, atol=_TIME_TOLERANCE_SEC):
            raise GuidedContinuousRwdPhasicDetectionError(
                f"Chunk {chunk_id} reconstructed first global time does not "
                "equal its own recorded window_start_sec."
            )
        if not np.isclose(global_time[-1], window_end, atol=_TIME_TOLERANCE_SEC):
            raise GuidedContinuousRwdPhasicDetectionError(
                f"Chunk {chunk_id} reconstructed global time does not reach "
                "its own recorded window_end_sec."
            )
        expected_next_window_start = window_end

        event_parts.append(event_values)
        time_parts.append(global_time)

    assert source_file is not None and cache_fs_hz is not None
    event_trace = np.concatenate(event_parts)
    global_time_sec = np.concatenate(time_parts)
    if event_trace.shape[0] != target_sample_count:
        raise GuidedContinuousRwdPhasicDetectionError(
            "The reconstructed ROI trace does not cover the accepted target "
            f"sample count: reconstructed={event_trace.shape[0]}, "
            f"expected={target_sample_count}."
        )
    if global_time_sec.shape[0] != target_sample_count:
        raise GuidedContinuousRwdPhasicDetectionError(
            "The reconstructed global time axis does not cover the accepted "
            f"target sample count: reconstructed={global_time_sec.shape[0]}, "
            f"expected={target_sample_count}."
        )

    # Final defense-in-depth check over the fully concatenated global grid,
    # not merely per-chunk: acceptable at the current real-data scale.
    if not np.all(np.isfinite(global_time_sec)):
        raise GuidedContinuousRwdPhasicDetectionError(
            "The reconstructed global time axis contains non-finite values."
        )
    if not np.all(np.diff(global_time_sec) > 0):
        raise GuidedContinuousRwdPhasicDetectionError(
            "The reconstructed global time axis is not strictly increasing."
        )
    _assert_uniform_cadence(
        global_time_sec, cache_fs_hz, context="Reconstructed global time axis"
    )
    expected_final_time = (target_sample_count - 1) / cache_fs_hz
    if not np.isclose(global_time_sec[-1], expected_final_time, atol=_TIME_TOLERANCE_SEC):
        raise GuidedContinuousRwdPhasicDetectionError(
            "The reconstructed global time axis's final sample does not match "
            "the accepted target sample count at the accepted recording origin: "
            f"observed={global_time_sec[-1]!r}, expected={expected_final_time!r}."
        )
    return event_trace, global_time_sec, cache_fs_hz, source_file


def _detect_roi(
    roi_id: str,
    event_trace: np.ndarray,
    global_time_sec: np.ndarray,
    *,
    fs_hz: float,
    source_file: str,
    config: Config,
) -> GuidedContinuousRwdPhasicRoiDetection:
    """Run the existing, unmodified phasic detector once on one unbroken
    ROI trace spanning the whole recording."""
    event_signal_field = str(getattr(config, "event_signal", "dff"))
    local_time = global_time_sec - global_time_sec[0]
    chunk = Chunk(
        chunk_id=0,
        source_file=source_file,
        format="rwd",
        time_sec=local_time,
        uv_raw=np.full((event_trace.shape[0], 1), np.nan, dtype=np.float64),
        sig_raw=np.full((event_trace.shape[0], 1), np.nan, dtype=np.float64),
        dff=event_trace.reshape(-1, 1) if event_signal_field == "dff" else None,
        delta_f=event_trace.reshape(-1, 1) if event_signal_field == "delta_f" else None,
        fs_hz=fs_hz,
        channel_names=[roi_id],
        metadata={},
    )
    features = extract_features(chunk, config)
    row = features.iloc[0]

    peak_indices, polarities = get_peak_indices_for_trace(
        event_trace, fs_hz, config, return_polarities=True
    )
    if int(len(peak_indices)) != int(row["peak_count"]):
        raise GuidedContinuousRwdPhasicDetectionError(
            f"ROI {roi_id!r}: independently detected peak count "
            f"({len(peak_indices)}) does not match extract_features' own "
            f"count ({row['peak_count']!r})."
        )

    return GuidedContinuousRwdPhasicRoiDetection(
        roi_id=roi_id,
        mean=float(row["mean"]),
        median=float(row["median"]),
        std=float(row["std"]),
        mad=float(row["mad"]),
        event_count=int(row["peak_count"]),
        auc=float(row["auc"]),
        peak_global_times_sec=global_time_sec[peak_indices].astype(np.float64),
        peak_polarities=polarities.astype(np.int64),
    )


def detect_guided_continuous_rwd_phasic_features(
    corrected_cache_path: str,
    *,
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    config: Config,
    cancellation_requested: Callable[[], bool] | None = None,
) -> GuidedContinuousRwdPhasicDetectionResult:
    """Run the existing phasic detector once per ROI on one accepted,
    finalized D1 corrected cache, reconstructed as one continuous recording.

    Processes one ROI's arrays at a time (``roi_at_a_time``): loads and
    concatenates that ROI's event-signal values and true global time axis
    across every storage chunk, runs the existing, unmodified detector
    (``core.feature_extraction.extract_features`` /
    ``get_peak_indices_for_trace``) exactly once on that unbroken array, then
    releases the arrays before moving to the next ROI. This reproduces
    "the existing detector run once on one unbroken trace" exactly, invariant
    to the arbitrary 600-second D1 storage-chunk boundaries.

    Raises ``GuidedContinuousRwdPhasicDetectionError`` if the cache does not
    match the accepted authorities, is not contiguous/complete, or if
    cancellation is requested. Publishes nothing to disk; this is the
    in-memory detection kernel only (see module docstring).
    """
    if not os.path.isfile(corrected_cache_path):
        raise GuidedContinuousRwdPhasicDetectionError(
            f"The corrected cache does not exist: {corrected_cache_path!r}."
        )

    expected_fs_hz = float(1.0 / float(target_grid.cadence_fraction))

    cache = open_phasic_cache(corrected_cache_path)
    try:
        included_roi_ids, chunk_ids, completion_identity = _validate_cache_authorities(
            cache, review_binding=review_binding, target_grid=target_grid
        )

        event_signal_field = str(getattr(config, "event_signal", "dff"))
        cache_field = "dff" if event_signal_field == "dff" else "delta_f"

        per_roi: dict[str, GuidedContinuousRwdPhasicRoiDetection] = {}
        global_start = global_end = None
        recording_source_file: str | None = None
        validated_fs_hz: float | None = None
        for roi_id in included_roi_ids:
            _check_cancellation(cancellation_requested)
            event_trace, global_time_sec, fs_hz, source_file = _reconstruct_roi_trace(
                cache,
                roi_id,
                chunk_ids,
                event_signal_field=cache_field,
                target_sample_count=target_grid.target_sample_count,
                expected_fs_hz=expected_fs_hz,
            )
            if recording_source_file is None:
                recording_source_file = source_file
            elif source_file != recording_source_file:
                raise GuidedContinuousRwdPhasicDetectionError(
                    f"ROI {roi_id!r} stores a different source_file "
                    f"({source_file!r}) than the recording's established value "
                    f"({recording_source_file!r})."
                )
            if validated_fs_hz is None:
                validated_fs_hz = fs_hz
            elif fs_hz != validated_fs_hz:
                raise GuidedContinuousRwdPhasicDetectionError(
                    f"ROI {roi_id!r} reconstructs with sampling rate {fs_hz!r}, "
                    f"which does not match the recording's established stored "
                    f"rate {validated_fs_hz!r}."
                )
            _check_cancellation(cancellation_requested)
            detection = _detect_roi(
                roi_id,
                event_trace,
                global_time_sec,
                fs_hz=fs_hz,
                source_file=source_file,
                config=config,
            )
            _check_cancellation(cancellation_requested)
            per_roi[roi_id] = detection
            if global_start is None:
                global_start = float(global_time_sec[0])
                global_end = float(global_time_sec[-1])
            del event_trace, global_time_sec
    finally:
        cache.close()

    assert global_start is not None and global_end is not None and validated_fs_hz is not None
    detector_parameter_identity = compute_feature_config_digest(
        feature_fields_from_config(config)
    )
    return GuidedContinuousRwdPhasicDetectionResult(
        recording_identity=review_binding.recording.recording_identity,
        target_grid_identity=target_grid.target_grid_identity,
        completion_identity=completion_identity,
        sampling_rate_hz=validated_fs_hz,
        included_roi_ids=included_roi_ids,
        target_sample_count=target_grid.target_sample_count,
        global_time_start_sec=global_start,
        global_time_end_sec=global_end,
        execution_strategy=EXECUTION_STRATEGY_ROI_AT_A_TIME,
        detector_parameter_identity=detector_parameter_identity,
        per_roi=per_roi,
    )
