"""Adapter-neutral consumed-recording evidence and comparator (B1 completion).

Separates two concepts that must never be conflated:

- ``NormalizedRecordingDescription`` (``guided_normalized_recording.py``) is
  what the scientist AUTHORIZED at Setup check.
- ``NormalizedConsumedRecordingEvidence`` (this module) is what one specific
  analysis branch's execution actually CONSUMED, established from durable,
  execution-generated run-directory artifacts only -- never by re-reading
  the original source tree.

``compare_requested_and_consumed_normalized_recording`` is the one
adapter-neutral comparator: it imports nothing RWD-specific (no C8 keys, no
HDF5 paths, no RWD suffix rules, no folder-timestamp regex, no cache
filenames) and understands only the two normalized dataclasses.
``build_rwd_consumed_normalized_recording_evidence`` is the RWD-specific
adapter that builds the consumed side from real RWD execution artifacts
(the branch's own C8 input-completeness ledger and its own HDF5 trace
cache) -- the one place in this module allowed to know about any of that.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from photometry_pipeline.guided_normalized_recording import (
    NormalizedRecordingDescription,
    SESSION_DISPOSITION_EXCLUDED,
    SESSION_DISPOSITION_MISSING,
    SESSION_DISPOSITION_PROCESS,
)


EVIDENCE_OBSERVED = "observed"
EVIDENCE_UNAVAILABLE = "unavailable"


class NormalizedConsumedEvidenceError(ValueError):
    """Consumed evidence could not be built from the run directory."""

    def __init__(self, category: str, message: str, **context: Any) -> None:
        self.category = str(category)
        self.context = dict(context)
        super().__init__(message)


@dataclass(frozen=True)
class NormalizedConsumedRoiResolution:
    """One ROI's actually-consumed channel resolution for one session."""

    roi_id: str
    resolved_signal_source: str | None
    resolved_reference_source: str | None


@dataclass(frozen=True)
class NormalizedConsumedSession:
    """One session's actually-consumed facts, for one analysis branch."""

    chronological_position: int
    disposition: str
    consumed_source_reference: str
    evidence_availability: str
    content_digest: str | None
    size_bytes: int | None
    cache_chunk_id: int | None
    observed_duration_sec: float | None
    fs_hz: float | None
    resolved_time_column: str | None
    resolved_header_row: int | None
    resolved_timestamp_unit: str | None
    roi_resolutions: tuple[NormalizedConsumedRoiResolution, ...] = ()
    # Appended after the pre-existing optional ROI field to preserve
    # positional construction compatibility for older adapter-neutral users.
    output_time_basis: str | None = None


@dataclass(frozen=True)
class NormalizedConsumedRecordingEvidence:
    """One analysis branch's consumed recording evidence."""

    adapter_format: str
    analysis_branch: str
    sessions: tuple[NormalizedConsumedSession, ...]
    processed_roi_ids: tuple[str, ...]
    parser_policy_satisfied: bool
    parser_policy_failure_category: str | None = None


def _normalize_path_for_comparison(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


# ---------------------------------------------------------------------------
# Adapter-neutral comparator
# ---------------------------------------------------------------------------


def compare_requested_and_consumed_normalized_recording(
    requested: NormalizedRecordingDescription,
    consumed: NormalizedConsumedRecordingEvidence,
) -> str:
    """Compare authorized vs. consumed normalized facts. "" when reconciled.

    Adapter-neutral: understands only the two dataclasses above -- every
    format-specific detail of how ``consumed`` was built is the calling
    adapter's responsibility, never this function's.
    """
    if consumed.adapter_format != requested.adapter_format:
        return (
            f"{consumed.analysis_branch} analysis consumed evidence adapter_format "
            f"{consumed.adapter_format!r} does not match the authorized recording's "
            f"adapter_format {requested.adapter_format!r}"
        )
    if not consumed.parser_policy_satisfied:
        category = consumed.parser_policy_failure_category or "unspecified"
        return (
            f"{consumed.analysis_branch} analysis consumed parser/channel resolution "
            f"violates the authorized parser policy ({category})"
        )

    requested_by_position = {item.chronological_position: item for item in requested.sessions}
    consumed_by_position = {item.chronological_position: item for item in consumed.sessions}
    if set(requested_by_position) != set(consumed_by_position):
        missing = sorted(set(requested_by_position) - set(consumed_by_position))
        extra = sorted(set(consumed_by_position) - set(requested_by_position))
        return (
            f"{consumed.analysis_branch} analysis session set does not match the "
            f"authorized recording (missing positions={missing}, extra positions={extra})"
        )

    for position in sorted(requested_by_position):
        req = requested_by_position[position]
        con = consumed_by_position[position]
        if _normalize_path_for_comparison(
            req.canonical_source_reference
        ) != _normalize_path_for_comparison(con.consumed_source_reference):
            return (
                f"{consumed.analysis_branch} analysis session {position} source "
                "identity does not match the authorized recording"
            )
        if con.disposition != req.disposition:
            return (
                f"{consumed.analysis_branch} analysis session {position} disposition "
                f"{con.disposition!r} does not match the authorized disposition "
                f"{req.disposition!r}"
            )

        if req.disposition == SESSION_DISPOSITION_PROCESS:
            if con.evidence_availability != EVIDENCE_OBSERVED:
                return (
                    f"{consumed.analysis_branch} analysis session {position} has no "
                    "consumed source evidence, but is authorized as an ordinary "
                    "processed session"
                )
            if con.cache_chunk_id is None:
                return (
                    f"{consumed.analysis_branch} analysis session {position} has no "
                    "canonical cache chunk evidence"
                )
            if not req.content_digest or not con.content_digest:
                return (
                    f"{consumed.analysis_branch} analysis session {position} has "
                    "unavailable source content digest evidence"
                )
            if req.content_digest != con.content_digest:
                return (
                    f"{consumed.analysis_branch} analysis session {position} consumed "
                    "source content digest does not match the authorized digest"
                )
            if con.size_bytes is None or req.size_bytes < 0:
                return (
                    f"{consumed.analysis_branch} analysis session {position} has "
                    "unavailable source size evidence"
                )
            if req.size_bytes != con.size_bytes:
                return (
                    f"{consumed.analysis_branch} analysis session {position} consumed "
                    "source size does not match the authorized size"
                )
            if con.fs_hz is None:
                return (
                    f"{consumed.analysis_branch} analysis session {position} has no "
                    "consumed output sampling rate evidence"
                )
            if (
                requested.sampling.target_fs_hz is not None
                and con.fs_hz != requested.sampling.target_fs_hz
            ):
                return (
                    f"{consumed.analysis_branch} analysis session {position} output "
                    f"sampling rate {con.fs_hz!r} does not match the authorized target "
                    f"rate {requested.sampling.target_fs_hz!r}"
                )
            if con.output_time_basis is None:
                return (
                    f"{consumed.analysis_branch} analysis session {position} has no "
                    "consumed output time basis evidence"
                )
            if con.output_time_basis != requested.sampling.time_basis:
                return (
                    f"{consumed.analysis_branch} analysis session {position} output "
                    f"time basis {con.output_time_basis!r} does not match the "
                    f"authorized time basis {requested.sampling.time_basis!r}"
                )
            if not con.resolved_time_column:
                return (
                    f"{consumed.analysis_branch} analysis session {position} has no "
                    "consumed time-column evidence"
                )
            if con.resolved_header_row is None:
                return (
                    f"{consumed.analysis_branch} analysis session {position} has no "
                    "consumed header-row evidence"
                )
            if not con.resolved_timestamp_unit:
                return (
                    f"{consumed.analysis_branch} analysis session {position} has no "
                    "consumed timestamp-unit evidence"
                )
        elif req.disposition == SESSION_DISPOSITION_EXCLUDED:
            if con.cache_chunk_id is not None:
                return (
                    f"{consumed.analysis_branch} analysis session {position} is the "
                    "authorized excluded final session but was processed as an "
                    "ordinary session"
                )
            if con.evidence_availability != EVIDENCE_OBSERVED:
                return (
                    f"{consumed.analysis_branch} analysis session {position} is the "
                    "authorized excluded final session but its source evidence is "
                    "unavailable"
                )
            if not req.content_digest or con.content_digest != req.content_digest:
                return (
                    f"{consumed.analysis_branch} analysis session {position} excluded "
                    "source content digest evidence does not match authorization"
                )
            if con.size_bytes is None or con.size_bytes != req.size_bytes:
                return (
                    f"{consumed.analysis_branch} analysis session {position} excluded "
                    "source size evidence does not match authorization"
                )
        elif req.disposition == SESSION_DISPOSITION_MISSING:
            if con.cache_chunk_id is not None:
                return (
                    f"{consumed.analysis_branch} analysis session {position} is an "
                    "authorized missing session but was processed as an ordinary "
                    "session"
                )
            if con.evidence_availability != EVIDENCE_OBSERVED:
                return (
                    f"{consumed.analysis_branch} analysis session {position} is an "
                    "authorized missing session but its authorization evidence is "
                    "unavailable"
                )
            if not req.content_digest or con.content_digest != req.content_digest:
                return (
                    f"{consumed.analysis_branch} analysis session {position} missing "
                    "source content digest evidence does not match authorization"
                )
            if con.size_bytes is None or con.size_bytes != req.size_bytes:
                return (
                    f"{consumed.analysis_branch} analysis session {position} missing "
                    "source size evidence does not match authorization"
                )

    included_roi_ids = {item.roi_id for item in requested.roi_channels if item.included}
    if set(consumed.processed_roi_ids) != included_roi_ids:
        extra_roi = sorted(set(consumed.processed_roi_ids) - included_roi_ids)
        missing_roi = sorted(included_roi_ids - set(consumed.processed_roi_ids))
        return (
            f"{consumed.analysis_branch} analysis processed ROI set does not match "
            f"the authorized included ROI set (extra={extra_roi}, missing={missing_roi})"
        )

    requested_rois = {
        item.roi_id: item for item in requested.roi_channels if item.included
    }
    for session in consumed.sessions:
        if session.disposition != SESSION_DISPOSITION_PROCESS:
            continue
        consumed_rois = {item.roi_id: item for item in session.roi_resolutions}
        if set(consumed_rois) != included_roi_ids:
            return (
                f"{consumed.analysis_branch} analysis session "
                f"{session.chronological_position} consumed ROI channel evidence "
                "does not cover exactly the authorized included ROIs"
            )
        for roi_id, requested_roi in requested_rois.items():
            consumed_roi = consumed_rois[roi_id]
            if consumed_roi.resolved_signal_source != requested_roi.signal_channel_identity:
                return (
                    f"{consumed.analysis_branch} analysis session "
                    f"{session.chronological_position} ROI {roi_id} consumed signal "
                    "source does not match the authorized signal channel"
                )
            if consumed_roi.resolved_reference_source != requested_roi.reference_channel_identity:
                return (
                    f"{consumed.analysis_branch} analysis session "
                    f"{session.chronological_position} ROI {roi_id} consumed reference "
                    "source does not match the authorized reference channel"
                )

    return ""


def compare_consumed_normalized_recording_branches(
    phasic: NormalizedConsumedRecordingEvidence,
    tonic: NormalizedConsumedRecordingEvidence,
) -> str:
    """Reconcile actual parsing of the same source session across branches.

    Pairing is by chronological position plus normalized source identity, never
    by an unordered set of rates or by branch-local cache chunk ids.
    """
    phasic_by_position = {item.chronological_position: item for item in phasic.sessions}
    tonic_by_position = {item.chronological_position: item for item in tonic.sessions}
    if set(phasic_by_position) != set(tonic_by_position):
        return "cross_branch_session_set_mismatch"

    for position in sorted(phasic_by_position):
        left = phasic_by_position[position]
        right = tonic_by_position[position]
        if (
            left.consumed_source_reference != right.consumed_source_reference
            or left.disposition != right.disposition
        ):
            return f"cross_branch_session_identity_mismatch:{position}"
        if left.disposition != SESSION_DISPOSITION_PROCESS:
            continue
        if left.resolved_time_column != right.resolved_time_column:
            return f"cross_branch_time_column_mismatch:{position}"
        if left.resolved_header_row != right.resolved_header_row:
            return f"cross_branch_header_row_mismatch:{position}"
        if left.resolved_timestamp_unit != right.resolved_timestamp_unit:
            return f"cross_branch_timestamp_unit_mismatch:{position}"
        if left.fs_hz != right.fs_hz:
            return f"cross_branch_output_sampling_rate_mismatch:{position}"
        if left.output_time_basis != right.output_time_basis:
            return f"cross_branch_output_time_basis_mismatch:{position}"
        left_rois = {item.roi_id: item for item in left.roi_resolutions}
        right_rois = {item.roi_id: item for item in right.roi_resolutions}
        if set(left_rois) != set(right_rois):
            return f"cross_branch_roi_channel_set_mismatch:{position}"
        for roi_id in sorted(left_rois):
            if (
                left_rois[roi_id].resolved_signal_source
                != right_rois[roi_id].resolved_signal_source
            ):
                return f"cross_branch_signal_source_mismatch:{position}:{roi_id}"
            if (
                left_rois[roi_id].resolved_reference_source
                != right_rois[roi_id].resolved_reference_source
            ):
                return f"cross_branch_reference_source_mismatch:{position}:{roi_id}"
    return ""


# ---------------------------------------------------------------------------
# RWD-specific consumed-evidence adapter
# ---------------------------------------------------------------------------

_RWD_DISPOSITION_MAP: dict[str, str] = {}


def _disposition_map() -> dict[str, str]:
    global _RWD_DISPOSITION_MAP
    if not _RWD_DISPOSITION_MAP:
        from photometry_pipeline.input_processing_completeness import (
            DISPOSITION_AUTHORIZED_EXCLUSION,
            DISPOSITION_AUTHORIZED_MISSING,
            DISPOSITION_PROCESS,
        )

        _RWD_DISPOSITION_MAP = {
            DISPOSITION_PROCESS: SESSION_DISPOSITION_PROCESS,
            DISPOSITION_AUTHORIZED_EXCLUSION: SESSION_DISPOSITION_EXCLUDED,
            DISPOSITION_AUTHORIZED_MISSING: SESSION_DISPOSITION_MISSING,
        }
    return _RWD_DISPOSITION_MAP


def build_rwd_consumed_normalized_recording_evidence(
    *,
    run_dir: str,
    analysis_kind: str,
    requested: NormalizedRecordingDescription,
) -> NormalizedConsumedRecordingEvidence:
    """Build one branch's consumed evidence from durable run-directory
    artifacts only -- never the live source tree.

    ``analysis_kind`` is ``"phasic"`` or ``"tonic"``. Reads only the
    branch's own ``input_processing_completeness.json`` (C8) and its own
    HDF5 trace cache (including the per-chunk parser/channel attrs
    ``io.hdf5_cache.HDF5CacheWriter.add_chunk`` now stamps every chunk).
    Raises ``NormalizedConsumedEvidenceError`` on missing or malformed
    run-directory artifacts -- fails closed, mirroring the rest of this
    codebase's terminal verification.
    """
    from photometry_pipeline.input_processing_completeness import (
        read_input_completeness,
    )
    from photometry_pipeline.io.hdf5_cache_reader import (
        CacheReadError,
        list_cache_rois,
        load_cache_chunk_attrs,
        open_phasic_cache,
        open_tonic_cache,
    )

    analysis_dir = os.path.join(run_dir, "_analysis", f"{analysis_kind}_out")
    payload, error = read_input_completeness(analysis_dir)
    if payload is None:
        raise NormalizedConsumedEvidenceError(
            "missing_input_completeness_ledger",
            f"the {analysis_kind} analysis input-completeness record is {error}",
        )
    expected = payload.get("expected")
    processed = payload.get("processed")
    if not isinstance(expected, list) or not isinstance(processed, list):
        raise NormalizedConsumedEvidenceError(
            "malformed_input_completeness_ledger",
            f"the {analysis_kind} analysis input-completeness record is malformed",
        )
    processed_by_index: dict[int, dict[str, Any]] = {}
    for record in processed:
        if isinstance(record, dict) and isinstance(record.get("index"), int):
            processed_by_index[int(record["index"])] = record

    cache_path = os.path.join(analysis_dir, f"{analysis_kind}_trace_cache.h5")
    if not os.path.isfile(cache_path):
        raise NormalizedConsumedEvidenceError(
            "missing_trace_cache",
            f"the {analysis_kind} analysis canonical trace cache is missing",
        )
    opener = open_phasic_cache if analysis_kind == "phasic" else open_tonic_cache
    disposition_map = _disposition_map()

    sessions: list[NormalizedConsumedSession] = []
    parser_policy_satisfied = True
    parser_policy_failure_category: str | None = None

    try:
        cache = opener(cache_path)
    except CacheReadError as exc:
        raise NormalizedConsumedEvidenceError(
            "malformed_trace_cache",
            f"the {analysis_kind} analysis canonical trace cache could not be read: {exc}",
        ) from exc

    try:
        try:
            cache_rois = tuple(sorted(list_cache_rois(cache)))
        except CacheReadError as exc:
            raise NormalizedConsumedEvidenceError(
                "malformed_trace_cache",
                f"the {analysis_kind} analysis canonical trace cache ROI index is "
                f"unreadable: {exc}",
            ) from exc

        expected_entries = sorted(
            (entry for entry in expected if isinstance(entry, dict)),
            key=lambda entry: int(entry.get("index", 0)),
        )
        for entry in expected_entries:
            index = int(entry.get("index", 0))
            raw_disposition = str(entry.get("disposition", ""))
            disposition = disposition_map.get(raw_disposition)
            if disposition is None:
                raise NormalizedConsumedEvidenceError(
                    "unknown_session_disposition",
                    f"session {index} has an unrecognized disposition {raw_disposition!r}",
                )
            source = str(entry.get("source", ""))
            size_bytes = entry.get("size_bytes")
            content_digest = entry.get("sha256")
            evidence_availability = (
                EVIDENCE_OBSERVED
                if isinstance(content_digest, str) and content_digest
                else EVIDENCE_UNAVAILABLE
            )
            processed_record = processed_by_index.get(index)

            cache_chunk_id: int | None = None
            observed_duration_sec: float | None = None
            fs_hz: float | None = None
            resolved_time_column: str | None = None
            resolved_header_row: int | None = None
            resolved_timestamp_unit: str | None = None
            output_time_basis: str | None = None
            roi_resolutions: list[NormalizedConsumedRoiResolution] = []

            if disposition == SESSION_DISPOSITION_PROCESS:
                if processed_record is None:
                    raise NormalizedConsumedEvidenceError(
                        "missing_processed_record",
                        f"session {index} has disposition 'process' but no processed "
                        "record",
                    )
                cache_chunk_id = int(processed_record["cache_chunk_id"])
                per_roi_attrs: list[tuple[str, dict[str, Any]]] = []
                for roi in cache_rois:
                    try:
                        attrs = load_cache_chunk_attrs(cache, roi, cache_chunk_id)
                    except CacheReadError as exc:
                        raise NormalizedConsumedEvidenceError(
                            "missing_cache_chunk_evidence",
                            f"session {index} ROI {roi!r} has no canonical cache "
                            f"chunk evidence: {exc}",
                        ) from exc
                    per_roi_attrs.append((roi, attrs))
                    sig = attrs.get("resolved_signal_source")
                    ref = attrs.get("resolved_reference_source")
                    roi_resolutions.append(
                        NormalizedConsumedRoiResolution(
                            roi_id=roi,
                            resolved_signal_source=(str(sig) if sig else None),
                            resolved_reference_source=(str(ref) if ref else None),
                        )
                    )
                    if sig is not None and not any(
                        str(sig).endswith(suffix)
                        for suffix in requested.authorized_signal_suffix_candidates
                    ):
                        parser_policy_satisfied = False
                        parser_policy_failure_category = "unauthorized_signal_source"
                    if ref is not None and not any(
                        str(ref).endswith(suffix)
                        for suffix in requested.authorized_uv_suffix_candidates
                    ):
                        parser_policy_satisfied = False
                        parser_policy_failure_category = "unauthorized_reference_source"

                if per_roi_attrs:
                    _, first_attrs = per_roi_attrs[0]
                    raw_fs_hz = first_attrs.get("fs_hz")
                    fs_hz = float(raw_fs_hz) if raw_fs_hz is not None else None
                    raw_time_col = first_attrs.get("resolved_time_column")
                    resolved_time_column = str(raw_time_col) if raw_time_col else None
                    raw_header_row = first_attrs.get("resolved_header_row")
                    resolved_header_row = (
                        int(raw_header_row) if raw_header_row is not None else None
                    )
                    raw_unit = first_attrs.get("resolved_timestamp_unit")
                    resolved_timestamp_unit = str(raw_unit) if raw_unit else None
                    raw_time_basis = first_attrs.get("output_time_basis")
                    output_time_basis = str(raw_time_basis) if raw_time_basis else None
                    for _roi_name, other_attrs in per_roi_attrs[1:]:
                        if any(
                            other_attrs.get(key) != first_attrs.get(key)
                            for key in (
                                "fs_hz",
                                "resolved_time_column",
                                "resolved_header_row",
                                "resolved_timestamp_unit",
                                "output_time_basis",
                            )
                        ):
                            parser_policy_satisfied = False
                            parser_policy_failure_category = (
                                "intra_branch_parser_evidence_mismatch"
                            )
                    if (
                        resolved_time_column is not None
                        and resolved_time_column
                        not in requested.authorized_time_column_candidates
                    ):
                        parser_policy_satisfied = False
                        parser_policy_failure_category = "unauthorized_time_column"
                    try:
                        time_sec = cache[
                            f"roi/{per_roi_attrs[0][0]}/chunk_{cache_chunk_id}/time_sec"
                        ][()]
                        if len(time_sec) > 0:
                            observed_duration_sec = float(time_sec[-1])
                    except Exception:
                        observed_duration_sec = None

            sessions.append(
                NormalizedConsumedSession(
                    chronological_position=index,
                    disposition=disposition,
                    consumed_source_reference=source,
                    evidence_availability=evidence_availability,
                    content_digest=(
                        str(content_digest) if evidence_availability == EVIDENCE_OBSERVED else None
                    ),
                    size_bytes=(
                        int(size_bytes)
                        if isinstance(size_bytes, int) and not isinstance(size_bytes, bool) and size_bytes >= 0
                        else None
                    ),
                    cache_chunk_id=cache_chunk_id,
                    observed_duration_sec=observed_duration_sec,
                    fs_hz=fs_hz,
                    resolved_time_column=resolved_time_column,
                    resolved_header_row=resolved_header_row,
                    resolved_timestamp_unit=resolved_timestamp_unit,
                    output_time_basis=output_time_basis,
                    roi_resolutions=tuple(roi_resolutions),
                )
            )

        processed_roi_ids = cache_rois
    finally:
        cache.close()

    return NormalizedConsumedRecordingEvidence(
        adapter_format="rwd",
        analysis_branch=analysis_kind,
        sessions=tuple(sessions),
        processed_roi_ids=processed_roi_ids,
        parser_policy_satisfied=parser_policy_satisfied,
        parser_policy_failure_category=parser_policy_failure_category,
    )
