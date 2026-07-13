"""Shared, format-neutral normalized recording description (B1).

Format-specific adapters (currently: RWD only) populate one shared model.
Downstream Guided code that only needs the *scientific* facts of a
recording -- session order, timing, ROI/channel identity, sampling -- can
consume this model instead of re-deriving RWD folder-naming, NPM
file-layout, or custom-tabular column-mapping conventions.

This module does not implement a second discovery, ordering, or timing
system. The RWD adapter here (``build_rwd_normalized_recording_description``)
consumes the already-authoritative A2 chronology
(``io.rwd_source_snapshot.GuidedRwdSourceCandidateSnapshot``, whose
candidate order already comes from ``io.rwd_chronology``) and the existing
folder-timestamp parser (``input_processing_completeness.resolve_session_start_time``)
rather than re-parsing or re-sorting anything.

Only RWD is implemented. NPM and custom-tabular adapters are out of scope
for this milestone; ``SUPPORTED_ADAPTER_FORMATS`` is the single place a
future adapter would register itself.

Identity rule (resolves the A2-noted ambiguity): this schema uses
**implementation-bound identity** (Option A). The canonical identity
includes ``adapter_format`` and ``adapter_contract_version`` alongside the
normalized scientific facts, as one combined digest -- matching the
existing A1 (``guided_plan_identity``) pattern of one combined digest for
"what was validated," and the existing startup-preflight pattern of exact
digest matching. Two different adapter formats/versions therefore produce
different identities even for scientifically equivalent facts, by design.
Only ``adapter_evidence`` (raw, format-specific provenance, e.g. which RWD
folder token supplied a timestamp) is excluded from the identity, because
it is diagnostic metadata, not a normalized fact and not a derivation
contract. If cross-adapter equivalence is ever needed, that requires a
separate, explicitly-scoped identity -- it is not implied here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import os
from typing import Any, Mapping

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.input_processing_completeness import (
    resolve_session_start_time,
)
from photometry_pipeline.io.adapters import (
    rwd_authorized_suffix_candidates,
    rwd_authorized_time_column_candidates,
)


NORMALIZED_RECORDING_SCHEMA_NAME = "guided_normalized_recording_description"
NORMALIZED_RECORDING_SCHEMA_VERSION = "v1"
NORMALIZED_RECORDING_IDENTITY_DOMAIN = "guided-normalized-recording-description:v1"
RWD_ADAPTER_CONTRACT_VERSION = "rwd_normalized_recording_adapter.v1"

# Where an RWD session's authoritative_source_start_time evidence came
# from. Today there is exactly one supported source; the field exists so a
# later evidence source (or a later format) doesn't silently get
# conflated with this one.
RWD_FOLDER_TIMESTAMP_EVIDENCE = "rwd_session_folder_canonical_timestamp"

SESSION_DISPOSITION_PROCESS = "process"
SESSION_DISPOSITION_MISSING = "missing"
SESSION_DISPOSITION_EXCLUDED = "excluded"
_SESSION_DISPOSITIONS = frozenset(
    {SESSION_DISPOSITION_PROCESS, SESSION_DISPOSITION_MISSING, SESSION_DISPOSITION_EXCLUDED}
)

# The only adapter formats this schema currently accepts. A future NPM or
# custom-tabular adapter registers here and supplies its own
# build_<format>_normalized_recording_description function; nothing else
# in this module needs to change.
SUPPORTED_ADAPTER_FORMATS = ("rwd",)


class NormalizedRecordingError(ValueError):
    """A normalized recording description could not be built, is invalid,
    or could not be safely deserialized."""

    def __init__(self, category: str, message: str, **context: Any) -> None:
        self.category = str(category)
        self.context = dict(context)
        super().__init__(message)


@dataclass(frozen=True)
class NormalizedSourceSession:
    """One session's normalized scientific facts.

    Deliberately format-neutral vocabulary: no field name here presumes
    RWD folder conventions, NPM file layout, or custom-tabular columns.

    Timing fields are deliberately kept distinct even though, for clean
    RWD data, several of them carry the same value today:

    - ``authoritative_source_start_time`` is the acquisition start time
      established from this session's *own* source evidence (for RWD: its
      folder's canonical timestamp token). This is an observation, not a
      prediction.
    - ``source_timing_evidence`` names *where* that observation came from.
    - ``expected_timeline_start_time`` is what the confirmed cadence
      (``sessions_per_hour``) predicts for this position, anchored at the
      first session's authoritative start. It is a schedule prediction,
      not an observation, and is None when no schedule model is
      available. For RWD today every session's own folder identity is
      required (even missing/excluded ones), so this is always
      computable, but the two concepts must not be collapsed: a future
      adapter (or a gap-detection feature) may have one without the
      other.
    """

    stable_source_identity: str
    canonical_source_reference: str
    chronological_position: int
    authoritative_source_start_time: str
    source_timing_evidence: str
    expected_timeline_start_time: str | None
    expected_duration_sec: float | None
    observed_duration_sec: float | None
    disposition: str
    size_bytes: int
    content_digest: str

    def __post_init__(self) -> None:
        if not self.stable_source_identity:
            raise NormalizedRecordingError(
                "invalid_session_identity", "stable_source_identity is required."
            )
        if self.disposition not in _SESSION_DISPOSITIONS:
            raise NormalizedRecordingError(
                "invalid_session_disposition",
                f"Unsupported session disposition: {self.disposition!r}.",
            )
        if self.chronological_position < 0:
            raise NormalizedRecordingError(
                "invalid_chronological_position",
                "chronological_position must be >= 0.",
            )
        if not self.authoritative_source_start_time:
            # Every session -- process, missing, or excluded -- must carry
            # an established authoritative start time; a session whose
            # time cannot be established is refused earlier, by the
            # adapter itself, rather than represented with a
            # blank/fabricated value here.
            raise NormalizedRecordingError(
                "unresolved_session_time",
                "authoritative_source_start_time is required.",
                stable_source_identity=self.stable_source_identity,
            )
        if not self.source_timing_evidence:
            raise NormalizedRecordingError(
                "unresolved_timing_evidence",
                "source_timing_evidence is required.",
                stable_source_identity=self.stable_source_identity,
            )


@dataclass(frozen=True)
class NormalizedRoiChannel:
    """One ROI's included/excluded state and channel identity."""

    roi_id: str
    included: bool
    signal_channel_identity: str
    reference_channel_identity: str

    def __post_init__(self) -> None:
        if not self.roi_id:
            raise NormalizedRecordingError("invalid_roi_id", "roi_id is required.")
        if not self.signal_channel_identity or not self.reference_channel_identity:
            raise NormalizedRecordingError(
                "invalid_channel_pairing",
                f"ROI {self.roi_id!r} is missing a signal or reference channel identity.",
            )


@dataclass(frozen=True)
class NormalizedSamplingContract:
    """The validated facts execution needs, independent of file format."""

    time_basis: str
    parser_contract_identity: str
    sessions_per_hour: int | None
    session_duration_sec: float | None
    # The target output rate is an authorized production fact when the
    # acquisition contract carries it.  It remains optional for older
    # format-neutral/unit-test descriptions that predate this field.
    target_fs_hz: float | None = None

    def __post_init__(self) -> None:
        if not self.time_basis:
            raise NormalizedRecordingError("invalid_time_basis", "time_basis is required.")
        if not self.parser_contract_identity:
            raise NormalizedRecordingError(
                "invalid_parser_contract_identity",
                "parser_contract_identity is required.",
            )


@dataclass(frozen=True)
class NormalizedRecordingDescription:
    """The one shared, format-neutral recording-description model.

    See the module docstring for the identity rule (Option A,
    implementation-bound). ``adapter_evidence`` is the only field excluded
    from the canonical identity -- it is raw, format-specific provenance
    retained for diagnostics (see ``build_normalized_recording_description_payload``).
    """

    schema_name: str
    schema_version: str
    adapter_format: str
    adapter_contract_version: str
    acquisition_mode: str
    timeline_anchor_mode: str
    recording_source_identity: str
    source_evidence_identity: str
    sessions: tuple[NormalizedSourceSession, ...]
    roi_channels: tuple[NormalizedRoiChannel, ...]
    sampling: NormalizedSamplingContract
    # The exact, expanded candidate tuples real per-session execution-time
    # parsing will accept, captured once from the frozen authorized
    # configuration (see io.adapters.rwd_authorized_time_column_candidates /
    # rwd_authorized_suffix_candidates) and persisted verbatim. This is
    # deliberately distinct from io.rwd_contract's separate, config-independent
    # preflight parser contract: real execution heterogeneously resolves a
    # different member of these tuples per source session, so terminal
    # verification checks per-session membership against this persisted
    # content -- never a re-derivation from whatever the installed
    # application build's fallback constants currently are.
    authorized_time_column_candidates: tuple[str, ...] = ()
    authorized_uv_suffix_candidates: tuple[str, ...] = ()
    authorized_signal_suffix_candidates: tuple[str, ...] = ()
    adapter_evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_name != NORMALIZED_RECORDING_SCHEMA_NAME:
            raise NormalizedRecordingError(
                "unsupported_schema", "Unsupported normalized recording schema name."
            )
        if self.schema_version != NORMALIZED_RECORDING_SCHEMA_VERSION:
            raise NormalizedRecordingError(
                "unsupported_schema", "Unsupported normalized recording schema version."
            )
        if not self.adapter_format or not self.adapter_contract_version:
            # The model itself is format-neutral: it validates that *some*
            # adapter identified itself, not *which* adapter. Restricting
            # to currently-supported formats (SUPPORTED_ADAPTER_FORMATS)
            # is a production-boundary policy, enforced where Setup check
            # decides which adapters it currently accepts -- not baked
            # into this shared model, so a test-only fake adapter can
            # exercise the shared boundary without a registry change.
            raise NormalizedRecordingError(
                "invalid_adapter_identity",
                "adapter_format and adapter_contract_version are required.",
            )
        if not self.recording_source_identity or not self.source_evidence_identity:
            raise NormalizedRecordingError(
                "invalid_recording_identity",
                "recording_source_identity and source_evidence_identity are required.",
            )
        if not self.sessions:
            raise NormalizedRecordingError(
                "no_sessions", "At least one normalized session is required."
            )
        positions = [item.chronological_position for item in self.sessions]
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            raise NormalizedRecordingError(
                "invalid_session_order",
                "Session chronological positions must be unique and ascending.",
            )
        identities = [item.stable_source_identity for item in self.sessions]
        if len(set(identities)) != len(identities):
            raise NormalizedRecordingError(
                "duplicate_session_identity", "Duplicate session stable_source_identity."
            )
        excluded = [
            item for item in self.sessions if item.disposition == SESSION_DISPOSITION_EXCLUDED
        ]
        if len(excluded) > 1:
            raise NormalizedRecordingError(
                "multiple_excluded_sessions",
                "At most one session may have disposition 'excluded'.",
            )
        if excluded and excluded[0].chronological_position != positions[-1]:
            raise NormalizedRecordingError(
                "excluded_session_not_final",
                "The excluded session must be the chronologically final session.",
                stable_source_identity=excluded[0].stable_source_identity,
            )
        if not self.roi_channels:
            raise NormalizedRecordingError(
                "no_roi_channels", "At least one ROI/channel entry is required."
            )
        roi_ids = [item.roi_id for item in self.roi_channels]
        if len(set(roi_ids)) != len(roi_ids):
            raise NormalizedRecordingError(
                "duplicate_roi", "Duplicate ROI id in roi_channels."
            )
        if not any(item.included for item in self.roi_channels):
            raise NormalizedRecordingError(
                "no_included_roi", "At least one ROI must be included."
            )


def _session_payload(session: NormalizedSourceSession) -> dict[str, Any]:
    return {
        "stable_source_identity": session.stable_source_identity,
        "canonical_source_reference": session.canonical_source_reference,
        "chronological_position": session.chronological_position,
        "authoritative_source_start_time": session.authoritative_source_start_time,
        "source_timing_evidence": session.source_timing_evidence,
        "expected_timeline_start_time": session.expected_timeline_start_time,
        "expected_duration_sec": session.expected_duration_sec,
        "observed_duration_sec": session.observed_duration_sec,
        "disposition": session.disposition,
        "size_bytes": session.size_bytes,
        "content_digest": session.content_digest,
    }


def _roi_payload(roi: NormalizedRoiChannel) -> dict[str, Any]:
    return {
        "roi_id": roi.roi_id,
        "included": roi.included,
        "signal_channel_identity": roi.signal_channel_identity,
        "reference_channel_identity": roi.reference_channel_identity,
    }


def build_normalized_recording_description_payload(
    description: NormalizedRecordingDescription,
) -> dict[str, Any]:
    """The canonical, JSON-encodable payload the identity digest is computed
    over. Exposed separately from the digest for testability."""
    if not isinstance(description, NormalizedRecordingDescription):
        raise NormalizedRecordingError(
            "invalid_description",
            "description must be a NormalizedRecordingDescription.",
        )
    sampling_payload = {
        "time_basis": description.sampling.time_basis,
        "parser_contract_identity": description.sampling.parser_contract_identity,
        "sessions_per_hour": description.sampling.sessions_per_hour,
        "session_duration_sec": description.sampling.session_duration_sec,
    }
    if description.sampling.target_fs_hz is not None:
        sampling_payload["target_fs_hz"] = description.sampling.target_fs_hz
    return {
        "schema_name": description.schema_name,
        "schema_version": description.schema_version,
        "adapter_format": description.adapter_format,
        "adapter_contract_version": description.adapter_contract_version,
        "acquisition_mode": description.acquisition_mode,
        "timeline_anchor_mode": description.timeline_anchor_mode,
        "recording_source_identity": description.recording_source_identity,
        "source_evidence_identity": description.source_evidence_identity,
        "sessions": [_session_payload(item) for item in description.sessions],
        "roi_channels": [_roi_payload(item) for item in description.roi_channels],
        "sampling": sampling_payload,
        "authorized_time_column_candidates": list(description.authorized_time_column_candidates),
        "authorized_uv_suffix_candidates": list(description.authorized_uv_suffix_candidates),
        "authorized_signal_suffix_candidates": list(
            description.authorized_signal_suffix_candidates
        ),
    }


def compute_normalized_recording_description_identity(
    description: NormalizedRecordingDescription,
) -> str:
    """Deterministic identity: construction order, object identity, and
    ``adapter_evidence`` content do not affect it -- only the normalized
    scientific facts and the adapter's own identity do (see the module
    docstring for the Option A identity rule)."""
    payload = build_normalized_recording_description_payload(description)
    payload_bytes = encode_canonical_value(payload)
    domain = NORMALIZED_RECORDING_IDENTITY_DOMAIN.encode("utf-8")
    return hashlib.sha256(domain + b"\x00" + payload_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Canonical serialization (persistence boundary)
# ---------------------------------------------------------------------------


def serialize_normalized_recording_description(
    description: NormalizedRecordingDescription,
) -> dict[str, Any]:
    """The full, persistable representation: the identity-bearing payload
    plus ``adapter_evidence`` and the pre-computed identity itself, so a
    reader can verify content without recomputing from scratch first."""
    payload = build_normalized_recording_description_payload(description)
    payload["adapter_evidence"] = dict(description.adapter_evidence)
    payload["normalized_recording_description_identity"] = (
        compute_normalized_recording_description_identity(description)
    )
    return payload


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise NormalizedRecordingError(
            "malformed_serialized_field", f"{key!r} is required and must be a non-empty string."
        )
    return value


def _optional_str(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise NormalizedRecordingError(
            "malformed_serialized_field", f"{key!r} must be a string or null."
        )
    return value


def _optional_float(mapping: Mapping[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizedRecordingError(
            "malformed_serialized_field", f"{key!r} must be a number or null."
        )
    return float(value)


def _str_tuple(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key, ())
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise NormalizedRecordingError(
            "malformed_serialized_field",
            f"{key!r} must be a list of non-empty strings.",
        )
    return tuple(value)


def deserialize_normalized_recording_description(
    payload: Any,
) -> NormalizedRecordingDescription:
    """The inverse of ``serialize_normalized_recording_description``.

    Fails closed: an unknown schema version, a missing required field, a
    malformed field, or a stored identity that does not match the
    reconstructed content all raise ``NormalizedRecordingError`` rather
    than silently proceeding or guessing.
    """
    if not isinstance(payload, Mapping):
        raise NormalizedRecordingError(
            "malformed_serialized_recording", "Serialized recording payload must be an object."
        )
    schema_name = payload.get("schema_name")
    if schema_name != NORMALIZED_RECORDING_SCHEMA_NAME:
        raise NormalizedRecordingError(
            "unsupported_schema", "Unsupported normalized recording schema name."
        )
    schema_version = payload.get("schema_version")
    if schema_version != NORMALIZED_RECORDING_SCHEMA_VERSION:
        raise NormalizedRecordingError(
            "unsupported_schema_version",
            f"Unsupported normalized recording schema version: {schema_version!r}.",
        )
    try:
        raw_sessions = payload["sessions"]
        raw_roi_channels = payload["roi_channels"]
        raw_sampling = payload["sampling"]
        stored_identity = _required_str(
            payload, "normalized_recording_description_identity"
        )
        sessions = tuple(
            NormalizedSourceSession(
                stable_source_identity=_required_str(item, "stable_source_identity"),
                canonical_source_reference=_required_str(
                    item, "canonical_source_reference"
                ),
                chronological_position=int(item["chronological_position"]),
                authoritative_source_start_time=_required_str(
                    item, "authoritative_source_start_time"
                ),
                source_timing_evidence=_required_str(item, "source_timing_evidence"),
                expected_timeline_start_time=_optional_str(
                    item, "expected_timeline_start_time"
                ),
                expected_duration_sec=_optional_float(item, "expected_duration_sec"),
                observed_duration_sec=_optional_float(item, "observed_duration_sec"),
                disposition=_required_str(item, "disposition"),
                size_bytes=int(item["size_bytes"]),
                content_digest=_required_str(item, "content_digest"),
            )
            for item in raw_sessions
        )
        roi_channels = tuple(
            NormalizedRoiChannel(
                roi_id=_required_str(item, "roi_id"),
                included=bool(item["included"]),
                signal_channel_identity=_required_str(item, "signal_channel_identity"),
                reference_channel_identity=_required_str(
                    item, "reference_channel_identity"
                ),
            )
            for item in raw_roi_channels
        )
        sampling = NormalizedSamplingContract(
            time_basis=_required_str(raw_sampling, "time_basis"),
            parser_contract_identity=_required_str(
                raw_sampling, "parser_contract_identity"
            ),
            sessions_per_hour=(
                int(raw_sampling["sessions_per_hour"])
                if raw_sampling.get("sessions_per_hour") is not None
                else None
            ),
            session_duration_sec=_optional_float(raw_sampling, "session_duration_sec"),
            target_fs_hz=_optional_float(raw_sampling, "target_fs_hz"),
        )
        description = NormalizedRecordingDescription(
            schema_name=schema_name,
            schema_version=schema_version,
            adapter_format=_required_str(payload, "adapter_format"),
            adapter_contract_version=_required_str(payload, "adapter_contract_version"),
            acquisition_mode=_required_str(payload, "acquisition_mode"),
            timeline_anchor_mode=_required_str(payload, "timeline_anchor_mode"),
            recording_source_identity=_required_str(payload, "recording_source_identity"),
            source_evidence_identity=_required_str(payload, "source_evidence_identity"),
            sessions=sessions,
            roi_channels=roi_channels,
            sampling=sampling,
            authorized_time_column_candidates=_str_tuple(
                payload, "authorized_time_column_candidates"
            ),
            authorized_uv_suffix_candidates=_str_tuple(
                payload, "authorized_uv_suffix_candidates"
            ),
            authorized_signal_suffix_candidates=_str_tuple(
                payload, "authorized_signal_suffix_candidates"
            ),
            adapter_evidence=dict(payload.get("adapter_evidence") or {}),
        )
    except NormalizedRecordingError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise NormalizedRecordingError(
            "malformed_serialized_recording",
            f"Serialized recording payload is malformed: {exc}",
        ) from exc

    recomputed_identity = compute_normalized_recording_description_identity(description)
    if recomputed_identity != stored_identity:
        raise NormalizedRecordingError(
            "serialized_identity_mismatch",
            "The stored recording identity does not match its own persisted content.",
        )
    return description


# ---------------------------------------------------------------------------
# RWD adapter
# ---------------------------------------------------------------------------


def build_rwd_normalized_recording_description(
    *,
    source_root_canonical: str,
    candidate_snapshot: Any,
    session_duration_sec: float,
    sessions_per_hour: int,
    timeline_anchor_mode: str,
    acquisition_mode: str,
    discovered_roi_ids: tuple[str, ...],
    included_roi_ids: tuple[str, ...],
    rwd_time_col: str,
    uv_suffix: str,
    sig_suffix: str,
    parser_contract_digest: str,
    target_fs_hz: float | None = None,
    missing_canonical_relative_paths: tuple[str, ...] = (),
    excluded_canonical_relative_path: str | None = None,
) -> NormalizedRecordingDescription:
    """Translate already-validated RWD facts into the shared model.

    ``candidate_snapshot`` must be a
    ``io.rwd_source_snapshot.GuidedRwdSourceCandidateSnapshot`` (or
    structurally equivalent object exposing ``.candidates`` in
    authoritative chronological order plus the two candidate-set/content
    digests) -- i.e. the object A2's chronology already produced. This
    function does not re-discover, re-sort, or re-hash the source; it only
    maps already-authoritative facts into normalized vocabulary.

    ``excluded_canonical_relative_path``, when given, must identify the
    same session the accepted production final-exclusion rule identifies
    (the chronologically final candidate -- see
    ``guided_execution_payloads.py``'s ``candidate_files[-1]`` rule). A
    path that does not match any discovered candidate, or that is not the
    chronologically final one, fails closed rather than silently being
    ignored or applied to the wrong session.
    """
    if (
        session_duration_sec is None
        or isinstance(session_duration_sec, bool)
        or float(session_duration_sec) <= 0
    ):
        raise NormalizedRecordingError(
            "invalid_session_duration", "session_duration_sec must be positive."
        )
    if not discovered_roi_ids or not included_roi_ids:
        raise NormalizedRecordingError(
            "invalid_roi_scope", "discovered_roi_ids and included_roi_ids are required."
        )
    if not parser_contract_digest:
        raise NormalizedRecordingError(
            "invalid_parser_contract_digest", "parser_contract_digest is required."
        )
    if target_fs_hz is not None and (
        isinstance(target_fs_hz, bool)
        or float(target_fs_hz) <= 0
    ):
        raise NormalizedRecordingError(
            "invalid_target_sampling_rate", "target_fs_hz must be positive when supplied."
        )

    candidates = tuple(candidate_snapshot.candidates)
    all_paths = {item.canonical_relative_path for item in candidates}
    missing_set = set(missing_canonical_relative_paths)
    unknown_missing = missing_set - all_paths
    if unknown_missing:
        raise NormalizedRecordingError(
            "unmatched_missing_session_identity",
            "An approved missing session does not match a discovered candidate.",
            unmatched=tuple(sorted(unknown_missing)),
        )
    if excluded_canonical_relative_path is not None:
        if excluded_canonical_relative_path not in all_paths:
            raise NormalizedRecordingError(
                "unmatched_excluded_session_identity",
                "The excluded final session does not match a discovered candidate.",
                excluded_canonical_relative_path=excluded_canonical_relative_path,
            )
        if candidates[-1].canonical_relative_path != excluded_canonical_relative_path:
            raise NormalizedRecordingError(
                "excluded_session_not_final",
                "The excluded session identity is not the chronologically final "
                "discovered session.",
                excluded_canonical_relative_path=excluded_canonical_relative_path,
            )
        if excluded_canonical_relative_path in missing_set:
            raise NormalizedRecordingError(
                "session_both_missing_and_excluded",
                "A session cannot be both an approved missing session and the "
                "excluded final session.",
                canonical_relative_path=excluded_canonical_relative_path,
            )

    included_set = set(included_roi_ids)
    cadence_sec = 3600.0 / float(sessions_per_hour) if sessions_per_hour else None
    anchor_start: datetime | None = None
    sessions: list[NormalizedSourceSession] = []
    for position, candidate in enumerate(candidates):
        source_path = os.path.join(
            source_root_canonical, *candidate.canonical_relative_path.split("/")
        )
        start_dt = resolve_session_start_time(source_path)
        if start_dt is None:
            # The A2 chronology check already refuses an unparseable
            # session folder name before a candidate snapshot can exist;
            # this is defensive, not a normally reachable path.
            raise NormalizedRecordingError(
                "unresolvable_session_time",
                "A recording session's acquisition time could not be established.",
                canonical_relative_path=candidate.canonical_relative_path,
            )
        if position == 0:
            anchor_start = start_dt
        expected_timeline_start: str | None = None
        if anchor_start is not None and cadence_sec is not None:
            expected_timeline_start = (
                anchor_start + timedelta(seconds=position * cadence_sec)
            ).isoformat()

        if candidate.canonical_relative_path == excluded_canonical_relative_path:
            disposition = SESSION_DISPOSITION_EXCLUDED
        elif candidate.canonical_relative_path in missing_set:
            disposition = SESSION_DISPOSITION_MISSING
        else:
            disposition = SESSION_DISPOSITION_PROCESS
        sessions.append(
            NormalizedSourceSession(
                stable_source_identity=candidate.canonical_relative_path,
                canonical_source_reference=source_path,
                chronological_position=position,
                authoritative_source_start_time=start_dt.isoformat(),
                source_timing_evidence=RWD_FOLDER_TIMESTAMP_EVIDENCE,
                expected_timeline_start_time=expected_timeline_start,
                expected_duration_sec=float(session_duration_sec),
                observed_duration_sec=None,
                disposition=disposition,
                size_bytes=candidate.size_bytes,
                content_digest=candidate.sha256_content_digest,
            )
        )

    roi_channels = tuple(
        NormalizedRoiChannel(
            roi_id=roi_id,
            included=roi_id in included_set,
            signal_channel_identity=f"{roi_id}{sig_suffix}",
            reference_channel_identity=f"{roi_id}{uv_suffix}",
        )
        for roi_id in discovered_roi_ids
    )

    authorized_time_column_candidates = rwd_authorized_time_column_candidates(rwd_time_col)
    authorized_uv_suffix_candidates, authorized_signal_suffix_candidates = (
        rwd_authorized_suffix_candidates(uv_suffix, sig_suffix)
    )

    return NormalizedRecordingDescription(
        schema_name=NORMALIZED_RECORDING_SCHEMA_NAME,
        schema_version=NORMALIZED_RECORDING_SCHEMA_VERSION,
        adapter_format="rwd",
        adapter_contract_version=RWD_ADAPTER_CONTRACT_VERSION,
        acquisition_mode=acquisition_mode,
        timeline_anchor_mode=timeline_anchor_mode,
        recording_source_identity=source_root_canonical,
        source_evidence_identity=candidate_snapshot.source_candidate_content_digest,
        sessions=tuple(sessions),
        roi_channels=roi_channels,
        sampling=NormalizedSamplingContract(
            time_basis="relative_seconds_since_session_start",
            parser_contract_identity=parser_contract_digest,
            sessions_per_hour=sessions_per_hour,
            session_duration_sec=float(session_duration_sec),
            target_fs_hz=(float(target_fs_hz) if target_fs_hz is not None else None),
        ),
        authorized_time_column_candidates=authorized_time_column_candidates,
        authorized_uv_suffix_candidates=authorized_uv_suffix_candidates,
        authorized_signal_suffix_candidates=authorized_signal_suffix_candidates,
        adapter_evidence={
            "rwd_time_col": rwd_time_col,
            "uv_suffix": uv_suffix,
            "sig_suffix": sig_suffix,
            "source_candidate_set_digest": candidate_snapshot.source_candidate_set_digest,
        },
    )


def rebuild_normalized_recording_description_from_intent(
    intent: Any,
) -> NormalizedRecordingDescription:
    """Rebuild a NormalizedRecordingDescription from an authorized
    GuidedProductionExecutionIntent's frozen granular fields.

    ``intent`` is expected to be a
    ``guided_production_mapping.GuidedProductionExecutionIntent`` (typed as
    ``Any`` here to avoid importing that module, which itself imports this
    one). Every field this reads was already frozen onto the intent at
    authorization time, so this performs zero filesystem I/O -- it is a
    pure re-derivation, not a second source scan. Callers verify the
    result by recomputing its identity with
    ``compute_normalized_recording_description_identity`` and comparing
    against ``intent.normalized_recording_description_identity``; this
    function only rebuilds, it does not itself verify.

    Shared by ``guided_execution_payloads.derive_guided_execution_payloads``
    (verify-by-rebuild at authorization time) and
    ``guided_startup_transaction.plan_guided_startup_transaction``
    (verify-by-rebuild at startup-materialization time) so the exact same
    rebuild logic is never hand-copied twice.
    """
    from types import SimpleNamespace

    excluded_path = (
        intent.input_source.candidate_files[-1].canonical_relative_path
        if intent.acquisition.exclude_incomplete_final_rwd_chunk
        and intent.input_source.candidate_files
        else None
    )
    target_fs_hz = next(
        (
            item.value
            for item in intent.acquisition.semantic_values
            if item.field_name == "target_fs_hz"
        ),
        None,
    )
    return build_rwd_normalized_recording_description(
        source_root_canonical=intent.input_source.source_root_canonical,
        candidate_snapshot=SimpleNamespace(
            candidates=intent.input_source.candidate_files,
            source_candidate_set_digest=(
                intent.input_source.source_candidate_set_digest
            ),
            source_candidate_content_digest=(
                intent.input_source.source_candidate_content_digest
            ),
        ),
        session_duration_sec=intent.acquisition.session_duration_sec,
        sessions_per_hour=intent.acquisition.sessions_per_hour,
        timeline_anchor_mode=intent.acquisition.timeline_anchor_mode,
        acquisition_mode=intent.acquisition.acquisition_mode,
        discovered_roi_ids=intent.roi_scope.discovered_roi_ids,
        included_roi_ids=intent.roi_scope.included_roi_ids,
        rwd_time_col=intent.acquisition.rwd_time_col,
        uv_suffix=intent.acquisition.uv_suffix,
        sig_suffix=intent.acquisition.sig_suffix,
        parser_contract_digest=intent.parser.parser_contract_digest,
        target_fs_hz=target_fs_hz,
        missing_canonical_relative_paths=tuple(
            item.canonical_relative_path
            for item in intent.input_source.approved_missing_candidates
        ),
        excluded_canonical_relative_path=excluded_path,
    )
