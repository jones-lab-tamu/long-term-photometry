"""Shared, format-neutral normalized recording description (B1).

Format-specific adapters (currently: RWD and the authorized intermittent NPM
adapter) populate one shared model.
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

Custom-tabular adapters remain out of scope for this milestone;
``SUPPORTED_ADAPTER_FORMATS`` is the single place a future adapter would
register itself.

Identity rule (resolves the A2-noted ambiguity): this schema uses
**implementation-bound identity** (Option A). The canonical identity
includes ``adapter_format`` and ``adapter_contract_version`` alongside the
normalized scientific facts, as one combined digest -- matching the
existing A1 (``guided_plan_identity``) pattern of one combined digest for
"what was validated," and the existing startup-preflight pattern of exact
digest matching. Two different adapter formats/versions therefore produce
different identities even for scientifically equivalent facts, by design.
For RWD, ``adapter_evidence`` (raw, format-specific provenance, e.g. which
folder token supplied a timestamp) remains excluded from the identity. For
NPM, the adapter's validated per-session resolved-evidence projection is
included because timestamp-column resolution, physical ROI mapping, support
geometry, and output time basis are execution-defining facts. The remaining
raw NPM evidence envelope remains diagnostic. If cross-adapter equivalence is
ever needed, that requires a separate, explicitly-scoped identity -- it is
not implied here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import os
from types import MappingProxyType
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
NPM_ADAPTER_CONTRACT_VERSION = "npm_normalized_recording_adapter.v1"
NPM_PARSER_CONTRACT_SCHEMA_NAME = "npm_normalized_parser_contract"
NPM_PARSER_CONTRACT_SCHEMA_VERSION = "v1"
NPM_PARSER_CONTRACT_DIGEST_DOMAIN = "npm-normalized-parser-contract:v1"
NPM_OUTPUT_TIME_BASIS = "relative_seconds_since_uv_signal_overlap_origin"
NPM_SESSION_RESOLVED_EVIDENCE_SCHEMA_VERSION = "npm_session_resolved_evidence.v1"
NPM_SESSION_RESOLVED_EVIDENCE_IDENTITY_DOMAIN = (
    "npm-session-resolved-evidence:v1"
)

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
SUPPORTED_ADAPTER_FORMATS = ("rwd", "npm")


def _canonical_parser_content(value: Any) -> Any:
    """Copy parser policy into recursively immutable canonical containers."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not value == value:
            raise NormalizedRecordingError(
                "parser_contract_content_invalid",
                "NPM parser policy cannot contain non-finite numbers.",
            )
        if isinstance(value, float) and value in (float("inf"), float("-inf")):
            raise NormalizedRecordingError(
                "parser_contract_content_invalid",
                "NPM parser policy cannot contain non-finite numbers.",
            )
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise NormalizedRecordingError(
                "parser_contract_content_invalid",
                "NPM parser policy object keys must be strings.",
            )
        return MappingProxyType(
            {key: _canonical_parser_content(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_parser_content(item) for item in value)
    raise NormalizedRecordingError(
        "parser_contract_content_invalid",
        "NPM parser policy contains an unsupported value type.",
        received_type=type(value).__name__,
    )


def _thaw_parser_content(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_parser_content(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_parser_content(item) for item in value]
    return value


def compute_npm_parser_contract_digest(content: Mapping[str, Any]) -> str:
    """Digest the complete, canonical NPM parser policy content."""
    if not isinstance(content, Mapping):
        raise NormalizedRecordingError(
            "parser_contract_content_invalid",
            "NPM parser policy content must be an object.",
        )
    encoded = encode_canonical_value(_thaw_parser_content(content))
    return hashlib.sha256(
        NPM_PARSER_CONTRACT_DIGEST_DOMAIN.encode("utf-8")
        + b"\x00"
        + encoded
    ).hexdigest()


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
    source_column: str | None = None

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
    # NPM's complete historical parser policy is part of the normalized
    # sampling contract.  RWD descriptions intentionally leave this absent
    # so their serialized bytes and identity remain backward-compatible.
    parser_contract_content: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.time_basis:
            raise NormalizedRecordingError("invalid_time_basis", "time_basis is required.")
        if not self.parser_contract_identity:
            raise NormalizedRecordingError(
                "invalid_parser_contract_identity",
                "parser_contract_identity is required.",
            )
        if self.parser_contract_content is not None:
            if not isinstance(self.parser_contract_content, Mapping):
                raise NormalizedRecordingError(
                    "parser_contract_content_invalid",
                    "NPM parser policy content must be an object.",
                )
            content = _canonical_parser_content(self.parser_contract_content)
            if content.get("schema_name") != NPM_PARSER_CONTRACT_SCHEMA_NAME:
                raise NormalizedRecordingError(
                    "unsupported_parser_policy_schema",
                    "Unsupported NPM parser policy schema.",
                )
            if content.get("schema_version") != NPM_PARSER_CONTRACT_SCHEMA_VERSION:
                raise NormalizedRecordingError(
                    "unsupported_parser_policy_version",
                    "Unsupported NPM parser policy version.",
                )
            if not isinstance(content.get("sampling"), Mapping):
                raise NormalizedRecordingError(
                    "parser_contract_content_invalid",
                    "NPM parser policy must contain a sampling object.",
                )
            expected_digest = compute_npm_parser_contract_digest(content)
            if self.parser_contract_identity != expected_digest:
                raise NormalizedRecordingError(
                    "parser_contract_content_digest_mismatch",
                    "NPM parser policy content does not match its digest.",
                    expected_digest=expected_digest,
                    received_digest=self.parser_contract_identity,
                )
            object.__setattr__(self, "parser_contract_content", content)


@dataclass(frozen=True)
class NormalizedRecordingDescription:
    """The one shared, format-neutral recording-description model.

    See the module docstring for the identity rule (Option A,
    implementation-bound). RWD adapter evidence is diagnostic and excluded;
    NPM's validated per-session resolved-evidence projection is identity-
    bearing while the remaining adapter evidence is retained for diagnostics.
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
        if self.adapter_format == "npm":
            parser_sampling = self.sampling.parser_contract_content.get(
                "sampling"
            ) if isinstance(self.sampling.parser_contract_content, Mapping) else None
            authorized_led_column = (
                parser_sampling.get("led_column")
                if isinstance(parser_sampling, Mapping)
                else None
            )
            authorized_led_values = (
                parser_sampling.get("led_values")
                if isinstance(parser_sampling, Mapping)
                else None
            )
            if (
                not isinstance(authorized_led_column, str)
                or not authorized_led_column
                or not isinstance(authorized_led_values, Mapping)
                or isinstance(authorized_led_values.get("uv"), bool)
                or not isinstance(authorized_led_values.get("uv"), int)
                or isinstance(authorized_led_values.get("signal"), bool)
                or not isinstance(authorized_led_values.get("signal"), int)
            ):
                raise NormalizedRecordingError(
                    "invalid_channel_pairing",
                    "NPM parser policy does not contain authorized LED selector facts.",
                )
            raw_mapping = self.adapter_evidence.get(
                "physical_to_canonical_roi_mapping"
            )
            if not isinstance(raw_mapping, (list, tuple)) or not raw_mapping:
                raise NormalizedRecordingError(
                    "npm_physical_roi_mapping_missing",
                    "NPM normalized recording is missing its frozen physical ROI mapping.",
                )
            frozen_mapping: list[tuple[str, str]] = []
            for item in raw_mapping:
                if not isinstance(item, Mapping):
                    raise NormalizedRecordingError(
                        "npm_physical_roi_mapping_invalid",
                        "NPM physical ROI mapping entries must be objects.",
                    )
                canonical_roi_id = item.get("canonical_roi_id")
                physical_source_column = item.get("physical_source_column")
                if (
                    not isinstance(canonical_roi_id, str)
                    or not canonical_roi_id
                    or not isinstance(physical_source_column, str)
                    or not physical_source_column
                ):
                    raise NormalizedRecordingError(
                        "npm_physical_roi_mapping_invalid",
                        "NPM physical ROI mapping entries must identify both ROI forms.",
                    )
                frozen_mapping.append(
                    (canonical_roi_id, physical_source_column)
                )
            if (
                len({canonical for canonical, _physical in frozen_mapping})
                != len(frozen_mapping)
                or len({physical for _canonical, physical in frozen_mapping})
                != len(frozen_mapping)
                or tuple(canonical for canonical, _physical in frozen_mapping)
                != tuple(channel.roi_id for channel in self.roi_channels)
            ):
                raise NormalizedRecordingError(
                    "npm_physical_roi_mapping_mismatch",
                    "NPM frozen physical ROI mapping does not match the canonical ROI order.",
                )
            mapping_by_roi = dict(frozen_mapping)
            for channel in self.roi_channels:
                try:
                    signal_source, signal_column, signal_value = _parse_npm_channel_identity(
                        channel.signal_channel_identity
                    )
                    reference_source, reference_column, reference_value = _parse_npm_channel_identity(
                        channel.reference_channel_identity
                    )
                except (TypeError, ValueError, KeyError) as exc:
                    raise NormalizedRecordingError(
                        "invalid_channel_pairing",
                        "NPM channel identity is not canonical JSON with the v1 schema.",
                        roi_id=channel.roi_id,
                    ) from exc
                if mapping_by_roi.get(channel.roi_id) != signal_source:
                    raise NormalizedRecordingError(
                        "npm_physical_roi_mapping_mismatch",
                        "NPM channel identity does not match its frozen physical ROI mapping.",
                        roi_id=channel.roi_id,
                    )
                if (
                    signal_value != authorized_led_values["signal"]
                    or reference_value != authorized_led_values["uv"]
                    or signal_column != authorized_led_column
                    or reference_column != authorized_led_column
                    or signal_column != reference_column
                    or signal_source != reference_source
                    or not channel.source_column
                    or channel.source_column != signal_source
                ):
                    raise NormalizedRecordingError(
                        "invalid_channel_pairing",
                        "NPM channel identity does not match its frozen physical source mapping.",
                        roi_id=channel.roi_id,
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
    payload = {
        "roi_id": roi.roi_id,
        "included": roi.included,
        "signal_channel_identity": roi.signal_channel_identity,
        "reference_channel_identity": roi.reference_channel_identity,
    }
    if roi.source_column is not None:
        payload["source_column"] = roi.source_column
    return payload


def _npm_session_resolved_evidence_projection(
    description: NormalizedRecordingDescription,
) -> list[dict[str, Any]]:
    """Return the NPM facts resolved independently for each source session.

    These facts are execution-defining for NPM: the authorized timestamp
    column, physical ROI inventory/mapping, support geometry, and output time
    basis must remain bound to the exact session order.  They are kept in the
    format-specific evidence envelope for compatibility, and this validated
    projection is additionally included in the NPM identity payload.
    """
    if description.adapter_format != "npm":
        return []

    raw_sessions = description.adapter_evidence.get("npm_sessions")
    if not isinstance(raw_sessions, (list, tuple)):
        raise NormalizedRecordingError(
            "npm_per_session_evidence_not_identity_bound",
            "NPM per-session resolved evidence is missing.",
        )
    if len(raw_sessions) != len(description.sessions):
        raise NormalizedRecordingError(
            "npm_per_session_evidence_not_identity_bound",
            "NPM per-session resolved evidence does not cover every session.",
        )

    raw_mapping = description.adapter_evidence.get(
        "physical_to_canonical_roi_mapping"
    )
    if not isinstance(raw_mapping, (list, tuple)) or not raw_mapping:
        raise NormalizedRecordingError(
            "npm_per_session_evidence_not_identity_bound",
            "NPM physical-to-canonical ROI mapping is missing.",
        )
    global_mapping: list[dict[str, str]] = []
    physical_columns: list[str] = []
    canonical_roi_ids: list[str] = []
    for item in raw_mapping:
        if not isinstance(item, Mapping):
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM physical-to-canonical ROI mapping is malformed.",
            )
        canonical_roi_id = item.get("canonical_roi_id")
        physical_source_column = item.get("physical_source_column")
        if (
            not isinstance(canonical_roi_id, str)
            or not canonical_roi_id
            or not isinstance(physical_source_column, str)
            or not physical_source_column
        ):
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM physical-to-canonical ROI mapping is incomplete.",
            )
        canonical_roi_ids.append(canonical_roi_id)
        physical_columns.append(physical_source_column)
        global_mapping.append(
            {
                "canonical_roi_id": canonical_roi_id,
                "physical_source_column": physical_source_column,
            }
        )
    if (
        len(set(canonical_roi_ids)) != len(canonical_roi_ids)
        or len(set(physical_columns)) != len(physical_columns)
        or tuple(canonical_roi_ids)
        != tuple(channel.roi_id for channel in description.roi_channels)
    ):
        raise NormalizedRecordingError(
            "npm_per_session_evidence_not_identity_bound",
            "NPM physical-to-canonical ROI mapping is not one-to-one.",
        )

    required_fields = (
        "canonical_relative_path",
        "resolved_timestamp_column",
        "timestamp_unit",
        "physical_roi_inventory",
        "physical_to_canonical_roi_mapping",
        "support_policy",
        "support_policy_identity",
        "overlap_origin_absolute",
        "resolved_support_start_offset_sec",
        "resolved_support_end_offset_sec",
        "resolved_support_start_absolute",
        "resolved_support_end_absolute",
        "output_time_basis",
    )
    projection: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    identity_mapping: list[dict[str, str]] | None = None
    identity_physical_columns: tuple[str, ...] | None = None
    for position, (session, raw) in enumerate(zip(description.sessions, raw_sessions)):
        if not isinstance(raw, Mapping):
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM per-session resolved evidence entries must be objects.",
                chronological_position=position,
            )
        if any(field_name not in raw for field_name in required_fields):
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM per-session resolved evidence is incomplete.",
                chronological_position=position,
            )
        path = raw.get("canonical_relative_path")
        if not isinstance(path, str) or not path or path in seen_paths:
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM per-session evidence paths must be unique and non-empty.",
                chronological_position=position,
            )
        seen_paths.add(path)
        if path != session.stable_source_identity:
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM per-session evidence order does not match normalized sessions.",
                chronological_position=position,
            )

        session_mapping = raw.get("physical_to_canonical_roi_mapping")
        if not isinstance(session_mapping, (list, tuple)) or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("canonical_roi_id"), str)
            or not item.get("canonical_roi_id")
            or not isinstance(item.get("physical_source_column"), str)
            or not item.get("physical_source_column")
            for item in session_mapping
        ):
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM per-session physical ROI mapping is missing or malformed.",
                chronological_position=position,
            )
        normalized_session_mapping = [
            {
                "canonical_roi_id": item["canonical_roi_id"],
                "physical_source_column": item["physical_source_column"],
            }
            for item in session_mapping
        ]
        if identity_mapping is None:
            identity_mapping = normalized_session_mapping
            identity_physical_columns = tuple(
                item["physical_source_column"] for item in identity_mapping
            )
        if normalized_session_mapping != identity_mapping:
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM per-session physical ROI mapping is inconsistent.",
                chronological_position=position,
            )
        inventory = raw.get("physical_roi_inventory")
        if not isinstance(inventory, (list, tuple)) or tuple(inventory) != identity_physical_columns:
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM physical ROI inventory is missing or inconsistent.",
                chronological_position=position,
            )
        support_policy = raw.get("support_policy")
        support_policy_identity = raw.get("support_policy_identity")
        if (
            not isinstance(support_policy, str)
            or not support_policy
            or not isinstance(support_policy_identity, str)
            or not support_policy_identity
        ):
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM support policy identity is missing.",
                chronological_position=position,
            )
        output_time_basis = raw.get("output_time_basis")
        if not isinstance(output_time_basis, str) or not output_time_basis:
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM output time basis is missing.",
                chronological_position=position,
            )

        projection.append(
            {
                "schema_version": NPM_SESSION_RESOLVED_EVIDENCE_SCHEMA_VERSION,
                "chronological_position": position,
                "canonical_relative_path": path,
                "resolved_timestamp_column": raw["resolved_timestamp_column"],
                "timestamp_unit": raw["timestamp_unit"],
                "physical_roi_inventory": list(identity_physical_columns or ()),
                "physical_to_canonical_roi_mapping": [
                    dict(item) for item in (identity_mapping or ())
                ],
                "support_policy": support_policy,
                "support_policy_identity": support_policy_identity,
                "overlap_origin_absolute": raw["overlap_origin_absolute"],
                "resolved_support_start_offset_sec": raw[
                    "resolved_support_start_offset_sec"
                ],
                "resolved_support_end_offset_sec": raw[
                    "resolved_support_end_offset_sec"
                ],
                "resolved_support_start_absolute": raw[
                    "resolved_support_start_absolute"
                ],
                "resolved_support_end_absolute": raw["resolved_support_end_absolute"],
                "output_time_basis": output_time_basis,
            }
        )
    return projection


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
    if description.sampling.parser_contract_content is not None:
        sampling_payload["parser_contract_content"] = _thaw_parser_content(
            description.sampling.parser_contract_content
        )
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
        **(
            {
                "npm_per_session_resolved_evidence": (
                    _npm_session_resolved_evidence_projection(description)
                ),
                "npm_physical_to_canonical_roi_mapping": [
                    dict(item)
                    for item in description.adapter_evidence.get(
                        "physical_to_canonical_roi_mapping", ()
                    )
                ],
            }
            if description.adapter_format == "npm"
            else {}
        ),
    }


def compute_normalized_recording_description_identity(
    description: NormalizedRecordingDescription,
) -> str:
    """Deterministic identity: construction order, object identity, and
    ``adapter_evidence`` content do not affect it for RWD. For NPM, only the
    validated per-session resolved-evidence projection affects it alongside
    the normalized scientific facts and adapter identity."""
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
                source_column=_optional_str(item, "source_column"),
            )
            for item in raw_roi_channels
        )
        parser_contract_content = raw_sampling.get("parser_contract_content")
        if parser_contract_content is None and payload.get("adapter_format") == "npm":
            raise NormalizedRecordingError(
                "parser_contract_content_required",
                "NPM normalized descriptions require parser policy content.",
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
            parser_contract_content=parser_contract_content,
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
    if description.adapter_format == "npm":
        persisted_projection = payload.get("npm_per_session_resolved_evidence")
        expected_projection = _npm_session_resolved_evidence_projection(description)
        if persisted_projection != expected_projection:
            raise NormalizedRecordingError(
                "serialized_npm_session_evidence_mismatch",
                "Persisted NPM per-session evidence does not match its adapter evidence.",
            )
        if payload.get("npm_physical_to_canonical_roi_mapping") != description.adapter_evidence.get(
            "physical_to_canonical_roi_mapping"
        ):
            raise NormalizedRecordingError(
                "serialized_npm_mapping_mismatch",
                "Persisted NPM physical ROI mapping does not match its adapter evidence.",
            )
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


def _npm_channel_identity(
    *, source_column: str, led_value: int, selector_column: str
) -> str:
    payload = {
        "selector": {
            "column": selector_column,
            "operator": "eq",
            "value": led_value,
        },
        "source_column": source_column,
    }
    return "npm-channel:v1:" + json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _parse_npm_channel_identity(identity: str) -> tuple[str, str, int]:
    prefix = "npm-channel:v1:"
    if not isinstance(identity, str) or not identity.startswith(prefix):
        raise ValueError("NPM channel identity has an unsupported schema/version.")
    payload = json.loads(identity[len(prefix) :])
    if not isinstance(payload, dict) or set(payload) != {"selector", "source_column"}:
        raise ValueError("NPM channel identity payload has an unsupported shape.")
    selector = payload.get("selector")
    source_column = payload.get("source_column")
    if (
        not isinstance(selector, dict)
        or set(selector) != {"column", "operator", "value"}
        or selector.get("operator") != "eq"
        or not isinstance(selector.get("column"), str)
        or not selector.get("column")
        or isinstance(selector.get("value"), bool)
        or not isinstance(selector.get("value"), int)
        or not isinstance(source_column, str)
        or not source_column
    ):
        raise ValueError("NPM channel identity selector is invalid.")
    expected = _npm_channel_identity(
        source_column=source_column,
        led_value=int(selector["value"]),
        selector_column=selector["column"],
    )
    if identity != expected:
        raise ValueError("NPM channel identity JSON is not canonical.")
    return source_column, selector["column"], int(selector["value"])


def build_npm_normalized_recording_description(
    *,
    source_snapshot: Any,
    session_inspections: Mapping[str, Any],
    parser_contract_content: Mapping[str, Any],
    session_duration_sec: float,
    sessions_per_hour: int,
    acquisition_mode: str = "intermittent",
    timeline_anchor_mode: str = "civil",
    discovered_roi_ids: tuple[str, ...],
    included_roi_ids: tuple[str, ...],
    target_fs_hz: float,
    physical_to_canonical_roi_mapping: tuple[tuple[str, str], ...] | None = None,
) -> NormalizedRecordingDescription:
    """Build the authorized NPM normalized description from frozen facts.

    ``source_snapshot`` owns filename chronology and content identity;
    ``session_inspections`` owns parser-resolved support/timing facts.  This
    function performs no discovery and does not reread the source files.
    """
    if not discovered_roi_ids or not included_roi_ids:
        raise NormalizedRecordingError(
            "invalid_roi_scope", "NPM discovered and included ROI ids are required."
        )
    if sessions_per_hour <= 0 or session_duration_sec <= 0 or target_fs_hz <= 0:
        raise NormalizedRecordingError(
            "invalid_sampling_contract",
            "NPM session cadence, duration, and target sampling rate must be positive.",
        )
    content = _canonical_parser_content(parser_contract_content)
    if not isinstance(content, Mapping):
        raise NormalizedRecordingError(
            "parser_contract_content_invalid", "NPM parser policy must be an object."
        )
    sampling_content = content.get("sampling")
    if not isinstance(sampling_content, Mapping):
        raise NormalizedRecordingError(
            "parser_contract_content_invalid",
            "NPM parser policy must contain sampling selector facts.",
        )
    led_column = sampling_content.get("led_column")
    led_values = sampling_content.get("led_values")
    if (
        not isinstance(led_column, str)
        or not led_column
        or not isinstance(led_values, Mapping)
        or isinstance(led_values.get("uv"), bool)
        or not isinstance(led_values.get("uv"), int)
        or isinstance(led_values.get("signal"), bool)
        or not isinstance(led_values.get("signal"), int)
    ):
        raise NormalizedRecordingError(
            "parser_contract_content_invalid",
            "NPM parser policy must contain authorized LED selector facts.",
        )
    candidates = tuple(source_snapshot.candidates)
    if not candidates:
        raise NormalizedRecordingError("no_sessions", "NPM source snapshot is empty.")

    first_inspection = session_inspections.get(candidates[0].canonical_relative_path)
    if first_inspection is None:
        raise NormalizedRecordingError(
            "missing_session_inspection",
            "NPM parser inspection facts are missing for the first source session.",
        )
    frozen_mapping = tuple(
        physical_to_canonical_roi_mapping
        if physical_to_canonical_roi_mapping is not None
        else getattr(first_inspection, "physical_to_canonical_roi_mapping", ())
    )
    if not frozen_mapping or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not all(isinstance(value, str) and value for value in item)
        for item in frozen_mapping
    ):
        raise NormalizedRecordingError(
            "npm_physical_roi_mapping_missing",
            "NPM normalized construction requires an authoritative physical ROI mapping.",
        )
    mapped_roi_ids = tuple(canonical for canonical, _physical in frozen_mapping)
    if mapped_roi_ids != tuple(discovered_roi_ids):
        raise NormalizedRecordingError(
            "npm_physical_roi_mapping_mismatch",
            "The authoritative NPM physical ROI mapping disagrees with the discovered ROI scope.",
        )
    physical_columns = tuple(physical for _canonical, physical in frozen_mapping)
    if len(set(mapped_roi_ids)) != len(mapped_roi_ids) or len(set(physical_columns)) != len(physical_columns):
        raise NormalizedRecordingError(
            "npm_physical_roi_mapping_invalid",
            "The authoritative NPM physical ROI mapping must be one-to-one.",
        )

    cadence_sec = 3600.0 / float(sessions_per_hour)
    anchor_start = candidates[0].authoritative_source_start_time
    try:
        anchor_dt = datetime.fromisoformat(anchor_start)
    except (TypeError, ValueError) as exc:
        raise NormalizedRecordingError(
            "unresolvable_session_time",
            "NPM filename chronology contains an invalid timestamp.",
        ) from exc

    sessions: list[NormalizedSourceSession] = []
    adapter_session_evidence: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates):
        inspection = session_inspections.get(candidate.canonical_relative_path)
        if inspection is None:
            raise NormalizedRecordingError(
                "missing_session_inspection",
                "NPM parser inspection facts are missing for a source session.",
                canonical_relative_path=candidate.canonical_relative_path,
            )
        inspection_mapping = tuple(
            getattr(inspection, "physical_to_canonical_roi_mapping", ())
        )
        if inspection_mapping != frozen_mapping:
            raise NormalizedRecordingError(
                "npm_physical_roi_mapping_mismatch",
                "NPM physical ROI mappings differ between admitted sessions.",
                canonical_relative_path=candidate.canonical_relative_path,
            )
        inspection_roi_columns = tuple(getattr(inspection, "roi_columns", ()))
        if inspection_roi_columns != physical_columns:
            raise NormalizedRecordingError(
                "npm_per_session_evidence_not_identity_bound",
                "NPM physical ROI inventory differs between admitted sessions.",
                canonical_relative_path=candidate.canonical_relative_path,
            )
        expected = (anchor_dt + timedelta(seconds=position * cadence_sec)).isoformat()
        actual = candidate.authoritative_source_start_time
        try:
            actual_elapsed_sec = (
                datetime.fromisoformat(actual) - anchor_dt
            ).total_seconds()
        except (TypeError, ValueError) as exc:
            raise NormalizedRecordingError(
                "unresolvable_session_time",
                "NPM filename chronology contains an invalid timestamp.",
            ) from exc
        sessions.append(
            NormalizedSourceSession(
                stable_source_identity=candidate.canonical_relative_path,
                canonical_source_reference=os.path.join(
                    source_snapshot.source_root_canonical,
                    *candidate.canonical_relative_path.split("/"),
                ),
                chronological_position=position,
                authoritative_source_start_time=actual,
                source_timing_evidence="npm_filename_timestamp_yyyy_mm_dd_thh_mm_ss",
                expected_timeline_start_time=expected,
                expected_duration_sec=float(session_duration_sec),
                observed_duration_sec=float(inspection.observed_duration_sec),
                disposition=SESSION_DISPOSITION_PROCESS,
                size_bytes=int(candidate.size_bytes),
                content_digest=candidate.sha256_content_digest,
            )
        )
        adapter_session_evidence.append(
            {
                "schema_version": NPM_SESSION_RESOLVED_EVIDENCE_SCHEMA_VERSION,
                "canonical_relative_path": candidate.canonical_relative_path,
                "resolved_timestamp_column": inspection.resolved_timestamp_column,
                "timestamp_unit": inspection.timestamp_unit,
                "physical_roi_inventory": list(physical_columns),
                "physical_to_canonical_roi_mapping": [
                    {
                        "canonical_roi_id": canonical_roi_id,
                        "physical_source_column": physical_source_column,
                    }
                    for canonical_roi_id, physical_source_column in frozen_mapping
                ],
                "overlap_origin_absolute": inspection.overlap_origin_absolute,
                "resolved_support_start_offset_sec": inspection.resolved_support_start_offset_sec,
                "resolved_support_end_offset_sec": inspection.resolved_support_end_offset_sec,
                "resolved_support_start_absolute": inspection.resolved_support_start_absolute,
                "resolved_support_end_absolute": inspection.resolved_support_end_absolute,
                "observed_duration_sec": inspection.observed_duration_sec,
                "output_time_basis": inspection.output_time_basis,
                "support_policy": inspection.support_policy,
                "support_policy_identity": hashlib.sha256(
                    (
                        "npm-support-policy:v1\x00"
                        + str(inspection.support_policy)
                    ).encode("utf-8")
                ).hexdigest(),
                "warning_categories": list(inspection.warning_categories),
                "actual_elapsed_sec": actual_elapsed_sec,
                "nominal_expected_elapsed_sec": position * cadence_sec,
            }
        )

    included_set = set(included_roi_ids)
    roi_channels = tuple(
        NormalizedRoiChannel(
            roi_id=canonical_roi_id,
            included=canonical_roi_id in included_set,
            signal_channel_identity=_npm_channel_identity(
                source_column=physical_source_column,
                led_value=int(led_values["signal"]),
                selector_column=led_column,
            ),
            reference_channel_identity=_npm_channel_identity(
                source_column=physical_source_column,
                led_value=int(led_values["uv"]),
                selector_column=led_column,
            ),
            source_column=physical_source_column,
        )
        for canonical_roi_id, physical_source_column in frozen_mapping
    )
    return NormalizedRecordingDescription(
        schema_name=NORMALIZED_RECORDING_SCHEMA_NAME,
        schema_version=NORMALIZED_RECORDING_SCHEMA_VERSION,
        adapter_format="npm",
        adapter_contract_version=NPM_ADAPTER_CONTRACT_VERSION,
        acquisition_mode=acquisition_mode,
        timeline_anchor_mode=timeline_anchor_mode,
        recording_source_identity=source_snapshot.source_root_canonical,
        source_evidence_identity=source_snapshot.source_candidate_content_digest,
        sessions=tuple(sessions),
        roi_channels=roi_channels,
        sampling=NormalizedSamplingContract(
            time_basis=NPM_OUTPUT_TIME_BASIS,
            parser_contract_identity=compute_npm_parser_contract_digest(content),
            sessions_per_hour=int(sessions_per_hour),
            session_duration_sec=float(session_duration_sec),
            target_fs_hz=float(target_fs_hz),
            parser_contract_content=content,
        ),
        adapter_evidence={
            "npm_source_candidate_set_digest": source_snapshot.source_candidate_set_digest,
            "npm_source_candidate_content_digest": source_snapshot.source_candidate_content_digest,
            "npm_sessions": adapter_session_evidence,
            "physical_to_canonical_roi_mapping": [
                {
                    "canonical_roi_id": canonical_roi_id,
                    "physical_source_column": physical_source_column,
                }
                for canonical_roi_id, physical_source_column in frozen_mapping
            ],
            "output_time_basis": NPM_OUTPUT_TIME_BASIS,
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
