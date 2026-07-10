"""Input-processing completeness record and accounting (4J16k41 / C8).

Every chunk admitted by preflight into the frozen production input set must reach
exactly one explicit terminal disposition:

- ``process``            -- an admitted chunk the run must process successfully;
- ``authorized_exclusion`` -- deliberately excluded by a narrow, user-authorized
  policy (today: the incomplete final RWD chunk, and only that);
- failed                 -- a processing failure, which terminates the run.

There are four explicit session outcomes: ``process`` (processed/valid),
``authorized_missing_corrupted`` (scientist-approved missing/corrupted session
preserved in its time slot), ``authorized_exclusion`` (the separately tracked
approved final incomplete-RWD exclusion), and unresolved failure.  The latter
raises :class:`InputProcessingError` and stops the run; it is never turned into a
warning-and-continue that omits the chunk from outputs.

The record this module builds (``input_processing_completeness.json``) is written
fail-closed by the Pipeline and bound into the terminal-completion contract, so a
run whose processed set does not exactly reconcile with its admitted set cannot
be finalized or reloaded as successful.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

INPUT_COMPLETENESS_FILENAME = "input_processing_completeness.json"
INPUT_COMPLETENESS_CONTRACT_VERSION = "input_processing_completeness.v1"

# One run-wide frozen production input manifest, written by the wrapper before
# any analysis subprocess launches and consumed identically by every one of
# them, so phasic and tonic cannot analyze two different chunk sets (4J16k41b).
FROZEN_INPUT_MANIFEST_FILENAME = "input_manifest.json"
FROZEN_INPUT_MANIFEST_CONTRACT_VERSION = "frozen_input_manifest.v1"

# The only authorized automatic exclusion policy.
POLICY_INCOMPLETE_FINAL_RWD_CHUNK = "incomplete_final_rwd_chunk"

DISPOSITION_PROCESS = "process"
DISPOSITION_AUTHORIZED_EXCLUSION = "authorized_exclusion"
# A scientist-approved corrupted/missing session that stays in its chronological
# slot as an explicit missing interval -- never removed, never zero (4J16k41c).
DISPOSITION_AUTHORIZED_MISSING = "authorized_missing_corrupted"

_TERMINAL_DISPOSITIONS = frozenset(
    {DISPOSITION_PROCESS, DISPOSITION_AUTHORIZED_EXCLUSION, DISPOSITION_AUTHORIZED_MISSING}
)

# Above this size a per-chunk digest would add minutes to a long run; record size
# only and say so, exactly as the terminal artifact contract does.
DIGEST_MAX_BYTES = 64 * 1024 * 1024
DIGEST_OMITTED_LARGE = "large_source_size_verified_only"


# A session's own validated acquisition timestamp. The RWD session folder name
# is the canonical production case. Generic datetime-token parsing remains
# available to legacy plotting/layout code, but it is not sufficient to
# authorize a missing NPM/custom session.
_SESSION_FOLDER_PATTERN = re.compile(r"(\d{4})[_-](\d{2})[_-](\d{2})-(\d{2})[_:](\d{2})[_:](\d{2})")
_LOOSE_DATETIME_PATTERNS = [
    re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})[-_ T](\d{2})[_:](\d{2})[_:](\d{2})"),
    re.compile(r"(\d{4})(\d{2})(\d{2})[-_T](\d{2})(\d{2})(\d{2})"),
]


def resolve_session_start_time(source: str) -> datetime | None:
    """Validated acquisition start time for a session, from its own identity.

    Prefers the canonical RWD session folder token ``YYYY_MM_DD-HH_MM_SS``; falls
    back to a datetime token elsewhere in the path for compatibility with
    existing layout/provenance readers.  The missing-session authorization gate
    separately requires ``input_format == 'rwd'``.
    """
    norm = str(source).replace("\\", "/")
    folder = norm.rsplit("/", 2)[-2] if "/" in norm else norm
    match = _SESSION_FOLDER_PATTERN.search(folder) or _SESSION_FOLDER_PATTERN.search(norm)
    if not match:
        for pattern in _LOOSE_DATETIME_PATTERNS:
            match = pattern.search(os.path.basename(norm)) or pattern.search(norm)
            if match:
                break
    if not match:
        return None
    try:
        return datetime(*[int(part) for part in match.groups()])
    except (ValueError, TypeError):
        return None


class InputProcessingError(RuntimeError):
    """An admitted chunk could not be processed, so the run must fail.

    Carries enough identity for an internal log while exposing a plain
    scientist-facing sentence that names the affected recording segment without
    leaking parser internals, stack traces, or schema detail.
    """

    def __init__(
        self,
        *,
        chunk_index: int | None,
        source: str,
        phase: str,
        category: str,
        reason: str,
    ):
        self.chunk_index = chunk_index
        self.source = source
        self.phase = phase
        self.category = category
        self.reason = reason
        super().__init__(
            f"input_processing_failed[{category}] "
            f"chunk_index={chunk_index} phase={phase} source={source!r}: {reason}"
        )

    def scientist_message(self) -> str:
        where = (
            f"recording segment {self.chunk_index + 1}"
            if isinstance(self.chunk_index, int)
            else "a recording segment"
        )
        label = os.path.basename(os.path.dirname(str(self.source))) or os.path.basename(
            str(self.source)
        )
        return (
            "Analysis stopped because part of the recording could not be processed. "
            f"The affected part is {where}"
            + (f" ({label})" if label else "")
            + ". No results were produced for this run; the recording was left unchanged."
        )


# ----------------------------------------------------------------------
# Source identity
# ----------------------------------------------------------------------


def _normalize_source(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: str) -> dict[str, Any]:
    """Stable identity for one source file: normalized path, size, and digest.

    Raises FileNotFoundError if the source is absent, so a frozen manifest cannot
    be built over a file that already does not exist.
    """
    norm = _normalize_source(path)
    size = os.path.getsize(norm)
    identity: dict[str, Any] = {
        "source": norm,
        "size_bytes": int(size),
        "sha256": None,
    }
    if size <= DIGEST_MAX_BYTES:
        identity["sha256"] = _sha256_file(norm)
    else:
        identity["digest_omitted_reason"] = DIGEST_OMITTED_LARGE
    return identity


def expected_entries_digest(expected: list[dict[str, Any]]) -> str:
    """Stable identity for an ordered admitted set.

    Depends only on the ordered (source, size, digest, disposition, policy) tuple
    of each chunk, so any change in ordered paths, sizes/identities, dispositions,
    or the authorized exclusion changes the digest.
    """
    canonical = [
        [
            str(entry.get("source", "")),
            int(entry.get("size_bytes", -1)),
            str(entry.get("sha256") or ""),
            str(entry.get("disposition", "")),
            str(entry.get("policy", "")),
            str(entry.get("expected_start_time", "")),
            "" if entry.get("expected_duration_sec") is None else str(entry.get("expected_duration_sec")),
        ]
        for entry in expected
    ]
    blob = json.dumps(canonical, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def source_drift_reason(entry: dict[str, Any], *, size_only: bool = False) -> str:
    """Compare a frozen expected entry against the file on disk now. "" when unchanged.

    ``size_only`` is the cheap guard used on every chunk load (missing/resized);
    the full form additionally re-hashes to catch a same-size content swap and is
    reserved for bounded checks.
    """
    norm = _normalize_source(str(entry.get("source", "")))
    if not os.path.isfile(norm):
        return "the source file is missing"
    recorded_size = entry.get("size_bytes")
    if isinstance(recorded_size, int) and os.path.getsize(norm) != recorded_size:
        return "the source file changed size since it was validated"
    if size_only:
        return ""
    recorded_digest = entry.get("sha256")
    if isinstance(recorded_digest, str) and recorded_digest:
        if _sha256_file(norm) != recorded_digest:
            return "the source file contents changed since it was validated"
    return ""


# ----------------------------------------------------------------------
# Writer-side accountant
# ----------------------------------------------------------------------


@dataclass
class _ExpectedChunk:
    """One expected session and its full session-index entry.

    ``data`` is the authoritative per-session record: source identity, chronological
    index, disposition, timing, and (for missing sessions) failure/authorization
    metadata. It round-trips verbatim into the session index and the completeness
    record, so nothing about a session -- including its time slot -- is lost.
    """

    index: int
    data: dict[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.data.get("disposition", ""))

    @property
    def policy(self) -> str:
        return str(self.data.get("policy", ""))

    @property
    def identity(self) -> dict[str, Any]:
        return self.data

    @property
    def source(self) -> str:
        return str(self.data.get("source", ""))


@dataclass
class InputProcessingAccountant:
    """Tracks one terminal disposition per admitted chunk, fail-closed.

    Built once from the frozen admitted set. The Pipeline calls
    :meth:`before_load` at each chunk (source-drift guard), :meth:`mark_processed`
    when a chunk finishes, and :meth:`fail` on any processing exception (which
    raises). :meth:`finalize` refuses to produce a record unless every
    process-disposition chunk has exactly one successful processing record.
    """

    acquisition_mode: str
    input_format: str
    frozen_manifest_digest: str = ""
    _expected: list[_ExpectedChunk] = field(default_factory=list)
    _processed: dict[int, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_admitted_manifest(
        cls,
        *,
        acquisition_mode: str,
        input_format: str,
        ordered_sources: list[str],
        excluded_source: str | None = None,
        exclusion_policy: str = "",
    ) -> "InputProcessingAccountant":
        """Freeze the admitted set into per-chunk expected records.

        ``ordered_sources`` is the full discovered, chronologically ordered set
        (admitted-for-processing chunks in order). ``excluded_source``, when
        given, is the single authorized exclusion and must be the final entry.
        """
        accountant = cls(acquisition_mode=acquisition_mode, input_format=input_format)
        excluded_norm = _normalize_source(excluded_source) if excluded_source else None
        for index, source in enumerate(ordered_sources):
            identity = source_identity(source)
            is_excluded = excluded_norm is not None and identity["source"] == excluded_norm
            data = dict(identity)
            data["index"] = index
            data["disposition"] = (
                DISPOSITION_AUTHORIZED_EXCLUSION if is_excluded else DISPOSITION_PROCESS
            )
            if is_excluded:
                data["policy"] = exclusion_policy
            accountant._expected.append(_ExpectedChunk(index=index, data=data))
        accountant.frozen_manifest_digest = expected_entries_digest(
            accountant._expected_payload()
        )
        return accountant

    @classmethod
    def from_frozen_manifest(cls, manifest: dict[str, Any]) -> "InputProcessingAccountant":
        """Build an accountant that consumes a supplied run-wide frozen manifest.

        The admitted set (including approved missing sessions) and its digest come
        from the manifest, not from an independent rediscovery, so every analysis
        in the run is held to the same expected sessions. The Pipeline separately
        verifies its own discovered set matches this manifest.
        """
        accountant = cls(
            acquisition_mode=str(manifest.get("acquisition_mode", "")),
            input_format=str(manifest.get("input_format", "")),
        )
        for entry in manifest.get("expected", []):
            accountant._expected.append(
                _ExpectedChunk(index=int(entry["index"]), data=dict(entry))
            )
        accountant.frozen_manifest_digest = str(manifest.get("digest", ""))
        return accountant

    def _expected_payload(self) -> list[dict[str, Any]]:
        return [dict(entry.data) for entry in self._expected]

    def ordered_process_sources(self) -> list[str]:
        """Normalized sources the run must load and process, in admitted order."""
        return [
            entry.source
            for entry in self._expected
            if entry.disposition == DISPOSITION_PROCESS
        ]

    def missing_sessions(self) -> list[dict[str, Any]]:
        """Authorized missing/corrupted sessions, in chronological order."""
        return [
            dict(entry.data)
            for entry in self._expected
            if entry.disposition == DISPOSITION_AUTHORIZED_MISSING
        ]

    def session_index_by_source(self) -> dict[str, int]:
        """Map each normalized source to its authoritative chronological session index."""
        return {entry.source: entry.index for entry in self._expected}

    def _entry_for_source(self, source: str) -> _ExpectedChunk | None:
        norm = _normalize_source(source)
        for entry in self._expected:
            if entry.source == norm:
                return entry
        return None

    def before_load(self, source: str, *, phase: str) -> None:
        """Fail closed if an admitted source drifted from its frozen identity."""
        entry = self._entry_for_source(source)
        if entry is None:
            raise InputProcessingError(
                chunk_index=None,
                source=source,
                phase=phase,
                category="unexpected_source",
                reason="a source not in the frozen admitted input set was reached",
            )
        drift = source_drift_reason(entry.data, size_only=True)
        if drift:
            raise InputProcessingError(
                chunk_index=entry.index,
                source=source,
                phase=phase,
                category="source_drift",
                reason=drift,
            )

    def mark_processed(self, source: str, *, cache_chunk_id: int) -> None:
        """Record one successful processing of an admitted chunk, exactly once.

        Raises if the chunk was already processed (no silent overwrite), if the
        source does not match the expected source for its index, if the chunk is
        an authorized exclusion, or if the cache contribution id is not a unique
        integer. Two admitted chunks can never claim the same cache slot.
        """
        entry = self._entry_for_source(source)
        if entry is None:
            raise InputProcessingError(
                chunk_index=None,
                source=source,
                phase="processing",
                category="unexpected_source",
                reason="processed a source not in the frozen admitted input set",
            )
        if entry.disposition != DISPOSITION_PROCESS:
            raise InputProcessingError(
                chunk_index=entry.index,
                source=source,
                phase="processing",
                category="excluded_chunk_processed",
                reason="an authorized-excluded chunk was processed",
            )
        if entry.index in self._processed:
            raise InputProcessingError(
                chunk_index=entry.index,
                source=source,
                phase="processing",
                category="duplicate_processing",
                reason="an admitted chunk was processed more than once",
            )
        if _normalize_source(source) != entry.source:
            raise InputProcessingError(
                chunk_index=entry.index,
                source=source,
                phase="processing",
                category="source_identity_mismatch",
                reason="the processed source does not match the expected source for its index",
            )
        if not isinstance(cache_chunk_id, int) or isinstance(cache_chunk_id, bool):
            raise InputProcessingError(
                chunk_index=entry.index,
                source=source,
                phase="processing",
                category="invalid_cache_chunk_id",
                reason="the cache contribution id is not an integer",
            )
        for existing in self._processed.values():
            if existing["cache_chunk_id"] == int(cache_chunk_id):
                raise InputProcessingError(
                    chunk_index=entry.index,
                    source=source,
                    phase="processing",
                    category="duplicate_cache_chunk_id",
                    reason="two admitted chunks claim the same cache contribution",
                )
        self._processed[entry.index] = {
            "index": entry.index,
            "source": entry.source,
            "cache_chunk_id": int(cache_chunk_id),
        }

    def fail(self, *, source: str, phase: str, category: str, reason: str) -> "InputProcessingError":
        entry = self._entry_for_source(source)
        return InputProcessingError(
            chunk_index=entry.index if entry else None,
            source=source,
            phase=phase,
            category=category,
            reason=reason,
        )

    def finalize(self) -> dict[str, Any]:
        """Build the completeness record, or raise if the admitted set is not fully processed."""
        for entry in self._expected:
            if entry.disposition == DISPOSITION_PROCESS and entry.index not in self._processed:
                raise InputProcessingError(
                    chunk_index=entry.index,
                    source=entry.source,
                    phase="finalize",
                    category="unprocessed_admitted_chunk",
                    reason="an admitted chunk never reached a successful processing record",
                )

        return {
            "contract_version": INPUT_COMPLETENESS_CONTRACT_VERSION,
            "acquisition_mode": self.acquisition_mode,
            "input_format": self.input_format,
            "frozen_manifest_digest": self.frozen_manifest_digest,
            "expected": self._expected_payload(),
            "processed": [self._processed[i] for i in sorted(self._processed)],
            "missing": self.missing_sessions(),
        }

    def write(self, output_dir: str) -> str:
        payload = self.finalize()
        path = os.path.join(output_dir, INPUT_COMPLETENESS_FILENAME)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
        return path


# ----------------------------------------------------------------------
# Run-wide frozen input manifest
# ----------------------------------------------------------------------


def build_session_index(
    *,
    acquisition_mode: str,
    input_format: str,
    ordered_sources: list[str],
    missing_sources: list[str] | tuple[str, ...] = (),
    missing_metadata: dict[str, dict[str, Any]] | None = None,
    excluded_source: str | None = None,
    exclusion_policy: str = "",
    expected_duration_sec: float | None = None,
) -> dict[str, Any]:
    """Freeze the run-wide session index covering every expected session.

    ``ordered_sources`` is the FULL chronological discovered set: process, approved
    missing, and the single final exclusion (which must be last). Every session
    keeps its chronological index and validated timing, so an approved missing
    session stays in its slot rather than being removed.

    Raises if an approved missing session has no validated own-identity timestamp
    -- such a session cannot be placed safely and cannot be authorized missing.
    """
    excluded_norm = _normalize_source(excluded_source) if excluded_source else None
    missing_norm = {_normalize_source(s) for s in missing_sources}
    missing_metadata = {
        _normalize_source(k): v for k, v in (missing_metadata or {}).items()
    }

    # A time-preserving missing interval is a scientific claim about acquisition
    # chronology, not a generic filename convenience.  Only timestamped RWD
    # session folders are validated for this release.  NPM/custom paths may
    # still run normally; they simply cannot authorize a corrupted session as a
    # missing chronological interval until a format-specific timing validator
    # exists.
    if missing_norm and str(input_format).strip().lower() != "rwd":
        raise InputProcessingError(
            chunk_index=None,
            source=sorted(missing_norm)[0],
            phase="freeze",
            category="unsupported_missing_session_timing",
            reason=(
                "approved missing-session continuation is supported only for "
                "timestamped intermittent RWD session folders"
            ),
        )

    expected: list[dict[str, Any]] = []
    for index, source in enumerate(ordered_sources):
        norm = _normalize_source(source)
        start_dt = resolve_session_start_time(source)
        entry: dict[str, Any] = {
            "index": index,
            "expected_start_time": start_dt.isoformat() if start_dt else "",
            "expected_duration_sec": (
                float(expected_duration_sec) if expected_duration_sec is not None else None
            ),
        }

        if excluded_norm is not None and norm == excluded_norm:
            entry.update(source_identity(source))
            entry["disposition"] = DISPOSITION_AUTHORIZED_EXCLUSION
            entry["policy"] = exclusion_policy or POLICY_INCOMPLETE_FINAL_RWD_CHUNK
        elif norm in missing_norm:
            if start_dt is None:
                raise InputProcessingError(
                    chunk_index=index,
                    source=source,
                    phase="freeze",
                    category="unresolvable_missing_session_time",
                    reason=(
                        "an approved missing session has no validated acquisition "
                        "timestamp of its own; its chronology cannot be established"
                    ),
                )
            # A missing/corrupted source is deliberately not opened; record its
            # on-disk identity when present, but never fabricate contents.
            if os.path.isfile(norm):
                entry.update(source_identity(source))
            else:
                entry["source"] = norm
                entry["size_bytes"] = -1
                entry["sha256"] = None
            entry["disposition"] = DISPOSITION_AUTHORIZED_MISSING
            meta = missing_metadata.get(norm, {})
            entry["failure_category"] = str(meta.get("failure_category", "corrupted_session"))
            entry["reason"] = str(meta.get("reason", "approved corrupted/missing session"))
            entry["authorization_source"] = str(meta.get("authorization_source", "run_config"))
        else:
            entry.update(source_identity(source))
            entry["disposition"] = DISPOSITION_PROCESS

        expected.append(entry)

    return {
        "contract_version": FROZEN_INPUT_MANIFEST_CONTRACT_VERSION,
        "acquisition_mode": str(acquisition_mode),
        "input_format": str(input_format),
        "expected": expected,
        "digest": expected_entries_digest(expected),
    }


def build_frozen_input_manifest(
    *,
    acquisition_mode: str,
    input_format: str,
    ordered_sources: list[str],
    excluded_source: str | None = None,
    exclusion_policy: str = "",
    expected_duration_sec: float | None = None,
) -> dict[str, Any]:
    """Freeze a session index with no approved missing sessions (compat shim)."""
    return build_session_index(
        acquisition_mode=acquisition_mode,
        input_format=input_format,
        ordered_sources=ordered_sources,
        excluded_source=excluded_source,
        exclusion_policy=exclusion_policy,
        expected_duration_sec=expected_duration_sec,
    )


def write_frozen_input_manifest(run_dir: str, manifest: dict[str, Any]) -> str:
    path = os.path.join(run_dir, FROZEN_INPUT_MANIFEST_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(tmp, path)
    return path


def load_frozen_input_manifest(path: str) -> tuple[dict[str, Any] | None, str]:
    """Load and structurally validate a frozen input manifest."""
    if not os.path.isfile(path):
        return None, "missing"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        return None, f"malformed ({exc})"
    if not isinstance(manifest, dict):
        return None, "root is not a JSON object"
    if manifest.get("contract_version") != FROZEN_INPUT_MANIFEST_CONTRACT_VERSION:
        return None, "unsupported frozen-manifest version"
    expected = manifest.get("expected")
    if not isinstance(expected, list) or not expected:
        return None, "frozen manifest lists no admitted chunks"
    recomputed = expected_entries_digest(expected)
    if manifest.get("digest") != recomputed:
        return None, "frozen manifest digest does not match its admitted set"
    return manifest, ""


# ----------------------------------------------------------------------
# Reader-side validation
# ----------------------------------------------------------------------


def validate_input_completeness(payload: Any) -> str:
    """Validate a completeness record's internal accounting. "" when sound.

    Checks: supported version; a contiguous, unique, ordered expected set; at most
    one authorized exclusion and it is the final chronological chunk; every
    process-disposition chunk processed exactly once, by its own expected source
    and a unique cache contribution; no duplicate or stray processed record;
    processed + authorized_exclusions == expected.
    """
    if not isinstance(payload, dict):
        return "the input-completeness record is not an object"
    if payload.get("contract_version") != INPUT_COMPLETENESS_CONTRACT_VERSION:
        return "the input-completeness record declares an unsupported version"

    expected = payload.get("expected")
    if not isinstance(expected, list) or not expected:
        return "the input-completeness record lists no admitted chunks"

    process_indices: set[int] = set()
    exclusion_indices: list[int] = []
    missing_indices: set[int] = set()
    seen_indices: set[int] = set()
    source_by_index: dict[int, str] = {}
    for entry in expected:
        if not isinstance(entry, dict):
            return "the input-completeness record contains an unreadable chunk entry"
        index = entry.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            return "an admitted chunk entry has no integer index"
        if index in seen_indices:
            return f"admitted chunk index {index} is duplicated"
        seen_indices.add(index)
        source_by_index[index] = _normalize_source(str(entry.get("source", "")))
        disposition = entry.get("disposition")
        if disposition == DISPOSITION_PROCESS:
            process_indices.add(index)
        elif disposition == DISPOSITION_AUTHORIZED_EXCLUSION:
            if not str(entry.get("policy", "")).strip():
                return f"admitted chunk index {index} is excluded without a policy"
            exclusion_indices.append(index)
        elif disposition == DISPOSITION_AUTHORIZED_MISSING:
            # An approved missing session must keep its chronological slot with a
            # validated timestamp and an explicit authorization -- never zero,
            # never silently dropped.
            if not str(entry.get("expected_start_time", "")).strip():
                return f"missing session index {index} has no validated timestamp"
            if not str(entry.get("authorization_source", "")).strip():
                return f"missing session index {index} has no authorization source"
            missing_indices.add(index)
        else:
            return f"admitted chunk index {index} has no valid disposition"

    n = len(expected)
    if seen_indices != set(range(n)):
        return "admitted chunk indices are not a contiguous 0..N-1 range"

    if len(exclusion_indices) > 1:
        return "more than one authorized exclusion is recorded"
    if exclusion_indices and exclusion_indices[0] != n - 1:
        return "an authorized exclusion is not the final chronological chunk"

    processed = payload.get("processed")
    if not isinstance(processed, list):
        return "the input-completeness record has no processed list"

    processed_indices: set[int] = set()
    cache_chunk_ids: set[int] = set()
    for record in processed:
        if not isinstance(record, dict):
            return "a processed record is unreadable"
        index = record.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            return "a processed record has no integer index"
        if index in processed_indices:
            return f"processed chunk index {index} is recorded more than once"
        processed_indices.add(index)
        if index not in process_indices:
            return f"processed chunk index {index} was not an admitted, non-excluded chunk"
        # The processed source must be the expected source for that index -- a
        # correct index carrying another chunk's source is a swap, not a match.
        if _normalize_source(str(record.get("source", ""))) != source_by_index.get(index):
            return f"processed chunk index {index} records a different source than admitted"
        cache_id = record.get("cache_chunk_id")
        if not isinstance(cache_id, int) or isinstance(cache_id, bool):
            return f"processed chunk index {index} has no integer cache contribution id"
        if cache_id in cache_chunk_ids:
            return f"cache contribution id {cache_id} is claimed by two admitted chunks"
        cache_chunk_ids.add(cache_id)

    unprocessed = process_indices - processed_indices
    if unprocessed:
        return f"admitted chunks were never processed: {sorted(unprocessed)}"

    # The explicit missing list must exactly mirror the missing dispositions, so a
    # consumer that reads only the missing list still sees every gap in place.
    missing_list = payload.get("missing", [])
    if not isinstance(missing_list, list):
        return "the input-completeness record has an unreadable missing list"
    listed_missing = set()
    for record in missing_list:
        if not isinstance(record, dict) or not isinstance(record.get("index"), int):
            return "a missing-session record is unreadable"
        listed_missing.add(record["index"])
    if listed_missing != missing_indices:
        return "the missing-session list does not match the missing dispositions"

    return ""


def read_input_completeness(analysis_out_dir: str) -> tuple[dict[str, Any] | None, str]:
    """Read and validate the record in an analysis output directory."""
    path = os.path.join(analysis_out_dir, INPUT_COMPLETENESS_FILENAME)
    if not os.path.isfile(path):
        return None, "missing"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        return None, f"malformed ({exc})"
    error = validate_input_completeness(payload)
    if error:
        return None, error
    return payload, ""
