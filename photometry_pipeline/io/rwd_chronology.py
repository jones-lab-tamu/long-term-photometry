"""Authoritative chronological ordering for intermittent RWD sessions.

This is the single ordering rule shared by production discovery
(``io.adapters.discover_rwd_chunks``, consumed by ``Pipeline.discover_files``
and the GUI preview path) and Guided validation/execution candidate
snapshots (``io.rwd_source_snapshot``, whose digests are frozen at Setup
check and re-verified at startup). Both call sites route through this
module so the sequence checked during Setup check is provably the same
sequence Pipeline execution consumes -- there is exactly one place that
decides "session A happened before session B."

The authoritative timing evidence is the RWD session folder name itself:
a fixed-width, zero-padded, naive (timezone-free) local wall-clock token
``YYYY_MM_DD-HH_MM_SS``. This is the RWD format's own documented naming
contract (see ``discover_rwd_chunks`` and the legacy synthetic-dataset
generator), not an invented convention. No other absolute-time evidence is
available inside an RWD chunk: the per-row time column is elapsed seconds
since that chunk's own acquisition start, not a wall-clock date, so there
is nothing else in the current RWD format to cross-check the folder name
against.

Ordering fails closed rather than silently falling back to filesystem
enumeration order, lexical order, or another incidental key, when a folder
name does not match this format or when two folders resolve to the exact
same timestamp.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Sequence, TypeVar

RWD_SESSION_FOLDER_TIMESTAMP_FORMAT = "%Y_%m_%d-%H_%M_%S"
RWD_CHRONOLOGY_ORDERING_RULE_VERSION = "rwd_session_folder_authoritative_timestamp.v1"

T = TypeVar("T")


class RwdChronologyError(ValueError):
    """A candidate RWD session set could not be placed into one trustworthy
    chronological order.

    ``category`` is a stable machine-readable reason
    ("malformed_session_timestamp" or "duplicate_session_timestamp") for
    callers that want to classify the failure; the exception message
    itself is written in plain, scientist-facing language.
    """

    def __init__(self, category: str, message: str, **context: object) -> None:
        self.category = str(category)
        self.context = dict(context)
        super().__init__(message)


def parse_rwd_session_folder_timestamp(folder_name: str) -> datetime | None:
    """Parse the authoritative acquisition start time from an RWD session
    folder name, or return None if it is not the canonical token.

    The full folder name must match ``YYYY_MM_DD-HH_MM_SS`` exactly -- a
    renamed or decorated folder name (an added prefix/suffix) is not
    treated as a trustworthy acquisition timestamp.
    """
    if not isinstance(folder_name, str) or not folder_name:
        return None
    try:
        return datetime.strptime(folder_name, RWD_SESSION_FOLDER_TIMESTAMP_FORMAT)
    except ValueError:
        return None


def order_rwd_session_candidates(
    candidates: Sequence[T],
    *,
    name_of: Callable[[T], str],
) -> list[T]:
    """Return `candidates` in one authoritative, deterministic chronological
    order derived from each candidate's RWD session-folder timestamp.

    Independent of input order (a genuine sort keyed on parsed timestamp),
    so filesystem enumeration order never influences the result. Raises
    ``RwdChronologyError`` instead of silently accepting an untrustworthy
    order when:

    - a candidate's folder name does not parse as the canonical RWD
      session timestamp ("malformed_session_timestamp");
    - two distinct candidates parse to the identical timestamp, which this
      RWD contract has no scientific tie-breaker to resolve
      ("duplicate_session_timestamp").
    """
    parsed: list[tuple[datetime, str, T]] = []
    malformed: list[str] = []
    for candidate in candidates:
        name = name_of(candidate)
        timestamp = parse_rwd_session_folder_timestamp(name)
        if timestamp is None:
            malformed.append(name)
            continue
        parsed.append((timestamp, name, candidate))

    if malformed:
        offenders = tuple(sorted(malformed))
        raise RwdChronologyError(
            "malformed_session_timestamp",
            "The app cannot determine when this recording session occurred "
            "because its folder name does not match the expected "
            "recording-time format: " + ", ".join(offenders),
            malformed_session_names=offenders,
        )

    by_timestamp: dict[datetime, list[str]] = {}
    for timestamp, name, _candidate in parsed:
        by_timestamp.setdefault(timestamp, []).append(name)
    duplicates = {ts: names for ts, names in by_timestamp.items() if len(names) > 1}
    if duplicates:
        first_timestamp = sorted(duplicates)[0]
        colliding = tuple(sorted(duplicates[first_timestamp]))
        raise RwdChronologyError(
            "duplicate_session_timestamp",
            "Two recording sessions appear to have the same start time: "
            + " and ".join(colliding),
            colliding_session_names=colliding,
        )

    parsed.sort(key=lambda item: item[0])
    return [candidate for _timestamp, _name, candidate in parsed]


def find_rwd_session_overlaps(
    ordered_sessions: Sequence[tuple[str, datetime]],
    *,
    session_duration_sec: float,
) -> list[tuple[str, str]]:
    """Return (earlier_name, later_name) pairs whose session intervals
    overlap.

    ``ordered_sessions`` must already be in chronological order (as
    returned by ``order_rwd_session_candidates``); this function does not
    re-sort. Each session is treated as occupying
    ``[start, start + session_duration_sec)`` using the single confirmed
    session duration -- the best validated duration evidence the current
    architecture has for intermittent RWD.
    """
    duration = timedelta(seconds=float(session_duration_sec))
    overlaps: list[tuple[str, str]] = []
    for (name_a, start_a), (name_b, start_b) in zip(
        ordered_sessions, ordered_sessions[1:]
    ):
        if start_b < start_a + duration:
            overlaps.append((name_a, name_b))
    return overlaps
