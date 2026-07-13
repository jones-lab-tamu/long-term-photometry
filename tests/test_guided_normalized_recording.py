"""Shared normalized recording-description contract (B1).

Covers the model's canonical identity rules (including the corrected,
distinct timing semantics and the resolved Option A identity rule), the
canonical serialize/deserialize boundary, the RWD adapter that translates
already-authoritative A2 source/chronology/final-exclusion facts into it,
and a fake non-RWD adapter proving the shape is genuinely format-neutral.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from photometry_pipeline.guided_normalized_recording import (
    NormalizedRecordingDescription,
    NormalizedRecordingError,
    NormalizedRoiChannel,
    NormalizedSamplingContract,
    NormalizedSourceSession,
    NORMALIZED_RECORDING_SCHEMA_NAME,
    NORMALIZED_RECORDING_SCHEMA_VERSION,
    RWD_FOLDER_TIMESTAMP_EVIDENCE,
    SESSION_DISPOSITION_EXCLUDED,
    SESSION_DISPOSITION_MISSING,
    SESSION_DISPOSITION_PROCESS,
    build_normalized_recording_description_payload,
    build_rwd_normalized_recording_description,
    compute_normalized_recording_description_identity,
    deserialize_normalized_recording_description,
    serialize_normalized_recording_description,
)
from photometry_pipeline.io.rwd_source_snapshot import (
    build_rwd_source_candidate_snapshot,
)


def _session(
    *,
    stable_source_identity="2025_01_01-00_00_00/fluorescence.csv",
    position=0,
    start="2025-01-01T00:00:00",
    timing_evidence=RWD_FOLDER_TIMESTAMP_EVIDENCE,
    expected_timeline_start=None,
    disposition=SESSION_DISPOSITION_PROCESS,
    duration=600.0,
    size=100,
    digest=None,
) -> NormalizedSourceSession:
    return NormalizedSourceSession(
        stable_source_identity=stable_source_identity,
        canonical_source_reference=f"/data/{stable_source_identity}",
        chronological_position=position,
        authoritative_source_start_time=start,
        source_timing_evidence=timing_evidence,
        expected_timeline_start_time=expected_timeline_start,
        expected_duration_sec=duration,
        observed_duration_sec=None,
        disposition=disposition,
        size_bytes=size,
        content_digest=digest or ("a" * 64),
    )


def _roi(roi_id="CH1", included=True) -> NormalizedRoiChannel:
    return NormalizedRoiChannel(
        roi_id=roi_id,
        included=included,
        signal_channel_identity=f"{roi_id}-470",
        reference_channel_identity=f"{roi_id}-410",
    )


def _description(
    *,
    sessions=None,
    roi_channels=None,
    adapter_format="rwd",
    source_evidence_identity="b" * 64,
    parser_identity="c" * 64,
    sessions_per_hour=6,
    session_duration_sec=600.0,
) -> NormalizedRecordingDescription:
    return NormalizedRecordingDescription(
        schema_name=NORMALIZED_RECORDING_SCHEMA_NAME,
        schema_version=NORMALIZED_RECORDING_SCHEMA_VERSION,
        adapter_format=adapter_format,
        adapter_contract_version="test_adapter.v1",
        acquisition_mode="intermittent",
        timeline_anchor_mode="civil",
        recording_source_identity="/data/root",
        source_evidence_identity=source_evidence_identity,
        sessions=(_session(),) if sessions is None else sessions,
        roi_channels=(_roi(),) if roi_channels is None else roi_channels,
        sampling=NormalizedSamplingContract(
            time_basis="relative_seconds_since_session_start",
            parser_contract_identity=parser_identity,
            sessions_per_hour=sessions_per_hour,
            session_duration_sec=session_duration_sec,
        ),
        adapter_evidence={"some": "evidence"},
    )


# ---------------------------------------------------------------------------
# A. Normalized model tests
# ---------------------------------------------------------------------------


def test_identity_is_deterministic():
    d = _description()
    assert (
        compute_normalized_recording_description_identity(d)
        == compute_normalized_recording_description_identity(d)
    )


def test_identity_independent_of_construction_order():
    sessions_a = (_session(stable_source_identity="a/fluorescence.csv", position=0),)
    sessions_b = tuple(sessions_a)  # separate tuple object, same content
    d1 = _description(sessions=sessions_a)
    d2 = _description(sessions=sessions_b)
    assert d1 is not d2
    assert (
        compute_normalized_recording_description_identity(d1)
        == compute_normalized_recording_description_identity(d2)
    )


def test_identity_changes_with_source_membership():
    d1 = _description(
        sessions=(_session(stable_source_identity="a/fluorescence.csv", position=0),)
    )
    d2 = _description(
        sessions=(
            _session(stable_source_identity="a/fluorescence.csv", position=0),
            _session(
                stable_source_identity="b/fluorescence.csv",
                position=1,
                start="2025-01-01T00:10:00",
            ),
        )
    )
    assert (
        compute_normalized_recording_description_identity(d1)
        != compute_normalized_recording_description_identity(d2)
    )


def test_identity_changes_with_chronology():
    a = _session(stable_source_identity="a/fluorescence.csv", position=0, start="2025-01-01T00:00:00")
    b = _session(stable_source_identity="b/fluorescence.csv", position=1, start="2025-01-01T00:10:00")
    forward = _description(sessions=(a, b))
    swapped = _description(
        sessions=(
            replace(b, chronological_position=0, authoritative_source_start_time="2025-01-01T00:00:00"),
            replace(a, chronological_position=1, authoritative_source_start_time="2025-01-01T00:10:00"),
        )
    )
    assert (
        compute_normalized_recording_description_identity(forward)
        != compute_normalized_recording_description_identity(swapped)
    )


def test_identity_changes_with_stable_source_identity():
    d1 = _description(sessions=(_session(stable_source_identity="a/fluorescence.csv"),))
    d2 = _description(sessions=(_session(stable_source_identity="a_renamed/fluorescence.csv"),))
    assert (
        compute_normalized_recording_description_identity(d1)
        != compute_normalized_recording_description_identity(d2)
    )


def test_identity_changes_with_timing_evidence():
    d1 = _description(sessions=(_session(start="2025-01-01T00:00:00"),))
    d2 = _description(sessions=(_session(start="2025-01-01T00:05:00"),))
    assert (
        compute_normalized_recording_description_identity(d1)
        != compute_normalized_recording_description_identity(d2)
    )


def test_authoritative_start_and_expected_timeline_start_remain_distinct():
    """The two concepts must not be conflated into one field, and a
    difference between them (e.g. real acquisition drift from the nominal
    schedule) must be representable and identity-relevant."""
    same_value = _session(
        start="2025-01-01T00:00:00", expected_timeline_start="2025-01-01T00:00:00"
    )
    assert (
        same_value.authoritative_source_start_time
        == same_value.expected_timeline_start_time
    )

    drifted = _session(
        start="2025-01-01T00:00:07", expected_timeline_start="2025-01-01T00:00:00"
    )
    assert (
        drifted.authoritative_source_start_time
        != drifted.expected_timeline_start_time
    )
    d1 = _description(sessions=(same_value,))
    d2 = _description(sessions=(drifted,))
    assert (
        compute_normalized_recording_description_identity(d1)
        != compute_normalized_recording_description_identity(d2)
    )


def test_identity_changes_with_roi_inventory():
    d1 = _description(roi_channels=(_roi("CH1"),))
    d2 = _description(roi_channels=(_roi("CH1"), _roi("CH2")))
    assert (
        compute_normalized_recording_description_identity(d1)
        != compute_normalized_recording_description_identity(d2)
    )


def test_identity_changes_with_channel_pairing():
    d1 = _description(roi_channels=(_roi("CH1"),))
    swapped_roi = replace(
        _roi("CH1"),
        signal_channel_identity="CH1-410",
        reference_channel_identity="CH1-470",
    )
    d2 = _description(roi_channels=(swapped_roi,))
    assert (
        compute_normalized_recording_description_identity(d1)
        != compute_normalized_recording_description_identity(d2)
    )


def test_identity_changes_with_sampling_assumptions():
    d1 = _description(sessions_per_hour=6)
    d2 = _description(sessions_per_hour=4)
    assert (
        compute_normalized_recording_description_identity(d1)
        != compute_normalized_recording_description_identity(d2)
    )


def test_identity_changes_with_missing_or_excluded_state():
    process = _description(sessions=(_session(disposition=SESSION_DISPOSITION_PROCESS),))
    missing = _description(sessions=(_session(disposition=SESSION_DISPOSITION_MISSING),))
    excluded = _description(sessions=(_session(disposition=SESSION_DISPOSITION_EXCLUDED),))
    ids = {
        compute_normalized_recording_description_identity(process),
        compute_normalized_recording_description_identity(missing),
        compute_normalized_recording_description_identity(excluded),
    }
    assert len(ids) == 3


def test_excluded_and_missing_dispositions_are_mutually_coherent():
    """A single session's disposition is one of exactly three values, so
    it can never itself be both -- but the description as a whole must
    also refuse an excluded session that isn't the true chronological
    final one, and more than one excluded session."""
    with pytest.raises(NormalizedRecordingError, match="final"):
        _description(
            sessions=(
                _session(
                    stable_source_identity="a/fluorescence.csv",
                    position=0,
                    disposition=SESSION_DISPOSITION_EXCLUDED,
                ),
                _session(
                    stable_source_identity="b/fluorescence.csv",
                    position=1,
                    start="2025-01-01T00:10:00",
                ),
            )
        )
    with pytest.raises(NormalizedRecordingError, match="one session"):
        _description(
            sessions=(
                _session(
                    stable_source_identity="a/fluorescence.csv",
                    position=0,
                    disposition=SESSION_DISPOSITION_EXCLUDED,
                ),
                _session(
                    stable_source_identity="b/fluorescence.csv",
                    position=1,
                    start="2025-01-01T00:10:00",
                    disposition=SESSION_DISPOSITION_EXCLUDED,
                ),
            )
        )


def test_adapter_format_and_version_are_part_of_the_identity():
    """Option A (implementation-bound identity, documented in the module
    docstring): two different adapter formats/versions producing
    scientifically identical facts must NOT share one identity."""
    d1 = _description(adapter_format="rwd")
    d2 = replace(d1, adapter_format="a_different_adapter")
    assert (
        compute_normalized_recording_description_identity(d1)
        != compute_normalized_recording_description_identity(d2)
    )
    d3 = replace(d1, adapter_contract_version="a_different_version")
    assert (
        compute_normalized_recording_description_identity(d1)
        != compute_normalized_recording_description_identity(d3)
    )


def test_adapter_evidence_does_not_enter_identity():
    """adapter_evidence is raw provenance, not a normalized fact or part
    of the derivation contract -- it alone is excluded from the identity
    (see the module docstring's Option A rationale)."""
    d1 = _description()
    d2 = replace(d1, adapter_evidence={"totally": "different", "evidence": 123})
    assert (
        compute_normalized_recording_description_identity(d1)
        == compute_normalized_recording_description_identity(d2)
    )


def test_required_fields_fail_closed_when_missing():
    with pytest.raises(NormalizedRecordingError, match="schema"):
        NormalizedRecordingDescription(
            schema_name="wrong",
            schema_version=NORMALIZED_RECORDING_SCHEMA_VERSION,
            adapter_format="rwd",
            adapter_contract_version="v1",
            acquisition_mode="intermittent",
            timeline_anchor_mode="civil",
            recording_source_identity="/data",
            source_evidence_identity="a" * 64,
            sessions=(_session(),),
            roi_channels=(_roi(),),
            sampling=NormalizedSamplingContract(
                time_basis="relative_seconds_since_session_start",
                parser_contract_identity="a" * 64,
                sessions_per_hour=6,
                session_duration_sec=600.0,
            ),
        )


def test_unknown_schema_version_refuses_safely():
    with pytest.raises(NormalizedRecordingError):
        NormalizedRecordingDescription(
            schema_name=NORMALIZED_RECORDING_SCHEMA_NAME,
            schema_version="v999_unknown",
            adapter_format="rwd",
            adapter_contract_version="v1",
            acquisition_mode="intermittent",
            timeline_anchor_mode="civil",
            recording_source_identity="/data",
            source_evidence_identity="a" * 64,
            sessions=(_session(),),
            roi_channels=(_roi(),),
            sampling=NormalizedSamplingContract(
                time_basis="relative_seconds_since_session_start",
                parser_contract_identity="a" * 64,
                sessions_per_hour=6,
                session_duration_sec=600.0,
            ),
        )


def test_empty_sessions_fails_closed():
    with pytest.raises(NormalizedRecordingError, match="session"):
        _description(sessions=())


def test_no_included_roi_fails_closed():
    with pytest.raises(NormalizedRecordingError, match="included"):
        _description(roi_channels=(_roi("CH1", included=False),))


def test_duplicate_chronological_position_fails_closed():
    with pytest.raises(NormalizedRecordingError):
        _description(
            sessions=(
                _session(stable_source_identity="a/fluorescence.csv", position=0),
                _session(stable_source_identity="b/fluorescence.csv", position=0),
            )
        )


def test_canonical_payload_round_trips_identity():
    d = _description()
    payload = build_normalized_recording_description_payload(d)
    # The payload is exactly what the identity is computed over; hashing
    # it again independently must reproduce the same identity.
    import hashlib

    from photometry_pipeline.guided_identity import encode_canonical_value
    from photometry_pipeline.guided_normalized_recording import (
        NORMALIZED_RECORDING_IDENTITY_DOMAIN,
    )

    recomputed = hashlib.sha256(
        NORMALIZED_RECORDING_IDENTITY_DOMAIN.encode("utf-8")
        + b"\x00"
        + encode_canonical_value(payload)
    ).hexdigest()
    assert recomputed == compute_normalized_recording_description_identity(d)
    assert "adapter_evidence" not in payload


def test_irrelevant_display_state_does_not_enter_identity():
    d1 = replace(_description(), adapter_evidence={"a": 1, "b": 2})
    d2 = replace(_description(), adapter_evidence={"b": 2, "a": 1})
    assert (
        compute_normalized_recording_description_identity(d1)
        == compute_normalized_recording_description_identity(d2)
    )


# ---------------------------------------------------------------------------
# A2. Canonical serialization round trip
# ---------------------------------------------------------------------------


def test_serialize_deserialize_round_trip_preserves_identity():
    d = _description()
    payload = serialize_normalized_recording_description(d)
    restored = deserialize_normalized_recording_description(payload)
    assert (
        compute_normalized_recording_description_identity(restored)
        == compute_normalized_recording_description_identity(d)
    )
    assert restored.adapter_evidence == dict(d.adapter_evidence)


def test_deserialize_refuses_unknown_schema_version():
    d = _description()
    payload = serialize_normalized_recording_description(d)
    payload["schema_version"] = "v999_unknown"
    with pytest.raises(NormalizedRecordingError, match="schema"):
        deserialize_normalized_recording_description(payload)


def test_deserialize_refuses_malformed_required_field():
    d = _description()
    payload = serialize_normalized_recording_description(d)
    del payload["sessions"][0]["stable_source_identity"]
    with pytest.raises(NormalizedRecordingError):
        deserialize_normalized_recording_description(payload)


def test_deserialize_refuses_tampered_content_identity_mismatch():
    """A payload whose stored identity doesn't match its own content
    (e.g. edited after the fact) must fail closed, not silently trust the
    stored identity or silently recompute and proceed."""
    d = _description()
    payload = serialize_normalized_recording_description(d)
    payload["sessions"][0]["authoritative_source_start_time"] = "2099-01-01T00:00:00"
    with pytest.raises(NormalizedRecordingError, match="identity"):
        deserialize_normalized_recording_description(payload)


def test_deserialize_refuses_non_object_payload():
    with pytest.raises(NormalizedRecordingError):
        deserialize_normalized_recording_description("not a dict")
    with pytest.raises(NormalizedRecordingError):
        deserialize_normalized_recording_description(None)


# ---------------------------------------------------------------------------
# B. RWD adapter tests
# ---------------------------------------------------------------------------


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("TimeStamp,Region0-410,Region0-470\n0,1.0,2.0\n", encoding="utf-8")


def test_rwd_adapter_translates_authoritative_order_and_identity(tmp_path: Path):
    names = ["2025_05_01-09_00_00", "2025_05_01-08_00_00", "2025_05_01-10_00_00"]
    for name in names:
        _touch(tmp_path / name / "fluorescence.csv")
    snapshot = build_rwd_source_candidate_snapshot(str(tmp_path))

    description = build_rwd_normalized_recording_description(
        source_root_canonical=snapshot.source_root_canonical,
        candidate_snapshot=snapshot,
        session_duration_sec=600.0,
        sessions_per_hour=6,
        timeline_anchor_mode="civil",
        acquisition_mode="intermittent",
        discovered_roi_ids=("Region0",),
        included_roi_ids=("Region0",),
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        parser_contract_digest="d" * 64,
    )

    # Authoritative A2 chronological order, not filesystem/name order.
    ordered_names = [s.stable_source_identity.split("/")[0] for s in description.sessions]
    assert ordered_names == sorted(names)
    assert [s.chronological_position for s in description.sessions] == [0, 1, 2]
    assert all(s.disposition == SESSION_DISPOSITION_PROCESS for s in description.sessions)
    assert all(s.authoritative_source_start_time for s in description.sessions)
    assert all(
        s.source_timing_evidence == RWD_FOLDER_TIMESTAMP_EVIDENCE
        for s in description.sessions
    )
    # First session anchors the schedule, so its expected timeline start
    # equals its own authoritative start; later ones are schedule
    # predictions from that anchor at the confirmed sessions_per_hour.
    first = description.sessions[0]
    assert first.expected_timeline_start_time == first.authoritative_source_start_time

    roi = description.roi_channels[0]
    assert roi.roi_id == "Region0"
    assert roi.signal_channel_identity == "Region0-470"
    assert roi.reference_channel_identity == "Region0-410"
    assert roi.included is True

    # Format-specific evidence retained for provenance, but not baked into
    # the shared identity payload.
    assert description.adapter_evidence["rwd_time_col"] == "TimeStamp"
    payload = build_normalized_recording_description_payload(description)
    assert "rwd_time_col" not in str(payload.keys())


def test_rwd_adapter_reflects_missing_and_excluded_disposition(tmp_path: Path):
    names = ["2025_05_02-00_00_00", "2025_05_02-00_10_00", "2025_05_02-00_20_00"]
    for name in names:
        _touch(tmp_path / name / "fluorescence.csv")
    snapshot = build_rwd_source_candidate_snapshot(str(tmp_path))
    missing_rel = "2025_05_02-00_10_00/fluorescence.csv"
    excluded_rel = "2025_05_02-00_20_00/fluorescence.csv"

    description = build_rwd_normalized_recording_description(
        source_root_canonical=snapshot.source_root_canonical,
        candidate_snapshot=snapshot,
        session_duration_sec=600.0,
        sessions_per_hour=6,
        timeline_anchor_mode="civil",
        acquisition_mode="intermittent",
        discovered_roi_ids=("Region0",),
        included_roi_ids=("Region0",),
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        parser_contract_digest="d" * 64,
        missing_canonical_relative_paths=(missing_rel,),
        excluded_canonical_relative_path=excluded_rel,
    )

    by_identity = {s.stable_source_identity: s for s in description.sessions}
    assert by_identity["2025_05_02-00_00_00/fluorescence.csv"].disposition == SESSION_DISPOSITION_PROCESS
    assert by_identity[missing_rel].disposition == SESSION_DISPOSITION_MISSING
    assert by_identity[excluded_rel].disposition == SESSION_DISPOSITION_EXCLUDED
    # Missing/excluded sessions keep their true chronological position.
    assert by_identity[missing_rel].chronological_position == 1
    assert by_identity[excluded_rel].chronological_position == 2


def test_rwd_adapter_excluded_session_must_be_true_final_session(tmp_path: Path):
    names = ["2025_05_06-00_00_00", "2025_05_06-00_10_00"]
    for name in names:
        _touch(tmp_path / name / "fluorescence.csv")
    snapshot = build_rwd_source_candidate_snapshot(str(tmp_path))
    non_final_rel = "2025_05_06-00_00_00/fluorescence.csv"
    with pytest.raises(NormalizedRecordingError, match="final"):
        build_rwd_normalized_recording_description(
            source_root_canonical=snapshot.source_root_canonical,
            candidate_snapshot=snapshot,
            session_duration_sec=600.0,
            sessions_per_hour=6,
            timeline_anchor_mode="civil",
            acquisition_mode="intermittent",
            discovered_roi_ids=("Region0",),
            included_roi_ids=("Region0",),
            rwd_time_col="TimeStamp",
            uv_suffix="-410",
            sig_suffix="-470",
            parser_contract_digest="d" * 64,
            excluded_canonical_relative_path=non_final_rel,
        )


def test_rwd_adapter_excluded_identity_must_match_a_discovered_candidate(tmp_path: Path):
    _touch(tmp_path / "2025_05_07-00_00_00" / "fluorescence.csv")
    snapshot = build_rwd_source_candidate_snapshot(str(tmp_path))
    with pytest.raises(NormalizedRecordingError, match="excluded"):
        build_rwd_normalized_recording_description(
            source_root_canonical=snapshot.source_root_canonical,
            candidate_snapshot=snapshot,
            session_duration_sec=600.0,
            sessions_per_hour=6,
            timeline_anchor_mode="civil",
            acquisition_mode="intermittent",
            discovered_roi_ids=("Region0",),
            included_roi_ids=("Region0",),
            rwd_time_col="TimeStamp",
            uv_suffix="-410",
            sig_suffix="-470",
            parser_contract_digest="d" * 64,
            excluded_canonical_relative_path="2099_01_01-00_00_00/fluorescence.csv",
        )


def test_rwd_adapter_session_cannot_be_both_missing_and_excluded(tmp_path: Path):
    _touch(tmp_path / "2025_05_08-00_00_00" / "fluorescence.csv")
    snapshot = build_rwd_source_candidate_snapshot(str(tmp_path))
    rel = "2025_05_08-00_00_00/fluorescence.csv"
    with pytest.raises(NormalizedRecordingError, match="both"):
        build_rwd_normalized_recording_description(
            source_root_canonical=snapshot.source_root_canonical,
            candidate_snapshot=snapshot,
            session_duration_sec=600.0,
            sessions_per_hour=6,
            timeline_anchor_mode="civil",
            acquisition_mode="intermittent",
            discovered_roi_ids=("Region0",),
            included_roi_ids=("Region0",),
            rwd_time_col="TimeStamp",
            uv_suffix="-410",
            sig_suffix="-470",
            parser_contract_digest="d" * 64,
            missing_canonical_relative_paths=(rel,),
            excluded_canonical_relative_path=rel,
        )


def test_rwd_adapter_reuses_a2_chronology_not_a_second_sort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Shuffled filesystem enumeration must not change the normalized
    description's session order or identity -- the adapter must not
    perform its own, second ordering pass."""
    import os
    import random

    names = ["2025_05_03-00_00_00", "2025_05_03-00_10_00", "2025_05_03-00_20_00"]
    for name in names:
        _touch(tmp_path / name / "fluorescence.csv")

    baseline_snapshot = build_rwd_source_candidate_snapshot(str(tmp_path))
    baseline = build_rwd_normalized_recording_description(
        source_root_canonical=baseline_snapshot.source_root_canonical,
        candidate_snapshot=baseline_snapshot,
        session_duration_sec=600.0,
        sessions_per_hour=6,
        timeline_anchor_mode="civil",
        acquisition_mode="intermittent",
        discovered_roi_ids=("Region0",),
        included_roi_ids=("Region0",),
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        parser_contract_digest="d" * 64,
    )

    real_scandir = os.scandir

    class _ShuffledScandir:
        def __init__(self, path):
            self._entries = list(real_scandir(path))
            random.shuffle(self._entries)

        def __enter__(self):
            return iter(self._entries)

        def __exit__(self, *exc):
            return False

    import photometry_pipeline.io.rwd_source_snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module.os, "scandir", _ShuffledScandir)
    shuffled_snapshot = build_rwd_source_candidate_snapshot(str(tmp_path))
    shuffled = build_rwd_normalized_recording_description(
        source_root_canonical=shuffled_snapshot.source_root_canonical,
        candidate_snapshot=shuffled_snapshot,
        session_duration_sec=600.0,
        sessions_per_hour=6,
        timeline_anchor_mode="civil",
        acquisition_mode="intermittent",
        discovered_roi_ids=("Region0",),
        included_roi_ids=("Region0",),
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        parser_contract_digest="d" * 64,
    )

    assert [s.stable_source_identity for s in baseline.sessions] == [
        s.stable_source_identity for s in shuffled.sessions
    ]
    assert (
        compute_normalized_recording_description_identity(baseline)
        == compute_normalized_recording_description_identity(shuffled)
    )


def test_rwd_adapter_invalid_session_duration_fails_closed(tmp_path: Path):
    _touch(tmp_path / "2025_05_04-00_00_00" / "fluorescence.csv")
    snapshot = build_rwd_source_candidate_snapshot(str(tmp_path))
    with pytest.raises(NormalizedRecordingError, match="session_duration_sec"):
        build_rwd_normalized_recording_description(
            source_root_canonical=snapshot.source_root_canonical,
            candidate_snapshot=snapshot,
            session_duration_sec=0.0,
            sessions_per_hour=6,
            timeline_anchor_mode="civil",
            acquisition_mode="intermittent",
            discovered_roi_ids=("Region0",),
            included_roi_ids=("Region0",),
            rwd_time_col="TimeStamp",
            uv_suffix="-410",
            sig_suffix="-470",
            parser_contract_digest="d" * 64,
        )


# ---------------------------------------------------------------------------
# C. Shared downstream boundary: fake non-RWD adapter
# ---------------------------------------------------------------------------


def _build_fake_adapter_description() -> NormalizedRecordingDescription:
    """A test-only, deliberately non-RWD-shaped adapter: sessions keyed by
    a synthetic vendor record id, not an RWD folder timestamp; ROI/channel
    names not using RWD's ``-410``/``-470`` suffix convention. Proves the
    shared model does not require RWD vocabulary. The real shared
    production consumer this drives is exercised in
    tests/test_guided_backend_validation_materialization.py and
    tests/test_guided_production_mapping.py (identity/threading), not a
    local fake summarizer -- see task B1-completion item 10.
    """
    sessions = tuple(
        NormalizedSourceSession(
            stable_source_identity=f"vendor-record-{index:03d}",
            canonical_source_reference=f"vendor://acquisition/{index:03d}",
            chronological_position=index,
            authoritative_source_start_time=f"2030-06-{index + 1:02d}T00:00:00",
            source_timing_evidence="fake_vendor_acquisition_log_timestamp",
            expected_timeline_start_time=f"2030-06-{index + 1:02d}T00:00:00",
            expected_duration_sec=45.0,
            observed_duration_sec=45.0,
            disposition=SESSION_DISPOSITION_PROCESS,
            size_bytes=4096,
            content_digest="f" * 64,
        )
        for index in range(3)
    )
    roi_channels = (
        NormalizedRoiChannel(
            roi_id="site-1",
            included=True,
            signal_channel_identity="vendor:green",
            reference_channel_identity="vendor:isosbestic",
        ),
    )
    return NormalizedRecordingDescription(
        schema_name=NORMALIZED_RECORDING_SCHEMA_NAME,
        schema_version=NORMALIZED_RECORDING_SCHEMA_VERSION,
        adapter_format="fake_test_vendor_format",
        adapter_contract_version="fake_test_adapter.v1",
        acquisition_mode="intermittent",
        timeline_anchor_mode="civil",
        recording_source_identity="vendor://acquisition-root",
        source_evidence_identity="e" * 64,
        sessions=sessions,
        roi_channels=roi_channels,
        sampling=NormalizedSamplingContract(
            time_basis="relative_seconds_since_session_start",
            parser_contract_identity="f" * 64,
            sessions_per_hour=80,
            session_duration_sec=45.0,
        ),
        adapter_evidence={"vendor_specific_blob": "opaque"},
    )


def test_fake_non_rwd_adapter_identity_and_serialization_are_format_neutral(tmp_path: Path):
    """Boundary test: the shared identity computation and canonical
    serialize/deserialize round trip -- both real shared production
    machinery, not test-local helpers -- work on a synthetic non-RWD
    adapter result with no RWD vocabulary (no folder timestamps, no
    -410/-470 suffixes)."""
    fake = _build_fake_adapter_description()

    identity = compute_normalized_recording_description_identity(fake)
    assert len(identity) == 64

    payload = serialize_normalized_recording_description(fake)
    restored = deserialize_normalized_recording_description(payload)
    assert (
        compute_normalized_recording_description_identity(restored) == identity
    )
    assert [s.stable_source_identity for s in restored.sessions] == [
        "vendor-record-000",
        "vendor-record-001",
        "vendor-record-002",
    ]
