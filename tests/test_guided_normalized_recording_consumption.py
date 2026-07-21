"""Adapter-neutral consumed-recording evidence and comparator (B1 completion).

Covers: the pure comparator's mismatch categories (exercised with a fake
non-RWD requested/consumed pair, proving genuine format neutrality -- no
RWD adapter is ever touched), and the real RWD consumed-evidence adapter
against a genuine Pipeline-produced run directory (proving the per-chunk
HDF5 evidence and C8 ledger are correctly translated and reconciled).
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.guided_normalized_recording import (
    NormalizedRecordingDescription,
    NormalizedRoiChannel,
    NormalizedSamplingContract,
    NormalizedSourceSession,
    NORMALIZED_RECORDING_SCHEMA_NAME,
    NORMALIZED_RECORDING_SCHEMA_VERSION,
    RWD_FOLDER_TIMESTAMP_EVIDENCE,
    SESSION_DISPOSITION_EXCLUDED,
    SESSION_DISPOSITION_MISSING,
    SESSION_DISPOSITION_PROCESS,
    build_npm_normalized_recording_description,
    build_rwd_normalized_recording_description,
)
from photometry_pipeline.guided_normalized_recording_consumption import (
    EVIDENCE_OBSERVED,
    EVIDENCE_UNAVAILABLE,
    NormalizedConsumedEvidenceError,
    NormalizedConsumedRecordingEvidence,
    NormalizedConsumedRoiResolution,
    NormalizedConsumedSession,
    build_npm_consumed_normalized_recording_evidence,
    build_rwd_consumed_normalized_recording_evidence,
    compare_consumed_normalized_recording_branches,
    compare_requested_and_consumed_normalized_recording,
)
from photometry_pipeline.io.npm_contract import (
    NpmParserContract,
    inspect_npm_csv,
)
from photometry_pipeline.io.npm_source_snapshot import (
    build_npm_source_candidate_snapshot,
)


# ---------------------------------------------------------------------------
# A. Adapter-neutral comparator, exercised with a fake non-RWD pair
# ---------------------------------------------------------------------------


def _fake_requested(
    *, sessions=None, roi_channels=None, adapter_format="lab_rig_fake_adapter"
) -> NormalizedRecordingDescription:
    return NormalizedRecordingDescription(
        schema_name=NORMALIZED_RECORDING_SCHEMA_NAME,
        schema_version=NORMALIZED_RECORDING_SCHEMA_VERSION,
        adapter_format=adapter_format,
        adapter_contract_version="lab_rig_fake_adapter.v1",
        acquisition_mode="intermittent",
        timeline_anchor_mode="civil",
        recording_source_identity="lab-rig-3:channel-A",
        source_evidence_identity="e" * 64,
        sessions=(
            (
                NormalizedSourceSession(
                    stable_source_identity="session-0",
                    canonical_source_reference="lab-rig-3://session-0",
                    chronological_position=0,
                    authoritative_source_start_time="2025-01-01T00:00:00",
                    source_timing_evidence="lab_rig_beacon_timestamp",
                    expected_timeline_start_time=None,
                    expected_duration_sec=600.0,
                    observed_duration_sec=None,
                    disposition=SESSION_DISPOSITION_PROCESS,
                    size_bytes=100,
                    content_digest="a" * 64,
                ),
            )
            if sessions is None
            else sessions
        ),
        roi_channels=(
            (
                NormalizedRoiChannel(
                    roi_id="Neuron1",
                    included=True,
                    signal_channel_identity="Neuron1.green",
                    reference_channel_identity="Neuron1.isosbestic",
                ),
            )
            if roi_channels is None
            else roi_channels
        ),
        sampling=NormalizedSamplingContract(
            time_basis="relative_seconds_since_session_start",
            parser_contract_identity="p" * 64,
            sessions_per_hour=6,
            session_duration_sec=600.0,
        ),
        authorized_time_column_candidates=("BeaconTime",),
        authorized_uv_suffix_candidates=(".isosbestic",),
        authorized_signal_suffix_candidates=(".green",),
    )


def _fake_consumed_session(
    *,
    position=0,
    disposition=SESSION_DISPOSITION_PROCESS,
    source="lab-rig-3://session-0",
    digest="a" * 64,
    size=100,
    chunk_id=0,
    fs_hz=20.0,
    resolved_time_column="BeaconTime",
    signal="Neuron1.green",
    reference="Neuron1.isosbestic",
    evidence_availability=EVIDENCE_OBSERVED,
) -> NormalizedConsumedSession:
    return NormalizedConsumedSession(
        chronological_position=position,
        disposition=disposition,
        consumed_source_reference=source,
        evidence_availability=evidence_availability,
        content_digest=digest,
        size_bytes=size,
        cache_chunk_id=chunk_id,
        observed_duration_sec=599.9,
        fs_hz=fs_hz,
        resolved_time_column=resolved_time_column,
        resolved_header_row=0,
        resolved_timestamp_unit="seconds",
        output_time_basis="relative_seconds_since_session_start",
        roi_resolutions=(
            NormalizedConsumedRoiResolution(
                roi_id="Neuron1",
                resolved_signal_source=signal,
                resolved_reference_source=reference,
            ),
        ),
    )


def _fake_consumed(
    *, sessions=None, processed_roi_ids=("Neuron1",), parser_policy_satisfied=True,
    parser_policy_failure_category=None, adapter_format="lab_rig_fake_adapter",
) -> NormalizedConsumedRecordingEvidence:
    return NormalizedConsumedRecordingEvidence(
        adapter_format=adapter_format,
        analysis_branch="phasic",
        sessions=(_fake_consumed_session(),) if sessions is None else sessions,
        processed_roi_ids=processed_roi_ids,
        parser_policy_satisfied=parser_policy_satisfied,
        parser_policy_failure_category=parser_policy_failure_category,
    )


def test_fake_non_rwd_exact_match_reconciles():
    requested = _fake_requested()
    consumed = _fake_consumed()
    assert compare_requested_and_consumed_normalized_recording(requested, consumed) == ""


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("resolved_time_column", "cross_branch_time_column_mismatch"),
        ("resolved_header_row", "cross_branch_header_row_mismatch"),
        ("resolved_timestamp_unit", "cross_branch_timestamp_unit_mismatch"),
        ("fs_hz", "cross_branch_output_sampling_rate_mismatch"),
        ("output_time_basis", "cross_branch_output_time_basis_mismatch"),
    ],
)
def test_cross_branch_reconciliation_is_session_specific(field, expected):
    left = _fake_consumed().sessions[0]
    right = replace(
        left,
        **{
            field: (
                21.0
                if field == "fs_hz"
                else "alternate-time"
                if field == "output_time_basis"
                else 1
                if field == "resolved_header_row"
                else "TimeStampAlternate"
            )
        },
    )
    result = compare_consumed_normalized_recording_branches(
        _fake_consumed(sessions=(left,)),
        replace(
            _fake_consumed(sessions=(right,)),
            analysis_branch="tonic",
        ),
    )
    assert expected in result


def test_cross_branch_reconciliation_catches_signal_and_reference_resolution():
    left = _fake_consumed().sessions[0]
    right_roi = replace(
        left.roi_resolutions[0],
        resolved_signal_source="Neuron1.other-green",
        resolved_reference_source="Neuron1.other-isosbestic",
    )
    right = replace(left, roi_resolutions=(right_roi,))
    result = compare_consumed_normalized_recording_branches(
        _fake_consumed(sessions=(left,)),
        replace(_fake_consumed(sessions=(right,)), analysis_branch="tonic"),
    )
    assert "cross_branch_signal_source_mismatch" in result


def test_cross_branch_reconciliation_does_not_compare_unordered_rates():
    first = _fake_consumed().sessions[0]
    second = replace(
        first,
        chronological_position=1,
        consumed_source_reference="lab-rig-3://session-1",
        fs_hz=30.0,
    )
    phasic = _fake_consumed(sessions=(replace(first, fs_hz=20.0), second))
    tonic = replace(
        _fake_consumed(
            sessions=(
                replace(first, fs_hz=30.0),
                replace(second, fs_hz=20.0),
            )
        ),
        analysis_branch="tonic",
    )
    assert "cross_branch_output_sampling_rate_mismatch" in (
        compare_consumed_normalized_recording_branches(phasic, tonic)
    )


def test_fake_non_rwd_adapter_format_mismatch_refuses():
    requested = _fake_requested()
    consumed = _fake_consumed(adapter_format="other_adapter")
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "adapter_format" in result


def test_fake_non_rwd_parser_policy_failure_refuses():
    requested = _fake_requested()
    consumed = _fake_consumed(
        parser_policy_satisfied=False,
        parser_policy_failure_category="unauthorized_time_column",
    )
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "parser policy" in result
    assert "unauthorized_time_column" in result


def test_fake_non_rwd_session_membership_mismatch_refuses():
    requested = _fake_requested()
    consumed = _fake_consumed(sessions=(_fake_consumed_session(position=1),))
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "session set" in result


def test_fake_non_rwd_disposition_mismatch_refuses():
    requested = _fake_requested()
    consumed = _fake_consumed(
        sessions=(_fake_consumed_session(disposition=SESSION_DISPOSITION_MISSING, chunk_id=None),)
    )
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "disposition" in result


def test_fake_non_rwd_content_digest_mismatch_refuses():
    requested = _fake_requested()
    consumed = _fake_consumed(sessions=(_fake_consumed_session(digest="f" * 64),))
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "digest" in result


def test_fake_non_rwd_roi_set_mismatch_refuses():
    requested = _fake_requested()
    consumed = _fake_consumed(processed_roi_ids=("Neuron1", "UnauthorizedRoi"))
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "ROI" in result


def test_fake_non_rwd_missing_session_processed_as_ordinary_refuses():
    requested = _fake_requested(
        sessions=(
            replace(
                _fake_requested().sessions[0],
                disposition=SESSION_DISPOSITION_MISSING,
            ),
        )
    )
    consumed = _fake_consumed(
        sessions=(_fake_consumed_session(disposition=SESSION_DISPOSITION_MISSING, chunk_id=0),)
    )
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "processed as an ordinary" in result


def test_fake_non_rwd_excluded_final_session_source_evidence_unavailable_refuses():
    requested = _fake_requested(
        sessions=(
            replace(
                _fake_requested().sessions[0],
                disposition=SESSION_DISPOSITION_EXCLUDED,
            ),
        )
    )
    consumed = _fake_consumed(
        sessions=(
            _fake_consumed_session(
                disposition=SESSION_DISPOSITION_EXCLUDED,
                chunk_id=None,
                evidence_availability=EVIDENCE_UNAVAILABLE,
                digest=None,
            ),
        )
    )
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "excluded final session" in result


def test_comparator_module_has_no_rwd_specific_names():
    """The comparator itself must never import RWD internals -- only the
    RWD-specific adapter function is allowed to."""
    import photometry_pipeline.guided_normalized_recording_consumption as mod
    import inspect

    source = inspect.getsource(mod.compare_requested_and_consumed_normalized_recording)
    for forbidden in (
        "rwd", "RWD", "fluorescence.csv", "-410", "-470", "hdf5", "h5py",
        "input_processing_completeness", "roi/", "chunk_{",
    ):
        assert forbidden not in source, f"comparator references {forbidden!r}"


# ---------------------------------------------------------------------------
# B. RWD-specific consumed-evidence adapter, against a real Pipeline run
# ---------------------------------------------------------------------------


def _write_rwd_source(path, *, n=200, fs=20.0):
    df = pd.DataFrame(
        {
            "TimeStamp": np.arange(n) / fs,
            "Region0-410": np.random.rand(n) + 1.0,
            "Region0-470": np.random.rand(n) + 1.0,
        }
    )
    df.to_csv(path, index=False)


@pytest.fixture
def real_rwd_run(tmp_path):
    root = tmp_path / "input"
    session_dir = root / "2025_01_01-00_00_00"
    session_dir.mkdir(parents=True)
    fpath = session_dir / "fluorescence.csv"
    _write_rwd_source(fpath)
    digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
    size = fpath.stat().st_size

    cfg = Config(
        target_fs_hz=20.0,
        chunk_duration_sec=10.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
    )
    run_dir = tmp_path / "run"
    out = run_dir / "_analysis" / "phasic_out"
    Pipeline(cfg, mode="phasic").run(
        str(root), str(out), force_format="rwd", recursive=True, sessions_per_hour=1
    )

    candidate = SimpleNamespace(
        canonical_relative_path="2025_01_01-00_00_00/fluorescence.csv",
        size_bytes=size,
        sha256_content_digest=digest,
    )
    snapshot = SimpleNamespace(
        candidates=(candidate,),
        source_candidate_set_digest="s" * 64,
        source_candidate_content_digest="c" * 64,
    )
    requested = build_rwd_normalized_recording_description(
        source_root_canonical=str(root),
        candidate_snapshot=snapshot,
        session_duration_sec=10.0,
        sessions_per_hour=1,
        timeline_anchor_mode="none",
        acquisition_mode="intermittent",
        discovered_roi_ids=("Region0",),
        included_roi_ids=("Region0",),
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        parser_contract_digest="d" * 64,
    )
    return str(run_dir), requested


def test_rwd_adapter_and_comparator_reconcile_a_real_run(real_rwd_run):
    run_dir, requested = real_rwd_run
    consumed = build_rwd_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    assert consumed.parser_policy_satisfied is True
    assert compare_requested_and_consumed_normalized_recording(requested, consumed) == ""


def test_rwd_adapter_missing_ledger_refuses(tmp_path, real_rwd_run):
    run_dir, requested = real_rwd_run
    ledger = os.path.join(run_dir, "_analysis", "phasic_out", "input_processing_completeness.json")
    os.remove(ledger)
    with pytest.raises(NormalizedConsumedEvidenceError):
        build_rwd_consumed_normalized_recording_evidence(
            run_dir=run_dir, analysis_kind="phasic", requested=requested
        )


def test_rwd_consumed_digest_tamper_caught_by_comparator(real_rwd_run):
    run_dir, requested = real_rwd_run
    ledger_path = os.path.join(
        run_dir, "_analysis", "phasic_out", "input_processing_completeness.json"
    )
    payload = json.loads(open(ledger_path, encoding="utf-8").read())
    payload["expected"][0]["sha256"] = "0" * 64
    with open(ledger_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    consumed = build_rwd_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "digest" in result


def test_rwd_consumed_unauthorized_time_column_caught(real_rwd_run):
    run_dir, requested = real_rwd_run
    tampered_requested = replace(
        requested, authorized_time_column_candidates=("SomeOtherColumn",)
    )
    consumed = build_rwd_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=tampered_requested
    )
    assert consumed.parser_policy_satisfied is False
    assert consumed.parser_policy_failure_category == "unauthorized_time_column"
    result = compare_requested_and_consumed_normalized_recording(
        tampered_requested, consumed
    )
    assert "parser policy" in result


def test_rwd_consumed_unauthorized_signal_source_caught(real_rwd_run):
    run_dir, requested = real_rwd_run
    tampered_requested = replace(
        requested, authorized_signal_suffix_candidates=("-999",)
    )
    consumed = build_rwd_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=tampered_requested
    )
    assert consumed.parser_policy_satisfied is False
    assert consumed.parser_policy_failure_category == "unauthorized_signal_source"


def test_rwd_consumed_extra_roi_processed_caught(real_rwd_run):
    run_dir, requested = real_rwd_run
    consumed = build_rwd_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    tampered_consumed = replace(consumed, processed_roi_ids=("Region0", "RegionGhost"))
    result = compare_requested_and_consumed_normalized_recording(
        requested, tampered_consumed
    )
    assert "ROI" in result


def test_rwd_still_requires_resolved_time_column_after_npm_repair(real_rwd_run):
    """Regression proof: RWD's existence checks for resolved_time_column/
    header_row/timestamp_unit remain fully required -- unaffected by the
    new NPM-only comparator guard."""
    run_dir, requested = real_rwd_run
    consumed = build_rwd_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    tampered_session = replace(consumed.sessions[0], resolved_time_column=None)
    tampered_consumed = replace(
        consumed, sessions=(tampered_session,) + consumed.sessions[1:]
    )
    result = compare_requested_and_consumed_normalized_recording(
        requested, tampered_consumed
    )
    assert "time-column" in result


def test_rwd_still_requires_roi_channel_resolution_after_npm_repair(real_rwd_run):
    """Regression proof: RWD's per-session per-ROI signal/reference source
    comparison remains fully required -- unaffected by the new NPM-only
    comparator guard."""
    run_dir, requested = real_rwd_run
    consumed = build_rwd_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    tampered_roi = replace(
        consumed.sessions[0].roi_resolutions[0],
        resolved_signal_source="Region0-999",
    )
    tampered_session = replace(consumed.sessions[0], roi_resolutions=(tampered_roi,))
    tampered_consumed = replace(
        consumed, sessions=(tampered_session,) + consumed.sessions[1:]
    )
    result = compare_requested_and_consumed_normalized_recording(
        requested, tampered_consumed
    )
    assert "signal source" in result


# ---------------------------------------------------------------------------
# C. NPM-specific consumed-evidence adapter, against a real Pipeline run
# ---------------------------------------------------------------------------


def _write_npm_source(path, *, n_seconds: float = 6.0, rate_hz: float = 2.0) -> None:
    step = 1.0 / (2.0 * rate_hz)
    rows = ["Timestamp,LedState,Region0G"]
    t = 0.0
    led = 1
    n_samples = int(n_seconds * 2.0 * rate_hz)
    for i in range(n_samples):
        rows.append(f"{t:.4f},{led},{10.0 + 0.01 * i:.4f}")
        t += step
        led = 2 if led == 1 else 1
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _npm_config() -> Config:
    return Config(
        target_fs_hz=2.0,
        chunk_duration_sec=6.0,
        npm_time_axis="system_timestamp",
        npm_system_ts_col="Timestamp",
        npm_computer_ts_col="Timestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
        adapter_value_nan_policy="strict",
        timestamp_cv_max=0.02,
    )


@pytest.fixture
def real_npm_run(tmp_path):
    root = tmp_path / "npm_input"
    root.mkdir()
    path = root / "photometryData2025-03-05T15_37_44.csv"
    _write_npm_source(path)

    cfg = _npm_config()
    contract = NpmParserContract.from_config(cfg)
    content = contract.content()
    inspection = inspect_npm_csv(str(path), contract)
    snapshot = build_npm_source_candidate_snapshot(str(root))

    requested = build_npm_normalized_recording_description(
        source_snapshot=snapshot,
        session_inspections={
            snapshot.candidates[0].canonical_relative_path: inspection
        },
        parser_contract_content=content,
        session_duration_sec=6.0,
        sessions_per_hour=1,
        discovered_roi_ids=inspection.roi_ids,
        included_roi_ids=inspection.roi_ids,
        target_fs_hz=2.0,
    )

    run_dir = tmp_path / "run"
    out = run_dir / "_analysis" / "phasic_out"
    Pipeline(cfg, mode="phasic").run(
        str(root), str(out), force_format="npm", recursive=True, sessions_per_hour=1
    )
    return str(run_dir), requested


def test_npm_adapter_and_comparator_reconcile_a_real_run(real_npm_run):
    run_dir, requested = real_npm_run
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    assert consumed.adapter_format == "npm"
    assert compare_requested_and_consumed_normalized_recording(requested, consumed) == ""


def test_npm_adapter_does_not_copy_requested_channel_identities(real_npm_run):
    """The exact correction this repair makes: NPM consumed evidence must
    never claim execution independently resolved facts it did not
    observe."""
    run_dir, requested = real_npm_run
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    for session in consumed.sessions:
        assert session.resolved_time_column is None
        assert session.resolved_header_row is None
        assert session.resolved_timestamp_unit is None
        assert session.roi_resolutions == ()


def test_npm_absent_rwd_only_resolution_fields_do_not_invalidate_completion(
    real_npm_run,
):
    """The exact scenario this repair fixes: NPM's genuinely absent
    RWD-only resolution fields must not refuse an otherwise truthful NPM
    completion."""
    run_dir, requested = real_npm_run
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    assert all(s.resolved_time_column is None for s in consumed.sessions)
    assert all(s.resolved_header_row is None for s in consumed.sessions)
    assert all(s.resolved_timestamp_unit is None for s in consumed.sessions)
    assert compare_requested_and_consumed_normalized_recording(requested, consumed) == ""


def test_npm_adapter_format_mismatch_refuses(real_npm_run):
    run_dir, requested = real_npm_run
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    tampered_consumed = replace(consumed, adapter_format="rwd")
    result = compare_requested_and_consumed_normalized_recording(
        requested, tampered_consumed
    )
    assert "adapter_format" in result


def test_npm_adapter_missing_ledger_refuses(real_npm_run):
    run_dir, requested = real_npm_run
    ledger = os.path.join(
        run_dir, "_analysis", "phasic_out", "input_processing_completeness.json"
    )
    os.remove(ledger)
    with pytest.raises(NormalizedConsumedEvidenceError):
        build_npm_consumed_normalized_recording_evidence(
            run_dir=run_dir, analysis_kind="phasic", requested=requested
        )


def test_npm_adapter_missing_cache_refuses(real_npm_run):
    run_dir, requested = real_npm_run
    cache_path = os.path.join(run_dir, "_analysis", "phasic_out", "phasic_trace_cache.h5")
    os.remove(cache_path)
    with pytest.raises(NormalizedConsumedEvidenceError):
        build_npm_consumed_normalized_recording_evidence(
            run_dir=run_dir, analysis_kind="phasic", requested=requested
        )


def test_npm_missing_tonic_evidence_refuses(real_npm_run):
    """This fixture only produces a phasic branch; requesting tonic
    consumed evidence must fail closed, proving missing-branch detection
    for NPM works the same as it does for RWD."""
    run_dir, requested = real_npm_run
    with pytest.raises(NormalizedConsumedEvidenceError):
        build_npm_consumed_normalized_recording_evidence(
            run_dir=run_dir, analysis_kind="tonic", requested=requested
        )


def test_npm_malformed_ledger_refuses(real_npm_run):
    run_dir, requested = real_npm_run
    ledger = os.path.join(
        run_dir, "_analysis", "phasic_out", "input_processing_completeness.json"
    )
    with open(ledger, "w", encoding="utf-8") as handle:
        handle.write("{not valid json")
    with pytest.raises(NormalizedConsumedEvidenceError):
        build_npm_consumed_normalized_recording_evidence(
            run_dir=run_dir, analysis_kind="phasic", requested=requested
        )


def test_npm_consumed_digest_tamper_caught_by_comparator(real_npm_run):
    run_dir, requested = real_npm_run
    ledger_path = os.path.join(
        run_dir, "_analysis", "phasic_out", "input_processing_completeness.json"
    )
    payload = json.loads(open(ledger_path, encoding="utf-8").read())
    payload["expected"][0]["sha256"] = "0" * 64
    with open(ledger_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "digest" in result


def test_npm_consumed_size_mismatch_caught_by_comparator(real_npm_run):
    run_dir, requested = real_npm_run
    ledger_path = os.path.join(
        run_dir, "_analysis", "phasic_out", "input_processing_completeness.json"
    )
    payload = json.loads(open(ledger_path, encoding="utf-8").read())
    payload["expected"][0]["size_bytes"] = payload["expected"][0]["size_bytes"] + 7
    with open(ledger_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "size" in result


def test_npm_consumed_source_identity_mismatch_caught(real_npm_run):
    run_dir, requested = real_npm_run
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    tampered_session = replace(
        consumed.sessions[0], consumed_source_reference="some/other/path.csv"
    )
    tampered_consumed = replace(
        consumed, sessions=(tampered_session,) + consumed.sessions[1:]
    )
    result = compare_requested_and_consumed_normalized_recording(
        requested, tampered_consumed
    )
    assert "source" in result and "identity" in result


def test_npm_consumed_missing_cache_chunk_identity_caught(real_npm_run):
    run_dir, requested = real_npm_run
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    tampered_session = replace(consumed.sessions[0], cache_chunk_id=None)
    tampered_consumed = replace(
        consumed, sessions=(tampered_session,) + consumed.sessions[1:]
    )
    result = compare_requested_and_consumed_normalized_recording(
        requested, tampered_consumed
    )
    assert "cache chunk" in result


def test_npm_consumed_sampling_rate_mismatch_caught(real_npm_run):
    run_dir, requested = real_npm_run
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    tampered_requested = replace(
        requested, sampling=replace(requested.sampling, target_fs_hz=999.0)
    )
    result = compare_requested_and_consumed_normalized_recording(
        tampered_requested, consumed
    )
    assert "sampling rate" in result


def test_npm_consumed_output_time_basis_mismatch_caught(real_npm_run):
    run_dir, requested = real_npm_run
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    tampered_requested = replace(
        requested, sampling=replace(requested.sampling, time_basis="something_else")
    )
    result = compare_requested_and_consumed_normalized_recording(
        tampered_requested, consumed
    )
    assert "time basis" in result


def test_npm_consumed_roi_set_mismatch_caught(real_npm_run):
    run_dir, requested = real_npm_run
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    tampered_consumed = replace(
        consumed, processed_roi_ids=consumed.processed_roi_ids + ("RegionGhost",)
    )
    result = compare_requested_and_consumed_normalized_recording(
        requested, tampered_consumed
    )
    assert "ROI" in result


def test_npm_cross_branch_disagreement_on_observable_field_caught(real_npm_run):
    """Phasic/tonic disagreement on an NPM-observable field (sampling
    rate) is still caught by the shared, unmodified cross-branch
    comparator."""
    run_dir, requested = real_npm_run
    phasic = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    tonic = replace(phasic, analysis_branch="tonic")
    tampered_session = replace(tonic.sessions[0], fs_hz=999.0)
    tampered_tonic = replace(
        tonic, sessions=(tampered_session,) + tonic.sessions[1:]
    )
    result = compare_consumed_normalized_recording_branches(phasic, tampered_tonic)
    assert "cross_branch_output_sampling_rate_mismatch" in result


def test_npm_cross_branch_agreement_reconciles(real_npm_run):
    """Two branches with identical NPM-observable evidence (as if phasic
    and tonic both genuinely consumed the same authorized recording)
    reconcile cleanly -- proving absent RWD-only fields (None on both
    sides) never cause a false cross-branch mismatch."""
    run_dir, requested = real_npm_run
    phasic = build_npm_consumed_normalized_recording_evidence(
        run_dir=run_dir, analysis_kind="phasic", requested=requested
    )
    tonic = replace(phasic, analysis_branch="tonic")
    assert compare_consumed_normalized_recording_branches(phasic, tonic) == ""
