from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import photometry_pipeline.io.adapters as adapters
import photometry_pipeline.pipeline as pipeline_module
from photometry_pipeline.config import Config
from photometry_pipeline.guided_npm_authorized_adapter import (
    GuidedNpmAuthorizedRuntime,
    build_guided_npm_authorized_runtime,
    load_guided_npm_authorized_chunk,
    verify_guided_npm_authorized_input,
    verify_guided_npm_authorized_runtime,
)
from photometry_pipeline.guided_npm_worker_prelaunch_claim import (
    GuidedNpmWorkerPrelaunchClaim,
    claim_guided_npm_worker_for_prelaunch,
)
from photometry_pipeline.guided_npm_production_execution_request import (
    compute_guided_npm_production_adapter_runtime_projection_identity,
    compute_guided_npm_production_execution_request_identity,
    compute_guided_npm_production_session_runtime_identity,
    compute_guided_npm_production_session_sequence_identity,
    compute_guided_npm_production_source_content_identity,
    compute_guided_npm_production_source_membership_identity,
    compute_guided_npm_production_source_runtime_projection_identity,
    compute_guided_npm_production_timing_runtime_projection_identity,
)
from photometry_pipeline.guided_npm_worker_request import (
    compute_guided_npm_live_freshness_evidence_identity,
    compute_guided_npm_live_verified_file_sequence_identity,
    compute_guided_npm_live_verified_source_file_identity,
    compute_guided_npm_worker_request_identity,
    verify_guided_npm_worker_request,
)
from photometry_pipeline.guided_npm_worker_request_materialization import (
    GuidedNpmWorkerRequestMaterializationReceipt,
    materialize_guided_npm_worker_request,
)
from photometry_pipeline.io.adapters import load_npm_authorized_bytes
from photometry_pipeline.io.npm_contract import NpmParserContract
from photometry_pipeline.pipeline import Pipeline
from tests.test_guided_npm_production_execution_request import (
    _request,
    _two_roi_payload,
)
from tests.test_guided_npm_startup_payload import _payload
from tests.test_guided_npm_worker_prelaunch_claim import _claim


def _runtime(tmp_path: Path) -> tuple[GuidedNpmWorkerPrelaunchClaim, GuidedNpmAuthorizedRuntime]:
    _, _, claim = _claim(tmp_path)
    assert isinstance(claim, GuidedNpmWorkerPrelaunchClaim)
    runtime = build_guided_npm_authorized_runtime(claim.worker_request)
    verify_guided_npm_authorized_runtime(runtime, claim.worker_request)
    return claim, runtime


def _two_roi_runtime(tmp_path: Path) -> GuidedNpmAuthorizedRuntime:
    payload = _two_roi_payload(_payload(tmp_path))
    _, _, execution = _request(tmp_path, payload)
    receipt = materialize_guided_npm_worker_request(
        execution,
        current_application_build_identity=execution.application_build_identity,
    )
    assert isinstance(receipt, GuidedNpmWorkerRequestMaterializationReceipt)
    claim = claim_guided_npm_worker_for_prelaunch(
        receipt.worker_request_artifact_path,
        receipt,
        current_application_build_identity=execution.application_build_identity,
    )
    assert isinstance(claim, GuidedNpmWorkerPrelaunchClaim)
    return build_guided_npm_authorized_runtime(claim.worker_request)


def _explicit_csv(*, extra_roi: bool = False, missing_roi: bool = False) -> bytes:
    columns = ["SystemTimestamp", "Timestamp", "LedState", "Region4G"]
    if not missing_roi:
        columns.append("Region0G")
    if extra_roi:
        columns.append("Region9G")
    rows = []
    for time, legacy, led, four, zero in (
        (100.0, 900.0, 1, 40.0, 10.0),
        (100.5, 901.0, 2, 400.0, 100.0),
        (101.0, 902.0, 1, 41.0, 11.0),
        (101.5, 903.0, 2, 401.0, 101.0),
        (102.0, 904.0, 1, 42.0, 12.0),
        (102.5, 905.0, 2, 402.0, 102.0),
    ):
        values = [time, legacy, led, four]
        if not missing_roi:
            values.append(zero)
        if extra_roi:
            values.append(999.0)
        rows.append(",".join(map(str, values)))
    return (",".join(columns) + "\n" + "\n".join(rows) + "\n").encode()


def _explicit_contract() -> NpmParserContract:
    return NpmParserContract(
        npm_time_axis="system_timestamp",
        npm_system_ts_col="SystemTimestamp",
        npm_computer_ts_col="Timestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
        target_fs_hz=2.0,
        session_duration_sec=2.0,
        allow_partial_final_chunk=False,
        adapter_value_nan_policy="strict",
        timestamp_cv_max=0.02,
    )


def _mixed_gap_runtime(tmp_path: Path) -> GuidedNpmAuthorizedRuntime:
    """Three exact sources whose frozen actual chronology differs from nominal."""
    _, base = _runtime(tmp_path)
    root = tmp_path / "mixed_gap_sources"
    root.mkdir()
    content_rows = ["SystemTimestamp,Timestamp,LedState,Region2G"]
    row_index = 0
    for sample_index in range(13):
        timestamp = 100.0 + (sample_index * 0.5)
        for led in (1, 2):
            value = 10.0 + sample_index + (20.0 if led == 2 else 0.0)
            content_rows.append(f"{timestamp},{900 + row_index},{led},{value}")
            row_index += 1
    content = ("\n".join(content_rows) + "\n").encode()
    paths = tuple(
        os.path.abspath(root / name).lower()
        for name in ("z_last_by_name.csv", "a_first_by_name.csv", "m_middle.csv")
    )
    for path in paths:
        Path(path).write_bytes(content)
    contract = replace(
        base.authorized_input.parser_contract,
        session_duration_sec=6.0,
    )
    config = replace(base.config, chunk_duration_sec=6.0)
    probe = load_npm_authorized_bytes(
        paths[0],
        content,
        config,
        0,
        contract=contract,
        resolved_timestamp_column="SystemTimestamp",
        reference_led_value=1,
        signal_led_value=2,
        physical_to_canonical_roi_mapping=base.authorized_input.physical_to_canonical_roi_map,
    )
    metadata = probe.metadata
    actual = (0.0, 90.0, 150.0)
    nominal = (0.0, 60.0, 120.0)
    source_starts = (
        "2030-01-03T00:00:00",
        "2030-01-01T00:00:00",
        "2030-01-02T00:00:00",
    )
    sessions = tuple(
        replace(
            base.authorized_input.ordered_sessions[0],
            chronological_position=index,
            source_path=path,
            canonical_relative_path=Path(path).name,
            source_size_bytes=len(content),
            source_sha256=hashlib.sha256(content).hexdigest(),
            authoritative_source_start_time=source_starts[index],
            actual_elapsed_sec=actual[index],
            nominal_expected_elapsed_sec=nominal[index],
            resolved_timestamp_column="SystemTimestamp",
            overlap_origin_absolute=metadata["npm_overlap_origin_absolute"],
            support_start_offset_sec=metadata["npm_resolved_support_start_offset_sec"],
            support_end_offset_sec=metadata["npm_resolved_support_end_offset_sec"],
            support_start_absolute=metadata["npm_resolved_support_start_absolute"],
            support_end_absolute=metadata["npm_resolved_support_end_absolute"],
            observed_support_duration_sec=metadata["npm_observed_duration_sec"],
            canonical_session_runtime_identity=hashlib.sha256(
                f"{path}|{actual[index]}|{nominal[index]}".encode()
            ).hexdigest(),
        )
        for index, path in enumerate(paths)
    )
    parser_json = json.dumps(
        contract.content(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    authorized = replace(
        base.authorized_input,
        source_root_path=os.path.abspath(root).lower(),
        run_directory_path=os.path.abspath(tmp_path / "mixed_gap_output").lower(),
        ordered_sessions=sessions,
        ordered_session_paths=paths,
        ordered_session_identities=tuple(
            item.canonical_session_runtime_identity for item in sessions
        ),
        chronological_positions=(0, 1, 2),
        authoritative_source_start_times=source_starts,
        actual_elapsed_sec_by_chunk=actual,
        nominal_expected_elapsed_sec_by_chunk=nominal,
        parser_contract=contract,
        parser_contract_identity=contract.digest,
        parser_contract_content_json=parser_json,
        configured_session_duration_sec=6.0,
        sessions_per_hour=60,
    )
    runtime = replace(base, authorized_input=authorized, config=config)
    verify_guided_npm_authorized_input(runtime.authorized_input)
    return runtime


def _worker_with_changed_actual_chronology(worker, actual_elapsed):
    """Reidentify one independently valid worker over the exact same files."""
    execution = worker.execution_request
    sessions = []
    for session, elapsed in zip(
        execution.source_runtime_projection.ordered_sessions,
        actual_elapsed,
        strict=True,
    ):
        changed = replace(
            session,
            actual_elapsed_sec=float(elapsed),
            canonical_session_runtime_identity="0" * 64,
        )
        sessions.append(
            replace(
                changed,
                canonical_session_runtime_identity=(
                    compute_guided_npm_production_session_runtime_identity(changed)
                ),
            )
        )
    sessions = tuple(sessions)
    source = replace(
        execution.source_runtime_projection,
        ordered_sessions=sessions,
        runtime_source_membership_identity=(
            compute_guided_npm_production_source_membership_identity(sessions)
        ),
        runtime_source_content_identity=(
            compute_guided_npm_production_source_content_identity(sessions)
        ),
        runtime_session_sequence_identity=(
            compute_guided_npm_production_session_sequence_identity(sessions)
        ),
        canonical_source_runtime_projection_identity="0" * 64,
    )
    source = replace(
        source,
        canonical_source_runtime_projection_identity=(
            compute_guided_npm_production_source_runtime_projection_identity(source)
        ),
    )
    timing = replace(
        execution.timing_runtime_projection,
        ordered_actual_elapsed_sec=tuple(float(value) for value in actual_elapsed),
        source_session_sequence_identity=source.runtime_session_sequence_identity,
        canonical_timing_runtime_projection_identity="0" * 64,
    )
    timing = replace(
        timing,
        canonical_timing_runtime_projection_identity=(
            compute_guided_npm_production_timing_runtime_projection_identity(timing)
        ),
    )
    adapter = replace(
        execution.adapter_runtime_projection,
        source_runtime_projection_identity=(
            source.canonical_source_runtime_projection_identity
        ),
        timing_runtime_projection_identity=(
            timing.canonical_timing_runtime_projection_identity
        ),
        canonical_adapter_runtime_projection_identity="0" * 64,
    )
    adapter = replace(
        adapter,
        canonical_adapter_runtime_projection_identity=(
            compute_guided_npm_production_adapter_runtime_projection_identity(adapter)
        ),
    )
    execution = replace(
        execution,
        source_runtime_projection=source,
        timing_runtime_projection=timing,
        adapter_runtime_projection=adapter,
        canonical_execution_request_identity="0" * 64,
    )
    execution = replace(
        execution,
        canonical_execution_request_identity=(
            compute_guided_npm_production_execution_request_identity(execution)
        ),
    )
    live_files = []
    for live, session in zip(
        worker.live_freshness_evidence.ordered_verified_files,
        sessions,
        strict=True,
    ):
        changed = replace(
            live,
            source_runtime_session_identity=session.canonical_session_runtime_identity,
            canonical_live_verified_source_file_identity="0" * 64,
        )
        live_files.append(
            replace(
                changed,
                canonical_live_verified_source_file_identity=(
                    compute_guided_npm_live_verified_source_file_identity(changed)
                ),
            )
        )
    live_files = tuple(live_files)
    freshness = replace(
        worker.live_freshness_evidence,
        expected_runtime_source_projection_identity=(
            source.canonical_source_runtime_projection_identity
        ),
        expected_runtime_membership_identity=source.runtime_source_membership_identity,
        expected_runtime_content_identity=source.runtime_source_content_identity,
        expected_runtime_session_sequence_identity=source.runtime_session_sequence_identity,
        ordered_verified_files=live_files,
        live_membership_identity=source.runtime_source_membership_identity,
        live_content_identity=source.runtime_source_content_identity,
        live_session_sequence_identity=source.runtime_session_sequence_identity,
        live_verified_file_sequence_identity=(
            compute_guided_npm_live_verified_file_sequence_identity(live_files)
        ),
        canonical_live_freshness_evidence_identity="0" * 64,
    )
    freshness = replace(
        freshness,
        canonical_live_freshness_evidence_identity=(
            compute_guided_npm_live_freshness_evidence_identity(freshness)
        ),
    )
    changed_worker = replace(
        worker,
        source_execution_request_identity=execution.canonical_execution_request_identity,
        execution_request=execution,
        live_freshness_evidence=freshness,
        canonical_worker_request_identity="0" * 64,
    )
    changed_worker = replace(
        changed_worker,
        canonical_worker_request_identity=(
            compute_guided_npm_worker_request_identity(changed_worker)
        ),
    )
    verify_guided_npm_worker_request(changed_worker)
    return changed_worker


def test_authorized_runtime_uses_exact_worker_session_sequence(tmp_path):
    claim, runtime = _runtime(tmp_path)
    source = claim.worker_request.execution_request.source_runtime_projection
    authorized = runtime.authorized_input
    assert authorized.ordered_session_paths == tuple(
        session.source_path for session in source.ordered_sessions
    )
    assert authorized.chronological_positions == tuple(range(len(source.ordered_sessions)))
    assert authorized.ordered_session_identities == tuple(
        session.canonical_session_runtime_identity for session in source.ordered_sessions
    )


def test_authorized_order_reaches_sink_without_sorting(monkeypatch, tmp_path):
    _, runtime = _runtime(tmp_path)
    authorized = runtime.authorized_input
    monkeypatch.setattr(adapters, "sort_npm_files", lambda *_: pytest.fail("legacy sort"))
    observed = []
    for index, path in enumerate(authorized.ordered_session_paths):
        chunk = load_guided_npm_authorized_chunk(
            authorized, path, runtime.config, index
        )
        observed.append(chunk.source_file)
    assert tuple(observed) == authorized.ordered_session_paths


def test_no_directory_discovery_or_legacy_npm_loader(monkeypatch, tmp_path):
    _, runtime = _runtime(tmp_path)
    pipeline = Pipeline(
        runtime.config,
        mode=runtime.mode,
        per_roi_correction=runtime.per_roi_correction,
        per_roi_feature_config=runtime.per_roi_feature_config,
        per_roi_feature_provenance=runtime.per_roi_feature_provenance,
    )
    pipeline._guided_npm_authorized_runtime = runtime
    monkeypatch.setattr(pipeline_module.glob, "glob", lambda *a, **k: pytest.fail("glob"))
    monkeypatch.setattr(pipeline_module, "sort_npm_files", lambda *a, **k: pytest.fail("sort"))
    monkeypatch.setattr(pipeline_module, "load_chunk", lambda *a, **k: pytest.fail("legacy load"))
    monkeypatch.setattr(adapters, "_load_npm", lambda *a, **k: pytest.fail("legacy npm"))
    monkeypatch.setattr(adapters, "_parse_vendor_npm_timestamp", lambda *a, **k: pytest.fail("filename timestamp"))
    monkeypatch.setattr(adapters, "_npm_roi_sort_key", lambda *a, **k: pytest.fail("ROI sort"))
    monkeypatch.setattr(adapters, "_create_canonical_names", lambda *a, **k: pytest.fail("positional ROI identity"))
    monkeypatch.setattr(
        adapters,
        "resolve_npm_support_geometry",
        lambda *a, **k: pytest.fail("timing geometry recomputation"),
    )
    monkeypatch.setattr(Path, "glob", lambda *a, **k: pytest.fail("Path.glob"))
    monkeypatch.setattr(Path, "rglob", lambda *a, **k: pytest.fail("Path.rglob"))
    monkeypatch.setattr(os, "scandir", lambda *a, **k: pytest.fail("scandir"))
    chunk = pipeline._load_entry_chunk(
        runtime.authorized_input.ordered_session_paths[0], 0, "npm"
    )
    assert tuple(chunk.channel_names) == runtime.authorized_input.canonical_roi_ids


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_path", "missing_path", "added_path", "duplicate_position", "gap_position"],
)
def test_authorized_membership_tampering_refuses_pure_verification(tmp_path, mutation):
    claim, runtime = _runtime(tmp_path)
    authorized = runtime.authorized_input
    if mutation == "duplicate_path":
        changed = replace(
            authorized,
            ordered_session_paths=(authorized.ordered_session_paths[0],) * len(authorized.ordered_session_paths),
        )
    elif mutation == "missing_path":
        changed = replace(authorized, ordered_session_paths=authorized.ordered_session_paths[:-1])
    elif mutation == "added_path":
        changed = replace(authorized, ordered_session_paths=authorized.ordered_session_paths + (os.path.abspath("extra.csv"),))
    elif mutation == "duplicate_position":
        changed = replace(authorized, chronological_positions=(0,) * len(authorized.chronological_positions))
    else:
        changed = replace(authorized, chronological_positions=tuple(index + 1 for index in authorized.chronological_positions))
    with pytest.raises(ValueError, match="authorized_npm_runtime_authority_mismatch"):
        verify_guided_npm_authorized_runtime(
            replace(runtime, authorized_input=changed), claim.worker_request
        )


def test_missing_or_renamed_authorized_source_refuses(tmp_path):
    _, runtime = _runtime(tmp_path)
    path = Path(runtime.authorized_input.ordered_session_paths[0])
    renamed = path.with_name("renamed.csv")
    path.rename(renamed)
    with pytest.raises(ValueError, match="authorized_npm_source_unavailable"):
        load_guided_npm_authorized_chunk(
            runtime.authorized_input,
            os.fspath(path),
            runtime.config,
            0,
        )


def test_added_and_nested_csv_are_never_admitted_by_adapter(tmp_path):
    _, runtime = _runtime(tmp_path)
    root = Path(runtime.authorized_input.source_root_path)
    extra = root / "extra.csv"
    nested = root / "nested" / "nested.csv"
    extra.write_text("unexpected", encoding="utf-8")
    nested.parent.mkdir()
    nested.write_text("unexpected", encoding="utf-8")
    assert os.fspath(extra) not in runtime.authorized_input.ordered_session_paths
    assert os.fspath(nested) not in runtime.authorized_input.ordered_session_paths
    observed = []
    for index, path in enumerate(runtime.authorized_input.ordered_session_paths):
        observed.append(
            load_guided_npm_authorized_chunk(
                runtime.authorized_input, path, runtime.config, index
            ).source_file
        )
    assert tuple(observed) == runtime.authorized_input.ordered_session_paths


def test_explicit_parser_uses_frozen_timestamp_not_autodetected_candidate():
    content = _explicit_csv()
    chunk = load_npm_authorized_bytes(
        "/authority/session.csv",
        content,
        Config(target_fs_hz=2.0, chunk_duration_sec=2.0),
        0,
        contract=_explicit_contract(),
        resolved_timestamp_column="SystemTimestamp",
        reference_led_value=1,
        signal_led_value=2,
        physical_to_canonical_roi_mapping=(
            ("Region0G", "ROI_07"),
            ("Region4G", "ROI_02"),
        ),
    )
    assert chunk.metadata["npm_resolved_timestamp_column"] == "SystemTimestamp"
    assert chunk.metadata["npm_overlap_origin_absolute"] < 200.0


def test_parser_runtime_binds_complete_contract_and_led_authority(tmp_path):
    claim, runtime = _runtime(tmp_path)
    parser = claim.worker_request.execution_request.parser_runtime_projection
    authorized = runtime.authorized_input
    assert authorized.parser_contract_identity == parser.parser_policy_identity
    assert authorized.parser_contract_content_json == parser.parser_policy_content_json
    assert authorized.parser_contract.timestamp_column_candidates == parser.ordered_timestamp_candidates
    assert authorized.parser_contract.npm_led_col == parser.led_state_column
    assert authorized.parser_contract.npm_region_prefix == parser.roi_prefix
    assert authorized.parser_contract.npm_region_suffix == parser.roi_suffix
    assert authorized.parser_contract.adapter_value_nan_policy == parser.roi_value_nan_policy
    assert (authorized.reference_led_value, authorized.signal_led_value) == (
        parser.reference_led_value,
        parser.signal_led_value,
    )


def test_explicit_loader_uses_nondefault_led_values_when_authorized():
    content = _explicit_csv().replace(b",1,", b",UV,").replace(b",2,", b",SIG,")
    chunk = load_npm_authorized_bytes(
        "/authority/session.csv",
        content,
        Config(target_fs_hz=2.0, chunk_duration_sec=2.0),
        0,
        contract=_explicit_contract(),
        resolved_timestamp_column="SystemTimestamp",
        reference_led_value="UV",
        signal_led_value="SIG",
        physical_to_canonical_roi_mapping=(("Region0G", "ROI_07"), ("Region4G", "ROI_02")),
    )
    assert chunk.channel_names == ["ROI_07", "ROI_02"]


def test_explicit_physical_to_canonical_mapping_ignores_dataframe_column_order():
    chunk = load_npm_authorized_bytes(
        "/authority/session.csv",
        _explicit_csv(),
        Config(target_fs_hz=2.0, chunk_duration_sec=2.0),
        0,
        contract=_explicit_contract(),
        resolved_timestamp_column="SystemTimestamp",
        reference_led_value=1,
        signal_led_value=2,
        physical_to_canonical_roi_mapping=(
            ("Region0G", "ROI_07"),
            ("Region4G", "ROI_02"),
        ),
    )
    assert chunk.channel_names == ["ROI_07", "ROI_02"]
    assert chunk.metadata["roi_map"] == {
        "ROI_07": {"raw_col": "Region0G"},
        "ROI_02": {"raw_col": "Region4G"},
    }
    assert chunk.uv_raw[0, 0] == 11.0
    assert chunk.uv_raw[0, 1] == 41.0


@pytest.mark.parametrize(
    ("content", "mapping", "error"),
    [
        (_explicit_csv(missing_roi=True), (("Region0G", "ROI_07"), ("Region4G", "ROI_02")), "inventory"),
        (_explicit_csv(extra_roi=True), (("Region0G", "ROI_07"), ("Region4G", "ROI_02")), "inventory"),
        (_explicit_csv(), (("Region0G", "ROI_07"), ("Region0G", "ROI_02")), "mapping"),
        (_explicit_csv(), (("Region0G", "ROI_07"), ("Region4G", "ROI_07")), "mapping"),
    ],
)
def test_physical_inventory_and_mapping_mismatches_refuse(content, mapping, error):
    with pytest.raises(ValueError, match=error):
        load_npm_authorized_bytes(
            "/authority/session.csv",
            content,
            Config(target_fs_hz=2.0, chunk_duration_sec=2.0),
            0,
            contract=_explicit_contract(),
            resolved_timestamp_column="SystemTimestamp",
            reference_led_value=1,
            signal_led_value=2,
            physical_to_canonical_roi_mapping=mapping,
        )


def test_selected_scope_is_applied_after_complete_inventory_mapping(tmp_path):
    chunk = load_npm_authorized_bytes(
        "/authority/session.csv",
        _explicit_csv(),
        Config(target_fs_hz=2.0, chunk_duration_sec=2.0),
        0,
        contract=_explicit_contract(),
        resolved_timestamp_column="SystemTimestamp",
        reference_led_value=1,
        signal_led_value=2,
        physical_to_canonical_roi_mapping=(("Region0G", "ROI_07"), ("Region4G", "ROI_02")),
    )
    complete = tuple(chunk.channel_names)
    pipeline = Pipeline(Config(target_fs_hz=2.0, chunk_duration_sec=2.0))
    pipeline._selected_rois = ["ROI_07"]
    selected = pipeline._apply_roi_filter(chunk)
    assert complete == ("ROI_07", "ROI_02")
    assert tuple(selected.channel_names) == ("ROI_07",)
    assert "ROI_02" not in selected.channel_names


@pytest.mark.parametrize("field", ["canonical_roi_ids", "selected_canonical_roi_ids", "physical_to_canonical_roi_map"])
def test_unknown_or_missing_canonical_roi_authority_refuses(tmp_path, field):
    claim, runtime = _runtime(tmp_path)
    authorized = runtime.authorized_input
    value = {
        "canonical_roi_ids": ("UNKNOWN",),
        "selected_canonical_roi_ids": ("UNKNOWN",),
        "physical_to_canonical_roi_map": ((authorized.physical_roi_ids[0], "UNKNOWN"),),
    }[field]
    with pytest.raises(ValueError, match="authorized_npm_runtime_authority_mismatch"):
        verify_guided_npm_authorized_runtime(
            replace(runtime, authorized_input=replace(authorized, **{field: value})),
            claim.worker_request,
        )


def test_timing_authority_and_loaded_metadata_reconcile(tmp_path):
    claim, runtime = _runtime(tmp_path)
    authorized = runtime.authorized_input
    assert runtime.config.target_fs_hz == authorized.target_fs_hz
    assert runtime.config.chunk_duration_sec == authorized.configured_session_duration_sec
    chunk = load_guided_npm_authorized_chunk(
        authorized, authorized.ordered_session_paths[0], runtime.config, 0
    )
    session = authorized.ordered_sessions[0]
    assert chunk.metadata["npm_overlap_origin_absolute"] == session.overlap_origin_absolute
    assert chunk.metadata["npm_support_policy"] == session.support_policy
    assert authorized.sessions_per_hour == (
        claim.worker_request.execution_request.timing_runtime_projection.sessions_per_hour
    )
    assert authorized.gap_policy and authorized.overlap_policy and authorized.chronology_policy


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_fs_hz", 999.0),
        ("configured_session_duration_sec", 999.0),
        ("sessions_per_hour", 999),
        ("chronology_policy", "other"),
        ("gap_policy", "other"),
        ("overlap_policy", "other"),
        ("output_time_basis", "other"),
    ],
)
def test_timing_authority_tampering_refuses_pure_verification(tmp_path, field, value):
    claim, runtime = _runtime(tmp_path)
    with pytest.raises(ValueError, match="authorized_npm_runtime_authority_mismatch"):
        verify_guided_npm_authorized_runtime(
            replace(
                runtime,
                authorized_input=replace(runtime.authorized_input, **{field: value}),
            ),
            claim.worker_request,
        )


def test_mixed_correction_and_feature_authority_is_exact_per_selected_roi(tmp_path):
    runtime = _two_roi_runtime(tmp_path)
    selected = runtime.authorized_input.selected_canonical_roi_ids
    assert tuple(runtime.per_roi_correction) == selected
    assert tuple(runtime.per_roi_feature_config) == selected
    assert tuple(runtime.per_roi_feature_provenance) == selected
    assert tuple(
        spec.strategy_family for spec in runtime.per_roi_correction.values()
    ) == ("dynamic_fit", "signal_only_f0")


def test_config_audit_is_complete_explicit_and_has_no_unsafe_defaults(tmp_path):
    _, runtime = _runtime(tmp_path)
    audit = runtime.config_field_audit
    assert tuple(item.field_name for item in audit) == tuple(item.name for item in fields(Config))
    assert {item.classification for item in audit} <= {
        "frozen_worker_authority",
        "fixed_build_invariant",
        "irrelevant_to_guided_npm",
    }
    assert all(item.authority for item in audit)


def test_pipeline_authorized_api_is_mutually_exclusive_and_preserves_exact_list(monkeypatch, tmp_path):
    _, runtime = _runtime(tmp_path)
    pipeline = Pipeline(
        runtime.config,
        mode=runtime.mode,
        per_roi_correction=runtime.per_roi_correction,
        per_roi_feature_config=runtime.per_roi_feature_config,
        per_roi_feature_provenance=runtime.per_roi_feature_provenance,
    )
    observed = {}

    def fake_run(input_dir, output_dir, **kwargs):
        observed.update(input_dir=input_dir, output_dir=output_dir, kwargs=kwargs)
        observed["runtime"] = pipeline._guided_npm_authorized_runtime

    monkeypatch.setattr(pipeline, "run", fake_run)
    pipeline.run_guided_npm_authorized(
        runtime, runtime.authorized_input.run_directory_path
    )
    assert observed["runtime"] is runtime
    assert observed["input_dir"] == runtime.authorized_input.source_root_path
    assert observed["kwargs"] == {
        "force_format": "npm",
        "recursive": False,
        "glob_pattern": "__guided_npm_authorized_no_discovery__",
        "include_rois": None,
        "exclude_rois": None,
        "traces_only": False,
        "sessions_per_hour": runtime.authorized_input.sessions_per_hour,
    }
    assert pipeline._guided_npm_authorized_runtime is None


def test_full_authorized_pipeline_reaches_existing_numerical_path_without_discovery(
    monkeypatch, tmp_path
):
    _, runtime = _runtime(tmp_path)
    pipeline = Pipeline(
        runtime.config,
        mode=runtime.mode,
        per_roi_correction=runtime.per_roi_correction,
        per_roi_feature_config=runtime.per_roi_feature_config,
        per_roi_feature_provenance=runtime.per_roi_feature_provenance,
    )
    monkeypatch.setattr(
        pipeline,
        "discover_files",
        lambda *a, **k: pytest.fail("legacy discovery"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "sort_npm_files",
        lambda *a, **k: pytest.fail("legacy sorting"),
    )
    monkeypatch.setattr(
        adapters,
        "_load_npm",
        lambda *a, **k: pytest.fail("legacy NPM loader"),
    )
    pipeline.run_guided_npm_authorized(
        runtime, runtime.authorized_input.run_directory_path
    )
    output = Path(runtime.authorized_input.run_directory_path)
    assert pipeline.file_list == list(runtime.authorized_input.ordered_session_paths)
    assert pipeline.roi_selection["selected_rois"] == list(
        runtime.authorized_input.selected_canonical_roi_ids
    )
    assert (output / "phasic_trace_cache.h5").is_file()
    assert (output / "run_report.json").is_file()


def test_production_pipeline_uses_actual_mixed_gap_chronology_for_cache_and_features(
    monkeypatch, tmp_path
):
    runtime = _mixed_gap_runtime(tmp_path)
    pipeline = Pipeline(
        runtime.config,
        mode=runtime.mode,
        per_roi_correction=runtime.per_roi_correction,
        per_roi_feature_config=runtime.per_roi_feature_config,
        per_roi_feature_provenance=runtime.per_roi_feature_provenance,
    )
    monkeypatch.setattr(
        pipeline,
        "_resolve_legacy_chunk_elapsed_offset_sec",
        lambda *a, **k: pytest.fail("nominal/legacy offset inference"),
    )
    monkeypatch.setattr(
        adapters,
        "_parse_vendor_npm_timestamp",
        lambda *a, **k: pytest.fail("filename chronology inference"),
    )
    monkeypatch.setattr(
        adapters,
        "resolve_npm_support_geometry",
        lambda *a, **k: pytest.fail("source timestamp chronology reconstruction"),
    )
    observed_event_times = []

    def feature_sink(chunk, config, per_roi_config=None):
        event_index = int(np.argmin(np.abs(chunk.time_sec - (chunk.time_sec[0] + 5.0))))
        observed_event_times.append(float(chunk.time_sec[event_index]))
        return pd.DataFrame(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "roi": chunk.channel_names[0],
                    "mean": 1.0,
                    "median": 1.0,
                    "std": 1.0,
                    "mad": 1.0,
                    "peak_count": 1,
                    "auc": 1.0,
                }
            ]
        )

    monkeypatch.setattr(
        pipeline_module.feature_extraction, "extract_features", feature_sink
    )
    pipeline.run_guided_npm_authorized(
        runtime, runtime.authorized_input.run_directory_path
    )

    cache_path = Path(runtime.authorized_input.run_directory_path) / "phasic_trace_cache.h5"
    roi = runtime.authorized_input.selected_canonical_roi_ids[0]
    starts = []
    combined = []
    with h5py.File(cache_path, "r") as cache:
        for chunk_id in range(3):
            group = cache[f"roi/{roi}/chunk_{chunk_id}"]
            time_sec = group["time_sec"][()]
            starts.append(float(time_sec[0]))
            combined.extend(float(value) for value in time_sec)
            assert group.attrs["guided_npm_chronological_position"] == chunk_id
            assert group.attrs["guided_npm_actual_elapsed_sec"] == (0.0, 90.0, 150.0)[chunk_id]
            assert group.attrs["guided_npm_nominal_expected_elapsed_sec"] == (0.0, 60.0, 120.0)[chunk_id]
            assert group.attrs["guided_npm_cross_session_time_authority"] == "frozen_worker_projection"
    assert starts == [0.0, 90.0, 150.0]
    assert starts != [0.0, 60.0, 120.0]
    assert observed_event_times == [5.0, 95.0, 155.0]
    assert not any(6.0 < value < 90.0 for value in combined)
    assert not any(96.0 < value < 150.0 for value in combined)

    report = json.loads(
        (Path(runtime.authorized_input.run_directory_path) / "run_report.json").read_text()
    )
    chronology = report["derived_settings"]["guided_npm_cross_session_chronology"]
    assert [item["actual_elapsed_sec"] for item in chronology["sessions"]] == [0.0, 90.0, 150.0]
    assert chronology["sessions_per_hour_role"] == "nominal_cadence_only"


def test_authorized_chunk_metadata_is_exact_before_and_after_pipeline_binding(tmp_path):
    runtime = _mixed_gap_runtime(tmp_path)
    pipeline = Pipeline(runtime.config)
    pipeline._guided_npm_authorized_runtime = runtime
    chunk = pipeline._load_entry_chunk(
        runtime.authorized_input.ordered_session_paths[1], 1, "npm"
    )
    assert chunk.time_sec[0] == 90.0
    assert chunk.metadata["guided_npm_chronological_position"] == 1
    assert chunk.metadata["guided_npm_actual_elapsed_sec"] == 90.0
    assert chunk.metadata["guided_npm_nominal_expected_elapsed_sec"] == 60.0
    assert chunk.metadata["guided_npm_authoritative_source_start_time"] == "2030-01-01T00:00:00"
    assert chunk.metadata["guided_npm_within_session_start_sec"] == 0.0
    assert chunk.metadata["guided_npm_recording_time_start_sec"] == 90.0


@pytest.mark.parametrize(
    "mutation",
    [
        "actual_value",
        "actual_order",
        "actual_length",
        "position",
        "path_pairing",
        "nominal_as_actual",
        "nonfinite",
        "decreasing",
        "duplicate",
    ],
)
def test_authorized_chronology_semantic_tampering_refuses(tmp_path, mutation):
    runtime = _mixed_gap_runtime(tmp_path)
    authorized = runtime.authorized_input
    changes = {}
    if mutation == "actual_value":
        changes["actual_elapsed_sec_by_chunk"] = (0.0, 91.0, 150.0)
    elif mutation == "actual_order":
        changes["actual_elapsed_sec_by_chunk"] = (0.0, 150.0, 90.0)
    elif mutation == "actual_length":
        changes["actual_elapsed_sec_by_chunk"] = (0.0, 90.0)
    elif mutation == "position":
        sessions = list(authorized.ordered_sessions)
        sessions[1] = replace(sessions[1], chronological_position=2)
        changes["ordered_sessions"] = tuple(sessions)
    elif mutation == "path_pairing":
        changes["ordered_session_paths"] = (
            authorized.ordered_session_paths[1],
            authorized.ordered_session_paths[0],
            authorized.ordered_session_paths[2],
        )
    elif mutation == "nominal_as_actual":
        changes["actual_elapsed_sec_by_chunk"] = authorized.nominal_expected_elapsed_sec_by_chunk
    elif mutation == "nonfinite":
        sessions = list(authorized.ordered_sessions)
        sessions[1] = replace(sessions[1], actual_elapsed_sec=float("nan"))
        changes["ordered_sessions"] = tuple(sessions)
        changes["actual_elapsed_sec_by_chunk"] = (0.0, float("nan"), 150.0)
    elif mutation == "decreasing":
        sessions = list(authorized.ordered_sessions)
        sessions[2] = replace(sessions[2], actual_elapsed_sec=80.0)
        changes["ordered_sessions"] = tuple(sessions)
        changes["actual_elapsed_sec_by_chunk"] = (0.0, 90.0, 80.0)
    else:
        sessions = list(authorized.ordered_sessions)
        sessions[2] = replace(sessions[2], actual_elapsed_sec=90.0)
        changes["ordered_sessions"] = tuple(sessions)
        changes["actual_elapsed_sec_by_chunk"] = (0.0, 90.0, 90.0)
    with pytest.raises(ValueError, match="authorized_npm_chronology_invalid"):
        verify_guided_npm_authorized_input(replace(authorized, **changes))


def test_runtime_chronology_remains_bound_to_its_worker(tmp_path):
    claim_a, runtime_a = _runtime(tmp_path)
    worker_b = _worker_with_changed_actual_chronology(
        claim_a.worker_request, (0.0, 900.0)
    )
    runtime_b = build_guided_npm_authorized_runtime(worker_b)
    assert runtime_a.authorized_input.ordered_session_paths == (
        runtime_b.authorized_input.ordered_session_paths
    )
    assert runtime_a.authorized_input.actual_elapsed_sec_by_chunk == (0.0, 600.0)
    assert runtime_b.authorized_input.actual_elapsed_sec_by_chunk == (0.0, 900.0)
    verify_guided_npm_authorized_runtime(runtime_a, claim_a.worker_request)
    verify_guided_npm_authorized_runtime(runtime_b, worker_b)
    with pytest.raises(ValueError, match="authorized_npm_runtime_authority_mismatch"):
        verify_guided_npm_authorized_runtime(runtime_a, worker_b)
