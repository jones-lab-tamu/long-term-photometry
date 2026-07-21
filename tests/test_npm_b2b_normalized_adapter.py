from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.guided_normalized_recording import (
    NormalizedRecordingError,
    build_npm_normalized_recording_description,
    compute_normalized_recording_description_identity,
    deserialize_normalized_recording_description,
    serialize_normalized_recording_description,
)
from photometry_pipeline.guided_backend_validation_materialization import (
    classify_npm_cadence_intervals,
)
from photometry_pipeline.io.adapters import load_chunk
from photometry_pipeline.io.npm_contract import (
    NpmParserContract,
    inspect_npm_csv,
)
from photometry_pipeline.io.npm_source_snapshot import (
    NpmSourceSnapshotError,
    build_npm_source_candidate_snapshot,
)


def _write_staggered(
    path: Path,
    physical_roi_columns: tuple[str, ...] = ("Region0G",),
) -> None:
    rows = (
        (0, 100.0, 1, 10.0),
        (1, 100.5, 2, 100.0),
        (2, 101.0, 1, 11.0),
        (3, 101.5, 2, 101.0),
        (4, 102.0, 1, 12.0),
        (5, 102.5, 2, 102.0),
    )
    lines = [
        ",".join(("FrameCounter", "Timestamp", "LedState", *physical_roi_columns))
    ]
    for frame_counter, timestamp, led_state, value in rows:
        lines.append(
            ",".join(
                (
                    str(frame_counter),
                    str(timestamp),
                    str(led_state),
                    *(str(value) for _ in physical_roi_columns),
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config(*, strict: bool) -> Config:
    return Config(
        target_fs_hz=2.0,
        chunk_duration_sec=2.0,
        allow_partial_final_chunk=not strict,
        npm_time_axis="system_timestamp",
        npm_system_ts_col="SystemTimestamp",
        npm_computer_ts_col="ComputerTimestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
        adapter_value_nan_policy="strict",
        timestamp_cv_max=0.02,
    )


def test_npm_overlap_origin_grid_is_preserved(tmp_path: Path):
    path = tmp_path / "photometryData2025-03-05T15_37_44.csv"
    _write_staggered(path)

    strict = load_chunk(str(path), "npm", _config(strict=True), 0)
    permissive = load_chunk(str(path), "npm", _config(strict=False), 0)

    np.testing.assert_allclose(strict.time_sec, [0.5, 1.0])
    assert strict.metadata["output_time_basis"] == (
        "relative_seconds_since_uv_signal_overlap_origin"
    )
    assert strict.metadata["npm_overlap_origin_absolute"] == 100.5
    assert strict.metadata["npm_resolved_support_start_offset_sec"] == 0.5
    assert strict.metadata["npm_resolved_support_end_offset_sec"] == 1.0
    np.testing.assert_allclose(permissive.time_sec, [0.0, 0.5, 1.0, 1.5])
    assert permissive.metadata["npm_resolved_support_start_offset_sec"] == 0.0


def test_npm_snapshot_orders_by_filename_timestamp_and_refuses_duplicates(tmp_path: Path):
    for name in (
        "z2025-03-05T15_37_46.csv",
        "a2025-03-05T15_37_44.csv",
        "m2025-03-05T15_37_45.csv",
    ):
        _write_staggered(tmp_path / name)
    snapshot = build_npm_source_candidate_snapshot(str(tmp_path))
    assert [item.canonical_relative_path for item in snapshot.candidates] == [
        "a2025-03-05t15_37_44.csv",
        "m2025-03-05t15_37_45.csv",
        "z2025-03-05t15_37_46.csv",
    ]
    assert [item.authoritative_source_start_time for item in snapshot.candidates] == [
        "2025-03-05T15:37:44",
        "2025-03-05T15:37:45",
        "2025-03-05T15:37:46",
    ]

    _write_staggered(tmp_path / "duplicate2025-03-05T15_37_44.csv")
    with pytest.raises(NpmSourceSnapshotError) as exc:
        build_npm_source_candidate_snapshot(str(tmp_path))
    assert exc.value.category == "duplicate_filename_timestamp"


def test_npm_parser_policy_is_immutable_and_identity_bound(tmp_path: Path):
    path = tmp_path / "photometryData2025-03-05T15_37_44.csv"
    _write_staggered(path)
    contract = NpmParserContract.from_config(_config(strict=True))
    content = contract.content()
    inspection = inspect_npm_csv(str(path), contract)
    snapshot = build_npm_source_candidate_snapshot(str(tmp_path))
    description = build_npm_normalized_recording_description(
        source_snapshot=snapshot,
        session_inspections={snapshot.candidates[0].canonical_relative_path: inspection},
        parser_contract_content=content,
        session_duration_sec=2.0,
        sessions_per_hour=1,
        discovered_roi_ids=inspection.roi_ids,
        included_roi_ids=inspection.roi_ids,
        target_fs_hz=2.0,
    )
    assert description.sampling.parser_contract_content is not content
    with pytest.raises(TypeError):
        description.sampling.parser_contract_content["sampling"] = {}  # type: ignore[index]
    assert description.roi_channels[0].signal_channel_identity == (
        'npm-channel:v1:{"selector":{"column":"LedState","operator":"eq","value":2},"source_column":"Region0G"}'
    )

    serialized = serialize_normalized_recording_description(description)
    restored = deserialize_normalized_recording_description(serialized)
    assert restored == description
    tampered = dict(serialized)
    tampered["sampling"] = dict(serialized["sampling"])
    tampered["sampling"]["parser_contract_content"] = dict(
        serialized["sampling"]["parser_contract_content"]
    )
    tampered["sampling"]["parser_contract_content"]["sampling"] = {}
    with pytest.raises(NormalizedRecordingError) as exc:
        deserialize_normalized_recording_description(tampered)
    assert exc.value.category == "parser_contract_content_digest_mismatch"


def _build_npm_description_for_columns(
    tmp_path: Path,
    physical_roi_columns: tuple[str, ...],
):
    path = tmp_path / "photometryData2025-03-05T15_37_44.csv"
    _write_staggered(path, physical_roi_columns)
    contract = NpmParserContract.from_config(_config(strict=True))
    snapshot = build_npm_source_candidate_snapshot(str(tmp_path))
    inspection = inspect_npm_csv(str(path), contract)
    description = build_npm_normalized_recording_description(
        source_snapshot=snapshot,
        session_inspections={snapshot.candidates[0].canonical_relative_path: inspection},
        parser_contract_content=contract.content(),
        session_duration_sec=2.0,
        sessions_per_hour=1,
        discovered_roi_ids=inspection.roi_ids,
        included_roi_ids=inspection.roi_ids,
        target_fs_hz=2.0,
    )
    return description, snapshot, inspection, contract


def test_npm_physical_roi_columns_are_naturally_ordered_before_canonicalization(
    tmp_path: Path,
):
    description, _snapshot, inspection, _contract = _build_npm_description_for_columns(
        tmp_path,
        ("Region10G", "Region2G"),
    )

    assert inspection.roi_columns == ("Region2G", "Region10G")
    assert inspection.roi_ids == ("Region0", "Region1")
    assert inspection.physical_to_canonical_roi_mapping == (
        ("Region0", "Region2G"),
        ("Region1", "Region10G"),
    )
    assert [channel.source_column for channel in description.roi_channels] == [
        "Region2G",
        "Region10G",
    ]
    assert '"source_column":"Region2G"' in (
        description.roi_channels[0].signal_channel_identity
    )
    assert '"source_column":"Region10G"' in (
        description.roi_channels[1].reference_channel_identity
    )


def test_npm_physical_roi_mapping_must_match_across_sessions(tmp_path: Path):
    first = tmp_path / "photometryData2025-03-05T15_37_44.csv"
    second = tmp_path / "photometryData2025-03-05T15_37_45.csv"
    _write_staggered(first, ("Region2G", "Region10G"))
    _write_staggered(second, ("Region2G", "Region11G"))
    contract = NpmParserContract.from_config(_config(strict=True))
    snapshot = build_npm_source_candidate_snapshot(str(tmp_path))
    inspections = {
        item.canonical_relative_path: inspect_npm_csv(
            str(tmp_path / item.canonical_relative_path), contract
        )
        for item in snapshot.candidates
    }

    with pytest.raises(NormalizedRecordingError) as exc:
        build_npm_normalized_recording_description(
            source_snapshot=snapshot,
            session_inspections=inspections,
            parser_contract_content=contract.content(),
            session_duration_sec=2.0,
            sessions_per_hour=3600,
            discovered_roi_ids=("Region0", "Region1"),
            included_roi_ids=("Region0", "Region1"),
            target_fs_hz=2.0,
        )
    assert exc.value.category == "npm_physical_roi_mapping_mismatch"


def test_npm_physical_source_column_is_identity_bearing(tmp_path: Path):
    description, _snapshot, _inspection, _contract = _build_npm_description_for_columns(
        tmp_path,
        ("Region2G", "Region10G"),
    )
    changed = replace(
        description,
        roi_channels=(
            description.roi_channels[0],
            replace(
                description.roi_channels[1],
                source_column="Region11G",
                signal_channel_identity=description.roi_channels[
                    1
                ].signal_channel_identity.replace("Region10G", "Region11G"),
                reference_channel_identity=description.roi_channels[
                    1
                ].reference_channel_identity.replace("Region10G", "Region11G"),
            ),
        ),
        adapter_evidence={
            **description.adapter_evidence,
            "physical_to_canonical_roi_mapping":[
                {
                    "canonical_roi_id": "Region0",
                    "physical_source_column": "Region2G",
                },
                {
                    "canonical_roi_id": "Region1",
                    "physical_source_column": "Region11G",
                },
            ],
        },
    )
    assert compute_normalized_recording_description_identity(changed) != (
        compute_normalized_recording_description_identity(description)
    )


def test_npm_channel_validation_refuses_canonical_source_reconstruction(
    tmp_path: Path,
):
    description, _snapshot, _inspection, _contract = _build_npm_description_for_columns(
        tmp_path,
        ("Region2G", "Region10G"),
    )
    tampered_channel = replace(
        description.roi_channels[0],
        source_column="Region0G",
        signal_channel_identity=description.roi_channels[
            0
        ].signal_channel_identity.replace("Region2G", "Region0G"),
        reference_channel_identity=description.roi_channels[
            0
        ].reference_channel_identity.replace("Region2G", "Region0G"),
    )
    with pytest.raises(NormalizedRecordingError) as exc:
        replace(
            description,
            roi_channels=(tampered_channel, description.roi_channels[1]),
        )
    assert exc.value.category == "npm_physical_roi_mapping_mismatch"


def test_npm_channel_selector_uses_authorized_led_column(tmp_path: Path):
    path = tmp_path / "photometryData2025-03-05T15_37_44.csv"
    path.write_text(
        "Timestamp,State,Region2G\n"
        "100.0,1,10.0\n"
        "100.5,2,100.0\n"
        "101.0,1,11.0\n"
        "101.5,2,101.0\n"
        "102.0,1,12.0\n"
        "102.5,2,102.0\n",
        encoding="utf-8",
    )
    config = replace(_config(strict=True), npm_led_col="State")
    contract = NpmParserContract.from_config(config)
    snapshot = build_npm_source_candidate_snapshot(str(tmp_path))
    inspection = inspect_npm_csv(str(path), contract)
    description = build_npm_normalized_recording_description(
        source_snapshot=snapshot,
        session_inspections={snapshot.candidates[0].canonical_relative_path: inspection},
        parser_contract_content=contract.content(),
        session_duration_sec=2.0,
        sessions_per_hour=1,
        discovered_roi_ids=inspection.roi_ids,
        included_roi_ids=inspection.roi_ids,
        target_fs_hz=2.0,
    )

    assert '"column":"State"' in description.roi_channels[0].signal_channel_identity


def test_npm_normalized_description_preserves_actual_and_nominal_timeline(
    tmp_path: Path,
):
    for name in (
        "photometryData2025-03-05T15_37_44.csv",
        "photometryData2025-03-05T15_39_14.csv",
        "photometryData2025-03-05T15_40_14.csv",
    ):
        _write_staggered(tmp_path / name, ("Region2G", "Region10G"))
    contract = NpmParserContract.from_config(_config(strict=True))
    snapshot = build_npm_source_candidate_snapshot(str(tmp_path))
    inspections = {
        item.canonical_relative_path: inspect_npm_csv(
            str(tmp_path / item.canonical_relative_path), contract
        )
        for item in snapshot.candidates
    }

    description = build_npm_normalized_recording_description(
        source_snapshot=snapshot,
        session_inspections=inspections,
        parser_contract_content=contract.content(),
        session_duration_sec=20.0,
        sessions_per_hour=60,
        discovered_roi_ids=("Region0", "Region1"),
        included_roi_ids=("Region0", "Region1"),
        target_fs_hz=2.0,
    )
    assert [session.authoritative_source_start_time for session in description.sessions] == [
        "2025-03-05T15:37:44",
        "2025-03-05T15:39:14",
        "2025-03-05T15:40:14",
    ]
    assert [session.expected_timeline_start_time for session in description.sessions] == [
        "2025-03-05T15:37:44",
        "2025-03-05T15:38:44",
        "2025-03-05T15:39:44",
    ]
    assert [item["actual_elapsed_sec"] for item in description.adapter_evidence["npm_sessions"]] == [
        0.0,
        90.0,
        150.0,
    ]
    assert [
        item["nominal_expected_elapsed_sec"]
        for item in description.adapter_evidence["npm_sessions"]
    ] == [0.0, 60.0, 120.0]


@pytest.mark.parametrize(
    ("starts", "classifications"),
    (
        ((0, 90, 150), ("npm_schedule_gap", "nominal")),
        ((0, 30, 90), ("npm_early_session", "nominal")),
        ((0, 90, 130), ("npm_schedule_gap", "npm_early_session")),
        ((0, 60, 120), ("nominal", "nominal")),
    ),
)
def test_npm_cadence_classification_uses_adjacent_intervals(starts, classifications):
    evidence = classify_npm_cadence_intervals(
        tuple(datetime.fromtimestamp(value) for value in starts),
        nominal_interval_sec=60.0,
    )
    assert tuple(item.classification for item in evidence) == classifications
    assert tuple(item.actual_interval_sec for item in evidence) == tuple(
        right - left for left, right in zip(starts, starts[1:])
    )
