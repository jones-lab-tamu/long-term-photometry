from __future__ import annotations

import builtins
from dataclasses import replace
import inspect
import json
import math
from pathlib import Path

import pytest

import photometry_pipeline.guided_continuous_rwd_recording as subject
from photometry_pipeline.guided_continuous_rwd_recording import (
    ACQUISITION_MODE,
    EXECUTION_ADMISSION_STATUS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SOURCE_FORMAT,
    TIME_BASIS,
    UNRESOLVED_ADMISSION_CHECKS,
    ContinuousRwdRecordingAuthorityError,
    build_guided_continuous_rwd_recording_description,
    deserialize_guided_continuous_rwd_recording_description,
    serialize_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.io.rwd_continuous_source import (
    CADENCE_EVIDENCE_POLICY_VERSION,
    INSPECTION_CONTRACT_NAME,
    INSPECTION_CONTRACT_VERSION,
    ContinuousRwdCadenceQuantile,
    ContinuousRwdChannelEvidence,
    ContinuousRwdInspectionResult,
    ContinuousRwdIntervalEvidence,
    ContinuousRwdParserFacts,
    ContinuousRwdRoiPair,
    ContinuousRwdSourceIdentity,
    ContinuousRwdTimeAxisEvidence,
)


def _inspection() -> ContinuousRwdInspectionResult:
    duration = 625.0
    return ContinuousRwdInspectionResult(
        contract_name=INSPECTION_CONTRACT_NAME,
        contract_version=INSPECTION_CONTRACT_VERSION,
        status="completed",
        outcome_category="inspection_completed",
        scientist_summary="Inspection completed.",
        source_identity=ContinuousRwdSourceIdentity(
            identity_policy_version="continuous-rwd-source-identity.v1",
            selected_folder_canonical="C:\\data\\recording",
            fluorescence_path_canonical="C:\\data\\recording\\Fluorescence.csv",
            file_size_bytes=123456,
            modification_time_ns=987654321,
            sha256="a" * 64,
            stable_source_identity="b" * 64,
        ),
        parser_facts=ContinuousRwdParserFacts(
            header_row_index=1,
            time_column="TimeStamp",
            raw_columns=("TimeStamp", "1-410", "1-470", "2-410", "2-470", "10-410", "10-470"),
            timestamp_unit="milliseconds",
            timestamp_scale_to_seconds=0.001,
        ),
        time_axis=ContinuousRwdTimeAxisEvidence(
            total_data_row_count=12501,
            valid_timestamp_count=12501,
            raw_first_timestamp=1000.0,
            raw_last_timestamp=626000.0,
            normalized_first_seconds=0.0,
            normalized_last_seconds=duration,
            measured_duration_seconds=duration,
            minimum_duration_seconds=600.0,
            duration_product_classification="meets_product_minimum",
            positive_interval_count=12500,
            nominal_cadence_seconds=0.05,
            minimum_positive_dt_seconds=0.049,
            maximum_positive_dt_seconds=0.08,
            mean_positive_dt_seconds=0.05,
            standard_deviation_positive_dt_seconds=0.001,
            coefficient_of_variation=0.02,
            quantiles=(
                ContinuousRwdCadenceQuantile(0.001, 0.049),
                ContinuousRwdCadenceQuantile(0.5, 0.05),
                ContinuousRwdCadenceQuantile(0.999, 0.08),
            ),
            quantile_method="deterministic_reservoir_linear.v1",
            quantile_sample_count=12500,
            duplicate_timestamp_count=0,
            backward_timestamp_count=0,
            nonnumeric_timestamp_count=0,
            nonfinite_timestamp_count=0,
            unusually_long_interval_count=2,
            unusually_short_interval_count=2,
            largest_unusual_intervals=(
                ContinuousRwdIntervalEvidence(900, 0.08, 1.6),
                ContinuousRwdIntervalEvidence(850, 0.075, 1.5),
            ),
            smallest_unusual_intervals=(
                ContinuousRwdIntervalEvidence(400, 0.02, 0.4),
                ContinuousRwdIntervalEvidence(450, 0.025, 0.5),
            ),
            cadence_evidence_policy_version=CADENCE_EVIDENCE_POLICY_VERSION,
        ),
        channels=ContinuousRwdChannelEvidence(
            roi_pairs=(
                ContinuousRwdRoiPair("10", "10-410", "10-470"),
                ContinuousRwdRoiPair("2", "2-410", "2-470"),
                ContinuousRwdRoiPair("1", "1-410", "1-470"),
            ),
            unmatched_channel_columns=(),
            selected_value_count=75006,
            nonnumeric_selected_value_count=0,
            nonfinite_selected_value_count=0,
            malformed_row_count=0,
        ),
        findings=(),
        source_stable=True,
        full_file_passes=2,
    )


def _build(inspection=None, included=("2", "1")):
    return build_guided_continuous_rwd_recording_description(
        inspection or _inspection(), included_roi_ids=included
    )


def test_valid_construction_freezes_exact_authority_and_pending_admission():
    description = _build()
    assert (
        description.schema_name,
        description.schema_version,
        description.source_format,
        description.acquisition_mode,
    ) == (SCHEMA_NAME, SCHEMA_VERSION, SOURCE_FORMAT, ACQUISITION_MODE)
    assert description.execution_admission_status == EXECUTION_ADMISSION_STATUS
    assert description.unresolved_admission_checks == UNRESOLVED_ADMISSION_CHECKS
    assert description.source.sha256 == "a" * 64
    assert description.source.file_size_bytes == 123456
    assert description.source.total_data_row_count == 12501
    assert description.source.raw_columns[0] == "TimeStamp"
    assert tuple(item.roi_id for item in description.roi.available_roi_channels) == ("1", "10", "2")
    assert description.roi.included_roi_ids == ("1", "2")
    assert description.roi.excluded_roi_ids == ("10",)
    assert description.roi.available_roi_channels[0].reference_column == "1-410"
    assert description.roi.available_roi_channels[0].signal_column == "1-470"
    assert description.time.normalized_elapsed_origin_seconds == 0.0
    assert description.time.normalized_elapsed_end_seconds == 625.0
    assert description.time.measured_support_start_seconds == 0.0
    assert description.time.measured_support_end_seconds == 625.0
    assert description.time.time_basis == TIME_BASIS


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: replace(value, contract_name="wrong"), "contract name"),
        (lambda value: replace(value, contract_version="v999"), "contract version"),
        (lambda value: replace(value, status="failed"), "completed successfully"),
        (lambda value: replace(value, outcome_category="inspection_incomplete"), "completed successfully"),
        (lambda value: replace(value, source_stable=False), "stable"),
        (lambda value: replace(value, source_identity=None), "incomplete"),
        (lambda value: replace(value, parser_facts=None), "incomplete"),
        (lambda value: replace(value, time_axis=None), "incomplete"),
        (lambda value: replace(value, channels=None), "incomplete"),
        (lambda value: replace(value, time_axis=replace(value.time_axis, measured_duration_seconds=599.9)), "600-second"),
        (lambda value: replace(value, time_axis=replace(value.time_axis, duplicate_timestamp_count=1)), "duplicate"),
        (lambda value: replace(value, time_axis=replace(value.time_axis, backward_timestamp_count=1)), "backward"),
        (lambda value: replace(value, time_axis=replace(value.time_axis, nonnumeric_timestamp_count=1)), "nonnumeric timestamp"),
        (lambda value: replace(value, time_axis=replace(value.time_axis, nonfinite_timestamp_count=1)), "nonfinite timestamp"),
        (lambda value: replace(value, channels=replace(value.channels, malformed_row_count=1)), "malformed row"),
        (lambda value: replace(value, channels=replace(value.channels, nonnumeric_selected_value_count=1)), "nonnumeric selected"),
        (lambda value: replace(value, channels=replace(value.channels, nonfinite_selected_value_count=1)), "nonfinite selected"),
        (lambda value: replace(value, channels=replace(value.channels, unmatched_channel_columns=("3-470",))), "unmatched"),
        (lambda value: replace(value, channels=replace(value.channels, roi_pairs=())), "ROI pair"),
    ],
)
def test_refuses_ineligible_or_incomplete_inspection(change, message):
    with pytest.raises(ContinuousRwdRecordingAuthorityError, match=message):
        _build(change(_inspection()))


@pytest.mark.parametrize(
    ("time_changes", "parser_changes", "message"),
    [
        ({"normalized_first_seconds": 1.0}, {}, "origin"),
        ({"normalized_last_seconds": 624.0}, {}, "end"),
        (
            {
                "measured_duration_seconds": 624.5,
                "normalized_last_seconds": 624.5,
            },
            {},
            "raw timestamps",
        ),
        ({"valid_timestamp_count": 12502}, {}, "exceed"),
        ({"total_data_row_count": 12502}, {}, "row counts"),
        ({"positive_interval_count": 12499}, {}, "minus one"),
        ({"raw_first_timestamp": math.inf}, {}, "finite"),
        ({"raw_last_timestamp": 999.0}, {}, "earlier"),
        ({}, {"timestamp_scale_to_seconds": 0.0}, "positive"),
    ],
)
def test_refuses_internally_incoherent_cr1a_time_evidence(
    time_changes, parser_changes, message
):
    inspection = _inspection()
    inspection = replace(
        inspection,
        time_axis=replace(inspection.time_axis, **time_changes),
        parser_facts=replace(inspection.parser_facts, **parser_changes),
    )
    with pytest.raises(ContinuousRwdRecordingAuthorityError, match=message):
        _build(inspection)


def _cadence_inspection(**changes):
    inspection = _inspection()
    return replace(
        inspection,
        time_axis=replace(inspection.time_axis, **changes),
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"nominal_cadence_seconds": 0.0}, "greater than zero"),
        ({"nominal_cadence_seconds": -0.05}, "greater than zero"),
        ({"minimum_positive_dt_seconds": 0.0}, "greater than zero"),
        ({"maximum_positive_dt_seconds": 0.0}, "greater than zero"),
        ({"mean_positive_dt_seconds": 0.0}, "greater than zero"),
        ({"nominal_cadence_seconds": 0.081}, "Nominal"),
        ({"mean_positive_dt_seconds": 0.081}, "Mean"),
        ({"standard_deviation_positive_dt_seconds": -0.001}, "standard deviation"),
        ({"coefficient_of_variation": -0.02}, "coefficient"),
        ({"quantile_sample_count": 12501}, "sample count"),
        ({"quantiles": ()}, "At least one"),
        (
            {"quantiles": (ContinuousRwdCadenceQuantile(-0.1, 0.049),)},
            "between zero and one",
        ),
        (
            {"quantiles": (ContinuousRwdCadenceQuantile(1.1, 0.049),)},
            "between zero and one",
        ),
        (
            {
                "quantiles": (
                    ContinuousRwdCadenceQuantile(0.001, 0.06),
                    ContinuousRwdCadenceQuantile(0.5, 0.05),
                )
            },
            "nondecreasing",
        ),
        (
            {"quantiles": (ContinuousRwdCadenceQuantile(0.5, 0.048),)},
            "between minimum and maximum",
        ),
        ({"unusually_long_interval_count": 0}, "cannot exceed"),
        (
            {
                "largest_unusual_intervals": (
                    ContinuousRwdIntervalEvidence(900, 0.0, 1.6),
                )
            },
            "dt must be greater than zero",
        ),
        (
            {
                "smallest_unusual_intervals": (
                    ContinuousRwdIntervalEvidence(400, 0.02, 0.0),
                )
            },
            "multiple must be greater than zero",
        ),
    ],
)
def test_refuses_internally_incoherent_cr1a_cadence_evidence(changes, message):
    with pytest.raises(ContinuousRwdRecordingAuthorityError, match=message):
        _build(_cadence_inspection(**changes))


def test_refuses_cr1a_quantiles_supplied_out_of_probability_order():
    inspection = _inspection()
    inspection = _cadence_inspection(
        quantiles=tuple(reversed(inspection.time_axis.quantiles))
    )

    with pytest.raises(
        ContinuousRwdRecordingAuthorityError,
        match="strictly ascending",
    ):
        _build(inspection)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("largest_unusual_intervals", "Largest unusual intervals"),
        ("smallest_unusual_intervals", "Smallest unusual intervals"),
    ],
)
def test_refuses_cr1a_unusual_intervals_supplied_in_wrong_order(field, message):
    inspection = _inspection()
    supplied = getattr(inspection.time_axis, field)
    assert len(supplied) == 2
    inspection = _cadence_inspection(**{field: tuple(reversed(supplied))})

    with pytest.raises(ContinuousRwdRecordingAuthorityError, match=message):
        _build(inspection)


def test_correctly_ordered_cr1a_cadence_builds_and_round_trips_unchanged():
    inspection = _inspection()
    description = _build(inspection)
    restored = deserialize_guided_continuous_rwd_recording_description(
        serialize_guided_continuous_rwd_recording_description(description)
    )

    assert tuple(
        (item.probability, item.dt_seconds) for item in description.cadence.quantiles
    ) == tuple(
        (item.probability, item.dt_seconds) for item in inspection.time_axis.quantiles
    )
    assert tuple(
        (item.row_index, item.dt_seconds)
        for item in description.cadence.largest_unusual_intervals
    ) == tuple(
        (item.row_index, item.dt_seconds)
        for item in inspection.time_axis.largest_unusual_intervals
    )
    assert tuple(
        (item.row_index, item.dt_seconds)
        for item in description.cadence.smallest_unusual_intervals
    ) == tuple(
        (item.row_index, item.dt_seconds)
        for item in inspection.time_axis.smallest_unusual_intervals
    )
    assert restored == description


@pytest.mark.parametrize(
    ("included", "message"),
    [
        ((), "At least one"),
        (("missing",), "exist"),
        (("1", "1"), "Duplicate"),
    ],
)
def test_refuses_invalid_roi_choices(included, message):
    with pytest.raises(ContinuousRwdRecordingAuthorityError, match=message):
        _build(included=included)


@pytest.mark.parametrize(
    "pairs",
    [
        (
            ContinuousRwdRoiPair("1", "1-410", "1-470"),
            ContinuousRwdRoiPair("1", "2-410", "2-470"),
        ),
        (
            ContinuousRwdRoiPair("1", "same-410", "1-470"),
            ContinuousRwdRoiPair("2", "same-410", "2-470"),
        ),
        (
            ContinuousRwdRoiPair("1", "1-410", "same-470"),
            ContinuousRwdRoiPair("2", "2-410", "same-470"),
        ),
        (
            ContinuousRwdRoiPair("1", "shared", "1-470"),
            ContinuousRwdRoiPair("2", "2-410", "shared"),
        ),
    ],
)
def test_refuses_duplicate_roi_ids_and_reused_columns(pairs):
    inspection = _inspection()
    inspection = replace(inspection, channels=replace(inspection.channels, roi_pairs=pairs))
    with pytest.raises(ContinuousRwdRecordingAuthorityError):
        _build(inspection, included=(pairs[0].roi_id,))


def test_roi_order_is_canonical_and_inclusion_is_identity_bearing():
    first = _build(included=("2", "1"))
    reordered = _build(included=("1", "2"))
    changed = _build(included=("1",))
    assert first == reordered
    assert first.recording_identity == reordered.recording_identity
    assert first.recording_identity != changed.recording_identity


def test_identical_inputs_produce_identical_all_level_identities():
    first, second = _build(), _build()
    assert first == second
    assert first.source.source_content_identity == second.source.source_content_identity
    assert first.source.parser_interpretation_identity == second.source.parser_interpretation_identity
    assert first.cadence.cadence_evidence_identity == second.cadence.cadence_evidence_identity


@pytest.mark.parametrize("field", ["sha256", "file_size_bytes"])
def test_source_content_change_changes_source_and_recording_identity(field):
    baseline = _build()
    inspection = _inspection()
    replacement = "c" * 64 if field == "sha256" else 123457
    inspection = replace(inspection, source_identity=replace(inspection.source_identity, **{field: replacement}))
    changed = _build(inspection)
    assert changed.source.source_content_identity != baseline.source.source_content_identity
    assert changed.recording_identity != baseline.recording_identity


@pytest.mark.parametrize(
    "changes",
    [
        {"header_row_index": 2},
        {"time_column": "AlternateTime", "raw_columns": ("TimeStamp", "AlternateTime", "1-410", "1-470", "2-410", "2-470", "10-410", "10-470")},
        {"timestamp_unit": "seconds"},
    ],
)
def test_parser_change_changes_parser_and_recording_identity(changes):
    baseline = _build()
    inspection = _inspection()
    inspection = replace(inspection, parser_facts=replace(inspection.parser_facts, **changes))
    changed = _build(inspection)
    assert changed.source.parser_interpretation_identity != baseline.source.parser_interpretation_identity
    assert changed.recording_identity != baseline.recording_identity


@pytest.mark.parametrize(
    "changes",
    [
        {"selected_folder_canonical": "D:\\moved", "fluorescence_path_canonical": "D:\\moved\\Fluorescence.csv"},
        {"modification_time_ns": 111},
        {"stable_source_identity": "d" * 64},
    ],
)
def test_provenance_only_change_does_not_change_scientific_identities(changes):
    baseline = _build()
    inspection = _inspection()
    inspection = replace(inspection, source_identity=replace(inspection.source_identity, **changes))
    changed = _build(inspection)
    assert changed.source.source_content_identity == baseline.source.source_content_identity
    assert changed.source.parser_interpretation_identity == baseline.source.parser_interpretation_identity
    assert changed.cadence.cadence_evidence_identity == baseline.cadence.cadence_evidence_identity
    assert changed.recording_identity == baseline.recording_identity
    assert serialize_guided_continuous_rwd_recording_description(changed)["source"] != serialize_guided_continuous_rwd_recording_description(baseline)["source"]


def test_cadence_change_changes_cadence_and_recording_identity():
    baseline = _build()
    inspection = _inspection()
    inspection = replace(inspection, time_axis=replace(inspection.time_axis, mean_positive_dt_seconds=0.051))
    changed = _build(inspection)
    assert changed.cadence.cadence_evidence_identity != baseline.cadence.cadence_evidence_identity
    assert changed.recording_identity != baseline.recording_identity


def test_builder_rejects_plan_and_execution_concepts_at_call_boundary():
    with pytest.raises(TypeError):
        build_guided_continuous_rwd_recording_description(_inspection(), included_roi_ids=("1",), target_fs_hz=20.0)
    with pytest.raises(TypeError):
        build_guided_continuous_rwd_recording_description(_inspection(), included_roi_ids=("1",), block_size=1000)


def test_serialization_is_deterministic_json_compatible_and_round_trips():
    description = _build()
    first = serialize_guided_continuous_rwd_recording_description(description)
    second = serialize_guided_continuous_rwd_recording_description(description)
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(second, sort_keys=True, separators=(",", ":"))
    assert deserialize_guided_continuous_rwd_recording_description(first) == description
    assert deserialize_guided_continuous_rwd_recording_description(
        json.loads(json.dumps(first))
    ) == description


@pytest.mark.parametrize(
    "tamper",
    [
        lambda payload: payload["source"].update(source_content_identity="0" * 64),
        lambda payload: payload["source"].update(parser_interpretation_identity="0" * 64),
        lambda payload: payload["cadence"].update(cadence_evidence_identity="0" * 64),
        lambda payload: payload.update(recording_identity="0" * 64),
    ],
)
def test_deserialization_refuses_each_stored_identity_mismatch(tamper):
    payload = serialize_guided_continuous_rwd_recording_description(_build())
    tamper(payload)
    with pytest.raises(ContinuousRwdRecordingAuthorityError, match="identity mismatch"):
        deserialize_guided_continuous_rwd_recording_description(payload)


def test_deserialization_refuses_unsupported_schema_missing_and_unknown_fields():
    payload = serialize_guided_continuous_rwd_recording_description(_build())
    payload["schema_version"] = "v999"
    with pytest.raises(ContinuousRwdRecordingAuthorityError, match="metadata"):
        deserialize_guided_continuous_rwd_recording_description(payload)
    payload = serialize_guided_continuous_rwd_recording_description(_build())
    del payload["time"]
    with pytest.raises(ContinuousRwdRecordingAuthorityError, match="exactly"):
        deserialize_guided_continuous_rwd_recording_description(payload)
    payload = serialize_guided_continuous_rwd_recording_description(_build())
    payload["future"] = True
    with pytest.raises(ContinuousRwdRecordingAuthorityError, match="exactly"):
        deserialize_guided_continuous_rwd_recording_description(payload)


def test_deserialization_refuses_nonfinite_and_inconsistent_roi_partition():
    payload = serialize_guided_continuous_rwd_recording_description(_build())
    payload["time"]["measured_duration_seconds"] = math.inf
    with pytest.raises(ContinuousRwdRecordingAuthorityError, match="finite"):
        deserialize_guided_continuous_rwd_recording_description(payload)
    payload = serialize_guided_continuous_rwd_recording_description(_build())
    payload["roi"]["excluded_roi_ids"] = []
    with pytest.raises(ContinuousRwdRecordingAuthorityError, match="cover"):
        deserialize_guided_continuous_rwd_recording_description(payload)


def _refresh_serialized_cadence_and_recording_identities(payload):
    cadence_payload = dict(payload["cadence"])
    cadence_payload.pop("cadence_evidence_identity")
    cadence_identity = subject._digest(
        subject.CADENCE_EVIDENCE_IDENTITY_DOMAIN,
        cadence_payload,
    )
    payload["cadence"]["cadence_evidence_identity"] = cadence_identity
    recording_payload = {
        "schema_name": payload["schema_name"],
        "schema_version": payload["schema_version"],
        "source_format": payload["source_format"],
        "acquisition_mode": payload["acquisition_mode"],
        "execution_admission_status": payload["execution_admission_status"],
        "unresolved_admission_checks": payload["unresolved_admission_checks"],
        "source_content_identity": payload["source"]["source_content_identity"],
        "parser_interpretation_identity": payload["source"][
            "parser_interpretation_identity"
        ],
        "cadence_evidence_identity": cadence_identity,
        "row_authority": {
            "total_data_row_count": payload["source"]["total_data_row_count"],
            "valid_timestamp_count": payload["source"]["valid_timestamp_count"],
        },
        "time_authority": payload["time"],
        "roi_authority": payload["roi"],
    }
    payload["recording_identity"] = subject._digest(
        subject.RECORDING_IDENTITY_DOMAIN,
        recording_payload,
    )


def test_deserialization_refuses_invalid_science_even_with_refreshed_identities():
    payload = serialize_guided_continuous_rwd_recording_description(_build())
    payload["cadence"]["standard_deviation_positive_dt_seconds"] = -0.001
    _refresh_serialized_cadence_and_recording_identities(payload)

    with pytest.raises(
        ContinuousRwdRecordingAuthorityError,
        match="standard deviation must be nonnegative",
    ):
        deserialize_guided_continuous_rwd_recording_description(payload)


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_serialized_payload_contains_no_session_or_disguised_segment_keys():
    keys = set(_all_keys(serialize_guided_continuous_rwd_recording_description(_build())))
    forbidden = {
        "session", "sessions", "session_index", "session_start", "session_duration",
        "sessions_per_hour", "expected_session_count", "missing_sessions",
        "excluded_final_session", "segment", "segments", "chunk", "chunks",
    }
    assert keys.isdisjoint(forbidden)


def test_build_serialize_deserialize_never_use_filesystem_or_inspector(monkeypatch, tmp_path):
    before = tuple(tmp_path.iterdir())
    def forbidden(*args, **kwargs):
        raise AssertionError("filesystem or inspector access is forbidden")
    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, "open", forbidden)
        guarded.setattr(Path, "exists", forbidden)
        guarded.setattr(Path, "is_file", forbidden)
        guarded.setattr(Path, "is_dir", forbidden)
        guarded.setattr(Path, "stat", forbidden)
        guarded.setattr(Path, "open", forbidden)
        guarded.setattr(Path, "iterdir", forbidden)
        guarded.setattr(
            "photometry_pipeline.io.rwd_continuous_source.inspect_continuous_rwd_acquisition_folder",
            forbidden,
        )
        description = _build()
        payload = serialize_guided_continuous_rwd_recording_description(description)
        assert deserialize_guided_continuous_rwd_recording_description(payload) == description
    assert tuple(tmp_path.iterdir()) == before


def test_module_has_no_gui_or_existing_normalized_recording_dependency():
    source = inspect.getsource(subject)
    assert "gui" not in subject.__dict__
    assert "guided_normalized_recording" not in source
    assert "inspect_continuous_rwd_acquisition_folder" not in source
