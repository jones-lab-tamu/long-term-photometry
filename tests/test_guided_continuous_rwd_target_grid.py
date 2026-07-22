from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError, asdict, fields, replace
from decimal import Decimal
from fractions import Fraction
import inspect
import math
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_continuous_rwd_target_grid as subject
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    CONTINUITY_PASSED,
    ContinuousRwdDiscontinuityEvaluation,
    ContinuousRwdDiscontinuityExample,
)
from photometry_pipeline.guided_continuous_rwd_discontinuity_policy import (
    POLICY_NAME,
    POLICY_VERSION,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    GuidedContinuousRwdRecordingDescription,
    build_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.io.rwd_continuous_source import (
    CADENCE_EVIDENCE_POLICY_VERSION,
    IDENTITY_POLICY_VERSION,
    INSPECTION_CONTRACT_NAME,
    INSPECTION_CONTRACT_VERSION,
    MINIMUM_DURATION_SEC,
    ContinuousRwdCadenceQuantile,
    ContinuousRwdChannelEvidence,
    ContinuousRwdInspectionResult,
    ContinuousRwdParserFacts,
    ContinuousRwdRoiPair,
    ContinuousRwdSourceIdentity,
    ContinuousRwdTimeAxisEvidence,
)


def _authorities(
    *,
    duration: float = 600.0,
    cadence: float = 0.1,
    row_count: int = 101,
    folder: str = r"C:\accepted\recording",
) -> tuple[GuidedContinuousRwdRecordingDescription, ContinuousRwdDiscontinuityEvaluation]:
    intervals = row_count - 1
    quantile_count = min(intervals, 100)
    inspection = ContinuousRwdInspectionResult(
        contract_name=INSPECTION_CONTRACT_NAME,
        contract_version=INSPECTION_CONTRACT_VERSION,
        status="completed",
        outcome_category="inspection_completed",
        scientist_summary="Inspection completed.",
        source_identity=ContinuousRwdSourceIdentity(
            identity_policy_version=IDENTITY_POLICY_VERSION,
            selected_folder_canonical=folder,
            fluorescence_path_canonical=str(Path(folder) / "Fluorescence.csv"),
            file_size_bytes=1234,
            modification_time_ns=5678,
            sha256="a" * 64,
            stable_source_identity="b" * 64,
        ),
        parser_facts=ContinuousRwdParserFacts(
            header_row_index=0,
            time_column="Time(s)",
            raw_columns=("Time(s)", "ROI1-410", "ROI1-470"),
            timestamp_unit="seconds",
            timestamp_scale_to_seconds=1.0,
        ),
        time_axis=ContinuousRwdTimeAxisEvidence(
            total_data_row_count=row_count,
            valid_timestamp_count=row_count,
            raw_first_timestamp=1000.0,
            raw_last_timestamp=1000.0 + duration,
            normalized_first_seconds=0.0,
            normalized_last_seconds=duration,
            measured_duration_seconds=duration,
            minimum_duration_seconds=MINIMUM_DURATION_SEC,
            duration_product_classification="meets_product_minimum",
            positive_interval_count=intervals,
            nominal_cadence_seconds=cadence,
            minimum_positive_dt_seconds=cadence,
            maximum_positive_dt_seconds=cadence,
            mean_positive_dt_seconds=cadence,
            standard_deviation_positive_dt_seconds=0.0,
            coefficient_of_variation=0.0,
            quantiles=tuple(
                ContinuousRwdCadenceQuantile(probability, cadence)
                for probability in (0.001, 0.01, 0.5, 0.99, 0.999)
            ),
            quantile_method="deterministic_reservoir_linear.v1",
            quantile_sample_count=quantile_count,
            duplicate_timestamp_count=0,
            backward_timestamp_count=0,
            nonnumeric_timestamp_count=0,
            nonfinite_timestamp_count=0,
            unusually_long_interval_count=0,
            unusually_short_interval_count=0,
            largest_unusual_intervals=(),
            smallest_unusual_intervals=(),
            cadence_evidence_policy_version=CADENCE_EVIDENCE_POLICY_VERSION,
        ),
        channels=ContinuousRwdChannelEvidence(
            roi_pairs=(ContinuousRwdRoiPair("ROI1", "ROI1-410", "ROI1-470"),),
            unmatched_channel_columns=(),
            selected_value_count=row_count * 2,
            nonnumeric_selected_value_count=0,
            nonfinite_selected_value_count=0,
            malformed_row_count=0,
        ),
        findings=(),
        source_stable=True,
        full_file_passes=2,
    )
    recording = build_guided_continuous_rwd_recording_description(
        inspection,
        included_roi_ids=("ROI1",),
    )
    evaluation = ContinuousRwdDiscontinuityEvaluation(
        outcome=CONTINUITY_PASSED,
        recording_identity=recording.recording_identity,
        source_content_identity=recording.source.source_content_identity,
        parser_interpretation_identity=recording.source.parser_interpretation_identity,
        cadence_evidence_identity=recording.cadence.cadence_evidence_identity,
        policy_name=POLICY_NAME,
        policy_version=POLICY_VERSION,
        nominal_cadence_seconds=recording.cadence.nominal_cadence_seconds,
        tolerance_seconds=0.002,
        valid_row_count_evaluated=row_count,
        positive_interval_count_evaluated=intervals,
        normal_interval_count=intervals,
        short_interval_anomaly_count=0,
        material_long_interval_count=0,
        maximum_short_residual_seconds=None,
        maximum_long_residual_seconds=None,
        short_examples=(),
        long_examples=(),
        observed_source_sha256=recording.source.sha256,
        observed_source_size_bytes=recording.source.file_size_bytes,
        failure_reason=None,
    )
    return recording, evaluation


@pytest.fixture
def valid_case():
    return _authorities()


def _build(case):
    return subject.build_guided_continuous_rwd_target_grid(*case)


def test_valid_construction_is_frozen_minimal_and_deterministic(valid_case):
    first = _build(valid_case)
    second = _build(valid_case)
    assert first == second
    assert (first.schema_name, first.schema_version) == (
        subject.SCHEMA_NAME,
        subject.SCHEMA_VERSION,
    )
    assert (first.grid_policy_name, first.grid_policy_version) == (
        subject.GRID_POLICY_NAME,
        subject.GRID_POLICY_VERSION,
    )
    assert first.recording_identity == valid_case[0].recording_identity
    assert first.continuity_evaluation_identity == (
        subject.compute_continuous_rwd_discontinuity_evaluation_identity(valid_case[1])
    )
    assert len(first.continuity_evaluation_identity) == 64
    assert len(first.target_grid_identity) == 64
    for identity in (
        first.recording_identity,
        first.continuity_evaluation_identity,
        first.target_grid_identity,
    ):
        assert len(identity) == 64
        assert identity == identity.lower()
        assert set(identity) <= set("0123456789abcdef")
    assert set(asdict(first)) == {
        "schema_name", "schema_version", "grid_policy_name", "grid_policy_version",
        "recording_identity", "continuity_evaluation_identity",
        "cadence_seconds_numerator", "cadence_seconds_denominator",
        "source_support_end_seconds_numerator",
        "source_support_end_seconds_denominator", "target_sample_count",
        "target_grid_identity",
    }
    assert not any("array" in name or "timestamps" in name for name in asdict(first))
    with pytest.raises(FrozenInstanceError):
        first.target_sample_count = 1


def test_exact_fraction_origin_count_and_endpoint_invariants(valid_case):
    grid = _build(valid_case)
    assert grid.origin_elapsed_seconds == 0.0
    assert grid.cadence_fraction == Fraction(1, 10)
    assert grid.source_support_end_fraction == Fraction(600, 1)
    assert math.gcd(grid.cadence_seconds_numerator, grid.cadence_seconds_denominator) == 1
    assert math.gcd(
        grid.source_support_end_seconds_numerator,
        grid.source_support_end_seconds_denominator,
    ) == 1
    assert grid.target_sample_count == 6001
    assert grid.last_target_index == 6000
    assert grid.last_target_elapsed_fraction == grid.source_support_end_fraction
    assert grid.next_target_elapsed_fraction > grid.source_support_end_fraction


@pytest.mark.parametrize(
    ("duration", "cadence", "expected_count", "expected_last"),
    [
        (600.0, 0.1, 6001, 600.0),
        (math.nextafter(600.0, math.inf), 0.1, 6001, 600.0),
        (600.0, 1.0, 601, 600.0),
        (600.0, 0.000025, 24_000_001, 600.0),
        (5_000_000.0, 0.1, 50_000_001, 5_000_000.0),
        (347_345.634853, 0.099997, 3_473_561, 347_345.57932),
    ],
)
def test_builder_boundary_arithmetic(duration, cadence, expected_count, expected_last):
    grid = _build(_authorities(duration=duration, cadence=cadence))
    assert grid.target_sample_count == expected_count
    assert grid.last_target_elapsed_seconds == expected_last
    assert grid.last_target_elapsed_fraction <= grid.source_support_end_fraction
    assert grid.next_target_elapsed_fraction > grid.source_support_end_fraction


def test_next_float_below_minimum_uses_exact_scalar_count_without_binary_off_by_one():
    duration = math.nextafter(600.0, -math.inf)
    support = Fraction(Decimal(str(duration)))
    cadence = Fraction(Decimal(str(0.1)))
    assert subject._resolve_target_sample_count(support, cadence) == 6000


def test_cadence_is_exact_b1_authority_not_conventional_or_duration_derived():
    recording, evaluation = _authorities(duration=601.0, cadence=0.099997, row_count=17)
    grid = _build((recording, evaluation))
    assert grid.cadence_fraction == Fraction(99997, 1_000_000)
    assert grid.cadence_seconds != 0.1
    assert grid.cadence_seconds != recording.time.measured_duration_seconds / 16
    assert grid.target_sample_count != recording.source.valid_timestamp_count


@pytest.mark.parametrize(
    "recording",
    [None, object(), "recording"],
)
def test_wrong_b1_type_is_refused(recording, valid_case):
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="recording must"):
        subject.build_guided_continuous_rwd_target_grid(recording, valid_case[1])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": "v999"}, "recording authority"),
        ({"source_format": "npm"}, "recording authority"),
        ({"acquisition_mode": "intermittent"}, "recording authority"),
        ({"execution_admission_status": "accepted"}, "recording authority"),
        ({"unresolved_admission_checks": ()}, "recording authority"),
    ],
)
def test_malformed_b1_envelope_is_refused(valid_case, change, message):
    malformed = replace(valid_case[0], **change)
    with pytest.raises(subject.ContinuousRwdTargetGridError, match=message):
        subject.build_guided_continuous_rwd_target_grid(malformed, valid_case[1])


@pytest.mark.parametrize("cadence", [math.nan, math.inf, 0.0, -0.1])
def test_invalid_b1_cadence_is_refused(valid_case, cadence):
    malformed_cadence = replace(valid_case[0].cadence, nominal_cadence_seconds=cadence)
    malformed = replace(valid_case[0], cadence=malformed_cadence)
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="recording authority"):
        subject.build_guided_continuous_rwd_target_grid(malformed, valid_case[1])


@pytest.mark.parametrize("support", [math.nan, math.inf, 0.0, -1.0])
def test_invalid_b1_support_is_refused(valid_case, support):
    malformed_time = replace(
        valid_case[0].time,
        measured_support_end_seconds=support,
        measured_duration_seconds=support,
        normalized_elapsed_end_seconds=support,
    )
    malformed = replace(valid_case[0], time=malformed_time)
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="recording authority"):
        subject.build_guided_continuous_rwd_target_grid(malformed, valid_case[1])


@pytest.mark.parametrize(
    "time_change",
    [
        {"normalized_elapsed_origin_seconds": 1.0},
        {"time_basis": "absolute"},
        {"measured_support_end_seconds": 599.0},
    ],
)
def test_malformed_b1_time_contract_is_refused(valid_case, time_change):
    malformed = replace(valid_case[0], time=replace(valid_case[0].time, **time_change))
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="recording authority"):
        subject.build_guided_continuous_rwd_target_grid(malformed, valid_case[1])


def test_wrong_b2_type_is_refused(valid_case):
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="continuity_evaluation"):
        subject.build_guided_continuous_rwd_target_grid(valid_case[0], object())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"outcome": "material_long_interval_detected"}, "must have passed"),
        ({"failure_reason": "failure"}, "failure reason"),
        ({"recording_identity": "0" * 64}, "recording identity"),
        ({"source_content_identity": "0" * 64}, "source-content identity"),
        ({"parser_interpretation_identity": "0" * 64}, "parser identity"),
        ({"cadence_evidence_identity": "0" * 64}, "cadence identity"),
        ({"policy_name": "other"}, "unsupported policy"),
        ({"policy_version": "v999"}, "unsupported policy"),
        ({"nominal_cadence_seconds": 0.2}, "nominal cadence"),
        ({"valid_row_count_evaluated": 99}, "valid-row count"),
        ({"positive_interval_count_evaluated": 99}, "interval count"),
        ({"normal_interval_count": 99}, "normal-interval count"),
        ({"short_interval_anomaly_count": 1}, "short-interval evidence"),
        ({"material_long_interval_count": 1}, "long-interval evidence"),
    ],
)
def test_incompatible_b2_authority_is_refused(valid_case, change, message):
    malformed = replace(valid_case[1], **change)
    with pytest.raises(subject.ContinuousRwdTargetGridError, match=message):
        subject.build_guided_continuous_rwd_target_grid(valid_case[0], malformed)


def _example(category: str) -> ContinuousRwdDiscontinuityExample:
    return ContinuousRwdDiscontinuityExample(
        category, 2, 1, 0.0, 0.2, 0.2, 0.1, 0.002, 0.1
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"short_examples": (_example("short_interval_anomaly"),)}, "short-interval"),
        ({"long_examples": (_example("material_long_interval"),)}, "long-interval"),
    ],
)
def test_passing_b2_must_not_retain_examples(valid_case, change, message):
    with pytest.raises(subject.ContinuousRwdTargetGridError, match=message):
        subject.build_guided_continuous_rwd_target_grid(
            valid_case[0], replace(valid_case[1], **change)
        )


def test_c1_does_not_recompute_b2a_tolerance_or_residual_invariants(valid_case):
    changed = replace(
        valid_case[1],
        tolerance_seconds=987.0,
        maximum_short_residual_seconds=123.0,
        maximum_long_residual_seconds=456.0,
    )
    grid = subject.build_guided_continuous_rwd_target_grid(valid_case[0], changed)
    assert grid.continuity_evaluation_identity != _build(valid_case).continuity_evaluation_identity


def test_every_b2_field_is_continuity_identity_bearing(valid_case):
    evaluation = valid_case[1]
    original = subject.compute_continuous_rwd_discontinuity_evaluation_identity(evaluation)
    replacements = {
        "outcome": "changed", "recording_identity": "0" * 64,
        "source_content_identity": "0" * 64,
        "parser_interpretation_identity": "0" * 64,
        "cadence_evidence_identity": "0" * 64, "policy_name": "changed",
        "policy_version": "changed", "nominal_cadence_seconds": 0.2,
        "tolerance_seconds": 0.3, "valid_row_count_evaluated": 102,
        "positive_interval_count_evaluated": 101, "normal_interval_count": 99,
        "short_interval_anomaly_count": 1, "material_long_interval_count": 1,
        "maximum_short_residual_seconds": 0.1,
        "maximum_long_residual_seconds": 0.1,
        "short_examples": (_example("short_interval_anomaly"),),
        "long_examples": (_example("material_long_interval"),),
        "observed_source_sha256": "0" * 64, "observed_source_size_bytes": 999,
        "failure_reason": "changed",
    }
    assert set(replacements) == {field.name for field in fields(evaluation)}
    for name, value in replacements.items():
        changed = replace(evaluation, **{name: value})
        assert subject.compute_continuous_rwd_discontinuity_evaluation_identity(changed) != original


@pytest.mark.parametrize(
    "changes",
    [
        {"cadence_seconds_numerator": 2},
        {"source_support_end_seconds_numerator": 601},
        {"target_sample_count": 6000},
        {"grid_policy_version": "v2"},
    ],
)
def test_each_grid_scalar_or_policy_change_changes_identity(valid_case, changes):
    grid = _build(valid_case)
    changed = replace(grid, **changes)
    assert subject.compute_guided_continuous_rwd_target_grid_identity(changed) != (
        grid.target_grid_identity
    )


def test_unsupported_policy_and_identity_mismatch_are_refused(valid_case):
    grid = _build(valid_case)
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="metadata"):
        subject._validate_target_grid_description(
            replace(grid, grid_policy_version="v2")
        )
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="identity mismatch"):
        subject._validate_target_grid_description(
            replace(grid, target_grid_identity="0" * 64)
        )


def _resign_grid_description(description, **changes):
    changed = replace(description, **changes, target_grid_identity="")
    return replace(
        changed,
        target_grid_identity=(
            subject.compute_guided_continuous_rwd_target_grid_identity(changed)
        ),
    )


@pytest.mark.parametrize(
    "identity",
    [123, "a" * 63, "A" * 64, "g" * 64],
)
def test_malformed_recording_identity_is_refused_even_when_grid_is_resigned(
    valid_case, identity
):
    malformed = _resign_grid_description(
        _build(valid_case), recording_identity=identity
    )
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="Recording identity"):
        subject._validate_target_grid_description(malformed)


@pytest.mark.parametrize(
    "identity",
    [None, "b" * 12, "B" * 64, "-" * 64],
)
def test_malformed_continuity_identity_is_refused_even_when_grid_is_resigned(
    valid_case, identity
):
    malformed = _resign_grid_description(
        _build(valid_case), continuity_evaluation_identity=identity
    )
    with pytest.raises(
        subject.ContinuousRwdTargetGridError,
        match="Continuity-evaluation identity",
    ):
        subject._validate_target_grid_description(malformed)


@pytest.mark.parametrize("identity", [object(), "c" * 8, "C" * 64, "z" * 64])
def test_malformed_target_grid_identity_form_is_refused(valid_case, identity):
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="Target-grid identity"):
        subject._validate_target_grid_description(
            replace(_build(valid_case), target_grid_identity=identity)
        )


def test_path_only_relocation_does_not_change_grid_identity(valid_case):
    recording, evaluation = valid_case
    relocated_source = replace(
        recording.source,
        selected_folder_canonical=r"D:\relocated",
        fluorescence_path_canonical=r"D:\relocated\Fluorescence.csv",
        modification_time_ns=9999,
        stable_source_identity="c" * 64,
    )
    relocated = replace(recording, source=relocated_source)
    assert relocated.recording_identity == recording.recording_identity
    assert _build((relocated, evaluation)).target_grid_identity == _build(valid_case).target_grid_identity


def test_fewer_than_two_samples_and_overflow_are_refused():
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="at least two"):
        _build(_authorities(duration=600.0, cadence=1000.0))
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="64-bit"):
        _build(_authorities(duration=600.0, cadence=1e-20))


def _signed_scalar_description(*, support: int, count: int):
    draft = subject.GuidedContinuousRwdTargetGridDescription(
        subject.SCHEMA_NAME, subject.SCHEMA_VERSION,
        subject.GRID_POLICY_NAME, subject.GRID_POLICY_VERSION,
        "a" * 64, "b" * 64, 1, 1, support, 1, count, "",
    )
    return replace(
        draft,
        target_grid_identity=subject.compute_guided_continuous_rwd_target_grid_identity(draft),
    )


def test_signed_64_bit_count_limit_and_exact_endpoint_rules():
    maximum = subject.MAX_TARGET_SAMPLE_COUNT
    accepted = _signed_scalar_description(support=maximum - 1, count=maximum)
    subject._validate_target_grid_description(accepted)
    assert accepted.last_target_elapsed_fraction == accepted.source_support_end_fraction

    above = _signed_scalar_description(support=maximum, count=maximum + 1)
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="64-bit"):
        subject._validate_target_grid_description(above)

    omitted = _signed_scalar_description(support=10, count=10)
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="omits"):
        subject._validate_target_grid_description(omitted)

    outside = _signed_scalar_description(support=10, count=12)
    with pytest.raises(subject.ContinuousRwdTargetGridError, match="outside"):
        subject._validate_target_grid_description(outside)


def test_global_index_time_is_partition_independent_and_half_open_ranges_compose(valid_case):
    grid = _build(valid_case)
    ranges = ((0, 7), (7, 101), (101, grid.target_sample_count))
    indices = [index for start, stop in ranges for index in range(start, stop)]
    assert indices == list(range(grid.target_sample_count))
    probe = 100
    direct = probe * grid.cadence_fraction
    from_first_partition = (0 + probe) * grid.cadence_fraction
    from_later_partition = (7 + (probe - 7)) * grid.cadence_fraction
    assert direct == from_first_partition == from_later_partition


def test_builder_is_scalar_only_and_performs_no_filesystem_work(valid_case, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("filesystem access is forbidden")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "stat", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    assert _build(valid_case).target_sample_count == 6001


def test_module_has_only_bounded_scalar_contract_dependencies():
    tree = ast.parse(inspect.getsource(subject))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    from_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {
        "numpy", "pandas", "h5py", "gui", "pathlib",
        "photometry_pipeline.io.rwd_continuous_source",
        "photometry_pipeline.io.adapters", "photometry_pipeline.pipeline",
    }
    assert imports.isdisjoint(forbidden)
    assert from_imports.isdisjoint(forbidden)
    source = inspect.getsource(subject).lower()
    for token in ("np.arange", "np.linspace", "interpol", "resampl", "source_path"):
        assert token not in source
    assert "numpy.ndarray" not in source
    assert set(inspect.signature(subject.build_guided_continuous_rwd_target_grid).parameters) == {
        "recording", "continuity_evaluation"
    }
