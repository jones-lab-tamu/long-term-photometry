from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import math
from pathlib import Path

import pytest

import photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation as subject
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    CONTINUITY_PASSED,
    EVALUATION_INTERRUPTED,
    MATERIAL_LONG_INTERVAL_DETECTED,
    SHORT_AND_LONG_DISCONTINUITIES_DETECTED,
    SHORT_INTERVAL_ANOMALY_DETECTED,
    SOURCE_CHANGED_OR_MISMATCHED,
    ContinuousRwdDiscontinuityEvaluationError,
    evaluate_continuous_rwd_timestamp_continuity,
)
from photometry_pipeline.guided_continuous_rwd_discontinuity_policy import (
    POLICY_NAME,
    POLICY_VERSION,
    classify_continuous_rwd_interval,
    resolve_continuous_rwd_discontinuity_tolerance,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    build_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.io.rwd_continuous_source import (
    CADENCE_EVIDENCE_POLICY_VERSION,
    INSPECTION_CONTRACT_NAME,
    INSPECTION_CONTRACT_VERSION,
    ContinuousRwdCadenceQuantile,
    ContinuousRwdChannelEvidence,
    ContinuousRwdInspectionResult,
    ContinuousRwdParserFacts,
    ContinuousRwdRoiPair,
    ContinuousRwdSourceIdentity,
    ContinuousRwdTimeAxisEvidence,
    inspect_continuous_rwd_acquisition_folder,
)


def _source_case(
    tmp_path: Path,
    *,
    nominal: float = 0.1,
    special_dts: tuple[float, ...] = (),
    special_at_end: bool = False,
    scale: float = 1.0,
    first_raw: float = 1000.0,
    name: str = "Fluorescence.csv",
):
    regular_count = max(
        math.ceil((600.0 - sum(special_dts)) / nominal) + 2,
        1,
    )
    dts = ((nominal,) * regular_count + special_dts) if special_at_end else (
        special_dts + (nominal,) * regular_count
    )
    timestamps = [first_raw]
    for dt in dts:
        timestamps.append(timestamps[-1] + dt / scale)
    time_column = "TimeStamp" if scale == 0.001 else "Time(s)"
    columns = (time_column, "CH1-410", "CH1-470")
    lines = ["synthetic preamble", ",".join(columns)]
    lines.extend(f"{value:.12f},1,2" for value in timestamps)
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path = tmp_path / name
    path.write_bytes(payload)
    facts = path.stat()

    scaled_dts = tuple(
        (right - left) * scale
        for left, right in zip(timestamps, timestamps[1:])
    )
    duration = (timestamps[-1] - timestamps[0]) * scale
    mean = sum(scaled_dts) / len(scaled_dts)
    variance = sum((value - mean) ** 2 for value in scaled_dts) / len(scaled_dts)
    inspection = ContinuousRwdInspectionResult(
        contract_name=INSPECTION_CONTRACT_NAME,
        contract_version=INSPECTION_CONTRACT_VERSION,
        status="completed",
        outcome_category="inspection_completed",
        scientist_summary="Synthetic inspection completed.",
        source_identity=ContinuousRwdSourceIdentity(
            identity_policy_version="continuous-rwd-source-identity.v1",
            selected_folder_canonical=str(tmp_path.resolve()),
            fluorescence_path_canonical=str(path.resolve()),
            file_size_bytes=len(payload),
            modification_time_ns=facts.st_mtime_ns,
            sha256=hashlib.sha256(payload).hexdigest(),
            stable_source_identity="b" * 64,
        ),
        parser_facts=ContinuousRwdParserFacts(
            header_row_index=1,
            time_column=time_column,
            raw_columns=columns,
            timestamp_unit="milliseconds" if scale == 0.001 else "seconds",
            timestamp_scale_to_seconds=scale,
        ),
        time_axis=ContinuousRwdTimeAxisEvidence(
            total_data_row_count=len(timestamps),
            valid_timestamp_count=len(timestamps),
            raw_first_timestamp=timestamps[0],
            raw_last_timestamp=timestamps[-1],
            normalized_first_seconds=0.0,
            normalized_last_seconds=duration,
            measured_duration_seconds=duration,
            minimum_duration_seconds=600.0,
            duration_product_classification="meets_product_minimum",
            positive_interval_count=len(scaled_dts),
            nominal_cadence_seconds=nominal,
            minimum_positive_dt_seconds=min(*scaled_dts, nominal),
            maximum_positive_dt_seconds=max(*scaled_dts, nominal),
            mean_positive_dt_seconds=mean,
            standard_deviation_positive_dt_seconds=math.sqrt(variance),
            coefficient_of_variation=math.sqrt(variance) / mean,
            quantiles=(
                ContinuousRwdCadenceQuantile(0.001, nominal),
                ContinuousRwdCadenceQuantile(0.01, nominal),
                ContinuousRwdCadenceQuantile(0.5, nominal),
                ContinuousRwdCadenceQuantile(0.99, nominal),
                ContinuousRwdCadenceQuantile(0.999, nominal),
            ),
            quantile_method="deterministic_reservoir_linear.v1",
            quantile_sample_count=len(scaled_dts),
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
            roi_pairs=(ContinuousRwdRoiPair("CH1", "CH1-410", "CH1-470"),),
            unmatched_channel_columns=(),
            selected_value_count=len(timestamps) * 2,
            nonnumeric_selected_value_count=0,
            nonfinite_selected_value_count=0,
            malformed_row_count=0,
        ),
        findings=(),
        source_stable=True,
        full_file_passes=2,
    )
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("CH1",)
    )
    return path, recording


def _cr1_a_source_case(tmp_path: Path, *, trailing_delimiter: bool):
    folder = tmp_path / (
        "cr1-a-trailing" if trailing_delimiter else "cr1-a-no-trailing"
    )
    folder.mkdir()
    path = folder / "Fluorescence.csv"
    terminal = "," if trailing_delimiter else ""
    rows = "".join(
        f"{second},1,2{terminal}\n" for second in range(601)
    )
    payload = (
        ("p" * 700)
        + "\n"
        + f"\ufeff  Time(s)  , CH1-410 , CH1-470 {terminal}\n"
        + rows
    )
    path.write_bytes(payload.encode("utf-8"))
    inspection = inspect_continuous_rwd_acquisition_folder(folder)
    assert inspection.status == "completed"
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("CH1",)
    )
    return path, inspection, recording


def test_cr1_a_trailing_delimiter_authority_passes_exact_b2b_parity(tmp_path):
    path, inspection, recording = _cr1_a_source_case(
        tmp_path, trailing_delimiter=True
    )
    expected = ("Time(s)", "CH1-410", "CH1-470", "")
    assert inspection.parser_facts.raw_columns == expected
    assert recording.source.raw_columns == expected
    assert inspection.channels.malformed_row_count == 0

    result = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=path
    )
    assert result.outcome == CONTINUITY_PASSED
    assert result.failure_reason is None
    assert result.valid_row_count_evaluated == 601
    assert result.positive_interval_count_evaluated == 600


@pytest.mark.parametrize(
    ("accepted_trailing", "current_trailing"),
    [(True, False), (False, True)],
)
def test_header_and_data_terminal_field_count_must_match_frozen_authority(
    tmp_path, accepted_trailing, current_trailing
):
    path, _, recording = _cr1_a_source_case(
        tmp_path, trailing_delimiter=accepted_trailing
    )
    lines = path.read_bytes().splitlines()
    data = lines[2:]
    if current_trailing:
        changed_data = [line + b"," for line in data]
        lines[0] = lines[0][:-len(data)]
    else:
        changed_data = [line[:-1] for line in data]
        lines[0] += b"p" * len(data)
    changed = b"\n".join(lines[:2] + changed_data) + b"\n"
    assert len(changed) == recording.source.file_size_bytes
    path.write_bytes(changed)

    result = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=path
    )
    assert result.outcome == SOURCE_CHANGED_OR_MISMATCHED
    assert result.failure_reason == "malformed_data_row"


def test_two_terminal_empty_header_fields_remain_refused_by_cr1_a(tmp_path):
    folder = tmp_path / "double-terminal"
    folder.mkdir()
    path = folder / "Fluorescence.csv"
    path.write_text(
        "Time(s),CH1-410,CH1-470,,\n0,1,2,,\n600,1,2,,\n",
        encoding="utf-8",
    )
    inspection = inspect_continuous_rwd_acquisition_folder(folder)
    assert inspection.status == "failed"
    assert inspection.outcome_category == "unsupported_or_ambiguous_header"


def test_interior_empty_selected_field_cannot_produce_continuity_success(tmp_path):
    path, _, recording = _cr1_a_source_case(
        tmp_path, trailing_delimiter=True
    )
    _same_size_replace(path, b"0,1,2,\n", b"0,,02,\n")
    cr1_a = inspect_continuous_rwd_acquisition_folder(path.parent)
    assert cr1_a.outcome_category == "selected_channel_parse_failure"

    result = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=path
    )
    assert result.outcome == SOURCE_CHANGED_OR_MISMATCHED
    assert result.failure_reason == "source_sha256_mismatch"


@pytest.mark.parametrize("nominal", [0.1, 0.05, 0.025])
def test_valid_10_20_and_40_hz_sources_pass(tmp_path, nominal):
    path, recording = _source_case(tmp_path, nominal=nominal)
    result = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=path
    )
    assert result.outcome == CONTINUITY_PASSED
    assert result.valid_row_count_evaluated == recording.source.valid_timestamp_count
    assert result.positive_interval_count_evaluated == recording.cadence.positive_interval_count
    assert result.normal_interval_count == result.positive_interval_count_evaluated
    assert result.short_examples == result.long_examples == ()
    assert result.maximum_short_residual_seconds is None
    assert result.maximum_long_residual_seconds is None


def test_jitter_identities_policy_endpoints_and_counts_are_bound(tmp_path):
    path, recording = _source_case(
        tmp_path, special_dts=(0.099, 0.101, 0.1)
    )
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    tolerance = resolve_continuous_rwd_discontinuity_tolerance(recording.cadence)
    assert result.outcome == CONTINUITY_PASSED
    assert (
        result.recording_identity,
        result.source_content_identity,
        result.parser_interpretation_identity,
        result.cadence_evidence_identity,
    ) == (
        recording.recording_identity,
        recording.source.source_content_identity,
        recording.source.parser_interpretation_identity,
        recording.cadence.cadence_evidence_identity,
    )
    assert (result.policy_name, result.policy_version) == (POLICY_NAME, POLICY_VERSION)
    assert result.tolerance_seconds == tolerance.final_tolerance_seconds
    assert result.observed_source_sha256 == recording.source.sha256
    assert result.observed_source_size_bytes == recording.source.file_size_bytes
    assert result.positive_interval_count_evaluated == result.valid_row_count_evaluated - 1


def test_moved_byte_identical_source_passes_without_changing_identity(tmp_path):
    path, recording = _source_case(tmp_path)
    moved = tmp_path / "moved.csv"
    moved.write_bytes(path.read_bytes())
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=moved)
    assert result.outcome == CONTINUITY_PASSED
    assert result.recording_identity == recording.recording_identity
    assert str(moved) not in repr(result)


@pytest.mark.parametrize(
    ("dts", "outcome", "short", "long"),
    [
        ((0.05,), SHORT_INTERVAL_ANOMALY_DETECTED, 1, 0),
        ((0.2,), MATERIAL_LONG_INTERVAL_DETECTED, 0, 1),
        ((0.05, 0.2), SHORT_AND_LONG_DISCONTINUITIES_DETECTED, 1, 1),
        ((0.098, 0.102), CONTINUITY_PASSED, 0, 0),
        ((0.0979, 0.1021), SHORT_AND_LONG_DISCONTINUITIES_DETECTED, 1, 1),
        ((0.2, 0.175), MATERIAL_LONG_INTERVAL_DETECTED, 0, 2),
    ],
)
def test_interval_outcomes_and_boundaries(tmp_path, dts, outcome, short, long):
    path, recording = _source_case(tmp_path, special_dts=dts, scale=0.001)
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.outcome == outcome
    assert result.short_interval_anomaly_count == short
    assert result.material_long_interval_count == long
    assert (
        result.normal_interval_count + short + long
        == result.positive_interval_count_evaluated
    )


def test_extreme_examples_are_deterministic_bounded_and_count_every_failure(tmp_path):
    short = tuple(0.02 + index * 0.001 for index in range(15))
    long = tuple(0.3 + index * 0.01 for index in range(15))
    path, recording = _source_case(tmp_path, special_dts=short + long)
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.short_interval_anomaly_count == 15
    assert result.material_long_interval_count == 15
    assert len(result.short_examples) == len(result.long_examples) == 10
    assert [item.dt_seconds for item in result.short_examples] == pytest.approx(short[:10])
    assert [item.dt_seconds for item in result.long_examples] == pytest.approx(
        tuple(reversed(long[-10:]))
    )
    assert result.maximum_short_residual_seconds == pytest.approx(0.08)
    assert result.maximum_long_residual_seconds == pytest.approx(0.34)


def test_tied_examples_prefer_earliest_row_and_elapsed_location_is_normalized(tmp_path):
    path, recording = _source_case(
        tmp_path, special_dts=(0.05,) * 12, scale=0.001
    )
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert [item.data_row_number for item in result.short_examples] == list(range(2, 12))
    first = result.short_examples[0]
    assert first.previous_data_row_number == 1
    assert first.previous_elapsed_seconds == pytest.approx(0.0)
    assert first.current_elapsed_seconds == pytest.approx(0.05)


def test_last_possible_failure_has_last_data_row_number(tmp_path):
    path, recording = _source_case(
        tmp_path, special_dts=(0.2,), special_at_end=True
    )
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.long_examples[0].data_row_number == recording.source.valid_timestamp_count


def test_millisecond_scale_is_frozen_and_reused(tmp_path):
    path, recording = _source_case(
        tmp_path, nominal=0.05, scale=0.001, first_raw=1000.0
    )
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert recording.time.raw_timestamp_unit == "milliseconds"
    assert recording.time.raw_timestamp_scale_to_seconds == 0.001
    assert result.nominal_cadence_seconds == 0.05
    assert result.outcome == CONTINUITY_PASSED


def _same_size_replace(path: Path, old: bytes, new: bytes):
    payload = path.read_bytes()
    assert len(old) == len(new) and old in payload
    path.write_bytes(payload.replace(old, new, 1))


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        (b"Time(s),CH1-410,CH1-470", b"CH1-410,Time(s),CH1-470", "header_mismatch"),
        (b"Time(s)", b"Clock()", "header_mismatch"),
        (b"1000.000000000000", b"1001.000000000000", "first_endpoint_mismatch"),
    ],
)
def test_changed_frozen_header_or_first_endpoint_is_refused(
    tmp_path, old, new, reason
):
    path, recording = _source_case(tmp_path)
    _same_size_replace(path, old, new)
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.outcome == SOURCE_CHANGED_OR_MISMATCHED
    assert result.failure_reason == reason
    assert result.valid_row_count_evaluated == 0


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        (b"xxxx.xxxxxxxxxxxx", "nonnumeric_timestamp"),
        (b"nan              ", "nonfinite_timestamp"),
    ],
)
def test_invalid_timestamp_is_source_mismatch(tmp_path, replacement, reason):
    path, recording = _source_case(tmp_path)
    _same_size_replace(path, b"1000.000000000000", replacement)
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.outcome == SOURCE_CHANGED_OR_MISMATCHED
    assert result.failure_reason == reason


@pytest.mark.parametrize(
    ("second", "reason"),
    [
        ("1000.000000000000", "duplicate_timestamp"),
        ("0999.000000000000", "backward_timestamp"),
    ],
)
def test_nonpositive_timestamp_is_source_mismatch(tmp_path, second, reason):
    path, recording = _source_case(tmp_path)
    lines = path.read_bytes().splitlines()
    lines[3] = second.encode("ascii") + b",1,2"
    path.write_bytes(b"\n".join(lines) + b"\n")
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.outcome == SOURCE_CHANGED_OR_MISMATCHED
    assert result.failure_reason == reason


def test_alternate_time_like_column_is_never_selected(tmp_path):
    path, recording = _source_case(tmp_path)
    _same_size_replace(path, b"Time(s)", b"OtherTm")
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.outcome == SOURCE_CHANGED_OR_MISMATCHED
    assert result.failure_reason == "header_mismatch"


def test_malformed_row_and_unexpected_row_count_are_refused(tmp_path):
    path, recording = _source_case(tmp_path)
    payload = path.read_bytes()
    _same_size_replace(path, b",1,2\n", b",1  \n")
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.failure_reason == "malformed_data_row"
    path.write_bytes(payload + b"10000,1,2\n")
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.failure_reason == "starting_size_mismatch"


def test_same_size_changed_data_bytes_are_refused_by_sha(tmp_path):
    path, recording = _source_case(tmp_path)
    _same_size_replace(path, b",1,2\n", b",3,4\n")
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.outcome == SOURCE_CHANGED_OR_MISMATCHED
    assert result.failure_reason == "source_sha256_mismatch"
    assert result.observed_source_sha256 != recording.source.sha256


def test_size_mismatch_missing_path_and_directory_are_refused(tmp_path):
    path, recording = _source_case(tmp_path)
    path.write_bytes(path.read_bytes() + b"x")
    assert evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=path
    ).failure_reason == "starting_size_mismatch"
    assert evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=tmp_path / "missing"
    ).failure_reason == "path_not_regular_file"
    assert evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=tmp_path
    ).failure_reason == "path_not_regular_file"


@pytest.mark.parametrize("changed_field", ["mtime_ns", "size"])
def test_before_after_source_change_is_deterministically_refused(
    tmp_path, monkeypatch, changed_field
):
    path, recording = _source_case(tmp_path)
    real_stat = subject._stat_source
    calls = 0

    def changed(current):
        nonlocal calls
        calls += 1
        result = real_stat(current)
        if calls == 1:
            return result
        return replace(result, **{changed_field: getattr(result, changed_field) + 1})

    monkeypatch.setattr(subject, "_stat_source", changed)
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.outcome == SOURCE_CHANGED_OR_MISMATCHED
    assert result.failure_reason == "source_changed_during_evaluation"


def test_cancellation_before_open_during_scan_and_before_success(tmp_path, monkeypatch):
    path, recording = _source_case(tmp_path, nominal=0.05)
    opened = 0
    real_open = Path.open

    def counted(self, *args, **kwargs):
        nonlocal opened
        if self == path:
            opened += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    immediate = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=path, cancellation_requested=lambda: True
    )
    assert immediate.outcome == EVALUATION_INTERRUPTED and opened == 0

    path, recording = _source_case(tmp_path, nominal=0.1)
    checks = 0
    def during():
        nonlocal checks
        checks += 1
        return checks == 2
    middle = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=path, cancellation_requested=during
    )
    assert middle.outcome == EVALUATION_INTERRUPTED
    assert middle.valid_row_count_evaluated == 0

    checks = 0
    def final():
        nonlocal checks
        checks += 1
        return checks == 2
    ending = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=path, cancellation_requested=final
    )
    assert ending.outcome == EVALUATION_INTERRUPTED
    assert ending.failure_reason == "evaluation_cancelled"


def test_data_is_opened_once_and_classifier_runs_for_every_interval(tmp_path, monkeypatch):
    path, recording = _source_case(tmp_path)
    opens = classifications = 0
    real_open = Path.open
    real_classifier = subject.classify_continuous_rwd_interval

    def counted_open(self, *args, **kwargs):
        nonlocal opens
        if self == path and args and args[0] == "rb":
            opens += 1
        return real_open(self, *args, **kwargs)

    def counted_classifier(*args, **kwargs):
        nonlocal classifications
        classifications += 1
        return real_classifier(*args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    monkeypatch.setattr(subject, "classify_continuous_rwd_interval", counted_classifier)
    result = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=path)
    assert result.outcome == CONTINUITY_PASSED
    assert opens == 1
    assert classifications == recording.cadence.positive_interval_count


def test_invalid_authority_and_callback_raise_narrow_exception(tmp_path):
    path, recording = _source_case(tmp_path)
    with pytest.raises(ContinuousRwdDiscontinuityEvaluationError):
        evaluate_continuous_rwd_timestamp_continuity(object(), source_path=path)
    with pytest.raises(ContinuousRwdDiscontinuityEvaluationError):
        evaluate_continuous_rwd_timestamp_continuity(
            recording, source_path=path, cancellation_requested=False
        )


def test_module_has_no_forbidden_dependencies_or_serialization_surface():
    tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    imports = {
        node.names[0].name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = ("pandas", "numpy", "metadata", "resampl", "target_grid", "pipeline.execution")
    assert not any(any(token in name.lower() for token in forbidden) for name in imports)
    assert not any(name == "gui" or name.startswith("gui.") for name in imports)
    public_functions = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    assert public_functions == ["evaluate_continuous_rwd_timestamp_continuity"]
    assert not any("serializ" in node.name.lower() for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    assert "inspect_continuous_rwd_acquisition_folder" not in Path(subject.__file__).read_text(encoding="utf-8")
