from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import inspect
import math
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from photometry_pipeline import guided_continuous_rwd_block_plan as block_subject
from photometry_pipeline import guided_continuous_rwd_target_grid as grid_subject
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    CONTINUITY_PASSED,
    evaluate_continuous_rwd_timestamp_continuity,
)
from photometry_pipeline.guided_continuous_rwd_projection import (
    GuidedContinuousRwdProjectedBlock,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    build_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    GuidedContinuousRwdReviewBinding,
    build_guided_continuous_rwd_review_binding,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from photometry_pipeline.io import rwd_continuous_projection_reader as subject
from photometry_pipeline.io.rwd_continuous_source import (
    inspect_continuous_rwd_acquisition_folder,
)


@dataclass(frozen=True)
class _Case:
    source: Path
    binding: GuidedContinuousRwdReviewBinding
    grid: grid_subject.GuidedContinuousRwdTargetGridDescription
    plan: block_subject.GuidedContinuousRwdBlockPlan


def _draft(folder: Path, roi_ids=("CH1", "CH2", "CH3")):
    return GuidedNewAnalysisDraftPlan(
        input_source_path=str(folder),
        resolved_input_source_path=str(folder),
        input_format="rwd",
        acquisition_mode="continuous",
        continuous_window_sec=600.0,
        continuous_step_sec=600.0,
        discovered_roi_ids=list(roi_ids),
        included_roi_ids=["CH2", "CH1"],
        excluded_roi_ids=["CH3"],
        output_base_path=str(folder / "output"),
        global_correction_strategy="global_linear_regression",
        feature_event_profile_id="default",
        feature_event_values={"peak_threshold_k": 8.0},
    )


def _source_text(
    *,
    terminal: bool = False,
    timestamp_column: str = "Time(s)",
    raw_times=None,
) -> str:
    if raw_times is None:
        raw_times = [1000, 101000, 201000, 301000, 401000, 501000, 601000]
    suffix = "," if terminal else ""
    header = (
        f"{timestamp_column},CH1-410,CH1-470,CH2-410,CH2-470,"
        f"CH3-410,CH3-470,Notes{suffix}\n"
    )
    rows = []
    for index, timestamp in enumerate(raw_times):
        rows.append(
            f"{timestamp},{10 + index},{20 + 2 * index},"
            f"{100 + 3 * index},{200 + 4 * index},"
            f"{1000 + index},{2000 + index},unselected{suffix}\n"
        )
    return "metadata padding for same-size mutations................\n" + header + "".join(rows)


def _build_case(
    tmp_path: Path,
    *,
    terminal: bool = False,
    timestamp_column: str = "Time(s)",
    raw_times=None,
) -> _Case:
    folder = tmp_path / "recording"
    folder.mkdir(parents=True)
    source = folder / "Fluorescence.csv"
    source.write_text(
        _source_text(
            terminal=terminal,
            timestamp_column=timestamp_column,
            raw_times=raw_times,
        ),
        encoding="utf-8",
        newline="",
    )
    inspection = inspect_continuous_rwd_acquisition_folder(folder)
    assert inspection.status == "completed"
    recording = build_guided_continuous_rwd_recording_description(
        inspection,
        included_roi_ids=("CH2", "CH1"),
    )
    evaluation = evaluate_continuous_rwd_timestamp_continuity(
        recording,
        source_path=source,
    )
    assert evaluation.outcome == CONTINUITY_PASSED
    binding = build_guided_continuous_rwd_review_binding(
        _draft(folder),
        recording=recording,
        continuity_evaluation=evaluation,
        current_source_path=source,
    )
    grid = grid_subject.build_guided_continuous_rwd_target_grid(
        recording,
        evaluation,
    )
    plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    return _Case(source, binding, grid, plan)


@pytest.fixture(autouse=True)
def _small_c2_blocks(monkeypatch):
    monkeypatch.setattr(block_subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", 3)


@pytest.fixture
def case(tmp_path):
    return _build_case(tmp_path)


def _iterator(case: _Case, **changes):
    values = dict(
        review_binding=case.binding,
        target_grid=case.grid,
        block_plan=case.plan,
    )
    values.update(changes)
    return subject.iter_project_guided_continuous_rwd_blocks(**values)


def _mutate_same_size(source: Path, transform):
    original = source.read_text(encoding="utf-8")
    preamble, body = original.split("\n", 1)
    changed_body = transform(body)
    delta = len(body.encode()) - len(changed_body.encode())
    if delta >= 0:
        changed_preamble = preamble + (" " * delta)
    else:
        changed_preamble = preamble[:delta]
    changed = changed_preamble + "\n" + changed_body
    assert len(changed.encode()) == len(original.encode())
    source.write_text(changed, encoding="utf-8", newline="")


def test_valid_source_yields_all_canonical_projected_blocks(case):
    results = list(_iterator(case))
    assert len(results) == case.plan.block_count == 3
    assert all(isinstance(result, GuidedContinuousRwdProjectedBlock) for result in results)
    assert [result.block_index for result in results] == [0, 1, 2]
    assert [
        (result.start_target_index, result.stop_target_index) for result in results
    ] == [(0, 3), (3, 6), (6, 7)]
    assert [index for result in results for index in range(
        result.start_target_index, result.stop_target_index
    )] == list(range(case.grid.target_sample_count))


def test_exact_provenance_shapes_roi_order_and_raw_amplitudes(case):
    results = list(_iterator(case))
    recording = case.binding.recording
    assert [(item.source_row_start, item.source_row_stop) for item in results] == [
        (0, 3), (3, 6), (6, 7)
    ]
    for result, block in zip(results, case.plan.iter_blocks()):
        assert result.recording_identity == recording.recording_identity
        assert result.source_content_identity == recording.source.source_content_identity
        assert result.target_grid_identity == case.grid.target_grid_identity
        assert result.included_roi_ids == ("CH1", "CH2")
        assert (result.start_target_index, result.stop_target_index) == (
            block.start_target_index,
            block.stop_target_index,
        )
    combined_control = np.vstack([item.control_values for item in results])
    combined_signal = np.vstack([item.signal_values for item in results])
    np.testing.assert_array_equal(combined_control[:, 0], np.arange(10, 17))
    np.testing.assert_array_equal(combined_signal[:, 0], np.arange(20, 34, 2))
    np.testing.assert_array_equal(combined_control[:, 1], np.arange(100, 121, 3))
    np.testing.assert_array_equal(combined_signal[:, 1], np.arange(200, 225, 4))


def test_nonzero_origin_normalization_subtracts_before_scaling(tmp_path):
    case = _build_case(
        tmp_path,
        timestamp_column="TimeStamp",
        raw_times=[1000, 101000, 201000, 301000, 401000, 501000, 601000],
    )
    times = np.concatenate([item.target_elapsed_seconds for item in _iterator(case)])
    np.testing.assert_array_equal(times, np.arange(0.0, 601.0, 100.0))
    assert times[0] == 0.0


def test_source_opens_once_and_c3a_is_called_once_per_block(case, monkeypatch):
    opens = []
    calls = []
    original_open = Path.open
    original_project = subject.project_guided_continuous_rwd_block

    def tracked_open(path, *args, **kwargs):
        if path == case.source:
            opens.append((args, kwargs))
        return original_open(path, *args, **kwargs)

    def tracked_project(*args, **kwargs):
        calls.append((args[2].block_index, kwargs["source_row_start"], kwargs["source_row_stop"]))
        return original_project(*args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(subject, "project_guided_continuous_rwd_block", tracked_project)
    results = list(_iterator(case))
    assert len(opens) == 1
    assert opens[0][0] == ("rb",)
    assert calls == [(0, 0, 3), (1, 3, 6), (2, 6, 7)]
    assert len(results) == len(calls)


def test_final_verification_is_deferred_until_exhaustion(case):
    iterator = _iterator(case)
    first = next(iterator)
    assert isinstance(first, GuidedContinuousRwdProjectedBlock)
    assert not hasattr(first, "complete_source_verified")
    _mutate_same_size(case.source, lambda body: body.replace(",10,20,", ",11,20,", 1))
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(iterator)
    assert caught.value.category in {"source_instability", "source_content_mismatch"}


def test_early_abandonment_has_no_success_indicator(case):
    iterator = _iterator(case)
    first = next(iterator)
    assert not hasattr(iterator, "complete")
    assert not hasattr(first, "complete_source_verified")
    iterator.close()


def test_module_documents_provisional_yield_semantics():
    documentation = inspect.getdoc(subject).lower()
    assert "provisional" in documentation
    assert "exhausts normally" in documentation
    assert "incomplete projection run" in documentation


@pytest.mark.parametrize(
    "transform",
    [
        lambda body: body.replace("Time(s)", "Mime(s)", 1),
        lambda body: body.replace("Time(s)", "TimeStamp", 1),
        lambda body: body.replace("CH1-410,CH1-470", "CH1-470,CH1-410", 1),
        lambda body: body.replace("CH2-410", "MISSING", 1),
        lambda body: body.replace(",Notes\n", ",Notes,Extra\n", 1),
    ],
)
def test_header_drift_is_refused_before_projection(case, transform):
    _mutate_same_size(case.source, transform)
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case))
    assert caught.value.category == "parser_header_mismatch"


def test_wrong_header_row_is_refused(case):
    _mutate_same_size(case.source, lambda body: "not,the,header\n" + body)
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case))
    assert caught.value.category == "parser_header_mismatch"


def test_terminal_empty_field_parity_succeeds_when_accepted(tmp_path):
    case = _build_case(tmp_path, terminal=True)
    assert case.binding.recording.source.raw_columns[-1] == ""
    assert len(list(_iterator(case))) == 3


@pytest.mark.parametrize("terminal", [False, True])
def test_terminal_empty_field_drift_is_refused(tmp_path, terminal):
    case = _build_case(tmp_path, terminal=terminal)
    if terminal:
        _mutate_same_size(case.source, lambda body: body.replace(",\n", "\n", 1))
    else:
        _mutate_same_size(case.source, lambda body: body.replace("Notes\n", "Notes,\n", 1))
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case))
    assert caught.value.category == "parser_header_mismatch"


def test_bom_and_surrounding_header_whitespace_follow_committed_normalization(tmp_path):
    folder = tmp_path / "recording"
    folder.mkdir()
    source = folder / "Fluorescence.csv"
    text = _source_text().replace("Time(s),", "\ufeff  Time(s)  ,", 1)
    source.write_text(text, encoding="utf-8", newline="")
    inspection = inspect_continuous_rwd_acquisition_folder(folder)
    assert inspection.status == "completed"
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("CH1", "CH2")
    )
    evaluation = evaluate_continuous_rwd_timestamp_continuity(recording, source_path=source)
    binding = build_guided_continuous_rwd_review_binding(
        _draft(folder), recording=recording, continuity_evaluation=evaluation,
        current_source_path=source,
    )
    grid = grid_subject.build_guided_continuous_rwd_target_grid(recording, evaluation)
    plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    assert len(list(subject.iter_project_guided_continuous_rwd_blocks(binding, grid, plan))) == 3


def test_malformed_csv_quoting_and_utf8_are_refused(case):
    _mutate_same_size(case.source, lambda body: body.replace("unselected\n", '"unterminated\n', 1))
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case))
    assert caught.value.category == "malformed_data_row"

    case = _build_case(case.source.parent.parent / "second")
    data = bytearray(case.source.read_bytes())
    data[data.index(b"unselected")] = 0xFF
    case.source.write_bytes(data)
    with pytest.raises(subject.ContinuousRwdProjectionReaderError):
        list(_iterator(case))


@pytest.mark.parametrize(
    ("old", "new", "category"),
    [
        (",unselected\n", "\n", "malformed_data_row"),
        (",unselected\n", ",unselected,extra\n", "malformed_data_row"),
        (",10,20,", ",,20,", "selected_channel_value_violation"),
        ("1000,10,", "oops,10,", "timestamp_violation"),
        (",10,20,", ",xx,20,", "selected_channel_value_violation"),
        (",100,200,", ",100,xxx,", "selected_channel_value_violation"),
        ("1000,10,", "nan,10,", "timestamp_violation"),
        ("1000,10,", "inf,10,", "timestamp_violation"),
        (",10,20,", ",nan,20,", "selected_channel_value_violation"),
        (",100,200,", ",100,inf,", "selected_channel_value_violation"),
    ],
)
def test_malformed_rows_and_selected_values_fail_closed(case, old, new, category):
    _mutate_same_size(case.source, lambda body: body.replace(old, new, 1))
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case))
    assert caught.value.category == category


def test_unselected_malformed_content_is_not_numerically_parsed(case):
    _mutate_same_size(case.source, lambda body: body.replace("unselected", "bad-number", 1))
    iterator = _iterator(case)
    assert isinstance(next(iterator), GuidedContinuousRwdProjectedBlock)
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(iterator)
    assert caught.value.category == "source_content_mismatch"


def test_excluded_roi_values_are_not_parsed_into_projection(case):
    _mutate_same_size(case.source, lambda body: body.replace(",1000,2000,", ",xxxx,yyyy,", 1))
    iterator = _iterator(case)
    assert next(iterator).included_roi_ids == ("CH1", "CH2")
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(iterator)
    assert caught.value.category == "source_content_mismatch"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("101000,", "1000,"),
        ("201000,", "50000,"),
    ],
)
def test_duplicate_and_backward_timestamps_are_refused(case, old, new):
    _mutate_same_size(case.source, lambda body: body.replace(old, new, 1))
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case))
    assert caught.value.category == "timestamp_violation"


def test_first_origin_and_final_endpoint_drift_are_refused(case):
    _mutate_same_size(case.source, lambda body: body.replace("1000,", "1001,", 1))
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case))
    assert caught.value.category == "timestamp_violation"

    case = _build_case(case.source.parent.parent / "second")
    _mutate_same_size(case.source, lambda body: body.replace("601000,", "601001,", 1))
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case))
    assert caught.value.category == "timestamp_violation"


def test_middle_blocks_reuse_bracketing_rows_without_target_overlap(tmp_path):
    case = _build_case(
        tmp_path,
        raw_times=[1000, 101000, 191000, 311000, 401000, 501000, 601000],
    )
    assert case.binding.continuity_evaluation.outcome == CONTINUITY_PASSED
    results = list(_iterator(case))
    bounds = [(item.source_row_start, item.source_row_stop) for item in results]
    assert bounds[0] == (0, 4)
    assert bounds[1][0] == 2
    assert set(range(*bounds[0])) & set(range(*bounds[1]))
    assert results[0].stop_target_index == results[1].start_target_index


def test_final_block_before_endpoint_still_consumes_trailing_rows(case):
    iterator = _iterator(case)
    first, second, final = next(iterator), next(iterator), next(iterator)
    assert final.target_elapsed_seconds[-1] == case.grid.last_target_elapsed_seconds
    assert final.target_elapsed_seconds[-1] == 600.0
    with pytest.raises(StopIteration):
        next(iterator)
    assert [first.block_index, second.block_index, final.block_index] == [0, 1, 2]


def test_final_target_before_accepted_endpoint_is_verified(tmp_path):
    case = _build_case(
        tmp_path,
        raw_times=[1000, 101000, 201000, 301000, 401000, 501000, 601050],
    )
    assert case.grid.last_target_elapsed_seconds == 600.0
    assert case.binding.recording.time.measured_support_end_seconds == pytest.approx(600.05)
    results = list(_iterator(case))
    assert results[-1].target_elapsed_seconds[-1] == 600.0
    assert results[-1].source_row_stop == 7


def test_same_size_early_eof_refuses_missing_right_support(case):
    def remove_final_row(body):
        lines = body.splitlines(keepends=True)
        return "".join(lines[:-1])

    _mutate_same_size(case.source, remove_final_row)
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case))
    assert caught.value.category == "insufficient_right_support"


def test_buffer_is_bounded_and_ceiling_is_enforced(case, monkeypatch):
    retained = []
    original = subject.project_guided_continuous_rwd_block

    def tracked(*args, **kwargs):
        retained.append(kwargs["source_elapsed_seconds"].size)
        return original(*args, **kwargs)

    monkeypatch.setattr(subject, "project_guided_continuous_rwd_block", tracked)
    list(_iterator(case))
    assert retained == [3, 3, 1]
    assert max(retained) < case.binding.recording.source.valid_timestamp_count

    monkeypatch.setattr(subject, "_buffer_row_limit", lambda *_args: 1)
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case))
    assert caught.value.category == "bounded_buffer_limit_exceeded"


def test_missing_nonfile_starting_size_and_relocated_source(case, tmp_path):
    missing = replace(case.binding, current_source_path=str(tmp_path / "Fluorescence.csv"))
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case, review_binding=missing))
    assert caught.value.category == "source_unavailable"

    directory = tmp_path / "directory" / "Fluorescence.csv"
    directory.mkdir(parents=True)
    nonfile = replace(case.binding, current_source_path=str(directory))
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case, review_binding=nonfile))
    assert caught.value.category == "source_unavailable"

    enlarged = case.source.read_bytes() + b"x"
    case.source.write_bytes(enlarged)
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case))
    assert caught.value.category == "source_content_mismatch"


def test_content_identical_relocation_ignores_historical_mtime(case, tmp_path):
    relocated = tmp_path / "relocated" / "Fluorescence.csv"
    relocated.parent.mkdir()
    shutil.copyfile(case.source, relocated)
    os.utime(relocated, ns=(1_000_000_000, 2_000_000_000))
    binding = build_guided_continuous_rwd_review_binding(
        _draft(relocated.parent),
        recording=case.binding.recording,
        continuity_evaluation=case.binding.continuity_evaluation,
        current_source_path=relocated,
    )
    assert len(list(_iterator(case, review_binding=binding))) == 3


@pytest.mark.parametrize("mutation", ["touch", "truncate", "enlarge"])
def test_source_changed_after_provisional_yield_is_refused(case, mutation):
    iterator = _iterator(case)
    next(iterator)
    if mutation == "touch":
        facts = case.source.stat()
        os.utime(case.source, ns=(facts.st_atime_ns, facts.st_mtime_ns + 1_000_000))
    elif mutation == "truncate":
        case.source.write_bytes(case.source.read_bytes()[:-1])
    else:
        case.source.write_bytes(case.source.read_bytes() + b"x")
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(iterator)
    assert caught.value.category in {
        "source_instability", "source_content_mismatch", "malformed_data_row"
    }


def test_same_size_valid_content_change_fails_final_sha(case):
    _mutate_same_size(case.source, lambda body: body.replace(",10,20,", ",11,20,", 1))
    iterator = _iterator(case)
    yielded = []
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        while True:
            yielded.append(next(iterator))
    assert caught.value.category == "source_content_mismatch"
    assert yielded


@pytest.mark.parametrize(
    "changes",
    [
        {"outcome": "not-passed"},
        {"failure_reason": "bad"},
        {"recording_identity": "e" * 64},
        {"source_content_identity": "e" * 64},
        {"parser_interpretation_identity": "e" * 64},
        {"cadence_evidence_identity": "e" * 64},
        {"valid_row_count_evaluated": 8},
        {"positive_interval_count_evaluated": 7},
        {"normal_interval_count": 5},
        {"observed_source_size_bytes": 1},
        {"observed_source_sha256": "e" * 64},
    ],
)
def test_b2_binding_mismatches_are_refused_before_open(case, changes, monkeypatch):
    evaluation = replace(case.binding.continuity_evaluation, **changes)
    binding = replace(case.binding, continuity_evaluation=evaluation)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: pytest.fail("opened"))
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        subject.iter_project_guided_continuous_rwd_blocks(binding, case.grid, case.plan)
    assert caught.value.category == "invalid_authority_binding"


def test_wrong_b3_and_malformed_b1_are_refused(case):
    with pytest.raises(subject.ContinuousRwdProjectionReaderError):
        subject.iter_project_guided_continuous_rwd_blocks(object(), case.grid, case.plan)
    malformed_recording = replace(case.binding.recording, schema_version="bad")
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        subject.iter_project_guided_continuous_rwd_blocks(
            replace(case.binding, recording=malformed_recording), case.grid, case.plan
        )
    assert caught.value.__cause__ is not None


def _resign_grid(grid, **changes):
    changed = replace(grid, **changes, target_grid_identity="")
    return replace(
        changed,
        target_grid_identity=grid_subject.compute_guided_continuous_rwd_target_grid_identity(
            changed
        ),
    )


def test_c1_and_c2_cross_binding_refusals(case):
    malformed_grid = replace(case.grid, schema_version="bad")
    with pytest.raises(subject.ContinuousRwdProjectionReaderError):
        subject.iter_project_guided_continuous_rwd_blocks(
            case.binding, malformed_grid, case.plan
        )
    recording_mismatch = _resign_grid(case.grid, recording_identity="e" * 64)
    continuity_mismatch = _resign_grid(
        case.grid, continuity_evaluation_identity="e" * 64
    )
    for grid in (recording_mismatch, continuity_mismatch):
        with pytest.raises(subject.ContinuousRwdProjectionReaderError):
            subject.iter_project_guided_continuous_rwd_blocks(
                case.binding,
                grid,
                block_subject.build_guided_continuous_rwd_block_plan(grid),
            )
    with pytest.raises(subject.ContinuousRwdProjectionReaderError):
        subject.iter_project_guided_continuous_rwd_blocks(
            case.binding, case.grid, replace(case.plan, schema_version="bad")
        )
    for plan in (
        replace(case.plan, target_grid_identity="e" * 64),
        replace(case.plan, target_sample_count=8),
    ):
        with pytest.raises(subject.ContinuousRwdProjectionReaderError):
            subject.iter_project_guided_continuous_rwd_blocks(
                case.binding, case.grid, plan
            )


def test_b3_filename_and_roi_authority_refusals(case):
    with pytest.raises(subject.ContinuousRwdProjectionReaderError):
        subject.iter_project_guided_continuous_rwd_blocks(
            replace(case.binding, current_source_path=str(case.source.with_name("Other.csv"))),
            case.grid,
            case.plan,
        )
    malformed_roi = replace(
        case.binding.recording.roi,
        included_roi_ids=("CH1",),
    )
    malformed_recording = replace(case.binding.recording, roi=malformed_roi)
    with pytest.raises(subject.ContinuousRwdProjectionReaderError):
        subject.iter_project_guided_continuous_rwd_blocks(
            replace(case.binding, recording=malformed_recording), case.grid, case.plan
        )


class _CancelOnCall:
    def __init__(self, call):
        self.call = call
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls == self.call


@pytest.mark.parametrize("call", [1, 2, 3])
def test_cancellation_before_open_after_header_or_before_projection(case, call):
    callback = _CancelOnCall(call)
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case, cancellation_requested=callback))
    assert caught.value.category == "projection_interrupted"


def test_cancellation_during_rows_and_after_yield(case, monkeypatch):
    monkeypatch.setattr(subject, "_CANCELLATION_ROW_INTERVAL", 1)
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case, cancellation_requested=_CancelOnCall(3)))
    assert caught.value.category == "projection_interrupted"

    monkeypatch.setattr(subject, "_CANCELLATION_ROW_INTERVAL", 10_000)
    callback = _CancelOnCall(4)
    iterator = _iterator(case, cancellation_requested=callback)
    assert next(iterator).block_index == 0
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        next(iterator)
    assert caught.value.category == "projection_interrupted"


def test_cancellation_before_final_verification_and_callback_failure(case):
    callback = _CancelOnCall(9)
    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case, cancellation_requested=callback))
    assert caught.value.category == "projection_interrupted"

    def broken():
        raise RuntimeError("callback failure")

    with pytest.raises(subject.ContinuousRwdProjectionReaderError) as caught:
        list(_iterator(case, cancellation_requested=broken))
    assert caught.value.category == "projection_interrupted"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_public_signature_dependencies_and_scope_are_narrow():
    assert set(inspect.signature(subject.iter_project_guided_continuous_rwd_blocks).parameters) == {
        "review_binding", "target_grid", "block_plan", "cancellation_requested"
    }
    tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    forbidden = {
        "pandas", "h5py", "scipy", "threading", "multiprocessing", "gui",
        "photometry_pipeline.pipeline", "photometry_pipeline.io.hdf5_cache",
        "photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation.evaluate_continuous_rwd_timestamp_continuity",
    }
    assert imports.isdisjoint(forbidden)
    source = inspect.getsource(subject)
    for token in (
        "evaluate_continuous_rwd_timestamp_continuity(", "_project_bounded_arrays(",
        "pandas", "h5py", "checkpoint", "serialize", "output_path",
        "block_size", "projection_policy", "random_block", "Thread", "Process",
    ):
        assert token not in source
    assert "project_guided_continuous_rwd_block(" in source
    assert not any(name.startswith("project_block_at") for name in vars(subject))
