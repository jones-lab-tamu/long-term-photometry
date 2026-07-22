from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
import inspect
from pathlib import Path

import numpy as np
import pytest

from photometry_pipeline import guided_continuous_rwd_block_plan as block_subject
from photometry_pipeline import guided_continuous_rwd_projection as subject
from photometry_pipeline import guided_continuous_rwd_target_grid as grid_subject


def _grid(
    count: int = 31,
    *,
    cadence: Fraction = Fraction(1, 10),
) -> grid_subject.GuidedContinuousRwdTargetGridDescription:
    support = (count - 1) * cadence
    draft = grid_subject.GuidedContinuousRwdTargetGridDescription(
        schema_name=grid_subject.SCHEMA_NAME,
        schema_version=grid_subject.SCHEMA_VERSION,
        grid_policy_name=grid_subject.GRID_POLICY_NAME,
        grid_policy_version=grid_subject.GRID_POLICY_VERSION,
        recording_identity="a" * 64,
        continuity_evaluation_identity="b" * 64,
        cadence_seconds_numerator=cadence.numerator,
        cadence_seconds_denominator=cadence.denominator,
        source_support_end_seconds_numerator=support.numerator,
        source_support_end_seconds_denominator=support.denominator,
        target_sample_count=count,
        target_grid_identity="",
    )
    return replace(
        draft,
        target_grid_identity=(
            grid_subject.compute_guided_continuous_rwd_target_grid_identity(draft)
        ),
    )


def _case(count: int = 31, *, cadence: Fraction = Fraction(1, 10)):
    grid = _grid(count, cadence=cadence)
    plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    return grid, plan, plan.block_at(0)


def _arrays():
    elapsed = np.array([0.0, 0.8, 2.2, 3.0], dtype=np.float64)
    control = np.column_stack((10.0 + elapsed, 100.0 + 2.0 * elapsed))
    signal = np.column_stack((20.0 + 3.0 * elapsed, 200.0 + 4.0 * elapsed))
    return elapsed, control, signal


def _project(**changes):
    grid, plan, block = _case()
    elapsed, control, signal = _arrays()
    arguments = dict(
        target_grid=grid,
        block_plan=plan,
        block=block,
        recording_identity="c" * 64,
        source_content_identity="d" * 64,
        included_roi_ids=("ROI2", "ROI1"),
        source_row_start=17,
        source_row_stop=21,
        source_elapsed_seconds=elapsed,
        source_control_values=control,
        source_signal_values=signal,
    )
    arguments.update(changes)
    return subject.project_guided_continuous_rwd_block(**arguments)


def test_valid_contract_is_frozen_minimal_and_retains_exact_provenance():
    result = _project()
    assert (result.schema_name, result.schema_version) == (
        "guided_continuous_rwd_projected_block",
        "v1",
    )
    assert (result.projection_policy_name, result.projection_policy_version) == (
        "continuous-rwd-linear-projection",
        "v1",
    )
    assert result.recording_identity == "c" * 64
    assert result.source_content_identity == "d" * 64
    assert result.target_grid_identity == _grid().target_grid_identity
    assert (result.block_index, result.start_target_index, result.stop_target_index) == (
        0,
        0,
        31,
    )
    assert result.included_roi_ids == ("ROI2", "ROI1")
    assert (result.source_row_start, result.source_row_stop) == (17, 21)
    assert {field.name for field in fields(result)} == {
        "schema_name", "schema_version", "projection_policy_name",
        "projection_policy_version", "recording_identity",
        "source_content_identity", "target_grid_identity", "block_index",
        "start_target_index", "stop_target_index", "included_roi_ids",
        "source_row_start", "source_row_stop", "target_elapsed_seconds",
        "control_values", "signal_values",
    }
    for forbidden in (
        "projected_block_identity", "source_path", "parser", "continuity",
        "correction", "feature", "cache", "output",
    ):
        assert not hasattr(result, forbidden)
    with pytest.raises(FrozenInstanceError):
        result.block_index = 1


def test_array_shapes_dtypes_read_only_and_mutation_refusal():
    result = _project()
    assert result.target_elapsed_seconds.shape == (31,)
    assert result.control_values.shape == (31, 2)
    assert result.signal_values.shape == (31, 2)
    for array in (
        result.target_elapsed_seconds,
        result.control_values,
        result.signal_values,
    ):
        assert array.dtype == np.float64
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = -1.0


def test_caller_arrays_are_not_mutated_or_exposed():
    elapsed, control, signal = _arrays()
    originals = tuple(array.copy() for array in (elapsed, control, signal))
    result = _project(
        source_elapsed_seconds=elapsed,
        source_control_values=control,
        source_signal_values=signal,
    )
    for supplied, original in zip((elapsed, control, signal), originals):
        np.testing.assert_array_equal(supplied, original)
        assert supplied.flags.writeable
    assert not np.shares_memory(result.target_elapsed_seconds, elapsed)
    assert not np.shares_memory(result.control_values, control)
    assert not np.shares_memory(result.signal_values, signal)


def test_global_target_coordinates_and_middle_range_use_no_local_origin():
    grid = _grid(200_005, cadence=Fraction(1, 10))
    plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    block = plan.block_at(1)
    start = block.start_target_index
    stop = block.stop_target_index
    elapsed = np.array([start / 10.0, (stop - 1) / 10.0])
    values = np.column_stack((elapsed, elapsed + 1.0))
    result = subject.project_guided_continuous_rwd_block(
        grid, plan, block,
        recording_identity="c" * 64,
        source_content_identity="d" * 64,
        included_roi_ids=("ROI1", "ROI2"),
        source_row_start=50,
        source_row_stop=52,
        source_elapsed_seconds=elapsed,
        source_control_values=values,
        source_signal_values=values + 10.0,
    )
    assert result.target_elapsed_seconds[0] == start * 0.1
    assert result.target_elapsed_seconds[-1] == (stop - 1) * 0.1
    assert result.target_elapsed_seconds[0] != 0.0
    expected = np.arange(start, stop, dtype=np.int64).astype(np.float64) * 0.1
    np.testing.assert_array_equal(result.target_elapsed_seconds, expected)


def test_representative_high_index_coordinates_use_one_global_multiplication():
    grid = _grid(3_000_005, cadence=Fraction(99997, 1_000_000))
    start, stop = 2_345_670, 2_345_675
    target = subject._target_coordinates(grid, start, stop)
    expected = np.arange(start, stop, dtype=np.int64).astype(np.float64) * np.float64(
        grid.cadence_fraction
    )
    np.testing.assert_array_equal(target, expected)


def test_collapsed_float64_target_coordinates_are_refused():
    maximum = grid_subject.MAX_TARGET_SAMPLE_COUNT
    grid = _grid(maximum, cadence=Fraction(1, 1))
    with pytest.raises(subject.ContinuousRwdProjectionError, match="resolution"):
        subject._target_coordinates(grid, maximum - 100_000, maximum)


def test_nonfinite_float64_target_coordinates_are_refused():
    grid = _grid(2, cadence=Fraction(10**400, 1))
    with pytest.raises(subject.ContinuousRwdProjectionError, match="finite float64"):
        subject._target_coordinates(grid, 0, 2)


def test_exact_hits_midpoints_irregular_intervals_and_roi_order():
    result = _project()
    times = result.target_elapsed_seconds
    expected_control = np.column_stack((10.0 + times, 100.0 + 2.0 * times))
    expected_signal = np.column_stack((20.0 + 3.0 * times, 200.0 + 4.0 * times))
    np.testing.assert_allclose(result.control_values, expected_control)
    np.testing.assert_allclose(result.signal_values, expected_signal)
    assert result.control_values[0, 0] == 10.0
    assert result.control_values[8, 0] == 10.8
    assert result.control_values[15, 0] == 11.5
    assert result.signal_values[22, 1] == 208.8
    assert np.all(np.isfinite(result.control_values))
    assert np.all(np.isfinite(result.signal_values))


def test_control_signal_are_independent_and_amplitudes_are_not_normalized():
    result = _project()
    assert result.control_values[10, 0] == 11.0
    assert result.signal_values[10, 0] == 23.0
    assert result.control_values[10, 1] == 102.0
    assert result.signal_values[10, 1] == 204.0


@pytest.mark.parametrize(
    ("elapsed", "count", "expected_last"),
    [
        (np.array([0.0, 3.0]), 31, 3.0),
        (np.array([0.0, 3.5]), 31, 3.0),
    ],
)
def test_first_and_final_exact_or_bracketed_support(elapsed, count, expected_last):
    grid, plan, block = _case(count)
    values = np.column_stack((elapsed + 1.0, elapsed + 2.0))
    result = subject.project_guided_continuous_rwd_block(
        grid, plan, block,
        recording_identity="c" * 64,
        source_content_identity="d" * 64,
        included_roi_ids=("R1", "R2"),
        source_row_start=0,
        source_row_stop=elapsed.size,
        source_elapsed_seconds=elapsed,
        source_control_values=values,
        source_signal_values=values + 10.0,
    )
    assert result.target_elapsed_seconds[0] == 0.0
    assert result.target_elapsed_seconds[-1] == expected_last


def test_one_sample_final_block_accepts_one_exact_support_row():
    count = block_subject.MAXIMUM_OWNED_SAMPLES_PER_BLOCK + 1
    grid = _grid(count)
    plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    block = plan.block_at(1)
    elapsed = np.array([count / 10.0 - 0.1])
    values = np.array([[7.0, 11.0]])
    result = subject.project_guided_continuous_rwd_block(
        grid, plan, block,
        recording_identity="c" * 64,
        source_content_identity="d" * 64,
        included_roi_ids=("R1", "R2"),
        source_row_start=123,
        source_row_stop=124,
        source_elapsed_seconds=elapsed,
        source_control_values=values,
        source_signal_values=values + 10.0,
    )
    assert result.target_elapsed_seconds.shape == (1,)
    assert result.target_elapsed_seconds[0] == block.start_target_index * 0.1
    np.testing.assert_array_equal(result.control_values, values)


@pytest.mark.parametrize(
    ("elapsed", "message"),
    [
        (np.array([0.1, 3.0]), "at or before"),
        (np.array([0.0, 2.9]), "at or after"),
    ],
)
def test_missing_support_refuses_extrapolation_without_endpoint_clamping(
    elapsed, message
):
    values = np.column_stack((elapsed, elapsed + 1.0))
    with pytest.raises(subject.ContinuousRwdProjectionError, match=message):
        _project(
            source_row_stop=17 + elapsed.size,
            source_elapsed_seconds=elapsed,
            source_control_values=values,
            source_signal_values=values,
        )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("source_elapsed_seconds", np.zeros((2, 2)), "1-dimensional"),
        ("source_control_values", np.zeros(4), "2-dimensional"),
        ("source_signal_values", np.zeros(4), "2-dimensional"),
        ("source_elapsed_seconds", np.array([], dtype=float), "at least one"),
        ("source_elapsed_seconds", np.array([0.0, 0.8, 0.8, 3.0]), "strictly"),
        ("source_elapsed_seconds", np.array([0.0, 0.8, 0.7, 3.0]), "strictly"),
        ("source_elapsed_seconds", np.array(["0", "1"]), "numeric real"),
        ("source_control_values", np.array([["1", "2"]] * 4), "numeric real"),
        ("source_signal_values", np.array([["1", "2"]] * 4), "numeric real"),
        ("source_elapsed_seconds", np.array([0.0, np.nan, 2.2, 3.0]), "finite"),
        ("source_elapsed_seconds", np.array([0.0, np.inf, 2.2, 3.0]), "finite"),
        ("source_control_values", np.array([[1.0, np.nan]] * 4), "finite"),
        ("source_signal_values", np.array([[1.0, np.inf]] * 4), "finite"),
    ],
)
def test_source_array_rank_numeric_finite_and_monotonic_refusals(
    argument, value, message
):
    changes = {argument: value}
    if argument == "source_elapsed_seconds" and value.ndim == 1 and value.size:
        changes["source_row_stop"] = 17 + value.shape[0]
    with pytest.raises(subject.ContinuousRwdProjectionError, match=message):
        _project(**changes)


def test_source_row_and_matrix_shape_mismatches_are_refused():
    elapsed, control, signal = _arrays()
    cases = (
        ({"source_row_stop": 22}, "provenance"),
        ({"source_control_values": control[:-1]}, "control matrix shape"),
        ({"source_signal_values": signal[:, :1]}, "signal matrix shape"),
    )
    for changes, message in cases:
        with pytest.raises(subject.ContinuousRwdProjectionError, match=message):
            _project(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"source_row_start": True}, {"source_row_stop": 21.0},
        {"source_row_start": -1}, {"source_row_start": 21},
        {"source_row_stop": 17},
    ],
)
def test_invalid_source_row_bounds_are_refused(changes):
    with pytest.raises(subject.ContinuousRwdProjectionError, match="Source-row bounds"):
        _project(**changes)


@pytest.mark.parametrize("name", ["recording_identity", "source_content_identity"])
@pytest.mark.parametrize("value", [None, "a" * 63, "A" * 64, "g" * 64])
def test_malformed_scalar_identities_are_refused(name, value):
    with pytest.raises(subject.ContinuousRwdProjectionError, match="identity"):
        _project(**{name: value})


@pytest.mark.parametrize(
    "roi_ids",
    [None, [], (), ("R1", "R1"), ("",), (1,)],
)
def test_roi_contract_refusals(roi_ids):
    with pytest.raises(subject.ContinuousRwdProjectionError):
        _project(included_roi_ids=roi_ids)


def test_wrong_or_malformed_c1_is_refused_with_translated_cause():
    with pytest.raises(subject.ContinuousRwdProjectionError, match="target_grid"):
        _project(target_grid=object())
    malformed = replace(_grid(), schema_version="bad")
    with pytest.raises(subject.ContinuousRwdProjectionError, match="C1") as caught:
        _project(target_grid=malformed)
    assert isinstance(caught.value.__cause__, grid_subject.ContinuousRwdTargetGridError)


def test_wrong_or_malformed_c2_plan_is_refused_with_translated_cause():
    with pytest.raises(subject.ContinuousRwdProjectionError, match="block_plan"):
        _project(block_plan=object())
    grid, plan, _block = _case()
    malformed = replace(plan, schema_version="bad")
    with pytest.raises(subject.ContinuousRwdProjectionError, match="C2") as caught:
        _project(target_grid=grid, block_plan=malformed)
    assert isinstance(caught.value.__cause__, block_subject.ContinuousRwdBlockPlanError)


def test_c1_c2_identity_count_and_descriptor_cross_binding_refusals():
    grid, plan, block = _case()
    other_grid, other_plan, _ = _case(32)
    cases = (
        ({"block_plan": replace(plan, target_grid_identity="e" * 64)}, "identity"),
        ({"block_plan": replace(plan, target_sample_count=32)}, "count"),
        ({"block": object()}, "descriptor"),
        ({"block": replace(block, stop_target_index=30)}, "descriptor"),
        ({"target_grid": other_grid, "block_plan": other_plan, "block": block}, "descriptor"),
    )
    for changes, message in cases:
        with pytest.raises(subject.ContinuousRwdProjectionError, match=message):
            _project(**changes)


def _replacement_result(result, **changes):
    values = {field.name: getattr(result, field.name) for field in fields(result)}
    values.update(changes)
    return subject.GuidedContinuousRwdProjectedBlock(**values)


def _validate_result(result):
    grid, plan, block = _case()
    subject._validate_projected_block(
        result,
        grid,
        plan,
        block,
        _arrays()[0],
        recording_identity="c" * 64,
        source_content_identity="d" * 64,
        included_roi_ids=("ROI2", "ROI1"),
        source_row_start=17,
        source_row_stop=21,
    )


def _readonly(array):
    array.setflags(write=False)
    return array


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": "bad"},
        {"projection_policy_version": "bad"},
        {"recording_identity": "A" * 64},
        {"target_grid_identity": "0" * 64},
        {"block_index": 1},
        {"included_roi_ids": ("R", "R")},
        {"source_row_start": -1},
    ],
)
def test_result_validator_refuses_malformed_scalar_contract(changes):
    result = _project()
    with pytest.raises(subject.ContinuousRwdProjectionError):
        _validate_result(_replacement_result(result, **changes))


def test_result_validator_accepts_exact_supplied_provenance():
    _validate_result(_project())


@pytest.mark.parametrize(
    "changes",
    [
        {"recording_identity": "e" * 64},
        {"source_content_identity": "f" * 64},
        {"included_roi_ids": ("ROI1", "ROI3")},
        {"source_row_start": 117, "source_row_stop": 121},
        {"source_row_start": 18},
        {"source_row_stop": 22},
    ],
)
def test_result_validator_refuses_valid_form_provenance_substitution(changes):
    with pytest.raises(subject.ContinuousRwdProjectionError, match="provenance"):
        _validate_result(_replacement_result(_project(), **changes))


def test_result_validator_refuses_writable_wrong_dtype_shape_nonfinite_and_time_drift():
    result = _project()
    cases = []
    for name in ("target_elapsed_seconds", "control_values", "signal_values"):
        writable = getattr(result, name).copy()
        cases.append({name: writable})
    nonfinite_signal = _readonly(np.full_like(result.signal_values, np.nan))
    mismatched_target = result.target_elapsed_seconds.copy()
    mismatched_target[5] += 0.01
    mismatched_target = _readonly(mismatched_target)
    nonmonotonic_target = result.target_elapsed_seconds.copy()
    nonmonotonic_target[5] = nonmonotonic_target[4]
    nonmonotonic_target = _readonly(nonmonotonic_target)
    cases.extend(
        (
            {"target_elapsed_seconds": result.target_elapsed_seconds.astype(np.float32)},
            {"control_values": result.control_values[:-1]},
            {"signal_values": nonfinite_signal},
            {"target_elapsed_seconds": mismatched_target},
            {"target_elapsed_seconds": nonmonotonic_target},
        )
    )
    for changes in cases:
        with pytest.raises(subject.ContinuousRwdProjectionError):
            _validate_result(_replacement_result(result, **changes))


def test_partition_independence_for_exact_between_boundary_final_and_multiple_rois():
    grid = _grid(101)
    elapsed, control, signal = _arrays()
    ranges = ((0, 31), (7, 24), (8, 23), (10, 11), (22, 23))
    projections = {}
    for start, stop in ranges:
        target, projected_control, projected_signal = subject._project_bounded_arrays(
            grid, start, stop, elapsed, control, signal
        )
        for offset, index in enumerate(range(start, stop)):
            value = (target[offset], projected_control[offset], projected_signal[offset])
            if index in projections:
                prior = projections[index]
                assert value[0] == prior[0]
                np.testing.assert_array_equal(value[1], prior[1])
                np.testing.assert_array_equal(value[2], prior[2])
            else:
                projections[index] = value
    assert projections[8][0] == 0.8
    assert projections[10][0] == 1.0
    assert projections[22][0] == 2.2
    assert projections[10][1].shape == (2,)
    assert projections[10][2].shape == (2,)


def test_module_is_pure_bounded_and_uses_explicit_interpolation_sentinels(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("filesystem work is forbidden")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    assert _project().target_elapsed_seconds.size == 31

    tree = ast.parse(inspect.getsource(subject))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    forbidden_imports = {
        "csv", "pathlib", "pandas", "h5py", "threading", "multiprocessing",
        "gui", "photometry_pipeline.guided_continuous_rwd_review_binding",
        "photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation",
        "photometry_pipeline.io.rwd_continuous_source",
        "photometry_pipeline.io.hdf5_cache", "photometry_pipeline.pipeline",
    }
    assert imported.isdisjoint(forbidden_imports)
    source_text = inspect.getsource(subject)
    assert "left=np.nan" in source_text and "right=np.nan" in source_text
    for token in (
        "source_path", "sha256", "cancel", "progress", "serialize", "to_json",
        "correction", "feature", "cache", "DataFrame", "scipy",
    ):
        assert token not in source_text
    assert set(inspect.signature(subject.project_guided_continuous_rwd_block).parameters) == {
        "target_grid", "block_plan", "block", "recording_identity",
        "source_content_identity", "included_roi_ids", "source_row_start",
        "source_row_stop", "source_elapsed_seconds", "source_control_values",
        "source_signal_values",
    }
