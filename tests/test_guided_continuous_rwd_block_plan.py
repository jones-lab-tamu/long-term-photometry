from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
import inspect
import os
from pathlib import Path

import pytest

from photometry_pipeline import guided_continuous_rwd_block_plan as subject
from photometry_pipeline import guided_continuous_rwd_target_grid as target_grid_subject


B = subject.MAXIMUM_OWNED_SAMPLES_PER_BLOCK


def _target_grid(
    target_sample_count: int = 250_123,
    *,
    cadence: Fraction = Fraction(1, 10),
) -> target_grid_subject.GuidedContinuousRwdTargetGridDescription:
    support = (target_sample_count - 1) * cadence
    draft = target_grid_subject.GuidedContinuousRwdTargetGridDescription(
        schema_name=target_grid_subject.SCHEMA_NAME,
        schema_version=target_grid_subject.SCHEMA_VERSION,
        grid_policy_name=target_grid_subject.GRID_POLICY_NAME,
        grid_policy_version=target_grid_subject.GRID_POLICY_VERSION,
        recording_identity="a" * 64,
        continuity_evaluation_identity="b" * 64,
        cadence_seconds_numerator=cadence.numerator,
        cadence_seconds_denominator=cadence.denominator,
        source_support_end_seconds_numerator=support.numerator,
        source_support_end_seconds_denominator=support.denominator,
        target_sample_count=target_sample_count,
        target_grid_identity="",
    )
    grid = replace(
        draft,
        target_grid_identity=(
            target_grid_subject.compute_guided_continuous_rwd_target_grid_identity(
                draft
            )
        ),
    )
    target_grid_subject._validate_target_grid_description(grid)
    return grid


def _plan(target_sample_count: int = 250_123):
    return subject.build_guided_continuous_rwd_block_plan(
        _target_grid(target_sample_count)
    )


def test_valid_plan_contains_only_irreducible_frozen_authority():
    grid = _target_grid()
    plan = subject.build_guided_continuous_rwd_block_plan(grid)

    assert plan.schema_name == "guided_continuous_rwd_block_plan"
    assert plan.schema_version == "v1"
    assert plan.block_policy_name == "continuous-rwd-fixed-owned-sample-blocks"
    assert plan.block_policy_version == "v1"
    assert plan.target_grid_identity == grid.target_grid_identity
    assert plan.target_sample_count == grid.target_sample_count
    assert plan.maximum_owned_samples_per_block == 100_000
    assert plan.block_count == 3
    assert {field.name for field in fields(plan)} == {
        "schema_name",
        "schema_version",
        "block_policy_name",
        "block_policy_version",
        "target_grid_identity",
        "target_sample_count",
    }
    assert not any(isinstance(value, (list, tuple)) for value in plan.__dict__.values())
    assert not hasattr(plan, "block_plan_identity")
    assert not hasattr(plan, "plan_identity")

    with pytest.raises(FrozenInstanceError):
        plan.target_sample_count = 2


def test_descriptor_contains_only_frozen_ownership_fields():
    block = _plan().block_at(0)
    assert {field.name for field in fields(block)} == {
        "block_index",
        "start_target_index",
        "stop_target_index",
    }
    assert block.owned_sample_count == B
    with pytest.raises(FrozenInstanceError):
        block.stop_target_index = 1


@pytest.mark.parametrize(
    ("count", "expected_ranges"),
    [
        (2, [(0, 2)]),
        (B - 1, [(0, B - 1)]),
        (B, [(0, B)]),
        (B + 1, [(0, B), (B, B + 1)]),
        (2 * B, [(0, B), (B, 2 * B)]),
        (2 * B + 17, [(0, B), (B, 2 * B), (2 * B, 2 * B + 17)]),
    ],
)
def test_exact_partition_shapes(count, expected_ranges):
    plan = _plan(count)
    blocks = list(plan.iter_blocks())
    assert plan.block_count == len(expected_ranges)
    assert [
        (block.start_target_index, block.stop_target_index) for block in blocks
    ] == expected_ranges
    assert blocks[0].start_target_index == 0
    assert blocks[-1].stop_target_index == count
    assert all(0 < block.owned_sample_count <= B for block in blocks)
    assert all(
        earlier.stop_target_index == later.start_target_index
        for earlier, later in zip(blocks, blocks[1:])
    )


def test_block_count_uses_subtraction_safe_integer_formula():
    for count in (2, B - 1, B, B + 1, 3 * B, 3 * B + 1):
        plan = _plan(count)
        assert plan.block_count == 1 + (count - 1) // B
    assert subject._resolve_block_count(7, 1) == 7


@pytest.mark.parametrize("bad_size", [True, False, 0, -1, 1.0, "1", None])
def test_private_count_arithmetic_refuses_invalid_block_sizes(bad_size):
    with pytest.raises(subject.ContinuousRwdBlockPlanError):
        subject._resolve_block_count(10, bad_size)


def test_random_access_is_canonical_constant_shape_and_repeatable():
    plan = _plan(7 * B + 23)
    expected = {
        0: (0, B),
        3: (3 * B, 4 * B),
        7: (7 * B, 7 * B + 23),
    }
    for index, (start, stop) in expected.items():
        first = plan.block_at(index)
        second = plan.block_at(index)
        assert first == second
        assert first is not second
        assert first.block_index == index
        assert first.start_target_index == start
        assert first.stop_target_index == stop


@pytest.mark.parametrize("bad_index", [True, False, 1.0, "1", None, -1, 3, 4])
def test_random_access_refuses_invalid_indices(bad_index):
    plan = _plan(2 * B + 1)
    with pytest.raises(subject.ContinuousRwdBlockPlanError):
        plan.block_at(bad_index)


def test_iteration_is_fresh_ordered_repeatable_and_independent():
    plan = _plan(3 * B + 9)
    first_iterator = plan.iter_blocks()
    assert iter(first_iterator) is first_iterator
    first_value = next(first_iterator)

    complete_after_partial = list(plan.iter_blocks())
    repeated = list(plan.iter_blocks())
    assert first_value == plan.block_at(0)
    assert complete_after_partial == repeated
    assert [block.block_index for block in repeated] == list(range(plan.block_count))
    assert len(repeated) == plan.block_count
    assert repeated == [plan.block_at(index) for index in range(plan.block_count)]
    assert not hasattr(plan, "current_block_index")
    assert not hasattr(plan, "iterator_position")


def test_iteration_allocates_descriptors_lazily(monkeypatch):
    plan = _plan(4 * B + 1)
    original = subject.GuidedContinuousRwdBlockPlan.block_at
    calls: list[int] = []

    def tracked(self, block_index):
        calls.append(block_index)
        return original(self, block_index)

    monkeypatch.setattr(subject.GuidedContinuousRwdBlockPlan, "block_at", tracked)
    iterator = plan.iter_blocks()
    assert calls == []
    assert next(iterator).block_index == 0
    assert calls == [0]
    assert next(iterator).block_index == 1
    assert calls == [0, 1]


def test_moderate_plan_has_exactly_one_owner_for_every_target_index():
    count = 3 * B + 29
    plan = _plan(count)
    ownership = bytearray(count)
    blocks = list(plan.iter_blocks())

    for block in blocks:
        for index in range(block.start_target_index, block.stop_target_index):
            ownership[index] += 1

    assert ownership == bytearray([1]) * count
    for earlier, later in zip(blocks, blocks[1:]):
        boundary = later.start_target_index
        assert boundary == earlier.stop_target_index
        assert boundary not in range(
            earlier.start_target_index, earlier.stop_target_index
        )
        assert boundary in range(later.start_target_index, later.stop_target_index)
    assert count - 1 in range(
        blocks[-1].start_target_index, blocks[-1].stop_target_index
    )
    assert blocks[-1].stop_target_index not in range(
        blocks[-1].start_target_index, blocks[-1].stop_target_index
    )


def test_global_target_times_remain_exactly_c1_derived():
    grid = _target_grid(4 * B + 31, cadence=Fraction(1, 10))
    plan = subject.build_guided_continuous_rwd_block_plan(grid)

    for index in (0, B - 1, B, B + 7, 3 * B, grid.last_target_index):
        block = plan.block_at(min(index // B, plan.block_count - 1))
        local_offset = index - block.start_target_index
        via_block_ownership = (
            block.start_target_index + local_offset
        ) * grid.cadence_fraction
        assert via_block_ownership == index * grid.cadence_fraction

    plan_names = {field.name for field in fields(plan)}
    block_names = {field.name for field in fields(plan.block_at(0))}
    forbidden = {"cadence", "elapsed", "duration", "local_time", "block_start_time"}
    assert not any(any(token in name for token in forbidden) for name in plan_names)
    assert not any(any(token in name for token in forbidden) for name in block_names)


def test_builder_refuses_wrong_type_and_translates_invalid_c1_authority():
    with pytest.raises(subject.ContinuousRwdBlockPlanError, match="target_grid"):
        subject.build_guided_continuous_rwd_block_plan(object())

    malformed = replace(_target_grid(), schema_version="bad")
    with pytest.raises(
        subject.ContinuousRwdBlockPlanError,
        match="target-grid authority is invalid",
    ) as caught:
        subject.build_guided_continuous_rwd_block_plan(malformed)
    assert isinstance(caught.value.__cause__, target_grid_subject.ContinuousRwdTargetGridError)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_name": "bad"},
        {"schema_version": "bad"},
        {"block_policy_name": "bad"},
        {"block_policy_version": "bad"},
        {"target_grid_identity": 7},
        {"target_grid_identity": "a" * 63},
        {"target_grid_identity": "A" * 64},
        {"target_grid_identity": "g" * 64},
        {"target_sample_count": True},
        {"target_sample_count": 2.0},
        {"target_sample_count": "2"},
        {"target_sample_count": 1},
        {"target_sample_count": target_grid_subject.MAX_TARGET_SAMPLE_COUNT + 1},
    ],
)
def test_plan_validator_refuses_malformed_authority(changes):
    malformed = replace(_plan(), **changes)
    with pytest.raises(subject.ContinuousRwdBlockPlanError):
        subject._validate_block_plan(malformed)


@pytest.mark.parametrize("bad_constant", [True, False, 0, -1, 1.5, "100000"])
def test_plan_validator_refuses_invalid_fixed_policy_constant(monkeypatch, bad_constant):
    plan = _plan()
    monkeypatch.setattr(subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", bad_constant)
    with pytest.raises(subject.ContinuousRwdBlockPlanError):
        subject._validate_block_plan(plan)


def test_plan_validator_refuses_wrong_plan_type():
    with pytest.raises(subject.ContinuousRwdBlockPlanError):
        subject._validate_block_plan(object())


@pytest.mark.parametrize(
    "description",
    [
        object(),
        subject.GuidedContinuousRwdBlockDescription(True, 0, B),
        subject.GuidedContinuousRwdBlockDescription(0, False, B),
        subject.GuidedContinuousRwdBlockDescription(0, 0, True),
        subject.GuidedContinuousRwdBlockDescription(0.0, 0, B),
        subject.GuidedContinuousRwdBlockDescription(0, 0.0, B),
        subject.GuidedContinuousRwdBlockDescription(0, 0, float(B)),
        subject.GuidedContinuousRwdBlockDescription(0, -1, B),
        subject.GuidedContinuousRwdBlockDescription(0, 0, 0),
        subject.GuidedContinuousRwdBlockDescription(0, 0, B + 1),
        subject.GuidedContinuousRwdBlockDescription(0, 1, B),
        subject.GuidedContinuousRwdBlockDescription(0, 0, B - 1),
        subject.GuidedContinuousRwdBlockDescription(1, 0, B),
        subject.GuidedContinuousRwdBlockDescription(2, 2 * B, 3 * B + 1),
    ],
)
def test_descriptor_validator_refuses_noncanonical_descriptions(description):
    plan = _plan(2 * B + 1)
    with pytest.raises(subject.ContinuousRwdBlockPlanError):
        subject._validate_block_description(plan, description)


def test_descriptor_validator_accepts_every_canonical_moderate_descriptor():
    plan = _plan(3 * B + 1)
    for description in plan.iter_blocks():
        subject._validate_block_description(plan, description)


def test_signed_64_bit_maximum_plan_is_scalar_and_sampled_algebraically():
    maximum = target_grid_subject.MAX_TARGET_SAMPLE_COUNT
    plan = subject.build_guided_continuous_rwd_block_plan(
        _target_grid(maximum, cadence=Fraction(1, 1))
    )

    assert plan.target_sample_count == maximum
    assert plan.block_count == 92_233_720_368_548
    assert {field.name for field in fields(plan)} == {
        "schema_name",
        "schema_version",
        "block_policy_name",
        "block_policy_version",
        "target_grid_identity",
        "target_sample_count",
    }

    sampled = sorted(
        {
            0,
            1,
            plan.block_count // 2,
            plan.block_count - 2,
            plan.block_count - 1,
        }
    )
    blocks = [plan.block_at(index) for index in sampled]
    for index, block in zip(sampled, blocks):
        assert block.block_index == index
        assert block.start_target_index == index * B
        assert block.stop_target_index == min((index + 1) * B, maximum)
        assert 0 < block.owned_sample_count <= B
    assert blocks[-1].stop_target_index == maximum

    adjacent_index = plan.block_count // 2
    earlier = plan.block_at(adjacent_index)
    later = plan.block_at(adjacent_index + 1)
    assert earlier.stop_target_index == later.start_target_index
    assert not any(isinstance(value, (list, tuple)) for value in plan.__dict__.values())


def test_ownership_contract_contains_no_context_or_source_fields():
    plan_fields = {field.name for field in fields(subject.GuidedContinuousRwdBlockPlan)}
    block_fields = {
        field.name for field in fields(subject.GuidedContinuousRwdBlockDescription)
    }
    forbidden = {
        "context",
        "overlap",
        "halo",
        "padding",
        "source_row",
        "source_path",
        "cadence",
        "elapsed",
    }
    assert not any(any(token in name for token in forbidden) for name in plan_fields)
    assert not any(any(token in name for token in forbidden) for name in block_fields)
    assert "contextual reads never change the owned output range" in inspect.getdoc(subject)


def test_builder_and_access_perform_no_filesystem_work(monkeypatch):
    grid = _target_grid()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("filesystem access is forbidden")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "stat", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    plan = subject.build_guided_continuous_rwd_block_plan(grid)
    assert plan.block_at(0).start_target_index == 0
    assert next(plan.iter_blocks()).block_index == 0


def test_module_dependencies_are_pure_and_bounded():
    source = inspect.getsource(subject)
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots <= {"__future__", "dataclasses", "typing", "photometry_pipeline"}
    assert imported_roots.isdisjoint(
        {
            "os",
            "pathlib",
            "numpy",
            "pandas",
            "h5py",
            "gui",
            "threading",
            "multiprocessing",
            "json",
        }
    )
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_leaves = {module.rsplit(".", 1)[-1] for module in imported_modules}
    assert not any(
        token in leaf
        for leaf in imported_leaves
        for token in ("adapter", "parser", "pipeline", "interpol", "resampl", "output")
    )
    assert not any(
        name in source
        for name in (
            "def serialize",
            "def deserialize",
            "def save",
            "def load",
            "to_json",
            "from_json",
        )
    )
