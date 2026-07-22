"""Immutable target-index ownership for Guided continuous RWD processing.

Each block owns results only for its half-open target-index range
``[start_target_index, stop_target_index)``. Later processing may read context
outside that range, but contextual reads never change the owned output range.

This module partitions the already accepted C1 grid. It does not define a
timebase, inspect a source, generate target coordinates, or model processing
context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from photometry_pipeline.guided_continuous_rwd_target_grid import (
    MAX_TARGET_SAMPLE_COUNT,
    ContinuousRwdTargetGridError,
    GuidedContinuousRwdTargetGridDescription,
    _validate_target_grid_description,
)


SCHEMA_NAME = "guided_continuous_rwd_block_plan"
SCHEMA_VERSION = "v1"
BLOCK_POLICY_NAME = "continuous-rwd-fixed-owned-sample-blocks"
BLOCK_POLICY_VERSION = "v1"
MAXIMUM_OWNED_SAMPLES_PER_BLOCK = 100_000


class ContinuousRwdBlockPlanError(ValueError):
    """The accepted C1 grid cannot establish one valid ownership plan."""


@dataclass(frozen=True)
class GuidedContinuousRwdBlockDescription:
    block_index: int
    start_target_index: int
    stop_target_index: int

    @property
    def owned_sample_count(self) -> int:
        return self.stop_target_index - self.start_target_index


@dataclass(frozen=True)
class GuidedContinuousRwdBlockPlan:
    schema_name: str
    schema_version: str
    block_policy_name: str
    block_policy_version: str
    target_grid_identity: str
    target_sample_count: int

    @property
    def maximum_owned_samples_per_block(self) -> int:
        return MAXIMUM_OWNED_SAMPLES_PER_BLOCK

    @property
    def block_count(self) -> int:
        return _resolve_block_count(
            self.target_sample_count,
            self.maximum_owned_samples_per_block,
        )

    def block_at(self, block_index: int) -> GuidedContinuousRwdBlockDescription:
        """Return one canonical descriptor in constant time."""
        _validate_block_plan(self)
        _validate_block_index(self, block_index)
        description = _canonical_block_description(self, block_index)
        _validate_block_description_fields(self, description)
        return description

    def iter_blocks(self) -> Iterator[GuidedContinuousRwdBlockDescription]:
        """Return a fresh, ordered, lazy iterator over canonical descriptors."""
        _validate_block_plan(self)
        return (self.block_at(index) for index in range(self.block_count))


def _fail(message: str) -> None:
    raise ContinuousRwdBlockPlanError(message)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_identity_text(value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(
            "Target-grid identity must be a lowercase 64-character "
            "hexadecimal identity."
        )


def _validate_fixed_policy_constant() -> None:
    if (
        not _is_integer(MAXIMUM_OWNED_SAMPLES_PER_BLOCK)
        or MAXIMUM_OWNED_SAMPLES_PER_BLOCK <= 0
    ):
        _fail("Maximum owned samples per block must be a positive integer.")


def _resolve_block_count(target_sample_count: int, block_size: int) -> int:
    if not _is_integer(target_sample_count) or target_sample_count <= 0:
        _fail("Target sample count must be a positive integer.")
    if not _is_integer(block_size) or block_size <= 0:
        _fail("Block size must be a positive integer.")
    return 1 + (target_sample_count - 1) // block_size


def _validate_block_index(
    plan: GuidedContinuousRwdBlockPlan,
    block_index: object,
) -> None:
    if not _is_integer(block_index):
        _fail("Block index must be an integer.")
    if block_index < 0 or block_index >= plan.block_count:
        _fail(
            f"Block index {block_index} is outside [0, {plan.block_count})."
        )


def _canonical_block_description(
    plan: GuidedContinuousRwdBlockPlan,
    block_index: int,
) -> GuidedContinuousRwdBlockDescription:
    block_size = plan.maximum_owned_samples_per_block
    start = block_index * block_size
    stop = min(start + block_size, plan.target_sample_count)
    return GuidedContinuousRwdBlockDescription(
        block_index=block_index,
        start_target_index=start,
        stop_target_index=stop,
    )


def _validate_block_description_fields(
    plan: GuidedContinuousRwdBlockPlan,
    description: object,
) -> None:
    if not isinstance(description, GuidedContinuousRwdBlockDescription):
        _fail("description must be a GuidedContinuousRwdBlockDescription.")
    for value, name in (
        (description.block_index, "Block index"),
        (description.start_target_index, "Start target index"),
        (description.stop_target_index, "Stop target index"),
    ):
        if not _is_integer(value):
            _fail(f"{name} must be an integer.")

    _validate_block_index(plan, description.block_index)
    expected = _canonical_block_description(plan, description.block_index)
    if description.start_target_index != expected.start_target_index:
        _fail("Block start target index is not canonical for its block index.")
    if description.stop_target_index != expected.stop_target_index:
        _fail("Block stop target index is not canonical for its block index.")
    if not (
        0
        <= description.start_target_index
        < description.stop_target_index
        <= plan.target_sample_count
    ):
        _fail("Block owned range must be nonempty and within the target grid.")
    if description.owned_sample_count > plan.maximum_owned_samples_per_block:
        _fail("Block owned range exceeds the fixed v1 maximum.")


def _validate_block_description(
    plan: GuidedContinuousRwdBlockPlan,
    description: object,
) -> None:
    """Validate a descriptor against one accepted scalar plan."""
    _validate_block_plan(plan)
    _validate_block_description_fields(plan, description)


def _validate_block_plan(plan: object) -> None:
    if not isinstance(plan, GuidedContinuousRwdBlockPlan):
        _fail("plan must be a GuidedContinuousRwdBlockPlan.")
    if plan.schema_name != SCHEMA_NAME or plan.schema_version != SCHEMA_VERSION:
        _fail("Unsupported continuous RWD block-plan schema.")
    if (
        plan.block_policy_name != BLOCK_POLICY_NAME
        or plan.block_policy_version != BLOCK_POLICY_VERSION
    ):
        _fail("Unsupported continuous RWD block policy.")
    _validate_identity_text(plan.target_grid_identity)
    if not _is_integer(plan.target_sample_count):
        _fail("Target sample count must be an integer.")
    if plan.target_sample_count < 2:
        _fail("Target sample count must be at least two.")
    if plan.target_sample_count > MAX_TARGET_SAMPLE_COUNT:
        _fail("Target sample count exceeds the signed 64-bit index limit.")
    _validate_fixed_policy_constant()

    block_count = plan.block_count
    if block_count < 1:
        _fail("Block count must be at least one.")
    first = _canonical_block_description(plan, 0)
    final = _canonical_block_description(plan, block_count - 1)
    _validate_block_description_fields(plan, first)
    _validate_block_description_fields(plan, final)
    if first.start_target_index != 0:
        _fail("First block must start at target index zero.")
    if final.stop_target_index != plan.target_sample_count:
        _fail("Final block must stop at the target sample count.")


def build_guided_continuous_rwd_block_plan(
    target_grid: GuidedContinuousRwdTargetGridDescription,
) -> GuidedContinuousRwdBlockPlan:
    """Build a frozen scalar ownership plan from one accepted C1 target grid."""
    if not isinstance(target_grid, GuidedContinuousRwdTargetGridDescription):
        _fail("target_grid must be a GuidedContinuousRwdTargetGridDescription.")
    try:
        _validate_target_grid_description(target_grid)
    except ContinuousRwdTargetGridError as exc:
        raise ContinuousRwdBlockPlanError(
            "Continuous RWD target-grid authority is invalid."
        ) from exc

    plan = GuidedContinuousRwdBlockPlan(
        schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        block_policy_name=BLOCK_POLICY_NAME,
        block_policy_version=BLOCK_POLICY_VERSION,
        target_grid_identity=target_grid.target_grid_identity,
        target_sample_count=target_grid.target_sample_count,
    )
    _validate_block_plan(plan)
    return plan
