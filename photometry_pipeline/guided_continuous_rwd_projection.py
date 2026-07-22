"""Pure one-block projection onto the accepted continuous RWD target grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from photometry_pipeline.guided_continuous_rwd_block_plan import (
    ContinuousRwdBlockPlanError,
    GuidedContinuousRwdBlockDescription,
    GuidedContinuousRwdBlockPlan,
    _validate_block_description,
    _validate_block_plan,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    ContinuousRwdTargetGridError,
    GuidedContinuousRwdTargetGridDescription,
    _validate_target_grid_description,
)


SCHEMA_NAME = "guided_continuous_rwd_projected_block"
SCHEMA_VERSION = "v1"
PROJECTION_POLICY_NAME = "continuous-rwd-linear-projection"
PROJECTION_POLICY_VERSION = "v1"


class ContinuousRwdProjectionError(ValueError):
    """One bounded source interval cannot establish the requested projection."""


@dataclass(frozen=True)
class GuidedContinuousRwdProjectedBlock:
    schema_name: str
    schema_version: str
    projection_policy_name: str
    projection_policy_version: str
    recording_identity: str
    source_content_identity: str
    target_grid_identity: str
    block_index: int
    start_target_index: int
    stop_target_index: int
    included_roi_ids: tuple[str, ...]
    source_row_start: int
    source_row_stop: int
    target_elapsed_seconds: np.ndarray
    control_values: np.ndarray
    signal_values: np.ndarray


def _fail(message: str) -> None:
    raise ContinuousRwdProjectionError(message)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_identity_text(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be a lowercase 64-character hexadecimal identity.")


def _validate_roi_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        _fail("included_roi_ids must be a tuple.")
    if not value:
        _fail("included_roi_ids must contain at least one ROI ID.")
    if any(not isinstance(roi_id, str) or not roi_id for roi_id in value):
        _fail("Every included ROI ID must be a nonempty string.")
    if len(set(value)) != len(value):
        _fail("included_roi_ids must not contain duplicates.")
    return value


def _validate_source_rows(source_row_start: object, source_row_stop: object) -> int:
    if not _is_integer(source_row_start) or not _is_integer(source_row_stop):
        _fail("Source-row bounds must be integers.")
    if source_row_start < 0 or source_row_start >= source_row_stop:
        _fail("Source-row bounds must define a nonempty nonnegative half-open range.")
    return source_row_stop - source_row_start


def _as_float64_numeric_array(
    value: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        _fail(f"{name} must be a NumPy array.")
    if value.ndim != ndim:
        _fail(f"{name} must be {ndim}-dimensional.")
    if value.dtype.kind not in "iuf":
        _fail(f"{name} must contain numeric real values.")
    converted = np.array(value, dtype=np.float64, copy=True)
    if converted.shape != value.shape:
        _fail(f"{name} shape changed during float64 conversion.")
    if not np.all(np.isfinite(converted)):
        _fail(f"{name} must contain only finite values.")
    return converted


def _validate_source_arrays(
    source_elapsed_seconds: object,
    source_control_values: object,
    source_signal_values: object,
    *,
    source_row_count: int,
    roi_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elapsed = _as_float64_numeric_array(
        source_elapsed_seconds,
        name="source_elapsed_seconds",
        ndim=1,
    )
    control = _as_float64_numeric_array(
        source_control_values,
        name="source_control_values",
        ndim=2,
    )
    signal = _as_float64_numeric_array(
        source_signal_values,
        name="source_signal_values",
        ndim=2,
    )
    if elapsed.shape[0] == 0:
        _fail("Source support must contain at least one row.")
    if elapsed.shape[0] != source_row_count:
        _fail("Source-row provenance does not match the timestamp row count.")
    expected_matrix_shape = (source_row_count, roi_count)
    if control.shape != expected_matrix_shape:
        _fail("Source control matrix shape does not match source rows and ROI order.")
    if signal.shape != expected_matrix_shape:
        _fail("Source signal matrix shape does not match source rows and ROI order.")
    if elapsed.size > 1 and not np.all(np.diff(elapsed) > 0.0):
        _fail("Source timestamps must be strictly increasing.")
    return elapsed, control, signal


def _target_coordinates(
    target_grid: GuidedContinuousRwdTargetGridDescription,
    start_target_index: int,
    stop_target_index: int,
) -> np.ndarray:
    owned_count = stop_target_index - start_target_index
    global_indices = np.arange(
        start_target_index,
        stop_target_index,
        dtype=np.int64,
    )
    try:
        cadence = np.float64(target_grid.cadence_fraction)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ContinuousRwdProjectionError(
            "Generated target coordinates must be finite float64 values."
        ) from exc
    target = global_indices.astype(np.float64) * cadence
    if target.shape != (owned_count,):
        _fail("Generated target-coordinate count does not match the owned range.")
    if not np.all(np.isfinite(target)):
        _fail("Generated target coordinates must be finite.")
    if target.size > 1 and not np.all(np.diff(target) > 0.0):
        _fail("Float64 target-coordinate resolution is insufficient at this range.")
    expected_first = np.float64(start_target_index) * cadence
    expected_last = np.float64(stop_target_index - 1) * cadence
    if target[0] != expected_first or target[-1] != expected_last:
        _fail("Generated target coordinates do not match the global-index formula.")
    return target


def _project_bounded_arrays(
    target_grid: GuidedContinuousRwdTargetGridDescription,
    start_target_index: int,
    stop_target_index: int,
    source_elapsed_seconds: np.ndarray,
    source_control_values: np.ndarray,
    source_signal_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one arbitrary global range; public ownership validation is external."""
    target = _target_coordinates(target_grid, start_target_index, stop_target_index)
    if source_elapsed_seconds[0] > target[0]:
        _fail("Source support does not include a sample at or before the first target.")
    if source_elapsed_seconds[-1] < target[-1]:
        _fail("Source support does not include a sample at or after the last target.")

    shape = (target.shape[0], source_control_values.shape[1])
    projected_control = np.empty(shape, dtype=np.float64)
    projected_signal = np.empty(shape, dtype=np.float64)
    for roi_index in range(shape[1]):
        projected_control[:, roi_index] = np.interp(
            target,
            source_elapsed_seconds,
            source_control_values[:, roi_index],
            left=np.nan,
            right=np.nan,
        )
        projected_signal[:, roi_index] = np.interp(
            target,
            source_elapsed_seconds,
            source_signal_values[:, roi_index],
            left=np.nan,
            right=np.nan,
        )
    if not np.all(np.isfinite(projected_control)) or not np.all(
        np.isfinite(projected_signal)
    ):
        _fail("Projection attempted extrapolation or produced nonfinite values.")
    return target, projected_control, projected_signal


def _validate_projected_block(
    result: object,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    block_plan: GuidedContinuousRwdBlockPlan,
    block: GuidedContinuousRwdBlockDescription,
    source_elapsed_seconds: np.ndarray,
    *,
    recording_identity: str,
    source_content_identity: str,
    included_roi_ids: tuple[str, ...],
    source_row_start: int,
    source_row_stop: int,
) -> None:
    if not isinstance(result, GuidedContinuousRwdProjectedBlock):
        _fail("result must be a GuidedContinuousRwdProjectedBlock.")
    if result.schema_name != SCHEMA_NAME or result.schema_version != SCHEMA_VERSION:
        _fail("Unsupported projected-block schema.")
    if (
        result.projection_policy_name != PROJECTION_POLICY_NAME
        or result.projection_policy_version != PROJECTION_POLICY_VERSION
    ):
        _fail("Unsupported projection policy.")
    _validate_identity_text(result.recording_identity, "Recording identity")
    _validate_identity_text(result.source_content_identity, "Source-content identity")
    _validate_identity_text(result.target_grid_identity, "Target-grid identity")
    if result.recording_identity != recording_identity:
        _fail("Projected recording identity does not match supplied provenance.")
    if result.source_content_identity != source_content_identity:
        _fail("Projected source-content identity does not match supplied provenance.")
    try:
        _validate_block_description(block_plan, block)
    except ContinuousRwdBlockPlanError as exc:
        raise ContinuousRwdProjectionError("Canonical block descriptor is invalid.") from exc
    if result.target_grid_identity != target_grid.target_grid_identity:
        _fail("Projected target-grid identity does not match C1.")
    if (
        result.block_index != block.block_index
        or result.start_target_index != block.start_target_index
        or result.stop_target_index != block.stop_target_index
    ):
        _fail("Projected block range does not match the canonical C2 descriptor.")
    roi_ids = _validate_roi_ids(result.included_roi_ids)
    if result.included_roi_ids != included_roi_ids:
        _fail("Projected ROI order does not match supplied provenance.")
    source_row_count = _validate_source_rows(
        result.source_row_start,
        result.source_row_stop,
    )
    if (
        result.source_row_start != source_row_start
        or result.source_row_stop != source_row_stop
    ):
        _fail("Projected source-row bounds do not match supplied provenance.")
    if source_row_count != source_elapsed_seconds.shape[0]:
        _fail("Projected source-row provenance does not match supplied support.")
    owned_count = block.owned_sample_count
    expected_shapes = (
        (result.target_elapsed_seconds, (owned_count,), "target_elapsed_seconds"),
        (result.control_values, (owned_count, len(roi_ids)), "control_values"),
        (result.signal_values, (owned_count, len(roi_ids)), "signal_values"),
    )
    for array, shape, name in expected_shapes:
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            _fail(f"{name} must be a NumPy float64 array.")
        if array.shape != shape:
            _fail(f"{name} has the wrong shape.")
        if array.flags.writeable:
            _fail(f"{name} must be read-only.")
        if not np.all(np.isfinite(array)):
            _fail(f"{name} must contain only finite values.")
    expected_target = _target_coordinates(
        target_grid,
        block.start_target_index,
        block.stop_target_index,
    )
    if result.target_elapsed_seconds.size > 1 and not np.all(
        np.diff(result.target_elapsed_seconds) > 0.0
    ):
        _fail("Stored target coordinates must be strictly increasing.")
    if not np.array_equal(result.target_elapsed_seconds, expected_target):
        _fail("Stored target coordinates do not match C1 global indices.")
    if (
        source_elapsed_seconds[0] > result.target_elapsed_seconds[0]
        or source_elapsed_seconds[-1] < result.target_elapsed_seconds[-1]
    ):
        _fail("Stored target coordinates are not bracketed by source support.")


def project_guided_continuous_rwd_block(
    target_grid: GuidedContinuousRwdTargetGridDescription,
    block_plan: GuidedContinuousRwdBlockPlan,
    block: GuidedContinuousRwdBlockDescription,
    *,
    recording_identity: str,
    source_content_identity: str,
    included_roi_ids: tuple[str, ...],
    source_row_start: int,
    source_row_stop: int,
    source_elapsed_seconds: np.ndarray,
    source_control_values: np.ndarray,
    source_signal_values: np.ndarray,
) -> GuidedContinuousRwdProjectedBlock:
    """Project already parsed bounded source values onto one canonical C2 block."""
    if not isinstance(target_grid, GuidedContinuousRwdTargetGridDescription):
        _fail("target_grid must be a GuidedContinuousRwdTargetGridDescription.")
    try:
        _validate_target_grid_description(target_grid)
    except (ContinuousRwdTargetGridError, TypeError, ValueError) as exc:
        raise ContinuousRwdProjectionError("C1 target-grid authority is invalid.") from exc
    if not isinstance(block_plan, GuidedContinuousRwdBlockPlan):
        _fail("block_plan must be a GuidedContinuousRwdBlockPlan.")
    try:
        _validate_block_plan(block_plan)
    except (ContinuousRwdBlockPlanError, TypeError, ValueError) as exc:
        raise ContinuousRwdProjectionError("C2 block-plan authority is invalid.") from exc
    if block_plan.target_grid_identity != target_grid.target_grid_identity:
        _fail("C2 target-grid identity does not match C1.")
    if block_plan.target_sample_count != target_grid.target_sample_count:
        _fail("C2 target sample count does not match C1.")
    try:
        _validate_block_description(block_plan, block)
    except (ContinuousRwdBlockPlanError, TypeError, ValueError) as exc:
        raise ContinuousRwdProjectionError("C2 block descriptor is invalid.") from exc

    _validate_identity_text(recording_identity, "Recording identity")
    _validate_identity_text(source_content_identity, "Source-content identity")
    roi_ids = _validate_roi_ids(included_roi_ids)
    source_row_count = _validate_source_rows(source_row_start, source_row_stop)
    elapsed, control, signal = _validate_source_arrays(
        source_elapsed_seconds,
        source_control_values,
        source_signal_values,
        source_row_count=source_row_count,
        roi_count=len(roi_ids),
    )
    target, projected_control, projected_signal = _project_bounded_arrays(
        target_grid,
        block.start_target_index,
        block.stop_target_index,
        elapsed,
        control,
        signal,
    )
    for array in (target, projected_control, projected_signal):
        array.setflags(write=False)
    result = GuidedContinuousRwdProjectedBlock(
        schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        projection_policy_name=PROJECTION_POLICY_NAME,
        projection_policy_version=PROJECTION_POLICY_VERSION,
        recording_identity=recording_identity,
        source_content_identity=source_content_identity,
        target_grid_identity=target_grid.target_grid_identity,
        block_index=block.block_index,
        start_target_index=block.start_target_index,
        stop_target_index=block.stop_target_index,
        included_roi_ids=roi_ids,
        source_row_start=source_row_start,
        source_row_stop=source_row_stop,
        target_elapsed_seconds=target,
        control_values=projected_control,
        signal_values=projected_signal,
    )
    _validate_projected_block(
        result,
        target_grid,
        block_plan,
        block,
        elapsed,
        recording_identity=recording_identity,
        source_content_identity=source_content_identity,
        included_roi_ids=roi_ids,
        source_row_start=source_row_start,
        source_row_stop=source_row_stop,
    )
    return result
