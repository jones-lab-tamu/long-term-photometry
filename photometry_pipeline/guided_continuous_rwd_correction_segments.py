"""Deterministic CR1 correction segments and recording-global dynamic F0.

This module stops at CR1-C4a.  It assembles raw projected segments and prepares
the scalar control-derived denominator required by later dynamic correction; it
does not calculate fitted control, delta-F, or dF/F.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np

from photometry_pipeline.config import Config
from photometry_pipeline.core.baseline import DeterministicReservoir
from photometry_pipeline.core.preprocessing import get_lowpass_sos
from photometry_pipeline.core.types import PerRoiCorrectionSpec
from photometry_pipeline.guided_continuous_rwd_block_plan import (
    GuidedContinuousRwdBlockPlan,
)
from photometry_pipeline.guided_continuous_rwd_projection import (
    GuidedContinuousRwdProjectedBlock,
    PROJECTION_POLICY_NAME,
    PROJECTION_POLICY_VERSION,
    SCHEMA_NAME as PROJECTED_BLOCK_SCHEMA_NAME,
    SCHEMA_VERSION as PROJECTED_BLOCK_SCHEMA_VERSION,
    _target_coordinates,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    GuidedContinuousRwdReviewBinding,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    GuidedContinuousRwdTargetGridDescription,
)
from photometry_pipeline.guided_correction_payload import correction_payload_identity
from photometry_pipeline.guided_execution_payloads import (
    GUIDED_CONFIG_DEFAULT_OVERRIDES,
    GuidedExecutionStartupMappingContract,
    build_guided_execution_startup_mapping_contract,
)
from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    build_guided_per_roi_production_strategy_map,
)
from photometry_pipeline.guided_plan_identity import (
    compute_guided_new_analysis_draft_plan_identity,
)
from photometry_pipeline.guided_production_mapping import (
    GuidedProductionPerRoiStrategy,
    guided_production_strategy_map_to_correction_specs,
)
from photometry_pipeline.io.rwd_continuous_projection_reader import (
    ContinuousRwdProjectionReaderError,
    _validate_authorities,
)


SEGMENT_SCHEMA_NAME = "guided_continuous_rwd_correction_segment_plan"
SEGMENT_SCHEMA_VERSION = "v1"
SEGMENT_POLICY_NAME = "continuous-rwd-fixed-rounded-sample-correction-segments"
SEGMENT_POLICY_VERSION = "v1"
FINAL_SHORT_POLICY = "retain-if-viable-otherwise-merge-preceding"
SEGMENT_IDENTITY_DOMAIN = "guided-continuous-rwd-correction-segment-plan:v1"

RAW_SEGMENT_SCHEMA_NAME = "guided_continuous_rwd_raw_correction_segment"
RAW_SEGMENT_SCHEMA_VERSION = "v1"

F0_SCHEMA_NAME = "guided_continuous_rwd_dynamic_f0_authority"
F0_SCHEMA_VERSION = "v1"
F0_POLICY_NAME = "raw-control-deterministic-reservoir-percentile"
F0_POLICY_VERSION = "v1"
F0_IDENTITY_DOMAIN = "guided-continuous-rwd-dynamic-f0-authority:v1"
CORRECTION_CONTRACT_IDENTITY_DOMAIN = (
    "guided-continuous-rwd-correction-contract:v1"
)

DYNAMIC_STRATEGIES = (
    "global_linear_regression",
    "robust_global_event_reject",
    "adaptive_event_gated_regression",
)
SIGNAL_ONLY_STRATEGY = "signal_only_f0"
FINITE_VALUE_POLICY = "discard-nonfinite-before-reservoir"
STORAGE_DTYPE = "float32"

ERROR_CATEGORIES = frozenset(
    {
        "invalid_segment_plan_binding",
        "invalid_segment_descriptor",
        "invalid_segment_order",
        "target_coverage_mismatch",
        "projected_block_order_mismatch",
        "projected_roi_order_mismatch",
        "projected_shape_mismatch",
        "target_time_mismatch",
        "segment_assembly_incomplete",
        "unsupported_correction_strategy",
        "invalid_correction_settings",
        "no_finite_control_support",
        "invalid_final_f0",
        "f0_preparation_interrupted",
        "source_verification_failed",
        "accepted_correction_binding_mismatch",
    }
)


class GuidedContinuousRwdCorrectionSegmentError(ValueError):
    """A narrow CR1-C4a planning, assembly, or F0 preparation refusal."""

    def __init__(self, category: str, message: str, **context: Any) -> None:
        if category not in ERROR_CATEGORIES:
            raise ValueError(f"Unsupported CR1-C4a error category: {category!r}")
        super().__init__(message)
        self.category = category
        self.context = dict(context)


@dataclass(frozen=True)
class GuidedContinuousRwdCorrectionBinding:
    roi_id: str
    strategy_family: str
    selected_strategy: str
    dynamic_fit_mode: str | None
    parameter_identity: str
    evidence_identity: str


@dataclass(frozen=True)
class GuidedContinuousRwdCorrectionSegmentDescription:
    segment_index: int
    start_target_index: int
    stop_target_index: int
    is_final: bool
    absorbed_short_tail: bool

    @property
    def sample_count(self) -> int:
        return self.stop_target_index - self.start_target_index


@dataclass(frozen=True)
class GuidedContinuousRwdCorrectionSegmentPlan:
    schema_name: str
    schema_version: str
    policy_name: str
    policy_version: str
    target_grid_identity: str
    accepted_guided_plan_identity: str
    correction_payload_identity: str
    fixed_correction_settings_identity: str
    correction_contract_identity: str
    segment_duration_seconds_numerator: int
    segment_duration_seconds_denominator: int
    cadence_seconds_numerator: int
    cadence_seconds_denominator: int
    nominal_segment_sample_count: int
    target_sample_count: int
    shared_minimum_viable_sample_count: int
    final_short_policy: str
    segment_count: int
    descriptors: tuple[GuidedContinuousRwdCorrectionSegmentDescription, ...]
    plan_identity: str


@dataclass(frozen=True)
class GuidedContinuousRwdRawCorrectionSegment:
    schema_name: str
    schema_version: str
    recording_identity: str
    source_content_identity: str
    target_grid_identity: str
    correction_segment_plan_identity: str
    segment_index: int
    start_target_index: int
    stop_target_index: int
    included_roi_ids: tuple[str, ...]
    target_elapsed_seconds: np.ndarray
    control_values: np.ndarray
    signal_values: np.ndarray


@dataclass(frozen=True)
class GuidedContinuousRwdDynamicF0Value:
    roi_id: str
    strategy: str
    finite_value_count: int
    retained_value_count: int
    scalar_f0: float


@dataclass(frozen=True)
class GuidedContinuousRwdDynamicF0Authority:
    schema_name: str
    schema_version: str
    policy_name: str
    policy_version: str
    recording_identity: str
    source_content_identity: str
    target_grid_identity: str
    correction_segment_plan_identity: str
    accepted_guided_plan_identity: str
    correction_payload_identity: str
    fixed_correction_settings_identity: str
    canonical_roi_order: tuple[str, ...]
    dynamic_roi_ids: tuple[str, ...]
    correction_bindings: tuple[GuidedContinuousRwdCorrectionBinding, ...]
    percentile: float
    seed: int
    capacity: int
    storage_dtype: str
    finite_value_policy: str
    values: tuple[GuidedContinuousRwdDynamicF0Value, ...]
    f0_min_value: float
    finalized: bool
    completion_state: str
    authority_identity: str


@dataclass(frozen=True)
class _AcceptedCorrectionContext:
    accepted_guided_plan_identity: str
    continuous_window_sec: float
    roi_order: tuple[str, ...]
    bindings: tuple[GuidedContinuousRwdCorrectionBinding, ...]
    correction_specs: Mapping[str, PerRoiCorrectionSpec]
    correction_payload_identity: str
    fixed_correction_settings_identity: str
    correction_contract_identity: str
    config: Config


def _raise(category: str, message: str, **context: Any) -> None:
    raise GuidedContinuousRwdCorrectionSegmentError(category, message, **context)


def _binding_mismatch(
    field: str,
    *,
    expected: Any,
    actual: Any,
    reason: str,
) -> None:
    _raise(
        "accepted_correction_binding_mismatch",
        f"{field} does not match the accepted Guided correction authority.",
        field=field,
        expected=expected,
        actual=actual,
        reason=reason,
    )


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _identity(value: object, name: str, category: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _raise(category, f"{name} must be a lowercase 64-character identity.")


def _digest(domain: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + encode_canonical_value(dict(payload))
    ).hexdigest()


def _positive_decimal_fraction(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _raise("invalid_correction_settings", f"{name} must be numeric.")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        _raise(
            "invalid_correction_settings",
            f"{name} must be finite and greater than zero.",
        )
    try:
        result = Fraction(Decimal(str(value)))
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise GuidedContinuousRwdCorrectionSegmentError(
            "invalid_correction_settings",
            f"{name} cannot be represented as an exact decimal fraction.",
        ) from exc
    return result


def _rounded_segment_sample_count(duration: Fraction, cadence: Fraction) -> int:
    if not isinstance(duration, Fraction) or duration <= 0:
        _raise("invalid_correction_settings", "duration must be a positive Fraction.")
    if not isinstance(cadence, Fraction) or cadence <= 0:
        _raise("invalid_correction_settings", "cadence must be a positive Fraction.")
    # Python/Fraction round is the repository's existing ties-to-even convention.
    return round(duration / cadence)


def _canonical_correction_bindings(
    roi_order: tuple[str, ...],
    per_roi_correction: Mapping[str, PerRoiCorrectionSpec],
) -> tuple[GuidedContinuousRwdCorrectionBinding, ...]:
    if (
        not isinstance(per_roi_correction, Mapping)
        or set(per_roi_correction) != set(roi_order)
        or len(per_roi_correction) != len(roi_order)
    ):
        _raise(
            "invalid_segment_plan_binding",
            "Per-ROI correction coverage must match the canonical included ROI order.",
        )
    bindings: list[GuidedContinuousRwdCorrectionBinding] = []
    for roi_id in roi_order:
        spec = per_roi_correction.get(roi_id)
        if not isinstance(spec, PerRoiCorrectionSpec) or spec.roi_id != roi_id:
            _raise(
                "invalid_segment_plan_binding",
                "Per-ROI correction entry does not match its canonical ROI key.",
                roi=roi_id,
            )
        if spec.strategy_family == "dynamic_fit":
            if spec.selected_strategy not in DYNAMIC_STRATEGIES:
                _raise(
                    "unsupported_correction_strategy",
                    "Dynamic correction strategy is outside the accepted Guided subset.",
                    roi=roi_id,
                    strategy=spec.selected_strategy,
                )
        elif (
            spec.strategy_family != SIGNAL_ONLY_STRATEGY
            or spec.selected_strategy != SIGNAL_ONLY_STRATEGY
        ):
            _raise(
                "unsupported_correction_strategy",
                "Correction strategy is outside the accepted Guided subset.",
                roi=roi_id,
                strategy=spec.selected_strategy,
            )
        bindings.append(
            GuidedContinuousRwdCorrectionBinding(
                roi_id=roi_id,
                strategy_family=spec.strategy_family,
                selected_strategy=spec.selected_strategy,
                dynamic_fit_mode=spec.dynamic_fit_mode,
                parameter_identity=spec.parameter_identity,
                evidence_identity=spec.evidence_identity,
            )
        )
    return tuple(bindings)


def _correction_contract_identity(
    *,
    accepted_guided_plan_identity: str,
    correction_payload_identity_value: str,
    fixed_correction_settings_identity: str,
) -> str:
    return _digest(
        CORRECTION_CONTRACT_IDENTITY_DOMAIN,
        {
            "accepted_guided_plan_identity": accepted_guided_plan_identity,
            "correction_payload_identity": correction_payload_identity_value,
            "fixed_correction_settings_identity": fixed_correction_settings_identity,
        },
    )


_C4A_FIXED_SETTING_NAMES = (
    "seed",
    "lowpass_hz",
    "filter_order",
    "signal_only_f0_min_window_samples",
    "signal_only_f0_min_coverage_fraction",
    "baseline_method",
    "baseline_percentile",
    "f0_min_value",
)


def _resolve_fixed_correction_config(
    startup_mapping_contract: object,
) -> tuple[Config, str]:
    canonical = build_guided_execution_startup_mapping_contract()
    if not isinstance(startup_mapping_contract, GuidedExecutionStartupMappingContract):
        _binding_mismatch(
            "startup_mapping_contract",
            expected="GuidedExecutionStartupMappingContract",
            actual=type(startup_mapping_contract).__name__,
            reason="wrong_type",
        )
    if startup_mapping_contract != canonical:
        _binding_mismatch(
            "startup_mapping_contract",
            expected=canonical.contract_version,
            actual=getattr(startup_mapping_contract, "contract_version", None),
            reason="fixed_settings_contract_mismatch",
        )
    overrides = {
        item.name: item.value for item in startup_mapping_contract.fixed_config_overrides
    }
    settings: dict[str, Any] = {}
    for name in _C4A_FIXED_SETTING_NAMES:
        if name not in overrides or overrides[name] != GUIDED_CONFIG_DEFAULT_OVERRIDES[name]:
            _binding_mismatch(
                name,
                expected=GUIDED_CONFIG_DEFAULT_OVERRIDES[name],
                actual=overrides.get(name),
                reason="fixed_correction_setting_mismatch",
            )
        settings[name] = overrides[name]
    config = Config()
    for name, value in settings.items():
        setattr(config, name, value)
    identity = _digest(
        "guided-continuous-rwd-fixed-correction-settings:v1",
        {
            "schema_name": startup_mapping_contract.schema_name,
            "schema_version": startup_mapping_contract.schema_version,
            "contract_version": startup_mapping_contract.contract_version,
            "config_mapping_contract_version": (
                startup_mapping_contract.config_mapping_contract_version
            ),
            "settings": settings,
        },
    )
    return config, identity


def _resolve_accepted_correction_context(
    review_binding: GuidedContinuousRwdReviewBinding,
    accepted_draft: object,
    startup_mapping_contract: object,
) -> _AcceptedCorrectionContext:
    if not isinstance(accepted_draft, GuidedNewAnalysisDraftPlan):
        _binding_mismatch(
            "accepted_draft",
            expected="GuidedNewAnalysisDraftPlan",
            actual=type(accepted_draft).__name__,
            reason="wrong_type",
        )
    try:
        actual_plan_identity = compute_guided_new_analysis_draft_plan_identity(
            accepted_draft
        )
    except (TypeError, ValueError, AttributeError) as exc:
        raise GuidedContinuousRwdCorrectionSegmentError(
            "accepted_correction_binding_mismatch",
            "Accepted Guided draft identity cannot be recomputed.",
            field="accepted_guided_plan_identity",
            expected=review_binding.draft_plan_identity,
            actual=None,
            reason="identity_recomputation_failed",
        ) from exc
    if actual_plan_identity != review_binding.draft_plan_identity:
        _binding_mismatch(
            "accepted_guided_plan_identity",
            expected=review_binding.draft_plan_identity,
            actual=actual_plan_identity,
            reason="draft_plan_identity_mismatch",
        )
    if accepted_draft.input_format != "rwd" or accepted_draft.acquisition_mode != "continuous":
        _binding_mismatch(
            "acquisition",
            expected="continuous_rwd",
            actual=f"{accepted_draft.acquisition_mode}_{accepted_draft.input_format}",
            reason="unsupported_accepted_draft",
        )
    roi_order = review_binding.recording.roi.included_roi_ids
    if tuple(sorted(str(item) for item in accepted_draft.included_roi_ids)) != roi_order:
        _binding_mismatch(
            "included_roi_ids",
            expected=roi_order,
            actual=tuple(accepted_draft.included_roi_ids),
            reason="roi_scope_mismatch",
        )
    strategy_map = build_guided_per_roi_production_strategy_map(accepted_draft)
    if (
        not strategy_map.execution_routing_supported
        or strategy_map.blocking_categories
        or tuple(item.roi_id for item in strategy_map.entries) != tuple(
            str(item) for item in accepted_draft.included_roi_ids
        )
    ):
        _binding_mismatch(
            "per_roi_correction_strategy_map",
            expected="complete_current_explicit_supported_map",
            actual=strategy_map.blocking_categories,
            reason="accepted_strategy_map_unusable",
        )
    production_entries = tuple(
        GuidedProductionPerRoiStrategy(
            roi_id=entry.roi_id,
            strategy_family=entry.strategy_family,
            dynamic_fit_mode=entry.dynamic_fit_mode,
            selected_strategy=entry.selected_strategy,
            evidence_source_type=entry.evidence_source_type,
            evidence_reference_json=json.dumps(
                entry.evidence_reference,
                sort_keys=True,
                default=str,
            ),
            explicit_user_mark=entry.explicit_user_mark,
            current_or_stale=entry.current_or_stale,
        )
        for entry in strategy_map.entries
    )
    try:
        unordered_specs = guided_production_strategy_map_to_correction_specs(
            production_entries
        )
    except (TypeError, ValueError) as exc:
        raise GuidedContinuousRwdCorrectionSegmentError(
            "accepted_correction_binding_mismatch",
            "Accepted Guided strategy map cannot establish correction specifications.",
            field="per_roi_correction_strategy_map",
            expected="production-convertible map",
            actual=None,
            reason="production_adapter_refused",
        ) from exc
    specs = {roi_id: unordered_specs[roi_id] for roi_id in roi_order}
    bindings = _canonical_correction_bindings(roi_order, specs)
    payload_identity = correction_payload_identity(roi_order, specs)
    config, fixed_identity = _resolve_fixed_correction_config(
        startup_mapping_contract
    )
    contract_identity = _correction_contract_identity(
        accepted_guided_plan_identity=actual_plan_identity,
        correction_payload_identity_value=payload_identity,
        fixed_correction_settings_identity=fixed_identity,
    )
    return _AcceptedCorrectionContext(
        accepted_guided_plan_identity=actual_plan_identity,
        continuous_window_sec=accepted_draft.continuous_window_sec,
        roi_order=roi_order,
        bindings=bindings,
        correction_specs=specs,
        correction_payload_identity=payload_identity,
        fixed_correction_settings_identity=fixed_identity,
        correction_contract_identity=contract_identity,
        config=config,
    )


def _correction_payload_identity_from_bindings(
    roi_order: tuple[str, ...],
    bindings: tuple[GuidedContinuousRwdCorrectionBinding, ...],
) -> str:
    specs = {
        binding.roi_id: PerRoiCorrectionSpec(
            roi_id=binding.roi_id,
            strategy_family=binding.strategy_family,
            selected_strategy=binding.selected_strategy,
            dynamic_fit_mode=binding.dynamic_fit_mode,
            parameter_identity=binding.parameter_identity,
            evidence_identity=binding.evidence_identity,
        )
        for binding in bindings
    }
    return correction_payload_identity(roi_order, specs)


def _minimum_filter_sample_count(cadence: Fraction, config: Config) -> int:
    if not _integer(config.filter_order) or config.filter_order <= 0:
        _raise("invalid_correction_settings", "filter_order must be positive.")
    lowpass = float(config.lowpass_hz)
    if not math.isfinite(lowpass) or lowpass <= 0.0:
        _raise("invalid_correction_settings", "lowpass_hz must be positive and finite.")
    sampling_rate = float(1 / cadence)
    try:
        sos = get_lowpass_sos(sampling_rate, lowpass, config.filter_order)
    except (TypeError, ValueError) as exc:
        raise GuidedContinuousRwdCorrectionSegmentError(
            "invalid_correction_settings", "Low-pass filter settings are invalid."
        ) from exc
    if sos is None:
        return 1
    # This is SciPy sosfiltfilt's default pad length, including its correction
    # for odd-order filters represented by poles/zeros at the SOS origin.
    ntaps = 2 * len(sos) + 1
    ntaps -= min(int(np.sum(sos[:, 2] == 0.0)), int(np.sum(sos[:, 5] == 0.0)))
    return 3 * ntaps + 1


def _shared_minimum_viable_sample_count(
    cadence: Fraction,
    bindings: tuple[GuidedContinuousRwdCorrectionBinding, ...],
    config: Config,
) -> int:
    coverage = float(config.signal_only_f0_min_coverage_fraction)
    if not math.isfinite(coverage) or not 0.0 < coverage <= 1.0:
        _raise(
            "invalid_correction_settings",
            "signal_only_f0_min_coverage_fraction must lie in (0, 1].",
        )
    if (
        not _integer(config.signal_only_f0_min_window_samples)
        or config.signal_only_f0_min_window_samples <= 0
    ):
        _raise(
            "invalid_correction_settings",
            "signal_only_f0_min_window_samples must be positive.",
        )
    filter_minimum = _minimum_filter_sample_count(cadence, config)
    minimums = []
    for binding in bindings:
        if binding.strategy_family == "dynamic_fit":
            # Robust and adaptive accept the existing OLS fallback.  All finite
            # C3 input therefore needs two pairs plus a filter-safe trace.
            minimums.append(max(2, filter_minimum))
        else:
            # The candidate and production wrapper require ten finite samples.
            # Window sizes are clipped to the segment and add no larger minimum.
            minimums.append(10)
    return max(minimums)


def _canonical_segment_ranges(
    target_sample_count: int,
    nominal_segment_sample_count: int,
    shared_minimum_viable_sample_count: int,
) -> tuple[tuple[tuple[int, int], ...], bool]:
    for value, name in (
        (target_sample_count, "target_sample_count"),
        (nominal_segment_sample_count, "nominal_segment_sample_count"),
        (shared_minimum_viable_sample_count, "shared_minimum_viable_sample_count"),
    ):
        if not _integer(value) or value <= 0:
            _raise("invalid_correction_settings", f"{name} must be a positive integer.")
    if target_sample_count < shared_minimum_viable_sample_count:
        _raise(
            "invalid_correction_settings",
            "The complete target grid is below the shared strategy viability minimum.",
            target_sample_count=target_sample_count,
            shared_minimum=shared_minimum_viable_sample_count,
        )
    if nominal_segment_sample_count < shared_minimum_viable_sample_count:
        _raise(
            "invalid_correction_settings",
            "The nominal correction segment is below the shared strategy viability minimum.",
            nominal_segment_sample_count=nominal_segment_sample_count,
            shared_minimum=shared_minimum_viable_sample_count,
        )
    ranges = [
        (start, min(start + nominal_segment_sample_count, target_sample_count))
        for start in range(0, target_sample_count, nominal_segment_sample_count)
    ]
    absorbed = False
    if (
        len(ranges) > 1
        and ranges[-1][1] - ranges[-1][0] < shared_minimum_viable_sample_count
    ):
        ranges[-2] = (ranges[-2][0], ranges[-1][1])
        ranges.pop()
        absorbed = True
    return tuple(ranges), absorbed


def _segment_plan_identity_payload(
    plan: GuidedContinuousRwdCorrectionSegmentPlan,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in asdict(plan).items()
        if key != "plan_identity"
    }


def compute_guided_continuous_rwd_correction_segment_plan_identity(
    plan: GuidedContinuousRwdCorrectionSegmentPlan,
) -> str:
    if not isinstance(plan, GuidedContinuousRwdCorrectionSegmentPlan):
        _raise("invalid_segment_plan_binding", "Segment plan has the wrong type.")
    return _digest(SEGMENT_IDENTITY_DOMAIN, _segment_plan_identity_payload(plan))


def _validate_segment_plan(
    plan: object,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    review_binding: GuidedContinuousRwdReviewBinding,
    accepted_context: _AcceptedCorrectionContext | None = None,
) -> None:
    if not isinstance(plan, GuidedContinuousRwdCorrectionSegmentPlan):
        _raise("invalid_segment_plan_binding", "Segment plan has the wrong type.")
    if (
        plan.schema_name != SEGMENT_SCHEMA_NAME
        or plan.schema_version != SEGMENT_SCHEMA_VERSION
        or plan.policy_name != SEGMENT_POLICY_NAME
        or plan.policy_version != SEGMENT_POLICY_VERSION
        or plan.final_short_policy != FINAL_SHORT_POLICY
    ):
        _raise("invalid_segment_plan_binding", "Segment plan policy is unsupported.")
    for value, name in (
        (plan.target_grid_identity, "Target-grid identity"),
        (plan.accepted_guided_plan_identity, "Accepted Guided plan identity"),
        (plan.correction_payload_identity, "Correction-payload identity"),
        (plan.fixed_correction_settings_identity, "Fixed-settings identity"),
        (plan.correction_contract_identity, "Correction-contract identity"),
        (plan.plan_identity, "Segment-plan identity"),
    ):
        _identity(value, name, "invalid_segment_plan_binding")
    if (
        plan.target_grid_identity != target_grid.target_grid_identity
        or plan.accepted_guided_plan_identity != review_binding.draft_plan_identity
        or plan.target_sample_count != target_grid.target_sample_count
        or plan.cadence_seconds_numerator != target_grid.cadence_seconds_numerator
        or plan.cadence_seconds_denominator != target_grid.cadence_seconds_denominator
    ):
        _raise("invalid_segment_plan_binding", "Segment plan does not match C1.")
    if accepted_context is not None and (
        plan.accepted_guided_plan_identity
        != accepted_context.accepted_guided_plan_identity
        or plan.correction_payload_identity
        != accepted_context.correction_payload_identity
        or plan.fixed_correction_settings_identity
        != accepted_context.fixed_correction_settings_identity
        or plan.correction_contract_identity
        != accepted_context.correction_contract_identity
    ):
        _binding_mismatch(
            "segment_plan_correction_provenance",
            expected=accepted_context.correction_contract_identity,
            actual=plan.correction_contract_identity,
            reason="segment_plan_upstream_binding_mismatch",
        )
    integers = (
        plan.segment_duration_seconds_numerator,
        plan.segment_duration_seconds_denominator,
        plan.cadence_seconds_numerator,
        plan.cadence_seconds_denominator,
        plan.nominal_segment_sample_count,
        plan.target_sample_count,
        plan.shared_minimum_viable_sample_count,
        plan.segment_count,
    )
    if any(not _integer(value) or value <= 0 for value in integers):
        _raise("invalid_segment_plan_binding", "Segment plan counts must be positive integers.")
    duration = Fraction(
        plan.segment_duration_seconds_numerator,
        plan.segment_duration_seconds_denominator,
    )
    cadence = Fraction(plan.cadence_seconds_numerator, plan.cadence_seconds_denominator)
    if (
        duration.numerator != plan.segment_duration_seconds_numerator
        or duration.denominator != plan.segment_duration_seconds_denominator
        or cadence.numerator != plan.cadence_seconds_numerator
        or cadence.denominator != plan.cadence_seconds_denominator
        or _rounded_segment_sample_count(duration, cadence)
        != plan.nominal_segment_sample_count
    ):
        _raise("invalid_segment_plan_binding", "Segment plan exact arithmetic is inconsistent.")
    if not isinstance(plan.descriptors, tuple) or len(plan.descriptors) != plan.segment_count:
        _raise("invalid_segment_descriptor", "Segment descriptor count is inconsistent.")
    expected_start = 0
    for index, descriptor in enumerate(plan.descriptors):
        if not isinstance(descriptor, GuidedContinuousRwdCorrectionSegmentDescription):
            _raise("invalid_segment_descriptor", "Segment descriptor has the wrong type.")
        if (
            not _integer(descriptor.segment_index)
            or not _integer(descriptor.start_target_index)
            or not _integer(descriptor.stop_target_index)
            or type(descriptor.is_final) is not bool
            or type(descriptor.absorbed_short_tail) is not bool
            or descriptor.segment_index != index
            or descriptor.start_target_index != expected_start
            or descriptor.stop_target_index <= descriptor.start_target_index
            or descriptor.stop_target_index > plan.target_sample_count
            or descriptor.is_final != (index == plan.segment_count - 1)
            or (descriptor.absorbed_short_tail and not descriptor.is_final)
        ):
            _raise("invalid_segment_descriptor", "Segment descriptor is not canonical.")
        if not descriptor.is_final and descriptor.sample_count != plan.nominal_segment_sample_count:
            _raise("invalid_segment_descriptor", "Non-final segment has a non-nominal length.")
        if descriptor.sample_count < plan.shared_minimum_viable_sample_count:
            _raise("invalid_segment_descriptor", "Segment is below the shared viability minimum.")
        if descriptor.absorbed_short_tail and descriptor.sample_count <= plan.nominal_segment_sample_count:
            _raise("invalid_segment_descriptor", "Tail-absorption marker is inconsistent.")
        expected_start = descriptor.stop_target_index
    if expected_start != plan.target_sample_count:
        _raise("target_coverage_mismatch", "Segment plan does not cover the complete target grid.")
    expected_ranges, expected_absorbed = _canonical_segment_ranges(
        plan.target_sample_count,
        plan.nominal_segment_sample_count,
        plan.shared_minimum_viable_sample_count,
    )
    stored_ranges = tuple(
        (item.start_target_index, item.stop_target_index)
        for item in plan.descriptors
    )
    if stored_ranges != expected_ranges or plan.descriptors[-1].absorbed_short_tail != expected_absorbed:
        _raise("invalid_segment_descriptor", "Segment descriptors are not the canonical partition.")
    if plan.plan_identity != compute_guided_continuous_rwd_correction_segment_plan_identity(plan):
        _raise("invalid_segment_plan_binding", "Segment-plan identity mismatch.")


def build_guided_continuous_rwd_correction_segment_plan(
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    *,
    accepted_draft: GuidedNewAnalysisDraftPlan,
    startup_mapping_contract: GuidedExecutionStartupMappingContract,
) -> GuidedContinuousRwdCorrectionSegmentPlan:
    """Build the deterministic shared correction-segment authority."""
    if not isinstance(review_binding, GuidedContinuousRwdReviewBinding):
        _raise("invalid_segment_plan_binding", "B3 Review binding has the wrong type.")
    accepted = _resolve_accepted_correction_context(
        review_binding,
        accepted_draft,
        startup_mapping_contract,
    )
    # Reuse the committed C1/C2/C3 validator with a canonical temporary C2 plan.
    from photometry_pipeline.guided_continuous_rwd_block_plan import (
        build_guided_continuous_rwd_block_plan,
    )

    block_plan = build_guided_continuous_rwd_block_plan(target_grid)
    try:
        _validate_authorities(review_binding, target_grid, block_plan)
    except Exception as exc:
        raise GuidedContinuousRwdCorrectionSegmentError(
            "invalid_segment_plan_binding", "Accepted B3/C1 authorities are incompatible."
        ) from exc
    bindings = accepted.bindings
    config = accepted.config
    duration = _positive_decimal_fraction(
        accepted.continuous_window_sec, "continuous_window_sec"
    )
    cadence = target_grid.cadence_fraction
    nominal = _rounded_segment_sample_count(duration, cadence)
    if nominal <= 0:
        _raise("invalid_correction_settings", "Rounded segment sample count must be positive.")
    minimum = _shared_minimum_viable_sample_count(cadence, bindings, config)
    count = target_grid.target_sample_count
    ranges, absorbed = _canonical_segment_ranges(count, nominal, minimum)
    descriptors = tuple(
        GuidedContinuousRwdCorrectionSegmentDescription(
            segment_index=index,
            start_target_index=start,
            stop_target_index=stop,
            is_final=index == len(ranges) - 1,
            absorbed_short_tail=absorbed and index == len(ranges) - 1,
        )
        for index, (start, stop) in enumerate(ranges)
    )
    draft = GuidedContinuousRwdCorrectionSegmentPlan(
        schema_name=SEGMENT_SCHEMA_NAME,
        schema_version=SEGMENT_SCHEMA_VERSION,
        policy_name=SEGMENT_POLICY_NAME,
        policy_version=SEGMENT_POLICY_VERSION,
        target_grid_identity=target_grid.target_grid_identity,
        accepted_guided_plan_identity=accepted.accepted_guided_plan_identity,
        correction_payload_identity=accepted.correction_payload_identity,
        fixed_correction_settings_identity=accepted.fixed_correction_settings_identity,
        correction_contract_identity=accepted.correction_contract_identity,
        segment_duration_seconds_numerator=duration.numerator,
        segment_duration_seconds_denominator=duration.denominator,
        cadence_seconds_numerator=cadence.numerator,
        cadence_seconds_denominator=cadence.denominator,
        nominal_segment_sample_count=nominal,
        target_sample_count=count,
        shared_minimum_viable_sample_count=minimum,
        final_short_policy=FINAL_SHORT_POLICY,
        segment_count=len(descriptors),
        descriptors=descriptors,
        plan_identity="",
    )
    plan = GuidedContinuousRwdCorrectionSegmentPlan(
        **{
            **draft.__dict__,
            "plan_identity": compute_guided_continuous_rwd_correction_segment_plan_identity(draft),
        }
    )
    _validate_segment_plan(plan, target_grid, review_binding, accepted)
    return plan


def _check_cancellation(callback: Callable[[], bool] | None) -> None:
    if callback is None:
        return
    try:
        interrupted = callback()
    except Exception as exc:
        raise GuidedContinuousRwdCorrectionSegmentError(
            "f0_preparation_interrupted", "Cancellation callback failed."
        ) from exc
    if interrupted:
        _raise("f0_preparation_interrupted", "Continuous F0 preparation was cancelled.")


def _validate_projected_block_for_assembly(
    block: object,
    *,
    expected_index: int,
    expected_start: int,
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    block_plan: GuidedContinuousRwdBlockPlan,
    roi_order: tuple[str, ...],
) -> GuidedContinuousRwdProjectedBlock:
    if not isinstance(block, GuidedContinuousRwdProjectedBlock):
        _raise("projected_shape_mismatch", "Projected item has the wrong type.")
    if (
        block.schema_name != PROJECTED_BLOCK_SCHEMA_NAME
        or block.schema_version != PROJECTED_BLOCK_SCHEMA_VERSION
        or block.projection_policy_name != PROJECTION_POLICY_NAME
        or block.projection_policy_version != PROJECTION_POLICY_VERSION
    ):
        _raise("invalid_segment_plan_binding", "Projected block policy is unsupported.")
    if expected_index >= block_plan.block_count:
        _raise("projected_block_order_mismatch", "Projected traversal contains extra blocks.")
    expected = block_plan.block_at(expected_index)
    if (
        block.block_index != expected_index
        or block.start_target_index != expected.start_target_index
        or block.stop_target_index != expected.stop_target_index
        or block.start_target_index != expected_start
    ):
        _raise("projected_block_order_mismatch", "Projected block order or range is not canonical.")
    recording = review_binding.recording
    if (
        block.recording_identity != recording.recording_identity
        or block.source_content_identity != recording.source.source_content_identity
        or block.target_grid_identity != target_grid.target_grid_identity
    ):
        _raise("invalid_segment_plan_binding", "Projected block provenance is incompatible.")
    if block.included_roi_ids != roi_order:
        _raise("projected_roi_order_mismatch", "Projected ROI order does not match B1.")
    count = expected.stop_target_index - expected.start_target_index
    expected_shapes = (
        (block.target_elapsed_seconds, (count,)),
        (block.control_values, (count, len(roi_order))),
        (block.signal_values, (count, len(roi_order))),
    )
    for array, shape in expected_shapes:
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.float64
            or array.shape != shape
            or array.flags.writeable
            or not np.all(np.isfinite(array))
        ):
            _raise("projected_shape_mismatch", "Projected block arrays violate C3 shape rules.")
    expected_time = _target_coordinates(
        target_grid, expected.start_target_index, expected.stop_target_index
    )
    if not np.array_equal(block.target_elapsed_seconds, expected_time):
        _raise("target_time_mismatch", "Projected target times do not match C1 indices.")
    return block


def iter_assemble_guided_continuous_rwd_correction_segments(
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    block_plan: GuidedContinuousRwdBlockPlan,
    segment_plan: GuidedContinuousRwdCorrectionSegmentPlan,
    projected_blocks: Iterable[GuidedContinuousRwdProjectedBlock],
    *,
    accepted_draft: GuidedNewAnalysisDraftPlan,
    startup_mapping_contract: GuidedExecutionStartupMappingContract,
    cancellation_requested: Callable[[], bool] | None = None,
) -> Iterator[GuidedContinuousRwdRawCorrectionSegment]:
    """Yield provisional bounded raw segments; normal exhaustion proves completion."""
    if cancellation_requested is not None and not callable(cancellation_requested):
        _raise("f0_preparation_interrupted", "cancellation_requested must be callable or None.")
    try:
        authorities = _validate_authorities(review_binding, target_grid, block_plan)
    except Exception as exc:
        raise GuidedContinuousRwdCorrectionSegmentError(
            "invalid_segment_plan_binding", "Accepted B3/C1/C2 authorities are incompatible."
        ) from exc
    accepted = _resolve_accepted_correction_context(
        review_binding,
        accepted_draft,
        startup_mapping_contract,
    )
    _validate_segment_plan(
        segment_plan, target_grid, review_binding, accepted
    )
    roi_order = authorities.included_roi_ids
    iterator = iter(projected_blocks)
    expected_block_index = 0
    expected_target_index = 0
    descriptor_index = 0
    time_parts: list[np.ndarray] = []
    control_parts: list[np.ndarray] = []
    signal_parts: list[np.ndarray] = []
    filled = 0
    _check_cancellation(cancellation_requested)
    try:
        for raw_block in iterator:
            block = _validate_projected_block_for_assembly(
                raw_block,
                expected_index=expected_block_index,
                expected_start=expected_target_index,
                review_binding=review_binding,
                target_grid=target_grid,
                block_plan=block_plan,
                roi_order=roi_order,
            )
            offset = 0
            while offset < block.target_elapsed_seconds.size:
                if descriptor_index >= segment_plan.segment_count:
                    _raise("target_coverage_mismatch", "Projected data exceeds the segment plan.")
                descriptor = segment_plan.descriptors[descriptor_index]
                needed = descriptor.sample_count - filled
                take = min(needed, block.target_elapsed_seconds.size - offset)
                stop = offset + take
                time_parts.append(block.target_elapsed_seconds[offset:stop])
                control_parts.append(block.control_values[offset:stop, :])
                signal_parts.append(block.signal_values[offset:stop, :])
                filled += take
                offset = stop
                expected_target_index += take
                if filled == descriptor.sample_count:
                    arrays = [
                        np.concatenate(time_parts).astype(np.float64, copy=False),
                        np.concatenate(control_parts, axis=0).astype(np.float64, copy=False),
                        np.concatenate(signal_parts, axis=0).astype(np.float64, copy=False),
                    ]
                    for array in arrays:
                        array.setflags(write=False)
                    segment = GuidedContinuousRwdRawCorrectionSegment(
                        schema_name=RAW_SEGMENT_SCHEMA_NAME,
                        schema_version=RAW_SEGMENT_SCHEMA_VERSION,
                        recording_identity=review_binding.recording.recording_identity,
                        source_content_identity=(
                            review_binding.recording.source.source_content_identity
                        ),
                        target_grid_identity=target_grid.target_grid_identity,
                        correction_segment_plan_identity=segment_plan.plan_identity,
                        segment_index=descriptor.segment_index,
                        start_target_index=descriptor.start_target_index,
                        stop_target_index=descriptor.stop_target_index,
                        included_roi_ids=roi_order,
                        target_elapsed_seconds=arrays[0],
                        control_values=arrays[1],
                        signal_values=arrays[2],
                    )
                    _check_cancellation(cancellation_requested)
                    yield segment
                    _check_cancellation(cancellation_requested)
                    descriptor_index += 1
                    filled = 0
                    time_parts.clear()
                    control_parts.clear()
                    signal_parts.clear()
            expected_block_index += 1
        # Reaching here, rather than merely receiving the final owned range,
        # proves that C3b completed its late source verification.
    except ContinuousRwdProjectionReaderError:
        raise
    if expected_block_index != block_plan.block_count:
        _raise("segment_assembly_incomplete", "Projected traversal ended before all C2 blocks.")
    if filled or time_parts or control_parts or signal_parts:
        _raise("segment_assembly_incomplete", "Projected traversal ended within a segment.")
    if (
        descriptor_index != segment_plan.segment_count
        or expected_target_index != target_grid.target_sample_count
    ):
        _raise("target_coverage_mismatch", "Assembled segments do not cover the target grid.")


def _f0_authority_payload(
    authority: GuidedContinuousRwdDynamicF0Authority,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in asdict(authority).items()
        if key != "authority_identity"
    }


def compute_guided_continuous_rwd_dynamic_f0_authority_identity(
    authority: GuidedContinuousRwdDynamicF0Authority,
) -> str:
    if not isinstance(authority, GuidedContinuousRwdDynamicF0Authority):
        _raise("invalid_final_f0", "F0 authority has the wrong type.")
    return _digest(F0_IDENTITY_DOMAIN, _f0_authority_payload(authority))


def _validate_dynamic_f0_authority(
    authority: object,
    *,
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    segment_plan: GuidedContinuousRwdCorrectionSegmentPlan,
    accepted_context: _AcceptedCorrectionContext,
) -> None:
    if not isinstance(authority, GuidedContinuousRwdDynamicF0Authority):
        _raise("invalid_final_f0", "F0 authority has the wrong type.")
    _validate_segment_plan(
        segment_plan,
        target_grid,
        review_binding,
        accepted_context,
    )
    recording = review_binding.recording
    if (
        authority.schema_name != F0_SCHEMA_NAME
        or authority.schema_version != F0_SCHEMA_VERSION
        or authority.policy_name != F0_POLICY_NAME
        or authority.policy_version != F0_POLICY_VERSION
        or authority.recording_identity != recording.recording_identity
        or authority.source_content_identity != recording.source.source_content_identity
        or authority.target_grid_identity != target_grid.target_grid_identity
        or authority.correction_segment_plan_identity != segment_plan.plan_identity
        or authority.canonical_roi_order != recording.roi.included_roi_ids
        or not authority.finalized
    ):
        _raise("invalid_final_f0", "F0 authority provenance or policy is invalid.")
    for field, actual, expected in (
        (
            "accepted_guided_plan_identity",
            authority.accepted_guided_plan_identity,
            accepted_context.accepted_guided_plan_identity,
        ),
        (
            "correction_payload_identity",
            authority.correction_payload_identity,
            accepted_context.correction_payload_identity,
        ),
        (
            "fixed_correction_settings_identity",
            authority.fixed_correction_settings_identity,
            accepted_context.fixed_correction_settings_identity,
        ),
    ):
        if actual != expected:
            _binding_mismatch(
                field,
                expected=expected,
                actual=actual,
                reason="f0_authority_upstream_identity_mismatch",
            )
    for value, name in (
        (authority.recording_identity, "Recording identity"),
        (authority.source_content_identity, "Source-content identity"),
        (authority.target_grid_identity, "Target-grid identity"),
        (authority.correction_segment_plan_identity, "Segment-plan identity"),
        (authority.accepted_guided_plan_identity, "Accepted Guided plan identity"),
        (authority.correction_payload_identity, "Correction-payload identity"),
        (authority.fixed_correction_settings_identity, "Fixed-settings identity"),
    ):
        _identity(value, name, "invalid_final_f0")
    _identity(authority.authority_identity, "F0 authority identity", "invalid_final_f0")
    if (
        not isinstance(authority.canonical_roi_order, tuple)
        or not authority.canonical_roi_order
        or len(set(authority.canonical_roi_order)) != len(authority.canonical_roi_order)
        or not isinstance(authority.dynamic_roi_ids, tuple)
        or len(set(authority.dynamic_roi_ids)) != len(authority.dynamic_roi_ids)
        or not isinstance(authority.correction_bindings, tuple)
        or tuple(item.roi_id for item in authority.correction_bindings)
        != authority.canonical_roi_order
    ):
        _raise("invalid_final_f0", "F0 authority ROI ordering is invalid.")
    for binding in authority.correction_bindings:
        if not isinstance(binding, GuidedContinuousRwdCorrectionBinding):
            _raise("invalid_final_f0", "F0 correction binding has the wrong type.")
        if (
            binding.strategy_family == "dynamic_fit"
            and (
                binding.selected_strategy not in DYNAMIC_STRATEGIES
                or binding.dynamic_fit_mode != binding.selected_strategy
            )
        ) or (
            binding.strategy_family == SIGNAL_ONLY_STRATEGY
            and (
                binding.selected_strategy != SIGNAL_ONLY_STRATEGY
                or binding.dynamic_fit_mode is not None
            )
        ) or binding.strategy_family not in {"dynamic_fit", SIGNAL_ONLY_STRATEGY}:
            _raise("invalid_final_f0", "F0 correction binding strategy is invalid.")
        if not isinstance(binding.parameter_identity, str) or not isinstance(binding.evidence_identity, str):
            _raise("invalid_final_f0", "F0 correction binding identities must be strings.")
    dynamic_from_bindings = tuple(
        item.roi_id
        for item in authority.correction_bindings
        if item.strategy_family == "dynamic_fit"
    )
    if dynamic_from_bindings != authority.dynamic_roi_ids:
        _raise("invalid_final_f0", "F0 authority dynamic ROI subset is inconsistent.")
    try:
        expected_payload_identity = _correction_payload_identity_from_bindings(
            authority.canonical_roi_order,
            authority.correction_bindings,
        )
    except (TypeError, ValueError) as exc:
        raise GuidedContinuousRwdCorrectionSegmentError(
            "invalid_final_f0", "F0 correction bindings cannot establish a payload identity."
        ) from exc
    if authority.correction_payload_identity != expected_payload_identity:
        _raise("invalid_final_f0", "F0 correction bindings do not match the segment plan.")
    if authority.correction_bindings != accepted_context.bindings:
        _binding_mismatch(
            "f0_authority_correction_bindings",
            expected=accepted_context.bindings,
            actual=authority.correction_bindings,
            reason="f0_authority_upstream_binding_mismatch",
        )
    if tuple(item.roi_id for item in authority.values) != authority.dynamic_roi_ids:
        _raise("invalid_final_f0", "F0 value order does not match the dynamic ROI subset.")
    binding_by_roi = {item.roi_id: item for item in authority.correction_bindings}
    if authority.storage_dtype != STORAGE_DTYPE or authority.finite_value_policy != FINITE_VALUE_POLICY:
        _raise("invalid_final_f0", "F0 estimator storage policy is invalid.")
    if (
        not _integer(authority.capacity)
        or authority.capacity <= 0
        or not _integer(authority.seed)
        or not math.isfinite(authority.f0_min_value)
        or authority.f0_min_value < 0.0
        or type(authority.finalized) is not bool
    ):
        _raise("invalid_final_f0", "F0 reservoir capacity is invalid.")
    if not math.isfinite(authority.percentile) or not 0.0 <= authority.percentile <= 100.0:
        _raise("invalid_final_f0", "F0 percentile is invalid.")
    for value in authority.values:
        if (
            not _integer(value.finite_value_count)
            or value.finite_value_count <= 0
            or value.retained_value_count != min(value.finite_value_count, authority.capacity)
            or not math.isfinite(value.scalar_f0)
            or value.scalar_f0 <= authority.f0_min_value
            or value.strategy != binding_by_roi[value.roi_id].selected_strategy
        ):
            _raise("invalid_final_f0", "F0 scalar evidence is invalid.", roi=value.roi_id)
    if authority.dynamic_roi_ids and authority.completion_state != "complete_source_verified":
        _raise("invalid_final_f0", "Dynamic F0 authority lacks verified source completion.")
    if not authority.dynamic_roi_ids and authority.completion_state != "not_required_all_signal_only":
        _raise("invalid_final_f0", "Empty F0 authority has an invalid completion state.")
    if authority.authority_identity != compute_guided_continuous_rwd_dynamic_f0_authority_identity(authority):
        _raise("invalid_final_f0", "F0 authority identity mismatch.")


def prepare_guided_continuous_rwd_dynamic_f0_authority(
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    block_plan: GuidedContinuousRwdBlockPlan,
    segment_plan: GuidedContinuousRwdCorrectionSegmentPlan,
    projected_blocks: Iterable[GuidedContinuousRwdProjectedBlock],
    *,
    accepted_draft: GuidedNewAnalysisDraftPlan,
    startup_mapping_contract: GuidedExecutionStartupMappingContract,
    cancellation_requested: Callable[[], bool] | None = None,
) -> GuidedContinuousRwdDynamicF0Authority:
    """Consume one complete C3b traversal and finalize dynamic raw-control F0."""
    _check_cancellation(cancellation_requested)
    try:
        _validate_authorities(review_binding, target_grid, block_plan)
    except Exception as exc:
        raise GuidedContinuousRwdCorrectionSegmentError(
            "invalid_segment_plan_binding", "Accepted B3/C1/C2 authorities are incompatible."
        ) from exc
    accepted = _resolve_accepted_correction_context(
        review_binding,
        accepted_draft,
        startup_mapping_contract,
    )
    roi_order = accepted.roi_order
    bindings = accepted.bindings
    config = accepted.config
    _validate_segment_plan(segment_plan, target_grid, review_binding, accepted)
    dynamic_roi_ids = tuple(
        binding.roi_id for binding in bindings if binding.strategy_family == "dynamic_fit"
    )
    values: tuple[GuidedContinuousRwdDynamicF0Value, ...]
    completion_state: str
    percentile = float(config.baseline_percentile)
    threshold = float(config.f0_min_value)
    capacity = DeterministicReservoir(seed=config.seed).capacity
    if (
        config.baseline_method != "uv_raw_percentile_session"
        or
        not math.isfinite(percentile)
        or not 0.0 <= percentile <= 100.0
        or not math.isfinite(threshold)
        or threshold < 0.0
        or not _integer(config.seed)
    ):
        _raise("invalid_correction_settings", "F0 estimator settings are invalid.")
    if not dynamic_roi_ids:
        values = ()
        completion_state = "not_required_all_signal_only"
    else:
        reservoir = DeterministicReservoir(seed=config.seed, capacity=capacity)
        roi_indices = {roi_id: roi_order.index(roi_id) for roi_id in dynamic_roi_ids}
        try:
            segments = iter_assemble_guided_continuous_rwd_correction_segments(
                review_binding,
                target_grid,
                block_plan,
                segment_plan,
                projected_blocks,
                accepted_draft=accepted_draft,
                startup_mapping_contract=startup_mapping_contract,
                cancellation_requested=cancellation_requested,
            )
            for segment in segments:
                _check_cancellation(cancellation_requested)
                for roi_id in dynamic_roi_ids:
                    _check_cancellation(cancellation_requested)
                    reservoir.add(roi_id, segment.control_values[:, roi_indices[roi_id]])
                _check_cancellation(cancellation_requested)
        except GuidedContinuousRwdCorrectionSegmentError:
            raise
        except ContinuousRwdProjectionReaderError as exc:
            if exc.category == "projection_interrupted":
                raise GuidedContinuousRwdCorrectionSegmentError(
                    "f0_preparation_interrupted", "C3b projection was cancelled."
                ) from exc
            raise GuidedContinuousRwdCorrectionSegmentError(
                "source_verification_failed", "C3b source verification failed."
            ) from exc
        _check_cancellation(cancellation_requested)
        result_values = []
        binding_by_roi = {item.roi_id: item for item in bindings}
        for roi_id in dynamic_roi_ids:
            finite_count = reservoir.count.get(roi_id, 0)
            retained_count = min(finite_count, capacity)
            strategy = binding_by_roi[roi_id].selected_strategy
            if finite_count <= 0:
                _raise(
                    "no_finite_control_support",
                    "Dynamic ROI has no finite raw-control support.",
                    roi=roi_id,
                    strategy=strategy,
                    finite_count=finite_count,
                    retained_count=retained_count,
                    percentile=percentile,
                    threshold=threshold,
                    reason="no_finite_raw_control_values",
                )
            scalar = float(reservoir.get_percentile(roi_id, percentile))
            if not math.isfinite(scalar) or scalar <= threshold:
                _raise(
                    "invalid_final_f0",
                    "Dynamic ROI finalized F0 is nonfinite or below threshold.",
                    roi=roi_id,
                    strategy=strategy,
                    finite_count=finite_count,
                    retained_count=retained_count,
                    percentile=percentile,
                    threshold=threshold,
                    reason="invalid_scalar_f0",
                )
            result_values.append(
                GuidedContinuousRwdDynamicF0Value(
                    roi_id=roi_id,
                    strategy=strategy,
                    finite_value_count=finite_count,
                    retained_value_count=retained_count,
                    scalar_f0=scalar,
                )
            )
        values = tuple(result_values)
        completion_state = "complete_source_verified"
    _check_cancellation(cancellation_requested)
    draft = GuidedContinuousRwdDynamicF0Authority(
        schema_name=F0_SCHEMA_NAME,
        schema_version=F0_SCHEMA_VERSION,
        policy_name=F0_POLICY_NAME,
        policy_version=F0_POLICY_VERSION,
        recording_identity=review_binding.recording.recording_identity,
        source_content_identity=review_binding.recording.source.source_content_identity,
        target_grid_identity=target_grid.target_grid_identity,
        correction_segment_plan_identity=segment_plan.plan_identity,
        accepted_guided_plan_identity=accepted.accepted_guided_plan_identity,
        correction_payload_identity=segment_plan.correction_payload_identity,
        fixed_correction_settings_identity=accepted.fixed_correction_settings_identity,
        canonical_roi_order=roi_order,
        dynamic_roi_ids=dynamic_roi_ids,
        correction_bindings=bindings,
        percentile=percentile,
        seed=config.seed,
        capacity=capacity,
        storage_dtype=STORAGE_DTYPE,
        finite_value_policy=FINITE_VALUE_POLICY,
        values=values,
        f0_min_value=threshold,
        finalized=True,
        completion_state=completion_state,
        authority_identity="",
    )
    authority = GuidedContinuousRwdDynamicF0Authority(
        **{
            **draft.__dict__,
            "authority_identity": compute_guided_continuous_rwd_dynamic_f0_authority_identity(draft),
        }
    )
    _check_cancellation(cancellation_requested)
    _validate_dynamic_f0_authority(
        authority,
        review_binding=review_binding,
        target_grid=target_grid,
        segment_plan=segment_plan,
        accepted_context=accepted,
    )
    return authority
