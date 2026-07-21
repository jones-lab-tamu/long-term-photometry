"""Immutable accepted-recording authority for one continuous RWD source.

This module is deliberately pure: it consumes completed CR1-A evidence and
explicit ROI choices, performs no filesystem access, and grants no execution
authority while material-gap admission remains unresolved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.io.rwd_continuous_source import (
    INSPECTION_CONTRACT_NAME,
    INSPECTION_CONTRACT_VERSION,
    MINIMUM_DURATION_SEC,
    ContinuousRwdInspectionResult,
)


SCHEMA_NAME = "guided_continuous_rwd_recording_description"
SCHEMA_VERSION = "v1"
SOURCE_FORMAT = "rwd"
ACQUISITION_MODE = "continuous"
EXECUTION_ADMISSION_STATUS = "pending_material_gap_policy"
UNRESOLVED_ADMISSION_CHECKS = ("material_gap_policy",)
ROI_SELECTION_RULE_VERSION = "continuous-rwd-roi-selection.v1"
TIME_BASIS = "recording_relative_seconds_from_first_valid_source_timestamp"
SOURCE_CONTENT_IDENTITY_DOMAIN = "guided-continuous-rwd-source-content:v1"
PARSER_INTERPRETATION_IDENTITY_DOMAIN = (
    "guided-continuous-rwd-parser-interpretation:v1"
)
CADENCE_EVIDENCE_IDENTITY_DOMAIN = "guided-continuous-rwd-cadence-evidence:v1"
RECORDING_IDENTITY_DOMAIN = "guided-continuous-rwd-recording-description:v1"


class ContinuousRwdRecordingAuthorityError(ValueError):
    """CR1-A evidence cannot establish the requested recording authority."""


@dataclass(frozen=True)
class ContinuousRwdSourceBinding:
    inspection_contract_name: str
    inspection_contract_version: str
    sha256: str
    file_size_bytes: int
    total_data_row_count: int
    valid_timestamp_count: int
    header_row_index: int
    raw_columns: tuple[str, ...]
    selected_time_column: str
    source_content_identity: str
    parser_interpretation_identity: str
    selected_folder_canonical: str
    fluorescence_path_canonical: str
    modification_time_ns: int
    stable_source_identity: str


@dataclass(frozen=True)
class ContinuousRwdRoiChannel:
    roi_id: str
    reference_column: str
    signal_column: str


@dataclass(frozen=True)
class ContinuousRwdRoiAuthority:
    selection_rule_version: str
    available_roi_channels: tuple[ContinuousRwdRoiChannel, ...]
    included_roi_ids: tuple[str, ...]
    excluded_roi_ids: tuple[str, ...]


@dataclass(frozen=True)
class ContinuousRwdTimeAuthority:
    raw_timestamp_unit: str
    raw_timestamp_scale_to_seconds: float
    raw_first_timestamp: float
    raw_last_timestamp: float
    normalized_elapsed_origin_seconds: float
    normalized_elapsed_end_seconds: float
    measured_support_start_seconds: float
    measured_support_end_seconds: float
    measured_duration_seconds: float
    time_basis: str


@dataclass(frozen=True)
class ContinuousRwdCadenceQuantile:
    probability: float
    dt_seconds: float


@dataclass(frozen=True)
class ContinuousRwdUnusualInterval:
    row_index: int
    dt_seconds: float
    estimated_expected_sample_multiple: float


@dataclass(frozen=True)
class ContinuousRwdCadenceAuthority:
    cadence_evidence_policy_version: str
    positive_interval_count: int
    nominal_cadence_seconds: float
    minimum_positive_dt_seconds: float
    maximum_positive_dt_seconds: float
    mean_positive_dt_seconds: float
    standard_deviation_positive_dt_seconds: float
    coefficient_of_variation: float
    quantile_method: str
    quantile_sample_count: int
    quantiles: tuple[ContinuousRwdCadenceQuantile, ...]
    unusually_long_interval_count: int
    unusually_short_interval_count: int
    largest_unusual_intervals: tuple[ContinuousRwdUnusualInterval, ...]
    smallest_unusual_intervals: tuple[ContinuousRwdUnusualInterval, ...]
    cadence_evidence_identity: str


@dataclass(frozen=True)
class GuidedContinuousRwdRecordingDescription:
    schema_name: str
    schema_version: str
    source_format: str
    acquisition_mode: str
    execution_admission_status: str
    unresolved_admission_checks: tuple[str, ...]
    source: ContinuousRwdSourceBinding
    roi: ContinuousRwdRoiAuthority
    time: ContinuousRwdTimeAuthority
    cadence: ContinuousRwdCadenceAuthority
    recording_identity: str


def _fail(message: str) -> None:
    raise ContinuousRwdRecordingAuthorityError(message)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite.")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}.")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{name} must be a non-empty string.")
    return value


def _digest(domain: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + encode_canonical_value(dict(payload))
    ).hexdigest()


def _roi_payload(roi: ContinuousRwdRoiAuthority) -> dict[str, Any]:
    return {
        "selection_rule_version": roi.selection_rule_version,
        "available_roi_channels": [asdict(item) for item in roi.available_roi_channels],
        "included_roi_ids": list(roi.included_roi_ids),
        "excluded_roi_ids": list(roi.excluded_roi_ids),
    }


def _time_payload(time: ContinuousRwdTimeAuthority) -> dict[str, Any]:
    return asdict(time)


def _cadence_payload(
    cadence: ContinuousRwdCadenceAuthority, *, include_identity: bool = False
) -> dict[str, Any]:
    payload = asdict(cadence)
    identity = payload.pop("cadence_evidence_identity")
    payload["quantiles"] = [asdict(item) for item in cadence.quantiles]
    payload["largest_unusual_intervals"] = [
        asdict(item) for item in cadence.largest_unusual_intervals
    ]
    payload["smallest_unusual_intervals"] = [
        asdict(item) for item in cadence.smallest_unusual_intervals
    ]
    if include_identity:
        payload["cadence_evidence_identity"] = identity
    return payload


def _source_content_payload(source: ContinuousRwdSourceBinding) -> dict[str, Any]:
    return {"sha256": source.sha256, "file_size_bytes": source.file_size_bytes}


def _parser_payload(
    inspection: ContinuousRwdInspectionResult,
    channels: tuple[ContinuousRwdRoiChannel, ...],
) -> dict[str, Any]:
    parser = inspection.parser_facts
    assert parser is not None
    return {
        "inspection_contract_name": inspection.contract_name,
        "inspection_contract_version": inspection.contract_version,
        "header_row_index": parser.header_row_index,
        "raw_columns": list(parser.raw_columns),
        "selected_time_column": parser.time_column,
        "timestamp_unit": parser.timestamp_unit,
        "timestamp_scale_to_seconds": parser.timestamp_scale_to_seconds,
        "available_roi_channels": [asdict(item) for item in channels],
        "time_basis": TIME_BASIS,
    }


def _stored_parser_payload(
    description: GuidedContinuousRwdRecordingDescription,
) -> dict[str, Any]:
    return {
        "inspection_contract_name": description.source.inspection_contract_name,
        "inspection_contract_version": description.source.inspection_contract_version,
        "header_row_index": description.source.header_row_index,
        "raw_columns": list(description.source.raw_columns),
        "selected_time_column": description.source.selected_time_column,
        "timestamp_unit": description.time.raw_timestamp_unit,
        "timestamp_scale_to_seconds": description.time.raw_timestamp_scale_to_seconds,
        "available_roi_channels": [
            asdict(item) for item in description.roi.available_roi_channels
        ],
        "time_basis": description.time.time_basis,
    }


def _validate_sha(value: str, name: str) -> None:
    if len(value) != 64 or value.lower() != value:
        _fail(f"{name} must be a lowercase SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError:
        _fail(f"{name} must be a lowercase SHA-256 digest.")


def _validate_roi(roi: ContinuousRwdRoiAuthority) -> None:
    if roi.selection_rule_version != ROI_SELECTION_RULE_VERSION:
        _fail("Unsupported ROI selection-rule version.")
    available_ids = tuple(item.roi_id for item in roi.available_roi_channels)
    if not available_ids or tuple(sorted(available_ids)) != available_ids:
        _fail("Available ROI inventory must be non-empty and canonically ordered.")
    if len(set(available_ids)) != len(available_ids):
        _fail("Duplicate available ROI IDs are not allowed.")
    columns: list[str] = []
    for item in roi.available_roi_channels:
        _string(item.roi_id, "ROI ID")
        columns.extend(
            (_string(item.reference_column, "reference column"),
             _string(item.signal_column, "signal column"))
        )
    if len(set(columns)) != len(columns):
        _fail("A source column cannot be assigned to more than one ROI or role.")
    if not roi.included_roi_ids:
        _fail("At least one ROI must be included.")
    if tuple(sorted(roi.included_roi_ids)) != roi.included_roi_ids:
        _fail("Included ROI IDs must be canonically ordered.")
    if tuple(sorted(roi.excluded_roi_ids)) != roi.excluded_roi_ids:
        _fail("Excluded ROI IDs must be canonically ordered.")
    if len(set(roi.included_roi_ids)) != len(roi.included_roi_ids):
        _fail("Duplicate included ROI IDs are not allowed.")
    if set(roi.included_roi_ids) | set(roi.excluded_roi_ids) != set(available_ids):
        _fail("Included and excluded ROI IDs must cover the available inventory.")
    if set(roi.included_roi_ids) & set(roi.excluded_roi_ids):
        _fail("Included and excluded ROI IDs must be disjoint.")


def _validate_time_evidence(
    *,
    raw_first_timestamp: Any,
    raw_last_timestamp: Any,
    timestamp_scale_to_seconds: Any,
    normalized_first_seconds: Any,
    normalized_last_seconds: Any,
    measured_duration_seconds: Any,
    total_data_row_count: Any,
    valid_timestamp_count: Any,
    positive_interval_count: Any,
) -> None:
    first = _finite(raw_first_timestamp, "raw first timestamp")
    last = _finite(raw_last_timestamp, "raw last timestamp")
    scale = _finite(timestamp_scale_to_seconds, "timestamp scale")
    if scale <= 0.0:
        _fail("Timestamp scale must be positive.")
    duration = _finite(measured_duration_seconds, "measured duration")
    if duration < MINIMUM_DURATION_SEC:
        _fail("Recording duration is below the 600-second CR1 minimum.")
    if last < first:
        _fail("Raw last timestamp must not be earlier than raw first timestamp.")
    if normalized_first_seconds != 0.0:
        _fail("Normalized elapsed origin must be exactly zero.")
    if normalized_last_seconds != duration:
        _fail("Normalized elapsed end must equal measured duration.")
    expected_duration = (last - first) * scale
    if not math.isclose(
        duration,
        expected_duration,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        _fail("Measured duration must agree with raw timestamps and timestamp scale.")
    total = _integer(total_data_row_count, "total row count", minimum=1)
    valid = _integer(valid_timestamp_count, "valid timestamp count", minimum=1)
    if valid > total:
        _fail("Valid timestamp count cannot exceed total row count.")
    if total != valid:
        _fail("Clean CR1-A evidence requires total and valid row counts to agree.")
    intervals = _integer(
        positive_interval_count,
        "positive interval count",
        minimum=1,
    )
    if intervals != valid - 1:
        _fail("Positive interval count must equal valid timestamp count minus one.")


def _validate_time(time: ContinuousRwdTimeAuthority) -> None:
    _string(time.raw_timestamp_unit, "raw timestamp unit")
    if time.measured_support_start_seconds != 0.0:
        _fail("Measured support start must be exactly zero.")
    if time.measured_support_end_seconds != time.measured_duration_seconds:
        _fail("Measured support end must equal measured duration.")
    if time.time_basis != TIME_BASIS:
        _fail("Unsupported normalized time basis.")


def _validate_cadence(cadence: ContinuousRwdCadenceAuthority) -> None:
    _string(cadence.cadence_evidence_policy_version, "cadence policy version")
    interval_count = _integer(
        cadence.positive_interval_count, "positive interval count", minimum=1
    )
    nominal = _finite(cadence.nominal_cadence_seconds, "nominal cadence")
    minimum = _finite(cadence.minimum_positive_dt_seconds, "minimum positive dt")
    maximum = _finite(cadence.maximum_positive_dt_seconds, "maximum positive dt")
    mean = _finite(cadence.mean_positive_dt_seconds, "mean positive dt")
    if min(nominal, minimum, maximum, mean) <= 0.0:
        _fail("Cadence and positive-dt values must be greater than zero.")
    if not minimum <= nominal <= maximum:
        _fail("Nominal cadence must lie between minimum and maximum positive dt.")
    if not minimum <= mean <= maximum:
        _fail("Mean cadence must lie between minimum and maximum positive dt.")
    standard_deviation = _finite(
        cadence.standard_deviation_positive_dt_seconds,
        "standard deviation positive dt",
    )
    if standard_deviation < 0.0:
        _fail("Cadence standard deviation must be nonnegative.")
    coefficient_of_variation = _finite(
        cadence.coefficient_of_variation,
        "coefficient of variation",
    )
    if coefficient_of_variation < 0.0:
        _fail("Cadence coefficient of variation must be nonnegative.")
    sample_count = _integer(
        cadence.quantile_sample_count, "quantile sample count", minimum=1
    )
    if sample_count > interval_count:
        _fail("Quantile sample count cannot exceed positive interval count.")
    long_count = _integer(
        cadence.unusually_long_interval_count, "long interval count"
    )
    short_count = _integer(
        cadence.unusually_short_interval_count, "short interval count"
    )
    _string(cadence.quantile_method, "quantile method")
    if not cadence.quantiles:
        _fail("At least one cadence quantile is required.")
    probabilities = tuple(item.probability for item in cadence.quantiles)
    quantile_values = tuple(item.dt_seconds for item in cadence.quantiles)
    for item in cadence.quantiles:
        probability = _finite(item.probability, "quantile probability")
        if not 0.0 <= probability <= 1.0:
            _fail("Cadence quantile probabilities must lie between zero and one.")
        value = _finite(item.dt_seconds, "quantile dt")
        if value <= 0.0:
            _fail("Cadence quantile dt values must be greater than zero.")
        if not minimum <= value <= maximum:
            _fail("Cadence quantile dt must lie between minimum and maximum positive dt.")
    if any(left >= right for left, right in zip(probabilities, probabilities[1:])):
        _fail("Cadence quantile probabilities must be unique and strictly ascending.")
    if any(left > right for left, right in zip(quantile_values, quantile_values[1:])):
        _fail("Cadence quantile dt values must be nondecreasing.")
    for items, count, reverse, name in (
        (cadence.largest_unusual_intervals, long_count, True, "largest unusual intervals"),
        (cadence.smallest_unusual_intervals, short_count, False, "smallest unusual intervals"),
    ):
        if len(items) > count:
            _fail(f"Retained {name} cannot exceed their total count.")
        for item in items:
            _integer(item.row_index, "unusual interval row index", minimum=1)
            if _finite(item.dt_seconds, "unusual interval dt") <= 0.0:
                _fail("Unusual interval dt must be greater than zero.")
            if _finite(item.estimated_expected_sample_multiple, "sample multiple") <= 0.0:
                _fail("Unusual interval expected-sample multiple must be greater than zero.")
        expected = tuple(sorted(items, key=lambda item: ((-item.dt_seconds if reverse else item.dt_seconds), item.row_index)))
        if items != expected:
            _fail(f"{name.capitalize()} are not deterministically ordered.")
    expected_identity = _digest(CADENCE_EVIDENCE_IDENTITY_DOMAIN, _cadence_payload(cadence))
    if cadence.cadence_evidence_identity != expected_identity:
        _fail("Cadence-evidence identity mismatch.")


def _validate_description(description: GuidedContinuousRwdRecordingDescription) -> None:
    if not isinstance(description, GuidedContinuousRwdRecordingDescription):
        _fail("description must be a GuidedContinuousRwdRecordingDescription.")
    expected = (SCHEMA_NAME, SCHEMA_VERSION, SOURCE_FORMAT, ACQUISITION_MODE,
                EXECUTION_ADMISSION_STATUS, UNRESOLVED_ADMISSION_CHECKS)
    actual = (description.schema_name, description.schema_version,
              description.source_format, description.acquisition_mode,
              description.execution_admission_status,
              description.unresolved_admission_checks)
    if actual != expected:
        _fail("Unsupported continuous recording-description metadata.")
    if (
        description.source.inspection_contract_name != INSPECTION_CONTRACT_NAME
        or description.source.inspection_contract_version
        != INSPECTION_CONTRACT_VERSION
    ):
        _fail("Unsupported CR1-A inspection contract metadata.")
    _validate_sha(description.source.sha256, "source SHA-256")
    _validate_sha(description.source.source_content_identity, "source-content identity")
    _validate_sha(description.source.parser_interpretation_identity, "parser identity")
    _integer(description.source.file_size_bytes, "file size", minimum=1)
    _integer(description.source.total_data_row_count, "total row count", minimum=1)
    _integer(description.source.valid_timestamp_count, "valid timestamp count", minimum=1)
    _integer(description.source.header_row_index, "header row index")
    if (
        not isinstance(description.source.raw_columns, tuple)
        or not description.source.raw_columns
        or any(not isinstance(item, str) or not item for item in description.source.raw_columns)
    ):
        _fail("Raw columns must be a non-empty tuple of strings.")
    _string(description.source.selected_time_column, "selected time column")
    if description.source.selected_time_column not in description.source.raw_columns:
        _fail("Selected time column must be present in raw columns.")
    for name in ("selected_folder_canonical", "fluorescence_path_canonical",
                 "stable_source_identity"):
        _string(getattr(description.source, name), name)
    _integer(description.source.modification_time_ns, "modification time")
    expected_source = _digest(SOURCE_CONTENT_IDENTITY_DOMAIN, _source_content_payload(description.source))
    if description.source.source_content_identity != expected_source:
        _fail("Source-content identity mismatch.")
    _validate_roi(description.roi)
    _validate_time_evidence(
        raw_first_timestamp=description.time.raw_first_timestamp,
        raw_last_timestamp=description.time.raw_last_timestamp,
        timestamp_scale_to_seconds=description.time.raw_timestamp_scale_to_seconds,
        normalized_first_seconds=description.time.normalized_elapsed_origin_seconds,
        normalized_last_seconds=description.time.normalized_elapsed_end_seconds,
        measured_duration_seconds=description.time.measured_duration_seconds,
        total_data_row_count=description.source.total_data_row_count,
        valid_timestamp_count=description.source.valid_timestamp_count,
        positive_interval_count=description.cadence.positive_interval_count,
    )
    _validate_time(description.time)
    _validate_cadence(description.cadence)
    expected_parser = _digest(
        PARSER_INTERPRETATION_IDENTITY_DOMAIN,
        _stored_parser_payload(description),
    )
    if description.source.parser_interpretation_identity != expected_parser:
        _fail("Parser-interpretation identity mismatch.")
    if compute_guided_continuous_rwd_recording_identity(
        build_guided_continuous_rwd_recording_identity_payload(description)
    ) != description.recording_identity:
        _fail("Accepted-recording identity mismatch.")


def build_guided_continuous_rwd_recording_description(
    inspection: ContinuousRwdInspectionResult,
    *,
    included_roi_ids: tuple[str, ...],
) -> GuidedContinuousRwdRecordingDescription:
    if not isinstance(inspection, ContinuousRwdInspectionResult):
        _fail("inspection must be a ContinuousRwdInspectionResult.")
    if inspection.contract_name != INSPECTION_CONTRACT_NAME:
        _fail("Unsupported CR1-A inspection contract name.")
    if inspection.contract_version != INSPECTION_CONTRACT_VERSION:
        _fail("Unsupported CR1-A inspection contract version.")
    if inspection.status != "completed" or inspection.outcome_category != "inspection_completed":
        _fail("CR1-A inspection must have completed successfully.")
    if not inspection.source_stable:
        _fail("CR1-A source must be stable.")
    if inspection.source_identity is None or inspection.parser_facts is None or inspection.time_axis is None or inspection.channels is None:
        _fail("CR1-A inspection evidence is incomplete.")
    source = inspection.source_identity
    parser = inspection.parser_facts
    time_axis = inspection.time_axis
    channel_evidence = inspection.channels
    rejected_counts = {
        "duplicate timestamps": time_axis.duplicate_timestamp_count,
        "backward timestamps": time_axis.backward_timestamp_count,
        "nonnumeric timestamps": time_axis.nonnumeric_timestamp_count,
        "nonfinite timestamps": time_axis.nonfinite_timestamp_count,
        "malformed rows": channel_evidence.malformed_row_count,
        "nonnumeric selected-channel values": channel_evidence.nonnumeric_selected_value_count,
        "nonfinite selected-channel values": channel_evidence.nonfinite_selected_value_count,
    }
    for name, count in rejected_counts.items():
        if count != 0:
            _fail(f"CR1-A evidence contains {name}.")
    _validate_time_evidence(
        raw_first_timestamp=time_axis.raw_first_timestamp,
        raw_last_timestamp=time_axis.raw_last_timestamp,
        timestamp_scale_to_seconds=parser.timestamp_scale_to_seconds,
        normalized_first_seconds=time_axis.normalized_first_seconds,
        normalized_last_seconds=time_axis.normalized_last_seconds,
        measured_duration_seconds=time_axis.measured_duration_seconds,
        total_data_row_count=time_axis.total_data_row_count,
        valid_timestamp_count=time_axis.valid_timestamp_count,
        positive_interval_count=time_axis.positive_interval_count,
    )
    if channel_evidence.unmatched_channel_columns:
        _fail("CR1-A evidence contains unmatched ROI channel columns.")
    if not isinstance(included_roi_ids, tuple):
        _fail("included_roi_ids must be a tuple.")
    if not included_roi_ids:
        _fail("At least one ROI must be included.")
    if len(set(included_roi_ids)) != len(included_roi_ids):
        _fail("Duplicate included ROI IDs are not allowed.")
    available = tuple(sorted((ContinuousRwdRoiChannel(item.roi_id, item.reference_column, item.signal_column) for item in channel_evidence.roi_pairs), key=lambda item: item.roi_id))
    available_ids = {item.roi_id for item in available}
    if not available:
        _fail("At least one complete ROI pair is required.")
    unknown = set(included_roi_ids) - available_ids
    if unknown:
        _fail("Included ROI IDs must exist in the CR1-A inventory.")
    included = tuple(sorted(included_roi_ids))
    roi = ContinuousRwdRoiAuthority(
        ROI_SELECTION_RULE_VERSION, available, included,
        tuple(item.roi_id for item in available if item.roi_id not in set(included)),
    )
    _validate_roi(roi)
    source_content_identity = _digest(SOURCE_CONTENT_IDENTITY_DOMAIN, {
        "sha256": source.sha256, "file_size_bytes": source.file_size_bytes,
    })
    parser_identity = _digest(PARSER_INTERPRETATION_IDENTITY_DOMAIN, _parser_payload(inspection, available))
    source_binding = ContinuousRwdSourceBinding(
        inspection.contract_name, inspection.contract_version, source.sha256,
        source.file_size_bytes, time_axis.total_data_row_count,
        time_axis.valid_timestamp_count, parser.header_row_index,
        tuple(parser.raw_columns), parser.time_column,
        source_content_identity, parser_identity,
        source.selected_folder_canonical, source.fluorescence_path_canonical,
        source.modification_time_ns, source.stable_source_identity,
    )
    time = ContinuousRwdTimeAuthority(
        parser.timestamp_unit, parser.timestamp_scale_to_seconds,
        time_axis.raw_first_timestamp, time_axis.raw_last_timestamp, 0.0,
        time_axis.measured_duration_seconds, 0.0,
        time_axis.measured_duration_seconds, time_axis.measured_duration_seconds,
        TIME_BASIS,
    )
    cadence_without_identity = ContinuousRwdCadenceAuthority(
        time_axis.cadence_evidence_policy_version, time_axis.positive_interval_count,
        time_axis.nominal_cadence_seconds, time_axis.minimum_positive_dt_seconds,
        time_axis.maximum_positive_dt_seconds, time_axis.mean_positive_dt_seconds,
        time_axis.standard_deviation_positive_dt_seconds,
        time_axis.coefficient_of_variation, time_axis.quantile_method,
        time_axis.quantile_sample_count,
        tuple(
            ContinuousRwdCadenceQuantile(item.probability, item.dt_seconds)
            for item in time_axis.quantiles
        ),
        time_axis.unusually_long_interval_count, time_axis.unusually_short_interval_count,
        tuple(
            ContinuousRwdUnusualInterval(
                item.row_index,
                item.dt_seconds,
                item.estimated_expected_sample_multiple,
            )
            for item in time_axis.largest_unusual_intervals
        ),
        tuple(
            ContinuousRwdUnusualInterval(
                item.row_index,
                item.dt_seconds,
                item.estimated_expected_sample_multiple,
            )
            for item in time_axis.smallest_unusual_intervals
        ),
        "",
    )
    cadence = ContinuousRwdCadenceAuthority(
        **{**asdict(cadence_without_identity), "quantiles": cadence_without_identity.quantiles,
           "largest_unusual_intervals": cadence_without_identity.largest_unusual_intervals,
           "smallest_unusual_intervals": cadence_without_identity.smallest_unusual_intervals,
           "cadence_evidence_identity": _digest(CADENCE_EVIDENCE_IDENTITY_DOMAIN, _cadence_payload(cadence_without_identity))}
    )
    draft = GuidedContinuousRwdRecordingDescription(
        SCHEMA_NAME, SCHEMA_VERSION, SOURCE_FORMAT, ACQUISITION_MODE,
        EXECUTION_ADMISSION_STATUS, UNRESOLVED_ADMISSION_CHECKS,
        source_binding, roi, time, cadence, "",
    )
    description = GuidedContinuousRwdRecordingDescription(
        **{**draft.__dict__, "recording_identity": compute_guided_continuous_rwd_recording_identity(build_guided_continuous_rwd_recording_identity_payload(draft))}
    )
    _validate_description(description)
    return description


def build_guided_continuous_rwd_recording_identity_payload(
    description: GuidedContinuousRwdRecordingDescription,
) -> dict[str, Any]:
    return {
        "schema_name": description.schema_name,
        "schema_version": description.schema_version,
        "source_format": description.source_format,
        "acquisition_mode": description.acquisition_mode,
        "execution_admission_status": description.execution_admission_status,
        "unresolved_admission_checks": list(description.unresolved_admission_checks),
        "source_content_identity": description.source.source_content_identity,
        "parser_interpretation_identity": description.source.parser_interpretation_identity,
        "cadence_evidence_identity": description.cadence.cadence_evidence_identity,
        "row_authority": {
            "total_data_row_count": description.source.total_data_row_count,
            "valid_timestamp_count": description.source.valid_timestamp_count,
        },
        "time_authority": _time_payload(description.time),
        "roi_authority": _roi_payload(description.roi),
    }


def compute_guided_continuous_rwd_recording_identity(
    description_or_payload: GuidedContinuousRwdRecordingDescription | Mapping[str, Any],
) -> str:
    payload = (build_guided_continuous_rwd_recording_identity_payload(description_or_payload)
               if isinstance(description_or_payload, GuidedContinuousRwdRecordingDescription)
               else dict(description_or_payload))
    return _digest(RECORDING_IDENTITY_DOMAIN, payload)


def serialize_guided_continuous_rwd_recording_description(
    description: GuidedContinuousRwdRecordingDescription,
) -> dict[str, Any]:
    _validate_description(description)
    source_payload = asdict(description.source)
    source_payload["raw_columns"] = list(description.source.raw_columns)
    return {
        "schema_name": description.schema_name,
        "schema_version": description.schema_version,
        "source_format": description.source_format,
        "acquisition_mode": description.acquisition_mode,
        "execution_admission_status": description.execution_admission_status,
        "unresolved_admission_checks": list(description.unresolved_admission_checks),
        "source": source_payload,
        "roi": _roi_payload(description.roi),
        "time": _time_payload(description.time),
        "cadence": _cadence_payload(description.cadence, include_identity=True),
        "recording_identity": description.recording_identity,
    }


def _strict(mapping: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping) or set(mapping) != keys:
        _fail(f"{name} must contain exactly the v1 fields.")
    return mapping


def deserialize_guided_continuous_rwd_recording_description(
    serialized: Any,
) -> GuidedContinuousRwdRecordingDescription:
    top = _strict(serialized, {"schema_name", "schema_version", "source_format", "acquisition_mode", "execution_admission_status", "unresolved_admission_checks", "source", "roi", "time", "cadence", "recording_identity"}, "serialized description")
    source_raw = _strict(top["source"], set(ContinuousRwdSourceBinding.__dataclass_fields__), "source")
    roi_raw = _strict(top["roi"], {"selection_rule_version", "available_roi_channels", "included_roi_ids", "excluded_roi_ids"}, "ROI authority")
    time_raw = _strict(top["time"], set(ContinuousRwdTimeAuthority.__dataclass_fields__), "time authority")
    cadence_raw = _strict(top["cadence"], set(ContinuousRwdCadenceAuthority.__dataclass_fields__), "cadence authority")
    try:
        channels = tuple(ContinuousRwdRoiChannel(**_strict(item, set(ContinuousRwdRoiChannel.__dataclass_fields__), "ROI channel")) for item in roi_raw["available_roi_channels"])
        quantiles = tuple(ContinuousRwdCadenceQuantile(**_strict(item, set(ContinuousRwdCadenceQuantile.__dataclass_fields__), "cadence quantile")) for item in cadence_raw["quantiles"])
        largest = tuple(ContinuousRwdUnusualInterval(**_strict(item, set(ContinuousRwdUnusualInterval.__dataclass_fields__), "unusual interval")) for item in cadence_raw["largest_unusual_intervals"])
        smallest = tuple(ContinuousRwdUnusualInterval(**_strict(item, set(ContinuousRwdUnusualInterval.__dataclass_fields__), "unusual interval")) for item in cadence_raw["smallest_unusual_intervals"])
        cadence_values = dict(cadence_raw)
        cadence_values.update(quantiles=quantiles, largest_unusual_intervals=largest, smallest_unusual_intervals=smallest)
        description = GuidedContinuousRwdRecordingDescription(
            schema_name=top["schema_name"], schema_version=top["schema_version"],
            source_format=top["source_format"], acquisition_mode=top["acquisition_mode"],
            execution_admission_status=top["execution_admission_status"],
            unresolved_admission_checks=tuple(top["unresolved_admission_checks"]),
            source=ContinuousRwdSourceBinding(
                **{**source_raw, "raw_columns": tuple(source_raw["raw_columns"])}
            ),
            roi=ContinuousRwdRoiAuthority(
                roi_raw["selection_rule_version"], channels,
                tuple(roi_raw["included_roi_ids"]), tuple(roi_raw["excluded_roi_ids"]),
            ),
            time=ContinuousRwdTimeAuthority(**time_raw),
            cadence=ContinuousRwdCadenceAuthority(**cadence_values),
            recording_identity=top["recording_identity"],
        )
    except ContinuousRwdRecordingAuthorityError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ContinuousRwdRecordingAuthorityError("Serialized description is malformed.") from exc
    _validate_description(description)
    return description
