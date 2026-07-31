"""Read-only, bounded inspection of one continuous generic CSV recording.

This mirrors :mod:`photometry_pipeline.io.rwd_continuous_source` for a plain
CSV file whose columns the scientist has already mapped in the ordinary Guided
CSV controls. It establishes source facts only: it does not create a Guided
plan, authorize execution, process signals, or write beside the source.

The recording description built from this inspection is the same normalized
object the continuous route already consumes, so nothing downstream changes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Sequence

from .rwd_contract import RwdHeaderContractInspection, RwdRoiChannelPair
from .rwd_continuous_source import (
    CADENCE_EVIDENCE_POLICY_VERSION,
    MINIMUM_DURATION_SEC,
    QUANTILE_PROBABILITIES,
    ContinuousRwdCadenceQuantile,
    ContinuousRwdChannelEvidence,
    ContinuousRwdFinding,
    ContinuousRwdInspectionResult,
    ContinuousRwdParserFacts,
    ContinuousRwdRoiPair,
    ContinuousRwdTimeAxisEvidence,
    _canonical,
    _facts,
    _identity,
    _Interrupted,
    _quantile,
    _RunningIntervals,
    _scan_and_hash,
    _scan_interval_outliers,
)


INSPECTION_CONTRACT_NAME = "continuous_csv_source_inspection"
INSPECTION_CONTRACT_VERSION = "v1"

# The scientist selects the unit in the ordinary Guided CSV controls, so the
# cadence is read straight from the selected elapsed-time column rather than
# guessed from vendor metadata.
TIMESTAMP_UNIT_SCALES = {"seconds": 1.0, "milliseconds": 0.001}


@dataclass(frozen=True)
class ContinuousCsvRoiSelection:
    """One mapped ROI: its signal column and its reference column."""

    roi_id: str
    signal_column: str
    reference_column: str


def _result(category: str, summary: str, **values: object) -> ContinuousRwdInspectionResult:
    return ContinuousRwdInspectionResult(
        contract_name=INSPECTION_CONTRACT_NAME,
        contract_version=INSPECTION_CONTRACT_VERSION,
        status="failed",
        outcome_category=category,
        scientist_summary=summary,
        **values,
    )


def candidate_csv_files(folder: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Every plain CSV in the selected folder, in stable order."""
    try:
        entries = sorted(Path(folder).iterdir(), key=lambda item: item.name.casefold())
    except OSError:
        return ()
    return tuple(
        item
        for item in entries
        if item.is_file() and item.suffix.casefold() == ".csv"
    )


def _header_inspection(
    columns: Sequence[str],
    time_column: str,
    roi_selections: Sequence[ContinuousCsvRoiSelection],
) -> RwdHeaderContractInspection:
    """Describe the mapped columns using the existing header-contract shape."""
    return RwdHeaderContractInspection(
        inspection_schema_name="continuous_csv_header_mapping",
        inspection_schema_version="v1",
        header_row_index=0,
        selected_time_column=str(time_column),
        normalized_raw_columns=tuple(str(name) for name in columns),
        roi_channel_pairs=tuple(
            RwdRoiChannelPair(
                roi_id=str(selection.roi_id),
                raw_uv_column=str(selection.reference_column),
                raw_signal_column=str(selection.signal_column),
                # Generic CSV columns are chosen by the scientist, not matched
                # by a vendor channel suffix.
                matched_uv_suffix="",
                matched_signal_suffix="",
            )
            for selection in roi_selections
        ),
        roi_ids=tuple(str(selection.roi_id) for selection in roi_selections),
        acceptable_for_strict_identity=True,
    )


def _mapping_refusal(
    columns: Sequence[str],
    time_column: str,
    roi_selections: Sequence[ContinuousCsvRoiSelection],
) -> tuple[str, str] | None:
    """Scientist-facing refusal for an unusable column mapping."""
    if not columns or any(not str(name).strip() for name in columns):
        return ("unusable_header", "The first row of the CSV file is not a usable column header.")
    if len(set(columns)) != len(columns):
        return ("unusable_header", "The CSV file has repeated column names, so columns cannot be identified.")
    if not roi_selections:
        return ("no_selected_rois", "Select at least one ROI with a signal column and a reference column.")
    if time_column not in columns:
        return ("selected_column_missing", f"The selected time column '{time_column}' is not in the CSV file.")
    used: dict[str, str] = {time_column: "the time column"}
    for selection in roi_selections:
        if selection.signal_column == selection.reference_column:
            return (
                "signal_reference_identical",
                f"ROI '{selection.roi_id}' uses the same column for signal and reference. "
                "Choose a different reference column.",
            )
        for column, role in (
            (selection.signal_column, f"the signal column for ROI '{selection.roi_id}'"),
            (selection.reference_column, f"the reference column for ROI '{selection.roi_id}'"),
        ):
            if column not in columns:
                return ("selected_column_missing", f"The selected column '{column}' is not in the CSV file.")
            if column in used:
                return (
                    "selected_column_reused",
                    f"Column '{column}' is already used as {used[column]} and cannot also be {role}.",
                )
            used[column] = role
    return None


def inspect_continuous_csv_recording(
    source_file: str | os.PathLike[str],
    *,
    time_column: str,
    time_unit: str,
    roi_selections: Sequence[ContinuousCsvRoiSelection],
    cancellation_check: Callable[[], bool] | None = None,
) -> ContinuousRwdInspectionResult:
    """Inspect one mapped CSV file as one uninterrupted continuous recording."""
    source = Path(source_file)
    unit = str(time_unit).strip().lower()
    if unit not in TIMESTAMP_UNIT_SCALES:
        return _result(
            "unsupported_time_unit",
            "Select elapsed time in seconds or milliseconds.",
        )
    scale = TIMESTAMP_UNIT_SCALES[unit]
    if not source.is_file():
        return _result("file_inaccessible", "The selected CSV file could not be read.")
    try:
        source_canonical = _canonical(source)
        folder_canonical = _canonical(source.parent)
        before = _facts(source)
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            columns = tuple(name.strip() for name in next(reader, []))
    except (OSError, UnicodeError, csv.Error, StopIteration):
        return _result("file_inaccessible", "The selected CSV file could not be read.")

    refusal = _mapping_refusal(columns, str(time_column), roi_selections)
    if refusal is not None:
        return _result(*refusal)

    header = _header_inspection(columns, str(time_column), roi_selections)
    try:
        _cancelled = cancellation_check
        if _cancelled is not None and _cancelled():
            raise _Interrupted
        scan, sha256 = _scan_and_hash(source, header, cancellation_check)
        middle = _facts(source)
        if middle != before:
            return _result(
                "source_changed_during_inspection",
                "The file changed while it was being inspected. Select a completed recording and inspect it again.",
                full_file_passes=1,
            )
        intervals = scan["intervals"]
        assert isinstance(intervals, _RunningIntervals)
        if not scan["row_count"] or not scan["valid_count"] or not intervals.count:
            return _result(
                "no_usable_rows",
                "The CSV file contains no usable recording rows.",
                full_file_passes=1,
            )
        samples = sorted(intervals.samples)
        median_raw = _quantile(samples, 0.5)
        median_sec = median_raw * scale
        long_count, short_count, largest, smallest = _scan_interval_outliers(
            source, header, scale, median_sec, cancellation_check
        )
        after = _facts(source)
        if after != before:
            return _result(
                "source_changed_during_inspection",
                "The file changed while it was being inspected. Select a completed recording and inspect it again.",
                full_file_passes=2,
            )
    except _Interrupted:
        return _result("inspection_interrupted", "Inspection was interrupted before it completed.")
    except (OSError, UnicodeError):
        return _result("file_inaccessible", "The CSV file could not be inspected completely.")
    except (csv.Error, ValueError):
        return _result(
            "inspection_incomplete",
            "Inspection could not be completed because the CSV data is malformed or unsupported.",
        )

    first_raw = float(scan["first"])
    last_raw = float(scan["last"])
    duration = (last_raw - first_raw) * scale
    findings: list[ContinuousRwdFinding] = []
    outcome = "inspection_completed"
    summary = "Continuous CSV source inspection completed. This does not authorize analysis."
    categories = (
        (int(scan["malformed"]), "inconsistent_roi_channel_structure", "Rows with inconsistent column counts were found."),
        (int(scan["nonnumeric_time"]), "nonnumeric_or_nonfinite_time", "Nonnumeric timestamps were found."),
        (int(scan["nonfinite_time"]), "nonnumeric_or_nonfinite_time", "Nonfinite timestamps were found."),
        (int(scan["duplicates"]), "duplicate_timestamps_present", "Duplicate recording timestamps were found."),
        (int(scan["backwards"]), "backward_timestamps_present", "The recording timestamps move backward and cannot be treated as one uninterrupted recording."),
        (int(scan["nonnumeric_selected"]), "selected_channel_parse_failure", "Nonnumeric values were found in the selected signal or reference columns."),
        (int(scan["nonfinite_selected"]), "selected_channel_parse_failure", "Nonfinite values were found in the selected signal or reference columns."),
    )
    for count, category, message in categories:
        if count:
            findings.append(ContinuousRwdFinding(category, message, count=count))
            if outcome == "inspection_completed":
                outcome, summary = category, message
    if duration < MINIMUM_DURATION_SEC:
        findings.append(
            ContinuousRwdFinding(
                "below_minimum_duration",
                "The recording is shorter than the 10-minute minimum for this long-term analysis workflow.",
            )
        )
    quantiles = tuple(
        ContinuousRwdCadenceQuantile(probability, _quantile(samples, probability) * scale)
        for probability in QUANTILE_PROBABILITIES
    )
    std_sec = intervals.standard_deviation * scale
    mean_sec = intervals.mean * scale
    time_axis = ContinuousRwdTimeAxisEvidence(
        total_data_row_count=int(scan["row_count"]),
        valid_timestamp_count=int(scan["valid_count"]),
        raw_first_timestamp=first_raw,
        raw_last_timestamp=last_raw,
        normalized_first_seconds=0.0,
        normalized_last_seconds=duration,
        measured_duration_seconds=duration,
        minimum_duration_seconds=MINIMUM_DURATION_SEC,
        duration_product_classification=(
            "meets_product_minimum" if duration >= MINIMUM_DURATION_SEC else "below_product_minimum"
        ),
        positive_interval_count=intervals.count,
        nominal_cadence_seconds=median_sec,
        minimum_positive_dt_seconds=intervals.minimum * scale,
        maximum_positive_dt_seconds=intervals.maximum * scale,
        mean_positive_dt_seconds=mean_sec,
        standard_deviation_positive_dt_seconds=std_sec,
        coefficient_of_variation=std_sec / mean_sec if mean_sec else float("nan"),
        quantiles=quantiles,
        quantile_method="deterministic_reservoir_linear.v1",
        quantile_sample_count=len(samples),
        duplicate_timestamp_count=int(scan["duplicates"]),
        backward_timestamp_count=int(scan["backwards"]),
        nonnumeric_timestamp_count=int(scan["nonnumeric_time"]),
        nonfinite_timestamp_count=int(scan["nonfinite_time"]),
        unusually_long_interval_count=long_count,
        unusually_short_interval_count=short_count,
        largest_unusual_intervals=largest,
        smallest_unusual_intervals=smallest,
        cadence_evidence_policy_version=CADENCE_EVIDENCE_POLICY_VERSION,
    )
    channels = ContinuousRwdChannelEvidence(
        roi_pairs=tuple(
            ContinuousRwdRoiPair(pair.roi_id, pair.raw_uv_column, pair.raw_signal_column)
            for pair in header.roi_channel_pairs
        ),
        unmatched_channel_columns=(),
        selected_value_count=int(scan["selected_count"]),
        nonnumeric_selected_value_count=int(scan["nonnumeric_selected"]),
        nonfinite_selected_value_count=int(scan["nonfinite_selected"]),
        malformed_row_count=int(scan["malformed"]),
    )
    return ContinuousRwdInspectionResult(
        contract_name=INSPECTION_CONTRACT_NAME,
        contract_version=INSPECTION_CONTRACT_VERSION,
        status="completed" if outcome == "inspection_completed" else "failed",
        outcome_category=outcome,
        scientist_summary=summary,
        source_identity=_identity(folder_canonical, source_canonical, before, sha256),
        parser_facts=ContinuousRwdParserFacts(
            header_row_index=header.header_row_index,
            time_column=str(header.selected_time_column),
            raw_columns=header.normalized_raw_columns,
            timestamp_unit=unit,
            timestamp_scale_to_seconds=scale,
        ),
        time_axis=time_axis,
        channels=channels,
        findings=tuple(findings),
        source_stable=True,
        full_file_passes=2,
    )
