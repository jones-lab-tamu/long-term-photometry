"""Read-only, bounded inspection of one continuous RWD acquisition folder.

This module establishes source facts only.  It does not create a Guided plan,
authorize execution, process signals, or write beside the selected source.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import random
from typing import Callable, Iterable, cast

from .adapters import (
    _extract_rwd_metadata_context,
    _resolve_rwd_timestamp_scale_from_median_dt,
)
from .rwd_contract import (
    RwdHeaderContractInspection,
    RwdHeaderInspectionError,
    RwdHeaderParsingContract,
    inspect_rwd_header_contract,
)


INSPECTION_CONTRACT_NAME = "continuous_rwd_source_inspection"
INSPECTION_CONTRACT_VERSION = "v1"
IDENTITY_POLICY_VERSION = "continuous-rwd-source-identity.v1"
CADENCE_EVIDENCE_POLICY_VERSION = "relative-to-measured-median.v1"
MINIMUM_DURATION_SEC = 600.0
QUANTILE_PROBABILITIES = (0.001, 0.01, 0.5, 0.99, 0.999)
DT_SAMPLE_CAPACITY = 100_000
EXTREME_INTERVAL_CAPACITY = 10
LONG_INTERVAL_MEDIAN_MULTIPLE = 1.5
SHORT_INTERVAL_MEDIAN_MULTIPLE = 0.5

_PARSER_CONTRACT = RwdHeaderParsingContract(
    time_column_candidates=("Time(s)", "TimeStamp"),
    uv_suffix_candidates=("-410",),
    signal_suffix_candidates=("-470",),
)


@dataclass(frozen=True)
class ContinuousRwdFinding:
    category: str
    summary: str
    detail: str = ""
    count: int = 0


@dataclass(frozen=True)
class ContinuousRwdSourceIdentity:
    identity_policy_version: str
    selected_folder_canonical: str
    fluorescence_path_canonical: str
    file_size_bytes: int
    modification_time_ns: int
    sha256: str
    stable_source_identity: str


@dataclass(frozen=True)
class ContinuousRwdParserFacts:
    header_row_index: int
    time_column: str
    raw_columns: tuple[str, ...]
    timestamp_unit: str
    timestamp_scale_to_seconds: float


@dataclass(frozen=True)
class ContinuousRwdRoiPair:
    roi_id: str
    reference_column: str
    signal_column: str


@dataclass(frozen=True)
class ContinuousRwdChannelEvidence:
    roi_pairs: tuple[ContinuousRwdRoiPair, ...]
    unmatched_channel_columns: tuple[str, ...]
    selected_value_count: int
    nonnumeric_selected_value_count: int
    nonfinite_selected_value_count: int
    malformed_row_count: int


@dataclass(frozen=True)
class ContinuousRwdCadenceQuantile:
    probability: float
    dt_seconds: float


@dataclass(frozen=True)
class ContinuousRwdIntervalEvidence:
    row_index: int
    dt_seconds: float
    estimated_expected_sample_multiple: float


@dataclass(frozen=True)
class ContinuousRwdTimeAxisEvidence:
    total_data_row_count: int
    valid_timestamp_count: int
    raw_first_timestamp: float
    raw_last_timestamp: float
    normalized_first_seconds: float
    normalized_last_seconds: float
    measured_duration_seconds: float
    minimum_duration_seconds: float
    duration_product_classification: str
    positive_interval_count: int
    nominal_cadence_seconds: float
    minimum_positive_dt_seconds: float
    maximum_positive_dt_seconds: float
    mean_positive_dt_seconds: float
    standard_deviation_positive_dt_seconds: float
    coefficient_of_variation: float
    quantiles: tuple[ContinuousRwdCadenceQuantile, ...]
    quantile_method: str
    quantile_sample_count: int
    duplicate_timestamp_count: int
    backward_timestamp_count: int
    nonnumeric_timestamp_count: int
    nonfinite_timestamp_count: int
    unusually_long_interval_count: int
    unusually_short_interval_count: int
    largest_unusual_intervals: tuple[ContinuousRwdIntervalEvidence, ...]
    smallest_unusual_intervals: tuple[ContinuousRwdIntervalEvidence, ...]
    cadence_evidence_policy_version: str


@dataclass(frozen=True)
class ContinuousRwdInspectionResult:
    contract_name: str
    contract_version: str
    status: str
    outcome_category: str
    scientist_summary: str
    source_identity: ContinuousRwdSourceIdentity | None = None
    parser_facts: ContinuousRwdParserFacts | None = None
    time_axis: ContinuousRwdTimeAxisEvidence | None = None
    channels: ContinuousRwdChannelEvidence | None = None
    findings: tuple[ContinuousRwdFinding, ...] = ()
    source_stable: bool = False
    full_file_passes: int = 0

    @property
    def inspection_completed(self) -> bool:
        return self.status == "completed"


@dataclass(frozen=True)
class _FileFacts:
    size: int
    mtime_ns: int
    device: int
    inode: int


class _Interrupted(Exception):
    pass


class _RunningIntervals:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.samples: list[float] = []
        self.rng = random.Random(0)

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        if len(self.samples) < DT_SAMPLE_CAPACITY:
            self.samples.append(value)
        else:
            slot = self.rng.randrange(self.count)
            if slot < DT_SAMPLE_CAPACITY:
                self.samples[slot] = value

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(self.m2 / self.count) if self.count else math.nan


def _result(category: str, summary: str, **values: object) -> ContinuousRwdInspectionResult:
    return ContinuousRwdInspectionResult(
        contract_name=INSPECTION_CONTRACT_NAME,
        contract_version=INSPECTION_CONTRACT_VERSION,
        status="failed",
        outcome_category=category,
        scientist_summary=summary,
        **values,
    )


def _facts(path: Path) -> _FileFacts:
    stat = path.stat()
    return _FileFacts(
        size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        device=int(stat.st_dev),
        inode=int(stat.st_ino),
    )


def _canonical(path: Path) -> str:
    return str(path.resolve(strict=True))


def _folder_entries(path: Path) -> tuple[Path, ...]:
    return tuple(path.iterdir())


def _cancelled(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise _Interrupted


def _matching_fluorescence(entries: Iterable[Path]) -> tuple[Path, ...]:
    if os.name == "nt":
        return tuple(item for item in entries if item.name.casefold() == "fluorescence.csv")
    return tuple(item for item in entries if item.name == "Fluorescence.csv")


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _prefix_columns(path: Path) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        for index, row in enumerate(csv.reader(handle)):
            if index >= _PARSER_CONTRACT.header_search_line_limit:
                break
            rows.append(tuple(str(value).lstrip("\ufeff").strip() for value in row))
    return tuple(rows)


def _classify_header_failure(path: Path) -> tuple[str, str]:
    try:
        rows = _prefix_columns(path)
    except (OSError, UnicodeError, csv.Error):
        return "file_inaccessible", "The fluorescence file could not be read."
    time_candidates = set(_PARSER_CONTRACT.time_column_candidates)
    suffixes = _PARSER_CONTRACT.uv_suffix_candidates + _PARSER_CONTRACT.signal_suffix_candidates
    has_time = any(time_candidates.intersection(row) for row in rows)
    has_channel = any(any(column.endswith(suffixes) for column in row) for row in rows)
    if has_channel and not has_time:
        return "no_supported_time_column", "The file does not contain a supported RWD time column."
    if has_time:
        return "inconsistent_roi_channel_structure", "The RWD fluorescence header does not contain an unambiguous supported ROI channel pair."
    return "unsupported_or_ambiguous_header", "The file does not contain a supported RWD fluorescence header."


def _identity(folder: str, source: str, facts: _FileFacts, sha256: str) -> ContinuousRwdSourceIdentity:
    payload = "\0".join(
        (
            IDENTITY_POLICY_VERSION,
            folder,
            source,
            str(facts.size),
            str(facts.mtime_ns),
            sha256,
        )
    ).encode("utf-8")
    return ContinuousRwdSourceIdentity(
        identity_policy_version=IDENTITY_POLICY_VERSION,
        selected_folder_canonical=folder,
        fluorescence_path_canonical=source,
        file_size_bytes=facts.size,
        modification_time_ns=facts.mtime_ns,
        sha256=sha256,
        stable_source_identity=hashlib.sha256(payload).hexdigest(),
    )


def _scan_and_hash(
    path: Path,
    header: RwdHeaderContractInspection,
    cancellation_check: Callable[[], bool] | None,
) -> tuple[dict[str, object], str]:
    time_index = header.normalized_raw_columns.index(str(header.selected_time_column))
    selected_indices = tuple(
        index
        for pair in header.roi_channel_pairs
        for index in (
            header.normalized_raw_columns.index(pair.raw_uv_column),
            header.normalized_raw_columns.index(pair.raw_signal_column),
        )
    )
    digest = hashlib.sha256()
    intervals = _RunningIntervals()
    row_count = valid_count = duplicate_count = backward_count = 0
    nonnumeric_time = nonfinite_time = 0
    selected_count = nonnumeric_selected = nonfinite_selected = malformed = 0
    first = last = previous = None

    def lines():
        with path.open("rb") as raw:
            for raw_line in raw:
                digest.update(raw_line)
                try:
                    yield raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("source_encoding_invalid") from exc

    reader = csv.reader(lines(), strict=True)
    try:
        for _ in range(header.header_row_index + 1):
            next(reader)
        for row_index, row in enumerate(reader, start=1):
            row_count += 1
            if row_count % 10_000 == 0:
                _cancelled(cancellation_check)
            if len(row) != len(header.normalized_raw_columns):
                malformed += 1
                continue
            raw_time = row[time_index].strip()
            try:
                timestamp = float(raw_time)
            except ValueError:
                nonnumeric_time += 1
                continue
            if not math.isfinite(timestamp):
                nonfinite_time += 1
                continue
            valid_count += 1
            if first is None:
                first = timestamp
            if previous is not None:
                dt = timestamp - previous
                if dt > 0.0:
                    intervals.add(dt)
                elif dt == 0.0:
                    duplicate_count += 1
                else:
                    backward_count += 1
            previous = timestamp
            last = timestamp
            for index in selected_indices:
                selected_count += 1
                try:
                    value = float(row[index].strip())
                except ValueError:
                    nonnumeric_selected += 1
                    continue
                if not math.isfinite(value):
                    nonfinite_selected += 1
    except (csv.Error, OSError, UnicodeError, ValueError) as exc:
        if str(exc) == "source_encoding_invalid":
            raise
        raise ValueError("source_scan_failed") from exc
    _cancelled(cancellation_check)
    return (
        {
            "row_count": row_count,
            "valid_count": valid_count,
            "first": first,
            "last": last,
            "duplicates": duplicate_count,
            "backwards": backward_count,
            "nonnumeric_time": nonnumeric_time,
            "nonfinite_time": nonfinite_time,
            "selected_count": selected_count,
            "nonnumeric_selected": nonnumeric_selected,
            "nonfinite_selected": nonfinite_selected,
            "malformed": malformed,
            "intervals": intervals,
        },
        digest.hexdigest(),
    )


def _scan_interval_outliers(
    path: Path,
    header: RwdHeaderContractInspection,
    scale: float,
    median_sec: float,
    cancellation_check: Callable[[], bool] | None,
) -> tuple[int, int, tuple[ContinuousRwdIntervalEvidence, ...], tuple[ContinuousRwdIntervalEvidence, ...]]:
    time_index = header.normalized_raw_columns.index(str(header.selected_time_column))
    long_threshold = median_sec * LONG_INTERVAL_MEDIAN_MULTIPLE
    short_threshold = median_sec * SHORT_INTERVAL_MEDIAN_MULTIPLE
    long_count = short_count = 0
    largest: list[ContinuousRwdIntervalEvidence] = []
    smallest: list[ContinuousRwdIntervalEvidence] = []
    previous: float | None = None
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        for _ in range(header.header_row_index + 1):
            next(reader)
        for row_index, row in enumerate(reader, start=1):
            if row_index % 10_000 == 0:
                _cancelled(cancellation_check)
            if len(row) != len(header.normalized_raw_columns):
                continue
            try:
                current = float(row[time_index].strip())
            except ValueError:
                continue
            if not math.isfinite(current):
                continue
            if previous is not None:
                dt_sec = (current - previous) * scale
                if dt_sec > long_threshold:
                    long_count += 1
                    largest.append(ContinuousRwdIntervalEvidence(row_index, dt_sec, dt_sec / median_sec))
                    largest = sorted(largest, key=lambda item: (-item.dt_seconds, item.row_index))[:EXTREME_INTERVAL_CAPACITY]
                elif 0.0 < dt_sec < short_threshold:
                    short_count += 1
                    smallest.append(ContinuousRwdIntervalEvidence(row_index, dt_sec, dt_sec / median_sec))
                    smallest = sorted(smallest, key=lambda item: (item.dt_seconds, item.row_index))[:EXTREME_INTERVAL_CAPACITY]
            previous = current
    _cancelled(cancellation_check)
    return long_count, short_count, tuple(largest), tuple(smallest)


def inspect_continuous_rwd_acquisition_folder(
    path: str | Path,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> ContinuousRwdInspectionResult:
    """Inspect the directly selected RWD acquisition folder without writing."""
    selected = Path(path)
    if not selected.exists():
        return _result("selected_path_missing", "The selected folder does not exist.")
    if not selected.is_dir():
        return _result("selected_path_not_directory", "Select an RWD acquisition folder, not a file.")
    try:
        folder_canonical = _canonical(selected)
        matches = _matching_fluorescence(_folder_entries(selected))
    except OSError:
        return _result("file_inaccessible", "The selected folder could not be read.")
    if not matches:
        return _result(
            "fluorescence_csv_missing",
            "Select the RWD acquisition folder that directly contains Fluorescence.csv.",
        )
    if len(matches) != 1:
        return _result(
            "ambiguous_fluorescence_candidates",
            "More than one Fluorescence.csv candidate was found in the selected folder.",
        )
    source = matches[0]
    if not source.is_file():
        return _result("file_inaccessible", "Fluorescence.csv is not a readable regular file.")
    try:
        source_canonical = _canonical(source)
        before = _facts(source)
        header = inspect_rwd_header_contract(
            source_canonical,
            parsing_contract=_PARSER_CONTRACT,
        )
    except RwdHeaderInspectionError:
        category, summary = _classify_header_failure(source)
        return _result(category, summary)
    except (OSError, UnicodeError):
        return _result("file_inaccessible", "The fluorescence file could not be read.")
    if not header.acceptable_for_strict_identity or header.selected_time_column is None:
        return _result(
            "unsupported_or_ambiguous_header",
            "The RWD fluorescence header is ambiguous and cannot be inspected safely.",
        )

    paired_columns = {
        value
        for pair in header.roi_channel_pairs
        for value in (pair.raw_uv_column, pair.raw_signal_column)
    }
    suffixes = _PARSER_CONTRACT.uv_suffix_candidates + _PARSER_CONTRACT.signal_suffix_candidates
    unmatched = tuple(
        column
        for column in header.normalized_raw_columns
        if column.endswith(suffixes) and column not in paired_columns
    )
    try:
        _cancelled(cancellation_check)
        scan, sha256 = _scan_and_hash(source, header, cancellation_check)
        middle = _facts(source)
        if middle != before:
            return _result(
                "source_changed_during_inspection",
                "The source changed while it was being inspected. Stop the acquisition or select a completed recording and inspect it again.",
                full_file_passes=1,
            )
        intervals = cast(_RunningIntervals, scan["intervals"])
        if not scan["row_count"] or not scan["valid_count"] or not intervals.count:
            return _result("no_usable_rows", "The fluorescence file contains no usable recording rows.", full_file_passes=1)
        samples = sorted(intervals.samples)
        median_raw = _quantile(samples, 0.5)
        metadata_fps, enabled_count = _extract_rwd_metadata_context(source_canonical, header.header_row_index)
        scale, timestamp_unit = _resolve_rwd_timestamp_scale_from_median_dt(
            median_raw, metadata_fps, enabled_count
        )
        median_sec = median_raw * scale
        long_count, short_count, largest, smallest = _scan_interval_outliers(
            source, header, scale, median_sec, cancellation_check
        )
        after = _facts(source)
        if after != before:
            return _result(
                "source_changed_during_inspection",
                "The source changed while it was being inspected. Stop the acquisition or select a completed recording and inspect it again.",
                full_file_passes=2,
            )
    except _Interrupted:
        return _result("inspection_interrupted", "Inspection was interrupted before it completed.")
    except (OSError, UnicodeError):
        return _result("file_inaccessible", "The fluorescence file could not be inspected completely.")
    except (csv.Error, ValueError):
        return _result(
            "inspection_incomplete",
            "Inspection could not be completed because the fluorescence data is malformed or unsupported.",
        )

    first_raw = float(scan["first"])
    last_raw = float(scan["last"])
    duration = (last_raw - first_raw) * scale
    findings: list[ContinuousRwdFinding] = []
    outcome = "inspection_completed"
    summary = "Continuous RWD source inspection completed. This does not authorize analysis."
    categories = (
        (int(scan["malformed"]), "inconsistent_roi_channel_structure", "Rows with inconsistent column counts were found."),
        (len(unmatched), "inconsistent_roi_channel_structure", "One or more RWD channels do not form a complete supported ROI pair."),
        (int(scan["nonnumeric_time"]), "nonnumeric_or_nonfinite_time", "Nonnumeric timestamps were found."),
        (int(scan["nonfinite_time"]), "nonnumeric_or_nonfinite_time", "Nonfinite timestamps were found."),
        (int(scan["duplicates"]), "duplicate_timestamps_present", "Duplicate recording timestamps were found."),
        (int(scan["backwards"]), "backward_timestamps_present", "The recording timestamps move backward and cannot be treated as one uninterrupted recording."),
        (int(scan["nonnumeric_selected"]), "selected_channel_parse_failure", "Nonnumeric values were found in selected fluorescence channels."),
        (int(scan["nonfinite_selected"]), "selected_channel_parse_failure", "Nonfinite values were found in selected fluorescence channels."),
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
        duration_product_classification=("meets_product_minimum" if duration >= MINIMUM_DURATION_SEC else "below_product_minimum"),
        positive_interval_count=intervals.count,
        nominal_cadence_seconds=median_sec,
        minimum_positive_dt_seconds=intervals.minimum * scale,
        maximum_positive_dt_seconds=intervals.maximum * scale,
        mean_positive_dt_seconds=mean_sec,
        standard_deviation_positive_dt_seconds=std_sec,
        coefficient_of_variation=std_sec / mean_sec,
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
        unmatched_channel_columns=unmatched,
        selected_value_count=int(scan["selected_count"]),
        nonnumeric_selected_value_count=int(scan["nonnumeric_selected"]),
        nonfinite_selected_value_count=int(scan["nonfinite_selected"]),
        malformed_row_count=int(scan["malformed"]),
    )
    identity = _identity(folder_canonical, source_canonical, before, sha256)
    return ContinuousRwdInspectionResult(
        contract_name=INSPECTION_CONTRACT_NAME,
        contract_version=INSPECTION_CONTRACT_VERSION,
        status="completed" if outcome == "inspection_completed" else "failed",
        outcome_category=outcome,
        scientist_summary=summary,
        source_identity=identity,
        parser_facts=ContinuousRwdParserFacts(
            header_row_index=header.header_row_index,
            time_column=header.selected_time_column,
            raw_columns=header.normalized_raw_columns,
            timestamp_unit=timestamp_unit,
            timestamp_scale_to_seconds=scale,
        ),
        time_axis=time_axis,
        channels=channels,
        findings=tuple(findings),
        source_stable=True,
        full_file_passes=2,
    )
