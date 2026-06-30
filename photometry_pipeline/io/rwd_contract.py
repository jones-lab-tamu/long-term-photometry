"""Strict, public, read-only RWD header contract inspection.

This module does not instantiate Config, load signal data, write files, invoke
the runner, or authorize validation or execution.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


PARSING_CONTRACT_SCHEMA_NAME = "rwd_header_parsing_contract"
PARSING_CONTRACT_SCHEMA_VERSION = "v1"
INSPECTION_SCHEMA_NAME = "rwd_header_contract_inspection"
INSPECTION_SCHEMA_VERSION = "v1"
COLUMN_NORMALIZATION_RULE = "strip_whitespace_and_bom.v1"
ROI_NAME_RULE = "exact_case_sensitive_reject_casefold_collisions.v1"
AMBIGUITY_POLICY = "reject_all.v1"


def _canonical_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, (tuple, list)):
        return all(_canonical_safe(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _canonical_safe(item)
            for key, item in value.items()
        )
    return False


def _freeze_context_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_context_value(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_context_value(item) for item in value)
    return value


def _immutable_context(context: Mapping[str, Any]) -> Mapping[str, Any]:
    copied = dict(context)
    if not _canonical_safe(copied):
        raise TypeError("RWD header finding context must contain canonical-safe values.")
    return _freeze_context_value(copied)


class RwdHeaderInspectionError(ValueError):
    """Categorized failure to inspect an RWD header."""

    def __init__(
        self,
        category: str,
        message: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.category = str(category)
        self.message = str(message)
        self.context = _immutable_context(context or {})
        super().__init__(self.message)


@dataclass(frozen=True)
class RwdHeaderParsingContract:
    schema_name: str = PARSING_CONTRACT_SCHEMA_NAME
    schema_version: str = PARSING_CONTRACT_SCHEMA_VERSION
    header_search_line_limit: int = 60
    time_column_candidates: tuple[str, ...] = ()
    uv_suffix_candidates: tuple[str, ...] = ()
    signal_suffix_candidates: tuple[str, ...] = ()
    column_normalization_rule: str = COLUMN_NORMALIZATION_RULE
    roi_name_rule: str = ROI_NAME_RULE
    ambiguity_policy: str = AMBIGUITY_POLICY
    unresolved_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "time_column_candidates",
            "uv_suffix_candidates",
            "signal_suffix_candidates",
            "unresolved_inputs",
        ):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                raise _invalid_contract(
                    f"{name} must be a tuple or list.",
                    field=name,
                    received_type=type(value).__name__,
                )
            object.__setattr__(self, name, tuple(value))
        _validate_parsing_contract(self)


@dataclass(frozen=True)
class RwdRoiChannelPair:
    roi_id: str
    raw_uv_column: str
    raw_signal_column: str
    matched_uv_suffix: str
    matched_signal_suffix: str


@dataclass(frozen=True)
class RwdHeaderFinding:
    category: str
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.category or not self.message:
            raise ValueError("RWD header findings require category and message.")
        object.__setattr__(self, "context", _immutable_context(self.context))


@dataclass(frozen=True)
class RwdHeaderContractInspection:
    inspection_schema_name: str
    inspection_schema_version: str
    header_row_index: int
    selected_time_column: str | None
    normalized_raw_columns: tuple[str, ...]
    roi_channel_pairs: tuple[RwdRoiChannelPair, ...]
    roi_ids: tuple[str, ...]
    duplicate_raw_columns: tuple[RwdHeaderFinding, ...] = ()
    duplicate_roi_bases: tuple[RwdHeaderFinding, ...] = ()
    ambiguous_time_columns: tuple[RwdHeaderFinding, ...] = ()
    ambiguous_pairings: tuple[RwdHeaderFinding, ...] = ()
    reused_columns: tuple[RwdHeaderFinding, ...] = ()
    casefold_roi_collisions: tuple[RwdHeaderFinding, ...] = ()
    invalid_roi_ids: tuple[RwdHeaderFinding, ...] = ()
    warnings: tuple[RwdHeaderFinding, ...] = ()
    blocking_findings: tuple[RwdHeaderFinding, ...] = ()
    acceptable_for_strict_identity: bool = False


@dataclass(frozen=True)
class _PairCandidate:
    roi_id: str
    uv_index: int
    signal_index: int
    raw_uv_column: str
    raw_signal_column: str
    uv_suffix: str
    signal_suffix: str


def _invalid_contract(message: str, **context: Any) -> RwdHeaderInspectionError:
    return RwdHeaderInspectionError("invalid_rwd_parsing_contract", message, context)


def _validate_candidate_tuple(name: str, values: tuple[str, ...]) -> None:
    if not values:
        raise _invalid_contract(f"{name} must not be empty.", field=name)
    if any(not isinstance(value, str) or not value for value in values):
        raise _invalid_contract(
            f"{name} must contain non-empty strings.",
            field=name,
        )
    if any(value != value.strip() for value in values):
        raise _invalid_contract(
            f"{name} must not contain surrounding whitespace.",
            field=name,
        )
    if len(set(values)) != len(values):
        raise _invalid_contract(
            f"{name} must not contain duplicates.",
            field=name,
        )


def _validate_parsing_contract(contract: RwdHeaderParsingContract) -> None:
    if contract.schema_name != PARSING_CONTRACT_SCHEMA_NAME:
        raise _invalid_contract("Unsupported parsing contract schema name.")
    if contract.schema_version != PARSING_CONTRACT_SCHEMA_VERSION:
        raise _invalid_contract("Unsupported parsing contract schema version.")
    if (
        isinstance(contract.header_search_line_limit, bool)
        or not isinstance(contract.header_search_line_limit, int)
        or contract.header_search_line_limit <= 0
    ):
        raise _invalid_contract("header_search_line_limit must be a positive integer.")
    _validate_candidate_tuple("time_column_candidates", contract.time_column_candidates)
    _validate_candidate_tuple("uv_suffix_candidates", contract.uv_suffix_candidates)
    _validate_candidate_tuple(
        "signal_suffix_candidates",
        contract.signal_suffix_candidates,
    )
    overlap = sorted(
        set(contract.uv_suffix_candidates) & set(contract.signal_suffix_candidates)
    )
    if overlap:
        raise _invalid_contract(
            "UV and signal suffix candidates must not overlap.",
            overlapping_suffixes=overlap,
        )
    if contract.column_normalization_rule != COLUMN_NORMALIZATION_RULE:
        raise _invalid_contract("Unsupported column normalization rule.")
    if contract.roi_name_rule != ROI_NAME_RULE:
        raise _invalid_contract("Unsupported ROI name rule.")
    if contract.ambiguity_policy != AMBIGUITY_POLICY:
        raise _invalid_contract("Unsupported ambiguity policy.")
    if contract.unresolved_inputs:
        raise _invalid_contract(
            "Parsing contract has unresolved inputs.",
            unresolved_inputs=contract.unresolved_inputs,
        )


def _normalize_column(value: str) -> str:
    return str(value).lstrip("\ufeff").strip()


def _parse_line(line: str, line_index: int) -> tuple[str, ...]:
    try:
        fields = next(csv.reader([line], strict=True))
    except (csv.Error, StopIteration) as exc:
        raise RwdHeaderInspectionError(
            "malformed_rwd_csv_header",
            f"Unable to parse candidate RWD header line {line_index}.",
            {"line_index": line_index, "reason": str(exc)},
        ) from exc
    return tuple(_normalize_column(field) for field in fields)


def _enumerate_pair_candidates(
    columns: tuple[str, ...],
    contract: RwdHeaderParsingContract,
) -> tuple[_PairCandidate, ...]:
    candidates: list[_PairCandidate] = []
    for uv_index, uv_column in enumerate(columns):
        for uv_suffix in contract.uv_suffix_candidates:
            if not uv_column.endswith(uv_suffix):
                continue
            roi_id = uv_column[: -len(uv_suffix)]
            for signal_suffix in contract.signal_suffix_candidates:
                expected_signal = f"{roi_id}{signal_suffix}"
                for signal_index, signal_column in enumerate(columns):
                    if signal_column == expected_signal:
                        candidates.append(
                            _PairCandidate(
                                roi_id=roi_id,
                                uv_index=uv_index,
                                signal_index=signal_index,
                                raw_uv_column=uv_column,
                                raw_signal_column=signal_column,
                                uv_suffix=uv_suffix,
                                signal_suffix=signal_suffix,
                            )
                        )
    return tuple(candidates)


def _finding(category: str, message: str, **context: Any) -> RwdHeaderFinding:
    return RwdHeaderFinding(category, message, context)


def _duplicate_column_findings(
    columns: tuple[str, ...],
) -> tuple[RwdHeaderFinding, ...]:
    findings: list[RwdHeaderFinding] = []
    for column in sorted(set(columns)):
        indices = tuple(index for index, value in enumerate(columns) if value == column)
        if len(indices) > 1:
            findings.append(
                _finding(
                    "duplicate_raw_column",
                    f"Raw RWD header column {column!r} appears more than once.",
                    column=column,
                    indices=indices,
                )
            )
    return tuple(findings)


def _invalid_roi_reason(roi_id: str) -> str | None:
    if not roi_id:
        return "empty"
    if roi_id != roi_id.strip():
        return "surrounding_whitespace"
    if any(ord(char) < 32 or ord(char) == 127 for char in roi_id):
        return "control_character"
    if "/" in roi_id or "\\" in roi_id:
        return "path_separator"
    if "\x00" in roi_id:
        return "nul_character"
    return None


def _build_inspection(
    *,
    header_row_index: int,
    columns: tuple[str, ...],
    time_matches: tuple[str, ...],
    pair_candidates: tuple[_PairCandidate, ...],
) -> RwdHeaderContractInspection:
    duplicate_raw_columns = _duplicate_column_findings(columns)

    ambiguous_time_columns: tuple[RwdHeaderFinding, ...] = ()
    selected_time_column: str | None = None
    if len(time_matches) == 1:
        selected_time_column = time_matches[0]
    elif len(time_matches) > 1:
        ambiguous_time_columns = (
            _finding(
                "ambiguous_time_columns",
                "RWD header contains multiple supported time columns.",
                time_columns=time_matches,
            ),
        )

    candidates_by_base: dict[str, list[_PairCandidate]] = {}
    for candidate in pair_candidates:
        candidates_by_base.setdefault(candidate.roi_id, []).append(candidate)

    duplicate_roi_bases: list[RwdHeaderFinding] = []
    ambiguous_pairings: list[RwdHeaderFinding] = []
    for roi_id in sorted(candidates_by_base):
        base_candidates = candidates_by_base[roi_id]
        if len(base_candidates) > 1:
            duplicate_roi_bases.append(
                _finding(
                    "duplicate_roi_base",
                    f"ROI base {roi_id!r} has multiple channel-pair candidates.",
                    roi_id=roi_id,
                    candidate_count=len(base_candidates),
                )
            )
            ambiguous_pairings.append(
                _finding(
                    "ambiguous_suffix_pairing",
                    f"ROI base {roi_id!r} has ambiguous UV/signal suffix pairing.",
                    roi_id=roi_id,
                    pairings=tuple(
                        (
                            candidate.raw_uv_column,
                            candidate.raw_signal_column,
                            candidate.uv_suffix,
                            candidate.signal_suffix,
                        )
                        for candidate in base_candidates
                    ),
                )
            )

    candidates_by_column: dict[int, list[_PairCandidate]] = {}
    for candidate in pair_candidates:
        candidates_by_column.setdefault(candidate.uv_index, []).append(candidate)
        candidates_by_column.setdefault(candidate.signal_index, []).append(candidate)

    reused_columns: list[RwdHeaderFinding] = []
    reused_indices: set[int] = set()
    for index in sorted(candidates_by_column):
        uses = candidates_by_column[index]
        if len(uses) > 1:
            reused_indices.add(index)
            reused_columns.append(
                _finding(
                    "reused_raw_column",
                    f"Raw column {columns[index]!r} participates in multiple pairings.",
                    column=columns[index],
                    column_index=index,
                    use_count=len(uses),
                )
            )

    spellings_by_casefold: dict[str, set[str]] = {}
    for roi_id in candidates_by_base:
        spellings_by_casefold.setdefault(roi_id.casefold(), set()).add(roi_id)
    casefold_roi_collisions: list[RwdHeaderFinding] = []
    collided_ids: set[str] = set()
    for folded in sorted(spellings_by_casefold):
        spellings = tuple(sorted(spellings_by_casefold[folded]))
        if len(spellings) > 1:
            collided_ids.update(spellings)
            casefold_roi_collisions.append(
                _finding(
                    "roi_casefold_collision",
                    "ROI IDs differ only by case.",
                    roi_ids=spellings,
                )
            )

    invalid_roi_ids: list[RwdHeaderFinding] = []
    invalid_ids: set[str] = set()
    for roi_id in sorted(candidates_by_base):
        reason = _invalid_roi_reason(roi_id)
        if reason is not None:
            invalid_ids.add(roi_id)
            invalid_roi_ids.append(
                _finding(
                    "invalid_roi_id",
                    f"ROI ID {roi_id!r} is invalid for strict identity.",
                    roi_id=roi_id,
                    reason=reason,
                )
            )

    accepted: list[RwdRoiChannelPair] = []
    for roi_id in sorted(candidates_by_base):
        base_candidates = candidates_by_base[roi_id]
        if len(base_candidates) != 1:
            continue
        candidate = base_candidates[0]
        if (
            candidate.uv_index in reused_indices
            or candidate.signal_index in reused_indices
            or roi_id in collided_ids
            or roi_id in invalid_ids
        ):
            continue
        accepted.append(
            RwdRoiChannelPair(
                roi_id=roi_id,
                raw_uv_column=candidate.raw_uv_column,
                raw_signal_column=candidate.raw_signal_column,
                matched_uv_suffix=candidate.uv_suffix,
                matched_signal_suffix=candidate.signal_suffix,
            )
        )

    blocking = (
        duplicate_raw_columns
        + tuple(duplicate_roi_bases)
        + ambiguous_time_columns
        + tuple(ambiguous_pairings)
        + tuple(reused_columns)
        + tuple(casefold_roi_collisions)
        + tuple(invalid_roi_ids)
    )
    acceptable = (
        selected_time_column is not None
        and bool(accepted)
        and not blocking
    )
    return RwdHeaderContractInspection(
        inspection_schema_name=INSPECTION_SCHEMA_NAME,
        inspection_schema_version=INSPECTION_SCHEMA_VERSION,
        header_row_index=header_row_index,
        selected_time_column=selected_time_column,
        normalized_raw_columns=columns,
        roi_channel_pairs=tuple(accepted),
        roi_ids=tuple(pair.roi_id for pair in accepted),
        duplicate_raw_columns=duplicate_raw_columns,
        duplicate_roi_bases=tuple(duplicate_roi_bases),
        ambiguous_time_columns=ambiguous_time_columns,
        ambiguous_pairings=tuple(ambiguous_pairings),
        reused_columns=tuple(reused_columns),
        casefold_roi_collisions=tuple(casefold_roi_collisions),
        invalid_roi_ids=tuple(invalid_roi_ids),
        warnings=(),
        blocking_findings=blocking,
        acceptable_for_strict_identity=acceptable,
    )


def inspect_rwd_header_contract(
    path: str,
    *,
    parsing_contract: RwdHeaderParsingContract,
) -> RwdHeaderContractInspection:
    """Inspect one RWD CSV header without loading signal rows or writing files."""
    if not isinstance(parsing_contract, RwdHeaderParsingContract):
        raise _invalid_contract(
            "parsing_contract must be an RwdHeaderParsingContract."
        )
    if not isinstance(path, str) or not path:
        raise RwdHeaderInspectionError(
            "rwd_header_path_missing",
            "RWD header path is required.",
        )

    file_path = Path(path)
    try:
        handle = file_path.open("r", encoding="utf-8", errors="strict", newline="")
    except (OSError, UnicodeError) as exc:
        raise RwdHeaderInspectionError(
            "rwd_header_unreadable",
            f"Unable to open RWD header file: {path}",
            {"path": path, "reason": str(exc)},
        ) from exc

    try:
        with handle:
            for line_index in range(parsing_contract.header_search_line_limit):
                try:
                    line = next(handle)
                except StopIteration:
                    break
                except (OSError, UnicodeError) as exc:
                    raise RwdHeaderInspectionError(
                        "rwd_header_unreadable",
                        f"Unable to read RWD header file: {path}",
                        {"path": path, "line_index": line_index, "reason": str(exc)},
                    ) from exc

                columns = _parse_line(line, line_index)
                time_matches = tuple(
                    candidate
                    for candidate in parsing_contract.time_column_candidates
                    if candidate in columns
                )
                pair_candidates = _enumerate_pair_candidates(
                    columns,
                    parsing_contract,
                )
                if time_matches and pair_candidates:
                    return _build_inspection(
                        header_row_index=line_index,
                        columns=columns,
                        time_matches=time_matches,
                        pair_candidates=pair_candidates,
                    )
    finally:
        if not handle.closed:
            handle.close()

    raise RwdHeaderInspectionError(
        "rwd_header_not_found",
        "No RWD header with a supported time column and UV/signal pair was found.",
        {
            "path": path,
            "header_search_line_limit": parsing_contract.header_search_line_limit,
        },
    )
