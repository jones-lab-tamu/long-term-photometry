"""Versioned NPM parser policy and per-file read-only inspection facts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from photometry_pipeline.config import Config
NPM_OUTPUT_TIME_BASIS = "relative_seconds_since_uv_signal_overlap_origin"
NPM_PARSER_CONTRACT_SCHEMA_NAME = "npm_normalized_parser_contract"
NPM_PARSER_CONTRACT_SCHEMA_VERSION = "v1"


NPM_TIMESTAMP_UNIT = "seconds"
NPM_SUPPORT_POLICY_STRICT = "strict_overlap_inner_support"
NPM_SUPPORT_POLICY_PERMISSIVE = "permissive_overlap_from_t0"
NPM_CHANNEL_SELECTOR_COLUMN = "LedState"
NPM_CHANNEL_SELECTOR_OPERATOR = "eq"
NPM_CHANNEL_UV_VALUE = 1
NPM_CHANNEL_SIGNAL_VALUE = 2
NPM_TIMESTAMP_MONOTONIC_POLICY = "finite_subset_strictly_increasing"
NPM_ROI_ORDER_POLICY = "numeric_index_then_lexical"


class NpmParserContractError(ValueError):
    def __init__(self, category: str, message: str, **context: Any) -> None:
        self.category = category
        self.context = dict(context)
        super().__init__(message)


def _finite_positive(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NpmParserContractError("parser_contract_invalid", f"{field_name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise NpmParserContractError("parser_contract_invalid", f"{field_name} must be positive and finite.")
    return result


@dataclass(frozen=True)
class NpmParserContract:
    """All NPM parsing/resampling behavior frozen for one Setup check."""

    npm_time_axis: str = "system_timestamp"
    npm_system_ts_col: str = "SystemTimestamp"
    npm_computer_ts_col: str = "ComputerTimestamp"
    npm_led_col: str = "LedState"
    npm_region_prefix: str = "Region"
    npm_region_suffix: str = "G"
    target_fs_hz: float = 40.0
    session_duration_sec: float = 600.0
    allow_partial_final_chunk: bool = False
    adapter_value_nan_policy: str = "strict"
    timestamp_cv_max: float = 0.02

    def __post_init__(self) -> None:
        if self.npm_time_axis not in {"system_timestamp", "computer_timestamp"}:
            raise NpmParserContractError("parser_contract_invalid", "Unsupported NPM time axis.")
        for name in (
            "npm_system_ts_col",
            "npm_computer_ts_col",
            "npm_led_col",
            "npm_region_prefix",
            "npm_region_suffix",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise NpmParserContractError("parser_contract_invalid", f"{name} is required.")
        if self.adapter_value_nan_policy not in {"strict", "mask"}:
            raise NpmParserContractError("parser_contract_invalid", "Unsupported NPM value NaN policy.")
        if not isinstance(self.allow_partial_final_chunk, bool):
            raise NpmParserContractError("parser_contract_invalid", "allow_partial_final_chunk must be boolean.")
        _finite_positive(self.target_fs_hz, "target_fs_hz")
        _finite_positive(self.session_duration_sec, "session_duration_sec")
        if isinstance(self.timestamp_cv_max, bool) or not isinstance(self.timestamp_cv_max, (int, float)):
            raise NpmParserContractError("parser_contract_invalid", "timestamp_cv_max must be numeric.")
        if not math.isfinite(float(self.timestamp_cv_max)) or float(self.timestamp_cv_max) < 0:
            raise NpmParserContractError("parser_contract_invalid", "timestamp_cv_max must be finite and non-negative.")

    @property
    def support_policy(self) -> str:
        return (
            NPM_SUPPORT_POLICY_PERMISSIVE
            if self.allow_partial_final_chunk
            else NPM_SUPPORT_POLICY_STRICT
        )

    @property
    def timestamp_column_candidates(self) -> tuple[str, ...]:
        if self.npm_time_axis == "system_timestamp":
            return (self.npm_system_ts_col, "Timestamp")
        return (self.npm_computer_ts_col,)

    def content(self) -> dict[str, Any]:
        """Return canonical JSON-compatible policy content."""
        return {
            "schema_name": NPM_PARSER_CONTRACT_SCHEMA_NAME,
            "schema_version": NPM_PARSER_CONTRACT_SCHEMA_VERSION,
            "sampling": {
                "time_axis": self.npm_time_axis,
                "timestamp_column_candidates": list(self.timestamp_column_candidates),
                "system_timestamp_column": self.npm_system_ts_col,
                "computer_timestamp_column": self.npm_computer_ts_col,
                "timestamp_unit": NPM_TIMESTAMP_UNIT,
                "timestamp_finite_policy": "finite_values_are_authoritative; nonfinite_rows_are_excluded",
                "timestamp_monotonic_policy": NPM_TIMESTAMP_MONOTONIC_POLICY,
                "timestamp_cv_max": float(self.timestamp_cv_max),
                "led_column": self.npm_led_col,
                "led_values": {"uv": NPM_CHANNEL_UV_VALUE, "signal": NPM_CHANNEL_SIGNAL_VALUE},
                "channel_selector": {
                    "column": self.npm_led_col,
                    "operator": NPM_CHANNEL_SELECTOR_OPERATOR,
                },
                "region_prefix": self.npm_region_prefix,
                "region_suffix": self.npm_region_suffix,
                "roi_order_policy": NPM_ROI_ORDER_POLICY,
                "roi_nan_policy": self.adapter_value_nan_policy,
                "target_fs_hz": float(self.target_fs_hz),
                "session_duration_sec": float(self.session_duration_sec),
                "allow_partial_final_chunk": self.allow_partial_final_chunk,
                "support_policy": self.support_policy,
                "output_time_basis": NPM_OUTPUT_TIME_BASIS,
                "filename_chronology": {
                    "timestamp_pattern": r"YYYY-MM-DDTHH_MM_SS",
                    "exactly_one_per_filename": True,
                    "sort": "ascending_parsed_timestamp",
                    "timezone": "naive",
                    "gaps": "accept_warn_npm_schedule_gap",
                    "early_sessions": "accept_warn_npm_early_session",
                    "overlap": "refuse_against_configured_session_duration_only",
                },
                "alignment": "uv_signal_overlap_origin",
                "interpolation": {
                    "method": "linear",
                    "outside_support": "nan",
                },
            },
        }

    @property
    def digest(self) -> str:
        from photometry_pipeline.guided_normalized_recording import (
            compute_npm_parser_contract_digest,
        )

        return compute_npm_parser_contract_digest(self.content())

    @classmethod
    def from_config(cls, config: Config, *, session_duration_sec: float | None = None) -> "NpmParserContract":
        return cls(
            npm_time_axis=str(config.npm_time_axis),
            npm_system_ts_col=str(config.npm_system_ts_col),
            npm_computer_ts_col=str(config.npm_computer_ts_col),
            npm_led_col=str(config.npm_led_col),
            npm_region_prefix=str(config.npm_region_prefix),
            npm_region_suffix=str(config.npm_region_suffix),
            target_fs_hz=float(config.target_fs_hz),
            session_duration_sec=float(
                config.chunk_duration_sec if session_duration_sec is None else session_duration_sec
            ),
            allow_partial_final_chunk=bool(config.allow_partial_final_chunk),
            adapter_value_nan_policy=str(config.adapter_value_nan_policy),
            timestamp_cv_max=float(config.timestamp_cv_max),
        )


@dataclass(frozen=True)
class NpmSessionInspection:
    path: str
    resolved_timestamp_column: str
    timestamp_unit: str
    roi_ids: tuple[str, ...]
    roi_columns: tuple[str, ...]
    physical_to_canonical_roi_mapping: tuple[tuple[str, str], ...]
    overlap_origin_absolute: float
    resolved_support_start_offset_sec: float
    resolved_support_end_offset_sec: float
    resolved_support_start_absolute: float
    resolved_support_end_absolute: float
    observed_duration_sec: float
    output_time_basis: str
    support_policy: str
    warning_categories: tuple[str, ...] = ()


def _roi_key(column: str, prefix: str, suffix: str) -> tuple[int, int, str]:
    if column.startswith(prefix) and column.endswith(suffix):
        core = column[len(prefix) : len(column) - len(suffix) if suffix else None]
        if core.isdigit():
            return 0, int(core), column
    return 1, 0, column


def _physical_roi_index(column: str, prefix: str, suffix: str) -> int:
    if not column.startswith(prefix) or not column.endswith(suffix):
        raise NpmParserContractError(
            "npm_roi_column_invalid",
            "An NPM ROI column does not match the configured physical ROI naming rule.",
            column=column,
        )
    core = column[len(prefix) : len(column) - len(suffix) if suffix else None]
    if not core.isdigit():
        raise NpmParserContractError(
            "npm_roi_column_invalid",
            "NPM physical ROI columns must contain a numeric source index.",
            column=column,
        )
    return int(core)


def _resolve_time_column(columns: tuple[str, ...], contract: NpmParserContract) -> str:
    matches = tuple(item for item in contract.timestamp_column_candidates if item in columns)
    if not matches:
        raise NpmParserContractError(
            "npm_time_column_missing",
            "The configured NPM timestamp column is missing.",
            candidates=contract.timestamp_column_candidates,
        )
    # Ordered candidates are an explicit historical policy; the first is authoritative.
    return matches[0]


def _finite_stream(values: np.ndarray, label: str) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        raise NpmParserContractError(
            "npm_insufficient_stream_support",
            f"NPM {label} stream has fewer than two finite timestamps.",
        )
    if np.any(np.diff(finite) <= 0.0):
        raise NpmParserContractError(
            "npm_non_monotonic_timestamp",
            f"NPM {label} timestamps are not strictly increasing in their finite subset.",
        )
    return finite


def _stream_cv(values: np.ndarray) -> float:
    dt = np.diff(values)
    return float(np.std(dt) / np.mean(dt)) if dt.size and np.mean(dt) > 0 else float("inf")


def resolve_npm_support_geometry(
    uv_abs: np.ndarray,
    sig_abs: np.ndarray,
    contract: NpmParserContract,
) -> dict[str, float | np.ndarray]:
    """Resolve the approved overlap-origin geometry without interpolation."""
    uv = _finite_stream(uv_abs, "UV")
    sig = _finite_stream(sig_abs, "signal")
    t0 = max(float(uv[0]), float(sig[0]))
    uv_rel = uv - t0
    sig_rel = sig - t0
    if contract.allow_partial_final_chunk:
        support_end = min(float(uv_rel[-1]), float(sig_rel[-1]))
        if support_end <= 0:
            raise NpmParserContractError("npm_no_support", "NPM streams have no positive common support.")
        return {
            "t0_abs": t0,
            "overlap_origin_absolute": t0,
            "uv_rel_t0": uv_rel,
            "sig_rel_t0": sig_rel,
            "overlap_start_rel_t0": 0.0,
            "overlap_end_rel_t0": support_end,
            "inner_start_rel_overlap": 0.0,
            "inner_end_rel_overlap": support_end,
            "resolved_support_start_absolute": t0,
            "resolved_support_end_absolute": t0 + support_end,
            "observed_duration_sec": support_end,
        }

    target_dt = 1.0 / float(contract.target_fs_hz)
    uv_dt = float(np.median(np.diff(uv)))
    sig_dt = float(np.median(np.diff(sig)))
    support_tol = max(target_dt, uv_dt, sig_dt)
    uv_start, sig_start = float(uv_rel[0]), float(sig_rel[0])
    uv_end, sig_end = float(uv_rel[-1]), float(sig_rel[-1])
    if abs(uv_start - sig_start) > support_tol or abs(uv_end - sig_end) > support_tol:
        raise NpmParserContractError(
            "npm_support_mismatch",
            "NPM UV and signal support differ beyond the configured tolerance.",
        )
    overlap_start = max(uv_start, sig_start)
    overlap_end = min(uv_end, sig_end)
    uv_inner = uv[(uv_rel >= overlap_start) & (uv_rel <= overlap_end)] - t0 - overlap_start
    sig_inner = sig[(sig_rel >= overlap_start) & (sig_rel <= overlap_end)] - t0 - overlap_start
    if uv_inner.size < 2 or sig_inner.size < 2:
        raise NpmParserContractError(
            "npm_insufficient_overlap_support",
            "NPM UV/signal overlap contains fewer than two samples per stream.",
        )
    inner_start = max(float(uv_inner[0]), float(sig_inner[0]))
    inner_end = min(float(uv_inner[-1]), float(sig_inner[-1]))
    if inner_end <= inner_start:
        raise NpmParserContractError("npm_no_support", "NPM streams have no positive inner support.")
    return {
        "t0_abs": t0,
        "overlap_origin_absolute": t0 + overlap_start,
        "uv_rel_t0": uv - t0,
        "sig_rel_t0": sig - t0,
        "overlap_start_rel_t0": overlap_start,
        "overlap_end_rel_t0": overlap_end,
        "inner_start_rel_overlap": inner_start,
        "inner_end_rel_overlap": inner_end,
        "resolved_support_start_absolute": t0 + overlap_start + inner_start,
        "resolved_support_end_absolute": t0 + overlap_start + inner_end,
        "observed_duration_sec": inner_end - inner_start,
        "support_tolerance_sec": support_tol,
    }


def inspect_npm_csv(path: str, contract: NpmParserContract) -> NpmSessionInspection:
    """Inspect one NPM file using the frozen policy; no output is generated."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise NpmParserContractError("npm_csv_unreadable", "Unable to read an NPM CSV file.", path=path) from exc
    df.columns = [str(column).strip().lstrip("\ufeff") for column in df.columns]
    columns = tuple(str(column) for column in df.columns)
    time_col = _resolve_time_column(columns, contract)
    if contract.npm_led_col not in columns:
        raise NpmParserContractError("npm_led_column_missing", "The configured NPM LED column is missing.")
    roi_columns = tuple(
        sorted(
            (
                column
                for column in columns
                if column.startswith(contract.npm_region_prefix)
                and column.endswith(contract.npm_region_suffix)
            ),
            key=lambda column: _roi_key(column, contract.npm_region_prefix, contract.npm_region_suffix),
        )
    )
    if not roi_columns:
        raise NpmParserContractError("npm_roi_columns_missing", "No NPM ROI columns were found.")
    try:
        physical_indices = tuple(
            _physical_roi_index(
                column,
                contract.npm_region_prefix,
                contract.npm_region_suffix,
            )
            for column in roi_columns
        )
    except NpmParserContractError:
        raise
    if len(set(physical_indices)) != len(physical_indices):
        raise NpmParserContractError(
            "npm_roi_column_invalid",
            "NPM physical ROI source indices must be unique.",
        )
    physical_to_canonical_roi_mapping = tuple(
        (
            f"{contract.npm_region_prefix}{position}",
            physical_column,
        )
        for position, physical_column in enumerate(roi_columns)
    )
    time_values = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    led_values = pd.to_numeric(df[contract.npm_led_col], errors="coerce").to_numpy(dtype=float)
    uv_rows = led_values == 1
    sig_rows = led_values == 2
    uv_abs = _finite_stream(time_values[uv_rows], "UV")
    sig_abs = _finite_stream(time_values[sig_rows], "signal")
    if _stream_cv(uv_abs) > float(contract.timestamp_cv_max) or _stream_cv(sig_abs) > float(contract.timestamp_cv_max):
        raise NpmParserContractError(
            "npm_timestamp_cv_exceeded",
            "NPM timestamp cadence exceeds the configured CV tolerance.",
        )
    warnings: list[str] = []
    finite_time_rows = np.isfinite(time_values)
    for label, mask in (("uv", uv_rows & finite_time_rows), ("signal", sig_rows & finite_time_rows)):
        values_by_roi = df.loc[mask, list(roi_columns)].apply(
            pd.to_numeric, errors="coerce"
        )
        for roi_column in roi_columns:
            values = values_by_roi[roi_column].to_numpy(dtype=float)
            invalid = int(np.count_nonzero(~np.isfinite(values)))
            if invalid:
                if contract.adapter_value_nan_policy == "strict":
                    raise NpmParserContractError(
                        "npm_roi_value_nonfinite",
                        f"NPM {label} ROI values contain non-finite values.",
                    )
                warnings.append("npm_roi_values_masked")
                if int(np.count_nonzero(np.isfinite(values))) < 2:
                    warnings.append("npm_roi_all_nan_after_mask")
    geometry = resolve_npm_support_geometry(uv_abs, sig_abs, contract)
    return NpmSessionInspection(
        path=str(Path(path)),
        resolved_timestamp_column=time_col,
        timestamp_unit=NPM_TIMESTAMP_UNIT,
        roi_ids=tuple(
            canonical_roi_id
            for canonical_roi_id, _physical_column in physical_to_canonical_roi_mapping
        ),
        roi_columns=roi_columns,
        physical_to_canonical_roi_mapping=physical_to_canonical_roi_mapping,
        overlap_origin_absolute=float(geometry["overlap_origin_absolute"]),
        resolved_support_start_offset_sec=float(geometry["inner_start_rel_overlap"]),
        resolved_support_end_offset_sec=float(geometry["inner_end_rel_overlap"]),
        resolved_support_start_absolute=float(geometry["resolved_support_start_absolute"]),
        resolved_support_end_absolute=float(geometry["resolved_support_end_absolute"]),
        observed_duration_sec=float(geometry["observed_duration_sec"]),
        output_time_basis=NPM_OUTPUT_TIME_BASIS,
        support_policy=contract.support_policy,
        warning_categories=tuple(dict.fromkeys(warnings)),
    )
