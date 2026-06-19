"""Guided diagnostic-cache state and artifact contracts.

This module is intentionally UI-free. It models the request and artifact
records needed before Guided can generate diagnostic caches from raw input.
It does not launch analysis, create directories, or write outputs unless a
caller explicitly uses the JSON file helper functions.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DIAGNOSTIC_CACHE_SCHEMA_VERSION = "guided_diagnostic_cache.v1"
DIAGNOSTIC_CACHE_PURPOSE = "guided_diagnostic_cache"
DIAGNOSTIC_CACHE_SCOPE_FULL = "full_selected_input"
DIAGNOSTIC_CACHE_SCOPE_FIRST_N = "first_n"
DIAGNOSTIC_CACHE_ARTIFACT_FILENAME = "guided_diagnostic_cache_artifact.json"
DIAGNOSTIC_CACHE_PROVENANCE_FILENAME = "guided_diagnostic_cache_provenance.json"
DIAGNOSTIC_CACHE_REQUEST_FILENAME = "guided_diagnostic_cache_request.json"
SUPPORTED_INPUT_FORMATS = {"auto", "rwd", "npm", "custom_tabular"}
SUPPORTED_ACQUISITION_MODES = {"intermittent", "continuous"}
SUPPORTED_DIAGNOSTIC_SCOPES = {
    DIAGNOSTIC_CACHE_SCOPE_FULL,
    DIAGNOSTIC_CACHE_SCOPE_FIRST_N,
}


class DiagnosticCacheContractError(ValueError):
    """Raised when diagnostic-cache contract data is invalid."""


@dataclass(frozen=True)
class DiagnosticCacheStatus:
    ok: bool
    code: str
    message: str
    warnings: tuple[str, ...] = ()
    stale: bool = False
    missing_artifacts: tuple[str, ...] = ()
    stale_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosticCacheResolvedSource:
    source_type: str
    cache_id: str
    cache_root_path: str
    phasic_trace_cache_path: str
    config_used_path: str
    request_json_path: str
    status_marker_path: str = ""
    run_report_path: str = ""
    artifact_record_path: str = ""
    provenance_path: str = ""
    included_roi_ids: tuple[str, ...] = ()
    excluded_roi_ids: tuple[str, ...] = ()
    roi_inventory: tuple[str, ...] = ()
    session_chunk_inventory_summary: dict[str, Any] = field(default_factory=dict)
    source_setup_signature: str = ""
    diagnostic_scope_signature: str = ""
    build_request_signature: str = ""
    warnings: tuple[str, ...] = ()

    def phasic_cache_source_args(self) -> dict[str, str]:
        return {
            "source_type": self.source_type,
            "cache_root_path": self.cache_root_path,
            "phasic_trace_cache_path": self.phasic_trace_cache_path,
            "config_used_path": self.config_used_path,
        }

    def as_phasic_cache_paths(self) -> dict[str, str]:
        return {
            "cache_root_path": self.cache_root_path,
            "phasic_trace_cache_path": self.phasic_trace_cache_path,
            "config_used_path": self.config_used_path,
        }


@dataclass(frozen=True)
class DiagnosticCacheResolveResult:
    status: DiagnosticCacheStatus
    source: DiagnosticCacheResolvedSource | None = None

    @property
    def ok(self) -> bool:
        return self.status.ok


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise DiagnosticCacheContractError(f"{field_name} must be a list, not a string")
    try:
        items = tuple(str(item).strip() for item in value)
    except TypeError as exc:
        raise DiagnosticCacheContractError(f"{field_name} must be iterable") from exc
    if any(not item for item in items):
        raise DiagnosticCacheContractError(f"{field_name} must not contain empty values")
    return items


def _normal_path(path: str | os.PathLike[str] | None) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    return os.path.normcase(os.path.abspath(os.path.normpath(text)))


def _normal_format(value: str) -> str:
    return str(value or "").strip().lower()


def _normal_mode(value: str) -> str:
    return str(value or "").strip().lower()


def _normal_scope(value: str) -> str:
    return str(value or "").strip().lower()


def _normal_roi_set(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _float_or_none(value: float | int | str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_none(value: int | str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _signature(data: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _require_mapping(data: Any, label: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise DiagnosticCacheContractError(f"{label} must be an object")
    return data


def _require_text(data: dict[str, Any], key: str) -> str:
    value = str(data.get(key, "") or "").strip()
    if not value:
        raise DiagnosticCacheContractError(f"{key} is required")
    return value


def _optional_text(data: dict[str, Any], key: str) -> str:
    return str(data.get(key, "") or "").strip()


def _validate_request_values(request: "DiagnosticCacheBuildRequest") -> None:
    if not str(request.raw_input_path or "").strip():
        raise DiagnosticCacheContractError("raw_input_path is required")
    if not str(request.output_base or "").strip():
        raise DiagnosticCacheContractError("output_base is required")
    if not str(request.requested_cache_path or "").strip():
        raise DiagnosticCacheContractError("requested_cache_path is required")
    if _normal_format(request.input_format) not in SUPPORTED_INPUT_FORMATS:
        raise DiagnosticCacheContractError(f"unsupported input_format: {request.input_format}")
    if _normal_mode(request.acquisition_mode) not in SUPPORTED_ACQUISITION_MODES:
        raise DiagnosticCacheContractError(
            f"unsupported acquisition_mode: {request.acquisition_mode}"
        )
    if not request.included_roi_ids:
        raise DiagnosticCacheContractError("included_roi_ids must contain at least one ROI")
    if _normal_scope(request.diagnostic_scope) not in SUPPORTED_DIAGNOSTIC_SCOPES:
        raise DiagnosticCacheContractError(
            f"unsupported diagnostic_scope: {request.diagnostic_scope}"
        )
    if request.diagnostic_scope == DIAGNOSTIC_CACHE_SCOPE_FIRST_N:
        if request.preview_first_n is None or int(request.preview_first_n) <= 0:
            raise DiagnosticCacheContractError(
                "preview_first_n must be a positive integer for first_n scope"
            )
    if request.continuous_window_sec is not None and float(request.continuous_window_sec) <= 0:
        raise DiagnosticCacheContractError("continuous_window_sec must be > 0")
    if request.continuous_step_sec is not None and float(request.continuous_step_sec) <= 0:
        raise DiagnosticCacheContractError("continuous_step_sec must be > 0")


@dataclass(frozen=True)
class DiagnosticCacheBuildRequest:
    raw_input_path: str
    input_format: str
    acquisition_mode: str
    included_roi_ids: tuple[str, ...]
    output_base: str
    requested_cache_path: str
    sessions_per_hour: int | None = None
    session_duration_sec: float | None = None
    continuous_window_sec: float = 600.0
    continuous_step_sec: float = 600.0
    allow_partial_final_window: bool = False
    excluded_roi_ids: tuple[str, ...] = ()
    baseline_config_source_path: str = ""
    baseline_config_source_kind: str = ""
    config_identity: str = ""
    diagnostic_scope: str = DIAGNOSTIC_CACHE_SCOPE_FULL
    preview_first_n: int | None = None
    requested_at_utc: str = ""
    schema_version: str = DIAGNOSTIC_CACHE_SCHEMA_VERSION
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_format", _normal_format(self.input_format))
        object.__setattr__(self, "acquisition_mode", _normal_mode(self.acquisition_mode))
        object.__setattr__(self, "diagnostic_scope", _normal_scope(self.diagnostic_scope))
        object.__setattr__(self, "included_roi_ids", _as_tuple(self.included_roi_ids, "included_roi_ids"))
        object.__setattr__(self, "excluded_roi_ids", _as_tuple(self.excluded_roi_ids, "excluded_roi_ids"))
        object.__setattr__(self, "warnings", _as_tuple(self.warnings, "warnings"))
        if not self.requested_at_utc:
            object.__setattr__(self, "requested_at_utc", utc_now_iso())
        _validate_request_values(self)

    def source_setup_signature_payload(self) -> dict[str, Any]:
        return {
            "raw_input_path": _normal_path(self.raw_input_path),
            "input_format": self.input_format,
            "acquisition_mode": self.acquisition_mode,
            "sessions_per_hour": self.sessions_per_hour,
            "session_duration_sec": self.session_duration_sec,
            "continuous_window_sec": self.continuous_window_sec,
            "continuous_step_sec": self.continuous_step_sec,
            "allow_partial_final_window": bool(self.allow_partial_final_window),
            "included_roi_ids": list(_normal_roi_set(self.included_roi_ids)),
            "excluded_roi_ids": list(_normal_roi_set(self.excluded_roi_ids)),
            "baseline_config_source_path": _normal_path(self.baseline_config_source_path),
            "baseline_config_source_kind": str(self.baseline_config_source_kind or ""),
            "config_identity": str(self.config_identity or ""),
        }

    def diagnostic_scope_signature_payload(self) -> dict[str, Any]:
        return {
            "diagnostic_scope": self.diagnostic_scope,
            "preview_first_n": self.preview_first_n,
        }

    def request_signature_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_setup": self.source_setup_signature_payload(),
            "diagnostic_scope": self.diagnostic_scope_signature_payload(),
            "output_base": _normal_path(self.output_base),
            "requested_cache_path": _normal_path(self.requested_cache_path),
        }

    @property
    def source_setup_signature(self) -> str:
        return _signature(self.source_setup_signature_payload())

    @property
    def diagnostic_scope_signature(self) -> str:
        return _signature(self.diagnostic_scope_signature_payload())

    @property
    def request_signature(self) -> str:
        return _signature(self.request_signature_payload())

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "raw_input_path": self.raw_input_path,
            "input_format": self.input_format,
            "acquisition_mode": self.acquisition_mode,
            "sessions_per_hour": self.sessions_per_hour,
            "session_duration_sec": self.session_duration_sec,
            "continuous_window_sec": self.continuous_window_sec,
            "continuous_step_sec": self.continuous_step_sec,
            "allow_partial_final_window": bool(self.allow_partial_final_window),
            "included_roi_ids": list(self.included_roi_ids),
            "excluded_roi_ids": list(self.excluded_roi_ids),
            "baseline_config_source_path": self.baseline_config_source_path,
            "baseline_config_source_kind": self.baseline_config_source_kind,
            "config_identity": self.config_identity,
            "diagnostic_scope": self.diagnostic_scope,
            "preview_first_n": self.preview_first_n,
            "output_base": self.output_base,
            "requested_cache_path": self.requested_cache_path,
            "requested_at_utc": self.requested_at_utc,
            "warnings": list(self.warnings),
            "signatures": {
                "source_setup_signature": self.source_setup_signature,
                "diagnostic_scope_signature": self.diagnostic_scope_signature,
                "request_signature": self.request_signature,
            },
        }

    @classmethod
    def from_json_dict(cls, data: Any) -> "DiagnosticCacheBuildRequest":
        data = _require_mapping(data, "DiagnosticCacheBuildRequest")
        schema = _optional_text(data, "schema_version") or DIAGNOSTIC_CACHE_SCHEMA_VERSION
        if schema != DIAGNOSTIC_CACHE_SCHEMA_VERSION:
            raise DiagnosticCacheContractError(f"unsupported schema_version: {schema}")
        return cls(
            schema_version=schema,
            raw_input_path=_require_text(data, "raw_input_path"),
            input_format=_require_text(data, "input_format"),
            acquisition_mode=_require_text(data, "acquisition_mode"),
            sessions_per_hour=_int_or_none(data.get("sessions_per_hour")),
            session_duration_sec=_float_or_none(data.get("session_duration_sec")),
            continuous_window_sec=float(data.get("continuous_window_sec", 600.0)),
            continuous_step_sec=float(data.get("continuous_step_sec", 600.0)),
            allow_partial_final_window=bool(data.get("allow_partial_final_window", False)),
            included_roi_ids=_as_tuple(data.get("included_roi_ids", ()), "included_roi_ids"),
            excluded_roi_ids=_as_tuple(data.get("excluded_roi_ids", ()), "excluded_roi_ids"),
            baseline_config_source_path=_optional_text(data, "baseline_config_source_path"),
            baseline_config_source_kind=_optional_text(data, "baseline_config_source_kind"),
            config_identity=_optional_text(data, "config_identity"),
            diagnostic_scope=_optional_text(data, "diagnostic_scope") or DIAGNOSTIC_CACHE_SCOPE_FULL,
            preview_first_n=_int_or_none(data.get("preview_first_n")),
            output_base=_optional_text(data, "output_base"),
            requested_cache_path=_optional_text(data, "requested_cache_path"),
            requested_at_utc=_optional_text(data, "requested_at_utc"),
            warnings=_as_tuple(data.get("warnings", ()), "warnings"),
        )


@dataclass(frozen=True)
class DiagnosticCacheArtifactRecord:
    cache_id: str
    source_path: str
    source_setup_signature: str
    build_request_signature: str
    diagnostic_scope_signature: str
    cache_root_path: str
    phasic_trace_cache_path: str
    config_used_path: str
    purpose: str = DIAGNOSTIC_CACHE_PURPOSE
    production_analysis: bool = False
    status_marker_path: str = ""
    run_report_path: str = ""
    effective_config_path: str = ""
    request_json_path: str = ""
    roi_inventory: tuple[str, ...] = ()
    included_roi_ids: tuple[str, ...] = ()
    excluded_roi_ids: tuple[str, ...] = ()
    session_chunk_inventory_summary: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = ""
    artifact_contract_version: str = DIAGNOSTIC_CACHE_SCHEMA_VERSION
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.cache_id or "").strip():
            raise DiagnosticCacheContractError("cache_id is required")
        if not str(self.source_path or "").strip():
            raise DiagnosticCacheContractError("source_path is required")
        for field_name in (
            "source_setup_signature",
            "build_request_signature",
            "diagnostic_scope_signature",
            "cache_root_path",
            "phasic_trace_cache_path",
            "config_used_path",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise DiagnosticCacheContractError(f"{field_name} is required")
        object.__setattr__(self, "roi_inventory", _as_tuple(self.roi_inventory, "roi_inventory"))
        object.__setattr__(self, "included_roi_ids", _as_tuple(self.included_roi_ids, "included_roi_ids"))
        object.__setattr__(self, "excluded_roi_ids", _as_tuple(self.excluded_roi_ids, "excluded_roi_ids"))
        object.__setattr__(self, "warnings", _as_tuple(self.warnings, "warnings"))
        if not isinstance(self.session_chunk_inventory_summary, dict):
            raise DiagnosticCacheContractError("session_chunk_inventory_summary must be an object")
        if not self.created_at_utc:
            object.__setattr__(self, "created_at_utc", utc_now_iso())

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "artifact_contract_version": self.artifact_contract_version,
            "cache_id": self.cache_id,
            "purpose": self.purpose,
            "production_analysis": bool(self.production_analysis),
            "source_path": self.source_path,
            "source_setup_signature": self.source_setup_signature,
            "build_request_signature": self.build_request_signature,
            "diagnostic_scope_signature": self.diagnostic_scope_signature,
            "cache_root_path": self.cache_root_path,
            "status_marker_path": self.status_marker_path,
            "run_report_path": self.run_report_path,
            "phasic_trace_cache_path": self.phasic_trace_cache_path,
            "config_used_path": self.config_used_path,
            "effective_config_path": self.effective_config_path,
            "request_json_path": self.request_json_path,
            "roi_inventory": list(self.roi_inventory),
            "included_roi_ids": list(self.included_roi_ids),
            "excluded_roi_ids": list(self.excluded_roi_ids),
            "session_chunk_inventory_summary": dict(self.session_chunk_inventory_summary),
            "created_at_utc": self.created_at_utc,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_json_dict(cls, data: Any) -> "DiagnosticCacheArtifactRecord":
        data = _require_mapping(data, "DiagnosticCacheArtifactRecord")
        version = _optional_text(data, "artifact_contract_version") or DIAGNOSTIC_CACHE_SCHEMA_VERSION
        if version != DIAGNOSTIC_CACHE_SCHEMA_VERSION:
            raise DiagnosticCacheContractError(f"unsupported artifact_contract_version: {version}")
        return cls(
            artifact_contract_version=version,
            cache_id=_require_text(data, "cache_id"),
            purpose=_optional_text(data, "purpose") or DIAGNOSTIC_CACHE_PURPOSE,
            production_analysis=bool(data.get("production_analysis", False)),
            source_path=_require_text(data, "source_path"),
            source_setup_signature=_require_text(data, "source_setup_signature"),
            build_request_signature=_require_text(data, "build_request_signature"),
            diagnostic_scope_signature=_require_text(data, "diagnostic_scope_signature"),
            cache_root_path=_require_text(data, "cache_root_path"),
            status_marker_path=_optional_text(data, "status_marker_path"),
            run_report_path=_optional_text(data, "run_report_path"),
            phasic_trace_cache_path=_require_text(data, "phasic_trace_cache_path"),
            config_used_path=_require_text(data, "config_used_path"),
            effective_config_path=_optional_text(data, "effective_config_path"),
            request_json_path=_optional_text(data, "request_json_path"),
            roi_inventory=_as_tuple(data.get("roi_inventory", ()), "roi_inventory"),
            included_roi_ids=_as_tuple(data.get("included_roi_ids", ()), "included_roi_ids"),
            excluded_roi_ids=_as_tuple(data.get("excluded_roi_ids", ()), "excluded_roi_ids"),
            session_chunk_inventory_summary=dict(data.get("session_chunk_inventory_summary") or {}),
            created_at_utc=_optional_text(data, "created_at_utc"),
            warnings=_as_tuple(data.get("warnings", ()), "warnings"),
        )


def validate_diagnostic_cache_artifact(
    record: DiagnosticCacheArtifactRecord,
    *,
    require_status_marker: bool = True,
    require_request_json: bool = False,
) -> DiagnosticCacheStatus:
    missing: list[str] = []
    warnings = list(record.warnings)
    if record.purpose != DIAGNOSTIC_CACHE_PURPOSE:
        return DiagnosticCacheStatus(
            ok=False,
            code="invalid_purpose",
            message="Diagnostic cache artifact purpose is not guided_diagnostic_cache.",
            warnings=tuple(warnings),
        )
    if bool(record.production_analysis):
        return DiagnosticCacheStatus(
            ok=False,
            code="production_analysis_not_allowed",
            message="Diagnostic cache artifact must not be marked as production analysis.",
            warnings=tuple(warnings),
        )
    if not record.source_path:
        missing.append("source_path")
    if not os.path.isdir(record.cache_root_path):
        missing.append("cache_root_path")
    if require_status_marker and (
        not record.status_marker_path or not os.path.isfile(record.status_marker_path)
    ):
        missing.append("status_marker_path")
    if not os.path.isfile(record.phasic_trace_cache_path):
        missing.append("phasic_trace_cache_path")
    if not os.path.isfile(record.config_used_path):
        missing.append("config_used_path")
    if require_request_json and (
        not record.request_json_path or not os.path.isfile(record.request_json_path)
    ):
        missing.append("request_json_path")
    if missing:
        return DiagnosticCacheStatus(
            ok=False,
            code="missing_artifacts",
            message="Diagnostic cache artifact is missing required files or metadata.",
            warnings=tuple(warnings),
            missing_artifacts=tuple(missing),
        )
    return DiagnosticCacheStatus(
        ok=True,
        code="ok",
        message="Diagnostic cache artifact is ready.",
        warnings=tuple(warnings),
    )


def _status_from_contract_error(exc: Exception) -> DiagnosticCacheStatus:
    message = str(exc)
    if "unsupported artifact_contract_version" in message or "unsupported schema_version" in message:
        return DiagnosticCacheStatus(
            ok=False,
            code="unsupported_schema_version",
            message=message,
        )
    return DiagnosticCacheStatus(
        ok=False,
        code="invalid_artifact_record",
        message=message,
    )


def _load_artifact_record_for_resolution(
    source: DiagnosticCacheArtifactRecord | str | os.PathLike[str],
) -> tuple[DiagnosticCacheArtifactRecord | None, str, DiagnosticCacheStatus | None]:
    if isinstance(source, DiagnosticCacheArtifactRecord):
        return source, "", None

    path = Path(source)
    artifact_path = path / DIAGNOSTIC_CACHE_ARTIFACT_FILENAME if path.is_dir() else path
    if not artifact_path.exists():
        return (
            None,
            str(artifact_path),
            DiagnosticCacheStatus(
                ok=False,
                code="missing_artifact_record",
                message=(
                    "Diagnostic cache artifact identity file is required: "
                    f"{artifact_path}"
                ),
                missing_artifacts=("artifact_record_path",),
            ),
        )
    if not artifact_path.is_file():
        return (
            None,
            str(artifact_path),
            DiagnosticCacheStatus(
                ok=False,
                code="missing_artifact_record",
                message=f"Diagnostic cache artifact identity path is not a file: {artifact_path}",
                missing_artifacts=("artifact_record_path",),
            ),
        )
    try:
        record = read_artifact_record_json(artifact_path)
    except json.JSONDecodeError as exc:
        return (
            None,
            str(artifact_path),
            DiagnosticCacheStatus(
                ok=False,
                code="invalid_json",
                message=f"Diagnostic cache artifact JSON is invalid: {exc}",
            ),
        )
    except DiagnosticCacheContractError as exc:
        return None, str(artifact_path), _status_from_contract_error(exc)
    except OSError as exc:
        return (
            None,
            str(artifact_path),
            DiagnosticCacheStatus(
                ok=False,
                code="invalid_artifact_record",
                message=f"Diagnostic cache artifact could not be read: {exc}",
            ),
        )
    return record, str(artifact_path), None


def resolve_diagnostic_cache_source(
    source: DiagnosticCacheArtifactRecord | str | os.PathLike[str],
) -> DiagnosticCacheResolveResult:
    record, artifact_path, load_status = _load_artifact_record_for_resolution(source)
    if load_status is not None:
        return DiagnosticCacheResolveResult(load_status)
    if record is None:
        return DiagnosticCacheResolveResult(
            DiagnosticCacheStatus(
                ok=False,
                code="invalid_artifact_record",
                message="Diagnostic cache artifact record could not be resolved.",
            )
        )

    validation = validate_diagnostic_cache_artifact(
        record,
        require_status_marker=False,
        require_request_json=True,
    )
    if not validation.ok:
        return DiagnosticCacheResolveResult(validation)

    provenance_path = ""
    if artifact_path:
        candidate = Path(artifact_path).with_name(DIAGNOSTIC_CACHE_PROVENANCE_FILENAME)
        if candidate.exists():
            provenance_path = str(candidate)
    else:
        candidate = Path(record.cache_root_path) / DIAGNOSTIC_CACHE_PROVENANCE_FILENAME
        if candidate.exists():
            provenance_path = str(candidate)

    resolved = DiagnosticCacheResolvedSource(
        source_type="diagnostic_cache",
        cache_id=record.cache_id,
        cache_root_path=record.cache_root_path,
        phasic_trace_cache_path=record.phasic_trace_cache_path,
        config_used_path=record.config_used_path,
        status_marker_path=record.status_marker_path if os.path.isfile(record.status_marker_path) else "",
        run_report_path=record.run_report_path if os.path.isfile(record.run_report_path) else "",
        request_json_path=record.request_json_path,
        artifact_record_path=artifact_path,
        provenance_path=provenance_path,
        included_roi_ids=record.included_roi_ids,
        excluded_roi_ids=record.excluded_roi_ids,
        roi_inventory=record.roi_inventory,
        session_chunk_inventory_summary=dict(record.session_chunk_inventory_summary),
        source_setup_signature=record.source_setup_signature,
        diagnostic_scope_signature=record.diagnostic_scope_signature,
        build_request_signature=record.build_request_signature,
        warnings=record.warnings,
    )
    return DiagnosticCacheResolveResult(
        DiagnosticCacheStatus(
            ok=True,
            code="ok",
            message="Diagnostic cache source resolved.",
            warnings=record.warnings,
        ),
        source=resolved,
    )


def resolve_diagnostic_cache_phasic_paths(
    source: DiagnosticCacheArtifactRecord | str | os.PathLike[str],
) -> DiagnosticCacheResolveResult:
    return resolve_diagnostic_cache_source(source)


def compare_request_to_artifact(
    request: DiagnosticCacheBuildRequest,
    record: DiagnosticCacheArtifactRecord,
) -> DiagnosticCacheStatus:
    stale_reasons: list[str] = []
    old_source_payload = _extract_source_payload_from_artifact(record)
    new_source_payload = request.source_setup_signature_payload()
    if request.source_setup_signature != record.source_setup_signature:
        reason_map = {
            "raw_input_path": "source path changed",
            "input_format": "input format changed",
            "acquisition_mode": "acquisition mode changed",
            "sessions_per_hour": "acquisition structure changed",
            "session_duration_sec": "acquisition structure changed",
            "continuous_window_sec": "acquisition structure changed",
            "continuous_step_sec": "acquisition structure changed",
            "allow_partial_final_window": "acquisition structure changed",
            "included_roi_ids": "ROI inclusion/exclusion changed",
            "excluded_roi_ids": "ROI inclusion/exclusion changed",
            "baseline_config_source_path": "baseline/config source changed",
            "baseline_config_source_kind": "baseline/config source changed",
            "config_identity": "baseline/config source changed",
        }
        for key, reason in reason_map.items():
            if old_source_payload.get(key) != new_source_payload.get(key):
                stale_reasons.append(reason)
    if request.diagnostic_scope_signature != record.diagnostic_scope_signature:
        stale_reasons.append("diagnostic scope changed")
    if request.request_signature != record.build_request_signature:
        stale_reasons.append("build request changed")
    stale_reasons = tuple(dict.fromkeys(stale_reasons))
    if stale_reasons:
        return DiagnosticCacheStatus(
            ok=False,
            code="stale",
            message="Diagnostic cache is stale for the current build request.",
            stale=True,
            stale_reasons=stale_reasons,
            warnings=record.warnings,
        )
    return DiagnosticCacheStatus(
        ok=True,
        code="current",
        message="Diagnostic cache matches the current build request.",
        warnings=record.warnings,
    )


def _extract_source_payload_from_artifact(record: DiagnosticCacheArtifactRecord) -> dict[str, Any]:
    payload = dict(record.session_chunk_inventory_summary.get("source_setup_signature_payload") or {})
    if payload:
        return payload
    return {
        "raw_input_path": _normal_path(record.source_path),
        "input_format": None,
        "acquisition_mode": None,
        "sessions_per_hour": None,
        "session_duration_sec": None,
        "continuous_window_sec": None,
        "continuous_step_sec": None,
        "allow_partial_final_window": None,
        "included_roi_ids": list(_normal_roi_set(record.included_roi_ids)),
        "excluded_roi_ids": list(_normal_roi_set(record.excluded_roi_ids)),
        "baseline_config_source_path": None,
        "baseline_config_source_kind": None,
        "config_identity": None,
    }


def artifact_record_from_request(
    request: DiagnosticCacheBuildRequest,
    *,
    cache_id: str,
    cache_root_path: str,
    phasic_trace_cache_path: str,
    config_used_path: str,
    purpose: str = DIAGNOSTIC_CACHE_PURPOSE,
    production_analysis: bool = False,
    status_marker_path: str = "",
    run_report_path: str = "",
    effective_config_path: str = "",
    request_json_path: str = "",
    roi_inventory: tuple[str, ...] | list[str] = (),
    session_chunk_inventory_summary: dict[str, Any] | None = None,
    created_at_utc: str = "",
    warnings: tuple[str, ...] | list[str] = (),
) -> DiagnosticCacheArtifactRecord:
    summary = dict(session_chunk_inventory_summary or {})
    summary.setdefault("source_setup_signature_payload", request.source_setup_signature_payload())
    summary.setdefault("diagnostic_scope_signature_payload", request.diagnostic_scope_signature_payload())
    return DiagnosticCacheArtifactRecord(
        cache_id=cache_id,
        purpose=purpose,
        production_analysis=production_analysis,
        source_path=request.raw_input_path,
        source_setup_signature=request.source_setup_signature,
        build_request_signature=request.request_signature,
        diagnostic_scope_signature=request.diagnostic_scope_signature,
        cache_root_path=cache_root_path,
        status_marker_path=status_marker_path,
        run_report_path=run_report_path,
        phasic_trace_cache_path=phasic_trace_cache_path,
        config_used_path=config_used_path,
        effective_config_path=effective_config_path,
        request_json_path=request_json_path,
        roi_inventory=tuple(roi_inventory),
        included_roi_ids=request.included_roi_ids,
        excluded_roi_ids=request.excluded_roi_ids,
        session_chunk_inventory_summary=summary,
        created_at_utc=created_at_utc,
        warnings=tuple(warnings),
    )


def write_json_file(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    Path(path).write_text(text + "\n", encoding="utf-8")


def read_json_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _require_mapping(data, "JSON file")


def write_build_request_json(
    path: str | os.PathLike[str],
    request: DiagnosticCacheBuildRequest,
) -> None:
    write_json_file(path, request.to_json_dict())


def read_build_request_json(path: str | os.PathLike[str]) -> DiagnosticCacheBuildRequest:
    return DiagnosticCacheBuildRequest.from_json_dict(read_json_file(path))


def write_artifact_record_json(
    path: str | os.PathLike[str],
    record: DiagnosticCacheArtifactRecord,
) -> None:
    write_json_file(path, record.to_json_dict())


def read_artifact_record_json(path: str | os.PathLike[str]) -> DiagnosticCacheArtifactRecord:
    return DiagnosticCacheArtifactRecord.from_json_dict(read_json_file(path))
