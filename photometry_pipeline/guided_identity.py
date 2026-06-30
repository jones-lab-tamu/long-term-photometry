"""Pure canonical identity foundation for future Guided backend validation.

This identity is non-authorizing until the deferred source, default/config,
dataset, feature/event, and output-safety identities are defined and included.
It does not inspect the filesystem, import GUI code, generate executable
artifacts, or replace the older non-authorizing draft fingerprint returned by
guided_validation_request.compute_request_identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any

from photometry_pipeline.guided_new_analysis_plan import (
    FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES,
)
from photometry_pipeline.guided_validation_request import GuidedValidationRequest


IDENTITY_SCHEMA_NAME = "guided_validation_request_identity"
IDENTITY_SCHEMA_VERSION = "v1"
CANONICALIZATION_ALGORITHM_VERSION = "typed_json_utf8.v1"
GUIDED_VALIDATION_REQUEST_SCHEMA_VERSION = "guided_validation_request.v1"

DEFERRED_CANONICAL_IDENTITY_FIELDS = (
    "app_build_identity",
    "backend_default_config_identity",
    "dataset_contract_identity",
    "dynamic_fit_parameter_contract_identity",
    "feature_event_profile_identity",
    "output_safety_classifier_identity",
    "roi_inventory_identity",
    "source_content_identity",
)
DEFERRED_CANONICAL_IDENTITY_STATUS = "unresolved_future_identity_inputs"

_WINDOWS_DRIVE_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_SUPPORTED_SOURCE_FORMATS = {"rwd", "RWD"}
_SUPPORTED_ACQUISITION_MODES = {"intermittent"}
_SUPPORTED_TIMELINE_ANCHOR_MODES = {"civil"}
_SUPPORTED_EXECUTION_MODES = {"phasic"}
_SUPPORTED_RUN_PROFILES = {"full"}
_SUPPORTED_STRATEGY_SCOPES = {"global"}
_SUPPORTED_CORRECTION_STRATEGIES = {"dynamic_fit"}
_SUPPORTED_OUTPUT_PATH_ROLES = {"output_base"}
_SUPPORTED_OUTPUT_CREATION_TIMINGS = {"future_execution_start_only"}
_SUPPORTED_RUN_DIRECTORY_STRATEGIES = {
    "derive_unique_run_id_under_output_base",
}


class GuidedIdentityError(ValueError):
    """Raised when canonical identity construction cannot be made unambiguous."""


@dataclass(frozen=True)
class CanonicalPath:
    canonical_path: str
    path_style: str  # windows_drive / windows_unc / posix


@dataclass(frozen=True)
class GuidedCanonicalValidationIdentityPayload:
    """Minimal non-authorizing payload with unresolved future identity inputs."""

    identity_schema_name: str
    identity_schema_version: str
    canonicalization_algorithm_version: str
    subset_contract_version: str
    request_schema_version: str
    source_path_canonical: str
    source_path_style: str
    source_format: str
    acquisition_mode: str
    sessions_per_hour: int
    session_duration_sec: float
    exclude_incomplete_final_rwd_chunk: bool
    timeline_anchor_mode: str
    included_roi_ids: tuple[str, ...]
    execution_mode: str
    run_profile: str
    traces_only: bool
    strategy_scope: str
    global_correction_strategy: str
    dynamic_fit_mode: str
    output_base_path_canonical: str
    output_base_path_style: str
    output_path_role: str
    output_creation_timing: str
    run_directory_strategy: str
    output_overwrite: bool
    deferred_fields: tuple[str, ...] = DEFERRED_CANONICAL_IDENTITY_FIELDS
    deferred_fields_status: str = DEFERRED_CANONICAL_IDENTITY_STATUS


def _resolve_lexical_segments(segments: list[str], *, root_name: str) -> list[str]:
    resolved: list[str] = []
    for segment in segments:
        if not segment or segment == ".":
            continue
        if segment == "..":
            if not resolved:
                raise GuidedIdentityError(
                    f"Path traversal above {root_name} is not allowed."
                )
            resolved.pop()
            continue
        resolved.append(segment)
    return resolved


def canonicalize_absolute_path(path: str) -> CanonicalPath:
    """Canonicalize an absolute path lexically without touching the filesystem."""
    if not isinstance(path, str) or not path:
        raise GuidedIdentityError("Path is required and must be a non-empty string.")
    if "\x00" in path:
        raise GuidedIdentityError("Path must not contain a NUL character.")

    if _WINDOWS_DRIVE_ABSOLUTE_RE.match(path):
        drive = path[0].lower()
        segments = re.split(r"[\\/]+", path[3:])
        resolved = _resolve_lexical_segments(segments, root_name="Windows drive root")
        canonical = f"{drive}:\\"
        if resolved:
            canonical += "\\".join(segment.casefold() for segment in resolved)
        return CanonicalPath(canonical, "windows_drive")

    if _WINDOWS_DRIVE_PREFIX_RE.match(path):
        raise GuidedIdentityError("Windows drive-relative paths are not supported.")

    if path.startswith("\\\\") or path.startswith("//"):
        segments = re.split(r"[\\/]+", path[2:])
        if len(segments) < 2 or not segments[0] or not segments[1]:
            raise GuidedIdentityError(
                "UNC paths require both a server and share component."
            )
        server, share = segments[0].casefold(), segments[1].casefold()
        resolved = _resolve_lexical_segments(
            segments[2:],
            root_name="UNC share root",
        )
        canonical = f"\\\\{server}\\{share}"
        if resolved:
            canonical += "\\" + "\\".join(
                segment.casefold() for segment in resolved
            )
        return CanonicalPath(canonical, "windows_unc")

    if path.startswith("/"):
        if "\\" in path:
            raise GuidedIdentityError(
                "Mixed POSIX and Windows path separators are ambiguous."
            )
        resolved = _resolve_lexical_segments(
            path[1:].split("/"),
            root_name="POSIX root",
        )
        canonical = "/" + "/".join(resolved)
        return CanonicalPath(canonical, "posix")

    if "\\" in path:
        raise GuidedIdentityError(
            "Relative or ambiguous Windows-style paths are not supported."
        )
    raise GuidedIdentityError("Relative paths are not supported.")


def _normalize_canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GuidedIdentityError("Non-finite numeric values are not supported.")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_canonical_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise GuidedIdentityError("Canonical object keys must be strings.")
        return {
            key: _normalize_canonical_value(item)
            for key, item in value.items()
        }
    raise GuidedIdentityError(
        f"Unsupported canonical value type: {type(value).__name__}."
    )


def encode_canonical_value(value: Any) -> bytes:
    """Encode supported typed values deterministically as canonical UTF-8 JSON."""
    normalized = _normalize_canonical_value(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def encode_canonical_identity_payload(
    payload: GuidedCanonicalValidationIdentityPayload,
) -> bytes:
    if not isinstance(payload, GuidedCanonicalValidationIdentityPayload):
        raise GuidedIdentityError(
            "payload must be a GuidedCanonicalValidationIdentityPayload."
        )
    _validate_payload_envelope(payload)
    return encode_canonical_value(asdict(payload))


def _require_supported(name: str, value: str, supported: set[str]) -> None:
    if not isinstance(value, str) or not value:
        raise GuidedIdentityError(f"{name} is required.")
    if value not in supported:
        raise GuidedIdentityError(f"Unsupported {name}: {value!r}.")


def _validate_payload_envelope(
    payload: GuidedCanonicalValidationIdentityPayload,
) -> None:
    if payload.identity_schema_name != IDENTITY_SCHEMA_NAME:
        raise GuidedIdentityError("Unsupported identity schema name.")
    if not payload.identity_schema_version:
        raise GuidedIdentityError("Identity schema version is required.")
    if not payload.canonicalization_algorithm_version:
        raise GuidedIdentityError(
            "Canonicalization algorithm version is required."
        )
    if not payload.subset_contract_version:
        raise GuidedIdentityError("Subset contract version is required.")
    if not payload.request_schema_version:
        raise GuidedIdentityError("Request schema version is required.")

    source_path = canonicalize_absolute_path(payload.source_path_canonical)
    if (
        source_path.canonical_path != payload.source_path_canonical
        or source_path.path_style != payload.source_path_style
    ):
        raise GuidedIdentityError("Source path is not in canonical form.")
    output_path = canonicalize_absolute_path(payload.output_base_path_canonical)
    if (
        output_path.canonical_path != payload.output_base_path_canonical
        or output_path.path_style != payload.output_base_path_style
    ):
        raise GuidedIdentityError("Output base path is not in canonical form.")

    _require_supported(
        "source format",
        payload.source_format,
        {"rwd"},
    )
    _require_supported(
        "acquisition mode",
        payload.acquisition_mode,
        _SUPPORTED_ACQUISITION_MODES,
    )
    _require_supported(
        "timeline anchor mode",
        payload.timeline_anchor_mode,
        _SUPPORTED_TIMELINE_ANCHOR_MODES,
    )
    _require_supported(
        "execution mode",
        payload.execution_mode,
        _SUPPORTED_EXECUTION_MODES,
    )
    _require_supported(
        "run profile",
        payload.run_profile,
        _SUPPORTED_RUN_PROFILES,
    )
    _require_supported(
        "strategy scope",
        payload.strategy_scope,
        _SUPPORTED_STRATEGY_SCOPES,
    )
    _require_supported(
        "global correction strategy",
        payload.global_correction_strategy,
        _SUPPORTED_CORRECTION_STRATEGIES,
    )
    _require_supported(
        "dynamic fit mode",
        payload.dynamic_fit_mode,
        FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES,
    )
    _require_supported(
        "output path role",
        payload.output_path_role,
        _SUPPORTED_OUTPUT_PATH_ROLES,
    )
    _require_supported(
        "output creation timing",
        payload.output_creation_timing,
        _SUPPORTED_OUTPUT_CREATION_TIMINGS,
    )
    _require_supported(
        "run directory strategy",
        payload.run_directory_strategy,
        _SUPPORTED_RUN_DIRECTORY_STRATEGIES,
    )

    if isinstance(payload.sessions_per_hour, bool) or not isinstance(
        payload.sessions_per_hour, int
    ):
        raise GuidedIdentityError("sessions_per_hour must be an integer.")
    if payload.sessions_per_hour <= 0:
        raise GuidedIdentityError("sessions_per_hour must be positive.")
    if isinstance(payload.session_duration_sec, bool) or not isinstance(
        payload.session_duration_sec, (int, float)
    ):
        raise GuidedIdentityError("session_duration_sec must be numeric.")
    if not math.isfinite(float(payload.session_duration_sec)):
        raise GuidedIdentityError("session_duration_sec must be finite.")
    if payload.session_duration_sec <= 0:
        raise GuidedIdentityError("session_duration_sec must be positive.")
    if not isinstance(payload.exclude_incomplete_final_rwd_chunk, bool):
        raise GuidedIdentityError(
            "exclude_incomplete_final_rwd_chunk must be boolean."
        )
    if not isinstance(payload.traces_only, bool) or payload.traces_only:
        raise GuidedIdentityError(
            "The first subset requires traces_only to be false."
        )
    if not isinstance(payload.output_overwrite, bool) or payload.output_overwrite:
        raise GuidedIdentityError(
            "The first subset requires output_overwrite to be false."
        )
    if not payload.included_roi_ids:
        raise GuidedIdentityError("At least one included ROI ID is required.")
    if any(not isinstance(roi, str) or not roi for roi in payload.included_roi_ids):
        raise GuidedIdentityError("Included ROI IDs must be non-empty strings.")
    if len(set(payload.included_roi_ids)) != len(payload.included_roi_ids):
        raise GuidedIdentityError("Duplicate included ROI IDs are not allowed.")
    if tuple(sorted(payload.included_roi_ids)) != payload.included_roi_ids:
        raise GuidedIdentityError("Included ROI IDs must be in canonical order.")
    if payload.deferred_fields != DEFERRED_CANONICAL_IDENTITY_FIELDS:
        raise GuidedIdentityError("Deferred identity fields do not match the schema.")
    if payload.deferred_fields_status != DEFERRED_CANONICAL_IDENTITY_STATUS:
        raise GuidedIdentityError(
            "Deferred identity fields must remain explicitly unresolved."
        )


def build_canonical_guided_identity_payload_from_request(
    request: GuidedValidationRequest,
) -> GuidedCanonicalValidationIdentityPayload:
    """Compile the minimal first-subset identity without validation or I/O."""
    if not isinstance(request, GuidedValidationRequest):
        raise GuidedIdentityError("request must be a GuidedValidationRequest.")
    if request.source_path is None:
        raise GuidedIdentityError("source_path is required.")
    if request.output_base_path is None:
        raise GuidedIdentityError("output_base_path is required.")
    if request.sessions_per_hour is None:
        raise GuidedIdentityError("sessions_per_hour is required.")
    if request.session_duration_sec is None:
        raise GuidedIdentityError("session_duration_sec is required.")
    if request.global_correction_strategy is None:
        raise GuidedIdentityError("global_correction_strategy is required.")
    if request.dynamic_fit_mode is None:
        raise GuidedIdentityError("dynamic_fit_mode is required.")

    source_path = canonicalize_absolute_path(request.source_path)
    output_path = canonicalize_absolute_path(request.output_base_path)
    roi_ids = tuple(request.included_roi_ids)
    if len(set(roi_ids)) != len(roi_ids):
        raise GuidedIdentityError("Duplicate included ROI IDs are not allowed.")

    source_format = request.source_format
    if source_format not in _SUPPORTED_SOURCE_FORMATS:
        raise GuidedIdentityError(
            f"Unsupported source format: {source_format!r}."
        )

    payload = GuidedCanonicalValidationIdentityPayload(
        identity_schema_name=IDENTITY_SCHEMA_NAME,
        identity_schema_version=IDENTITY_SCHEMA_VERSION,
        canonicalization_algorithm_version=CANONICALIZATION_ALGORITHM_VERSION,
        subset_contract_version=request.subset_contract_version,
        request_schema_version=GUIDED_VALIDATION_REQUEST_SCHEMA_VERSION,
        source_path_canonical=source_path.canonical_path,
        source_path_style=source_path.path_style,
        source_format=source_format.lower(),
        acquisition_mode=request.acquisition_mode,
        sessions_per_hour=request.sessions_per_hour,
        session_duration_sec=request.session_duration_sec,
        exclude_incomplete_final_rwd_chunk=(
            request.exclude_incomplete_final_rwd_chunk
        ),
        timeline_anchor_mode=request.timeline_anchor_mode,
        included_roi_ids=tuple(sorted(roi_ids)),
        execution_mode=request.execution_mode,
        run_profile=request.run_profile,
        traces_only=request.traces_only,
        strategy_scope=request.strategy_scope,
        global_correction_strategy=request.global_correction_strategy,
        dynamic_fit_mode=request.dynamic_fit_mode,
        output_base_path_canonical=output_path.canonical_path,
        output_base_path_style=output_path.path_style,
        output_path_role=request.output_path_role,
        output_creation_timing=request.output_creation_timing,
        run_directory_strategy=request.run_directory_strategy,
        output_overwrite=request.output_overwrite,
    )
    _validate_payload_envelope(payload)
    return payload


def compute_canonical_guided_identity(
    payload: GuidedCanonicalValidationIdentityPayload,
) -> str:
    """Return a future identity foundation, never current Run authorization.

    The digest remains non-authorizing until every deferred identity input is
    resolved and a later backend-validation contract explicitly adopts it.
    """
    payload_bytes = encode_canonical_identity_payload(payload)
    domain = (
        f"{payload.identity_schema_name}:"
        f"{payload.identity_schema_version}:"
        f"{payload.canonicalization_algorithm_version}"
    ).encode("utf-8")
    return hashlib.sha256(domain + b"\x00" + payload_bytes).hexdigest()
