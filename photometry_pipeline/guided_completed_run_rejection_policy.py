"""Backend-neutral completed-run rejection policy for Guided diagnostic caches."""

import json
import os
from dataclasses import dataclass
from typing import Any

from photometry_pipeline.guided_diagnostic_cache import (
    DIAGNOSTIC_CACHE_ARTIFACT_FILENAME,
    DIAGNOSTIC_CACHE_PROVENANCE_FILENAME,
    DIAGNOSTIC_CACHE_PURPOSE,
    DIAGNOSTIC_CACHE_SCHEMA_VERSION,
)


GUIDED_DIAGNOSTIC_CACHE_INELIGIBLE = "guided_diagnostic_cache_ineligible"
MALFORMED_GUIDED_DIAGNOSTIC_CACHE_METADATA = (
    "malformed_guided_diagnostic_cache_metadata"
)
AMBIGUOUS_GUIDED_DIAGNOSTIC_CACHE_METADATA = (
    "ambiguous_guided_diagnostic_cache_metadata"
)


@dataclass(frozen=True)
class CompletedRunRejection:
    category: str
    message: str
    detail: str = ""

    def as_evidence(self) -> str:
        evidence = f"{self.category}: {self.message}"
        if self.detail:
            evidence += f" ({self.detail})"
        return evidence


def _diagnostic_cache_rejection(
    category: str,
    *,
    detail: str = "",
) -> CompletedRunRejection:
    if category == GUIDED_DIAGNOSTIC_CACHE_INELIGIBLE:
        message = (
            "This folder is a Guided diagnostic cache, not a completed analysis run."
        )
    else:
        message = (
            "This folder contains Guided diagnostic-cache metadata but it is "
            "malformed, so it cannot be opened as a completed run."
        )
    return CompletedRunRejection(category=category, message=message, detail=detail)


def _read_recognized_cache_metadata(path: str) -> tuple[dict[str, Any] | None, str]:
    if not os.path.isfile(path):
        return None, "recognized metadata path is not a regular file"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return None, f"recognized metadata could not be read: {exc}"
    if not isinstance(payload, dict):
        return None, "recognized metadata root is not a JSON object"
    return payload, ""


def _require_non_empty_text(payload: dict[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        return f"{field_name} must be a non-empty string"
    return None


def _validate_cache_artifact_boundary(
    payload: dict[str, Any],
) -> tuple[str, str]:
    version = payload.get("artifact_contract_version")
    if version != DIAGNOSTIC_CACHE_SCHEMA_VERSION:
        return "malformed", "artifact_contract_version is missing or unsupported"

    purpose_error = _require_non_empty_text(payload, "purpose")
    if purpose_error:
        return "malformed", purpose_error
    if payload["purpose"] != DIAGNOSTIC_CACHE_PURPOSE:
        return "malformed", "purpose does not identify a Guided diagnostic cache"

    production = payload.get("production_analysis")
    if not isinstance(production, bool):
        return "malformed", "production_analysis must be boolean"
    if production:
        return "ambiguous", "Guided diagnostic-cache metadata is marked production"

    for field_name in (
        "cache_id",
        "source_setup_signature",
        "build_request_signature",
        "diagnostic_scope_signature",
    ):
        error = _require_non_empty_text(payload, field_name)
        if error:
            return "malformed", error

    summary = payload.get("session_chunk_inventory_summary")
    if not isinstance(summary, dict):
        return "malformed", "session_chunk_inventory_summary must be a JSON object"

    preliminary = summary.get("preliminary_cache")
    summary_production = summary.get("production_analysis")
    if not isinstance(preliminary, bool):
        return "malformed", "preliminary_cache must be boolean"
    if not isinstance(summary_production, bool):
        return "malformed", "session production_analysis must be boolean"
    if not preliminary or summary_production:
        return "ambiguous", "preliminary and production cache flags contradict"

    return "", ""


def _validate_cache_provenance_boundary(
    payload: dict[str, Any],
) -> tuple[str, str]:
    version = payload.get("schema_version")
    if version != DIAGNOSTIC_CACHE_SCHEMA_VERSION:
        return "malformed", "schema_version is missing or unsupported"

    purpose_error = _require_non_empty_text(payload, "purpose")
    if purpose_error:
        return "malformed", purpose_error
    if payload["purpose"] != DIAGNOSTIC_CACHE_PURPOSE:
        return "malformed", "purpose does not identify a Guided diagnostic cache"

    preliminary = payload.get("preliminary_cache")
    production = payload.get("production_analysis")
    if not isinstance(preliminary, bool):
        return "malformed", "preliminary_cache must be boolean"
    if not isinstance(production, bool):
        return "malformed", "production_analysis must be boolean"
    if not preliminary or production:
        return "ambiguous", "preliminary and production provenance flags contradict"

    if not isinstance(payload.get("build_request"), dict):
        return "malformed", "build_request must be a JSON object"
    nested_artifact = payload.get("artifact")
    if not isinstance(nested_artifact, dict):
        return "malformed", "artifact must be a JSON object"
    root_nested_fields = (
        ("schema version", "schema_version", "artifact_contract_version"),
        ("purpose", "purpose", "purpose"),
        ("production status", "production_analysis", "production_analysis"),
    )
    for label, root_field, nested_field in root_nested_fields:
        if (
            nested_field in nested_artifact
            and payload[root_field] != nested_artifact[nested_field]
        ):
            return "ambiguous", f"provenance root and nested artifact disagree on {label}"
    return _validate_cache_artifact_boundary(nested_artifact)


def _cache_artifact_mismatch(
    artifact: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    nested = provenance["artifact"]
    comparisons = (
        ("artifact_contract_version", artifact.get("artifact_contract_version"), nested.get("artifact_contract_version")),
        ("cache_id", artifact.get("cache_id"), nested.get("cache_id")),
        ("purpose", artifact.get("purpose"), nested.get("purpose")),
        ("production_analysis", artifact.get("production_analysis"), nested.get("production_analysis")),
        ("cache_root_path", artifact.get("cache_root_path"), nested.get("cache_root_path")),
        ("source_setup_signature", artifact.get("source_setup_signature"), nested.get("source_setup_signature")),
        ("build_request_signature", artifact.get("build_request_signature"), nested.get("build_request_signature")),
        ("diagnostic_scope_signature", artifact.get("diagnostic_scope_signature"), nested.get("diagnostic_scope_signature")),
    )
    for field_name, artifact_value, provenance_value in comparisons:
        if artifact_value != provenance_value:
            return f"artifact and provenance disagree on {field_name}"
    if provenance.get("schema_version") != artifact.get("artifact_contract_version"):
        return "artifact and provenance disagree on schema version"
    if provenance.get("purpose") != artifact.get("purpose"):
        return "artifact and provenance disagree on purpose"
    if provenance.get("production_analysis") != artifact.get("production_analysis"):
        return "artifact and provenance disagree on production status"
    return ""


def detect_guided_diagnostic_cache_candidate(
    run_dir: str | os.PathLike[str],
) -> CompletedRunRejection | None:
    """Reject recognized Guided cache roots from completed-run loading."""
    root = os.path.realpath(os.fspath(run_dir))
    artifact_path = os.path.join(root, DIAGNOSTIC_CACHE_ARTIFACT_FILENAME)
    provenance_path = os.path.join(root, DIAGNOSTIC_CACHE_PROVENANCE_FILENAME)
    artifact_exists = os.path.lexists(artifact_path)
    provenance_exists = os.path.lexists(provenance_path)

    if not artifact_exists and not provenance_exists:
        return None
    if artifact_exists != provenance_exists:
        return _diagnostic_cache_rejection(
            AMBIGUOUS_GUIDED_DIAGNOSTIC_CACHE_METADATA,
            detail="recognized diagnostic-cache metadata pair is incomplete",
        )

    artifact, artifact_error = _read_recognized_cache_metadata(artifact_path)
    if artifact_error:
        return _diagnostic_cache_rejection(
            MALFORMED_GUIDED_DIAGNOSTIC_CACHE_METADATA,
            detail=artifact_error,
        )
    provenance, provenance_error = _read_recognized_cache_metadata(provenance_path)
    if provenance_error:
        return _diagnostic_cache_rejection(
            MALFORMED_GUIDED_DIAGNOSTIC_CACHE_METADATA,
            detail=provenance_error,
        )
    assert artifact is not None and provenance is not None

    artifact_kind, artifact_detail = _validate_cache_artifact_boundary(artifact)
    if artifact_kind:
        category = (
            AMBIGUOUS_GUIDED_DIAGNOSTIC_CACHE_METADATA
            if artifact_kind == "ambiguous"
            else MALFORMED_GUIDED_DIAGNOSTIC_CACHE_METADATA
        )
        return _diagnostic_cache_rejection(category, detail=artifact_detail)

    provenance_kind, provenance_detail = _validate_cache_provenance_boundary(
        provenance
    )
    if provenance_kind:
        category = (
            AMBIGUOUS_GUIDED_DIAGNOSTIC_CACHE_METADATA
            if provenance_kind == "ambiguous"
            else MALFORMED_GUIDED_DIAGNOSTIC_CACHE_METADATA
        )
        return _diagnostic_cache_rejection(category, detail=provenance_detail)

    mismatch = _cache_artifact_mismatch(artifact, provenance)
    if mismatch:
        return _diagnostic_cache_rejection(
            AMBIGUOUS_GUIDED_DIAGNOSTIC_CACHE_METADATA,
            detail=mismatch,
        )
    return _diagnostic_cache_rejection(GUIDED_DIAGNOSTIC_CACHE_INELIGIBLE)
