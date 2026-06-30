"""
run_report_parser.py

Pure functions for parsing run_report.json, determining preview mode,
and resolving quick-links safely within <run_dir>.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

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


def parse_run_report(report_path: str) -> Tuple[Dict[str, Any], str | None]:
    """
    Parse run_report.json and return (data_dict, error_string).
    If error_string is not None, parsing failed or file is missing.
    """
    if not os.path.isfile(report_path):
        return {}, f"File missing at {report_path}"
        
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {}, f"Parse error: {e}"
        
    if not isinstance(data, dict):
        return {}, "Root of run_report.json is not a JSON object"
        
    return data, None


def _read_json_dict(path: str) -> Tuple[Dict[str, Any], str | None]:
    """Read a JSON object from path; return (dict, err)."""
    if not os.path.isfile(path):
        return {}, f"File missing at {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {}, f"Parse error: {e}"
    if not isinstance(data, dict):
        return {}, f"Root of JSON file is not an object: {path}"
    return data, None


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


def is_preflight_or_ineligible(run_dir: str) -> Tuple[bool, str]:
    """
    Determine if run_dir is marked as a non-production preflight or completed-run-ineligible.

    NOTE: The following are reserved loader-side rejection markers only:
    - preflight_marker.json
    - preflight.manifest
    - completed_run_ineligible: true
    - preflight: true

    These markers are used solely to trigger completed-run loader rejection.
    They are not production acceptance evidence and do not define a full preflight schema.
    """
    run_dir = os.path.realpath(run_dir)
    # Check for explicit preflight marker file
    if os.path.isfile(os.path.join(run_dir, "preflight_marker.json")) or os.path.isfile(os.path.join(run_dir, "preflight.manifest")):
        return True, "Directory contains preflight marker file."

    # Check status.json
    status_path = os.path.join(run_dir, "status.json")
    status_data, status_err = _read_json_dict(status_path)
    if status_err is None:
        if status_data.get("completed_run_ineligible") is True or status_data.get("preflight") is True:
            return True, "status.json marks directory as completed-run-ineligible/preflight."

    # Check run_report.json
    report_path = os.path.join(run_dir, "run_report.json")
    report, report_err = parse_run_report(report_path)
    if report_err is None:
        if report.get("completed_run_ineligible") is True or report.get("preflight") is True:
            return True, "run_report.json marks directory as completed-run-ineligible/preflight."
        run_ctx = report.get("run_context", {})
        if isinstance(run_ctx, dict):
            if run_ctx.get("completed_run_ineligible") is True or run_ctx.get("preflight") is True:
                return True, "run_report.json run_context marks directory as completed-run-ineligible/preflight."

    return False, ""


def detect_metadata_conflict(run_dir: str) -> Tuple[bool, str]:
    """
    Check if any recognized metadata file explicitly reports a non-successful,
    failed, aborted, active, or in-progress state.
    """
    run_dir = os.path.realpath(run_dir)

    conflict_statuses = {"failed", "error", "cancelled", "aborted", "in-progress", "active", "running"}
    conflict_phases = {"aborted", "cancelled", "in-progress", "active", "running", "non-final"}

    # 1) Check run_report.json
    report_path = os.path.join(run_dir, "run_report.json")
    report, report_err = parse_run_report(report_path)
    if report_err is None:
        status_vals = [
            str(report.get("status", "")).strip().lower(),
            str(report.get("run_status", "")).strip().lower(),
            str(report.get("final_status", "")).strip().lower(),
            str(report.get("result", "")).strip().lower(),
        ]
        phase_vals = [
            str(report.get("phase", "")).strip().lower(),
            str(report.get("run_phase", "")).strip().lower(),
            str(report.get("final_phase", "")).strip().lower(),
        ]

        for val in status_vals:
            if val in conflict_statuses:
                return True, f"run_report.json reports status={val}"
        for val in phase_vals:
            if val in conflict_phases:
                return True, f"run_report.json reports phase={val}"

        run_ctx = report.get("run_context", {})
        if isinstance(run_ctx, dict):
            ctx_status = str(run_ctx.get("status", "")).strip().lower()
            ctx_phase = str(run_ctx.get("phase", "")).strip().lower()
            if ctx_status in conflict_statuses:
                return True, f"run_report.json run_context reports status={ctx_status}"
            if ctx_phase in conflict_phases:
                return True, f"run_report.json run_context reports phase={ctx_phase}"

    # 2) Check status.json
    status_path = os.path.join(run_dir, "status.json")
    status_data, status_err = _read_json_dict(status_path)
    if status_err is None:
        status_val = str(status_data.get("status", "")).strip().lower()
        phase_val = str(status_data.get("phase", "")).strip().lower()
        if status_val in conflict_statuses:
            return True, f"status.json reports status={status_val}"
        if phase_val in conflict_phases:
            return True, f"status.json reports phase={phase_val}"

    # 3) Check MANIFEST.json
    manifest_path = os.path.join(run_dir, "MANIFEST.json")
    manifest, manifest_err = _read_json_dict(manifest_path)
    if manifest_err is None:
        manifest_status = str(manifest.get("status", "")).strip().lower()
        if manifest_status in conflict_statuses:
            return True, f"MANIFEST.json reports status={manifest_status}"

    return False, ""


def is_successful_completed_run_dir(run_dir: str) -> Tuple[bool, str]:
    """
    Determine whether run_dir represents a completed successful run.

    Evidence precedence:
      1) run_report.json explicitly indicates success/completion
      2) status.json indicates schema_version=1, phase=final, status=success
      3) MANIFEST.json status indicates success/completion
    """
    run_dir = os.path.realpath(run_dir)
    if not os.path.isdir(run_dir):
        return False, f"Directory does not exist: {run_dir}"

    cache_rejection = detect_guided_diagnostic_cache_candidate(run_dir)
    if cache_rejection is not None:
        return False, cache_rejection.as_evidence()

    ineligible, inel_reason = is_preflight_or_ineligible(run_dir)
    if ineligible:
        return False, f"Directory is flagged as non-production/completed-run-ineligible: {inel_reason}"

    has_conflict, conflict_reason = detect_metadata_conflict(run_dir)
    if has_conflict:
        return False, f"Directory contains conflicting metadata: {conflict_reason}"

    # 1) run_report.json explicit success/completion metadata
    report_path = os.path.join(run_dir, "run_report.json")
    report, report_err = parse_run_report(report_path)
    if report_err is None:
        status_tokens = [
            str(report.get("status", "")).strip().lower(),
            str(report.get("run_status", "")).strip().lower(),
            str(report.get("final_status", "")).strip().lower(),
            str(report.get("result", "")).strip().lower(),
        ]
        phase_tokens = [
            str(report.get("phase", "")).strip().lower(),
            str(report.get("run_phase", "")).strip().lower(),
            str(report.get("final_phase", "")).strip().lower(),
        ]
        success_tokens = {"success", "complete", "completed", "done"}
        if any(tok in success_tokens for tok in status_tokens if tok):
            if not any(phase_tokens) or any(tok in {"final", "complete", "completed", "done"} for tok in phase_tokens if tok):
                return True, "run_report.json indicates a successful completed run."

        run_ctx = report.get("run_context", {})
        if isinstance(run_ctx, dict):
            ctx_status = str(run_ctx.get("status", "")).strip().lower()
            ctx_phase = str(run_ctx.get("phase", "")).strip().lower()
            if ctx_status in success_tokens and (not ctx_phase or ctx_phase in {"final", "complete", "completed", "done"}):
                return True, "run_report.json run_context indicates a successful completed run."

    # 2) status.json terminal success contract
    status_path = os.path.join(run_dir, "status.json")
    status_data, status_err = _read_json_dict(status_path)
    if status_err is None:
        schema_ok = (status_data.get("schema_version") == 1)
        phase_ok = str(status_data.get("phase", "")).strip().lower() == "final"
        status_ok = str(status_data.get("status", "")).strip().lower() == "success"
        if schema_ok and phase_ok and status_ok:
            return True, "status.json indicates final success."

    # 3) MANIFEST.json success/completion status
    manifest_path = os.path.join(run_dir, "MANIFEST.json")
    manifest, manifest_err = _read_json_dict(manifest_path)
    if manifest_err is None:
        manifest_status = str(manifest.get("status", "")).strip().lower()
        if manifest_status in {"success", "complete", "completed"}:
            return True, "MANIFEST.json indicates successful completion."

    reasons = []
    if report_err:
        reasons.append(f"run_report.json: {report_err}")
    else:
        reasons.append("run_report.json present but does not explicitly report successful completion.")
    if status_err:
        reasons.append(f"status.json: {status_err}")
    else:
        reasons.append("status.json present but does not match terminal success contract (schema_version=1, phase=final, status=success).")
    if manifest_err:
        reasons.append(f"MANIFEST.json: {manifest_err}")
    else:
        reasons.append("MANIFEST.json present but status is not success/completed.")
    reasons.append("Select a run directory that contains final-success metadata.")
    return False, " | ".join(reasons)


def get_preview_mode(report_data: Dict[str, Any]) -> bool:
    """
    Determine if the run was a preview run based strictly on run_report.json.
    """
    return get_run_type(report_data) == "preview"


def get_run_type(report_data: Dict[str, Any]) -> str:
    """
    Return normalized run_type from run_report context.
    Falls back to 'full' when absent/unknown.
    """
    ctx = report_data.get("run_context", {})
    if not isinstance(ctx, dict):
        return "full"
    raw = str(ctx.get("run_type", "")).strip().lower()
    if raw in {"full", "preview", "tuning_prep"}:
        return raw
    return "full"


def get_summary_fields(report_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Extract explicitly allowlisted generic fields for the Run Summary.
    """
    allowed_keys = ["run_type", "event_signal", "lowpass_hz", "baseline_method"]
    cfg = report_data.get("configuration", {})
    ctx = report_data.get("run_context", {})
    
    # Merge context and configuration maps for easy lookup
    merged = {}
    if isinstance(cfg, dict):
        merged.update(cfg)
    if isinstance(ctx, dict):
        merged.update(ctx)
        
    fields = []
    for k in allowed_keys:
        if k in merged:
            fields.append((k, str(merged[k])))
            
    return fields


def resolve_region_deliverables(run_dir: str) -> List[Dict[str, Any]]:
    """
    Dynamically discover region folders in the run_root.
    A folder is a region if it contains 'summary', 'day_plots', or 'tables'.
    Returns a list of dicts: {'name': str, 'path': str, 'subfolders': List[Tuple[str, str, str]]}
    """
    run_dir = os.path.realpath(run_dir)
    regions = []
    
    if not os.path.isdir(run_dir):
        return []

    # Potential region candidates are subdirectories of the run root
    try:
        candidates = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    except OSError:
        return []

    for d in sorted(candidates):
        if d.startswith(".") or d.startswith("_"):
            continue # Skip internal/hidden
            
        reg_path = os.path.join(run_dir, d)
        
        # Check for semantic subfolders
        subfolders = []
        for sub in ["summary", "day_plots", "tables"]:
            sub_path = os.path.join(reg_path, sub)
            if os.path.isdir(sub_path):
                label = sub.replace("_", " ").title()
                subfolders.append((label, sub_path, "ok"))
        
        if subfolders:
            regions.append({
                "name": d,
                "path": reg_path,
                "subfolders": subfolders
            })
            
    return regions


def resolve_internal_artifacts(run_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find internal/advanced artifacts under _analysis/.
    """
    run_dir = os.path.realpath(run_dir)
    analysis_dir = os.path.join(run_dir, "_analysis")
    links = []
    
    if not os.path.isdir(analysis_dir):
        return []
        
    targets = [
        ("phasic_out", "Phasic Analysis (Internal)"),
        ("tonic_out", "Tonic Analysis (Internal)")
    ]
    
    for rel, label in targets:
        p = os.path.join(analysis_dir, rel)
        if os.path.isdir(p):
            links.append((label, p, "ok"))
            
    return links


def _add_link(run_dir: str, links: List[Tuple[str, str, str]], label: str, rel_path: str):
    """Internal helper to safely resolve a link within run_dir."""
    # Defense-in-depth: Reject explicit traversal segments
    normalized_rel = rel_path.replace("\\", "/")
    if ".." in normalized_rel.split("/"):
         links.append((label, rel_path, "missing/invalid (directory traversal rejected)"))
         return

    # Join and normalize
    target_path = os.path.realpath(os.path.join(run_dir, rel_path))
    
    # Enforce run_dir enclosure
    try:
        is_inside = os.path.commonpath([run_dir, target_path]) == run_dir
    except (ValueError, OSError):
        is_inside = False
        
    if not is_inside:
        links.append((label, target_path, f"missing/invalid (outside run_dir: {target_path})"))
        return
        
    if os.path.exists(target_path):
        links.append((label, target_path, "ok"))
    else:
        links.append((label, target_path, "missing (does not exist)"))


def resolve_primary_artifacts(run_dir: str, report_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Resolve high-level root-level artifacts (config, status, etc).
    Also processes explicit 'artifacts' map from report_data for backward compatibility.
    """
    run_dir = os.path.realpath(run_dir)
    links = []
    
    # Explicitly enumerated artifacts in report
    artifacts = report_data.get("artifacts", {})
    if isinstance(artifacts, dict):
        for name, rel_path in artifacts.items():
            if isinstance(rel_path, str):
                _add_link(run_dir, links, f"Artifact: {name}", rel_path)

    # Standard root-level files
    targets = [
        ("config_effective.yaml", "Effective Config"),
        ("status.json", "Run Status"),
        ("MANIFEST.json", "Output Manifest")
    ]
    
    for rel, label in targets:
        # We only add if it exists or if we want to show it as missing
        # For primary artifacts, we only show if exists or explicitly listed
        p = os.path.join(run_dir, rel)
        if os.path.exists(p):
            _add_link(run_dir, links, label, rel)
            
    return links


def resolve_quick_links(run_dir: str, report_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Backward compatibility wrapper for root-level artifact resolution."""
    return resolve_primary_artifacts(run_dir, report_data)


def classify_completed_run_candidate(run_dir: str) -> Tuple[bool, str]:
    """
    Classify if a run directory satisfies the completed-run contract.
    Combines is_successful_completed_run_dir success metadata checks with
    resolve_region_deliverables region verification.
    """
    ok, evidence = is_successful_completed_run_dir(run_dir)
    if not ok:
        return False, evidence

    regions = resolve_region_deliverables(run_dir)
    if not regions:
        return False, "Completed-run metadata found, but no region deliverables (summary, day_plots, or tables folders) discovered."

    return True, evidence
