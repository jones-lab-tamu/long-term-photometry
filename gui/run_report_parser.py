"""
run_report_parser.py

Pure functions for parsing run_report.json, determining preview mode,
and resolving quick-links safely within <run_dir>.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

from photometry_pipeline.guided_completed_run_rejection_policy import (
    AMBIGUOUS_GUIDED_DIAGNOSTIC_CACHE_METADATA,
    CompletedRunRejection,
    GUIDED_DIAGNOSTIC_CACHE_INELIGIBLE,
    MALFORMED_GUIDED_DIAGNOSTIC_CACHE_METADATA,
    detect_guided_diagnostic_cache_candidate,
)
from photometry_pipeline.run_completion_contract import (
    TERMINAL_SUCCESS_LEGACY,
    classify_run_terminal_state,
)


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

    Acceptance is decided by the single terminal-completion contract in
    photometry_pipeline.run_completion_contract: a current run must present one
    coherent, verified terminal set (final success status, mandatory run report,
    final manifest, all mandatory artifacts, matching run identity, verified
    artifact identities). A run from an earlier version of the app is accepted
    only when its historical run report positively identifies it; missing or
    malformed metadata is corrupt, never legacy.
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

    classification = classify_run_terminal_state(run_dir)
    if not classification.is_success:
        return False, classification.reason
    return True, classification.reason


def classify_completed_run_terminal_state(run_dir: str):
    """Expose the full terminal classification (successful / failed / interrupted /
    corrupted / legacy) for callers that must distinguish them."""
    return classify_run_terminal_state(run_dir)


def is_continuous_rwd_run_mode(run_mode: Dict[str, Any]) -> bool:
    """True only for a completed CR1 continuous-RWD run (the accepted
    ``guided_continuous_rwd_{correction,tonic,phasic,combined}`` producers),
    never for the older chunked "continuous acquisition" full-pipeline mode.

    Both share ``acquisition_mode == "continuous"`` and
    ``deliverable_profile == "continuous"`` in the normalized run mode, so
    neither field alone distinguishes them; ``run_profile`` does, since only
    the CR1 continuous-RWD producers use the ``guided_continuous_rwd_``
    prefix (see CR1-E1-B handoff section 4/7 and the completed-run routing
    in ``gui/run_report_viewer.py`` / ``gui/main_window.py``).
    """
    return str(run_mode.get("run_profile", "")).startswith("guided_continuous_rwd_")


def get_scientist_completion_summary(run_dir: str, classification=None) -> str:
    """Return plain-language completion text for the existing Review surface.

    This deliberately reads the shared session-index record only to translate
    approved gaps into scientist-facing terms.  It does not expose internal
    implementation vocabulary in the normal summary.
    """
    if classification is None:
        classification = classify_run_terminal_state(run_dir)
    if not getattr(classification, "completed_with_missing", False):
        if getattr(classification, "is_success", False):
            return "Completed successfully."
        return str(getattr(classification, "reason", "Run could not be loaded."))

    expected = None
    candidates = [
        os.path.join(run_dir, "input_manifest.json"),
        os.path.join(run_dir, "_analysis", "phasic_out", "input_processing_completeness.json"),
        os.path.join(run_dir, "_analysis", "tonic_out", "input_processing_completeness.json"),
    ]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and isinstance(payload.get("expected"), list):
                expected = payload["expected"]
                break
        except (OSError, ValueError, TypeError):
            continue

    missing_count = int(getattr(classification, "missing_session_count", 0))
    exclusion_count = int(getattr(classification, "final_exclusion_count", 0))
    if missing_count and exclusion_count:
        headline = "Completed with missing sessions and an incomplete final session excluded."
    elif exclusion_count:
        headline = "Completed with an incomplete final session excluded."
    else:
        headline = "Completed with missing sessions."
    lines = [headline]
    if missing_count:
        lines.append(
            f"{missing_count} missing session(s) were approved and kept in their original time positions."
        )
    if exclusion_count:
        lines.append(
            f"{exclusion_count} incomplete final session(s) were excluded from analysis."
        )
    affected = []
    for entry in expected or []:
        disposition = str(entry.get("disposition", ""))
        if disposition not in {"authorized_missing_corrupted", "authorized_exclusion"}:
            continue
        number = int(entry.get("index", 0)) + 1
        timestamp = str(entry.get("expected_start_time", "")).strip()
        duration = entry.get("expected_duration_sec")
        reason = str(entry.get("reason", "")).strip()
        label = f"Session {number}"
        if timestamp:
            try:
                label += f" ({datetime.fromisoformat(timestamp).isoformat(sep=' ')})"
            except ValueError:
                label += f" ({timestamp})"
        if duration is not None:
            label += f", expected duration {float(duration):g}s"
        if disposition == "authorized_exclusion":
            reason_text = "final incomplete session excluded"
        else:
            reason_text = reason or "approved missing/corrupted session"
        affected.append(f"{label}: {reason_text}")
    if affected:
        lines.append("Affected sessions:")
        lines.extend(f"• {item}" for item in affected)
    return "\n".join(lines)


def completed_run_verification_is_unavailable(run_dir: str) -> bool:
    """True when the run loads successfully but predates current verification."""
    return classify_run_terminal_state(run_dir).state == TERMINAL_SUCCESS_LEGACY


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
