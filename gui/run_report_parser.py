"""
run_report_parser.py

Pure functions for parsing run_report.json, determining preview mode,
and resolving quick-links safely within <run_dir>.
"""

import json
import os
from typing import Dict, Any, List, Tuple


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
    ctx = report_data.get("run_context", {})
    if not isinstance(ctx, dict):
        return False
    return ctx.get("run_type") == "preview"


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
