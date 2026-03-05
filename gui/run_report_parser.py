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


def resolve_quick_links(run_dir: str, report_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Resolve quick links to specific subdirectories/files in the run_dir.
    Returns a list of tuples: (Label, Path, Status)
    Status is either "ok" or "missing/invalid (outside run_dir)".
    
    Priority 1: Explicit paths in report_data (e.g., config_effective.yaml)
    Priority 2: Fallback subdirectories.
    """
    run_dir = os.path.realpath(run_dir)
    links = []
    
    def add_link(label: str, rel_path: str):
        # Defense-in-depth: Reject explicit traversal segments in the input string
        # before any normalization occurs.
        normalized_rel = rel_path.replace("\\", "/")
        if ".." in normalized_rel.split("/"):
             links.append((label, rel_path, "missing/invalid (directory traversal rejected)"))
             return

        # Join and normalize to resolve absolute paths or harmless internal dots
        # Use realpath on candidate as well to resolve symlinks
        target_path = os.path.realpath(os.path.join(run_dir, rel_path))
        
        # Enforce run_dir enclosure
        try:
            is_inside = os.path.commonpath([run_dir, target_path]) == run_dir
        except (ValueError, OSError):
            # Different drives or invalid paths
            is_inside = False
            
        if not is_inside:
            links.append((label, target_path, f"missing/invalid (outside run_dir: {target_path})"))
            return
            
        if os.path.exists(target_path):
            links.append((label, target_path, "ok"))
        else:
            links.append((label, target_path, "missing (does not exist)"))
            
    # Priority 1: explicitly enumerated output paths in report_data
    # Determine what explicit keys might exist. Actually the spec says:
    # "Use explicit artifact paths enumerated inside run_report.json if provided."
    # Let's assume common paths like "config_effective.yaml", "gui_run_spec.json", "status.json"
    # Actually, the report might enumeration things under a "deliverables" or "artifacts" key.
    # For now, let's look for known top-level outputs listed in the report.
    artifacts = report_data.get("artifacts", {})
    if isinstance(artifacts, dict):
        for name, rel_path in artifacts.items():
            if isinstance(rel_path, str):
                add_link(f"Artifact: {name}", rel_path)

    # Priority 2: Fallback directories
    fallbacks = [
        ("traces/", "Traces Folder"),
        ("features/", "Features Folder"),
        ("qc_summary/", "QC Summary Folder"),
        ("viz/", "Visualizations Folder")
    ]
    
    for rel_path, label in fallbacks:
        add_link(label, rel_path)
        
    return links
