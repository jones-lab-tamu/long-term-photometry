"""Orchestrate the post-phasic applied-dF/F batch generation for Guided runs."""

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


class GuidedAppliedDffOrchestrationError(ValueError):
    """Raised when applied-dF/F orchestration validation fails."""
    pass


def build_guided_applied_dff_manifest_rows(
    strategy_map_payload: dict,
    applied_dff_root: Path,
) -> list[dict]:
    """
    Build the exact batch manifest rows for run_applied_dff_batch.py.
    
    Enforces the exactly-one-ROI rule before returning any rows.
    Raises GuidedAppliedDffOrchestrationError if validation fails.
    """
    if strategy_map_payload.get("production_strategy_map_version") != "per_roi_correction_strategy_map.v1":
        raise GuidedAppliedDffOrchestrationError("production_strategy_map_version must be exactly per_roi_correction_strategy_map.v1")

    included_roi_ids = set(strategy_map_payload.get("included_roi_ids", []))
    entries = strategy_map_payload.get("per_roi_production_strategy_map", [])

    if not included_roi_ids:
        raise GuidedAppliedDffOrchestrationError("Included ROI set cannot be empty.")

    map_roi_ids = [entry.get("roi_id") for entry in entries]
    
    # Check for missing ROIs
    missing_rois = included_roi_ids - set(map_roi_ids)
    if missing_rois:
        raise GuidedAppliedDffOrchestrationError(f"Missing ROIs in strategy map: {missing_rois}")
        
    # Check for duplicate ROIs
    if len(map_roi_ids) != len(set(map_roi_ids)):
        seen = set()
        duplicates = set(x for x in map_roi_ids if x in seen or seen.add(x))
        raise GuidedAppliedDffOrchestrationError(f"Duplicate ROIs in strategy map: {duplicates}")

    # Check for extra ROIs
    extra_rois = set(map_roi_ids) - included_roi_ids
    if extra_rois:
        raise GuidedAppliedDffOrchestrationError(f"Extra ROIs in strategy map not in included_roi_ids: {extra_rois}")

    rows = []
    output_names = set()
    output_dirs = set()

    from photometry_pipeline.guided_new_analysis_plan import FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES
    SUPPORTED_DYNAMIC_FIT_MODES = FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES

    used_dynamic_fit_modes = set()

    for entry in entries:
        roi_id = entry.get("roi_id")
        strategy_family = entry.get("strategy_family")
        explicit_user_mark = entry.get("explicit_user_mark")
        current_or_stale = entry.get("current_or_stale")
        selected_strategy = entry.get("selected_strategy")
        dynamic_fit_mode = entry.get("dynamic_fit_mode")
        
        if not roi_id:
            raise GuidedAppliedDffOrchestrationError("Strategy map entry missing roi_id.")

        if strategy_family not in ("dynamic_fit", "signal_only_f0"):
            raise GuidedAppliedDffOrchestrationError(f"Unsupported strategy_family: {strategy_family}")

        if not explicit_user_mark:
            raise GuidedAppliedDffOrchestrationError(f"Non-explicit entry for ROI {roi_id}")

        if current_or_stale != "current":
            raise GuidedAppliedDffOrchestrationError(f"Stale entry for ROI {roi_id}")

        if not selected_strategy:
            raise GuidedAppliedDffOrchestrationError(f"Missing selected_strategy for ROI {roi_id}")

        if strategy_family == "dynamic_fit":
            if selected_strategy not in SUPPORTED_DYNAMIC_FIT_MODES:
                raise GuidedAppliedDffOrchestrationError(f"Unsupported dynamic_fit selected_strategy: {selected_strategy}")
            if dynamic_fit_mode not in SUPPORTED_DYNAMIC_FIT_MODES:
                raise GuidedAppliedDffOrchestrationError(f"Unsupported dynamic_fit_mode: {dynamic_fit_mode}")
            if selected_strategy != dynamic_fit_mode:
                raise GuidedAppliedDffOrchestrationError(f"Mismatch between selected_strategy ({selected_strategy}) and dynamic_fit_mode ({dynamic_fit_mode}) for ROI {roi_id}")
            used_dynamic_fit_modes.add(dynamic_fit_mode)
            batch_strategy = "dynamic_fit"

        elif strategy_family == "signal_only_f0":
            if selected_strategy != "signal_only_f0":
                raise GuidedAppliedDffOrchestrationError(f"Unsupported signal_only_f0 selected_strategy: {selected_strategy}")
            if dynamic_fit_mode is not None:
                raise GuidedAppliedDffOrchestrationError(f"signal_only_f0 entry for ROI {roi_id} must have dynamic_fit_mode=None")
            batch_strategy = "signal_only_f0"

        # Sanitize ROI name to ensure valid output name (e.g. replace spaces with underscores)
        safe_roi = str(roi_id).replace(" ", "_").replace("/", "_").replace("\\", "_")
        output_name = f"{safe_roi}_{batch_strategy}"

        if output_name in output_names:
            raise GuidedAppliedDffOrchestrationError(f"Duplicate output_name generated: {output_name}")
        output_names.add(output_name)

        output_dir = (applied_dff_root / output_name).resolve()

        # Prevent escaping applied_dff root
        try:
            output_dir.relative_to(applied_dff_root.resolve())
        except ValueError:
            raise GuidedAppliedDffOrchestrationError(f"output_dir escapes applied_dff root: {output_dir}")

        if str(output_dir) in output_dirs:
            raise GuidedAppliedDffOrchestrationError(f"Duplicate output_dir generated: {output_dir}")
        output_dirs.add(str(output_dir))

        # Only provide the columns that _parse_csv_manifest expects
        row = {
            "roi": roi_id,
            "strategy": batch_strategy,
            "output_name": output_name,
        }
        rows.append(row)

    if len(used_dynamic_fit_modes) > 1:
        raise GuidedAppliedDffOrchestrationError(
            "Mixed dynamic_fit modes cannot be executed by the current "
            "applied-dF/F batch manifest because manifest rows do not carry "
            f"a per-ROI dynamic-fit mode. Found: {used_dynamic_fit_modes}"
        )

    return rows


def run_guided_applied_dff_orchestration_if_enabled(
    run_dir: str,
    phasic_out: str,
    run_cmd_callable=None,
    on_enabled=None,
) -> bool:
    """
    Orchestrates the applied_dff batch execution if enabled.
    Returns True if orchestration ran, False if skipped safely.
    Raises GuidedAppliedDffOrchestrationError if it fails closed.
    """
    run_path = Path(run_dir)
    strategy_map_path = run_path / "guided_correction_strategy_map.json"
    
    if not strategy_map_path.exists():
        return False
        
    try:
        strategy_map_payload = json.loads(strategy_map_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GuidedAppliedDffOrchestrationError("guided_correction_strategy_map.json is malformed") from exc

    if "applied_dff_orchestration_enabled" not in strategy_map_payload:
        raise GuidedAppliedDffOrchestrationError("guided_correction_strategy_map.json is missing applied_dff_orchestration_enabled flag")

    if strategy_map_payload.get("applied_dff_orchestration_enabled") is not True:
        return False
        
    if on_enabled:
        on_enabled()
        
    phasic_cache_path = Path(phasic_out) / "phasic_trace_cache.h5"
    phasic_cache_exists = phasic_cache_path.is_file()
    
    applied_dff_root = run_path / "applied_dff"
    applied_dff_root.mkdir(parents=True, exist_ok=True)
    manifest_csv_path = applied_dff_root / "batch_manifest.csv"
    prov_path = applied_dff_root / "guided_applied_dff_provenance.json"
    
    created_at = datetime.now(timezone.utc).isoformat()
    
    prov_payload = {
        "schema_version": "v1",
        "created_at_utc": created_at,
        "completed_at_utc": None,
        "run_dir": str(run_path),
        "phasic_cache_path": str(phasic_cache_path),
        "phasic_cache_exists_at_start": phasic_cache_exists,
        "phasic_cache_size_bytes": phasic_cache_path.stat().st_size if phasic_cache_exists else None,
        "production_strategy_map_version": strategy_map_payload.get("production_strategy_map_version", ""),
        "requested_strategy_map": strategy_map_payload,
        "batch_manifest_path": str(manifest_csv_path),
        "applied_output_root": str(applied_dff_root),
        "batch_command": [],
        "batch_returncode": None,
        "batch_summary_path": None,
        "orchestration_capability_enabled": True,
        "failure_policy": "fail_whole_guided_run_on_any_applied_failure",
        "overall_status": "running",
        "rows": [],
        "production_analysis": True,
        "preview_only": False,
        "error": None,
    }
    
    try:
        if not phasic_cache_exists:
            raise GuidedAppliedDffOrchestrationError(f"Missing phasic cache file: {phasic_cache_path}")
            
        rows = build_guided_applied_dff_manifest_rows(strategy_map_payload, applied_dff_root)
        
        # Populate provenance rows
        prov_rows = []
        for entry in strategy_map_payload.get("per_roi_production_strategy_map", []):
            safe_roi = str(entry["roi_id"]).replace(" ", "_").replace("/", "_").replace("\\", "_")
            output_name = f"{safe_roi}_{entry['strategy_family']}"
            output_dir = applied_dff_root / output_name
            prov_rows.append({
                "roi_id": entry["roi_id"],
                "batch_roi": entry["roi_id"],
                "strategy_family": entry["strategy_family"],
                "selected_strategy": entry["selected_strategy"],
                "dynamic_fit_mode": entry["dynamic_fit_mode"],
                "batch_strategy": entry["strategy_family"],
                "output_name": output_name,
                "output_dir": str(output_dir),
                "status": "pending",
                "pipeline_summary_path": None,
                "error": None,
            })
        prov_payload["rows"] = prov_rows
        
        with prov_path.open("w", encoding="utf-8") as f:
            json.dump(prov_payload, f, indent=2)
            
        with manifest_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["roi", "strategy", "output_name"])
            writer.writeheader()
            writer.writerows(rows)
            
        script_dir = Path(__file__).resolve().parent
        batch_script_path = script_dir.parent / "tools" / "run_applied_dff_batch.py"

        applied_cmd = [
            sys.executable,
            str(batch_script_path),
            "--phasic-out", str(phasic_out),
            "--manifest", str(manifest_csv_path),
            "--output-root", str(applied_dff_root)
        ]
        prov_payload["batch_command"] = applied_cmd
        with prov_path.open("w", encoding="utf-8") as f:
            json.dump(prov_payload, f, indent=2)
        
        if run_cmd_callable:
            retcode = run_cmd_callable(applied_cmd)
        else:
            proc = subprocess.run(applied_cmd, check=False)
            retcode = proc.returncode

        prov_payload["batch_returncode"] = retcode
        if retcode != 0:
            raise GuidedAppliedDffOrchestrationError(f"Subprocess returned {retcode}")

        summary_path = applied_dff_root / "batch_summary.json"
        if summary_path.exists():
            prov_payload["batch_summary_path"] = str(summary_path)

        prov_payload["overall_status"] = "succeeded"
        for row in prov_payload["rows"]:
            row["status"] = "succeeded"
        prov_payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        with prov_path.open("w", encoding="utf-8") as f:
            json.dump(prov_payload, f, indent=2)
        return True
        
    except Exception as exc:
        prov_payload["overall_status"] = "failed"
        prov_payload["error"] = str(exc)
        for row in prov_payload["rows"]:
            if row["status"] == "pending":
                row["status"] = "failed"
        prov_payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        with prov_path.open("w", encoding="utf-8") as f:
            json.dump(prov_payload, f, indent=2)
        raise GuidedAppliedDffOrchestrationError(f"Orchestration failed: {exc}") from exc
