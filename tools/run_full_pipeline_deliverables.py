#!/usr/bin/env python3
"""
Run Full Pipeline & Package Deliverables
========================================

Orchestrates the full photometry pipeline and packages outputs for delivery.
Enforces deterministic session duration and strict output naming.

Supports two output modes:
  Legacy:  --out <DIR>           (run_dir == DIR, existing behavior)
  GUI:     --out-base <DIR>      (engine creates run_dir = DIR/<run_id>)

Usage:
    python tools/run_full_pipeline_deliverables.py --input <IN> --out <OUT> --config <CFG> --format <FMT>
    python tools/run_full_pipeline_deliverables.py --input <IN> --out-base <BASE> --config <CFG> --format <FMT>
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
import copy
import glob
import re
import pandas as pd
import numpy as np
import time
import secrets
import errno
from datetime import datetime, timezone

try:
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    HAS_PYARROW_CSV = True
except Exception:
    pa = None
    pa_csv = None
    HAS_PYARROW_CSV = False

# Self-contained repo root bootstrap
from pathlib import Path
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from photometry_pipeline.config import Config
    from photometry_pipeline.core.utils import natural_sort_key
    from photometry_pipeline.core.events import EventEmitter
    from photometry_pipeline.core.tonic_output import (
        TONIC_OUTPUT_MODE_PRESERVE_RAW,
        apply_tonic_output_mode_to_session,
        normalize_tonic_output_mode,
    )
    from photometry_pipeline.core.tonic_timeline import (
        TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
        TONIC_TIMELINE_MODE_REAL_ELAPSED,
        build_tonic_chunk_time_axis,
        normalize_tonic_timeline_mode,
        remap_gapfree_axis_to_elapsed_span,
    )
    from photometry_pipeline.io.hdf5_cache_reader import (
        open_tonic_cache, open_phasic_cache, resolve_cache_roi, load_cache_chunk_fields, list_cache_chunk_ids
    )
except ImportError:
    print("ERROR: Could not import photometry_pipeline. Ensure script is in tools/ and repo root is accessible.", flush=True)
    raise SystemExit(1)


# ======================================================================
# Helpers
# ======================================================================

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _extract_cmd_label(cmd):
    """Extract a human-readable label from command list."""
    if not cmd:
        return "unknown"
    # Try to get the script name if it's a python call
    if len(cmd) > 1 and ("python" in cmd[0].lower() or cmd[0].endswith("python.exe")):
        return os.path.basename(cmd[1])
    return os.path.basename(cmd[0])


def run_cmd(cmd, roi_label=None):
    label = _extract_cmd_label(cmd)
    roi_info = f" roi={roi_label}" if roi_label else ""
    started_utc = _utc_now_iso()
    t0 = time.perf_counter()
    
    print(f"TIMING START cmd={label}{roi_info} at {started_utc}", flush=True)
    print(f"Running: {' '.join(cmd)}", flush=True)
    
    try:
        subprocess.check_call(cmd)
        returncode = 0
    except subprocess.CalledProcessError:
        # check_call raises on non-zero, but we want to be explicit if we were using run
        raise

    elapsed = time.perf_counter() - t0
    finished_utc = _utc_now_iso()
    print(f"TIMING DONE cmd={label}{roi_info} elapsed_sec={elapsed:.3f}", flush=True)
    
    return {
        "cmd": cmd,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_sec": elapsed,
        "returncode": returncode
    }


def _log_roi_timing_detail(roi, bucket, elapsed_sec):
    print(f"TIMING DETAIL roi={roi} bucket={bucket} elapsed_sec={elapsed_sec:.3f}", flush=True)


def _accumulate_roi_bucket(roi, bucket_totals, bucket, elapsed_sec):
    bucket_totals[bucket] = bucket_totals.get(bucket, 0.0) + elapsed_sec
    _log_roi_timing_detail(roi, bucket, elapsed_sec)


def _log_roi_timing_metric(roi, name, value):
    print(f"TIMING METRIC roi={roi} name={name} value={value}", flush=True)


def _phase_start(status_data, phase_name):
    """Record phase start in status and print log."""
    now_utc = _utc_now_iso()
    status_data["timing"]["current_phase"] = phase_name
    status_data["timing"]["phase_started_utc"] = now_utc
    print(f"TIMING START phase={phase_name} at {now_utc}", flush=True)
    return time.perf_counter(), now_utc


def _phase_done(status_data, manifest, phase_name, t0, started_utc, status_path=None):
    """Record phase completion in status and manifest, and print log."""
    elapsed = time.perf_counter() - t0
    finished_utc = _utc_now_iso()
    
    # Update status
    status_data["timing"]["last_completed_phase"] = phase_name
    status_data["timing"]["last_phase_elapsed_sec"] = elapsed
    status_data["timing"]["current_phase"] = None
    status_data["timing"]["phase_started_utc"] = None
    
    # Update manifest
    if "timing" not in manifest:
        manifest["timing"] = {"phases": {}}
    manifest["timing"]["phases"][phase_name] = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_sec": elapsed
    }
    
    if status_path:
        _write_status_json(status_path, status_data)
    
    print(f"TIMING DONE phase={phase_name} elapsed_sec={elapsed:.3f}", flush=True)


def _generate_run_id():
    """Generate a deterministic-format run_id: run_YYYYMMDD_HHMMSS_<8hex>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{secrets.token_hex(4)}"


def _is_windows_lock_error(exc):
    """Best-effort detection for transient Windows file-lock contention."""
    if isinstance(exc, PermissionError):
        return True
    if not isinstance(exc, OSError):
        return False
    winerror = getattr(exc, "winerror", None)
    if winerror in (5, 32):  # ERROR_ACCESS_DENIED / ERROR_SHARING_VIOLATION
        return True
    err_no = getattr(exc, "errno", None)
    return err_no in (errno.EACCES, errno.EPERM)


def _atomic_write_json(path, obj, *, replace_retries=0, replace_retry_delay_sec=0.05):
    """Write JSON atomically via tmp + os.replace, with bounded replace retries."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    retries = max(0, int(replace_retries))
    for attempt in range(retries + 1):
        try:
            os.replace(tmp, path)
            return
        except OSError as e:
            is_retryable = _is_windows_lock_error(e)
            if attempt >= retries or not is_retryable:
                raise
            time.sleep(replace_retry_delay_sec)


def _write_status_json(path, obj):
    """
    Status-writer wrapper: keep atomic tmp+replace semantics, but tolerate
    short Windows lock windows from concurrent status polling.
    """
    _atomic_write_json(
        path,
        obj,
        replace_retries=20,          # ~1s bounded retry window at 50ms
        replace_retry_delay_sec=0.05
    )

def _normalize_event_dict(event: dict) -> dict:
    """Ensure event dict has schema_version: 1 (int), overriding any existing key."""
    obj = event.copy()
    obj["schema_version"] = 1
    return obj


# EventEmitter is now imported from photometry_pipeline.core.events

# ======================================================================
# Cancel Flag
# ======================================================================

def should_cancel(cancel_flag_path):
    """Return True if the cancel flag file exists."""
    return cancel_flag_path and os.path.exists(cancel_flag_path)


def check_cancel(cancel_flag_path, emitter, stage, manifest_path, manifest):
    """If cancel flag exists, emit event, write cancelled manifest, exit 130."""
    if not should_cancel(cancel_flag_path):
        return
    emitter.emit(stage, "cancelled", f"Cancel flag detected at stage: {stage}",
                 error_code="CANCELLED")
    manifest_data = manifest.copy()
    _atomic_write_json(manifest_path, manifest_data)
    emitter.close()
    raise SystemExit(130)

def _cleanup_run_outputs_in_place(run_dir, emitter=None):
    """
    Perform in-place cleanup of tool-generated outputs while preserving
    GUI-owned files and the root run directory.
    """
    msg = "overwrite: in-place cleanup of generated outputs"
    if emitter:
        emitter.emit("engine", "info", msg)
    else:
        print(msg, flush=True)

    # Tool-generated artifacts to remove if they exist.
    to_remove = [
        "status.json",
        "run_metadata.json",
        "run_report.json",
        "MANIFEST.json",
        "events.ndjson",
        "features.csv",
        "summary.csv",
        "discovery.json",
        "_analysis",
        "phasic_output",
        "tonic_output"
    ]

    for item in to_remove:
        path = os.path.join(run_dir, item)
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except OSError as e:
                print(f"WARNING: Could not remove directory {path}: {e}", flush=True)
        elif os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                print(f"WARNING: Could not remove file {path}: {e}", flush=True)


RUN_PROFILE_CHOICES = ("full", "tuning_prep")
TUNING_PREP_SKIPPED_PHASES = [
    "analysis.tonic_analysis",
    "plots.tonic_overview",
    "plots.tonic_df_timeseries_table",
    "plots.phasic_time_series_summary",
    "plots.phasic_dayplot_bundle",
]
TUNING_PREP_SKIPPED_OUTPUT_PATTERNS = [
    "_analysis/tonic_out/tonic_trace_cache.h5",
    "summary/tonic_overview.png",
    "tables/tonic_df_timeseries.csv",
    "summary/phasic_peak_rate_timeseries.png",
    "summary/phasic_auc_timeseries.png",
    "tables/phasic_peak_rate_timeseries.csv",
    "tables/phasic_auc_timeseries.csv",
    "day_plots/phasic_sig_iso_day_*.png",
    "day_plots/phasic_dynamic_fit_day_*.png",
    "day_plots/phasic_dFF_day_*.png",
    "day_plots/phasic_stacked_day_*.png",
]
TUNING_PREP_ARTIFACT_CONTRACT = {
    "stage": "stage_2_selective_skip",
    "required_for_completed_load": [
        "status.json (schema_version=1, phase=final, status=success) or equivalent run_report/MANIFEST success evidence",
        "run_report.json",
    ],
    "required_for_post_run_tuning": [
        "_analysis/phasic_out/phasic_trace_cache.h5",
        "_analysis/phasic_out/config_used.yaml",
        "ROI/session entries readable from phasic cache",
    ],
    "workflows_guaranteed": [
        "results workspace load",
        "downstream event-detection tuning",
        "correction retune",
    ],
    "intentionally_skipped_phases": TUNING_PREP_SKIPPED_PHASES,
    "intentionally_skipped_outputs": TUNING_PREP_SKIPPED_OUTPUT_PATTERNS,
    "not_promised": [
        "full production deliverable package (day-plot families and extended per-ROI summaries)",
    ],
}


def _resolve_effective_run_type(*, run_profile: str, preview_first_n) -> str:
    """Resolve externally visible run_type from profile + preview state."""
    if run_profile == "tuning_prep":
        return "tuning_prep"
    return "preview" if preview_first_n is not None else "full"


def _artifact_contract_for_profile(run_profile: str):
    """Return a JSON-safe artifact contract map for the selected run profile."""
    if run_profile == "tuning_prep":
        return copy.deepcopy(TUNING_PREP_ARTIFACT_CONTRACT)
    return None


def _skip_plan_for_profile(run_profile: str, *, run_tonic_mode: bool, run_phasic_mode: bool):
    """Return metadata describing intentionally skipped outputs for a profile."""
    if run_profile != "tuning_prep":
        return None

    skipped_phases = []
    skipped_outputs = []
    if run_tonic_mode:
        skipped_phases.extend([
            "analysis.tonic_analysis",
            "plots.tonic_overview",
            "plots.tonic_df_timeseries_table",
        ])
        skipped_outputs.extend([
            "_analysis/tonic_out/tonic_trace_cache.h5",
            "summary/tonic_overview.png",
            "tables/tonic_df_timeseries.csv",
        ])
    if run_phasic_mode:
        skipped_phases.extend([
            "plots.phasic_time_series_summary",
            "plots.phasic_dayplot_bundle",
        ])
        skipped_outputs.extend([
            "summary/phasic_peak_rate_timeseries.png",
            "summary/phasic_auc_timeseries.png",
            "tables/phasic_peak_rate_timeseries.csv",
            "tables/phasic_auc_timeseries.csv",
            "day_plots/phasic_sig_iso_day_*.png",
            "day_plots/phasic_dynamic_fit_day_*.png",
            "day_plots/phasic_dFF_day_*.png",
            "day_plots/phasic_stacked_day_*.png",
        ])

    return {
        "profile": "tuning_prep",
        "forced_traces_only": True,
        "skipped_phases": skipped_phases,
        "skipped_outputs": skipped_outputs,
    }


def _ensure_root_run_report(
    run_dir,
    phasic_out,
    tonic_out,
    emitter,
    run_type=None,
    run_profile=None,
    artifact_contract=None,
    intentional_skips=None,
    sessions_per_hour=None,
    sessions_per_hour_source=None,
    timeline_anchor_mode="civil",
    fixed_daily_anchor_clock=None,
):
    """
    Ensure <run_dir>/run_report.json exists at root before terminal status.
    Ordered requirement for Step 8: Strict Ordering Gate.
    Returns: bool (True if root run_report.json exists after check/copy).
    """
    if not run_dir or not os.path.isdir(run_dir):
        return False
        
    report_path = os.path.join(run_dir, "run_report.json")
    
    # Try to find a source if not at root
    if not os.path.exists(report_path):
        search_paths = []
        if phasic_out:
            search_paths.append(os.path.join(phasic_out, "run_report.json"))
        if tonic_out:
            search_paths.append(os.path.join(tonic_out, "run_report.json"))
        
        for src in search_paths:
            if os.path.exists(src):
                try:
                    shutil.copy2(src, report_path)
                    if emitter:
                        emitter.emit("package", "done", f"Hardened copy: {os.path.basename(src)} -> root")
                    break
                except Exception as e:
                    if emitter:
                        emitter.emit("package", "error", f"Failed to copy report to root: {e}")

    # STAMPING: Authoritative Audit Stamp
    if os.path.exists(report_path):
        try:
             with open(report_path, 'r') as f:
                 repo = json.load(f)
             
             # Inject authoritative metadata into run_context and derived_settings
             run_ctx = repo.setdefault("run_context", {})
             if isinstance(run_ctx, dict):
                 run_ctx['sessions_per_hour'] = sessions_per_hour
                 run_ctx['sessions_per_hour_source'] = sessions_per_hour_source
                 run_ctx['timeline_anchor_mode'] = timeline_anchor_mode
                 run_ctx['fixed_daily_anchor_clock'] = fixed_daily_anchor_clock
                 if run_type:
                     run_ctx["run_type"] = str(run_type)
                 if run_profile:
                     run_ctx["run_profile"] = str(run_profile)
                 if artifact_contract is not None:
                     run_ctx["artifact_contract"] = artifact_contract
                 if intentional_skips is not None:
                     run_ctx["intentional_skips"] = intentional_skips
             
             derived_settings = repo.setdefault("derived_settings", {})
             if isinstance(derived_settings, dict):
                 derived_settings['sessions_per_hour'] = sessions_per_hour
                 derived_settings['sessions_per_hour_source'] = sessions_per_hour_source
                 derived_settings['timeline_anchor_mode'] = timeline_anchor_mode
                 derived_settings['fixed_daily_anchor_clock'] = fixed_daily_anchor_clock

             if artifact_contract is not None:
                 repo["run_mode_contract"] = {
                     "run_profile": str(run_profile or ""),
                     "run_type": str(run_type or ""),
                     "artifact_contract": artifact_contract,
                 }
             if intentional_skips is not None:
                 repo.setdefault("run_mode_contract", {})
                 if isinstance(repo["run_mode_contract"], dict):
                     repo["run_mode_contract"]["intentional_skips"] = intentional_skips
                 
             with open(report_path, 'w') as f:
                 json.dump(repo, f, indent=2)
             
             if emitter and sessions_per_hour_source:
                 emitter.emit(
                     "package",
                     "audit",
                     "Stamped report with authoritative run context "
                     f"(run_type={run_type}, run_profile={run_profile}, sessions_per_hour_source={sessions_per_hour_source})",
                 )
        except Exception as e:
             if emitter:
                 emitter.emit("package", "error", f"Failed to stamp report: {e}")
                 
    if not os.path.exists(report_path) and emitter:
        emitter.emit("package", "notice", "run_report.json sourcing skipped (missing artifacts)")
    
    return os.path.exists(report_path)

# ======================================================================
# Arg Parsing
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    # Output mode: --out (legacy) OR --out-base (GUI mode)
    parser.add_argument('--out', default=None,
                        help="Legacy: explicit run directory path")
    parser.add_argument('--out-base', default=None,
                        help="GUI mode: base directory; engine creates <out-base>/<run-id>")
    parser.add_argument('--run-id', default=None,
                        help="Optional run ID for --out-base mode (no path separators)")
    parser.add_argument('--config', required=True)
    parser.add_argument('--format', required=True, choices=['rwd', 'npm', 'auto'])
    parser.add_argument('--mode', choices=['both', 'tonic', 'phasic'], default='both', help="Analysis mode (both, tonic, or phasic)")
    parser.add_argument(
        '--run-type',
        choices=list(RUN_PROFILE_CHOICES),
        default='full',
        help=(
            "Formal run profile. 'full' is standard production packaging. "
            "'tuning_prep' guarantees artifacts required for results inspection + post-run tuning workflows."
        ),
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--include-rois', type=str, default=None, help="Comma-separated list of ROIs to process exclusively")
    parser.add_argument('--exclude-rois', type=str, default=None, help="Comma-separated list of ROIs to ignore")
    parser.add_argument('--traces-only', action='store_true', help="Run traces and QC, skip feature extraction (features.csv) and feature-dependent summaries.")
    parser.add_argument('--sessions-per-hour', type=int, help="Force sessions per hour (integer)")
    parser.add_argument('--session-duration-s', type=float, help="Recording duration in seconds (data length per chunk). If provided, validated against traces.")
    parser.add_argument(
        '--timeline-anchor-mode',
        choices=['civil', 'elapsed', 'fixed_daily_anchor'],
        default='civil',
        help="Anchor semantics for phasic/dayplot hour/day placement."
    )
    parser.add_argument(
        '--fixed-daily-anchor-clock',
        default=None,
        help="Anchor clock for fixed_daily_anchor mode (HH:MM or HH:MM:SS)."
    )
    parser.add_argument('--smooth-window-s', type=float, default=1.0)
    parser.add_argument('--sig-iso-render-mode', choices=['qc', 'full'], default='qc',
                        help="Render mode for sig/iso day plots (qc or full).")
    parser.add_argument('--dff-render-mode', choices=['qc', 'full'], default='qc',
                        help="Render mode for dFF day plots (qc or full).")
    parser.add_argument('--stacked-render-mode', choices=['qc', 'full'], default='qc',
                        help="Render mode for stacked day plots (qc or full).")
    parser.add_argument('--event-signal', type=str, choices=['dff', 'delta_f'], help="Signal to use for peak/event detection")
    parser.add_argument('--representative-session-index', type=int, default=None, help="Force a specific session index for representative artifacts (0-based)")
    parser.add_argument('--preview-first-n', type=int, default=None, help="Preview mode: process only the first N discovered sessions (after discovery/sort).")
    parser.add_argument('--validate-only', action='store_true',
                        help=(
                            "Validate inputs and exit without analysis. "
                            "Legacy --out: no run_dir, no directories, no MANIFEST. "
                            "GUI --out-base: may create run_dir for auto paths."
                        ))
    parser.add_argument('--events', default='auto',
                        help=(
                            "Events NDJSON path, or 'auto' (run_dir/events.ndjson). "
                            "In legacy validate-only: 'auto' disables events "
                            "(run_dir not created); explicit path writes only if "
                            "parent directory already exists."
                        ))
    parser.add_argument('--discover', action='store_true',
                        help=(
                            "Discovery mode: emit JSON with discovered sessions and ROIs. "
                            "Exits 0 without creating files or running analysis."
                        ))
    parser.add_argument('--cancel-flag', default='auto',
                        help="Cancel flag path, or 'auto' for run_dir/CANCEL.REQUESTED")
    return parser.parse_args()

# ======================================================================
# Input Validation
# ======================================================================

def validate_inputs(args):
    """Cheap preflight checks. Raises RuntimeError on any problem."""
    # Input path
    if not os.path.isdir(args.input):
        raise RuntimeError(f"Input directory does not exist or is not a directory: {args.input}")

    # Config path
    if not os.path.isfile(args.config):
        raise RuntimeError(f"Config file does not exist or is not a file: {args.config}")

    # Format (already constrained by argparse choices, but belt-and-suspenders)
    if args.format not in ('rwd', 'npm', 'auto'):
        raise RuntimeError(f"Invalid format: {args.format}")

    run_profile = str(getattr(args, "run_type", "full") or "full").strip().lower()
    if run_profile == "tuning_prep" and getattr(args, "mode", "both") == "tonic":
        raise RuntimeError(
            "--run-type tuning_prep requires phasic-capable mode ('both' or 'phasic') "
            "to preserve downstream/correction tuning artifacts."
        )

    # sessions_per_hour
    if args.sessions_per_hour is not None:
        if args.sessions_per_hour < 1:
            raise RuntimeError(f"--sessions-per-hour must be >= 1, got {args.sessions_per_hour}")

    # session_duration_s
    if args.session_duration_s is not None:
        if args.session_duration_s <= 0:
            raise RuntimeError(f"--session-duration-s must be > 0, got {args.session_duration_s}")

    if args.timeline_anchor_mode == "fixed_daily_anchor":
        if args.fixed_daily_anchor_clock is None or not str(args.fixed_daily_anchor_clock).strip():
            raise RuntimeError(
                "--fixed-daily-anchor-clock is required when --timeline-anchor-mode fixed_daily_anchor."
            )

    # Impossible schedule (only when both are provided)
    if args.sessions_per_hour is not None and args.session_duration_s is not None:
        stride_s = 3600.0 / args.sessions_per_hour
        if args.session_duration_s > stride_s + 1e-6:
            raise RuntimeError(
                f"Impossible schedule: Duration {args.session_duration_s:.2f}s > "
                f"Stride {stride_s:.2f}s (SPH={args.sessions_per_hour}).")

# ======================================================================
# Run-dir Resolution
# ======================================================================

def resolve_run_dir(args):
    """Resolve run_dir and run_id from args. Returns (run_dir, run_id, is_gui_mode)."""
    if args.out and args.out_base:
        print("Error: --out and --out-base are mutually exclusive.", file=sys.stderr)
        raise SystemExit(2)
    if not args.out and not args.out_base:
        print("Error: one of --out or --out-base is required.", file=sys.stderr)
        raise SystemExit(2)
    if args.run_id and not args.out_base:
        print("Error: --run-id requires --out-base.", file=sys.stderr)
        raise SystemExit(2)

    if args.out_base:
        # GUI mode
        run_id = args.run_id if args.run_id else _generate_run_id()
        # Validate run_id: no path separators
        if os.sep in run_id or (os.altsep and os.altsep in run_id):
            print(f"Error: --run-id must not contain path separators, got: {run_id}",
                  file=sys.stderr)
            raise SystemExit(2)
        run_dir = os.path.abspath(os.path.join(args.out_base, run_id))
        return run_dir, run_id, True
    else:
        # Legacy mode
        run_dir = os.path.abspath(args.out)
        run_id = os.path.basename(run_dir)
        return run_dir, run_id, False


def resolve_paths(args, run_dir):
    """Resolve events_path and cancel_flag_path."""
    if args.events == "auto":
        events_path = os.path.join(run_dir, "events.ndjson")
    else:
        events_path = os.path.abspath(args.events)

    if args.cancel_flag == "auto":
        cancel_flag_path = os.path.join(run_dir, "CANCEL.REQUESTED")
    else:
        cancel_flag_path = os.path.abspath(args.cancel_flag)

    return events_path, cancel_flag_path

# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    
    # Timing: Authoritative Total Runtime
    t0_total = time.perf_counter()
    started_utc_total = _utc_now_iso()

    # Initialize variables for safe use in except blocks (Strict Ordering Gate)
    phasic_out = None
    tonic_out = None
    emitter = None

    # ============================================================
    # Discovery Preflight
    # ============================================================
    if args.discover:
        # We perform discovery and print JSON to stdout, then exit 0.
        # No run_dir is created, no events emitted.
        from photometry_pipeline.discovery import discover_inputs
        try:
            cfg = Config.from_yaml(args.config)
            # Apply any CLI overrides that discovery might care about
            if args.preview_first_n is not None:
                cfg.preview_first_n = args.preview_first_n
                
            res = discover_inputs(
                input_dir=args.input,
                config=cfg,
                force_format=args.format,
                preview_first_n=cfg.preview_first_n
            )
            print(json.dumps(res, indent=2))
            raise SystemExit(0)
        except Exception as e:
            # Output error strictly to stderr so stdout JSON remains clean if expected
            print(f"Error during discovery: {e}", file=sys.stderr)
            raise SystemExit(1)

    # -- Resolve run directory --
    run_dir, run_id, is_gui_mode = resolve_run_dir(args)
    events_path, cancel_flag_path = resolve_paths(args, run_dir)
    manifest_path = os.path.join(run_dir, "MANIFEST.json")

    # ============================================================
    # 2. Resolve Effective Params (sessions_per_hour, event_signal, etc.)
    # ============================================================
    # Resolve early so it is available for status.json in both validate-only and full run.

    # sessions_per_hour resolution (source-auditable)
    resolved_sessions_per_hour = None
    sessions_per_hour_source = None

    if args.sessions_per_hour is not None:
        resolved_sessions_per_hour = args.sessions_per_hour
        sessions_per_hour_source = "user-provided"
    else:
        try:
            cfg = Config.from_yaml(args.config)
            config_sph = getattr(cfg, "sessions_per_hour", None)
            if config_sph is not None:
                resolved_sessions_per_hour = config_sph
                sessions_per_hour_source = "config"
        except Exception:
            pass

    # Log the source clearly to stdout (auditable line)
    if resolved_sessions_per_hour is not None:
        print(f"Using sessions_per_hour={resolved_sessions_per_hour} ({sessions_per_hour_source})", flush=True)
    else:
        print("Using sessions_per_hour=None (no source found)", flush=True)

    # Determine effective event signal, excursion polarity, representative index, and preview for stamping
    effective_event_signal = args.event_signal
    effective_signal_excursion_polarity = "positive"
    effective_representative_index = args.representative_session_index
    effective_preview_first_n = args.preview_first_n
    effective_tonic_output_mode = TONIC_OUTPUT_MODE_PRESERVE_RAW
    effective_tonic_timeline_mode = TONIC_TIMELINE_MODE_REAL_ELAPSED
    
    if (
        effective_event_signal is None
        or effective_representative_index is None
        or effective_preview_first_n is None
        or effective_tonic_output_mode == TONIC_OUTPUT_MODE_PRESERVE_RAW
        or effective_tonic_timeline_mode == TONIC_TIMELINE_MODE_REAL_ELAPSED
    ):
        try:
            cfg = Config.from_yaml(args.config)
            if effective_event_signal is None:
                effective_event_signal = getattr(cfg, "event_signal", "dff")
            effective_signal_excursion_polarity = str(
                getattr(cfg, "signal_excursion_polarity", "positive")
            )
            if effective_representative_index is None:
                effective_representative_index = getattr(cfg, "representative_session_index", None)
            if effective_preview_first_n is None:
                effective_preview_first_n = getattr(cfg, "preview_first_n", None)
            effective_tonic_output_mode = normalize_tonic_output_mode(
                getattr(cfg, "tonic_output_mode", TONIC_OUTPUT_MODE_PRESERVE_RAW)
            )
            effective_tonic_timeline_mode = normalize_tonic_timeline_mode(
                getattr(cfg, "tonic_timeline_mode", TONIC_TIMELINE_MODE_REAL_ELAPSED)
            )
        except Exception as e:
            print(f"WARNING: Failed to parse config for runner stamping: {e}", flush=True)
            if effective_event_signal is None: 
                effective_event_signal = "dff"
            effective_signal_excursion_polarity = "positive"
            # others remain as given or None
            effective_tonic_output_mode = TONIC_OUTPUT_MODE_PRESERVE_RAW
            effective_tonic_timeline_mode = TONIC_TIMELINE_MODE_REAL_ELAPSED

    print(
        f"Using tonic_output_mode={effective_tonic_output_mode}",
        flush=True,
    )
    print(
        f"Using tonic_timeline_mode={effective_tonic_timeline_mode}",
        flush=True,
    )
    effective_run_type = _resolve_effective_run_type(
        run_profile=args.run_type,
        preview_first_n=effective_preview_first_n,
    )
    effective_artifact_contract = _artifact_contract_for_profile(args.run_type)
    tune_prep_light_mode = bool(args.run_type == "tuning_prep")
    selected_tonic_mode = args.mode in ('both', 'tonic')
    run_tonic_mode = selected_tonic_mode and not tune_prep_light_mode
    run_phasic_mode = args.mode in ('both', 'phasic')
    effective_traces_only = bool(args.traces_only or args.run_type == "tuning_prep")
    effective_skip_plan = _skip_plan_for_profile(
        args.run_type,
        run_tonic_mode=selected_tonic_mode,
        run_phasic_mode=run_phasic_mode,
    )
    print(
        f"Using run_profile={args.run_type} (effective run_type={effective_run_type})",
        flush=True,
    )
    if effective_traces_only and not args.traces_only and args.run_type == "tuning_prep":
        print(
            "Tuning-prep profile: forcing traces-only analysis for lighter execution.",
            flush=True,
        )
    if tune_prep_light_mode and selected_tonic_mode:
        print(
            "Tuning-prep profile: skipping tonic analysis/cache generation by contract.",
            flush=True,
        )

    print(f"RUN_DIR: {run_dir}", flush=True)

    # -- Build base manifest dict --
    manifest = {
        'tool': 'run_full_pipeline_deliverables',
        'timestamp': datetime.now().isoformat(),
        'run_id': run_id,
        'run_profile': args.run_type,
        'run_type': effective_run_type,
        'run_dir': run_dir,
        'events_path': events_path,
        'cancel_flag_path': cancel_flag_path,
        'args': vars(args),
        'timeline_anchor_mode': args.timeline_anchor_mode,
        'fixed_daily_anchor_clock': args.fixed_daily_anchor_clock,
        'intentional_skips': effective_skip_plan,
        'commands': [],
        'regions': [],
        'deliverables': {}
    }

    # ============================================================
    # 0. Validate-only preflight
    # ============================================================
    if args.validate_only:
        # Side-effect policy:
        #   --out-base mode: create run_dir to support auto events/cancel paths.
        #   Legacy --out mode: do NOT create run_dir. Disable events if
        #     --events is "auto" (would require creating run_dir).
        vo_events_path = events_path
        vo_allow_makedirs = True
        
        # Enforce Step 8 contract: validate-only MUST create run_dir to house status.json
        os.makedirs(run_dir, exist_ok=True)

        if not is_gui_mode:
            # Legacy mode: specifically silence 'auto' events to prevent clutter
            # if the user wasn't expecting directory side effects beyond status.json.
            if args.events == "auto":
                vo_events_path = None
                print("VALIDATE-ONLY: events disabled (legacy --out mode, "
                      "--events auto); to enable, pass --events <PATH>", flush=True)
            else:
                # Explicit path: check parent directory
                parent = os.path.dirname(events_path) or "."
                if not os.path.isdir(parent):
                    vo_events_path = None
                    print("VALIDATE-ONLY: events disabled (legacy --out mode); "
                          "parent directory does not exist and will not be "
                          "created because legacy validate-only creates "
                          "no directories", flush=True)
            vo_allow_makedirs = False

        emitter = EventEmitter(vo_events_path, run_id, run_dir,
                               allow_makedirs=vo_allow_makedirs)
        emitter.emit("engine", "start", "Engine starting (validate-only)")
        emitter.emit("validate", "start", "Validating inputs")

        # --- Helpers for GUI validate-only status.json ---
        vo_t0 = time.time()
        vo_created_utc = datetime.now(timezone.utc).isoformat()
        vo_out_base = os.path.abspath(args.out_base) if args.out_base else None

        def _vo_status_template():
            """Return the base status dict for validate-only mode."""
            return {
                "schema_version": 1,
                "run_type": "validate_only",
                "run_profile": args.run_type,
                "run_id": run_id,
                "phase": "running",
                "status": "running",
                "created_utc": vo_created_utc,
                "finished_utc": None,
                "duration_sec": 0.0,
                "input_dir": os.path.abspath(args.input),
                "out_base": vo_out_base,
                "run_root": run_dir,
                "output_package": run_dir,
                "command": " ".join(sys.argv),
                "config_path": os.path.abspath(args.config),
                "format": args.format,
                "sessions_per_hour": resolved_sessions_per_hour,
                "sessions_per_hour_source": sessions_per_hour_source,
                "timeline_anchor_mode": args.timeline_anchor_mode,
                "fixed_daily_anchor_clock": args.fixed_daily_anchor_clock,
                "events_mode": args.events,
                "outputs": {
                    "manifest_json": None,
                    "events_ndjson": vo_events_path,
                    "region_dirs": []
                },
                "errors": [],
                "warnings": [],
                "validation": {"result": "validate_only"},
                "artifact_contract": effective_artifact_contract,
                "intentional_skips": effective_skip_plan,
                "traces_only": effective_traces_only,
            }

        def _vo_write_final_status(status, error_msg=None):
            """Finalize and write status.json for GUI validate-only."""
            sd = _vo_status_template()
            sd["phase"] = "final"
            sd["status"] = status
            sd["finished_utc"] = datetime.now(timezone.utc).isoformat()
            sd["duration_sec"] = time.time() - vo_t0
            if error_msg:
                sd["errors"].append(str(error_msg))
            vo_status_path = os.path.join(run_dir, "status.json")
            _write_status_json(vo_status_path, sd)

        try:
            validate_inputs(args)
        except RuntimeError as e:
            emitter.emit("validate", "error", str(e), error_code="VALIDATION_FAILED")
            emitter.close()
            # Always emit status.json even in legacy mode for GUI/automated observers
            _vo_write_final_status("error", error_msg=str(e))
            print(str(e), file=sys.stderr)
            raise SystemExit(1)

        emitter.emit("validate", "done", "Validation passed")

        # Build the argv that a real run would use
        argv = [sys.executable, "tools/run_full_pipeline_deliverables.py",
                "--input", args.input]
        if args.out:
            argv.extend(["--out", args.out])
            if args.overwrite:
                argv.append("--overwrite")
        elif args.out_base:
            argv.extend(["--out-base", args.out_base])
            if args.run_id:
                argv.extend(["--run-id", args.run_id])
        argv.extend(["--config", args.config, "--format", args.format])
        if args.include_rois:
            argv.extend(["--include-rois", args.include_rois])
        if args.exclude_rois:
            argv.extend(["--exclude-rois", args.exclude_rois])
        if resolved_sessions_per_hour is not None:
            argv.extend(["--sessions-per-hour", str(resolved_sessions_per_hour)])
        if args.session_duration_s is not None:
            argv.extend(["--session-duration-s", str(args.session_duration_s)])
        if args.timeline_anchor_mode != "civil":
            argv.extend(["--timeline-anchor-mode", str(args.timeline_anchor_mode)])
        if args.timeline_anchor_mode == "fixed_daily_anchor" and args.fixed_daily_anchor_clock:
            argv.extend(["--fixed-daily-anchor-clock", str(args.fixed_daily_anchor_clock)])
        argv.extend(["--smooth-window-s", str(args.smooth_window_s)])
        argv.extend(["--sig-iso-render-mode", str(args.sig_iso_render_mode)])
        argv.extend(["--dff-render-mode", str(args.dff_render_mode)])
        argv.extend(["--stacked-render-mode", str(args.stacked_render_mode)])
        if effective_traces_only:
            argv.append("--traces-only")
        if args.run_type != "full":
            argv.extend(["--run-type", str(args.run_type)])

        print("VALIDATE-ONLY: OK", flush=True)
        print(f"VALIDATE-ONLY: run_dir={run_dir}", flush=True)
        print(f"VALIDATE-ONLY: events_path={vo_events_path}", flush=True)
        print(f"VALIDATE-ONLY: cancel_flag_path={cancel_flag_path}", flush=True)
        print(f"VALIDATE-ONLY: argv={json.dumps(argv)}", flush=True)

        emitter.emit("engine", "done", "Validate-only complete")
        emitter.close()
        # Always emit status.json even in legacy mode for GUI/automated observers
        _vo_write_final_status("success")
        raise SystemExit(0)

    # ============================================================
    # 1. Setup run directory
    # ============================================================
    if is_gui_mode:
        # GUI mode: --overwrite is meaningless (each run gets a unique run_id)
        if args.overwrite:
            print("WARNING: --overwrite is ignored in --out-base mode")
    else:
        # Legacy mode: handle overwrite
        if os.path.exists(run_dir):
            if not args.overwrite:
                print(f"Error: Output directory {run_dir} exists. Use --overwrite.")
                raise SystemExit(1)

            # In-Place Overwrite (Fix B1v5)
            # We do NOT delete the root run_dir because it contains GUI-owned files
            # and open log handles.
            _cleanup_run_outputs_in_place(run_dir, emitter=None)

    os.makedirs(run_dir, exist_ok=True)

    # ============================================================
    # 2. Status & Emitter Setup
    # ============================================================

    # -- Status tracking (strict contract) --
    # Phase: "running" (no status field) -> "final" (status="success"|"error")
    status_data = {
        "schema_version": 1,
        "run_type": "full",
        "run_profile": args.run_type,
        "run_id": run_id,
        "phase": "running", 
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": None,
        "duration_sec": 0.0,
        "input_dir": os.path.abspath(args.input),
        "out_base": os.path.abspath(args.out_base) if args.out_base else os.path.abspath(os.path.dirname(run_dir)),
        "run_root": run_dir,
        "output_package": run_dir,
        "command": " ".join(sys.argv),
        "config_path": os.path.abspath(args.config),
        "format": args.format,
        "sessions_per_hour": resolved_sessions_per_hour,
        "sessions_per_hour_source": sessions_per_hour_source,
        "timeline_anchor_mode": args.timeline_anchor_mode,
        "fixed_daily_anchor_clock": args.fixed_daily_anchor_clock,
        "events_mode": args.events,
        "outputs": {
            "manifest_json": manifest_path,
            "events_ndjson": events_path,
            "region_dirs": []
        },
        "errors": [],
        "warnings": [],
        "validation": None,
        "artifact_contract": effective_artifact_contract,
        "intentional_skips": effective_skip_plan,
        "features_extracted": False if effective_traces_only else None,
        "traces_only": effective_traces_only,
        "timing": {
            "current_phase": None,
            "phase_started_utc": None,
            "last_completed_phase": None,
            "last_phase_elapsed_sec": None
        }
    }
    t0_status = time.time()
    status_path = os.path.join(run_dir, "status.json")

    def _finalize_status(state="success", error_msg=None):
        """Update and write status.json atomically with final state."""
        status_data["phase"] = "final"
        status_data["finished_utc"] = datetime.now(timezone.utc).isoformat()
        status_data["duration_sec"] = time.time() - t0_status
        
        if error_msg:
            status_data["errors"].append(str(error_msg))
            
        # Discover region dirs (any child dir except _analysis and hidden)
        if os.path.isdir(run_dir):
            try:
                subs = []
                for d in os.listdir(run_dir):
                    p = os.path.join(run_dir, d)
                    if os.path.isdir(p) and d not in ("_analysis",) and not d.startswith("."):
                        subs.append(p)
                status_data["outputs"]["region_dirs"] = sorted(subs)
            except OSError:
                pass

        status_data["status"] = state
        _write_status_json(status_path, status_data)

    def _write_status_update(phase):
        """Perform a non-terminal status update with current duration."""
        status_data["phase"] = phase
        status_data["status"] = "running"
        status_data["duration_sec"] = time.time() - t0_status
        _write_status_json(status_path, status_data)

    # Update status_data with resolved preview info before initial write
    status_data["run_type"] = effective_run_type
    status_data["preview"] = {"selector": "first_n", "first_n": effective_preview_first_n} if effective_preview_first_n is not None else None

    # Initial write (phase="running")
    _write_status_json(status_path, status_data)
    _write_status_update("initializing")

    # -- Open event emitter --
    emitter = EventEmitter(events_path, run_id, run_dir, file_mode="w")
    emitter.emit("engine", "start", "Engine starting")
    emitter.emit("engine", "context", "Run context initialized", payload={
        "run_type": effective_run_type,
        "run_profile": args.run_type,
        "artifact_contract": effective_artifact_contract,
        "intentional_skips": effective_skip_plan,
        "features_extracted": False if effective_traces_only else None, 
        "preview": {"selector": "first_n", "first_n": effective_preview_first_n} if effective_preview_first_n is not None else None, 
        "traces_only": effective_traces_only, 
        "event_signal": effective_event_signal,
        "signal_excursion_polarity": effective_signal_excursion_polarity,
        "representative_session_index": effective_representative_index,
        "sessions_per_hour": resolved_sessions_per_hour,
        "sessions_per_hour_source": sessions_per_hour_source,
        "timeline_anchor_mode": args.timeline_anchor_mode,
        "fixed_daily_anchor_clock": args.fixed_daily_anchor_clock,
    })
    substantive_work_completed = False

    try:
        # -- Check cancel immediately --
        check_cancel(cancel_flag_path, emitter, "engine_start", manifest_path, manifest)
    
        analysis_dir = os.path.join(run_dir, '_analysis')
        tonic_out = os.path.join(analysis_dir, 'tonic_out')
        phasic_out = os.path.join(analysis_dir, 'phasic_out')
        if tune_prep_light_mode and selected_tonic_mode:
            emitter.emit(
                "tonic",
                "audit",
                "Tuning-prep contract: tonic analysis/cache generation intentionally skipped.",
                payload={"skipped_output": "_analysis/tonic_out/tonic_trace_cache.h5"},
            )

        # ============================================================
        # 3. Validate
        # ============================================================
        t_phase, started_utc_phase = _phase_start(status_data, "validate")
        _write_status_update("validating")
        emitter.emit("validate", "start", "Validating inputs")
        validate_inputs(args)
        emitter.emit("validate", "done", "Validation passed")
        _phase_done(status_data, manifest, "validate", t_phase, started_utc_phase, status_path=status_path)

        check_cancel(cancel_flag_path, emitter, "validate", manifest_path, manifest)

        # ============================================================
        # 4. Tonic Analysis
        # ============================================================
        # Locate analyze_photometry.py relative to this script
        # This script is in tools/, analyze_photometry.py is in root/
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(tools_dir)
        analyze_script = os.path.join(root_dir, 'analyze_photometry.py')
        
        if run_tonic_mode:
            t_phase, started_utc_phase = _phase_start(status_data, "tonic_analysis")
            _write_status_update("tonic_analysis")
            emitter.emit("tonic", "start", "Running tonic analysis")
            emitter.close()  # Release file lock so subprocess can append events
            cmd_tonic = [sys.executable, analyze_script,
                         '--input', args.input,
                         '--out', tonic_out,
                         '--config', args.config,
                         '--mode', 'tonic',
                         '--format', args.format,
                         '--recursive', '--overwrite']
            if args.include_rois: cmd_tonic.extend(['--include-rois', args.include_rois])
            if args.exclude_rois: cmd_tonic.extend(['--exclude-rois', args.exclude_rois])
            if effective_traces_only: cmd_tonic.append('--traces-only')
            if args.event_signal: cmd_tonic.extend(['--event-signal', args.event_signal])
            if args.representative_session_index is not None: cmd_tonic.extend(['--representative-session-index', str(args.representative_session_index)])
            if args.preview_first_n is not None:
                cmd_tonic.extend(['--preview-first-n', str(args.preview_first_n)])
            if resolved_sessions_per_hour is not None: 
                cmd_tonic.extend(['--sessions-per-hour', str(resolved_sessions_per_hour)])
            if events_path: cmd_tonic.extend(['--events-path', events_path])
            try:
                manifest['commands'].append(run_cmd(cmd_tonic))
            finally:
                emitter = EventEmitter(events_path, run_id, run_dir, file_mode="a")
            emitter.emit("tonic", "done", "Tonic analysis complete")
            _phase_done(status_data, manifest, "tonic_analysis", t_phase, started_utc_phase, status_path=status_path)

            check_cancel(cancel_flag_path, emitter, "tonic", manifest_path, manifest)

        # ============================================================
        # 5. Phasic Analysis
        # ============================================================
        if run_phasic_mode:
            t_phase, started_utc_phase = _phase_start(status_data, "phasic_analysis")
            _write_status_update("phasic_analysis")
            emitter.emit("phasic", "start", "Running phasic analysis")
            emitter.close()  # Release file lock so subprocess can append events
            cmd_phasic = [sys.executable, analyze_script,
                          '--input', args.input,
                          '--out', phasic_out,
                          '--config', args.config,
                          '--mode', 'phasic',
                          '--format', args.format,
                          '--recursive', '--overwrite']
            if args.include_rois: cmd_phasic.extend(['--include-rois', args.include_rois])
            if args.exclude_rois: cmd_phasic.extend(['--exclude-rois', args.exclude_rois])
            if effective_traces_only: cmd_phasic.append('--traces-only')
            if args.event_signal: cmd_phasic.extend(['--event-signal', args.event_signal])
            if args.representative_session_index is not None: cmd_phasic.extend(['--representative-session-index', str(args.representative_session_index)])
            if args.preview_first_n is not None: cmd_phasic.extend(['--preview-first-n', str(args.preview_first_n)])
            if resolved_sessions_per_hour is not None: 
                cmd_phasic.extend(['--sessions-per-hour', str(resolved_sessions_per_hour)])
            if events_path: cmd_phasic.extend(['--events-path', events_path])
            try:
                manifest['commands'].append(run_cmd(cmd_phasic))
            finally:
                emitter = EventEmitter(events_path, run_id, run_dir, file_mode="a")
            emitter.emit("phasic", "done", "Phasic analysis complete")
            _phase_done(status_data, manifest, "phasic_analysis", t_phase, started_utc_phase, status_path=status_path)

            check_cancel(cancel_flag_path, emitter, "phasic", manifest_path, manifest)

        # ============================================================
        # 6. Session / Stride Computation
        # ============================================================
        t_phase, started_utc_phase = _phase_start(status_data, "session_compute")
        
        # Timing Resolution Cache Measurement (Mandatory)
        if run_tonic_mode and not run_phasic_mode:
            # Use HDF5 Cache for Tonic Timing
            cache_path = os.path.join(tonic_out, 'tonic_trace_cache.h5')
            cache = open_tonic_cache(cache_path)
            try:
                # Use first chunk to derive duration
                cids = list_cache_chunk_ids(cache)
                if not cids:
                    raise RuntimeError(f"No chunks found in tonic cache: {cache_path}")
                
                # We need an ROI to load data. The timing is SHARED across ROIs.
                roi_to_use = resolve_cache_roi(cache)
                t_arr = load_cache_chunk_fields(cache, roi_to_use, cids[0], ['time_sec'])[0]
                
                if len(t_arr) < 2:
                    raise RuntimeError(f"First tonic chunk is too short.")
                
                actual_trace_duration_s = t_arr[-1] - t_arr[0]
            finally:
                cache.close()
        else:
            # Phasic (Migrated to HDF5 Cache)
            cache_path = os.path.join(phasic_out, 'phasic_trace_cache.h5')
            if not os.path.exists(cache_path):
                raise RuntimeError(f"Phasic cache not found: {cache_path}")
            
            cache = open_phasic_cache(cache_path)
            try:
                roi = resolve_cache_roi(cache, None)
                cids = list_cache_chunk_ids(cache)
                if not cids:
                    raise RuntimeError(f"No chunks found in phasic cache: {cache_path}")
                
                cid0 = cids[0]
                # We only need time_sec to resolve duration
                t_arr, = load_cache_chunk_fields(cache, roi, cid0, ['time_sec'])
                
                if len(t_arr) < 2:
                    raise RuntimeError(f"First phasic chunk is too short.")
                
                actual_trace_duration_s = t_arr[-1] - t_arr[0]
            finally:
                cache.close()

        if not np.isfinite(actual_trace_duration_s) or actual_trace_duration_s <= 0:
            raise RuntimeError(f"Invalid actual_trace_duration_s: {actual_trace_duration_s}")

        # Stride Inference
        stride_s = None

        if resolved_sessions_per_hour is not None:
             sessions_per_hour = resolved_sessions_per_hour
             stride_s = 3600.0 / sessions_per_hour
             # Already logged at top of main
        else:
             # This script currently requires sessions_per_hour for delivery packaging
             raise RuntimeError("Cannot infer session stride (duty-cycled acquisition) without sessions_per_hour. Please provide it explicitly via GUI/CLI.")

        # Validation
        computed_sph = int(round(3600.0 / stride_s))
        if computed_sph < 1 or computed_sph > 12:
             raise RuntimeError(f"Invalid sessions per hour: {computed_sph}")

        if abs(stride_s * computed_sph - 3600.0) > 2.0:
             raise RuntimeError(f"Stride {stride_s:.2f}s not compatible with integer sessions/hr {computed_sph}")

        if args.session_duration_s is not None:
             if args.session_duration_s <= 0:
                  raise RuntimeError(f"Provided session duration must be > 0, got {args.session_duration_s}")

             # Validate actual duration against user input
             tol = max(2.0, 0.005 * args.session_duration_s)
             diff = abs(actual_trace_duration_s - args.session_duration_s)
             if diff > tol:
                   raise RuntimeError(f"Session Duration Mismatch! Provided: {args.session_duration_s:.2f}s, Trace (Cache): {actual_trace_duration_s:.2f}s (Diff: {diff:.2f}s, Tol: {tol:.2f}s).")

             session_duration_s = args.session_duration_s
        else:
             session_duration_s = actual_trace_duration_s

        # Impossible Schedule Check
        if session_duration_s > (stride_s + 1e-6):
             raise RuntimeError(f"Impossible schedule: Duration {session_duration_s:.2f}s > Stride {stride_s:.2f}s (SPH={sessions_per_hour}).")

        sessions_per_hour = computed_sph
        manifest['sessions_per_hour'] = sessions_per_hour
        manifest['sessions_per_hour_source'] = sessions_per_hour_source
        manifest['session_duration_s'] = session_duration_s
        manifest['session_stride_s'] = stride_s
        print(f"Deterministic Sessions Per Hour: {sessions_per_hour} (Stride={stride_s:.1f}s, Dur={session_duration_s:.1f}s)", flush=True)
        _phase_done(status_data, manifest, "session_compute", t_phase, started_utc_phase, status_path=status_path)

        check_cancel(cancel_flag_path, emitter, "session_compute", manifest_path, manifest)

        # 7. Per-Region Processing (Plots & Packaging)
        # ============================================================
        t_phase, started_utc_phase = _phase_start(status_data, "plots_total")
        _write_status_update("plots")
        emitter.emit("plots", "start", "Generating per-ROI deliverables")
        if tune_prep_light_mode and effective_skip_plan is not None:
            emitter.emit(
                "plots",
                "audit",
                "Tuning-prep selective skipping enabled for nonessential outputs.",
                payload=effective_skip_plan,
            )

        def _regions_from_run_report(report_path):
            if not report_path or not os.path.exists(report_path):
                return None
            try:
                rr = json.load(open(report_path, 'r'))
                roi_sel = rr.get('roi_selection', {})
                if roi_sel.get('selected_rois'):
                    return list(roi_sel['selected_rois'])
                if roi_sel.get('discovered_rois'):
                    return list(roi_sel['discovered_rois'])
            except Exception as e:
                print(f"WARNING: Failed to read roi_selection from {report_path}: {e}", flush=True)
            return None

        has_features = False
        df_feat = None
        regions = None
        if run_phasic_mode:
            feats_csv = os.path.join(phasic_out, 'features', 'features.csv')
            has_features = os.path.exists(feats_csv)
            if run_phasic_mode and has_features and not tune_prep_light_mode:
                df_feat = pd.read_csv(feats_csv)
                regions = sorted(df_feat['roi'].unique())
            else:
                regions = _regions_from_run_report(os.path.join(phasic_out, 'run_report.json'))

        # Fallback to tonic report only when tonic mode ran.
        if regions is None and run_tonic_mode:
            regions = _regions_from_run_report(os.path.join(tonic_out, 'run_report.json'))

        if regions is None:
            regions = []
            print("WARNING: Could not determine ROIs for packaging", flush=True)

        manifest['regions'] = regions

        for roi in regions:
            t_roi = time.perf_counter()
            started_utc_roi = _utc_now_iso()
            print(f"TIMING START roi={roi} at {started_utc_roi}", flush=True)
            roi_bucket_totals = {}
            roi_child_script_elapsed = 0.0

            t_bucket = time.perf_counter()
            check_cancel(cancel_flag_path, emitter, "plots", manifest_path, manifest)
            _write_status_update(f"plot_{roi}")

            print(f"Processing ROI: {roi}", flush=True)
            emitter.emit("plots", "progress", f"Processing ROI: {roi}", roi=roi)
            reg_dir = os.path.join(run_dir, roi)
            s_dir = os.path.join(reg_dir, "summary")
            d_dir = os.path.join(reg_dir, "day_plots")
            t_dir = os.path.join(reg_dir, "tables")
            _accumulate_roi_bucket(roi, roi_bucket_totals, "roi_setup", time.perf_counter() - t_bucket)

            t_bucket = time.perf_counter()
            for d in [reg_dir, s_dir, d_dir, t_dir]:
                os.makedirs(d, exist_ok=True)
            _accumulate_roi_bucket(roi, roi_bucket_totals, "roi_directory_prepare", time.perf_counter() - t_bucket)
            files_written = []

            manifest['deliverables'][roi] = {}
            if run_phasic_mode:
                t_bucket = time.perf_counter()
                cid_diag = None
                if has_features and df_feat is not None:
                    roi_feat = df_feat[df_feat['roi'] == roi].copy()
                    roi_feat = roi_feat.sort_values('chunk_id')
                    if not roi_feat.empty:
                        # Diagnostic Selection (Day 0, H 12, S 0)
                        diag_idx = 12 * sessions_per_hour
                        candidates = roi_feat['chunk_id'].values
                        if len(candidates) > diag_idx:
                            cid_diag = int(candidates[diag_idx])
                        else:
                            cid_diag = int(candidates[0])
                if cid_diag is None:
                    cache_path_diag = os.path.join(phasic_out, 'phasic_trace_cache.h5')
                    with open_phasic_cache(cache_path_diag) as cache_diag:
                        chunk_ids_diag = list_cache_chunk_ids(cache_diag)
                    if not chunk_ids_diag:
                        raise RuntimeError(
                            f"CRITICAL: No chunks found in phasic cache for ROI={roi}."
                        )
                    cid_diag = int(chunk_ids_diag[0])
                manifest['deliverables'][roi]['diagnostic_chunk_id'] = int(cid_diag)
                _accumulate_roi_bucket(roi, roi_bucket_totals, "roi_feature_selection", time.perf_counter() - t_bucket)

                # A. Phasic Correction Impact (4-panel + session CSV)
                cmd_impact = [sys.executable, 'tools/plot_phasic_correction_impact.py',
                              '--analysis-out', phasic_out,
                              '--roi', roi,
                              '--chunk-id', str(cid_diag),
                              '--out', os.path.join(s_dir, "phasic_correction_impact.png")]
                cmd_result = run_cmd(cmd_impact, roi_label=roi)
                manifest['commands'].append(cmd_result)
                roi_child_script_elapsed += cmd_result["elapsed_sec"]
                files_written.append("summary/phasic_correction_impact.png")

                t_bucket = time.perf_counter()
                cache_path_p = os.path.join(phasic_out, 'phasic_trace_cache.h5')
                if not os.path.exists(cache_path_p):
                    raise RuntimeError(f"CRITICAL: Phasic cache missing for required session CSV: {cache_path_p}")

                try:
                    with open_phasic_cache(cache_path_p) as f_p:
                        fields = ['time_sec', 'sig_raw', 'uv_raw', 'fit_ref', 'dff']
                        t, sig, uv, fit, dff = load_cache_chunk_fields(f_p, roi, int(cid_diag), fields)

                        df_c = pd.DataFrame({
                            't_s': t,
                            'sig_raw': sig,
                            'iso_raw': uv,
                            'iso_fit_dynamic': fit,
                            'dff_dynamic': dff,
                            'region': roi,
                            'chunk_id': int(cid_diag)
                        })

                        keep = ['t_s', 'sig_raw', 'iso_raw', 'iso_fit_dynamic', 'dff_dynamic', 'region', 'chunk_id']
                        df_c[keep].to_csv(os.path.join(t_dir, "phasic_correction_impact_session.csv"), index=False)
                        files_written.append("tables/phasic_correction_impact_session.csv")
                except Exception as e:
                    raise RuntimeError(f"CRITICAL: Failed to read phasic cache for session CSV (ROI={roi}, chunk={cid_diag}): {e}")
                _accumulate_roi_bucket(roi, roi_bucket_totals, "phasic_session_csv_packaging", time.perf_counter() - t_bucket)

            if run_tonic_mode and not tune_prep_light_mode:
                # B. Tonic Overview
                out_tonic = os.path.join(s_dir, "tonic_overview.png")
                cmd_tonic_roi = [sys.executable, 'tools/plot_tonic_48h.py',
                                 '--analysis-out', tonic_out, '--roi', roi,
                                 '--out', out_tonic,
                                 '--tonic-output-mode', effective_tonic_output_mode,
                                 '--tonic-timeline-mode', effective_tonic_timeline_mode]
                if args.input:
                    cmd_tonic_roi.extend(['--input', args.input])
                if args.format:
                    cmd_tonic_roi.extend(['--format', args.format])
                if resolved_sessions_per_hour is not None:
                    cmd_tonic_roi.extend(['--sessions-per-hour', str(resolved_sessions_per_hour)])
                cmd_result = run_cmd(cmd_tonic_roi, roi_label=roi)
                roi_child_script_elapsed += cmd_result["elapsed_sec"]

                t_bucket = time.perf_counter()
                if os.path.exists(out_tonic):
                    files_written.append("summary/tonic_overview.png")
                _accumulate_roi_bucket(roi, roi_bucket_totals, "tonic_image_packaging", time.perf_counter() - t_bucket)

                # Tonic DF Timeseries (Migrated to HDF5 Cache)
                t_bucket = time.perf_counter()
                tonic_csv_path = os.path.join(t_dir, "tonic_df_timeseries.csv")
                cache_path = os.path.join(tonic_out, 'tonic_trace_cache.h5')
                tonic_subbucket_totals = {
                    "cache_path_check": 0.0,
                    "cache_open": 0.0,
                    "chunk_enumeration": 0.0,
                    "per_chunk_cache_read": 0.0,
                    "per_chunk_transform": 0.0,
                    "row_accumulation": 0.0,
                    "dataframe_concat": 0.0,
                    "csv_write": 0.0,
                    "cache_close": 0.0
                }
                chunks_processed = 0
                rows_written = 0
                frames_accumulated = 0
                tonic_mode_fallback_count = 0
                csv_file_handle = None
                t_secs_mode = []
                t_secs_real = []
                d_vals = []
                use_streaming_csv = (
                    HAS_PYARROW_CSV
                    and effective_tonic_timeline_mode == TONIC_TIMELINE_MODE_REAL_ELAPSED
                )
                arrow_write_header = None
                arrow_write_no_header = None
                if use_streaming_csv:
                    arrow_write_header = pa_csv.WriteOptions(include_header=True, delimiter=',')
                    arrow_write_no_header = pa_csv.WriteOptions(include_header=False, delimiter=',')

                t_sub = time.perf_counter()
                if os.path.exists(cache_path):
                    tonic_subbucket_totals["cache_path_check"] += time.perf_counter() - t_sub

                    t_sub = time.perf_counter()
                    cache = open_tonic_cache(cache_path)
                    tonic_subbucket_totals["cache_open"] += time.perf_counter() - t_sub
                    try:
                        t_sub = time.perf_counter()
                        cids = list_cache_chunk_ids(cache)
                        
                        from photometry_pipeline.utils.timeline import map_cached_sources_to_schedule_positions
                                
                        source_files = []
                        if "meta" in cache and "source_files" in cache["meta"]:
                            source_files = [f.decode('utf-8') if isinstance(f, bytes) else f for f in cache["meta"]["source_files"][:]]

                        # Rescale timeline offsets using exactly matched position
                        actual_positions = map_cached_sources_to_schedule_positions(
                            args.input, args.format, source_files, cids
                        )
                        prev_chunk_end_sec = None
                        prev_dt_sec = None
                        prev_real_end_sec = None
                        prev_real_dt_sec = None

                        tonic_subbucket_totals["chunk_enumeration"] += time.perf_counter() - t_sub
                        for i, cid in enumerate(cids):
                            actual_schedule_idx = actual_positions[i] if i < len(actual_positions) else None
                            
                            chunks_processed += 1
                            t_sub = time.perf_counter()
                            if effective_tonic_output_mode == TONIC_OUTPUT_MODE_PRESERVE_RAW:
                                t_arr, df_arr = load_cache_chunk_fields(
                                    cache, roi, cid, ['time_sec', 'deltaF']
                                )
                            else:
                                t_arr, sig_arr, uv_arr, df_raw_arr = load_cache_chunk_fields(
                                    cache, roi, cid, ['time_sec', 'sig_raw', 'uv_raw', 'deltaF']
                                )
                                _, _, df_arr, tonic_mode_meta = apply_tonic_output_mode_to_session(
                                    time_sec=t_arr,
                                    sig_raw=sig_arr,
                                    uv_raw=uv_arr,
                                    deltaf_raw=df_raw_arr,
                                    mode_raw=effective_tonic_output_mode,
                                )
                                tonic_mode_fallback_count += int(
                                    tonic_mode_meta.get("fallback_count", 0)
                                )
                            tonic_subbucket_totals["per_chunk_cache_read"] += time.perf_counter() - t_sub

                            t_sub = time.perf_counter()
                            t_abs_real, timeline_state_real = build_tonic_chunk_time_axis(
                                time_sec_local=t_arr,
                                timeline_mode_raw=TONIC_TIMELINE_MODE_REAL_ELAPSED,
                                chunk_sequence_index=int(cid),
                                actual_schedule_index=actual_schedule_idx,
                                stride_sec=stride_s,
                                prev_chunk_end_sec=prev_real_end_sec,
                                prev_dt_sec=prev_real_dt_sec,
                            )
                            prev_real_end_sec = timeline_state_real["prev_chunk_end_sec"]
                            prev_real_dt_sec = timeline_state_real["prev_dt_sec"]
                            if effective_tonic_timeline_mode == TONIC_TIMELINE_MODE_REAL_ELAPSED:
                                t_abs = t_abs_real
                            else:
                                t_abs, timeline_state = build_tonic_chunk_time_axis(
                                    time_sec_local=t_arr,
                                    timeline_mode_raw=TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
                                    chunk_sequence_index=int(cid),
                                    actual_schedule_index=actual_schedule_idx,
                                    stride_sec=stride_s,
                                    prev_chunk_end_sec=prev_chunk_end_sec,
                                    prev_dt_sec=prev_dt_sec,
                                )
                                prev_chunk_end_sec = timeline_state["prev_chunk_end_sec"]
                                prev_dt_sec = timeline_state["prev_dt_sec"]
                            tonic_subbucket_totals["per_chunk_transform"] += time.perf_counter() - t_sub

                            t_sub = time.perf_counter()
                            rows_written += len(t_abs)
                            frames_accumulated += 1
                            tonic_subbucket_totals["row_accumulation"] += time.perf_counter() - t_sub

                            if use_streaming_csv:
                                t_sub = time.perf_counter()
                                if csv_file_handle is None:
                                    csv_file_handle = pa.OSFile(tonic_csv_path, 'wb')
                                table = pa.table({
                                    'time_hours': pa.array(t_abs / 3600.0),
                                    'tonic_df': pa.array(df_arr)
                                })
                                write_opts = arrow_write_header if chunks_processed == 1 else arrow_write_no_header
                                pa_csv.write_csv(table, csv_file_handle, write_options=write_opts)
                                tonic_subbucket_totals["csv_write"] += time.perf_counter() - t_sub
                            else:
                                t_sub = time.perf_counter()
                                t_secs_mode.append(np.asarray(t_abs, dtype=float))
                                t_secs_real.append(np.asarray(t_abs_real, dtype=float))
                                d_vals.append(np.asarray(df_arr, dtype=float))
                                tonic_subbucket_totals["row_accumulation"] += time.perf_counter() - t_sub
                    finally:
                        if csv_file_handle is not None:
                            t_sub = time.perf_counter()
                            csv_file_handle.close()
                            tonic_subbucket_totals["csv_write"] += time.perf_counter() - t_sub
                        t_sub = time.perf_counter()
                        cache.close()
                        tonic_subbucket_totals["cache_close"] += time.perf_counter() - t_sub
                else:
                    tonic_subbucket_totals["cache_path_check"] += time.perf_counter() - t_sub

                if t_secs_mode:
                    t_sub = time.perf_counter()
                    full_t_mode = np.concatenate(t_secs_mode)
                    full_t_real = np.concatenate(t_secs_real)
                    if effective_tonic_timeline_mode != TONIC_TIMELINE_MODE_REAL_ELAPSED:
                        full_t_mode = remap_gapfree_axis_to_elapsed_span(
                            full_t_mode,
                            elapsed_start_sec=float(np.nanmin(full_t_real)),
                            elapsed_end_sec=float(np.nanmax(full_t_real)),
                        )
                    full_tonic = pd.DataFrame(
                        {
                            'time_hours': full_t_mode / 3600.0,
                            'tonic_df': np.concatenate(d_vals),
                        }
                    )
                    tonic_subbucket_totals["dataframe_concat"] += time.perf_counter() - t_sub

                    t_sub = time.perf_counter()
                    full_tonic.to_csv(tonic_csv_path, index=False)
                    tonic_subbucket_totals["csv_write"] += time.perf_counter() - t_sub

                if frames_accumulated > 0 and os.path.exists(tonic_csv_path):
                    files_written.append("tables/tonic_df_timeseries.csv")

                tonic_table_total_elapsed = time.perf_counter() - t_bucket
                tonic_subbucket_sum = sum(tonic_subbucket_totals.values())
                tonic_subbucket_remainder = tonic_table_total_elapsed - tonic_subbucket_sum
                for subbucket_name, subbucket_elapsed in tonic_subbucket_totals.items():
                    _log_roi_timing_detail(roi, f"tonic_table_packaging.{subbucket_name}", subbucket_elapsed)
                _log_roi_timing_detail(roi, "tonic_table_packaging.subbucket_sum", tonic_subbucket_sum)
                _log_roi_timing_detail(roi, "tonic_table_packaging.subbucket_remainder", tonic_subbucket_remainder)
                _log_roi_timing_metric(roi, "tonic_table_packaging.chunks_processed", chunks_processed)
                _log_roi_timing_metric(roi, "tonic_table_packaging.rows_written", rows_written)
                _log_roi_timing_metric(roi, "tonic_table_packaging.frames_accumulated", frames_accumulated)
                _log_roi_timing_metric(
                    roi,
                    "tonic_table_packaging.tonic_mode_fallback_count",
                    tonic_mode_fallback_count,
                )
                _accumulate_roi_bucket(roi, roi_bucket_totals, "tonic_table_packaging", tonic_table_total_elapsed)
            elif run_tonic_mode and tune_prep_light_mode:
                manifest['deliverables'][roi].setdefault('intentionally_skipped', [])
                manifest['deliverables'][roi]['intentionally_skipped'].extend([
                    "summary/tonic_overview.png",
                    "tables/tonic_df_timeseries.csv",
                ])

            if run_phasic_mode and has_features:
                # C. Phasic Time Series (Plots & CSV) — requires features
                out_rate_png = os.path.join(s_dir, "phasic_peak_rate_timeseries.png")
                out_auc_png = os.path.join(s_dir, "phasic_auc_timeseries.png")
                out_rate_csv = os.path.join(t_dir, "phasic_peak_rate_timeseries.csv")
                out_auc_csv = os.path.join(t_dir, "phasic_auc_timeseries.csv")

                cmd_ts = [sys.executable, 'tools/plot_phasic_time_series_summary.py',
                          '--analysis-out', phasic_out,
                          '--roi', roi,
                          '--sessions-per-hour', str(sessions_per_hour),
                          '--session-duration-s', str(session_duration_s),
                          '--out-rate-png', out_rate_png,
                          '--out-auc-png', out_auc_png,
                          '--out-rate-csv', out_rate_csv,
                          '--out-auc-csv', out_auc_csv]
                if args.timeline_anchor_mode != "civil":
                    cmd_ts.extend(['--timeline-anchor-mode', str(args.timeline_anchor_mode)])
                if args.timeline_anchor_mode == "fixed_daily_anchor" and args.fixed_daily_anchor_clock:
                    cmd_ts.extend(['--fixed-daily-anchor-clock', str(args.fixed_daily_anchor_clock)])
                cmd_result = run_cmd(cmd_ts, roi_label=roi)
                manifest['commands'].append(cmd_result)
                roi_child_script_elapsed += cmd_result["elapsed_sec"]

                # Record Results
                t_bucket = time.perf_counter()
                for dst in [out_rate_png, out_auc_png, out_rate_csv, out_auc_csv]:
                    if os.path.exists(dst):
                        rel_path = os.path.relpath(dst, reg_dir).replace('\\', '/')
                        files_written.append(rel_path)
                _accumulate_roi_bucket(roi, roi_bucket_totals, "phasic_ts_packaging", time.perf_counter() - t_bucket)
            elif run_phasic_mode and tune_prep_light_mode:
                manifest['deliverables'][roi].setdefault('intentionally_skipped', [])
                manifest['deliverables'][roi]['intentionally_skipped'].extend([
                    "summary/phasic_peak_rate_timeseries.png",
                    "summary/phasic_auc_timeseries.png",
                    "tables/phasic_peak_rate_timeseries.csv",
                    "tables/phasic_auc_timeseries.csv",
                ])

            t_bucket = time.perf_counter()
            check_cancel(cancel_flag_path, emitter, "plots", manifest_path, manifest)
            _accumulate_roi_bucket(roi, roi_bucket_totals, "roi_cancel_check", time.perf_counter() - t_bucket)

            s_dff = []
            s_sig = []
            s_dyn = []
            s_stk = []
            if run_phasic_mode and not tune_prep_light_mode:
                # D. Per-Day Plots (Sig/Iso, dFF, Stacked) via Unified Bundle Driver
                cmd_bundle = [sys.executable, 'tools/plot_phasic_dayplot_bundle.py',
                              '--analysis-out', phasic_out,
                              '--roi', roi,
                              '--output-dir', d_dir,
                              '--sessions-per-hour', str(sessions_per_hour),
                              '--session-duration-s', str(session_duration_s),
                              '--smooth-window-s', str(args.smooth_window_s),
                              '--sig-iso-render-mode', str(args.sig_iso_render_mode),
                              '--dff-render-mode', str(args.dff_render_mode),
                              '--stacked-render-mode', str(args.stacked_render_mode)]
                if args.timeline_anchor_mode != "civil":
                    cmd_bundle.extend(['--timeline-anchor-mode', str(args.timeline_anchor_mode)])
                if args.timeline_anchor_mode == "fixed_daily_anchor" and args.fixed_daily_anchor_clock:
                    cmd_bundle.extend(['--fixed-daily-anchor-clock', str(args.fixed_daily_anchor_clock)])

                if not has_features:
                    cmd_bundle.extend(['--no-write-dff-grid', '--no-write-stacked'])

                cmd_result = run_cmd(cmd_bundle, roi_label=roi)
                manifest['commands'].append(cmd_result)
                roi_child_script_elapsed += cmd_result["elapsed_sec"]

                # Collect Per-Day Files
                t_bucket = time.perf_counter()
                days_dff = set()
                days_sig_iso = set()

                # Verify dFF
                if has_features:
                    for f in glob.glob(os.path.join(d_dir, "phasic_dFF_day_*.png")):
                        files_written.append(f"day_plots/{os.path.basename(f)}")
                        m = re.match(r'phasic_dFF_day_(\d+)\.png', os.path.basename(f))
                        if m:
                            days_dff.add(m.group(1))

                # Verify Sig/Iso
                for f in glob.glob(os.path.join(d_dir, "phasic_sig_iso_day_*.png")):
                     files_written.append(f"day_plots/{os.path.basename(f)}")
                     m = re.match(r'phasic_sig_iso_day_(\d+)\.png', os.path.basename(f))
                     if m:
                         days_sig_iso.add(m.group(1))

                # Verify Dynamic Fit
                days_dynamic_fit = set()
                for f in glob.glob(os.path.join(d_dir, "phasic_dynamic_fit_day_*.png")):
                    files_written.append(f"day_plots/{os.path.basename(f)}")
                    m = re.match(r'phasic_dynamic_fit_day_(\d+)\.png', os.path.basename(f))
                    if m:
                        days_dynamic_fit.add(m.group(1))

                # Stacked are already in d_dir, just verify
                days_stacked = set()
                for f in glob.glob(os.path.join(d_dir, "phasic_stacked_day_*.png")):
                    files_written.append(f"day_plots/{os.path.basename(f)}")
                    m = re.match(r'phasic_stacked_day_(\d+)\.png', os.path.basename(f))
                    if m:
                        days_stacked.add(m.group(1))
                _accumulate_roi_bucket(roi, roi_bucket_totals, "dayplot_file_discovery", time.perf_counter() - t_bucket)

                s_dff = sorted(list(days_dff))
                s_sig = sorted(list(days_sig_iso))
                s_dyn = sorted(list(days_dynamic_fit))
                s_stk = sorted(list(days_stacked))
            elif run_phasic_mode and tune_prep_light_mode:
                manifest['deliverables'][roi].setdefault('intentionally_skipped', [])
                manifest['deliverables'][roi]['intentionally_skipped'].extend([
                    "day_plots/phasic_sig_iso_day_*.png",
                    "day_plots/phasic_dynamic_fit_day_*.png",
                    "day_plots/phasic_dFF_day_*.png",
                    "day_plots/phasic_stacked_day_*.png",
                ])

            t_bucket = time.perf_counter()
            manifest['deliverables'][roi]['days_dff'] = s_dff
            manifest['deliverables'][roi]['days_sig_iso'] = s_sig
            manifest['deliverables'][roi]['days_dynamic_fit'] = s_dyn
            manifest['deliverables'][roi]['days_stacked'] = s_stk

            # Consistency check (restored)
            if run_phasic_mode and has_features and not (s_dff == s_sig == s_stk):
                 raise RuntimeError(f"Inconsistent day sets for ROI {roi}: DFF={s_dff}, SigIso={s_sig}, Stacked={s_stk}")
            if run_phasic_mode and s_dyn and s_sig and s_dyn != s_sig:
                 raise RuntimeError(f"Inconsistent day sets for ROI {roi}: DynamicFit={s_dyn}, SigIso={s_sig}")

            manifest['deliverables'][roi]['files'] = sorted(list(set(files_written)))
            if run_phasic_mode:
                manifest['deliverables'][roi]['days_generated'] = s_dff if has_features else s_stk
            else:
                manifest['deliverables'][roi]['days_generated'] = []
            _accumulate_roi_bucket(roi, roi_bucket_totals, "roi_manifest_bookkeeping", time.perf_counter() - t_bucket)

            # ROI timing finalize
            elapsed_roi = time.perf_counter() - t_roi
            finished_utc_roi = _utc_now_iso()
            manifest['deliverables'][roi]['timing'] = {
                "started_utc": started_utc_roi,
                "finished_utc": finished_utc_roi,
                "elapsed_sec": elapsed_roi
            }
            explicit_in_process_elapsed = sum(roi_bucket_totals.values())
            residual_remainder = elapsed_roi - roi_child_script_elapsed - explicit_in_process_elapsed
            _log_roi_timing_detail(roi, "child_script_timings_sum", roi_child_script_elapsed)
            _log_roi_timing_detail(roi, "explicit_in_process_sum", explicit_in_process_elapsed)
            _log_roi_timing_detail(roi, "residual_remainder", residual_remainder)
            print(f"TIMING DONE roi={roi} elapsed_sec={elapsed_roi:.3f}", flush=True)

        emitter.emit("plots", "done", "All ROI deliverables complete")
        _phase_done(status_data, manifest, "plots_total", t_phase, started_utc_phase, status_path=status_path)

        check_cancel(cancel_flag_path, emitter, "package", manifest_path, manifest)
        
        # ============================================================
        # 8. Write Manifest (LAST, atomic)
        # ============================================================
        t_phase, started_utc_phase = _phase_start(status_data, "manifest_write")
        emitter.emit("package", "start", "Writing final manifest")
        
        # Add authoritative total runtime to manifest timing
        manifest["timing"]["total_runtime_sec"] = time.perf_counter() - t0_total
        
        _atomic_write_json(manifest_path, manifest)
        emitter.emit("package", "done", "Manifest written")
        _phase_done(status_data, manifest, "manifest_write", t_phase, started_utc_phase, status_path=status_path)

        # ============================================================
        # 9. Finalize Artifacts (Strict Ordering Gate)
        # ============================================================
        t_phase, started_utc_phase = _phase_start(status_data, "finalize_artifacts")
        # ============================================================
        # Finalize Status (Success)
        # ============================================================
        ok = _ensure_root_run_report(run_dir, phasic_out, tonic_out, emitter,
                                     run_type=effective_run_type,
                                     run_profile=args.run_type,
                                     artifact_contract=effective_artifact_contract,
                                     intentional_skips=effective_skip_plan,
                                     sessions_per_hour=resolved_sessions_per_hour,
                                     sessions_per_hour_source=sessions_per_hour_source,
                                     timeline_anchor_mode=args.timeline_anchor_mode,
                                     fixed_daily_anchor_clock=args.fixed_daily_anchor_clock)
        err_payload = None
        if not ok:
            msg = "STRICT ORDERING VIOLATION: root run_report.json missing at terminal finalize"
            emitter.emit("package", "error", msg)
            err_payload = msg
            
        _phase_done(status_data, manifest, "finalize_artifacts", t_phase, started_utc_phase, status_path=status_path)

        # Final manifest update for end-to-end timing
        manifest["timing"]["total_runtime_sec"] = time.perf_counter() - t0_total
        _atomic_write_json(manifest_path, manifest)

        substantive_work_completed = True
        _finalize_status("success", error_msg=err_payload)
        emitter.emit("engine", "done", "Execution complete")
        emitter.close()
    
    except SystemExit as se:
        # Re-raise sys.exit calls (from check_cancel) WITHOUT catching them
        # BUT ensure the ordering gate holds for cancellation too.
        ok = False
        try:
             # Pass explicit variables as requested (not locals())
             ok = _ensure_root_run_report(run_dir, phasic_out, tonic_out, emitter,
                                          run_type=effective_run_type,
                                          run_profile=args.run_type,
                                          artifact_contract=effective_artifact_contract,
                                          intentional_skips=effective_skip_plan,
                                          sessions_per_hour=resolved_sessions_per_hour,
                                          sessions_per_hour_source=sessions_per_hour_source,
                                          timeline_anchor_mode=args.timeline_anchor_mode,
                                          fixed_daily_anchor_clock=args.fixed_daily_anchor_clock)
        except Exception:
             pass
             
        err_payload = "CANCELLED"
        if not ok:
            msg = "STRICT ORDERING VIOLATION: root run_report.json missing at terminal finalize"
            if emitter:
                emitter.emit("package", "error", msg)
            err_payload = f"CANCELLED | {msg}"

        _finalize_status("cancelled", error_msg=err_payload)
        if emitter:
            emitter.close()
        raise se
    except Exception as e:
        # --- Catch-all Error Handling ---
        import traceback
        traceback.print_exc()

        # If substantive work already completed, avoid rewriting terminal status as
        # generic pipeline failure when the remaining error is status-write bookkeeping.
        if substantive_work_completed and _is_windows_lock_error(e):
            if emitter:
                emitter.emit(
                    "engine",
                    "error",
                    "Substantive work completed, but terminal status.json write failed",
                    error_code="STATUS_WRITE_FAILED_POST_SUCCESS"
                )
            print(
                "ERROR: Substantive pipeline work completed, but terminal status.json "
                "could not be finalized due file-lock contention.",
                flush=True
            )
            raise SystemExit(1)
        
        # ============================================================
        # Finalize Artifacts (Strict Ordering Gate) - Error path
        # ============================================================
        ok = False
        try:
             ok = _ensure_root_run_report(run_dir, phasic_out, tonic_out, emitter,
                                          run_type=effective_run_type,
                                          run_profile=args.run_type,
                                          artifact_contract=effective_artifact_contract,
                                          intentional_skips=effective_skip_plan,
                                          sessions_per_hour=resolved_sessions_per_hour,
                                          sessions_per_hour_source=sessions_per_hour_source,
                                          timeline_anchor_mode=args.timeline_anchor_mode,
                                          fixed_daily_anchor_clock=args.fixed_daily_anchor_clock)
        except Exception:
             pass

        msg_body = str(e)
        if not ok:
            viol_msg = "STRICT ORDERING VIOLATION: root run_report.json missing at terminal finalize"
            if emitter:
                emitter.emit("package", "error", viol_msg)
            msg_body = f"{msg_body} | {viol_msg}"

        # --- Finalize Status: Error ---
        try:
            _finalize_status("error", error_msg=msg_body)
        except Exception as status_err:
            print(f"ERROR: Failed to finalize error status.json: {status_err}", flush=True)
        raise SystemExit(1)

if __name__ == '__main__':
    main()
