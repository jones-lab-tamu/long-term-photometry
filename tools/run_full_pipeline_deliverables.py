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
import glob
import re
import pandas as pd
import numpy as np
import time
import secrets
from datetime import datetime, timezone

# Self-contained repo root bootstrap
from pathlib import Path
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from photometry_pipeline.config import Config
    from photometry_pipeline.core.utils import natural_sort_key
    from photometry_pipeline.core.events import EventEmitter
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
        _atomic_write_json(status_path, status_data)
    
    print(f"TIMING DONE phase={phase_name} elapsed_sec={elapsed:.3f}", flush=True)


def _generate_run_id():
    """Generate a deterministic-format run_id: run_YYYYMMDD_HHMMSS_<8hex>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{secrets.token_hex(4)}"


def _atomic_write_json(path, obj):
    """Write JSON atomically via tmp + os.replace."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

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


def _ensure_root_run_report(run_dir, phasic_out, tonic_out, emitter, sessions_per_hour=None, sessions_per_hour_source=None):
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
             if 'run_context' in repo:
                 repo['run_context']['sessions_per_hour'] = sessions_per_hour
                 repo['run_context']['sessions_per_hour_source'] = sessions_per_hour_source
             
             if 'derived_settings' in repo:
                 repo['derived_settings']['sessions_per_hour'] = sessions_per_hour
                 repo['derived_settings']['sessions_per_hour_source'] = sessions_per_hour_source
                 
             with open(report_path, 'w') as f:
                 json.dump(repo, f, indent=2)
             
             if emitter and sessions_per_hour_source:
                 emitter.emit("package", "audit", f"Stamped report with sessions_per_hour_source={sessions_per_hour_source}")
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
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--include-rois', type=str, default=None, help="Comma-separated list of ROIs to process exclusively")
    parser.add_argument('--exclude-rois', type=str, default=None, help="Comma-separated list of ROIs to ignore")
    parser.add_argument('--traces-only', action='store_true', help="Run traces and QC, skip feature extraction (features.csv) and feature-dependent summaries.")
    parser.add_argument('--sessions-per-hour', type=int, help="Force sessions per hour (integer)")
    parser.add_argument('--session-duration-s', type=float, help="Recording duration in seconds (data length per chunk). If provided, validated against traces.")
    parser.add_argument('--smooth-window-s', type=float, default=1.0)
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

    # sessions_per_hour
    if args.sessions_per_hour is not None:
        if args.sessions_per_hour < 1:
            raise RuntimeError(f"--sessions-per-hour must be >= 1, got {args.sessions_per_hour}")

    # session_duration_s
    if args.session_duration_s is not None:
        if args.session_duration_s <= 0:
            raise RuntimeError(f"--session-duration-s must be > 0, got {args.session_duration_s}")

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

    # Determine effective event signal, representative index, and preview for stamping
    effective_event_signal = args.event_signal
    effective_representative_index = args.representative_session_index
    effective_preview_first_n = args.preview_first_n
    
    if effective_event_signal is None or effective_representative_index is None or effective_preview_first_n is None:
        try:
            cfg = Config.from_yaml(args.config)
            if effective_event_signal is None:
                effective_event_signal = getattr(cfg, "event_signal", "dff")
            if effective_representative_index is None:
                effective_representative_index = getattr(cfg, "representative_session_index", None)
            if effective_preview_first_n is None:
                effective_preview_first_n = getattr(cfg, "preview_first_n", None)
        except Exception as e:
            print(f"WARNING: Failed to parse config for runner stamping: {e}", flush=True)
            if effective_event_signal is None: 
                effective_event_signal = "dff"
            # others remain as given or None

    print(f"RUN_DIR: {run_dir}", flush=True)

    # -- Build base manifest dict --
    manifest = {
        'tool': 'run_full_pipeline_deliverables',
        'timestamp': datetime.now().isoformat(),
        'run_id': run_id,
        'run_dir': run_dir,
        'events_path': events_path,
        'cancel_flag_path': cancel_flag_path,
        'args': vars(args),
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
                "events_mode": args.events,
                "outputs": {
                    "manifest_json": None,
                    "events_ndjson": vo_events_path,
                    "region_dirs": []
                },
                "errors": [],
                "warnings": [],
                "validation": {"result": "validate_only"}
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
            _atomic_write_json(vo_status_path, sd)

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
        argv.extend(["--smooth-window-s", str(args.smooth_window_s)])

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
        "events_mode": args.events,
        "outputs": {
            "manifest_json": manifest_path,
            "events_ndjson": events_path,
            "region_dirs": []
        },
        "errors": [],
        "warnings": [],
        "validation": None,
        "features_extracted": False if args.traces_only else None,
        "traces_only": args.traces_only,
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
        _atomic_write_json(status_path, status_data)

    def _write_status_update(phase):
        """Perform a non-terminal status update with current duration."""
        status_data["phase"] = phase
        status_data["status"] = "running"
        status_data["duration_sec"] = time.time() - t0_status
        _atomic_write_json(status_path, status_data)

    # Update status_data with resolved preview info before initial write
    status_data["run_type"] = "preview" if effective_preview_first_n is not None else "full"
    status_data["preview"] = {"selector": "first_n", "first_n": effective_preview_first_n} if effective_preview_first_n is not None else None

    # Initial write (phase="running")
    _atomic_write_json(status_path, status_data)
    _write_status_update("initializing")

    # -- Open event emitter --
    emitter = EventEmitter(events_path, run_id, run_dir, file_mode="w")
    emitter.emit("engine", "start", "Engine starting")
    emitter.emit("engine", "context", "Run context initialized", payload={
        "run_type": "preview" if effective_preview_first_n is not None else "full", 
        "features_extracted": False if args.traces_only else None, 
        "preview": {"selector": "first_n", "first_n": effective_preview_first_n} if effective_preview_first_n is not None else None, 
        "traces_only": args.traces_only, 
        "event_signal": effective_event_signal,
        "representative_session_index": effective_representative_index,
        "sessions_per_hour": resolved_sessions_per_hour,
        "sessions_per_hour_source": sessions_per_hour_source
    })

    try:
        # -- Check cancel immediately --
        check_cancel(cancel_flag_path, emitter, "engine_start", manifest_path, manifest)
    
        analysis_dir = os.path.join(run_dir, '_analysis')
        tonic_out = os.path.join(analysis_dir, 'tonic_out')
        phasic_out = os.path.join(analysis_dir, 'phasic_out')

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
        
        if args.mode in ('both', 'tonic'):
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
            if args.traces_only: cmd_tonic.append('--traces-only')
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
        if args.mode in ('both', 'phasic'):
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
            if args.traces_only: cmd_phasic.append('--traces-only')
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
        if args.mode == 'tonic':
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

        feats_csv = os.path.join(phasic_out, 'features', 'features.csv')
        has_features = os.path.exists(feats_csv)

        if has_features:
            df_feat = pd.read_csv(feats_csv)
            regions = sorted(df_feat['roi'].unique())
        else:
            # traces-only: derive ROIs from canonical roi_selection in run_report.json
            regions = None
            report_path = os.path.join(phasic_out, 'run_report.json')
            if os.path.exists(report_path):
                try:
                    rr = json.load(open(report_path, 'r'))
                    roi_sel = rr.get('roi_selection', {})
                    if roi_sel.get('selected_rois'):
                        regions = list(roi_sel['selected_rois'])
                    elif roi_sel.get('discovered_rois'):
                        regions = list(roi_sel['discovered_rois'])
                except Exception as e:
                    print(f"WARNING: Failed to read roi_selection from run_report.json: {e}", flush=True)
            if regions is None:
                print("WARNING: roi_selection not found in run_report.json; falling back to tonic_out", flush=True)
                report_path_t = os.path.join(tonic_out, 'run_report.json')
                if os.path.exists(report_path_t):
                    try:
                        rr = json.load(open(report_path_t, 'r'))
                        roi_sel = rr.get('roi_selection', {})
                        if roi_sel.get('selected_rois'):
                            regions = list(roi_sel['selected_rois'])
                        elif roi_sel.get('discovered_rois'):
                            regions = list(roi_sel['discovered_rois'])
                    except Exception:
                        pass
            if regions is None:
                regions = []
                print("WARNING: Could not determine ROIs for traces-only packaging", flush=True)
            df_feat = None
        manifest['regions'] = regions

        for roi in regions:
            t_roi = time.perf_counter()
            started_utc_roi = _utc_now_iso()
            print(f"TIMING START roi={roi} at {started_utc_roi}", flush=True)
            
            check_cancel(cancel_flag_path, emitter, "plots", manifest_path, manifest)
            _write_status_update(f"plot_{roi}")

            print(f"Processing ROI: {roi}", flush=True)
            emitter.emit("plots", "progress", f"Processing ROI: {roi}", roi=roi)
            reg_dir = os.path.join(run_dir, roi)
            s_dir = os.path.join(reg_dir, "summary")
            d_dir = os.path.join(reg_dir, "day_plots")
            t_dir = os.path.join(reg_dir, "tables")
            
            for d in [reg_dir, s_dir, d_dir, t_dir]:
                os.makedirs(d, exist_ok=True)
            files_written = []

            if has_features:
                roi_feat = df_feat[df_feat['roi'] == roi].copy()
                roi_feat = roi_feat.sort_values('chunk_id')

                # Diagnostic Selection (Day 0, H 12, S 0)
                diag_idx = 12 * sessions_per_hour
                candidates = roi_feat['chunk_id'].values

                if len(candidates) > diag_idx:
                    cid_diag = candidates[diag_idx]
                else:
                    cid_diag = candidates[0]

                manifest['deliverables'][roi] = {'diagnostic_chunk_id': int(cid_diag)}

                # A. Phasic Correction Impact (3-Panel)
                cmd_impact = [sys.executable, 'tools/plot_phasic_correction_impact.py',
                              '--analysis-out', phasic_out,
                              '--roi', roi,
                              '--chunk-id', str(cid_diag),
                              '--out', os.path.join(s_dir, "phasic_correction_impact.png")]
                manifest['commands'].append(run_cmd(cmd_impact, roi_label=roi))
                files_written.append("summary/phasic_correction_impact.png")

                # Correction Data CSV
                c_csv = os.path.join(phasic_out, 'phasic_intermediates', f"chunk_{cid_diag:04d}_{roi}.csv")
                if not os.path.exists(c_csv):
                     c_csv = os.path.join(phasic_out, 'phasic_intermediates', f"chunk_{cid_diag}_{roi}.csv")

                if os.path.exists(c_csv):
                    df_c = pd.read_csv(c_csv)
                    rename_map = {'time_sec': 't_s', 'fit_ref': 'iso_fit_dynamic', 'dff': 'dff_dynamic'}
                    df_c = df_c.rename(columns=rename_map)
                    keep = ['t_s', 'sig_raw', 'iso_raw', 'iso_fit_dynamic', 'dff_dynamic', 'region', 'chunk_id']
                    df_c['region'] = roi
                    df_c['chunk_id'] = cid_diag
                    df_c[keep].to_csv(os.path.join(t_dir, "phasic_correction_impact_session.csv"), index=False)
                    files_written.append("tables/phasic_correction_impact_session.csv")
            else:
                manifest['deliverables'][roi] = {}

            # B. Tonic Overview
            out_tonic = os.path.join(s_dir, "tonic_overview.png")
            cmd_tonic_roi = [sys.executable, 'tools/plot_tonic_48h.py',
                             '--analysis-out', tonic_out, '--roi', roi,
                             '--out', out_tonic]
            run_cmd(cmd_tonic_roi, roi_label=roi)

            if os.path.exists(out_tonic):
                files_written.append("summary/tonic_overview.png")

            # Tonic DF Timeseries (Migrated to HDF5 Cache)
            t_rows = []
            cache_path = os.path.join(tonic_out, 'tonic_trace_cache.h5')
            if os.path.exists(cache_path):
                cache = open_tonic_cache(cache_path)
                try:
                    cids = list_cache_chunk_ids(cache)
                    for i, cid in enumerate(cids):
                        t_arr, df_arr = load_cache_chunk_fields(cache, roi, cid, ['time_sec', 'deltaF'])
                        t_abs = (i * stride_s) + t_arr
                        df_sub = pd.DataFrame({
                            'time_hours': t_abs / 3600.0,
                            'tonic_df': df_arr
                        })
                        t_rows.append(df_sub)
                finally:
                    cache.close()

            if t_rows:
                full_tonic = pd.concat(t_rows, ignore_index=True)
                full_tonic.to_csv(os.path.join(t_dir, "tonic_df_timeseries.csv"), index=False)
                files_written.append("tables/tonic_df_timeseries.csv")

            if has_features:
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
                manifest['commands'].append(run_cmd(cmd_ts, roi_label=roi))

                # Record Results
                for dst in [out_rate_png, out_auc_png, out_rate_csv, out_auc_csv]:
                    if os.path.exists(dst):
                        rel_path = os.path.relpath(dst, reg_dir).replace('\\', '/')
                        files_written.append(rel_path)

            check_cancel(cancel_flag_path, emitter, "plots", manifest_path, manifest)

            # D. Per-Day Plots (Sig/Iso, dFF, Stacked) via Unified Bundle Driver
            cmd_bundle = [sys.executable, 'tools/plot_phasic_dayplot_bundle.py',
                          '--analysis-out', phasic_out,
                          '--roi', roi,
                          '--output-dir', d_dir,
                          '--sessions-per-hour', str(sessions_per_hour),
                          '--session-duration-s', str(session_duration_s),
                          '--smooth-window-s', str(args.smooth_window_s)]
                          
            if not has_features:
                cmd_bundle.extend(['--no-write-dff-grid', '--no-write-stacked'])
                
            manifest['commands'].append(run_cmd(cmd_bundle, roi_label=roi))

            # Collect Per-Day Files
            days_generated = set()
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

            # Stacked are already in d_dir, just verify
            days_stacked = set()
            for f in glob.glob(os.path.join(d_dir, "phasic_stacked_day_*.png")):
                files_written.append(f"day_plots/{os.path.basename(f)}")
                m = re.match(r'phasic_stacked_day_(\d+)\.png', os.path.basename(f))
                if m:
                    days_stacked.add(m.group(1))

            s_dff = sorted(list(days_dff))
            s_sig = sorted(list(days_sig_iso))
            s_stk = sorted(list(days_stacked))
            
            manifest['deliverables'][roi]['days_dff'] = s_dff
            manifest['deliverables'][roi]['days_sig_iso'] = s_sig
            manifest['deliverables'][roi]['days_stacked'] = s_stk

            # Consistency check (restored)
            if has_features and not (s_dff == s_sig == s_stk):
                 raise RuntimeError(f"Inconsistent day sets for ROI {roi}: DFF={s_dff}, SigIso={s_sig}, Stacked={s_stk}")

            manifest['deliverables'][roi]['files'] = sorted(list(set(files_written)))
            manifest['deliverables'][roi]['days_generated'] = s_dff if has_features else s_stk

            # ROI timing finalize
            elapsed_roi = time.perf_counter() - t_roi
            finished_utc_roi = _utc_now_iso()
            manifest['deliverables'][roi]['timing'] = {
                "started_utc": started_utc_roi,
                "finished_utc": finished_utc_roi,
                "elapsed_sec": elapsed_roi
            }
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
                                     sessions_per_hour=resolved_sessions_per_hour,
                                     sessions_per_hour_source=sessions_per_hour_source)
        err_payload = None
        if not ok:
            msg = "STRICT ORDERING VIOLATION: root run_report.json missing at terminal finalize"
            emitter.emit("package", "error", msg)
            err_payload = msg
            
        _phase_done(status_data, manifest, "finalize_artifacts", t_phase, started_utc_phase, status_path=status_path)

        # Final manifest update for end-to-end timing
        manifest["timing"]["total_runtime_sec"] = time.perf_counter() - t0_total
        _atomic_write_json(manifest_path, manifest)

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
                                          sessions_per_hour=resolved_sessions_per_hour,
                                          sessions_per_hour_source=sessions_per_hour_source)
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
        
        # ============================================================
        # Finalize Artifacts (Strict Ordering Gate) - Error path
        # ============================================================
        ok = False
        try:
             ok = _ensure_root_run_report(run_dir, phasic_out, tonic_out, emitter,
                                          sessions_per_hour=resolved_sessions_per_hour,
                                          sessions_per_hour_source=sessions_per_hour_source)
        except Exception:
             pass

        msg_body = str(e)
        if not ok:
            viol_msg = "STRICT ORDERING VIOLATION: root run_report.json missing at terminal finalize"
            if emitter:
                emitter.emit("package", "error", viol_msg)
            msg_body = f"{msg_body} | {viol_msg}"

        # --- Finalize Status: Error ---
        _finalize_status("error", error_msg=msg_body)
        raise SystemExit(1)

if __name__ == '__main__':
    main()
