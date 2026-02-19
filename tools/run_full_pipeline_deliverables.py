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


# ======================================================================
# Helpers
# ======================================================================

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    return cmd


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




# ======================================================================
# NDJSON Event Emitter
# ======================================================================

class EventEmitter:
    """Writes NDJSON events to events_path, one JSON object per line."""

    def __init__(self, events_path, run_id, run_dir, file_mode="a",
                 allow_makedirs=True):
        self._run_id = run_id
        self._run_dir = run_dir
        self._fh = None
        if events_path:
            parent = os.path.dirname(events_path) or "."
            if allow_makedirs:
                os.makedirs(parent, exist_ok=True)
                self._fh = open(events_path, file_mode, encoding="utf-8")
            else:
                # Side-effect free: only open if parent already exists
                if os.path.isdir(parent):
                    self._fh = open(events_path, file_mode, encoding="utf-8")
                # else: stay disabled (self._fh remains None)

    def emit(self, stage, event_type, message, **kwargs):
        """Emit one NDJSON event line. schema_version: 1 is forced."""
        raw_obj = {
            "time_iso": datetime.now().isoformat(),
            "run_id": self._run_id,
            "run_dir": self._run_dir,
            "stage": stage,
            "type": event_type,
            "message": message,
            **kwargs,
        }
        
        # Producer-side discipline: force schema_version: 1 (int)
        obj = _normalize_event_dict(raw_obj)
        
        if self._fh:
            # separators=(",", ":") for compact NDJSON (standard compliant)
            self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

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
    sys.exit(130)

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
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--sessions-per-hour', type=int, help="Force sessions per hour (integer)")
    parser.add_argument('--session-duration-s', type=float, help="Recording duration in seconds (data length per chunk). If provided, validated against traces.")
    parser.add_argument('--smooth-window-s', type=float, default=1.0)
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
        sys.exit(2)
    if not args.out and not args.out_base:
        print("Error: one of --out or --out-base is required.", file=sys.stderr)
        sys.exit(2)
    if args.run_id and not args.out_base:
        print("Error: --run-id requires --out-base.", file=sys.stderr)
        sys.exit(2)

    if args.out_base:
        # GUI mode
        run_id = args.run_id if args.run_id else _generate_run_id()
        # Validate run_id: no path separators
        if os.sep in run_id or (os.altsep and os.altsep in run_id):
            print(f"Error: --run-id must not contain path separators, got: {run_id}",
                  file=sys.stderr)
            sys.exit(2)
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

    # -- Resolve run directory --
    run_dir, run_id, is_gui_mode = resolve_run_dir(args)
    events_path, cancel_flag_path = resolve_paths(args, run_dir)
    manifest_path = os.path.join(run_dir, "MANIFEST.json")

    print(f"RUN_DIR: {run_dir}")

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
        if is_gui_mode:
            os.makedirs(run_dir, exist_ok=True)
        else:
            # Legacy mode: side-effect free policy.
            # a) --events auto: disable events entirely.
            # b) --events <explicit>: only write if parent dir already exists;
            #    never create directories.
            if args.events == "auto":
                vo_events_path = None
                print("VALIDATE-ONLY: events disabled (legacy --out mode, "
                      "--events auto); to enable, pass --events <PATH> to "
                      "an existing parent directory "
                      "(no directories will be created)")
            else:
                # Explicit path: check parent directory
                parent = os.path.dirname(events_path) or "."
                if not os.path.isdir(parent):
                    vo_events_path = None
                    print("VALIDATE-ONLY: events disabled (legacy --out mode); "
                          "parent directory does not exist and will not be "
                          "created because legacy validate-only creates "
                          "no directories")
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
                "run_id": run_id,
                "phase": "running",
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
                "sessions_per_hour": args.sessions_per_hour,
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
            if is_gui_mode:
                _vo_write_final_status("error", error_msg=str(e))
            print(str(e), file=sys.stderr)
            sys.exit(1)

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
        if args.sessions_per_hour is not None:
            argv.extend(["--sessions-per-hour", str(args.sessions_per_hour)])
        if args.session_duration_s is not None:
            argv.extend(["--session-duration-s", str(args.session_duration_s)])
        argv.extend(["--smooth-window-s", str(args.smooth_window_s)])

        print("VALIDATE-ONLY: OK")
        print(f"VALIDATE-ONLY: run_dir={run_dir}")
        print(f"VALIDATE-ONLY: events_path={vo_events_path}")
        print(f"VALIDATE-ONLY: cancel_flag_path={cancel_flag_path}")
        print(f"VALIDATE-ONLY: argv={json.dumps(argv)}")

        emitter.emit("engine", "done", "Validate-only complete")
        emitter.close()
        if is_gui_mode:
            _vo_write_final_status("success")
        sys.exit(0)

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
                sys.exit(1)

            # Robust rmtree
            max_retries = 5
            for i in range(max_retries):
                try:
                    shutil.rmtree(run_dir)
                    break
                except OSError as e:
                    if i == max_retries - 1:
                        print(f"Error: Failed to delete {run_dir} after retries: {e}")
                        sys.exit(1)
                    time.sleep(0.5 * (i + 1))

    os.makedirs(run_dir, exist_ok=True)

    # -- Open event emitter (write mode: fresh file per run) --
    # ============================================================
    # 2. Status & Emitter Setup
    # ============================================================

    # -- Status tracking (strict contract) --
    # Phase: "running" (no status field) -> "final" (status="success"|"error")
    status_data = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "running", 
        # "status": "..." # OMITTED per Design 1 while running
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
        "sessions_per_hour": args.sessions_per_hour,
        "events_mode": args.events,
        "outputs": {
            "manifest_json": manifest_path,
            "events_ndjson": events_path,
            "region_dirs": []
        },
        "errors": [],
        "warnings": [],
        "validation": None
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

    # Initial write (phase="running")
    _atomic_write_json(status_path, status_data)

    # -- Open event emitter --
    emitter = EventEmitter(events_path, run_id, run_dir, file_mode="w")
    emitter.emit("engine", "start", "Engine starting")

    try:
        # -- Check cancel immediately --
        check_cancel(cancel_flag_path, emitter, "engine_start", manifest_path, manifest)
    
        analysis_dir = os.path.join(run_dir, '_analysis')
        tonic_out = os.path.join(analysis_dir, 'tonic_out')
        phasic_out = os.path.join(analysis_dir, 'phasic_out')

        # ============================================================
        # 3. Validate
        # ============================================================
        emitter.emit("validate", "start", "Validating inputs")
        validate_inputs(args)
        emitter.emit("validate", "done", "Validation passed")

        check_cancel(cancel_flag_path, emitter, "validate", manifest_path, manifest)

        # ============================================================
        # 4. Tonic Analysis
        # ============================================================
        # Locate analyze_photometry.py relative to this script
        # This script is in tools/, analyze_photometry.py is in root/
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(tools_dir)
        analyze_script = os.path.join(root_dir, 'analyze_photometry.py')
        
        emitter.emit("tonic", "start", "Running tonic analysis")
        cmd_tonic = [sys.executable, analyze_script,
                     '--input', args.input,
                     '--out', tonic_out,
                     '--config', args.config,
                     '--mode', 'tonic',
                     '--format', args.format,
                     '--recursive', '--overwrite']
        manifest['commands'].append(run_cmd(cmd_tonic))
        emitter.emit("tonic", "done", "Tonic analysis complete")

        check_cancel(cancel_flag_path, emitter, "tonic", manifest_path, manifest)

        # ============================================================
        # 5. Phasic Analysis
        # ============================================================
        emitter.emit("phasic", "start", "Running phasic analysis")
        cmd_phasic = [sys.executable, analyze_script,
                      '--input', args.input,
                      '--out', phasic_out,
                      '--config', args.config,
                      '--mode', 'phasic',
                      '--format', args.format,
                      '--recursive', '--overwrite']
        manifest['commands'].append(run_cmd(cmd_phasic))
        emitter.emit("phasic", "done", "Phasic analysis complete")

        check_cancel(cancel_flag_path, emitter, "phasic", manifest_path, manifest)

        # ============================================================
        # 6. Session / Stride Computation
        # ============================================================
        trace_files = sorted(glob.glob(os.path.join(phasic_out, 'traces', 'chunk_*.csv')))
        if not trace_files:
            raise RuntimeError("No phases traces found.")

        # Inspect first trace for duration
        df0 = pd.read_csv(trace_files[0])
        if 'time_sec' not in df0.columns or len(df0) < 2:
             raise RuntimeError(f"Trace {trace_files[0]} missing time_sec or too short.")

        time_sec = df0['time_sec'].values
        trace_duration_s = time_sec[-1] - time_sec[0]

        if not np.isfinite(trace_duration_s) or trace_duration_s <= 0:
            raise RuntimeError(f"Invalid trace_duration_s: {trace_duration_s}")

        # Stride Inference
        stride_s = None

        if args.sessions_per_hour:
             sessions_per_hour = args.sessions_per_hour
             stride_s = 3600.0 / sessions_per_hour
        else:
             raise RuntimeError("Cannot infer session stride (duty-cycled acquisition) without --sessions-per-hour. Please provide it explicitly.")

        # Validation
        computed_sph = int(round(3600.0 / stride_s))
        if computed_sph < 1 or computed_sph > 12:
             raise RuntimeError(f"Invalid sessions per hour: {computed_sph}")

        if abs(stride_s * computed_sph - 3600.0) > 2.0:
             raise RuntimeError(f"Stride {stride_s:.2f}s not compatible with integer sessions/hr {computed_sph}")

        if args.session_duration_s is not None:
             if args.session_duration_s <= 0:
                  raise RuntimeError(f"Provided session duration must be > 0, got {args.session_duration_s}")

             # Validate against trace
             tol = max(2.0, 0.005 * args.session_duration_s)
             diff = abs(trace_duration_s - args.session_duration_s)
             if diff > tol:
                  raise RuntimeError(f"Session Duration Mismatch! Provided: {args.session_duration_s:.2f}s, Trace: {trace_duration_s:.2f}s (Diff: {diff:.2f}s, Tol: {tol:.2f}s). File: {trace_files[0]}")

             session_duration_s = args.session_duration_s
        else:
             session_duration_s = trace_duration_s

        # Impossible Schedule Check
        if session_duration_s > (stride_s + 1e-6):
             raise RuntimeError(f"Impossible schedule: Duration {session_duration_s:.2f}s > Stride {stride_s:.2f}s (SPH={sessions_per_hour}).")

        sessions_per_hour = computed_sph
        manifest['sessions_per_hour'] = sessions_per_hour
        manifest['session_duration_s'] = session_duration_s
        manifest['session_stride_s'] = stride_s
        print(f"Deterministic Sessions Per Hour: {sessions_per_hour} (Stride={stride_s:.1f}s, Dur={session_duration_s:.1f}s)")

        check_cancel(cancel_flag_path, emitter, "session_compute", manifest_path, manifest)

        # ============================================================
        # 7. Per-Region Processing (Plots & Packaging)
        # ============================================================
        emitter.emit("plots", "start", "Generating per-ROI deliverables")

        feats_csv = os.path.join(phasic_out, 'features', 'features.csv')
        df_feat = pd.read_csv(feats_csv)
        regions = sorted(df_feat['roi'].unique())
        manifest['regions'] = regions

        for roi in regions:
            check_cancel(cancel_flag_path, emitter, "plots", manifest_path, manifest)

            print(f"Processing ROI: {roi}")
            emitter.emit("plots", "progress", f"Processing ROI: {roi}", roi=roi)
            reg_dir = os.path.join(run_dir, roi)
            os.makedirs(reg_dir, exist_ok=True)
            files_written = []

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
                          '--out', os.path.join(reg_dir, "phasic_correction_impact.png")]
            manifest['commands'].append(run_cmd(cmd_impact))
            files_written.append("phasic_correction_impact.png")

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
                df_c[keep].to_csv(os.path.join(reg_dir, "phasic_correction_impact_session.csv"), index=False)
                files_written.append("phasic_correction_impact_session.csv")

            # B. Tonic Overview
            cmd_tonic_roi = [sys.executable, 'tools/plot_tonic_48h.py',
                             '--analysis-out', tonic_out, '--roi', roi]
            run_cmd(cmd_tonic_roi)

            src_tonic = os.path.join(tonic_out, 'tonic_qc', f"tonic_48h_overview_{roi}.png")
            if os.path.exists(src_tonic):
                shutil.copy2(src_tonic, os.path.join(reg_dir, "tonic_overview.png"))
                files_written.append("tonic_overview.png")

            # Tonic CSV
            tonic_files = sorted(glob.glob(os.path.join(tonic_out, 'traces', 'chunk_*.csv')))
            t_rows = []
            fs_tonic = 20.0

            for i, tf in enumerate(tonic_files):
                tdf = pd.read_csv(tf)
                col_d = f"{roi}_deltaF"
                if col_d in tdf.columns:
                    n_pts = len(tdf)
                    t_abs = (i * stride_s) + tdf['time_sec'].values
                    df_sub = pd.DataFrame({
                        'time_hours': t_abs / 3600.0,
                        'tonic_df': tdf[col_d].values
                    })
                    t_rows.append(df_sub)

            if t_rows:
                full_tonic = pd.concat(t_rows, ignore_index=True)
                full_tonic.to_csv(os.path.join(reg_dir, "tonic_df_timeseries.csv"), index=False)
                files_written.append("tonic_df_timeseries.csv")

            # C. Phasic Time Series (Plots & CSV)
            ts_dir = os.path.join(phasic_out, f'viz_{roi}')
            cmd_ts = [sys.executable, 'tools/plot_phasic_time_series_summary.py',
                      '--analysis-out', phasic_out,
                      '--roi', roi,
                      '--sessions-per-hour', str(sessions_per_hour),
                      '--session-duration-s', str(session_duration_s),
                      '--out-dir', ts_dir,
                      '--export-csv']
            manifest['commands'].append(run_cmd(cmd_ts))

            # Copy Results
            pairs = [
                ("fig_phasic_peak_rate_timeseries.png", "phasic_peak_rate_timeseries.png"),
                ("fig_phasic_auc_timeseries.png", "phasic_auc_timeseries.png"),
                ("phasic_peak_rate_timeseries.csv", "phasic_peak_rate_timeseries.csv"),
                ("phasic_auc_timeseries.csv", "phasic_auc_timeseries.csv")
            ]
            for src_name, dst_name in pairs:
                s = os.path.join(ts_dir, src_name)
                if os.path.exists(s):
                    shutil.copy2(s, os.path.join(reg_dir, dst_name))
                    files_written.append(dst_name)

            check_cancel(cancel_flag_path, emitter, "plots", manifest_path, manifest)

            # D. Per-Day Plots (Sig/Iso, dFF, Stacked)

            # 1. dFF Grid
            qc_dir = os.path.join(phasic_out, f'qc_dff_{roi}')
            cmd_qc = [sys.executable, 'tools/plot_phasic_qc_grid.py',
                      '--analysis-out', phasic_out,
                      '--roi', roi,
                      '--mode', 'dff',
                      '--sessions-per-hour', str(sessions_per_hour),
                      '--output-dir', qc_dir]
            run_cmd(cmd_qc)

            # 2. Sig/Iso Grid
            sess_dir = os.path.join(phasic_out, f'session_qc_{roi}')
            cmd_sess = [sys.executable, 'tools/plot_session_grid.py',
                        '--analysis-out', phasic_out,
                        '--roi', roi,
                        '--sessions-per-hour', str(sessions_per_hour),
                        '--session-duration-s', str(session_duration_s)]
            run_cmd(cmd_sess)

            # 3. Stacked
            cmd_stack = [sys.executable, 'tools/plot_phasic_stacked_day_smoothed.py',
                         '--analysis-out', phasic_out,
                         '--roi', roi,
                         '--out-dir', reg_dir,
                         '--sessions-per-hour', str(sessions_per_hour),
                         '--smooth-window-s', str(args.smooth_window_s)]
            manifest['commands'].append(run_cmd(cmd_stack))

            # Collect Per-Day Files
            days_generated = set()
            days_dff = set()
            days_sig_iso = set()

            # Copy dFF
            qc_dir_roi = os.path.join(phasic_out, f'qc_dff_{roi}')
            for f in glob.glob(os.path.join(qc_dir_roi, "day_*.png")):
                m = re.match(r'day_(\d+)\.png', os.path.basename(f))
                if m:
                    day_idx = m.group(1)
                    dst = f"phasic_dFF_day_{day_idx}.png"
                    shutil.copy2(f, os.path.join(reg_dir, dst))
                    files_written.append(dst)
                    days_dff.add(day_idx)

            # Copy Sig/Iso
            sess_out_base = os.path.join(phasic_out, 'session_qc')
            for f in glob.glob(os.path.join(sess_out_base, f"day_*_raw_iso_{roi}.png")):
                 m = re.match(r'day_(\d+)_raw_iso_', os.path.basename(f))
                 if m:
                     day_idx = m.group(1)
                     dst = f"phasic_sig_iso_day_{day_idx}.png"
                     shutil.copy2(f, os.path.join(reg_dir, dst))
                     files_written.append(dst)
                     days_sig_iso.add(day_idx)

            # Stacked are already in reg_dir, just verify
            days_stacked = set()
            for f in glob.glob(os.path.join(reg_dir, "phasic_stacked_day_*.png")):
                files_written.append(os.path.basename(f))
                m = re.match(r'phasic_stacked_day_(\d+)\.png', os.path.basename(f))
                if m:
                    days_stacked.add(m.group(1))

            # Sort sets for consistency check
            s_dff = sorted(list(days_dff))
            s_sig = sorted(list(days_sig_iso))
            s_stk = sorted(list(days_stacked))

            manifest['deliverables'][roi]['days_dff'] = s_dff
            manifest['deliverables'][roi]['days_sig_iso'] = s_sig
            manifest['deliverables'][roi]['days_stacked'] = s_stk

            if not (s_dff == s_sig == s_stk):
                 raise RuntimeError(f"Inconsistent day sets for ROI {roi}: DFF={s_dff}, SigIso={s_sig}, Stacked={s_stk}")

            manifest['deliverables'][roi]['files'] = sorted(list(set(files_written)))
            manifest['deliverables'][roi]['days_generated'] = s_dff

        emitter.emit("plots", "done", "All ROI deliverables complete")

        check_cancel(cancel_flag_path, emitter, "package", manifest_path, manifest)

        # ============================================================
        # 8. Write Manifest (LAST, atomic)
        # ============================================================
        emitter.emit("package", "start", "Writing final manifest")
        _atomic_write_json(manifest_path, manifest)
        emitter.emit("package", "done", "Manifest written")

        emitter.emit("engine", "done", "Deliverables package complete")
        emitter.close()
        print("Deliverables Package Complete.")
        
        # --- Finalize Status: Success ---
        _finalize_status("success")

    except SystemExit:
        # Re-raise sys.exit calls (from check_cancel) without catching them
        _finalize_status("cancelled", error_msg="CANCELLED")
        raise
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

        # Write failed manifest
        try:
            _atomic_write_json(manifest_path, manifest)
        except Exception:
            pass  # best effort

        emitter.emit("engine", "error", str(e), error_code="EXCEPTION")
        emitter.close()
        
        # --- Finalize Status: Error ---
        _finalize_status("error", error_msg=str(e))
        sys.exit(1)

if __name__ == '__main__':
    main()
