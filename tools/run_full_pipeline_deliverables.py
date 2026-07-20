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
    from photometry_pipeline.io.adapters import (
        load_chunk,
        sniff_format,
        estimate_continuous_source_duration,
        plan_continuous_windows_for_source,
    )
    from photometry_pipeline.guided_manifest_current_facts import (
        build_guided_manifest_current_facts,
    )
    from photometry_pipeline.guided_manifest_verification import (
        GuidedManifestCliContext,
        load_guided_candidate_manifest,
        verify_guided_candidate_manifest_consumption,
    )
    from photometry_pipeline.guided_new_analysis_plan import (
        FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES,
    )
    from photometry_pipeline.guided_normalized_recording import (
        NormalizedRecordingError,
        deserialize_normalized_recording_description,
    )
    from photometry_pipeline.guided_startup_claim import (
        claim_guided_preallocated_startup,
        validate_guided_preallocated_startup,
    )
    from photometry_pipeline.guided_startup_transaction import (
        GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME,
        GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME,
    )
    from photometry_pipeline.run_completion_contract import (
        COMPLETION_KEY,
        FAMILY_PHASIC_DAY_PLOTS,
        FAMILY_PHASIC_TIMESERIES,
        FAMILY_TONIC_OVERVIEW,
        FAMILY_TONIC_TIMESERIES,
        PROFILE_CONTINUOUS,
        PROFILE_FULL_INTERMITTENT,
        PROFILE_TUNING_PREP,
        RunCompletionError,
        build_continuous_window_index,
        build_manifest_completion_block,
        build_report_completion_block,
        build_status_completion_block,
        normalize_run_mode,
        sha256_file,
        verify_terminal_set_before_status,
    )
except ImportError:
    print("ERROR: Could not import photometry_pipeline. Ensure script is in tools/ and repo root is accessible.", flush=True)
    raise SystemExit(1)


# ======================================================================
# Helpers
# ======================================================================

# Test-only in-process seam. It is not configurable from CLI and is inactive
# in production. Tests use it to stop after initial status, before analysis.
_GUIDED_TEST_STOP_AFTER_INITIAL_STATUS = None

# Test-only in-process seam. Inactive in production. Failure-injection tests
# call it at each named finalization checkpoint to assert that a crash there
# can never leave a directory that reloads as a successful run.
_TEST_FINALIZATION_HOOK = None


def _finalization_checkpoint(name):
    if _TEST_FINALIZATION_HOOK is not None:
        _TEST_FINALIZATION_HOOK(name)


# Optional artifacts recorded in the final manifest when present. Their absence
# never blocks completion; they are listed so the manifest describes the whole
# output package rather than only its mandatory core.
OPTIONAL_MANIFEST_ARTIFACTS = [
    "events.ndjson",
    "config_effective.yaml",
    "gui_run_spec.json",
]


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


def _ensure_phase_timing_containers(status_data):
    timing = status_data.setdefault("timing", {})
    timing.setdefault("phase_history", [])
    timing.setdefault("phase_elapsed_sec", {})
    return timing


def _phase_start(status_data, phase_name, emitter=None):
    """Record phase start in status and print log."""
    now_utc = _utc_now_iso()
    timing = _ensure_phase_timing_containers(status_data)
    timing["current_phase"] = phase_name
    timing["phase_started_utc"] = now_utc
    if emitter is not None:
        emitter.emit(
            "timing",
            "timing_start",
            f"Timing started for phase: {phase_name}",
            payload={"phase": phase_name, "started_utc": now_utc},
        )
    print(f"TIMING START phase={phase_name} at {now_utc}", flush=True)
    return time.perf_counter(), now_utc


def _phase_done(status_data, manifest, phase_name, t0, started_utc, status_path=None, emitter=None):
    """Record phase completion in status and manifest, and print log."""
    elapsed = time.perf_counter() - t0
    finished_utc = _utc_now_iso()
    record = {
        "phase": phase_name,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_sec": elapsed,
    }
    
    # Update status
    timing = _ensure_phase_timing_containers(status_data)
    timing["last_completed_phase"] = phase_name
    timing["last_phase_elapsed_sec"] = elapsed
    timing["current_phase"] = None
    timing["phase_started_utc"] = None
    timing["phase_history"].append(dict(record))
    timing["phase_elapsed_sec"][phase_name] = elapsed
    
    # Update manifest
    if "timing" not in manifest:
        manifest["timing"] = {"phases": {}}
    manifest["timing"]["phases"][phase_name] = dict(record)
    
    if status_path:
        _write_status_json(status_path, status_data)

    if emitter is not None:
        emitter.emit(
            "timing",
            "timing_done",
            f"Timing completed for phase: {phase_name}",
            payload=dict(record),
        )
    
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
INTERMITTENT_ONLY_OUTPUT_KEYS = [
    "session-slot dayplots",
    "stacked session views",
    "duty-cycle occupancy plots",
    "anchored phasic summaries requiring sessions_per_hour",
    "outputs requiring duration <= stride assumptions",
    "dayplot rerender paths requiring sessions-per-hour metadata",
]
CONTINUOUS_NPM_UNSUPPORTED_MESSAGE = (
    "Continuous acquisition mode is not yet implemented for NPM/interleaved inputs."
)
CONTINUOUS_AUTO_FORMAT_MESSAGE = (
    "Continuous mode with --format auto is ambiguous for mixed/unknown inputs. "
    "Use --format rwd or --format custom_tabular."
)
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


def intermittent_only_output_message() -> str:
    """Stable user-facing message for intermittent-only output families."""
    return (
        "This output is designed for intermittent/session-based recordings and is "
        "not available in continuous mode. Use continuous elapsed-time outputs when "
        "available."
    )


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


def _continuous_row_counts_from_writer(summary_result):
    """Per-family, per-ROI window-row counts, as continuous_outputs reported them.

    Read from the writer's own return value, never by re-scanning the output
    directory, so the index describes what was written rather than what survived.
    """
    from photometry_pipeline.run_completion_contract import (
        FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY,
        FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY,
    )

    counts = {}
    for kind, family in (
        ("phasic", FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY),
        ("tonic", FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY),
    ):
        sub = (summary_result or {}).get(kind) or {}
        row_counts = sub.get("row_counts") or {}
        counts[family] = {
            str(roi): int(count)
            for roi, count in row_counts.items()
            if roi != "all_rois"
        }
    return counts


def _freeze_run_input_manifest(*, input_dir, config_path, run_dir, force_format):
    """Freeze one ordered production input manifest for the whole run.

    Uses the Pipeline's own discovery so the frozen ordered set and authorized
    exclusion are identical to what each analysis subprocess resolves; each
    subprocess additionally re-verifies its discovery against this manifest.
    Returns the written manifest path.
    """
    import os as _os

    from photometry_pipeline.config import Config as _Config
    from photometry_pipeline.pipeline import Pipeline as _Pipeline
    from photometry_pipeline.input_processing_completeness import (
        POLICY_INCOMPLETE_FINAL_RWD_CHUNK as _POLICY,
        build_session_index as _build,
        write_frozen_input_manifest as _write,
    )

    def _norm(path):
        return _os.path.normcase(_os.path.abspath(_os.path.normpath(str(path))))

    cfg = _Config.from_yaml(config_path)
    probe = _Pipeline(cfg, mode="phasic")
    try:
        probe.discover_files(input_dir, recursive=True, force_format=force_format)
    except ValueError:
        # An empty/non-RWD input is still rejected by the analysis subprocess;
        # there is no admitted set to freeze here.  Defer that clear failure so
        # command construction and diagnostics retain their historical order.
        # Never defer when the user requested an authorized omission, because
        # that request itself requires a discovered, timestamped source.
        if getattr(cfg, "authorized_missing_sessions", None):
            raise
        return None
    # Continuous inputs carry their own window index; do not freeze here.
    if probe._is_continuous_mode_enabled():
        return None

    ordered_admitted = list(probe.file_list)
    excluded = probe._authorized_exclusion
    ordered_sources = ordered_admitted + ([excluded] if excluded is not None else [])

    # Resolve ``auto`` before writing the shared index.  Missing-session
    # authorization is deliberately supported only for validated timestamped
    # RWD folders; passing the literal ``auto`` here would weaken that gate.
    try:
        resolved_input_format = probe._get_format(ordered_admitted[0], force_format)
    except Exception:
        resolved_input_format = str(force_format)

    missing_norm = sorted(
        {_norm(p) for p in (getattr(cfg, "authorized_missing_sessions", []) or []) if str(p).strip()}
    )
    expected_duration = float(getattr(cfg, "chunk_duration_sec", 0.0)) or None

    manifest = _build(
        acquisition_mode="intermittent",
        input_format=str(resolved_input_format),
        ordered_sources=ordered_sources,
        missing_sources=missing_norm,
        excluded_source=excluded,
        exclusion_policy=(_POLICY if excluded is not None else ""),
        expected_duration_sec=expected_duration,
    )
    return _write(run_dir, manifest)


def _regions_from_run_report(report_path):
    """The ROI set an analysis declared it selected, as recorded in its run report.

    This is the analysis's own statement of what it processed. It is never the
    set of region directories that happen to exist, so it cannot be changed by
    deleting a deliverable.
    """
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
    acquisition_mode="intermittent",
    continuous_window_sec=600.0,
    continuous_step_sec=600.0,
    allow_partial_final_window=False,
    acquisition_mode_source=None,
    timeline_anchor_mode="civil",
    fixed_daily_anchor_clock=None,
    run_id=None,
):
    """
    Ensure <run_dir>/run_report.json exists at root before terminal status.
    Ordered requirement for Step 8: Strict Ordering Gate.

    When run_id is supplied the report is stamped with the completion contract it
    satisfies, so a reader can tell a run produced by this build (which MUST have
    a coherent terminal set) from a positively-identified historical run.

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
                    # Stage then replace: an interrupted copy must never leave a
                    # truncated report where a complete one is expected.
                    staged = report_path + ".tmp"
                    shutil.copy2(src, staged)
                    os.replace(staged, report_path)
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
                 run_ctx['acquisition_mode'] = acquisition_mode
                 run_ctx['continuous_window_sec'] = continuous_window_sec
                 run_ctx['continuous_step_sec'] = continuous_step_sec
                 run_ctx['allow_partial_final_window'] = allow_partial_final_window
                 run_ctx['acquisition_mode_source'] = acquisition_mode_source
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
                 derived_settings['acquisition_mode'] = acquisition_mode
                 derived_settings['continuous_window_sec'] = continuous_window_sec
                 derived_settings['continuous_step_sec'] = continuous_step_sec
                 derived_settings['allow_partial_final_window'] = allow_partial_final_window
                 derived_settings['acquisition_mode_source'] = acquisition_mode_source
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

             # Declare the completion contract this run is held to. Present on
             # every terminal path (success, error, cancel) so a damaged current
             # run is never silently downgraded to "legacy" on reload.
             if run_id:
                 repo["completion_contract"] = build_report_completion_block(run_id=run_id)

             _atomic_write_json(report_path, repo)

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
    parser.add_argument('--format', required=True, choices=['rwd', 'npm', 'custom_tabular', 'auto'])
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
        '--acquisition-mode',
        choices=['intermittent', 'continuous'],
        default=None,
        help="Acquisition structure: intermittent/session-based or continuous recording.",
    )
    parser.add_argument(
        '--continuous-window-sec',
        type=float,
        default=None,
        help="Continuous mode window duration in seconds.",
    )
    parser.add_argument(
        '--continuous-step-sec',
        type=float,
        default=None,
        help="Continuous mode step duration in seconds (phase 1 requires step == window).",
    )
    partial_group = parser.add_mutually_exclusive_group()
    partial_group.add_argument(
        '--allow-partial-final-window',
        dest='allow_partial_final_window',
        action='store_true',
        help="Continuous mode only: include a trailing undersized final window.",
    )
    partial_group.add_argument(
        '--no-allow-partial-final-window',
        dest='allow_partial_final_window',
        action='store_false',
        help="Continuous mode only: drop a trailing undersized final window.",
    )
    parser.set_defaults(allow_partial_final_window=None)
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
    parser.add_argument(
        '--export-display-series-csv',
        action='store_true',
        help=(
            "Advanced export: write long-format CSVs of plotted display-series data "
            "for selected figures. Off by default."
        ),
    )
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
    parser.add_argument(
        '--guided-candidate-manifest',
        default=None,
        help="Internal/backend use only: exact Guided candidate manifest.",
    )
    parser.add_argument(
        '--guided-preallocated-run-dir',
        action='store_true',
        help=(
            "Internal/backend use only: claim a Guided startup-preallocated "
            "run directory."
        ),
    )
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
    if args.format not in ('rwd', 'npm', 'custom_tabular', 'auto'):
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
    acquisition_mode = getattr(args, "acquisition_mode", "intermittent")
    continuous_window_sec = getattr(args, "continuous_window_sec", 600.0)
    continuous_step_sec = getattr(args, "continuous_step_sec", 600.0)
    if acquisition_mode not in {"intermittent", "continuous"}:
        raise RuntimeError(
            "acquisition_mode must be 'intermittent' or 'continuous'."
        )
    if continuous_window_sec is None or float(continuous_window_sec) <= 0:
        raise RuntimeError(
            f"continuous_window_sec must be > 0, got {continuous_window_sec}"
        )
    if continuous_step_sec is None or float(continuous_step_sec) <= 0:
        raise RuntimeError(
            f"continuous_step_sec must be > 0, got {continuous_step_sec}"
        )
    if abs(float(continuous_step_sec) - float(continuous_window_sec)) > 1e-9:
        raise RuntimeError(
            "continuous_step_sec must equal continuous_window_sec in this version; "
            "overlapping/sliding windows are not yet supported."
        )

    if acquisition_mode == "continuous":
        try:
            cfg = Config.from_yaml(args.config)
        except Exception as e:
            raise RuntimeError(
                f"Could not parse config for continuous-mode validation: {e}"
            ) from e
        cfg.acquisition_mode = "continuous"
        cfg.continuous_window_sec = float(continuous_window_sec)
        cfg.continuous_step_sec = float(continuous_step_sec)
        cfg.allow_partial_final_window = bool(
            getattr(args, "allow_partial_final_window", cfg.allow_partial_final_window)
        )
        _resolve_continuous_format(str(args.input), str(args.format), cfg)

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

    # Deep contract validation is intentionally limited to validate-only
    # custom_tabular runs, so full-run preflight remains cheap.
    if (
        str(getattr(args, "format", "")).strip().lower() == "custom_tabular"
        and bool(getattr(args, "validate_only", False))
    ):
        _validate_custom_tabular_contract_validate_only(
            input_dir=str(args.input),
            config_path=str(args.config),
        )


def verify_guided_manifest_before_output(args):
    """Verify internal Guided manifest mode before run-dir resolution/allocation."""
    if not getattr(args, "guided_candidate_manifest", None):
        return None
    loaded = load_guided_candidate_manifest(args.guided_candidate_manifest)
    if not loaded.accepted or loaded.manifest is None:
        detail = (
            loaded.blocking_issues[0].category
            if loaded.blocking_issues
            else "guided_manifest_load_failed"
        )
        raise RuntimeError(f"Guided manifest verification refused: {detail}")
    manifest = loaded.manifest
    cfg = Config.from_yaml(args.config)
    acquisition_mode = (
        args.acquisition_mode
        if args.acquisition_mode is not None
        else getattr(cfg, "acquisition_mode", "intermittent")
    )
    if acquisition_mode != "intermittent":
        raise RuntimeError("Guided manifest execution requires intermittent acquisition.")
    if str(getattr(cfg, "dynamic_fit_mode", "")) not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES:
        raise RuntimeError("Guided manifest execution has unsupported dynamic-fit mode.")
    effective_preview_first_n = (
        args.preview_first_n
        if args.preview_first_n is not None
        else getattr(cfg, "preview_first_n", None)
    )
    facts = build_guided_manifest_current_facts(
        source_root=args.input,
        config=cfg,
        manifest_included_roi_ids=manifest.included_roi_ids,
        source_format=args.format,
    )
    include_rois = (
        tuple(item.strip() for item in args.include_rois.split(",") if item.strip())
        if args.include_rois
        else None
    )
    exclude_rois = (
        tuple(item.strip() for item in args.exclude_rois.split(",") if item.strip())
        if args.exclude_rois
        else ()
    )
    verified = verify_guided_candidate_manifest_consumption(
        manifest=manifest,
        source_root=args.input,
        current_candidates=facts.current_candidates,
        current_roi_inventory=facts.current_roi_inventory,
        cli_context=GuidedManifestCliContext(
            input_format=args.format,
            mode=args.mode,
            run_type=args.run_type,
            traces_only=bool(args.traces_only),
            discover=bool(args.discover),
            validate_only=bool(args.validate_only),
            overwrite=bool(args.overwrite),
            preview_first_n=effective_preview_first_n,
            requested_include_rois=include_rois,
            requested_exclude_rois=exclude_rois,
        ),
    )
    if not verified.accepted:
        detail = (
            verified.blocking_issues[0].category
            if verified.blocking_issues
            else "guided_manifest_verification_failed"
        )
        raise RuntimeError(f"Guided manifest verification refused: {detail}")
    return facts, verified


def verify_guided_normalized_recording_description_before_output(args, facts, verified):
    """Cross-check the persisted normalized recording description against
    the same freshly-verified candidate/ROI facts the manifest check just
    established -- no second filesystem scan.

    Runs immediately after ``verify_guided_manifest_before_output`` at its
    one call site, reusing ``facts``/``verified`` already computed there.
    By the time this runs, ``validate_guided_preallocated_startup`` (called
    earlier, at the preallocated-mode gate) has already proven
    ``guided_normalized_recording_description.json`` is byte-identical to
    what Setup-check authorization and startup materialization wrote --
    this function checks that its *content* still matches the live source
    tree the manifest check just re-verified, catching drift between
    Setup-check authorization and wrapper launch for exactly the facts
    that are freshly re-derivable at this boundary. Session_duration_sec,
    cadence, and parser fields have no independent fresh source here; their
    integrity is already proven by the same startup-artifact hash chain
    that protects config_effective.yaml.
    """
    if facts is None or verified is None:
        return None
    normalized_path = os.path.join(
        args.out, GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME
    )
    try:
        with open(normalized_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Guided normalized recording description is unreadable: {exc}"
        ) from exc
    try:
        requested = deserialize_normalized_recording_description(payload)
    except NormalizedRecordingError as exc:
        raise RuntimeError(
            f"Guided normalized recording description verification refused: {exc}"
        ) from exc

    verified_paths = {
        item.canonical_relative_path for item in verified.verified_candidates
    }
    for session in requested.sessions:
        if session.disposition in ("process", "excluded"):
            if session.stable_source_identity not in verified_paths:
                raise RuntimeError(
                    "Guided normalized recording description verification refused: "
                    f"session {session.stable_source_identity!r} is no longer present "
                    "among the live source candidates."
                )

    discovered = set(facts.current_roi_inventory.discovered_roi_ids)
    included = set(facts.current_roi_inventory.included_roi_ids)
    requested_included = {
        item.roi_id for item in requested.roi_channels if item.included
    }
    requested_discovered = {item.roi_id for item in requested.roi_channels}
    if requested_discovered != discovered:
        raise RuntimeError(
            "Guided normalized recording description verification refused: "
            "the discovered ROI inventory no longer matches the authorized "
            "recording description."
        )
    if requested_included != included:
        raise RuntimeError(
            "Guided normalized recording description verification refused: "
            "the included ROI selection no longer matches the authorized "
            "recording description."
        )
    return requested


def validate_guided_preallocated_mode_args(args):
    """Validate the internal preallocated handoff flags without writing."""
    if not getattr(args, "guided_preallocated_run_dir", False):
        return None
    # The prepared command is the immutable startup authority.  A caller that
    # mutates an otherwise-supported analysis mode after Setup check must be
    # rejected as an internal handoff conflict, rather than being allowed to
    # reach the claim validator as a merely different-but-valid mode.
    out_value = getattr(args, "out", None)
    if out_value:
        command_path = Path(out_value) / "command_invoked.txt"
        try:
            command_values = command_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            command_values = []
    else:
        command_values = []
    mode_positions = [
        index for index, value in enumerate(command_values) if value == "--mode"
    ]
    if (
        len(mode_positions) == 1
        and mode_positions[0] + 1 < len(command_values)
        and command_values[mode_positions[0] + 1] != getattr(args, "mode", None)
    ):
        raise RuntimeError(
            "Guided preallocated startup handoff refused: analysis mode does not "
            "match the prepared startup command."
        )
    conflicts = (
        (not getattr(args, "guided_candidate_manifest", None), "manifest required"),
        (not getattr(args, "out", None), "--out required"),
        (bool(getattr(args, "out_base", None)), "--out-base prohibited"),
        (bool(getattr(args, "overwrite", False)), "overwrite prohibited"),
        (
            getattr(args, "format", None) not in {"rwd", "npm"},
            "a supported input format is required",
        ),
        (
            getattr(args, "mode", None) not in {"phasic", "tonic", "both"},
            "supported analysis mode required",
        ),
        (getattr(args, "run_type", None) != "full", "full run type required"),
        (
            getattr(args, "preview_first_n", None) is not None,
            "preview prohibited",
        ),
        (bool(getattr(args, "discover", False)), "discover prohibited"),
        (bool(getattr(args, "validate_only", False)), "validate-only prohibited"),
        (bool(getattr(args, "traces_only", False)), "traces-only prohibited"),
        (
            getattr(args, "acquisition_mode", None)
            not in (None, "intermittent"),
            "continuous acquisition prohibited",
        ),
        (getattr(args, "include_rois", None) is not None, "ROI override prohibited"),
        (getattr(args, "exclude_rois", None) is not None, "ROI override prohibited"),
    )
    for failed, detail in conflicts:
        if failed:
            raise RuntimeError(
                f"Guided preallocated startup handoff refused: {detail}."
            )
    return validate_guided_preallocated_startup(
        input_dir=args.input,
        output_dir=args.out,
        config_path=args.config,
        manifest_path=args.guided_candidate_manifest,
        expected_mode=args.mode,
    )


def _append_guided_manifest_to_analysis_command(cmd, args, *, mode):
    """Thread one authorized native map into both analysis branches."""
    manifest_path = getattr(args, "guided_candidate_manifest", None)
    if manifest_path and mode in {"phasic", "tonic"}:
        cmd.extend(["--guided-candidate-manifest", manifest_path])


def _dayplot_provenance_args(feature_provenance_mode, feature_provenance_path):
    """Exact provenance flags handed to every per-ROI day-plot process.

    The classification is made once by this wrapper; the child never infers it.
    A current run always carries the record path alongside the mode.
    """
    args = ['--provenance-mode', str(feature_provenance_mode)]
    if feature_provenance_path:
        args.extend(['--feature-event-provenance', str(feature_provenance_path)])
    return args


def _resolve_feature_provenance_for_plots(
    phasic_out, plotted_rois, *, require_current=True, emitter=None
):
    """Classify the analysis output once and validate its per-ROI settings record.

    Returns (mode, provenance_path).

    This wrapper just invoked the current analysis build, so `require_current` is
    True on the production path: the run MUST declare the exact supported contract
    version and MUST carry a complete, digest-consistent record. A missing or
    malformed run_report.json, an absent or unsupported contract version, a
    missing provenance file, a missing ROI entry, or any digest mismatch fails the
    run here -- it is never downgraded to "legacy" and never verified against the
    global config_used.yaml.

    `require_current=False` is only for intentionally processing a run that is
    POSITIVELY recognized as predating the contract. Even then an unknown
    classification fails closed; absence of metadata is not a legacy signal.

    The record is additionally bound to config_used.yaml: the recorded global
    Default digest must match the loaded global configuration, and every Default
    ROI must carry exactly that digest.
    """
    from photometry_pipeline.feature_event_provenance import (
        FeatureEventProvenanceError,
        PROVENANCE_MODE_CURRENT,
        PROVENANCE_MODE_LEGACY,
        PROVENANCE_MODE_UNKNOWN,
        classify_provenance_contract,
        load_feature_event_provenance,
        resolve_roi_entry,
        verify_global_default_identity,
    )

    mode, provenance_path, reason = classify_provenance_contract(phasic_out)

    if mode == PROVENANCE_MODE_UNKNOWN:
        raise RuntimeError(
            "Detector-aware day plotting refused: the analysis output does not "
            f"positively identify its feature-settings contract ({reason}). "
            "Refusing to fall back to the global configuration."
        )

    if mode == PROVENANCE_MODE_LEGACY:
        if require_current:
            raise RuntimeError(
                "Detector-aware day plotting refused: this run was produced by the "
                "current analysis build but its report does not declare the current "
                "per-ROI feature-settings contract."
            )
        if emitter is not None:
            emitter.emit(
                "plots",
                "audit",
                "Legacy analysis output: no per-ROI feature-settings record exists. "
                "Day plots use the global configuration and make no ROI-specific "
                "verification claim.",
                payload={"feature_provenance_mode": PROVENANCE_MODE_LEGACY},
            )
        return PROVENANCE_MODE_LEGACY, None

    try:
        payload = load_feature_event_provenance(provenance_path)
        global_config = Config.from_yaml(os.path.join(phasic_out, "config_used.yaml"))
        verify_global_default_identity(payload, global_config)
        for roi in plotted_rois:
            resolve_roi_entry(payload, roi)
    except FeatureEventProvenanceError as exc:
        raise RuntimeError(
            "Detector-aware day plotting refused: this run declares the current "
            f"per-ROI feature-settings contract but its record is unusable: {exc}"
        ) from exc

    if emitter is not None:
        emitter.emit(
            "plots",
            "audit",
            "Per-ROI feature-settings record verified for all plotted ROIs.",
            payload={
                "feature_provenance_mode": PROVENANCE_MODE_CURRENT,
                "roi_count": len(payload.get("rois", [])),
            },
        )
    return PROVENANCE_MODE_CURRENT, provenance_path


def _load_guided_per_roi_feature_event_overrides(run_dir):
    """Load complete effective per-ROI feature-config fields for Custom ROIs.

    guided_startup_materialization.py writes GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME
    into the Guided run directory only when the plan has at least one Custom
    (source="override") ROI. Returns None when the artifact is absent (a
    global-only Guided run, or a plain non-Guided invocation).

    Returns the artifact's per_roi_effective_feature_config_fields_for_overrides
    mapping -- COMPLETE effective FEATURE_EVENT_CONFIG_FIELDS dicts, never the
    sparse per_roi_override_config_fields.

    Note: this wrapper's own retired "Guided Post-Phasic Applied-dF/F
    Orchestration" stage (formerly Section 7.5) was this helper's only
    caller and has been removed; the function is retained as a
    general-purpose, independently-tested loader for the native
    per-ROI feature-config artifact.
    """
    path = os.path.join(run_dir, GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("per_roi_effective_feature_config_fields_for_overrides") or None


def _discover_custom_tabular_csv_files(input_dir: str) -> list:
    pattern = os.path.join(input_dir, "*.csv")
    files = glob.glob(pattern)
    files.sort(key=natural_sort_key)
    if not files:
        raise RuntimeError(
            "custom_tabular validation failed: no CSV files were found in the input directory. "
            "custom_tabular expects one CSV per session/chunk."
        )
    return files


def _select_custom_tabular_validation_subset(files: list) -> tuple[list, bool]:
    """Return (files_to_validate, was_sampled)."""
    n = len(files)
    if n <= 200:
        return list(files), False

    sample_size = 12
    # Include first and last and a spread of interior files.
    interior_needed = max(0, sample_size - 2)
    indices = {0, n - 1}
    if interior_needed > 0:
        stride = max(1, (n - 2) // interior_needed)
        idx = 1
        while len(indices) < sample_size and idx < n - 1:
            indices.add(idx)
            idx += stride
        # Ensure we fill any shortfall from the tail inward.
        tail = n - 2
        while len(indices) < sample_size and tail > 0:
            indices.add(tail)
            tail -= 1

    selected = [files[i] for i in sorted(indices)]
    return selected, True


def _validate_custom_tabular_contract_validate_only(input_dir: str, config_path: str) -> None:
    """Deep custom_tabular contract validation used by validate-only mode."""
    try:
        cfg = Config.from_yaml(config_path)
    except Exception as e:
        raise RuntimeError(
            f"custom_tabular validation failed: could not load config '{config_path}': {e}"
        ) from e

    files = _discover_custom_tabular_csv_files(input_dir)
    selected, sampled = _select_custom_tabular_validation_subset(files)

    errors = []
    n_valid = 0
    for idx, fpath in enumerate(selected):
        try:
            load_chunk(fpath, "custom_tabular", cfg, chunk_id=idx)
            n_valid += 1
        except Exception as e:
            fname = os.path.basename(fpath)
            detail = str(e).strip() or repr(e)
            errors.append(f"- {fname}: {detail}")

    if errors:
        scope_line = (
            f"Checked {len(selected)} of {len(files)} file(s) (sampled for scale)."
            if sampled
            else f"Checked all {len(files)} discovered file(s)."
        )
        header = "custom_tabular validate-only contract check failed."
        if n_valid == 0:
            body = "No valid custom_tabular files could be parsed."
        else:
            body = (
                f"{len(errors)} file(s) failed custom_tabular contract validation "
                f"while {n_valid} file(s) passed."
            )
        first_errors = "\n".join(errors[:5])
        raise RuntimeError(
            f"{header}\n{scope_line}\n{body}\n\nFirst file-level errors:\n{first_errors}"
        )

    if sampled:
        print(
            f"VALIDATE-ONLY: custom_tabular contract check validated {len(selected)}/"
            f"{len(files)} files (sampled).",
            flush=True,
        )
    else:
        print(
            f"VALIDATE-ONLY: custom_tabular contract check validated all {len(files)} files.",
            flush=True,
        )


def _discover_continuous_sources(input_dir: str, fmt: str) -> list:
    fmt_l = str(fmt).strip().lower()
    if fmt_l == "rwd":
        from photometry_pipeline.io.adapters import discover_rwd_chunks

        files = discover_rwd_chunks(input_dir)
        files.sort(key=natural_sort_key)
        return files
    if fmt_l == "custom_tabular":
        files = glob.glob(os.path.join(input_dir, "*.csv"))
        files.sort(key=natural_sort_key)
        if not files:
            raise RuntimeError(
                f"No files found in {input_dir}"
            )
        return files
    raise RuntimeError(
        f"Continuous acquisition mode is unsupported for format '{fmt}'."
    )


def _resolve_continuous_format(input_dir: str, fmt: str, cfg: Config) -> str:
    fmt_l = str(fmt).strip().lower()
    if fmt_l in {"rwd", "custom_tabular"}:
        return fmt_l
    if fmt_l == "npm":
        raise RuntimeError(CONTINUOUS_NPM_UNSUPPORTED_MESSAGE)
    if fmt_l != "auto":
        raise RuntimeError(f"Invalid format: {fmt}")

    csv_candidates = glob.glob(os.path.join(input_dir, "*.csv"))
    csv_candidates.sort(key=natural_sort_key)
    direct_fluorescence = os.path.join(input_dir, "fluorescence.csv")
    if os.path.isfile(direct_fluorescence):
        csv_candidates.append(direct_fluorescence)
    if not csv_candidates:
        raise RuntimeError(CONTINUOUS_AUTO_FORMAT_MESSAGE)

    sniffed = []
    for c in csv_candidates[:20]:
        fmt_guess = sniff_format(c, cfg)
        if fmt_guess:
            sniffed.append(fmt_guess)
    sniffed = sorted(set(sniffed))
    if sniffed == ["rwd"]:
        return "rwd"
    if sniffed == ["custom_tabular"]:
        return "custom_tabular"
    if sniffed == ["npm"]:
        raise RuntimeError(CONTINUOUS_NPM_UNSUPPORTED_MESSAGE)
    raise RuntimeError(CONTINUOUS_AUTO_FORMAT_MESSAGE)


def _plan_continuous_windows_summary(input_dir: str, fmt: str, cfg: Config) -> dict:
    resolved_format = _resolve_continuous_format(input_dir, fmt, cfg)
    source_files = _discover_continuous_sources(input_dir, resolved_format)
    source_cache = {}
    per_source = []
    planned_windows = 0
    partial_windows = 0
    dropped_partial_windows = 0

    for src in source_files:
        duration_info = estimate_continuous_source_duration(
            src,
            resolved_format,
            cfg,
            source_cache=source_cache,
        )
        windows = plan_continuous_windows_for_source(
            src,
            resolved_format,
            cfg,
            source_cache=source_cache,
        )
        planned_windows += len(windows)
        partial_count = int(sum(1 for w in windows if bool(w.get("is_partial_final_window", False))))
        partial_windows += partial_count
        expected_full = int(
            np.floor(
                float(duration_info["duration_sec"]) / float(cfg.continuous_window_sec)
            )
        )
        remainder = float(duration_info["duration_sec"]) - (
            expected_full * float(cfg.continuous_window_sec)
        )
        if remainder > 1e-9 and not bool(cfg.allow_partial_final_window):
            dropped_partial_windows += 1
        per_source.append(
            {
                "source_file": os.path.abspath(src),
                "duration_sec": float(duration_info["duration_sec"]),
                "median_dt_sec": float(duration_info["median_dt_sec"]),
                "window_count": len(windows),
                "partial_window_count": partial_count,
            }
        )

    return {
        "acquisition_mode": "continuous",
        "resolved_format": resolved_format,
        "source_file_count": len(source_files),
        "continuous_window_sec": float(cfg.continuous_window_sec),
        "continuous_step_sec": float(cfg.continuous_step_sec),
        "allow_partial_final_window": bool(cfg.allow_partial_final_window),
        "planned_window_count": int(planned_windows),
        "partial_window_count": int(partial_windows),
        "dropped_partial_window_count": int(dropped_partial_windows),
        "per_source": per_source,
    }

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

    preallocated_validation = None
    try:
        preallocated_validation = validate_guided_preallocated_mode_args(args)
        if (
            preallocated_validation is not None
            and not preallocated_validation.accepted
        ):
            issue = preallocated_validation.blocking_issues[0]
            raise RuntimeError(
                f"Guided preallocated startup handoff refused: {issue.category}"
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    # Internal Guided execution must verify live source identity before run-dir
    # resolution, allocation, status creation, or any subprocess launch.
    try:
        manifest_verification = verify_guided_manifest_before_output(args)
        if manifest_verification is not None:
            guided_facts, guided_verified = manifest_verification
            verify_guided_normalized_recording_description_before_output(
                args, guided_facts, guided_verified
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

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

    cfg_for_resolution = None
    cfg_parse_failed = False
    try:
        cfg_for_resolution = Config.from_yaml(args.config)
    except Exception as e:
        cfg_parse_failed = True
        print(f"WARNING: Failed to parse config for runner stamping: {e}", flush=True)

    # Resolve acquisition-mode planning fields (CLI overrides config).
    effective_acquisition_mode = "intermittent"
    effective_continuous_window_sec = 600.0
    effective_continuous_step_sec = 600.0
    effective_allow_partial_final_window = False
    acquisition_mode_source = "default"
    if cfg_for_resolution is not None:
        effective_acquisition_mode = str(
            getattr(cfg_for_resolution, "acquisition_mode", "intermittent")
        )
        effective_continuous_window_sec = float(
            getattr(cfg_for_resolution, "continuous_window_sec", 600.0)
        )
        effective_continuous_step_sec = float(
            getattr(cfg_for_resolution, "continuous_step_sec", 600.0)
        )
        effective_allow_partial_final_window = bool(
            getattr(cfg_for_resolution, "allow_partial_final_window", False)
        )
        acquisition_mode_source = "config"

    if args.acquisition_mode is not None:
        effective_acquisition_mode = str(args.acquisition_mode)
        acquisition_mode_source = "user-provided"
    if args.continuous_window_sec is not None:
        effective_continuous_window_sec = float(args.continuous_window_sec)
    if args.continuous_step_sec is not None:
        effective_continuous_step_sec = float(args.continuous_step_sec)
    if args.allow_partial_final_window is not None:
        effective_allow_partial_final_window = bool(args.allow_partial_final_window)

    # Canonicalize args so validation + provenance all reference one resolved set.
    args.acquisition_mode = effective_acquisition_mode
    args.continuous_window_sec = effective_continuous_window_sec
    args.continuous_step_sec = effective_continuous_step_sec
    args.allow_partial_final_window = effective_allow_partial_final_window

    print(
        "Using acquisition_mode="
        f"{effective_acquisition_mode} ({acquisition_mode_source}), "
        f"continuous_window_sec={effective_continuous_window_sec}, "
        f"continuous_step_sec={effective_continuous_step_sec}, "
        f"allow_partial_final_window={effective_allow_partial_final_window}",
        flush=True,
    )

    # Determine effective event signal, excursion polarity, representative index, and preview for stamping
    effective_event_signal = args.event_signal
    effective_signal_excursion_polarity = "positive"
    effective_representative_index = args.representative_session_index
    effective_preview_first_n = args.preview_first_n
    effective_tonic_output_mode = TONIC_OUTPUT_MODE_PRESERVE_RAW
    effective_tonic_timeline_mode = TONIC_TIMELINE_MODE_REAL_ELAPSED
    effective_export_display_series_csv = bool(args.export_display_series_csv)
    
    if (
        effective_event_signal is None
        or effective_representative_index is None
        or effective_preview_first_n is None
        or effective_tonic_output_mode == TONIC_OUTPUT_MODE_PRESERVE_RAW
        or effective_tonic_timeline_mode == TONIC_TIMELINE_MODE_REAL_ELAPSED
    ):
        try:
            if cfg_for_resolution is not None:
                cfg = cfg_for_resolution
            elif cfg_parse_failed:
                raise RuntimeError("config_parse_failed")
            else:
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
            if not effective_export_display_series_csv:
                effective_export_display_series_csv = bool(
                    getattr(cfg, "export_display_series_csv", False)
                )
        except Exception as e:
            print(f"WARNING: Failed to parse config for runner stamping: {e}", flush=True)
            if effective_event_signal is None: 
                effective_event_signal = "dff"
            effective_signal_excursion_polarity = "positive"
            # others remain as given or None
            effective_tonic_output_mode = TONIC_OUTPUT_MODE_PRESERVE_RAW
            effective_tonic_timeline_mode = TONIC_TIMELINE_MODE_REAL_ELAPSED
            effective_export_display_series_csv = bool(args.export_display_series_csv)

    print(
        f"Using tonic_output_mode={effective_tonic_output_mode}",
        flush=True,
    )
    print(
        f"Using tonic_timeline_mode={effective_tonic_timeline_mode}",
        flush=True,
    )
    print(
        "Using export_display_series_csv="
        f"{'enabled' if effective_export_display_series_csv else 'disabled'}",
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
    continuous_mode = bool(effective_acquisition_mode == "continuous")
    analysis_force_format = str(args.format)
    effective_traces_only = bool(args.traces_only or args.run_type == "tuning_prep")
    # The ROIs this run actually processes. Populated once the analysis outputs
    # declare them, and read at finalization to decide which per-ROI deliverables
    # were owed. Never derived from the region directories on disk.
    expected_rois = []
    # Set when the continuous writer has run, together with the per-family window
    # row counts it reported writing.
    continuous_outputs_ran = False
    continuous_row_counts = {}
    # Set when the wrapper froze one run-wide input manifest shared by every
    # analysis subprocess (4J16k41b).
    frozen_input_manifest_path = None
    shared_input_manifest = False
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
        'acquisition_mode': effective_acquisition_mode,
        'continuous_window_sec': effective_continuous_window_sec,
        'continuous_step_sec': effective_continuous_step_sec,
        'allow_partial_final_window': effective_allow_partial_final_window,
        'acquisition_mode_source': acquisition_mode_source,
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
        continuous_plan_summary = None

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
                "acquisition_mode": effective_acquisition_mode,
                "continuous_window_sec": effective_continuous_window_sec,
                "continuous_step_sec": effective_continuous_step_sec,
                "allow_partial_final_window": effective_allow_partial_final_window,
                "acquisition_mode_source": acquisition_mode_source,
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
                "continuous_plan": continuous_plan_summary,
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
            if effective_acquisition_mode == "continuous":
                cfg_plan = Config.from_yaml(args.config)
                cfg_plan.acquisition_mode = "continuous"
                cfg_plan.continuous_window_sec = float(effective_continuous_window_sec)
                cfg_plan.continuous_step_sec = float(effective_continuous_step_sec)
                cfg_plan.allow_partial_final_window = bool(
                    effective_allow_partial_final_window
                )
                continuous_plan_summary = _plan_continuous_windows_summary(
                    input_dir=str(args.input),
                    fmt=str(args.format),
                    cfg=cfg_plan,
                )
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
        if effective_acquisition_mode != "intermittent":
            argv.extend(["--acquisition-mode", str(effective_acquisition_mode)])
        if (
            effective_acquisition_mode != "intermittent"
            or float(effective_continuous_window_sec) != 600.0
        ):
            argv.extend(["--continuous-window-sec", str(effective_continuous_window_sec)])
        if (
            effective_acquisition_mode != "intermittent"
            or float(effective_continuous_step_sec) != 600.0
        ):
            argv.extend(["--continuous-step-sec", str(effective_continuous_step_sec)])
        if effective_acquisition_mode != "intermittent" or bool(effective_allow_partial_final_window):
            argv.append(
                "--allow-partial-final-window"
                if effective_allow_partial_final_window
                else "--no-allow-partial-final-window"
            )
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
        print(
            "VALIDATE-ONLY: acquisition plan "
            f"(mode={effective_acquisition_mode}, "
            f"window_sec={effective_continuous_window_sec}, "
            f"step_sec={effective_continuous_step_sec}, "
            f"allow_partial_final_window={effective_allow_partial_final_window})",
            flush=True,
        )
        if effective_acquisition_mode == "continuous" and continuous_plan_summary is not None:
            print(
                "VALIDATE-ONLY: continuous planning "
                f"resolved_format={continuous_plan_summary['resolved_format']} "
                f"source_files={continuous_plan_summary['source_file_count']} "
                f"planned_windows={continuous_plan_summary['planned_window_count']} "
                f"partial_windows={continuous_plan_summary['partial_window_count']} "
                f"dropped_partial_windows={continuous_plan_summary['dropped_partial_window_count']}",
                flush=True,
            )
            for src in continuous_plan_summary.get("per_source", []):
                print(
                    "VALIDATE-ONLY: continuous source "
                    f"path={src['source_file']} duration_sec={src['duration_sec']:.6f} "
                    f"median_dt_sec={src['median_dt_sec']:.6f} "
                    f"window_count={src['window_count']} "
                    f"partial_window_count={src['partial_window_count']}",
                    flush=True,
                )
            print(
                f"VALIDATE-ONLY: NOTE: {intermittent_only_output_message()}",
                flush=True,
            )
        print(f"VALIDATE-ONLY: argv={json.dumps(argv)}", flush=True)

        emitter.emit("engine", "done", "Validate-only complete")
        emitter.close()
        # Always emit status.json even in legacy mode for GUI/automated observers
        _vo_write_final_status("success")
        raise SystemExit(0)

    # ============================================================
    # 1. Setup run directory
    # ============================================================
    if args.guided_preallocated_run_dir:
        # Startup owns this existing final directory. Never create, clean, or
        # overwrite its root after the exclusive claim above.
        pass
    elif is_gui_mode:
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

    if not args.guided_preallocated_run_dir:
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
        "acquisition_mode": effective_acquisition_mode,
        "continuous_window_sec": effective_continuous_window_sec,
        "continuous_step_sec": effective_continuous_step_sec,
        "allow_partial_final_window": effective_allow_partial_final_window,
        "acquisition_mode_source": acquisition_mode_source,
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
        "failure_details": [],
        "validation": None,
        "artifact_contract": effective_artifact_contract,
        "intentional_skips": effective_skip_plan,
        "features_extracted": False if effective_traces_only else None,
        "traces_only": effective_traces_only,
        "timing": {
            "current_phase": None,
            "phase_started_utc": None,
            "last_completed_phase": None,
            "last_phase_elapsed_sec": None,
            "phase_history": [],
            "phase_elapsed_sec": {},
        }
    }
    t0_status = time.time()
    status_path = os.path.join(run_dir, "status.json")

    # Set the first time any terminal status is written, so a later handler
    # cannot relabel an already-recorded outcome (an injected finalization
    # failure must stay "error", not become "cancelled").
    terminal_flags = {"written": False}

    def _finalize_status(state="success", error_msg=None, completion=None, failure_details=None):
        """Update and write status.json atomically with final state."""
        status_data["phase"] = "final"
        status_data["finished_utc"] = datetime.now(timezone.utc).isoformat()
        status_data["duration_sec"] = time.time() - t0_status

        if error_msg:
            status_data["errors"].append(str(error_msg))

        if failure_details is not None:
            status_data["failure_details"] = list(failure_details)
        else:
            status_data["failure_details"] = status_data.get("failure_details", [])

        # Only a verified terminal set carries the completion block. Its absence
        # on a "success" status is itself a contradiction the loader rejects.
        if completion is not None:
            status_data[COMPLETION_KEY] = completion
        else:
            status_data.pop(COMPLETION_KEY, None)

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
        terminal_flags["written"] = True

    def _write_status_update(phase):
        """Perform a non-terminal status update with current duration."""
        status_data["phase"] = phase
        status_data["status"] = "running"
        status_data["duration_sec"] = time.time() - t0_status
        _write_status_json(status_path, status_data)

    def _skipped_deliverable_families():
        """Deliverable families this run deliberately did not produce."""
        families = set()
        if effective_skip_plan:
            for skipped in effective_skip_plan.get("skipped_outputs", []) or []:
                path = str(skipped)
                if "tonic_overview" in path:
                    families.add(FAMILY_TONIC_OVERVIEW)
                elif "tonic_df_timeseries" in path:
                    families.add(FAMILY_TONIC_TIMESERIES)
                elif "phasic_peak_rate_timeseries" in path or "phasic_auc_timeseries" in path:
                    families.add(FAMILY_PHASIC_TIMESERIES)
                elif "day_plots/" in path:
                    families.add(FAMILY_PHASIC_DAY_PLOTS)
        return sorted(families)

    def _produced_deliverable_files():
        """Per-ROI files this run wrote, recorded so the manifest describes the whole
        package. Mandatory members are promoted to required by the contract itself;
        the rest (extra day plots, session CSVs) stay optional."""
        produced = []
        for roi, record in (manifest.get("deliverables") or {}).items():
            if not isinstance(record, dict):
                continue
            for rel in record.get("files", []) or []:
                produced.append(f"{roi}/{rel}")
        return produced

    def _produced_continuous_files():
        """Continuous outputs the writer reported producing, recorded as optional.

        The mandatory window tables are promoted to required by the contract; the
        plots are skippable when a column has no finite values, so they stay
        optional.
        """
        outputs = status_data.get("continuous_outputs") or {}
        produced = []
        for key in ("summary_tables", "summary_plots", "trace_overview_plots"):
            produced.extend(str(rel) for rel in outputs.get(key, []) or [])
        return produced

    def _execution_run_mode():
        """What this run was asked to do and which phases it executed.

        Nothing here is read back from an output file. Feature extraction runs
        whenever a phasic analysis runs without --traces-only, and it always
        writes features.csv plus the per-ROI settings record beside it, so its
        absence is a real failure and can never excuse itself. The expected ROI
        set is the set the wrapper actually processed, not whichever region
        directories happen to exist at finalization.
        """
        if continuous_mode:
            deliverable_profile = PROFILE_CONTINUOUS
        elif tune_prep_light_mode:
            deliverable_profile = PROFILE_TUNING_PREP
        else:
            deliverable_profile = PROFILE_FULL_INTERMITTENT

        return normalize_run_mode(
            run_profile=args.run_type,
            run_type=effective_run_type,
            acquisition_mode=effective_acquisition_mode,
            traces_only=effective_traces_only,
            phasic_analysis=bool(run_phasic_mode),
            tonic_analysis=bool(run_tonic_mode),
            feature_extraction_ran=bool(run_phasic_mode and not effective_traces_only),
            deliverable_profile=deliverable_profile,
            expected_rois=list(expected_rois),
            skipped_deliverable_families=_skipped_deliverable_families(),
            continuous_outputs_ran=bool(continuous_outputs_ran),
            # Intermittent, non-preview analyses process source data chunk by
            # chunk and must account for every admitted chunk (4J16k41 / C8). The
            # Pipeline writes the completeness record for every non-preview
            # intermittent run (including traces-only tuning-prep).
            chunked_input_processing=bool(
                not continuous_mode
                and effective_run_type != "preview"
                and (run_phasic_mode or run_tonic_mode)
            ),
            shared_input_manifest=bool(shared_input_manifest),
        )

    def _finalize_terminal_success():
        """Write the terminal set in the only order that cannot lie.

        1. mandatory artifacts are already on disk (analysis + deliverables done)
        2. the run report is placed at the root and stamped with its contract
        3. the final manifest is generated from the files that actually exist
        4. the whole terminal set is validated
        5. only then is the success status written, pinning the manifest

        Any failure writes an error status and exits non-zero. The directory is
        never left reloadable as a successful run.
        """
        emitter.emit("package", "start", "Finalizing run outputs")
        _finalization_checkpoint("before_report_finalize")

        report_ok = _ensure_root_run_report(
            run_dir, phasic_out, tonic_out, emitter,
            run_type=effective_run_type,
            run_profile=args.run_type,
            artifact_contract=effective_artifact_contract,
            intentional_skips=effective_skip_plan,
            sessions_per_hour=resolved_sessions_per_hour,
            sessions_per_hour_source=sessions_per_hour_source,
            acquisition_mode=effective_acquisition_mode,
            continuous_window_sec=effective_continuous_window_sec,
            continuous_step_sec=effective_continuous_step_sec,
            allow_partial_final_window=effective_allow_partial_final_window,
            acquisition_mode_source=acquisition_mode_source,
            timeline_anchor_mode=args.timeline_anchor_mode,
            fixed_daily_anchor_clock=args.fixed_daily_anchor_clock,
            run_id=run_id,
        )

        terminal_error = ""
        if not report_ok:
            terminal_error = "mandatory run_report.json is missing at terminal finalize"
        else:
            try:
                run_mode = _execution_run_mode()

                t_mw, started_mw = _phase_start(status_data, "manifest_write", emitter=emitter)
                emitter.emit("package", "start", "Writing final manifest")
                manifest["timing"]["total_runtime_sec"] = time.perf_counter() - t0_total
                # Raises if a mandatory output for this run mode is absent, so a
                # final manifest can never be minted for an incomplete run.
                continuous_index = (
                    build_continuous_window_index(
                        run_dir,
                        run_mode=run_mode,
                        row_counts_by_family=continuous_row_counts,
                    )
                    if continuous_outputs_ran
                    else None
                )
                manifest[COMPLETION_KEY] = build_manifest_completion_block(
                    run_dir,
                    run_id=run_id,
                    run_mode=run_mode,
                    finalized_utc=_utc_now_iso(),
                    optional_artifacts=(
                        OPTIONAL_MANIFEST_ARTIFACTS
                        + _produced_deliverable_files()
                        + _produced_continuous_files()
                    ),
                    continuous_index=continuous_index,
                )
                _phase_done(status_data, manifest, "manifest_write", t_mw, started_mw, emitter=emitter)

                t_fa, started_fa = _phase_start(status_data, "finalize_artifacts", emitter=emitter)
                _phase_done(status_data, manifest, "finalize_artifacts", t_fa, started_fa, emitter=emitter)

                _finalization_checkpoint("before_manifest_write")
                _atomic_write_json(manifest_path, manifest)
                _finalization_checkpoint("after_manifest_write")
                emitter.emit("package", "done", "Manifest written")
                _write_status_json(status_path, status_data)

                terminal_error = verify_terminal_set_before_status(
                    run_dir, run_id=run_id, run_mode=run_mode
                )
            except RunCompletionError as exc:
                terminal_error = str(exc)

        if terminal_error:
            emitter.emit(
                "package",
                "error",
                f"Run outputs are incomplete, so this run was not marked successful: {terminal_error}",
                error_code="TERMINAL_VALIDATION_FAILED",
            )
            _finalize_status("error", error_msg=f"TERMINAL_VALIDATION_FAILED: {terminal_error}")
            # This is the actual terminal cause of a nonzero exit here. Unlike
            # status.json/events.ndjson (files), stderr is the only channel a
            # caller that only captures process.stderr (e.g. the Guided
            # startup orchestration's blocking-issue message) ever sees --
            # without this, only earlier, unrelated stderr output (such as a
            # successful plotting phase's library warnings) would be visible,
            # silently hiding why a run that produced every ordinary
            # deliverable was still correctly rejected.
            print(f"Error: TERMINAL_VALIDATION_FAILED: {terminal_error}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        _finalization_checkpoint("before_success_status")
        _finalize_status(
            "success",
            completion=build_status_completion_block(
                run_id=run_id,
                manifest_sha256=sha256_file(manifest_path),
            ),
        )
        _finalization_checkpoint("after_success_status")
        emitter.emit("package", "done", "Run outputs finalized and verified")

    # Update status_data with resolved preview info before initial write
    status_data["run_type"] = effective_run_type
    status_data["preview"] = {"selector": "first_n", "first_n": effective_preview_first_n} if effective_preview_first_n is not None else None

    if preallocated_validation is not None:
        # Consume the one-shot startup only after every earlier non-writing
        # preflight has passed and immediately before production status.
        preallocated_validation = validate_guided_preallocated_mode_args(args)
        if not preallocated_validation.accepted:
            issue = preallocated_validation.blocking_issues[0]
            print(
                "Error: Guided preallocated startup handoff refused: "
                f"{issue.category}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        claim_result = claim_guided_preallocated_startup(
            preallocated_validation,
            claimed_utc=_utc_now_iso(),
            process_id=os.getpid(),
        )
        if not claim_result.claimed:
            issue = claim_result.blocking_issues[0]
            print(
                "Error: Guided preallocated startup handoff refused: "
                f"{issue.category}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Initial write (phase="running")
    _write_status_json(status_path, status_data)
    _write_status_update("initializing")
    if _GUIDED_TEST_STOP_AFTER_INITIAL_STATUS is not None:
        _GUIDED_TEST_STOP_AFTER_INITIAL_STATUS(
            run_dir=run_dir,
            status_path=status_path,
        )

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
        "acquisition_mode": effective_acquisition_mode,
        "continuous_window_sec": effective_continuous_window_sec,
        "continuous_step_sec": effective_continuous_step_sec,
        "allow_partial_final_window": effective_allow_partial_final_window,
        "acquisition_mode_source": acquisition_mode_source,
        "timeline_anchor_mode": args.timeline_anchor_mode,
        "fixed_daily_anchor_clock": args.fixed_daily_anchor_clock,
    })
    substantive_work_completed = False
    continuous_full_plan_summary = None

    try:
        if continuous_mode:
            cfg_plan = Config.from_yaml(args.config)
            cfg_plan.acquisition_mode = "continuous"
            cfg_plan.continuous_window_sec = float(effective_continuous_window_sec)
            cfg_plan.continuous_step_sec = float(effective_continuous_step_sec)
            cfg_plan.allow_partial_final_window = bool(effective_allow_partial_final_window)
            continuous_full_plan_summary = _plan_continuous_windows_summary(
                input_dir=str(args.input),
                fmt=str(args.format),
                cfg=cfg_plan,
            )
            analysis_force_format = str(continuous_full_plan_summary["resolved_format"])
            manifest["continuous_plan"] = continuous_full_plan_summary
            emitter.emit(
                "engine",
                "audit",
                "Continuous mode planning resolved.",
                payload=continuous_full_plan_summary,
            )
            emitter.emit(
                "engine",
                "notice",
                intermittent_only_output_message(),
                payload={"intermittent_only_outputs": INTERMITTENT_ONLY_OUTPUT_KEYS},
            )

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
        t_phase, started_utc_phase = _phase_start(status_data, "validate", emitter=emitter)
        _write_status_update("validating")
        emitter.emit("validate", "start", "Validating inputs")
        validate_inputs(args)
        emitter.emit("validate", "done", "Validation passed")
        _phase_done(status_data, manifest, "validate", t_phase, started_utc_phase, status_path=status_path, emitter=emitter)

        check_cancel(cancel_flag_path, emitter, "validate", manifest_path, manifest)

        # ============================================================
        # 4. Tonic Analysis
        # ============================================================
        # Locate analyze_photometry.py relative to this script
        # This script is in tools/, analyze_photometry.py is in root/
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(tools_dir)
        analyze_script = os.path.join(root_dir, 'analyze_photometry.py')

        def _append_continuous_analysis_args(cmd):
            """Propagate wrapper-resolved continuous settings to analysis subprocesses."""
            if effective_acquisition_mode != "continuous":
                return
            cmd.extend(["--acquisition-mode", "continuous"])
            cmd.extend(["--continuous-window-sec", str(effective_continuous_window_sec)])
            cmd.extend(["--continuous-step-sec", str(effective_continuous_step_sec)])
            cmd.append(
                "--allow-partial-final-window"
                if effective_allow_partial_final_window
                else "--no-allow-partial-final-window"
            )

        # Freeze ONE run-wide production input manifest before launching any
        # analysis subprocess, so phasic and tonic are held to the same admitted
        # chunk set (4J16k41b). Guided runs resolve their inputs from a verified
        # candidate manifest, not by discovery, and are phasic-only, so they keep
        # their single self-frozen record.
        frozen_input_manifest_path = None
        shared_input_manifest = bool(
            not continuous_mode
            and effective_run_type != "preview"
            and (run_phasic_mode or run_tonic_mode)
            and not getattr(args, "guided_candidate_manifest", None)
        )
        if shared_input_manifest:
            frozen_input_manifest_path = _freeze_run_input_manifest(
                input_dir=args.input,
                config_path=args.config,
                run_dir=run_dir,
                force_format=analysis_force_format,
            )

        def _append_frozen_manifest_arg(cmd):
            if frozen_input_manifest_path:
                cmd.extend(["--frozen-input-manifest", frozen_input_manifest_path])

        if run_tonic_mode:
            t_phase, started_utc_phase = _phase_start(status_data, "tonic_analysis", emitter=emitter)
            _write_status_update("tonic_analysis")
            emitter.emit("tonic", "start", "Running tonic analysis")
            emitter.close()  # Release file lock so subprocess can append events
            cmd_tonic = [sys.executable, analyze_script,
                         '--input', args.input,
                         '--out', tonic_out,
                         '--config', args.config,
                         '--mode', 'tonic',
                         '--format', analysis_force_format,
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
            _append_continuous_analysis_args(cmd_tonic)
            _append_frozen_manifest_arg(cmd_tonic)
            _append_guided_manifest_to_analysis_command(
                cmd_tonic, args, mode="tonic"
            )
            if events_path: cmd_tonic.extend(['--events-path', events_path])
            try:
                manifest['commands'].append(run_cmd(cmd_tonic))
            finally:
                emitter = EventEmitter(events_path, run_id, run_dir, file_mode="a")
            emitter.emit("tonic", "done", "Tonic analysis complete")
            _phase_done(status_data, manifest, "tonic_analysis", t_phase, started_utc_phase, status_path=status_path, emitter=emitter)

            check_cancel(cancel_flag_path, emitter, "tonic", manifest_path, manifest)

        # ============================================================
        # 5. Phasic Analysis
        # ============================================================
        if run_phasic_mode:
            t_phase, started_utc_phase = _phase_start(status_data, "phasic_analysis", emitter=emitter)
            _write_status_update("phasic_analysis")
            emitter.emit("phasic", "start", "Running phasic analysis")
            emitter.close()  # Release file lock so subprocess can append events
            cmd_phasic = [sys.executable, analyze_script,
                          '--input', args.input,
                          '--out', phasic_out,
                          '--config', args.config,
                          '--mode', 'phasic',
                          '--format', analysis_force_format,
                          '--recursive']
            # analyze_photometry.py rejects --overwrite together with
            # --guided-candidate-manifest (its Guided execution state must
            # match the caller's already-enforced no-overwrite contract).
            # phasic_out is a freshly allocated subdirectory of the Guided
            # run directory in that case, so --overwrite is never needed.
            if not getattr(args, "guided_candidate_manifest", None):
                cmd_phasic.append('--overwrite')
            if args.include_rois: cmd_phasic.extend(['--include-rois', args.include_rois])
            if args.exclude_rois: cmd_phasic.extend(['--exclude-rois', args.exclude_rois])
            if effective_traces_only: cmd_phasic.append('--traces-only')
            if args.event_signal: cmd_phasic.extend(['--event-signal', args.event_signal])
            if args.representative_session_index is not None: cmd_phasic.extend(['--representative-session-index', str(args.representative_session_index)])
            if args.preview_first_n is not None: cmd_phasic.extend(['--preview-first-n', str(args.preview_first_n)])
            if resolved_sessions_per_hour is not None:
                cmd_phasic.extend(['--sessions-per-hour', str(resolved_sessions_per_hour)])
            _append_continuous_analysis_args(cmd_phasic)
            _append_frozen_manifest_arg(cmd_phasic)
            _append_guided_manifest_to_analysis_command(
                cmd_phasic, args, mode="phasic"
            )
            if events_path: cmd_phasic.extend(['--events-path', events_path])
            try:
                manifest['commands'].append(run_cmd(cmd_phasic))
            finally:
                emitter = EventEmitter(events_path, run_id, run_dir, file_mode="a")
            emitter.emit("phasic", "done", "Phasic analysis complete")
            _phase_done(status_data, manifest, "phasic_analysis", t_phase, started_utc_phase, status_path=status_path, emitter=emitter)

            check_cancel(cancel_flag_path, emitter, "phasic", manifest_path, manifest)

        if continuous_mode:
            t_phase, started_utc_phase = _phase_start(status_data, "continuous_summary_tables", emitter=emitter)
            _write_status_update("continuous_summary_tables")
            emitter.emit("continuous_outputs", "start", "Generating continuous summary tables")
            from photometry_pipeline.continuous_outputs import (
                generate_continuous_summary_plots,
                generate_continuous_summary_tables,
                generate_continuous_trace_overview_plots,
            )

            # The analyzed ROI set, as each analysis declared it. Frozen here,
            # before any continuous deliverable is written, so a deleted table
            # can never shrink what this run is held to.
            expected_rois = (
                (_regions_from_run_report(os.path.join(phasic_out, 'run_report.json'))
                 if run_phasic_mode else None)
                or (_regions_from_run_report(os.path.join(tonic_out, 'run_report.json'))
                    if run_tonic_mode else None)
                or []
            )

            summary_result = generate_continuous_summary_tables(
                run_dir,
                tonic_out_dir=tonic_out if run_tonic_mode else None,
                phasic_out_dir=phasic_out if run_phasic_mode else None,
                mode=str(args.mode),
                logger=lambda msg: emitter.emit("continuous_outputs", "notice", str(msg)),
            )
            continuous_outputs_ran = True
            continuous_row_counts = _continuous_row_counts_from_writer(summary_result)
            status_data["continuous_outputs"] = {
                "summary_tables_generated": bool(summary_result.get("summary_tables_generated", False)),
                "summary_tables": list(summary_result.get("summary_tables", [])),
                "summary_skips": list(summary_result.get("summary_skips", [])),
            }
            emitter.emit(
                "continuous_outputs",
                "done",
                "Continuous summary table generation complete",
                payload=status_data["continuous_outputs"],
            )
            _phase_done(
                status_data,
                manifest,
                "continuous_summary_tables",
                t_phase,
                started_utc_phase,
                status_path=status_path,
                emitter=emitter,
            )

            t_phase, started_utc_phase = _phase_start(status_data, "continuous_summary_plots", emitter=emitter)
            _write_status_update("continuous_summary_plots")
            emitter.emit("continuous_outputs", "start", "Generating continuous summary plots")
            plot_result = generate_continuous_summary_plots(
                run_dir,
                mode=str(args.mode),
                logger=lambda msg: emitter.emit("continuous_outputs", "notice", str(msg)),
            )
            status_data["continuous_outputs"].update(
                {
                    "summary_plots_generated": bool(plot_result.get("summary_plots_generated", False)),
                    "summary_plots": list(plot_result.get("summary_plots", [])),
                    "plot_skips": list(plot_result.get("plot_skips", [])),
                }
            )
            emitter.emit(
                "continuous_outputs",
                "done",
                "Continuous summary plot generation complete",
                payload={
                    "summary_plots_generated": status_data["continuous_outputs"]["summary_plots_generated"],
                    "summary_plots": status_data["continuous_outputs"]["summary_plots"],
                    "plot_skips": status_data["continuous_outputs"]["plot_skips"],
                },
            )
            _phase_done(
                status_data,
                manifest,
                "continuous_summary_plots",
                t_phase,
                started_utc_phase,
                status_path=status_path,
                emitter=emitter,
            )

            t_phase, started_utc_phase = _phase_start(status_data, "continuous_trace_overview_plots", emitter=emitter)
            _write_status_update("continuous_trace_overview_plots")
            emitter.emit("continuous_outputs", "start", "Generating continuous trace overview plots")
            trace_overview_result = generate_continuous_trace_overview_plots(
                run_dir,
                mode=str(args.mode),
                logger=lambda msg: emitter.emit("continuous_outputs", "notice", str(msg)),
            )
            status_data["continuous_outputs"].update(
                {
                    "trace_overview_plots_generated": bool(
                        trace_overview_result.get("generated", False)
                    ),
                    "trace_overview_plots": list(trace_overview_result.get("plots", [])),
                    "trace_overview_skips": list(trace_overview_result.get("skips", [])),
                    "trace_overview_details": dict(trace_overview_result.get("details", {})),
                }
            )
            emitter.emit(
                "continuous_outputs",
                "done",
                "Continuous trace overview plot generation complete",
                payload={
                    "trace_overview_plots_generated": status_data["continuous_outputs"][
                        "trace_overview_plots_generated"
                    ],
                    "trace_overview_plots": status_data["continuous_outputs"][
                        "trace_overview_plots"
                    ],
                    "trace_overview_skips": status_data["continuous_outputs"][
                        "trace_overview_skips"
                    ],
                },
            )
            _phase_done(
                status_data,
                manifest,
                "continuous_trace_overview_plots",
                t_phase,
                started_utc_phase,
                status_path=status_path,
                emitter=emitter,
            )

            manifest["sessions_per_hour"] = resolved_sessions_per_hour
            manifest["sessions_per_hour_source"] = sessions_per_hour_source
            manifest["session_duration_s"] = None
            manifest["session_stride_s"] = None
            manifest["continuous_outputs"] = {
                "analysis_caches_generated": True,
                "summary_tables_generated": bool(summary_result.get("summary_tables_generated", False)),
                "summary_tables": list(summary_result.get("summary_tables", [])),
                "summary_skips": list(summary_result.get("summary_skips", [])),
                "summary_plots_generated": bool(plot_result.get("summary_plots_generated", False)),
                "summary_plots": list(plot_result.get("summary_plots", [])),
                "plot_skips": list(plot_result.get("plot_skips", [])),
                "trace_overview_plots_generated": bool(trace_overview_result.get("generated", False)),
                "trace_overview_plots": list(trace_overview_result.get("plots", [])),
                "trace_overview_skips": list(trace_overview_result.get("skips", [])),
                "trace_overview_details": dict(trace_overview_result.get("details", {})),
                "summary_details": {
                    "phasic": summary_result.get("phasic"),
                    "tonic": summary_result.get("tonic"),
                },
                "plot_details": {
                    "phasic": plot_result.get("phasic"),
                    "tonic": plot_result.get("tonic"),
                },
                "intermittent_only_outputs_skipped": INTERMITTENT_ONLY_OUTPUT_KEYS,
                "guidance": intermittent_only_output_message(),
            }
            emitter.emit(
                "session_compute",
                "skipped",
                "Session/stride computation skipped in continuous mode.",
            )
            emitter.emit(
                "plots",
                "skipped",
                intermittent_only_output_message(),
                payload={"intermittent_only_outputs": INTERMITTENT_ONLY_OUTPUT_KEYS},
            )

            substantive_work_completed = True
            _finalize_terminal_success()
            emitter.emit("engine", "done", "Execution complete")
            emitter.close()
            return

        # ============================================================
        # 6. Session / Stride Computation
        # ============================================================
        t_phase, started_utc_phase = _phase_start(status_data, "session_compute", emitter=emitter)
        
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
        _phase_done(status_data, manifest, "session_compute", t_phase, started_utc_phase, status_path=status_path, emitter=emitter)

        check_cancel(cancel_flag_path, emitter, "session_compute", manifest_path, manifest)

        # 7. Per-Region Processing (Plots & Packaging)
        # ============================================================
        t_phase, started_utc_phase = _phase_start(status_data, "plots_total", emitter=emitter)
        _write_status_update("plots")
        emitter.emit("plots", "start", "Generating per-ROI deliverables")

        # Resolve the ROI set before it is used to classify per-ROI feature
        # provenance below.
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
        # Freeze the analyzed ROI set now, while it is still derived from what the
        # analysis reported, not from the deliverable directories it will fill.
        expected_rois = list(regions)

        # Classify the run ONCE, before any ROI plot process is launched. A
        # current-contract run must carry a complete per-ROI record of the
        # settings actually consumed; a missing/invalid record fails the run
        # here rather than letting each child silently verify against the
        # global config_used.yaml (4J16k39b).
        # Detector-aware verification only happens when this run produced
        # features. When it did, this wrapper knows it just invoked the current
        # analysis build, so the current contract is REQUIRED: damaged metadata
        # fails the run instead of silently degrading to global settings.
        _will_verify_detectors = (
            run_phasic_mode
            and not tune_prep_light_mode
            and os.path.isfile(os.path.join(phasic_out, "features", "features.csv"))
        )
        if _will_verify_detectors:
            feature_provenance_mode, feature_provenance_path = (
                _resolve_feature_provenance_for_plots(
                    phasic_out,
                    regions,
                    require_current=True,
                    emitter=emitter,
                )
            )
        else:
            # No features were extracted, so no strict peak-count verification and
            # no per-ROI detector replay occurs for this run.
            feature_provenance_mode, feature_provenance_path = "legacy", None
        if tune_prep_light_mode and effective_skip_plan is not None:
            emitter.emit(
                "plots",
                "audit",
                "Tuning-prep selective skipping enabled for nonessential outputs.",
                payload=effective_skip_plan,
            )

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
                if effective_export_display_series_csv:
                    cmd_tonic_roi.append('--export-display-series-csv')
                    cmd_tonic_roi.extend(['--source-run-profile', str(args.run_type)])
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
                session_indices = []
                session_statuses = []
                session_sources = []
                missing_tonic_sessions = []
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
                        from photometry_pipeline.viz.phasic_data_prep import build_authoritative_plot_sessions
                                
                        source_files = []
                        if "meta" in cache and "source_files" in cache["meta"]:
                            source_files = [f.decode('utf-8') if isinstance(f, bytes) else f for f in cache["meta"]["source_files"][:]]

                        authoritative_tonic_sessions = build_authoritative_plot_sessions(
                            tonic_out, cids, source_files
                        )
                        cache_to_session = {
                            int(item["cache_chunk_id"]): int(item["session_index"])
                            for item in (authoritative_tonic_sessions or [])
                            if item.get("cache_chunk_id") is not None
                        }
                        missing_tonic_sessions = [
                            item for item in (authoritative_tonic_sessions or [])
                            if item.get("status") != "valid"
                        ]
                        if missing_tonic_sessions:
                            # Current missing-session rows carry explicit session
                            # metadata and markers; use the bounded fallback
                            # writer rather than a two-column streaming shortcut.
                            use_streaming_csv = False

                        # Rescale timeline offsets using exactly matched position
                        actual_positions = (
                            [cache_to_session.get(int(cid), int(cid)) for cid in cids]
                            if authoritative_tonic_sessions is not None
                            else map_cached_sources_to_schedule_positions(
                                args.input, args.format, source_files, cids
                            )
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
                                session_indices.append(
                                    np.full(len(t_abs), int(actual_schedule_idx if actual_schedule_idx is not None else cid), dtype=int)
                                )
                                session_statuses.append(np.full(len(t_abs), "valid", dtype=object))
                                session_sources.append(np.full(len(t_abs), str(source_files[i] if i < len(source_files) else ""), dtype=object))
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
                            'session_index': np.concatenate(session_indices),
                            'status': np.concatenate(session_statuses),
                            'source_file': np.concatenate(session_sources),
                            'row_kind': 'sample',
                        }
                    )
                    if missing_tonic_sessions:
                        marker_rows = []
                        base_start = min(
                            (
                                item.get("expected_start_time")
                                for item in (authoritative_tonic_sessions or missing_tonic_sessions)
                                if item.get("expected_start_time") is not None
                            ),
                            default=None,
                        )
                        for item in missing_tonic_sessions:
                            start = item.get("expected_start_time")
                            if base_start is not None and start is not None:
                                marker_time = (start - base_start).total_seconds() / 3600.0
                            else:
                                marker_time = float(item.get("session_index", 0)) * (
                                    1.0 / float(resolved_sessions_per_hour or 1.0)
                                )
                            marker_rows.append(
                                {
                                    "time_hours": marker_time,
                                    "tonic_df": np.nan,
                                    "session_index": int(item["session_index"]),
                                    "status": str(item.get("status", "missing_corrupted")),
                                    "source_file": str(item.get("source_file", "")),
                                    "row_kind": "missing_session_marker",
                                }
                            )
                        full_tonic = pd.concat(
                            [full_tonic, pd.DataFrame(marker_rows)],
                            ignore_index=True,
                        ).sort_values(["time_hours", "session_index"], kind="stable")
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
                              '--stacked-render-mode', str(args.stacked_render_mode),
                              ]
                # The run was classified once, before any ROI plot was launched.
                # Child processes never infer the contract from file presence.
                cmd_bundle.extend(
                    _dayplot_provenance_args(
                        feature_provenance_mode, feature_provenance_path
                    )
                )
                if effective_export_display_series_csv:
                    cmd_bundle.append('--export-display-series-csv')
                    cmd_bundle.extend(['--source-run-profile', str(args.run_type)])
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
        _phase_done(status_data, manifest, "plots_total", t_phase, started_utc_phase, status_path=status_path, emitter=emitter)
        check_cancel(cancel_flag_path, emitter, "package", manifest_path, manifest)

        # Removed: Section 7.5 "Guided Post-Phasic Applied-dF/F
        # Orchestration" used to run here unconditionally for every
        # invocation of this wrapper (Guided current-native, Guided
        # positive-legacy, and plain non-Guided --out invocations alike),
        # gated only by whether guided_correction_strategy_map.json
        # happened to be present -- an absence-based skip, not an explicit
        # classification. guided_startup_materialization.py no longer
        # writes that artifact for any transaction (native or legacy), so
        # no supported workflow can reach this stage any more; the retired
        # entry point (photometry_pipeline/guided_applied_dff_orchestration.py)
        # has no remaining caller. Positive legacy applied-dF/F evidence
        # from already-completed runs remains loadable for Review via
        # guided_completed_applied_dff_reload.py, which does not execute
        # anything. Standalone applied-dF/F production (Full Control's
        # "Applied dF/F Explicit Batch" group, and the tools/*applied_dff*
        # CLI chain) calls tools/run_applied_dff_batch.py directly and does
        # not go through this wrapper.

        # ============================================================
        # 8. Ordered, fail-closed finalization
        #    (report -> final manifest -> terminal validation -> success)
        # ============================================================
        substantive_work_completed = True
        _finalize_terminal_success()
        emitter.emit("engine", "done", "Execution complete")
        emitter.close()

    except SystemExit as se:
        # A terminal state already on disk is the truth. Never relabel it: an
        # injected finalization failure stays an error, and an applied-dF/F
        # failure is not reported to the scientist as a cancellation.
        if terminal_flags["written"]:
            if emitter:
                emitter.close()
            raise se

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
                                          acquisition_mode=effective_acquisition_mode,
                                          continuous_window_sec=effective_continuous_window_sec,
                                          continuous_step_sec=effective_continuous_step_sec,
                                          allow_partial_final_window=effective_allow_partial_final_window,
                                          acquisition_mode_source=acquisition_mode_source,
                                          timeline_anchor_mode=args.timeline_anchor_mode,
                                          fixed_daily_anchor_clock=args.fixed_daily_anchor_clock,
                                          run_id=run_id)
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

        # A terminal status is only ever written after the whole terminal set has
        # been validated. A later failure is post-run bookkeeping: report it, but
        # never contradict an outcome already recorded on disk.
        if terminal_flags["written"]:
            if emitter:
                emitter.emit(
                    "engine",
                    "error",
                    f"Failure after the run's outcome was recorded: {e}",
                    error_code="POST_TERMINAL_FAILURE",
                )
                emitter.close()
            raise SystemExit(0 if status_data.get("status") == "success" else 1)

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
                                          acquisition_mode=effective_acquisition_mode,
                                          continuous_window_sec=effective_continuous_window_sec,
                                          continuous_step_sec=effective_continuous_step_sec,
                                          allow_partial_final_window=effective_allow_partial_final_window,
                                          acquisition_mode_source=acquisition_mode_source,
                                          timeline_anchor_mode=args.timeline_anchor_mode,
                                          fixed_daily_anchor_clock=args.fixed_daily_anchor_clock,
                                          run_id=run_id)
        except Exception:
             pass

        msg_body = str(e)
        if not ok:
            viol_msg = "STRICT ORDERING VIOLATION: root run_report.json missing at terminal finalize"
            if emitter:
                emitter.emit("package", "error", viol_msg)
            msg_body = f"{msg_body} | {viol_msg}"

        from photometry_pipeline.input_processing_completeness import InputProcessingError

        failure_details_list = []

        # 1. Check if the subprocess wrote an input_processing_error.json
        error_file_paths = []
        if 'run_dir' in locals():
            error_file_paths = [
                os.path.join(run_dir, '_analysis', 'phasic_out', 'input_processing_error.json'),
                os.path.join(run_dir, '_analysis', 'tonic_out', 'input_processing_error.json'),
            ]
        for path in error_file_paths:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        err_data = json.load(handle)
                    cat = err_data.get("category")
                    phase = err_data.get("phase")
                    source = err_data.get("source")
                    s_idx = err_data.get("session_index")
                    reason = err_data.get("reason")

                    if (
                        isinstance(cat, str)
                        and isinstance(phase, str)
                        and isinstance(source, str)
                        and not isinstance(s_idx, bool)
                        and isinstance(s_idx, int)
                        and s_idx >= 0
                        and isinstance(reason, str)
                    ):
                        eligible = (
                            cat == "processing_exception"
                            and phase in ("pass1", "pass1a", "pass1b", "tonic_pass1c", "pass2")
                        )
                        failure_details_list.append({
                            "failure_type": "input_processing_failure",
                            "category": cat,
                            "phase": phase,
                            "source": source,
                            "session_index": s_idx,
                            "reason": reason,
                            "eligible_for_missing_session_authorization": eligible
                        })
                except Exception as read_err:
                    print(f"ERROR: Failed to read {path}: {read_err}")

        # 2. Check if the error is an in-process InputProcessingError (for completeness)
        if isinstance(e, InputProcessingError):
            eligible = (
                e.category == "processing_exception"
                and not isinstance(e.chunk_index, bool)
                and isinstance(e.chunk_index, int)
                and e.chunk_index >= 0
                and bool(e.source)
                and e.phase in ("pass1", "pass1a", "pass1b", "tonic_pass1c", "pass2")
            )
            norm_source = os.path.normpath(str(e.source)).replace("\\", "/")
            failure_details_list.append({
                "failure_type": "input_processing_failure",
                "category": e.category,
                "phase": e.phase,
                "source": norm_source,
                "session_index": e.chunk_index,
                "reason": str(e.reason),
                "eligible_for_missing_session_authorization": eligible
            })

        # Deduplicate: the same underlying failure must not be reported twice
        # even though it may be observed both via a subprocess-written
        # input_processing_error.json and via an in-process
        # InputProcessingError. Keyed by the complete structured identity so a
        # genuinely distinct failure is never dropped.
        deduped_failure_details = []
        seen_failure_keys = set()
        for detail in failure_details_list:
            key = (
                detail.get("failure_type"),
                detail.get("category"),
                detail.get("phase"),
                os.path.normpath(str(detail.get("source", ""))).replace("\\", "/"),
                detail.get("session_index"),
                detail.get("reason"),
            )
            if key in seen_failure_keys:
                continue
            seen_failure_keys.add(key)
            deduped_failure_details.append(detail)
        failure_details_list = deduped_failure_details

        # --- Finalize Status: Error ---
        try:
            _finalize_status("error", error_msg=msg_body, failure_details=failure_details_list)
        except Exception as status_err:
            print(f"ERROR: Failed to finalize error status.json: {status_err}", flush=True)
        raise SystemExit(1)

if __name__ == '__main__':
    main()
