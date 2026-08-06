"""
run_report_parser.py

Pure functions for parsing run_report.json, determining preview mode,
and resolving quick-links safely within <run_dir>.
"""

import json
import os
import re
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
    COMPLETION_KEY,
    CONTINUOUS_INDEX_KEY,
    CONTINUOUS_FAMILY_FILENAMES,
    GUIDED_CONTINUOUS_RUN_PROFILES,
    PROFILE_CONTINUOUS,
    expected_continuous_families,
    guided_continuous_saved_artifact_specs,
    TERMINAL_SUCCESS_LEGACY,
    classify_run_terminal_state,
)


class SavedArtifactIndexError(RuntimeError):
    """A current native Guided continuous artifact package is not readable."""


_CONTINUOUS_ARTIFACT_LABELS = {
    "phasic_correction_impact.png": "Correction Verification",
    "tonic_overview.png": "Slow Signal Summary",
    "phasic_auc_timeseries.png": "Corrected Signal Area Over Time",
    "phasic_peak_rate_timeseries.png": "Detected Event Rate Over Time",
    "continuous_phasic_window_summary.csv": "Event Window Summary",
    "continuous_tonic_window_summary.csv": "Slow Signal Window Summary",
    "continuous_phasic_events.csv": "Detected Events",
}

_CONTINUOUS_ARTIFACT_ORDER = {
    "phasic_correction_impact.png": 10,
    "tonic_overview.png": 20,
    "phasic_auc_timeseries.png": 30,
    "phasic_peak_rate_timeseries.png": 40,
    "continuous_phasic_window_summary.csv": 50,
    "continuous_tonic_window_summary.csv": 60,
    "continuous_phasic_events.csv": 70,
}

_CONTINUOUS_DAY_PLOT_FAMILY_LABELS = {
    "sampled_signal_reference": "Signal / Reference",
    "sampled_correction_reference": "Correction Reference",
    "sampled_phasic_dff": "dF/F",
    "sampled_stacked": "Stacked dF/F",
}
_CONTINUOUS_DAY_PLOT_FILENAME_PATTERNS = {
    "sampled_signal_reference": re.compile(r"phasic_sig_iso_day_\d{3}\.png"),
    "sampled_correction_reference": re.compile(
        r"phasic_correction_reference_day_\d{3}\.png"
    ),
    "sampled_phasic_dff": re.compile(r"phasic_dFF_day_\d{3}\.png"),
    "sampled_stacked": re.compile(r"phasic_stacked_day_\d{3}\.png"),
}


def _manifest_completion_run_mode(run_dir: str) -> Dict[str, Any]:
    """Read only the completion-declared run mode, without directory discovery."""
    manifest, error = _read_json_dict(os.path.join(run_dir, "MANIFEST.json"))
    if error is not None:
        return {}
    completion = manifest.get(COMPLETION_KEY)
    if not isinstance(completion, dict):
        return {}
    run_mode = completion.get("run_mode")
    return dict(run_mode) if isinstance(run_mode, dict) else {}


def declared_completed_run_mode(run_dir: str) -> Dict[str, Any]:
    """Return the run mode pinned by the completion manifest, if present."""
    return _manifest_completion_run_mode(os.path.realpath(run_dir))


def is_guided_continuous_saved_artifact_run_mode(
    run_mode: Dict[str, Any]
) -> bool:
    """True only for native Guided continuous runs with the I3 package."""
    return (
        str(run_mode.get("run_profile", "")) in GUIDED_CONTINUOUS_RUN_PROFILES
        and str(run_mode.get("acquisition_mode", "")) == "continuous"
        and str(run_mode.get("deliverable_profile", "")) == PROFILE_CONTINUOUS
        and bool(run_mode.get("continuous_outputs_ran"))
    )


def _continuous_artifact_path(
    run_dir: str, relative_path: str, *, description: str
) -> str:
    """Resolve one manifest-declared path inside the completed run."""
    normalized = str(relative_path or "").replace("\\", "/").strip("/")
    if (
        not normalized
        or normalized.startswith("/")
        or ".." in normalized.split("/")
    ):
        raise SavedArtifactIndexError(
            f"This completed analysis has an invalid {description} path."
        )
    root = os.path.realpath(run_dir)
    path = os.path.realpath(os.path.join(root, *normalized.split("/")))
    try:
        inside = os.path.normcase(os.path.commonpath([root, path])) == os.path.normcase(root)
    except (OSError, ValueError):
        inside = False
    try:
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
    except OSError:
        exists = False
        size = 0
    if not inside or not exists:
        raise SavedArtifactIndexError(
            f"This completed analysis cannot be opened because the required "
            f"{description} is missing or invalid."
        )
    if size <= 0:
        raise SavedArtifactIndexError(
            f"This completed analysis cannot be opened because the required "
            f"{description} is empty."
        )
    return path


def _validate_continuous_saved_image(path: str, label: str) -> None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size[0] <= 0 or image.size[1] <= 0:
                raise ValueError("image dimensions are not positive")
    except Exception as exc:
        raise SavedArtifactIndexError(
            f"This completed analysis cannot be opened because the required "
            f"{label} image is missing or invalid."
        ) from exc


def build_guided_continuous_saved_artifact_index(
    run_dir: str,
    *,
    classification=None,
) -> Dict[str, Any]:
    """Adapt the I3 completion index to the existing generic Results viewer.

    The manifest is the only source of artifact membership and ROI order here.
    This function never scans ROI directories and never opens an analysis cache.
    """
    resolved = os.path.realpath(str(run_dir))
    if classification is None:
        classification = classify_run_terminal_state(resolved)
    if not getattr(classification, "is_success", False):
        raise SavedArtifactIndexError(
            "This completed analysis cannot be opened because it is not a "
            f"verified successful run ({getattr(classification, 'reason', 'validation failed')})."
        )
    if not getattr(classification, "is_current", False):
        raise SavedArtifactIndexError(
            "This completed analysis is not a current native Guided result and "
            "cannot be opened in the saved-artifact viewer."
        )

    manifest, manifest_error = _read_json_dict(os.path.join(resolved, "MANIFEST.json"))
    if manifest_error is not None:
        raise SavedArtifactIndexError(
            "This completed analysis cannot be opened because its output manifest "
            "is missing or invalid."
        )
    completion = manifest.get(COMPLETION_KEY)
    if not isinstance(completion, dict):
        raise SavedArtifactIndexError(
            "This completed analysis cannot be opened because its completion record is missing."
        )
    run_mode = completion.get("run_mode")
    if not isinstance(run_mode, dict) or not is_guided_continuous_saved_artifact_run_mode(run_mode):
        raise SavedArtifactIndexError(
            "This completed analysis is not a native Guided continuous artifact package."
        )
    expected_rois = tuple(str(roi) for roi in (run_mode.get("expected_rois") or ()))
    if not expected_rois:
        raise SavedArtifactIndexError(
            "This completed analysis has no manifest-declared regions of interest."
        )

    deliverables = completion.get("deliverables")
    if not isinstance(deliverables, dict):
        raise SavedArtifactIndexError(
            "This completed analysis has no manifest-declared saved Results artifacts."
        )
    index = deliverables.get(CONTINUOUS_INDEX_KEY)
    if not isinstance(index, dict):
        raise SavedArtifactIndexError(
            "This completed analysis has no manifest-declared continuous Results index."
        )

    saved_records = index.get("saved_artifacts")
    if not isinstance(saved_records, list):
        raise SavedArtifactIndexError(
            "This completed analysis has no manifest-declared saved figure records."
        )
    saved_by_path: Dict[str, Dict[str, Any]] = {}
    for record in saved_records:
        if not isinstance(record, dict):
            raise SavedArtifactIndexError(
                "This completed analysis has an unreadable saved figure record."
            )
        relative_path = str(record.get("relative_path", "")).replace("\\", "/")
        if relative_path in saved_by_path:
            raise SavedArtifactIndexError(
                f"This completed analysis declares the saved figure more than once: {relative_path}."
            )
        saved_by_path[relative_path] = record

    report, report_error = _read_json_dict(os.path.join(resolved, "run_report.json"))
    if report_error is not None:
        raise SavedArtifactIndexError(
            "This completed analysis cannot be opened because its run report is missing or invalid."
        )

    artifacts: List[Dict[str, Any]] = []
    for roi in expected_rois:
        for family, filename, analysis_family in guided_continuous_saved_artifact_specs(run_mode):
            relative_path = f"{roi}/summary/{filename}"
            record = saved_by_path.get(relative_path)
            label = _CONTINUOUS_ARTIFACT_LABELS.get(filename, filename)
            if record is None:
                raise SavedArtifactIndexError(
                    f"This completed analysis cannot be opened because the required "
                    f"{label} image for {roi} is missing from its completion record."
                )
            if (
                str(record.get("roi", "")) != roi
                or str(record.get("family", "")) != family
                or str(record.get("analysis_family", "")) != analysis_family
                or str(record.get("artifact_type", "")) != "image"
            ):
                raise SavedArtifactIndexError(
                    f"This completed analysis has invalid provenance for the required "
                    f"{label} image for {roi}."
                )
            path = _continuous_artifact_path(
                resolved, relative_path, description=f"{label} image for {roi}"
            )
            _validate_continuous_saved_image(path, label)
            artifacts.append(
                {
                    **dict(record),
                    "roi": roi,
                    "label": label,
                    "relative_path": relative_path,
                    "path": path,
                    "artifact_type": "image",
                    "analysis_applicability": analysis_family,
                    "order": _CONTINUOUS_ARTIFACT_ORDER[filename],
                }
            )

    # Continuous sampled Day Plots are optional image records in the existing
    # manifest-backed saved-artifact index.  They are admitted only from that
    # index; this parser never scans day_plots folders or opens analysis caches.
    for relative_path, record in saved_by_path.items():
        parts = relative_path.split("/")
        if len(parts) != 3 or parts[1] != "day_plots":
            continue
        family = str(record.get("family") or "")
        if family not in _CONTINUOUS_DAY_PLOT_FAMILY_LABELS:
            continue
        roi = parts[0]
        filename = parts[2]
        label = _CONTINUOUS_DAY_PLOT_FAMILY_LABELS[family]
        if roi not in expected_rois:
            raise SavedArtifactIndexError(
                f"This completed analysis has a sampled Day Plot for an unknown ROI: {roi}."
            )
        if not _CONTINUOUS_DAY_PLOT_FILENAME_PATTERNS[family].fullmatch(filename):
            raise SavedArtifactIndexError(
                f"This completed analysis has an invalid sampled Day Plot filename for {roi}."
            )
        if (
            str(record.get("roi", "")) != roi
            or str(record.get("analysis_family", "")) != "phasic"
            or str(record.get("artifact_type", "")) != "image"
            or str(record.get("label", "")) != label
        ):
            raise SavedArtifactIndexError(
                f"This completed analysis has invalid provenance for the sampled "
                f"{label} image for {roi}."
            )
        day_index = record.get("day_index")
        if isinstance(day_index, bool) or not isinstance(day_index, int) or day_index < 0:
            raise SavedArtifactIndexError(
                f"This completed analysis has an invalid sampled Day Plot day for {roi}."
            )
        path = _continuous_artifact_path(
            resolved,
            relative_path,
            description=f"sampled {label} image for {roi}",
        )
        _validate_continuous_saved_image(path, label)
        artifacts.append(
            {
                **dict(record),
                "roi": roi,
                "label": label,
                "relative_path": relative_path,
                "path": path,
                "artifact_type": "image",
                "analysis_applicability": "phasic",
                "order": int(record.get("order", 0)),
            }
        )

    for family in expected_continuous_families(run_mode):
        families = index.get("families")
        if not isinstance(families, dict):
            raise SavedArtifactIndexError(
                "This completed analysis has an invalid manifest-declared table index."
            )
        family_record = families.get(family)
        if not isinstance(family_record, dict):
            raise SavedArtifactIndexError(
                f"This completed analysis has no manifest-declared {family} table family."
            )
        paths = family_record.get("relative_paths")
        if not isinstance(paths, dict):
            raise SavedArtifactIndexError(
                f"This completed analysis has an invalid manifest-declared {family} table family."
            )
        filename = CONTINUOUS_FAMILY_FILENAMES[family]
        analysis_family = "phasic" if family.startswith("continuous_phasic") else "tonic"
        label = _CONTINUOUS_ARTIFACT_LABELS[filename]
        for roi in expected_rois:
            relative_path = str(paths.get(roi, "")).replace("\\", "/")
            expected_path = f"{roi}/tables/{filename}"
            if relative_path != expected_path:
                raise SavedArtifactIndexError(
                    f"This completed analysis has an invalid {label} path for {roi}."
                )
            path = _continuous_artifact_path(
                resolved, relative_path, description=f"{label} table for {roi}"
            )
            artifacts.append(
                {
                    "roi": roi,
                    "label": label,
                    "relative_path": relative_path,
                    "path": path,
                    "artifact_type": "table",
                    "analysis_applicability": analysis_family,
                    "order": _CONTINUOUS_ARTIFACT_ORDER[filename],
                }
            )

    if run_mode.get("phasic_analysis") and run_mode.get("feature_extraction_ran"):
        event_relative_path = "_analysis/phasic_out/features/continuous_phasic_events.csv"
        completion_artifacts = completion.get("artifacts")
        event_records = [
            record
            for record in (completion_artifacts if isinstance(completion_artifacts, list) else [])
            if isinstance(record, dict)
            and str(record.get("relative_path", "")).replace("\\", "/") == event_relative_path
        ]
        if len(event_records) != 1:
            raise SavedArtifactIndexError(
                "This completed analysis cannot be opened because its saved "
                "Detected events table is missing from the completion record."
            )
        event_path = _continuous_artifact_path(
            resolved, event_relative_path, description="Detected events table"
        )
        artifacts.append(
            {
                "roi": None,
                "label": "Detected Events",
                "relative_path": event_relative_path,
                "path": event_path,
                "artifact_type": "table",
                "analysis_applicability": "phasic",
                "scope": "run",
                "order": _CONTINUOUS_ARTIFACT_ORDER["continuous_phasic_events.csv"],
            }
        )

    timeline = manifest.get("timeline")
    if not isinstance(timeline, dict):
        timeline = report.get("timeline") if isinstance(report.get("timeline"), dict) else {}
    window_timing = index.get("window_timing")
    if not isinstance(window_timing, dict):
        window_timing = {}
    return {
        "run_dir": resolved,
        "run_id": str(getattr(classification, "run_id", "") or ""),
        "run_mode": dict(run_mode),
        "roi_order": expected_rois,
        "timeline": dict(timeline),
        "window_timing": dict(window_timing),
        "artifacts": sorted(
            artifacts,
            key=lambda item: (
                int(item.get("order", 0)),
                expected_rois.index(item["roi"]) if item.get("roi") in expected_rois else -1,
            ),
        ),
    }


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
    recorded gaps into scientist-facing terms.  It does not expose internal
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
        os.path.join(run_dir, "_analysis", "phasic_out", "input_processing_completeness.json"),
        os.path.join(run_dir, "_analysis", "tonic_out", "input_processing_completeness.json"),
        os.path.join(run_dir, "input_manifest.json"),
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
        headline = "Completed with missing sessions and a legacy final-session exclusion."
    elif exclusion_count:
        headline = "Completed with a legacy final-session exclusion."
    else:
        headline = "Completed with missing sessions."
    lines = [headline]
    if missing_count:
        lines.append(
            f"{missing_count} missing session(s) were recorded and kept in their original time positions."
        )
    if exclusion_count:
        lines.append(
            f"{exclusion_count} legacy final session(s) were excluded from analysis."
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
            reason_text = "legacy final incomplete session excluded"
        else:
            reason_text = reason or "session could not be processed"
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

    # Native Guided continuous runs own an exhaustive manifest/completion
    # index.  Their completed-run gate must not rediscover ROI folders to
    # decide whether they are openable; the saved-artifact adapter performs
    # the authoritative package validation instead.
    classification = classify_run_terminal_state(run_dir)
    run_mode = classification.run_mode or declared_completed_run_mode(run_dir)
    if is_guided_continuous_saved_artifact_run_mode(run_mode):
        try:
            build_guided_continuous_saved_artifact_index(
                run_dir, classification=classification
            )
        except SavedArtifactIndexError as exc:
            return False, str(exc)
        return True, evidence

    regions = resolve_region_deliverables(run_dir)
    if not regions:
        return False, "Completed-run metadata found, but no region deliverables (summary, day_plots, or tables folders) discovered."

    return True, evidence
