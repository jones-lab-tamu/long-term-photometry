"""Terminal completion contract for production runs (4J16k40).

A run directory is only a *successful current run* when one coherent terminal
set exists on disk:

1. ``status.json``   -- terminal status, ``phase=final``, ``status=success``
2. ``run_report.json`` -- parses, and declares the supported completion contract
3. ``MANIFEST.json`` -- parses, is marked final, and enumerates the artifacts
4. every mandatory artifact for the run's *actual* mode is present
5. run identity agrees across status, report, and manifest
6. recorded artifact identities (size, and digest where recorded) verify
7. no contradictory failed / interrupted / in-progress marker is present

Nothing here infers success from directory shape or from a single favoured
file. Absence or corruption of metadata is never treated as "legacy": a legacy
run must be *positively* identified by the historical run-report shape.

The wrapper (``tools/run_full_pipeline_deliverables.py``) builds the manifest
completion block from the files it actually wrote, validates the terminal set,
and only then writes the success status. Readers (GUI completed-run loading,
tuning gates, correction preview) classify with
:func:`classify_run_terminal_state` so there is exactly one definition of
success.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

# Bumped whenever the terminal set's meaning changes. A run declaring any other
# value is not something this build knows how to verify, so it fails closed.
COMPLETION_CONTRACT_VERSION = "run_completion.v1"

STATUS_FILENAME = "status.json"
MANIFEST_FILENAME = "MANIFEST.json"
RUN_REPORT_FILENAME = "run_report.json"

# Key under which the contract block is stamped into each terminal file.
COMPLETION_KEY = "completion"
REPORT_COMPLETION_KEY = "completion_contract"

# Hashing a multi-gigabyte trace cache at finalize would add minutes to every
# days-to-weeks run. Above this size we record size only and say so explicitly
# rather than pretending a digest was checked.
DIGEST_MAX_BYTES = 64 * 1024 * 1024
DIGEST_OMITTED_LARGE = "large_binary_size_verified_only"

# Terminal classifications.
TERMINAL_SUCCESS_CURRENT = "success_current"
TERMINAL_SUCCESS_LEGACY = "success_legacy"
TERMINAL_FAILED = "failed"
TERMINAL_INTERRUPTED = "interrupted"
TERMINAL_CORRUPTED = "corrupted"
TERMINAL_NOT_A_RUN = "not_a_run"

SUCCESS_STATES = (TERMINAL_SUCCESS_CURRENT, TERMINAL_SUCCESS_LEGACY)

_FAILED_STATUS_TOKENS = {"failed", "error", "failure"}
_INTERRUPTED_STATUS_TOKENS = {"cancelled", "canceled", "aborted", "running", "in-progress", "active"}
_INTERRUPTED_PHASE_TOKENS = {"cancelled", "canceled", "aborted", "running", "in-progress", "active", "non-final"}
_LEGACY_SUCCESS_TOKENS = {"success", "complete", "completed", "done"}
_LEGACY_FINAL_TOKENS = {"final", "complete", "completed", "done"}


class RunCompletionError(RuntimeError):
    """The terminal set could not be built or does not verify."""


@dataclass(frozen=True)
class TerminalClassification:
    """Outcome of inspecting a run directory's terminal set."""

    state: str
    reason: str
    run_id: str = ""
    contract_version: str = ""
    run_mode: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.state in SUCCESS_STATES

    @property
    def is_current(self) -> bool:
        return self.state == TERMINAL_SUCCESS_CURRENT

    @property
    def is_legacy(self) -> bool:
        return self.state == TERMINAL_SUCCESS_LEGACY


# ----------------------------------------------------------------------
# Artifact identity
# ----------------------------------------------------------------------


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_artifact_record(run_dir: str, rel_path: str, *, required: bool) -> dict[str, Any] | None:
    """Describe one artifact as it exists on disk, or None when absent.

    Digests are recorded for artifacts small enough that hashing them is not a
    meaningful cost at the end of a long run; larger artifacts record their size
    and state plainly that no digest was taken.
    """
    abs_path = os.path.join(run_dir, _to_os_rel(rel_path))
    if not os.path.isfile(abs_path):
        return None
    size = os.path.getsize(abs_path)
    record: dict[str, Any] = {
        "relative_path": _to_posix_rel(rel_path),
        "required": bool(required),
        "size_bytes": int(size),
        "sha256": None,
    }
    if size <= DIGEST_MAX_BYTES:
        record["sha256"] = sha256_file(abs_path)
    else:
        record["digest_omitted_reason"] = DIGEST_OMITTED_LARGE
    return record


def _to_posix_rel(rel_path: str) -> str:
    return str(rel_path).replace("\\", "/")


def _to_os_rel(rel_path: str) -> str:
    return os.path.join(*_to_posix_rel(rel_path).split("/"))


# ----------------------------------------------------------------------
# Mandatory artifacts by actual run mode
# ----------------------------------------------------------------------


# Which scientist-facing deliverable package the run promised to produce.
PROFILE_FULL_INTERMITTENT = "full_intermittent"
PROFILE_TUNING_PREP = "tuning_prep"
PROFILE_CONTINUOUS = "continuous"

# Named families of per-ROI deliverables, so an intentional skip can be recorded
# by name instead of by enumerating the files it would have produced.
FAMILY_PHASIC_CORRECTION_IMPACT = "phasic_correction_impact"
FAMILY_TONIC_OVERVIEW = "tonic_overview"
FAMILY_TONIC_TIMESERIES = "tonic_df_timeseries"
FAMILY_PHASIC_TIMESERIES = "phasic_timeseries"
FAMILY_PHASIC_DAY_PLOTS = "phasic_day_plots"

# Continuous mode packages its science as one exhaustive per-ROI window table per
# family: every analysis window is a row, so there are no dynamically named files
# to enumerate. The tables are the index.
FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY = "continuous_phasic_window_summary"
FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY = "continuous_tonic_window_summary"

CONTINUOUS_FAMILY_FILENAMES = {
    FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY: "continuous_phasic_window_summary.csv",
    FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY: "continuous_tonic_window_summary.csv",
}

CONTINUOUS_INDEX_KEY = "continuous_window_index"

# Day indices are offsets from the first chunk's own date, so day 000 exists for
# every ROI that produced any day plot at all. Requiring the day-000 members of
# each family gives a stable check without guessing how many days a recording
# spans -- the wrapper already enforces that the per-family day sets agree.
_DAY_ZERO = "000"


def normalize_run_mode(
    *,
    run_profile: str,
    run_type: str,
    acquisition_mode: str,
    traces_only: bool,
    phasic_analysis: bool,
    tonic_analysis: bool,
    feature_extraction_ran: bool,
    deliverable_profile: str,
    expected_rois: list[str] | tuple[str, ...] = (),
    skipped_deliverable_families: list[str] | tuple[str, ...] = (),
    continuous_outputs_ran: bool = False,
) -> dict[str, Any]:
    """The execution facts that decide the mandatory artifact set.

    Every field here describes what the run was asked to do and which phases it
    executed. None of them is derived from the presence of an output file, so a
    deleted output can never quietly excuse itself by changing the run mode.
    """
    return {
        "run_profile": str(run_profile or ""),
        "run_type": str(run_type or ""),
        "acquisition_mode": str(acquisition_mode or ""),
        "traces_only": bool(traces_only),
        "phasic_analysis": bool(phasic_analysis),
        "tonic_analysis": bool(tonic_analysis),
        "feature_extraction_ran": bool(feature_extraction_ran),
        "deliverable_profile": str(deliverable_profile or ""),
        "expected_rois": [str(roi) for roi in expected_rois],
        "skipped_deliverable_families": sorted({str(f) for f in skipped_deliverable_families}),
        "continuous_outputs_ran": bool(continuous_outputs_ran),
    }


def expected_continuous_families(run_mode: dict[str, Any]) -> list[str]:
    """Continuous window-summary families this run promised to write.

    Derived from which analyses ran, never from which tables survive.
    `generate_continuous_phasic_summary` needs features.csv, so a traces-only
    continuous run promises no phasic summary; `generate_continuous_tonic_summary`
    needs only the tonic cache.
    """
    if run_mode.get("deliverable_profile") != PROFILE_CONTINUOUS:
        return []
    if not run_mode.get("continuous_outputs_ran"):
        return []

    skipped = set(run_mode.get("skipped_deliverable_families") or ())
    families = []
    if run_mode.get("phasic_analysis") and run_mode.get("feature_extraction_ran"):
        families.append(FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY)
    if run_mode.get("tonic_analysis"):
        families.append(FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY)
    return [family for family in families if family not in skipped]


def required_core_artifacts_for_run_mode(run_mode: dict[str, Any]) -> list[str]:
    """Mandatory internal analysis artifacts implied by the phases that executed."""
    required = [RUN_REPORT_FILENAME]

    if run_mode.get("phasic_analysis"):
        required += [
            "_analysis/phasic_out/run_report.json",
            "_analysis/phasic_out/config_used.yaml",
            "_analysis/phasic_out/phasic_trace_cache.h5",
        ]
    if run_mode.get("tonic_analysis"):
        required += [
            "_analysis/tonic_out/run_report.json",
            "_analysis/tonic_out/config_used.yaml",
            "_analysis/tonic_out/tonic_trace_cache.h5",
        ]
    # Feature extraction writes one row per ROI per chunk, so it produces
    # features.csv (and, since 4J16k39, the per-ROI settings record beside it)
    # whenever it runs -- a recording with no detected events yields rows with
    # peak_count 0, never a missing file. Requiring these whenever extraction ran
    # is therefore sound, and absence is always a real failure.
    if run_mode.get("feature_extraction_ran"):
        required += [
            "_analysis/phasic_out/features/features.csv",
            "_analysis/phasic_out/features/feature_event_provenance.json",
        ]
    return required


def required_deliverables_for_run_mode(run_mode: dict[str, Any]) -> list[str]:
    """Mandatory scientist-facing deliverables for the selected production profile.

    The expected ROI set comes from the ROIs the run actually analyzed, never
    from whichever region directories happen to exist. Families the run
    deliberately skipped are recorded by name and are not required.

    The full intermittent profile promises a per-ROI plot/table package;
    continuous mode promises one exhaustive per-ROI window table per family;
    tuning-prep promises only traces, cache, and report.
    """
    profile = run_mode.get("deliverable_profile")
    if profile == PROFILE_CONTINUOUS:
        return [
            f"{roi}/tables/{CONTINUOUS_FAMILY_FILENAMES[family]}"
            for roi in run_mode.get("expected_rois") or []
            for family in expected_continuous_families(run_mode)
        ]
    if profile != PROFILE_FULL_INTERMITTENT:
        return []

    skipped = set(run_mode.get("skipped_deliverable_families") or ())
    phasic = bool(run_mode.get("phasic_analysis"))
    tonic = bool(run_mode.get("tonic_analysis"))
    features = bool(run_mode.get("feature_extraction_ran"))

    required: list[str] = []
    for roi in run_mode.get("expected_rois") or []:
        if phasic and FAMILY_PHASIC_CORRECTION_IMPACT not in skipped:
            required.append(f"{roi}/summary/phasic_correction_impact.png")
        if tonic and FAMILY_TONIC_OVERVIEW not in skipped:
            required.append(f"{roi}/summary/tonic_overview.png")
        if tonic and FAMILY_TONIC_TIMESERIES not in skipped:
            required.append(f"{roi}/tables/tonic_df_timeseries.csv")
        if features and FAMILY_PHASIC_TIMESERIES not in skipped:
            required += [
                f"{roi}/summary/phasic_peak_rate_timeseries.png",
                f"{roi}/summary/phasic_auc_timeseries.png",
                f"{roi}/tables/phasic_peak_rate_timeseries.csv",
                f"{roi}/tables/phasic_auc_timeseries.csv",
            ]
        if phasic and FAMILY_PHASIC_DAY_PLOTS not in skipped:
            required.append(f"{roi}/day_plots/phasic_sig_iso_day_{_DAY_ZERO}.png")
            if features:
                required += [
                    f"{roi}/day_plots/phasic_dFF_day_{_DAY_ZERO}.png",
                    f"{roi}/day_plots/phasic_stacked_day_{_DAY_ZERO}.png",
                ]
    return required


def required_artifacts_for_run_mode(run_mode: dict[str, Any]) -> list[str]:
    """Every mandatory relative path for this run mode: internal, then deliverables."""
    return required_core_artifacts_for_run_mode(run_mode) + required_deliverables_for_run_mode(
        run_mode
    )


def run_mode_structural_error(run_mode: dict[str, Any]) -> str:
    """Reject a run mode that cannot describe a real run, before it excuses anything."""
    profile = run_mode.get("deliverable_profile")
    if profile not in (PROFILE_FULL_INTERMITTENT, PROFILE_TUNING_PREP, PROFILE_CONTINUOUS):
        return f"unknown deliverable profile {profile!r}"
    if run_mode.get("feature_extraction_ran") and not run_mode.get("phasic_analysis"):
        return "feature extraction is recorded as having run without phasic analysis"
    if run_mode.get("feature_extraction_ran") and run_mode.get("traces_only"):
        return "feature extraction is recorded as having run for a traces-only run"
    if profile == PROFILE_FULL_INTERMITTENT and not run_mode.get("expected_rois"):
        return "a full production run analyzed no ROIs, so it produced no deliverables"
    if run_mode.get("continuous_outputs_ran") and profile != PROFILE_CONTINUOUS:
        return "continuous outputs are recorded as having run outside continuous mode"
    if expected_continuous_families(run_mode) and not run_mode.get("expected_rois"):
        return "a continuous run promised window summaries but analyzed no ROIs"
    return ""


def build_continuous_window_index(
    run_dir: str,
    *,
    run_mode: dict[str, Any],
    row_counts_by_family: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """The exhaustive index of continuous window outputs, from the writer's own report.

    `row_counts_by_family` maps family -> {roi: number of window rows written},
    taken from what `continuous_outputs` reported writing. It is never rebuilt by
    scanning the output directory.
    """
    families: dict[str, Any] = {}
    for family in expected_continuous_families(run_mode):
        filename = CONTINUOUS_FAMILY_FILENAMES[family]
        counts = row_counts_by_family.get(family, {})
        families[family] = {
            "relative_paths": {
                str(roi): f"{roi}/tables/{filename}"
                for roi in run_mode.get("expected_rois") or []
            },
            "window_row_counts": {str(roi): int(count) for roi, count in counts.items()},
        }
    return {
        "families": families,
        "skipped_families": list(run_mode.get("skipped_deliverable_families") or []),
    }


def continuous_index_error(run_mode: dict[str, Any], deliverables: Any) -> str:
    """Validate the continuous index against what the run promised. "" when sound."""
    families = expected_continuous_families(run_mode)
    if not families:
        return ""

    if not isinstance(deliverables, dict):
        return "the output manifest records no continuous window index"
    index = deliverables.get(CONTINUOUS_INDEX_KEY)
    if not isinstance(index, dict):
        return "the continuous window index is missing"
    indexed = index.get("families")
    if not isinstance(indexed, dict):
        return "the continuous window index lists no output families"

    expected_rois = [str(roi) for roi in run_mode.get("expected_rois") or []]
    for family in families:
        entry = indexed.get(family)
        if not isinstance(entry, dict):
            return f"the continuous window index omits the {family} outputs"
        paths = entry.get("relative_paths")
        counts = entry.get("window_row_counts")
        if not isinstance(paths, dict) or not isinstance(counts, dict):
            return f"the continuous window index for {family} is unreadable"

        filename = CONTINUOUS_FAMILY_FILENAMES[family]
        for roi in expected_rois:
            expected_path = f"{roi}/tables/{filename}"
            if _to_posix_rel(str(paths.get(roi, ""))) != expected_path:
                return f"the continuous window index does not account for {expected_path}"
            # A continuous window table always carries one row per analysis
            # window. Zero rows means the window analysis produced nothing for
            # this ROI, which is not a completed continuous run.
            count = counts.get(roi)
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                return f"the continuous window index records no analysis windows for ROI {roi}"
    return ""


def build_manifest_deliverables_block(
    run_mode: dict[str, Any],
    *,
    continuous_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize, for a reader, what the run owed and what it deliberately skipped."""
    block = {
        "profile": run_mode.get("deliverable_profile", ""),
        "expected_rois": list(run_mode.get("expected_rois") or []),
        "required": [_to_posix_rel(p) for p in required_deliverables_for_run_mode(run_mode)],
        "intentionally_skipped_families": list(
            run_mode.get("skipped_deliverable_families") or []
        ),
    }
    if continuous_index is not None:
        block[CONTINUOUS_INDEX_KEY] = continuous_index
    return block


# ----------------------------------------------------------------------
# Building the terminal set (writer side)
# ----------------------------------------------------------------------


def build_manifest_completion_block(
    run_dir: str,
    *,
    run_id: str,
    run_mode: dict[str, Any],
    finalized_utc: str,
    optional_artifacts: list[str] | None = None,
    continuous_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the manifest's terminal block from the files actually on disk.

    Raises RunCompletionError when the run mode is incoherent, when its
    continuous index does not account for the outputs it promised, or when a
    mandatory artifact is absent, so the caller cannot mint a final manifest for
    an incomplete run.
    """
    structural_error = run_mode_structural_error(run_mode)
    if structural_error:
        raise RunCompletionError(f"Run mode does not describe a completable run: {structural_error}")

    deliverables = build_manifest_deliverables_block(run_mode, continuous_index=continuous_index)
    index_error = continuous_index_error(run_mode, deliverables)
    if index_error:
        raise RunCompletionError(f"Continuous window outputs are incomplete: {index_error}")

    required = required_artifacts_for_run_mode(run_mode)
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []

    for rel_path in required:
        record = build_artifact_record(run_dir, rel_path, required=True)
        if record is None:
            missing.append(_to_posix_rel(rel_path))
        else:
            artifacts.append(record)

    if missing:
        raise RunCompletionError(
            "Mandatory outputs for this run mode are missing: " + ", ".join(sorted(missing))
        )

    seen = {rec["relative_path"] for rec in artifacts}
    for rel_path in optional_artifacts or []:
        posix = _to_posix_rel(rel_path)
        if posix in seen:
            continue
        record = build_artifact_record(run_dir, rel_path, required=False)
        if record is not None:
            artifacts.append(record)
            seen.add(posix)

    return {
        "completion_contract_version": COMPLETION_CONTRACT_VERSION,
        "final": True,
        "run_id": str(run_id),
        "finalized_utc": str(finalized_utc),
        "run_mode": dict(run_mode),
        "deliverables": deliverables,
        "artifacts": sorted(artifacts, key=lambda rec: rec["relative_path"]),
    }


def build_report_completion_block(*, run_id: str) -> dict[str, Any]:
    """The block stamped into run_report.json so the report declares its contract."""
    return {
        "contract_version": COMPLETION_CONTRACT_VERSION,
        "run_id": str(run_id),
    }


def build_status_completion_block(*, run_id: str, manifest_sha256: str) -> dict[str, Any]:
    """The block stamped into the terminal status.

    Binding the manifest digest into the status closes the chain: status pins
    the manifest, the manifest pins the report, the config snapshot, and every
    other mandatory artifact.
    """
    return {
        "completion_contract_version": COMPLETION_CONTRACT_VERSION,
        "run_id": str(run_id),
        "manifest_sha256": str(manifest_sha256),
    }


def verify_terminal_set_before_status(
    run_dir: str,
    *,
    run_id: str,
    run_mode: dict[str, Any],
) -> str:
    """Verify everything a successful run needs *except* its final status.

    The wrapper calls this immediately before writing the success status, so a
    run that cannot pass it never gets one. Returns "" when the terminal set
    verifies, otherwise an actionable internal reason.
    """
    report, report_err = _read_json_object(os.path.join(run_dir, RUN_REPORT_FILENAME))
    if report is None:
        return f"mandatory {RUN_REPORT_FILENAME} is {report_err}"
    report_block = report.get(REPORT_COMPLETION_KEY)
    if not isinstance(report_block, dict):
        return f"{RUN_REPORT_FILENAME} does not declare a completion contract"
    if report_block.get("contract_version") != COMPLETION_CONTRACT_VERSION:
        return (
            f"{RUN_REPORT_FILENAME} declares completion contract "
            f"{report_block.get('contract_version')!r}; expected {COMPLETION_CONTRACT_VERSION!r}"
        )
    if str(report_block.get("run_id", "")) != str(run_id):
        return f"{RUN_REPORT_FILENAME} run identity does not match this run"

    manifest, manifest_err = _read_json_object(os.path.join(run_dir, MANIFEST_FILENAME))
    if manifest is None:
        return f"mandatory {MANIFEST_FILENAME} is {manifest_err}"
    manifest_block = manifest.get(COMPLETION_KEY)
    if not isinstance(manifest_block, dict):
        return f"{MANIFEST_FILENAME} does not declare a completion contract"
    if manifest_block.get("completion_contract_version") != COMPLETION_CONTRACT_VERSION:
        return f"{MANIFEST_FILENAME} declares an unsupported completion contract"
    if manifest_block.get("final") is not True:
        return f"{MANIFEST_FILENAME} is not marked final"
    if str(manifest_block.get("run_id", "")) != str(run_id):
        return f"{MANIFEST_FILENAME} run identity does not match this run"
    if manifest_block.get("run_mode") != dict(run_mode):
        return f"{MANIFEST_FILENAME} run mode does not match what this run executed"
    structural_error = run_mode_structural_error(run_mode)
    if structural_error:
        return f"run mode does not describe a completable run: {structural_error}"
    index_error = continuous_index_error(run_mode, manifest_block.get("deliverables"))
    if index_error:
        return f"continuous window outputs are incomplete: {index_error}"

    artifacts = manifest_block.get("artifacts")
    if not isinstance(artifacts, list):
        return f"{MANIFEST_FILENAME} does not list this run's outputs"

    listed_required = {
        str(entry.get("relative_path", ""))
        for entry in artifacts
        if isinstance(entry, dict) and entry.get("required") is True
    }
    unlisted = sorted({_to_posix_rel(p) for p in required_artifacts_for_run_mode(run_mode)} - listed_required)
    if unlisted:
        return f"{MANIFEST_FILENAME} omits mandatory outputs: " + ", ".join(unlisted)

    return _verify_recorded_artifacts(run_dir, artifacts)


# ----------------------------------------------------------------------
# Reading the terminal set (reader side)
# ----------------------------------------------------------------------


def _read_json_object(path: str) -> tuple[dict[str, Any] | None, str]:
    """Return (data, error). data is None when absent or unusable."""
    if not os.path.isfile(path):
        return None, "missing"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - any read/parse failure is malformed
        return None, f"malformed ({exc})"
    if not isinstance(data, dict):
        return None, "malformed (root is not a JSON object)"
    return data, ""


def _declared_contract_versions(
    status: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collect the completion-contract version each terminal file declares."""
    declared: dict[str, Any] = {}
    if status is not None and isinstance(status.get(COMPLETION_KEY), dict):
        declared[STATUS_FILENAME] = status[COMPLETION_KEY].get("completion_contract_version")
    if manifest is not None and isinstance(manifest.get(COMPLETION_KEY), dict):
        declared[MANIFEST_FILENAME] = manifest[COMPLETION_KEY].get("completion_contract_version")
    if report is not None and isinstance(report.get(REPORT_COMPLETION_KEY), dict):
        declared[RUN_REPORT_FILENAME] = report[REPORT_COMPLETION_KEY].get("contract_version")
    return declared


def _status_token(status: dict[str, Any]) -> str:
    return str(status.get("status", "")).strip().lower()


def _phase_token(status: dict[str, Any]) -> str:
    return str(status.get("phase", "")).strip().lower()


def classify_run_terminal_state(run_dir: str) -> TerminalClassification:
    """Classify a run directory's terminal set. This is the only definition of success."""
    run_dir = os.path.realpath(str(run_dir))
    if not os.path.isdir(run_dir):
        return TerminalClassification(TERMINAL_NOT_A_RUN, f"Directory does not exist: {run_dir}")

    status, status_err = _read_json_object(os.path.join(run_dir, STATUS_FILENAME))
    manifest, manifest_err = _read_json_object(os.path.join(run_dir, MANIFEST_FILENAME))
    report, report_err = _read_json_object(os.path.join(run_dir, RUN_REPORT_FILENAME))

    if status is None and manifest is None and report is None:
        if status_err == manifest_err == report_err == "missing":
            return TerminalClassification(
                TERMINAL_NOT_A_RUN,
                "This folder has no run status, run report, or output manifest, "
                "so it is not an analysis run folder.",
            )
        return TerminalClassification(
            TERMINAL_CORRUPTED,
            "Every run record in this folder is unreadable, so the run cannot be verified.",
        )

    declared = _declared_contract_versions(status, manifest, report)
    if declared:
        unsupported = {
            name: version
            for name, version in declared.items()
            if version != COMPLETION_CONTRACT_VERSION
        }
        if unsupported:
            detail = ", ".join(f"{name} declares {version!r}" for name, version in sorted(unsupported.items()))
            return TerminalClassification(
                TERMINAL_CORRUPTED,
                "This run declares a completion record this version of the app cannot verify "
                f"({detail}); expected {COMPLETION_CONTRACT_VERSION!r}.",
                contract_version=COMPLETION_CONTRACT_VERSION,
            )
        return _classify_current(
            run_dir,
            status=status,
            status_err=status_err,
            manifest=manifest,
            manifest_err=manifest_err,
            report=report,
            report_err=report_err,
        )

    return _classify_without_current_contract(
        run_dir,
        status=status,
        manifest=manifest,
        report=report,
        report_err=report_err,
    )


def _classify_current(
    run_dir: str,
    *,
    status: dict[str, Any] | None,
    status_err: str,
    manifest: dict[str, Any] | None,
    manifest_err: str,
    report: dict[str, Any] | None,
    report_err: str,
) -> TerminalClassification:
    """Strict validation for a run that claims the current completion contract."""
    version = COMPLETION_CONTRACT_VERSION

    def corrupted(reason: str, **kw: Any) -> TerminalClassification:
        return TerminalClassification(TERMINAL_CORRUPTED, reason, contract_version=version, **kw)

    if status is None:
        return TerminalClassification(
            TERMINAL_INTERRUPTED if status_err == "missing" else TERMINAL_CORRUPTED,
            f"The run status is {status_err}, so this run has no verified outcome.",
            contract_version=version,
        )

    # A non-terminal or explicitly bad status settles the classification before
    # anything else is inspected. A finished-looking directory with a running
    # status is an interrupted run, not a success.
    state_token = _status_token(status)
    phase_token = _phase_token(status)
    if state_token in _FAILED_STATUS_TOKENS:
        return TerminalClassification(
            TERMINAL_FAILED, "This run finished with an error.", contract_version=version
        )
    if state_token in _INTERRUPTED_STATUS_TOKENS or phase_token in _INTERRUPTED_PHASE_TOKENS:
        return TerminalClassification(
            TERMINAL_INTERRUPTED,
            "This run did not finish; it was interrupted or is still running.",
            contract_version=version,
        )
    if phase_token != "final":
        return TerminalClassification(
            TERMINAL_INTERRUPTED,
            "This run never reached its final step.",
            contract_version=version,
        )
    if state_token != "success":
        return TerminalClassification(
            TERMINAL_INTERRUPTED,
            f"The run status does not report success (reported {state_token or 'nothing'}).",
            contract_version=version,
        )
    errors = status.get("errors")
    if isinstance(errors, list) and errors:
        return corrupted(
            "This run is recorded as successful but also recorded errors, "
            "so its outcome is contradictory."
        )

    if report is None:
        return corrupted(
            f"The mandatory run report is {report_err}. A current run cannot be opened "
            "as successful without it."
        )
    if manifest is None:
        return corrupted(
            f"The mandatory output manifest is {manifest_err}. A current run cannot be "
            "opened as successful without it."
        )

    report_block = report.get(REPORT_COMPLETION_KEY)
    if not isinstance(report_block, dict):
        return corrupted("The run report does not declare a completion record.")

    manifest_block = manifest.get(COMPLETION_KEY)
    if not isinstance(manifest_block, dict):
        return corrupted("The output manifest does not declare a completion record.")
    if manifest_block.get("final") is not True:
        return TerminalClassification(
            TERMINAL_INTERRUPTED,
            "The output manifest was never finalized, so this run stopped before it finished.",
            contract_version=version,
        )

    status_block = status.get(COMPLETION_KEY)
    if not isinstance(status_block, dict):
        return corrupted("The run status does not declare a completion record.")

    # 5. Run identity must agree across every terminal file.
    ids = {
        STATUS_FILENAME: str(status_block.get("run_id", "") or status.get("run_id", "") or ""),
        MANIFEST_FILENAME: str(manifest_block.get("run_id", "") or ""),
        RUN_REPORT_FILENAME: str(report_block.get("run_id", "") or ""),
    }
    if not all(ids.values()):
        blank = sorted(name for name, value in ids.items() if not value)
        return corrupted("These run records carry no run identity: " + ", ".join(blank))
    if len(set(ids.values())) != 1:
        detail = ", ".join(f"{name}={value}" for name, value in sorted(ids.items()))
        return corrupted(f"These run records describe different runs ({detail}).")
    run_id = ids[STATUS_FILENAME]

    # 6. The status pins the manifest; the manifest pins everything else.
    recorded_manifest_digest = str(status_block.get("manifest_sha256", "") or "")
    if not recorded_manifest_digest:
        return corrupted("The run status does not record the identity of the output manifest.", run_id=run_id)
    actual_manifest_digest = sha256_file(os.path.join(run_dir, MANIFEST_FILENAME))
    if actual_manifest_digest != recorded_manifest_digest:
        return corrupted(
            "The output manifest has changed since the run finished, so its contents "
            "no longer match the completed run.",
            run_id=run_id,
        )

    run_mode = manifest_block.get("run_mode")
    if not isinstance(run_mode, dict):
        return corrupted("The output manifest does not record what this run actually did.", run_id=run_id)

    structural_error = run_mode_structural_error(run_mode)
    if structural_error:
        return corrupted(
            f"The output manifest describes an impossible run ({structural_error}).",
            run_id=run_id,
        )

    index_error = continuous_index_error(run_mode, manifest_block.get("deliverables"))
    if index_error:
        return corrupted(
            f"This continuous run's window outputs are not accounted for: {index_error}.",
            run_id=run_id,
        )

    mismatch = _run_mode_disagreement(run_mode, status)
    if mismatch:
        return corrupted(
            f"The run status and the output manifest disagree about the run ({mismatch}).",
            run_id=run_id,
        )

    artifacts = manifest_block.get("artifacts")
    if not isinstance(artifacts, list):
        return corrupted("The output manifest does not list this run's outputs.", run_id=run_id)

    listed_required = {
        str(entry.get("relative_path", ""))
        for entry in artifacts
        if isinstance(entry, dict) and entry.get("required") is True
    }
    expected_required = {_to_posix_rel(p) for p in required_artifacts_for_run_mode(run_mode)}
    unlisted = sorted(expected_required - listed_required)
    if unlisted:
        return corrupted(
            "The output manifest does not account for outputs this run was required to "
            "produce: " + ", ".join(unlisted),
            run_id=run_id,
        )

    verification_error = _verify_recorded_artifacts(run_dir, artifacts)
    if verification_error:
        return corrupted(verification_error, run_id=run_id)

    return TerminalClassification(
        TERMINAL_SUCCESS_CURRENT,
        "This run finished successfully and all of its outputs were verified.",
        run_id=run_id,
        contract_version=version,
        run_mode=dict(run_mode),
    )


def _run_mode_disagreement(run_mode: dict[str, Any], status: dict[str, Any]) -> str:
    """Compare the manifest's declared mode with the status's own record of it."""
    checks = (
        ("run_profile", run_mode.get("run_profile"), status.get("run_profile")),
        ("run_type", run_mode.get("run_type"), status.get("run_type")),
        ("acquisition_mode", run_mode.get("acquisition_mode"), status.get("acquisition_mode")),
        ("traces_only", run_mode.get("traces_only"), status.get("traces_only")),
    )
    for name, manifest_value, status_value in checks:
        if status_value is None:
            continue
        if isinstance(manifest_value, bool) or isinstance(status_value, bool):
            if bool(manifest_value) != bool(status_value):
                return f"{name}"
        elif str(manifest_value or "") != str(status_value or ""):
            return f"{name}"
    return ""


def _verify_recorded_artifacts(run_dir: str, artifacts: list[Any]) -> str:
    """Verify every mandatory artifact the manifest records. Optional ones may be absent."""
    for entry in artifacts:
        if not isinstance(entry, dict):
            return "The output manifest contains an unreadable output record."
        rel_path = str(entry.get("relative_path", ""))
        if not rel_path:
            return "The output manifest contains an output record with no path."
        required = entry.get("required") is True
        abs_path = os.path.join(run_dir, _to_os_rel(rel_path))

        if not os.path.isfile(abs_path):
            if required:
                return f"A required output of this run is missing: {rel_path}"
            continue

        recorded_size = entry.get("size_bytes")
        if isinstance(recorded_size, int) and os.path.getsize(abs_path) != recorded_size:
            if required:
                return f"A required output of this run has changed since it finished: {rel_path}"
            continue

        recorded_digest = entry.get("sha256")
        if required and isinstance(recorded_digest, str) and recorded_digest:
            if sha256_file(abs_path) != recorded_digest:
                return f"A required output of this run has changed since it finished: {rel_path}"
    return ""


# ----------------------------------------------------------------------
# Runs that do not claim the current contract
# ----------------------------------------------------------------------


def _is_positively_legacy_report(report: dict[str, Any]) -> bool:
    """A real historical run report, not merely a file we failed to understand.

    The pre-contract pipeline always wrote both an ``analytical_contract`` and a
    ``configuration`` snapshot. Requiring both means an empty, truncated, or
    hand-made metadata file cannot masquerade as a historical run.

    A report that declares either the completion contract or the per-ROI
    feature-settings contract (4J16k39) was written by a current build. If such a
    run reaches this function its completion record has been lost, and it is
    damaged -- never legacy.
    """
    if REPORT_COMPLETION_KEY in report:
        return False
    if "feature_event_provenance" in report:
        return False
    return isinstance(report.get("analytical_contract"), dict) and isinstance(
        report.get("configuration"), dict
    )


_CURRENT_BUILD_REPORT_MARKERS = (REPORT_COMPLETION_KEY, "feature_event_provenance")


def _declares_current_build(report: dict[str, Any]) -> bool:
    return any(marker in report for marker in _CURRENT_BUILD_REPORT_MARKERS)


def _find_legacy_report(run_dir: str, root_report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Locate a positively-identified historical run report for this directory.

    Every report in the run is inspected. If *any* of them was written by a
    current build, the run is damaged rather than historical, and no legacy
    report is returned no matter what the others look like.
    """
    reports = [root_report]
    # Runs predating the root-report ordering gate only wrote the report beside
    # their analysis outputs.
    for rel in ("_analysis/phasic_out", "_analysis/tonic_out"):
        nested, _err = _read_json_object(os.path.join(run_dir, _to_os_rel(rel), RUN_REPORT_FILENAME))
        reports.append(nested)

    present = [report for report in reports if report is not None]
    if any(_declares_current_build(report) for report in present):
        return None
    for report in present:
        if _is_positively_legacy_report(report):
            return report
    return None


def _legacy_conflict(
    status: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    report: dict[str, Any] | None,
) -> TerminalClassification | None:
    """Detect an explicitly failed or unfinished legacy run."""
    sources: list[tuple[str, str, str]] = []
    if status is not None:
        sources.append((STATUS_FILENAME, _status_token(status), _phase_token(status)))
    if manifest is not None:
        sources.append((MANIFEST_FILENAME, str(manifest.get("status", "")).strip().lower(), ""))
    if report is not None:
        run_ctx = report.get("run_context") if isinstance(report.get("run_context"), dict) else {}
        for holder in (report, run_ctx):
            sources.append(
                (
                    RUN_REPORT_FILENAME,
                    str(holder.get("status", "")).strip().lower(),
                    str(holder.get("phase", "")).strip().lower(),
                )
            )

    for _name, state_token, phase_token in sources:
        if state_token in _FAILED_STATUS_TOKENS:
            return TerminalClassification(TERMINAL_FAILED, "This run finished with an error.")
        if state_token in _INTERRUPTED_STATUS_TOKENS or phase_token in _INTERRUPTED_PHASE_TOKENS:
            return TerminalClassification(
                TERMINAL_INTERRUPTED,
                "This run did not finish; it was interrupted or is still running.",
            )
    return None


def _legacy_declares_success(
    status: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    report: dict[str, Any] | None,
) -> bool:
    if status is not None:
        if (
            status.get("schema_version") == 1
            and _phase_token(status) == "final"
            and _status_token(status) == "success"
        ):
            return True
    if report is not None:
        holders = [report]
        if isinstance(report.get("run_context"), dict):
            holders.append(report["run_context"])
        for holder in holders:
            state_token = str(holder.get("status", "")).strip().lower()
            phase_token = str(holder.get("phase", "")).strip().lower()
            if state_token in _LEGACY_SUCCESS_TOKENS and (
                not phase_token or phase_token in _LEGACY_FINAL_TOKENS
            ):
                return True
    if manifest is not None:
        if str(manifest.get("status", "")).strip().lower() in _LEGACY_SUCCESS_TOKENS:
            return True
    return False


def _classify_without_current_contract(
    run_dir: str,
    *,
    status: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    report: dict[str, Any] | None,
    report_err: str,
) -> TerminalClassification:
    """Classify a run that makes no current-contract claim.

    It is legacy only when a historical run report positively identifies it.
    Missing or malformed metadata is corrupt, never legacy.
    """
    conflict = _legacy_conflict(status, manifest, report)
    if conflict is not None:
        return conflict

    legacy_report = _find_legacy_report(run_dir, report)
    if legacy_report is None:
        if report is None and report_err != "missing":
            return TerminalClassification(
                TERMINAL_CORRUPTED,
                f"This run's report is {report_err}, so the run cannot be verified.",
            )
        return TerminalClassification(
            TERMINAL_CORRUPTED,
            "This folder does not contain a complete record of a finished run, and it "
            "cannot be positively identified as a run from an earlier version of the app.",
        )

    if not _legacy_declares_success(status, manifest, legacy_report if report is None else report):
        return TerminalClassification(
            TERMINAL_INTERRUPTED,
            "This run from an earlier version of the app has no record of finishing successfully.",
        )

    return TerminalClassification(
        TERMINAL_SUCCESS_LEGACY,
        "This run was produced by an earlier version of the app. It reports success, but "
        "its outputs cannot be verified against the current completeness checks.",
    )
