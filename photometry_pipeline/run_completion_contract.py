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
import math
from dataclasses import dataclass, field
from typing import Any

from photometry_pipeline.input_processing_completeness import (
    FROZEN_INPUT_MANIFEST_FILENAME,
    INPUT_COMPLETENESS_FILENAME,
    expected_entries_digest,
    load_frozen_input_manifest,
    read_input_completeness,
)
from photometry_pipeline.guided_normalized_recording import (
    NormalizedRecordingError,
    deserialize_normalized_recording_description,
)
from photometry_pipeline.guided_normalized_recording_consumption import (
    NormalizedConsumedEvidenceError,
    build_npm_consumed_normalized_recording_evidence,
    build_rwd_consumed_normalized_recording_evidence,
    compare_consumed_normalized_recording_branches,
    compare_requested_and_consumed_normalized_recording,
)
from photometry_pipeline.guided_startup_transaction import (
    GUIDED_CANDIDATE_MANIFEST_FILENAME,
    GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME,
    GUIDED_PER_ROI_CORRECTION_FILENAME,
    GUIDED_STARTUP_PROVENANCE_FILENAME,
    GUIDED_STARTUP_STATUS_FILENAME,
)
from photometry_pipeline.guided_startup_claim import (
    GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME,
)

# B1: the Guided markers that cannot validly exist for a non-Guided (Full
# Control / legacy) run -- each is written exclusively by guided_startup_*.py
# or the wrapper's Guided-only code paths. Presence of any one of these
# establishes "this is a Guided run" for classify_guided_current_native_state
# below; GUIDED_PER_ROI_CORRECTION_FILENAME is conditional (native-correction
# runs only) but still definitive when present. Generic artifacts that a
# Full Control run can also produce (config_effective.yaml, command_invoked.txt,
# run_metadata.json, run_report.json, MANIFEST.json, status.json, HDF5 caches)
# are deliberately excluded -- they must never establish Guided identity alone.
GUIDED_DEFINITIVE_MARKER_FILENAMES = (
    GUIDED_CANDIDATE_MANIFEST_FILENAME,
    GUIDED_STARTUP_PROVENANCE_FILENAME,
    GUIDED_STARTUP_STATUS_FILENAME,
    GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME,
    GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME,
    GUIDED_PER_ROI_CORRECTION_FILENAME,
)

GUIDED_CURRENT_NATIVE_STATE_MIXED = "mixed"
GUIDED_CURRENT_NATIVE_STATE_CURRENT_NATIVE = "current_native"
GUIDED_CURRENT_NATIVE_STATE_CORRUPTED = "corrupted"
GUIDED_CURRENT_NATIVE_STATE_LEGACY = "legacy"
GUIDED_CURRENT_NATIVE_STATE_NOT_GUIDED = "not_guided"

# Bumped whenever the terminal set's meaning changes. A run declaring any other
# value is not something this build knows how to verify, so it fails closed.
COMPLETION_CONTRACT_VERSION = "run_completion.v1"
# Versioned run-level production correction provenance.  This is deliberately
# separate from the terminal-set version: provenance can evolve while the
# outer status/manifest contract remains compatible.
CORRECTION_PROVENANCE_SCHEMA_VERSION = "correction_provenance.v1"

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
# A run that finished successfully but contains scientist-approved missing
# sessions and/or an authorized final exclusion -- a distinct outcome, never
# indistinguishable clean success (4J16k41c).
TERMINAL_SUCCESS_WITH_MISSING = "success_with_missing"
TERMINAL_FAILED = "failed"
TERMINAL_INTERRUPTED = "interrupted"
TERMINAL_CORRUPTED = "corrupted"
TERMINAL_NOT_A_RUN = "not_a_run"

SUCCESS_STATES = (
    TERMINAL_SUCCESS_CURRENT,
    TERMINAL_SUCCESS_WITH_MISSING,
    TERMINAL_SUCCESS_LEGACY,
)

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
    missing_session_count: int = 0
    final_exclusion_count: int = 0

    @property
    def is_success(self) -> bool:
        return self.state in SUCCESS_STATES

    @property
    def is_current(self) -> bool:
        return self.state in (TERMINAL_SUCCESS_CURRENT, TERMINAL_SUCCESS_WITH_MISSING)

    @property
    def completed_with_missing(self) -> bool:
        return self.state == TERMINAL_SUCCESS_WITH_MISSING

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
    chunked_input_processing: bool = False,
    shared_input_manifest: bool = False,
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
        "chunked_input_processing": bool(chunked_input_processing),
        "shared_input_manifest": bool(shared_input_manifest),
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
    # Chunked intermittent analyses must account for every admitted input chunk
    # (4J16k41 / C8): the record proves nothing was silently omitted.
    if run_mode.get("chunked_input_processing"):
        if run_mode.get("phasic_analysis"):
            required.append(f"_analysis/phasic_out/{INPUT_COMPLETENESS_FILENAME}")
        if run_mode.get("tonic_analysis"):
            required.append(f"_analysis/tonic_out/{INPUT_COMPLETENESS_FILENAME}")
    # The one run-wide frozen input manifest binds every analysis to the same
    # admitted chunk set (4J16k41b).
    if run_mode.get("shared_input_manifest"):
        required.append(FROZEN_INPUT_MANIFEST_FILENAME)
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
    if run_mode.get("shared_input_manifest") and not run_mode.get("chunked_input_processing"):
        return "a shared input manifest is recorded for a run that processes no chunks"
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


def input_completeness_error(run_dir: str, run_mode: dict[str, Any]) -> str:
    """Validate the input-processing completeness records this run mode requires.

    For each chunked intermittent analysis that ran, the record must exist and
    reconcile: every admitted, non-excluded chunk processed exactly once, the one
    authorized exclusion is the final chronological chunk, and no chunk is missing
    or duplicated. "" when sound.
    """
    if not run_mode.get("chunked_input_processing"):
        return ""

    checks = []
    if run_mode.get("phasic_analysis"):
        checks.append(("phasic", os.path.join(run_dir, "_analysis", "phasic_out")))
    if run_mode.get("tonic_analysis"):
        checks.append(("tonic", os.path.join(run_dir, "_analysis", "tonic_out")))

    payloads: dict[str, Any] = {}
    for label, analysis_dir in checks:
        payload, error = read_input_completeness(analysis_dir)
        if payload is None:
            return f"the {label} analysis input-completeness record is {error}"
        payloads[label] = payload

    # When the wrapper froze one run-wide input manifest, every analysis must be
    # bound to it: same admitted chunks, order, sizes/identities, dispositions,
    # and authorized exclusion -- proven by a shared digest, not a chunk count.
    if run_mode.get("shared_input_manifest"):
        manifest, error = load_frozen_input_manifest(
            os.path.join(run_dir, FROZEN_INPUT_MANIFEST_FILENAME)
        )
        if manifest is None:
            return f"the run-wide frozen input manifest is {error}"
        run_digest = manifest["digest"]
        for label, payload in payloads.items():
            recorded = str(payload.get("frozen_manifest_digest", ""))
            if not recorded:
                return f"the {label} analysis does not record the frozen input identity"
            # The record's own admitted set must hash to the digest it claims,
            # so a tampered expected set cannot ride a copied digest field.
            if expected_entries_digest(payload.get("expected", [])) != recorded:
                return f"the {label} analysis record does not match its own frozen identity"
            if recorded != run_digest:
                return f"the {label} analysis used a different input set than the run manifest"
    return ""


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


def _text_value(value: Any) -> str:
    """Decode an HDF5 scalar/string attribute without changing its identity."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode(errors="replace")
    return str(value)


def _normalized_source(value: Any) -> str:
    """Use the same path identity semantics as C8 source accounting."""
    return os.path.normcase(os.path.abspath(os.path.normpath(_text_value(value))))


def _load_authoritative_analysis_sessions(
    run_dir: str, run_mode: dict[str, Any], analysis_kind: str
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, str]:
    """Load C8's expected and processed phasic session identities."""
    analysis_dir = os.path.join(run_dir, "_analysis", f"{analysis_kind}_out")
    payload, error = read_input_completeness(analysis_dir)
    if payload is None:
        return None, None, f"the {analysis_kind} analysis input-completeness record is {error}"

    expected = payload.get("expected")
    processed = payload.get("processed")
    if run_mode.get("shared_input_manifest"):
        manifest, manifest_error = load_frozen_input_manifest(
            os.path.join(run_dir, FROZEN_INPUT_MANIFEST_FILENAME)
        )
        if manifest is None:
            return None, None, f"the run-wide frozen input manifest is {manifest_error}"
        expected = manifest.get("expected")
    if not isinstance(expected, list) or not isinstance(processed, list):
        return None, None, f"the authoritative {analysis_kind} session index is malformed"
    return expected, processed, ""


def _correction_completion_error_for_analysis(
    run_dir: str, run_mode: dict[str, Any], analysis_kind: str
) -> str:
    """Verify requested/consumed correction evidence for a current phasic run.

    Older current runs that predate the versioned provenance section remain
    compatible: when neither ``run_metadata.json`` nor the nested report claims
    this section, the existing completion contract continues to decide success.
    Once the section is claimed, every field and every processed ROI/session is
    checked fail-closed against C8 and the canonical HDF5 cache.
    """
    analysis_dir = os.path.join(run_dir, "_analysis", f"{analysis_kind}_out")
    metadata, _metadata_error = _read_json_object(
        os.path.join(analysis_dir, "run_metadata.json")
    )
    nested_report, _ = _read_json_object(os.path.join(analysis_dir, RUN_REPORT_FILENAME))
    metadata_has_provenance = (
        isinstance(metadata, dict) and "correction_provenance" in metadata
    )
    report_has_provenance = False
    nested_provenance = None
    if isinstance(nested_report, dict):
        derived = nested_report.get("derived_settings")
        if isinstance(derived, dict):
            report_has_provenance = "correction_provenance" in derived
            if report_has_provenance:
                nested_provenance = derived["correction_provenance"]

    # Presence is a three-state contract across the two independent files.  A
    # present-but-null/list value still counts as a claim and must not be
    # mistaken for legacy absence.
    if not metadata_has_provenance and not report_has_provenance:
        # Deliberate pre-provenance compatibility path.
        return ""
    if metadata_has_provenance != report_has_provenance:
        present = "run_metadata.json" if metadata_has_provenance else "run_report.json"
        absent = "run_report.json" if metadata_has_provenance else "run_metadata.json"
        return (
            f"correction_provenance is present in {present} but absent from {absent}"
        )

    assert isinstance(metadata, dict)
    provenance = metadata["correction_provenance"]
    if not isinstance(provenance, dict) or not isinstance(nested_provenance, dict):
        return "both phasic correction_provenance values must be dictionaries"
    if provenance.get("schema_version") != CORRECTION_PROVENANCE_SCHEMA_VERSION:
        return "correction_provenance declares an unsupported schema version"
    if nested_provenance.get("schema_version") != CORRECTION_PROVENANCE_SCHEMA_VERSION:
        return "phasic report correction_provenance declares an unsupported schema version"
    if nested_provenance != provenance:
        return "phasic report and run_metadata correction provenance disagree"
    if provenance.get("analysis_mode") != analysis_kind:
        return f"correction_provenance is not bound to {analysis_kind} analysis"
    source = provenance.get("source")
    if source not in {"explicit_per_roi_map", "legacy_uniform_translation"}:
        return f"correction_provenance has an unknown source {source!r}"
    if analysis_kind == "tonic" and source == "legacy_uniform_translation":
        # Historical tonic uses its recording-global robust correction stage,
        # not the native per-ROI cache evidence verified below.
        return ""

    expected_rois = [str(roi) for roi in (run_mode.get("expected_rois") or [])]
    if len(expected_rois) != len(set(expected_rois)):
        return "the terminal run mode contains duplicate expected ROI identities"
    included = provenance.get("included_roi_ids")
    if not isinstance(included, list) or [str(roi) for roi in included] != expected_rois:
        return "correction_provenance included ROI identities do not match the run mode"
    requested = provenance.get("requested_by_roi")
    if not isinstance(requested, list):
        return "correction_provenance requested_by_roi is not a list"
    if len(requested) != len(expected_rois):
        return "correction_provenance does not contain exactly one entry per expected ROI"

    from photometry_pipeline.core.types import (
        CORRECTION_STRATEGY_FAMILIES,
        RESOLVED_DYNAMIC_FIT_MODES,
    )

    requested_by_roi: dict[str, dict[str, Any]] = {}
    for record in requested:
        if not isinstance(record, dict):
            return "correction_provenance contains an unreadable ROI entry"
        roi = record.get("roi_id")
        if not isinstance(roi, str) or roi in requested_by_roi:
            return "correction_provenance contains a missing or duplicate ROI identity"
        if roi not in set(expected_rois):
            return f"correction_provenance contains unknown ROI {roi!r}"
        family = record.get("strategy_family")
        selected = record.get("selected_strategy")
        mode = record.get("dynamic_fit_mode")
        if family not in CORRECTION_STRATEGY_FAMILIES:
            return f"correction_provenance has unknown strategy_family for ROI {roi!r}"
        if not isinstance(selected, str) or not selected:
            return f"correction_provenance has no selected_strategy for ROI {roi!r}"
        if family == "dynamic_fit":
            if mode not in RESOLVED_DYNAMIC_FIT_MODES or selected != mode:
                return f"correction_provenance has an invalid dynamic-fit mode for ROI {roi!r}"
        else:
            if selected != "signal_only_f0" or mode is not None:
                return f"correction_provenance has invalid Signal-Only fields for ROI {roi!r}"
        for identity_key in ("parameter_identity", "evidence_identity"):
            if not isinstance(record.get(identity_key, ""), str):
                return f"correction_provenance has malformed {identity_key} for ROI {roi!r}"
        requested_by_roi[roi] = record
    if set(requested_by_roi) != set(expected_rois):
        return "correction_provenance ROI coverage is incomplete"

    # A preview/continuous run without C8's intermittent session index cannot
    # claim the per-session completion check.  Its existing artifact contract
    # remains authoritative; full chunked runs continue below.
    if not run_mode.get("chunked_input_processing"):
        return ""

    expected, processed, session_error = _load_authoritative_analysis_sessions(
        run_dir, run_mode, analysis_kind
    )
    if session_error:
        return session_error
    assert expected is not None and processed is not None
    expected_by_index = {
        int(entry.get("index")): entry
        for entry in expected
        if isinstance(entry, dict) and isinstance(entry.get("index"), int)
    }
    process_entries = [
        entry
        for entry in expected
        if isinstance(entry, dict) and entry.get("disposition") == "process"
    ]
    from photometry_pipeline.input_processing_completeness import resolve_session_start_time

    processed_by_index: dict[int, dict[str, Any]] = {}
    for record in processed:
        if not isinstance(record, dict) or not isinstance(record.get("index"), int):
            return "the phasic processed session index contains an unreadable entry"
        index = int(record["index"])
        if index in processed_by_index:
            return f"the phasic processed session index duplicates session {index}"
        processed_by_index[index] = record
    for entry in process_entries:
        index = int(entry["index"])
        record = processed_by_index.get(index)
        if record is None:
            return f"processed session {index} is missing from the authoritative index"
        if _normalized_source(record.get("source", "")) != _normalized_source(entry.get("source", "")):
            return f"processed session {index} source does not match the authoritative index"
        expected_start = str(entry.get("expected_start_time", "")).strip()
        source_start = resolve_session_start_time(str(entry.get("source", "")))
        if expected_start and source_start is not None and source_start.isoformat() != expected_start:
            return f"processed session {index} timestamp does not match its source identity"

    cache_path = os.path.join(analysis_dir, f"{analysis_kind}_trace_cache.h5")
    if not os.path.isfile(cache_path):
        return "the phasic canonical trace cache is missing"

    try:
        import h5py
        import numpy as np

        coverage = float(provenance.get("finite_coverage_fraction", 0.80))
        if not math.isfinite(coverage) or not 0.0 < coverage <= 1.0:
            return "correction_provenance has an invalid finite-coverage policy"
        with h5py.File(cache_path, "r") as cache:
            roi_root = cache.get("roi")
            meta = cache.get("meta")
            if roi_root is None or meta is None:
                return "the phasic canonical trace cache lacks roi/meta groups"
            cache_rois = [str(roi) for roi in roi_root.keys()]
            if len(cache_rois) != len(set(cache_rois)) or set(cache_rois) != set(expected_rois):
                return "the phasic cache ROI identities do not match requested correction provenance"
            if "rois" in meta:
                meta_rois = [_text_value(value) for value in np.asarray(meta["rois"][()]).reshape(-1)]
                if len(meta_rois) != len(set(meta_rois)) or set(meta_rois) != set(expected_rois):
                    return "the phasic cache meta ROI identities are incomplete or unknown"

            if "chunk_ids" not in meta:
                return "the phasic cache has no canonical chunk identity index"
            meta_chunk_ids = [int(value) for value in np.asarray(meta["chunk_ids"][()]).reshape(-1)]
            if len(meta_chunk_ids) != len(set(meta_chunk_ids)):
                return "the phasic cache chunk identity index contains duplicates"
            expected_chunk_ids = {
                int(record["cache_chunk_id"])
                for record in processed_by_index.values()
                if isinstance(record.get("cache_chunk_id"), int)
                and not isinstance(record.get("cache_chunk_id"), bool)
            }
            if set(meta_chunk_ids) != expected_chunk_ids:
                return "the phasic cache chunk identities do not match processed C8 sessions"
            source_by_cache_id = {
                int(record["cache_chunk_id"]): _normalized_source(record.get("source", ""))
                for record in processed_by_index.values()
                if isinstance(record.get("cache_chunk_id"), int)
            }
            if "source_files" in meta:
                meta_sources = [_normalized_source(value) for value in np.asarray(meta["source_files"][()]).reshape(-1)]
                if len(meta_sources) != len(meta_chunk_ids):
                    return "the phasic cache source identity index is incomplete"
                for cache_id, meta_source in zip(meta_chunk_ids, meta_sources):
                    if meta_source != source_by_cache_id.get(cache_id, ""):
                        return "the phasic cache source identity index does not match C8"
            expected_entry_by_cache_id = {
                int(record["cache_chunk_id"]): expected_by_index[int(index)]
                for index, record in processed_by_index.items()
                if isinstance(record.get("cache_chunk_id"), int)
                and int(index) in expected_by_index
            }

            for roi in expected_rois:
                roi_group = roi_root[roi]
                chunk_names = list(roi_group.keys())
                chunk_ids: list[int] = []
                for name in chunk_names:
                    if not str(name).startswith("chunk_"):
                        return f"ROI {roi!r} has an unknown cache member {name!r}"
                    try:
                        chunk_ids.append(int(str(name)[len("chunk_"):]))
                    except ValueError:
                        return f"ROI {roi!r} has an invalid cache chunk identity {name!r}"
                if len(chunk_ids) != len(set(chunk_ids)) or set(chunk_ids) != expected_chunk_ids:
                    return f"ROI {roi!r} cache sessions do not match processed C8 sessions"

                requested_record = requested_by_roi[roi]
                family = requested_record["strategy_family"]
                selected = requested_record["selected_strategy"]
                requested_mode = requested_record.get("dynamic_fit_mode")
                for cache_id in sorted(expected_chunk_ids):
                    group = roi_group[f"chunk_{cache_id}"]
                    source = _normalized_source(group.attrs.get("source_file", ""))
                    if source != source_by_cache_id.get(cache_id, ""):
                        return f"ROI {roi!r} session {cache_id} source identity does not match C8"
                    required_datasets = ("time_sec", "sig_raw", "dff")
                    if any(name not in group for name in required_datasets):
                        return f"ROI {roi!r} session {cache_id} lacks a canonical correction dataset"
                    time_sec = np.asarray(group["time_sec"][()])
                    sig_raw = np.asarray(group["sig_raw"][()])
                    dff = np.asarray(group["dff"][()])
                    n = int(time_sec.size)
                    if (
                        time_sec.ndim != 1
                        or sig_raw.ndim != 1
                        or dff.ndim != 1
                        or n == 0
                        or sig_raw.size != n
                        or dff.size != n
                    ):
                        return f"ROI {roi!r} session {cache_id} canonical dataset shapes disagree"
                    if not np.all(np.isfinite(time_sec)) or time_sec[0] < 0.0:
                        return f"ROI {roi!r} session {cache_id} has invalid canonical time identity"
                    # Canonical time origin is not universally 0.0: RWD's
                    # strict grid (io.adapters._resample_strict) always
                    # starts at sample index 0 by construction, but NPM's
                    # strict grid (io.adapters._resolve_npm_strict_grid) is
                    # anchored to the UV/signal overlap origin and is
                    # "intentionally not re-zeroed to inner_start: staggered
                    # streams may therefore produce a first output time
                    # greater than zero" (see that function's own docstring
                    # comment). Both formats' real construction algorithms
                    # guarantee the first sample lands exactly on the
                    # target-rate grid (an integer multiple of 1/fs_hz), so
                    # that -- not a hardcoded zero -- is the genuine
                    # format-neutral invariant to verify here.
                    fs_attr = group.attrs.get("fs_hz")
                    try:
                        fs_val = float(fs_attr)
                    except (TypeError, ValueError):
                        fs_val = float("nan")
                    if not math.isfinite(fs_val) or fs_val <= 0.0:
                        return f"ROI {roi!r} session {cache_id} has invalid canonical time identity"
                    samples_from_origin = float(time_sec[0]) * fs_val
                    nearest_sample = round(samples_from_origin)
                    if nearest_sample < 0 or abs(samples_from_origin - nearest_sample) > 1e-6:
                        return f"ROI {roi!r} session {cache_id} has invalid canonical time identity"
                    if n > 1 and not np.all(np.diff(time_sec) > 0):
                        return f"ROI {roi!r} session {cache_id} time is not strictly increasing"
                    expected_entry = expected_entry_by_cache_id.get(cache_id)
                    expected_duration = (
                        expected_entry.get("expected_duration_sec")
                        if isinstance(expected_entry, dict)
                        else None
                    )
                    if expected_duration is not None and n > 1:
                        try:
                            duration = float(expected_duration)
                            dt = float(np.median(np.diff(time_sec)))
                            if abs(float(time_sec[-1]) - duration) > max(
                                0.10, 2.0 * abs(dt), 0.05 * abs(duration)
                            ):
                                return f"ROI {roi!r} session {cache_id} elapsed time does not match C8"
                        except (TypeError, ValueError):
                            return f"ROI {roi!r} session {cache_id} has malformed expected duration"
                    coverage_required = max(1, int(math.ceil(coverage * n)))
                    if int(np.sum(np.isfinite(dff))) < coverage_required:
                        return f"ROI {roi!r} session {cache_id} canonical dF/F coverage is insufficient"

                    if _text_value(group.attrs.get("correction_execution_status", "")) != "consumed":
                        return f"ROI {roi!r} session {cache_id} has no consumed correction status"
                    if _text_value(group.attrs.get("correction_strategy_family", "")) != family:
                        return f"ROI {roi!r} session {cache_id} strategy family mismatches the request"
                    if _text_value(group.attrs.get("correction_selected_strategy", "")) != selected:
                        return f"ROI {roi!r} session {cache_id} selected strategy mismatches the request"
                    for identity_key, attr_name in (
                        ("parameter_identity", "correction_parameter_identity"),
                        ("evidence_identity", "correction_evidence_identity"),
                    ):
                        requested_identity = requested_record.get(identity_key, "")
                        if requested_identity and _text_value(group.attrs.get(attr_name, "")) != requested_identity:
                            return f"ROI {roi!r} session {cache_id} {identity_key} mismatches the request"

                    dynamic_mode = _text_value(group.attrs.get("correction_dynamic_fit_mode", ""))
                    if family == "dynamic_fit":
                        if dynamic_mode != requested_mode:
                            return f"ROI {roi!r} session {cache_id} dynamic-fit mode mismatches the request"
                        if _text_value(group.attrs.get("dynamic_fit_mode_resolved", "")) != requested_mode:
                            return f"ROI {roi!r} session {cache_id} lacks resolved dynamic-fit mode metadata"
                        if not _text_value(group.attrs.get("dynamic_fit_engine", "")):
                            return f"ROI {roi!r} session {cache_id} lacks dynamic-fit engine metadata"
                        if "fit_ref" not in group:
                            return f"ROI {roi!r} session {cache_id} lacks fit_ref"
                        fit_ref = np.asarray(group["fit_ref"][()])
                        if fit_ref.ndim != 1 or fit_ref.size != n or int(np.sum(np.isfinite(fit_ref))) < coverage_required:
                            return f"ROI {roi!r} session {cache_id} fit_ref coverage is insufficient"
                        if "signal_only_f0_baseline" in group or "signal_only_f0_production_available" in group.attrs:
                            return f"ROI {roi!r} session {cache_id} carries orphan Signal-Only production evidence"
                        if any(
                            key in group.attrs
                            for key in (
                                "signal_only_f0_production_formula",
                                "signal_only_f0_production_baseline_source",
                            )
                        ):
                            return f"ROI {roi!r} session {cache_id} carries orphan Signal-Only production metadata"
                    else:
                        if dynamic_mode:
                            return f"ROI {roi!r} session {cache_id} Signal-Only entry carries dynamic-fit mode"
                        if any(
                            key in group.attrs
                            for key in ("dynamic_fit_mode_resolved", "dynamic_fit_engine")
                        ):
                            return f"ROI {roi!r} session {cache_id} Signal-Only entry carries dynamic-fit attribution"
                        if "signal_only_f0_baseline" not in group:
                            return f"ROI {roi!r} session {cache_id} lacks production F0 baseline"
                        baseline = np.asarray(group["signal_only_f0_baseline"][()])
                        signal_only_min_required = max(10, coverage_required)
                        if baseline.ndim != 1 or baseline.size != n or int(np.sum(np.isfinite(baseline))) < signal_only_min_required:
                            return f"ROI {roi!r} session {cache_id} production F0 coverage is insufficient"
                        if not bool(group.attrs.get("signal_only_f0_production_available", False)):
                            return f"ROI {roi!r} session {cache_id} lacks production Signal-Only QC"
                        if _text_value(group.attrs.get("signal_only_f0_production_baseline_source", "")) != "signal_only_f0_candidate_uncapped":
                            return f"ROI {roi!r} session {cache_id} has the wrong Signal-Only baseline source"
                        if _text_value(group.attrs.get("signal_only_f0_production_formula", "")) != "100 * (signal - f0) / f0":
                            return f"ROI {roi!r} session {cache_id} has the wrong Signal-Only formula"
                        if "fit_ref" in group and np.any(np.isfinite(np.asarray(group["fit_ref"][()]))):
                            return f"ROI {roi!r} session {cache_id} Signal-Only fit_ref is not all NaN"
    except Exception as exc:  # noqa: BLE001 - malformed cache must fail closed
        if isinstance(exc, (ValueError, KeyError, OSError)):
            return f"the {analysis_kind} canonical correction cache is malformed: {exc}"
        return f"the {analysis_kind} canonical correction cache could not be verified: {exc}"
    return ""


def correction_completion_error(run_dir: str, run_mode: dict[str, Any]) -> str:
    """Verify every analysis branch that claims native correction provenance."""
    for analysis_kind, enabled_key in (
        ("tonic", "tonic_analysis"),
        ("phasic", "phasic_analysis"),
    ):
        if not run_mode.get(enabled_key):
            continue
        error = _correction_completion_error_for_analysis(
            run_dir, run_mode, analysis_kind
        )
        if error:
            return error
    return ""


def _load_requested_normalized_recording(run_dir: str):
    """Deserialize and identity-verify the run's authorized normalized
    recording description. Returns the description, or raises
    NormalizedRecordingError with an actionable category on any failure."""
    path = os.path.join(run_dir, GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise NormalizedRecordingError(
            "unreadable_normalized_recording_description",
            f"the normalized recording description is unreadable: {exc}",
        ) from exc
    return deserialize_normalized_recording_description(payload)


def normalized_recording_completion_error(run_dir: str, run_mode: dict[str, Any]) -> str:
    """Verify the run's authorized normalized recording description against
    what each enabled analysis branch actually consumed.

    Reuses the shared adapter-neutral comparator
    (guided_normalized_recording_consumption.compare_requested_and_consumed_normalized_recording)
    for every enabled branch independently, then verifies both branches
    consumed the same authorized session set and the same actually-used
    cadence when both are enabled (chronology/ROI-membership facts are
    orthogonal to correction strategy, so unlike correction_completion_error
    there is no tonic legacy-uniform shortcut here).

    "" when the run is not a Guided run at all (no normalized recording
    description and no candidate manifest present -- an ordinary Full
    Control/legacy run, entirely unaffected). A candidate manifest present
    without a normalized recording description is a current Guided run
    whose mandatory provenance is missing, and fails closed.
    """
    state = classify_guided_current_native_state(run_dir)
    normalized_path = os.path.join(
        run_dir, GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME
    )
    if state == GUIDED_CURRENT_NATIVE_STATE_NOT_GUIDED:
        return ""
    if state == GUIDED_CURRENT_NATIVE_STATE_LEGACY:
        return ""
    if state == GUIDED_CURRENT_NATIVE_STATE_MIXED:
        return "mixed Guided and legacy provenance"
    if not os.path.isfile(normalized_path):
        return (
            "this Guided run has no normalized recording description despite "
            "carrying definitive Guided provenance"
        )

    try:
        requested = _load_requested_normalized_recording(run_dir)
    except NormalizedRecordingError as exc:
        return f"the normalized recording description could not be verified: {exc}"

    enabled_branches = [
        analysis_kind
        for analysis_kind, enabled_key in (
            ("phasic", "phasic_analysis"),
            ("tonic", "tonic_analysis"),
        )
        if run_mode.get(enabled_key)
    ]
    if not enabled_branches:
        return ""

    if requested.adapter_format == "rwd":
        build_consumed_evidence = build_rwd_consumed_normalized_recording_evidence
    elif requested.adapter_format == "npm":
        build_consumed_evidence = build_npm_consumed_normalized_recording_evidence
    else:
        return (
            f"unsupported adapter format {requested.adapter_format!r} for "
            "completion verification"
        )

    consumed_by_branch = {}
    for analysis_kind in enabled_branches:
        try:
            consumed = build_consumed_evidence(
                run_dir=run_dir, analysis_kind=analysis_kind, requested=requested
            )
        except NormalizedConsumedEvidenceError as exc:
            return (
                f"the {analysis_kind} analysis consumed normalized recording "
                f"evidence could not be established: {exc}"
            )
        comparison_error = compare_requested_and_consumed_normalized_recording(
            requested, consumed
        )
        if comparison_error:
            return comparison_error
        consumed_by_branch[analysis_kind] = consumed

    if "phasic" in consumed_by_branch and "tonic" in consumed_by_branch:
        phasic_consumed = consumed_by_branch["phasic"]
        tonic_consumed = consumed_by_branch["tonic"]
        cross_branch_error = compare_consumed_normalized_recording_branches(
            phasic_consumed, tonic_consumed
        )
        if cross_branch_error:
            return cross_branch_error

    return ""


def classify_guided_current_native_state(run_dir: str, run_mode: dict[str, Any] | None = None) -> str:
    """Classify a run directory's Guided provenance, mutually exclusively.

    One shared artifact-family classifier used by both terminal classification
    and completed Review, so the two never disagree.  Evidence verification is
    deliberately a separate step (``normalized_recording_completion_error``)
    so this classifier cannot recurse through terminal classification.
    Precedence:

    1. A definitive Guided marker coexists with a positively-identified
       legacy report shape -> "mixed" (the only path to "mixed" -- mere
       absence of one file is never "mixed").
    2. Any definitive Guided marker present -> this run is Guided-shaped:
       complete evidence and successful verification -> "current_native";
       anything missing, malformed, or contradictory (including a partial
       definitive marker set) -> "corrupted".
    3. No definitive marker at all: a positively-identified legacy shape
       -> "legacy"; otherwise -> "not_guided" (an ordinary current Full
       Control run, classified independently by classify_run_terminal_state).
    """
    present_markers = [
        name
        for name in GUIDED_DEFINITIVE_MARKER_FILENAMES
        if os.path.isfile(os.path.join(run_dir, name))
    ]
    root_report, _ = _read_json_object(os.path.join(run_dir, RUN_REPORT_FILENAME))
    legacy_report = _find_legacy_report(run_dir, root_report)

    if present_markers and legacy_report is not None:
        return GUIDED_CURRENT_NATIVE_STATE_MIXED

    if present_markers:
        mandatory = (
            GUIDED_CANDIDATE_MANIFEST_FILENAME,
            GUIDED_STARTUP_PROVENANCE_FILENAME,
            GUIDED_STARTUP_STATUS_FILENAME,
            GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME,
        )
        if not all(
            os.path.isfile(os.path.join(run_dir, name)) for name in mandatory
        ):
            return GUIDED_CURRENT_NATIVE_STATE_CORRUPTED
        try:
            _load_requested_normalized_recording(run_dir)
        except NormalizedRecordingError:
            return GUIDED_CURRENT_NATIVE_STATE_CORRUPTED
        return GUIDED_CURRENT_NATIVE_STATE_CURRENT_NATIVE

    if legacy_report is not None:
        return GUIDED_CURRENT_NATIVE_STATE_LEGACY
    return GUIDED_CURRENT_NATIVE_STATE_NOT_GUIDED


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
    completeness_error = input_completeness_error(run_dir, run_mode)
    if completeness_error:
        return f"input chunks are not fully accounted for: {completeness_error}"
    correction_error = correction_completion_error(run_dir, run_mode)
    if correction_error:
        return f"correction evidence is incomplete or inconsistent: {correction_error}"
    normalized_recording_error = normalized_recording_completion_error(run_dir, run_mode)
    if normalized_recording_error:
        return (
            "normalized recording provenance is incomplete or inconsistent: "
            f"{normalized_recording_error}"
        )

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

    guided_state = classify_guided_current_native_state(run_dir)
    if guided_state == GUIDED_CURRENT_NATIVE_STATE_MIXED:
        return TerminalClassification(
            TERMINAL_CORRUPTED,
            "This run mixes definitive Guided provenance with a positively identified "
            "legacy result, so its provenance cannot be trusted.",
        )
    if guided_state == GUIDED_CURRENT_NATIVE_STATE_CORRUPTED:
        return TerminalClassification(
            TERMINAL_CORRUPTED,
            "This run carries incomplete or malformed definitive Guided provenance.",
        )

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
            guided_state=guided_state,
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
    guided_state: str,
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

    completeness_error = input_completeness_error(run_dir, run_mode)
    if completeness_error:
        return corrupted(
            f"This run's input chunks are not fully accounted for: {completeness_error}.",
            run_id=run_id,
        )

    correction_error = correction_completion_error(run_dir, run_mode)
    if correction_error:
        return corrupted(
            f"This run's correction evidence is incomplete or inconsistent: {correction_error}.",
            run_id=run_id,
        )

    if guided_state == GUIDED_CURRENT_NATIVE_STATE_CURRENT_NATIVE:
        normalized_recording_error = normalized_recording_completion_error(
            run_dir, run_mode
        )
        if normalized_recording_error:
            return corrupted(
                "This run's normalized recording provenance is incomplete or "
                f"inconsistent: {normalized_recording_error}.",
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

    # A run with scientist-approved missing sessions or an authorized final
    # exclusion completes as a distinct outcome, never indistinguishable clean
    # success.
    missing_count, exclusion_count = _authorized_gap_counts(run_dir, run_mode)
    if missing_count or exclusion_count:
        pieces = []
        if missing_count:
            pieces.append(f"{missing_count} approved missing session(s)")
        if exclusion_count:
            pieces.append("an incomplete final chunk excluded")
        return TerminalClassification(
            TERMINAL_SUCCESS_WITH_MISSING,
            "This run finished successfully but is completed with "
            + " and ".join(pieces)
            + "; those intervals are preserved as explicit gaps.",
            run_id=run_id,
            contract_version=version,
            run_mode=dict(run_mode),
            missing_session_count=missing_count,
            final_exclusion_count=exclusion_count,
        )

    return TerminalClassification(
        TERMINAL_SUCCESS_CURRENT,
        "This run finished successfully and all of its outputs were verified.",
        run_id=run_id,
        contract_version=version,
        run_mode=dict(run_mode),
    )


def _authorized_gap_counts(run_dir: str, run_mode: dict[str, Any]) -> tuple[int, int]:
    """Count approved missing sessions and authorized final exclusions in the run.

    Read from the run-wide session index (the frozen manifest) when present, else
    from an analysis completeness record. Counts are per-session, run-wide.
    """
    from photometry_pipeline.input_processing_completeness import (
        DISPOSITION_AUTHORIZED_EXCLUSION,
        DISPOSITION_AUTHORIZED_MISSING,
        load_frozen_input_manifest,
    )

    expected = None
    if run_mode.get("shared_input_manifest"):
        manifest, _err = load_frozen_input_manifest(
            os.path.join(run_dir, FROZEN_INPUT_MANIFEST_FILENAME)
        )
        if manifest is not None:
            expected = manifest.get("expected")
    if expected is None:
        for analysis in ("phasic_out", "tonic_out"):
            payload, _err = read_input_completeness(
                os.path.join(run_dir, "_analysis", analysis)
            )
            if payload is not None:
                expected = payload.get("expected")
                break
    if not isinstance(expected, list):
        return 0, 0
    missing = sum(1 for e in expected if e.get("disposition") == DISPOSITION_AUTHORIZED_MISSING)
    exclusions = sum(
        1 for e in expected if e.get("disposition") == DISPOSITION_AUTHORIZED_EXCLUSION
    )
    return missing, exclusions


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
