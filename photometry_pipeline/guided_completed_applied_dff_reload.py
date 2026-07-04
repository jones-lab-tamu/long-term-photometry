"""Module for loading and validating completed-run applied-dF/F provenance."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from photometry_pipeline.guided_applied_dff_orchestration import (
    GuidedAppliedDffOrchestrationError,
    build_guided_applied_dff_manifest_rows,
)


@dataclass(frozen=True)
class GuidedCompletedAppliedDffRow:
    roi_id: str
    batch_roi: str
    strategy_family: str
    selected_strategy: str
    dynamic_fit_mode: str | None
    batch_strategy: str
    output_name: str
    output_dir: str
    status: str
    pipeline_summary_path: str | None
    output_dir_exists: bool
    pipeline_summary_exists: bool
    error: str | None = None


@dataclass(frozen=True)
class GuidedCompletedAppliedDffIssue:
    category: str
    severity: str  # "info", "warning", "blocking"
    message: str
    path: str | None = None


@dataclass(frozen=True)
class GuidedCompletedAppliedDffState:
    present: bool
    provenance_path: str | None = None
    schema_version: str | None = None
    overall_status: str | None = None
    production_analysis: bool | None = None
    preview_only: bool | None = None
    orchestration_capability_enabled: bool | None = None
    production_strategy_map_version: str | None = None
    phasic_cache_path: str | None = None
    phasic_cache_exists_at_load: bool | None = None
    batch_manifest_path: str | None = None
    batch_manifest_exists_at_load: bool | None = None
    applied_output_root: str | None = None
    applied_output_root_exists_at_load: bool | None = None
    batch_summary_path: str | None = None
    batch_summary_exists_at_load: bool | None = None
    batch_returncode: int | None = None
    failure_policy: str | None = None
    requested_strategy_map: dict | None = None
    rows: tuple[GuidedCompletedAppliedDffRow, ...] = ()
    issues: tuple[GuidedCompletedAppliedDffIssue, ...] = ()

    @classmethod
    def absent(cls) -> GuidedCompletedAppliedDffState:
        """Create a default absent state cleanly."""
        return cls(
            present=False,
            issues=(
                GuidedCompletedAppliedDffIssue(
                    category="applied_dff_provenance_not_present",
                    severity="info",
                    message="Applied-dF/F provenance file is not present.",
                ),
            ),
        )

    @property
    def has_blocking_issues(self) -> bool:
        """Check if there are any blocking issues."""
        return any(issue.severity == "blocking" for issue in self.issues)

    def blocking_issues(self) -> list[GuidedCompletedAppliedDffIssue]:
        """Return all blocking issues."""
        return [issue for issue in self.issues if issue.severity == "blocking"]


def format_guided_completed_applied_dff_summary(
    state: GuidedCompletedAppliedDffState,
) -> str:
    """Format completed-run applied-dF/F state for read-only review."""
    if not state.present:
        return (
            "Applied dF/F: not present\n"
            "Guided applied-dF/F provenance was not found."
        )

    blocking_count = sum(
        issue.severity == "blocking" for issue in state.issues
    )
    warning_count = sum(
        issue.severity == "warning" for issue in state.issues
    )
    if blocking_count:
        display_status = "invalid provenance"
    elif state.overall_status in {"succeeded", "failed", "running"}:
        display_status = state.overall_status
    else:
        display_status = "unknown"

    def yes_no(value: bool | None) -> str:
        if value is True:
            return "yes"
        if value is False:
            return "no"
        return "unknown"

    lines = [
        f"Applied dF/F: {display_status}",
        f"Overall status: {state.overall_status or 'unknown'}",
        f"Production analysis: {yes_no(state.production_analysis)}",
        f"Preview only: {yes_no(state.preview_only)}",
        "Orchestration capability enabled: "
        f"{yes_no(state.orchestration_capability_enabled)}",
        f"Issues: {blocking_count} blocking, {warning_count} warning",
    ]

    if state.rows:
        lines.append("ROI strategies:")
        for row in state.rows:
            strategy = (
                f"{row.strategy_family}; selected {row.selected_strategy}"
            )
            if row.dynamic_fit_mode:
                strategy += f"; dynamic-fit mode {row.dynamic_fit_mode}"
            output_state = (
                "output present" if row.output_dir_exists else "output missing"
            )
            summary_state = (
                "summary present"
                if row.pipeline_summary_exists
                else "summary missing"
                if row.pipeline_summary_path
                else "summary not recorded"
            )
            lines.append(
                f"- {row.roi_id}: {strategy}; batch {row.batch_strategy}; "
                f"status {row.status}; {output_state}; {summary_state}"
            )

    visible_issues = [
        issue
        for issue in state.issues
        if issue.severity in {"blocking", "warning"}
    ]
    if visible_issues:
        lines.append("Issues:")
        lines.extend(
            f"- {issue.severity}: {issue.category} — {issue.message}"
            for issue in visible_issues
        )
    return "\n".join(lines)


def _is_subpath(child: Path, parent: Path) -> bool:
    """Check if child path is under parent path."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resolve_from_run(value: object, run_dir: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (run_dir / path).resolve()


def load_guided_completed_applied_dff_state(run_dir: str | Path | None) -> GuidedCompletedAppliedDffState:
    """
    Read-only loader for guided completed run applied-dF/F provenance.
    """
    if run_dir is None:
        return GuidedCompletedAppliedDffState.absent()

    run_dir_resolved = Path(run_dir).resolve()
    provenance_path = run_dir_resolved / "applied_dff" / "guided_applied_dff_provenance.json"

    if not provenance_path.exists():
        return GuidedCompletedAppliedDffState.absent()

    issues: list[GuidedCompletedAppliedDffIssue] = []

    # 1. Parse JSON
    try:
        prov_data = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return GuidedCompletedAppliedDffState(
            present=True,
            provenance_path=str(provenance_path),
            issues=(
                GuidedCompletedAppliedDffIssue(
                    category="applied_dff_provenance_malformed",
                    severity="blocking",
                    message=f"Applied-dF/F provenance JSON is malformed: {exc}",
                    path=str(provenance_path),
                ),
            ),
        )

    if not isinstance(prov_data, dict):
        return GuidedCompletedAppliedDffState(
            present=True,
            provenance_path=str(provenance_path),
            issues=(
                GuidedCompletedAppliedDffIssue(
                    category="applied_dff_provenance_malformed",
                    severity="blocking",
                    message="Applied-dF/F provenance root is not a JSON object",
                    path=str(provenance_path),
                ),
            ),
        )

    # 2. Check Required Fields
    required_fields = [
        "schema_version",
        "overall_status",
        "production_analysis",
        "preview_only",
        "orchestration_capability_enabled",
        "production_strategy_map_version",
        "requested_strategy_map",
        "phasic_cache_path",
        "batch_manifest_path",
        "applied_output_root",
        "rows",
    ]
    missing_fields = [f for f in required_fields if f not in prov_data]
    for field_name in missing_fields:
        issues.append(
            GuidedCompletedAppliedDffIssue(
                category="applied_dff_provenance_missing_required_field",
                severity="blocking",
                message=f"Missing required field: '{field_name}'",
                path=str(provenance_path),
            )
        )

    # Extract fields with safe defaults for missing ones
    schema_version = prov_data.get("schema_version")
    overall_status = prov_data.get("overall_status")
    production_analysis = prov_data.get("production_analysis")
    preview_only = prov_data.get("preview_only")
    orchestration_capability_enabled = prov_data.get("orchestration_capability_enabled")
    production_strategy_map_version = prov_data.get("production_strategy_map_version")
    phasic_cache_path_str = prov_data.get("phasic_cache_path")
    batch_manifest_path_str = prov_data.get("batch_manifest_path")
    applied_output_root_str = prov_data.get("applied_output_root")
    batch_summary_path_str = prov_data.get("batch_summary_path")
    batch_returncode = prov_data.get("batch_returncode")
    failure_policy = prov_data.get("failure_policy")
    requested_strategy_map = prov_data.get("requested_strategy_map")
    raw_rows = prov_data.get("rows", [])

    # 3. Validate Schema Version
    if "schema_version" in prov_data and schema_version != "v1":
        issues.append(
            GuidedCompletedAppliedDffIssue(
                category="applied_dff_provenance_unsupported_schema_version",
                severity="blocking",
                message=f"Unsupported schema version: '{schema_version}' (only 'v1' is supported)",
                path=str(provenance_path),
            )
        )

    if (
        "production_strategy_map_version" in prov_data
        and production_strategy_map_version
        != "per_roi_correction_strategy_map.v1"
    ):
        issues.append(
            GuidedCompletedAppliedDffIssue(
                category=(
                    "applied_dff_provenance_unsupported_strategy_map_version"
                ),
                severity="blocking",
                message=(
                    "Top-level provenance production_strategy_map_version "
                    "must be exactly 'per_roi_correction_strategy_map.v1'; "
                    f"got '{production_strategy_map_version}'."
                ),
                path=str(provenance_path),
            )
        )

    # 4. Production / Preview Validation
    if "production_analysis" in prov_data and production_analysis is not True:
        issues.append(
            GuidedCompletedAppliedDffIssue(
                category="applied_dff_provenance_not_production_analysis",
                severity="blocking",
                message="Provenance production_analysis flag must be True.",
                path=str(provenance_path),
            )
        )
    if "preview_only" in prov_data and preview_only is not False:
        issues.append(
            GuidedCompletedAppliedDffIssue(
                category="applied_dff_provenance_marked_preview_only",
                severity="blocking",
                message="Provenance preview_only flag must be False.",
                path=str(provenance_path),
            )
        )

    # 5. Capability Gate Validation
    if "orchestration_capability_enabled" in prov_data and orchestration_capability_enabled is not True:
        issues.append(
            GuidedCompletedAppliedDffIssue(
                category="applied_dff_orchestration_capability_not_enabled",
                severity="blocking",
                message="Provenance orchestration_capability_enabled flag must be True.",
                path=str(provenance_path),
            )
        )

    # 6. Overall Status Validation
    if "overall_status" in prov_data:
        if overall_status not in ("running", "succeeded", "failed"):
            issues.append(
                GuidedCompletedAppliedDffIssue(
                    category="applied_dff_provenance_invalid_status",
                    severity="blocking",
                    message=f"Invalid overall_status: '{overall_status}'",
                    path=str(provenance_path),
                )
            )
        elif overall_status == "running":
            issues.append(
                GuidedCompletedAppliedDffIssue(
                    category="applied_dff_provenance_still_running_in_completed_run",
                    severity="blocking",
                    message="Orchestration status is still 'running' in a completed run.",
                    path=str(provenance_path),
                )
            )
        elif overall_status == "failed":
            error_msg = prov_data.get("error") or "Unknown error"
            issues.append(
                GuidedCompletedAppliedDffIssue(
                    category="applied_dff_orchestration_failed",
                    severity="warning",
                    message=f"Applied dF/F batch orchestration failed: {error_msg}",
                    path=str(provenance_path),
                )
            )

    # 7. Resolve paths before enforcing exact-location and containment rules.
    applied_dff_dir = (run_dir_resolved / "applied_dff").resolve()
    phasic_cache_exists_at_load = None
    if phasic_cache_path_str:
        p_cache = _resolve_from_run(phasic_cache_path_str, run_dir_resolved)
        expected_cache = (
            run_dir_resolved
            / "_analysis"
            / "phasic_out"
            / "phasic_trace_cache.h5"
        ).resolve()
        if p_cache != expected_cache:
            issues.append(
                GuidedCompletedAppliedDffIssue(
                    category="applied_dff_phasic_cache_path_unexpected",
                    severity="blocking",
                    message=(
                        "phasic_cache_path must resolve to the production "
                        f"Guided cache location '{expected_cache}', got '{p_cache}'."
                    ),
                )
            )
        phasic_cache_exists_at_load = p_cache.is_file()

    batch_manifest_exists_at_load = None
    if batch_manifest_path_str:
        p_manifest = _resolve_from_run(
            batch_manifest_path_str, run_dir_resolved
        )
        if not _is_subpath(p_manifest, applied_dff_dir):
            issues.append(
                GuidedCompletedAppliedDffIssue(
                    category="path_escapes_run_dir",
                    severity="blocking",
                    message=(
                        "batch_manifest_path escapes applied_dff root: "
                        f"'{batch_manifest_path_str}'"
                    ),
                )
            )
        batch_manifest_exists_at_load = p_manifest.is_file()

    applied_output_root_exists_at_load = None
    applied_output_root_resolved = None
    if applied_output_root_str:
        p_out_root = _resolve_from_run(
            applied_output_root_str, run_dir_resolved
        )
        if not _is_subpath(p_out_root, applied_dff_dir):
            issues.append(
                GuidedCompletedAppliedDffIssue(
                    category="path_escapes_run_dir",
                    severity="blocking",
                    message=(
                        "applied_output_root escapes applied_dff root: "
                        f"'{applied_output_root_str}'"
                    ),
                )
            )
        applied_output_root_resolved = p_out_root
        applied_output_root_exists_at_load = p_out_root.is_dir()

    batch_summary_exists_at_load = None
    if batch_summary_path_str:
        p_summary = _resolve_from_run(batch_summary_path_str, run_dir_resolved)
        if not _is_subpath(p_summary, applied_dff_dir):
            issues.append(
                GuidedCompletedAppliedDffIssue(
                    category="path_escapes_run_dir",
                    severity="blocking",
                    message=(
                        "batch_summary_path escapes applied_dff root: "
                        f"'{batch_summary_path_str}'"
                    ),
                )
            )
        batch_summary_exists_at_load = p_summary.is_file()

    # 8. Row Validation
    parsed_rows: list[GuidedCompletedAppliedDffRow] = []
    if not isinstance(raw_rows, list):
        issues.append(
            GuidedCompletedAppliedDffIssue(
                category="applied_dff_rows_malformed",
                severity="blocking",
                message="Provenance rows must be a JSON list.",
                path=str(provenance_path),
            )
        )
    else:
        for idx, row_data in enumerate(raw_rows):
            if not isinstance(row_data, dict):
                issues.append(
                    GuidedCompletedAppliedDffIssue(
                        category="applied_dff_row_malformed",
                        severity="blocking",
                        message=f"Row index {idx} is not an object",
                    )
                )
                continue

            roi_id = row_data.get("roi_id", "")
            batch_roi = row_data.get("batch_roi", "")
            strategy_family = row_data.get("strategy_family", "")
            selected_strategy = row_data.get("selected_strategy", "")
            dynamic_fit_mode = row_data.get("dynamic_fit_mode")
            batch_strategy = row_data.get("batch_strategy", "")
            output_name = row_data.get("output_name", "")
            output_dir_str = row_data.get("output_dir", "")
            status = row_data.get("status", "")
            pipeline_summary_path = row_data.get("pipeline_summary_path")
            row_err = row_data.get("error")

            required_row_fields = (
                "roi_id",
                "batch_roi",
                "strategy_family",
                "selected_strategy",
                "batch_strategy",
                "output_name",
                "output_dir",
                "status",
            )
            for field_name in required_row_fields:
                if not str(row_data.get(field_name) or "").strip():
                    issues.append(
                        GuidedCompletedAppliedDffIssue(
                            category="applied_dff_row_missing_required_field",
                            severity="blocking",
                            message=(
                                f"Row index {idx} is missing required nonblank "
                                f"field '{field_name}'."
                            ),
                        )
                    )

            if not roi_id:
                issues.append(
                    GuidedCompletedAppliedDffIssue(
                        category="applied_dff_row_missing_roi_id",
                        severity="blocking",
                        message=f"Row index {idx} is missing 'roi_id'",
                    )
                )

            # Strategy checks
            if strategy_family not in ("dynamic_fit", "signal_only_f0"):
                issues.append(
                    GuidedCompletedAppliedDffIssue(
                        category="applied_dff_row_strategy_invalid",
                        severity="blocking",
                        message=f"Row index {idx} has invalid strategy_family: '{strategy_family}'",
                    )
                )
            if batch_strategy not in ("dynamic_fit", "signal_only_f0"):
                issues.append(
                    GuidedCompletedAppliedDffIssue(
                        category="applied_dff_row_strategy_invalid",
                        severity="blocking",
                        message=f"Row index {idx} has invalid batch_strategy: '{batch_strategy}'",
                    )
                )
            if strategy_family and batch_strategy and strategy_family != batch_strategy:
                issues.append(
                    GuidedCompletedAppliedDffIssue(
                        category="applied_dff_row_strategy_invalid",
                        severity="blocking",
                        message=f"Row index {idx} strategy_family '{strategy_family}' != batch_strategy '{batch_strategy}'",
                    )
                )

            if status not in {"pending", "succeeded", "failed", "skipped"}:
                issues.append(
                    GuidedCompletedAppliedDffIssue(
                        category="applied_dff_row_status_invalid",
                        severity="blocking",
                        message=f"Row index {idx} has invalid status '{status}'.",
                    )
                )

            # Row status consistency check
            if overall_status == "succeeded" and status != "succeeded":
                issues.append(
                    GuidedCompletedAppliedDffIssue(
                        category="applied_dff_row_status_inconsistent",
                        severity="blocking",
                        message=f"Row for ROI '{roi_id}' has status '{status}' in a succeeded run.",
                    )
                )

            # Row output dir existence and escaping
            output_dir_exists = False
            if output_dir_str:
                p_row_out = Path(output_dir_str)
                if not p_row_out.is_absolute():
                    p_row_out = (
                        (applied_output_root_resolved or applied_dff_dir)
                        / p_row_out
                    )
                p_row_out = p_row_out.resolve()

                if applied_output_root_resolved:
                    if not _is_subpath(p_row_out, applied_output_root_resolved):
                        issues.append(
                            GuidedCompletedAppliedDffIssue(
                                category="applied_dff_row_output_dir_escapes_root",
                                severity="blocking",
                                message=f"Row output_dir escapes root: '{output_dir_str}'",
                            )
                        )
                output_dir_exists = p_row_out.is_dir()

            # Pipeline summary existence check (Optional / Warning only)
            pipeline_summary_exists = False
            if pipeline_summary_path:
                p_row_summary = Path(pipeline_summary_path)
                if not p_row_summary.is_absolute():
                    p_row_summary = (
                        (applied_output_root_resolved or applied_dff_dir)
                        / p_row_summary
                    )
                p_row_summary = p_row_summary.resolve()

                if applied_output_root_resolved:
                    if not _is_subpath(p_row_summary, applied_output_root_resolved):
                        issues.append(
                            GuidedCompletedAppliedDffIssue(
                                category="applied_dff_row_output_dir_escapes_root",
                                severity="blocking",
                                message=f"Row pipeline_summary_path escapes root: '{pipeline_summary_path}'",
                            )
                        )
                pipeline_summary_exists = p_row_summary.is_file()
                if not pipeline_summary_exists:
                    issues.append(
                        GuidedCompletedAppliedDffIssue(
                            category="applied_dff_row_summary_missing",
                            severity="warning",
                            message=f"Pipeline summary missing for ROI '{roi_id}': '{pipeline_summary_path}'",
                        )
                    )

            parsed_rows.append(
                GuidedCompletedAppliedDffRow(
                    roi_id=roi_id,
                    batch_roi=batch_roi,
                    strategy_family=strategy_family,
                    selected_strategy=selected_strategy,
                    dynamic_fit_mode=dynamic_fit_mode,
                    batch_strategy=batch_strategy,
                    output_name=output_name,
                    output_dir=output_dir_str,
                    status=status,
                    pipeline_summary_path=pipeline_summary_path,
                    output_dir_exists=output_dir_exists,
                    pipeline_summary_exists=pipeline_summary_exists,
                    error=row_err,
                )
            )

    # 9. Cross-check requested_strategy_map
    if isinstance(requested_strategy_map, dict):
        map_version = requested_strategy_map.get("production_strategy_map_version")
        if map_version != "per_roi_correction_strategy_map.v1":
            issues.append(
                GuidedCompletedAppliedDffIssue(
                    category="applied_dff_provenance_invalid_map_version",
                    severity="blocking",
                    message=f"Invalid map version: '{map_version}' (must be 'per_roi_correction_strategy_map.v1')",
                    path=str(provenance_path),
                )
            )

        if requested_strategy_map.get("applied_dff_orchestration_enabled") is not True:
            issues.append(
                GuidedCompletedAppliedDffIssue(
                    category="applied_dff_provenance_orchestration_not_enabled",
                    severity="blocking",
                    message="requested_strategy_map.applied_dff_orchestration_enabled must be True",
                    path=str(provenance_path),
                )
            )

        # ROI alignment
        req_rois = set(requested_strategy_map.get("included_roi_ids", []))
        row_rois = {row.roi_id for row in parsed_rows}
        if req_rois != row_rois:
            issues.append(
                GuidedCompletedAppliedDffIssue(
                    category="applied_dff_provenance_roi_mismatch",
                    severity="blocking",
                    message=f"included_roi_ids {req_rois} do not match parsed rows {row_rois}",
                    path=str(provenance_path),
                )
            )

        # Strategy validations
        choices = requested_strategy_map.get("per_roi_production_strategy_map", [])
        if isinstance(choices, list):
            has_dynamic_fit = False
            dynamic_fit_modes = set()
            for idx, choice in enumerate(choices):
                choice_roi = choice.get("roi_id")
                choice_family = choice.get("strategy_family")
                choice_df_mode = choice.get("dynamic_fit_mode")
                choice_selected = choice.get("selected_strategy")

                if choice.get("explicit_user_mark") is not True:
                    issues.append(
                        GuidedCompletedAppliedDffIssue(
                            category="applied_dff_provenance_non_explicit_strategy",
                            severity="blocking",
                            message=f"Strategy for ROI '{choice_roi}' must be explicitly marked.",
                        )
                    )
                if choice.get("current_or_stale") != "current":
                    issues.append(
                        GuidedCompletedAppliedDffIssue(
                            category="applied_dff_provenance_stale_strategy",
                            severity="blocking",
                            message=f"Strategy for ROI '{choice_roi}' is stale.",
                        )
                    )

                if choice_family == "dynamic_fit":
                    has_dynamic_fit = True
                    if choice_df_mode:
                        dynamic_fit_modes.add(choice_df_mode)
                    if choice_selected != choice_df_mode:
                        issues.append(
                            GuidedCompletedAppliedDffIssue(
                                category="applied_dff_provenance_strategy_mismatch",
                                severity="blocking",
                                message=f"selected_strategy '{choice_selected}' != dynamic_fit_mode '{choice_df_mode}' for ROI '{choice_roi}'",
                            )
                        )

            if not has_dynamic_fit:
                issues.append(
                    GuidedCompletedAppliedDffIssue(
                        category="applied_dff_provenance_no_dynamic_fit",
                        severity="blocking",
                        message="At least one dynamic_fit row is required. All-signal_only_f0 maps are unsupported.",
                    )
                )

            if len(dynamic_fit_modes) > 1:
                issues.append(
                    GuidedCompletedAppliedDffIssue(
                        category="applied_dff_provenance_mixed_dynamic_fit",
                        severity="blocking",
                        message=f"Mixed dynamic_fit modes are unsupported: {dynamic_fit_modes}",
                    )
                )

        if applied_output_root_resolved is not None:
            try:
                build_guided_applied_dff_manifest_rows(
                    requested_strategy_map,
                    applied_output_root_resolved,
                )
            except GuidedAppliedDffOrchestrationError as exc:
                issues.append(
                    GuidedCompletedAppliedDffIssue(
                        category="applied_dff_requested_strategy_map_invalid",
                        severity="blocking",
                        message=(
                            "Requested strategy map cannot reproduce a valid "
                            f"production batch manifest: {exc}"
                        ),
                        path=str(provenance_path),
                    )
                )
    elif "requested_strategy_map" in prov_data:
        issues.append(
            GuidedCompletedAppliedDffIssue(
                category="applied_dff_requested_strategy_map_invalid",
                severity="blocking",
                message="requested_strategy_map must be a JSON object.",
                path=str(provenance_path),
            )
        )

    return GuidedCompletedAppliedDffState(
        present=True,
        provenance_path=str(provenance_path),
        schema_version=schema_version,
        overall_status=overall_status,
        production_analysis=production_analysis,
        preview_only=preview_only,
        orchestration_capability_enabled=orchestration_capability_enabled,
        production_strategy_map_version=production_strategy_map_version,
        phasic_cache_path=phasic_cache_path_str,
        phasic_cache_exists_at_load=phasic_cache_exists_at_load,
        batch_manifest_path=batch_manifest_path_str,
        batch_manifest_exists_at_load=batch_manifest_exists_at_load,
        applied_output_root=applied_output_root_str,
        applied_output_root_exists_at_load=applied_output_root_exists_at_load,
        batch_summary_path=batch_summary_path_str,
        batch_summary_exists_at_load=batch_summary_exists_at_load,
        batch_returncode=batch_returncode,
        failure_policy=failure_policy,
        requested_strategy_map=requested_strategy_map,
        rows=tuple(parsed_rows),
        issues=tuple(issues),
    )
