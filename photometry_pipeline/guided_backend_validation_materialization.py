"""Read-only materialized facts layer for Guided Mode validation.

This module performs all read-only I/O (files presence, size checks, and digests)
to gather filesystem facts into immutable dataclasses before pure compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Callable

# Importing needed elements from request contract and draft plan
from photometry_pipeline.guided_backend_validation_request import (
    GuidedBackendSourceCandidateFile,
    GuidedBackendSourceSnapshotFacts,
    GuidedBackendIncompleteFinalClassificationFacts,
    GuidedBackendValidationMaterializedFacts,
    GuidedBackendParserFacts,
    GuidedBackendEvidenceReferenceFacts,
    GuidedBackendDiagnosticCacheFacts,
    GuidedBackendOutputFacts,
    GuidedBackendTypedFieldValue,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    FEATURE_EVENT_CONFIG_FIELDS,
    build_guided_feature_event_effective_values_preview,
)

# Importing RWD helpers
from photometry_pipeline.io.rwd_source_snapshot import (
    build_rwd_source_candidate_snapshot,
    compute_rwd_source_candidate_set_digest,
    compute_rwd_source_candidate_content_digest,
    make_not_requested_incomplete_final_chunk_classification,
    compute_incomplete_final_chunk_classification_digest,
    RwdSourceSnapshotError,
)
from photometry_pipeline.io.rwd_contract import (
    RwdHeaderParsingContract,
    compute_rwd_header_parsing_contract_digest,
)

# Constants for Stage 2b
GUIDED_BACKEND_VALIDATION_MATERIALIZATION_SCOPE = (
    "guided_rwd_intermittent_phasic_full_materialization"
)
GUIDED_BACKEND_VALIDATION_MATERIALIZER_VERSION = (
    "guided_backend_validation_materializer.v1"
)
GUIDED_BACKEND_VALIDATION_MATERIALIZATION_STAGE = (
    "stage_2b_parser_and_feature_event"
)

STAGE_2B_VALID_ISSUES = {
    "missing_source",
    "unsupported_source_format",
    "source_snapshot_unavailable",
    "source_snapshot_cancelled",
    "source_snapshot_unstable",
    "source_snapshot_digest_mismatch",
    "unsupported_incomplete_final_exclusion",
    "incomplete_final_classification_mismatch",
    "materialization_cancelled",
    "materializer_internal_error",
    "unsupported_stage_2a_state",
    # Stage 2b categories
    "parser_contract_missing",
    "parser_digest_unavailable",
    "parser_unresolved_inputs",
    "feature_event_profile_missing",
    "feature_event_profile_stale",
    "feature_event_unapplied_changes",
    "unresolved_feature_event_effective_value",
    "feature_event_effective_value_helper_unsafe",
}


@dataclass(frozen=True)
class GuidedBackendValidationMaterializationIssue:
    category: str
    section: str
    message: str
    detail_code: str | None = None
    debug_context: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.category, str) or not self.category:
            raise ValueError("category must be a non-empty string.")
        if self.category not in STAGE_2B_VALID_ISSUES:
            raise ValueError(f"Unsupported materialization issue category: {self.category}")
        if not isinstance(self.section, str) or not self.section:
            raise ValueError("section must be a non-empty string.")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string.")
        if not isinstance(self.debug_context, tuple):
            raise ValueError("debug_context must be a tuple.")


@dataclass(frozen=True)
class GuidedBackendValidationMaterializationSuccess:
    facts: GuidedBackendValidationMaterializedFacts
    materialization_scope: str = GUIDED_BACKEND_VALIDATION_MATERIALIZATION_SCOPE
    materializer_version: str = GUIDED_BACKEND_VALIDATION_MATERIALIZER_VERSION
    warning_categories: tuple[str, ...] = ()
    status: str = field(default="materialized", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.facts, GuidedBackendValidationMaterializedFacts):
            raise TypeError("facts must be an instance of GuidedBackendValidationMaterializedFacts.")


@dataclass(frozen=True)
class GuidedBackendValidationMaterializationFailure:
    blocking_issues: tuple[GuidedBackendValidationMaterializationIssue, ...]
    warning_categories: tuple[str, ...] = ()
    status: str = field(default="refused", init=False)
    no_usable_facts: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.blocking_issues, tuple):
            raise ValueError("blocking_issues must be a tuple.")
        if not self.blocking_issues:
            raise ValueError("At least one blocking issue is required for failure.")


def _failure(
    category: str,
    section: str,
    message: str,
    *,
    detail_code: str | None = None,
) -> GuidedBackendValidationMaterializationFailure:
    return GuidedBackendValidationMaterializationFailure(
        blocking_issues=(
            GuidedBackendValidationMaterializationIssue(
                category=category,
                section=section,
                message=message,
                detail_code=detail_code,
            ),
        )
    )


def materialize_guided_backend_validation_facts(
    draft: GuidedNewAnalysisDraftPlan,
    *,
    parser_contract: RwdHeaderParsingContract | None = None,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedBackendValidationMaterializationSuccess | GuidedBackendValidationMaterializationFailure:
    """Materialize the Stage 2b subset of validation facts (read-only).

    Currently compiles:
      - Source snapshot facts
      - Incomplete-final not_requested classification facts
      - Parser contract facts
      - Feature/event effective values

    Remaining fact groups are marked unavailable, and compiler handoff will block.
    """
    if not isinstance(draft, GuidedNewAnalysisDraftPlan):
        return _failure(
            "unsupported_stage_2a_state",
            "draft",
            "A GuidedNewAnalysisDraftPlan is required.",
            detail_code="invalid_draft_type",
        )

    # 1. Cancellation check before starting
    if cancellation_check and cancellation_check():
        return _failure(
            "materialization_cancelled",
            "cancellation",
            "Materialization was cancelled before starting.",
            detail_code="cancelled_preflight",
        )

    # 2. Input Format and Acquisition Mode Audit
    if draft.input_format not in ("rwd", "auto"):
        return _failure(
            "unsupported_source_format",
            "source",
            f"Input format '{draft.input_format}' is not supported by Guided Mode.",
            detail_code="format_unsupported",
        )

    if draft.acquisition_mode != "intermittent":
        return _failure(
            "unsupported_stage_2a_state",
            "source",
            f"Acquisition mode '{draft.acquisition_mode}' is not supported by the first subset.",
            detail_code="acquisition_mode_unsupported",
        )

    # 3. Path Audit
    source_path = draft.resolved_input_source_path or draft.input_source_path
    if not source_path:
        return _failure(
            "missing_source",
            "source",
            "No source path specified in the draft plan.",
            detail_code="source_path_missing",
        )

    # 4. Parser Contract Presence Check (Before Snapshot)
    if parser_contract is None:
        return _failure(
            "parser_contract_missing",
            "parser",
            "Parser contract is required for materialization.",
            detail_code="parser_contract_missing",
        )

    if parser_contract.unresolved_inputs:
        return _failure(
            "parser_unresolved_inputs",
            "parser",
            "Parser contract has unresolved inputs.",
            detail_code="unresolved_inputs",
        )

    try:
        parser_contract_digest = compute_rwd_header_parsing_contract_digest(parser_contract)
    except Exception as exc:
        return _failure(
            "parser_digest_unavailable",
            "parser",
            f"Failed to compute parser contract digest: {exc}",
            detail_code="digest_error",
        )

    parser_facts = GuidedBackendParserFacts(
        available=True,
        parser_contract_digest=parser_contract_digest,
        unresolved_inputs=(),
    )

    # 5. Incomplete-Final Policy Audit (Before Snapshot)
    if draft.exclude_incomplete_final_rwd_chunk:
        return _failure(
            "unsupported_incomplete_final_exclusion",
            "incomplete_final",
            "Excluding incomplete final chunk is not supported by the first subset.",
            detail_code="exclusion_enabled_unsupported",
        )

    # 6. Source Snapshot Materialization (I/O Read-Only)
    try:
        snapshot = build_rwd_source_candidate_snapshot(
            source_path,
            cancellation_check=cancellation_check,
        )
    except RwdSourceSnapshotError as exc:
        category = "source_snapshot_unavailable"
        if exc.category in ("source_root_missing", "source_root_not_directory"):
            category = "missing_source"
        elif exc.category == "unstable_filesystem_facts":
            category = "source_snapshot_unstable"
        elif exc.category in ("snapshot_cancelled", "source_candidate_snapshot_cancelled"):
            category = "source_snapshot_cancelled"

        return _failure(
            category,
            "source",
            f"Failed to build source candidate snapshot: {exc.message}",
            detail_code=exc.category,
        )
    except Exception as exc:
        return _failure(
            "materializer_internal_error",
            "source",
            f"Internal error during snapshot build: {exc}",
            detail_code="internal_snapshot_error",
        )

    # 7. Cancellation check after potentially expensive snapshot build
    if cancellation_check and cancellation_check():
        return _failure(
            "materialization_cancelled",
            "cancellation",
            "Materialization was cancelled after snapshot generation.",
            detail_code="cancelled_post_snapshot",
        )

    # Map candidate files into request contract type
    backend_candidate_files = []
    for f in snapshot.candidates:
        backend_candidate_files.append(
            GuidedBackendSourceCandidateFile(
                canonical_relative_path=f.canonical_relative_path,
                size_bytes=f.size_bytes,
                sha256_content_digest=f.sha256_content_digest,
            )
        )

    source_snapshot_facts = GuidedBackendSourceSnapshotFacts(
        available=True,
        source_root_canonical=snapshot.source_root_canonical,
        source_candidate_set_digest=snapshot.source_candidate_set_digest,
        source_candidate_content_digest=snapshot.source_candidate_content_digest,
        candidate_files=tuple(backend_candidate_files),
        stale=False,
    )

    # 8. Incomplete-Final not_requested Classification Materialization

    try:
        classification = make_not_requested_incomplete_final_chunk_classification(snapshot)
        classification_digest = compute_incomplete_final_chunk_classification_digest(classification)
    except Exception as exc:
        return _failure(
            "materializer_internal_error",
            "incomplete_final",
            f"Failed to generate incomplete-final classification: {exc}",
            detail_code="internal_classification_error",
        )

    classification_facts = GuidedBackendIncompleteFinalClassificationFacts(
        available=True,
        classification_status="not_requested",
        classification_digest=classification_digest,
        source_candidate_set_digest=snapshot.source_candidate_set_digest,
        source_candidate_content_digest=snapshot.source_candidate_content_digest,
    )

    # 9. Cancellation check before feature/event work
    if cancellation_check and cancellation_check():
        return _failure(
            "materialization_cancelled",
            "cancellation",
            "Materialization was cancelled before feature/event compilation.",
            detail_code="cancelled_pre_feature",
        )

    # 10. Feature/Event Effective-Value Materialization
    # Check profile status on the draft
    if draft.feature_event_profile_status in ("missing", "unavailable"):
        return _failure(
            "feature_event_profile_missing",
            "feature_event",
            "Feature/event profile is missing from the draft plan.",
            detail_code="profile_missing",
        )
    elif draft.feature_event_profile_status == "stale":
        return _failure(
            "feature_event_profile_stale",
            "feature_event",
            "Feature/event profile is stale.",
            detail_code="profile_stale",
        )
    elif (
        not draft.feature_event_explicitly_applied
        or draft.feature_event_profile_status != "applied"
        or draft.feature_event_stale_reasons
    ):
        return _failure(
            "feature_event_unapplied_changes",
            "feature_event",
            "Feature/event profile has unapplied changes.",
            detail_code="unapplied_changes",
        )

    # Use build_guided_feature_event_effective_values_preview (pure, no Config instantiation)
    try:
        preview = build_guided_feature_event_effective_values_preview(draft)
    except Exception as exc:
        return _failure(
            "materializer_internal_error",
            "feature_event",
            f"Feature/event effective values preview builder failed: {exc}",
            detail_code="helper_failed",
        )

    # Validate that we do not silently use backend_config_default for active fields.
    # If any active field has source "backend_config_default" or is unresolved, block.
    effective_values_list = preview.get("effective_values", [])
    backend_typed_values = []

    source_map = {
        "applied_guided_profile": "explicit",
        "backend_config_default": "backend_default",
        "unresolved": "unresolved",
    }

    for item in effective_values_list:
        field_name = item["field_name"]
        effective_value = item["effective_value"]
        source = item["source"]
        activity = item["active_or_inactive"]

        if activity == "active":
            if source != "applied_guided_profile" or effective_value is None:
                return _failure(
                    "unresolved_feature_event_effective_value",
                    "feature_event",
                    f"Required feature/event value for '{field_name}' is unresolved or relies on a silent default.",
                    detail_code="unresolved_value",
                )

        mapped_source = source_map.get(source, "unresolved")

        backend_typed_values.append(
            GuidedBackendTypedFieldValue(
                field_name=field_name,
                value_type=type(effective_value).__name__ if effective_value is not None else "NoneType",
                value=effective_value,
                source_classification=mapped_source,
            )
        )

    # 11. Cancellation check after feature/event work
    if cancellation_check and cancellation_check():
        return _failure(
            "materialization_cancelled",
            "cancellation",
            "Materialization was cancelled after feature/event compilation.",
            detail_code="cancelled_post_feature",
        )

    # 12. Unresolved required inputs lists the other fact groups for Stage 2b
    unresolved = (
        "diagnostic_cache_facts",
        "output_facts",
        "evidence_references",
    )

    facts = GuidedBackendValidationMaterializedFacts(
        source_snapshot=source_snapshot_facts,
        incomplete_final_classification=classification_facts,
        parser=parser_facts,
        diagnostic_cache=GuidedBackendDiagnosticCacheFacts(available=False),
        output=GuidedBackendOutputFacts(available=False),
        evidence_references=GuidedBackendEvidenceReferenceFacts(complete=False),
        effective_feature_event_values=tuple(backend_typed_values),
        complete_for_compilation=False,
        unresolved_required_inputs=unresolved,
    )

    return GuidedBackendValidationMaterializationSuccess(facts=facts)
