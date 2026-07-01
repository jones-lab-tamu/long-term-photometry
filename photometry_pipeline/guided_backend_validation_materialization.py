"""Read-only materialized facts layer for Guided Mode validation.

This module performs all read-only I/O (files presence, size checks, and digests)
to gather filesystem facts into immutable dataclasses before pure compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
import hashlib
import json
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
    GuidedBackendAcquisitionDatasetFacts,
    GuidedBackendRoiScopeFacts,
    GuidedBackendConfirmedStrategyMarkFacts,
    GuidedBackendCorrectionFacts,
    GuidedBackendFeatureEventFacts,
    GuidedBackendEvidenceReferenceFacts,
    GuidedBackendEvidenceReference,
    GuidedBackendDiagnosticCacheFacts,
    GuidedBackendOutputFacts,
    GuidedBackendOutputRelationship,
    GuidedBackendTypedFieldValue,
    GUIDED_BACKEND_FEATURE_EVENT_PROFILE_SCHEMA_VERSION,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    FEATURE_EVENT_CONFIG_FIELDS,
    FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES,
    FORBIDDEN_CORRECTION_STRATEGIES,
    build_guided_feature_event_effective_values_preview,
    classify_output_base_safety_ownership,
)
from photometry_pipeline.guided_completed_run_rejection_policy import (
    AMBIGUOUS_GUIDED_DIAGNOSTIC_CACHE_METADATA,
    GUIDED_DIAGNOSTIC_CACHE_INELIGIBLE,
    MALFORMED_GUIDED_DIAGNOSTIC_CACHE_METADATA,
    detect_guided_diagnostic_cache_candidate,
)
from photometry_pipeline.guided_diagnostic_cache import (
    DIAGNOSTIC_CACHE_ARTIFACT_FILENAME,
    DIAGNOSTIC_CACHE_PROVENANCE_FILENAME,
    DIAGNOSTIC_CACHE_PURPOSE,
    DIAGNOSTIC_CACHE_SCHEMA_VERSION,
    resolve_diagnostic_cache_source,
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
    "stage_2d_output_facts"
)
GUIDED_BACKEND_OUTPUT_SAFETY_CLASSIFIER_VERSION = (
    "guided_output_base_safety_ownership.v1"
)

STAGE_2C_VALID_ISSUES = {
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
    # Stage 2c categories
    "diagnostic_cache_missing",
    "diagnostic_cache_resolver_failed",
    "diagnostic_cache_metadata_missing",
    "diagnostic_cache_metadata_malformed",
    "diagnostic_cache_metadata_ambiguous",
    "diagnostic_cache_not_completed_run_ineligible",
    "diagnostic_cache_completed_run_accepted",
    "diagnostic_cache_purpose_mismatch",
    "diagnostic_cache_schema_unsupported",
    "diagnostic_cache_not_preliminary",
    "diagnostic_cache_marked_production",
    "diagnostic_cache_source_setup_mismatch",
    "diagnostic_cache_scope_mismatch",
    "diagnostic_cache_build_request_mismatch",
    "diagnostic_cache_artifact_digest_unavailable",
    "diagnostic_cache_provenance_digest_unavailable",
    "evidence_reference_missing_or_stale",
    "evidence_reference_cache_mismatch",
    "evidence_reference_roi_mismatch",
    "evidence_reference_strategy_mismatch",
    "missing_confirmed_strategy_mark",
    "duplicate_confirmed_strategy_mark",
    "stale_strategy_mark",
    "non_explicit_strategy_mark",
    "mixed_dynamic_fit_modes",
    "signal_only_not_supported_for_validate",
    "forbidden_strategy_state",
    # Stage 2d categories
    "output_policy_missing",
    "output_policy_stale",
    "output_policy_unapplied_changes",
    "output_base_missing",
    "output_base_invalid",
    "output_protected_root_context_incomplete",
    "output_overlaps_source",
    "output_overlaps_diagnostic_cache",
    "output_overlaps_completed_run",
    "output_overlaps_protected_root",
    "output_overwrite_not_allowed",
    "output_precreate_not_allowed",
    "output_safety_facts_unavailable",
    "output_materialization_requires_write",
    # Request-ready fact completion categories
    "dataset_facts_missing",
    "dataset_facts_stale",
    "dataset_semantic_value_unresolved",
    "dataset_source_signature_mismatch",
    "parser_contract_fields_unavailable",
    "roi_scope_missing",
    "roi_scope_invalid",
    "correction_facts_missing",
    "dynamic_fit_parameter_unresolved",
    "dynamic_fit_parameter_contract_mismatch",
    "feature_event_profile_identity_unavailable",
    "feature_event_activity_unavailable",
    "source_path_style_unavailable",
}

# Backward-compatible name retained for callers/tests from Stage 2b.
STAGE_2B_VALID_ISSUES = STAGE_2C_VALID_ISSUES


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
        if self.category not in STAGE_2C_VALID_ISSUES:
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


def _read_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "metadata root is not a JSON object"
    return payload, ""


def _semantic_json_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _materialize_diagnostic_cache(
    draft: GuidedNewAnalysisDraftPlan,
) -> tuple[
    GuidedBackendDiagnosticCacheFacts | None,
    dict[str, Any] | None,
    GuidedBackendValidationMaterializationFailure | None,
]:
    cache_pointer = draft.artifact_record_path or draft.cache_root_path
    if not cache_pointer:
        return None, None, _failure(
            "diagnostic_cache_missing",
            "diagnostic_cache",
            "The draft plan does not identify a diagnostic cache.",
            detail_code="cache_pointer_missing",
        )

    pointer = Path(cache_pointer)
    root = pointer.parent if pointer.name == DIAGNOSTIC_CACHE_ARTIFACT_FILENAME else pointer
    cache_root_canonical = os.path.realpath(os.fspath(root))

    rejection = detect_guided_diagnostic_cache_candidate(cache_root_canonical)
    if rejection is None:
        return None, None, _failure(
            "diagnostic_cache_not_completed_run_ineligible",
            "diagnostic_cache",
            "The selected folder is not identified as an ineligible Guided diagnostic cache.",
            detail_code="completed_run_rejection_missing",
        )
    if rejection.category == MALFORMED_GUIDED_DIAGNOSTIC_CACHE_METADATA:
        if "unsupported" in rejection.detail:
            category = "diagnostic_cache_schema_unsupported"
        elif "purpose" in rejection.detail:
            category = "diagnostic_cache_purpose_mismatch"
        else:
            category = "diagnostic_cache_metadata_malformed"
        return None, None, _failure(
            category,
            "diagnostic_cache",
            rejection.message,
            detail_code=rejection.category,
        )
    if rejection.category == AMBIGUOUS_GUIDED_DIAGNOSTIC_CACHE_METADATA:
        boundary_artifact, _ = _read_json_object(
            Path(cache_root_canonical) / DIAGNOSTIC_CACHE_ARTIFACT_FILENAME
        )
        boundary_provenance, _ = _read_json_object(
            Path(cache_root_canonical) / DIAGNOSTIC_CACHE_PROVENANCE_FILENAME
        )
        boundary_summary = (
            boundary_artifact.get("session_chunk_inventory_summary")
            if boundary_artifact
            else None
        )
        if (
            (boundary_artifact and boundary_artifact.get("production_analysis") is True)
            or (boundary_provenance and boundary_provenance.get("production_analysis") is True)
            or (
                isinstance(boundary_summary, dict)
                and boundary_summary.get("production_analysis") is True
            )
        ):
            category = "diagnostic_cache_marked_production"
        elif (
            (boundary_provenance and boundary_provenance.get("preliminary_cache") is False)
            or (
                isinstance(boundary_summary, dict)
                and boundary_summary.get("preliminary_cache") is False
            )
        ):
            category = "diagnostic_cache_not_preliminary"
        else:
            category = ""
        detail_categories = (
            ("source_setup_signature", "diagnostic_cache_source_setup_mismatch"),
            ("diagnostic_scope_signature", "diagnostic_cache_scope_mismatch"),
            ("build_request_signature", "diagnostic_cache_build_request_mismatch"),
            ("cache_id", "evidence_reference_cache_mismatch"),
            ("preliminary", "diagnostic_cache_not_preliminary"),
            ("production", "diagnostic_cache_marked_production"),
        )
        if not category:
            category = next(
                (
                    mapped
                    for marker, mapped in detail_categories
                    if marker in rejection.detail
                ),
                "diagnostic_cache_metadata_ambiguous",
            )
        return None, None, _failure(
            category,
            "diagnostic_cache",
            rejection.message,
            detail_code=rejection.category,
        )
    if rejection.category != GUIDED_DIAGNOSTIC_CACHE_INELIGIBLE:
        return None, None, _failure(
            "diagnostic_cache_not_completed_run_ineligible",
            "diagnostic_cache",
            "The selected cache does not carry the required completed-run rejection category.",
            detail_code=rejection.category,
        )

    resolved = resolve_diagnostic_cache_source(cache_pointer)
    if not resolved.ok or resolved.source is None:
        return None, None, _failure(
            "diagnostic_cache_resolver_failed",
            "diagnostic_cache",
            resolved.status.message,
            detail_code=resolved.status.code,
        )
    source = resolved.source
    resolved_root = os.path.realpath(source.cache_root_path)
    if resolved_root != cache_root_canonical:
        return None, None, _failure(
            "diagnostic_cache_resolver_failed",
            "diagnostic_cache",
            "Resolved diagnostic-cache root disagrees with the selected cache root.",
            detail_code="cache_root_mismatch",
        )
    if draft.stale_or_current != "current" or draft.stale_reasons:
        return None, None, _failure(
            "diagnostic_cache_resolver_failed",
            "diagnostic_cache",
            "The draft diagnostic-cache reference is not current.",
            detail_code="cache_not_current",
        )

    artifact_path = Path(source.artifact_record_path)
    provenance_path = Path(source.provenance_path)
    if not artifact_path.is_file() or not provenance_path.is_file():
        return None, None, _failure(
            "diagnostic_cache_metadata_missing",
            "diagnostic_cache",
            "Diagnostic-cache artifact and provenance metadata are required.",
            detail_code="identity_file_missing",
        )
    artifact, artifact_error = _read_json_object(artifact_path)
    if artifact_error or artifact is None:
        return None, None, _failure(
            "diagnostic_cache_metadata_malformed",
            "diagnostic_cache",
            f"Diagnostic-cache artifact metadata is invalid: {artifact_error}",
            detail_code="artifact_invalid",
        )
    provenance, provenance_error = _read_json_object(provenance_path)
    if provenance_error or provenance is None:
        return None, None, _failure(
            "diagnostic_cache_metadata_malformed",
            "diagnostic_cache",
            f"Diagnostic-cache provenance metadata is invalid: {provenance_error}",
            detail_code="provenance_invalid",
        )

    if (
        artifact.get("artifact_contract_version") != DIAGNOSTIC_CACHE_SCHEMA_VERSION
        or provenance.get("schema_version") != DIAGNOSTIC_CACHE_SCHEMA_VERSION
    ):
        return None, None, _failure(
            "diagnostic_cache_schema_unsupported",
            "diagnostic_cache",
            "Diagnostic-cache artifact or provenance schema is unsupported.",
            detail_code="schema_unsupported",
        )
    if (
        artifact.get("purpose") != DIAGNOSTIC_CACHE_PURPOSE
        or provenance.get("purpose") != DIAGNOSTIC_CACHE_PURPOSE
    ):
        return None, None, _failure(
            "diagnostic_cache_purpose_mismatch",
            "diagnostic_cache",
            "Diagnostic-cache artifact or provenance purpose is invalid.",
            detail_code="purpose_mismatch",
        )

    summary = artifact.get("session_chunk_inventory_summary")
    if not isinstance(summary, dict):
        return None, None, _failure(
            "diagnostic_cache_metadata_malformed",
            "diagnostic_cache",
            "Diagnostic-cache session/chunk inventory summary is invalid.",
            detail_code="inventory_summary_invalid",
        )
    if summary.get("preliminary_cache") is not True or provenance.get("preliminary_cache") is not True:
        return None, None, _failure(
            "diagnostic_cache_not_preliminary",
            "diagnostic_cache",
            "Diagnostic-cache metadata is not marked preliminary.",
            detail_code="preliminary_cache_false",
        )
    if (
        artifact.get("production_analysis") is not False
        or summary.get("production_analysis") is not False
        or provenance.get("production_analysis") is not False
    ):
        return None, None, _failure(
            "diagnostic_cache_marked_production",
            "diagnostic_cache",
            "Diagnostic-cache metadata is marked as production analysis.",
            detail_code="production_analysis_true",
        )

    identity_fields = (
        ("cache_id", "evidence_reference_cache_mismatch"),
        ("source_setup_signature", "diagnostic_cache_source_setup_mismatch"),
        ("diagnostic_scope_signature", "diagnostic_cache_scope_mismatch"),
        ("build_request_signature", "diagnostic_cache_build_request_mismatch"),
    )
    nested_artifact = provenance.get("artifact")
    if not isinstance(nested_artifact, dict):
        return None, None, _failure(
            "diagnostic_cache_metadata_malformed",
            "diagnostic_cache",
            "Diagnostic-cache provenance does not contain artifact identity.",
            detail_code="nested_artifact_missing",
        )
    for field_name, category in identity_fields:
        artifact_value = artifact.get(field_name)
        if not isinstance(artifact_value, str) or not artifact_value:
            return None, None, _failure(
                "diagnostic_cache_metadata_malformed",
                "diagnostic_cache",
                f"Diagnostic-cache {field_name} is missing.",
                detail_code=f"{field_name}_missing",
            )
        if nested_artifact.get(field_name) != artifact_value:
            return None, None, _failure(
                category,
                "diagnostic_cache",
                f"Artifact and provenance disagree on {field_name}.",
                detail_code=f"{field_name}_metadata_mismatch",
            )

    draft_identity = (
        ("cache_id", draft.cache_id, artifact["cache_id"], "evidence_reference_cache_mismatch"),
        ("source_setup_signature", draft.source_setup_signature, artifact["source_setup_signature"], "diagnostic_cache_source_setup_mismatch"),
        ("diagnostic_scope_signature", draft.diagnostic_scope_signature, artifact["diagnostic_scope_signature"], "diagnostic_cache_scope_mismatch"),
        ("build_request_signature", draft.build_request_signature, artifact["build_request_signature"], "diagnostic_cache_build_request_mismatch"),
    )
    for field_name, draft_value, artifact_value, category in draft_identity:
        if not draft_value or draft_value != artifact_value:
            return None, None, _failure(
                category,
                "diagnostic_cache",
                f"Draft and diagnostic cache disagree on {field_name}.",
                detail_code=f"{field_name}_draft_mismatch",
            )

    try:
        artifact_digest = _semantic_json_digest(artifact)
    except Exception:
        return None, None, _failure(
            "diagnostic_cache_artifact_digest_unavailable",
            "diagnostic_cache",
            "The diagnostic-cache artifact semantic digest could not be computed.",
            detail_code="artifact_digest_failed",
        )
    try:
        provenance_digest = _semantic_json_digest(provenance)
    except Exception:
        return None, None, _failure(
            "diagnostic_cache_provenance_digest_unavailable",
            "diagnostic_cache",
            "The diagnostic-cache provenance semantic digest could not be computed.",
            detail_code="provenance_digest_failed",
        )

    return (
        GuidedBackendDiagnosticCacheFacts(
            available=True,
            cache_id=artifact["cache_id"],
            cache_root_canonical=cache_root_canonical,
            artifact_semantic_digest=artifact_digest,
            provenance_semantic_digest=provenance_digest,
            completed_run_rejection_category=rejection.category,
            resolver_status="current",
            source_setup_signature=artifact["source_setup_signature"],
            diagnostic_scope_signature=artifact["diagnostic_scope_signature"],
            build_request_signature=artifact["build_request_signature"],
            preliminary_cache=True,
            production_analysis=False,
        ),
        artifact,
        None,
    )


def _materialize_evidence_references(
    draft: GuidedNewAnalysisDraftPlan,
    cache_facts: GuidedBackendDiagnosticCacheFacts,
    artifact: dict[str, Any],
) -> tuple[
    GuidedBackendEvidenceReferenceFacts | None,
    GuidedBackendValidationMaterializationFailure | None,
]:
    included = tuple(draft.included_roi_ids)
    if not included:
        return None, _failure(
            "missing_confirmed_strategy_mark",
            "evidence_references",
            "The draft has no included ROIs with confirmed strategy marks.",
            detail_code="included_roi_ids_missing",
        )

    choices_by_roi: dict[str, list[Any]] = {roi: [] for roi in included}
    for choice in draft.per_roi_correction_strategy_choices:
        if choice.roi_id in choices_by_roi:
            choices_by_roi[choice.roi_id].append(choice)

    references: list[GuidedBackendEvidenceReference] = []
    modes: set[str] = set()
    evidence_reference_id = draft.correction_preview_result_id
    if not evidence_reference_id or draft.correction_preview_status != "current":
        return None, _failure(
            "evidence_reference_missing_or_stale",
            "evidence_references",
            "Current correction-preview evidence is required.",
            detail_code="correction_preview_missing_or_stale",
        )
    if draft.correction_preview_source_cache_id != cache_facts.cache_id:
        return None, _failure(
            "evidence_reference_cache_mismatch",
            "evidence_references",
            "Correction-preview evidence does not reference the resolved diagnostic cache.",
            detail_code="preview_cache_mismatch",
        )

    inventory = artifact.get("session_chunk_inventory_summary", {}).get(
        "evidence_references"
    )
    if not isinstance(inventory, list) or not inventory:
        return None, _failure(
            "evidence_reference_missing_or_stale",
            "evidence_references",
            "The diagnostic cache does not contain a usable evidence-reference inventory.",
            detail_code="cache_evidence_inventory_missing",
        )
    inventory_entries = inventory

    for roi_id in included:
        choices = choices_by_roi[roi_id]
        if not choices:
            return None, _failure(
                "missing_confirmed_strategy_mark",
                "evidence_references",
                f"Included ROI '{roi_id}' has no confirmed strategy mark.",
                detail_code="strategy_mark_missing",
            )
        if len(choices) != 1:
            return None, _failure(
                "duplicate_confirmed_strategy_mark",
                "evidence_references",
                f"Included ROI '{roi_id}' has duplicate confirmed strategy marks.",
                detail_code="strategy_mark_duplicate",
            )
        choice = choices[0]
        if choice.current_or_stale != "current":
            return None, _failure(
                "stale_strategy_mark",
                "evidence_references",
                f"Strategy mark for ROI '{roi_id}' is stale.",
                detail_code="strategy_mark_stale",
            )
        if not choice.explicit_user_mark:
            return None, _failure(
                "non_explicit_strategy_mark",
                "evidence_references",
                f"Strategy mark for ROI '{roi_id}' is not explicit.",
                detail_code="strategy_mark_not_explicit",
            )
        if choice.selected_strategy == "signal_only_f0":
            return None, _failure(
                "signal_only_not_supported_for_validate",
                "evidence_references",
                "Signal-Only is not supported by the first validation subset.",
                detail_code="signal_only",
            )
        if (
            choice.selected_strategy in FORBIDDEN_CORRECTION_STRATEGIES
            or choice.selected_strategy not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES
        ):
            return None, _failure(
                "forbidden_strategy_state",
                "evidence_references",
                f"Strategy '{choice.selected_strategy}' is not supported.",
                detail_code="strategy_forbidden",
            )
        modes.add(choice.selected_strategy)

        if choice.diagnostic_cache_id != cache_facts.cache_id:
            return None, _failure(
                "evidence_reference_cache_mismatch",
                "evidence_references",
                f"Strategy mark for ROI '{roi_id}' references a different cache.",
                detail_code="mark_cache_id_mismatch",
            )
        if choice.diagnostic_cache_root and os.path.realpath(choice.diagnostic_cache_root) != cache_facts.cache_root_canonical:
            return None, _failure(
                "evidence_reference_cache_mismatch",
                "evidence_references",
                f"Strategy mark for ROI '{roi_id}' references a different cache root.",
                detail_code="mark_cache_root_mismatch",
            )
        signature_checks = (
            (choice.source_setup_signature, cache_facts.source_setup_signature, "diagnostic_cache_source_setup_mismatch"),
            (choice.diagnostic_scope_signature, cache_facts.diagnostic_scope_signature, "diagnostic_cache_scope_mismatch"),
            (choice.build_request_signature, cache_facts.build_request_signature, "diagnostic_cache_build_request_mismatch"),
        )
        for actual, expected, category in signature_checks:
            if actual != expected:
                return None, _failure(
                    category,
                    "evidence_references",
                    f"Strategy mark for ROI '{roi_id}' does not match cache identity.",
                    detail_code="mark_signature_mismatch",
                )

        reference_matches = [
            entry
            for entry in inventory_entries
            if isinstance(entry, dict)
            and entry.get("evidence_reference_id") == evidence_reference_id
        ]
        if not reference_matches:
            return None, _failure(
                "evidence_reference_missing_or_stale",
                "evidence_references",
                f"Cache evidence reference '{evidence_reference_id}' is missing.",
                detail_code="cache_evidence_reference_missing",
            )
        roi_matches = [
            entry for entry in reference_matches if entry.get("roi_id") == roi_id
        ]
        if not roi_matches:
            return None, _failure(
                "evidence_reference_roi_mismatch",
                "evidence_references",
                f"Cache evidence does not bind reference '{evidence_reference_id}' to ROI '{roi_id}'.",
                detail_code="cache_evidence_roi_mismatch",
            )
        if not any(
            entry.get("diagnostic_cache_id") == cache_facts.cache_id
            for entry in roi_matches
        ):
            return None, _failure(
                "evidence_reference_cache_mismatch",
                "evidence_references",
                "Cache evidence references a different diagnostic cache.",
                detail_code="cache_evidence_cache_mismatch",
            )
        identity_matches = [
            entry
            for entry in roi_matches
            if entry.get("diagnostic_cache_id") == cache_facts.cache_id
        ]
        if not any(
            entry.get("selected_strategy", entry.get("dynamic_fit_mode"))
            == choice.selected_strategy
            for entry in identity_matches
        ):
            return None, _failure(
                "evidence_reference_strategy_mismatch",
                "evidence_references",
                f"Cache evidence strategy does not match ROI '{roi_id}'.",
                detail_code="cache_evidence_strategy_mismatch",
            )

        references.append(
            GuidedBackendEvidenceReference(
                evidence_reference_id=evidence_reference_id,
                evidence_kind="correction_preview",
                diagnostic_cache_id=cache_facts.cache_id,
                source_setup_signature=cache_facts.source_setup_signature,
                diagnostic_scope_signature=cache_facts.diagnostic_scope_signature,
                build_request_signature=cache_facts.build_request_signature,
                evidence_chunk=choice.evidence_chunk,
                roi_id=roi_id,
                selected_dynamic_fit_mode=choice.selected_strategy,
                current=True,
            )
        )

    if len(modes) != 1:
        return None, _failure(
            "mixed_dynamic_fit_modes",
            "evidence_references",
            "Included ROIs must use one supported dynamic-fit mode.",
            detail_code="mixed_modes",
        )
    return GuidedBackendEvidenceReferenceFacts(
        references=tuple(references),
        complete=True,
    ), None


def _materialize_dataset_and_roi_facts(
    draft: GuidedNewAnalysisDraftPlan,
    source_facts: GuidedBackendSourceSnapshotFacts,
    cache_facts: GuidedBackendDiagnosticCacheFacts,
) -> tuple[
    GuidedBackendAcquisitionDatasetFacts | None,
    GuidedBackendRoiScopeFacts | None,
    GuidedBackendValidationMaterializationFailure | None,
]:
    snapshot = draft.dataset_contract_snapshot
    if snapshot.status == "stale" or snapshot.stale_reasons:
        return None, None, _failure(
            "dataset_facts_stale",
            "dataset",
            "The applied dataset contract snapshot is stale.",
            detail_code="dataset_snapshot_stale",
        )
    if (
        not snapshot.current_applied
        or snapshot.input_format != "rwd"
        or snapshot.resolved_input_format != "rwd"
        or snapshot.acquisition_mode != "intermittent"
    ):
        return None, None, _failure(
            "dataset_facts_missing",
            "dataset",
            "A current applied RWD/intermittent dataset contract is required.",
            detail_code="dataset_snapshot_missing_or_invalid",
        )
    if (
        draft.execution_intent.timeline_anchor_mode != "civil"
        or draft.execution_intent.fixed_daily_anchor_clock is not None
        or draft.execution_intent.execution_mode != "phasic"
        or draft.execution_intent.run_profile != "full"
    ):
        return None, None, _failure(
            "dataset_facts_missing",
            "dataset",
            "The execution intent is outside the first request-ready subset.",
            detail_code="execution_intent_unsupported",
        )

    identity = snapshot.source_identity
    required_semantic_names = ("rwd_time_col", "uv_suffix", "sig_suffix")
    semantic_values: list[GuidedBackendTypedFieldValue] = []
    for field_name in sorted(snapshot.contract_values):
        value = snapshot.contract_values[field_name]
        if value is not None and not isinstance(value, (str, bool, int, float)):
            return None, None, _failure(
                "dataset_semantic_value_unresolved",
                "dataset",
                f"Dataset semantic value '{field_name}' is not a canonical scalar.",
                detail_code="dataset_value_not_scalar",
            )
        semantic_values.append(
            GuidedBackendTypedFieldValue(
                field_name=field_name,
                value_type=type(value).__name__,
                value=value,
                source_classification="applied_dataset_contract",
            )
        )
    missing_semantics = [
        name
        for name in required_semantic_names
        if not isinstance(snapshot.contract_values.get(name), str)
        or not str(snapshot.contract_values.get(name) or "").strip()
    ]
    if missing_semantics:
        return None, None, _failure(
            "dataset_semantic_value_unresolved",
            "dataset",
            "Required RWD dataset semantic values are missing.",
            detail_code="required_rwd_semantics_missing",
        )
    if (
        identity.sessions_per_hour is None
        or identity.sessions_per_hour <= 0
        or identity.session_duration_sec is None
        or identity.session_duration_sec <= 0
    ):
        return None, None, _failure(
            "dataset_semantic_value_unresolved",
            "dataset",
            "Session cadence and duration must be resolved.",
            detail_code="session_timing_missing",
        )
    if (
        identity.sessions_per_hour != draft.sessions_per_hour
        or identity.session_duration_sec != draft.session_duration_sec
        or identity.allow_partial_final_window is not False
        or identity.exclude_incomplete_final_rwd_chunk is not False
    ):
        return None, None, _failure(
            "dataset_facts_stale",
            "dataset",
            "Dataset timing or incomplete-final policy is stale.",
            detail_code="dataset_timing_policy_mismatch",
        )
    if (
        identity.source_setup_signature != cache_facts.source_setup_signature
        or identity.diagnostic_cache_contract_identity
        != cache_facts.build_request_signature
    ):
        return None, None, _failure(
            "dataset_source_signature_mismatch",
            "dataset",
            "Dataset source identity does not match the diagnostic cache.",
            detail_code="dataset_cache_identity_mismatch",
        )

    discovered = tuple(draft.discovered_roi_ids)
    included = tuple(draft.included_roi_ids)
    excluded = tuple(draft.excluded_roi_ids)
    if not discovered or not included:
        return None, None, _failure(
            "roi_scope_missing",
            "roi_scope",
            "Discovered and included ROI sets are required.",
            detail_code="roi_inventory_missing",
        )
    if (
        len(discovered) != len(set(discovered))
        or len(included) != len(set(included))
        or len(excluded) != len(set(excluded))
        or set(included) & set(excluded)
        or set(included) | set(excluded) != set(discovered)
    ):
        return None, None, _failure(
            "roi_scope_invalid",
            "roi_scope",
            "Included and excluded ROI sets must uniquely partition discovered ROIs.",
            detail_code="roi_partition_invalid",
        )
    if (
        tuple(identity.discovered_roi_ids) != discovered
        or tuple(identity.included_roi_ids) != included
    ):
        return None, None, _failure(
            "dataset_facts_stale",
            "dataset",
            "Dataset snapshot ROI identity is stale.",
            detail_code="dataset_roi_identity_mismatch",
        )

    dataset_facts = GuidedBackendAcquisitionDatasetFacts(
        available=True,
        acquisition_mode="intermittent",
        sessions_per_hour=identity.sessions_per_hour,
        session_duration_sec=float(identity.session_duration_sec),
        timeline_anchor_mode=draft.execution_intent.timeline_anchor_mode,
        fixed_daily_anchor_clock=draft.execution_intent.fixed_daily_anchor_clock,
        allow_partial_final_window=bool(identity.allow_partial_final_window),
        exclude_incomplete_final_rwd_chunk=bool(
            identity.exclude_incomplete_final_rwd_chunk
        ),
        dataset_snapshot_schema_version=snapshot.schema_version,
        dataset_status=snapshot.status,
        dataset_current_applied=snapshot.current_applied,
        rwd_time_col=str(snapshot.contract_values["rwd_time_col"]),
        uv_suffix=str(snapshot.contract_values["uv_suffix"]),
        sig_suffix=str(snapshot.contract_values["sig_suffix"]),
        semantic_values=tuple(semantic_values),
        dataset_source_setup_signature=str(identity.source_setup_signature),
        diagnostic_cache_contract_identity=str(
            identity.diagnostic_cache_contract_identity
        ),
        validation_issue_categories=tuple(snapshot.validation_issues),
        stale_reason_categories=tuple(snapshot.stale_reasons),
    )
    roi_facts = GuidedBackendRoiScopeFacts(
        available=True,
        discovered_roi_ids=discovered,
        included_roi_ids=included,
        excluded_roi_ids=excluded,
        selection_mode="include",
        inventory_status="plan_inventory_current_for_snapshot",
        inventory_source_content_digest=source_facts.source_candidate_content_digest,
        roi_inventory_identity_status="deferred_not_authoritative",
    )
    return dataset_facts, roi_facts, None


def _materialize_correction_facts(
    draft: GuidedNewAnalysisDraftPlan,
    roi_facts: GuidedBackendRoiScopeFacts,
    cache_facts: GuidedBackendDiagnosticCacheFacts,
    evidence_facts: GuidedBackendEvidenceReferenceFacts,
) -> tuple[
    GuidedBackendCorrectionFacts | None,
    GuidedBackendValidationMaterializationFailure | None,
]:
    contract = draft.dynamic_fit_parameter_contract
    if contract.unresolved_parameters:
        return None, _failure(
            "dynamic_fit_parameter_unresolved",
            "correction",
            "Dynamic-fit parameter contract contains unresolved parameters.",
            detail_code="dynamic_fit_parameters_unresolved",
        )
    evidence_by_roi = {
        reference.roi_id: reference for reference in evidence_facts.references
    }
    modes = {
        reference.selected_dynamic_fit_mode
        for reference in evidence_facts.references
    }
    if len(modes) != 1:
        return None, _failure(
            "correction_facts_missing",
            "correction",
            "Correction evidence does not resolve to one global mode.",
            detail_code="global_mode_unresolved",
        )
    global_mode = next(iter(modes))
    if contract.dynamic_fit_mode != global_mode:
        return None, _failure(
            "dynamic_fit_parameter_contract_mismatch",
            "correction",
            "Dynamic-fit parameter contract does not match confirmed marks.",
            detail_code="dynamic_fit_mode_mismatch",
        )

    parameter_values: list[GuidedBackendTypedFieldValue] = []
    excluded_parameter_fields = {
        "schema_version",
        "unresolved_parameters",
        "provenance",
    }
    for contract_field in dataclass_fields(contract):
        field_name = contract_field.name
        if field_name in excluded_parameter_fields:
            continue
        value = getattr(contract, field_name)
        parameter_values.append(
            GuidedBackendTypedFieldValue(
                field_name=field_name,
                value_type=type(value).__name__,
                value=value,
                source_classification="applied_dynamic_fit_contract",
            )
        )

    choices_by_roi = {
        choice.roi_id: choice
        for choice in draft.per_roi_correction_strategy_choices
        if choice.roi_id in set(roi_facts.included_roi_ids)
    }
    marks: list[GuidedBackendConfirmedStrategyMarkFacts] = []
    for roi_id in roi_facts.included_roi_ids:
        choice = choices_by_roi.get(roi_id)
        evidence = evidence_by_roi.get(roi_id)
        if choice is None or evidence is None:
            return None, _failure(
                "correction_facts_missing",
                "correction",
                f"Request-ready correction binding is missing for ROI '{roi_id}'.",
                detail_code="confirmed_binding_missing",
            )
        if (
            not choice.explicit_user_mark
            or choice.current_or_stale != "current"
            or evidence.current is not True
            or evidence.diagnostic_cache_id != cache_facts.cache_id
        ):
            return None, _failure(
                "correction_facts_missing",
                "correction",
                f"Confirmed strategy binding is not current for ROI '{roi_id}'.",
                detail_code="confirmed_binding_not_current",
            )
        marks.append(
            GuidedBackendConfirmedStrategyMarkFacts(
                roi_id=roi_id,
                selected_dynamic_fit_mode=global_mode,
                diagnostic_cache_id=cache_facts.cache_id,
                source_setup_signature=cache_facts.source_setup_signature,
                diagnostic_scope_signature=cache_facts.diagnostic_scope_signature,
                build_request_signature=cache_facts.build_request_signature,
                evidence_reference_id=evidence.evidence_reference_id,
                evidence_chunk=evidence.evidence_chunk,
                explicit_user_mark=True,
                current=True,
            )
        )
    return GuidedBackendCorrectionFacts(
        available=True,
        global_dynamic_fit_mode=global_mode,
        dynamic_fit_parameter_values=tuple(parameter_values),
        confirmed_marks=tuple(marks),
    ), None


def _materialize_output_facts(
    draft: GuidedNewAnalysisDraftPlan,
    source_facts: GuidedBackendSourceSnapshotFacts,
    cache_facts: GuidedBackendDiagnosticCacheFacts,
    *,
    additional_protected_roots: tuple[tuple[str, str], ...],
) -> tuple[
    GuidedBackendOutputFacts | None,
    GuidedBackendValidationMaterializationFailure | None,
]:
    if draft.output_policy_status in ("missing", "unavailable"):
        return None, _failure(
            "output_policy_missing",
            "output",
            "An applied output policy is required.",
            detail_code="output_policy_missing",
        )
    if draft.output_policy_status == "stale" or draft.output_policy_stale_reasons:
        return None, _failure(
            "output_policy_stale",
            "output",
            "The applied output policy is stale.",
            detail_code="output_policy_stale",
        )
    if (
        draft.output_policy_status != "applied"
        or draft.output_policy_explicitly_applied is not True
        or draft.output_policy_validation_issues
    ):
        return None, _failure(
            "output_policy_unapplied_changes",
            "output",
            "The output policy is invalid or has unapplied changes.",
            detail_code="output_policy_not_current",
        )

    output_base = str(draft.output_policy_path or "").strip()
    if not output_base:
        return None, _failure(
            "output_base_missing",
            "output",
            "The applied output policy has no output base.",
            detail_code="output_base_missing",
        )

    policy = draft.output_creation_policy
    if policy.overwrite is not False:
        return None, _failure(
            "output_overwrite_not_allowed",
            "output",
            "Overwrite is not allowed for the first Guided validation subset.",
            detail_code="overwrite_requested",
        )
    if policy.precreate_during_preview is not False:
        return None, _failure(
            "output_precreate_not_allowed",
            "output",
            "Output precreation is not allowed during materialization.",
            detail_code="precreate_requested",
        )
    if (
        policy.path_role != "output_base"
        or policy.creation_timing != "future_execution_start_only"
        or policy.run_directory_strategy
        != "derive_unique_run_id_under_output_base"
        or policy.gui_preflight_writes_enabled is not False
    ):
        return None, _failure(
            "output_safety_facts_unavailable",
            "output",
            "The output creation policy is not supported by the first subset.",
            detail_code="output_ownership_policy_mismatch",
        )

    if (
        not source_facts.available
        or not source_facts.source_root_canonical
        or not cache_facts.available
        or not cache_facts.cache_root_canonical
    ):
        return None, _failure(
            "output_protected_root_context_incomplete",
            "output",
            "Source and diagnostic-cache protected-root facts are required.",
            detail_code="fact_derived_roots_missing",
        )
    if not isinstance(additional_protected_roots, tuple):
        return None, _failure(
            "output_protected_root_context_incomplete",
            "output",
            "Additional protected roots must be supplied as a tuple.",
            detail_code="additional_roots_invalid",
        )

    protected_roots: list[tuple[str, str]] = [
        ("diagnostic_cache", cache_facts.cache_root_canonical)
    ]
    for item in additional_protected_roots:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0].strip()
            or not isinstance(item[1], str)
            or not item[1].strip()
        ):
            return None, _failure(
                "output_protected_root_context_incomplete",
                "output",
                "An additional protected-root entry is invalid.",
                detail_code="additional_root_entry_invalid",
            )
        try:
            canonical_root = os.path.realpath(item[1])
        except (OSError, TypeError, ValueError):
            return None, _failure(
                "output_protected_root_context_incomplete",
                "output",
                "An additional protected-root path cannot be canonicalized.",
                detail_code="additional_root_path_invalid",
            )
        protected_roots.append((item[0].strip(), canonical_root))

    try:
        classification = classify_output_base_safety_ownership(
            output_base=output_base,
            source_path=source_facts.source_root_canonical,
            output_policy_status=draft.output_policy_status,
            output_policy_explicitly_applied=draft.output_policy_explicitly_applied,
            output_policy_validation_issues=tuple(
                draft.output_policy_validation_issues
            ),
            output_policy_stale_reasons=tuple(draft.output_policy_stale_reasons),
            path_role=policy.path_role,
            run_directory_strategy=policy.run_directory_strategy,
            overwrite_requested=policy.overwrite,
            precreate_during_preview=policy.precreate_during_preview,
            protected_roots=tuple(protected_roots),
            protected_root_context_complete=True,
            filesystem_facts=None,
            write_context="backend_validation_materialization",
        )
    except Exception as exc:
        return None, _failure(
            "output_safety_facts_unavailable",
            "output",
            f"Output safety classification failed: {exc}",
            detail_code="classifier_failed",
        )

    blockers = tuple(classification.get("blocker_categories") or ())
    if blockers:
        if "output_base_missing" in blockers:
            category = "output_base_missing"
        elif (
            "output_base_relative" in blockers
            or "output_path_style_mismatch" in blockers
        ):
            category = "output_base_invalid"
        elif "unsafe_overwrite_for_guided_first_subset" in blockers:
            category = "output_overwrite_not_allowed"
        elif "unsafe_source_output_relationship" in blockers:
            category = "output_overlaps_source"
        elif "unsafe_protected_output_location" in blockers:
            unsafe_root_kinds = {
                item.get("root_kind")
                for item in classification.get("protected_root_relationships", ())
                if item.get("status") in ("unsafe", "unknown_mixed_path_style")
            }
            if "diagnostic_cache" in unsafe_root_kinds:
                category = "output_overlaps_diagnostic_cache"
            elif "completed_run" in unsafe_root_kinds:
                category = "output_overlaps_completed_run"
            else:
                category = "output_overlaps_protected_root"
        else:
            category = "output_safety_facts_unavailable"
        return None, _failure(
            category,
            "output",
            "The output base failed read-only safety classification.",
            detail_code=str(classification.get("output_safety_status") or "blocked"),
        )

    if (
        classification.get("output_safety_status")
        != "output_base_ready_for_runner_owned_future_mapping"
        or classification.get("future_output_owner") != "runner"
        or classification.get("no_directory_creation") is not True
        or classification.get("no_directory_reservation") is not True
        or classification.get("no_files_written") is not True
    ):
        return None, _failure(
            "output_safety_facts_unavailable",
            "output",
            "Output safety classification did not establish runner-owned future mapping.",
            detail_code="classifier_contract_incomplete",
        )

    relationships: list[GuidedBackendOutputRelationship] = []
    for relationship in classification.get("path_relationships", ()):
        relationships.append(
            GuidedBackendOutputRelationship(
                relationship=str(relationship.get("relationship") or ""),
                root_kind="source",
                status=str(relationship.get("status") or ""),
            )
        )
    for relationship in classification.get("protected_root_relationships", ()):
        relationships.append(
            GuidedBackendOutputRelationship(
                relationship="output_base_vs_protected_root",
                root_kind=str(relationship.get("root_kind") or ""),
                status=str(relationship.get("status") or ""),
            )
        )

    path_relationships = classification.get("path_relationships") or ()
    output_path_style = ""
    if path_relationships:
        evidence = path_relationships[0].get("evidence") or {}
        output_path_style = str(evidence.get("output_path_style") or "")
    if not output_path_style:
        return None, _failure(
            "output_base_invalid",
            "output",
            "The output base path style could not be determined.",
            detail_code="path_style_missing",
        )
    try:
        output_base_canonical = os.path.realpath(os.path.abspath(output_base))
    except (OSError, TypeError, ValueError):
        return None, _failure(
            "output_base_invalid",
            "output",
            "The output base cannot be canonicalized.",
            detail_code="output_base_canonicalization_failed",
        )

    return (
        GuidedBackendOutputFacts(
            available=True,
            output_base_canonical=output_base_canonical,
            output_base_path_style=output_path_style,
            path_role=str(classification["path_role"]),
            future_output_owner=str(classification["future_output_owner"]),
            run_directory_strategy=str(classification["run_directory_strategy"]),
            creation_timing=policy.creation_timing,
            overwrite=False,
            precreate=False,
            policy_status=draft.output_policy_status,
            policy_current=True,
            safety_classifier_version=GUIDED_BACKEND_OUTPUT_SAFETY_CLASSIFIER_VERSION,
            protected_root_context_complete=True,
            relationships=tuple(relationships),
            blocker_categories=(),
            filesystem_fact_scope=(
                "read_only_path_relationships_no_writability_probe"
            ),
        ),
        None,
    )


def materialize_guided_backend_validation_facts(
    draft: GuidedNewAnalysisDraftPlan,
    *,
    parser_contract: RwdHeaderParsingContract | None = None,
    cancellation_check: Callable[[], bool] | None = None,
    additional_protected_roots: tuple[tuple[str, str], ...] = (),
) -> GuidedBackendValidationMaterializationSuccess | GuidedBackendValidationMaterializationFailure:
    """Materialize the Stage 2d subset of validation facts (read-only).

    Currently compiles:
      - Source snapshot facts
      - Incomplete-final not_requested classification facts
      - Parser contract facts
      - Feature/event effective values
      - Diagnostic-cache facts
      - Evidence-reference facts
      - Output safety/ownership facts

    Compiler request population remains deferred, so compiler handoff blocks.
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
        schema_name=parser_contract.schema_name,
        schema_version=parser_contract.schema_version,
        header_search_line_limit=parser_contract.header_search_line_limit,
        time_column_candidates=tuple(parser_contract.time_column_candidates),
        uv_suffix_candidates=tuple(parser_contract.uv_suffix_candidates),
        signal_suffix_candidates=tuple(parser_contract.signal_suffix_candidates),
        column_normalization_rule=parser_contract.column_normalization_rule,
        roi_name_rule=parser_contract.roi_name_rule,
        ambiguity_policy=parser_contract.ambiguity_policy,
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

    source_root_text = snapshot.source_root_canonical
    normalized_source_root = source_root_text.replace("\\", "/")
    if (
        len(normalized_source_root) >= 3
        and normalized_source_root[1] == ":"
        and normalized_source_root[2] == "/"
    ):
        source_root_path_style = "windows_drive"
    elif normalized_source_root.startswith("/"):
        source_root_path_style = "posix"
    else:
        return _failure(
            "source_path_style_unavailable",
            "source",
            "The canonical source-root path style is unavailable.",
            detail_code="source_path_style_unknown",
        )

    source_snapshot_facts = GuidedBackendSourceSnapshotFacts(
        available=True,
        source_root_canonical=snapshot.source_root_canonical,
        source_root_path_style=source_root_path_style,
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

    if not draft.feature_event_profile_id:
        return _failure(
            "feature_event_profile_identity_unavailable",
            "feature_event",
            "The applied feature/event profile has no semantic profile identity.",
            detail_code="profile_id_missing",
        )
    active_fields = tuple(
        item["field_name"]
        for item in effective_values_list
        if item["active_or_inactive"] == "active"
    )
    inactive_fields = tuple(
        item["field_name"]
        for item in effective_values_list
        if item["active_or_inactive"] != "active"
    )
    if (
        len(active_fields) + len(inactive_fields) != len(backend_typed_values)
        or not active_fields
    ):
        return _failure(
            "feature_event_activity_unavailable",
            "feature_event",
            "Feature/event active and inactive field classifications are incomplete.",
            detail_code="feature_activity_incomplete",
        )
    feature_event_facts = GuidedBackendFeatureEventFacts(
        available=True,
        profile_schema_version=(
            GUIDED_BACKEND_FEATURE_EVENT_PROFILE_SCHEMA_VERSION
        ),
        profile_id=draft.feature_event_profile_id,
        effective_values=tuple(backend_typed_values),
        active_fields=active_fields,
        inactive_fields=inactive_fields,
        profile_status=draft.feature_event_profile_status,
        explicitly_applied=draft.feature_event_explicitly_applied,
        current=True,
        visible_unapplied_changes=False,
        validation_issue_categories=tuple(
            draft.feature_event_validation_issues
        ),
        stale_reason_categories=tuple(draft.feature_event_stale_reasons),
    )

    # 11. Cancellation check after feature/event work
    if cancellation_check and cancellation_check():
        return _failure(
            "materialization_cancelled",
            "cancellation",
            "Materialization was cancelled after feature/event compilation.",
            detail_code="cancelled_post_feature",
        )

    # 12. Diagnostic-cache and evidence-reference materialization
    cache_facts, cache_artifact, cache_failure = _materialize_diagnostic_cache(draft)
    if cache_failure is not None:
        return cache_failure
    assert cache_facts is not None and cache_artifact is not None

    evidence_facts, evidence_failure = _materialize_evidence_references(
        draft,
        cache_facts,
        cache_artifact,
    )
    if evidence_failure is not None:
        return evidence_failure
    assert evidence_facts is not None

    dataset_facts, roi_facts, dataset_failure = (
        _materialize_dataset_and_roi_facts(
            draft,
            source_snapshot_facts,
            cache_facts,
        )
    )
    if dataset_failure is not None:
        return dataset_failure
    assert dataset_facts is not None and roi_facts is not None

    correction_facts, correction_failure = _materialize_correction_facts(
        draft,
        roi_facts,
        cache_facts,
        evidence_facts,
    )
    if correction_failure is not None:
        return correction_failure
    assert correction_facts is not None

    output_facts, output_failure = _materialize_output_facts(
        draft,
        source_snapshot_facts,
        cache_facts,
        additional_protected_roots=additional_protected_roots,
    )
    if output_failure is not None:
        return output_failure
    assert output_facts is not None

    facts = GuidedBackendValidationMaterializedFacts(
        source_snapshot=source_snapshot_facts,
        incomplete_final_classification=classification_facts,
        parser=parser_facts,
        acquisition_dataset=dataset_facts,
        roi_scope=roi_facts,
        correction=correction_facts,
        diagnostic_cache=cache_facts,
        output=output_facts,
        evidence_references=evidence_facts,
        feature_event=feature_event_facts,
        effective_feature_event_values=tuple(backend_typed_values),
        complete_for_compilation=True,
        unresolved_required_inputs=(),
    )

    return GuidedBackendValidationMaterializationSuccess(facts=facts)
