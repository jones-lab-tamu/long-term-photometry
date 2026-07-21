"""Canonical, deterministic identity for a current Guided draft analysis plan.

This is the authoritative "current canonical plan identity" used to detect
staleness: any analysis-defining change to a GuidedNewAnalysisDraftPlan
changes this identity, independent of whether any UI callback remembered to
call an invalidation helper. It is pull-based (computed fresh, on demand,
from the plan's current field values) rather than push-based (a hand
maintained dirty flag or revision counter), so it cannot silently miss a
mutation path the way a manually incremented counter can.

Only fields that would change the produced analysis output are included.
Display-only state (cached ids, provenance/cache path strings, staleness
flags themselves, timestamps, labels, warnings/blocking-issue lists,
non-executable readiness flags) is intentionally excluded -- see
guided_new_analysis_plan.py's field-by-field classification for the audit
this module implements.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

from photometry_pipeline.guided_identity import (
    GuidedIdentityError,
    canonicalize_absolute_path,
    encode_canonical_value,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan

GUIDED_PLAN_IDENTITY_SCHEMA_NAME = "guided_new_analysis_draft_plan_identity"
GUIDED_PLAN_IDENTITY_SCHEMA_VERSION = "v3"


def _canonical_path_or_raw(path: str | None) -> dict[str, Any]:
    """Canonicalize an absolute path deterministically.

    Falls back to the raw string for a not-yet-valid in-progress draft
    value (e.g. empty, relative, or not yet browsed) rather than raising --
    an incomplete draft is simply a distinct identity from a complete one.
    """
    if not path:
        return {"raw": ""}
    try:
        canonical = canonicalize_absolute_path(path)
    except GuidedIdentityError:
        return {"raw": str(path)}
    return {"canonical": canonical.canonical_path, "style": canonical.path_style}


def _correction_choice_payload(choice) -> dict[str, Any]:
    return {
        "roi_id": str(choice.roi_id),
        "selected_strategy": str(choice.selected_strategy or ""),
        "current_or_stale": str(choice.current_or_stale or ""),
        "explicit_user_mark": bool(choice.explicit_user_mark),
        "evidence_reference": dict(choice.evidence_reference or {}),
    }


def _feature_event_choice_payload(choice) -> dict[str, Any]:
    return {
        "roi_id": str(choice.roi_id),
        "feature_event_profile_id": str(choice.feature_event_profile_id or ""),
        "config_fields": dict(choice.config_fields or {}),
        "current_or_stale": str(choice.current_or_stale or ""),
        "explicit_user_mark": bool(choice.explicit_user_mark),
    }


def _missing_session_approval_payload(approval) -> dict[str, Any]:
    return {
        "canonical_relative_path": str(approval.canonical_relative_path),
        "size_bytes": int(approval.size_bytes),
        "sha256_content_digest": str(approval.sha256_content_digest),
        "session_index": int(approval.session_index),
        "expected_start_time": str(approval.expected_start_time),
        "expected_duration_sec": float(approval.expected_duration_sec),
        "reason": str(approval.reason or ""),
    }


def _dynamic_fit_parameter_contract_payload(contract) -> dict[str, Any]:
    if contract is None:
        return {}
    payload = dataclasses.asdict(contract)
    # provenance is explanatory metadata about *why* a default was chosen,
    # not a value that changes the produced analysis; schema_version is a
    # contract-version pin, not a per-plan choice.
    payload.pop("provenance", None)
    payload.pop("schema_version", None)
    return payload


def _tonic_settings_contract_payload(contract) -> dict[str, Any]:
    if contract is None:
        return {}
    payload = dataclasses.asdict(contract)
    # provenance is explanatory metadata, not a value that changes the
    # produced analysis; schema_version is a contract-version pin.
    payload.pop("provenance", None)
    payload.pop("schema_version", None)
    return payload


def _dataset_contract_pairing_payload(snapshot) -> dict[str, Any]:
    """Exact signal/reference channel pairing and RWD parsing contract the
    user has confirmed, as it exists on the draft plan today.

    contract_values carries rwd_time_col/sig_suffix/uv_suffix (the signal
    vs. reference channel suffix pairing) and exclude_incomplete_final_rwd_chunk
    once the user has confirmed detected dataset settings (see
    _confirm_detected_dataset_settings_via_review_plan_button /
    _infer_dataset_contract_overrides in gui/main_window.py). No ordering
    policy is invented here -- the dict is taken exactly as currently
    validated.
    """
    contract_values = getattr(snapshot, "contract_values", None)
    return dict(contract_values) if isinstance(contract_values, dict) else {}


def _dataset_contract_source_manifest_payload(snapshot) -> dict[str, Any]:
    """The frozen discovered-session/source-manifest identity currently
    bound to the plan, exactly as GuidedNewAnalysisDatasetContractSnapshot
    carries it (source_setup_signature is a content-hash of the discovered
    raw input/format/acquisition/timing/ROI/output/config state at the
    moment it was last computed; config_fingerprint and
    diagnostic_cache_contract_identity bind the active baseline config and
    diagnostic-cache contract). This is the authoritative object the
    current architecture provides for this identity; see
    GuidedNewAnalysisDatasetContractSourceIdentity in
    guided_new_analysis_plan.py.
    """
    source_identity = getattr(snapshot, "source_identity", None)
    if source_identity is None:
        return {}
    return {
        "source_setup_signature": str(
            getattr(source_identity, "source_setup_signature", "") or ""
        ),
        "config_fingerprint": str(
            getattr(source_identity, "config_fingerprint", "") or ""
        ),
        "diagnostic_cache_contract_identity": str(
            getattr(source_identity, "diagnostic_cache_contract_identity", "")
            or ""
        ),
    }


def build_guided_new_analysis_draft_plan_identity_payload(
    plan: GuidedNewAnalysisDraftPlan,
) -> dict[str, Any]:
    """Build the plain, JSON-canonicalizable payload for `plan`'s identity.

    Exposed separately from the digest for testability and for callers that
    need the payload itself (e.g. to explain a mismatch), not just its hash.
    """
    if not isinstance(plan, GuidedNewAnalysisDraftPlan):
        raise TypeError("plan must be a GuidedNewAnalysisDraftPlan")

    execution_intent = plan.execution_intent
    output_creation_policy = plan.output_creation_policy
    dataset_contract_snapshot = plan.dataset_contract_snapshot

    return {
        "identity_schema_name": GUIDED_PLAN_IDENTITY_SCHEMA_NAME,
        "identity_schema_version": GUIDED_PLAN_IDENTITY_SCHEMA_VERSION,
        "recording": {
            "source_path": _canonical_path_or_raw(
                plan.resolved_input_source_path or plan.input_source_path
            ),
            "input_format": str(plan.input_format or ""),
            "acquisition_mode": str(plan.acquisition_mode or ""),
            "channel_pairing": _dataset_contract_pairing_payload(
                dataset_contract_snapshot
            ),
            "source_manifest": _dataset_contract_source_manifest_payload(
                dataset_contract_snapshot
            ),
        },
        "session_timing": {
            "sessions_per_hour": plan.sessions_per_hour,
            "session_duration_sec": plan.session_duration_sec,
            "continuous_window_sec": plan.continuous_window_sec,
            "continuous_step_sec": plan.continuous_step_sec,
            "allow_partial_final_window": bool(plan.allow_partial_final_window),
            "exclude_incomplete_final_rwd_chunk": bool(
                plan.exclude_incomplete_final_rwd_chunk
            ),
            "approved_missing_sessions": sorted(
                (
                    _missing_session_approval_payload(approval)
                    for approval in plan.approved_missing_sessions
                ),
                key=lambda item: item["canonical_relative_path"],
            ),
        },
        "roi_scope": {
            "included_roi_ids": sorted(str(roi) for roi in plan.included_roi_ids),
            "excluded_roi_ids": sorted(str(roi) for roi in plan.excluded_roi_ids),
        },
        "correction": {
            "global_correction_strategy": str(plan.global_correction_strategy or ""),
            "dynamic_fit_mode": str(plan.dynamic_fit_mode or ""),
            "dynamic_fit_parameter_contract": (
                _dynamic_fit_parameter_contract_payload(
                    plan.dynamic_fit_parameter_contract
                )
            ),
            "per_roi_correction_strategy_choices": sorted(
                (
                    _correction_choice_payload(choice)
                    for choice in plan.per_roi_correction_strategy_choices
                ),
                key=lambda item: item["roi_id"],
            ),
        },
        "tonic_settings": _tonic_settings_contract_payload(
            plan.tonic_settings_contract
        ),
        "feature_detection": {
            "execution_mode": str(
                getattr(execution_intent, "execution_mode", "") or ""
            ),
            "feature_event_profile_id": str(plan.feature_event_profile_id or ""),
            "feature_event_values": dict(plan.feature_event_values or {}),
            "per_roi_feature_event_choices": sorted(
                (
                    _feature_event_choice_payload(choice)
                    for choice in plan.per_roi_feature_event_choices
                ),
                key=lambda item: item["roi_id"],
            ),
        },
        "output": {
            "output_base_path": _canonical_path_or_raw(plan.output_base_path),
            "overwrite": bool(
                getattr(output_creation_policy, "overwrite", False)
            ),
        },
    }


def compute_guided_new_analysis_draft_plan_identity(
    plan: GuidedNewAnalysisDraftPlan,
) -> str:
    """Return the canonical, deterministic identity digest for `plan`.

    Two plans with identical analysis-defining field values produce an
    identical digest regardless of dict/tuple construction order, object
    identity, or any UI-only/display state. Any analysis-defining field
    change produces a different digest.
    """
    payload = build_guided_new_analysis_draft_plan_identity_payload(plan)
    payload_bytes = encode_canonical_value(payload)
    domain = (
        f"{GUIDED_PLAN_IDENTITY_SCHEMA_NAME}:{GUIDED_PLAN_IDENTITY_SCHEMA_VERSION}"
    ).encode("utf-8")
    return hashlib.sha256(domain + b"\x00" + payload_bytes).hexdigest()
