"""Minimal non-executing Guided Workflow run-plan contract model.

This module intentionally has no GUI, RunSpec, pipeline, validation, feature
extraction, or output-writing imports. It defines only the small data contract
needed to protect long-duration Guided Workflow scope boundaries before any
production execution is wired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "guided_run_plan.v1"

PLAN_MODES = {"new_analysis", "completed_run_planning"}
SOURCE_MODES = {"raw_input", "completed_run"}
ROI_STATUSES = {"planned", "incomplete", "excluded", "stale"}
FEATURE_EVENT_PROFILE_SCOPES = {"run", "roi", "selected_roi_group"}
EVIDENCE_ROLES = {"representative_evidence"}

RUNNABLE_CORRECTION_STRATEGIES = {
    "robust_global_event_reject",
    "adaptive_event_gated_regression",
    "global_linear_regression",
    "signal_only_f0",
}
FORBIDDEN_CORRECTION_STRATEGIES = {"auto", "needs_review", "no_correction"}
EXPLICIT_CHOICE_SOURCE = "explicit_user_mark"

FEATURE_EVENT_CONFIG_FIELDS = {
    "event_signal",
    "signal_excursion_polarity",
    "peak_threshold_method",
    "peak_threshold_k",
    "peak_threshold_percentile",
    "peak_threshold_abs",
    "peak_min_distance_sec",
    "peak_min_prominence_k",
    "peak_min_width_sec",
    "peak_pre_filter",
    "event_auc_baseline",
}


class GuidedRunPlanContractError(ValueError):
    """Raised when a Guided run-plan contract is invalid."""


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GuidedRunPlanContractError(f"{path} must be an object")
    return value


def _optional_mapping(value: Any, path: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _require_mapping(value, path)


def _require_list(value: Any, path: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise GuidedRunPlanContractError(f"{path} must be a list")
    return value


def _require_non_negative_int(value: Any, path: str) -> int:
    if value is None:
        raise GuidedRunPlanContractError(f"{path} is required")
    if isinstance(value, bool) or not isinstance(value, int):
        raise GuidedRunPlanContractError(f"{path} must be an integer evidence reference")
    if value < 0:
        raise GuidedRunPlanContractError(f"{path} must be non-negative")
    return int(value)


def is_runnable_correction_strategy(strategy: str) -> bool:
    return str(strategy or "").strip() in RUNNABLE_CORRECTION_STRATEGIES


def validate_correction_strategy(strategy: str) -> None:
    value = str(strategy or "").strip()
    if value in FORBIDDEN_CORRECTION_STRATEGIES:
        raise GuidedRunPlanContractError(f"forbidden runnable correction strategy: {value}")
    if value not in RUNNABLE_CORRECTION_STRATEGIES:
        raise GuidedRunPlanContractError(f"unknown runnable correction strategy: {value}")


@dataclass
class GuidedPlanSource:
    source_mode: str
    raw_input_dir: str | None = None
    completed_run_dir: str | None = None
    phasic_out_dir: str | None = None
    source_config_path: str | None = None
    source_run_report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_mode": self.source_mode,
            "raw_input_dir": self.raw_input_dir,
            "completed_run_dir": self.completed_run_dir,
            "phasic_out_dir": self.phasic_out_dir,
            "source_config_path": self.source_config_path,
            "source_run_report_path": self.source_run_report_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuidedPlanSource":
        data = _require_mapping(data, "source")
        return cls(
            source_mode=str(data.get("source_mode") or ""),
            raw_input_dir=data.get("raw_input_dir"),
            completed_run_dir=data.get("completed_run_dir"),
            phasic_out_dir=data.get("phasic_out_dir"),
            source_config_path=data.get("source_config_path"),
            source_run_report_path=data.get("source_run_report_path"),
        )


@dataclass
class EvidenceChunkReview:
    chunk_id: int
    role: str = "representative_evidence"
    diagnostic_artifact_paths: list[str] = field(default_factory=list)
    preview_artifact_paths: list[str] = field(default_factory=list)
    summary: str = ""
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": int(self.chunk_id),
            "role": self.role,
            "diagnostic_artifact_paths": list(self.diagnostic_artifact_paths),
            "preview_artifact_paths": list(self.preview_artifact_paths),
            "summary": self.summary,
            "stale": bool(self.stale),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceChunkReview":
        data = _require_mapping(data, "evidence item")
        return cls(
            chunk_id=_require_non_negative_int(data.get("chunk_id"), "evidence item chunk_id"),
            role=str(data.get("role") or "representative_evidence"),
            diagnostic_artifact_paths=[
                str(x)
                for x in _require_list(
                    data.get("diagnostic_artifact_paths", []),
                    "evidence item diagnostic_artifact_paths",
                )
            ],
            preview_artifact_paths=[
                str(x)
                for x in _require_list(
                    data.get("preview_artifact_paths", []),
                    "evidence item preview_artifact_paths",
                )
            ],
            summary=str(data.get("summary") or ""),
            stale=bool(data.get("stale", False)),
        )


@dataclass
class CorrectionStrategyChoice:
    strategy: str
    strategy_label: str = ""
    choice_source: str = EXPLICIT_CHOICE_SOURCE
    no_auto_selection: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "strategy_label": self.strategy_label,
            "choice_source": self.choice_source,
            "no_auto_selection": bool(self.no_auto_selection),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorrectionStrategyChoice":
        data = _require_mapping(data, "correction_strategy")
        return cls(
            strategy=str(data.get("strategy") or ""),
            strategy_label=str(data.get("strategy_label") or ""),
            choice_source=str(data.get("choice_source") or ""),
            no_auto_selection=bool(data.get("no_auto_selection", False)),
        )


@dataclass
class RoiPlanEntry:
    roi: str
    roi_status: str = "planned"
    correction_strategy: CorrectionStrategyChoice | None = None
    evidence: list[EvidenceChunkReview] = field(default_factory=list)
    feature_event_profile_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi": self.roi,
            "roi_status": self.roi_status,
            "correction_strategy": (
                self.correction_strategy.to_dict() if self.correction_strategy else None
            ),
            "evidence": [item.to_dict() for item in self.evidence],
            "feature_event_profile_id": self.feature_event_profile_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoiPlanEntry":
        data = _require_mapping(data, "roi_plan item")
        choice = data.get("correction_strategy")
        if choice is not None and not isinstance(choice, dict):
            raise GuidedRunPlanContractError("roi_plan item correction_strategy must be an object")
        return cls(
            roi=str(data.get("roi") or ""),
            roi_status=str(data.get("roi_status") or "planned"),
            correction_strategy=(
                CorrectionStrategyChoice.from_dict(choice) if isinstance(choice, dict) else None
            ),
            evidence=[
                EvidenceChunkReview.from_dict(item)
                for item in _require_list(data.get("evidence", []), "roi_plan item evidence")
            ],
            feature_event_profile_id=data.get("feature_event_profile_id"),
        )


@dataclass
class FeatureEventProfile:
    profile_id: str
    scope: str = "run"
    config_fields: dict[str, Any] = field(default_factory=dict)
    evidence_previews: list[EvidenceChunkReview] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "scope": self.scope,
            "config_fields": dict(self.config_fields),
            "evidence_previews": [item.to_dict() for item in self.evidence_previews],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureEventProfile":
        data = _require_mapping(data, "feature_event_profiles item")
        config_fields = data.get("config_fields") or {}
        if not isinstance(config_fields, dict):
            raise GuidedRunPlanContractError("feature_event_profiles item config_fields must be an object")
        return cls(
            profile_id=str(data.get("profile_id") or ""),
            scope=str(data.get("scope") or "run"),
            config_fields=dict(config_fields),
            evidence_previews=[
                EvidenceChunkReview.from_dict(item)
                for item in _require_list(
                    data.get("evidence_previews", []),
                    "feature_event_profiles item evidence_previews",
                )
            ],
        )


@dataclass
class OutputPolicy:
    output_root: str | None = None
    overwrite: bool = False
    separate_from_source_required: bool = True
    legacy_outputs_protected: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "overwrite": bool(self.overwrite),
            "separate_from_source_required": bool(self.separate_from_source_required),
            "legacy_outputs_protected": bool(self.legacy_outputs_protected),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OutputPolicy":
        data = _optional_mapping(data, "output_policy") or {}
        return cls(
            output_root=data.get("output_root"),
            overwrite=bool(data.get("overwrite", False)),
            separate_from_source_required=bool(data.get("separate_from_source_required", True)),
            legacy_outputs_protected=bool(data.get("legacy_outputs_protected", True)),
        )


@dataclass
class PlanProvenanceFlags:
    no_manifest_written: bool = True
    no_pipeline_execution: bool = True
    no_feature_extraction: bool = True
    no_auto_strategy_selection: bool = True
    no_applied_dff_outputs: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "no_manifest_written": bool(self.no_manifest_written),
            "no_pipeline_execution": bool(self.no_pipeline_execution),
            "no_feature_extraction": bool(self.no_feature_extraction),
            "no_auto_strategy_selection": bool(self.no_auto_strategy_selection),
            "no_applied_dff_outputs": bool(self.no_applied_dff_outputs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlanProvenanceFlags":
        data = _optional_mapping(data, "provenance") or {}
        return cls(
            no_manifest_written=bool(data.get("no_manifest_written", True)),
            no_pipeline_execution=bool(data.get("no_pipeline_execution", True)),
            no_feature_extraction=bool(data.get("no_feature_extraction", True)),
            no_auto_strategy_selection=bool(data.get("no_auto_strategy_selection", True)),
            no_applied_dff_outputs=bool(data.get("no_applied_dff_outputs", True)),
        )


@dataclass
class GuidedRunPlan:
    source: GuidedPlanSource
    roi_plan: list[RoiPlanEntry]
    schema_version: str = SCHEMA_VERSION
    plan_id: str = ""
    mode: str = "new_analysis"
    feature_event_profiles: list[FeatureEventProfile] = field(default_factory=list)
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)
    provenance: PlanProvenanceFlags = field(default_factory=PlanProvenanceFlags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "mode": self.mode,
            "source": self.source.to_dict(),
            "roi_plan": [entry.to_dict() for entry in self.roi_plan],
            "feature_event_profiles": [profile.to_dict() for profile in self.feature_event_profiles],
            "output_policy": self.output_policy.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuidedRunPlan":
        data = _require_mapping(data, "guided run plan")
        return cls(
            schema_version=str(data.get("schema_version") or ""),
            plan_id=str(data.get("plan_id") or ""),
            mode=str(data.get("mode") or ""),
            source=GuidedPlanSource.from_dict(data.get("source") or {}),
            roi_plan=[
                RoiPlanEntry.from_dict(item)
                for item in _require_list(data.get("roi_plan", []), "roi_plan")
            ],
            feature_event_profiles=[
                FeatureEventProfile.from_dict(item)
                for item in _require_list(
                    data.get("feature_event_profiles", []),
                    "feature_event_profiles",
                )
            ],
            output_policy=OutputPolicy.from_dict(data.get("output_policy")),
            provenance=PlanProvenanceFlags.from_dict(data.get("provenance")),
        )


def serialize_plan_to_dict(plan: GuidedRunPlan) -> dict[str, Any]:
    return plan.to_dict()


def deserialize_plan_from_dict(data: dict[str, Any]) -> GuidedRunPlan:
    return GuidedRunPlan.from_dict(data)


def validate_plan_contract(plan: GuidedRunPlan) -> list[str]:
    errors: list[str] = []

    if plan.schema_version != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {plan.schema_version}")
    if plan.mode not in PLAN_MODES:
        errors.append(f"invalid plan mode: {plan.mode}")
    if plan.source.source_mode not in SOURCE_MODES:
        errors.append(f"invalid source_mode: {plan.source.source_mode}")
    if plan.mode == "new_analysis" and plan.source.source_mode != "raw_input":
        errors.append("new_analysis plans must use source_mode raw_input")
    if plan.mode == "completed_run_planning" and plan.source.source_mode != "completed_run":
        errors.append("completed_run_planning plans must use source_mode completed_run")
    if plan.source.source_mode == "raw_input" and not plan.source.raw_input_dir:
        errors.append("raw_input source requires raw_input_dir")
    if plan.source.source_mode == "completed_run" and not plan.source.completed_run_dir:
        errors.append("completed_run source requires completed_run_dir")

    seen_rois: set[str] = set()
    for index, entry in enumerate(plan.roi_plan):
        roi = str(entry.roi or "").strip()
        if not roi:
            errors.append(f"roi_plan[{index}] missing roi")
        elif roi in seen_rois:
            errors.append(f"duplicate ROI plan entry: {roi}")
        else:
            seen_rois.add(roi)
        if entry.roi_status not in ROI_STATUSES:
            errors.append(f"roi_plan[{index}] invalid roi_status: {entry.roi_status}")
        if entry.correction_strategy is not None:
            _validate_choice(entry.correction_strategy, f"roi_plan[{index}]", errors)
        for ev_index, evidence in enumerate(entry.evidence):
            _validate_evidence(evidence, f"roi_plan[{index}].evidence[{ev_index}]", errors)

    for index, profile in enumerate(plan.feature_event_profiles):
        if not str(profile.profile_id or "").strip():
            errors.append(f"feature_event_profiles[{index}] missing profile_id")
        if profile.scope not in FEATURE_EVENT_PROFILE_SCOPES:
            errors.append(f"feature_event_profiles[{index}] invalid scope: {profile.scope}")
        unknown = sorted(set(profile.config_fields) - FEATURE_EVENT_CONFIG_FIELDS)
        if unknown:
            errors.append(f"feature_event_profiles[{index}] unknown config fields: {unknown}")
        for ev_index, evidence in enumerate(profile.evidence_previews):
            _validate_evidence(
                evidence,
                f"feature_event_profiles[{index}].evidence_previews[{ev_index}]",
                errors,
            )

    prov = plan.provenance
    if not prov.no_manifest_written:
        errors.append("plan provenance must record no_manifest_written")
    if not prov.no_pipeline_execution:
        errors.append("plan provenance must record no_pipeline_execution")
    if not prov.no_feature_extraction:
        errors.append("plan provenance must record no_feature_extraction")
    if not prov.no_auto_strategy_selection:
        errors.append("plan provenance must record no_auto_strategy_selection")
    if not prov.no_applied_dff_outputs:
        errors.append("plan provenance must record no_applied_dff_outputs")

    return errors


def assert_valid_plan_contract(plan: GuidedRunPlan) -> None:
    errors = validate_plan_contract(plan)
    if errors:
        raise GuidedRunPlanContractError("; ".join(errors))


def _validate_choice(choice: CorrectionStrategyChoice, prefix: str, errors: list[str]) -> None:
    try:
        validate_correction_strategy(choice.strategy)
    except GuidedRunPlanContractError as exc:
        errors.append(f"{prefix}: {exc}")
    if choice.choice_source != EXPLICIT_CHOICE_SOURCE:
        errors.append(f"{prefix}: correction choice_source must be {EXPLICIT_CHOICE_SOURCE}")
    if not choice.no_auto_selection:
        errors.append(f"{prefix}: no_auto_selection must be true")
    if choice.strategy == "signal_only_f0" and choice.choice_source != EXPLICIT_CHOICE_SOURCE:
        errors.append(f"{prefix}: signal_only_f0 must be an explicit user mark")


def _validate_evidence(evidence: EvidenceChunkReview, prefix: str, errors: list[str]) -> None:
    if evidence.role not in EVIDENCE_ROLES:
        errors.append(f"{prefix}: invalid evidence role: {evidence.role}")
    if evidence.chunk_id is None:
        errors.append(f"{prefix}: chunk_id is required")
        return
    if isinstance(evidence.chunk_id, bool) or not isinstance(evidence.chunk_id, int):
        errors.append(f"{prefix}: chunk_id must be an integer evidence reference")
        return
    if evidence.chunk_id < 0:
        errors.append(f"{prefix}: chunk_id must be non-negative")
