"""Persisted native evidence model for completed-run Review and plots.

This module is deliberately read-only.  It never invokes a correction engine or
reconstructs a Signal-Only baseline from raw fluorescence; current runs are
bound to the terminal verifier's two-copy requested provenance and the HDF5
consumed attributes/datasets that verifier already checked.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import yaml

from photometry_pipeline.input_processing_completeness import (
    DISPOSITION_AUTHORIZED_EXCLUSION,
    DISPOSITION_AUTHORIZED_MISSING,
    DISPOSITION_PROCESS,
    FROZEN_INPUT_MANIFEST_FILENAME,
    read_input_completeness,
    load_frozen_input_manifest,
)
from photometry_pipeline.io.hdf5_cache_reader import (
    open_phasic_cache,
    open_tonic_cache,
    list_cache_rois,
)
from photometry_pipeline.guided_completed_feature_event_reload import (
    load_guided_completed_feature_event_state,
)
from photometry_pipeline.guided_normalized_recording import (
    NormalizedRecordingDescription,
    NormalizedRecordingError,
    NormalizedRoiChannel,
    deserialize_normalized_recording_description,
)
from photometry_pipeline.guided_startup_transaction import (
    GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME,
)
from photometry_pipeline.run_completion_contract import (
    CORRECTION_PROVENANCE_SCHEMA_VERSION,
    GUIDED_CURRENT_NATIVE_STATE_CORRUPTED,
    GUIDED_CURRENT_NATIVE_STATE_CURRENT_NATIVE,
    GUIDED_CURRENT_NATIVE_STATE_MIXED,
    classify_run_terminal_state,
    classify_guided_current_native_state,
    correction_completion_error,
    normalized_recording_completion_error,
    review_with_warnings_eligibility,
)
from photometry_pipeline.core.types import (
    CORRECTION_STRATEGY_FAMILIES,
    RESOLVED_DYNAMIC_FIT_MODES,
)


SCIENTIST_STRATEGY_LABELS = {
    "global_linear_regression": "Global linear regression",
    "robust_global_event_reject": "Robust global fit with event rejection",
    "adaptive_event_gated_regression": "Adaptive event-gated regression",
    "rolling_filtered_to_raw": "Rolling regression (filtered to raw)",
    "rolling_filtered_to_filtered": "Rolling regression (filtered to filtered)",
}


@dataclass(frozen=True)
class AnalysisPlotContext:
    """Immutable source classification plus requested native authority."""

    source_kind: str
    requested_by_roi: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    schema_version: str | None = None


def resolve_analysis_plot_context(analysis_out: str | Path) -> AnalysisPlotContext:
    """Classify an analysis output before a plotter consumes its cache.

    A standalone analysis directory has no run-level terminal files and remains
    usable through its historical plotting path.  Once a directory is
    part of a current run (or carries native run metadata), plotting is strict:
    corrupted/incomplete terminal evidence cannot silently fall back to a
    global config or to ``fit_ref`` presence.
    """
    analysis_path = Path(analysis_out).expanduser().resolve()
    analysis_kind = (
        "tonic"
        if (analysis_path / "tonic_trace_cache.h5").is_file()
        else "phasic"
    )
    run_dir = analysis_path.parent.parent
    classification = classify_run_terminal_state(str(run_dir))
    if classification.is_current:
        metadata = _read_json(analysis_path / "run_metadata.json")
        report = _read_json(analysis_path / "run_report.json")
        requested, native_provenance = _validate_requested_provenance(
            metadata, report, analysis_kind
        )
        if (
            analysis_kind == "tonic"
            and isinstance(metadata, dict)
            and isinstance(metadata.get("correction_provenance"), dict)
            and metadata["correction_provenance"].get("source")
            == "legacy_uniform_translation"
        ):
            return AnalysisPlotContext(source_kind="legacy")
        return AnalysisPlotContext(
            source_kind="current",
            requested_by_roi=(
                _freeze_requested_by_roi(requested)
                if native_provenance
                else _freeze_requested_by_roi({})
            ),
            schema_version=(CORRECTION_PROVENANCE_SCHEMA_VERSION if native_provenance else None),
        )
    if classification.is_legacy:
        return AnalysisPlotContext(source_kind="legacy")

    terminal_files_present = any(
        (run_dir / name).is_file()
        for name in ("status.json", "MANIFEST.json", "run_report.json")
    )

    # A wrapper writes the phasic analysis and its two provenance copies before
    # it can write the final run-level terminal set.  Inspect those copies
    # before deciding that this is a historical standalone directory.
    metadata = _read_json(analysis_path / "run_metadata.json")
    report = _read_json(analysis_path / "run_report.json")
    try:
        requested, native_provenance = _validate_requested_provenance(
            metadata, report, analysis_kind
        )
    except CompletedRunReviewError as exc:
        raise CompletedRunReviewError(
            f"Current native correction provenance is unreadable ({exc})."
        ) from exc
    if (
        analysis_kind == "tonic"
        and isinstance(metadata, dict)
        and isinstance(metadata.get("correction_provenance"), dict)
        and metadata["correction_provenance"].get("source")
        == "legacy_uniform_translation"
    ):
        return AnalysisPlotContext(source_kind="legacy")
    if native_provenance:
        return AnalysisPlotContext(
            source_kind="current",
            requested_by_roi=_freeze_requested_by_roi(requested),
            schema_version=CORRECTION_PROVENANCE_SCHEMA_VERSION,
        )

    if not terminal_files_present:
        return AnalysisPlotContext(source_kind="standalone")
    raise CompletedRunReviewError(
        f"This analysis result cannot be verified for plotting ({classification.reason})."
    )


def classify_analysis_plot_source(analysis_out: str | Path) -> str:
    """Return the historical source-kind API backed by the shared context."""
    return resolve_analysis_plot_context(analysis_out).source_kind


def resolve_persisted_cache_strategy(
    cache: Any,
    roi: str,
    chunk_ids: list[int] | tuple[int, ...],
    *,
    strict_current: bool,
    requested_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one ROI's persisted strategy/reference contract.

    In strict current mode every cached session must carry matching consumed
    family/selection metadata and its family-specific reference dataset.  The
    non-strict branch preserves the positive legacy interpretation while still
    recognizing explicit strategy attributes in standalone native fixtures.
    """
    if not callable(getattr(cache, "get", None)):
        if strict_current:
            raise CompletedRunReviewError(
                f"Current cache exposes no strategy metadata for ROI {roi}."
            )
        return {
            "strategy_family": "dynamic_fit",
            "selected_strategy": "dynamic_fit",
            "dynamic_fit_mode": None,
            "field": "fit_ref",
            "label": "Fitted reference",
            "title": "Dynamic Fit (Raw Signal + Fitted reference)",
        }
    resolved: dict[str, Any] | None = None
    for chunk_id in chunk_ids:
        group = cache.get(f"roi/{roi}/chunk_{int(chunk_id)}")
        if group is None:
            if strict_current:
                raise CompletedRunReviewError(
                    f"Current cache is missing ROI/session {roi}/{chunk_id}."
                )
            continue
        family = str(_attr(group.attrs, "correction_strategy_family", "") or "").strip()
        selected = str(_attr(group.attrs, "correction_selected_strategy", "") or "").strip()
        mode = str(_attr(group.attrs, "correction_dynamic_fit_mode", "") or "").strip() or None
        if strict_current:
            if not isinstance(requested_record, Mapping):
                raise CompletedRunReviewError(
                    f"Current plot has no requested correction provenance for ROI {roi}."
                )
            if str(requested_record.get("roi_id", "")) != str(roi):
                raise CompletedRunReviewError(
                    f"Requested-versus-consumed correction mismatch for ROI {roi}: ROI identity differs."
                )
            if _attr(group.attrs, "correction_execution_status", "") != "consumed":
                raise CompletedRunReviewError(
                    f"Current cache ROI {roi}/{chunk_id} has no consumed correction status."
                )
            if family not in {"dynamic_fit", "signal_only_f0"}:
                raise CompletedRunReviewError(
                    f"Current cache ROI {roi} has no supported consumed strategy family."
                )
            if not selected:
                raise CompletedRunReviewError(
                    f"Current cache ROI {roi} has no consumed strategy selection."
                )
            if family == "signal_only_f0":
                if selected != "signal_only_f0":
                    raise CompletedRunReviewError(
                        f"Current cache ROI {roi} has inconsistent Signal-Only strategy metadata."
                    )
                if mode is not None or any(
                    _attr(group.attrs, key, "")
                    for key in ("dynamic_fit_mode_resolved", "dynamic_fit_engine")
                ):
                    raise CompletedRunReviewError(
                        f"Current cache ROI {roi} carries dynamic-fit attribution for Signal-Only."
                    )
                field = "signal_only_f0_baseline"
                label = "Signal-only F0 baseline"
            else:
                if selected not in RESOLVED_DYNAMIC_FIT_MODES or mode != selected:
                    raise CompletedRunReviewError(
                        f"Current cache ROI {roi} has an unsupported or inconsistent dynamic-fit strategy."
                    )
                if _attr(group.attrs, "dynamic_fit_mode_resolved", "") != selected:
                    raise CompletedRunReviewError(
                        f"Current cache ROI {roi} has no resolved dynamic-fit mode."
                    )
                field = "fit_ref"
                label = "Fitted reference"
            if field not in group:
                raise CompletedRunReviewError(
                    f"Current cache ROI {roi}/{chunk_id} is missing its {label}."
                )
            requested_family = str(requested_record.get("strategy_family", ""))
            requested_selected = str(requested_record.get("selected_strategy", ""))
            requested_mode = requested_record.get("dynamic_fit_mode")
            if (
                family != requested_family
                or selected != requested_selected
                or mode != requested_mode
            ):
                raise CompletedRunReviewError(
                    f"Requested-versus-consumed correction mismatch for ROI {roi}/{chunk_id}."
                )
            for requested_key, consumed_key in (
                ("parameter_identity", "correction_parameter_identity"),
                ("evidence_identity", "correction_evidence_identity"),
            ):
                requested_identity = requested_record.get(requested_key, "")
                if requested_identity and _attr(group.attrs, consumed_key, "") != requested_identity:
                    raise CompletedRunReviewError(
                        f"Requested-versus-consumed correction mismatch for ROI {roi}/{chunk_id}: {requested_key}."
                    )
            candidate = {
                "strategy_family": family,
                "selected_strategy": selected,
                "dynamic_fit_mode": mode,
                "parameter_identity": _attr(group.attrs, "correction_parameter_identity", ""),
                "evidence_identity": _attr(group.attrs, "correction_evidence_identity", ""),
                "field": field,
                "label": label,
                "title": (
                    "Signal-Only F0 (Raw Signal + Signal-only F0 baseline)"
                    if family == "signal_only_f0"
                    else "Dynamic Fit (Raw Signal + Fitted reference)"
                ),
            }
            if resolved is not None and (
                candidate["strategy_family"] != resolved["strategy_family"]
                or candidate["selected_strategy"] != resolved["selected_strategy"]
                or candidate["field"] != resolved["field"]
            ):
                raise CompletedRunReviewError(
                    f"Current cache ROI {roi} changes correction strategy across sessions."
                )
            resolved = candidate
        elif resolved is None:
            if family == "signal_only_f0" or selected == "signal_only_f0":
                resolved = {
                    "strategy_family": "signal_only_f0",
                    "selected_strategy": "signal_only_f0",
                    "dynamic_fit_mode": None,
                    "field": "signal_only_f0_baseline",
                    "label": "Signal-only F0 baseline",
                    "title": "Signal-Only F0 (Raw Signal + Signal-only F0 baseline)",
                }
            elif family == "dynamic_fit" or "fit_ref" in group:
                resolved = {
                    "strategy_family": "dynamic_fit",
                    "selected_strategy": selected or mode or "dynamic_fit",
                    "dynamic_fit_mode": mode,
                    "field": "fit_ref",
                    "label": "Fitted reference",
                    "title": "Dynamic Fit (Raw Signal + Fitted reference)",
                }
    if resolved is not None:
        return resolved
    if strict_current:
        raise CompletedRunReviewError(
            f"Current cache contains no strategy evidence for ROI {roi}."
        )
    return {
        "strategy_family": "dynamic_fit",
        "selected_strategy": "dynamic_fit",
        "dynamic_fit_mode": None,
        "field": "fit_ref",
        "label": "Fitted reference",
        "title": "Dynamic Fit (Raw Signal + Fitted reference)",
    }


class CompletedRunReviewError(RuntimeError):
    """The persisted completed result cannot be interpreted safely."""


@dataclass(frozen=True)
class CompletedReviewSession:
    analysis_branch: str
    roi_id: str
    session_index: int
    chunk_id: int | None
    source_file: str
    disposition: str
    missing_reason: str = ""
    time_sec: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float), repr=False)
    raw_signal: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float), repr=False)
    canonical_dff: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float), repr=False)
    strategy_family: str = ""
    selected_strategy: str = ""
    dynamic_fit_mode: str | None = None
    fitted_reference: np.ndarray | None = field(default=None, repr=False)
    production_f0_baseline: np.ndarray | None = field(default=None, repr=False)
    signal_only_qc: dict[str, Any] = field(default_factory=dict)
    processing_diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def processed(self) -> bool:
        return self.disposition == DISPOSITION_PROCESS

    @property
    def strategy_label(self) -> str:
        if self.strategy_family == "signal_only_f0":
            return "Signal-Only F0"
        return SCIENTIST_STRATEGY_LABELS.get(
            self.dynamic_fit_mode or self.selected_strategy,
            "Fitted reference",
        )

    @property
    def reference_label(self) -> str:
        return (
            "Signal-only F0 baseline"
            if self.strategy_family == "signal_only_f0"
            else "Fitted reference"
        )

    @property
    def correction_reference(self) -> np.ndarray | None:
        if self.strategy_family == "signal_only_f0":
            return self.production_f0_baseline
        return self.fitted_reference


@dataclass(frozen=True)
class CompletedRunReviewModel:
    run_dir: str
    rois: tuple[str, ...]
    sessions_by_roi: dict[str, tuple[CompletedReviewSession, ...]]
    requested_by_roi: dict[str, dict[str, Any]]
    feature_settings_by_roi: dict[str, dict[str, Any]]
    current_native: bool
    terminal_state: str
    analysis_branches: tuple[str, ...] = ("phasic",)
    sessions_by_branch_roi: dict[
        str, dict[str, tuple[CompletedReviewSession, ...]]
    ] = field(default_factory=dict)
    # Current-native Guided ROI authority, including excluded discovered ROIs.
    # ``rois`` remains the included, scientist-visible subset for compatibility.
    roi_inventory: tuple[NormalizedRoiChannel, ...] = ()
    # B1: the verified, authorized normalized recording description for
    # current-native Guided runs (input format, acquisition mode, ordered
    # session identities/dispositions, ROI/channel identities, sampling).
    # None for legacy/non-Guided runs -- their existing session/ROI
    # construction above is entirely unaffected. When present, it has
    # already been verified via normalized_recording_completion_error
    # against this run's own consumed C8/cache evidence before this model
    # was constructed, so the sessions_by_roi facts above are provably
    # consistent with it, not merely coincidentally matching.
    normalized_recording: NormalizedRecordingDescription | None = None

    @property
    def heterogeneous_correction(self) -> bool:
        families = {
            str(record.get("strategy_family", ""))
            for record in self.requested_by_roi.values()
        }
        selections = {
            str(record.get("selected_strategy", ""))
            for record in self.requested_by_roi.values()
        }
        return len(families) > 1 or len(selections) > 1

    def sessions_for_roi(
        self, roi_id: str, branch: str | None = None
    ) -> tuple[CompletedReviewSession, ...]:
        if branch is not None and self.sessions_by_branch_roi:
            return self.sessions_by_branch_roi.get(str(branch), {}).get(str(roi_id), ())
        return self.sessions_by_roi.get(str(roi_id), ())

    def strategy_label_for_roi(self, roi_id: str) -> str:
        records = self.sessions_for_roi(roi_id)
        for record in records:
            if record.processed:
                return record.strategy_label
        requested = self.requested_by_roi.get(str(roi_id), {})
        if requested.get("strategy_family") == "signal_only_f0":
            return "Signal-Only F0"
        return SCIENTIST_STRATEGY_LABELS.get(
            requested.get("dynamic_fit_mode") or requested.get("selected_strategy"),
            "Fitted reference",
        )

    def signal_only_qc_for_roi(self, roi_id: str) -> dict[str, Any]:
        """Return the persisted concise QC attributes for one ROI."""
        for record in self.sessions_for_roi(roi_id):
            if record.processed and record.strategy_family == "signal_only_f0":
                return dict(record.signal_only_qc)
        return {}

    def feature_settings_summary_for_roi(self, roi_id: str) -> str:
        """Return a scientist-facing summary of the selected ROI's settings."""
        row = self.feature_settings_by_roi.get(str(roi_id))
        if not row:
            if self.current_native:
                return "Feature detection settings are unavailable for this result."
            return "Feature detection settings are unavailable for this legacy result."
        fields = row.get("effective_config_fields")
        if not isinstance(fields, dict) or not fields:
            return "Feature detection settings are unavailable for this result."
        source = str(row.get("source", "")).strip().lower()
        source_label = "Custom feature settings" if source == "override" else "Default feature settings"
        method = str(fields.get("peak_threshold_method", "mean_std")).strip().lower()
        method_labels = {
            "mean_std": "mean ± SD threshold",
            "percentile": "percentile threshold",
            "absolute": "absolute threshold",
        }
        method_label = method_labels.get(method, "configured threshold")
        threshold_key = {
            "mean_std": "peak_threshold_k",
            "percentile": "peak_threshold_percentile",
            "absolute": "peak_threshold_abs",
        }.get(method)
        threshold = fields.get(threshold_key, "") if threshold_key else ""
        signal = str(fields.get("event_signal", "dff")).strip().lower()
        signal_label = {"dff": "dF/F", "delta_f": "delta F"}.get(signal, signal or "selected trace")
        threshold_text = f" ({threshold:g})" if isinstance(threshold, (int, float)) else (f" ({threshold})" if threshold else "")
        return f"{source_label}: {method_label}{threshold_text}; event signal {signal_label}."


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CompletedRunReviewError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompletedRunReviewError(f"{path.name} is not a JSON object")
    return value


def _attr(attrs: Any, key: str, default: Any = None) -> Any:
    value = attrs.get(key, default)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _same_source_identity(left: str, right: str) -> bool:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(left)))) == os.path.normcase(
        os.path.abspath(os.path.normpath(str(right)))
    )


def _load_sessions_index(
    run_dir: Path, analysis_kind: str = "phasic"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = run_dir / FROZEN_INPUT_MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest, error = load_frozen_input_manifest(str(manifest_path))
        if manifest is None:
            raise CompletedRunReviewError(f"Frozen session index is invalid: {error}")
        payload, payload_error = read_input_completeness(
            str(run_dir / "_analysis" / f"{analysis_kind}_out")
        )
        if payload is None:
            raise CompletedRunReviewError(
                f"{analysis_kind.capitalize()} session accounting is invalid: {payload_error}"
            )
        return list(manifest.get("expected", [])), list(payload.get("processed", []))

    payload, error = read_input_completeness(
        str(run_dir / "_analysis" / f"{analysis_kind}_out")
    )
    if payload is None:
        return [], []
    return list(payload.get("expected", [])), list(payload.get("processed", []))


def _load_feature_settings(phasic_dir: Path) -> dict[str, dict[str, Any]]:
    state = load_guided_completed_feature_event_state(phasic_dir.parent.parent)
    if not state.present or not state.valid:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in state.rows:
        out[str(row.roi)] = {
            "roi": str(row.roi),
            "source": str(row.source),
            "feature_event_profile_id": str(row.feature_event_profile_id),
            "effective_config_fields": dict(row.effective_config_fields),
        }
    return out


def _validate_requested_provenance(
    metadata: dict[str, Any] | None,
    report: dict[str, Any] | None,
    analysis_kind: str = "phasic",
) -> tuple[dict[str, dict[str, Any]], bool]:
    metadata_prov = metadata.get("correction_provenance") if metadata else None
    derived = report.get("derived_settings") if report else None
    report_prov = derived.get("correction_provenance") if isinstance(derived, dict) else None
    has_metadata = isinstance(metadata, dict) and "correction_provenance" in metadata
    has_report = isinstance(derived, dict) and "correction_provenance" in derived
    if not has_metadata and not has_report:
        return {}, False
    if has_metadata != has_report:
        raise CompletedRunReviewError(
            "Completed result has only one copy of correction settings."
        )
    if not isinstance(metadata_prov, dict) or not isinstance(report_prov, dict):
        raise CompletedRunReviewError(
            "Completed result correction settings are malformed."
        )
    if metadata_prov != report_prov:
        raise CompletedRunReviewError(
            "Completed result correction settings disagree between report and metadata."
        )
    if metadata_prov.get("schema_version") != CORRECTION_PROVENANCE_SCHEMA_VERSION:
        raise CompletedRunReviewError("Completed result correction settings are unsupported.")
    if metadata_prov.get("analysis_mode") != analysis_kind:
        raise CompletedRunReviewError(
            f"Completed result correction settings are not bound to {analysis_kind} analysis."
        )
    if metadata_prov.get("source") not in {
        "explicit_per_roi_map",
        "legacy_uniform_translation",
    }:
        raise CompletedRunReviewError("Completed result correction settings have an unsupported source.")
    records = metadata_prov.get("requested_by_roi")
    if not isinstance(records, list):
        raise CompletedRunReviewError("Completed result correction settings are incomplete.")
    included = metadata_prov.get("included_roi_ids")
    if not isinstance(included, list):
        raise CompletedRunReviewError("Completed result correction settings have incomplete ROI coverage.")
    if len(included) != len(set(str(roi) for roi in included)):
        raise CompletedRunReviewError("Completed result correction settings duplicate an included ROI.")
    by_roi: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not str(record.get("roi_id", "")):
            raise CompletedRunReviewError("Completed result correction settings contain an invalid ROI.")
        roi = str(record["roi_id"])
        if roi in by_roi:
            raise CompletedRunReviewError("Completed result correction settings duplicate an ROI.")
        family = record.get("strategy_family")
        selected = record.get("selected_strategy")
        mode = record.get("dynamic_fit_mode")
        if family not in CORRECTION_STRATEGY_FAMILIES:
            raise CompletedRunReviewError(
                f"Completed result correction settings have an unsupported strategy family for ROI {roi}."
            )
        if not isinstance(selected, str) or not selected:
            raise CompletedRunReviewError(
                f"Completed result correction settings have no selected strategy for ROI {roi}."
            )
        if family == "dynamic_fit":
            if mode not in RESOLVED_DYNAMIC_FIT_MODES or selected != mode:
                raise CompletedRunReviewError(
                    f"Completed result correction settings have an invalid dynamic-fit mode for ROI {roi}."
                )
        elif selected != "signal_only_f0" or mode is not None:
            raise CompletedRunReviewError(
                f"Completed result correction settings have invalid Signal-Only fields for ROI {roi}."
            )
        for identity_key in ("parameter_identity", "evidence_identity"):
            value = record.get(identity_key, "")
            if value is not None and not isinstance(value, str):
                raise CompletedRunReviewError(
                    f"Completed result correction settings have malformed {identity_key} for ROI {roi}."
                )
        by_roi[roi] = dict(record)
    if set(by_roi) != {str(roi) for roi in included}:
        raise CompletedRunReviewError(
            "Completed result correction settings do not cover exactly the included ROIs."
        )
    return by_roi, True


def _freeze_requested_by_roi(
    requested_by_roi: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    return MappingProxyType(
        {
            str(roi): MappingProxyType(dict(record))
            for roi, record in requested_by_roi.items()
        }
    )


def _load_verified_normalized_recording(
    resolved: Path, run_mode: dict[str, Any], *, required: bool = False
) -> NormalizedRecordingDescription | None:
    """Load and verify this run's authorized normalized recording
    description, following the same precedent already used for
    ``load_guided_completed_applied_dff_state``/
    ``load_guided_completed_feature_event_state`` (a Guided-produced,
    run-root JSON artifact read post-hoc for Review, never the live source
    tree). Returns None for a legacy/non-Guided run (the file is simply
    absent -- existing behavior entirely unchanged); raises
    CompletedRunReviewError fail-closed for a current-native Guided run
    whose provenance is missing, malformed, or contradicts its own
    consumed evidence.
    """
    normalized_path = resolved / GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME
    if not normalized_path.is_file():
        if required:
            raise CompletedRunReviewError(
                "Completed Guided result is missing its normalized recording provenance."
            )
        return None
    try:
        payload = json.loads(normalized_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CompletedRunReviewError(
            f"Completed result normalized recording provenance is unreadable ({exc})."
        ) from exc
    try:
        normalized_recording = deserialize_normalized_recording_description(payload)
    except NormalizedRecordingError as exc:
        raise CompletedRunReviewError(
            f"Completed result normalized recording provenance could not be "
            f"verified ({exc})."
        ) from exc
    reconciliation_error = normalized_recording_completion_error(
        str(resolved), run_mode
    )
    if reconciliation_error:
        raise CompletedRunReviewError(
            "Completed result normalized recording provenance disagrees with "
            f"its consumed evidence ({reconciliation_error})."
        )
    return normalized_recording


def _load_completed_branch_review(
    run_dir: str | Path, analysis_kind: str
) -> CompletedRunReviewModel:
    """Load one completed analysis branch exclusively from persisted evidence."""
    resolved = Path(run_dir).expanduser().resolve()
    classification = classify_run_terminal_state(str(resolved))
    if not classification.is_success:
        raise CompletedRunReviewError(
            f"This completed result cannot be verified ({classification.reason})."
        )
    guided_state = classify_guided_current_native_state(
        str(resolved), classification.run_mode or None
    )
    if guided_state in {
        GUIDED_CURRENT_NATIVE_STATE_MIXED,
        GUIDED_CURRENT_NATIVE_STATE_CORRUPTED,
    }:
        raise CompletedRunReviewError(
            "Completed result carries mixed or corrupted Guided provenance."
        )
    analysis_dir = resolved / "_analysis" / f"{analysis_kind}_out"
    cache_path = analysis_dir / f"{analysis_kind}_trace_cache.h5"
    if not cache_path.is_file():
        raise CompletedRunReviewError(
            f"Completed result has no {analysis_kind} canonical trace cache."
        )

    metadata = _read_json(analysis_dir / "run_metadata.json")
    report = _read_json(analysis_dir / "run_report.json")
    requested_by_roi, correction_native = _validate_requested_provenance(
        metadata, report, analysis_kind
    )
    guided_current_native = guided_state == GUIDED_CURRENT_NATIVE_STATE_CURRENT_NATIVE
    current_native = bool(correction_native or guided_current_native)
    if classification.is_current and not current_native:
        raise CompletedRunReviewError(
            f"The current {analysis_kind} result has no verified native correction settings."
        )
    if current_native:
        correction_error = correction_completion_error(
            str(resolved), classification.run_mode
        )
        if correction_error:
            raise CompletedRunReviewError(
                f"Completed result correction evidence could not be verified ({correction_error})."
            )

    normalized_recording = _load_verified_normalized_recording(
        resolved,
        classification.run_mode,
        required=guided_current_native,
    )

    expected, processed = _load_sessions_index(resolved, analysis_kind)
    processed_by_index = {
        int(record["index"]): record
        for record in processed
        if isinstance(record, dict) and isinstance(record.get("index"), int)
    }

    opener = open_phasic_cache if analysis_kind == "phasic" else open_tonic_cache
    with opener(str(cache_path)) as cache:
        cache_rois = [str(roi) for roi in list_cache_rois(cache)]
        roi_inventory = (
            tuple(normalized_recording.roi_channels)
            if guided_current_native and normalized_recording is not None
            else ()
        )
        if guided_current_native:
            rois = [item.roi_id for item in roi_inventory if item.included]
            if set(rois) != set(cache_rois):
                raise CompletedRunReviewError(
                    "Completed result cache ROI identities do not match the "
                    "authorized normalized ROI inventory."
                )
        elif current_native:
            rois = [str(roi) for roi in requested_by_roi]
            if set(rois) != set(cache_rois):
                raise CompletedRunReviewError("Completed result ROI identities do not match its correction settings.")
        else:
            # Positive legacy runs retain the historical uniform fit-only view.
            rois = cache_rois

        config_mode = ""
        if not current_native:
            try:
                config = yaml.safe_load((analysis_dir / "config_used.yaml").read_text(encoding="utf-8")) or {}
            except Exception:
                config = {}
            config_mode = str(config.get("dynamic_fit_mode", "")) if isinstance(config, dict) else ""

        sessions_by_roi: dict[str, list[CompletedReviewSession]] = {roi: [] for roi in rois}
        c8_expected_by_index = {
            int(entry["index"]): entry
            for entry in expected
            if isinstance(entry, dict) and isinstance(entry.get("index"), int)
        }
        if guided_current_native:
            assert normalized_recording is not None
            normalized_by_index = {
                item.chronological_position: item
                for item in normalized_recording.sessions
            }
            if set(normalized_by_index) != set(c8_expected_by_index):
                raise CompletedRunReviewError(
                    "Completed result C8 session accounting does not cover the "
                    "normalized chronological session inventory."
                )
            normalized_entries: list[dict[str, Any]] = []
            disposition_names = {
                "process": DISPOSITION_PROCESS,
                "missing": DISPOSITION_AUTHORIZED_MISSING,
                "excluded": DISPOSITION_AUTHORIZED_EXCLUSION,
            }
            for session in normalized_recording.sessions:
                c8_entry = c8_expected_by_index[session.chronological_position]
                if not _same_source_identity(
                    str(c8_entry.get("source", "")),
                    session.canonical_source_reference,
                ):
                    raise CompletedRunReviewError(
                        f"Completed result C8 source identity disagrees with normalized "
                        f"session {session.chronological_position}."
                    )
                if c8_entry.get("disposition") != disposition_names[session.disposition]:
                    raise CompletedRunReviewError(
                        f"Completed result C8 disposition disagrees with normalized "
                        f"session {session.chronological_position}."
                    )
                normalized_entries.append(
                    {
                        "index": session.chronological_position,
                        "source": session.canonical_source_reference,
                        "disposition": disposition_names[session.disposition],
                        # Reason is an outcome/diagnostic owned by C8, not by
                        # normalized recording provenance.
                        "reason": c8_entry.get(
                            "reason", c8_entry.get("failure_category", "")
                        ),
                    }
                )
            expected_entries = normalized_entries
        else:
            expected_entries = sorted(
                [entry for entry in expected if isinstance(entry, dict)],
                key=lambda entry: int(entry.get("index", 0)),
            )
        using_cache_fallback = False
        if not expected_entries:
            # Preview/legacy fallback: expose cache chunks as processed sessions.
            using_cache_fallback = True
            expected_entries = [
                {"index": int(chunk_id), "disposition": DISPOSITION_PROCESS}
                for chunk_id in cache["meta"]["chunk_ids"][()]
            ]

        for entry in expected_entries:
            session_index = int(entry.get("index", 0))
            disposition = str(entry.get("disposition", DISPOSITION_PROCESS))
            processed_record = processed_by_index.get(session_index)
            if guided_current_native:
                if disposition == DISPOSITION_PROCESS:
                    if processed_record is None:
                        raise CompletedRunReviewError(
                            f"Completed result is missing the processed C8 record for "
                            f"normalized session {session_index}."
                        )
                    if not _same_source_identity(
                        str(processed_record.get("source", "")),
                        str(entry.get("source", "")),
                    ):
                        raise CompletedRunReviewError(
                            f"Completed result processed C8 source disagrees with "
                            f"normalized session {session_index}."
                        )
                elif processed_record is not None:
                    raise CompletedRunReviewError(
                        f"Completed result has an ordinary processed C8 record for "
                        f"normalized non-processed session {session_index}."
                    )
            chunk_id = (
                int(processed_record["cache_chunk_id"])
                if isinstance(processed_record, dict)
                and isinstance(processed_record.get("cache_chunk_id"), int)
                else (session_index if using_cache_fallback else None)
            )
            source_file = str(
                entry.get("source", "")
                if guided_current_native
                else (processed_record or {}).get("source", entry.get("source", ""))
            )
            reason = str(entry.get("reason", entry.get("failure_category", "")) or "")
            processing_diagnostics = dict(processed_record or {})
            for roi in rois:
                if disposition != DISPOSITION_PROCESS:
                    sessions_by_roi[roi].append(
                        CompletedReviewSession(
                            analysis_branch=analysis_kind,
                            roi_id=roi,
                            session_index=session_index,
                            chunk_id=None,
                            source_file=source_file,
                            disposition=disposition,
                            missing_reason=reason,
                            strategy_family=(
                                requested_by_roi.get(roi, {}).get("strategy_family", "")
                            ),
                            selected_strategy=(
                                requested_by_roi.get(roi, {}).get("selected_strategy", "")
                            ),
                            processing_diagnostics=processing_diagnostics,
                        )
                    )
                    continue
                if chunk_id is None:
                    raise CompletedRunReviewError(
                        f"Processed session {session_index} has no canonical chunk identity."
                    )
                try:
                    arrays = cache[f"roi/{roi}/chunk_{chunk_id}"]
                except Exception as exc:
                    raise CompletedRunReviewError(
                        f"Completed result is missing ROI/session {roi}/{chunk_id}."
                    ) from exc
                trace_field = (
                    "dff"
                    if "dff" in arrays
                    else "deltaF"
                    if analysis_kind == "tonic" and not current_native
                    else "dff"
                )
                required = ("time_sec", "sig_raw", trace_field)
                if any(name not in arrays for name in required):
                    raise CompletedRunReviewError(
                        f"Completed result is missing canonical data for ROI/session {roi}/{chunk_id}."
                    )
                time_sec = np.asarray(arrays["time_sec"][()], dtype=float)
                raw_signal = np.asarray(arrays["sig_raw"][()], dtype=float)
                dff = np.asarray(arrays[trace_field][()], dtype=float)
                attrs = arrays.attrs
                if current_native:
                    request = requested_by_roi.get(roi)
                    if request is None:
                        raise CompletedRunReviewError(f"No requested correction settings for ROI {roi}.")
                    family = str(request.get("strategy_family", ""))
                    selected = str(request.get("selected_strategy", ""))
                    mode = request.get("dynamic_fit_mode")
                    if str(_attr(attrs, "correction_strategy_family", "")) != family:
                        raise CompletedRunReviewError(f"Consumed strategy mismatch for ROI {roi}.")
                    if str(_attr(attrs, "correction_selected_strategy", "")) != selected:
                        raise CompletedRunReviewError(f"Consumed selection mismatch for ROI {roi}.")
                else:
                    family = "dynamic_fit"
                    mode = str(_attr(attrs, "dynamic_fit_mode_resolved", config_mode) or config_mode)
                    selected = mode or "dynamic_fit"

                fit_ref = None
                baseline = None
                qc: dict[str, Any] = {}
                if family == "signal_only_f0":
                    if "signal_only_f0_baseline" not in arrays:
                        raise CompletedRunReviewError(f"Signal-Only baseline is missing for ROI {roi}.")
                    baseline = np.asarray(arrays["signal_only_f0_baseline"][()], dtype=float)
                    for key, value in attrs.items():
                        key_text = str(key)
                        if key_text.startswith("signal_only_f0_production_") or key_text in {
                            "signal_only_f0_status",
                            "signal_only_f0_warning",
                            "signal_only_f0_candidate_viability",
                            "signal_only_f0_candidate_confidence",
                            "signal_only_f0_anchor_status",
                            "signal_only_f0_flags",
                        }:
                            qc[key_text] = _attr(attrs, key_text, value)
                elif family == "dynamic_fit":
                    if "fit_ref" not in arrays and not (
                        analysis_kind == "tonic" and not current_native
                    ):
                        raise CompletedRunReviewError(f"Fitted reference is missing for ROI {roi}.")
                    if "fit_ref" in arrays:
                        fit_ref = np.asarray(arrays["fit_ref"][()], dtype=float)
                else:
                    raise CompletedRunReviewError(f"Unknown consumed correction family for ROI {roi}.")

                sessions_by_roi[roi].append(
                    CompletedReviewSession(
                        analysis_branch=analysis_kind,
                        roi_id=roi,
                        session_index=session_index,
                        chunk_id=chunk_id,
                        source_file=source_file,
                        disposition=disposition,
                        time_sec=time_sec,
                        raw_signal=raw_signal,
                        canonical_dff=dff,
                        strategy_family=family,
                        selected_strategy=selected,
                        dynamic_fit_mode=(str(mode) if mode is not None else None),
                        fitted_reference=fit_ref,
                        production_f0_baseline=baseline,
                        signal_only_qc=qc,
                        processing_diagnostics=processing_diagnostics,
                    )
                )

    return CompletedRunReviewModel(
        run_dir=str(resolved),
        rois=tuple(rois),
        sessions_by_roi={roi: tuple(rows) for roi, rows in sessions_by_roi.items()},
        requested_by_roi=requested_by_roi,
        feature_settings_by_roi=(
            _load_feature_settings(analysis_dir) if analysis_kind == "phasic" else {}
        ),
        current_native=current_native,
        terminal_state=classification.state,
        analysis_branches=(analysis_kind,),
        sessions_by_branch_roi={
            analysis_kind: {
                roi: tuple(rows) for roi, rows in sessions_by_roi.items()
            }
        },
        roi_inventory=roi_inventory,
        normalized_recording=normalized_recording,
    )


def load_completed_phasic_review(run_dir: str | Path) -> CompletedRunReviewModel:
    """Load the existing completed Review surface for tonic, phasic, or both."""
    resolved = Path(run_dir).expanduser().resolve()
    classification = classify_run_terminal_state(str(resolved))
    if not classification.is_success:
        raise CompletedRunReviewError(
            f"This completed result cannot be verified ({classification.reason})."
        )
    enabled = tuple(
        branch
        for branch, key in (("tonic", "tonic_analysis"), ("phasic", "phasic_analysis"))
        if classification.run_mode.get(key)
    )
    if not enabled and classification.is_legacy:
        enabled = tuple(
            branch
            for branch in ("tonic", "phasic")
            if (
                resolved / "_analysis" / f"{branch}_out" / f"{branch}_trace_cache.h5"
            ).is_file()
        )
    if not enabled:
        raise CompletedRunReviewError("Completed result has no reviewable analysis branch.")
    models = {
        branch: _load_completed_branch_review(resolved, branch)
        for branch in enabled
    }
    if len(models) == 1:
        return next(iter(models.values()))

    tonic = models["tonic"]
    phasic = models["phasic"]
    if tonic.current_native != phasic.current_native:
        raise CompletedRunReviewError(
            "Tonic and phasic correction evidence do not describe the same completed analysis."
        )
    if tonic.current_native and tonic.requested_by_roi != phasic.requested_by_roi:
        raise CompletedRunReviewError(
            "Tonic and phasic correction approaches disagree."
        )
    if set(tonic.rois) != set(phasic.rois):
        raise CompletedRunReviewError("Tonic and phasic ROI identities disagree.")
    if (tonic.normalized_recording is None) != (phasic.normalized_recording is None):
        raise CompletedRunReviewError(
            "Tonic and phasic normalized recording provenance do not describe the "
            "same completed analysis."
        )
    if (
        tonic.normalized_recording is not None
        and phasic.normalized_recording is not None
        and tonic.normalized_recording != phasic.normalized_recording
    ):
        raise CompletedRunReviewError(
            "Tonic and phasic normalized recording provenance disagree."
        )
    # The existing UI remains phasic-primary for feature/result images, while
    # branch-qualified sessions preserve both canonical trace sets.
    return CompletedRunReviewModel(
        run_dir=phasic.run_dir,
        rois=phasic.rois,
        sessions_by_roi=phasic.sessions_by_roi,
        requested_by_roi=phasic.requested_by_roi,
        feature_settings_by_roi=phasic.feature_settings_by_roi,
        current_native=phasic.current_native,
        terminal_state=phasic.terminal_state,
        analysis_branches=("tonic", "phasic"),
        sessions_by_branch_roi={
            "tonic": tonic.sessions_by_roi,
            "phasic": phasic.sessions_by_roi,
        },
        roi_inventory=phasic.roi_inventory,
        normalized_recording=phasic.normalized_recording,
    )


def _is_final_validation_failed_review_candidate(
    resolved: Path,
    *,
    status: dict[str, Any],
    completion: dict[str, Any],
    status_completion: dict[str, Any],
    report_completion: dict[str, Any],
) -> bool:
    """Cheap, HDF5-free screen for whether a failed run is even worth running
    the heavier read-only reviewability verification on (see
    `run_completion_contract.review_with_warnings_eligibility`).

    Passing this gate is only permission to *attempt* that heavier
    verification -- it is not itself permission to open Review. Every other
    failed run (interrupted, cancelled, missing records, mismatched identity,
    or a terminal failure that was never TERMINAL_VALIDATION_FAILED at all) is
    rejected here with no further work and no HDF5 access.
    """
    if status.get("phase") != "final" or status.get("status") != "error":
        return False
    errors = status.get("errors")
    if not isinstance(errors, list) or not any(
        str(entry).startswith("TERMINAL_VALIDATION_FAILED:") for entry in errors
    ):
        return False
    if (
        completion.get("completion_contract_version") != "run_completion.v1"
        or report_completion.get("contract_version") != "run_completion.v1"
    ):
        return False
    if completion.get("final") is not True:
        return False

    run_id = str(status.get("run_id", ""))
    manifest_run_id = str(completion.get("run_id", ""))
    report_run_id = str(report_completion.get("run_id", ""))
    if not run_id or run_id != manifest_run_id or run_id != report_run_id:
        return False

    # A failed run's status.json is written by `_finalize_status("error", ...)`
    # without a `completion` block at all (only the success path stamps one),
    # so there is normally nothing here to compare against. If some future
    # status shape does record a manifest digest, it must still agree.
    recorded_digest = (
        status_completion.get("manifest_sha256")
        if isinstance(status_completion, dict)
        else None
    )
    if recorded_digest:
        try:
            manifest_digest = hashlib.sha256(
                (resolved / "MANIFEST.json").read_bytes()
            ).hexdigest()
        except OSError:
            return False
        if recorded_digest != manifest_digest:
            return False

    run_mode = completion.get("run_mode", {})
    if not isinstance(run_mode, dict):
        return False
    enabled = [
        branch
        for branch, key in (("tonic", "tonic_analysis"), ("phasic", "phasic_analysis"))
        if run_mode.get(key)
    ]
    if not enabled:
        return False
    for branch in enabled:
        branch_dir = resolved / "_analysis" / f"{branch}_out"
        required = (
            branch_dir / "run_metadata.json",
            branch_dir / "run_report.json",
            branch_dir / f"{branch}_trace_cache.h5",
            branch_dir / "input_processing_completeness.json",
        )
        if any(not path.is_file() for path in required):
            return False

    deliverables = completion.get("deliverables")
    artifacts = completion.get("artifacts")
    if not isinstance(deliverables, dict) or not isinstance(artifacts, list):
        return False
    required_paths = deliverables.get("required")
    if not isinstance(required_paths, list):
        return False
    listed = {
        str(entry.get("relative_path", ""))
        for entry in artifacts
        if isinstance(entry, dict)
    }
    if any(str(rel_path) not in listed for rel_path in required_paths):
        return False

    for path in resolved.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".tmp", ".lock"):
            return False

    return True


TONIC_OUTPUT_MODE_SUMMARY_LABELS = {
    "preserve_raw_session_shape": "Preserved",
    "flatten_session_bleach_preserve_session_baseline": (
        "Within-session bleaching trend removed"
    ),
}
TONIC_TIMELINE_MODE_SUMMARY_LABELS = {
    "real_elapsed_time": "Real elapsed time",
    "gap_free_elapsed_time": "Gap-free elapsed time",
}


def format_tonic_settings_summary(tonic_settings: Mapping[str, str]) -> str:
    """Scientist-facing two-line summary of the tonic settings a run
    consumed. "" when no tonic settings evidence is available (e.g. a
    phasic-only run, or the consumed configuration could not be read)."""
    output_mode = tonic_settings.get("tonic_output_mode") if tonic_settings else None
    timeline_mode = (
        tonic_settings.get("tonic_timeline_mode") if tonic_settings else None
    )
    if not output_mode or not timeline_mode:
        return ""
    timeline_label = TONIC_TIMELINE_MODE_SUMMARY_LABELS.get(
        timeline_mode, str(timeline_mode)
    )
    shape_label = TONIC_OUTPUT_MODE_SUMMARY_LABELS.get(output_mode, str(output_mode))
    return f"Tonic timeline: {timeline_label}\nSession shape: {shape_label}"


def _format_duration_natural(duration_sec: float) -> str:
    """Render a persisted expected-duration value as scientist-facing text,
    in whichever unit is exact for that value -- never a hardcoded figure."""
    if duration_sec > 0 and duration_sec % 60 == 0:
        return f"{int(round(duration_sec / 60))}-minute"
    if duration_sec > 0 and duration_sec == int(duration_sec):
        return f"{int(duration_sec)}-second"
    return f"{duration_sec:.1f}-second"


def _duration_validation_warning_message(
    elapsed_time_warnings: list[dict[str, Any]], session_count: int
) -> str:
    """Build warning text only from facts actually present in the collected
    evidence -- no claim about ordering, trend, or a specific duration
    unless every affected session genuinely shares that duration."""
    plural = "session" if session_count == 1 else "sessions"
    verb = "was" if session_count == 1 else "were"
    expected_durations = sorted(
        {
            float(entry["expected_duration_sec"])
            for entry in elapsed_time_warnings
            if "expected_duration_sec" in entry
        }
    )
    if len(expected_durations) == 1:
        shorter_phrase = (
            "shorter than the expected "
            f"{_format_duration_natural(expected_durations[0])} length"
        )
    else:
        shorter_phrase = "shorter than expected"
    return (
        "Your plots and tables were generated and are available below. Some "
        f"recording sessions were {shorter_phrase}. {session_count} {plural} "
        f"{verb} affected. Review those sessions before relying on the results."
    )


def load_completed_review_overview(run_dir: str | Path) -> dict[str, Any]:
    """Load a compact Guided Review orientation without trace datasets.

    This deliberately reads only terminal/analysis metadata plus the persisted
    normalized recording and feature configuration. It never opens an HDF5
    cache or materializes a full-resolution trace.
    """
    resolved = Path(run_dir).expanduser().resolve()
    status = _read_json(resolved / "status.json")
    manifest_path = resolved / "MANIFEST.json"
    manifest = _read_json(manifest_path)
    top_report = _read_json(resolved / "run_report.json")
    completion = manifest.get("completion", {})
    status_completion = status.get("completion", {})
    report_completion = top_report.get("completion_contract", {})
    declares_current_contract = bool(
        status_completion.get("completion_contract_version")
        or completion.get("completion_contract_version")
        or report_completion.get("contract_version")
    )
    if not declares_current_contract:
        classification = classify_run_terminal_state(str(resolved))
        if not classification.is_success or not classification.is_legacy:
            raise CompletedRunReviewError(
                "This completed result could not be verified as a successful "
                f"legacy run ({classification.reason})."
            )
        enabled = [
            branch
            for branch in ("tonic", "phasic")
            if (
                resolved
                / "_analysis"
                / f"{branch}_out"
                / f"{branch}_trace_cache.h5"
            ).is_file()
        ]
        included_rois = [
            path.name
            for path in sorted(resolved.iterdir(), key=lambda item: item.name)
            if path.is_dir()
            and path.name != "_analysis"
            and any(
                (path / child).is_dir()
                for child in ("summary", "tables", "day_plots")
            )
        ]
        if not enabled or not included_rois:
            raise CompletedRunReviewError(
                "This legacy completed result has no reviewable analysis "
                "branch or ROI deliverables."
            )
        return {
            "run_dir": str(resolved),
            "terminal_state": classification.state,
            "format": "",
            "acquisition_mode": "",
            "analysis_branches": enabled,
            "included_rois": included_rois,
            "excluded_rois": [],
            "session_counts": {
                "total": 0,
                "processed": 0,
                "missing": 0,
                "excluded": 0,
            },
            "requested_by_roi": {},
            "feature_settings_by_roi": {},
            "full_resolution_traces_loaded": False,
        }
    run_id = str(status.get("run_id", ""))
    manifest_run_id = str(completion.get("run_id", ""))
    report_run_id = str(report_completion.get("run_id", ""))
    try:
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CompletedRunReviewError(
            f"Completed result manifest is unreadable ({exc})."
        ) from exc
    compact_terminal_ok = bool(
        status.get("phase") == "final"
        and status.get("status") == "success"
        and completion.get("final") is True
        and status_completion.get("completion_contract_version")
        == "run_completion.v1"
        and completion.get("completion_contract_version") == "run_completion.v1"
        and report_completion.get("contract_version") == "run_completion.v1"
        and run_id
        and run_id == manifest_run_id == report_run_id
        and status_completion.get("manifest_sha256") == manifest_digest
    )
    review_status = "success"
    affected_session_indices: list[int] = []
    duration_warnings: list[dict[str, Any]] = []
    if not compact_terminal_ok:
        candidate = _is_final_validation_failed_review_candidate(
            resolved,
            status=status,
            completion=completion,
            status_completion=status_completion,
            report_completion=report_completion,
        )
        fatal_error = "warning-review candidate gate failed"
        elapsed_time_warnings: list[dict[str, Any]] = []
        if candidate:
            candidate_run_mode = completion.get("run_mode", {})
            if isinstance(candidate_run_mode, dict):
                fatal_error, elapsed_time_warnings = review_with_warnings_eligibility(
                    str(resolved), run_id=run_id, run_mode=candidate_run_mode
                )
        # A run reaches here only because status.json recorded status=="error".
        # Every currently-passing terminal check would make the run structurally
        # eligible for success, so if current-code verification finds neither a
        # fatal error nor any collected elapsed-time warning, the persisted
        # error record disagrees with everything checkable today. That is not
        # a case product policy covers (see: never convert a failed run to
        # success), so it stays rejected exactly like today.
        if fatal_error or not elapsed_time_warnings:
            raise CompletedRunReviewError(
                "This completed result has incomplete or inconsistent completion metadata."
            )
        review_status = "reviewable_with_warning"
        duration_warnings = elapsed_time_warnings
        affected_session_indices = sorted(
            {int(entry["session_index"]) for entry in elapsed_time_warnings}
        )
    run_mode = completion.get("run_mode", {})
    if not isinstance(run_mode, dict):
        raise CompletedRunReviewError(
            "This completed result has unreadable run-mode metadata."
        )
    enabled = tuple(
        branch
        for branch, key in (
            ("tonic", "tonic_analysis"),
            ("phasic", "phasic_analysis"),
        )
        if run_mode.get(key)
    )
    if not enabled:
        raise CompletedRunReviewError(
            "This completed result has no reviewable analysis branch."
        )
    analysis_kind = "phasic" if "phasic" in enabled else "tonic"
    analysis_dir = resolved / "_analysis" / f"{analysis_kind}_out"
    for branch in enabled:
        branch_dir = resolved / "_analysis" / f"{branch}_out"
        required = (
            branch_dir / "run_metadata.json",
            branch_dir / "run_report.json",
            branch_dir / f"{branch}_trace_cache.h5",
        )
        if any(not path.is_file() for path in required):
            raise CompletedRunReviewError(
                f"The completed {branch} result is missing persisted Review evidence."
            )
    metadata = _read_json(analysis_dir / "run_metadata.json")
    report = _read_json(analysis_dir / "run_report.json")
    requested_by_roi, native = _validate_requested_provenance(
        metadata, report, analysis_kind
    )
    if not native:
        raise CompletedRunReviewError(
            "The completed result has no verified correction summary."
        )

    normalized_path = resolved / GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME
    normalized = None
    if normalized_path.is_file():
        try:
            normalized = deserialize_normalized_recording_description(
                json.loads(normalized_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, NormalizedRecordingError) as exc:
            raise CompletedRunReviewError(
                f"Completed result recording summary is unreadable ({exc})."
            ) from exc
    feature_settings = (
        _load_feature_settings(resolved / "_analysis" / "phasic_out")
        if "phasic" in enabled
        else {}
    )
    included_rois = list(requested_by_roi)
    excluded_rois: list[str] = []
    session_counts = {"total": 0, "processed": 0, "missing": 0, "excluded": 0}
    adapter_format = ""
    acquisition_mode = ""
    if normalized is not None:
        adapter_format = str(normalized.adapter_format)
        acquisition_mode = str(normalized.acquisition_mode)
        included_rois = [
            item.roi_id for item in normalized.roi_channels if item.included
        ]
        excluded_rois = [
            item.roi_id for item in normalized.roi_channels if not item.included
        ]
        session_counts["total"] = len(normalized.sessions)
        for session in normalized.sessions:
            disposition = str(session.disposition)
            count_key = {
                "process": "processed",
                "missing": "missing",
                "excluded": "excluded",
            }.get(disposition)
            if count_key:
                session_counts[count_key] += 1

    # Consumed evidence only -- never the GUI's current selection -- and only
    # when the tonic branch actually ran and its outputs are on disk. Present
    # for both a "success" and a "reviewable_with_warning" run, since a
    # duration warning does not affect the tonic settings that were used.
    tonic_settings: dict[str, str] = {}
    if "tonic" in enabled:
        tonic_config_used_path = (
            resolved / "_analysis" / "tonic_out" / "config_used.yaml"
        )
        try:
            tonic_config_used = yaml.safe_load(
                tonic_config_used_path.read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError):
            tonic_config_used = None
        if isinstance(tonic_config_used, dict):
            output_mode = tonic_config_used.get("tonic_output_mode")
            timeline_mode = tonic_config_used.get("tonic_timeline_mode")
            if isinstance(output_mode, str) and isinstance(timeline_mode, str):
                tonic_settings = {
                    "tonic_output_mode": output_mode,
                    "tonic_timeline_mode": timeline_mode,
                }

    overview = {
        "run_dir": str(resolved),
        "terminal_state": "success" if review_status == "success" else "failed",
        "review_status": review_status,
        "format": adapter_format,
        "acquisition_mode": acquisition_mode,
        "analysis_branches": list(enabled),
        "included_rois": list(included_rois),
        "excluded_rois": list(excluded_rois),
        "session_counts": dict(session_counts),
        "requested_by_roi": {
            str(roi): dict(record) for roi, record in requested_by_roi.items()
        },
        "feature_settings_by_roi": {
            str(roi): dict(record) for roi, record in feature_settings.items()
        },
        "tonic_settings": tonic_settings,
        "full_resolution_traces_loaded": False,
    }
    if review_status == "reviewable_with_warning":
        session_count = len(affected_session_indices)
        overview["validation_warning_title"] = (
            "Analysis completed with a validation warning"
        )
        overview["validation_warning_message"] = _duration_validation_warning_message(
            duration_warnings, session_count
        )
        overview["affected_session_count"] = session_count
        overview["first_affected_session_index"] = (
            affected_session_indices[0] if affected_session_indices else None
        )
        expected_durations = sorted(
            {
                float(entry["expected_duration_sec"])
                for entry in duration_warnings
                if "expected_duration_sec" in entry
            }
        )
        if len(expected_durations) == 1:
            overview["expected_session_duration_sec"] = expected_durations[0]
        shortfall_magnitudes = [
            -float(entry["duration_difference_sec"])
            for entry in duration_warnings
            if "duration_difference_sec" in entry
        ]
        if shortfall_magnitudes:
            overview["largest_duration_shortfall_sec"] = max(shortfall_magnitudes)
    return overview
