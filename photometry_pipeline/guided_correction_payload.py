"""Versioned native Guided per-ROI correction execution payload."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

from photometry_pipeline.core.types import PerRoiCorrectionSpec
from photometry_pipeline.guided_production_mapping import (
    GuidedProductionPerRoiStrategy,
    guided_production_strategy_map_to_correction_specs,
)
from photometry_pipeline.guided_new_analysis_plan import (
    validate_guided_effective_correction_parameters,
)


GUIDED_PER_ROI_CORRECTION_FILENAME = "guided_per_roi_correction.json"
GUIDED_PER_ROI_CORRECTION_SCHEMA_NAME = "guided_per_roi_correction"
GUIDED_PER_ROI_CORRECTION_SCHEMA_VERSION = "v1"


class GuidedCorrectionPayloadError(RuntimeError):
    pass


def _entry(spec: PerRoiCorrectionSpec) -> dict[str, object]:
    return {
        "roi_id": spec.roi_id,
        "strategy_family": spec.strategy_family,
        "selected_strategy": spec.selected_strategy,
        "dynamic_fit_mode": spec.dynamic_fit_mode,
        "parameter_identity": spec.parameter_identity,
        "evidence_identity": spec.evidence_identity,
        "effective_parameters": {
            name: value for name, value in spec.effective_parameters
        },
    }


def _semantic_basis(included_roi_ids: tuple[str, ...], specs: dict[str, PerRoiCorrectionSpec]) -> dict[str, object]:
    return {
        "schema_name": GUIDED_PER_ROI_CORRECTION_SCHEMA_NAME,
        "schema_version": GUIDED_PER_ROI_CORRECTION_SCHEMA_VERSION,
        "included_roi_ids": sorted(included_roi_ids),
        "per_roi_correction": [_entry(specs[roi]) for roi in sorted(specs)],
    }


def _legacy_entry(spec: PerRoiCorrectionSpec) -> dict[str, object]:
    return {
        "roi_id": spec.roi_id,
        "strategy_family": spec.strategy_family,
        "selected_strategy": spec.selected_strategy,
        "dynamic_fit_mode": spec.dynamic_fit_mode,
        "parameter_identity": spec.parameter_identity,
        "evidence_identity": spec.evidence_identity,
    }


def _legacy_correction_payload_identity(
    included_roi_ids: tuple[str, ...],
    specs: dict[str, PerRoiCorrectionSpec],
) -> str:
    basis = {
        "schema_name": GUIDED_PER_ROI_CORRECTION_SCHEMA_NAME,
        "schema_version": GUIDED_PER_ROI_CORRECTION_SCHEMA_VERSION,
        "included_roi_ids": sorted(included_roi_ids),
        "per_roi_correction": [
            _legacy_entry(specs[roi]) for roi in sorted(specs)
        ],
    }
    encoded = json.dumps(
        basis, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def correction_payload_identity(included_roi_ids: tuple[str, ...], specs: dict[str, PerRoiCorrectionSpec]) -> str:
    basis = _semantic_basis(included_roi_ids, specs)
    encoded = json.dumps(basis, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def serialize_guided_correction_payload(
    included_roi_ids: Iterable[str],
    entries: tuple[GuidedProductionPerRoiStrategy, ...],
) -> bytes:
    included = tuple(included_roi_ids)
    if len(included) != len(set(included)):
        raise GuidedCorrectionPayloadError("Included ROI identities contain duplicates.")
    try:
        specs = guided_production_strategy_map_to_correction_specs(entries)
    except (TypeError, ValueError) as exc:
        raise GuidedCorrectionPayloadError(str(exc)) from exc
    if set(specs) != set(included):
        raise GuidedCorrectionPayloadError("Per-ROI correction coverage does not exactly match included ROIs.")
    basis = _semantic_basis(included, specs)
    document = {**basis, "canonical_correction_payload_identity": correction_payload_identity(included, specs)}
    return (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def load_guided_correction_payload(path: str | Path, expected_roi_ids: Iterable[str]) -> dict[str, PerRoiCorrectionSpec]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise GuidedCorrectionPayloadError("Guided correction payload is missing or malformed.") from exc
    if not isinstance(document, dict):
        raise GuidedCorrectionPayloadError("Guided correction payload must be an object.")
    if document.get("schema_name") != GUIDED_PER_ROI_CORRECTION_SCHEMA_NAME or document.get("schema_version") != GUIDED_PER_ROI_CORRECTION_SCHEMA_VERSION:
        raise GuidedCorrectionPayloadError("Guided correction payload schema is unsupported.")
    included = document.get("included_roi_ids")
    raw_entries = document.get("per_roi_correction")
    if not isinstance(included, list) or not all(isinstance(x, str) and x for x in included) or len(included) != len(set(included)):
        raise GuidedCorrectionPayloadError("Guided correction included ROI set is malformed.")
    expected = tuple(expected_roi_ids)
    if len(expected) != len(set(expected)) or set(included) != set(expected):
        raise GuidedCorrectionPayloadError("Guided correction included ROI set is stale or incomplete.")
    if not isinstance(raw_entries, list):
        raise GuidedCorrectionPayloadError("Guided correction map is malformed.")
    specs: dict[str, PerRoiCorrectionSpec] = {}
    legacy_specs: dict[str, PerRoiCorrectionSpec] = {}
    # Older Guided payloads serialized these names even though they were not
    # part of the current Guided editable contract.  Keep this compatibility
    # path local to reopening an existing payload; current overrides and new
    # payloads continue to use the current field registry.
    legacy_effective_parameter_names = {
        "robust_event_reject_local_var_window_sec",
        "adaptive_event_gate_local_var_window_sec",
    }
    required = {
        "roi_id",
        "strategy_family",
        "selected_strategy",
        "dynamic_fit_mode",
        "parameter_identity",
        "evidence_identity",
    }
    allowed = required | {"effective_parameters"}
    try:
        for raw in raw_entries:
            if (
                not isinstance(raw, dict)
                or not required.issubset(set(raw))
                or not set(raw).issubset(allowed)
            ):
                raise GuidedCorrectionPayloadError("Guided correction entry is malformed.")
            if not isinstance(raw["parameter_identity"], str) or not isinstance(raw["evidence_identity"], str):
                raise GuidedCorrectionPayloadError("Guided correction identities must be strings.")
            has_effective_parameters = "effective_parameters" in raw
            raw_parameters = raw.get("effective_parameters", {})
            if has_effective_parameters and not isinstance(raw_parameters, Mapping):
                raise GuidedCorrectionPayloadError(
                    "Guided correction effective_parameters must be an object."
                )
            if isinstance(raw_parameters, Mapping) and any(
                not isinstance(name, str) for name in raw_parameters
            ):
                raise GuidedCorrectionPayloadError(
                    "Guided correction parameter names must be strings."
                )
            raw_effective_parameters = (
                tuple(raw_parameters.items())
                if isinstance(raw_parameters, Mapping)
                else ()
            )
            effective_parameters = tuple(
                (name, value)
                for name, value in raw_effective_parameters
                if name not in legacy_effective_parameter_names
            )
            strategy = str(raw.get("selected_strategy") or "")
            if has_effective_parameters:
                try:
                    effective_parameters = validate_guided_effective_correction_parameters(
                        strategy,
                        effective_parameters,
                    )
                except (TypeError, ValueError) as exc:
                    raise GuidedCorrectionPayloadError(str(exc)) from exc
            spec = PerRoiCorrectionSpec(
                roi_id=raw["roi_id"],
                strategy_family=raw["strategy_family"],
                selected_strategy=raw["selected_strategy"],
                dynamic_fit_mode=raw["dynamic_fit_mode"],
                parameter_identity=raw["parameter_identity"],
                evidence_identity=raw["evidence_identity"],
                effective_parameters=effective_parameters,
            )
            if spec.roi_id in specs:
                raise GuidedCorrectionPayloadError("Guided correction map contains a duplicate ROI.")
            specs[spec.roi_id] = spec
            if legacy_effective_parameter_names.intersection(
                name for name, _value in raw_effective_parameters
            ):
                legacy_specs[spec.roi_id] = PerRoiCorrectionSpec(
                    roi_id=raw["roi_id"],
                    strategy_family=raw["strategy_family"],
                    selected_strategy=raw["selected_strategy"],
                    dynamic_fit_mode=raw["dynamic_fit_mode"],
                    parameter_identity=raw["parameter_identity"],
                    evidence_identity=raw["evidence_identity"],
                    effective_parameters=raw_effective_parameters,
                )
    except (TypeError, ValueError) as exc:
        raise GuidedCorrectionPayloadError(str(exc)) from exc
    if set(specs) != set(included):
        raise GuidedCorrectionPayloadError("Guided correction coverage does not exactly match included ROIs.")
    identity = correction_payload_identity(tuple(included), specs)
    if document.get("canonical_correction_payload_identity") != identity:
        legacy_identity_specs = dict(specs)
        legacy_identity_specs.update(legacy_specs)
        legacy_parameter_identity_matches = bool(legacy_specs) and document.get(
            "canonical_correction_payload_identity"
        ) == correction_payload_identity(
            tuple(included), legacy_identity_specs
        )
        legacy_entries = all(
            isinstance(raw, dict) and "effective_parameters" not in raw
            for raw in raw_entries
        )
        if not legacy_parameter_identity_matches and (
            not legacy_entries
            or document.get("canonical_correction_payload_identity")
            != _legacy_correction_payload_identity(tuple(included), specs)
        ):
            raise GuidedCorrectionPayloadError(
                "Guided correction payload identity mismatch."
            )
    return specs
