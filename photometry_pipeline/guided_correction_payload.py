"""Versioned native Guided per-ROI correction execution payload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from photometry_pipeline.core.types import PerRoiCorrectionSpec
from photometry_pipeline.guided_production_mapping import (
    GuidedProductionPerRoiStrategy,
    guided_production_strategy_map_to_correction_specs,
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
    }


def _semantic_basis(included_roi_ids: tuple[str, ...], specs: dict[str, PerRoiCorrectionSpec]) -> dict[str, object]:
    return {
        "schema_name": GUIDED_PER_ROI_CORRECTION_SCHEMA_NAME,
        "schema_version": GUIDED_PER_ROI_CORRECTION_SCHEMA_VERSION,
        "included_roi_ids": sorted(included_roi_ids),
        "per_roi_correction": [_entry(specs[roi]) for roi in sorted(specs)],
    }


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
    required = {"roi_id", "strategy_family", "selected_strategy", "dynamic_fit_mode", "parameter_identity", "evidence_identity"}
    try:
        for raw in raw_entries:
            if not isinstance(raw, dict) or set(raw) != required:
                raise GuidedCorrectionPayloadError("Guided correction entry is malformed.")
            if not isinstance(raw["parameter_identity"], str) or not isinstance(raw["evidence_identity"], str):
                raise GuidedCorrectionPayloadError("Guided correction identities must be strings.")
            spec = PerRoiCorrectionSpec(**raw)
            if spec.roi_id in specs:
                raise GuidedCorrectionPayloadError("Guided correction map contains a duplicate ROI.")
            specs[spec.roi_id] = spec
    except (TypeError, ValueError) as exc:
        raise GuidedCorrectionPayloadError(str(exc)) from exc
    if set(specs) != set(included):
        raise GuidedCorrectionPayloadError("Guided correction coverage does not exactly match included ROIs.")
    identity = correction_payload_identity(tuple(included), specs)
    if document.get("canonical_correction_payload_identity") != identity:
        raise GuidedCorrectionPayloadError("Guided correction payload identity mismatch.")
    return specs
