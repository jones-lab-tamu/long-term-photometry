"""Authoritative record of the feature-detection settings actually consumed
per ROI (4J16k39b).

Contract
--------
- ``_analysis/phasic_out/config_used.yaml`` remains the complete confirmed
  GLOBAL Default configuration. It does NOT describe ROIs that used Custom
  settings.
- ``_analysis/phasic_out/features/feature_event_provenance.json`` is the
  authoritative record of the COMPLETE effective feature-detection
  configuration actually consumed for EACH analyzed ROI, including runs with no
  Custom ROI (a Default-only run records one Default entry per ROI).

Every current production run emits this file. Consumers must therefore
distinguish two very different situations:

- A **current** run (identified by an explicit contract-version signal) whose
  provenance file is missing, incomplete, or digest-mismatched must FAIL
  CLOSED. Absence is never evidence of "Default-only".
- A **legacy** run predates this contract and is recognized by the ABSENCE of
  the contract-version signal in its run report -- never merely by the absence
  of the provenance file. Such runs fall back to the global configuration and
  are labeled legacy/unknown; no ROI-specific claim is made for them.

This module is pure: it performs no I/O beyond reading a caller-supplied path,
and never writes.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from photometry_pipeline.feature_event_config import FEATURE_EVENT_CONFIG_FIELDS
from photometry_pipeline.guided_identity import encode_canonical_value


FEATURE_EVENT_PROVENANCE_FILENAME = "feature_event_provenance.json"
FEATURE_EVENT_PROVENANCE_SCHEMA_V2 = "guided_feature_event_provenance.v2"
FEATURE_EVENT_PROVENANCE_SCHEMA_V3 = "guided_feature_event_provenance.v3"
# Explicit signal stamped into run_report.json so a consumer can tell a current
# run (which MUST have provenance) from a legacy run (which never had it).
FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION = "feature_event_provenance.v3"

SOURCE_DEFAULT = "default"
SOURCE_OVERRIDE = "override"

PROVENANCE_MODE_CURRENT = "current"
PROVENANCE_MODE_LEGACY = "legacy"
# The metadata cannot positively establish either contract. Consumers must fail
# closed: a damaged current run must never be silently downgraded to legacy and
# verified against the global configuration.
PROVENANCE_MODE_UNKNOWN = "unknown"


class FeatureEventProvenanceError(RuntimeError):
    """The per-ROI consumed-settings record is missing, incomplete, or does not
    match the settings it claims to describe. Never substitute global settings."""


def feature_fields_from_config(config: Any) -> dict[str, Any]:
    """Extract the complete feature-detection field set from a Config object."""
    return {name: getattr(config, name) for name in sorted(FEATURE_EVENT_CONFIG_FIELDS)}


def compute_feature_config_digest(fields: dict[str, Any]) -> str:
    """Deterministic identity for a complete effective feature configuration.

    Depends only on the feature-detection field names and values. It is
    independent of dictionary insertion order (keys are canonically sorted),
    file paths, timestamps, and every non-feature Config field.
    """
    missing = set(FEATURE_EVENT_CONFIG_FIELDS) - set(fields)
    if missing:
        raise FeatureEventProvenanceError(
            f"Cannot digest an incomplete feature configuration; missing: {sorted(missing)}"
        )
    unknown = set(fields) - set(FEATURE_EVENT_CONFIG_FIELDS)
    if unknown:
        raise FeatureEventProvenanceError(
            f"Cannot digest unknown feature fields: {sorted(unknown)}"
        )
    canonical = encode_canonical_value(
        {name: fields[name] for name in sorted(fields)}
    )
    return hashlib.sha256(b"feature-config:v1\x00" + canonical).hexdigest()


def build_feature_event_provenance_payload(
    *,
    base_config: Any,
    analyzed_rois: list[str] | tuple[str, ...],
    per_roi_feature_config: dict[str, Any] | None = None,
    per_roi_source_details: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the provenance payload from the Config objects ACTUALLY consumed.

    base_config is the confirmed global Default configuration handed to
    Pipeline; per_roi_feature_config maps a Custom ROI to the complete Config
    object Pipeline used for that ROI. Nothing here is reconstructed from the
    Guided plan or the startup artifact, so Default-only and Custom runs follow
    exactly the same evidence path.

    per_roi_source_details optionally supplies the sparse user override and the
    profile id for an ROI; it is descriptive only and never determines the
    effective settings.
    """
    per_roi_feature_config = per_roi_feature_config or {}
    per_roi_source_details = per_roi_source_details or {}

    base_fields = feature_fields_from_config(base_config)
    global_digest = compute_feature_config_digest(base_fields)

    rois: list[dict[str, Any]] = []
    for roi in sorted(analyzed_rois):
        roi_config = per_roi_feature_config.get(roi, base_config)
        effective = feature_fields_from_config(roi_config)
        is_override = roi in per_roi_feature_config
        details = dict(per_roi_source_details.get(roi) or {})
        rois.append(
            {
                "roi": roi,
                "source": SOURCE_OVERRIDE if is_override else SOURCE_DEFAULT,
                "feature_event_profile_id": str(
                    details.get("feature_event_profile_id", "")
                ),
                "override_config_fields": dict(
                    details.get("override_config_fields") or {}
                ),
                "effective_config_fields": effective,
                "effective_config_digest": compute_feature_config_digest(effective),
            }
        )

    return {
        "schema_version": FEATURE_EVENT_PROVENANCE_SCHEMA_V3,
        "feature_event_provenance_contract_version": (
            FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION
        ),
        "global_default_config_fields": base_fields,
        "global_default_config_digest": global_digest,
        "rois": rois,
    }


def load_feature_event_provenance(path: str) -> dict[str, Any]:
    """Read and structurally validate a provenance file. Fails closed."""
    if not os.path.isfile(path):
        raise FeatureEventProvenanceError(
            f"Feature-detection provenance is missing: {path}"
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise FeatureEventProvenanceError(
            f"Feature-detection provenance is malformed: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise FeatureEventProvenanceError(
            f"Feature-detection provenance root is not an object: {path}"
        )
    if payload.get("schema_version") != FEATURE_EVENT_PROVENANCE_SCHEMA_V3:
        raise FeatureEventProvenanceError(
            "Feature-detection provenance uses an unsupported schema "
            f"{payload.get('schema_version')!r}; expected "
            f"{FEATURE_EVENT_PROVENANCE_SCHEMA_V3!r}."
        )
    if not isinstance(payload.get("rois"), list) or not payload["rois"]:
        raise FeatureEventProvenanceError(
            f"Feature-detection provenance records no ROI entries: {path}"
        )
    return payload


def resolve_roi_effective_fields(payload: dict[str, Any], roi: str) -> dict[str, Any]:
    """Return the complete effective feature configuration recorded for one ROI.

    Fails closed when the ROI has no entry, the entry is incomplete, or the
    recorded digest does not match the recorded settings. Never falls back to
    the global configuration.
    """
    entries = [
        entry
        for entry in payload.get("rois", [])
        if isinstance(entry, dict) and str(entry.get("roi", "")) == str(roi)
    ]
    if not entries:
        raise FeatureEventProvenanceError(
            f"Feature-detection provenance has no entry for ROI {roi!r}."
        )
    if len(entries) > 1:
        raise FeatureEventProvenanceError(
            f"Feature-detection provenance has duplicate entries for ROI {roi!r}."
        )
    entry = entries[0]
    fields = entry.get("effective_config_fields")
    if not isinstance(fields, dict):
        raise FeatureEventProvenanceError(
            f"ROI {roi!r} has no effective feature configuration recorded."
        )
    missing = set(FEATURE_EVENT_CONFIG_FIELDS) - set(fields)
    if missing:
        raise FeatureEventProvenanceError(
            f"ROI {roi!r} effective feature configuration is incomplete; "
            f"missing: {sorted(missing)}"
        )
    recorded = str(entry.get("effective_config_digest", ""))
    actual = compute_feature_config_digest(fields)
    if recorded != actual:
        raise FeatureEventProvenanceError(
            f"ROI {roi!r} feature-configuration digest does not match its "
            f"recorded settings (recorded {recorded!r}, computed {actual!r})."
        )
    return dict(fields)


def resolve_roi_entry(payload: dict[str, Any], roi: str) -> dict[str, Any]:
    """Return the validated provenance entry for one ROI (digest checked)."""
    resolve_roi_effective_fields(payload, roi)
    for entry in payload.get("rois", []):
        if isinstance(entry, dict) and str(entry.get("roi", "")) == str(roi):
            return entry
    raise FeatureEventProvenanceError(f"No provenance entry for ROI {roi!r}.")


def run_uses_current_provenance_contract(run_report: dict[str, Any] | None) -> bool:
    """True only for the EXACT supported contract version.

    An absent, empty, malformed, or unknown-future contract version is not a
    valid current contract -- and must not be silently reinterpreted as legacy.
    """
    if not isinstance(run_report, dict):
        return False
    section = run_report.get("feature_event_provenance")
    if not isinstance(section, dict):
        return False
    return section.get("contract_version") == FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION


def _run_report_is_positively_legacy(run_report: dict[str, Any]) -> bool:
    """A run report that is well-formed in the pre-contract shape.

    Legacy is a POSITIVE identification, not an inference from missing data: the
    report must parse, carry the long-standing run-report schema signals
    (`analytical_contract` + `configuration`, written by every historical run),
    and carry no feature-event-provenance section at all. A report that is
    missing, malformed, truncated, or that carries an unrecognized provenance
    section proves nothing and must be classified unknown.
    """
    if not isinstance(run_report, dict):
        return False
    if "feature_event_provenance" in run_report:
        return False
    return isinstance(run_report.get("analytical_contract"), dict) and isinstance(
        run_report.get("configuration"), dict
    )


def classify_provenance_contract(phasic_out: str) -> tuple[str, str | None, str]:
    """Classify an analysis output directory. Returns (mode, provenance_path, reason).

    - PROVENANCE_MODE_CURRENT: the report explicitly declares the exact supported
      contract version. The caller must require a complete, digest-consistent
      provenance file.
    - PROVENANCE_MODE_LEGACY: the report positively identifies a pre-contract run.
      The caller may use the global config_used.yaml and must make no
      ROI-specific verification claim.
    - PROVENANCE_MODE_UNKNOWN: the metadata cannot positively establish either.
      The caller MUST fail closed. A missing or malformed run report, or an
      unsupported contract version, lands here -- never in legacy.
    """
    report_path = os.path.join(phasic_out, "run_report.json")
    if not os.path.isfile(report_path):
        return (
            PROVENANCE_MODE_UNKNOWN,
            None,
            f"run_report.json is missing: {report_path}",
        )
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            run_report = json.load(handle)
    except Exception as exc:
        return (
            PROVENANCE_MODE_UNKNOWN,
            None,
            f"run_report.json is malformed: {exc}",
        )
    if not isinstance(run_report, dict):
        return PROVENANCE_MODE_UNKNOWN, None, "run_report.json root is not an object."

    section = run_report.get("feature_event_provenance")
    if isinstance(section, dict):
        version = section.get("contract_version")
        if version == FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION:
            return (
                PROVENANCE_MODE_CURRENT,
                os.path.join(phasic_out, "features", FEATURE_EVENT_PROVENANCE_FILENAME),
                "",
            )
        return (
            PROVENANCE_MODE_UNKNOWN,
            None,
            f"unsupported feature-provenance contract version {version!r}; "
            f"expected {FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION!r}",
        )
    if "feature_event_provenance" in run_report:
        return (
            PROVENANCE_MODE_UNKNOWN,
            None,
            "feature_event_provenance section is present but not an object.",
        )

    if _run_report_is_positively_legacy(run_report):
        return PROVENANCE_MODE_LEGACY, None, ""

    return (
        PROVENANCE_MODE_UNKNOWN,
        None,
        "run_report.json neither declares the current feature-provenance contract "
        "nor positively identifies a pre-contract run.",
    )


def verify_global_default_identity(payload: dict[str, Any], global_config: Any) -> str:
    """Bind the provenance record to the global configuration it claims to describe.

    Requires that:
      - the recorded global_default_config_fields are complete and hash to the
        recorded global_default_config_digest;
      - the feature fields of the loaded config_used.yaml hash to that same digest;
      - every ROI whose source is Default carries exactly that digest.

    A Custom ROI may legitimately differ, but its own complete effective settings
    must still hash to its own recorded digest (checked by resolve_roi_entry).

    Returns the verified global Default digest. Fails closed on any mismatch.
    """
    recorded_fields = payload.get("global_default_config_fields")
    if not isinstance(recorded_fields, dict):
        raise FeatureEventProvenanceError(
            "Provenance does not record the global Default feature configuration."
        )
    recomputed = compute_feature_config_digest(recorded_fields)
    recorded_digest = str(payload.get("global_default_config_digest", ""))
    if recomputed != recorded_digest:
        raise FeatureEventProvenanceError(
            "Recorded global Default configuration does not match its recorded "
            f"digest (recorded {recorded_digest!r}, computed {recomputed!r})."
        )

    actual = compute_feature_config_digest(feature_fields_from_config(global_config))
    if actual != recorded_digest:
        raise FeatureEventProvenanceError(
            "config_used.yaml does not match the global Default feature "
            "configuration recorded in the provenance "
            f"(config_used digest {actual!r}, recorded {recorded_digest!r})."
        )

    for entry in payload.get("rois", []):
        if not isinstance(entry, dict):
            raise FeatureEventProvenanceError("Provenance ROI entry is not an object.")
        roi = str(entry.get("roi", ""))
        # Validates completeness and the entry's own digest.
        resolve_roi_effective_fields(payload, roi)
        if entry.get("source") == SOURCE_DEFAULT:
            if str(entry.get("effective_config_digest", "")) != recorded_digest:
                raise FeatureEventProvenanceError(
                    f"Default ROI {roi!r} effective settings differ from the global "
                    "Default configuration."
                )

    return recorded_digest
