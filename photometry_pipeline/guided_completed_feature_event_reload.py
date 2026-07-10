"""Module for loading and formatting completed-run per-ROI feature-detection settings.

Reads features/feature_event_provenance.json (schema guided_feature_event_provenance.v2,
written by Pipeline only when per-ROI feature-detection settings were supplied for a
Guided Run -- see photometry_pipeline.pipeline.Pipeline._write_feature_event_provenance).
A global-only run writes no such file, so this loader's "not present" state is the
normal case for those runs, not an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FEATURE_EVENT_PROVENANCE_SCHEMA_VERSION = "guided_feature_event_provenance.v2"
# 4J16k39b: current runs record the settings actually consumed for every ROI.
# v2 runs predate that contract and only ever recorded per-ROI entries when a
# Custom ROI existed, so they are read but never treated as a complete record.
FEATURE_EVENT_PROVENANCE_SCHEMA_VERSION_V3 = "guided_feature_event_provenance.v3"
_SUPPORTED_PROVENANCE_SCHEMA_VERSIONS = (
    FEATURE_EVENT_PROVENANCE_SCHEMA_VERSION,
    FEATURE_EVENT_PROVENANCE_SCHEMA_VERSION_V3,
)


@dataclass(frozen=True)
class GuidedCompletedFeatureEventRow:
    roi: str
    source: str  # "default" or "override"
    feature_event_profile_id: str
    effective_config_fields: dict[str, Any]


@dataclass(frozen=True)
class GuidedCompletedFeatureEventState:
    present: bool
    valid: bool = True
    provenance_path: str | None = None
    warning: str | None = None
    rows: tuple[GuidedCompletedFeatureEventRow, ...] = ()

    @classmethod
    def absent(cls) -> "GuidedCompletedFeatureEventState":
        """Create the default absent state: no per-ROI feature file for this run."""
        return cls(present=False, valid=True)

    @property
    def has_custom_roi(self) -> bool:
        return any(row.source == "override" for row in self.rows)


def load_guided_completed_feature_event_state(
    run_dir: str | Path | None,
) -> GuidedCompletedFeatureEventState:
    """Read-only loader for a completed run's per-ROI feature-detection settings."""
    if run_dir is None:
        return GuidedCompletedFeatureEventState.absent()

    run_dir_resolved = Path(run_dir).resolve()
    path = (
        run_dir_resolved
        / "_analysis"
        / "phasic_out"
        / "features"
        / "feature_event_provenance.json"
    )
    if not path.is_file():
        return GuidedCompletedFeatureEventState.absent()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return GuidedCompletedFeatureEventState(
            present=True,
            valid=False,
            provenance_path=str(path),
            warning=(
                "The feature-detection settings recorded for this run could "
                "not be read."
            ),
        )

    if not isinstance(payload, dict):
        return GuidedCompletedFeatureEventState(
            present=True,
            valid=False,
            provenance_path=str(path),
            warning=(
                "The feature-detection settings recorded for this run are "
                "not in the expected format."
            ),
        )

    if payload.get("schema_version") not in _SUPPORTED_PROVENANCE_SCHEMA_VERSIONS:
        return GuidedCompletedFeatureEventState(
            present=True,
            valid=False,
            provenance_path=str(path),
            warning=(
                "The feature-detection settings recorded for this run use a "
                "format this version of the app does not recognize."
            ),
        )

    raw_rows = payload.get("rois")
    if not isinstance(raw_rows, list):
        return GuidedCompletedFeatureEventState(
            present=True,
            valid=False,
            provenance_path=str(path),
            warning=(
                "The feature-detection settings recorded for this run are "
                "missing per-ROI entries."
            ),
        )

    rows: list[GuidedCompletedFeatureEventRow] = []
    for entry in raw_rows:
        if not isinstance(entry, dict):
            continue
        roi = str(entry.get("roi", "")).strip()
        if not roi:
            continue
        source = str(entry.get("source", "")).strip()
        effective = entry.get("effective_config_fields")
        rows.append(
            GuidedCompletedFeatureEventRow(
                roi=roi,
                source=source if source in {"default", "override"} else "default",
                feature_event_profile_id=str(
                    entry.get("feature_event_profile_id", "")
                ),
                effective_config_fields=(
                    dict(effective) if isinstance(effective, dict) else {}
                ),
            )
        )
    rows.sort(key=lambda row: row.roi)

    return GuidedCompletedFeatureEventState(
        present=True,
        valid=True,
        provenance_path=str(path),
        rows=tuple(rows),
    )


def _effective_settings_summary_text(effective_config_fields: dict) -> str:
    """One-line scientist-facing summary, matching the Step 5 per-ROI table's
    own formatting convention (see gui/main_window.py's
    _guided_per_roi_feature_event_summary_text) so pre-run and post-run
    phrasing agree."""
    if not effective_config_fields:
        return "Settings used for this ROI are not available."
    method = str(effective_config_fields.get("peak_threshold_method", "mean_std"))
    if method == "percentile":
        threshold = effective_config_fields.get("peak_threshold_percentile", "")
    elif method == "absolute":
        threshold = effective_config_fields.get("peak_threshold_abs", "")
    else:
        threshold = effective_config_fields.get("peak_threshold_k", "")
    signal = effective_config_fields.get("event_signal", "dff")
    return f"{method} threshold ({threshold}) · {signal} signal"


def format_guided_completed_feature_event_summary(
    state: GuidedCompletedFeatureEventState,
) -> str:
    """Scientist-facing per-ROI feature-detection summary for completed-run Review.

    Empty string means the caller should hide this section entirely -- the
    normal case for a global-only run (no per-ROI feature file was written).
    """
    if not state.present:
        return ""
    if not state.valid:
        return (
            "Feature detection: settings used for this run are available in "
            "technical details below."
        )
    if not state.rows or not state.has_custom_roi:
        return "Feature detection: one default setting set was used for all ROIs."

    lines = ["Feature detection — settings used for this Run:"]
    for row in state.rows:
        label = "Custom" if row.source == "override" else "Default"
        lines.append(
            f"- {row.roi}: {label} — "
            f"{_effective_settings_summary_text(row.effective_config_fields)}"
        )
    return "\n".join(lines)


def format_guided_completed_feature_event_technical_details(
    state: GuidedCompletedFeatureEventState,
) -> str:
    """Full effective-settings breakdown for optional disclosure.

    Always reads effective_config_fields (the complete, resolved settings
    actually used for that ROI), never the sparse override fields a Custom
    profile may have only partially specified.
    """
    if not state.present:
        return "No per-ROI feature-detection settings were recorded for this run."
    if not state.valid:
        return state.warning or (
            "Feature-detection settings could not be displayed for this run."
        )
    if not state.rows:
        return "No per-ROI feature-detection entries were recorded for this run."

    lines = ["Effective settings — these are the settings used during Run:"]
    for row in state.rows:
        label = "Custom" if row.source == "override" else "Default"
        profile_bit = (
            f" (profile: {row.feature_event_profile_id})"
            if row.feature_event_profile_id
            else ""
        )
        lines.append(f"- {row.roi} — {label}{profile_bit}")
        for key in sorted(row.effective_config_fields):
            lines.append(f"    {key}: {row.effective_config_fields[key]}")
    return "\n".join(lines)
