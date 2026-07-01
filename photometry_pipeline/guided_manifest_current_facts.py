"""Read-only live RWD facts for Guided manifest consumption verification."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

from photometry_pipeline.config import Config
from photometry_pipeline.guided_execution_preflight import (
    compute_guided_strict_roi_inventory_digest,
)
from photometry_pipeline.guided_manifest_verification import (
    GuidedManifestCurrentCandidate,
    GuidedManifestCurrentRoiInventory,
)
from photometry_pipeline.io.rwd_contract import (
    RwdHeaderParsingContract,
    compute_rwd_header_parsing_contract_digest,
    inspect_rwd_header_contract,
)
from photometry_pipeline.io.rwd_source_snapshot import (
    build_rwd_source_candidate_snapshot,
)


GUIDED_FIRST_SUBSET_SELECTION_MODE = "include"


@dataclass(frozen=True)
class GuidedManifestCurrentFacts:
    current_candidates: tuple[GuidedManifestCurrentCandidate, ...]
    current_roi_inventory: GuidedManifestCurrentRoiInventory


def build_guided_manifest_current_facts(
    *,
    source_root: str | Path,
    config: Config,
    manifest_included_roi_ids: Sequence[str],
) -> GuidedManifestCurrentFacts:
    """Build current candidate and ROI facts without writes or allocation."""
    if not isinstance(config, Config):
        raise TypeError("config must be a Config.")
    if str(getattr(config, "acquisition_mode", "intermittent")) != "intermittent":
        raise ValueError("Guided manifest execution requires intermittent acquisition.")
    if bool(getattr(config, "exclude_incomplete_final_rwd_chunk", False)):
        raise ValueError(
            "Guided manifest execution does not support incomplete-final exclusion."
        )
    included = tuple(str(item) for item in manifest_included_roi_ids)
    if not included or any(not item.strip() for item in included):
        raise ValueError("Guided manifest execution requires included ROI IDs.")
    if len(included) != len(set(included)):
        raise ValueError("Guided manifest included ROI IDs contain duplicates.")

    root = os.fspath(source_root)
    snapshot = build_rwd_source_candidate_snapshot(root)
    parser_contract = RwdHeaderParsingContract(
        time_column_candidates=(str(config.rwd_time_col),),
        uv_suffix_candidates=(str(config.uv_suffix),),
        signal_suffix_candidates=(str(config.sig_suffix),),
    )
    parser_digest = compute_rwd_header_parsing_contract_digest(parser_contract)

    candidates = tuple(
        GuidedManifestCurrentCandidate(
            canonical_relative_path=item.canonical_relative_path,
            absolute_path=os.path.abspath(
                os.path.join(
                    snapshot.source_root_canonical,
                    *item.canonical_relative_path.split("/"),
                )
            ),
        )
        for item in snapshot.candidates
    )

    discovered: tuple[str, ...] | None = None
    for candidate in candidates:
        inspection = inspect_rwd_header_contract(
            candidate.absolute_path,
            parsing_contract=parser_contract,
        )
        if not inspection.acceptable_for_strict_identity:
            raise ValueError(
                "RWD header is not acceptable for strict Guided ROI identity."
            )
        current = tuple(inspection.roi_ids)
        if discovered is None:
            discovered = current
        elif current != discovered:
            raise ValueError(
                "Guided candidate files do not share one ordered ROI inventory."
            )
    if not discovered:
        raise ValueError("No live ROI inventory was discovered.")

    missing = tuple(item for item in included if item not in discovered)
    if missing:
        raise ValueError(
            f"Included ROIs are absent from the live source: {list(missing)}"
        )
    excluded = tuple(item for item in discovered if item not in included)
    strict_digest = compute_guided_strict_roi_inventory_digest(
        source_candidate_content_digest=snapshot.source_candidate_content_digest,
        parser_contract_digest=parser_digest,
        discovered_roi_ids=discovered,
        included_roi_ids=included,
        excluded_roi_ids=excluded,
        selection_mode=GUIDED_FIRST_SUBSET_SELECTION_MODE,
    )
    return GuidedManifestCurrentFacts(
        current_candidates=candidates,
        current_roi_inventory=GuidedManifestCurrentRoiInventory(
            discovered_roi_ids=discovered,
            included_roi_ids=included,
            excluded_roi_ids=excluded,
            parser_contract_digest=parser_digest,
            strict_roi_inventory_digest=strict_digest,
        ),
    )
