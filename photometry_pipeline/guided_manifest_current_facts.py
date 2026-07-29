"""Read-only live RWD facts for Guided manifest consumption verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Sequence

from photometry_pipeline.config import Config
from photometry_pipeline.guided_backend_validation_workflow import (
    GUIDED_BACKEND_RWD_SIGNAL_SUFFIX_CANDIDATES,
    GUIDED_BACKEND_RWD_TIME_COLUMN_CANDIDATES,
    GUIDED_BACKEND_RWD_UV_SUFFIX_CANDIDATES,
)
from photometry_pipeline.guided_execution_preflight import (
    compute_guided_strict_roi_inventory_digest,
)
from photometry_pipeline.guided_manifest_verification import (
    GuidedManifestCurrentCandidate,
    GuidedManifestCurrentRoiInventory,
)
from photometry_pipeline.guided_npm_execution_authority import (
    GUIDED_NPM_CANONICAL_ROI_NAMING_RULE,
    GuidedNpmRoiAuthority,
    GuidedNpmRoiMappingEntry,
    compute_guided_npm_roi_authority_identity,
)
from photometry_pipeline.io.npm_contract import NpmParserContract, inspect_npm_csv
from photometry_pipeline.io.npm_source_snapshot import (
    build_npm_source_candidate_snapshot,
)
from photometry_pipeline.io.rwd_contract import (
    RwdHeaderParsingContract,
    compute_rwd_header_parsing_contract_digest,
    inspect_rwd_header_contract,
)
from photometry_pipeline.io.rwd_source_snapshot import (
    build_rwd_source_candidate_snapshot,
)
from photometry_pipeline.io.custom_tabular_source_snapshot import (
    build_custom_tabular_source_candidate_snapshot,
)
from photometry_pipeline.io.adapters import inspect_custom_tabular_header
from photometry_pipeline.guided_normalized_recording import (
    compute_custom_tabular_parser_contract_digest,
)


GUIDED_FIRST_SUBSET_SELECTION_MODE = "include"
GUIDED_MANIFEST_CURRENT_FACTS_SUPPORTED_FORMATS = (
    "rwd",
    "npm",
    "custom_tabular",
)


@dataclass(frozen=True)
class GuidedManifestCurrentFacts:
    current_candidates: tuple[GuidedManifestCurrentCandidate, ...]
    current_roi_inventory: GuidedManifestCurrentRoiInventory


def build_guided_manifest_current_facts(
    *,
    source_root: str | Path,
    config: Config,
    manifest_included_roi_ids: Sequence[str],
    source_format: str,
) -> GuidedManifestCurrentFacts:
    """Build current candidate and ROI facts without writes or allocation.

    Dispatches strictly on ``source_format`` ("rwd" or "npm"); no format is
    inferred from filenames, config contents, or manifest shape.
    """
    if not isinstance(config, Config):
        raise TypeError("config must be a Config.")
    if str(getattr(config, "acquisition_mode", "intermittent")) != "intermittent":
        raise ValueError("Guided manifest execution requires intermittent acquisition.")
    included = tuple(str(item) for item in manifest_included_roi_ids)
    if not included or any(not item.strip() for item in included):
        raise ValueError("Guided manifest execution requires included ROI IDs.")
    if len(included) != len(set(included)):
        raise ValueError("Guided manifest included ROI IDs contain duplicates.")

    root = os.fspath(source_root)
    if source_format == "rwd":
        return _build_rwd_guided_manifest_current_facts(root, config, included)
    if source_format == "npm":
        return _build_npm_guided_manifest_current_facts(root, config, included)
    if source_format == "custom_tabular":
        return _build_custom_tabular_guided_manifest_current_facts(
            root, config, included
        )
    raise ValueError(
        f"Unsupported Guided manifest source format: {source_format!r}."
    )


def _build_rwd_guided_manifest_current_facts(
    root: str,
    config: Config,
    included: tuple[str, ...],
) -> GuidedManifestCurrentFacts:
    snapshot = build_rwd_source_candidate_snapshot(root)
    # Re-verify against the same candidate contract that was already
    # validated and embedded in the manifest's parser_contract_digest, not
    # a narrower reconstruction from the one resolved config value -- a
    # single-value contract can never match a multi-candidate one even
    # when the resolved column is among the candidates (4J16k19).
    parser_contract = RwdHeaderParsingContract(
        time_column_candidates=GUIDED_BACKEND_RWD_TIME_COLUMN_CANDIDATES,
        uv_suffix_candidates=GUIDED_BACKEND_RWD_UV_SUFFIX_CANDIDATES,
        signal_suffix_candidates=GUIDED_BACKEND_RWD_SIGNAL_SUFFIX_CANDIDATES,
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


def _build_npm_guided_manifest_current_facts(
    root: str,
    config: Config,
    included: tuple[str, ...],
) -> GuidedManifestCurrentFacts:
    snapshot = build_npm_source_candidate_snapshot(root)
    # Re-derive the parser policy from the same Config fields the accepted
    # NPM authority's recording policy was itself built from (see
    # guided_npm_startup_bridge._npm_config_payload_values), not a narrower
    # reconstruction -- NPM's parser policy is config-driven, unlike RWD's
    # fixed candidate-suffix list.
    parser_contract = NpmParserContract.from_config(config)
    parser_digest = parser_contract.digest

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
    physical_to_canonical: tuple[tuple[str, str], ...] | None = None
    for candidate in candidates:
        inspection = inspect_npm_csv(candidate.absolute_path, parser_contract)
        current = tuple(inspection.roi_ids)
        current_mapping = tuple(inspection.physical_to_canonical_roi_mapping)
        if discovered is None:
            discovered = current
            physical_to_canonical = current_mapping
        elif current != discovered or current_mapping != physical_to_canonical:
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

    # Mirror guided_npm_execution_authority._build_roi_authority's exact
    # field derivation, fed with live-discovered values instead of the
    # already-normalized session values it uses at authorization time, so
    # the resulting identity is comparable to the one already embedded in
    # the manifest by guided_npm_startup_bridge.compile_npm_generic_execution_payloads.
    included_set = set(included)
    roi_mapping = tuple(
        GuidedNpmRoiMappingEntry(
            physical_source_column=physical_column,
            canonical_roi_id=canonical_roi_id,
        )
        for canonical_roi_id, physical_column in physical_to_canonical
    )
    selected_map = tuple(
        item for item in roi_mapping if item.canonical_roi_id in included_set
    )
    roi_authority = GuidedNpmRoiAuthority(
        complete_canonical_roi_ids=discovered,
        selected_canonical_roi_ids=included,
        excluded_canonical_roi_ids=excluded,
        complete_physical_source_columns=tuple(
            item.physical_source_column for item in roi_mapping
        ),
        physical_to_canonical_roi_mapping=roi_mapping,
        selected_physical_source_columns=tuple(
            item.physical_source_column for item in selected_map
        ),
        selected_physical_to_canonical_roi_mapping=selected_map,
        roi_ordering_rule=str(parser_contract.content()["sampling"]["roi_order_policy"]),
        canonical_roi_naming_rule=GUIDED_NPM_CANONICAL_ROI_NAMING_RULE,
        canonical_roi_authority_identity="0" * 64,
    )
    roi_authority = replace(
        roi_authority,
        canonical_roi_authority_identity=compute_guided_npm_roi_authority_identity(
            roi_authority
        ),
    )

    return GuidedManifestCurrentFacts(
        current_candidates=candidates,
        current_roi_inventory=GuidedManifestCurrentRoiInventory(
            discovered_roi_ids=discovered,
            included_roi_ids=included,
            excluded_roi_ids=excluded,
            parser_contract_digest=parser_digest,
            strict_roi_inventory_digest=roi_authority.canonical_roi_authority_identity,
        ),
    )


def _build_custom_tabular_guided_manifest_current_facts(
    root: str,
    config: Config,
    included: tuple[str, ...],
) -> GuidedManifestCurrentFacts:
    snapshot = build_custom_tabular_source_candidate_snapshot(root)
    mappings = json.loads(config.custom_tabular_roi_mapping_json)
    interpretation = {
        "time_column": config.custom_tabular_time_col,
        "time_unit": config.custom_tabular_time_unit,
        "time_scale_to_seconds": (
            1.0 if config.custom_tabular_time_unit == "seconds" else 0.001
        ),
        "header_rule": "ordinary_first_row",
        "delimiter": "comma",
        "roi_mappings": mappings,
        "chronology_authority": "confirmed_filename_order",
    }
    parser_digest = compute_custom_tabular_parser_contract_digest(interpretation)
    discovered = tuple(str(item["roi_id"]) for item in mappings)
    required = [
        config.custom_tabular_time_col,
        *[
            item[key]
            for item in mappings
            for key in ("signal_column", "reference_column")
        ],
    ]
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
    for candidate in candidates:
        headers = inspect_custom_tabular_header(candidate.absolute_path)
        missing = [column for column in required if column not in headers]
        if missing:
            raise ValueError(
                f"The selected column {missing[0]!r} is missing from "
                f"{candidate.canonical_relative_path}."
            )
    missing_rois = tuple(item for item in included if item not in discovered)
    if missing_rois:
        raise ValueError(
            f"Included ROIs are absent from the live source: {list(missing_rois)}"
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
