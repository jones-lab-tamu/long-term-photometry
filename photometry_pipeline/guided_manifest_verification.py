import hashlib
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Sequence

from photometry_pipeline.guided_identity import encode_canonical_value, canonicalize_absolute_path
from photometry_pipeline.io.npm_source_snapshot import (
    build_npm_source_candidate_snapshot,
)
from photometry_pipeline.io.rwd_source_snapshot import (
    compute_rwd_source_candidate_set_digest,
    compute_rwd_source_candidate_content_digest,
    GuidedRwdSourceCandidateFile,
)


# Deterministic statuses
GUIDED_MANIFEST_STATUS_ACCEPTED = "accepted"
GUIDED_MANIFEST_STATUS_REFUSED = "refused"

# Allowed schema / contract version strings
GUIDED_MANIFEST_SCHEMA_NAME = "guided_runner_candidate_manifest"
GUIDED_MANIFEST_SCHEMA_VERSION = "v1"
GUIDED_CANDIDATE_CONSUMPTION_CONTRACT_VERSION = "exact_candidate_manifest_consumption.v1"


@dataclass(frozen=True)
class GuidedManifestCandidateFile:
    canonical_relative_path: str
    size_bytes: int
    sha256_content_digest: str


@dataclass(frozen=True)
class GuidedCandidateManifestForRunner:
    manifest_schema_name: str
    manifest_schema_version: str
    candidate_consumption_contract_version: str
    source_root_canonical: str
    source_candidate_set_digest: str
    source_candidate_content_digest: str
    candidate_files: tuple[GuidedManifestCandidateFile, ...]
    parser_contract_digest: str
    discovered_roi_ids: tuple[str, ...]
    included_roi_ids: tuple[str, ...]
    excluded_roi_ids: tuple[str, ...]
    strict_roi_inventory_digest: str
    candidate_preflight_identity: str
    roi_preflight_identity: str
    canonical_candidate_manifest_payload_identity: str


@dataclass(frozen=True)
class GuidedManifestCurrentCandidate:
    canonical_relative_path: str
    absolute_path: str


@dataclass(frozen=True)
class GuidedManifestCurrentRoiInventory:
    discovered_roi_ids: tuple[str, ...]
    included_roi_ids: tuple[str, ...]
    excluded_roi_ids: tuple[str, ...]
    parser_contract_digest: str
    strict_roi_inventory_digest: str


@dataclass(frozen=True)
class GuidedManifestCliContext:
    input_format: str
    mode: str
    run_type: str
    traces_only: bool
    discover: bool
    validate_only: bool
    overwrite: bool
    preview_first_n: int | None
    requested_include_rois: tuple[str, ...] | None
    requested_exclude_rois: tuple[str, ...]


@dataclass(frozen=True)
class GuidedManifestVerifiedCandidate:
    canonical_relative_path: str
    absolute_path: str
    size_bytes: int
    sha256_content_digest: str


@dataclass(frozen=True)
class GuidedManifestVerificationIssue:
    category: str
    section: str
    message: str
    detail_code: str = ""


@dataclass(frozen=True)
class GuidedManifestLoadResult:
    status: str
    accepted: bool
    manifest: GuidedCandidateManifestForRunner | None
    blocking_issues: tuple[GuidedManifestVerificationIssue, ...]


@dataclass(frozen=True)
class GuidedManifestVerificationResult:
    status: str
    accepted: bool
    manifest_identity: str | None
    verified_candidates: tuple[GuidedManifestVerifiedCandidate, ...]
    verified_included_roi_ids: tuple[str, ...]
    verified_excluded_roi_ids: tuple[str, ...]
    source_candidate_set_digest: str | None
    source_candidate_content_digest: str | None
    parser_contract_digest: str | None
    strict_roi_inventory_digest: str | None
    blocking_issues: tuple[GuidedManifestVerificationIssue, ...]


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.match(r"^[0-9a-f]{64}$", value))


def compute_guided_candidate_manifest_for_runner_identity(
    manifest: GuidedCandidateManifestForRunner,
) -> str:
    """Compute deterministic identity for GuidedCandidateManifestForRunner."""
    data = {
        "manifest_schema_name": manifest.manifest_schema_name,
        "manifest_schema_version": manifest.manifest_schema_version,
        "candidate_consumption_contract_version": manifest.candidate_consumption_contract_version,
        "source_root_canonical": manifest.source_root_canonical,
        "source_candidate_set_digest": manifest.source_candidate_set_digest,
        "source_candidate_content_digest": manifest.source_candidate_content_digest,
        "candidate_files": [
            {
                "canonical_relative_path": f.canonical_relative_path,
                "size_bytes": f.size_bytes,
                "sha256_content_digest": f.sha256_content_digest,
            }
            for f in manifest.candidate_files
        ],
        "parser_contract_digest": manifest.parser_contract_digest,
        "discovered_roi_ids": manifest.discovered_roi_ids,
        "included_roi_ids": manifest.included_roi_ids,
        "excluded_roi_ids": manifest.excluded_roi_ids,
        "strict_roi_inventory_digest": manifest.strict_roi_inventory_digest,
        "candidate_preflight_identity": manifest.candidate_preflight_identity,
        "roi_preflight_identity": manifest.roi_preflight_identity,
    }
    domain = b"guided-runner-candidate-manifest-payload:v1"
    payload_bytes = encode_canonical_value(data)
    return hashlib.sha256(domain + b"\x00" + payload_bytes).hexdigest()


def _is_safe_relative_path(path: str) -> bool:
    """Check that relative path is safe (not absolute, no drive letters, no traversal)."""
    if not isinstance(path, str) or not path.strip():
        return False
    # No drive letter
    if ":" in path:
        return False
    # No absolute roots
    if path.startswith("/") or path.startswith("\\"):
        return False
    # No traversal or empty segments or dots
    parts = re.split(r"[/\\]", path)
    for p in parts:
        if p in ("..", ".", ""):
            return False
    return True


def load_guided_candidate_manifest(path: str) -> GuidedManifestLoadResult:
    """Loads and strictly validates a guided candidate manifest file."""
    def _issue(category: str, msg: str) -> GuidedManifestLoadResult:
        return GuidedManifestLoadResult(
            status=GUIDED_MANIFEST_STATUS_REFUSED,
            accepted=False,
            manifest=None,
            blocking_issues=(GuidedManifestVerificationIssue(
                category=category,
                section="guided_manifest_loader",
                message=msg,
            ),),
        )

    if not os.path.exists(path):
        return _issue("guided_manifest_missing", f"Guided candidate manifest file not found: {path}.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return _issue("guided_manifest_schema_invalid", f"Invalid JSON payload: {e}")

    if not isinstance(data, dict):
        return _issue("guided_manifest_schema_invalid", "Manifest top-level value must be a JSON object.")

    # Strict top-level keys
    required_keys = {
        "manifest_schema_name",
        "manifest_schema_version",
        "candidate_consumption_contract_version",
        "source_root_canonical",
        "source_candidate_set_digest",
        "source_candidate_content_digest",
        "candidate_files",
        "parser_contract_digest",
        "discovered_roi_ids",
        "included_roi_ids",
        "excluded_roi_ids",
        "strict_roi_inventory_digest",
        "candidate_preflight_identity",
        "roi_preflight_identity",
        "canonical_candidate_manifest_payload_identity",
    }
    actual_keys = set(data.keys())
    if actual_keys != required_keys:
        return _issue("guided_manifest_schema_invalid", "Unknown or missing top-level keys in manifest.")

    # Validate top-level types and values
    schema_name = data["manifest_schema_name"]
    schema_version = data["manifest_schema_version"]
    contract_version = data["candidate_consumption_contract_version"]

    if schema_name != GUIDED_MANIFEST_SCHEMA_NAME:
        return _issue("guided_manifest_version_unsupported", f"Unsupported schema name: {schema_name}")
    if schema_version != GUIDED_MANIFEST_SCHEMA_VERSION:
        return _issue("guided_manifest_version_unsupported", f"Unsupported schema version: {schema_version}")
    if contract_version != GUIDED_CANDIDATE_CONSUMPTION_CONTRACT_VERSION:
        return _issue("guided_manifest_version_unsupported", f"Unsupported consumption contract: {contract_version}")

    # Validate digest string constraints
    for k in (
        "source_candidate_set_digest",
        "source_candidate_content_digest",
        "parser_contract_digest",
        "strict_roi_inventory_digest",
        "candidate_preflight_identity",
        "roi_preflight_identity",
        "canonical_candidate_manifest_payload_identity",
    ):
        if not _valid_sha256(data[k]):
            return _issue("guided_manifest_schema_invalid", f"Field {k} must be a lowercase 64-character hex string.")

    # Validate source root
    if not isinstance(data["source_root_canonical"], str) or not data["source_root_canonical"].strip():
        return _issue("guided_manifest_schema_invalid", "source_root_canonical must be a non-empty string.")

    # Validate ROI ID lists
    for k in ("discovered_roi_ids", "included_roi_ids", "excluded_roi_ids"):
        val = data[k]
        if not isinstance(val, list) or not all(isinstance(x, str) and x.strip() for x in val):
            return _issue("guided_manifest_schema_invalid", f"Field {k} must be a list of non-empty strings.")
        if len(val) != len(set(val)):
            return _issue("guided_manifest_schema_invalid", f"Field {k} contains duplicate ROI IDs.")

    # Validate candidate files list
    cand_list = data["candidate_files"]
    if not isinstance(cand_list, list) or not cand_list:
        return _issue("guided_manifest_schema_invalid", "candidate_files must be a non-empty list.")

    seen_relative_paths = set()
    candidate_objects = []
    for idx, cand in enumerate(cand_list):
        if not isinstance(cand, dict):
            return _issue("guided_manifest_schema_invalid", f"candidate_files[{idx}] must be a JSON object.")
        cand_keys = set(cand.keys())
        expected_cand_keys = {"canonical_relative_path", "size_bytes", "sha256_content_digest"}
        if cand_keys != expected_cand_keys:
            return _issue("guided_manifest_schema_invalid", f"candidate_files[{idx}] has missing or unknown fields.")

        rel_path = cand["canonical_relative_path"]
        size = cand["size_bytes"]
        digest = cand["sha256_content_digest"]

        if not _is_safe_relative_path(rel_path):
            return _issue("guided_manifest_schema_invalid", f"candidate_files[{idx}] contains unsafe relative path.")
        if rel_path in seen_relative_paths:
            return _issue("guided_manifest_schema_invalid", f"candidate_files[{idx}] contains duplicate path: {rel_path}.")
        seen_relative_paths.add(rel_path)

        if not isinstance(size, int) or size < 0:
            return _issue("guided_manifest_schema_invalid", f"candidate_files[{idx}] has invalid size_bytes: {size}.")
        if not _valid_sha256(digest):
            return _issue("guided_manifest_schema_invalid", f"candidate_files[{idx}] has invalid sha256_content_digest.")

        candidate_objects.append(GuidedManifestCandidateFile(
            canonical_relative_path=rel_path,
            size_bytes=size,
            sha256_content_digest=digest,
        ))

    manifest_obj = GuidedCandidateManifestForRunner(
        manifest_schema_name=schema_name,
        manifest_schema_version=schema_version,
        candidate_consumption_contract_version=contract_version,
        source_root_canonical=data["source_root_canonical"],
        source_candidate_set_digest=data["source_candidate_set_digest"],
        source_candidate_content_digest=data["source_candidate_content_digest"],
        candidate_files=tuple(candidate_objects),
        parser_contract_digest=data["parser_contract_digest"],
        discovered_roi_ids=tuple(data["discovered_roi_ids"]),
        included_roi_ids=tuple(data["included_roi_ids"]),
        excluded_roi_ids=tuple(data["excluded_roi_ids"]),
        strict_roi_inventory_digest=data["strict_roi_inventory_digest"],
        candidate_preflight_identity=data["candidate_preflight_identity"],
        roi_preflight_identity=data["roi_preflight_identity"],
        canonical_candidate_manifest_payload_identity=data["canonical_candidate_manifest_payload_identity"],
    )

    # Recompute and verify payload identity
    computed_id = compute_guided_candidate_manifest_for_runner_identity(manifest_obj)
    if computed_id != manifest_obj.canonical_candidate_manifest_payload_identity:
        return _issue(
            "guided_manifest_schema_invalid",
            "Manifest payload identity mismatch: computed identity does not match the stored value."
        )

    return GuidedManifestLoadResult(
        status=GUIDED_MANIFEST_STATUS_ACCEPTED,
        accepted=True,
        manifest=manifest_obj,
        blocking_issues=(),
    )


def verify_guided_candidate_manifest_consumption(
    *,
    manifest: GuidedCandidateManifestForRunner,
    source_root: str,
    current_candidates: tuple[GuidedManifestCurrentCandidate, ...],
    current_roi_inventory: GuidedManifestCurrentRoiInventory,
    cli_context: GuidedManifestCliContext,
) -> GuidedManifestVerificationResult:
    """Verifies that explicitly supplied current candidates and ROIs match the manifest."""
    
    def _issue(category: str, msg: str) -> GuidedManifestVerificationResult:
        return GuidedManifestVerificationResult(
            status=GUIDED_MANIFEST_STATUS_REFUSED,
            accepted=False,
            manifest_identity=manifest.canonical_candidate_manifest_payload_identity,
            verified_candidates=(),
            verified_included_roi_ids=(),
            verified_excluded_roi_ids=(),
            source_candidate_set_digest=None,
            source_candidate_content_digest=None,
            parser_contract_digest=None,
            strict_roi_inventory_digest=None,
            blocking_issues=(GuidedManifestVerificationIssue(
                category=category,
                section="guided_manifest_verifier",
                message=msg,
            ),),
        )

    # 1. Validate CLI context first (require first Guided subset)
    if cli_context.input_format not in ("rwd", "npm"):
        return _issue("guided_manifest_unsupported_mode", "Guided execution requires a supported input format.")
    if cli_context.mode not in {"phasic", "tonic", "both"}:
        return _issue("guided_manifest_unsupported_mode", "Guided execution requires a supported analysis mode.")
    if cli_context.run_type != "full":
        return _issue("guided_manifest_unsupported_mode", "Guided execution requires run_type full.")
    if cli_context.traces_only is not False:
        return _issue("guided_manifest_unsupported_mode", "Guided execution requires traces_only to be False.")
    if cli_context.discover is not False:
        return _issue("guided_manifest_cli_conflict", "Guided execution conflicts with discover flag.")
    if cli_context.validate_only is not False:
        return _issue("guided_manifest_cli_conflict", "Guided execution conflicts with validate_only flag.")
    if cli_context.overwrite is not False:
        return _issue("guided_manifest_cli_conflict", "Guided execution conflicts with overwrite flag.")
    if cli_context.preview_first_n is not None:
        return _issue("guided_manifest_cli_conflict", "Guided execution conflicts with preview_first_n.")
    if cli_context.requested_exclude_rois != ():
        return _issue("guided_manifest_cli_conflict", "Guided execution conflicts with requested exclude ROIs.")
    if cli_context.requested_include_rois is not None:
        if tuple(cli_context.requested_include_rois) != tuple(manifest.included_roi_ids):
            return _issue("guided_manifest_include_roi_mismatch", "CLI requested include ROIs mismatch manifest.")

    # 2. Source root canonicalization & comparison
    try:
        norm_source_root = os.path.normcase(os.path.abspath(os.path.normpath(source_root)))
        real_source_root = os.path.realpath(norm_source_root)
    except Exception as e:
        return _issue("guided_manifest_verification_internal_error", f"Failed to canonicalize source root: {e}")

    # The manifest source root canonical check
    manifest_source_root = os.path.normcase(os.path.abspath(os.path.normpath(manifest.source_root_canonical)))
    if norm_source_root != manifest_source_root:
        return _issue(
            "guided_manifest_source_root_mismatch",
            f"Source root mismatch: current {norm_source_root!r} vs manifest {manifest_source_root!r}."
        )

    # 3. Path traversals and containment validation
    for cand in current_candidates:
        if not _is_safe_relative_path(cand.canonical_relative_path):
            return _issue("guided_manifest_path_invalid", f"Unsafe current candidate relative path: {cand.canonical_relative_path}")
        
        # Verify containment and resolve real path
        try:
            abs_normalized = os.path.normcase(os.path.abspath(os.path.normpath(cand.absolute_path)))
            real_abs_path = os.path.realpath(abs_normalized)
        except Exception as e:
            return _issue("guided_manifest_path_invalid", f"Failed to resolve path for {cand.canonical_relative_path}: {e}")

        # Ensure absolute path matches resolved relative path under source root
        expected_abs = os.path.normcase(os.path.abspath(os.path.join(norm_source_root, cand.canonical_relative_path)))
        if abs_normalized != expected_abs:
            return _issue("guided_manifest_path_invalid", f"Candidate absolute path {abs_normalized} does not match expected relative path location.")

        # Commonpath containment checks (robust escape checks)
        try:
            common = os.path.commonpath([real_source_root, real_abs_path])
            if os.path.normcase(common) != os.path.normcase(real_source_root):
                return _issue("guided_manifest_path_invalid", f"Candidate escapes source root via symlink traversal.")
        except ValueError:
            return _issue("guided_manifest_path_invalid", f"Candidate absolute path has no common path with source root.")

    # 4. Compare current candidate list to manifest candidate list
    manifest_paths = [c.canonical_relative_path for c in manifest.candidate_files]
    current_paths = [c.canonical_relative_path for c in current_candidates]

    if set(current_paths) != set(manifest_paths):
        # Determine missing or extra
        extra = set(current_paths) - set(manifest_paths)
        missing = set(manifest_paths) - set(current_paths)
        if extra:
            return _issue("guided_manifest_extra_candidate_consumed", f"Extra candidates supplied: {extra}")
        if missing:
            return _issue("guided_manifest_missing_candidate", f"Missing candidates from manifest: {missing}")

    # Check ordering
    if current_paths != manifest_paths:
        return _issue("guided_manifest_candidate_set_mismatch", "Candidate files order mismatch.")

    # 5. Fresh file verification (stat and hash)
    verified_candidates = []
    manifest_map = {c.canonical_relative_path: c for c in manifest.candidate_files}

    for cand in current_candidates:
        m_entry = manifest_map[cand.canonical_relative_path]
        
        if not os.path.exists(cand.absolute_path):
            return _issue("guided_manifest_missing_candidate", f"Candidate file does not exist on disk: {cand.absolute_path}")

        try:
            stat_res = os.stat(cand.absolute_path)
            actual_size = stat_res.st_size
        except Exception as e:
            return _issue("guided_manifest_verification_internal_error", f"Failed to stat file: {e}")

        if actual_size != m_entry.size_bytes:
            return _issue(
                "guided_manifest_file_size_mismatch",
                f"Candidate file size mismatch for {cand.canonical_relative_path}. Expected {m_entry.size_bytes}, got {actual_size}."
            )

        # Compute SHA-256
        try:
            h = hashlib.sha256()
            with open(cand.absolute_path, "rb") as file_to_hash:
                while True:
                    chunk = file_to_hash.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
            actual_digest = h.hexdigest()
        except Exception as e:
            return _issue("guided_manifest_verification_internal_error", f"Failed to compute file hash: {e}")

        if actual_digest != m_entry.sha256_content_digest:
            return _issue(
                "guided_manifest_file_digest_mismatch",
                f"Candidate content digest mismatch for {cand.canonical_relative_path}."
            )

        verified_candidates.append(GuidedManifestVerifiedCandidate(
            canonical_relative_path=cand.canonical_relative_path,
            absolute_path=cand.absolute_path,
            size_bytes=actual_size,
            sha256_content_digest=actual_digest,
        ))

    # Recompute and compare aggregate digests using each format's own
    # source-snapshot digest algorithm. RWD's and NPM's digest domains and
    # inputs are disjoint (NPM's also binds each candidate's authoritative
    # filename timestamp, which RWD has no equivalent of), so there is no
    # single generic recomputation shared across formats here.
    try:
        path_style = canonicalize_absolute_path(manifest.source_root_canonical).path_style
        if cli_context.input_format == "npm":
            # A fresh, independent rebuild (mirroring the precedent already
            # established for RWD, where run_candidate_manifest_execution_
            # preflight and run_roi_execution_preflight each independently
            # call build_rwd_source_candidate_snapshot again rather than
            # reusing one shared snapshot). This also sidesteps needing to
            # re-derive each candidate's authoritative_source_start_time
            # from `verified_candidates[i].absolute_path`, which is
            # reconstructed from the case-folded canonical_relative_path on
            # Windows and is therefore not reliably parseable by
            # parse_npm_filename_timestamp (its "T" separator may already
            # be lower-cased).
            npm_snapshot = build_npm_source_candidate_snapshot(source_root)
            recomputed_set_digest = npm_snapshot.source_candidate_set_digest
            recomputed_content_digest = npm_snapshot.source_candidate_content_digest
        else:
            rwd_files = tuple(
                GuidedRwdSourceCandidateFile(
                    canonical_relative_path=v.canonical_relative_path,
                    size_bytes=v.size_bytes,
                    sha256_content_digest=v.sha256_content_digest,
                )
                for v in verified_candidates
            )
            semantic_payload = {
                "snapshot_schema_name": "guided_rwd_source_candidate_snapshot",
                "snapshot_schema_version": "v1",
                "discovery_rule_version": "immediate_child_exact_fluorescence_csv.v1",
                "path_canonicalization_version": "typed_json_utf8.v1",
                "relative_path_rule_version": "canonical_forward_slash_relative_path.v1",
                "digest_algorithm": "sha256",
                "source_root_canonical": manifest.source_root_canonical,
                "source_root_path_style": path_style,
                "source_format": "rwd",
                "acquisition_mode": "intermittent",
                "candidates": rwd_files,
                "ignored_files_policy": "ignore_non_target_entries_bounded_nested_root_check.v1",
                "build_mode": "read_only",
                "unresolved_inputs": (),
            }
            recomputed_set_digest = compute_rwd_source_candidate_set_digest(semantic_payload)
            recomputed_content_digest = compute_rwd_source_candidate_content_digest(semantic_payload)
    except Exception as e:
        return _issue("guided_manifest_verification_internal_error", f"Failed to compute aggregate digests: {e}")

    if recomputed_set_digest != manifest.source_candidate_set_digest:
        return _issue("guided_manifest_candidate_set_mismatch", "Candidate set digest mismatch.")
    if recomputed_content_digest != manifest.source_candidate_content_digest:
        return _issue("guided_manifest_candidate_content_mismatch", "Candidate content digest mismatch.")

    # 6. Parser and ROI comparisons
    if current_roi_inventory.parser_contract_digest != manifest.parser_contract_digest:
        return _issue("guided_manifest_parser_contract_mismatch", "Parser contract digest mismatch.")

    if tuple(current_roi_inventory.discovered_roi_ids) != tuple(manifest.discovered_roi_ids):
        return _issue("guided_manifest_roi_inventory_mismatch", "Discovered ROI inventory mismatch.")

    if tuple(current_roi_inventory.included_roi_ids) != tuple(manifest.included_roi_ids):
        return _issue("guided_manifest_include_roi_mismatch", "Included ROI list mismatch.")

    if tuple(current_roi_inventory.excluded_roi_ids) != tuple(manifest.excluded_roi_ids):
        return _issue("guided_manifest_exclude_roi_mismatch", "Excluded ROI list mismatch.")

    if current_roi_inventory.strict_roi_inventory_digest != manifest.strict_roi_inventory_digest:
        return _issue("guided_manifest_roi_inventory_mismatch", "Strict ROI inventory digest mismatch.")

    return GuidedManifestVerificationResult(
        status=GUIDED_MANIFEST_STATUS_ACCEPTED,
        accepted=True,
        manifest_identity=manifest.canonical_candidate_manifest_payload_identity,
        verified_candidates=tuple(verified_candidates),
        verified_included_roi_ids=manifest.included_roi_ids,
        verified_excluded_roi_ids=manifest.excluded_roi_ids,
        source_candidate_set_digest=manifest.source_candidate_set_digest,
        source_candidate_content_digest=manifest.source_candidate_content_digest,
        parser_contract_digest=manifest.parser_contract_digest,
        strict_roi_inventory_digest=manifest.strict_roi_inventory_digest,
        blocking_issues=(),
    )
