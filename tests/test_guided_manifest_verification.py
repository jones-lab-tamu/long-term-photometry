import ast
import builtins
import hashlib
import json
import os
import sys
from pathlib import Path
import pytest

import photometry_pipeline.guided_manifest_verification as verification
import photometry_pipeline.guided_execution_payloads as payloads
from photometry_pipeline.io.rwd_source_snapshot import (
    compute_rwd_source_candidate_set_digest,
    compute_rwd_source_candidate_content_digest,
    GuidedRwdSourceCandidateFile,
)


# Test data builders
def _valid_manifest_dict(tmp_path):
    from photometry_pipeline.guided_identity import canonicalize_absolute_path
    canon = canonicalize_absolute_path(str(tmp_path.resolve()))
    root = canon.canonical_path
    path_style = canon.path_style
    
    # Pre-calculate matching set/content digests using standard helpers to guarantee alignment
    c1 = GuidedRwdSourceCandidateFile(
        canonical_relative_path="session1/fluorescence.csv",
        size_bytes=100,
        sha256_content_digest=hashlib.sha256(b"x" * 100).hexdigest()
    )
    c2 = GuidedRwdSourceCandidateFile(
        canonical_relative_path="session2/fluorescence.csv",
        size_bytes=200,
        sha256_content_digest=hashlib.sha256(b"y" * 200).hexdigest()
    )
    
    semantic = {
        "snapshot_schema_name": "guided_rwd_source_candidate_snapshot",
        "snapshot_schema_version": "v1",
        "discovery_rule_version": "immediate_child_exact_fluorescence_csv.v1",
        "path_canonicalization_version": "typed_json_utf8.v1",
        "relative_path_rule_version": "canonical_forward_slash_relative_path.v1",
        "digest_algorithm": "sha256",
        "source_root_canonical": root,
        "source_root_path_style": path_style,
        "source_format": "rwd",
        "acquisition_mode": "intermittent",
        "candidates": (c1, c2),
        "ignored_files_policy": "ignore_non_target_entries_bounded_nested_root_check.v1",
        "build_mode": "read_only",
        "unresolved_inputs": (),
    }
    
    set_digest = compute_rwd_source_candidate_set_digest(semantic)
    content_digest = compute_rwd_source_candidate_content_digest(semantic)

    return {
        "manifest_schema_name": "guided_runner_candidate_manifest",
        "manifest_schema_version": "v1",
        "candidate_consumption_contract_version": "exact_candidate_manifest_consumption.v1",
        "source_root_canonical": root,
        "source_candidate_set_digest": set_digest,
        "source_candidate_content_digest": content_digest,
        "candidate_files": [
            {
                "canonical_relative_path": "session1/fluorescence.csv",
                "size_bytes": 100,
                "sha256_content_digest": c1.sha256_content_digest,
            },
            {
                "canonical_relative_path": "session2/fluorescence.csv",
                "size_bytes": 200,
                "sha256_content_digest": c2.sha256_content_digest,
            }
        ],
        "parser_contract_digest": "e" * 64,
        "discovered_roi_ids": ["ROI1", "ROI2"],
        "included_roi_ids": ["ROI1"],
        "excluded_roi_ids": ["ROI2"],
        "strict_roi_inventory_digest": "f" * 64,
        "candidate_preflight_identity": "a" * 64,
        "roi_preflight_identity": "b" * 64,
        "canonical_candidate_manifest_payload_identity": "0" * 64, # Placeholder
    }


def _write_manifest(tmp_path, manifest_dict):
    # Compute proper identity
    manifest_files = [
        verification.GuidedManifestCandidateFile(
            canonical_relative_path=f["canonical_relative_path"],
            size_bytes=f["size_bytes"],
            sha256_content_digest=f["sha256_content_digest"],
        )
        for f in manifest_dict["candidate_files"]
    ]
    manifest_obj = verification.GuidedCandidateManifestForRunner(
        manifest_schema_name=manifest_dict["manifest_schema_name"],
        manifest_schema_version=manifest_dict["manifest_schema_version"],
        candidate_consumption_contract_version=manifest_dict["candidate_consumption_contract_version"],
        source_root_canonical=manifest_dict["source_root_canonical"],
        source_candidate_set_digest=manifest_dict["source_candidate_set_digest"],
        source_candidate_content_digest=manifest_dict["source_candidate_content_digest"],
        candidate_files=tuple(manifest_files),
        parser_contract_digest=manifest_dict["parser_contract_digest"],
        discovered_roi_ids=tuple(manifest_dict["discovered_roi_ids"]),
        included_roi_ids=tuple(manifest_dict["included_roi_ids"]),
        excluded_roi_ids=tuple(manifest_dict["excluded_roi_ids"]),
        strict_roi_inventory_digest=manifest_dict["strict_roi_inventory_digest"],
        candidate_preflight_identity=manifest_dict["candidate_preflight_identity"],
        roi_preflight_identity=manifest_dict["roi_preflight_identity"],
        canonical_candidate_manifest_payload_identity="0" * 64,
    )
    identity = verification.compute_guided_candidate_manifest_for_runner_identity(manifest_obj)
    manifest_dict["canonical_candidate_manifest_payload_identity"] = identity
    
    path = tmp_path / "guided_candidate_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f)
    return path, identity


def _write_raw_manifest(tmp_path, manifest_dict):
    path = tmp_path / "guided_candidate_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f)
    return path


def _cli_context():
    return verification.GuidedManifestCliContext(
        input_format="rwd",
        mode="phasic",
        run_type="full",
        traces_only=False,
        discover=False,
        validate_only=False,
        overwrite=False,
        preview_first_n=None,
        requested_include_rois=None,
        requested_exclude_rois=(),
    )


def _current_roi_inventory():
    return verification.GuidedManifestCurrentRoiInventory(
        discovered_roi_ids=("ROI1", "ROI2"),
        included_roi_ids=("ROI1",),
        excluded_roi_ids=("ROI2",),
        parser_contract_digest="e" * 64,
        strict_roi_inventory_digest="f" * 64,
    )


# 1. Loader accepts valid manifest JSON.
def test_loader_accepts_valid_manifest(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, identity = _write_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is True
    assert res.status == verification.GUIDED_MANIFEST_STATUS_ACCEPTED
    assert res.manifest is not None
    assert res.manifest.canonical_candidate_manifest_payload_identity == identity


# 2. Loader refuses missing manifest path
def test_loader_refuses_missing_path():
    res = verification.load_guided_candidate_manifest("non_existent_file.json")
    assert res.accepted is False
    assert res.status == verification.GUIDED_MANIFEST_STATUS_REFUSED
    assert res.blocking_issues[0].category == "guided_manifest_missing"


# 3. Loader refuses invalid JSON
def test_loader_refuses_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("invalid json contents", encoding="utf-8")
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


# 4. Loader refuses missing required top-level field
def test_loader_refuses_missing_toplevel_field(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    del m_dict["parser_contract_digest"]
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


# 5. Loader refuses unknown top-level field
def test_loader_refuses_unknown_toplevel_field(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["unknown_field"] = "extra"
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


# 6. Loader refuses unknown candidate file field
def test_loader_refuses_unknown_candidate_field(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["candidate_files"][0]["extra_cand_field"] = "bad"
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


# 7. Loader refuses unsupported schema version
def test_loader_refuses_unsupported_schema_version(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["manifest_schema_version"] = "v99"
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_version_unsupported"


# 8. Loader refuses unsupported candidate consumption contract version
def test_loader_refuses_unsupported_contract_version(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["candidate_consumption_contract_version"] = "bad_version"
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_version_unsupported"


# 9. Loader refuses non-lowercase or non-64-character digest/identity values
def test_loader_refuses_invalid_digests(tmp_path):
    for bad_digest in ("A" * 64, "g" * 64, "abc", "123"):
        m_dict = _valid_manifest_dict(tmp_path)
        m_dict["parser_contract_digest"] = bad_digest
        path = _write_raw_manifest(tmp_path, m_dict)
        res = verification.load_guided_candidate_manifest(str(path))
        assert res.accepted is False
        assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


# 10. Loader refuses negative size_bytes
def test_loader_refuses_negative_size(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["candidate_files"][0]["size_bytes"] = -1
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


# 11. Loader refuses empty candidate_files
def test_loader_refuses_empty_candidates(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["candidate_files"] = []
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


# 12. Loader refuses duplicate candidate relative paths
def test_loader_refuses_duplicate_paths(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["candidate_files"].append(m_dict["candidate_files"][0])
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


# 13. Loader refuses duplicate ROI IDs
def test_loader_refuses_duplicate_rois(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["discovered_roi_ids"].append("ROI1")
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


# 14. Loader recomputes and verifies manifest payload identity
def test_loader_verifies_payload_identity(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    # Edit the json to have wrong identity
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["canonical_candidate_manifest_payload_identity"] = "f" * 64
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


# 15. Compatibility test with 4J14i manifest payload identity
def test_compatibility_with_4j14i_identity(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    # Build payload dataclasses
    manifest_files = [
        payloads.GuidedRunnerCandidateManifestEntry(
            canonical_relative_path=f["canonical_relative_path"],
            size_bytes=f["size_bytes"],
            sha256_content_digest=f["sha256_content_digest"],
        )
        for f in m_dict["candidate_files"]
    ]
    preflight_manifest = payloads.GuidedRunnerCandidateManifestPayload(
        manifest_schema_name=m_dict["manifest_schema_name"],
        manifest_schema_version=m_dict["manifest_schema_version"],
        candidate_consumption_contract_version=m_dict["candidate_consumption_contract_version"],
        source_root_canonical=m_dict["source_root_canonical"],
        source_candidate_set_digest=m_dict["source_candidate_set_digest"],
        source_candidate_content_digest=m_dict["source_candidate_content_digest"],
        candidate_files=tuple(manifest_files),
        parser_contract_digest=m_dict["parser_contract_digest"],
        discovered_roi_ids=tuple(m_dict["discovered_roi_ids"]),
        included_roi_ids=tuple(m_dict["included_roi_ids"]),
        excluded_roi_ids=tuple(m_dict["excluded_roi_ids"]),
        strict_roi_inventory_digest=m_dict["strict_roi_inventory_digest"],
        candidate_preflight_identity=m_dict["candidate_preflight_identity"],
        roi_preflight_identity=m_dict["roi_preflight_identity"],
        canonical_candidate_manifest_payload_identity="0" * 64,
    )
    preflight_id = payloads.compute_guided_runner_candidate_manifest_payload_identity(preflight_manifest)

    # Loader equivalent
    loader_files = [
        verification.GuidedManifestCandidateFile(
            canonical_relative_path=f["canonical_relative_path"],
            size_bytes=f["size_bytes"],
            sha256_content_digest=f["sha256_content_digest"],
        )
        for f in m_dict["candidate_files"]
    ]
    loader_manifest = verification.GuidedCandidateManifestForRunner(
        manifest_schema_name=m_dict["manifest_schema_name"],
        manifest_schema_version=m_dict["manifest_schema_version"],
        candidate_consumption_contract_version=m_dict["candidate_consumption_contract_version"],
        source_root_canonical=m_dict["source_root_canonical"],
        source_candidate_set_digest=m_dict["source_candidate_set_digest"],
        source_candidate_content_digest=m_dict["source_candidate_content_digest"],
        candidate_files=tuple(loader_files),
        parser_contract_digest=m_dict["parser_contract_digest"],
        discovered_roi_ids=tuple(m_dict["discovered_roi_ids"]),
        included_roi_ids=tuple(m_dict["included_roi_ids"]),
        excluded_roi_ids=tuple(m_dict["excluded_roi_ids"]),
        strict_roi_inventory_digest=m_dict["strict_roi_inventory_digest"],
        candidate_preflight_identity=m_dict["candidate_preflight_identity"],
        roi_preflight_identity=m_dict["roi_preflight_identity"],
        canonical_candidate_manifest_payload_identity="0" * 64,
    )
    loader_id = verification.compute_guided_candidate_manifest_for_runner_identity(loader_manifest)
    assert loader_id == preflight_id


# 16. Verifier accepts valid manifest/current candidates/current ROI inventory/CLI context
def test_verifier_accepts_valid_inputs(tmp_path):
    # Setup files on disk
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"y" * 200)

    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    
    load_res = verification.load_guided_candidate_manifest(str(path))
    assert load_res.accepted is True

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )

    result = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest,
        source_root=str(tmp_path),
        current_candidates=curr_candidates,
        current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context(),
    )
    assert result.accepted is True
    assert result.status == verification.GUIDED_MANIFEST_STATUS_ACCEPTED


# 17-25. CLI context refusals
def test_verifier_refuses_bad_cli_context(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    # Test format
    ctx = _cli_context()
    ctx = verification.GuidedManifestCliContext(**{**ctx.__dict__, "input_format": "npm"})
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=ctx
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_unsupported_mode"

    # Test mode
    ctx = _cli_context()
    ctx = verification.GuidedManifestCliContext(**{**ctx.__dict__, "mode": "tonic"})
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=ctx
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_unsupported_mode"


# Detailed Scenario Tests for CLI Context
def test_unsupported_run_type(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    ctx = _cli_context()
    ctx = verification.GuidedManifestCliContext(**{**ctx.__dict__, "run_type": "part"})
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=ctx
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_unsupported_mode"


def test_traces_only_true(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    ctx = _cli_context()
    ctx = verification.GuidedManifestCliContext(**{**ctx.__dict__, "traces_only": True})
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=ctx
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_unsupported_mode"


def test_discover_true(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    ctx = _cli_context()
    ctx = verification.GuidedManifestCliContext(**{**ctx.__dict__, "discover": True})
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=ctx
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_cli_conflict"


def test_validate_only_true(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    ctx = _cli_context()
    ctx = verification.GuidedManifestCliContext(**{**ctx.__dict__, "validate_only": True})
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=ctx
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_cli_conflict"


def test_overwrite_true(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    ctx = _cli_context()
    ctx = verification.GuidedManifestCliContext(**{**ctx.__dict__, "overwrite": True})
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=ctx
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_cli_conflict"


def test_preview_first_n_set(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    ctx = _cli_context()
    ctx = verification.GuidedManifestCliContext(**{**ctx.__dict__, "preview_first_n": 5})
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=ctx
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_cli_conflict"


def test_requested_exclude_rois_present(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    ctx = _cli_context()
    ctx = verification.GuidedManifestCliContext(**{**ctx.__dict__, "requested_exclude_rois": ("ROI2",)})
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=ctx
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_cli_conflict"


def test_requested_include_rois_matching_success(tmp_path):
    # Setup files on disk
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"y" * 200)

    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    
    ctx = _cli_context()
    ctx = verification.GuidedManifestCliContext(**{**ctx.__dict__, "requested_include_rois": ("ROI1",)})
    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=ctx
    )
    assert res.accepted is True


# 26. Verifier refuses mismatched requested_include_rois
def test_verifier_refuses_mismatched_include_rois(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    ctx = _cli_context()
    ctx = verification.GuidedManifestCliContext(**{**ctx.__dict__, "requested_include_rois": ("ROI1", "ROI2")})
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=ctx
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_include_roi_mismatch"


# 28. Verifier refuses source root mismatch
def test_verifier_refuses_source_root_mismatch(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root="different_root",
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_source_root_mismatch"


# Path validation tests
def test_verifier_path_validations(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["candidate_files"][0]["canonical_relative_path"] = "/absolute/path"
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


def test_drive_letter_candidate_path_refusal(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["candidate_files"][0]["canonical_relative_path"] = "C:\\session1\\fluorescence.csv"
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


def test_dotdot_traversal_refusal(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["candidate_files"][0]["canonical_relative_path"] = "session1/../fluorescence.csv"
    path = _write_raw_manifest(tmp_path, m_dict)
    res = verification.load_guided_candidate_manifest(str(path))
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_schema_invalid"


def test_current_candidate_unsafe_relative_path_refusal(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/../fluorescence.csv", # Unsafe traversal
            absolute_path=str(tmp_path / "session1/fluorescence.csv"),
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_path_invalid"


def test_current_candidate_absolute_path_mismatch_refusal(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(tmp_path / "session2/fluorescence.csv"), # Absolute path mismatch
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_path_invalid"


# 33. Verifier refuses missing current candidate
def test_verifier_refuses_missing_current_candidate(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=(), current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_missing_candidate"


# 34. Verifier refuses extra current candidate
def test_verifier_refuses_extra_current_candidate(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(tmp_path / "session1/fluorescence.csv"),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(tmp_path / "session2/fluorescence.csv"),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session3/fluorescence.csv",
            absolute_path=str(tmp_path / "session3/fluorescence.csv"),
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_extra_candidate_consumed"


# 35. Verifier refuses same set but different order
def test_verifier_refuses_different_order(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(tmp_path / "session2/fluorescence.csv"),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(tmp_path / "session1/fluorescence.csv"),
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_candidate_set_mismatch"


# 36. Verifier refuses missing candidate file on disk
def test_verifier_refuses_missing_file_on_disk(tmp_path):
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(tmp_path / "session1/fluorescence.csv"),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(tmp_path / "session2/fluorescence.csv"),
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_missing_candidate"


# 37. Verifier refuses file size mismatch
def test_verifier_refuses_size_mismatch(tmp_path):
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 99) # Incorrect size
    f2.write_bytes(b"y" * 200)

    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_file_size_mismatch"


# 38. Verifier refuses file SHA-256 digest mismatch
def test_verifier_refuses_digest_mismatch(tmp_path):
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"z" * 100) # Wrong contents
    f2.write_bytes(b"y" * 200)

    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["candidate_files"][0]["sha256_content_digest"] = "c" * 64
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_file_digest_mismatch"


# 39-43. ROI inventory mismatches
def test_verifier_refuses_roi_mismatches(tmp_path):
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"y" * 200)

    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )

    # Test discovered ROI mismatch
    bad_inventory = verification.GuidedManifestCurrentRoiInventory(
        discovered_roi_ids=("ROI1", "ROI_DIFFERENT"),
        included_roi_ids=("ROI1",),
        excluded_roi_ids=("ROI2",),
        parser_contract_digest="e" * 64,
        strict_roi_inventory_digest="f" * 64,
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=bad_inventory,
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_roi_inventory_mismatch"

    # Test included ROI tuple mismatch
    bad_inventory_inc = verification.GuidedManifestCurrentRoiInventory(
        discovered_roi_ids=("ROI1", "ROI2"),
        included_roi_ids=("ROI2",), # mismatch
        excluded_roi_ids=("ROI2",),
        parser_contract_digest="e" * 64,
        strict_roi_inventory_digest="f" * 64,
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=bad_inventory_inc,
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_include_roi_mismatch"

    # Test excluded ROI tuple mismatch
    bad_inventory_exc = verification.GuidedManifestCurrentRoiInventory(
        discovered_roi_ids=("ROI1", "ROI2"),
        included_roi_ids=("ROI1",),
        excluded_roi_ids=("ROI1",), # mismatch
        parser_contract_digest="e" * 64,
        strict_roi_inventory_digest="f" * 64,
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=bad_inventory_exc,
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_exclude_roi_mismatch"

    # Test strict ROI inventory digest mismatch
    bad_inventory_strict = verification.GuidedManifestCurrentRoiInventory(
        discovered_roi_ids=("ROI1", "ROI2"),
        included_roi_ids=("ROI1",),
        excluded_roi_ids=("ROI2",),
        parser_contract_digest="e" * 64,
        strict_roi_inventory_digest="bad_strict_digest" + "a" * 48,
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=bad_inventory_strict,
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_roi_inventory_mismatch"


# 46. Verifier does not discover files, tested by passing an extra file on disk
def test_verifier_does_not_discover_files(tmp_path):
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f3 = tmp_path / "session3" / "fluorescence.csv" # Extra file on disk
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f3.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"y" * 200)
    f3.write_bytes(b"z" * 300)

    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )

    # Verify that having an extra file on disk does NOT fail verification because the verifier does not scan/discover
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is True


# 48. No output side effects occur during load or verification
def test_no_output_side_effects(tmp_path, monkeypatch):
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"y" * 200)

    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    
    # Setup forbidden side effects monkeypatches
    def _fail(*args, **kwargs):
        pytest.fail("Side effect executed!")

    original_open = builtins.open
    def _fail_open(file, mode="r", *args, **kwargs):
        if any(ch in mode for ch in ("w", "a", "x", "+")):
            pytest.fail(f"Filesystem write/open attempted with mode {mode}!")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fail)
    monkeypatch.setattr(Path, "write_bytes", _fail)
    monkeypatch.setattr(Path, "mkdir", _fail)
    monkeypatch.setattr(Path, "touch", _fail)
    monkeypatch.setattr(os, "makedirs", _fail)
    monkeypatch.setattr(os, "mkdir", _fail)
    monkeypatch.setattr(builtins, "open", _fail_open)
    
    import tempfile
    monkeypatch.setattr(tempfile, "mktemp", _fail)
    monkeypatch.setattr(tempfile, "mkstemp", _fail)
    monkeypatch.setattr(tempfile, "mkdtemp", _fail)
    
    import subprocess
    monkeypatch.setattr(subprocess, "run", _fail)

    load_res = verification.load_guided_candidate_manifest(str(path))
    assert load_res.accepted is True

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is True


# 49. Import-boundary AST test
def test_import_boundary():
    file_path = Path(__file__).parent.parent / "photometry_pipeline" / "guided_manifest_verification.py"
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    forbidden_packages = {
        "tools.run_full_pipeline_deliverables",
        "photometry_pipeline.pipeline",
        "photometry_pipeline.discovery",
        "photometry_pipeline.io.adapters",
        "gui",
        "RunSpec",
        "subprocess",
        "output_allocator",
        "status_writer",
        "report_writer",
        "completed_run_loader",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imported = name.name
                for forbidden in forbidden_packages:
                    if imported == forbidden or imported.startswith(forbidden + "."):
                        pytest.fail(f"Forbidden import statement: {imported}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module
                for name in node.names:
                    imported = f"{module_name}.{name.name}"
                    for forbidden in forbidden_packages:
                        if (
                            module_name == forbidden 
                            or module_name.startswith(forbidden + ".")
                            or imported == forbidden 
                            or imported.startswith(forbidden + ".")
                        ):
                            pytest.fail(f"Forbidden import: {imported} (from {module_name})")


# 50. Windows normalization test
def test_windows_normalization(tmp_path):
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"y" * 200)

    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    
    load_res = verification.load_guided_candidate_manifest(str(path))
    assert load_res.accepted is True

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path).lower(), # Casing variation
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is True


# 51. Symlink escape test
def test_symlink_escape(tmp_path):
    # Try to create symlinks, skip if unsupported (e.g. Windows without admin privileges)
    try:
        source_dir = tmp_path / "source_root"
        escape_dir = tmp_path / "escape_root"
        source_dir.mkdir()
        escape_dir.mkdir()
        
        target_file = escape_dir / "secret.csv"
        target_file.write_bytes(b"x" * 100)
        
        # Symlink escaping root
        symlink_path = source_dir / "session1_fluorescence.csv"
        os.symlink(str(target_file), str(symlink_path))
    except (OSError, NotImplementedError, AttributeError) as e:
        pytest.skip(f"Symlinks are not fully supported on this platform: {e}")

    # Build manifest with correct set/content digests for this single candidate file
    c1 = GuidedRwdSourceCandidateFile(
        canonical_relative_path="session1_fluorescence.csv",
        size_bytes=100,
        sha256_content_digest=hashlib.sha256(b"x" * 100).hexdigest()
    )
    from photometry_pipeline.guided_identity import canonicalize_absolute_path
    canon = canonicalize_absolute_path(str(source_dir))
    root = canon.canonical_path
    path_style = canon.path_style
    semantic = {
        "snapshot_schema_name": "guided_rwd_source_candidate_snapshot",
        "snapshot_schema_version": "v1",
        "discovery_rule_version": "immediate_child_exact_fluorescence_csv.v1",
        "path_canonicalization_version": "typed_json_utf8.v1",
        "relative_path_rule_version": "canonical_forward_slash_relative_path.v1",
        "digest_algorithm": "sha256",
        "source_root_canonical": root,
        "source_root_path_style": path_style,
        "source_format": "rwd",
        "acquisition_mode": "intermittent",
        "candidates": (c1,),
        "ignored_files_policy": "ignore_non_target_entries_bounded_nested_root_check.v1",
        "build_mode": "read_only",
        "unresolved_inputs": (),
    }
    set_digest = compute_rwd_source_candidate_set_digest(semantic)
    content_digest = compute_rwd_source_candidate_content_digest(semantic)

    m_dict = _valid_manifest_dict(source_dir)
    m_dict["candidate_files"] = [{
        "canonical_relative_path": "session1_fluorescence.csv",
        "size_bytes": 100,
        "sha256_content_digest": c1.sha256_content_digest
    }]
    m_dict["source_root_canonical"] = root
    m_dict["source_candidate_set_digest"] = set_digest
    m_dict["source_candidate_content_digest"] = content_digest
    
    path, _ = _write_manifest(source_dir, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    assert load_res.accepted is True

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1_fluorescence.csv",
            absolute_path=str(symlink_path),
        ),
    )

    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(source_dir),
        current_candidates=curr_candidates, current_roi_inventory=verification.GuidedManifestCurrentRoiInventory(
            discovered_roi_ids=("ROI1", "ROI2"),
            included_roi_ids=("ROI1",),
            excluded_roi_ids=("ROI2",),
            parser_contract_digest="e" * 64,
            strict_roi_inventory_digest="f" * 64,
        ),
        cli_context=_cli_context()
    )
    # Refuses because the symlink realpath resolves outside the source root
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_path_invalid"


# 53. Unexpected hash/read exception becomes guided_manifest_verification_internal_error
def test_unexpected_exception_handling(tmp_path, monkeypatch):
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"y" * 200)

    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    assert load_res.accepted is True

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )

    # Monkeypatch open to raise PermissionError
    original_open = builtins.open
    def _mock_open(file, mode="r", *args, **kwargs):
        if str(file).endswith("fluorescence.csv") and "b" in mode:
            raise PermissionError("Access denied")
        return original_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", _mock_open)

    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=verification.GuidedManifestCurrentRoiInventory(
            discovered_roi_ids=("ROI1", "ROI2"),
            included_roi_ids=("ROI1",),
            excluded_roi_ids=("ROI2",),
            parser_contract_digest="e" * 64,
            strict_roi_inventory_digest="f" * 64,
        ),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_verification_internal_error"


# Aggregate candidate set/content digest mismatch tests
def test_aggregate_candidate_set_digest_mismatch(tmp_path):
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"y" * 200)

    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["source_candidate_set_digest"] = "f" * 64 # Incorrect set digest
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    assert load_res.accepted is True

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_candidate_set_mismatch"


def test_aggregate_candidate_content_digest_mismatch(tmp_path):
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"y" * 200)

    m_dict = _valid_manifest_dict(tmp_path)
    m_dict["source_candidate_content_digest"] = "f" * 64 # Incorrect content digest
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    assert load_res.accepted is True

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )
    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is False
    assert res.blocking_issues[0].category == "guided_manifest_candidate_content_mismatch"


def test_valid_manifest_from_helpers_verifies_successfully(tmp_path):
    # Setup files on disk
    f1 = tmp_path / "session1" / "fluorescence.csv"
    f2 = tmp_path / "session2" / "fluorescence.csv"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"y" * 200)

    # Let _valid_manifest_dict use the official helpers to generate matching digests
    m_dict = _valid_manifest_dict(tmp_path)
    path, _ = _write_manifest(tmp_path, m_dict)
    load_res = verification.load_guided_candidate_manifest(str(path))
    assert load_res.accepted is True

    curr_candidates = (
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session1/fluorescence.csv",
            absolute_path=str(f1),
        ),
        verification.GuidedManifestCurrentCandidate(
            canonical_relative_path="session2/fluorescence.csv",
            absolute_path=str(f2),
        ),
    )

    res = verification.verify_guided_candidate_manifest_consumption(
        manifest=load_res.manifest, source_root=str(tmp_path),
        current_candidates=curr_candidates, current_roi_inventory=_current_roi_inventory(),
        cli_context=_cli_context()
    )
    assert res.accepted is True
    assert res.status == verification.GUIDED_MANIFEST_STATUS_ACCEPTED
