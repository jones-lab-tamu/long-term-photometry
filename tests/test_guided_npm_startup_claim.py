from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import analyze_photometry
import photometry_pipeline.guided_npm_startup_claim as claim_module
from photometry_pipeline.guided_manifest_verification import (
    load_guided_candidate_manifest,
)
from photometry_pipeline.guided_npm_startup_claim import (
    GUIDED_NPM_STARTUP_CLAIM_SOURCE_DIRECT_ARTIFACT,
    GUIDED_NPM_STARTUP_CLAIM_SOURCE_PERSISTENCE_RECEIPT,
    GUIDED_NPM_STARTUP_WRAPPER_ARGUMENT,
    GUIDED_RWD_STARTUP_WRAPPER_ARGUMENT,
    GuidedNpmStartupClaimCancelled,
    GuidedNpmStartupClaimFailure,
    GuidedNpmStartupClaimReceipt,
    claim_guided_npm_startup_artifact,
    claim_guided_npm_startup_artifact_from_receipt,
    claim_guided_npm_startup_artifact_path,
    compute_guided_npm_startup_claim_receipt_identity,
    deserialize_guided_npm_startup_claim_receipt,
    serialize_guided_npm_startup_claim_receipt,
    validate_guided_startup_authority_argument_selection,
    verify_guided_npm_startup_claim_receipt,
)
from photometry_pipeline.guided_npm_startup_persistence import (
    GuidedNpmStartupPersistenceReceipt,
    compute_guided_npm_startup_persistence_receipt_identity,
    persist_guided_npm_startup_payload,
)
from photometry_pipeline.guided_production_mapping import (
    build_application_build_identity,
)

from tests.test_guided_npm_startup_payload import _payload


def _persist(tmp_path: Path):
    payload = _payload(tmp_path)
    receipt = persist_guided_npm_startup_payload(payload)
    assert isinstance(receipt, GuidedNpmStartupPersistenceReceipt)
    return payload, receipt


def _claim(tmp_path: Path):
    payload, receipt = _persist(tmp_path)
    claim = claim_guided_npm_startup_artifact(
        receipt,
        current_application_build_identity=payload.application_build_identity,
    )
    assert isinstance(claim, GuidedNpmStartupClaimReceipt)
    return payload, receipt, claim


def _issue(result, category: str):
    assert isinstance(result, GuidedNpmStartupClaimFailure)
    assert result.blocking_issues[0].category == category
    return result.blocking_issues[0]


def _reidentify_persistence_receipt(receipt, **changes):
    candidate = replace(
        receipt,
        **changes,
        canonical_persistence_receipt_identity="0" * 64,
    )
    return replace(
        candidate,
        canonical_persistence_receipt_identity=(
            compute_guided_npm_startup_persistence_receipt_identity(candidate)
        ),
    )


def test_valid_persisted_artifact_claims_without_execution(tmp_path):
    payload, receipt, claim = _claim(tmp_path)
    assert claim.source_persistence_receipt_identity == (
        receipt.canonical_persistence_receipt_identity
    )
    assert claim.claim_source_kind == GUIDED_NPM_STARTUP_CLAIM_SOURCE_PERSISTENCE_RECEIPT
    assert claim.source_startup_payload_identity == payload.canonical_startup_payload_identity
    assert claim.claimed_payload_identity == payload.canonical_startup_payload_identity
    assert claim.startup_artifact_path == receipt.startup_artifact_path
    assert claim.startup_artifact_sha256 == receipt.persisted_artifact_sha256
    assert claim.startup_artifact_size_bytes == receipt.persisted_size_bytes
    assert claim.wrapper_argument_name == "--guided-npm-startup-artifact"
    assert claim.wrapper_argument_path == receipt.startup_artifact_path
    assert claim.claim_status == "claimed_for_npm_startup"
    assert claim.startup_status == "claimed_not_executed"
    assert claim.runnable is False
    verify_guided_npm_startup_claim_receipt(claim)
    assert tuple(Path(receipt.run_directory_path).iterdir()) == (
        Path(receipt.startup_artifact_path),
    )
    with pytest.raises(FrozenInstanceError):
        claim.runnable = True


def test_claim_signature_has_no_artifact_path_override():
    assert tuple(inspect.signature(claim_guided_npm_startup_artifact).parameters) == (
        "receipt",
        "current_application_build_identity",
        "cancellation_check",
    )
    assert claim_guided_npm_startup_artifact is claim_guided_npm_startup_artifact_from_receipt


def test_claim_refuses_wrong_current_build(tmp_path):
    payload, receipt = _persist(tmp_path)
    current = payload.application_build_identity
    changed = build_application_build_identity(
        distribution_name=current.distribution_name,
        distribution_version=current.distribution_version + ".other",
        source_revision_kind=current.source_revision_kind,
        source_revision=current.source_revision,
        source_tree_state=current.source_tree_state,
        source_tree_digest=current.source_tree_digest,
        build_artifact_digest=current.build_artifact_digest,
        identity_provider_version=current.identity_provider_version,
    )
    _issue(
        claim_guided_npm_startup_artifact(
            receipt, current_application_build_identity=changed
        ),
        "claim_build_identity_mismatch",
    )


@pytest.mark.parametrize(
    ("mutation", "category"),
    (
        ("delete", "claim_artifact_missing"),
        ("rename", "claim_artifact_missing"),
        ("one_byte", "claim_artifact_digest_mismatch"),
        ("same_size", "claim_artifact_digest_mismatch"),
        ("formatting", "claim_artifact_size_mismatch"),
        ("directory", "claim_artifact_path_mismatch"),
    ),
)
def test_claim_time_artifact_mutation_refuses(tmp_path, mutation, category):
    payload, receipt = _persist(tmp_path)
    path = Path(receipt.startup_artifact_path)
    content = path.read_bytes()
    if mutation == "delete":
        path.unlink()
    elif mutation == "rename":
        path.rename(path.with_suffix(".moved"))
    elif mutation == "one_byte":
        path.write_bytes(bytes((content[0] ^ 1,)) + content[1:])
    elif mutation == "same_size":
        index = len(content) // 2
        path.write_bytes(content[:index] + bytes((content[index] ^ 1,)) + content[index + 1 :])
    elif mutation == "formatting":
        path.write_bytes(b" \n" + content)
    else:
        path.unlink()
        path.mkdir()
    _issue(
        claim_guided_npm_startup_artifact(
            receipt,
            current_application_build_identity=payload.application_build_identity,
        ),
        category,
    )


def test_claim_refuses_symlink_replacement_where_supported(tmp_path):
    payload, receipt = _persist(tmp_path)
    path = Path(receipt.startup_artifact_path)
    target = path.with_suffix(".target")
    target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")
    _issue(
        claim_guided_npm_startup_artifact(
            receipt,
            current_application_build_identity=payload.application_build_identity,
        ),
        "claim_artifact_path_mismatch",
    )


def test_claim_refuses_reidentified_receipt_plan_revision_and_payload_mismatch(tmp_path):
    payload, receipt = _persist(tmp_path)
    for field, value, category in (
        ("guided_plan_identity", "1" * 64, "claim_plan_identity_mismatch"),
        ("validation_revision", receipt.validation_revision + 1, "claim_validation_revision_mismatch"),
        ("source_startup_payload_identity", "1" * 64, "claim_payload_identity_mismatch"),
    ):
        changes = {field: value}
        if field == "source_startup_payload_identity":
            changes["readback_payload_identity"] = value
        changed = _reidentify_persistence_receipt(receipt, **changes)
        _issue(
            claim_guided_npm_startup_artifact(
                changed,
                current_application_build_identity=payload.application_build_identity,
            ),
            category,
        )


def test_npm_argument_selection_is_mutually_exclusive_and_explicit():
    assert validate_guided_startup_authority_argument_selection(
        guided_candidate_manifest=None,
        guided_npm_startup_artifact="npm.json",
    ) == "npm"
    assert validate_guided_startup_authority_argument_selection(
        guided_candidate_manifest="rwd.json",
        guided_npm_startup_artifact=None,
    ) == "rwd"
    with pytest.raises(ValueError, match="claim_argument_conflict"):
        validate_guided_startup_authority_argument_selection(
            guided_candidate_manifest="rwd.json",
            guided_npm_startup_artifact="npm.json",
        )
    with pytest.raises(ValueError, match="claim_argument_missing"):
        validate_guided_startup_authority_argument_selection(
            guided_candidate_manifest=None,
            guided_npm_startup_artifact=None,
        )
    assert GUIDED_NPM_STARTUP_WRAPPER_ARGUMENT != GUIDED_RWD_STARTUP_WRAPPER_ARGUMENT


def test_analyze_parser_claims_npm_argument_but_refuses_numerical_execution(
    tmp_path, monkeypatch, capsys
):
    payload, receipt = _persist(tmp_path)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("numerical execution path entered")
    monkeypatch.setattr(analyze_photometry, "Pipeline", forbidden)
    monkeypatch.setattr(analyze_photometry.Config, "from_yaml", forbidden)
    monkeypatch.setattr(analyze_photometry, "load_guided_per_roi_feature_settings", forbidden)
    monkeypatch.setattr(analyze_photometry, "load_guided_per_roi_correction", forbidden)
    monkeypatch.setattr(
        analyze_photometry,
        "resolve_application_build_identity",
        lambda **_kwargs: type(
            "Resolved", (),
            {"status": "resolved", "build_identity": payload.application_build_identity, "blocking_issues": ()},
        )(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_photometry.py",
            "--guided-npm-startup-artifact",
            receipt.startup_artifact_path,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        analyze_photometry.main()
    assert exc.value.code == 0
    assert "claimed successfully" in capsys.readouterr().out
    assert payload.runnable is False


def test_analyze_parser_rejects_both_startup_authority_arguments(
    tmp_path, monkeypatch
):
    _, receipt = _persist(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_photometry.py",
            "--input",
            "unused",
            "--config",
            "unused",
            "--out",
            "unused",
            "--guided-candidate-manifest",
            "rwd.json",
            "--guided-npm-startup-artifact",
            receipt.startup_artifact_path,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        analyze_photometry.main()
    assert exc.value.code == 2


def test_npm_and_rwd_schema_discrimination(tmp_path):
    payload, receipt = _persist(tmp_path)
    npm_path = Path(receipt.startup_artifact_path)
    rwd_loader = load_guided_candidate_manifest(os.fspath(npm_path))
    assert not rwd_loader.accepted

    rwd_bytes = b'{"schema_name":"guided_candidate_manifest","schema_version":"v1"}\n'
    npm_path.write_bytes(rwd_bytes)
    changed = _reidentify_persistence_receipt(
        receipt,
        serialized_payload_sha256=hashlib.sha256(rwd_bytes).hexdigest(),
        persisted_artifact_sha256=hashlib.sha256(rwd_bytes).hexdigest(),
        persisted_size_bytes=len(rwd_bytes),
    )
    _issue(
        claim_guided_npm_startup_artifact(
            changed,
            current_application_build_identity=payload.application_build_identity,
        ),
        "claim_artifact_schema_mismatch",
    )


def test_claim_receipt_serialization_round_trip_and_tampering(tmp_path):
    _, _, claim = _claim(tmp_path)
    serialized = serialize_guided_npm_startup_claim_receipt(claim)
    assert deserialize_guided_npm_startup_claim_receipt(serialized) == claim
    for field, value in (
        ("claim_schema_version", "v999"),
        ("startup_artifact_sha256", "1" * 64),
        ("wrapper_argument_path", os.fspath(tmp_path / "other.json")),
        ("runnable", True),
        ("canonical_claim_receipt_identity", "1" * 64),
    ):
        changed = copy.deepcopy(serialized)
        changed[field] = value
        with pytest.raises(ValueError, match="claim_receipt_serialization_invalid"):
            deserialize_guided_npm_startup_claim_receipt(changed)
    missing = copy.deepcopy(serialized)
    del missing["validation_revision"]
    with pytest.raises(ValueError, match="claim_receipt_serialization_invalid"):
        deserialize_guided_npm_startup_claim_receipt(missing)


def test_claim_receipt_identity_tampering_refuses(tmp_path):
    _, _, claim = _claim(tmp_path)
    changed = replace(claim, canonical_claim_receipt_identity="1" * 64)
    with pytest.raises(ValueError, match="claim_receipt_identity_mismatch"):
        verify_guided_npm_startup_claim_receipt(changed)


def test_claim_cancellation_never_claims(tmp_path):
    payload, receipt = _persist(tmp_path)
    result = claim_guided_npm_startup_artifact(
        receipt,
        current_application_build_identity=payload.application_build_identity,
        cancellation_check=lambda: True,
    )
    assert isinstance(result, GuidedNpmStartupClaimCancelled)
    assert tuple(Path(receipt.run_directory_path).iterdir()) == (
        Path(receipt.startup_artifact_path),
    )


def test_claim_module_has_no_rwd_manifest_dependency_or_claim_sidecar(tmp_path):
    _, receipt, _ = _claim(tmp_path)
    source = inspect.getsource(claim_module).lower()
    assert "guided_candidate_manifest.json" not in source
    assert "guided_startup_wrapper_claim.json" not in source
    assert tuple(Path(receipt.run_directory_path).iterdir()) == (
        Path(receipt.startup_artifact_path),
    )


def test_claim_does_not_launch_a_worker(tmp_path, monkeypatch):
    payload, receipt = _persist(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("worker launch attempted")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    claim = claim_guided_npm_startup_artifact(
        receipt,
        current_application_build_identity=payload.application_build_identity,
    )
    assert isinstance(claim, GuidedNpmStartupClaimReceipt)


def test_direct_artifact_claim_requires_no_persistence_receipt(tmp_path):
    payload, receipt = _persist(tmp_path)
    artifact_path = receipt.startup_artifact_path
    del receipt
    claim = claim_guided_npm_startup_artifact_path(
        artifact_path,
        current_application_build_identity=payload.application_build_identity,
    )
    assert isinstance(claim, GuidedNpmStartupClaimReceipt)
    assert claim.claim_source_kind == GUIDED_NPM_STARTUP_CLAIM_SOURCE_DIRECT_ARTIFACT
    assert claim.source_persistence_receipt_identity is None
    assert claim.source_startup_payload_identity == payload.canonical_startup_payload_identity
    assert claim.runnable is False


@pytest.mark.parametrize(
    ("mutation", "category"),
    (
        ("missing", "claim_artifact_missing"),
        ("renamed", "claim_artifact_path_invalid"),
        ("directory", "claim_artifact_not_regular"),
        ("one_byte", "claim_artifact_noncanonical"),
        ("same_length", "claim_artifact_schema_mismatch"),
        ("truncated", "claim_artifact_noncanonical"),
        ("appended", "claim_artifact_noncanonical"),
        ("pretty", "claim_artifact_noncanonical"),
        ("duplicate", "claim_artifact_noncanonical"),
        ("invalid_utf8", "claim_artifact_noncanonical"),
        ("wrong_schema", "claim_artifact_schema_mismatch"),
    ),
)
def test_direct_artifact_claim_mutations_refuse(tmp_path, mutation, category):
    payload, receipt = _persist(tmp_path)
    path = Path(receipt.startup_artifact_path)
    content = path.read_bytes()
    claim_path = path
    if mutation == "missing":
        path.unlink()
    elif mutation == "renamed":
        claim_path = path.with_name("renamed.json")
        path.rename(claim_path)
    elif mutation == "directory":
        path.unlink()
        path.mkdir()
    elif mutation == "one_byte":
        path.write_bytes(bytes((content[0] ^ 1,)) + content[1:])
    elif mutation == "same_length":
        marker = b'"intermittent"'
        path.write_bytes(content.replace(marker, b'"continuousxx"', 1))
    elif mutation == "truncated":
        path.write_bytes(content[:-1])
    elif mutation == "appended":
        path.write_bytes(content + b"x")
    elif mutation == "pretty":
        path.write_text(json.dumps(json.loads(content), indent=2), encoding="utf-8")
    elif mutation == "duplicate":
        path.write_bytes(content.replace(b"{", b'{"acquisition_mode":"intermittent",', 1))
    elif mutation == "invalid_utf8":
        path.write_bytes(b"\xff" + content[1:])
    else:
        path.write_bytes(content.replace(b'"guided_npm_startup_payload"', b'"guided_candidate_manifest"', 1))
    _issue(
        claim_guided_npm_startup_artifact_path(
            os.fspath(claim_path),
            current_application_build_identity=payload.application_build_identity,
        ),
        category,
    )


def test_direct_artifact_claim_refuses_build_mismatch(tmp_path):
    payload, receipt = _persist(tmp_path)
    current = payload.application_build_identity
    changed = build_application_build_identity(
        distribution_name=current.distribution_name,
        distribution_version=current.distribution_version,
        source_revision_kind=current.source_revision_kind,
        source_revision=current.source_revision,
        source_tree_state=current.source_tree_state,
        source_tree_digest=current.source_tree_digest,
        build_artifact_digest=current.build_artifact_digest,
        identity_provider_version=current.identity_provider_version + ".changed",
    )
    _issue(
        claim_guided_npm_startup_artifact_path(
            receipt.startup_artifact_path,
            current_application_build_identity=changed,
        ),
        "claim_build_identity_mismatch",
    )


def _reidentify_claim(claim, **changes):
    candidate = replace(claim, **changes, canonical_claim_receipt_identity="0" * 64)
    return replace(
        candidate,
        canonical_claim_receipt_identity=compute_guided_npm_startup_claim_receipt_identity(candidate),
    )


def test_direct_and_receipt_claim_serialization_and_provenance_rules(tmp_path):
    payload, receipt, receipt_claim = _claim(tmp_path)
    direct_claim = claim_guided_npm_startup_artifact_path(
        receipt.startup_artifact_path,
        current_application_build_identity=payload.application_build_identity,
    )
    assert isinstance(direct_claim, GuidedNpmStartupClaimReceipt)
    for claim in (receipt_claim, direct_claim):
        serialized = serialize_guided_npm_startup_claim_receipt(claim)
        assert deserialize_guided_npm_startup_claim_receipt(serialized) == claim
    with pytest.raises(ValueError):
        replace(direct_claim, claim_source_kind="unknown")
    with pytest.raises(ValueError):
        replace(receipt_claim, source_persistence_receipt_identity=None)
    with pytest.raises(ValueError):
        replace(direct_claim, source_persistence_receipt_identity="1" * 64)
    for field, value in (
        ("startup_artifact_path", os.fspath(tmp_path / "guided_npm_startup_payload.json")),
        ("startup_artifact_sha256", "1" * 64),
        ("claimed_payload_identity", "1" * 64),
        ("guided_plan_identity", "1" * 64),
        ("validation_revision", direct_claim.validation_revision + 1),
        ("wrapper_argument_name", "--other"),
        ("startup_status", "executed"),
        ("runnable", True),
    ):
        with pytest.raises(ValueError):
            verify_guided_npm_startup_claim_receipt(replace(direct_claim, **{field: value}))


def test_wrapper_claim_only_outcome_never_enters_numerical_paths(tmp_path, monkeypatch):
    payload, receipt = _persist(tmp_path)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("numerical path entered")
    monkeypatch.setattr(analyze_photometry, "Pipeline", forbidden)
    monkeypatch.setattr(analyze_photometry.Config, "from_yaml", forbidden)
    monkeypatch.setattr(analyze_photometry, "load_guided_per_roi_feature_settings", forbidden)
    monkeypatch.setattr(analyze_photometry, "load_guided_per_roi_correction", forbidden)
    outcome = analyze_photometry.claim_guided_npm_startup_for_wrapper(
        receipt.startup_artifact_path,
        current_application_build_identity=payload.application_build_identity,
    )
    assert outcome.execution_status == "claimed_execution_not_implemented"
    assert outcome.runnable is False


@pytest.mark.parametrize("invalid_kind", ("missing", "mutated", "rwd"))
def test_actual_main_distinguishes_claim_failure_without_execution(
    tmp_path, monkeypatch, capsys, invalid_kind
):
    payload, receipt = _persist(tmp_path)
    path = Path(receipt.startup_artifact_path)
    if invalid_kind == "missing":
        path.unlink()
    elif invalid_kind == "mutated":
        path.write_bytes(path.read_bytes() + b"x")
    else:
        path.write_bytes(b'{"schema_name":"guided_candidate_manifest","schema_version":"v1"}\n')
    monkeypatch.setattr(
        analyze_photometry,
        "resolve_application_build_identity",
        lambda **_kwargs: type("Resolved", (), {"status": "resolved", "build_identity": payload.application_build_identity, "blocking_issues": ()})(),
    )
    monkeypatch.setattr(analyze_photometry, "Pipeline", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("pipeline entered")))
    monkeypatch.setattr(sys, "argv", ["analyze_photometry.py", "--guided-npm-startup-artifact", os.fspath(path)])
    with pytest.raises(SystemExit) as exc:
        analyze_photometry.main()
    assert exc.value.code == analyze_photometry.GUIDED_NPM_CLAIM_FAILURE_EXIT_CODE
    assert "claim failed" in capsys.readouterr().out


def test_npm_cli_rejects_every_independent_authority_override(tmp_path, monkeypatch, capsys):
    _, receipt = _persist(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_photometry.py", "--guided-npm-startup-artifact", receipt.startup_artifact_path, "--input", "override"],
    )
    with pytest.raises(SystemExit) as exc:
        analyze_photometry.main()
    assert exc.value.code == analyze_photometry.GUIDED_NPM_CLAIM_FAILURE_EXIT_CODE
    assert "claim_argument_conflict" in capsys.readouterr().out


@pytest.mark.parametrize(
    "override",
    (
        ("--input", "source"),
        ("--format", "npm"),
        ("--acquisition-mode", "intermittent"),
        ("--mode", "tonic"),
        ("--sessions-per-hour", "12"),
        ("--out", "output"),
        ("--traces-only",),
    ),
)
def test_npm_cli_authority_fields_all_fail_closed(tmp_path, monkeypatch, override):
    _, receipt = _persist(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_photometry.py",
            "--guided-npm-startup-artifact",
            receipt.startup_artifact_path,
            *override,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        analyze_photometry.main()
    assert exc.value.code == analyze_photometry.GUIDED_NPM_CLAIM_FAILURE_EXIT_CODE


def test_direct_claim_requires_absolute_exact_unaliased_path(tmp_path):
    payload, receipt = _persist(tmp_path)
    path = Path(receipt.startup_artifact_path)
    for invalid in (
        path.name,
        os.fspath(path.with_name("other.json")),
        os.fspath(path.parent / ".." / path.parent.name / path.name),
    ):
        _issue(
            claim_guided_npm_startup_artifact_path(
                invalid,
                current_application_build_identity=payload.application_build_identity,
            ),
            "claim_artifact_path_invalid" if invalid != os.fspath(path.parent / ".." / path.parent.name / path.name) else "claim_artifact_alias_invalid",
        )


def test_direct_claim_rejects_symlink_where_supported(tmp_path):
    payload, receipt = _persist(tmp_path)
    path = Path(receipt.startup_artifact_path)
    target = path.with_suffix(".target")
    target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")
    _issue(
        claim_guided_npm_startup_artifact_path(
            os.fspath(path),
            current_application_build_identity=payload.application_build_identity,
        ),
        "claim_artifact_alias_invalid",
    )


@pytest.mark.parametrize(
    "mutation",
    ("unsupported_version", "top_identity", "nested_identity", "runnable"),
)
def test_direct_claim_refuses_canonical_json_with_invalid_payload_semantics(
    tmp_path, mutation
):
    payload, receipt = _persist(tmp_path)
    path = Path(receipt.startup_artifact_path)
    value = json.loads(path.read_bytes())
    if mutation == "unsupported_version":
        value["startup_schema_version"] = "v999"
    elif mutation == "top_identity":
        value["canonical_startup_payload_identity"] = "1" * 64
    elif mutation == "nested_identity":
        value["source_projection"]["canonical_source_projection_identity"] = "1" * 64
    else:
        value["runnable"] = True
    path.write_bytes(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    )
    _issue(
        claim_guided_npm_startup_artifact_path(
            os.fspath(path),
            current_application_build_identity=payload.application_build_identity,
        ),
        "claim_artifact_schema_mismatch",
    )


def test_claim_receipt_build_and_identity_tampering_refuses(tmp_path):
    _, _, claim = _claim(tmp_path)
    changed_build = replace(
        claim.application_build_identity,
        canonical_identity="1" * 64,
    )
    with pytest.raises(ValueError, match="application_build_identity_mismatch"):
        verify_guided_npm_startup_claim_receipt(
            replace(claim, application_build_identity=changed_build)
        )
    with pytest.raises(ValueError, match="claim_receipt_identity_mismatch"):
        verify_guided_npm_startup_claim_receipt(
            replace(claim, canonical_claim_receipt_identity="1" * 64)
        )


def test_claim_module_uses_public_shared_artifact_boundary():
    source = inspect.getsource(claim_module)
    assert "verify_guided_npm_startup_artifact_path" in source
    for private_name in (
        "_decode_canonical_payload_bytes",
        "_read_artifact_bytes",
        "_verify_build_identity",
    ):
        assert private_name not in source
