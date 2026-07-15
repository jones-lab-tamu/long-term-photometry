from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import threading

import pytest

import photometry_pipeline.guided_npm_worker_prelaunch_claim as module
from photometry_pipeline.guided_npm_worker_prelaunch_claim import (
    GuidedNpmPrelaunchFreshnessEvidence,
    GuidedNpmWorkerPrelaunchClaim,
    GuidedNpmWorkerPrelaunchClaimCancelled,
    GuidedNpmWorkerPrelaunchClaimFailure,
    claim_guided_npm_worker_for_prelaunch,
    compute_guided_npm_prelaunch_freshness_evidence_identity,
    compute_guided_npm_worker_prelaunch_claim_identity,
    stored_parent_and_name,
    stored_paths_equal,
    verify_stored_path_style,
    verify_guided_npm_prelaunch_freshness_evidence,
    verify_guided_npm_worker_prelaunch_claim,
)
from photometry_pipeline.guided_npm_worker_request import GUIDED_NPM_WORKER_REQUEST_FILENAME
from photometry_pipeline.guided_npm_worker_request_materialization import (
    GuidedNpmWorkerRequestMaterializationReceipt,
    compute_guided_npm_worker_request_materialization_receipt_identity,
    materialize_guided_npm_worker_request,
)
from photometry_pipeline.guided_production_mapping import build_application_build_identity
from tests.test_guided_npm_production_execution_request import _request


def _prepared(tmp_path):
    _, _, execution = _request(tmp_path)
    receipt = materialize_guided_npm_worker_request(
        execution,
        current_application_build_identity=execution.application_build_identity,
    )
    assert isinstance(receipt, GuidedNpmWorkerRequestMaterializationReceipt)
    return execution, receipt


def _claim(tmp_path, execution=None, receipt=None, **kwargs):
    if execution is None or receipt is None:
        execution, receipt = _prepared(tmp_path)
    result = claim_guided_npm_worker_for_prelaunch(
        kwargs.pop("path", receipt.worker_request_artifact_path),
        receipt,
        current_application_build_identity=kwargs.pop("build", execution.application_build_identity),
        **kwargs,
    )
    return execution, receipt, result


def _failure(result, category):
    assert isinstance(result, GuidedNpmWorkerPrelaunchClaimFailure)
    assert result.blocking_issues[0].category == category


def _reidentify_receipt(receipt, **changes):
    value = replace(receipt, **changes, canonical_materialization_receipt_identity="0" * 64)
    return replace(
        value,
        canonical_materialization_receipt_identity=compute_guided_npm_worker_request_materialization_receipt_identity(value),
    )


def _reidentify_evidence(evidence, **changes):
    value = replace(evidence, **changes, canonical_prelaunch_freshness_evidence_identity="0" * 64)
    return replace(
        value,
        canonical_prelaunch_freshness_evidence_identity=compute_guided_npm_prelaunch_freshness_evidence_identity(value),
    )


def _reidentify_claim(claim, **changes):
    value = replace(claim, **changes, canonical_prelaunch_claim_identity="0" * 64)
    return replace(value, canonical_prelaunch_claim_identity=compute_guided_npm_worker_prelaunch_claim_identity(value))


def _different_build(build, suffix="other"):
    return build_application_build_identity(
        distribution_name=build.distribution_name,
        distribution_version=build.distribution_version + "." + suffix,
        source_revision_kind=build.source_revision_kind,
        source_revision=build.source_revision,
        source_tree_state=build.source_tree_state,
        source_tree_digest=build.source_tree_digest,
        build_artifact_digest=build.build_artifact_digest,
        identity_provider_version=build.identity_provider_version,
    )


def test_exact_artifact_and_receipt_create_frozen_nonlaunched_claim_without_writes(tmp_path):
    execution, receipt = _prepared(tmp_path)
    run = Path(receipt.run_directory_path)
    before = {item.name for item in run.iterdir()}
    _, _, claim = _claim(tmp_path, execution, receipt)
    assert isinstance(claim, GuidedNpmWorkerPrelaunchClaim)
    verify_guided_npm_worker_prelaunch_claim(claim)
    assert claim.claim_status == "verified_for_prelaunch"
    assert (claim.launch_status, claim.execution_status, claim.completion_status, claim.runnable) == (
        "not_launched", "not_started", "not_available", False,
    )
    assert claim.worker_request.request_status == "constructed_for_worker"
    assert claim.materialization_receipt.materialization_status == "persisted_and_verified"
    assert {item.name for item in run.iterdir()} == before
    with pytest.raises(FrozenInstanceError):
        claim.runnable = True


@pytest.mark.parametrize("variant", [
    r"c:\users\jeff\analysis\guided_npm_worker_request.json",
    r"C:\USERS\JEFF\ANALYSIS\GUIDED_NPM_WORKER_REQUEST.JSON",
])
def test_windows_equivalent_stored_paths_compare_equal(variant):
    authority = r"C:\Users\Jeff\Analysis\guided_npm_worker_request.json"
    verify_stored_path_style(variant, "windows_drive")
    assert stored_paths_equal(authority, variant, "windows_drive")
    parent, name = stored_parent_and_name(variant, "windows_drive")
    assert stored_paths_equal(parent, r"C:\Users\Jeff\Analysis", "windows_drive")
    assert name.lower() == GUIDED_NPM_WORKER_REQUEST_FILENAME


@pytest.mark.parametrize("candidate", [
    r"D:\Users\Jeff\Analysis\guided_npm_worker_request.json",
    r"C:\Users\Jeff\Other\guided_npm_worker_request.json",
    r"C:\Users\Jeff\Analysis\other.json",
])
def test_windows_genuinely_different_stored_paths_compare_unequal(candidate):
    authority = r"C:\Users\Jeff\Analysis\guided_npm_worker_request.json"
    assert not stored_paths_equal(authority, candidate, "windows_drive")


@pytest.mark.parametrize("candidate", [
    r"C:Users\Jeff\Analysis\guided_npm_worker_request.json",
    r"C:\Users\Jeff\..\Analysis\guided_npm_worker_request.json",
    r"C:/Users/Jeff/Analysis/guided_npm_worker_request.json",
    "/tmp/guided_npm_worker_request.json",
    r"\\server\share\guided_npm_worker_request.json",
])
def test_windows_noncanonical_or_wrong_style_paths_refuse(candidate):
    with pytest.raises(ValueError):
        verify_stored_path_style(candidate, "windows_drive")


def test_posix_stored_paths_remain_case_sensitive():
    upper = "/tmp/Run/guided_npm_worker_request.json"
    lower = "/tmp/run/guided_npm_worker_request.json"
    verify_stored_path_style(upper, "posix_absolute")
    verify_stored_path_style(lower, "posix_absolute")
    assert not stored_paths_equal(upper, lower, "posix_absolute")
    with pytest.raises(ValueError):
        verify_stored_path_style(r"C:\tmp\guided_npm_worker_request.json", "posix_absolute")


@pytest.mark.skipif(os.name != "nt", reason="requires case-insensitive live Windows path access")
def test_equivalent_windows_caller_spelling_canonicalizes_identity(tmp_path):
    execution, receipt = _prepared(tmp_path)
    _, _, canonical_claim = _claim(tmp_path, execution, receipt)
    assert isinstance(canonical_claim, GuidedNpmWorkerPrelaunchClaim)
    alternate = receipt.worker_request_artifact_path.swapcase()
    _, _, alternate_claim = _claim(tmp_path, execution, receipt, path=alternate)
    assert isinstance(alternate_claim, GuidedNpmWorkerPrelaunchClaim)
    assert alternate_claim.worker_request_artifact_path == canonical_claim.worker_request_artifact_path
    assert alternate_claim.prelaunch_freshness_evidence.worker_artifact_path == canonical_claim.prelaunch_freshness_evidence.worker_artifact_path
    assert alternate_claim.canonical_prelaunch_claim_identity == canonical_claim.canonical_prelaunch_claim_identity
    assert (
        alternate_claim.prelaunch_freshness_evidence.canonical_prelaunch_freshness_evidence_identity
        == canonical_claim.prelaunch_freshness_evidence.canonical_prelaunch_freshness_evidence_identity
    )


def test_pure_verifiers_use_declared_path_semantics_for_case_only_variants(tmp_path):
    _, _, claim = _claim(tmp_path)
    assert isinstance(claim, GuidedNpmWorkerPrelaunchClaim)
    style = claim.worker_request.execution_request.output_runtime_projection.output_base_path_style
    if style == "windows_drive":
        evidence = claim.prelaunch_freshness_evidence
        alternate_path = evidence.worker_artifact_path.swapcase()
        evidence = _reidentify_evidence(
            evidence,
            worker_artifact_path=alternate_path,
            worker_artifact_pre_facts=replace(evidence.worker_artifact_pre_facts, canonical_path=alternate_path),
            worker_artifact_opened_facts=replace(evidence.worker_artifact_opened_facts, canonical_path=alternate_path),
            worker_artifact_post_read_facts=replace(evidence.worker_artifact_post_read_facts, canonical_path=alternate_path),
            worker_artifact_final_facts=replace(evidence.worker_artifact_final_facts, canonical_path=alternate_path),
        )
        verify_guided_npm_prelaunch_freshness_evidence(evidence, claim.worker_request, claim.materialization_receipt)
        changed_claim = _reidentify_claim(
            claim,
            worker_request_artifact_path=alternate_path,
            run_directory_path=claim.run_directory_path.swapcase(),
            prelaunch_freshness_evidence=evidence,
            source_prelaunch_freshness_evidence_identity=evidence.canonical_prelaunch_freshness_evidence_identity,
        )
        verify_guided_npm_worker_prelaunch_claim(changed_claim)
    else:
        changed = _reidentify_claim(
            claim,
            worker_request_artifact_path=claim.worker_request_artifact_path.replace("/tmp/", "/TMP/", 1),
        )
        with pytest.raises(ValueError):
            verify_guided_npm_worker_prelaunch_claim(changed)


def test_malformed_inputs_refuse_before_filesystem(monkeypatch, tmp_path):
    execution, receipt = _prepared(tmp_path)
    monkeypatch.setattr(module, "_stable_read_worker_artifact", lambda *args: pytest.fail("filesystem reached"))
    _failure(
        claim_guided_npm_worker_for_prelaunch(receipt.worker_request_artifact_path, object(), current_application_build_identity=execution.application_build_identity),
        "materialization_receipt_invalid",
    )
    _failure(
        claim_guided_npm_worker_for_prelaunch(receipt.worker_request_artifact_path, receipt, current_application_build_identity=object()),
        "current_build_invalid",
    )
    _, _, result = _claim(tmp_path, execution, receipt, path="relative.json")
    _failure(result, "worker_artifact_path_invalid")


@pytest.mark.parametrize("change,category", [
    ("missing", "worker_artifact_missing"),
    ("directory", "worker_artifact_not_regular"),
    ("byte", "worker_artifact_digest_mismatch"),
    ("same_size", "worker_artifact_digest_mismatch"),
    ("truncate", "worker_artifact_size_mismatch"),
    ("append", "worker_artifact_size_mismatch"),
])
def test_worker_artifact_refusals(tmp_path, change, category):
    execution, receipt = _prepared(tmp_path)
    path = Path(receipt.worker_request_artifact_path)
    content = path.read_bytes()
    if change == "missing":
        path.unlink()
    elif change == "directory":
        path.unlink(); path.mkdir()
    elif change in {"byte", "same_size"}:
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    elif change == "truncate":
        path.write_bytes(content[:-1])
    else:
        path.write_bytes(content + b"x")
    _, _, result = _claim(tmp_path, execution, receipt)
    _failure(result, category)


def test_noncanonical_worker_bytes_refuse_after_coherent_receipt_update(tmp_path):
    execution, receipt = _prepared(tmp_path)
    path = Path(receipt.worker_request_artifact_path)
    content = path.read_bytes().replace(b'"request_status":"constructed_for_worker"', b'"request_status": "constructed_for_worker"', 1)
    path.write_bytes(content)
    changed = _reidentify_receipt(
        receipt,
        worker_request_artifact_sha256=__import__("hashlib").sha256(content).hexdigest(),
        worker_request_artifact_size_bytes=len(content),
    )
    _, _, result = _claim(tmp_path, execution, changed)
    _failure(result, "worker_artifact_noncanonical")


def test_worker_symlink_refuses_when_supported(tmp_path):
    execution, receipt = _prepared(tmp_path)
    path = Path(receipt.worker_request_artifact_path)
    target = path.with_suffix(".target")
    path.rename(target)
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    _, _, result = _claim(tmp_path, execution, receipt)
    _failure(result, "worker_artifact_alias_invalid")


@pytest.mark.parametrize("field", ["worker_request_artifact_sha256", "worker_request_artifact_size_bytes"])
def test_receipt_observed_artifact_mismatch_refuses(tmp_path, field):
    execution, receipt = _prepared(tmp_path)
    value = "a" * 64 if field.endswith("sha256") else receipt.worker_request_artifact_size_bytes + 1
    changed = _reidentify_receipt(receipt, **{field: value})
    _, _, result = _claim(tmp_path, execution, changed)
    _failure(result, "worker_artifact_digest_mismatch" if field.endswith("sha256") else "worker_artifact_size_mismatch")


def test_structurally_invalid_receipt_and_path_mismatch_refuse_before_read(monkeypatch, tmp_path):
    execution, receipt = _prepared(tmp_path)
    monkeypatch.setattr(module, "_stable_read_worker_artifact", lambda *args: pytest.fail("artifact read"))
    _, _, result = _claim(tmp_path, execution, replace(receipt, runnable=True))
    _failure(result, "materialization_receipt_invalid")
    _, _, result = _claim(tmp_path, execution, receipt, path=os.fspath(Path(receipt.run_directory_path) / "other.json"))
    _failure(result, "worker_artifact_path_invalid")


@pytest.mark.parametrize("field,value_factory", [
    ("application_build_identity", lambda receipt: _different_build(receipt.application_build_identity)),
    ("guided_plan_identity", lambda receipt: "a" * 64),
    ("validation_revision", lambda receipt: receipt.validation_revision + 1),
])
def test_coherently_changed_receipt_authority_refuses_binding(tmp_path, field, value_factory):
    execution, receipt = _prepared(tmp_path)
    changed = _reidentify_receipt(receipt, **{field: value_factory(receipt)})
    _, _, result = _claim(tmp_path, execution, changed)
    _failure(result, "materialization_binding_mismatch")


def test_stale_receipt_identity_refuses_before_artifact_read(monkeypatch, tmp_path):
    execution, receipt = _prepared(tmp_path)
    monkeypatch.setattr(module, "_stable_read_worker_artifact", lambda *args: pytest.fail("artifact read"))
    stale = replace(receipt, canonical_materialization_receipt_identity="a" * 64)
    _, _, result = _claim(tmp_path, execution, stale)
    _failure(result, "materialization_receipt_invalid")


def test_receipt_for_another_worker_and_different_valid_authority_refuse(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"; a.mkdir(); b.mkdir()
    execution_a, receipt_a = _prepared(a)
    execution_b, receipt_b = _prepared(b)
    _, _, success = _claim(a, execution_a, receipt_a)
    assert isinstance(success, GuidedNpmWorkerPrelaunchClaim)
    _, _, wrong_receipt = _claim(a, execution_a, receipt_b, path=receipt_a.worker_request_artifact_path)
    _failure(wrong_receipt, "worker_artifact_path_invalid")
    _, _, wrong_artifact = _claim(b, execution_b, receipt_a, path=receipt_b.worker_request_artifact_path)
    _failure(wrong_artifact, "worker_artifact_path_invalid")


def test_copied_worker_artifact_never_retargets_authority(tmp_path):
    execution, receipt = _prepared(tmp_path)
    copied_dir = tmp_path / "copy"; copied_dir.mkdir()
    copied = copied_dir / GUIDED_NPM_WORKER_REQUEST_FILENAME
    shutil.copyfile(receipt.worker_request_artifact_path, copied)
    _, _, result = _claim(tmp_path, execution, receipt, path=os.fspath(copied))
    _failure(result, "worker_artifact_path_invalid")
    retargeted = _reidentify_receipt(
        receipt,
        run_directory_path=os.fspath(copied_dir),
        worker_request_artifact_path=os.fspath(copied),
    )
    _, _, result = _claim(tmp_path, execution, retargeted, path=os.fspath(copied))
    _failure(result, "worker_artifact_path_invalid")


@pytest.mark.parametrize("suffix", ["release", "revision", "dirty", "fallback"])
def test_current_build_mismatch_refuses_before_final_freshness(monkeypatch, tmp_path, suffix):
    execution, receipt = _prepared(tmp_path)
    monkeypatch.setattr(module, "verify_guided_npm_startup_artifact_live", lambda *args: pytest.fail("startup reached"))
    _, _, result = _claim(tmp_path, execution, receipt, build=_different_build(execution.application_build_identity, suffix))
    _failure(result, "current_build_mismatch")


@pytest.mark.parametrize("change,category", [
    ("missing", "startup_artifact_missing"),
    ("byte", "startup_artifact_mutated"),
    ("same_size", "startup_artifact_mutated"),
])
def test_final_startup_freshness_refuses_changes(tmp_path, change, category):
    execution, receipt = _prepared(tmp_path)
    path = Path(execution.startup_artifact_path)
    content = path.read_bytes()
    if change == "missing": path.unlink()
    else: path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    _, _, result = _claim(tmp_path, execution, receipt)
    _failure(result, category)


def test_startup_symlink_refuses_when_supported(tmp_path):
    execution, receipt = _prepared(tmp_path)
    path = Path(execution.startup_artifact_path)
    target = path.with_suffix(".target")
    path.rename(target)
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    _, _, result = _claim(tmp_path, execution, receipt)
    _failure(result, "startup_artifact_alias_invalid")


@pytest.mark.parametrize("change,category", [
    ("missing", "source_membership_missing"),
    ("extra", "source_membership_extra"),
    ("rename", "source_membership_missing"),
    ("hidden", "source_membership_extra"),
    ("temporary", "source_membership_extra"),
    ("byte", "source_file_digest_mismatch"),
    ("append", "source_file_size_mismatch"),
    ("truncate", "source_file_size_mismatch"),
    ("directory", "source_membership_missing"),
])
def test_final_source_freshness_refuses_changes(tmp_path, change, category):
    execution, receipt = _prepared(tmp_path)
    path = Path(execution.source_runtime_projection.ordered_sessions[0].source_path)
    content = path.read_bytes()
    if change == "missing": path.unlink()
    elif change == "extra": (path.parent / "extra.csv").write_bytes(b"x")
    elif change == "rename": path.rename(path.parent / "renamed.csv")
    elif change == "hidden": (path.parent / ".hidden.csv").write_bytes(b"x")
    elif change == "temporary": (path.parent / "~temp.csv").write_bytes(b"x")
    elif change == "byte": path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    elif change == "append": path.write_bytes(content + b"x")
    elif change == "truncate": path.write_bytes(content[:-1])
    else: path.unlink(); path.mkdir()
    _, _, result = _claim(tmp_path, execution, receipt)
    _failure(result, category)


def test_source_symlink_refuses_when_supported(tmp_path):
    execution, receipt = _prepared(tmp_path)
    path = Path(execution.source_runtime_projection.ordered_sessions[0].source_path)
    target = path.with_suffix(".target.csv")
    path.rename(target)
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    _, _, result = _claim(tmp_path, execution, receipt)
    _failure(result, "source_file_alias_invalid")


def test_non_csv_and_nested_csv_are_ignored_and_authority_order_is_preserved(monkeypatch, tmp_path):
    execution, receipt = _prepared(tmp_path)
    root = Path(execution.source_runtime_projection.source_root_canonical)
    (root / "notes.txt").write_text("ignored")
    nested = root / "nested"; nested.mkdir(); (nested / "ignored.csv").write_bytes(b"x")
    original = module.os.scandir
    monkeypatch.setattr(module.os, "scandir", lambda path: iter(reversed(list(original(path)))))
    _, _, claim = _claim(tmp_path, execution, receipt)
    assert isinstance(claim, GuidedNpmWorkerPrelaunchClaim)
    assert tuple(item.source_path for item in claim.prelaunch_freshness_evidence.source_freshness_evidence.ordered_verified_files) == execution.source_runtime_projection.ordered_source_paths


@pytest.mark.parametrize("field", [
    "worker_artifact_path", "worker_artifact_sha256", "worker_artifact_size_bytes",
    "startup_artifact_path", "startup_artifact_sha256", "startup_artifact_size_bytes",
    "current_application_build_identity", "freshness_status",
])
def test_outer_reidentified_prelaunch_evidence_tampering_refuses(tmp_path, field):
    _, _, claim = _claim(tmp_path)
    assert isinstance(claim, GuidedNpmWorkerPrelaunchClaim)
    evidence = claim.prelaunch_freshness_evidence
    if field.endswith("size_bytes"): value = getattr(evidence, field) + 1
    elif field.endswith("sha256"): value = "a" * 64
    elif field == "current_application_build_identity": value = _different_build(evidence.current_application_build_identity)
    elif field == "freshness_status": value = "stale"
    else: value = getattr(evidence, field) + ".changed"
    changed = _reidentify_evidence(evidence, **{field: value})
    with pytest.raises(ValueError):
        verify_guided_npm_prelaunch_freshness_evidence(changed, claim.worker_request, claim.materialization_receipt)


def test_worker_artifact_metadata_and_source_freshness_identity_tampering_refuse(tmp_path):
    _, _, claim = _claim(tmp_path)
    evidence = claim.prelaunch_freshness_evidence
    facts = replace(evidence.worker_artifact_opened_facts, mtime_ns=evidence.worker_artifact_opened_facts.mtime_ns + 1)
    changed = _reidentify_evidence(evidence, worker_artifact_opened_facts=facts)
    with pytest.raises(ValueError): verify_guided_npm_prelaunch_freshness_evidence(changed, claim.worker_request, claim.materialization_receipt)
    facts = replace(
        evidence.worker_artifact_final_facts,
        canonical_path=os.fspath(Path(evidence.worker_artifact_path).parent / "different.json"),
    )
    changed = _reidentify_evidence(evidence, worker_artifact_final_facts=facts)
    with pytest.raises(ValueError): verify_guided_npm_prelaunch_freshness_evidence(changed, claim.worker_request, claim.materialization_receipt)
    startup_facts = replace(
        evidence.startup_artifact_final_facts,
        canonical_path=evidence.startup_artifact_path + ".different",
    )
    changed = _reidentify_evidence(evidence, startup_artifact_final_facts=startup_facts)
    with pytest.raises(ValueError): verify_guided_npm_prelaunch_freshness_evidence(changed, claim.worker_request, claim.materialization_receipt)
    source = replace(evidence.source_freshness_evidence, canonical_live_freshness_evidence_identity="a" * 64)
    changed = _reidentify_evidence(evidence, source_freshness_evidence=source)
    with pytest.raises(ValueError): verify_guided_npm_prelaunch_freshness_evidence(changed, claim.worker_request, claim.materialization_receipt)


@pytest.mark.parametrize("field", [
    "source_worker_request_identity", "source_execution_request_identity",
    "source_materialization_receipt_identity", "source_prelaunch_freshness_evidence_identity",
    "worker_request_artifact_path", "worker_request_artifact_sha256",
    "worker_request_artifact_size_bytes", "application_build_identity", "guided_plan_identity",
    "validation_revision", "execution_mode", "run_directory_path", "claim_status",
    "launch_status", "execution_status", "completion_status", "runnable",
])
def test_outer_reidentified_prelaunch_claim_tampering_refuses(tmp_path, field):
    _, _, claim = _claim(tmp_path)
    assert isinstance(claim, GuidedNpmWorkerPrelaunchClaim)
    current = getattr(claim, field)
    if field.endswith("size_bytes") or field == "validation_revision": value = current + 1
    elif field == "application_build_identity": value = _different_build(current)
    elif field == "runnable": value = True
    elif field.endswith("identity") or field.endswith("sha256"): value = "a" * 64
    else: value = str(current) + ".changed"
    changed = _reidentify_claim(claim, **{field: value})
    with pytest.raises(ValueError): verify_guided_npm_worker_prelaunch_claim(changed)


def test_cancellation_returns_no_claim_and_changes_nothing(tmp_path):
    execution, receipt = _prepared(tmp_path)
    before = Path(receipt.worker_request_artifact_path).read_bytes()
    _, _, result = _claim(tmp_path, execution, receipt, cancellation_check=lambda: True)
    assert isinstance(result, GuidedNpmWorkerPrelaunchClaimCancelled)
    assert Path(receipt.worker_request_artifact_path).read_bytes() == before


def test_mutation_during_worker_streaming_refuses(monkeypatch, tmp_path):
    execution, receipt = _prepared(tmp_path)
    target = Path(receipt.worker_request_artifact_path)
    original_open = Path.open

    class Reader:
        def __init__(self, handle): self.handle, self.done = handle, False
        def __enter__(self): self.handle.__enter__(); return self
        def __exit__(self, *args): return self.handle.__exit__(*args)
        def fileno(self): return self.handle.fileno()
        def read(self, size=-1):
            block = self.handle.read(size)
            if block and not self.done:
                self.done = True; now = target.stat().st_mtime_ns; os.utime(target, ns=(now + 1_000_000, now + 1_000_000))
            return block
    monkeypatch.setattr(Path, "open", lambda path, *a, **k: Reader(original_open(path, *a, **k)) if module._same_stored_path(os.fspath(path), os.fspath(target), "windows_drive" if os.name == "nt" else "posix_absolute") else original_open(path, *a, **k))
    _, _, result = _claim(tmp_path, execution, receipt)
    _failure(result, "worker_artifact_mutated")


def test_mutation_after_read_and_after_binding_refuses(monkeypatch, tmp_path):
    execution, receipt = _prepared(tmp_path)
    worker_path = Path(receipt.worker_request_artifact_path)
    original_gate = module._final_live_gate
    state = {"done": False}
    def mutate_before_gate(path, expected, category):
        if category == "worker_artifact_mutated" and not state["done"]:
            state["done"] = True; now = worker_path.stat().st_mtime_ns; os.utime(worker_path, ns=(now + 1_000_000, now + 1_000_000))
        return original_gate(path, expected, category)
    monkeypatch.setattr(module, "_final_live_gate", mutate_before_gate)
    _, _, result = _claim(tmp_path, execution, receipt)
    _failure(result, "worker_artifact_mutated")


def test_mutation_during_claim_construction_final_gate_refuses(monkeypatch, tmp_path):
    execution, receipt = _prepared(tmp_path)
    source = Path(execution.source_runtime_projection.ordered_sessions[0].source_path)
    original_identity = module.compute_guided_npm_worker_prelaunch_claim_identity
    def mutate(value):
        digest = original_identity(value)
        now = source.stat().st_mtime_ns; os.utime(source, ns=(now + 1_000_000, now + 1_000_000))
        return digest
    monkeypatch.setattr(module, "compute_guided_npm_worker_prelaunch_claim_identity", mutate)
    _, _, result = _claim(tmp_path, execution, receipt)
    _failure(result, "source_file_mutated")


def test_pure_verifiers_use_no_filesystem(monkeypatch, tmp_path):
    _, _, claim = _claim(tmp_path)
    assert isinstance(claim, GuidedNpmWorkerPrelaunchClaim)
    for name in ("exists", "stat", "resolve", "open", "read_bytes"):
        monkeypatch.setattr(Path, name, lambda *a, _name=name, **k: pytest.fail(f"Path.{_name}"))
    for name in ("stat", "lstat", "scandir"):
        monkeypatch.setattr(os, name, lambda *a, _name=name, **k: pytest.fail(f"os.{_name}"))
    verify_guided_npm_prelaunch_freshness_evidence(claim.prelaunch_freshness_evidence, claim.worker_request, claim.materialization_receipt)
    verify_guided_npm_worker_prelaunch_claim(claim)


def test_no_source_reinterpretation_or_process_launch(monkeypatch, tmp_path):
    execution, receipt = _prepared(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Popen"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("run"))
    monkeypatch.setattr(multiprocessing, "Process", lambda *a, **k: pytest.fail("Process"))
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: pytest.fail("Thread"))
    _, _, claim = _claim(tmp_path, execution, receipt)
    assert isinstance(claim, GuidedNpmWorkerPrelaunchClaim)
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert all(token not in source for token in ("pandas", "read_csv", "Pipeline(", "extract_features(", "subprocess.", "multiprocessing."))
