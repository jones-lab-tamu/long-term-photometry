from __future__ import annotations

from dataclasses import replace
import hashlib
import multiprocessing
import os
from pathlib import Path
import subprocess
import threading
import inspect
import shutil

import pytest

import photometry_pipeline.guided_npm_worker_request_materialization as module
from photometry_pipeline.guided_npm_worker_request import (
    GUIDED_NPM_WORKER_REQUEST_FILENAME,
    compute_guided_npm_worker_request_identity,
    decode_canonical_guided_npm_worker_request_bytes,
)
from photometry_pipeline.guided_npm_worker_request_materialization import (
    GuidedNpmWorkerRequestMaterializationCancelled,
    GuidedNpmWorkerRequestMaterializationFailure,
    GuidedNpmWorkerRequestMaterializationReceipt,
    compute_guided_npm_worker_request_materialization_receipt_identity,
    materialize_guided_npm_worker_request,
    verify_guided_npm_worker_request_materialization_binding,
    verify_guided_npm_worker_request_materialization_receipt,
)
from photometry_pipeline.guided_production_mapping import build_application_build_identity
from tests.test_guided_npm_production_execution_request import _request


def _materialize(tmp_path, execution=None, **kwargs):
    if execution is None:
        _, _, execution = _request(tmp_path)
    return execution, materialize_guided_npm_worker_request(
        execution,
        current_application_build_identity=kwargs.pop("build", execution.application_build_identity),
        **kwargs,
    )


def _failure(result, category):
    assert isinstance(result, GuidedNpmWorkerRequestMaterializationFailure)
    assert result.blocking_issues[0].category == category


def _reidentify_receipt(receipt, **changes):
    changed = replace(
        receipt,
        **changes,
        canonical_materialization_receipt_identity="0" * 64,
    )
    return replace(
        changed,
        canonical_materialization_receipt_identity=(
            compute_guided_npm_worker_request_materialization_receipt_identity(changed)
        ),
    )


def _reidentify_worker(worker, **changes):
    changed = replace(worker, **changes, canonical_worker_request_identity="0" * 64)
    return replace(
        changed,
        canonical_worker_request_identity=compute_guided_npm_worker_request_identity(changed),
    )


def _receipt_and_worker(tmp_path):
    execution, receipt = _materialize(tmp_path)
    assert isinstance(receipt, GuidedNpmWorkerRequestMaterializationReceipt)
    worker = decode_canonical_guided_npm_worker_request_bytes(
        Path(receipt.worker_request_artifact_path).read_bytes()
    )
    return execution, receipt, worker


def _different_valid_build(build):
    return build_application_build_identity(
        distribution_name=build.distribution_name,
        distribution_version=build.distribution_version + ".binding-test",
        source_revision_kind=build.source_revision_kind,
        source_revision=build.source_revision,
        source_tree_state=build.source_tree_state,
        source_tree_digest=build.source_tree_digest,
        build_artifact_digest=build.build_artifact_digest,
        identity_provider_version=build.identity_provider_version,
    )


def test_materializes_exact_canonical_artifact_and_receipt(tmp_path):
    execution, result = _materialize(tmp_path)
    assert isinstance(result, GuidedNpmWorkerRequestMaterializationReceipt)
    verify_guided_npm_worker_request_materialization_receipt(result)
    path = Path(result.worker_request_artifact_path)
    assert path.name == GUIDED_NPM_WORKER_REQUEST_FILENAME
    assert path.parent == Path(execution.output_runtime_projection.run_directory_path)
    content = path.read_bytes()
    restored = decode_canonical_guided_npm_worker_request_bytes(content)
    assert hashlib.sha256(content).hexdigest() == result.worker_request_artifact_sha256
    assert len(content) == result.worker_request_artifact_size_bytes
    assert restored.execution_request == execution
    assert restored.canonical_worker_request_identity == result.readback_worker_request_identity
    assert restored.request_status == "constructed_for_worker"
    assert not hasattr(restored, "materialization_status")
    verify_guided_npm_worker_request_materialization_binding(result, restored)
    assert (result.materialization_status, result.launch_status, result.execution_status, result.completion_status, result.runnable) == (
        "persisted_and_verified", "not_launched", "not_started", "not_available", False,
    )


def test_materializer_binds_receipt_to_decoded_readback_before_success(monkeypatch, tmp_path):
    calls = []
    original = module.verify_guided_npm_worker_request_materialization_binding

    def recording_binding(receipt, worker):
        calls.append((receipt, worker))
        return original(receipt, worker)

    monkeypatch.setattr(
        module,
        "verify_guided_npm_worker_request_materialization_binding",
        recording_binding,
    )
    _, result = _materialize(tmp_path)
    assert isinstance(result, GuidedNpmWorkerRequestMaterializationReceipt)
    assert len(calls) == 1
    receipt, restored = calls[0]
    assert receipt == result
    assert restored.canonical_worker_request_identity == result.readback_worker_request_identity


def test_materializer_binding_failure_refuses_and_removes_owned_artifact(monkeypatch, tmp_path):
    _, _, execution = _request(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_guided_npm_worker_request_materialization_binding",
        lambda *args: (_ for _ in ()).throw(ValueError("injected_binding_failure")),
    )
    _, result = _materialize(tmp_path, execution)
    _failure(result, "worker_request_identity_mismatch")
    assert not Path(
        execution.output_runtime_projection.run_directory_path,
        GUIDED_NPM_WORKER_REQUEST_FILENAME,
    ).exists()


def test_invalid_request_and_current_build_refuse_before_filesystem(monkeypatch, tmp_path):
    _, _, execution = _request(tmp_path)
    monkeypatch.setattr(module, "_verify_startup_artifact", lambda *args: pytest.fail("filesystem reached"))
    _failure(materialize_guided_npm_worker_request(object(), current_application_build_identity=execution.application_build_identity), "execution_request_missing_or_invalid")
    _failure(materialize_guided_npm_worker_request(execution, current_application_build_identity=object()), "execution_request_build_mismatch")


@pytest.mark.parametrize("change,category", [
    ("missing", "source_membership_missing"),
    ("extra", "source_membership_extra"),
    ("rename", "source_membership_missing"),
    ("hidden", "source_membership_extra"),
    ("temporary", "source_membership_extra"),
    ("directory", "source_membership_missing"),
])
def test_membership_refusals(tmp_path, change, category):
    _, _, execution = _request(tmp_path)
    session = execution.source_runtime_projection.ordered_sessions[0]
    path = Path(session.source_path)
    if change == "missing":
        path.unlink()
    elif change == "extra":
        (path.parent / "extra.CSV").write_bytes(b"extra")
    elif change == "rename":
        path.rename(path.parent / "renamed.csv")
    elif change == "hidden":
        (path.parent / ".hidden.csv").write_bytes(b"hidden")
    elif change == "temporary":
        (path.parent / "~temporary.csv").write_bytes(b"temporary")
    else:
        path.unlink()
        path.mkdir()
    _, result = _materialize(tmp_path, execution)
    _failure(result, category)


def test_nested_csv_non_csv_and_directory_are_ignored(tmp_path):
    _, _, execution = _request(tmp_path)
    root = Path(execution.source_runtime_projection.source_root_canonical)
    (root / "notes.txt").write_text("ignored")
    nested = root / "nested"
    nested.mkdir()
    (nested / "ignored.csv").write_bytes(b"ignored")
    _, result = _materialize(tmp_path, execution)
    assert isinstance(result, GuidedNpmWorkerRequestMaterializationReceipt)


def test_symlinked_csv_refuses_when_supported(tmp_path):
    _, _, execution = _request(tmp_path)
    root = Path(execution.source_runtime_projection.source_root_canonical)
    link = root / "alias.csv"
    try:
        link.symlink_to(Path(execution.source_runtime_projection.ordered_sessions[0].source_path))
    except OSError:
        pytest.skip("symlink creation unavailable")
    _, result = _materialize(tmp_path, execution)
    _failure(result, "source_file_alias_invalid")


@pytest.mark.parametrize("change,category", [
    ("byte", "source_file_digest_mismatch"),
    ("append", "source_file_size_mismatch"),
    ("truncate", "source_file_size_mismatch"),
])
def test_source_content_changes_refuse(tmp_path, change, category):
    _, _, execution = _request(tmp_path)
    path = Path(execution.source_runtime_projection.ordered_sessions[0].source_path)
    content = path.read_bytes()
    if change == "byte":
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    elif change == "append":
        path.write_bytes(content + b"x")
    else:
        path.write_bytes(content[:-1])
    _, result = _materialize(tmp_path, execution)
    _failure(result, category)


def test_startup_artifact_missing_and_changed_refuse(tmp_path):
    _, _, execution = _request(tmp_path)
    startup = Path(execution.startup_artifact_path)
    startup.unlink()
    _, result = _materialize(tmp_path, execution)
    _failure(result, "startup_artifact_missing")

    other = tmp_path / "other"
    other.mkdir()
    _, _, execution = _request(other)
    startup = Path(execution.startup_artifact_path)
    content = startup.read_bytes()
    startup.write_bytes(content.replace(b'"runnable":false', b'"runnable":true ', 1))
    _, result = _materialize(tmp_path, execution)
    assert isinstance(result, GuidedNpmWorkerRequestMaterializationFailure)
    assert result.blocking_issues[0].category.startswith("startup_artifact_")


def test_startup_replacement_during_public_verification_refuses(monkeypatch, tmp_path):
    _, _, execution = _request(tmp_path)
    original = module.verify_guided_npm_startup_artifact_path

    def replace_after(path):
        verified = original(path)
        target = Path(path)
        content = target.read_bytes()
        target.unlink()
        target.write_bytes(content)
        return verified

    monkeypatch.setattr(module, "verify_guided_npm_startup_artifact_path", replace_after)
    _, result = _materialize(tmp_path, execution)
    _failure(result, "startup_artifact_mutated")


def test_source_mutation_after_hash_before_publication_refuses(monkeypatch, tmp_path):
    _, _, execution = _request(tmp_path)
    target = Path(execution.source_runtime_projection.ordered_sessions[0].source_path)
    original = module._verify_run_directory

    def mutate(request):
        result = original(request)
        os.utime(target, None)
        return result

    monkeypatch.setattr(module, "_verify_run_directory", mutate)
    _, result = _materialize(tmp_path, execution)
    _failure(result, "source_file_mutated")


def test_mtime_mutation_during_streaming_hash_refuses(monkeypatch, tmp_path):
    _, _, execution = _request(tmp_path)
    target = Path(execution.source_runtime_projection.ordered_sessions[0].source_path)
    original_open = Path.open

    class MutatingReader:
        def __init__(self, handle):
            self.handle = handle
            self.mutated = False

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def fileno(self):
            return self.handle.fileno()

        def read(self, size=-1):
            block = self.handle.read(size)
            if block and not self.mutated:
                self.mutated = True
                current = target.stat().st_mtime_ns
                os.utime(target, ns=(current + 1_000_000, current + 1_000_000))
            return block

    def patched_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return MutatingReader(handle) if module._same_path(path, target) else handle

    monkeypatch.setattr(Path, "open", patched_open)
    with pytest.raises(module._Refusal) as caught:
        module._stable_read(target, None)
    assert caught.value.category == "source_file_mutated"


def test_discovery_order_does_not_change_authority_order(monkeypatch, tmp_path):
    _, _, execution = _request(tmp_path)
    original = module.os.scandir

    def reversed_scandir(path):
        return iter(reversed(list(original(path))))

    monkeypatch.setattr(module.os, "scandir", reversed_scandir)
    _, result = _materialize(tmp_path, execution)
    assert isinstance(result, GuidedNpmWorkerRequestMaterializationReceipt)
    worker = decode_canonical_guided_npm_worker_request_bytes(Path(result.worker_request_artifact_path).read_bytes())
    assert tuple(item.source_path for item in worker.live_freshness_evidence.ordered_verified_files) == execution.source_runtime_projection.ordered_source_paths


@pytest.mark.parametrize("entry,category", [
    ("unknown.txt", "run_directory_dirty"),
    (GUIDED_NPM_WORKER_REQUEST_FILENAME, "worker_request_artifact_conflict"),
])
def test_dirty_run_directory_and_conflict_refuse(tmp_path, entry, category):
    _, _, execution = _request(tmp_path)
    run = Path(execution.output_runtime_projection.run_directory_path)
    (run / entry).write_bytes(b"foreign")
    _, result = _materialize(tmp_path, execution)
    _failure(result, category)
    assert (run / entry).read_bytes() == b"foreign"


@pytest.mark.parametrize("helper,category", [
    ("_write_temp", "worker_request_write_failed"),
    ("_publish_no_replace", "worker_request_publish_failed"),
    ("_fsync_directory", "worker_request_flush_failed"),
])
def test_atomic_failure_cleanup_preserves_startup(monkeypatch, tmp_path, helper, category):
    _, _, execution = _request(tmp_path)
    run = Path(execution.output_runtime_projection.run_directory_path)
    startup = Path(execution.startup_artifact_path).read_bytes()

    def fail(*args, **kwargs):
        module._refuse(category, "publication", "injected", "injected")

    monkeypatch.setattr(module, helper, fail)
    _, result = _materialize(tmp_path, execution)
    _failure(result, category)
    assert Path(execution.startup_artifact_path).read_bytes() == startup
    assert not (run / GUIDED_NPM_WORKER_REQUEST_FILENAME).exists()
    assert not list(run.glob("*.tmp"))


def test_cancellation_before_and_after_publication_cleans_owned_state(monkeypatch, tmp_path):
    _, _, execution = _request(tmp_path)
    _, result = _materialize(tmp_path, execution, cancellation_check=lambda: True)
    assert isinstance(result, GuidedNpmWorkerRequestMaterializationCancelled)

    later = tmp_path / "later"
    later.mkdir()
    _, _, execution = _request(later)
    state = {"cancel": False}
    original = module._publish_no_replace

    def publish(*args):
        original(*args)
        state["cancel"] = True

    monkeypatch.setattr(module, "_publish_no_replace", publish)
    _, result = _materialize(tmp_path, execution, cancellation_check=lambda: state["cancel"])
    assert isinstance(result, GuidedNpmWorkerRequestMaterializationCancelled)
    assert not Path(execution.output_runtime_projection.run_directory_path, GUIDED_NPM_WORKER_REQUEST_FILENAME).exists()


def test_no_parser_pipeline_or_worker_launch(monkeypatch, tmp_path):
    _, _, execution = _request(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("worker launched"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("subprocess run"))
    monkeypatch.setattr(multiprocessing, "Process", lambda *a, **k: pytest.fail("process created"))
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: pytest.fail("thread created"))
    _, result = _materialize(tmp_path, execution)
    assert isinstance(result, GuidedNpmWorkerRequestMaterializationReceipt)


def test_implementation_has_no_parser_loader_analysis_or_wrapper_boundary():
    source = inspect.getsource(module)
    forbidden = (
        "pandas", "read_csv", "resolve_npm_support_geometry(",
        "extract_features(", "Pipeline(", "subprocess.", "multiprocessing.",
    )
    assert all(value not in source for value in forbidden)
    wrapper = Path("analyze_photometry.py").read_text(encoding="utf-8")
    assert "guided_npm_worker_request_materialization" not in wrapper


def test_receipt_tampering_refuses(tmp_path):
    _, result = _materialize(tmp_path)
    assert isinstance(result, GuidedNpmWorkerRequestMaterializationReceipt)
    with pytest.raises(ValueError):
        verify_guided_npm_worker_request_materialization_receipt(replace(result, runnable=True))


@pytest.mark.parametrize(
    "field,value_factory,reidentify",
    [
        ("receipt_schema_version", lambda receipt: "v2", True),
        ("source_execution_request_identity", lambda receipt: "A" * 64, True),
        ("run_directory_path", lambda receipt: "relative-run", True),
        ("worker_request_artifact_path", lambda receipt: os.fspath(Path(receipt.run_directory_path).parent / GUIDED_NPM_WORKER_REQUEST_FILENAME), True),
        ("run_directory_path", lambda receipt: os.fspath(Path(receipt.run_directory_path).parent), True),
        ("worker_request_artifact_path", lambda receipt: os.fspath(Path(receipt.run_directory_path) / "wrong.json"), True),
        ("worker_request_artifact_sha256", lambda receipt: "A" * 64, True),
        ("worker_request_artifact_size_bytes", lambda receipt: 0, True),
        ("worker_request_artifact_size_bytes", lambda receipt: True, True),
        ("source_worker_request_identity", lambda receipt: "a" * 64, True),
        ("readback_worker_request_identity", lambda receipt: "a" * 64, True),
        ("source_execution_request_identity", lambda receipt: "a" * 64, False),
        ("source_live_freshness_evidence_identity", lambda receipt: "a" * 64, False),
        ("application_build_identity", lambda receipt: replace(receipt.application_build_identity, canonical_identity="a" * 64), True),
        ("guided_plan_identity", lambda receipt: "A" * 64, True),
        ("validation_revision", lambda receipt: -1, True),
        ("validation_revision", lambda receipt: True, True),
        ("materialization_status", lambda receipt: "not_persisted", True),
        ("launch_status", lambda receipt: "launched", True),
        ("execution_status", lambda receipt: "running", True),
        ("completion_status", lambda receipt: "completed", True),
        ("runnable", lambda receipt: True, True),
    ],
)
def test_outer_receipt_tampering_refuses(tmp_path, field, value_factory, reidentify):
    _, receipt = _materialize(tmp_path)
    assert isinstance(receipt, GuidedNpmWorkerRequestMaterializationReceipt)
    changes = {field: value_factory(receipt)}
    changed = _reidentify_receipt(receipt, **changes) if reidentify else replace(receipt, **changes)
    with pytest.raises((TypeError, ValueError)):
        verify_guided_npm_worker_request_materialization_receipt(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_request_artifact_sha256", "a" * 64),
        ("worker_request_artifact_size_bytes", 1),
        ("source_execution_request_identity", "a" * 64),
        ("source_live_freshness_evidence_identity", "a" * 64),
    ],
)
def test_receipt_identity_binds_artifact_and_provenance_fields(tmp_path, field, value):
    _, receipt = _materialize(tmp_path)
    assert isinstance(receipt, GuidedNpmWorkerRequestMaterializationReceipt)
    with pytest.raises(ValueError):
        verify_guided_npm_worker_request_materialization_receipt(replace(receipt, **{field: value}))


def test_stale_receipt_identity_refuses(tmp_path):
    _, receipt = _materialize(tmp_path)
    assert isinstance(receipt, GuidedNpmWorkerRequestMaterializationReceipt)
    with pytest.raises(ValueError, match="materialization_receipt_invalid"):
        verify_guided_npm_worker_request_materialization_receipt(
            replace(receipt, canonical_materialization_receipt_identity="a" * 64)
        )


def test_copied_worker_bytes_do_not_self_attest_materialization(tmp_path):
    _, receipt, original_worker = _receipt_and_worker(tmp_path)
    original = Path(receipt.worker_request_artifact_path)
    copied = tmp_path / "arbitrary-copy" / GUIDED_NPM_WORKER_REQUEST_FILENAME
    copied.parent.mkdir()
    shutil.copyfile(original, copied)

    worker = decode_canonical_guided_npm_worker_request_bytes(copied.read_bytes())
    assert worker.request_status == "constructed_for_worker"
    assert not hasattr(worker, "materialization_status")
    assert receipt.worker_request_artifact_path != os.fspath(copied)
    verify_guided_npm_worker_request_materialization_receipt(receipt)
    verify_guided_npm_worker_request_materialization_binding(receipt, worker)
    assert worker == original_worker

    copied_receipt = _reidentify_receipt(
        receipt,
        run_directory_path=os.fspath(copied.parent),
        worker_request_artifact_path=os.fspath(copied),
    )
    verify_guided_npm_worker_request_materialization_receipt(copied_receipt)
    with pytest.raises(ValueError, match="materialization_binding_run_directory_mismatch"):
        verify_guided_npm_worker_request_materialization_binding(copied_receipt, worker)


@pytest.mark.parametrize(
    "changes_factory,detail_code",
    [
        (
            lambda receipt, tmp_path: {
                "run_directory_path": os.fspath(tmp_path / "retargeted"),
                "worker_request_artifact_path": os.fspath(tmp_path / "retargeted" / GUIDED_NPM_WORKER_REQUEST_FILENAME),
            },
            "materialization_binding_run_directory_mismatch",
        ),
        (
            lambda receipt, tmp_path: {
                "source_worker_request_identity": "a" * 64,
                "readback_worker_request_identity": "a" * 64,
            },
            "materialization_binding_worker_identity_mismatch",
        ),
        (lambda receipt, tmp_path: {"source_execution_request_identity": "a" * 64}, "materialization_binding_execution_identity_mismatch"),
        (lambda receipt, tmp_path: {"source_live_freshness_evidence_identity": "a" * 64}, "materialization_binding_freshness_identity_mismatch"),
        (lambda receipt, tmp_path: {"application_build_identity": _different_valid_build(receipt.application_build_identity)}, "materialization_binding_build_mismatch"),
        (lambda receipt, tmp_path: {"guided_plan_identity": "a" * 64}, "materialization_binding_plan_mismatch"),
        (lambda receipt, tmp_path: {"validation_revision": receipt.validation_revision + 1}, "materialization_binding_revision_mismatch"),
    ],
)
def test_coherent_outer_reidentified_receipt_passes_structure_but_refuses_binding(
    tmp_path, changes_factory, detail_code
):
    _, receipt, worker = _receipt_and_worker(tmp_path)
    changed = _reidentify_receipt(receipt, **changes_factory(receipt, tmp_path))
    verify_guided_npm_worker_request_materialization_receipt(changed)
    with pytest.raises(ValueError, match=detail_code):
        verify_guided_npm_worker_request_materialization_binding(changed, worker)


@pytest.mark.parametrize(
    "changes",
    [
        {"launch_status": "launched"},
        {"runnable": True},
    ],
)
def test_invalid_receipt_state_refuses_both_structural_and_semantic_verification(tmp_path, changes):
    _, receipt, worker = _receipt_and_worker(tmp_path)
    changed = _reidentify_receipt(receipt, **changes)
    with pytest.raises(ValueError, match="materialization_receipt_invalid"):
        verify_guided_npm_worker_request_materialization_receipt(changed)
    with pytest.raises(ValueError, match="materialization_receipt_invalid"):
        verify_guided_npm_worker_request_materialization_binding(changed, worker)


def test_receipt_for_one_valid_worker_refuses_another_valid_worker(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _, receipt_a, worker_a = _receipt_and_worker(first)
    _, receipt_b, worker_b = _receipt_and_worker(second)
    verify_guided_npm_worker_request_materialization_binding(receipt_a, worker_a)
    verify_guided_npm_worker_request_materialization_binding(receipt_b, worker_b)
    with pytest.raises(ValueError, match="materialization_binding_worker_identity_mismatch"):
        verify_guided_npm_worker_request_materialization_binding(receipt_a, worker_b)


@pytest.mark.parametrize(
    "field,value_factory",
    [
        ("worker_request_artifact_path", lambda worker, other: other.worker_request_artifact_path),
        ("run_directory_path", lambda worker, other: other.run_directory_path),
        ("source_execution_request_identity", lambda worker, other: other.source_execution_request_identity),
        ("execution_request", lambda worker, other: other.execution_request),
        ("application_build_identity", lambda worker, other: _different_valid_build(worker.application_build_identity)),
        ("guided_plan_identity", lambda worker, other: "a" * 64),
        ("validation_revision", lambda worker, other: worker.validation_revision + 1),
        ("launch_status", lambda worker, other: "launched"),
        ("execution_status", lambda worker, other: "running"),
        ("completion_status", lambda worker, other: "completed"),
        ("runnable", lambda worker, other: True),
    ],
)
def test_worker_side_outer_reidentified_mismatch_refuses_binding(tmp_path, field, value_factory):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _, receipt, worker = _receipt_and_worker(first)
    _, _, other = _receipt_and_worker(second)
    changed = _reidentify_worker(worker, **{field: value_factory(worker, other)})
    with pytest.raises(ValueError):
        verify_guided_npm_worker_request_materialization_binding(receipt, changed)


def test_worker_side_freshness_identity_mismatch_refuses_binding(tmp_path):
    _, receipt, worker = _receipt_and_worker(tmp_path)
    evidence = replace(
        worker.live_freshness_evidence,
        canonical_live_freshness_evidence_identity="a" * 64,
    )
    changed = _reidentify_worker(worker, live_freshness_evidence=evidence)
    with pytest.raises(ValueError):
        verify_guided_npm_worker_request_materialization_binding(receipt, changed)


def test_structural_and_semantic_verifiers_perform_no_filesystem_access(monkeypatch, tmp_path):
    _, receipt, worker = _receipt_and_worker(tmp_path)
    for name in ("exists", "stat", "resolve", "open", "read_bytes"):
        monkeypatch.setattr(Path, name, lambda *args, _name=name, **kwargs: pytest.fail(f"Path.{_name}"))
    for name in ("stat", "lstat", "scandir"):
        monkeypatch.setattr(os, name, lambda *args, _name=name, **kwargs: pytest.fail(f"os.{_name}"))
    verify_guided_npm_worker_request_materialization_receipt(receipt)
    verify_guided_npm_worker_request_materialization_binding(receipt, worker)
