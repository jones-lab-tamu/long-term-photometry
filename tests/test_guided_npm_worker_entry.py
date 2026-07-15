from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_npm_worker_entry as entry_module
from photometry_pipeline.guided_npm_worker_acknowledgement import (
    read_and_verify_guided_npm_consumed_authority_receipt,
)
from photometry_pipeline.guided_npm_worker_entry import (
    GUIDED_NPM_WORKER_ENTRY_FAILED,
    GUIDED_NPM_WORKER_ENTRY_REFUSED,
    GUIDED_NPM_WORKER_ENTRY_SUCCESS,
    GUIDED_NPM_WORKER_ENTRY_TERMINAL_PUBLICATION_FAILED,
    run_guided_npm_worker_to_terminal_receipt,
)
from photometry_pipeline.guided_npm_worker_terminal import (
    GUIDED_NPM_TERMINAL_RECEIPT_FILENAME,
    expected_guided_npm_terminal_receipt_path,
    read_guided_npm_worker_terminal_receipt,
)
from photometry_pipeline.pipeline import Pipeline
from tests.test_guided_npm_worker_acknowledgement import _transaction


def _real(tmp_path: Path, *, pid: int | None = None):
    claim, worker, runtime, invocation, context, start, evidence, _ = _transaction(
        tmp_path, pid=pid or os.getpid()
    )
    return claim, worker, runtime, invocation, context, start, evidence


def test_success_returns_zero_and_publishes_exactly_one_receipt(tmp_path):
    claim, worker, runtime, invocation, context, start, evidence = _real(tmp_path)
    code, receipt = run_guided_npm_worker_to_terminal_receipt(worker, launch_context=context)
    assert code == GUIDED_NPM_WORKER_ENTRY_SUCCESS
    assert receipt.terminal_outcome == "completed"
    path = expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    assert Path(path).is_file()
    assert Path(path).name == GUIDED_NPM_TERMINAL_RECEIPT_FILENAME


def test_authority_refusal_before_launch_context_verification_returns_two(tmp_path):
    claim, worker, runtime, invocation, context, start, evidence = _real(tmp_path)
    assert (
        entry_module.main(
            ["--guided-npm-worker-request", worker.worker_request_artifact_path]
        )
        == GUIDED_NPM_WORKER_ENTRY_REFUSED
    )
    # No terminal receipt destination is even known at this refusal point.
    assert not Path(
        expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    ).exists()


def test_execution_failure_with_valid_failure_receipt_returns_three(tmp_path, monkeypatch):
    claim, worker, runtime, invocation, context, start, evidence = _real(tmp_path)
    monkeypatch.setattr(
        Pipeline, "run_pass_2", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    code, receipt = run_guided_npm_worker_to_terminal_receipt(worker, launch_context=context)
    assert code == GUIDED_NPM_WORKER_ENTRY_FAILED
    assert receipt.terminal_outcome == "failed_after_consumed_authority"
    assert receipt.completion_status == "failed"
    path = expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    assert Path(path).is_file()
    reread = read_guided_npm_worker_terminal_receipt(path, worker_request=worker)
    assert reread == receipt


def test_failure_before_consumed_authority_has_no_consumed_identity(tmp_path, monkeypatch):
    claim, worker, runtime, invocation, context, start, evidence = _real(tmp_path)
    monkeypatch.setattr(
        Pipeline, "run_pass_1", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    code, receipt = run_guided_npm_worker_to_terminal_receipt(worker, launch_context=context)
    assert code == GUIDED_NPM_WORKER_ENTRY_FAILED
    assert receipt.terminal_outcome == "failed_before_consumed_authority"
    assert receipt.source_consumed_authority_receipt_identity is None
    assert receipt.consumed_authority_status == "not_available"
    assert not (Path(worker.run_directory_path) / "guided_npm_consumed_authority_receipt.json").exists()


def test_output_finalization_failure_returns_three_with_finalization_outcome(tmp_path, monkeypatch):
    claim, worker, runtime, invocation, context, start, evidence = _real(tmp_path)
    monkeypatch.setattr(
        entry_module,
        "build_guided_npm_required_output_evidence",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("evidence boom")),
    )
    code, receipt = run_guided_npm_worker_to_terminal_receipt(worker, launch_context=context)
    assert code == GUIDED_NPM_WORKER_ENTRY_FAILED
    assert receipt.terminal_outcome == "failed_during_output_finalization"
    assert receipt.consumed_authority_status == "verified"
    assert receipt.completion_status == "failed"
    assert receipt.output_evidence == ()
    assert receipt.completed_run_evidence_identity is None


def test_success_computation_with_failed_terminal_publication_is_not_zero(tmp_path, monkeypatch):
    claim, worker, runtime, invocation, context, start, evidence = _real(tmp_path)
    path = expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    Path(path).write_text("pre-existing", encoding="utf-8")
    code, receipt = run_guided_npm_worker_to_terminal_receipt(worker, launch_context=context)
    assert code == GUIDED_NPM_WORKER_ENTRY_TERMINAL_PUBLICATION_FAILED
    assert code != GUIDED_NPM_WORKER_ENTRY_SUCCESS
    assert receipt is None
    assert Path(path).read_text(encoding="utf-8") == "pre-existing"


def test_failure_computation_with_failed_terminal_publication_returns_four(tmp_path, monkeypatch):
    claim, worker, runtime, invocation, context, start, evidence = _real(tmp_path)
    path = expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    Path(path).write_text("pre-existing", encoding="utf-8")
    monkeypatch.setattr(
        Pipeline, "run_pass_2", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    code, receipt = run_guided_npm_worker_to_terminal_receipt(worker, launch_context=context)
    assert code == GUIDED_NPM_WORKER_ENTRY_TERMINAL_PUBLICATION_FAILED
    assert receipt is None
    assert Path(path).read_text(encoding="utf-8") == "pre-existing"


def test_no_false_success_when_required_output_missing(tmp_path, monkeypatch):
    claim, worker, runtime, invocation, context, start, evidence = _real(tmp_path)
    original_run = Pipeline.run_guided_npm_authorized

    def run_then_delete(self, authorized_runtime, output_dir, **kwargs):
        result = original_run(self, authorized_runtime, output_dir, **kwargs)
        os.remove(os.path.join(output_dir, "run_metadata.json"))
        return result

    monkeypatch.setattr(Pipeline, "run_guided_npm_authorized", run_then_delete)
    code, receipt = run_guided_npm_worker_to_terminal_receipt(worker, launch_context=context)
    assert code != GUIDED_NPM_WORKER_ENTRY_SUCCESS
    assert receipt.terminal_outcome == "failed_during_output_finalization"


def test_no_automatic_retry_pipeline_dispatched_exactly_once(tmp_path, monkeypatch):
    claim, worker, runtime, invocation, context, start, evidence = _real(tmp_path)
    calls = []
    original_run = Pipeline.run_guided_npm_authorized

    def counting_run(self, authorized_runtime, output_dir, **kwargs):
        calls.append(1)
        return original_run(self, authorized_runtime, output_dir, **kwargs)

    monkeypatch.setattr(Pipeline, "run_guided_npm_authorized", counting_run)
    run_guided_npm_worker_to_terminal_receipt(worker, launch_context=context)
    assert len(calls) == 1


def test_malformed_launch_context_or_worker_refuses_before_terminal_destination(tmp_path):
    claim, worker, runtime, invocation, context, start, evidence = _real(tmp_path)
    assert (
        entry_module.main(
            [
                "--guided-npm-worker-request",
                worker.worker_request_artifact_path,
                "--guided-npm-launch-context",
                os.path.abspath("does-not-exist.json"),
            ]
        )
        == GUIDED_NPM_WORKER_ENTRY_REFUSED
    )


def test_main_delegates_exit_code_from_terminal_lifecycle(monkeypatch, tmp_path):
    claim, worker, runtime, invocation, context, start, evidence = _real(tmp_path)
    monkeypatch.setattr(
        entry_module, "load_verified_guided_npm_worker_request", lambda *a, **k: worker
    )
    monkeypatch.setattr(
        entry_module, "read_guided_npm_worker_launch_context", lambda *a, **k: context
    )
    monkeypatch.setattr(
        entry_module,
        "run_guided_npm_worker_to_terminal_receipt",
        lambda *a, **k: (GUIDED_NPM_WORKER_ENTRY_TERMINAL_PUBLICATION_FAILED, None),
    )
    result = entry_module.main(
        [
            "--guided-npm-worker-request",
            worker.worker_request_artifact_path,
            "--guided-npm-launch-context",
            invocation.launch_context_artifact_path,
        ]
    )
    assert result == GUIDED_NPM_WORKER_ENTRY_TERMINAL_PUBLICATION_FAILED


def test_parent_never_waits_polls_or_communicates():
    import inspect

    source = inspect.getsource(entry_module)
    for forbidden in (".wait(", ".poll(", ".communicate(", ".terminate(", ".kill("):
        assert forbidden not in source
