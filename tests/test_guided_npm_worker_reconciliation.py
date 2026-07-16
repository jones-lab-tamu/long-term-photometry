from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import pytest

import photometry_pipeline.guided_npm_worker_entry as entry_module
import photometry_pipeline.guided_npm_worker_reconciliation as reco_module
import photometry_pipeline.guided_npm_worker_terminal as terminal_module
from photometry_pipeline.guided_npm_worker_launch import (
    GuidedNpmLaunchedWorkerRuntime,
    GuidedNpmStartedProcessRuntime,
    GUIDED_NPM_LAUNCHER_KIND,
    launch_guided_npm_worker_runtime,
)
from photometry_pipeline.guided_npm_worker_reconciliation import (
    GuidedNpmReconciliationObservationError,
    GuidedNpmReconciliationPublicationError,
    GuidedNpmReconciliationTimeout,
    GuidedNpmWorkerReconciliationResult,
    OUTCOME_AUTHORITY_REFUSED,
    OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED,
    OUTCOME_CONSUMED_AUTHORITY_EVIDENCE_INVALID,
    OUTCOME_INDETERMINATE,
    OUTCOME_POST_LAUNCH_EVIDENCE_FAILED,
    OUTCOME_PROCESS_EXITED_ZERO_WITHOUT_TERMINAL_EVIDENCE,
    OUTCOME_PROCESS_FAILED_WITHOUT_TERMINAL_EVIDENCE,
    OUTCOME_PROCESS_IDENTITY_MISMATCH,
    OUTCOME_TERMINAL_EVIDENCE_INVALID,
    OUTCOME_TERMINAL_RECEIPT_PUBLICATION_FAILED,
    OUTCOME_VERIFIED_COMPLETED,
    OUTCOME_VERIFIED_FAILED_AFTER_CONSUMED,
    OUTCOME_VERIFIED_FAILED_BEFORE_CONSUMED,
    OUTCOME_VERIFIED_FAILED_DURING_OUTPUT_FINALIZATION,
    compute_guided_npm_worker_reconciliation_result_identity,
    decode_guided_npm_worker_reconciliation_result_bytes,
    expected_guided_npm_reconciliation_result_path,
    publish_guided_npm_worker_reconciliation_result,
    read_guided_npm_worker_reconciliation_result,
    reconcile_guided_npm_post_launch_runtime,
    reconcile_guided_npm_worker_runtime,
    serialize_guided_npm_worker_reconciliation_result,
    verify_guided_npm_worker_reconciliation_result,
)
from photometry_pipeline.guided_npm_worker_terminal import (
    expected_guided_npm_terminal_receipt_path,
    publish_guided_npm_worker_terminal_receipt,
)
from photometry_pipeline.pipeline import Pipeline
from tests.test_guided_npm_worker_acknowledgement import _transaction


class FakeProcessHandle:
    def __init__(self, pid, *, wait_result=0, wait_exception=None):
        self.pid = pid
        self._wait_result = wait_result
        self._wait_exception = wait_exception
        self.wait_calls = 0

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self._wait_exception is not None:
            raise self._wait_exception
        return self._wait_result


def _prepare(tmp_path: Path, *, pid: int | None = None):
    """A fresh, valid claim/worker/launch context -- nothing run yet."""
    claim, worker, runtime, invocation, context, start, evidence, _ = _transaction(
        tmp_path, pid=pid or os.getpid()
    )
    return claim, worker, invocation, context, start


def _runtime_for(claim, invocation, context, start, handle):
    return GuidedNpmLaunchedWorkerRuntime(claim, invocation, context, start, handle)


def _real_completed(tmp_path, *, pid: int | None = None):
    """Run the real child to a genuine, durable, exit-0 completed terminal receipt."""
    claim, worker, invocation, context, start = _prepare(tmp_path, pid=pid)
    code, terminal = entry_module.run_guided_npm_worker_to_terminal_receipt(
        worker, launch_context=context
    )
    assert code == entry_module.GUIDED_NPM_WORKER_ENTRY_SUCCESS
    return claim, worker, invocation, context, start, terminal


def _real_failure(tmp_path, *, monkeypatch, stage: str, pid: int | None = None):
    """Run the real child to a genuine exit-3 failure terminal receipt."""
    claim, worker, invocation, context, start = _prepare(tmp_path, pid=pid)
    if stage == "before_consumed":
        monkeypatch.setattr(
            Pipeline, "run_pass_1", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
    elif stage == "after_consumed":
        monkeypatch.setattr(
            Pipeline, "run_pass_2", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
    elif stage == "output_finalization":
        monkeypatch.setattr(
            entry_module,
            "build_guided_npm_required_output_evidence",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    else:
        raise AssertionError(stage)
    code, terminal = entry_module.run_guided_npm_worker_to_terminal_receipt(
        worker, launch_context=context
    )
    assert code == entry_module.GUIDED_NPM_WORKER_ENTRY_FAILED
    return claim, worker, invocation, context, start, terminal


# ---------------------------------------------------------------------------
# Real-process integration (section 35)
# ---------------------------------------------------------------------------


def test_real_subprocess_retains_handle_waits_and_reconciles_authority_refused(tmp_path):
    """A genuinely OS-spawned child, waited on via the real retained Popen handle.

    This fixture's worker-request artifact carries a fixed synthetic
    ``application_build_identity`` ("photometry-test") that can never equal
    the real, content-hash-dependent build identity this repository resolves
    at runtime (verified directly: they differ in every field). A truly
    separate OS process running the production entry point therefore always
    refuses with exit code 2 in this environment -- there is no fixture
    construction that lets an actually-spawned child reach Pass 1. This is a
    pre-existing characteristic of the test fixtures, not something this
    patch introduces. The exit-2 authority-refusal path is exactly as real
    and unmocked as any other: genuine Popen, genuine wait(), genuine exit
    code, genuine reconciliation.
    """
    claim, worker, invocation, context, start = _prepare(tmp_path)
    runtime = launch_guided_npm_worker_runtime(
        claim, current_application_build_identity=claim.application_build_identity
    )
    assert isinstance(runtime, GuidedNpmLaunchedWorkerRuntime)
    assert isinstance(runtime.process_handle, subprocess.Popen)

    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.observed_exit_code == 2
    assert result.observed_process_id == runtime.process_handle.pid
    assert result.final_outcome == OUTCOME_AUTHORITY_REFUSED
    assert result.terminal_evidence_status == "absent"
    assert result.consumed_authority_status == "absent"

    path = expected_guided_npm_reconciliation_result_path(worker.run_directory_path)
    assert Path(path).is_file()
    reread = read_guided_npm_worker_reconciliation_result(path)
    assert reread == result


# ---------------------------------------------------------------------------
# Verified completion (section 37) -- real Pipeline, real receipts, real
# outputs; only the outermost OS-process-spawn layer is a deterministic fake
# carrying this test process's own real PID (the same established pattern
# B2-D1/B2-D2A already use for their "real child" tests), because a truly
# separate OS-spawned child cannot reach Pass 1 in this environment (see
# above).
# ---------------------------------------------------------------------------


def test_verified_completed_real_pipeline_and_real_outputs(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)

    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_VERIFIED_COMPLETED
    assert result.observed_exit_code == 0
    assert result.observed_process_id == os.getpid() == start.process_id == terminal.observed_process_id
    assert result.consumed_authority_status == "verified"
    assert result.terminal_evidence_status == "verified"
    assert result.output_reconciliation_status == "verified"
    assert result.terminal_receipt_identity == terminal.canonical_terminal_receipt_identity
    assert result.completed_run_evidence_identity == terminal.completed_run_evidence_identity
    assert result.failure_category is None
    assert result.failure_detail_code is None
    assert handle.wait_calls == 1

    verify_guided_npm_worker_reconciliation_result(
        result,
        prelaunch_claim=claim,
        launch_invocation=invocation,
        execution_start_receipt=start,
        launch_context=context,
        consumed_authority_receipt=None if result.consumed_authority_status != "verified" else _read_consumed(context, claim, invocation, start),
        terminal_receipt=terminal,
    )

    # A second reconciliation call returns the same cached result without
    # waiting again or publishing a second artifact.
    result2 = reconcile_guided_npm_worker_runtime(runtime)
    assert result2 == result
    assert handle.wait_calls == 1


def _read_consumed(context, claim, invocation, start):
    from photometry_pipeline.guided_npm_worker_acknowledgement import (
        read_and_verify_guided_npm_consumed_authority_receipt,
    )

    return read_and_verify_guided_npm_consumed_authority_receipt(
        context.consumed_authority_receipt_path,
        prelaunch_claim=claim,
        launch_invocation=invocation,
        execution_start_receipt=start,
    )


def test_reconciliation_identity_is_deterministic(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert (
        compute_guided_npm_worker_reconciliation_result_identity(result)
        == result.canonical_reconciliation_result_identity
    )


def test_durable_publication_serializes_and_rereads(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    content = serialize_guided_npm_worker_reconciliation_result(result)
    assert decode_guided_npm_worker_reconciliation_result_bytes(content) == result


# ---------------------------------------------------------------------------
# Exit-code matrix (section 38)
# ---------------------------------------------------------------------------


def test_exit_zero_without_terminal_receipt(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_PROCESS_EXITED_ZERO_WITHOUT_TERMINAL_EVIDENCE
    assert result.terminal_evidence_status == "absent"


def test_exit_zero_with_failure_receipt_is_indeterminate(tmp_path, monkeypatch):
    claim, worker, invocation, context, start, terminal = _real_failure(
        tmp_path, monkeypatch=monkeypatch, stage="after_consumed"
    )
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_INDETERMINATE


def test_exit_two_without_terminal_receipt_is_authority_refused(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=2)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_AUTHORITY_REFUSED


@pytest.mark.parametrize(
    ("stage", "expected_outcome"),
    [
        ("before_consumed", OUTCOME_VERIFIED_FAILED_BEFORE_CONSUMED),
        ("after_consumed", OUTCOME_VERIFIED_FAILED_AFTER_CONSUMED),
        ("output_finalization", OUTCOME_VERIFIED_FAILED_DURING_OUTPUT_FINALIZATION),
    ],
)
def test_exit_three_with_valid_failure_receipt_maps_faithfully(
    tmp_path, monkeypatch, stage, expected_outcome
):
    claim, worker, invocation, context, start, terminal = _real_failure(
        tmp_path, monkeypatch=monkeypatch, stage=stage
    )
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == expected_outcome
    if stage == "before_consumed":
        assert result.consumed_authority_status == "absent"
    else:
        assert result.consumed_authority_status == "verified"


def test_exit_three_without_terminal_receipt(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_PROCESS_FAILED_WITHOUT_TERMINAL_EVIDENCE


def test_exit_three_with_completed_receipt_is_indeterminate(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_INDETERMINATE


def test_exit_four_without_terminal_receipt(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=4)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_TERMINAL_RECEIPT_PUBLICATION_FAILED


def test_exit_four_with_valid_terminal_receipt_is_indeterminate(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=4)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_INDETERMINATE


def test_unexpected_exit_code_without_terminal_receipt(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=17)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_PROCESS_FAILED_WITHOUT_TERMINAL_EVIDENCE


def test_wait_returns_bool_raises_observation_error(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=True)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    with pytest.raises(GuidedNpmReconciliationObservationError):
        reconcile_guided_npm_worker_runtime(runtime)


def test_wait_returns_none_raises_observation_error(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=None)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    with pytest.raises(GuidedNpmReconciliationObservationError):
        reconcile_guided_npm_worker_runtime(runtime)


def test_wait_raises_propagates_without_synthetic_exit_code(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_exception=RuntimeError("boom"))
    runtime = _runtime_for(claim, invocation, context, start, handle)
    with pytest.raises(RuntimeError, match="boom"):
        reconcile_guided_npm_worker_runtime(runtime)


def test_wait_timeout_does_not_kill_and_allows_later_reconciliation(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(
        os.getpid(), wait_exception=subprocess.TimeoutExpired(cmd="x", timeout=0.01)
    )
    runtime = _runtime_for(claim, invocation, context, start, handle)
    with pytest.raises(GuidedNpmReconciliationTimeout):
        reconcile_guided_npm_worker_runtime(runtime, timeout_sec=0.01)
    assert not Path(
        expected_guided_npm_reconciliation_result_path(worker.run_directory_path)
    ).exists()

    # A later call on the same runtime may still reconcile once it exits.
    handle._wait_exception = None
    handle._wait_result = 2
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_AUTHORITY_REFUSED


# ---------------------------------------------------------------------------
# Identity / PID mismatch (section 39)
# ---------------------------------------------------------------------------


def test_process_handle_pid_mismatch_with_start_receipt_refuses(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    mismatched_handle = FakeProcessHandle(start.process_id + 1, wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, mismatched_handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_PROCESS_IDENTITY_MISMATCH


def test_foreign_terminal_receipt_is_invalid_not_completed(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    claim, worker, invocation, context, start, _ = _real_completed(tmp_path / "a")
    other_claim, other_worker, other_invocation, other_context, other_start, other_terminal = (
        _real_completed(tmp_path / "b", pid=os.getpid())
    )
    # Point this run's terminal-receipt destination at the *other* run's
    # already-published, self-consistent, genuinely-valid receipt.
    foreign_path = expected_guided_npm_terminal_receipt_path(other_worker.run_directory_path)
    target_path = expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    Path(target_path).unlink()
    import shutil

    shutil.copyfile(foreign_path, target_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_TERMINAL_EVIDENCE_INVALID


# ---------------------------------------------------------------------------
# Evidence absence / corruption (section 40)
# ---------------------------------------------------------------------------


def test_consumed_receipt_malformed_is_invalid(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    Path(context.consumed_authority_receipt_path).write_text("{not json", encoding="utf-8")
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_CONSUMED_AUTHORITY_EVIDENCE_INVALID
    assert result.consumed_authority_status == "invalid"


def test_consumed_receipt_noncanonical_is_invalid(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    path = Path(context.consumed_authority_receipt_path)
    noncanonical = json.dumps(json.loads(path.read_bytes()), indent=2).encode()
    path.write_bytes(noncanonical)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_CONSUMED_AUTHORITY_EVIDENCE_INVALID


def test_terminal_receipt_malformed_is_invalid(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    path = expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    Path(path).write_text("{not json", encoding="utf-8")
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_TERMINAL_EVIDENCE_INVALID
    assert result.terminal_evidence_status == "invalid"


def test_terminal_receipt_noncanonical_is_invalid(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    path = expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    noncanonical = json.dumps(json.loads(Path(path).read_bytes()), indent=2).encode()
    Path(path).write_bytes(noncanonical)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_TERMINAL_EVIDENCE_INVALID


def test_failure_after_consumed_receipt_without_consumed_evidence_is_invalid(tmp_path, monkeypatch):
    claim, worker, invocation, context, start, terminal = _real_failure(
        tmp_path, monkeypatch=monkeypatch, stage="after_consumed"
    )
    # Delete the consumed-authority evidence this failure receipt requires.
    os.remove(context.consumed_authority_receipt_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_TERMINAL_EVIDENCE_INVALID


# ---------------------------------------------------------------------------
# Post-exit output mutation (section 41)
# ---------------------------------------------------------------------------


def _first_output_path(terminal) -> Path:
    return Path(terminal.output_evidence[0].output_path)


def test_output_unchanged_is_verified_completed(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_VERIFIED_COMPLETED


def test_output_missing_blocks_completion(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    _first_output_path(terminal).unlink()
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED
    assert result.output_reconciliation_status == "failed"


def test_output_changed_bytes_blocks_completion(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    path = _first_output_path(terminal)
    path.write_bytes(path.read_bytes() + b"tampered")
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED


def test_output_same_size_changed_bytes_blocks_completion(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    path = _first_output_path(terminal)
    original = path.read_bytes()
    flipped = bytes(b ^ 0xFF for b in original[:1]) + original[1:]
    path.write_bytes(flipped)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED


def test_output_replaced_same_bytes_different_inode_blocks_completion(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    path = _first_output_path(terminal)
    original = path.read_bytes()
    path.unlink()
    path.write_bytes(original)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    # Replacement changes device/inode identity even when content matches.
    assert result.final_outcome == OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED


def test_output_becomes_directory_blocks_completion(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    path = _first_output_path(terminal)
    path.unlink()
    path.mkdir()
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED


def test_output_becomes_symlink_blocks_completion(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    path = _first_output_path(terminal)
    target = path.with_name("elsewhere-" + path.name)
    target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED


def test_output_hard_link_added_blocks_completion(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    path = _first_output_path(terminal)
    link = path.with_name("extra-link-" + path.name)
    try:
        os.link(path, link)
    except OSError as exc:
        pytest.skip(f"hard link creation unavailable: {exc}")
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED


def test_terminal_record_reidentified_but_actual_file_differs_blocks_completion(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    path = expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    records = list(terminal.output_evidence)
    tampered_record = replace(
        records[0],
        source_sha256="1" * 64,
        canonical_output_record_identity="0" * 64,
    )
    tampered_record = replace(
        tampered_record,
        canonical_output_record_identity=(
            terminal_module.compute_guided_npm_terminal_output_record_identity(tampered_record)
        ),
    )
    records[0] = tampered_record
    tampered_evidence = tuple(records)
    completed_run_evidence_identity = (
        terminal_module.compute_guided_npm_completed_run_evidence_identity(
            source_worker_request_identity=terminal.source_worker_request_identity,
            source_execution_request_identity=terminal.source_execution_request_identity,
            source_consumed_authority_receipt_identity=(
                terminal.source_consumed_authority_receipt_identity
            ),
            guided_plan_identity=terminal.guided_plan_identity,
            validation_revision=terminal.validation_revision,
            output_evidence=tampered_evidence,
        )
    )
    tampered_terminal = replace(
        terminal,
        output_evidence=tampered_evidence,
        completed_run_evidence_identity=completed_run_evidence_identity,
        canonical_terminal_receipt_identity="0" * 64,
    )
    tampered_terminal = replace(
        tampered_terminal,
        canonical_terminal_receipt_identity=(
            terminal_module.compute_guided_npm_worker_terminal_receipt_identity(tampered_terminal)
        ),
    )
    Path(path).unlink()
    Path(path).write_bytes(
        terminal_module.serialize_guided_npm_worker_terminal_receipt(tampered_terminal)
    )
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    # The tampered receipt is internally self-consistent and passes the pure
    # verifier; only re-hashing the *actual* file catches the fraud.
    assert result.terminal_evidence_status == "verified"
    assert result.final_outcome == OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED


# ---------------------------------------------------------------------------
# Reconciliation schema tampering (section 42)
# ---------------------------------------------------------------------------


def _reidentify(result, **changes):
    changed = replace(result, **changes, canonical_reconciliation_result_identity="0" * 64)
    return replace(
        changed,
        canonical_reconciliation_result_identity=(
            compute_guided_npm_worker_reconciliation_result_identity(changed)
        ),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_prelaunch_claim_identity", "1" * 64),
        ("source_launch_invocation_identity", "1" * 64),
        ("source_execution_start_receipt_identity", "1" * 64),
        ("source_worker_request_identity", "1" * 64),
        ("source_execution_request_identity", "1" * 64),
        ("source_launch_context_identity", "1" * 64),
        ("guided_plan_identity", "other"),
        ("validation_revision", 999),
        ("execution_mode", "tonic"),
        ("run_directory_path", os.path.abspath("other-run")),
        ("observed_process_id", 999999),
        ("observed_exit_code", 17),
        ("consumed_authority_receipt_identity", "1" * 64),
        ("terminal_receipt_identity", "1" * 64),
        ("completed_run_evidence_identity", "1" * 64),
        ("process_terminal_status", "other"),
        ("consumed_authority_status", "invalid"),
        ("terminal_evidence_status", "invalid"),
        ("output_reconciliation_status", "failed"),
        ("final_outcome", "indeterminate"),
        ("failure_category", "some_category"),
    ],
)
def test_outer_reidentified_reconciliation_result_tampering_refuses(tmp_path, field, value):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    if field == "execution_mode" and value == result.execution_mode:
        value = "phasic" if value == "tonic" else "tonic"
    consumed = _read_consumed(context, claim, invocation, start)
    with pytest.raises(ValueError):
        verify_guided_npm_worker_reconciliation_result(
            _reidentify(result, **{field: value}),
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            consumed_authority_receipt=consumed,
            terminal_receipt=terminal,
        )


def test_reconciliation_result_identity_mismatch_refuses(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    consumed = _read_consumed(context, claim, invocation, start)
    with pytest.raises(ValueError, match="reconciliation_result_identity_mismatch"):
        verify_guided_npm_worker_reconciliation_result(
            replace(result, canonical_reconciliation_result_identity="1" * 64),
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            consumed_authority_receipt=consumed,
            terminal_receipt=terminal,
        )


# ---------------------------------------------------------------------------
# Coordinated final-outcome matrix tampering (B2-D2B narrow follow-up
# section 15): every scenario recomputes the outer identity so only the
# semantic matrix -- shared via _classify between orchestration and the pure
# verifier -- is the reason for refusal.
# ---------------------------------------------------------------------------


def test_matrix_authority_refused_with_wrong_exit_code_refuses(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=2)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_AUTHORITY_REFUSED
    tampered = _reidentify(result, observed_exit_code=3)
    with pytest.raises(ValueError, match="reconciliation_result_final_outcome_invalid"):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
        )


def test_matrix_authority_refused_with_terminal_verified_refuses(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=2)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_AUTHORITY_REFUSED

    (tmp_path / "foreign").mkdir()
    _, _, _, _, _, foreign_terminal = _real_completed(tmp_path / "foreign", pid=os.getpid())
    tampered = _reidentify(
        result,
        terminal_evidence_status="verified",
        terminal_receipt_identity=foreign_terminal.canonical_terminal_receipt_identity,
    )
    with pytest.raises(ValueError):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            terminal_receipt=foreign_terminal,
        )


def test_matrix_publication_failed_with_exit_zero_refuses(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=4)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_TERMINAL_RECEIPT_PUBLICATION_FAILED
    tampered = _reidentify(result, observed_exit_code=0)
    with pytest.raises(ValueError, match="reconciliation_result_final_outcome_invalid"):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
        )


def test_matrix_publication_failed_with_terminal_verified_refuses(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=4)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_TERMINAL_RECEIPT_PUBLICATION_FAILED

    (tmp_path / "foreign").mkdir()
    _, _, _, _, _, foreign_terminal = _real_completed(tmp_path / "foreign", pid=os.getpid())
    tampered = _reidentify(
        result,
        terminal_evidence_status="verified",
        terminal_receipt_identity=foreign_terminal.canonical_terminal_receipt_identity,
    )
    with pytest.raises(ValueError):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            terminal_receipt=foreign_terminal,
        )


def test_matrix_failed_after_consumed_with_consumed_absent_refuses(tmp_path, monkeypatch):
    claim, worker, invocation, context, start, terminal = _real_failure(
        tmp_path, monkeypatch=monkeypatch, stage="after_consumed"
    )
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_VERIFIED_FAILED_AFTER_CONSUMED
    tampered = _reidentify(
        result, consumed_authority_status="absent", consumed_authority_receipt_identity=None
    )
    with pytest.raises(ValueError, match="reconciliation_result_failed_after_consumed_invalid"):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            terminal_receipt=terminal,
        )


def test_matrix_failed_before_consumed_with_consumed_verified_refuses(tmp_path, monkeypatch):
    # Build the foreign completed transaction *before* monkeypatching
    # run_pass_1 for the primary failure scenario below -- monkeypatch only
    # reverts at test teardown, so it would otherwise still be active and
    # break the foreign run too.
    (tmp_path / "foreign").mkdir()
    (
        foreign_claim,
        foreign_worker,
        foreign_invocation,
        foreign_context,
        foreign_start,
        _,
    ) = _real_completed(tmp_path / "foreign", pid=os.getpid())
    foreign_consumed = _read_consumed(foreign_context, foreign_claim, foreign_invocation, foreign_start)

    claim, worker, invocation, context, start, terminal = _real_failure(
        tmp_path, monkeypatch=monkeypatch, stage="before_consumed"
    )
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_VERIFIED_FAILED_BEFORE_CONSUMED

    tampered = _reidentify(
        result,
        consumed_authority_status="verified",
        consumed_authority_receipt_identity=foreign_consumed.canonical_consumed_authority_receipt_identity,
    )
    with pytest.raises(ValueError, match="reconciliation_result_failed_before_consumed_invalid"):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            consumed_authority_receipt=foreign_consumed,
            terminal_receipt=terminal,
        )


def test_matrix_failed_during_output_finalization_with_exit_zero_refuses(tmp_path, monkeypatch):
    claim, worker, invocation, context, start, terminal = _real_failure(
        tmp_path, monkeypatch=monkeypatch, stage="output_finalization"
    )
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_VERIFIED_FAILED_DURING_OUTPUT_FINALIZATION
    consumed = _read_consumed(context, claim, invocation, start)
    tampered = _reidentify(result, observed_exit_code=0)
    with pytest.raises(ValueError, match="reconciliation_result_final_outcome_invalid"):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            consumed_authority_receipt=consumed,
            terminal_receipt=terminal,
        )


def test_matrix_exited_zero_no_terminal_with_wrong_exit_code_refuses(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_PROCESS_EXITED_ZERO_WITHOUT_TERMINAL_EVIDENCE
    tampered = _reidentify(result, observed_exit_code=3)
    with pytest.raises(ValueError, match="reconciliation_result_final_outcome_invalid"):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
        )


def test_matrix_process_failed_without_terminal_with_terminal_verified_refuses(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_PROCESS_FAILED_WITHOUT_TERMINAL_EVIDENCE

    (tmp_path / "foreign").mkdir()
    _, _, _, _, _, foreign_terminal = _real_completed(tmp_path / "foreign", pid=os.getpid())
    tampered = _reidentify(
        result,
        terminal_evidence_status="verified",
        terminal_receipt_identity=foreign_terminal.canonical_terminal_receipt_identity,
    )
    with pytest.raises(ValueError):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            terminal_receipt=foreign_terminal,
        )


def test_matrix_completed_output_integrity_failed_with_output_verified_refuses(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    _first_output_path(terminal).write_bytes(b"corrupted")
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED
    consumed = _read_consumed(context, claim, invocation, start)
    tampered = _reidentify(result, output_reconciliation_status="verified")
    # For this exact outcome pair, _classify's own exit==0 branch already
    # depends on output_ok to choose between verified_completed and
    # completed_output_integrity_failed, so the shared cross-check alone
    # already refuses this -- confirming no gap, not asserting which
    # specific message fires.
    with pytest.raises(ValueError):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            consumed_authority_receipt=consumed,
            terminal_receipt=terminal,
        )


def test_matrix_output_status_must_be_not_applicable_outside_output_outcomes(tmp_path):
    """_classify ignores output_ok for outcomes other than the two output
    ones, so tampering output_reconciliation_status away from
    not_applicable for e.g. authority_refused would slip past the shared
    cross-check alone -- this isolates the dedicated output-status rule."""
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=2)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_AUTHORITY_REFUSED
    assert result.output_reconciliation_status == "not_applicable"
    tampered = _reidentify(result, output_reconciliation_status="failed")
    with pytest.raises(ValueError, match="reconciliation_result_output_status_invalid"):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
        )


def test_matrix_process_identity_mismatch_with_all_pids_equal_refuses(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(start.process_id + 1, wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_PROCESS_IDENTITY_MISMATCH
    consumed = _read_consumed(context, claim, invocation, start)
    # Coordinated: make the stored PID agree with every other source while
    # still claiming a mismatch outcome.
    tampered = _reidentify(result, observed_process_id=start.process_id)
    with pytest.raises(ValueError, match="reconciliation_result_pid_mismatch_outcome_unjustified"):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            consumed_authority_receipt=consumed,
            terminal_receipt=terminal,
        )


def test_matrix_verified_completed_with_failure_fields_refuses(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_VERIFIED_COMPLETED
    consumed = _read_consumed(context, claim, invocation, start)
    tampered = _reidentify(result, failure_category="some_category", failure_detail_code="some_category")
    with pytest.raises(ValueError, match="reconciliation_result_completed_invalid"):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            consumed_authority_receipt=consumed,
            terminal_receipt=terminal,
        )


def test_matrix_indeterminate_for_a_more_specific_outcome_refuses(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_VERIFIED_COMPLETED
    consumed = _read_consumed(context, claim, invocation, start)
    tampered = _reidentify(
        result,
        final_outcome=OUTCOME_INDETERMINATE,
        failure_category=OUTCOME_INDETERMINATE,
        failure_detail_code=OUTCOME_INDETERMINATE,
        completed_run_evidence_identity=None,
    )
    with pytest.raises(ValueError, match="reconciliation_result_final_outcome_invalid"):
        verify_guided_npm_worker_reconciliation_result(
            tampered,
            prelaunch_claim=claim,
            launch_invocation=invocation,
            execution_start_receipt=start,
            launch_context=context,
            consumed_authority_receipt=consumed,
            terminal_receipt=terminal,
        )


# ---------------------------------------------------------------------------
# Publication mechanics (section 43)
# ---------------------------------------------------------------------------


def test_publish_refuses_preexisting_destination(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    path = expected_guided_npm_reconciliation_result_path(worker.run_directory_path)
    Path(path).write_text("conflict", encoding="utf-8")
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    with pytest.raises(GuidedNpmReconciliationPublicationError, match="destination_conflict"):
        reconcile_guided_npm_worker_runtime(runtime)
    assert Path(path).read_text(encoding="utf-8") == "conflict"


def test_publish_fsync_failure_leaves_no_final_artifact(monkeypatch, tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    monkeypatch.setattr(os, "fsync", lambda *_: (_ for _ in ()).throw(OSError("fsync")))
    with pytest.raises(OSError, match="fsync"):
        reconcile_guided_npm_worker_runtime(runtime)
    path = expected_guided_npm_reconciliation_result_path(worker.run_directory_path)
    assert not Path(path).exists()


def test_publish_reread_mismatch_never_overwrites(monkeypatch, tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    monkeypatch.setattr(reco_module, "_stable_read", lambda *_: b"{}\n")
    with pytest.raises(ValueError, match="decode_invalid"):
        publish_guided_npm_worker_reconciliation_result(result)
    path = expected_guided_npm_reconciliation_result_path(worker.run_directory_path)
    assert Path(path).is_file()
    monkeypatch.undo()
    with pytest.raises(GuidedNpmReconciliationPublicationError, match="destination_conflict"):
        publish_guided_npm_worker_reconciliation_result(result)


def test_strict_decoder_refuses_unknown_missing_and_duplicate_fields(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    payload = json.loads(serialize_guided_npm_worker_reconciliation_result(result))
    payload["unknown_field"] = 1
    with pytest.raises(ValueError, match="decode_invalid"):
        decode_guided_npm_worker_reconciliation_result_bytes(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    del payload["unknown_field"]
    del payload["execution_mode"]
    with pytest.raises(ValueError, match="decode_invalid"):
        decode_guided_npm_worker_reconciliation_result_bytes(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    with pytest.raises(ValueError, match="decode_invalid"):
        decode_guided_npm_worker_reconciliation_result_bytes(
            b'{"result_schema_name":"a","result_schema_name":"b"}\n'
        )
    noncanonical = json.dumps(
        json.loads(serialize_guided_npm_worker_reconciliation_result(result)), indent=2
    ).encode()
    with pytest.raises(ValueError, match="decode_invalid"):
        decode_guided_npm_worker_reconciliation_result_bytes(noncanonical)


# ---------------------------------------------------------------------------
# Computation/publication state separation (B2-D2B narrow follow-up
# sections 16-26)
# ---------------------------------------------------------------------------


def test_publish_false_then_publish_true_waits_once_publishes_once(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    path = expected_guided_npm_reconciliation_result_path(worker.run_directory_path)

    first = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert handle.wait_calls == 1
    assert not Path(path).exists()

    second = reconcile_guided_npm_worker_runtime(runtime, publish=True)
    assert handle.wait_calls == 1
    assert second == first
    assert Path(path).is_file()


def test_repeated_no_publication_calls_return_same_result_one_wait(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    path = expected_guided_npm_reconciliation_result_path(worker.run_directory_path)

    first = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    second = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert first == second
    assert handle.wait_calls == 1
    assert not Path(path).exists()


def test_repeated_publication_request_publishes_once(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)

    first = reconcile_guided_npm_worker_runtime(runtime, publish=True)
    second = reconcile_guided_npm_worker_runtime(runtime, publish=True)
    assert first == second
    assert handle.wait_calls == 1


def test_publication_failure_is_remembered_and_never_retried(monkeypatch, tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)

    publish_calls = []
    real_publish = reco_module.publish_guided_npm_worker_reconciliation_result

    def failing_publish(result):
        publish_calls.append(1)
        raise OSError("publish boom")

    monkeypatch.setattr(reco_module, "publish_guided_npm_worker_reconciliation_result", failing_publish)
    with pytest.raises(OSError, match="publish boom"):
        reconcile_guided_npm_worker_runtime(runtime, publish=True)
    assert len(publish_calls) == 1
    assert handle.wait_calls == 1

    # A second publish=True call must not touch the publisher again.
    with pytest.raises(GuidedNpmReconciliationPublicationError, match="reconciliation_publication_previously_failed"):
        reconcile_guided_npm_worker_runtime(runtime, publish=True)
    assert len(publish_calls) == 1
    assert handle.wait_calls == 1

    # publish=False still returns the computed in-memory result.
    monkeypatch.undo()
    result = reconcile_guided_npm_worker_runtime(runtime, publish=False)
    assert result.final_outcome == OUTCOME_VERIFIED_COMPLETED
    assert handle.wait_calls == 1
    path = expected_guided_npm_reconciliation_result_path(worker.run_directory_path)
    assert not Path(path).exists()


def test_indeterminate_reread_failure_is_remembered_and_never_retried(monkeypatch, tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)

    monkeypatch.setattr(reco_module, "_stable_read", lambda *_: b"{}\n")
    with pytest.raises(ValueError, match="decode_invalid"):
        reconcile_guided_npm_worker_runtime(runtime, publish=True)
    assert handle.wait_calls == 1
    path = expected_guided_npm_reconciliation_result_path(worker.run_directory_path)
    assert Path(path).is_file()  # the rename succeeded; only the reread verification failed

    monkeypatch.undo()
    # The indeterminate reread failure is remembered: a later publish=True
    # call must raise the same controlled failure without touching the
    # filesystem again (never a fresh destination_conflict from a retried
    # attempt).
    with pytest.raises(
        GuidedNpmReconciliationPublicationError, match="reconciliation_publication_previously_failed"
    ):
        reconcile_guided_npm_worker_runtime(runtime, publish=True)
    assert handle.wait_calls == 1
    assert Path(path).is_file()


def test_concurrent_reconciliation_of_same_runtime_waits_once_publishes_once(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)

    results = []
    errors = []
    barrier = threading.Barrier(4)

    def worker_fn():
        try:
            barrier.wait(timeout=5)
            results.append(reconcile_guided_npm_worker_runtime(runtime, publish=True))
        except Exception as exc:  # pragma: no cover - failure path recorded, not expected
            errors.append(exc)

    threads = [threading.Thread(target=worker_fn) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert len(results) == 4
    assert all(r == results[0] for r in results)
    assert handle.wait_calls == 1
    path = expected_guided_npm_reconciliation_result_path(worker.run_directory_path)
    assert Path(path).is_file()


def test_independent_runtimes_reconcile_without_blocking_each_other(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    claim_a, worker_a, invocation_a, context_a, start_a, _ = _real_completed(tmp_path / "a")
    claim_b, worker_b, invocation_b, context_b, start_b, _ = _real_completed(
        tmp_path / "b", pid=os.getpid()
    )

    a_may_finish = threading.Event()
    a_started_waiting = threading.Event()

    class BlockingHandle:
        def __init__(self, pid):
            self.pid = pid
            self.wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            a_started_waiting.set()
            a_may_finish.wait(timeout=5)
            return 0

    handle_a = BlockingHandle(os.getpid())
    handle_b = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime_a = _runtime_for(claim_a, invocation_a, context_a, start_a, handle_a)
    runtime_b = _runtime_for(claim_b, invocation_b, context_b, start_b, handle_b)

    results = {}

    def reconcile_a():
        results["a"] = reconcile_guided_npm_worker_runtime(runtime_a, publish=False)

    thread_a = threading.Thread(target=reconcile_a)
    thread_a.start()
    assert a_started_waiting.wait(timeout=5)

    # Runtime B must be able to reconcile fully while A is still blocked.
    result_b = reconcile_guided_npm_worker_runtime(runtime_b, publish=False)
    assert result_b.final_outcome == OUTCOME_VERIFIED_COMPLETED
    assert handle_b.wait_calls == 1

    a_may_finish.set()
    thread_a.join(timeout=5)
    assert results["a"].final_outcome == OUTCOME_VERIFIED_COMPLETED
    assert handle_a.wait_calls == 1


# ---------------------------------------------------------------------------
# Post-launch-confirmation-failure reconciliation (B2-D2B narrow follow-up
# sections 7-9)
# ---------------------------------------------------------------------------


def test_post_launch_reconciliation_never_becomes_completed_even_with_exit_zero_and_evidence(
    monkeypatch, tmp_path
):
    import photometry_pipeline.guided_npm_worker_launch as launch_module
    from photometry_pipeline.guided_npm_worker_launch import GuidedNpmPostLaunchRuntime

    claim, worker, invocation, context, start = _prepare(tmp_path)
    real_verify = launch_module.verify_guided_npm_worker_execution_start_receipt

    def flaky_verify(receipt, prelaunch_claim, invocation_):
        if receipt.process_id == os.getpid():
            raise ValueError("forced_verification_failure")
        return real_verify(receipt, prelaunch_claim, invocation_)

    monkeypatch.setattr(
        launch_module, "verify_guided_npm_worker_execution_start_receipt", flaky_verify
    )

    handle = FakeProcessHandle(os.getpid(), wait_result=0)

    def launcher(argv, *, cwd, shell):
        return GuidedNpmStartedProcessRuntime(os.getpid(), GUIDED_NPM_LAUNCHER_KIND, handle)

    runtime = launch_guided_npm_worker_runtime(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=launcher,
    )
    assert isinstance(runtime, GuidedNpmPostLaunchRuntime)

    monkeypatch.undo()
    # Even though the real child later runs to completion with a valid
    # terminal receipt and exit 0, the parent never durably confirmed the
    # launch, so this can never be treated as verified completion.
    code, terminal = entry_module.run_guided_npm_worker_to_terminal_receipt(
        worker, launch_context=context
    )
    assert code == entry_module.GUIDED_NPM_WORKER_ENTRY_SUCCESS
    assert terminal.terminal_outcome == "completed"

    result = reconcile_guided_npm_post_launch_runtime(runtime)
    assert result.final_outcome == OUTCOME_POST_LAUNCH_EVIDENCE_FAILED
    assert result.final_outcome != OUTCOME_VERIFIED_COMPLETED
    assert result.observed_exit_code == 0
    assert result.terminal_evidence_present is True
    assert result.consumed_authority_evidence_present is True


def test_post_launch_reconciliation_computes_once_and_caches(tmp_path):
    import photometry_pipeline.guided_npm_worker_launch as launch_module

    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    failure = launch_module.GuidedNpmWorkerPostLaunchFailure(
        (
            launch_module.GuidedNpmWorkerLaunchIssue(
                "process_identity_invalid",
                "post_launch",
                "test",
                "test_detail",
            ),
        ),
        invocation.canonical_launch_invocation_identity,
        os.getpid(),
    )
    runtime = launch_module.GuidedNpmPostLaunchRuntime(
        claim, invocation, context, handle, os.getpid(), failure
    )
    first = reconcile_guided_npm_post_launch_runtime(runtime)
    second = reconcile_guided_npm_post_launch_runtime(runtime)
    assert first == second
    assert handle.wait_calls == 1


# ---------------------------------------------------------------------------
# No-false-completion (section 44)
# ---------------------------------------------------------------------------


def test_exit_code_alone_does_not_determine_completion(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome != OUTCOME_VERIFIED_COMPLETED


def test_terminal_receipt_alone_does_not_determine_completion(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    # Exit code is wrong for a completed claim.
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome != OUTCOME_VERIFIED_COMPLETED


def test_consumed_receipt_alone_does_not_determine_completion(tmp_path, monkeypatch):
    claim, worker, invocation, context, start, terminal = _real_failure(
        tmp_path, monkeypatch=monkeypatch, stage="after_consumed"
    )
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.consumed_authority_status == "verified"
    assert result.final_outcome != OUTCOME_VERIFIED_COMPLETED


def test_outputs_alone_do_not_determine_completion(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome != OUTCOME_VERIFIED_COMPLETED


def test_completed_receipt_with_mismatched_pid_never_passes(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(start.process_id + 1, wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome == OUTCOME_PROCESS_IDENTITY_MISMATCH


def test_completed_receipt_with_changed_outputs_never_passes(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    _first_output_path(terminal).write_bytes(b"corrupted")
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome != OUTCOME_VERIFIED_COMPLETED


def test_exit_zero_with_missing_terminal_receipt_never_passes(tmp_path):
    claim, worker, invocation, context, start = _prepare(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome != OUTCOME_VERIFIED_COMPLETED


def test_exit_three_with_completed_receipt_never_passes(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    handle = FakeProcessHandle(os.getpid(), wait_result=3)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert result.final_outcome != OUTCOME_VERIFIED_COMPLETED


def test_result_says_completed_only_after_wait_returns(tmp_path):
    claim, worker, invocation, context, start, terminal = _real_completed(tmp_path)
    waited = {"called": False}
    real_wait = FakeProcessHandle.wait

    def tracking_wait(self, timeout=None):
        waited["called"] = True
        return real_wait(self, timeout=timeout)

    handle = FakeProcessHandle(os.getpid(), wait_result=0)
    handle.wait = tracking_wait.__get__(handle, FakeProcessHandle)
    runtime = _runtime_for(claim, invocation, context, start, handle)
    assert waited["called"] is False
    result = reconcile_guided_npm_worker_runtime(runtime)
    assert waited["called"] is True
    assert result.final_outcome == OUTCOME_VERIFIED_COMPLETED


# ---------------------------------------------------------------------------
# Ordinary NPM / RWD isolation (section 34)
# ---------------------------------------------------------------------------


def test_ordinary_npm_creates_no_reconciliation_result(tmp_path):
    from photometry_pipeline.config import Config
    from tests.test_guided_npm_authorized_adapter import _explicit_csv

    source = tmp_path / "ordinary"
    source.mkdir()
    (source / "session.csv").write_bytes(_explicit_csv())
    output = tmp_path / "ordinary-output"
    pipeline = Pipeline(Config(target_fs_hz=2.0, chunk_duration_sec=2.0))
    pipeline.run(os.fspath(source), os.fspath(output), force_format="npm")
    assert not (
        output / reco_module.GUIDED_NPM_RECONCILIATION_RESULT_FILENAME
    ).exists()
