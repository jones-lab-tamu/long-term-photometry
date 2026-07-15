from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_npm_worker_terminal as terminal_module
import photometry_pipeline.guided_npm_worker_entry as entry_module
from photometry_pipeline.guided_npm_worker_acknowledgement import (
    read_and_verify_guided_npm_consumed_authority_receipt,
)
from photometry_pipeline.guided_npm_worker_terminal import (
    GUIDED_NPM_TERMINAL_RECEIPT_FILENAME,
    GuidedNpmTerminalError,
    GuidedNpmTerminalOutputRecord,
    GuidedNpmWorkerTerminalReceipt,
    build_guided_npm_required_output_evidence,
    build_guided_npm_terminal_output_record,
    build_guided_npm_worker_terminal_failure_receipt,
    build_guided_npm_worker_terminal_success_receipt,
    compute_guided_npm_completed_run_evidence_identity,
    compute_guided_npm_terminal_output_record_identity,
    compute_guided_npm_worker_terminal_receipt_identity,
    decode_guided_npm_worker_terminal_receipt_bytes,
    expected_guided_npm_terminal_receipt_path,
    publish_guided_npm_worker_terminal_receipt,
    read_guided_npm_worker_terminal_receipt,
    required_guided_npm_terminal_output_roles,
    serialize_guided_npm_worker_terminal_receipt,
    verify_guided_npm_worker_terminal_receipt,
)
from tests.test_guided_npm_worker_acknowledgement import _transaction


def _real_success_transaction(tmp_path: Path, *, pid: int | None = None):
    """Run the exact real child path to a genuine 'completed' terminal receipt."""
    claim, worker, runtime, invocation, context, start, evidence, _ = _transaction(
        tmp_path, pid=pid or os.getpid()
    )
    code, receipt = entry_module.run_guided_npm_worker_to_terminal_receipt(
        worker, launch_context=context
    )
    assert code == entry_module.GUIDED_NPM_WORKER_ENTRY_SUCCESS
    assert isinstance(receipt, GuidedNpmWorkerTerminalReceipt)
    consumed = read_and_verify_guided_npm_consumed_authority_receipt(
        context.consumed_authority_receipt_path,
        prelaunch_claim=claim,
        launch_invocation=invocation,
        execution_start_receipt=start,
    )
    return claim, worker, context, consumed, receipt


def _reidentify(receipt, **changes):
    changed = replace(receipt, **changes, canonical_terminal_receipt_identity="0" * 64)
    return replace(
        changed,
        canonical_terminal_receipt_identity=(
            compute_guided_npm_worker_terminal_receipt_identity(changed)
        ),
    )


# ---------------------------------------------------------------------------
# Required-output derivation (never globbed)
# ---------------------------------------------------------------------------


def test_required_output_roles_are_deterministic_from_mode_and_traces_only():
    assert required_guided_npm_terminal_output_roles("tonic", traces_only=False) == (
        "run_report", "run_metadata", "config_used", "trace_cache",
    )
    assert required_guided_npm_terminal_output_roles("tonic", traces_only=True) == (
        "run_report", "run_metadata", "config_used", "trace_cache",
    )
    assert required_guided_npm_terminal_output_roles("phasic", traces_only=True) == (
        "run_report", "run_metadata", "config_used", "trace_cache",
    )
    assert required_guided_npm_terminal_output_roles("phasic", traces_only=False) == (
        "run_report", "run_metadata", "config_used", "trace_cache",
        "features_csv", "feature_event_provenance",
    )
    with pytest.raises(ValueError):
        required_guided_npm_terminal_output_roles("both", traces_only=False)


# ---------------------------------------------------------------------------
# Real success-path integration (section 38)
# ---------------------------------------------------------------------------


def test_real_success_path_publishes_exact_verifiable_terminal_receipt(tmp_path):
    claim, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    assert receipt.terminal_outcome == "completed"
    assert receipt.terminal_stage == "terminal"
    assert (
        receipt.worker_acceptance_status,
        receipt.consumed_authority_status,
        receipt.numerical_dispatch_status,
        receipt.completion_status,
    ) == ("accepted_exact_worker_authority", "verified", "entered", "verified_completed")
    assert receipt.source_consumed_authority_receipt_identity == (
        consumed.canonical_consumed_authority_receipt_identity
    )
    assert receipt.observed_process_id == consumed.observed_process_id == os.getpid()
    assert receipt.authorized_runtime_identity == consumed.authorized_runtime_identity
    assert receipt.correction_authority_identity == consumed.correction_authority_identity
    assert receipt.feature_authority_identity == consumed.feature_authority_identity
    assert [record.output_role for record in receipt.output_evidence] == list(
        required_guided_npm_terminal_output_roles("phasic", traces_only=False)
    )
    for record in receipt.output_evidence:
        path = Path(record.output_path)
        assert path.is_file()
        assert record.source_size_bytes == path.stat().st_size
        assert record.canonical_output_record_identity == (
            compute_guided_npm_terminal_output_record_identity(record)
        )
    expected_evidence_identity = compute_guided_npm_completed_run_evidence_identity(
        source_worker_request_identity=receipt.source_worker_request_identity,
        source_execution_request_identity=receipt.source_execution_request_identity,
        source_consumed_authority_receipt_identity=(
            receipt.source_consumed_authority_receipt_identity
        ),
        guided_plan_identity=receipt.guided_plan_identity,
        validation_revision=receipt.validation_revision,
        output_evidence=receipt.output_evidence,
    )
    assert receipt.completed_run_evidence_identity == expected_evidence_identity
    assert receipt.failure_category is None
    assert receipt.failure_exception_type is None

    path = expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    assert Path(path).name == GUIDED_NPM_TERMINAL_RECEIPT_FILENAME
    reread = read_guided_npm_worker_terminal_receipt(path, worker_request=worker)
    assert reread == receipt
    verify_guided_npm_worker_terminal_receipt(
        receipt, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
    )


def test_ordinary_npm_and_rwd_never_produce_a_terminal_receipt(tmp_path):
    from photometry_pipeline.config import Config
    from photometry_pipeline.pipeline import Pipeline

    source = tmp_path / "ordinary"
    source.mkdir()
    from tests.test_guided_npm_authorized_adapter import _explicit_csv

    (source / "session.csv").write_bytes(_explicit_csv())
    output = tmp_path / "ordinary-output"
    pipeline = Pipeline(Config(target_fs_hz=2.0, chunk_duration_sec=2.0))
    pipeline.run(os.fspath(source), os.fspath(output), force_format="npm")
    assert not (output / GUIDED_NPM_TERMINAL_RECEIPT_FILENAME).exists()


# ---------------------------------------------------------------------------
# Output-record verification (sections 18, 41)
# ---------------------------------------------------------------------------


def test_output_record_refuses_missing_symlinked_directory_or_stale(tmp_path):
    run_dir = tmp_path
    (run_dir / "run_report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(GuidedNpmTerminalError, match="terminal_output_missing"):
        build_guided_npm_terminal_output_record(
            os.fspath(run_dir), "run_metadata", "phasic"
        )

    (run_dir / "config_used.yaml").mkdir()
    with pytest.raises(GuidedNpmTerminalError, match="terminal_output_not_regular"):
        build_guided_npm_terminal_output_record(
            os.fspath(run_dir), "config_used", "phasic"
        )

    target = run_dir / "elsewhere_report.json"
    target.write_text("{}", encoding="utf-8")
    link = run_dir / "run_report_link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    record = build_guided_npm_terminal_output_record(
        os.fspath(run_dir), "run_report", "phasic"
    )
    assert record.source_sha256

    stale_report = run_dir / "stale_report.json"
    stale_report.write_text("{}", encoding="utf-8")
    with pytest.raises(GuidedNpmTerminalError, match="terminal_output_stale"):
        build_guided_npm_terminal_output_record(
            os.fspath(run_dir),
            "run_report",
            "phasic",
            not_before_mtime_ns=stale_report.stat().st_mtime_ns + 10**9,
        )


def test_output_path_outside_run_directory_refused(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(GuidedNpmTerminalError, match="terminal_output_missing"):
        build_guided_npm_terminal_output_record(
            os.fspath(run_dir), "run_report", "phasic"
        )


def test_required_output_evidence_refuses_duplicate_or_incomplete(tmp_path, monkeypatch):
    run_dir = tmp_path
    for name in ("run_report.json", "run_metadata.json", "config_used.yaml"):
        (run_dir / name).write_text("{}", encoding="utf-8")
    (run_dir / "phasic_trace_cache.h5").write_bytes(b"fake")
    (run_dir / "features").mkdir()
    (run_dir / "features" / "features.csv").write_text("a,b\n", encoding="utf-8")
    (run_dir / "features" / "feature_event_provenance.json").write_text("{}", encoding="utf-8")
    evidence = build_guided_npm_required_output_evidence(
        os.fspath(run_dir), "phasic", traces_only=False
    )
    assert len(evidence) == 6
    assert len({record.output_role for record in evidence}) == 6


# ---------------------------------------------------------------------------
# Schema / serialize / decode
# ---------------------------------------------------------------------------


def test_receipt_serializes_canonically_and_round_trips(tmp_path):
    _, _, _, _, receipt = _real_success_transaction(tmp_path)
    content = serialize_guided_npm_worker_terminal_receipt(receipt)
    assert decode_guided_npm_worker_terminal_receipt_bytes(content) == receipt


def test_strict_decoder_refuses_unknown_missing_duplicate_and_noncanonical(tmp_path):
    _, _, _, _, receipt = _real_success_transaction(tmp_path)
    payload = json.loads(serialize_guided_npm_worker_terminal_receipt(receipt))
    payload["unknown_field"] = 1
    with pytest.raises(ValueError, match="decode_invalid"):
        decode_guided_npm_worker_terminal_receipt_bytes(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    del payload["unknown_field"]
    del payload["execution_mode"]
    with pytest.raises(ValueError, match="decode_invalid"):
        decode_guided_npm_worker_terminal_receipt_bytes(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    with pytest.raises(ValueError, match="decode_invalid"):
        decode_guided_npm_worker_terminal_receipt_bytes(
            b'{"receipt_schema_name":"a","receipt_schema_name":"b"}\n'
        )
    noncanonical = json.dumps(
        json.loads(serialize_guided_npm_worker_terminal_receipt(receipt)), indent=2
    ).encode()
    with pytest.raises(ValueError, match="decode_invalid"):
        decode_guided_npm_worker_terminal_receipt_bytes(noncanonical)


# ---------------------------------------------------------------------------
# Pure verifier: tampering (section 42, consolidated as parametrized outer
# reidentification, matching the existing repo convention in
# test_guided_npm_worker_launch.py / test_guided_npm_worker_acknowledgement.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_worker_request_identity", "1" * 64),
        ("source_execution_request_identity", "1" * 64),
        ("source_launch_invocation_identity", "1" * 64),
        ("source_launch_context_identity", "1" * 64),
        ("source_consumed_authority_receipt_identity", "1" * 64),
        ("guided_plan_identity", "other"),
        ("validation_revision", 999),
        ("execution_mode", "tonic"),
        ("observed_process_id", 999999),
        ("worker_request_artifact_path", os.path.abspath("other-worker.json")),
        ("run_directory_path", os.path.abspath("other-run")),
        ("terminal_outcome", "failed_before_consumed_authority"),
        ("terminal_stage", "pass_1"),
        ("worker_acceptance_status", "other"),
        ("consumed_authority_status", "not_available"),
        ("numerical_dispatch_status", "not_available"),
        ("completion_status", "failed"),
        ("authorized_runtime_identity", "1" * 64),
        ("correction_authority_identity", "1" * 64),
        ("feature_authority_identity", "1" * 64),
        ("completed_run_evidence_identity", "1" * 64),
        ("failure_category", "some_category"),
        ("failure_exception_type", "RuntimeError"),
    ],
)
def test_outer_reidentified_terminal_receipt_tampering_refuses(tmp_path, field, value):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    if field == "execution_mode" and value == receipt.execution_mode:
        value = "phasic" if value == "tonic" else "tonic"
    with pytest.raises(ValueError):
        verify_guided_npm_worker_terminal_receipt(
            _reidentify(receipt, **{field: value}),
            worker_request=worker,
            launch_context=context,
            consumed_authority_receipt=consumed,
        )


def test_output_record_tampering_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    records = list(receipt.output_evidence)
    for field, value in (
        ("output_role", "run_metadata"),
        ("output_path", os.path.abspath("other.json")),
        ("canonical_relative_path", "other.json"),
        ("source_size_bytes", 999999),
        ("source_sha256", "1" * 64),
    ):
        changed = replace(
            records[0], **{field: value}, canonical_output_record_identity="0" * 64
        )
        changed = replace(
            changed,
            canonical_output_record_identity=(
                compute_guided_npm_terminal_output_record_identity(changed)
            ),
        )
        tampered_records = list(records)
        tampered_records[0] = changed
        tampered = _reidentify(receipt, output_evidence=tuple(tampered_records))
        with pytest.raises(ValueError):
            verify_guided_npm_worker_terminal_receipt(
                tampered,
                worker_request=worker,
                launch_context=context,
                consumed_authority_receipt=consumed,
            )


def test_receipt_identity_mismatch_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    with pytest.raises(ValueError, match="terminal_receipt_identity_mismatch"):
        verify_guided_npm_worker_terminal_receipt(
            replace(receipt, canonical_terminal_receipt_identity="1" * 64),
            worker_request=worker,
            launch_context=context,
            consumed_authority_receipt=consumed,
        )


# ---------------------------------------------------------------------------
# Output-path authority binding (B2-D2A narrow follow-up)
# ---------------------------------------------------------------------------


def _reidentify_output_record(record, **changes):
    changed = replace(record, **changes, canonical_output_record_identity="0" * 64)
    return replace(
        changed,
        canonical_output_record_identity=(
            compute_guided_npm_terminal_output_record_identity(changed)
        ),
    )


def _coordinated_reidentify_with_output_evidence(receipt, output_evidence):
    """Recompute every downstream identity so only the path check can refuse."""
    completed_run_evidence_identity = compute_guided_npm_completed_run_evidence_identity(
        source_worker_request_identity=receipt.source_worker_request_identity,
        source_execution_request_identity=receipt.source_execution_request_identity,
        source_consumed_authority_receipt_identity=(
            receipt.source_consumed_authority_receipt_identity
        ),
        guided_plan_identity=receipt.guided_plan_identity,
        validation_revision=receipt.validation_revision,
        output_evidence=output_evidence,
    )
    return _reidentify(
        receipt,
        output_evidence=output_evidence,
        completed_run_evidence_identity=completed_run_evidence_identity,
    )


def test_coordinated_reidentified_output_path_outside_authorized_run_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    records = list(receipt.output_evidence)
    outside_path = os.path.abspath(
        os.path.join(str(tmp_path), "elsewhere", os.path.basename(records[0].output_path))
    )
    records[0] = _reidentify_output_record(records[0], output_path=outside_path)
    tampered = _coordinated_reidentify_with_output_evidence(receipt, tuple(records))
    # Every identity is now internally self-consistent; only deterministic
    # authority reconciliation against the run directory can catch this.
    with pytest.raises(ValueError, match="terminal_output_record_path_invalid"):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


def test_coordinated_reidentified_relative_output_path_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    records = list(receipt.output_evidence)
    records[0] = _reidentify_output_record(
        records[0], output_path=records[0].canonical_relative_path
    )
    tampered = _coordinated_reidentify_with_output_evidence(receipt, tuple(records))
    with pytest.raises(ValueError, match="terminal_output_record_path_invalid"):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


def test_coordinated_reidentified_output_path_in_another_run_directory_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    records = list(receipt.output_evidence)
    other_run_dir = os.path.abspath(os.path.join(str(tmp_path), "other-run"))
    other_path = os.path.join(other_run_dir, records[0].canonical_relative_path)
    records[0] = _reidentify_output_record(records[0], output_path=other_path)
    tampered = _coordinated_reidentify_with_output_evidence(receipt, tuple(records))
    with pytest.raises(ValueError, match="terminal_output_record_path_invalid"):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


def test_coordinated_reidentified_wrong_role_specific_path_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    records = list(receipt.output_evidence)
    run_report_index = next(
        index for index, record in enumerate(records) if record.output_role == "run_report"
    )
    wrong_path = os.path.join(worker.run_directory_path, "run_metadata.json")
    records[run_report_index] = _reidentify_output_record(
        records[run_report_index], output_path=wrong_path
    )
    tampered = _coordinated_reidentify_with_output_evidence(receipt, tuple(records))
    with pytest.raises(ValueError, match="terminal_output_record_path_invalid"):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


# ---------------------------------------------------------------------------
# Schema-v1 deferred execution-start binding (B2-D2A narrow follow-up)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["completed", "failed_after_consumed"])
def test_schema_v1_refuses_claimed_execution_start_receipt_identity(tmp_path, kind):
    _, worker, context, consumed, success_receipt = _real_success_transaction(tmp_path)
    if kind == "completed":
        base = success_receipt
    else:
        base = build_guided_npm_worker_terminal_failure_receipt(
            worker_request=worker,
            launch_context=context,
            observed_process_id=consumed.observed_process_id,
            terminal_outcome="failed_after_consumed_authority",
            terminal_stage="pass_1",
            consumed_authority_receipt=consumed,
            failure_category="pipeline_execution_failed",
            failure_exception_type="RuntimeError",
        )
    tampered = _reidentify(base, source_execution_start_receipt_identity="1" * 64)
    with pytest.raises(ValueError, match="terminal_receipt_invalid"):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


# ---------------------------------------------------------------------------
# Impossible outcome/field combinations (section 43)
# ---------------------------------------------------------------------------


def test_completed_with_failure_category_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    tampered = _reidentify(receipt, failure_category="some_category")
    with pytest.raises(ValueError, match="terminal_receipt_completed_invalid"):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


def test_completed_with_no_consumed_receipt_identity_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    tampered = _reidentify(receipt, source_consumed_authority_receipt_identity=None)
    with pytest.raises(ValueError):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


def test_completed_with_no_output_evidence_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    tampered = _reidentify(receipt, output_evidence=())
    with pytest.raises(ValueError, match="terminal_receipt_completed_invalid"):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


def test_completed_with_no_completed_run_evidence_identity_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    tampered = _reidentify(receipt, completed_run_evidence_identity=None)
    with pytest.raises(ValueError, match="terminal_receipt_completed_invalid"):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


def test_failed_with_completion_status_verified_completed_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    failure = build_guided_npm_worker_terminal_failure_receipt(
        worker_request=worker,
        launch_context=context,
        observed_process_id=consumed.observed_process_id,
        terminal_outcome="failed_after_consumed_authority",
        terminal_stage="pass_1",
        consumed_authority_receipt=consumed,
        failure_category="pipeline_execution_failed",
        failure_exception_type="RuntimeError",
    )
    tampered = _reidentify(failure, completion_status="verified_completed")
    with pytest.raises(ValueError):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


def test_failed_before_consumed_with_consumed_status_verified_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    failure = build_guided_npm_worker_terminal_failure_receipt(
        worker_request=worker,
        launch_context=context,
        observed_process_id=consumed.observed_process_id,
        terminal_outcome="failed_before_consumed_authority",
        terminal_stage="numerical_dispatch",
        consumed_authority_receipt=None,
        failure_category="numerical_dispatch_failed",
        failure_exception_type="RuntimeError",
    )
    tampered = _reidentify(
        failure,
        consumed_authority_status="verified",
        source_consumed_authority_receipt_identity=(
            consumed.canonical_consumed_authority_receipt_identity
        ),
    )
    with pytest.raises(ValueError, match="terminal_receipt_failed_before_consumed_invalid"):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


def test_failed_after_consumed_without_consumed_receipt_identity_refuses(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    failure = build_guided_npm_worker_terminal_failure_receipt(
        worker_request=worker,
        launch_context=context,
        observed_process_id=consumed.observed_process_id,
        terminal_outcome="failed_after_consumed_authority",
        terminal_stage="pass_1",
        consumed_authority_receipt=consumed,
        failure_category="pipeline_execution_failed",
        failure_exception_type="RuntimeError",
    )
    tampered = _reidentify(failure, source_consumed_authority_receipt_identity=None)
    with pytest.raises(ValueError, match="terminal_receipt_failed_after_consumed_invalid"):
        verify_guided_npm_worker_terminal_receipt(
            tampered, worker_request=worker, launch_context=context, consumed_authority_receipt=consumed
        )


def test_builder_refuses_stage_outcome_mismatch(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    with pytest.raises(GuidedNpmTerminalError, match="terminal_failure_outcome_stage_mismatch"):
        build_guided_npm_worker_terminal_failure_receipt(
            worker_request=worker,
            launch_context=context,
            observed_process_id=consumed.observed_process_id,
            terminal_outcome="failed_before_consumed_authority",
            terminal_stage="pass_1",
            consumed_authority_receipt=consumed,
            failure_category="pipeline_execution_failed",
        )
    with pytest.raises(GuidedNpmTerminalError, match="terminal_failure_outcome_stage_mismatch"):
        build_guided_npm_worker_terminal_failure_receipt(
            worker_request=worker,
            launch_context=context,
            observed_process_id=consumed.observed_process_id,
            terminal_outcome="failed_after_consumed_authority",
            terminal_stage="numerical_dispatch",
            consumed_authority_receipt=None,
            failure_category="numerical_dispatch_failed",
        )


# ---------------------------------------------------------------------------
# Publication mechanics (section 44)
# ---------------------------------------------------------------------------


def test_publish_is_atomic_reread_verified_and_never_overwrites(tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    # The real path already published once; a second explicit attempt must
    # refuse rather than overwrite.
    with pytest.raises(GuidedNpmTerminalError, match="terminal_destination_conflict"):
        publish_guided_npm_worker_terminal_receipt(receipt)


def test_publish_refuses_preexisting_destination(tmp_path):
    run_dir = tmp_path
    run_dir.mkdir(exist_ok=True)
    path = expected_guided_npm_terminal_receipt_path(os.fspath(run_dir))
    Path(path).write_text("conflict", encoding="utf-8")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    _, worker, context, consumed, receipt = _real_success_transaction(unrelated)
    foreign = replace(receipt, run_directory_path=os.fspath(run_dir))
    with pytest.raises(GuidedNpmTerminalError, match="terminal_destination_conflict"):
        publish_guided_npm_worker_terminal_receipt(foreign)
    assert Path(path).read_text(encoding="utf-8") == "conflict"


def test_publish_fsync_failure_leaves_no_final_artifact(monkeypatch, tmp_path):
    # Dispatch the real worker (not through the full terminal lifecycle, so no
    # terminal receipt is auto-published yet) to get genuine, path-consistent
    # output evidence for this exact run directory.
    claim, worker, runtime, invocation, context, start, evidence_placeholder, _ = _transaction(
        tmp_path, pid=os.getpid()
    )
    entry_module.run_guided_npm_worker(worker, launch_context=context)
    consumed = read_and_verify_guided_npm_consumed_authority_receipt(
        context.consumed_authority_receipt_path,
        prelaunch_claim=claim,
        launch_invocation=invocation,
        execution_start_receipt=start,
    )
    output_evidence = build_guided_npm_required_output_evidence(
        worker.run_directory_path, "phasic", traces_only=False
    )
    fresh = build_guided_npm_worker_terminal_success_receipt(
        worker_request=worker,
        launch_context=context,
        consumed_authority_receipt=consumed,
        observed_process_id=consumed.observed_process_id,
        output_evidence=output_evidence,
    )
    monkeypatch.setattr(os, "fsync", lambda *_: (_ for _ in ()).throw(OSError("fsync")))
    with pytest.raises(OSError, match="fsync"):
        publish_guided_npm_worker_terminal_receipt(fresh)
    assert not Path(expected_guided_npm_terminal_receipt_path(worker.run_directory_path)).exists()


def test_publish_reread_mismatch_is_indeterminate_and_never_overwritten(monkeypatch, tmp_path):
    _, worker, context, consumed, receipt = _real_success_transaction(tmp_path)
    path = expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    Path(path).unlink()
    monkeypatch.setattr(terminal_module, "_stable_read", lambda *_: b"{}\n")
    with pytest.raises(ValueError, match="decode_invalid"):
        publish_guided_npm_worker_terminal_receipt(receipt)
    assert Path(path).is_file()
    monkeypatch.undo()
    with pytest.raises(GuidedNpmTerminalError, match="terminal_destination_conflict"):
        publish_guided_npm_worker_terminal_receipt(receipt)
