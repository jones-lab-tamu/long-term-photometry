from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_npm_worker_acknowledgement as ack_module
import photometry_pipeline.guided_npm_worker_launch as launch_module
import photometry_pipeline.guided_npm_worker_entry as entry_module
from photometry_pipeline.config import Config
from photometry_pipeline.guided_npm_authorized_adapter import (
    build_guided_npm_authorized_runtime,
    load_guided_npm_authorized_chunk_with_record,
)
from photometry_pipeline.guided_npm_worker_acknowledgement import (
    GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_FILENAME,
    GuidedNpmConsumedAuthorityEvidence,
    GuidedNpmWorkerConsumedAuthorityReceipt,
    build_guided_npm_worker_consumed_authority_receipt,
    build_guided_npm_worker_launch_context,
    compute_guided_npm_consumed_source_record_identity,
    compute_guided_npm_worker_consumed_authority_receipt_identity,
    decode_guided_npm_worker_consumed_authority_receipt_bytes,
    expected_guided_npm_consumed_authority_receipt_path,
    persist_guided_npm_worker_launch_context,
    publish_guided_npm_worker_consumed_authority_receipt,
    read_and_verify_guided_npm_consumed_authority_receipt,
    read_guided_npm_worker_launch_context,
    serialize_guided_npm_worker_consumed_authority_receipt,
    verify_guided_npm_child_consumed_authority_receipt,
    verify_guided_npm_worker_consumed_authority_receipt,
)
from photometry_pipeline.guided_npm_worker_launch import (
    GUIDED_NPM_LAUNCHER_KIND,
    GuidedNpmStartedProcess,
    build_guided_npm_worker_launch_invocation,
    launch_guided_npm_worker,
)
from photometry_pipeline.guided_npm_worker_entry import run_guided_npm_worker
from photometry_pipeline.pipeline import Pipeline
from tests.test_guided_npm_worker_prelaunch_claim import _claim
from tests.test_guided_npm_worker_prelaunch_claim import _different_build
from tests.test_guided_npm_authorized_adapter import _explicit_csv, _mixed_gap_runtime


def _transaction(tmp_path: Path, *, pid: int = 4242):
    _, _, claim = _claim(tmp_path)
    worker = claim.worker_request
    runtime = build_guided_npm_authorized_runtime(worker)
    invocation = build_guided_npm_worker_launch_invocation(
        claim,
        current_application_build_identity=claim.application_build_identity,
    )
    context = build_guided_npm_worker_launch_context(invocation)
    start = launch_module._build_execution_start_receipt(
        claim,
        invocation,
        GuidedNpmStartedProcess(pid, GUIDED_NPM_LAUNCHER_KIND),
    )
    records = tuple(
        load_guided_npm_authorized_chunk_with_record(
            runtime.authorized_input, path, runtime.config, position
        ).consumed_source_record
        for position, path in enumerate(runtime.authorized_input.ordered_session_paths)
    )
    evidence = GuidedNpmConsumedAuthorityEvidence(
        runtime.canonical_guided_npm_authorized_runtime_identity,
        runtime.correction_authority_identity,
        runtime.feature_authority_identity,
        records,
    )
    receipt = build_guided_npm_worker_consumed_authority_receipt(
        worker_request=worker,
        launch_context=context,
        evidence=evidence,
        observed_process_id=pid,
    )
    return claim, worker, runtime, invocation, context, start, evidence, receipt


def _reidentify_receipt(receipt, **changes):
    changed = replace(
        receipt,
        **changes,
        canonical_consumed_authority_receipt_identity="0" * 64,
    )
    return replace(
        changed,
        canonical_consumed_authority_receipt_identity=(
            compute_guided_npm_worker_consumed_authority_receipt_identity(changed)
        ),
    )


def test_consumed_source_records_are_exact_ordered_loader_observations(tmp_path):
    _, worker, runtime, _, _, _, evidence, _ = _transaction(tmp_path)
    sessions = worker.execution_request.source_runtime_projection.ordered_sessions
    records = evidence.consumed_source_records
    assert len(records) == len(sessions)
    assert tuple(item.chronological_position for item in records) == tuple(range(len(records)))
    assert tuple(item.source_path for item in records) == runtime.authorized_input.ordered_session_paths
    assert tuple(item.actual_elapsed_sec for item in records) == runtime.authorized_input.actual_elapsed_sec_by_chunk
    for record, session in zip(records, sessions, strict=True):
        assert record.source_size_bytes == Path(record.source_path).stat().st_size
        assert record.source_sha256 == session.source_sha256
        assert record.resolved_timestamp_column == session.resolved_timestamp_column
        assert record.physical_to_canonical_roi_map == runtime.authorized_input.physical_to_canonical_roi_map
        assert record.canonical_consumed_source_record_identity == compute_guided_npm_consumed_source_record_identity(record)


def test_runtime_and_dispatch_authority_identities_are_exact_and_deterministic(tmp_path):
    _, worker, runtime, _, _, _, evidence, receipt = _transaction(tmp_path)
    rebuilt = build_guided_npm_authorized_runtime(worker)
    assert runtime.canonical_guided_npm_authorized_runtime_identity == rebuilt.canonical_guided_npm_authorized_runtime_identity
    assert receipt.authorized_runtime_identity == runtime.canonical_guided_npm_authorized_runtime_identity
    assert receipt.correction_authority_identity == worker.execution_request.correction_runtime_projection.canonical_correction_runtime_projection_identity
    assert receipt.feature_authority_identity == worker.execution_request.feature_runtime_projection.canonical_feature_runtime_projection_identity
    assert evidence.numerical_dispatch_status == "entered"


def test_receipt_schema_statuses_and_canonical_round_trip(tmp_path):
    _, worker, _, invocation, context, start, evidence, receipt = _transaction(tmp_path)
    assert (
        receipt.worker_acceptance_status,
        receipt.consumed_authority_status,
        receipt.numerical_dispatch_status,
        receipt.completion_status,
    ) == ("accepted_exact_worker_authority", "verified", "entered", "not_available")
    content = serialize_guided_npm_worker_consumed_authority_receipt(receipt)
    assert decode_guided_npm_worker_consumed_authority_receipt_bytes(content) == receipt
    verify_guided_npm_child_consumed_authority_receipt(
        receipt, worker_request=worker, launch_context=context, evidence=evidence
    )
    verify_guided_npm_worker_consumed_authority_receipt(
        receipt,
        worker_request=worker,
        launch_invocation=invocation,
        execution_start_receipt=start,
    )


@pytest.mark.parametrize("pid", [True, 0, -1])
def test_child_pid_invalid_values_refuse(tmp_path, pid):
    _, worker, _, _, context, _, evidence, _ = _transaction(tmp_path)
    with pytest.raises(ValueError, match="child_consumed_authority_receipt_invalid"):
        build_guided_npm_worker_consumed_authority_receipt(
            worker_request=worker,
            launch_context=context,
            evidence=evidence,
            observed_process_id=pid,
        )


def test_pid_and_launch_transaction_must_match_start_receipt(tmp_path):
    _, worker, _, invocation, _, start, _, receipt = _transaction(tmp_path)
    wrong_pid = replace(start, process_id=start.process_id + 1)
    wrong_pid = replace(
        wrong_pid,
        canonical_execution_start_receipt_identity=(
            launch_module.compute_guided_npm_worker_execution_start_receipt_identity(wrong_pid)
        ),
    )
    with pytest.raises(ValueError, match="consumed_authority_receipt_invalid"):
        verify_guided_npm_worker_consumed_authority_receipt(
            receipt,
            worker_request=worker,
            launch_invocation=invocation,
            execution_start_receipt=wrong_pid,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_worker_request_identity", "1" * 64),
        ("source_execution_request_identity", "1" * 64),
        ("source_launch_invocation_identity", "1" * 64),
        ("source_launch_context_identity", "1" * 64),
        ("source_prelaunch_freshness_evidence_identity", "1" * 64),
        ("guided_plan_identity", "other"),
        ("validation_revision", 999),
        ("execution_mode", "tonic"),
        ("worker_request_artifact_path", "C:\\other\\worker.json"),
        ("run_directory_path", "C:\\other"),
        ("ordered_session_paths", ("C:\\other.csv",)),
        ("ordered_session_identities", ("1" * 64,)),
        ("chronological_positions", (1, 0)),
        ("actual_elapsed_sec_by_chunk", (0.0, 60.0)),
        ("parser_contract_identity", "1" * 64),
        ("complete_physical_roi_ids", ("other",)),
        ("complete_canonical_roi_ids", ("other",)),
        ("selected_canonical_roi_ids", ("other",)),
        ("physical_to_canonical_roi_map", (("other", "other"),)),
        ("correction_authority_identity", "1" * 64),
        ("feature_authority_identity", "1" * 64),
        ("authorized_runtime_identity", "1" * 64),
        ("worker_acceptance_status", "completed"),
        ("consumed_authority_status", "unverified"),
        ("numerical_dispatch_status", "completed"),
        ("completion_status", "complete"),
    ],
)
def test_outer_reidentified_receipt_tampering_refuses(tmp_path, field, value):
    _, worker, _, invocation, _, start, _, receipt = _transaction(tmp_path)
    if field == "execution_mode" and value == receipt.execution_mode:
        value = "phasic" if value == "tonic" else "tonic"
    with pytest.raises(ValueError):
        verify_guided_npm_worker_consumed_authority_receipt(
            _reidentify_receipt(receipt, **{field: value}),
            worker_request=worker,
            launch_invocation=invocation,
            execution_start_receipt=start,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chronological_position", 1),
        ("source_path", "C:\\other.csv"),
        ("source_size_bytes", 999),
        ("source_sha256", "1" * 64),
        ("canonical_session_runtime_identity", "1" * 64),
        ("resolved_timestamp_column", "other"),
        ("timestamp_unit", "milliseconds"),
        ("support_policy", "other"),
        ("output_time_basis", "other"),
        ("actual_elapsed_sec", 999.0),
        ("recording_time_start_sec", 999.0),
        ("recording_time_end_sec", 999.0),
        ("physical_roi_ids", ("other",)),
        ("observed_physical_roi_ids", ("other",)),
        ("canonical_roi_ids", ("other",)),
        ("physical_to_canonical_roi_map", (("other", "other"),)),
    ],
)
def test_outer_reidentified_source_record_tampering_refuses(tmp_path, field, value):
    _, worker, _, invocation, _, start, _, receipt = _transaction(tmp_path)
    records = list(receipt.consumed_source_records)
    changed = replace(
        records[0],
        **{field: value},
        canonical_consumed_source_record_identity="0" * 64,
    )
    records[0] = replace(
        changed,
        canonical_consumed_source_record_identity=(
            compute_guided_npm_consumed_source_record_identity(changed)
        ),
    )
    tampered = _reidentify_receipt(receipt, consumed_source_records=tuple(records))
    with pytest.raises(ValueError, match="consumed_source_record_invalid"):
        verify_guided_npm_worker_consumed_authority_receipt(
            tampered,
            worker_request=worker,
            launch_invocation=invocation,
            execution_start_receipt=start,
        )


def test_outer_reidentified_build_tampering_refuses(tmp_path):
    _, worker, _, invocation, _, start, _, receipt = _transaction(tmp_path)
    tampered = _reidentify_receipt(
        receipt,
        application_build_identity=_different_build(receipt.application_build_identity),
    )
    with pytest.raises(ValueError, match="consumed_authority_receipt_invalid"):
        verify_guided_npm_worker_consumed_authority_receipt(
            tampered,
            worker_request=worker,
            launch_invocation=invocation,
            execution_start_receipt=start,
        )


def test_different_worker_invocation_and_start_transactions_refuse(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    tx_a = _transaction(tmp_path / "a")
    tx_b = _transaction(tmp_path / "b")
    _, worker_a, _, invocation_a, _, start_a, _, receipt_a = tx_a
    _, worker_b, _, invocation_b, _, start_b, _, _ = tx_b
    verify_guided_npm_worker_consumed_authority_receipt(
        receipt_a,
        worker_request=worker_a,
        launch_invocation=invocation_a,
        execution_start_receipt=start_a,
    )
    for worker, invocation, start in (
        (worker_b, invocation_a, start_a),
        (worker_a, invocation_b, start_a),
        (worker_a, invocation_a, start_b),
    ):
        with pytest.raises(ValueError):
            verify_guided_npm_worker_consumed_authority_receipt(
                receipt_a,
                worker_request=worker,
                launch_invocation=invocation,
                execution_start_receipt=start,
            )


def test_launch_context_and_receipt_publish_atomically_and_parent_reads(tmp_path):
    claim, worker, _, invocation, context, start, _, receipt = _transaction(tmp_path)
    context_path = persist_guided_npm_worker_launch_context(context)
    assert read_guided_npm_worker_launch_context(
        context_path, worker_request=worker
    ) == context
    receipt_path = context.consumed_authority_receipt_path
    publish_guided_npm_worker_consumed_authority_receipt(
        receipt, receipt_path=receipt_path, launch_context=context
    )
    assert read_and_verify_guided_npm_consumed_authority_receipt(
        receipt_path,
        prelaunch_claim=claim,
        launch_invocation=invocation,
        execution_start_receipt=start,
    ) == receipt
    assert Path(receipt_path).name == GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_FILENAME


def test_parent_launch_persists_exact_context_before_exact_once_process_creation(tmp_path):
    claim, worker, _, invocation, context, _, _, _ = _transaction(tmp_path)
    calls = []
    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=lambda argv, **kwargs: (
            calls.append((argv, kwargs)),
            GuidedNpmStartedProcess(777, GUIDED_NPM_LAUNCHER_KIND),
        )[1],
    )
    assert result.process_id == 777
    assert calls == [
        (
            invocation.argument_vector,
            {"cwd": invocation.working_directory_path, "shell": False},
        )
    ]
    assert read_guided_npm_worker_launch_context(
        invocation.launch_context_artifact_path, worker_request=worker
    ) == context


def test_preexisting_launch_context_refuses_before_process_creation(tmp_path):
    claim, _, _, invocation, _, _, _, _ = _transaction(tmp_path)
    Path(invocation.launch_context_artifact_path).write_text("conflict", encoding="utf-8")
    calls = []
    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=lambda *a, **k: calls.append(1),
    )
    assert result.blocking_issues[0].category == "launch_context_persistence_failed"
    assert calls == []


def test_preexisting_receipt_refuses_without_overwrite(tmp_path):
    _, _, _, _, context, _, _, receipt = _transaction(tmp_path)
    path = Path(context.consumed_authority_receipt_path)
    path.write_text("other", encoding="utf-8")
    with pytest.raises(ValueError, match="destination_conflict"):
        publish_guided_npm_worker_consumed_authority_receipt(
            receipt, receipt_path=os.fspath(path), launch_context=context
        )
    assert path.read_text(encoding="utf-8") == "other"


def test_temporary_write_and_atomic_promotion_failures_leave_no_final_receipt(monkeypatch, tmp_path):
    _, _, _, _, context, _, _, receipt = _transaction(tmp_path)
    path = Path(context.consumed_authority_receipt_path)
    monkeypatch.setattr(os, "fsync", lambda *_: (_ for _ in ()).throw(OSError("fsync")))
    with pytest.raises(OSError, match="fsync"):
        publish_guided_npm_worker_consumed_authority_receipt(
            receipt, receipt_path=os.fspath(path), launch_context=context
        )
    assert not path.exists()


def test_atomic_promotion_failure_leaves_no_final_and_does_not_retry(monkeypatch, tmp_path):
    _, _, _, _, context, _, _, receipt = _transaction(tmp_path)
    path = Path(context.consumed_authority_receipt_path)
    calls = []

    def fail_once(*args):
        calls.append(args)
        raise OSError("promotion")

    monkeypatch.setattr(os, "rename", fail_once)
    with pytest.raises(OSError, match="promotion"):
        publish_guided_npm_worker_consumed_authority_receipt(
            receipt, receipt_path=os.fspath(path), launch_context=context
        )
    assert len(calls) == 1
    assert not path.exists()


def test_reread_failure_is_indeterminate_and_never_overwritten(monkeypatch, tmp_path):
    _, _, _, _, context, _, _, receipt = _transaction(tmp_path)
    path = Path(context.consumed_authority_receipt_path)
    monkeypatch.setattr(ack_module, "_stable_read", lambda *_: b"{}\n")
    with pytest.raises(ValueError, match="decode_invalid"):
        publish_guided_npm_worker_consumed_authority_receipt(
            receipt, receipt_path=os.fspath(path), launch_context=context
        )
    assert path.is_file()
    with pytest.raises(ValueError, match="destination_conflict"):
        publish_guided_npm_worker_consumed_authority_receipt(
            receipt, receipt_path=os.fspath(path), launch_context=context
        )


def test_partial_noncanonical_wrong_path_and_symlink_receipts_refuse(tmp_path):
    claim, _, _, invocation, context, start, _, receipt = _transaction(tmp_path)
    path = Path(context.consumed_authority_receipt_path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        read_and_verify_guided_npm_consumed_authority_receipt(
            os.fspath(path), prelaunch_claim=claim, launch_invocation=invocation,
            execution_start_receipt=start,
        )
    path.unlink()
    noncanonical = json.dumps(json.loads(serialize_guided_npm_worker_consumed_authority_receipt(receipt)), indent=2).encode()
    with pytest.raises(ValueError):
        decode_guided_npm_worker_consumed_authority_receipt_bytes(noncanonical)
    with pytest.raises(ValueError, match="path_invalid"):
        read_and_verify_guided_npm_consumed_authority_receipt(
            os.fspath(path.with_name("other.json")), prelaunch_claim=claim,
            launch_invocation=invocation, execution_start_receipt=start,
        )


def test_absent_and_symlink_receipt_reader_refuse(tmp_path):
    claim, _, _, invocation, context, start, _, _ = _transaction(tmp_path)
    path = Path(context.consumed_authority_receipt_path)
    with pytest.raises(OSError):
        read_and_verify_guided_npm_consumed_authority_receipt(
            os.fspath(path), prelaunch_claim=claim, launch_invocation=invocation,
            execution_start_receipt=start,
        )
    target = path.with_name("target.json")
    target.write_text("{}\n", encoding="utf-8")
    try:
        path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="not_regular"):
        read_and_verify_guided_npm_consumed_authority_receipt(
            os.fspath(path), prelaunch_claim=claim, launch_invocation=invocation,
            execution_start_receipt=start,
        )


def test_strict_decoder_refuses_unknown_missing_and_duplicate_fields(tmp_path):
    *_, receipt = _transaction(tmp_path)
    payload = json.loads(serialize_guided_npm_worker_consumed_authority_receipt(receipt))
    payload["unknown"] = 1
    with pytest.raises(ValueError, match="decode_invalid"):
        decode_guided_npm_worker_consumed_authority_receipt_bytes(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    del payload["unknown"]
    del payload["execution_mode"]
    with pytest.raises(ValueError, match="decode_invalid"):
        decode_guided_npm_worker_consumed_authority_receipt_bytes(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    with pytest.raises(ValueError, match="decode_invalid"):
        decode_guided_npm_worker_consumed_authority_receipt_bytes(
            b'{"receipt_schema_name":"a","receipt_schema_name":"b"}\n'
        )


def test_real_pipeline_hook_is_once_after_all_sources_and_before_pass2(monkeypatch, tmp_path):
    claim, worker, runtime, _, _, _, _, _ = _transaction(tmp_path)
    pipeline = Pipeline(
        runtime.config,
        mode=runtime.mode,
        per_roi_correction=runtime.per_roi_correction,
        per_roi_feature_config=runtime.per_roi_feature_config,
        per_roi_feature_provenance=runtime.per_roi_feature_provenance,
    )
    observed = []
    monkeypatch.setattr(
        pipeline,
        "run_pass_2",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("after acknowledgement")),
    )
    with pytest.raises(RuntimeError, match="after acknowledgement"):
        pipeline.run_guided_npm_authorized(
            runtime,
            runtime.authorized_input.run_directory_path,
            on_consumed_authority_verified=observed.append,
        )
    assert len(observed) == 1
    assert tuple(item.source_path for item in observed[0].consumed_source_records) == runtime.authorized_input.ordered_session_paths
    assert observed[0].numerical_dispatch_status == "entered"


def test_consumed_evidence_binds_actual_not_nominal_three_session_chronology(monkeypatch, tmp_path):
    runtime = _mixed_gap_runtime(tmp_path)
    pipeline = Pipeline(
        runtime.config,
        mode=runtime.mode,
        per_roi_correction=runtime.per_roi_correction,
        per_roi_feature_config=runtime.per_roi_feature_config,
        per_roi_feature_provenance=runtime.per_roi_feature_provenance,
    )
    observed = []
    monkeypatch.setattr(
        pipeline,
        "run_pass_2",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("after boundary")),
    )
    with pytest.raises(RuntimeError, match="after boundary"):
        pipeline.run_guided_npm_authorized(
            runtime,
            runtime.authorized_input.run_directory_path,
            on_consumed_authority_verified=observed.append,
        )
    actual = tuple(
        record.actual_elapsed_sec
        for record in observed[0].consumed_source_records
    )
    assert actual == (0.0, 90.0, 150.0)
    assert actual != runtime.authorized_input.nominal_expected_elapsed_sec_by_chunk


def test_real_child_publishes_pid_bound_receipt_from_actual_pipeline(tmp_path):
    claim, worker, _, invocation, context, start, _, _ = _transaction(
        tmp_path, pid=os.getpid()
    )
    run_guided_npm_worker(worker, launch_context=context)
    receipt = read_and_verify_guided_npm_consumed_authority_receipt(
        context.consumed_authority_receipt_path,
        prelaunch_claim=claim,
        launch_invocation=invocation,
        execution_start_receipt=start,
    )
    assert receipt.observed_process_id == os.getpid()
    assert receipt.completion_status == "not_available"


def test_failure_after_real_child_acknowledgement_leaves_valid_receipt(monkeypatch, tmp_path):
    claim, worker, _, invocation, context, start, _, _ = _transaction(
        tmp_path, pid=os.getpid()
    )
    monkeypatch.setattr(
        Pipeline,
        "run_pass_2",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("post-ack failure")),
    )
    with pytest.raises(RuntimeError, match="post-ack failure"):
        run_guided_npm_worker(worker, launch_context=context)
    assert Path(context.consumed_authority_receipt_path).is_file()
    receipt = read_and_verify_guided_npm_consumed_authority_receipt(
        context.consumed_authority_receipt_path,
        prelaunch_claim=claim,
        launch_invocation=invocation,
        execution_start_receipt=start,
    )
    assert receipt.consumed_authority_status == "verified"
    assert receipt.completion_status == "not_available"


def test_failure_before_pass1_completion_publishes_no_acknowledgement(monkeypatch, tmp_path):
    _, _, runtime, _, _, _, _, _ = _transaction(tmp_path)
    pipeline = Pipeline(
        runtime.config,
        mode=runtime.mode,
        per_roi_correction=runtime.per_roi_correction,
        per_roi_feature_config=runtime.per_roi_feature_config,
        per_roi_feature_provenance=runtime.per_roi_feature_provenance,
    )
    observed = []
    monkeypatch.setattr(
        pipeline,
        "run_pass_1",
        lambda *_: (_ for _ in ()).throw(RuntimeError("before boundary")),
    )
    with pytest.raises(RuntimeError, match="before boundary"):
        pipeline.run_guided_npm_authorized(
            runtime,
            runtime.authorized_input.run_directory_path,
            on_consumed_authority_verified=observed.append,
        )
    assert observed == []


def test_source_mutation_after_pass1_before_publication_refuses(monkeypatch, tmp_path):
    _, _, runtime, _, _, _, _, _ = _transaction(tmp_path)
    pipeline = Pipeline(
        runtime.config,
        mode=runtime.mode,
        per_roi_correction=runtime.per_roi_correction,
        per_roi_feature_config=runtime.per_roi_feature_config,
        per_roi_feature_provenance=runtime.per_roi_feature_provenance,
    )
    original = pipeline.run_pass_1

    def mutate_after_pass1(*args, **kwargs):
        result = original(*args, **kwargs)
        path = Path(runtime.authorized_input.ordered_session_paths[-1])
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(pipeline, "run_pass_1", mutate_after_pass1)
    observed = []
    with pytest.raises(ValueError, match="changed_before_acknowledgement"):
        pipeline.run_guided_npm_authorized(
            runtime,
            runtime.authorized_input.run_directory_path,
            on_consumed_authority_verified=observed.append,
        )
    assert observed == []


def test_source_mutation_before_child_load_produces_no_record(tmp_path):
    _, _, runtime, _, _, _, _, _ = _transaction(tmp_path)
    path = Path(runtime.authorized_input.ordered_session_paths[0])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source_identity_mismatch"):
        load_guided_npm_authorized_chunk_with_record(
            runtime.authorized_input, os.fspath(path), runtime.config, 0
        )


def test_source_mutation_during_stable_read_produces_no_record(monkeypatch, tmp_path):
    _, _, runtime, _, _, _, _, _ = _transaction(tmp_path)
    target = Path(runtime.authorized_input.ordered_session_paths[0])
    original_open = Path.open

    class MutatingHandle:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self):
            content = self.handle.read()
            with open(target, "ab") as mutation:
                mutation.write(b"\n")
            return content

    def mutating_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return MutatingHandle(handle) if path == target else handle

    monkeypatch.setattr(Path, "open", mutating_open)
    with pytest.raises(ValueError, match="changed_during_read"):
        load_guided_npm_authorized_chunk_with_record(
            runtime.authorized_input, os.fspath(target), runtime.config, 0
        )


def test_d1_reader_never_uses_process_or_output_lifecycle(monkeypatch, tmp_path):
    claim, _, _, invocation, context, start, _, receipt = _transaction(tmp_path)
    publish_guided_npm_worker_consumed_authority_receipt(
        receipt,
        receipt_path=context.consumed_authority_receipt_path,
        launch_context=context,
    )
    for name in ("wait", "communicate", "poll", "terminate", "kill"):
        monkeypatch.setattr(
            launch_module.subprocess.Popen,
            name,
            lambda *a, _name=name, **k: pytest.fail(f"process lifecycle {_name}"),
            raising=False,
        )
    assert read_and_verify_guided_npm_consumed_authority_receipt(
        context.consumed_authority_receipt_path,
        prelaunch_claim=claim,
        launch_invocation=invocation,
        execution_start_receipt=start,
    ) == receipt


def test_ordinary_pipeline_has_no_d1_callback_surface():
    pipeline = Pipeline(Config())
    assert pipeline._guided_npm_consumed_authority_callback is None
    assert pipeline._guided_npm_consumed_source_records is None


def test_ordinary_npm_run_never_invokes_d1_callback(tmp_path):
    source = tmp_path / "ordinary"
    source.mkdir()
    (source / "session.csv").write_bytes(_explicit_csv())
    output = tmp_path / "ordinary-output"
    pipeline = Pipeline(Config(target_fs_hz=2.0, chunk_duration_sec=2.0))
    pipeline._guided_npm_consumed_authority_callback = lambda *_: pytest.fail(
        "ordinary NPM acknowledgement"
    )
    pipeline.run(os.fspath(source), os.fspath(output), force_format="npm")
    assert not (output / GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_FILENAME).exists()


def test_smoke_route_never_accepts_or_writes_acknowledgement(tmp_path):
    assert entry_module.main([entry_module.GUIDED_NPM_WORKER_SMOKE_ARGUMENT]) == 0
    assert entry_module.main(
        [
            entry_module.GUIDED_NPM_WORKER_SMOKE_ARGUMENT,
            entry_module.GUIDED_NPM_LAUNCH_CONTEXT_ARGUMENT,
            os.fspath(tmp_path / "context.json"),
        ]
    ) == entry_module.GUIDED_NPM_WORKER_ENTRY_REFUSED
    assert not list(tmp_path.iterdir())


def test_remove_exact_launch_context_succeeds_when_unchanged(tmp_path):
    _, _, _, _, context, _, _, _ = _transaction(tmp_path)
    path = persist_guided_npm_worker_launch_context(context)
    assert ack_module.remove_exact_guided_npm_worker_launch_context(context) is True
    assert not Path(path).exists()


def test_remove_exact_launch_context_absent_is_a_noop(tmp_path):
    _, _, _, _, context, _, _, _ = _transaction(tmp_path)
    assert ack_module.remove_exact_guided_npm_worker_launch_context(context) is True
    path = persist_guided_npm_worker_launch_context(context)
    Path(path).unlink()
    assert ack_module.remove_exact_guided_npm_worker_launch_context(context) is True


def test_remove_exact_launch_context_confirmed_absence_via_stat(monkeypatch, tmp_path):
    _, _, _, _, context, _, _, _ = _transaction(tmp_path)
    path = persist_guided_npm_worker_launch_context(context)
    original_stat = Path.stat
    stat_calls = []
    unlink_calls = []

    def patched_stat(self, *args, **kwargs):
        if os.fspath(self) == path:
            stat_calls.append(1)
            raise FileNotFoundError()
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", patched_stat)
    monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: unlink_calls.append(1))
    assert ack_module.remove_exact_guided_npm_worker_launch_context(context) is True
    assert len(stat_calls) == 1
    assert unlink_calls == []


def test_remove_exact_launch_context_initial_permission_failure_raises_cleanup_error(
    monkeypatch, tmp_path
):
    _, _, _, _, context, _, _, _ = _transaction(tmp_path)
    path = persist_guided_npm_worker_launch_context(context)
    original_stat = Path.stat
    unlink_calls = []

    def patched_stat(self, *args, **kwargs):
        if os.fspath(self) == path:
            raise PermissionError("denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", patched_stat)
    monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: unlink_calls.append(1))
    with pytest.raises(
        ack_module.GuidedNpmLaunchContextCleanupError,
        match="launch_context_cleanup_initial_stat_failed",
    ):
        ack_module.remove_exact_guided_npm_worker_launch_context(context)
    assert unlink_calls == []
    monkeypatch.undo()
    assert Path(path).is_file()


def test_remove_exact_launch_context_initial_generic_oserror_raises_cleanup_error(
    monkeypatch, tmp_path
):
    _, _, _, _, context, _, _, _ = _transaction(tmp_path)
    path = persist_guided_npm_worker_launch_context(context)
    original_stat = Path.stat
    unlink_calls = []

    def patched_stat(self, *args, **kwargs):
        if os.fspath(self) == path:
            raise OSError("boom")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", patched_stat)
    monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: unlink_calls.append(1))
    with pytest.raises(ack_module.GuidedNpmLaunchContextCleanupError):
        ack_module.remove_exact_guided_npm_worker_launch_context(context)
    assert unlink_calls == []
    monkeypatch.undo()
    assert Path(path).is_file()


def test_remove_exact_launch_context_refuses_when_replaced_by_different_valid_context(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, _, _, _, context_a, _, _, _ = _transaction(tmp_path / "a")
    _, _, _, _, context_b, _, _, _ = _transaction(tmp_path / "b")
    path = Path(persist_guided_npm_worker_launch_context(context_a))
    path.write_bytes(ack_module.serialize_guided_npm_worker_launch_context(context_b))
    with pytest.raises(ack_module.GuidedNpmLaunchContextCleanupError):
        ack_module.remove_exact_guided_npm_worker_launch_context(context_a)
    assert path.is_file()


def test_remove_exact_launch_context_refuses_when_field_changed(tmp_path):
    _, _, _, _, context, _, _, _ = _transaction(tmp_path)
    path = Path(persist_guided_npm_worker_launch_context(context))
    data = json.loads(path.read_bytes())
    data["source_worker_request_identity"] = "1" * 64
    path.write_bytes(
        (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    with pytest.raises(ack_module.GuidedNpmLaunchContextCleanupError):
        ack_module.remove_exact_guided_npm_worker_launch_context(context)
    assert path.is_file()


def test_remove_exact_launch_context_refuses_when_malformed(tmp_path):
    _, _, _, _, context, _, _, _ = _transaction(tmp_path)
    path = Path(persist_guided_npm_worker_launch_context(context))
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ack_module.GuidedNpmLaunchContextCleanupError):
        ack_module.remove_exact_guided_npm_worker_launch_context(context)
    assert path.is_file()


def test_remove_exact_launch_context_refuses_noncanonical_bytes(tmp_path):
    _, _, _, _, context, _, _, _ = _transaction(tmp_path)
    path = Path(persist_guided_npm_worker_launch_context(context))
    canonical = ack_module.serialize_guided_npm_worker_launch_context(context)
    path.write_bytes(json.dumps(json.loads(canonical), indent=2).encode())
    with pytest.raises(ack_module.GuidedNpmLaunchContextCleanupError):
        ack_module.remove_exact_guided_npm_worker_launch_context(context)
    assert path.is_file()


def test_remove_exact_launch_context_refuses_when_symlinked(tmp_path):
    _, _, _, _, context, _, _, _ = _transaction(tmp_path)
    path = Path(persist_guided_npm_worker_launch_context(context))
    target = path.with_name("elsewhere-context.json")
    target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ack_module.GuidedNpmLaunchContextCleanupError):
        ack_module.remove_exact_guided_npm_worker_launch_context(context)
    assert path.is_symlink()


def test_remove_exact_launch_context_refuses_when_hardlinked(tmp_path):
    _, _, _, _, context, _, _, _ = _transaction(tmp_path)
    path = Path(persist_guided_npm_worker_launch_context(context))
    link = path.with_name("extra-link-context.json")
    try:
        os.link(path, link)
    except OSError as exc:
        pytest.skip(f"hard link creation unavailable: {exc}")
    with pytest.raises(ack_module.GuidedNpmLaunchContextCleanupError):
        ack_module.remove_exact_guided_npm_worker_launch_context(context)
    assert path.is_file()


def test_production_cli_refuses_without_launch_context_before_worker_read(monkeypatch):
    monkeypatch.setattr(
        entry_module,
        "load_verified_guided_npm_worker_request",
        lambda *_a, **_k: pytest.fail("worker read should not occur"),
    )
    assert entry_module.main(
        ["--guided-npm-worker-request", os.path.abspath("worker.json")]
    ) == entry_module.GUIDED_NPM_WORKER_ENTRY_REFUSED
