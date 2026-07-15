from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path

import pytest

from photometry_pipeline.guided_npm_worker_request import (
    GuidedNpmWorkerRequest,
    canonical_guided_npm_worker_request_bytes,
    compute_guided_npm_live_freshness_evidence_identity,
    compute_guided_npm_live_verified_file_sequence_identity,
    compute_guided_npm_live_verified_source_file_identity,
    compute_guided_npm_worker_request_identity,
    decode_canonical_guided_npm_worker_request_bytes,
    deserialize_guided_npm_worker_request,
    serialize_guided_npm_worker_request,
    verify_guided_npm_worker_request,
    build_guided_npm_worker_request,
)
import photometry_pipeline.guided_npm_worker_request_materialization as materialization_module
from photometry_pipeline.guided_npm_worker_request_materialization import (
    GuidedNpmWorkerRequestMaterializationReceipt,
    materialize_guided_npm_worker_request,
)
from tests.test_guided_npm_production_execution_request import _request


def _worker(tmp_path: Path):
    _, _, execution = _request(tmp_path)
    receipt = materialize_guided_npm_worker_request(
        execution,
        current_application_build_identity=execution.application_build_identity,
    )
    assert isinstance(receipt, GuidedNpmWorkerRequestMaterializationReceipt)
    return decode_canonical_guided_npm_worker_request_bytes(Path(receipt.worker_request_artifact_path).read_bytes())


def _in_memory_worker(tmp_path: Path):
    _, _, execution = _request(tmp_path)
    materialization_module._verify_startup_artifact(execution, None)
    evidence = materialization_module._verify_live_sources(execution, None)
    worker = build_guided_npm_worker_request(execution, evidence)
    return execution, worker


def _reidentify(worker, **changes):
    changed = replace(worker, **changes, canonical_worker_request_identity="0" * 64)
    return replace(changed, canonical_worker_request_identity=compute_guided_npm_worker_request_identity(changed))


def _reidentify_file(value, **changes):
    changed = replace(value, **changes, canonical_live_verified_source_file_identity="0" * 64)
    return replace(changed, canonical_live_verified_source_file_identity=compute_guided_npm_live_verified_source_file_identity(changed))


def _reidentify_evidence(value, **changes):
    changed = replace(value, **changes, canonical_live_freshness_evidence_identity="0" * 64)
    return replace(changed, canonical_live_freshness_evidence_identity=compute_guided_npm_live_freshness_evidence_identity(changed))


def test_worker_request_is_frozen_complete_and_non_runnable(tmp_path):
    worker = _worker(tmp_path)
    assert isinstance(worker, GuidedNpmWorkerRequest)
    assert worker.execution_request.canonical_execution_request_identity == worker.source_execution_request_identity
    assert worker.request_status == "constructed_for_worker"
    assert "persisted_and_verified" not in serialize_guided_npm_worker_request(worker).values()
    assert (worker.launch_status, worker.execution_status, worker.completion_status, worker.runnable) == (
        "not_launched", "not_started", "not_available", False,
    )
    with pytest.raises(FrozenInstanceError):
        worker.runnable = True


def test_deterministic_round_trip_and_canonical_bytes(tmp_path):
    worker = _worker(tmp_path)
    payload = serialize_guided_npm_worker_request(worker)
    assert deserialize_guided_npm_worker_request(copy.deepcopy(payload)) == worker
    first = canonical_guided_npm_worker_request_bytes(worker)
    assert first == canonical_guided_npm_worker_request_bytes(worker)
    assert first.endswith(b"\n") and not first.startswith(b"\xef\xbb\xbf")
    assert decode_canonical_guided_npm_worker_request_bytes(first) == worker


@pytest.mark.parametrize("field,value", [
    ("request_status", "persisted_and_verified"),
    ("launch_status", "launched"),
    ("execution_status", "running"),
    ("completion_status", "completed"),
    ("runnable", True),
    ("worker_request_artifact_filename", "request.json"),
    ("guided_plan_identity", "a" * 64),
    ("validation_revision", 999),
])
def test_outer_reidentified_state_or_provenance_tampering_refuses(tmp_path, field, value):
    worker = _worker(tmp_path)
    with pytest.raises(ValueError):
        verify_guided_npm_worker_request(_reidentify(worker, **{field: value}))


@pytest.mark.parametrize("field", [
    "observed_size_bytes", "observed_sha256", "source_path", "canonical_relative_path",
])
def test_outer_reidentified_live_file_tampering_refuses(tmp_path, field):
    worker = _worker(tmp_path)
    live = worker.live_freshness_evidence.ordered_verified_files[0]
    value = live.observed_size_bytes + 1 if field == "observed_size_bytes" else "a" * 64 if field == "observed_sha256" else live.source_path + ".changed"
    changed_live = _reidentify_file(live, **{field: value})
    files = (changed_live,) + worker.live_freshness_evidence.ordered_verified_files[1:]
    evidence = _reidentify_evidence(
        worker.live_freshness_evidence,
        ordered_verified_files=files,
        live_verified_file_sequence_identity=compute_guided_npm_live_verified_file_sequence_identity(files),
    )
    with pytest.raises(ValueError):
        verify_guided_npm_worker_request(_reidentify(worker, live_freshness_evidence=evidence))


def test_live_file_identity_fact_tampering_refuses(tmp_path):
    worker = _worker(tmp_path)
    live = worker.live_freshness_evidence.ordered_verified_files[0]
    facts = replace(live.opened_file_facts, mtime_ns=live.opened_file_facts.mtime_ns + 1)
    changed_live = _reidentify_file(live, opened_file_facts=facts)
    files = (changed_live,) + worker.live_freshness_evidence.ordered_verified_files[1:]
    evidence = _reidentify_evidence(worker.live_freshness_evidence, ordered_verified_files=files, live_verified_file_sequence_identity=compute_guided_npm_live_verified_file_sequence_identity(files))
    with pytest.raises(ValueError):
        verify_guided_npm_worker_request(_reidentify(worker, live_freshness_evidence=evidence))


@pytest.mark.parametrize("field", ["live_membership_identity", "live_content_identity", "live_session_sequence_identity", "freshness_status"])
def test_live_evidence_semantic_tampering_refuses(tmp_path, field):
    worker = _worker(tmp_path)
    value = "not_verified" if field == "freshness_status" else "a" * 64
    evidence = _reidentify_evidence(worker.live_freshness_evidence, **{field: value})
    with pytest.raises(ValueError):
        verify_guided_npm_worker_request(_reidentify(worker, live_freshness_evidence=evidence))


@pytest.mark.parametrize("mutation", ["missing", "extra", "domain", "schema", "nested_identity"])
def test_serialized_tampering_refuses(tmp_path, mutation):
    payload = copy.deepcopy(serialize_guided_npm_worker_request(_worker(tmp_path)))
    if mutation == "missing":
        del payload["launch_status"]
    elif mutation == "extra":
        payload["unknown"] = True
    elif mutation == "domain":
        payload["identity_domain"] = "wrong"
    elif mutation == "schema":
        payload["worker_request_schema_version"] = "v2"
    else:
        payload["live_freshness_evidence"]["ordered_verified_files"][0]["canonical_live_verified_source_file_identity"] = "a" * 64
    with pytest.raises(ValueError, match="worker_request_serialization_invalid"):
        deserialize_guided_npm_worker_request(payload)


@pytest.mark.parametrize("content", [
    b"\xef\xbb\xbf{}\n",
    b'{"identity_domain":"guided_npm_worker_request.v1","identity_domain":"guided_npm_worker_request.v1"}\n',
    b'{"value":NaN}\n',
    b"\xff\n",
    b"{} trailing",
])
def test_invalid_json_bytes_refuse(content):
    with pytest.raises(ValueError, match="worker_request_noncanonical"):
        decode_canonical_guided_npm_worker_request_bytes(content)


def test_noncanonical_format_and_reordered_live_sequence_refuse(tmp_path):
    worker = _worker(tmp_path)
    pretty = (json.dumps(serialize_guided_npm_worker_request(worker), indent=2) + "\n").encode()
    with pytest.raises(ValueError, match="worker_request_noncanonical"):
        decode_canonical_guided_npm_worker_request_bytes(pretty)
    payload = serialize_guided_npm_worker_request(worker)
    payload["live_freshness_evidence"]["ordered_verified_files"].reverse()
    with pytest.raises(ValueError):
        deserialize_guided_npm_worker_request(payload)


def test_worker_request_verifies_and_serializes_before_publication(tmp_path):
    execution, worker = _in_memory_worker(tmp_path)
    artifact = Path(execution.output_runtime_projection.run_directory_path, "guided_npm_worker_request.json")
    assert not artifact.exists()
    verify_guided_npm_worker_request(worker)
    content = canonical_guided_npm_worker_request_bytes(worker)
    assert decode_canonical_guided_npm_worker_request_bytes(content) == worker
    assert worker.request_status == "constructed_for_worker"
    assert not artifact.exists()


def test_worker_request_verification_performs_no_filesystem_access(monkeypatch, tmp_path):
    _, worker = _in_memory_worker(tmp_path)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("filesystem open"))
    monkeypatch.setattr(Path, "stat", lambda *args, **kwargs: pytest.fail("filesystem stat"))
    monkeypatch.setattr(Path, "exists", lambda *args, **kwargs: pytest.fail("filesystem exists"))
    monkeypatch.setattr(os, "stat", lambda *args, **kwargs: pytest.fail("os.stat"))
    verify_guided_npm_worker_request(worker)
    serialize_guided_npm_worker_request(worker)
