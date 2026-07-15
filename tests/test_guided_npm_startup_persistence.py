from __future__ import annotations

import builtins
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import pytest

import photometry_pipeline.guided_npm_startup_persistence as persistence_module
import photometry_pipeline.guided_npm_authorization as authorization_module
import photometry_pipeline.guided_npm_execution_authority as authority_module
import photometry_pipeline.guided_normalized_recording as normalized_module
import photometry_pipeline.io.npm_contract as npm_contract_module
import photometry_pipeline.io.npm_source_snapshot as source_snapshot_module
from photometry_pipeline.guided_npm_startup_payload import (
    GuidedNpmStartupPayload,
    compute_guided_npm_startup_execution_projection_identity,
    compute_guided_npm_startup_output_projection_identity,
    compute_guided_npm_startup_payload_identity,
)
from photometry_pipeline.guided_npm_startup_persistence import (
    GUIDED_NPM_STARTUP_ARTIFACT_FILENAME,
    GuidedNpmStartupPersistenceCancelled,
    GuidedNpmStartupPersistenceFailure,
    GuidedNpmStartupPersistenceReceipt,
    _PersistenceRefusal,
    canonical_guided_npm_startup_payload_bytes,
    compute_guided_npm_startup_persistence_receipt_identity,
    deserialize_guided_npm_startup_persistence_receipt,
    persist_guided_npm_startup_payload,
    serialize_guided_npm_startup_persistence_receipt,
    verify_guided_npm_startup_persistence_receipt,
    verify_persisted_guided_npm_startup_artifact,
)

from tests.test_guided_npm_startup_payload import _payload


def _persist(tmp_path: Path):
    payload = _payload(tmp_path)
    result = persist_guided_npm_startup_payload(payload)
    assert isinstance(result, GuidedNpmStartupPersistenceReceipt)
    return payload, result


def _issue(result, category: str):
    assert isinstance(result, GuidedNpmStartupPersistenceFailure)
    assert result.blocking_issues[0].category == category
    return result.blocking_issues[0]


def _empty_after_failure(tmp_path: Path) -> bool:
    output = tmp_path / "planned_outputs"
    return not output.exists() or not tuple(output.rglob("*"))


def _payload_with_output_base(payload, output_base: Path):
    output = replace(
        payload.output_projection,
        output_base_canonical=os.fspath(output_base.absolute()),
        canonical_output_projection_identity="0" * 64,
    )
    output = replace(
        output,
        canonical_output_projection_identity=(
            compute_guided_npm_startup_output_projection_identity(output)
        ),
    )
    execution = replace(
        payload.execution_projection,
        output_projection_identity=output.canonical_output_projection_identity,
        canonical_execution_projection_identity="0" * 64,
    )
    execution = replace(
        execution,
        canonical_execution_projection_identity=(
            compute_guided_npm_startup_execution_projection_identity(execution)
        ),
    )
    changed = replace(
        payload,
        output_projection=output,
        execution_projection=execution,
        canonical_startup_payload_identity="0" * 64,
    )
    return replace(
        changed,
        canonical_startup_payload_identity=(
            compute_guided_npm_startup_payload_identity(changed)
        ),
    )


def _reidentify_receipt(receipt, **changes):
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


def test_persists_one_exact_canonical_artifact_and_frozen_receipt(tmp_path):
    payload, receipt = _persist(tmp_path)
    run_dir = Path(receipt.run_directory_path)
    artifact = Path(receipt.startup_artifact_path)
    assert run_dir.parent == tmp_path / "planned_outputs"
    assert tuple(run_dir.iterdir()) == (artifact,)
    assert artifact.name == GUIDED_NPM_STARTUP_ARTIFACT_FILENAME
    content = artifact.read_bytes()
    assert content == canonical_guided_npm_startup_payload_bytes(payload)
    assert content.endswith(b"\n") and not content.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in content
    assert hashlib.sha256(content).hexdigest() == receipt.serialized_payload_sha256
    assert receipt.serialized_payload_sha256 == receipt.persisted_artifact_sha256
    assert receipt.persisted_size_bytes == len(content)
    assert receipt.readback_payload_identity == payload.canonical_startup_payload_identity
    assert verify_persisted_guided_npm_startup_artifact(receipt) == payload
    verify_guided_npm_startup_persistence_receipt(receipt)
    assert receipt.runnable is False
    with pytest.raises(FrozenInstanceError):
        receipt.runnable = True


def test_canonical_bytes_are_deterministic_sorted_utf8_and_compact(tmp_path):
    payload = _payload(tmp_path)
    first = canonical_guided_npm_startup_payload_bytes(payload)
    second = canonical_guided_npm_startup_payload_bytes(payload)
    assert first == second
    decoded = first.decode("utf-8")
    value = json.loads(decoded)
    reordered = dict(reversed(tuple(value.items())))
    assert json.dumps(
        reordered,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n" == first
    assert first.startswith(b'{"acquisition_mode":')


def test_invalid_payload_refuses_before_filesystem_mutation(tmp_path):
    payload = _payload(tmp_path)
    tampered = replace(payload, canonical_startup_payload_identity="1" * 64)
    _issue(
        persist_guided_npm_startup_payload(tampered),
        "startup_payload_identity_mismatch",
    )
    assert not (tmp_path / "planned_outputs").exists()


@pytest.mark.parametrize(
    ("target", "category"),
    (
        ("write", "startup_artifact_write_failed"),
        ("flush", "startup_artifact_flush_failed"),
        ("publish", "startup_artifact_publish_failed"),
        ("directory_fsync", "startup_artifact_flush_failed"),
        ("readback", "startup_artifact_readback_failed"),
    ),
)
def test_injected_atomic_failures_return_no_receipt_and_cleanup(
    tmp_path, monkeypatch, target, category
):
    payload = _payload(tmp_path)
    if target == "write":
        monkeypatch.setattr(
            persistence_module,
            "_write_temp_file",
            lambda *_: (_ for _ in ()).throw(
                _PersistenceRefusal(
                    category, "artifact", "injected", "injected_write"
                )
            ),
        )
    elif target == "flush":
        monkeypatch.setattr(
            persistence_module.os,
            "fsync",
            lambda *_: (_ for _ in ()).throw(OSError("injected flush")),
        )
    elif target == "publish":
        monkeypatch.setattr(
            persistence_module,
            "_publish_no_replace",
            lambda *_: (_ for _ in ()).throw(
                _PersistenceRefusal(
                    category, "artifact", "injected", "injected_publish"
                )
            ),
        )
    elif target == "directory_fsync":
        monkeypatch.setattr(
            persistence_module,
            "_fsync_directory",
            lambda *_: (_ for _ in ()).throw(OSError("injected directory fsync")),
        )
    else:
        monkeypatch.setattr(
            persistence_module,
            "_read_artifact_bytes",
            lambda *_: (_ for _ in ()).throw(OSError("injected readback")),
        )
    result = persist_guided_npm_startup_payload(payload)
    _issue(result, category)
    assert _empty_after_failure(tmp_path)


def test_partial_write_failure_removes_temporary_and_run_directory(tmp_path, monkeypatch):
    payload = _payload(tmp_path)

    def partial(path, content):
        path.write_bytes(content[:17])
        raise _PersistenceRefusal(
            "startup_artifact_write_failed",
            "artifact",
            "partial",
            "partial_write",
        )

    monkeypatch.setattr(persistence_module, "_write_temp_file", partial)
    _issue(
        persist_guided_npm_startup_payload(payload),
        "startup_artifact_write_failed",
    )
    assert _empty_after_failure(tmp_path)


def test_readback_byte_mutation_refuses_and_removes_owned_artifact(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    original = persistence_module._read_artifact_bytes

    def changed(path):
        content = original(path)
        return content[:-1] + bytes((content[-1] ^ 1,))

    monkeypatch.setattr(persistence_module, "_read_artifact_bytes", changed)
    _issue(
        persist_guided_npm_startup_payload(payload),
        "startup_artifact_digest_mismatch",
    )
    assert _empty_after_failure(tmp_path)


def test_cancellation_before_and_after_publication_never_returns_success(tmp_path, monkeypatch):
    (tmp_path / "before").mkdir()
    payload = _payload(tmp_path / "before")
    result = persist_guided_npm_startup_payload(
        payload, cancellation_check=lambda: True
    )
    assert isinstance(result, GuidedNpmStartupPersistenceCancelled)
    assert not (tmp_path / "before" / "planned_outputs").exists()

    (tmp_path / "after").mkdir()
    payload = _payload(tmp_path / "after")
    calls = 0

    def cancel_after_publish():
        nonlocal calls
        calls += 1
        return calls >= 6

    result = persist_guided_npm_startup_payload(
        payload, cancellation_check=cancel_after_publish
    )
    assert isinstance(result, GuidedNpmStartupPersistenceCancelled)
    assert _empty_after_failure(tmp_path / "after")


def test_output_base_file_missing_parent_and_source_overlap_refuse(tmp_path):
    (tmp_path / "base").mkdir()
    payload = _payload(tmp_path / "base")
    output_file = tmp_path / "output-file"
    output_file.write_text("foreign", encoding="utf-8")
    _issue(
        persist_guided_npm_startup_payload(
            _payload_with_output_base(payload, output_file)
        ),
        "output_base_not_directory",
    )
    missing = tmp_path / "missing-parent" / "output"
    _issue(
        persist_guided_npm_startup_payload(
            _payload_with_output_base(payload, missing)
        ),
        "output_base_missing",
    )
    overlap = Path(payload.source_projection.source_root_canonical)
    _issue(
        persist_guided_npm_startup_payload(
            _payload_with_output_base(payload, overlap)
        ),
        "output_path_unsafe",
    )


@pytest.mark.parametrize(
    ("denied_mode", "category"),
    ((os.R_OK, "output_base_unreadable"), (os.W_OK, "output_base_unwritable")),
)
def test_output_base_access_refuses(tmp_path, monkeypatch, denied_mode, category):
    payload = _payload(tmp_path)
    output = tmp_path / "planned_outputs"
    output.mkdir()

    def access(_path, mode):
        return mode != denied_mode

    monkeypatch.setattr(persistence_module.os, "access", access)
    _issue(persist_guided_npm_startup_payload(payload), category)
    assert tuple(output.iterdir()) == ()


def test_run_directory_collision_budget_refuses_without_reuse(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    output = tmp_path / "planned_outputs"
    output.mkdir()
    (output / ("guided_npm_run_" + "a" * 32)).mkdir()
    monkeypatch.setattr(persistence_module.secrets, "token_hex", lambda _: "a" * 32)
    _issue(
        persist_guided_npm_startup_payload(payload),
        "run_directory_conflict",
    )
    assert tuple(output.iterdir()) == (output / ("guided_npm_run_" + "a" * 32),)


def test_publication_race_does_not_overwrite_or_remove_foreign_artifact(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    original = persistence_module._publish_no_replace
    foreign = b"foreign"

    def race(temp_path, final_path):
        final_path.write_bytes(foreign)
        original(temp_path, final_path)

    monkeypatch.setattr(persistence_module, "_publish_no_replace", race)
    result = persist_guided_npm_startup_payload(payload)
    _issue(result, "startup_artifact_conflict")
    assert isinstance(result, GuidedNpmStartupPersistenceFailure)
    assert result.unverified_artifact_path is not None
    artifacts = list((tmp_path / "planned_outputs").rglob(GUIDED_NPM_STARTUP_ARTIFACT_FILENAME))
    assert len(artifacts) == 1 and artifacts[0].read_bytes() == foreign


def test_persistence_receipt_serialization_round_trip_and_tampering(tmp_path):
    _, receipt = _persist(tmp_path)
    serialized = serialize_guided_npm_startup_persistence_receipt(receipt)
    assert deserialize_guided_npm_startup_persistence_receipt(serialized) == receipt
    for field, value in (
        ("persistence_schema_version", "v999"),
        ("persisted_artifact_sha256", "1" * 64),
        ("run_directory_path", os.fspath(tmp_path / "other")),
        ("runnable", True),
        ("canonical_persistence_receipt_identity", "1" * 64),
    ):
        changed = copy.deepcopy(serialized)
        changed[field] = value
        with pytest.raises(ValueError, match="persistence_receipt_serialization_invalid"):
            deserialize_guided_npm_startup_persistence_receipt(changed)
    missing = copy.deepcopy(serialized)
    del missing["guided_plan_identity"]
    with pytest.raises(ValueError, match="persistence_receipt_serialization_invalid"):
        deserialize_guided_npm_startup_persistence_receipt(missing)


def test_live_verification_rejects_noncanonical_equivalent_json(tmp_path):
    payload, receipt = _persist(tmp_path)
    artifact = Path(receipt.startup_artifact_path)
    pretty = json.dumps(
        json.loads(artifact.read_text(encoding="utf-8")),
        sort_keys=False,
        indent=2,
    ).encode("utf-8")
    artifact.write_bytes(pretty)
    changed = _reidentify_receipt(
        receipt,
        serialized_payload_sha256=hashlib.sha256(pretty).hexdigest(),
        persisted_artifact_sha256=hashlib.sha256(pretty).hexdigest(),
        persisted_size_bytes=len(pretty),
    )
    with pytest.raises(ValueError, match="startup_artifact_noncanonical"):
        verify_persisted_guided_npm_startup_artifact(changed)
    assert payload.canonical_startup_payload_identity == receipt.source_startup_payload_identity


@pytest.mark.parametrize("mutation", ("trailing", "duplicate", "invalid_utf8"))
def test_live_verification_rejects_trailing_duplicate_and_invalid_utf8(
    tmp_path, mutation
):
    _, receipt = _persist(tmp_path)
    artifact = Path(receipt.startup_artifact_path)
    content = artifact.read_bytes()
    if mutation == "trailing":
        changed_bytes = content + b"x"
    elif mutation == "duplicate":
        changed_bytes = content.replace(
            b"{",
            b'{"acquisition_mode":"intermittent",',
            1,
        )
    else:
        changed_bytes = b"\xff" + content[1:]
    artifact.write_bytes(changed_bytes)
    digest = hashlib.sha256(changed_bytes).hexdigest()
    changed = _reidentify_receipt(
        receipt,
        serialized_payload_sha256=digest,
        persisted_artifact_sha256=digest,
        persisted_size_bytes=len(changed_bytes),
    )
    with pytest.raises(ValueError, match="startup_artifact_noncanonical"):
        verify_persisted_guided_npm_startup_artifact(changed)


def test_persistence_does_not_call_source_or_scientific_construction(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    source_root = Path(payload.source_projection.source_root_canonical).resolve()
    real_io_open = io.open

    def forbidden(*_args, **_kwargs):
        raise AssertionError("scientific/source function called")

    def reject_source_open(file, *args, **kwargs):
        candidate = Path(file).resolve(strict=False)
        try:
            candidate.relative_to(source_root)
        except ValueError:
            return real_io_open(file, *args, **kwargs)
        raise AssertionError("source file opened during B2-C5 persistence")

    for target, name in (
        (authorization_module, "authorize_guided_npm_execution_authority"),
        (authorization_module, "_verify_one_file"),
        (authority_module, "build_guided_npm_execution_authority"),
        (source_snapshot_module, "build_npm_source_candidate_snapshot"),
        (source_snapshot_module, "parse_npm_filename_timestamp"),
        (npm_contract_module, "inspect_npm_csv"),
        (npm_contract_module, "resolve_npm_support_geometry"),
        (normalized_module, "build_npm_normalized_recording_description"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(io, "open", reject_source_open)
    result = persist_guided_npm_startup_payload(payload)
    assert isinstance(result, GuidedNpmStartupPersistenceReceipt)
    names = {item.name for item in Path(result.run_directory_path).iterdir()}
    assert names == {GUIDED_NPM_STARTUP_ARTIFACT_FILENAME}
    forbidden_names = {
        "guided_candidate_manifest.json",
        "session_order.json",
        "parser_policy.json",
        "correction.json",
        "feature.json",
        "completion.json",
        "terminal.json",
    }
    assert names.isdisjoint(forbidden_names)


def test_receipt_identity_binds_every_field(tmp_path):
    _, receipt = _persist(tmp_path)
    changed = replace(
        receipt,
        canonical_persistence_receipt_identity="1" * 64,
    )
    with pytest.raises(ValueError, match="persistence_receipt_identity_mismatch"):
        verify_guided_npm_startup_persistence_receipt(changed)
