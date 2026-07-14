from __future__ import annotations

import builtins
import hashlib
import os
from dataclasses import replace
from pathlib import Path
import shutil

import pytest

import photometry_pipeline.guided_npm_authorization as authorization_module
import photometry_pipeline.guided_normalized_recording as normalized_recording
import photometry_pipeline.io.npm_contract as npm_contract
import photometry_pipeline.io.npm_source_snapshot as npm_source_snapshot
from photometry_pipeline.guided_npm_authorization import (
    GUIDED_NPM_AUTHORIZATION_HASH_CHUNK_SIZE,
    GUIDED_NPM_AUTHORIZATION_STATUS_AUTHORIZED_FOR_STARTUP_PREPARATION,
    GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_NAME,
    GuidedNpmExecutionAuthorization,
    GuidedNpmExecutionAuthorizationCancelled,
    GuidedNpmExecutionAuthorizationFailure,
    authorize_guided_npm_execution_authority,
    compute_guided_npm_execution_authorization_identity,
)
from photometry_pipeline.guided_npm_execution_authority import (
    GuidedNpmExecutionAuthority,
    compute_guided_npm_authorized_session_identity,
    compute_guided_npm_execution_authority_identity,
    compute_guided_npm_session_sequence_identity,
)
from photometry_pipeline.guided_production_mapping import (
    build_application_build_identity,
)

from tests.test_guided_npm_execution_authority import (
    _accepted_authority,
    _accepted_two_session_authority,
)


def _authorize(authority, **changes):
    return authorize_guided_npm_execution_authority(
        authority,
        expected_validation_revision=changes.get(
            "expected_validation_revision", authority.validation_revision
        ),
        expected_plan_identity=changes.get(
            "expected_plan_identity", authority.guided_plan_identity
        ),
        current_application_build_identity=changes.get(
            "current_application_build_identity", authority.application_build_identity
        ),
        cancellation_check=changes.get("cancellation_check"),
    )


def _issue(result, category: str):
    assert isinstance(
        result,
        (GuidedNpmExecutionAuthorizationFailure, GuidedNpmExecutionAuthorizationCancelled),
    )
    assert result.blocking_issues[0].category == category
    return result.blocking_issues[0]


def _unsafe_replace(instance, **changes):
    clone = object.__new__(type(instance))
    for name, value in instance.__dict__.items():
        object.__setattr__(clone, name, changes.get(name, value))
    for name, value in changes.items():
        if name not in instance.__dict__:
            object.__setattr__(clone, name, value)
    return clone


def _source_path(authority: GuidedNpmExecutionAuthority, position: int = 0) -> Path:
    return Path(authority.sessions[position].authorized_absolute_source_reference)


def _accepted_authority_at(path: Path):
    path.mkdir(parents=True)
    return _accepted_authority(path)


def _rebind_sessions(authority, sessions):
    candidate = replace(
        authority,
        sessions=tuple(sessions),
        session_sequence_identity="0" * 64,
        canonical_authority_identity="0" * 64,
    )
    candidate = replace(
        candidate,
        session_sequence_identity=compute_guided_npm_session_sequence_identity(
            candidate.sessions
        ),
    )
    return replace(
        candidate,
        canonical_authority_identity=compute_guided_npm_execution_authority_identity(
            candidate
        ),
    )


def _changed_build(authority, **changes):
    current = authority.application_build_identity
    values = {
        "distribution_name": current.distribution_name,
        "distribution_version": current.distribution_version,
        "source_revision_kind": current.source_revision_kind,
        "source_revision": current.source_revision,
        "source_tree_state": current.source_tree_state,
        "source_tree_digest": current.source_tree_digest,
        "build_artifact_digest": current.build_artifact_digest,
        "identity_provider_version": current.identity_provider_version,
    }
    values.update(changes)
    return build_application_build_identity(**values)


def test_valid_authority_is_authorized_for_startup_preparation_only(tmp_path: Path):
    authority = _accepted_two_session_authority(tmp_path)
    result = _authorize(authority)
    assert isinstance(result, GuidedNpmExecutionAuthorization)
    assert result.authorization_schema_name == GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_NAME
    assert result.authorization_status == (
        GUIDED_NPM_AUTHORIZATION_STATUS_AUTHORIZED_FOR_STARTUP_PREPARATION
    )
    assert result.startup_status == "not_materialized"
    assert result.runnable is False
    assert tuple(
        item.canonical_relative_path
        for item in result.verified_source_snapshot.ordered_files
    ) == tuple(item.canonical_relative_path for item in authority.sessions)
    assert result.source_authority_identity == authority.canonical_authority_identity
    with pytest.raises((AttributeError, TypeError)):
        result.runnable = True


@pytest.mark.parametrize(
    ("mutation", "category"),
    (
        ("wrong_type", "authority_missing_or_invalid"),
        ("schema", "authority_schema_unsupported"),
        ("top_identity", "authority_identity_mismatch"),
        ("nested_identity", "authority_identity_mismatch"),
        ("authorized_state", "authority_state_invalid"),
        ("startup_state", "authority_state_invalid"),
        ("runnable", "authority_state_invalid"),
    ),
)
def test_authority_is_fully_verified_before_source_inspection(
    tmp_path: Path, monkeypatch, mutation: str, category: str
):
    authority = _accepted_authority(tmp_path)
    if mutation == "wrong_type":
        candidate = object()
    elif mutation == "schema":
        candidate = _unsafe_replace(authority, authority_schema_version="v999")
    elif mutation == "top_identity":
        candidate = _unsafe_replace(authority, canonical_authority_identity="0" * 64)
    elif mutation == "nested_identity":
        policy = _unsafe_replace(
            authority.recording_policy, canonical_policy_identity="0" * 64
        )
        candidate = _unsafe_replace(authority, recording_policy=policy)
    elif mutation == "authorized_state":
        candidate = _unsafe_replace(
            authority, authorization_status="authorized_for_startup_preparation"
        )
    elif mutation == "startup_state":
        candidate = _unsafe_replace(authority, startup_status="materialized")
    else:
        candidate = _unsafe_replace(authority, runnable=True)

    monkeypatch.setattr(
        authorization_module,
        "discover_npm_source_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source inspection occurred before authority refusal")
        ),
    )
    if mutation == "wrong_type":
        result = authorize_guided_npm_execution_authority(
            candidate,
            expected_validation_revision=authority.validation_revision,
            expected_plan_identity=authority.guided_plan_identity,
            current_application_build_identity=authority.application_build_identity,
        )
    else:
        result = _authorize(candidate)
    _issue(result, category)


@pytest.mark.parametrize(
    ("changes", "category", "detail_code"),
    (
        (
            {"expected_validation_revision": None},
            "validation_revision_missing",
            "expected_validation_revision_missing",
        ),
        (
            {"expected_validation_revision": True},
            "validation_revision_missing",
            "expected_validation_revision_invalid",
        ),
        (
            {"expected_validation_revision": 99},
            "validation_revision_mismatch",
            "validation_revision_mismatch",
        ),
        (
            {"expected_plan_identity": ""},
            "guided_plan_identity_missing",
            "expected_plan_identity_invalid",
        ),
        (
            {"expected_plan_identity": "0" * 64},
            "guided_plan_identity_mismatch",
            "guided_plan_identity_mismatch",
        ),
        (
            {"expected_plan_identity": "request_identity"},
            "guided_plan_identity_mismatch",
            "request_identity_substituted_for_plan_identity",
        ),
    ),
)
def test_revision_and_plan_context_refusals(
    tmp_path: Path, changes: dict, category: str, detail_code: str
):
    authority = _accepted_authority(tmp_path)
    if changes.get("expected_plan_identity") == "request_identity":
        changes["expected_plan_identity"] = authority.source_request_identity
    issue = _issue(_authorize(authority, **changes), category)
    assert issue.detail_code == detail_code


@pytest.mark.parametrize(
    "build_changes",
    (
        {"source_revision": "changed-revision"},
        {
            "source_tree_state": "dirty_content_bound",
            "source_tree_digest": "1" * 64,
        },
        {"build_artifact_digest": "2" * 64},
    ),
)
def test_changed_build_facts_refuse(tmp_path: Path, build_changes: dict):
    authority = _accepted_authority(tmp_path)
    changed = _changed_build(authority, **build_changes)
    _issue(
        _authorize(authority, current_application_build_identity=changed),
        "application_build_identity_mismatch",
    )


def test_malformed_current_build_identity_refuses(tmp_path: Path):
    authority = _accepted_authority(tmp_path)
    malformed = _unsafe_replace(
        authority.application_build_identity, canonical_identity="0" * 64
    )
    _issue(
        _authorize(authority, current_application_build_identity=malformed),
        "application_build_identity_invalid",
    )


def test_new_matching_csv_refuses_but_unrelated_file_is_ignored(tmp_path: Path):
    authority = _accepted_authority(tmp_path)
    root = _source_path(authority).parent
    (root / "notes.txt").write_text("ignored", encoding="utf-8")
    assert isinstance(_authorize(authority), GuidedNpmExecutionAuthorization)
    (root / "new_matching_source.CSV").write_bytes(b"new")
    _issue(_authorize(authority), "source_set_extra_file")


def test_deleted_renamed_and_same_content_alternate_paths_refuse(tmp_path: Path):
    for operation in ("delete", "rename", "same_content_alternate"):
        case_root = tmp_path / operation
        authority = _accepted_authority_at(case_root)
        source = _source_path(authority)
        if operation == "delete":
            source.unlink()
        elif operation == "rename":
            source.rename(source.with_name("renamed.csv"))
        else:
            content = source.read_bytes()
            source.unlink()
            source.with_name("alternate.csv").write_bytes(content)
        _issue(_authorize(authority), "source_set_missing_file")


def test_reverse_filesystem_enumeration_keeps_authority_order_and_identity(
    tmp_path: Path, monkeypatch
):
    authority = _accepted_two_session_authority(tmp_path)
    baseline = _authorize(authority)
    assert isinstance(baseline, GuidedNpmExecutionAuthorization)
    real_discover = authorization_module.discover_npm_source_files

    def reverse_discovery(root):
        discovered = real_discover(root)
        return replace(discovered, files=tuple(reversed(discovered.files)))

    monkeypatch.setattr(
        authorization_module, "discover_npm_source_files", reverse_discovery
    )
    reversed_result = _authorize(authority)
    assert isinstance(reversed_result, GuidedNpmExecutionAuthorization)
    assert reversed_result.verified_source_snapshot == baseline.verified_source_snapshot
    assert (
        reversed_result.canonical_authorization_identity
        == baseline.canonical_authorization_identity
    )


def test_modification_times_do_not_determine_session_order(tmp_path: Path):
    authority = _accepted_two_session_authority(tmp_path)
    os.utime(_source_path(authority, 0), ns=(1_000_000_000, 9_000_000_000))
    os.utime(_source_path(authority, 1), ns=(1_000_000_000, 2_000_000_000))
    result = _authorize(authority)
    assert isinstance(result, GuidedNpmExecutionAuthorization)
    assert tuple(
        item.canonical_relative_path
        for item in result.verified_source_snapshot.ordered_files
    ) == tuple(item.canonical_relative_path for item in authority.sessions)


def test_duplicate_discovered_canonical_path_refuses(tmp_path: Path, monkeypatch):
    authority = _accepted_authority(tmp_path)
    real_discover = authorization_module.discover_npm_source_files

    def duplicate(root):
        discovered = real_discover(root)
        return replace(discovered, files=discovered.files + discovered.files[:1])

    monkeypatch.setattr(authorization_module, "discover_npm_source_files", duplicate)
    _issue(_authorize(authority), "source_set_duplicate_path")


def test_second_enumeration_detects_added_file(tmp_path: Path, monkeypatch):
    authority = _accepted_authority(tmp_path)
    root = _source_path(authority).parent
    real_discover = authorization_module.discover_npm_source_files
    calls = 0

    def mutate_on_second(source_root):
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / "appeared.csv").write_bytes(b"new")
        return real_discover(source_root)

    monkeypatch.setattr(
        authorization_module, "discover_npm_source_files", mutate_on_second
    )
    _issue(_authorize(authority), "source_set_changed_during_verification")


def test_second_enumeration_detects_removed_file(tmp_path: Path, monkeypatch):
    authority = _accepted_two_session_authority(tmp_path)
    real_discover = authorization_module.discover_npm_source_files
    calls = 0

    def mutate_on_second(source_root):
        nonlocal calls
        calls += 1
        if calls == 2:
            _source_path(authority, 1).unlink()
        return real_discover(source_root)

    monkeypatch.setattr(
        authorization_module, "discover_npm_source_files", mutate_on_second
    )
    _issue(_authorize(authority), "source_set_changed_during_verification")


def test_final_stat_detects_file_change_after_second_enumeration(
    tmp_path: Path, monkeypatch
):
    authority = _accepted_authority(tmp_path)
    path = _source_path(authority)
    real_discover = authorization_module.discover_npm_source_files
    calls = 0

    def mutate_after_second(source_root):
        nonlocal calls
        calls += 1
        result = real_discover(source_root)
        if calls == 2:
            current = path.stat()
            os.utime(
                path,
                ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
            )
        return result

    monkeypatch.setattr(
        authorization_module, "discover_npm_source_files", mutate_after_second
    )
    _issue(_authorize(authority), "source_file_changed_during_verification")


def test_size_and_same_size_digest_mutations_refuse(tmp_path: Path):
    size_authority = _accepted_authority_at(tmp_path / "size")
    with _source_path(size_authority).open("ab") as handle:
        handle.write(b"x")
    _issue(_authorize(size_authority), "source_file_size_mismatch")

    digest_authority = _accepted_authority_at(tmp_path / "digest")
    path = _source_path(digest_authority)
    content = bytearray(path.read_bytes())
    content[0] ^= 1
    path.write_bytes(content)
    _issue(_authorize(digest_authority), "source_file_digest_mismatch")


def test_zero_byte_authorized_csv_is_hashed_without_content_interpretation(tmp_path: Path):
    authority = _accepted_authority(tmp_path)
    path = _source_path(authority)
    path.write_bytes(b"")
    session = replace(
        authority.sessions[0],
        size_bytes=0,
        sha256_content_digest=hashlib.sha256(b"").hexdigest(),
        canonical_session_identity="0" * 64,
    )
    session = replace(
        session,
        canonical_session_identity=compute_guided_npm_authorized_session_identity(
            session
        ),
    )
    rebound = _rebind_sessions(authority, (session,))
    assert isinstance(_authorize(rebound), GuidedNpmExecutionAuthorization)


def test_hashing_uses_bounded_binary_reads(tmp_path: Path, monkeypatch):
    authority = _accepted_authority(tmp_path)
    authorized_path = _source_path(authority)
    real_open = builtins.open
    read_sizes: list[int] = []

    class GuardedReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, size=-1):
            read_sizes.append(size)
            assert size == GUIDED_NPM_AUTHORIZATION_HASH_CHUNK_SIZE
            return self.handle.read(size)

    def guarded_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if Path(path) == authorized_path and mode == "rb":
            return GuardedReader(handle)
        return handle

    monkeypatch.setattr(authorization_module, "open", guarded_open, raising=False)
    assert isinstance(_authorize(authority), GuidedNpmExecutionAuthorization)
    assert read_sizes


@pytest.mark.parametrize("mutation", ("size", "mtime"))
def test_file_mutation_during_hash_refuses(
    tmp_path: Path, monkeypatch, mutation: str
):
    authority = _accepted_authority(tmp_path)
    path = _source_path(authority)
    real_open = builtins.open
    changed = False

    class MutatingReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, size=-1):
            nonlocal changed
            data = self.handle.read(size)
            if data and not changed:
                changed = True
                if mutation == "size":
                    with real_open(path, "ab") as writer:
                        writer.write(b"x")
                else:
                    current = path.stat()
                    os.utime(
                        path,
                        ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
                    )
            return data

    def mutating_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if Path(file) == path and mode == "rb":
            return MutatingReader(handle)
        return handle

    monkeypatch.setattr(authorization_module, "open", mutating_open, raising=False)
    _issue(_authorize(authority), "source_file_changed_during_verification")


def test_file_replacement_between_stat_and_open_refuses(tmp_path: Path, monkeypatch):
    authority = _accepted_authority(tmp_path)
    path = _source_path(authority)
    real_open = builtins.open
    replaced = False

    def replacing_open(file, mode="r", *args, **kwargs):
        nonlocal replaced
        if Path(file) == path and mode == "rb" and not replaced:
            replaced = True
            replacement_path = path.with_suffix(".replacement")
            replacement_path.write_bytes(path.read_bytes())
            os.replace(replacement_path, path)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(authorization_module, "open", replacing_open, raising=False)
    _issue(_authorize(authority), "source_file_changed_during_verification")


def test_file_replacement_after_hash_before_post_stat_refuses(
    tmp_path: Path, monkeypatch
):
    authority = _accepted_authority(tmp_path)
    path = _source_path(authority)
    real_open = builtins.open

    class ReplacingOnClose:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.handle.close()
            replacement_path = path.with_suffix(".replacement")
            replacement_path.write_bytes(path.read_bytes())
            os.replace(replacement_path, path)

        def fileno(self):
            return self.handle.fileno()

        def read(self, size=-1):
            return self.handle.read(size)

    def replacing_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if Path(file) == path and mode == "rb":
            return ReplacingOnClose(handle)
        return handle

    monkeypatch.setattr(authorization_module, "open", replacing_open, raising=False)
    _issue(_authorize(authority), "source_file_changed_during_verification")


def test_missing_root_root_file_and_source_directory_refuse(tmp_path: Path):
    missing_authority = _accepted_authority_at(tmp_path / "missing")
    missing_root = _source_path(missing_authority).parent
    shutil.rmtree(missing_root)
    _issue(_authorize(missing_authority), "source_root_missing")

    file_authority = _accepted_authority_at(tmp_path / "file")
    file_root = _source_path(file_authority).parent
    shutil.rmtree(file_root)
    file_root.write_bytes(b"not a directory")
    _issue(_authorize(file_authority), "source_root_not_directory")

    directory_authority = _accepted_authority_at(tmp_path / "directory")
    source = _source_path(directory_authority)
    source.unlink()
    source.mkdir()
    _issue(_authorize(directory_authority), "source_file_not_regular")


def test_unreadable_root_discovery_and_source_file_refuse(tmp_path: Path, monkeypatch):
    root_authority = _accepted_authority_at(tmp_path / "root")
    monkeypatch.setattr(authorization_module.os, "access", lambda *_args: False)
    _issue(_authorize(root_authority), "source_root_unreadable")
    monkeypatch.undo()

    discovery_authority = _accepted_authority_at(tmp_path / "discovery")
    monkeypatch.setattr(
        authorization_module,
        "discover_npm_source_files",
        lambda *_args: (_ for _ in ()).throw(PermissionError("denied")),
    )
    _issue(_authorize(discovery_authority), "source_discovery_failed")
    monkeypatch.undo()

    file_authority = _accepted_authority_at(tmp_path / "source")
    source = _source_path(file_authority)
    real_open = builtins.open

    def unreadable_open(path, mode="r", *args, **kwargs):
        if Path(path) == source and mode == "rb":
            raise PermissionError("denied")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(authorization_module, "open", unreadable_open, raising=False)
    _issue(_authorize(file_authority), "source_file_unreadable")


def test_traversal_authority_is_refused_upstream_before_source_access(tmp_path: Path):
    authority = _accepted_authority(tmp_path)
    session = _unsafe_replace(
        authority.sessions[0], canonical_relative_path="../outside.csv"
    )
    tampered = _unsafe_replace(authority, sessions=(session,))
    _issue(_authorize(tampered), "authority_identity_mismatch")


def test_symlink_escape_and_broken_symlink_refuse_when_supported(tmp_path: Path):
    escape_authority = _accepted_authority_at(tmp_path / "escape")
    source = _source_path(escape_authority)
    outside = tmp_path / "outside.csv"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    try:
        os.symlink(outside, source)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")
    _issue(_authorize(escape_authority), "source_path_outside_root")

    broken_authority = _accepted_authority_at(tmp_path / "broken")
    broken = _source_path(broken_authority)
    broken.unlink()
    os.symlink(broken.with_name("missing.csv"), broken)
    _issue(_authorize(broken_authority), "source_path_alias_mismatch")


def test_no_csv_or_scientific_reinterpretation_is_called(tmp_path: Path, monkeypatch):
    authority = _accepted_authority(tmp_path)

    def fail(*_args, **_kwargs):
        raise AssertionError("B2-C3 attempted source reinterpretation")

    monkeypatch.setattr(npm_source_snapshot, "parse_npm_filename_timestamp", fail)
    monkeypatch.setattr(npm_source_snapshot, "_hash_stable", fail)
    monkeypatch.setattr(npm_contract, "inspect_npm_csv", fail)
    monkeypatch.setattr(
        normalized_recording, "build_npm_normalized_recording_description", fail
    )
    try:
        import pandas as pd

        monkeypatch.setattr(pd, "read_csv", fail)
    except ImportError:
        pass
    assert isinstance(_authorize(authority), GuidedNpmExecutionAuthorization)


def test_no_output_or_startup_mutation_occurs(tmp_path: Path, monkeypatch):
    authority = _accepted_authority(tmp_path)

    def fail(*_args, **_kwargs):
        raise AssertionError("B2-C3 attempted output or startup mutation")

    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    result = _authorize(authority)
    assert isinstance(result, GuidedNpmExecutionAuthorization)
    assert result.startup_status == "not_materialized"
    assert result.runnable is False


def test_cancellation_before_enumeration_and_during_hash(tmp_path: Path):
    before_authority = _accepted_authority_at(tmp_path / "before")
    result = _authorize(before_authority, cancellation_check=lambda: True)
    assert isinstance(result, GuidedNpmExecutionAuthorizationCancelled)

    hash_authority = _accepted_authority_at(tmp_path / "hash")
    calls = 0

    def during_hash():
        nonlocal calls
        calls += 1
        return calls == 5

    result = _authorize(hash_authority, cancellation_check=during_hash)
    assert isinstance(result, GuidedNpmExecutionAuthorizationCancelled)


def test_cancellation_between_files(tmp_path: Path, monkeypatch):
    authority = _accepted_two_session_authority(tmp_path)
    real_verify = authorization_module._verify_one_file
    cancelled = False
    verified_count = 0

    def wrapped(*args, **kwargs):
        nonlocal cancelled, verified_count
        result = real_verify(*args, **kwargs)
        verified_count += 1
        if verified_count == 1:
            cancelled = True
        return result

    monkeypatch.setattr(authorization_module, "_verify_one_file", wrapped)
    result = _authorize(authority, cancellation_check=lambda: cancelled)
    assert isinstance(result, GuidedNpmExecutionAuthorizationCancelled)
    assert verified_count == 1


def test_cancellation_after_verification_before_success(tmp_path: Path, monkeypatch):
    authority = _accepted_authority(tmp_path)
    real_discover = authorization_module.discover_npm_source_files
    calls = 0
    cancelled = False

    def wrapped(root):
        nonlocal calls, cancelled
        calls += 1
        result = real_discover(root)
        if calls == 2:
            cancelled = True
        return result

    monkeypatch.setattr(authorization_module, "discover_npm_source_files", wrapped)
    result = _authorize(authority, cancellation_check=lambda: cancelled)
    assert isinstance(result, GuidedNpmExecutionAuthorizationCancelled)


def test_snapshot_and_authorization_identities_are_deterministic(tmp_path: Path):
    authority = _accepted_two_session_authority(tmp_path)
    first = _authorize(authority)
    second = _authorize(authority)
    assert isinstance(first, GuidedNpmExecutionAuthorization)
    assert second == first
    assert first.verified_source_snapshot == second.verified_source_snapshot
    assert compute_guided_npm_execution_authorization_identity(first) == (
        first.canonical_authorization_identity
    )

    changed = replace(
        first,
        source_authority_identity="0" * 64,
        canonical_authorization_identity="0" * 64,
    )
    assert compute_guided_npm_execution_authorization_identity(changed) != (
        first.canonical_authorization_identity
    )
