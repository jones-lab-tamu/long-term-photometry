from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import os
from pathlib import Path

import pytest

from photometry_pipeline.io import rwd_source_snapshot
from photometry_pipeline.io.rwd_source_snapshot import (
    GuidedRwdSourceCandidateFile,
    GuidedRwdSourceCandidateSnapshot,
    RwdSourceSnapshotError,
    build_rwd_source_candidate_snapshot,
    compute_incomplete_final_chunk_classification_digest,
    compute_rwd_source_candidate_content_digest,
    compute_rwd_source_candidate_set_digest,
    make_not_requested_incomplete_final_chunk_classification,
)


def _write_session(root: Path, name: str, content: bytes) -> Path:
    session = root / name
    session.mkdir()
    target = session / "fluorescence.csv"
    target.write_bytes(content)
    return target


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    _write_session(root, "2025_01_01-00_00_00", b"abc\n")
    _write_session(root, "2025_01_01-00_10_00", b"def\n")
    return root


def _tree(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


def test_same_stable_files_produce_identical_snapshot_and_digests(tmp_path: Path):
    root = _dataset(tmp_path)
    first = build_rwd_source_candidate_snapshot(str(root))
    second = build_rwd_source_candidate_snapshot(str(root))
    assert first == second
    assert compute_rwd_source_candidate_set_digest(first) == first.source_candidate_set_digest
    assert compute_rwd_source_candidate_content_digest(first) == first.source_candidate_content_digest


def test_discovery_enumeration_order_does_not_change_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _dataset(tmp_path)
    first = build_rwd_source_candidate_snapshot(str(root))
    original = rwd_source_snapshot._scandir_entries

    def reversed_entries(path: str):
        return list(reversed(original(path)))

    monkeypatch.setattr(rwd_source_snapshot, "_scandir_entries", reversed_entries)
    second = build_rwd_source_candidate_snapshot(str(root))
    assert second.source_candidate_set_digest == first.source_candidate_set_digest
    assert second.source_candidate_content_digest == first.source_candidate_content_digest


def test_same_size_edit_preserves_set_digest_and_changes_content_digest(tmp_path: Path):
    root = _dataset(tmp_path)
    first = build_rwd_source_candidate_snapshot(str(root))
    target = root / "2025_01_01-00_00_00" / "fluorescence.csv"
    target.write_bytes(b"xyz\n")
    second = build_rwd_source_candidate_snapshot(str(root))
    assert second.source_candidate_set_digest == first.source_candidate_set_digest
    assert second.source_candidate_content_digest != first.source_candidate_content_digest


@pytest.mark.parametrize("change", ["add", "remove", "rename"])
def test_candidate_set_changes_affect_both_digests(tmp_path: Path, change: str):
    root = _dataset(tmp_path)
    first = build_rwd_source_candidate_snapshot(str(root))
    if change == "add":
        _write_session(root, "2025_01_01-00_20_00", b"ghi\n")
    elif change == "remove":
        target = root / "2025_01_01-00_10_00" / "fluorescence.csv"
        target.unlink()
        target.parent.rmdir()
    else:
        # Renamed to another canonical RWD session timestamp -- still a
        # trustworthy chronology, just a different session identity/time.
        (root / "2025_01_01-00_10_00").rename(root / "2025_01_01-00_15_00")
    second = build_rwd_source_candidate_snapshot(str(root))
    assert second.source_candidate_set_digest != first.source_candidate_set_digest
    assert second.source_candidate_content_digest != first.source_candidate_content_digest


def test_renamed_session_with_non_canonical_name_fails_closed(tmp_path: Path):
    """A folder renamed away from the canonical RWD timestamp format is not
    a trustworthy acquisition time and must refuse rather than silently
    keep its old (now-incorrect) position or digest it as data (A2)."""
    root = _dataset(tmp_path)
    (root / "2025_01_01-00_10_00").rename(root / "renamed_session")
    with pytest.raises(RwdSourceSnapshotError) as excinfo:
        build_rwd_source_candidate_snapshot(str(root))
    assert excinfo.value.category == "malformed_session_timestamp"


def test_ignored_file_changes_summary_but_not_semantic_digests(tmp_path: Path):
    root = _dataset(tmp_path)
    first = build_rwd_source_candidate_snapshot(str(root))
    (root / "ignored.txt").write_text("ignored", encoding="utf-8")
    second = build_rwd_source_candidate_snapshot(str(root))
    assert second.ignored_summary != first.ignored_summary
    assert second.source_candidate_set_digest == first.source_candidate_set_digest
    assert second.source_candidate_content_digest == first.source_candidate_content_digest


def test_root_level_fluorescence_blocks(tmp_path: Path):
    root = _dataset(tmp_path)
    (root / "fluorescence.csv").write_text("bad root", encoding="utf-8")
    with pytest.raises(RwdSourceSnapshotError) as excinfo:
        build_rwd_source_candidate_snapshot(str(root))
    assert excinfo.value.category == "unsupported_root_level_rwd_layout"


def test_bounded_nested_acquisition_root_blocks(tmp_path: Path):
    root = tmp_path / "raw"
    nested = root / "wrapper" / "session"
    nested.mkdir(parents=True)
    (nested / "fluorescence.csv").write_text("nested", encoding="utf-8")
    with pytest.raises(RwdSourceSnapshotError) as excinfo:
        build_rwd_source_candidate_snapshot(str(root))
    assert excinfo.value.category == "ambiguous_nested_rwd_root"


def test_zero_candidates_blocks(tmp_path: Path):
    root = tmp_path / "raw"
    root.mkdir()
    with pytest.raises(RwdSourceSnapshotError) as excinfo:
        build_rwd_source_candidate_snapshot(str(root))
    assert excinfo.value.category == "no_rwd_fluorescence_files"


def test_non_regular_target_blocks(tmp_path: Path):
    root = tmp_path / "raw"
    target = root / "session" / "fluorescence.csv"
    target.mkdir(parents=True)
    with pytest.raises(RwdSourceSnapshotError) as excinfo:
        build_rwd_source_candidate_snapshot(str(root))
    assert excinfo.value.category == "candidate_non_regular"


def test_unreadable_target_blocks_via_read_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _dataset(tmp_path)

    def deny_open(path: str):
        raise PermissionError("denied")

    monkeypatch.setattr(rwd_source_snapshot, "_open_candidate", deny_open)
    with pytest.raises(RwdSourceSnapshotError) as excinfo:
        build_rwd_source_candidate_snapshot(str(root))
    assert excinfo.value.category == "candidate_unreadable"


def test_repeated_pre_post_discovery_mismatch_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _dataset(tmp_path)
    original = rwd_source_snapshot._scan_candidates
    calls = 0

    def changing_scan(source_root: str, path_style: str):
        nonlocal calls
        calls += 1
        result = original(source_root, path_style)
        if calls % 2 == 0:
            return replace(result, candidates=result.candidates[:-1])
        return result

    monkeypatch.setattr(rwd_source_snapshot, "_scan_candidates", changing_scan)
    with pytest.raises(RwdSourceSnapshotError) as excinfo:
        build_rwd_source_candidate_snapshot(str(root))
    assert excinfo.value.category == "source_changed_during_snapshot"
    assert calls == 4


def test_file_mutation_after_read_retries_then_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _dataset(tmp_path)
    original = rwd_source_snapshot._read_candidate_digest
    mutations = 0

    def mutate_after_read(candidate, cancellation_check):
        nonlocal mutations
        result = original(candidate, cancellation_check)
        with open(candidate.runtime_path, "ab") as handle:
            handle.write(b"x")
        mutations += 1
        return result

    monkeypatch.setattr(
        rwd_source_snapshot,
        "_read_candidate_digest",
        mutate_after_read,
    )
    with pytest.raises(RwdSourceSnapshotError) as excinfo:
        build_rwd_source_candidate_snapshot(str(root))
    assert excinfo.value.category == "source_changed_during_snapshot"
    assert mutations == 2


def test_cancellation_returns_no_partial_snapshot(tmp_path: Path):
    root = _dataset(tmp_path)
    with pytest.raises(RwdSourceSnapshotError) as excinfo:
        build_rwd_source_candidate_snapshot(
            str(root),
            cancellation_check=lambda: True,
        )
    assert excinfo.value.category == "source_candidate_snapshot_cancelled"


def test_snapshot_operation_writes_no_files_or_directories(tmp_path: Path):
    root = _dataset(tmp_path)
    before = _tree(root)
    build_rwd_source_candidate_snapshot(str(root))
    assert _tree(root) == before


def test_set_digest_excludes_content_hash_but_content_digest_includes_it(tmp_path: Path):
    snapshot = build_rwd_source_candidate_snapshot(str(_dataset(tmp_path)))
    first = snapshot.candidates[0]
    replacement_digest = hashlib.sha256(b"replacement").hexdigest()
    changed_candidate = replace(first, sha256_content_digest=replacement_digest)
    changed_candidates = (changed_candidate,) + snapshot.candidates[1:]
    semantic = {
        "snapshot_schema_name": snapshot.snapshot_schema_name,
        "snapshot_schema_version": snapshot.snapshot_schema_version,
        "discovery_rule_version": snapshot.discovery_rule_version,
        "path_canonicalization_version": snapshot.path_canonicalization_version,
        "relative_path_rule_version": snapshot.relative_path_rule_version,
        "digest_algorithm": snapshot.digest_algorithm,
        "source_root_canonical": snapshot.source_root_canonical,
        "source_root_path_style": snapshot.source_root_path_style,
        "source_format": snapshot.source_format,
        "acquisition_mode": snapshot.acquisition_mode,
        "candidates": changed_candidates,
        "ignored_files_policy": snapshot.ignored_files_policy,
        "build_mode": snapshot.build_mode,
        "unresolved_inputs": (),
    }
    assert compute_rwd_source_candidate_set_digest(semantic) == snapshot.source_candidate_set_digest
    assert compute_rwd_source_candidate_content_digest(semantic) != snapshot.source_candidate_content_digest


@pytest.mark.parametrize("field", ["source_candidate_set_digest", "source_candidate_content_digest"])
def test_stored_snapshot_digest_mismatch_blocks(tmp_path: Path, field: str):
    snapshot = build_rwd_source_candidate_snapshot(str(_dataset(tmp_path)))
    malformed = replace(snapshot, **{field: "0" * 64})
    with pytest.raises(RwdSourceSnapshotError) as excinfo:
        if field == "source_candidate_set_digest":
            compute_rwd_source_candidate_set_digest(malformed)
        else:
            compute_rwd_source_candidate_content_digest(malformed)
    assert excinfo.value.category == "source_candidate_digest_mismatch"
    with pytest.raises(RwdSourceSnapshotError):
        make_not_requested_incomplete_final_chunk_classification(malformed)


def test_not_requested_classification_is_snapshot_bound_and_empty(tmp_path: Path):
    snapshot = build_rwd_source_candidate_snapshot(str(_dataset(tmp_path)))
    classification = make_not_requested_incomplete_final_chunk_classification(
        snapshot
    )
    assert classification.classification_status == "not_requested"
    assert classification.source_candidate_set_digest == snapshot.source_candidate_set_digest
    assert classification.source_candidate_content_digest == snapshot.source_candidate_content_digest
    assert classification.excluded_canonical_relative_path is None
    assert classification.reason is None
    assert classification.evidence is None
    assert classification.parsing_contract_digest is None
    assert classification.timing_contract_digest is None
    assert classification.unresolved_inputs == ()
    assert classification.policy.exclude_incomplete_final_rwd_chunk is False


def test_not_requested_does_not_inspect_headers_or_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    snapshot = build_rwd_source_candidate_snapshot(str(_dataset(tmp_path)))

    def forbidden(*args, **kwargs):
        raise AssertionError("not_requested must not inspect source content")

    monkeypatch.setattr(rwd_source_snapshot, "_open_candidate", forbidden)
    classification = make_not_requested_incomplete_final_chunk_classification(
        snapshot
    )
    assert classification.classification_status == "not_requested"


def test_classification_digest_is_deterministic_and_snapshot_sensitive(tmp_path: Path):
    root = _dataset(tmp_path)
    first_snapshot = build_rwd_source_candidate_snapshot(str(root))
    first = make_not_requested_incomplete_final_chunk_classification(first_snapshot)
    assert (
        compute_incomplete_final_chunk_classification_digest(first)
        == compute_incomplete_final_chunk_classification_digest(first)
    )
    (root / "2025_01_01-00_00_00" / "fluorescence.csv").write_bytes(b"xyz\n")
    second_snapshot = build_rwd_source_candidate_snapshot(str(root))
    second = make_not_requested_incomplete_final_chunk_classification(second_snapshot)
    assert (
        compute_incomplete_final_chunk_classification_digest(first)
        != compute_incomplete_final_chunk_classification_digest(second)
    )


def test_malformed_snapshot_blocks_classification(tmp_path: Path):
    snapshot = build_rwd_source_candidate_snapshot(str(_dataset(tmp_path)))
    malformed = replace(snapshot, candidates=())
    with pytest.raises(RwdSourceSnapshotError) as excinfo:
        make_not_requested_incomplete_final_chunk_classification(malformed)
    assert excinfo.value.category == "invalid_rwd_source_snapshot"


def test_module_has_no_forbidden_imports():
    source = Path(rwd_source_snapshot.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    forbidden = (
        "gui",
        "subprocess",
        "pandas",
        "yaml",
        "photometry_pipeline.config",
        "run_spec",
        "run_report_parser",
    )
    assert not any(
        imported == item or imported.startswith(f"{item}.")
        for imported in imports
        for item in forbidden
    )
    assert "Config(" not in source
    assert "RunSpec" not in source
