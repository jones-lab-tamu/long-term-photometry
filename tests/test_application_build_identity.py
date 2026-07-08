from __future__ import annotations

import ast
import builtins
import os
from pathlib import Path
import subprocess
import pytest

import importlib.metadata

from photometry_pipeline.application_build_identity import (
    resolve_application_build_identity,
    ApplicationBuildIdentityProviderResult,
    SOURCE_LAUNCH_VERSION_SENTINEL,
    _check_git_available,
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    if not _check_git_available():
        pytest.skip("Git command is not available in this environment.")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo_dir), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return repo_dir


def commit_file(repo: Path, filename: str, content: str) -> str:
    filepath = repo / filename
    filepath.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=str(repo), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "commit", "-m", f"Add {filename}"], cwd=str(repo), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), check=True, stdout=subprocess.PIPE, text=True)
    return res.stdout.strip()


# 1. Clean Git repo resolves
def test_clean_git_repo_resolves(git_repo: Path):
    commit_sha = commit_file(git_repo, "file.txt", "initial content")
    result = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
    )
    assert result.status == "resolved"
    assert result.build_identity is not None
    assert result.build_identity.source_revision_kind == "git"
    assert result.build_identity.source_revision == commit_sha
    assert result.build_identity.source_tree_state == "clean"
    assert result.build_identity.source_tree_digest is None
    assert result.build_identity.build_artifact_digest is None
    assert len(result.build_identity.canonical_identity) == 64


# 2. Dirty tracked file resolves when allowed
def test_dirty_tracked_file_resolves(git_repo: Path):
    commit_file(git_repo, "file.txt", "initial content")
    # Modify tracked file
    filepath = git_repo / "file.txt"
    filepath.write_text("modified content", encoding="utf-8")
    
    result = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    assert result.status == "resolved"
    assert result.build_identity is not None
    assert result.build_identity.source_tree_state == "dirty_content_bound"
    assert result.build_identity.source_tree_digest is not None
    
    digest_1 = result.build_identity.source_tree_digest
    
    # Digest changes when file content changes
    filepath.write_text("another modification", encoding="utf-8")
    result_2 = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    assert result_2.build_identity is not None
    assert result_2.build_identity.source_tree_digest != digest_1


# 3. Dirty tracked file refuses when dirty content-bound identity is not allowed
def test_dirty_tracked_file_refuses_when_not_allowed(git_repo: Path):
    commit_file(git_repo, "file.txt", "initial content")
    # Modify tracked file
    filepath = git_repo / "file.txt"
    filepath.write_text("modified content", encoding="utf-8")
    
    result = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=False,
    )
    assert result.status == "unresolved"
    assert result.build_identity is None
    assert result.blocking_issues[0].category == "git_tree_state_unavailable"


# 4. Untracked included file affects digest
def test_untracked_included_file_affects_digest(git_repo: Path):
    commit_file(git_repo, "file.txt", "initial content")
    result_clean = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
    )
    assert result_clean.build_identity is not None
    assert result_clean.build_identity.source_tree_state == "clean"

    # Add untracked included file
    untracked = git_repo / "new_file.txt"
    untracked.write_text("untracked content", encoding="utf-8")

    result_dirty = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    assert result_dirty.status == "resolved"
    assert result_dirty.build_identity is not None
    assert result_dirty.build_identity.source_tree_state == "dirty_content_bound"
    assert result_dirty.build_identity.source_tree_digest is not None


# 5. Ignored/cache/generated file does not affect digest
def test_ignored_cache_generated_file_does_not_affect_digest(git_repo: Path):
    commit_file(git_repo, "file.txt", "initial content")
    result_clean = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    assert result_clean.build_identity is not None
    assert result_clean.build_identity.source_tree_state == "clean"

    # Create excluded cache/generated files
    pycache = git_repo / "__pycache__"
    pycache.mkdir()
    (pycache / "file.pyc").write_text("binary", encoding="utf-8")
    
    pytest_cache = git_repo / ".pytest_cache"
    pytest_cache.mkdir()
    (pytest_cache / "test.json").write_text("{}", encoding="utf-8")

    diff_file = git_repo / "full_diff_test.txt"
    diff_file.write_text("diff content", encoding="utf-8")

    result_after = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    assert result_after.build_identity is not None
    # Still resolves as clean because all transient files are excluded
    assert result_after.build_identity.source_tree_state == "clean"


# 5b. Dirty files under ordinary folders like scratch or example_data do dirty the tree
def test_scratch_and_example_data_files_dirty_tree(git_repo: Path):
    commit_file(git_repo, "file.txt", "initial content")
    
    # Create file under scratch/
    scratch_dir = git_repo / "scratch"
    scratch_dir.mkdir()
    (scratch_dir / "note.txt").write_text("scratch note", encoding="utf-8")
    
    result = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    assert result.status == "resolved"
    assert result.build_identity is not None
    # Must be dirty because scratch/ is no longer excluded
    assert result.build_identity.source_tree_state == "dirty_content_bound"

    # Re-commit to clean the repository
    commit_file(git_repo, "scratch/note.txt", "scratch note")
    
    # Create file under example_data/
    example_data_dir = git_repo / "example_data"
    example_data_dir.mkdir()
    (example_data_dir / "example.txt").write_text("example data", encoding="utf-8")
    
    result_example = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    assert result_example.status == "resolved"
    assert result_example.build_identity is not None
    # Must be dirty because example_data/ is no longer excluded
    assert result_example.build_identity.source_tree_state == "dirty_content_bound"


# 6. Deterministic digest
def test_deterministic_digest(git_repo: Path):
    commit_file(git_repo, "file.txt", "initial content")
    untracked = git_repo / "new_file.txt"
    untracked.write_text("untracked content", encoding="utf-8")

    res_1 = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    res_2 = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    assert res_1.build_identity is not None
    assert res_2.build_identity is not None
    assert res_1.build_identity.source_tree_digest == res_2.build_identity.source_tree_digest


# 7. Non-Git path without packaged digest refuses
def test_non_git_path_without_packaged_digest_refuses(tmp_path: Path):
    non_git_dir = tmp_path / "nongit"
    non_git_dir.mkdir()
    result = resolve_application_build_identity(
        project_root=non_git_dir,
        distribution_version="1.0.0",
        packaged_artifact_digest=None,
    )
    assert result.status == "unresolved"
    assert result.build_identity is None
    assert result.blocking_issues[0].category in ("repository_root_not_found", "git_unavailable")


# 8. Packaged artifact fallback resolves
def test_packaged_artifact_fallback_resolves(tmp_path: Path):
    non_git_dir = tmp_path / "nongit"
    non_git_dir.mkdir()
    valid_digest = "a" * 64
    result = resolve_application_build_identity(
        project_root=non_git_dir,
        distribution_version="1.0.0",
        packaged_artifact_digest=valid_digest,
    )
    assert result.status == "resolved"
    assert result.build_identity is not None
    assert result.build_identity.source_revision_kind == "packaged_artifact"
    assert result.build_identity.distribution_version == "1.0.0"
    assert result.build_identity.source_tree_state == "unavailable"
    assert result.build_identity.build_artifact_digest == valid_digest


# 9. Invalid packaged digest refuses
def test_invalid_packaged_digest_refuses(tmp_path: Path):
    non_git_dir = tmp_path / "nongit"
    non_git_dir.mkdir()
    invalid_digest = "not_sha_256"
    result = resolve_application_build_identity(
        project_root=non_git_dir,
        distribution_version="1.0.0",
        packaged_artifact_digest=invalid_digest,
    )
    assert result.status == "unresolved"
    assert result.build_identity is None
    assert result.blocking_issues[0].category == "packaged_artifact_identity_unavailable"


# 10. Provider no writes
def test_provider_no_writes(git_repo: Path, monkeypatch: pytest.MonkeyPatch):
    commit_file(git_repo, "file.txt", "initial content")
    
    # Make the repository dirty
    (git_repo / "file.txt").write_text("dirty content", encoding="utf-8")
    
    # Add an untracked file to be read
    (git_repo / "untracked.txt").write_text("some content", encoding="utf-8")

    def raise_write_error(*args, **kwargs):
        raise AssertionError("Writes are strictly forbidden during identity resolution.")

    # Mock all Python write operations inside the repository
    original_open = builtins.open
    def mock_open(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "w+ax"):
            try:
                file_path = Path(file).resolve()
                if git_repo.resolve() in file_path.parents or file_path == git_repo.resolve():
                    raise_write_error()
            except Exception:
                pass
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", mock_open)
    monkeypatch.setattr(Path, "write_text", raise_write_error)
    monkeypatch.setattr(Path, "write_bytes", raise_write_error)
    monkeypatch.setattr(Path, "mkdir", raise_write_error)
    monkeypatch.setattr(Path, "touch", raise_write_error)
    monkeypatch.setattr(os, "makedirs", raise_write_error)
    
    # Subprocesses should not be blocked since they are read-only git calls.
    # Reads (via builtins.open, filepath.is_symlink(), os.readlink, stat) should succeed.
    result = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    assert result.status == "resolved"
    assert result.build_identity is not None
    assert result.build_identity.source_tree_state == "dirty_content_bound"
    assert result.build_identity.source_tree_digest is not None


# 11. Import boundary
def test_import_boundary():
    module_path = Path(__file__).parent.parent / "photometry_pipeline" / "application_build_identity.py"
    source = module_path.read_text(encoding="utf-8")
    
    imports = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
            
    prohibited = {
        "gui",
        "run_spec",
        "runner",
        "process_runner",
        "output_allocator",
        "config_writer",
        "artifact_writer",
        "report_writer",
        "status_writer",
        "manifest_writer",
    }
    
    for imported in imports:
        parts = imported.split(".")
        for p in prohibited:
            assert p not in parts, f"Prohibited import detected: {imported}"


# 12. Identity sensitivity
def test_identity_sensitivity(git_repo: Path):
    commit_file(git_repo, "file.txt", "initial content")
    
    res_base = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
    )
    
    # New commit changes canonical identity
    commit_2 = commit_file(git_repo, "file_2.txt", "more content")
    res_commit_changed = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
    )
    assert res_commit_changed.build_identity.canonical_identity != res_base.build_identity.canonical_identity
    
    # Distribution version change changes canonical identity
    res_ver_changed = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="2.0.0",
    )
    assert res_ver_changed.build_identity.canonical_identity != res_commit_changed.build_identity.canonical_identity

    # Dirty digest change changes canonical identity
    filepath = git_repo / "file.txt"
    filepath.write_text("modified", encoding="utf-8")
    res_dirty_1 = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    filepath.write_text("modified again", encoding="utf-8")
    res_dirty_2 = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
        allow_dirty_content_bound=True,
    )
    assert res_dirty_1.build_identity.canonical_identity != res_dirty_2.build_identity.canonical_identity


def _make_package_not_found_version(target_name: str):
    """A stand-in for importlib.metadata.version that raises
    PackageNotFoundError only for `target_name`, matching a real machine
    where that one distribution has no installed metadata (a source
    checkout with no pyproject.toml/setup.py and no prior `pip install`),
    without disturbing lookups for any other distribution."""
    original = importlib.metadata.version

    def fake_version(name, *args, **kwargs):
        if name == target_name:
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name, *args, **kwargs)

    return fake_version


# 13. Source launch (no installed package metadata, no explicit override)
# resolves a real git identity using the source-launch sentinel version.
def test_source_launch_without_package_metadata_resolves_git_identity(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    commit_sha = commit_file(git_repo, "file.txt", "initial content")
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        _make_package_not_found_version("photometry-pipeline"),
    )

    result = resolve_application_build_identity(project_root=git_repo)

    assert result.status == "resolved"
    assert result.build_identity is not None
    assert result.build_identity.distribution_version == (
        SOURCE_LAUNCH_VERSION_SENTINEL
    )
    assert result.build_identity.source_revision_kind == "git"
    assert result.build_identity.source_revision == commit_sha
    assert result.build_identity.source_tree_state == "clean"
    assert len(result.build_identity.canonical_identity) == 64


# 14. Source launch with a dirty tree still resolves, with the real dirty
# state and content digest computed exactly as an installed launch would.
def test_source_launch_without_package_metadata_resolves_dirty_git_identity(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    commit_file(git_repo, "file.txt", "initial content")
    (git_repo / "file.txt").write_text("modified content", encoding="utf-8")
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        _make_package_not_found_version("photometry-pipeline"),
    )

    result = resolve_application_build_identity(
        project_root=git_repo,
        allow_dirty_content_bound=True,
    )

    assert result.status == "resolved"
    assert result.build_identity is not None
    assert result.build_identity.distribution_version == (
        SOURCE_LAUNCH_VERSION_SENTINEL
    )
    assert result.build_identity.source_revision_kind == "git"
    assert result.build_identity.source_tree_state == "dirty_content_bound"
    assert result.build_identity.source_tree_digest is not None


# 15. An explicitly supplied distribution_version is completely unaffected
# by missing package metadata: importlib.metadata.version must not even be
# consulted, and existing explicit-version behavior is unchanged.
def test_explicit_distribution_version_bypasses_metadata_lookup_entirely(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    commit_sha = commit_file(git_repo, "file.txt", "initial content")

    def fail_if_called(name, *args, **kwargs):
        raise AssertionError(
            "importlib.metadata.version must not be called when "
            "distribution_version is explicitly supplied"
        )

    monkeypatch.setattr(importlib.metadata, "version", fail_if_called)

    result = resolve_application_build_identity(
        project_root=git_repo,
        distribution_version="1.0.0",
    )

    assert result.status == "resolved"
    assert result.build_identity is not None
    assert result.build_identity.distribution_version == "1.0.0"
    assert result.build_identity.source_revision == commit_sha


# 16. A source launch with no installed package metadata AND no git
# identity available (and no packaged-artifact fallback) still refuses,
# exactly as before -- the sentinel only ever substitutes for the missing
# *version string*; it never fabricates a source/git identity that
# genuinely cannot be resolved.
def test_source_launch_without_package_metadata_and_without_git_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    non_git_dir = tmp_path / "nongit"
    non_git_dir.mkdir()
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        _make_package_not_found_version("photometry-pipeline"),
    )

    result = resolve_application_build_identity(
        project_root=non_git_dir,
        packaged_artifact_digest=None,
    )

    assert result.status == "unresolved"
    assert result.build_identity is None
    assert result.blocking_issues[0].category in (
        "repository_root_not_found",
        "git_unavailable",
    )
