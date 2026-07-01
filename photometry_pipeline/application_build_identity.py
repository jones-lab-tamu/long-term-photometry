"""Application build-identity provider for Guided Mode validation.

This module resolves the current application source code/build identity
and returns an ApplicationBuildIdentity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
from typing import Any

from photometry_pipeline.guided_production_mapping import (
    APPLICATION_BUILD_IDENTITY_PROVIDER_VERSION,
    ApplicationBuildIdentity,
    build_application_build_identity,
)
from photometry_pipeline.guided_identity import encode_canonical_value


APPLICATION_BUILD_IDENTITY_PROVIDER_REFUSAL_CATEGORIES = (
    "repository_root_not_found",
    "git_unavailable",
    "git_revision_unavailable",
    "git_tree_state_unavailable",
    "dirty_tree_digest_unavailable",
    "packaged_artifact_identity_unavailable",
    "project_version_unavailable",
    "build_identity_internal_error",
)


@dataclass(frozen=True)
class ApplicationBuildIdentityProviderIssue:
    category: str
    section: str
    message: str
    detail_code: str = ""


@dataclass(frozen=True)
class ApplicationBuildIdentityProviderResult:
    status: str
    build_identity: ApplicationBuildIdentity | None
    blocking_issues: tuple[ApplicationBuildIdentityProviderIssue, ...]
    provider_version: str


def _unresolved(category: str, message: str, detail_code: str = "") -> ApplicationBuildIdentityProviderResult:
    issue = ApplicationBuildIdentityProviderIssue(
        category=category,
        section="build_identity",
        message=message,
        detail_code=detail_code,
    )
    return ApplicationBuildIdentityProviderResult(
        status="unresolved",
        build_identity=None,
        blocking_issues=(issue,),
        provider_version=APPLICATION_BUILD_IDENTITY_PROVIDER_VERSION,
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def is_excluded_path(rel_path_str: str) -> bool:
    # Convert to POSIX style for normalization
    rel_path_str = rel_path_str.replace("\\", "/")
    parts = rel_path_str.split("/")
    
    # Exclude directory names anywhere in the path
    excluded_dirs = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
        ".git",
    }
    for part in parts:
        if part in excluded_dirs:
            return True
            
    filename = parts[-1]
    
    # Exclude Python bytecode files
    if filename.endswith((".pyc", ".pyo", ".pyd")):
        return True
        
    # Exclude generated full_diff_*.txt files
    if filename.startswith("full_diff_") and filename.endswith(".txt"):
        return True
        
    return False


def _compute_file_entry(root: Path, rel_path: str) -> dict[str, Any] | None:
    filepath = root / rel_path
    try:
        if filepath.is_symlink():
            target = os.readlink(str(filepath))
            sha256_hex = hashlib.sha256(target.encode("utf-8")).hexdigest()
            size = len(target)
            ftype = "symlink"
        else:
            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            sha256_hex = h.hexdigest()
            size = filepath.stat().st_size
            ftype = "file"
        return {
            "path": rel_path,
            "type": ftype,
            "size": size,
            "sha256": sha256_hex,
        }
    except Exception:
        return None


def _check_git_available() -> bool:
    try:
        subprocess.run(
            ["git", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2,
        )
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        res = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout
    except Exception:
        pass
    return None


def resolve_application_build_identity(
    *,
    project_root: str | Path | None = None,
    distribution_name: str = "photometry-pipeline",
    distribution_version: str | None = None,
    allow_dirty_content_bound: bool = True,
    packaged_artifact_digest: str | None = None,
) -> ApplicationBuildIdentityProviderResult:
    """Resolve the current application/source/build identity in-memory."""
    try:
        # 1. Resolve project root
        if project_root is not None:
            resolved_root = Path(project_root).resolve()
        else:
            resolved_root = Path(__file__).resolve().parent.parent

        # 2. Resolve version
        version = distribution_version
        if not version:
            try:
                version = importlib.metadata.version(distribution_name)
            except importlib.metadata.PackageNotFoundError:
                pass
        
        if not version:
            return _unresolved(
                "project_version_unavailable",
                f"Could not resolve version for distribution '{distribution_name}'."
            )

        # 3. Check if Git is available
        git_avail = _check_git_available()
        
        is_git_repo = False
        if git_avail:
            # Check if project root is inside a git repo
            toplevel = _run_git(["git", "rev-parse", "--show-toplevel"], resolved_root)
            if toplevel is not None:
                is_git_repo = True
                repo_root = Path(toplevel.strip()).resolve()

        if is_git_repo:
            # 4. Resolve Git revision
            revision_out = _run_git(["git", "rev-parse", "HEAD"], resolved_root)
            if revision_out is None:
                return _unresolved(
                    "git_revision_unavailable",
                    "Failed to get revision HEAD via git."
                )
            source_revision = revision_out.strip()

            # 5. Check dirty state
            status_out = _run_git(["git", "status", "--porcelain"], resolved_root)
            if status_out is None:
                return _unresolved(
                    "git_tree_state_unavailable",
                    "Failed to run git status."
                )

            dirty_files = []
            for line in status_out.splitlines():
                if not line.strip():
                    continue
                # Format of line is e.g. " M filename" or "?? filename"
                rel_to_repo = line[3:].strip().strip('"')
                abs_path = (repo_root / rel_to_repo).resolve()
                try:
                    rel_to_project = abs_path.relative_to(resolved_root).as_posix()
                except ValueError:
                    continue
                if not is_excluded_path(rel_to_project):
                    dirty_files.append(rel_to_project)

            if len(dirty_files) == 0:
                # Clean Git repository
                identity = build_application_build_identity(
                    distribution_name=distribution_name,
                    distribution_version=version,
                    source_revision_kind="git",
                    source_revision=source_revision,
                    source_tree_state="clean",
                )
                return ApplicationBuildIdentityProviderResult(
                    status="resolved",
                    build_identity=identity,
                    blocking_issues=(),
                    provider_version=APPLICATION_BUILD_IDENTITY_PROVIDER_VERSION,
                )

            # Dirty repository
            if not allow_dirty_content_bound:
                return _unresolved(
                    "git_tree_state_unavailable",
                    "Dirty worktree is not allowed by policy."
                )

            # Compute source tree digest
            tracked_out = _run_git(["git", "ls-files", "-z"], resolved_root)
            untracked_out = _run_git(["git", "ls-files", "--others", "--exclude-standard", "-z"], resolved_root)

            if tracked_out is None or untracked_out is None:
                return _unresolved(
                    "dirty_tree_digest_unavailable",
                    "Failed to read file list from git."
                )

            all_paths = set()
            for out in (tracked_out, untracked_out):
                for p in out.split("\x00"):
                    if not p.strip():
                        continue
                    abs_path = (repo_root / p).resolve()
                    try:
                        rel_to_project = abs_path.relative_to(resolved_root).as_posix()
                    except ValueError:
                        continue
                    if not is_excluded_path(rel_to_project):
                        all_paths.add(rel_to_project)

            sorted_paths = sorted(all_paths)
            entries = []
            for path_str in sorted_paths:
                entry = _compute_file_entry(resolved_root, path_str)
                if entry is None:
                    return _unresolved(
                        "dirty_tree_digest_unavailable",
                        f"Failed to read file content for digest: {path_str}"
                    )
                entries.append(entry)

            # Canonical serialization
            domain = b"photometry-source-tree-digest:v1"
            payload_bytes = encode_canonical_value(entries)
            source_tree_digest = hashlib.sha256(domain + b"\x00" + payload_bytes).hexdigest()

            identity = build_application_build_identity(
                distribution_name=distribution_name,
                distribution_version=version,
                source_revision_kind="git",
                source_revision=source_revision,
                source_tree_state="dirty_content_bound",
                source_tree_digest=source_tree_digest,
            )
            return ApplicationBuildIdentityProviderResult(
                status="resolved",
                build_identity=identity,
                blocking_issues=(),
                provider_version=APPLICATION_BUILD_IDENTITY_PROVIDER_VERSION,
            )

        # 6. Non-Git fallback
        if packaged_artifact_digest is not None:
            if not _is_sha256(packaged_artifact_digest):
                return _unresolved(
                    "packaged_artifact_identity_unavailable",
                    "Provided packaged_artifact_digest is not a valid SHA-256 digest."
                )
            
            # Use distribution version as revision
            identity = build_application_build_identity(
                distribution_name=distribution_name,
                distribution_version=version,
                source_revision_kind="packaged_artifact",
                source_revision=version,
                source_tree_state="unavailable",
                build_artifact_digest=packaged_artifact_digest,
            )
            return ApplicationBuildIdentityProviderResult(
                status="resolved",
                build_identity=identity,
                blocking_issues=(),
                provider_version=APPLICATION_BUILD_IDENTITY_PROVIDER_VERSION,
            )

        # Packaged fallback not available or not requested
        refusal_cat = "packaged_artifact_identity_unavailable" if not git_avail else "repository_root_not_found"
        return _unresolved(
            refusal_cat,
            "Not a Git repository and no valid packaged_artifact_digest provided."
        )

    except Exception:
        return _unresolved(
            "build_identity_internal_error",
            "Application build identity could not be resolved.",
            detail_code="build_identity_provider_exception",
        )
