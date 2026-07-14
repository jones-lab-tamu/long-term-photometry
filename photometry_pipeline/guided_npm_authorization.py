"""B2-C3 live authorization for an immutable intermittent-NPM authority.

This is the first NPM boundary allowed to inspect the live source filesystem.
It verifies source membership and bytes only.  It does not parse CSV content,
reinterpret scientific facts, materialize startup, or authorize worker launch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
import hashlib
import ntpath
import os
import posixpath
import stat
from typing import Any

from photometry_pipeline.guided_identity import (
    GuidedIdentityError,
    canonicalize_absolute_path,
    encode_canonical_value,
)
from photometry_pipeline.guided_npm_execution_authority import (
    GUIDED_NPM_AUTHORIZATION_STATUS_NOT_AUTHORIZED,
    GUIDED_NPM_EXECUTION_AUTHORITY_CONTRACT_VERSION,
    GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_NAME,
    GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_VERSION,
    GUIDED_NPM_STARTUP_STATUS_NOT_MATERIALIZED,
    GuidedNpmExecutionAuthority,
    verify_guided_npm_execution_authority,
)
from photometry_pipeline.guided_production_mapping import (
    ApplicationBuildIdentity,
    build_application_build_identity,
)
from photometry_pipeline.io.npm_source_snapshot import (
    NPM_IGNORED_FILES_POLICY,
    NPM_SOURCE_DISCOVERY_RULE_VERSION,
    GuidedNpmDiscoveredSourceSet,
    NpmSourceSnapshotError,
    discover_npm_source_files,
)


GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_NAME = (
    "guided_npm_execution_authorization"
)
GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_VERSION = "v1"
GUIDED_NPM_EXECUTION_AUTHORIZATION_CONTRACT_VERSION = (
    "guided_npm_execution_authorization.v1"
)
GUIDED_NPM_EXECUTION_AUTHORIZATION_IDENTITY_DOMAIN = (
    "guided_npm_execution_authorization.v1"
)
GUIDED_NPM_AUTHORIZATION_STATUS_AUTHORIZED_FOR_STARTUP_PREPARATION = (
    "authorized_for_startup_preparation"
)
GUIDED_NPM_AUTHORIZATION_STARTUP_STATUS_NOT_MATERIALIZED = "not_materialized"
GUIDED_NPM_AUTHORIZATION_HASH_CHUNK_SIZE = 1024 * 1024

_SHA_HEX = frozenset("0123456789abcdef")

GUIDED_NPM_EXECUTION_AUTHORIZATION_REFUSAL_CATEGORIES = (
    "authority_missing_or_invalid",
    "authority_schema_unsupported",
    "authority_identity_mismatch",
    "authority_state_invalid",
    "validation_revision_missing",
    "validation_revision_mismatch",
    "guided_plan_identity_missing",
    "guided_plan_identity_mismatch",
    "application_build_identity_invalid",
    "application_build_identity_mismatch",
    "source_root_missing",
    "source_root_not_directory",
    "source_root_unreadable",
    "source_root_identity_mismatch",
    "source_path_outside_root",
    "source_path_alias_mismatch",
    "source_discovery_failed",
    "source_discovery_contract_mismatch",
    "source_set_missing_file",
    "source_set_extra_file",
    "source_set_duplicate_path",
    "source_set_changed_during_verification",
    "source_file_missing",
    "source_file_not_regular",
    "source_file_unreadable",
    "source_file_size_mismatch",
    "source_file_digest_mismatch",
    "source_file_changed_during_verification",
    "authorization_cancelled",
    "authorization_identity_mismatch",
    "authorization_internal_error",
)
_REFUSAL_CATEGORY_SET = frozenset(
    GUIDED_NPM_EXECUTION_AUTHORIZATION_REFUSAL_CATEGORIES
)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA_HEX


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Canonical mapping keys must be strings.")
        return {key: _canonical(value[key]) for key in value}
    if hasattr(value, "__dataclass_fields__"):
        return {
            item.name: _canonical(getattr(value, item.name)) for item in fields(value)
        }
    raise ValueError(f"Unsupported authorization identity value: {type(value).__name__}.")


def _digest(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("utf-8")
        + b"\x00"
        + encode_canonical_value(_canonical(value))
    ).hexdigest()


@dataclass(frozen=True)
class GuidedNpmExecutionAuthorizationIssue:
    category: str
    section: str
    message: str
    detail_code: str

    def __post_init__(self) -> None:
        if self.category not in _REFUSAL_CATEGORY_SET:
            raise ValueError("Unsupported NPM authorization refusal category.")
        if not all(_text(value) for value in (self.section, self.message, self.detail_code)):
            raise ValueError("Authorization issues require complete text fields.")


@dataclass(frozen=True)
class GuidedNpmExecutionAuthorizationFailure:
    blocking_issues: tuple[GuidedNpmExecutionAuthorizationIssue, ...]
    status: str = "refused"

    def __post_init__(self) -> None:
        if not self.blocking_issues or self.status != "refused":
            raise ValueError("Authorization failure requires blocking issues.")


@dataclass(frozen=True)
class GuidedNpmExecutionAuthorizationCancelled:
    blocking_issues: tuple[GuidedNpmExecutionAuthorizationIssue, ...]
    status: str = "cancelled"

    def __post_init__(self) -> None:
        if (
            len(self.blocking_issues) != 1
            or self.blocking_issues[0].category != "authorization_cancelled"
            or self.status != "cancelled"
        ):
            raise ValueError("Cancelled authorization requires one cancellation issue.")


@dataclass(frozen=True)
class GuidedNpmVerifiedSourceFile:
    chronological_position: int
    canonical_relative_path: str
    authorized_absolute_source_reference: str
    inspected_absolute_path: str
    expected_size_bytes: int
    observed_size_bytes: int
    expected_sha256_content_digest: str
    observed_sha256_content_digest: str
    pre_hash_stat_identity: str
    post_hash_stat_identity: str
    canonical_verified_file_identity: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.chronological_position, bool)
            or not isinstance(self.chronological_position, int)
            or self.chronological_position < 0
        ):
            raise ValueError("chronological_position must be non-negative.")
        if not all(
            _text(value)
            for value in (
                self.canonical_relative_path,
                self.authorized_absolute_source_reference,
                self.inspected_absolute_path,
            )
        ):
            raise ValueError("Verified source paths are required.")
        if (
            isinstance(self.expected_size_bytes, bool)
            or not isinstance(self.expected_size_bytes, int)
            or self.expected_size_bytes < 0
            or isinstance(self.observed_size_bytes, bool)
            or not isinstance(self.observed_size_bytes, int)
            or self.observed_size_bytes < 0
            or self.expected_size_bytes != self.observed_size_bytes
        ):
            raise ValueError("Verified source size is invalid.")
        for name in (
            "expected_sha256_content_digest",
            "observed_sha256_content_digest",
            "pre_hash_stat_identity",
            "post_hash_stat_identity",
            "canonical_verified_file_identity",
        ):
            if not _sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256.")
        if self.expected_sha256_content_digest != self.observed_sha256_content_digest:
            raise ValueError("Verified source digest does not match authority.")
        if self.pre_hash_stat_identity != self.post_hash_stat_identity:
            raise ValueError("Verified source stat identity is unstable.")


@dataclass(frozen=True)
class GuidedNpmVerifiedSourceSnapshot:
    source_root_canonical: str
    source_root_inspected: str
    discovery_contract_version: str
    ordered_files: tuple[GuidedNpmVerifiedSourceFile, ...]
    ordered_file_sequence_identity: str
    source_set_identity: str
    source_content_identity: str
    canonical_verified_snapshot_identity: str

    def __post_init__(self) -> None:
        if not all(
            _text(value)
            for value in (
                self.source_root_canonical,
                self.source_root_inspected,
                self.discovery_contract_version,
            )
        ):
            raise ValueError("Verified source snapshot metadata is incomplete.")
        if self.discovery_contract_version != NPM_SOURCE_DISCOVERY_RULE_VERSION:
            raise ValueError("Unsupported NPM discovery contract version.")
        if not isinstance(self.ordered_files, tuple) or not self.ordered_files:
            raise ValueError("Verified source snapshot requires ordered files.")
        if tuple(item.chronological_position for item in self.ordered_files) != tuple(
            range(len(self.ordered_files))
        ):
            raise ValueError("Verified source order is not contiguous.")
        if len({item.canonical_relative_path for item in self.ordered_files}) != len(
            self.ordered_files
        ):
            raise ValueError("Verified source paths must be unique.")
        for name in (
            "ordered_file_sequence_identity",
            "source_set_identity",
            "source_content_identity",
            "canonical_verified_snapshot_identity",
        ):
            if not _sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256.")


@dataclass(frozen=True)
class GuidedNpmExecutionAuthorization:
    authorization_schema_name: str
    authorization_schema_version: str
    authorization_contract_version: str
    source_authority_identity: str
    source_production_intent_identity: str
    source_request_identity: str
    validation_revision: int
    guided_plan_identity: str
    application_build_identity: ApplicationBuildIdentity
    source_root_canonical: str
    verified_source_snapshot: GuidedNpmVerifiedSourceSnapshot
    execution_mode: str
    selected_canonical_roi_ids: tuple[str, ...]
    correction_authority_identity: str
    feature_authority_identity: str
    output_authority_identity: str
    authorization_status: str
    startup_status: str
    runnable: bool
    canonical_authorization_identity: str

    def __post_init__(self) -> None:
        if self.authorization_schema_name != GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_NAME:
            raise ValueError("Unsupported NPM authorization schema name.")
        if self.authorization_schema_version != GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_VERSION:
            raise ValueError("Unsupported NPM authorization schema version.")
        if self.authorization_contract_version != GUIDED_NPM_EXECUTION_AUTHORIZATION_CONTRACT_VERSION:
            raise ValueError("Unsupported NPM authorization contract version.")
        for name in (
            "source_authority_identity",
            "source_production_intent_identity",
            "source_request_identity",
            "guided_plan_identity",
            "correction_authority_identity",
            "feature_authority_identity",
            "output_authority_identity",
            "canonical_authorization_identity",
        ):
            if not _sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256.")
        if (
            isinstance(self.validation_revision, bool)
            or not isinstance(self.validation_revision, int)
            or self.validation_revision < 0
        ):
            raise ValueError("validation_revision must be non-negative.")
        if not isinstance(self.application_build_identity, ApplicationBuildIdentity):
            raise ValueError("Application build identity is required.")
        if not isinstance(self.verified_source_snapshot, GuidedNpmVerifiedSourceSnapshot):
            raise ValueError("Verified source snapshot is required.")
        if not _text(self.source_root_canonical) or not _text(self.execution_mode):
            raise ValueError("Authorization source and execution mode are required.")
        if not isinstance(self.selected_canonical_roi_ids, tuple) or not self.selected_canonical_roi_ids:
            raise ValueError("Selected ROI authority is required.")
        if self.authorization_status != (
            GUIDED_NPM_AUTHORIZATION_STATUS_AUTHORIZED_FOR_STARTUP_PREPARATION
        ):
            raise ValueError("B2-C3 authorization state is invalid.")
        if self.startup_status != GUIDED_NPM_AUTHORIZATION_STARTUP_STATUS_NOT_MATERIALIZED:
            raise ValueError("B2-C3 cannot claim startup materialization.")
        if self.runnable is not False:
            raise ValueError("B2-C3 authorization cannot be runnable.")


GuidedNpmExecutionAuthorizationResult = (
    GuidedNpmExecutionAuthorization
    | GuidedNpmExecutionAuthorizationFailure
    | GuidedNpmExecutionAuthorizationCancelled
)


class _AuthorizationRefusal(ValueError):
    def __init__(
        self, category: str, section: str, message: str, detail_code: str
    ) -> None:
        self.category = category
        self.section = section
        self.message = message
        self.detail_code = detail_code
        super().__init__(message)


def _refuse(category: str, section: str, message: str, detail_code: str) -> None:
    if category not in _REFUSAL_CATEGORY_SET:
        category = "authorization_internal_error"
    raise _AuthorizationRefusal(category, section, message, detail_code)


def _issue(exc: _AuthorizationRefusal) -> GuidedNpmExecutionAuthorizationIssue:
    return GuidedNpmExecutionAuthorizationIssue(
        exc.category, exc.section, exc.message, exc.detail_code
    )


def _failure(exc: _AuthorizationRefusal) -> GuidedNpmExecutionAuthorizationFailure:
    return GuidedNpmExecutionAuthorizationFailure((_issue(exc),))


def _cancelled(
    exc: _AuthorizationRefusal,
) -> GuidedNpmExecutionAuthorizationCancelled:
    return GuidedNpmExecutionAuthorizationCancelled((_issue(exc),))


def compute_guided_npm_verified_source_file_identity(
    verified: GuidedNpmVerifiedSourceFile,
) -> str:
    if not isinstance(verified, GuidedNpmVerifiedSourceFile):
        raise ValueError("verified must be a GuidedNpmVerifiedSourceFile.")
    return _digest(
        "guided_npm_verified_source_file.v1",
        {
            item.name: getattr(verified, item.name)
            for item in fields(verified)
            if item.name != "canonical_verified_file_identity"
        },
    )


def compute_guided_npm_verified_source_sequence_identity(
    files: tuple[GuidedNpmVerifiedSourceFile, ...],
) -> str:
    if not isinstance(files, tuple):
        raise ValueError("files must be a tuple.")
    return _digest(
        "guided_npm_verified_source_sequence.v1",
        tuple(
            {
                "chronological_position": item.chronological_position,
                "canonical_relative_path": item.canonical_relative_path,
                "canonical_verified_file_identity": item.canonical_verified_file_identity,
            }
            for item in files
        ),
    )


def compute_guided_npm_verified_source_set_identity(
    files: tuple[GuidedNpmVerifiedSourceFile, ...],
) -> str:
    return _digest(
        "guided_npm_verified_source_set.v1",
        {
            "discovery_contract_version": NPM_SOURCE_DISCOVERY_RULE_VERSION,
            "canonical_relative_paths": sorted(
                item.canonical_relative_path for item in files
            ),
        },
    )


def compute_guided_npm_verified_source_content_identity(
    files: tuple[GuidedNpmVerifiedSourceFile, ...],
) -> str:
    return _digest(
        "guided_npm_verified_source_content.v1",
        tuple(
            {
                "canonical_relative_path": item.canonical_relative_path,
                "observed_size_bytes": item.observed_size_bytes,
                "observed_sha256_content_digest": item.observed_sha256_content_digest,
            }
            for item in sorted(files, key=lambda item: item.canonical_relative_path)
        ),
    )


def compute_guided_npm_verified_source_snapshot_identity(
    snapshot: GuidedNpmVerifiedSourceSnapshot,
) -> str:
    if not isinstance(snapshot, GuidedNpmVerifiedSourceSnapshot):
        raise ValueError("snapshot must be a GuidedNpmVerifiedSourceSnapshot.")
    return _digest(
        "guided_npm_verified_source_snapshot.v1",
        {
            item.name: getattr(snapshot, item.name)
            for item in fields(snapshot)
            if item.name != "canonical_verified_snapshot_identity"
        },
    )


def compute_guided_npm_execution_authorization_identity(
    authorization: GuidedNpmExecutionAuthorization,
) -> str:
    if not isinstance(authorization, GuidedNpmExecutionAuthorization):
        raise ValueError("authorization must be a GuidedNpmExecutionAuthorization.")
    return _digest(
        GUIDED_NPM_EXECUTION_AUTHORIZATION_IDENTITY_DOMAIN,
        {
            item.name: getattr(authorization, item.name)
            for item in fields(authorization)
            if item.name != "canonical_authorization_identity"
        },
    )


def _check_cancellation(cancellation_check: Callable[[], bool] | None) -> None:
    if cancellation_check is not None and cancellation_check():
        _refuse(
            "authorization_cancelled",
            "authorization",
            "NPM authorization was cancelled.",
            "authorization_cancelled",
        )


def _recompute_build_identity(identity: Any) -> ApplicationBuildIdentity:
    if type(identity) is not ApplicationBuildIdentity:
        _refuse(
            "application_build_identity_invalid",
            "guided_context",
            "The current application build identity is invalid.",
            "application_build_identity_type_invalid",
        )
    try:
        expected = build_application_build_identity(
            distribution_name=identity.distribution_name,
            distribution_version=identity.distribution_version,
            source_revision_kind=identity.source_revision_kind,
            source_revision=identity.source_revision,
            source_tree_state=identity.source_tree_state,
            source_tree_digest=identity.source_tree_digest,
            build_artifact_digest=identity.build_artifact_digest,
            identity_provider_version=identity.identity_provider_version,
        )
    except (TypeError, ValueError) as exc:
        _refuse(
            "application_build_identity_invalid",
            "guided_context",
            "The current application build identity is malformed.",
            "application_build_identity_recompute_failed",
        )
    if expected != identity:
        _refuse(
            "application_build_identity_invalid",
            "guided_context",
            "The current application build identity does not match its fields.",
            "application_build_identity_canonical_mismatch",
        )
    return expected


def _verify_authority_and_context(
    authority: Any,
    expected_validation_revision: Any,
    expected_plan_identity: Any,
    current_application_build_identity: Any,
) -> None:
    if type(authority) is not GuidedNpmExecutionAuthority:
        _refuse(
            "authority_missing_or_invalid",
            "authority",
            "A valid B2-C2 NPM execution authority is required.",
            "authority_type_invalid",
        )
    if (
        authority.authority_schema_name != GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_NAME
        or authority.authority_schema_version
        != GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_VERSION
        or authority.authority_contract_version
        != GUIDED_NPM_EXECUTION_AUTHORITY_CONTRACT_VERSION
    ):
        _refuse(
            "authority_schema_unsupported",
            "authority",
            "The NPM execution authority schema is unsupported.",
            "authority_schema_unsupported",
        )
    if (
        authority.authorization_status
        != GUIDED_NPM_AUTHORIZATION_STATUS_NOT_AUTHORIZED
        or authority.startup_status != GUIDED_NPM_STARTUP_STATUS_NOT_MATERIALIZED
        or authority.runnable is not False
    ):
        _refuse(
            "authority_state_invalid",
            "authority",
            "The B2-C2 authority has an invalid authorization state.",
            "authority_state_invalid",
        )
    try:
        verify_guided_npm_execution_authority(authority)
    except (TypeError, ValueError) as exc:
        _refuse(
            "authority_identity_mismatch",
            "authority",
            "The B2-C2 authority identity chain is invalid.",
            str(exc) or "authority_identity_chain_invalid",
        )

    if expected_validation_revision is None:
        _refuse(
            "validation_revision_missing",
            "guided_context",
            "The current validation revision is required.",
            "expected_validation_revision_missing",
        )
    if (
        isinstance(expected_validation_revision, bool)
        or not isinstance(expected_validation_revision, int)
        or expected_validation_revision < 0
    ):
        _refuse(
            "validation_revision_missing",
            "guided_context",
            "The current validation revision is invalid.",
            "expected_validation_revision_invalid",
        )
    if authority.validation_revision != expected_validation_revision:
        _refuse(
            "validation_revision_mismatch",
            "guided_context",
            "The current validation revision does not match the authority.",
            "validation_revision_mismatch",
        )

    if not _sha256(expected_plan_identity):
        _refuse(
            "guided_plan_identity_missing",
            "guided_context",
            "The current Guided plan identity is required.",
            "expected_plan_identity_invalid",
        )
    if expected_plan_identity == authority.source_request_identity:
        _refuse(
            "guided_plan_identity_mismatch",
            "guided_context",
            "The request identity cannot substitute for the Guided plan identity.",
            "request_identity_substituted_for_plan_identity",
        )
    if authority.guided_plan_identity != expected_plan_identity:
        _refuse(
            "guided_plan_identity_mismatch",
            "guided_context",
            "The current Guided plan identity does not match the authority.",
            "guided_plan_identity_mismatch",
        )

    current_build = _recompute_build_identity(current_application_build_identity)
    if (
        current_build.canonical_identity
        != authority.application_build_identity.canonical_identity
    ):
        _refuse(
            "application_build_identity_mismatch",
            "guided_context",
            "The current application build does not match the authority.",
            "application_build_identity_mismatch",
        )


def _windows_path(path: str) -> bool:
    return bool(ntpath.splitdrive(path)[0]) or "\\" in path


def _path_key(path: str, *, windows: bool) -> str:
    if windows:
        return ntpath.normcase(ntpath.normpath(path))
    return posixpath.normpath(path)


def _join_root(root: str, relative: str, *, windows: bool) -> str:
    parts = relative.split("/")
    module = ntpath if windows else posixpath
    return module.join(root, *parts)


def _derive_authorized_source_root(authority: GuidedNpmExecutionAuthority) -> str:
    roots: list[tuple[str, bool]] = []
    for session in authority.sessions:
        relative = session.canonical_relative_path
        absolute = session.authorized_absolute_source_reference
        if (
            not _text(relative)
            or relative.startswith(("/", "\\"))
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
        ):
            _refuse(
                "source_path_outside_root",
                "source",
                "An authorized source path is not a safe canonical relative path.",
                "authorized_relative_path_invalid",
            )
        windows = _windows_path(absolute)
        module = ntpath if windows else posixpath
        root = absolute
        for _part in relative.split("/"):
            root = module.dirname(root)
        if not root or _path_key(_join_root(root, relative, windows=windows), windows=windows) != _path_key(
            absolute, windows=windows
        ):
            _refuse(
                "source_root_identity_mismatch",
                "source",
                "The authorized absolute source reference is not rooted by its relative path.",
                "authorized_source_root_derivation_failed",
            )
        roots.append((root, windows))
    first_root, first_windows = roots[0]
    if any(
        windows != first_windows
        or _path_key(root, windows=windows)
        != _path_key(first_root, windows=first_windows)
        for root, windows in roots[1:]
    ):
        _refuse(
            "source_root_identity_mismatch",
            "source",
            "Authorized NPM sessions do not share one exact source root.",
            "authorized_source_roots_differ",
        )
    try:
        return canonicalize_absolute_path(first_root).canonical_path
    except GuidedIdentityError:
        _refuse(
            "source_root_identity_mismatch",
            "source",
            "The authorized NPM source root is not canonical.",
            "authorized_source_root_invalid",
        )


def _canonical_current_path(path: str) -> str:
    try:
        return canonicalize_absolute_path(os.path.abspath(os.path.normpath(path))).canonical_path
    except (GuidedIdentityError, OSError, ValueError):
        _refuse(
            "source_path_outside_root",
            "source",
            "A current NPM source path cannot be canonicalized.",
            "current_source_path_invalid",
        )


def _verify_source_root(source_root: str) -> str:
    if not os.path.exists(source_root):
        _refuse(
            "source_root_missing",
            "source",
            "The authorized NPM source root is missing.",
            "source_root_missing",
        )
    if not os.path.isdir(source_root):
        _refuse(
            "source_root_not_directory",
            "source",
            "The authorized NPM source root is not a directory.",
            "source_root_not_directory",
        )
    if os.path.islink(source_root):
        _refuse(
            "source_path_alias_mismatch",
            "source",
            "The authorized NPM source root cannot be a path alias.",
            "source_root_symlink_prohibited",
        )
    if not os.access(source_root, os.R_OK):
        _refuse(
            "source_root_unreadable",
            "source",
            "The authorized NPM source root is unreadable.",
            "source_root_unreadable",
        )
    lexical = _canonical_current_path(source_root)
    inspected = _canonical_current_path(os.path.realpath(source_root))
    if lexical != source_root:
        _refuse(
            "source_root_identity_mismatch",
            "source",
            "The live source root does not match the authority root.",
            "source_root_canonical_mismatch",
        )
    if inspected != lexical:
        _refuse(
            "source_path_alias_mismatch",
            "source",
            "The authorized source root resolves through a path alias.",
            "source_root_resolved_alias_mismatch",
        )
    return inspected


def _precheck_authorized_path_types(
    authority: GuidedNpmExecutionAuthority,
) -> None:
    """Classify aliases and non-files before membership-only discovery."""
    for session in authority.sessions:
        path = session.authorized_absolute_source_reference
        if not os.path.lexists(path):
            continue
        if os.path.islink(path):
            _refuse(
                "source_path_alias_mismatch",
                "source",
                "An authorized source path was replaced by a path alias.",
                "authorized_source_symlink_substitution",
            )
        try:
            current_stat = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            _refuse(
                "source_file_unreadable",
                "source",
                "An authorized source path cannot be inspected.",
                type(exc).__name__,
            )
        if not stat.S_ISREG(current_stat.st_mode):
            _refuse(
                "source_file_not_regular",
                "source",
                "An authorized source path is no longer a regular file.",
                "authorized_source_not_regular",
            )


def _path_is_within(path: str, root: str) -> bool:
    try:
        return os.path.normcase(os.path.commonpath([root, path])) == os.path.normcase(
            root
        )
    except (OSError, ValueError):
        return False


def _enumerate_source_set(source_root: str) -> GuidedNpmDiscoveredSourceSet:
    try:
        discovered = discover_npm_source_files(source_root)
    except NpmSourceSnapshotError as exc:
        if exc.category == "no_npm_csv_files":
            _refuse(
                "source_set_missing_file",
                "source",
                "The current NPM source set is missing authorized files.",
                "source_set_empty",
            )
        _refuse(
            "source_discovery_failed",
            "source",
            "The current NPM source set could not be enumerated.",
            exc.category,
        )
    except OSError as exc:
        _refuse(
            "source_discovery_failed",
            "source",
            "The current NPM source set could not be enumerated.",
            type(exc).__name__,
        )
    if (
        discovered.discovery_rule_version != NPM_SOURCE_DISCOVERY_RULE_VERSION
        or discovered.ignored_files_policy != NPM_IGNORED_FILES_POLICY
    ):
        _refuse(
            "source_discovery_contract_mismatch",
            "source",
            "The NPM discovery contract does not match B2-B.",
            "npm_discovery_contract_mismatch",
        )
    if discovered.source_root_canonical != source_root:
        _refuse(
            "source_root_identity_mismatch",
            "source",
            "The discovered NPM source root differs from the authority root.",
            "discovered_source_root_mismatch",
        )
    relative_paths = [item.canonical_relative_path for item in discovered.files]
    if len(set(relative_paths)) != len(relative_paths):
        _refuse(
            "source_set_duplicate_path",
            "source",
            "The current NPM source set contains a duplicate canonical path.",
            "source_set_duplicate_canonical_path",
        )
    real_root = os.path.realpath(source_root)
    for item in discovered.files:
        expected = _join_root(
            source_root,
            item.canonical_relative_path,
            windows=_windows_path(source_root),
        )
        if _canonical_current_path(item.absolute_path) != _canonical_current_path(expected):
            _refuse(
                "source_path_alias_mismatch",
                "source",
                "A discovered NPM path does not match its canonical root-relative path.",
                "discovered_absolute_path_mismatch",
            )
        if not _path_is_within(os.path.realpath(item.absolute_path), real_root):
            _refuse(
                "source_path_outside_root",
                "source",
                "A discovered NPM source path escapes the authorized root.",
                "discovered_path_outside_root",
            )
    return discovered


def _compare_initial_source_set(
    authority: GuidedNpmExecutionAuthority,
    source_root: str,
    discovered: GuidedNpmDiscoveredSourceSet,
) -> dict[str, str]:
    authorized = {item.canonical_relative_path for item in authority.sessions}
    current = {item.canonical_relative_path for item in discovered.files}
    missing = authorized - current
    extra = current - authorized
    if missing:
        missing_path = sorted(missing)[0]
        expected = _join_root(
            source_root, missing_path, windows=_windows_path(source_root)
        )
        if os.path.lexists(expected):
            if os.path.islink(expected):
                _refuse(
                    "source_path_alias_mismatch",
                    "source",
                    "An authorized source path was replaced by a path alias.",
                    "authorized_source_symlink_substitution",
                )
            _refuse(
                "source_file_not_regular",
                "source",
                "An authorized source path is no longer a regular file.",
                "authorized_source_not_regular",
            )
        _refuse(
            "source_set_missing_file",
            "source",
            "The current NPM source set is missing an authorized file.",
            "authorized_source_missing_from_set",
        )
    if extra:
        _refuse(
            "source_set_extra_file",
            "source",
            "A new matching NPM source file entered the source set.",
            "extra_npm_source_file",
        )
    return {item.canonical_relative_path: item.absolute_path for item in discovered.files}


def _stat_facts(value: os.stat_result) -> dict[str, int | None]:
    return {
        "size_bytes": int(value.st_size),
        "mtime_ns": int(
            getattr(value, "st_mtime_ns", int(value.st_mtime * 1_000_000_000))
        ),
        "device": int(value.st_dev) if hasattr(value, "st_dev") else None,
        "inode": int(value.st_ino) if hasattr(value, "st_ino") else None,
        "file_type": int(stat.S_IFMT(value.st_mode)),
    }


def _stat_identity(value: os.stat_result) -> str:
    return _digest("guided_npm_verified_source_stat.v1", _stat_facts(value))


def _verify_one_file(
    session: Any,
    discovered_path: str,
    source_root: str,
    source_root_inspected: str,
    cancellation_check: Callable[[], bool] | None,
) -> GuidedNpmVerifiedSourceFile:
    authorized_path = session.authorized_absolute_source_reference
    expected_path = _join_root(
        source_root,
        session.canonical_relative_path,
        windows=_windows_path(source_root),
    )
    if _canonical_current_path(authorized_path) != _canonical_current_path(expected_path):
        _refuse(
            "source_root_identity_mismatch",
            "source",
            "An authorized absolute source path no longer matches its root-relative path.",
            "authorized_absolute_path_mismatch",
        )
    if _canonical_current_path(discovered_path) != _canonical_current_path(authorized_path):
        _refuse(
            "source_path_alias_mismatch",
            "source",
            "The discovered source path is not the exact authorized path.",
            "discovered_authorized_path_mismatch",
        )
    if not os.path.lexists(authorized_path):
        _refuse(
            "source_file_missing",
            "source",
            "An authorized NPM source file is missing.",
            "authorized_source_file_missing",
        )
    if os.path.islink(authorized_path):
        _refuse(
            "source_path_alias_mismatch",
            "source",
            "An authorized NPM source file cannot be a path alias.",
            "authorized_source_file_symlink_prohibited",
        )
    real_path = os.path.realpath(authorized_path)
    if not _path_is_within(real_path, os.path.realpath(source_root)):
        _refuse(
            "source_path_outside_root",
            "source",
            "An authorized NPM source file resolves outside its source root.",
            "authorized_source_file_outside_root",
        )
    inspected_path = _canonical_current_path(real_path)
    if inspected_path != _canonical_current_path(authorized_path):
        _refuse(
            "source_path_alias_mismatch",
            "source",
            "An authorized NPM source path resolves to another path.",
            "authorized_source_file_resolved_alias_mismatch",
        )
    if source_root_inspected != _canonical_current_path(os.path.realpath(source_root)):
        _refuse(
            "source_root_identity_mismatch",
            "source",
            "The inspected source root changed during authorization.",
            "source_root_changed_during_verification",
        )
    try:
        pre_stat = os.stat(authorized_path, follow_symlinks=False)
    except FileNotFoundError:
        _refuse(
            "source_file_missing",
            "source",
            "An authorized NPM source file is missing.",
            "authorized_source_file_missing",
        )
    except OSError as exc:
        _refuse(
            "source_file_unreadable",
            "source",
            "An authorized NPM source file cannot be inspected.",
            type(exc).__name__,
        )
    if not stat.S_ISREG(pre_stat.st_mode):
        _refuse(
            "source_file_not_regular",
            "source",
            "An authorized NPM source path is not a regular file.",
            "authorized_source_file_not_regular",
        )
    if int(pre_stat.st_size) != session.size_bytes:
        _refuse(
            "source_file_size_mismatch",
            "source",
            "An authorized NPM source file size changed.",
            "source_file_size_mismatch",
        )

    _check_cancellation(cancellation_check)
    digest = hashlib.sha256()
    try:
        with open(authorized_path, "rb") as handle:
            opened_stat = os.fstat(handle.fileno())
            if _stat_facts(opened_stat) != _stat_facts(pre_stat):
                _refuse(
                    "source_file_changed_during_verification",
                    "source",
                    "An authorized source file changed before hashing began.",
                    "source_file_changed_between_stat_and_open",
                )
            while True:
                _check_cancellation(cancellation_check)
                chunk = handle.read(GUIDED_NPM_AUTHORIZATION_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
            _check_cancellation(cancellation_check)
            handle_post_stat = os.fstat(handle.fileno())
    except _AuthorizationRefusal:
        raise
    except FileNotFoundError:
        _refuse(
            "source_file_missing",
            "source",
            "An authorized NPM source file disappeared before hashing.",
            "source_file_missing_during_hash",
        )
    except OSError as exc:
        _refuse(
            "source_file_unreadable",
            "source",
            "An authorized NPM source file cannot be read for hashing.",
            type(exc).__name__,
        )
    try:
        post_stat = os.stat(authorized_path, follow_symlinks=False)
    except OSError:
        _refuse(
            "source_file_changed_during_verification",
            "source",
            "An authorized NPM source file changed after hashing.",
            "source_file_missing_after_hash",
        )
    stable = _stat_facts(pre_stat)
    if (
        _stat_facts(opened_stat) != stable
        or _stat_facts(handle_post_stat) != stable
        or _stat_facts(post_stat) != stable
        or int(post_stat.st_size) != session.size_bytes
    ):
        _refuse(
            "source_file_changed_during_verification",
            "source",
            "An authorized NPM source file changed while it was hashed.",
            "source_file_stat_identity_changed",
        )
    observed_digest = digest.hexdigest()
    if observed_digest != session.sha256_content_digest:
        _refuse(
            "source_file_digest_mismatch",
            "source",
            "An authorized NPM source file content digest changed.",
            "source_file_digest_mismatch",
        )
    pre_identity = _stat_identity(pre_stat)
    post_identity = _stat_identity(post_stat)
    verified = GuidedNpmVerifiedSourceFile(
        chronological_position=session.chronological_position,
        canonical_relative_path=session.canonical_relative_path,
        authorized_absolute_source_reference=authorized_path,
        inspected_absolute_path=inspected_path,
        expected_size_bytes=session.size_bytes,
        observed_size_bytes=int(post_stat.st_size),
        expected_sha256_content_digest=session.sha256_content_digest,
        observed_sha256_content_digest=observed_digest,
        pre_hash_stat_identity=pre_identity,
        post_hash_stat_identity=post_identity,
        canonical_verified_file_identity="0" * 64,
    )
    return replace(
        verified,
        canonical_verified_file_identity=(
            compute_guided_npm_verified_source_file_identity(verified)
        ),
    )


def _build_verified_snapshot(
    source_root: str,
    source_root_inspected: str,
    files: tuple[GuidedNpmVerifiedSourceFile, ...],
) -> GuidedNpmVerifiedSourceSnapshot:
    snapshot = GuidedNpmVerifiedSourceSnapshot(
        source_root_canonical=source_root,
        source_root_inspected=source_root_inspected,
        discovery_contract_version=NPM_SOURCE_DISCOVERY_RULE_VERSION,
        ordered_files=files,
        ordered_file_sequence_identity=(
            compute_guided_npm_verified_source_sequence_identity(files)
        ),
        source_set_identity=compute_guided_npm_verified_source_set_identity(files),
        source_content_identity=(
            compute_guided_npm_verified_source_content_identity(files)
        ),
        canonical_verified_snapshot_identity="0" * 64,
    )
    return replace(
        snapshot,
        canonical_verified_snapshot_identity=(
            compute_guided_npm_verified_source_snapshot_identity(snapshot)
        ),
    )


def _reverify_final_file_stats(
    authority: GuidedNpmExecutionAuthority,
    verified_files: tuple[GuidedNpmVerifiedSourceFile, ...],
    cancellation_check: Callable[[], bool] | None,
) -> None:
    verified_by_path = {
        item.canonical_relative_path: item for item in verified_files
    }
    for session in authority.sessions:
        _check_cancellation(cancellation_check)
        try:
            current = os.stat(
                session.authorized_absolute_source_reference,
                follow_symlinks=False,
            )
        except OSError:
            _refuse(
                "source_file_changed_during_verification",
                "source",
                "An authorized NPM source file changed before authorization completed.",
                "source_file_missing_during_final_stat_check",
            )
        verified = verified_by_path[session.canonical_relative_path]
        if (
            not stat.S_ISREG(current.st_mode)
            or _stat_identity(current) != verified.post_hash_stat_identity
        ):
            _refuse(
                "source_file_changed_during_verification",
                "source",
                "An authorized NPM source file changed before authorization completed.",
                "source_file_changed_after_hash",
            )
def _authorize(
    authority: GuidedNpmExecutionAuthority,
    *,
    expected_validation_revision: int,
    expected_plan_identity: str,
    current_application_build_identity: ApplicationBuildIdentity,
    cancellation_check: Callable[[], bool] | None,
) -> GuidedNpmExecutionAuthorization:
    _verify_authority_and_context(
        authority,
        expected_validation_revision,
        expected_plan_identity,
        current_application_build_identity,
    )
    source_root = _derive_authorized_source_root(authority)
    source_root_inspected = _verify_source_root(source_root)
    _check_cancellation(cancellation_check)
    _precheck_authorized_path_types(authority)

    _check_cancellation(cancellation_check)
    before = _enumerate_source_set(source_root)
    discovered_by_path = _compare_initial_source_set(authority, source_root, before)

    verified_files: list[GuidedNpmVerifiedSourceFile] = []
    for session in authority.sessions:
        _check_cancellation(cancellation_check)
        verified_files.append(
            _verify_one_file(
                session,
                discovered_by_path[session.canonical_relative_path],
                source_root,
                source_root_inspected,
                cancellation_check,
            )
        )

    _check_cancellation(cancellation_check)
    after = _enumerate_source_set(source_root)
    before_paths = {item.canonical_relative_path for item in before.files}
    after_paths = {item.canonical_relative_path for item in after.files}
    authorized_paths = {item.canonical_relative_path for item in authority.sessions}
    if before_paths != after_paths or after_paths != authorized_paths:
        _refuse(
            "source_set_changed_during_verification",
            "source",
            "The NPM source set changed during authorization.",
            "source_set_changed_between_enumerations",
        )

    verified_tuple = tuple(verified_files)
    _reverify_final_file_stats(authority, verified_tuple, cancellation_check)
    _check_cancellation(cancellation_check)
    snapshot = _build_verified_snapshot(
        source_root, source_root_inspected, verified_tuple
    )
    authorization = GuidedNpmExecutionAuthorization(
        authorization_schema_name=GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_NAME,
        authorization_schema_version=GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
        authorization_contract_version=(
            GUIDED_NPM_EXECUTION_AUTHORIZATION_CONTRACT_VERSION
        ),
        source_authority_identity=authority.canonical_authority_identity,
        source_production_intent_identity=authority.source_production_intent_identity,
        source_request_identity=authority.source_request_identity,
        validation_revision=authority.validation_revision,
        guided_plan_identity=authority.guided_plan_identity,
        application_build_identity=current_application_build_identity,
        source_root_canonical=source_root,
        verified_source_snapshot=snapshot,
        execution_mode=authority.execution_mode,
        selected_canonical_roi_ids=(
            authority.roi_authority.selected_canonical_roi_ids
        ),
        correction_authority_identity=(
            authority.correction_authority.canonical_correction_authority_identity
        ),
        feature_authority_identity=(
            authority.feature_authority.canonical_feature_authority_identity
        ),
        output_authority_identity=(
            authority.output_authority.canonical_output_authority_identity
        ),
        authorization_status=(
            GUIDED_NPM_AUTHORIZATION_STATUS_AUTHORIZED_FOR_STARTUP_PREPARATION
        ),
        startup_status=GUIDED_NPM_AUTHORIZATION_STARTUP_STATUS_NOT_MATERIALIZED,
        runnable=False,
        canonical_authorization_identity="0" * 64,
    )
    return replace(
        authorization,
        canonical_authorization_identity=(
            compute_guided_npm_execution_authorization_identity(authorization)
        ),
    )


def authorize_guided_npm_execution_authority(
    authority: GuidedNpmExecutionAuthority,
    *,
    expected_validation_revision: int,
    expected_plan_identity: str,
    current_application_build_identity: ApplicationBuildIdentity,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedNpmExecutionAuthorizationResult:
    """Authorize an immutable B2-C2 authority against current bytes and context."""
    try:
        return _authorize(
            authority,
            expected_validation_revision=expected_validation_revision,
            expected_plan_identity=expected_plan_identity,
            current_application_build_identity=current_application_build_identity,
            cancellation_check=cancellation_check,
        )
    except _AuthorizationRefusal as exc:
        if exc.category == "authorization_cancelled":
            return _cancelled(exc)
        return _failure(exc)
    except Exception as exc:
        return _failure(
            _AuthorizationRefusal(
                "authorization_internal_error",
                "authorization",
                "NPM authorization failed.",
                type(exc).__name__,
            )
        )
