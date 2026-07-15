"""B2-C5 durable persistence for the canonical NPM startup payload only."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_npm_startup_payload import (
    GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_NAME,
    GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_VERSION,
    GuidedNpmStartupPayload,
    deserialize_guided_npm_startup_payload,
    serialize_guided_npm_startup_payload,
    verify_guided_npm_startup_payload,
)
from photometry_pipeline.guided_production_mapping import (
    ApplicationBuildIdentity,
    build_application_build_identity,
)


GUIDED_NPM_STARTUP_ARTIFACT_FILENAME = "guided_npm_startup_payload.json"
GUIDED_NPM_STARTUP_PERSISTENCE_SCHEMA_NAME = (
    "guided_npm_startup_persistence_receipt"
)
GUIDED_NPM_STARTUP_PERSISTENCE_SCHEMA_VERSION = "v1"
GUIDED_NPM_STARTUP_PERSISTENCE_CONTRACT_VERSION = (
    "guided_npm_startup_persistence_receipt.v1"
)
GUIDED_NPM_STARTUP_PERSISTENCE_IDENTITY_DOMAIN = (
    "guided_npm_startup_persistence_receipt.v1"
)
GUIDED_NPM_STARTUP_PERSISTENCE_STATUS = "persisted_and_verified"
GUIDED_NPM_STARTUP_PERSISTENCE_CLAIM_STATUS = "not_claimed"
GUIDED_NPM_STARTUP_PERSISTENCE_STARTUP_STATUS = "materialized_not_claimed"

GUIDED_NPM_STARTUP_PERSISTENCE_REFUSAL_CATEGORIES = (
    "startup_payload_missing_or_invalid",
    "startup_payload_schema_unsupported",
    "startup_payload_identity_mismatch",
    "startup_payload_state_invalid",
    "output_base_missing",
    "output_base_not_directory",
    "output_base_unreadable",
    "output_base_unwritable",
    "output_path_unsafe",
    "run_directory_allocation_failed",
    "run_directory_conflict",
    "startup_artifact_conflict",
    "startup_artifact_path_invalid",
    "startup_artifact_write_failed",
    "startup_artifact_flush_failed",
    "startup_artifact_publish_failed",
    "startup_artifact_readback_failed",
    "startup_artifact_size_mismatch",
    "startup_artifact_digest_mismatch",
    "startup_artifact_bytes_mismatch",
    "startup_artifact_noncanonical",
    "startup_artifact_payload_identity_mismatch",
    "startup_artifact_schema_mismatch",
    "persistence_cancelled",
    "persistence_receipt_identity_mismatch",
    "persistence_internal_error",
)
_CATEGORY_SET = frozenset(GUIDED_NPM_STARTUP_PERSISTENCE_REFUSAL_CATEGORIES)
_HEX = frozenset("0123456789abcdef")
_RUN_ALLOCATION_ATTEMPTS = 8


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


@dataclass(frozen=True)
class GuidedNpmStartupPersistenceIssue:
    category: str
    section: str
    message: str
    detail_code: str

    def __post_init__(self) -> None:
        if self.category not in _CATEGORY_SET or not all(
            _text(value) for value in (self.section, self.message, self.detail_code)
        ):
            raise ValueError("Invalid NPM startup-persistence issue.")


@dataclass(frozen=True)
class GuidedNpmStartupPersistenceFailure:
    blocking_issues: tuple[GuidedNpmStartupPersistenceIssue, ...]
    unverified_artifact_path: str | None = None
    status: str = "refused"

    def __post_init__(self) -> None:
        if len(self.blocking_issues) != 1 or self.status != "refused":
            raise ValueError("Persistence failure requires one blocking issue.")


@dataclass(frozen=True)
class GuidedNpmStartupPersistenceCancelled:
    blocking_issues: tuple[GuidedNpmStartupPersistenceIssue, ...]
    persisted_artifact_path: str | None = None
    status: str = "cancelled"

    def __post_init__(self) -> None:
        if (
            len(self.blocking_issues) != 1
            or self.blocking_issues[0].category != "persistence_cancelled"
            or self.status != "cancelled"
        ):
            raise ValueError("Cancelled persistence requires one cancellation issue.")


@dataclass(frozen=True)
class GuidedNpmStartupPersistenceReceipt:
    persistence_schema_name: str
    persistence_schema_version: str
    persistence_contract_version: str
    source_startup_payload_identity: str
    source_authorization_identity: str
    source_authority_identity: str
    source_request_identity: str
    validation_revision: int
    guided_plan_identity: str
    application_build_identity: ApplicationBuildIdentity
    run_directory_path: str
    startup_artifact_path: str
    startup_artifact_filename: str
    serialized_payload_sha256: str
    persisted_artifact_sha256: str
    persisted_size_bytes: int
    readback_payload_identity: str
    canonical_persistence_receipt_identity: str
    persistence_status: str
    claim_status: str
    startup_status: str
    runnable: bool

    def __post_init__(self) -> None:
        if (
            self.persistence_schema_name
            != GUIDED_NPM_STARTUP_PERSISTENCE_SCHEMA_NAME
            or self.persistence_schema_version
            != GUIDED_NPM_STARTUP_PERSISTENCE_SCHEMA_VERSION
            or self.persistence_contract_version
            != GUIDED_NPM_STARTUP_PERSISTENCE_CONTRACT_VERSION
        ):
            raise ValueError("Unsupported NPM persistence-receipt schema.")
        for name in (
            "source_startup_payload_identity",
            "source_authorization_identity",
            "source_authority_identity",
            "source_request_identity",
            "guided_plan_identity",
            "serialized_payload_sha256",
            "persisted_artifact_sha256",
            "readback_payload_identity",
            "canonical_persistence_receipt_identity",
        ):
            if not _sha(getattr(self, name)):
                raise ValueError(f"{name} is invalid.")
        if (
            isinstance(self.validation_revision, bool)
            or not isinstance(self.validation_revision, int)
            or self.validation_revision < 0
            or isinstance(self.persisted_size_bytes, bool)
            or not isinstance(self.persisted_size_bytes, int)
            or self.persisted_size_bytes <= 0
        ):
            raise ValueError("Persistence receipt numeric field is invalid.")
        if not isinstance(self.application_build_identity, ApplicationBuildIdentity):
            raise ValueError("Persistence receipt build identity is invalid.")
        run_dir = Path(self.run_directory_path)
        artifact = Path(self.startup_artifact_path)
        if (
            not run_dir.is_absolute()
            or not artifact.is_absolute()
            or artifact.parent != run_dir
            or artifact.name != GUIDED_NPM_STARTUP_ARTIFACT_FILENAME
            or self.startup_artifact_filename
            != GUIDED_NPM_STARTUP_ARTIFACT_FILENAME
        ):
            raise ValueError("Persistence receipt paths are invalid.")
        if self.serialized_payload_sha256 != self.persisted_artifact_sha256:
            raise ValueError("Intended and persisted startup digests differ.")
        if self.source_startup_payload_identity != self.readback_payload_identity:
            raise ValueError("Source and readback payload identities differ.")
        if (
            self.persistence_status != GUIDED_NPM_STARTUP_PERSISTENCE_STATUS
            or self.claim_status != GUIDED_NPM_STARTUP_PERSISTENCE_CLAIM_STATUS
            or self.startup_status != GUIDED_NPM_STARTUP_PERSISTENCE_STARTUP_STATUS
            or self.runnable is not False
        ):
            raise ValueError("Persistence receipt state is invalid.")


GuidedNpmStartupPersistenceResult = (
    GuidedNpmStartupPersistenceReceipt
    | GuidedNpmStartupPersistenceFailure
    | GuidedNpmStartupPersistenceCancelled
)


@dataclass(frozen=True)
class GuidedNpmVerifiedStartupArtifact:
    """One exact, canonical, independently verified persisted NPM artifact."""

    startup_artifact_path: str
    startup_artifact_size_bytes: int
    startup_artifact_sha256: str
    payload: GuidedNpmStartupPayload

    def __post_init__(self) -> None:
        path = Path(self.startup_artifact_path)
        if not path.is_absolute() or path.name != GUIDED_NPM_STARTUP_ARTIFACT_FILENAME:
            raise ValueError("startup_artifact_path_invalid")
        if (
            isinstance(self.startup_artifact_size_bytes, bool)
            or not isinstance(self.startup_artifact_size_bytes, int)
            or self.startup_artifact_size_bytes <= 0
            or not _sha(self.startup_artifact_sha256)
            or type(self.payload) is not GuidedNpmStartupPayload
        ):
            raise ValueError("startup_artifact_verification_invalid")


class _PersistenceRefusal(ValueError):
    def __init__(
        self,
        category: str,
        section: str,
        message: str,
        detail_code: str,
        *,
        unverified_artifact_path: str | None = None,
    ) -> None:
        self.category = category
        self.section = section
        self.message = message
        self.detail_code = detail_code
        self.unverified_artifact_path = unverified_artifact_path
        super().__init__(message)


class _PersistenceCancelled(RuntimeError):
    def __init__(self, persisted_artifact_path: str | None = None) -> None:
        self.persisted_artifact_path = persisted_artifact_path
        super().__init__("persistence_cancelled")


def _refuse(
    category: str,
    section: str,
    message: str,
    detail_code: str,
    *,
    unverified_artifact_path: str | None = None,
) -> None:
    raise _PersistenceRefusal(
        category,
        section,
        message,
        detail_code,
        unverified_artifact_path=unverified_artifact_path,
    )


def _issue(exc: _PersistenceRefusal) -> GuidedNpmStartupPersistenceIssue:
    category = exc.category if exc.category in _CATEGORY_SET else "persistence_internal_error"
    return GuidedNpmStartupPersistenceIssue(
        category, exc.section, exc.message, exc.detail_code
    )


def _failure(exc: _PersistenceRefusal) -> GuidedNpmStartupPersistenceFailure:
    return GuidedNpmStartupPersistenceFailure(
        (_issue(exc),),
        unverified_artifact_path=exc.unverified_artifact_path,
    )


def _cancelled(exc: _PersistenceCancelled) -> GuidedNpmStartupPersistenceCancelled:
    issue = GuidedNpmStartupPersistenceIssue(
        "persistence_cancelled",
        "persistence",
        "NPM startup persistence was cancelled.",
        "persistence_cancelled",
    )
    return GuidedNpmStartupPersistenceCancelled(
        (issue,), persisted_artifact_path=exc.persisted_artifact_path
    )


def _check_cancelled(
    cancellation_check: Callable[[], bool] | None,
    *,
    persisted_artifact_path: str | None = None,
) -> None:
    if cancellation_check is not None and cancellation_check():
        raise _PersistenceCancelled(persisted_artifact_path)


def canonical_guided_npm_startup_payload_bytes(
    payload: GuidedNpmStartupPayload,
) -> bytes:
    """Canonical UTF-8 JSON: sorted keys, compact separators, UTF-8 text, LF."""
    verify_guided_npm_startup_payload(payload)
    return (
        json.dumps(
            serialize_guided_npm_startup_payload(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _decode_canonical_payload_bytes(content: bytes) -> GuidedNpmStartupPayload:
    try:
        text = content.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite_json_constant:{value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("startup_artifact_noncanonical") from exc
    if not isinstance(value, Mapping):
        raise ValueError("startup_artifact_schema_mismatch")
    try:
        payload = deserialize_guided_npm_startup_payload(value)
        verify_guided_npm_startup_payload(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("startup_artifact_schema_mismatch") from exc
    if canonical_guided_npm_startup_payload_bytes(payload) != content:
        raise ValueError("startup_artifact_noncanonical")
    return payload


def _identity_payload(receipt: GuidedNpmStartupPersistenceReceipt) -> dict[str, Any]:
    return {
        item.name: _serialize(getattr(receipt, item.name))
        for item in fields(receipt)
        if item.name != "canonical_persistence_receipt_identity"
    }


def compute_guided_npm_startup_persistence_receipt_identity(
    receipt: GuidedNpmStartupPersistenceReceipt,
) -> str:
    return hashlib.sha256(
        GUIDED_NPM_STARTUP_PERSISTENCE_IDENTITY_DOMAIN.encode("utf-8")
        + b"\x00"
        + encode_canonical_value(_identity_payload(receipt))
    ).hexdigest()


def verify_application_build_identity(identity: ApplicationBuildIdentity) -> None:
    """Verify the complete canonical application build identity."""
    if type(identity) is not ApplicationBuildIdentity:
        raise ValueError("application_build_identity_type_invalid")
    rebuilt = build_application_build_identity(
        distribution_name=identity.distribution_name,
        distribution_version=identity.distribution_version,
        source_revision_kind=identity.source_revision_kind,
        source_revision=identity.source_revision,
        source_tree_state=identity.source_tree_state,
        source_tree_digest=identity.source_tree_digest,
        build_artifact_digest=identity.build_artifact_digest,
        identity_provider_version=identity.identity_provider_version,
    )
    if rebuilt != identity:
        raise ValueError("application_build_identity_mismatch")


_verify_build_identity = verify_application_build_identity


def verify_guided_npm_startup_persistence_receipt(
    receipt: GuidedNpmStartupPersistenceReceipt,
) -> None:
    """Pure receipt verification; this function performs no filesystem I/O."""
    if type(receipt) is not GuidedNpmStartupPersistenceReceipt:
        raise ValueError("persistence_receipt_type_invalid")
    replace(receipt)
    _verify_build_identity(receipt.application_build_identity)
    if (
        compute_guided_npm_startup_persistence_receipt_identity(receipt)
        != receipt.canonical_persistence_receipt_identity
    ):
        raise ValueError("persistence_receipt_identity_mismatch")


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {item.name: _serialize(getattr(value, item.name)) for item in fields(value)}
    raise ValueError("persistence_receipt_serialization_invalid")


def serialize_guided_npm_startup_persistence_receipt(
    receipt: GuidedNpmStartupPersistenceReceipt,
) -> dict[str, Any]:
    verify_guided_npm_startup_persistence_receipt(receipt)
    return {
        "identity_domain": GUIDED_NPM_STARTUP_PERSISTENCE_IDENTITY_DOMAIN,
        **_serialize(receipt),
    }


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError("persistence_receipt_serialization_invalid")
    return payload[key]


def _build_identity(payload: Mapping[str, Any]) -> ApplicationBuildIdentity:
    return ApplicationBuildIdentity(
        schema_name=_required(payload, "schema_name"),
        schema_version=_required(payload, "schema_version"),
        identity_provider_version=_required(payload, "identity_provider_version"),
        distribution_name=_required(payload, "distribution_name"),
        distribution_version=_required(payload, "distribution_version"),
        source_revision_kind=_required(payload, "source_revision_kind"),
        source_revision=_required(payload, "source_revision"),
        source_tree_state=_required(payload, "source_tree_state"),
        source_tree_digest=payload.get("source_tree_digest"),
        build_artifact_digest=payload.get("build_artifact_digest"),
        canonical_identity=_required(payload, "canonical_identity"),
    )


def deserialize_guided_npm_startup_persistence_receipt(
    payload: Mapping[str, Any],
) -> GuidedNpmStartupPersistenceReceipt:
    if not isinstance(payload, Mapping):
        raise ValueError("persistence_receipt_serialization_invalid")
    try:
        if _required(payload, "identity_domain") != GUIDED_NPM_STARTUP_PERSISTENCE_IDENTITY_DOMAIN:
            raise ValueError("persistence_receipt_serialization_invalid")
        build = _required(payload, "application_build_identity")
        if not isinstance(build, Mapping):
            raise ValueError("persistence_receipt_serialization_invalid")
        receipt = GuidedNpmStartupPersistenceReceipt(
            **{
                item.name: (
                    _build_identity(build)
                    if item.name == "application_build_identity"
                    else _required(payload, item.name)
                )
                for item in fields(GuidedNpmStartupPersistenceReceipt)
            }
        )
        verify_guided_npm_startup_persistence_receipt(receipt)
        return receipt
    except (TypeError, KeyError, ValueError, OverflowError) as exc:
        if str(exc) == "persistence_receipt_serialization_invalid":
            raise
        raise ValueError("persistence_receipt_serialization_invalid") from exc


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.fspath(left)) == os.path.normcase(os.fspath(right))


def _relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_payload(payload: Any) -> GuidedNpmStartupPayload:
    if type(payload) is not GuidedNpmStartupPayload:
        _refuse(
            "startup_payload_missing_or_invalid",
            "payload",
            "A valid B2-C4 NPM startup payload is required.",
            "startup_payload_type_invalid",
        )
    if (
        payload.startup_schema_name != GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_NAME
        or payload.startup_schema_version != GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_VERSION
    ):
        _refuse(
            "startup_payload_schema_unsupported",
            "payload",
            "The NPM startup-payload schema is unsupported.",
            "startup_payload_schema_unsupported",
        )
    try:
        verify_guided_npm_startup_payload(payload)
    except (TypeError, ValueError) as exc:
        _refuse(
            "startup_payload_identity_mismatch",
            "payload",
            "The NPM startup payload identity chain is invalid.",
            str(exc) or "startup_payload_identity_mismatch",
        )
    output = payload.output_projection
    if (
        payload.payload_status != "constructed_in_memory"
        or payload.persistence_status != "not_persisted"
        or payload.claim_status != "not_claimed"
        or payload.startup_status != "not_materialized"
        or payload.runnable is not False
        or output.creation_timing != "future_execution_start_only"
        or output.overwrite is not False
        or output.precreate is not False
        or output.path_role != "output_base"
        or output.run_directory_strategy
        != "derive_unique_run_id_under_output_base"
        or output.future_output_owner != "runner"
        or output.protected_root_context_complete is not True
        or any(item.status != "safe" for item in output.relationships)
    ):
        _refuse(
            "startup_payload_state_invalid",
            "payload",
            "The NPM startup payload is not eligible for persistence.",
            "startup_payload_state_invalid",
        )
    return payload


def _prepare_output_base(payload: GuidedNpmStartupPayload) -> tuple[Path, bool]:
    output_input = Path(payload.output_projection.output_base_canonical)
    source_input = Path(payload.source_projection.source_root_canonical)
    expected_style = "windows_drive" if os.name == "nt" else "posix_absolute"
    if payload.output_projection.output_base_path_style != expected_style:
        _refuse(
            "output_path_unsafe",
            "output",
            "The output-base path style does not match this host.",
            "output_path_style_mismatch",
        )
    if not output_input.is_absolute() or not source_input.is_absolute():
        _refuse(
            "output_path_unsafe", "output", "Startup paths must be absolute.", "path_not_absolute"
        )
    existed = output_input.exists()
    if existed:
        if output_input.is_symlink() or not output_input.is_dir():
            _refuse(
                "output_base_not_directory",
                "output",
                "The output base is not a safe directory.",
                "output_base_not_directory",
            )
        if not os.access(output_input, os.R_OK):
            _refuse(
                "output_base_unreadable", "output", "The output base is unreadable.", "output_base_unreadable"
            )
        if not os.access(output_input, os.W_OK):
            _refuse(
                "output_base_unwritable", "output", "The output base is unwritable.", "output_base_unwritable"
            )
    else:
        parent = output_input.parent
        if not parent.is_dir() or parent.is_symlink():
            _refuse(
                "output_base_missing",
                "output",
                "The output base parent is unavailable.",
                "output_base_parent_missing",
            )
    try:
        output = output_input.resolve(strict=False)
        source = source_input.resolve(strict=False)
    except OSError:
        _refuse(
            "output_path_unsafe", "output", "Startup paths could not be resolved.", "path_resolution_failed"
        )
    if (
        _same_path(output, source)
        or _relative_to(output, source)
        or _relative_to(source, output)
    ):
        _refuse(
            "output_path_unsafe",
            "output",
            "Source and NPM startup output paths overlap.",
            "source_output_overlap",
        )
    return output, existed


def _allocate_run_directory(output_base: Path) -> Path:
    for _ in range(_RUN_ALLOCATION_ATTEMPTS):
        child = output_base / f"guided_npm_run_{secrets.token_hex(16)}"
        try:
            child.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        except OSError as exc:
            _refuse(
                "run_directory_allocation_failed",
                "allocation",
                "The NPM run directory could not be allocated.",
                type(exc).__name__,
            )
        try:
            resolved = child.resolve(strict=True)
            parent = output_base.resolve(strict=True)
        except OSError:
            _refuse(
                "output_path_unsafe", "allocation", "The allocated run directory is unsafe.", "run_directory_resolution_failed"
            )
        if not _same_path(resolved.parent, parent) or not _relative_to(resolved, parent):
            _refuse(
                "output_path_unsafe", "allocation", "The allocated run directory escaped its output base.", "run_directory_escape"
            )
        return resolved
    _refuse(
        "run_directory_conflict",
        "allocation",
        "Unique NPM run-directory allocation collided repeatedly.",
        "run_directory_collision_budget_exhausted",
    )


def _write_temp_file(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            written = handle.write(content)
            if written != len(content):
                raise OSError("partial_write")
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except OSError as exc:
                _refuse(
                    "startup_artifact_flush_failed",
                    "artifact",
                    "The NPM startup artifact could not be durably flushed.",
                    type(exc).__name__,
                )
    except FileExistsError:
        _refuse(
            "startup_artifact_conflict", "artifact", "The temporary startup artifact already exists.", "temporary_artifact_conflict"
        )
    except _PersistenceRefusal:
        raise
    except OSError as exc:
        _refuse(
            "startup_artifact_write_failed",
            "artifact",
            "The NPM startup artifact could not be written.",
            type(exc).__name__,
        )


def _publish_no_replace(temp_path: Path, final_path: Path) -> None:
    try:
        os.link(temp_path, final_path, follow_symlinks=False)
        temp_path.unlink()
    except FileExistsError:
        _refuse(
            "startup_artifact_conflict",
            "artifact",
            "The final NPM startup artifact already exists.",
            "final_artifact_conflict",
            unverified_artifact_path=os.fspath(final_path),
        )
    except OSError as exc:
        _refuse(
            "startup_artifact_publish_failed",
            "artifact",
            "The NPM startup artifact could not be atomically published.",
            type(exc).__name__,
        )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_artifact_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("startup_artifact_path_invalid")
    return path.read_bytes()


def verify_guided_npm_startup_artifact_path(
    startup_artifact_path: str,
) -> GuidedNpmVerifiedStartupArtifact:
    """Open and fully verify one explicitly named persisted NPM artifact."""
    if not isinstance(startup_artifact_path, str) or not startup_artifact_path.strip():
        raise ValueError("startup_artifact_path_invalid")
    path = Path(startup_artifact_path)
    if not path.is_absolute() or path.name != GUIDED_NPM_STARTUP_ARTIFACT_FILENAME:
        raise ValueError("startup_artifact_path_invalid")
    if not os.path.lexists(path):
        raise FileNotFoundError("startup_artifact_missing")
    if path.is_symlink():
        raise ValueError("startup_artifact_alias_invalid")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise OSError("startup_artifact_read_failed") from exc
    if not _same_path(path, resolved):
        raise ValueError("startup_artifact_alias_invalid")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise OSError("startup_artifact_read_failed") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("startup_artifact_not_regular")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise OSError("startup_artifact_read_failed") from exc
    payload = _decode_canonical_payload_bytes(content)
    if (
        payload.startup_schema_name != GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_NAME
        or payload.startup_schema_version != GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_VERSION
    ):
        raise ValueError("startup_artifact_schema_mismatch")
    return GuidedNpmVerifiedStartupArtifact(
        startup_artifact_path=os.fspath(resolved),
        startup_artifact_size_bytes=len(content),
        startup_artifact_sha256=hashlib.sha256(content).hexdigest(),
        payload=payload,
    )


def _safe_cleanup(
    *,
    temp_path: Path | None,
    final_path: Path | None,
    final_owned: bool,
    run_directory: Path | None,
    output_base: Path | None,
    output_base_owned: bool,
) -> None:
    if temp_path is not None:
        try:
            if temp_path.is_file() and not temp_path.is_symlink():
                temp_path.unlink()
        except OSError:
            pass
    if final_owned and final_path is not None:
        try:
            if final_path.is_file() and not final_path.is_symlink():
                final_path.unlink()
        except OSError:
            pass
    if run_directory is not None:
        try:
            run_directory.rmdir()
        except OSError:
            pass
    if output_base_owned and output_base is not None:
        try:
            output_base.rmdir()
        except OSError:
            pass


def verify_persisted_guided_npm_startup_artifact(
    receipt: GuidedNpmStartupPersistenceReceipt,
) -> GuidedNpmStartupPayload:
    """Live verification of the exact artifact path frozen in a receipt."""
    verify_guided_npm_startup_persistence_receipt(receipt)
    try:
        verified = verify_guided_npm_startup_artifact_path(
            receipt.startup_artifact_path
        )
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("startup_artifact_readback_failed") from exc
    except ValueError as exc:
        if str(exc) in {
            "startup_artifact_noncanonical",
            "startup_artifact_schema_mismatch",
        }:
            raise
        raise ValueError("startup_artifact_readback_failed") from exc
    if verified.startup_artifact_size_bytes != receipt.persisted_size_bytes:
        raise ValueError("startup_artifact_size_mismatch")
    if verified.startup_artifact_sha256 != receipt.persisted_artifact_sha256:
        raise ValueError("startup_artifact_digest_mismatch")
    payload = verified.payload
    if payload.canonical_startup_payload_identity != receipt.source_startup_payload_identity:
        raise ValueError("startup_artifact_payload_identity_mismatch")
    return payload


def persist_guided_npm_startup_payload(
    payload: GuidedNpmStartupPayload,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedNpmStartupPersistenceResult:
    """Allocate, atomically persist, read back, and verify one NPM artifact."""
    output_base: Path | None = None
    run_directory: Path | None = None
    temp_path: Path | None = None
    final_path: Path | None = None
    output_base_owned = False
    final_owned = False
    try:
        _check_cancelled(cancellation_check)
        payload = _validate_payload(payload)
        _check_cancelled(cancellation_check)
        intended = canonical_guided_npm_startup_payload_bytes(payload)
        intended_digest = hashlib.sha256(intended).hexdigest()
        output_base, output_preexisted = _prepare_output_base(payload)
        _check_cancelled(cancellation_check)
        if not output_preexisted:
            try:
                output_base.mkdir(exist_ok=False)
                output_base_owned = True
            except FileExistsError:
                _refuse(
                    "run_directory_conflict", "output", "The output base appeared during allocation.", "output_base_race"
                )
            except OSError as exc:
                _refuse(
                    "run_directory_allocation_failed", "output", "The output base could not be created.", type(exc).__name__
                )
        run_directory = _allocate_run_directory(output_base)
        _check_cancelled(cancellation_check)
        final_path = run_directory / GUIDED_NPM_STARTUP_ARTIFACT_FILENAME
        if os.path.lexists(final_path):
            _refuse(
                "startup_artifact_conflict", "artifact", "The startup artifact path already exists.", "startup_artifact_preexisting"
            )
        temp_path = run_directory / (
            f".{GUIDED_NPM_STARTUP_ARTIFACT_FILENAME}.{secrets.token_hex(8)}.tmp"
        )
        _write_temp_file(temp_path, intended)
        _check_cancelled(cancellation_check)
        _publish_no_replace(temp_path, final_path)
        temp_path = None
        final_owned = True
        try:
            _fsync_directory(run_directory)
        except OSError as exc:
            _refuse(
                "startup_artifact_flush_failed", "artifact", "The startup directory could not be durably flushed.", type(exc).__name__
            )
        _check_cancelled(
            cancellation_check,
            persisted_artifact_path=os.fspath(final_path),
        )
        try:
            readback = _read_artifact_bytes(final_path)
        except (OSError, ValueError) as exc:
            _refuse(
                "startup_artifact_readback_failed", "artifact", "The persisted startup artifact could not be read back.", type(exc).__name__
            )
        if len(readback) != len(intended):
            _refuse(
                "startup_artifact_size_mismatch", "artifact", "The persisted startup artifact size changed.", "readback_size_mismatch"
            )
        observed_digest = hashlib.sha256(readback).hexdigest()
        if observed_digest != intended_digest:
            _refuse(
                "startup_artifact_digest_mismatch", "artifact", "The persisted startup artifact digest changed.", "readback_digest_mismatch"
            )
        if readback != intended:
            _refuse(
                "startup_artifact_bytes_mismatch", "artifact", "The persisted startup bytes differ.", "readback_bytes_mismatch"
            )
        try:
            restored = _decode_canonical_payload_bytes(readback)
        except ValueError as exc:
            category = str(exc)
            if category not in {
                "startup_artifact_noncanonical",
                "startup_artifact_schema_mismatch",
            }:
                category = "startup_artifact_noncanonical"
            _refuse(category, "artifact", "The persisted startup artifact is invalid.", str(exc))
        if restored.canonical_startup_payload_identity != payload.canonical_startup_payload_identity:
            _refuse(
                "startup_artifact_payload_identity_mismatch", "artifact", "The readback payload identity differs.", "readback_payload_identity_mismatch"
            )
        _check_cancelled(
            cancellation_check,
            persisted_artifact_path=os.fspath(final_path),
        )
        receipt = GuidedNpmStartupPersistenceReceipt(
            persistence_schema_name=GUIDED_NPM_STARTUP_PERSISTENCE_SCHEMA_NAME,
            persistence_schema_version=GUIDED_NPM_STARTUP_PERSISTENCE_SCHEMA_VERSION,
            persistence_contract_version=GUIDED_NPM_STARTUP_PERSISTENCE_CONTRACT_VERSION,
            source_startup_payload_identity=payload.canonical_startup_payload_identity,
            source_authorization_identity=payload.source_authorization_identity,
            source_authority_identity=payload.source_authority_identity,
            source_request_identity=payload.source_request_identity,
            validation_revision=payload.validation_revision,
            guided_plan_identity=payload.guided_plan_identity,
            application_build_identity=payload.application_build_identity,
            run_directory_path=os.fspath(run_directory),
            startup_artifact_path=os.fspath(final_path),
            startup_artifact_filename=GUIDED_NPM_STARTUP_ARTIFACT_FILENAME,
            serialized_payload_sha256=intended_digest,
            persisted_artifact_sha256=observed_digest,
            persisted_size_bytes=len(readback),
            readback_payload_identity=restored.canonical_startup_payload_identity,
            canonical_persistence_receipt_identity="0" * 64,
            persistence_status=GUIDED_NPM_STARTUP_PERSISTENCE_STATUS,
            claim_status=GUIDED_NPM_STARTUP_PERSISTENCE_CLAIM_STATUS,
            startup_status=GUIDED_NPM_STARTUP_PERSISTENCE_STARTUP_STATUS,
            runnable=False,
        )
        receipt = replace(
            receipt,
            canonical_persistence_receipt_identity=(
                compute_guided_npm_startup_persistence_receipt_identity(receipt)
            ),
        )
        verify_guided_npm_startup_persistence_receipt(receipt)
        return receipt
    except _PersistenceCancelled as exc:
        _safe_cleanup(
            temp_path=temp_path,
            final_path=final_path,
            final_owned=final_owned,
            run_directory=run_directory,
            output_base=output_base,
            output_base_owned=output_base_owned,
        )
        return _cancelled(exc)
    except _PersistenceRefusal as exc:
        _safe_cleanup(
            temp_path=temp_path,
            final_path=final_path,
            final_owned=final_owned,
            run_directory=run_directory,
            output_base=output_base,
            output_base_owned=output_base_owned,
        )
        return _failure(exc)
    except Exception as exc:
        _safe_cleanup(
            temp_path=temp_path,
            final_path=final_path,
            final_owned=final_owned,
            run_directory=run_directory,
            output_base=output_base,
            output_base_owned=output_base_owned,
        )
        return _failure(
            _PersistenceRefusal(
                "persistence_internal_error",
                "persistence",
                "NPM startup persistence failed.",
                type(exc).__name__,
            )
        )
