"""B2-C5 NPM-specific claim of one durably persisted startup artifact."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
import hashlib
import os
from pathlib import Path
from typing import Any

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_npm_startup_payload import (
    GuidedNpmStartupPayload,
)
from photometry_pipeline.guided_npm_startup_persistence import (
    GUIDED_NPM_STARTUP_ARTIFACT_FILENAME,
    GuidedNpmVerifiedStartupArtifact,
    GuidedNpmStartupPersistenceReceipt,
    verify_application_build_identity,
    verify_guided_npm_startup_artifact_path,
    verify_guided_npm_startup_persistence_receipt,
)
from photometry_pipeline.guided_production_mapping import (
    ApplicationBuildIdentity,
)


GUIDED_NPM_STARTUP_WRAPPER_ARGUMENT = "--guided-npm-startup-artifact"
GUIDED_RWD_STARTUP_WRAPPER_ARGUMENT = "--guided-candidate-manifest"
GUIDED_NPM_STARTUP_CLAIM_SCHEMA_NAME = "guided_npm_startup_claim_receipt"
GUIDED_NPM_STARTUP_CLAIM_SCHEMA_VERSION = "v1"
GUIDED_NPM_STARTUP_CLAIM_CONTRACT_VERSION = "guided_npm_startup_claim_receipt.v1"
GUIDED_NPM_STARTUP_CLAIM_IDENTITY_DOMAIN = "guided_npm_startup_claim_receipt.v1"
GUIDED_NPM_STARTUP_CLAIM_STATUS = "claimed_for_npm_startup"
GUIDED_NPM_STARTUP_CLAIM_STARTUP_STATUS = "claimed_not_executed"
GUIDED_NPM_STARTUP_CLAIM_SOURCE_PERSISTENCE_RECEIPT = "persistence_receipt"
GUIDED_NPM_STARTUP_CLAIM_SOURCE_DIRECT_ARTIFACT = "direct_persisted_artifact"

GUIDED_NPM_STARTUP_CLAIM_REFUSAL_CATEGORIES = (
    "claim_receipt_missing_or_invalid",
    "claim_artifact_path_invalid",
    "claim_build_identity_mismatch",
    "claim_artifact_missing",
    "claim_artifact_not_regular",
    "claim_artifact_alias_invalid",
    "claim_artifact_read_failed",
    "claim_artifact_noncanonical",
    "claim_artifact_schema_mismatch",
    "claim_artifact_payload_identity_mismatch",
    "claim_artifact_path_mismatch",
    "claim_artifact_size_mismatch",
    "claim_artifact_digest_mismatch",
    "claim_payload_identity_mismatch",
    "claim_plan_identity_mismatch",
    "claim_validation_revision_mismatch",
    "claim_schema_mismatch",
    "claim_argument_conflict",
    "claim_cancelled",
    "claim_receipt_identity_mismatch",
    "claim_internal_error",
)
_CATEGORY_SET = frozenset(GUIDED_NPM_STARTUP_CLAIM_REFUSAL_CATEGORIES)
_HEX = frozenset("0123456789abcdef")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


@dataclass(frozen=True)
class GuidedNpmStartupClaimIssue:
    category: str
    section: str
    message: str
    detail_code: str

    def __post_init__(self) -> None:
        if self.category not in _CATEGORY_SET or not all(
            _text(value) for value in (self.section, self.message, self.detail_code)
        ):
            raise ValueError("Invalid NPM startup-claim issue.")


@dataclass(frozen=True)
class GuidedNpmStartupClaimFailure:
    blocking_issues: tuple[GuidedNpmStartupClaimIssue, ...]
    status: str = "refused"

    def __post_init__(self) -> None:
        if len(self.blocking_issues) != 1 or self.status != "refused":
            raise ValueError("Claim failure requires one blocking issue.")


@dataclass(frozen=True)
class GuidedNpmStartupClaimCancelled:
    blocking_issues: tuple[GuidedNpmStartupClaimIssue, ...]
    status: str = "cancelled"

    def __post_init__(self) -> None:
        if (
            len(self.blocking_issues) != 1
            or self.blocking_issues[0].category != "claim_cancelled"
            or self.status != "cancelled"
        ):
            raise ValueError("Cancelled claim requires one cancellation issue.")


@dataclass(frozen=True)
class GuidedNpmStartupClaimReceipt:
    claim_schema_name: str
    claim_schema_version: str
    claim_contract_version: str
    claim_source_kind: str
    source_persistence_receipt_identity: str | None
    source_startup_payload_identity: str
    startup_artifact_path: str
    startup_artifact_sha256: str
    startup_artifact_size_bytes: int
    claimed_payload_identity: str
    application_build_identity: ApplicationBuildIdentity
    guided_plan_identity: str
    validation_revision: int
    wrapper_argument_name: str
    wrapper_argument_path: str
    claim_status: str
    startup_status: str
    runnable: bool
    canonical_claim_receipt_identity: str

    def __post_init__(self) -> None:
        if (
            self.claim_schema_name != GUIDED_NPM_STARTUP_CLAIM_SCHEMA_NAME
            or self.claim_schema_version != GUIDED_NPM_STARTUP_CLAIM_SCHEMA_VERSION
            or self.claim_contract_version != GUIDED_NPM_STARTUP_CLAIM_CONTRACT_VERSION
        ):
            raise ValueError("Unsupported NPM claim-receipt schema.")
        for name in (
            "source_startup_payload_identity",
            "startup_artifact_sha256",
            "claimed_payload_identity",
            "guided_plan_identity",
            "canonical_claim_receipt_identity",
        ):
            if not _sha(getattr(self, name)):
                raise ValueError(f"{name} is invalid.")
        if self.claim_source_kind == GUIDED_NPM_STARTUP_CLAIM_SOURCE_PERSISTENCE_RECEIPT:
            if not _sha(self.source_persistence_receipt_identity):
                raise ValueError("Receipt-based claim provenance is invalid.")
        elif self.claim_source_kind == GUIDED_NPM_STARTUP_CLAIM_SOURCE_DIRECT_ARTIFACT:
            if self.source_persistence_receipt_identity is not None:
                raise ValueError("Direct-artifact claim provenance is invalid.")
        else:
            raise ValueError("Claim source kind is invalid.")
        if (
            isinstance(self.startup_artifact_size_bytes, bool)
            or not isinstance(self.startup_artifact_size_bytes, int)
            or self.startup_artifact_size_bytes <= 0
            or isinstance(self.validation_revision, bool)
            or not isinstance(self.validation_revision, int)
            or self.validation_revision < 0
        ):
            raise ValueError("Claim receipt numeric field is invalid.")
        if not isinstance(self.application_build_identity, ApplicationBuildIdentity):
            raise ValueError("Claim receipt build identity is invalid.")
        artifact = Path(self.startup_artifact_path)
        if (
            not artifact.is_absolute()
            or artifact.name != GUIDED_NPM_STARTUP_ARTIFACT_FILENAME
            or self.wrapper_argument_name != GUIDED_NPM_STARTUP_WRAPPER_ARGUMENT
            or self.wrapper_argument_path != self.startup_artifact_path
        ):
            raise ValueError("Claim receipt artifact argument is invalid.")
        if self.claimed_payload_identity != self.source_startup_payload_identity:
            raise ValueError("Claimed payload identity differs from persistence.")
        if (
            self.claim_status != GUIDED_NPM_STARTUP_CLAIM_STATUS
            or self.startup_status != GUIDED_NPM_STARTUP_CLAIM_STARTUP_STATUS
            or self.runnable is not False
        ):
            raise ValueError("Claim receipt state is invalid.")


GuidedNpmStartupClaimResult = (
    GuidedNpmStartupClaimReceipt
    | GuidedNpmStartupClaimFailure
    | GuidedNpmStartupClaimCancelled
)


class _ClaimRefusal(ValueError):
    def __init__(self, category: str, section: str, message: str, detail_code: str):
        self.category = category
        self.section = section
        self.message = message
        self.detail_code = detail_code
        super().__init__(message)


class _ClaimCancelled(RuntimeError):
    pass


def _refuse(category: str, section: str, message: str, detail_code: str) -> None:
    raise _ClaimRefusal(category, section, message, detail_code)


def _failure(exc: _ClaimRefusal) -> GuidedNpmStartupClaimFailure:
    category = exc.category if exc.category in _CATEGORY_SET else "claim_internal_error"
    return GuidedNpmStartupClaimFailure(
        (
            GuidedNpmStartupClaimIssue(
                category, exc.section, exc.message, exc.detail_code
            ),
        )
    )


def _cancelled() -> GuidedNpmStartupClaimCancelled:
    return GuidedNpmStartupClaimCancelled(
        (
            GuidedNpmStartupClaimIssue(
                "claim_cancelled",
                "claim",
                "NPM startup claim was cancelled.",
                "claim_cancelled",
            ),
        )
    )


def _check_cancelled(cancellation_check: Callable[[], bool] | None) -> None:
    if cancellation_check is not None and cancellation_check():
        raise _ClaimCancelled("claim_cancelled")


def validate_guided_startup_authority_argument_selection(
    *,
    guided_candidate_manifest: str | None,
    guided_npm_startup_artifact: str | None,
    guided_claim_required: bool = True,
) -> str | None:
    """Pure NPM/RWD wrapper-argument mutual-exclusion classification."""
    rwd = bool(guided_candidate_manifest)
    npm = bool(guided_npm_startup_artifact)
    if rwd and npm:
        raise ValueError("claim_argument_conflict")
    if guided_claim_required and not rwd and not npm:
        raise ValueError("claim_argument_missing")
    if npm:
        return "npm"
    if rwd:
        return "rwd"
    return None


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            item.name: _serialize(getattr(value, item.name))
            for item in fields(value)
        }
    raise ValueError("claim_receipt_serialization_invalid")


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError("claim_receipt_serialization_invalid")
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


def _identity_payload(receipt: GuidedNpmStartupClaimReceipt) -> dict[str, Any]:
    return {
        item.name: _serialize(getattr(receipt, item.name))
        for item in fields(receipt)
        if item.name != "canonical_claim_receipt_identity"
    }


def compute_guided_npm_startup_claim_receipt_identity(
    receipt: GuidedNpmStartupClaimReceipt,
) -> str:
    return hashlib.sha256(
        GUIDED_NPM_STARTUP_CLAIM_IDENTITY_DOMAIN.encode("utf-8")
        + b"\x00"
        + encode_canonical_value(_identity_payload(receipt))
    ).hexdigest()


def verify_guided_npm_startup_claim_receipt(
    receipt: GuidedNpmStartupClaimReceipt,
) -> None:
    """Pure claim-receipt verification; this function performs no I/O."""
    if type(receipt) is not GuidedNpmStartupClaimReceipt:
        raise ValueError("claim_receipt_type_invalid")
    replace(receipt)
    verify_application_build_identity(receipt.application_build_identity)
    if (
        compute_guided_npm_startup_claim_receipt_identity(receipt)
        != receipt.canonical_claim_receipt_identity
    ):
        raise ValueError("claim_receipt_identity_mismatch")


def serialize_guided_npm_startup_claim_receipt(
    receipt: GuidedNpmStartupClaimReceipt,
) -> dict[str, Any]:
    verify_guided_npm_startup_claim_receipt(receipt)
    return {
        "identity_domain": GUIDED_NPM_STARTUP_CLAIM_IDENTITY_DOMAIN,
        **_serialize(receipt),
    }


def deserialize_guided_npm_startup_claim_receipt(
    payload: Mapping[str, Any],
) -> GuidedNpmStartupClaimReceipt:
    if not isinstance(payload, Mapping):
        raise ValueError("claim_receipt_serialization_invalid")
    try:
        if _required(payload, "identity_domain") != GUIDED_NPM_STARTUP_CLAIM_IDENTITY_DOMAIN:
            raise ValueError("claim_receipt_serialization_invalid")
        build = _required(payload, "application_build_identity")
        if not isinstance(build, Mapping):
            raise ValueError("claim_receipt_serialization_invalid")
        receipt = GuidedNpmStartupClaimReceipt(
            **{
                item.name: (
                    _build_identity(build)
                    if item.name == "application_build_identity"
                    else _required(payload, item.name)
                )
                for item in fields(GuidedNpmStartupClaimReceipt)
            }
        )
        verify_guided_npm_startup_claim_receipt(receipt)
        return receipt
    except (TypeError, KeyError, ValueError, OverflowError) as exc:
        if str(exc) == "claim_receipt_serialization_invalid":
            raise
        raise ValueError("claim_receipt_serialization_invalid") from exc


def _validate_current_build(identity: ApplicationBuildIdentity) -> None:
    try:
        verify_application_build_identity(identity)
    except (TypeError, ValueError) as exc:
        _refuse(
            "claim_build_identity_mismatch",
            "build",
            "The current application build identity is invalid.",
            str(exc) or "claim_build_identity_invalid",
        )


def _verify_artifact_for_claim(path: str) -> GuidedNpmVerifiedStartupArtifact:
    try:
        return verify_guided_npm_startup_artifact_path(path)
    except FileNotFoundError as exc:
        _refuse("claim_artifact_missing", "artifact", "The NPM startup artifact is missing.", str(exc))
    except OSError as exc:
        _refuse("claim_artifact_read_failed", "artifact", "The NPM startup artifact could not be read.", str(exc))
    except ValueError as exc:
        detail = str(exc)
        category = {
            "startup_artifact_path_invalid": "claim_artifact_path_invalid",
            "startup_artifact_not_regular": "claim_artifact_not_regular",
            "startup_artifact_alias_invalid": "claim_artifact_alias_invalid",
            "startup_artifact_noncanonical": "claim_artifact_noncanonical",
            "startup_artifact_schema_mismatch": "claim_artifact_schema_mismatch",
        }.get(detail, "claim_artifact_schema_mismatch")
        _refuse(category, "artifact", "The NPM startup artifact could not be verified.", detail or category)
    raise AssertionError("unreachable")


def _build_claim_receipt(
    verified: GuidedNpmVerifiedStartupArtifact,
    *,
    current_application_build_identity: ApplicationBuildIdentity,
    claim_source_kind: str,
    source_persistence_receipt_identity: str | None,
) -> GuidedNpmStartupClaimReceipt:
    payload = verified.payload
    if payload.application_build_identity != current_application_build_identity:
        _refuse(
            "claim_build_identity_mismatch", "artifact",
            "The claimed payload build identity differs from the current build.",
            "claim_payload_build_identity_mismatch",
        )
    claim = GuidedNpmStartupClaimReceipt(
        claim_schema_name=GUIDED_NPM_STARTUP_CLAIM_SCHEMA_NAME,
        claim_schema_version=GUIDED_NPM_STARTUP_CLAIM_SCHEMA_VERSION,
        claim_contract_version=GUIDED_NPM_STARTUP_CLAIM_CONTRACT_VERSION,
        claim_source_kind=claim_source_kind,
        source_persistence_receipt_identity=source_persistence_receipt_identity,
        source_startup_payload_identity=payload.canonical_startup_payload_identity,
        startup_artifact_path=verified.startup_artifact_path,
        startup_artifact_sha256=verified.startup_artifact_sha256,
        startup_artifact_size_bytes=verified.startup_artifact_size_bytes,
        claimed_payload_identity=payload.canonical_startup_payload_identity,
        application_build_identity=current_application_build_identity,
        guided_plan_identity=payload.guided_plan_identity,
        validation_revision=payload.validation_revision,
        wrapper_argument_name=GUIDED_NPM_STARTUP_WRAPPER_ARGUMENT,
        wrapper_argument_path=verified.startup_artifact_path,
        claim_status=GUIDED_NPM_STARTUP_CLAIM_STATUS,
        startup_status=GUIDED_NPM_STARTUP_CLAIM_STARTUP_STATUS,
        runnable=False,
        canonical_claim_receipt_identity="0" * 64,
    )
    claim = replace(
        claim,
        canonical_claim_receipt_identity=compute_guided_npm_startup_claim_receipt_identity(claim),
    )
    verify_guided_npm_startup_claim_receipt(claim)
    return claim


def claim_guided_npm_startup_artifact_path(
    startup_artifact_path: str,
    *,
    current_application_build_identity: ApplicationBuildIdentity,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedNpmStartupClaimResult:
    """Claim a persisted artifact directly; no persistence receipt is required."""
    try:
        _check_cancelled(cancellation_check)
        _validate_current_build(current_application_build_identity)
        verified = _verify_artifact_for_claim(startup_artifact_path)
        _check_cancelled(cancellation_check)
        return _build_claim_receipt(
            verified,
            current_application_build_identity=current_application_build_identity,
            claim_source_kind=GUIDED_NPM_STARTUP_CLAIM_SOURCE_DIRECT_ARTIFACT,
            source_persistence_receipt_identity=None,
        )
    except _ClaimCancelled:
        return _cancelled()
    except _ClaimRefusal as exc:
        return _failure(exc)
    except Exception as exc:
        return _failure(_ClaimRefusal("claim_internal_error", "claim", "NPM startup claim failed.", type(exc).__name__))


def claim_guided_npm_startup_artifact_from_receipt(
    receipt: GuidedNpmStartupPersistenceReceipt,
    *,
    current_application_build_identity: ApplicationBuildIdentity,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedNpmStartupClaimResult:
    """Claim the exact receipt-bound NPM artifact without launching execution."""
    try:
        _check_cancelled(cancellation_check)
        if type(receipt) is not GuidedNpmStartupPersistenceReceipt:
            _refuse("claim_receipt_missing_or_invalid", "receipt", "A valid NPM persistence receipt is required.", "receipt_type_invalid")
        try:
            verify_guided_npm_startup_persistence_receipt(receipt)
        except (TypeError, ValueError) as exc:
            _refuse("claim_receipt_missing_or_invalid", "receipt", "The NPM persistence receipt is invalid.", str(exc) or "receipt_invalid")
        _validate_current_build(current_application_build_identity)
        if current_application_build_identity != receipt.application_build_identity:
            _refuse("claim_build_identity_mismatch", "build", "The current build differs from the persisted startup build.", "claim_build_identity_mismatch")
        path = Path(receipt.startup_artifact_path)
        if not os.path.lexists(path):
            _refuse("claim_artifact_missing", "artifact", "The NPM startup artifact is missing.", "claim_artifact_missing")
        if path.is_symlink() or not path.is_file():
            _refuse("claim_artifact_path_mismatch", "artifact", "The NPM startup artifact path is not a regular file.", "claim_artifact_alias_or_type_invalid")
        try:
            live_bytes = path.read_bytes()
        except OSError as exc:
            _refuse("claim_artifact_missing", "artifact", "The NPM startup artifact cannot be read.", type(exc).__name__)
        if len(live_bytes) != receipt.persisted_size_bytes:
            _refuse("claim_artifact_size_mismatch", "artifact", "The NPM startup artifact size changed.", "claim_artifact_size_mismatch")
        if hashlib.sha256(live_bytes).hexdigest() != receipt.persisted_artifact_sha256:
            _refuse("claim_artifact_digest_mismatch", "artifact", "The NPM startup artifact digest changed.", "claim_artifact_digest_mismatch")
        verified = _verify_artifact_for_claim(receipt.startup_artifact_path)
        if verified.startup_artifact_size_bytes != receipt.persisted_size_bytes:
            _refuse("claim_artifact_size_mismatch", "artifact", "The NPM startup artifact size changed.", "claim_artifact_size_mismatch")
        if verified.startup_artifact_sha256 != receipt.persisted_artifact_sha256:
            _refuse("claim_artifact_digest_mismatch", "artifact", "The NPM startup artifact digest changed.", "claim_artifact_digest_mismatch")
        payload = verified.payload
        if payload.canonical_startup_payload_identity != receipt.source_startup_payload_identity:
            _refuse("claim_payload_identity_mismatch", "artifact", "The claimed payload identity differs from persistence.", "claim_payload_identity_mismatch")
        if payload.guided_plan_identity != receipt.guided_plan_identity:
            _refuse("claim_plan_identity_mismatch", "artifact", "The claimed payload plan identity differs.", "claim_plan_identity_mismatch")
        if payload.validation_revision != receipt.validation_revision:
            _refuse("claim_validation_revision_mismatch", "artifact", "The claimed payload revision differs.", "claim_validation_revision_mismatch")
        _check_cancelled(cancellation_check)
        return _build_claim_receipt(
            verified,
            current_application_build_identity=current_application_build_identity,
            claim_source_kind=GUIDED_NPM_STARTUP_CLAIM_SOURCE_PERSISTENCE_RECEIPT,
            source_persistence_receipt_identity=receipt.canonical_persistence_receipt_identity,
        )
    except _ClaimCancelled:
        return _cancelled()
    except _ClaimRefusal as exc:
        return _failure(exc)
    except Exception as exc:
        return _failure(_ClaimRefusal("claim_internal_error", "claim", "NPM startup claim failed.", type(exc).__name__))


# Backward-compatible B2-C5 name: this remains the receipt-based entry point.
claim_guided_npm_startup_artifact = claim_guided_npm_startup_artifact_from_receipt
