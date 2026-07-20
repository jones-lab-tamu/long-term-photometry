"""Bridge NPM's already-accepted execution authority into the mature shared
Guided startup path (the same path RWD already uses for allocation,
materialization, wrapper launch, completion, and Results).

This module adds no new worker, execution-request family, identity function,
serialized authority, terminal receipt, output layout, or Results adapter. It
wraps the two existing, independently proven authority objects
(``GuidedRunAuthorizationResult`` for RWD, ``GuidedNpmProductionExecutionIntent``
+ ``GuidedNpmExecutionAuthority`` for NPM) behind one small in-memory
discriminated type, and compiles NPM's already-accepted intent/authority into
the same three generic startup payload types RWD's
``derive_guided_execution_payloads`` already produces
(``GuidedExecutionConfigPayload``, ``GuidedRunnerCandidateManifestPayload``,
``GuidedStartupProvenanceSeed``), reusing their existing canonicalization and
identity functions unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from typing import Any

from photometry_pipeline.config import Config
from photometry_pipeline.guided_execution_payloads import (
    GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE,
    GuidedConfigFieldValue,
    GuidedExecutionConfigPayload,
    GuidedExecutionPayloadDerivationResult,
    GuidedExecutionPayloadIssue,
    GuidedExecutionStartupMappingContract,
    GuidedRunnerCandidateManifestEntry,
    GuidedRunnerCandidateManifestPayload,
    GuidedStartupProvenanceSeed,
    compute_guided_execution_config_payload_identity,
    compute_guided_runner_candidate_manifest_payload_identity,
    compute_guided_startup_provenance_seed_identity,
    replace_config_payload_identity,
    replace_manifest_payload_identity,
    replace_provenance_seed_identity,
)
from photometry_pipeline.guided_npm_execution_authority import (
    GuidedNpmExecutionAuthority,
    compute_guided_npm_execution_authority_identity,
    verify_guided_npm_execution_authority,
)
from photometry_pipeline.guided_production_mapping import (
    GuidedNpmProductionExecutionIntent,
    compute_guided_npm_production_execution_intent_identity,
)
from photometry_pipeline.guided_run_authorization import (
    GuidedRunAuthorizationResult,
    compute_guided_run_authorization_identity,
)


_CORRECTION_CONFIG_NAMES = {
    "slope_constraint": "dynamic_fit_slope_constraint",
    "min_slope": "dynamic_fit_min_slope",
}


def _typed_values(values: tuple[Any, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        if item.field_name in result:
            raise ValueError("npm_startup_bridge_duplicate_typed_field")
        result[item.field_name] = item.value
    return result


@dataclass(frozen=True)
class GuidedStartupAuthority:
    """Carries exactly one of the two existing, independently proven Guided
    authority representations, unmodified: RWD's single
    ``GuidedRunAuthorizationResult`` (already nests its own production
    intent), or NPM's pair -- ``GuidedNpmProductionExecutionIntent`` plus the
    ``GuidedNpmExecutionAuthority`` built from it -- since NPM's architecture
    keeps those as two separate, identity-linked objects rather than one
    nested structure. Not a new identity: every accessor below reads a value
    that already exists on a wrapped object; nothing here is computed,
    cached, or re-derived."""

    rwd: GuidedRunAuthorizationResult | None = None
    npm_intent: GuidedNpmProductionExecutionIntent | None = None
    npm_authority: GuidedNpmExecutionAuthority | None = None

    def __post_init__(self) -> None:
        npm_present = self.npm_intent is not None or self.npm_authority is not None
        if (self.npm_intent is not None) != (self.npm_authority is not None):
            raise ValueError(
                "GuidedStartupAuthority.npm_intent and npm_authority must be "
                "supplied together."
            )
        if (self.rwd is not None) == npm_present:
            raise ValueError(
                "GuidedStartupAuthority must carry exactly one of rwd or "
                "npm_intent/npm_authority."
            )
        if self.rwd is not None and type(self.rwd) is not GuidedRunAuthorizationResult:
            raise ValueError("GuidedStartupAuthority.rwd has an invalid type.")
        if (
            self.npm_intent is not None
            and not isinstance(self.npm_intent, GuidedNpmProductionExecutionIntent)
        ):
            raise ValueError("GuidedStartupAuthority.npm_intent has an invalid type.")
        if (
            self.npm_authority is not None
            and type(self.npm_authority) is not GuidedNpmExecutionAuthority
        ):
            raise ValueError("GuidedStartupAuthority.npm_authority has an invalid type.")

    @property
    def is_npm(self) -> bool:
        return self.npm_intent is not None

    @property
    def source_format(self) -> str:
        if self.rwd is not None:
            return self.rwd.production_intent.input_source.source_format
        return self.npm_intent.source_format

    @property
    def execution_mode(self) -> str:
        if self.rwd is not None:
            return self.rwd.production_intent.execution_profile.execution_mode
        return self.npm_intent.execution_mode

    @property
    def sessions_per_hour(self) -> int:
        if self.rwd is not None:
            return self.rwd.production_intent.acquisition.sessions_per_hour
        return self.npm_authority.recording_policy.sessions_per_hour

    @property
    def source_root_canonical(self) -> str:
        if self.rwd is not None:
            return self.rwd.production_intent.input_source.source_root_canonical
        return self.npm_intent.source_root_canonical

    @property
    def output_base_canonical(self) -> str:
        if self.rwd is not None:
            return self.rwd.production_intent.output_policy.output_base_canonical
        return self.npm_intent.output_policy.output_base_canonical

    @property
    def overwrite(self) -> bool:
        if self.rwd is not None:
            return self.rwd.production_intent.output_policy.overwrite
        return self.npm_intent.output_policy.overwrite

    @property
    def protected_root_context_complete(self) -> bool:
        if self.rwd is not None:
            return self.rwd.production_intent.output_policy.protected_root_context_complete
        return self.npm_intent.output_policy.protected_root_context_complete

    @property
    def included_roi_ids(self) -> tuple[str, ...]:
        if self.rwd is not None:
            return self.rwd.production_intent.roi_scope.included_roi_ids
        return self.npm_intent.selected_roi_ids

    @property
    def per_roi_correction_strategy_map(self) -> tuple:
        """Both formats resolve to the identical shared
        ``GuidedProductionPerRoiStrategy`` tuple type, just under different
        field paths (RWD: ``intent.correction.per_roi_production_strategy_map``;
        NPM: ``intent.per_roi_correction_strategy_map``, flat). This is the
        one correction value genuinely uniform across formats -- unlike a
        "correction-map version" (RWD-only legacy-era concept NPM never
        had), which is intentionally not exposed here."""
        if self.rwd is not None:
            return self.rwd.production_intent.correction.per_roi_production_strategy_map
        return self.npm_intent.per_roi_correction_strategy_map

    @property
    def feature_event(self):
        if self.rwd is not None:
            return self.rwd.production_intent.feature_event
        return self.npm_intent.feature_event

    # -- Already-verified stored identities, read directly (not
    # recomputed here) -- callers that need a *fresh* recompute-and-cross-
    # check must go through guided_startup_transaction._gate_issue's
    # per-format authority helpers, which is where that verification
    # belongs. These accessors exist so plan_guided_startup_transaction's
    # single shared status/provenance/identity-envelope construction can
    # read one canonical value per concept regardless of format.

    @property
    def canonical_authorization_identity(self) -> str:
        if self.rwd is not None:
            return self.rwd.canonical_authorization_identity
        return self.npm_authority.canonical_authority_identity

    @property
    def production_intent_identity(self) -> str:
        if self.rwd is not None:
            return self.rwd.production_intent_identity
        return self.npm_intent.canonical_intent_identity

    @property
    def candidate_preflight_identity(self) -> str:
        if self.rwd is not None:
            return self.rwd.candidate_preflight_identity
        return self.npm_intent.source_snapshot_identity

    @property
    def roi_preflight_identity(self) -> str:
        if self.rwd is not None:
            return self.rwd.roi_preflight_identity
        return self.npm_authority.roi_authority.canonical_roi_authority_identity

    @property
    def fresh_request_identity(self) -> str:
        if self.rwd is not None:
            return self.rwd.fresh_request_identity
        return self.npm_intent.source_request_identity

    @property
    def application_build_identity(self) -> str:
        if self.rwd is not None:
            return self.rwd.application_build_identity
        return self.npm_intent.application_build_identity.canonical_identity

    @property
    def mapping_contract_version(self) -> str:
        if self.rwd is not None:
            return self.rwd.production_intent.mapping_contract_version
        return self.npm_intent.mapping_contract_version

    @property
    def runner_contract_version(self) -> str:
        if self.rwd is not None:
            return self.rwd.production_intent.runner_contract_version
        return self.npm_intent.runner_contract_version

    def verify_self_consistent(self) -> None:
        """Re-verify the wrapped authority's own existing identity chain.
        Uses only each format's existing identity/verification functions --
        no new identity is computed here."""
        if self.rwd is not None:
            if (
                compute_guided_run_authorization_identity(self.rwd)
                != self.rwd.canonical_authorization_identity
            ):
                raise ValueError("rwd_authorization_identity_mismatch")
            return
        if (
            compute_guided_npm_production_execution_intent_identity(self.npm_intent)
            != self.npm_intent.canonical_intent_identity
        ):
            raise ValueError("npm_intent_identity_mismatch")
        verify_guided_npm_execution_authority(self.npm_authority)
        if (
            compute_guided_npm_execution_authority_identity(self.npm_authority)
            != self.npm_authority.canonical_authority_identity
        ):
            raise ValueError("npm_authority_identity_mismatch")
        if (
            self.npm_authority.source_production_intent_identity
            != self.npm_intent.canonical_intent_identity
        ):
            raise ValueError("npm_authority_intent_identity_mismatch")


def _npm_parser_sampling(recording_policy) -> dict[str, Any]:
    try:
        content = json.loads(recording_policy.parser_policy_content_json)
        return content["sampling"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("npm_startup_bridge_parser_content_invalid") from exc


def _npm_config_payload_values(
    authority: GuidedNpmExecutionAuthority,
) -> tuple[GuidedConfigFieldValue, ...]:
    """Mirror guided_npm_authorized_adapter._config_and_audit's exact,
    already-proven field-sourcing rules -- the same mapping the bespoke
    worker itself uses -- but sourced from the accepted authority instead of
    a later-stage worker runtime projection. Real NPM parser values are used
    throughout; nothing is left at an RWD-context placeholder."""
    recording_policy = authority.recording_policy
    sampling = _npm_parser_sampling(recording_policy)
    defaults = Config()
    values = asdict(defaults)
    frozen: dict[str, Any] = {
        "acquisition_mode": "intermittent",
        "target_fs_hz": recording_policy.target_fs_hz,
        "chunk_duration_sec": recording_policy.configured_session_duration_sec,
        "allow_partial_final_chunk": bool(sampling.get("allow_partial_final_chunk", False)),
        "adapter_value_nan_policy": recording_policy.roi_value_nan_policy,
        "npm_time_axis": recording_policy.time_axis_mode,
        "npm_system_ts_col": sampling.get("system_timestamp_column", ""),
        "npm_computer_ts_col": sampling.get("computer_timestamp_column", ""),
        "npm_led_col": recording_policy.led_state_column,
        "npm_region_prefix": recording_policy.roi_prefix,
        "npm_region_suffix": recording_policy.roi_suffix,
        "timestamp_cv_max": sampling.get("timestamp_cv_max", defaults.timestamp_cv_max),
    }
    for name, value in _typed_values(
        authority.correction_authority.correction_parameter_values
    ).items():
        frozen[_CORRECTION_CONFIG_NAMES.get(name, name)] = value
    frozen.update(_typed_values(authority.feature_authority.effective_values))
    known = {item.name for item in fields(Config)}
    unknown = set(frozen) - known
    if unknown:
        raise ValueError("npm_startup_bridge_config_field_unmapped")
    values.update(frozen)
    # Construct every Config field explicitly, exactly like the bespoke
    # worker's own adapter does, so this proves every value is a valid,
    # self-consistent Config -- not merely a raw dict.
    config = Config(**values)
    return tuple(
        GuidedConfigFieldValue(item.name, getattr(config, item.name))
        for item in fields(Config)
    )


def compile_npm_generic_execution_payloads(
    intent: GuidedNpmProductionExecutionIntent,
    authority: GuidedNpmExecutionAuthority,
    *,
    startup_mapping_contract: GuidedExecutionStartupMappingContract,
) -> GuidedExecutionPayloadDerivationResult:
    """Compile NPM's already-accepted, immutable intent/authority into the
    same three generic startup payload types RWD's
    derive_guided_execution_payloads already produces. Performs no
    filesystem access, no allocation, and invokes nothing. Mirrors
    derive_guided_execution_payloads' own non-runnable terminal shape
    exactly, so the shared finalizer treats both results identically."""

    def _unresolved(
        category: str, message: str, detail_code: str = ""
    ) -> GuidedExecutionPayloadDerivationResult:
        issue = GuidedExecutionPayloadIssue(
            category=category,
            section="guided_npm_startup_bridge",
            message=message,
            detail_code=detail_code,
        )
        return GuidedExecutionPayloadDerivationResult(
            status="refused", ok=False, runnable=False,
            config_payload=None, candidate_manifest_payload=None, runner_request=None,
            provenance_seed=None, config_payload_identity=None,
            candidate_manifest_payload_identity=None, runner_request_identity=None,
            provenance_seed_identity=None, limiting_issues=(), blocking_issues=(issue,),
        )

    try:
        if not isinstance(intent, GuidedNpmProductionExecutionIntent):
            return _unresolved("payload_request_invalid", "Invalid NPM production intent type.")
        if type(authority) is not GuidedNpmExecutionAuthority:
            return _unresolved("payload_request_invalid", "Invalid NPM execution authority type.")
        try:
            intent_identity = compute_guided_npm_production_execution_intent_identity(intent)
        except ValueError as exc:
            return _unresolved("production_intent_identity_mismatch", "NPM production intent is invalid.", str(exc))
        if intent_identity != intent.canonical_intent_identity:
            return _unresolved("production_intent_identity_mismatch", "NPM production intent identity mismatch.")
        try:
            verify_guided_npm_execution_authority(authority)
        except ValueError as exc:
            return _unresolved("authorization_identity_mismatch", "NPM execution authority is invalid.", str(exc))
        if compute_guided_npm_execution_authority_identity(authority) != authority.canonical_authority_identity:
            return _unresolved("authorization_identity_mismatch", "NPM execution authority identity mismatch.")
        if authority.source_production_intent_identity != intent.canonical_intent_identity:
            return _unresolved("production_intent_identity_mismatch", "NPM authority does not match the accepted intent.")
        if intent.execution_mode != "both":
            # Guided Mode exposes no phasic-versus-tonic choice: an
            # authentic Guided NPM run always requires "both". This
            # restriction belongs only to the Guided startup bridge --
            # ordinary/backend and Full Control support for phasic-only or
            # tonic-only NPM execution is untouched elsewhere.
            return _unresolved("config_field_unsupported", "Guided NPM execution mode must be both.")
        if intent.output_policy.overwrite is not False:
            return _unresolved("config_field_unsupported", "output_overwrite must be False.")

        try:
            config_values = _npm_config_payload_values(authority)
        except ValueError as exc:
            return _unresolved("config_field_unsupported", "NPM config values could not be derived.", str(exc))
        provisional_config = GuidedExecutionConfigPayload(
            config_schema_name="photometry_pipeline_config",
            config_mapping_contract_version=startup_mapping_contract.config_mapping_contract_version,
            values=config_values,
            canonical_config_payload_identity="0" * 64,
        )
        config_id = compute_guided_execution_config_payload_identity(provisional_config)
        config_payload = replace_config_payload_identity(provisional_config, config_id)

        manifest_files = tuple(
            GuidedRunnerCandidateManifestEntry(
                canonical_relative_path=session.canonical_relative_path,
                size_bytes=session.size_bytes,
                sha256_content_digest=session.sha256_content_digest,
            )
            for session in authority.sessions
        )
        provisional_manifest = GuidedRunnerCandidateManifestPayload(
            manifest_schema_name="guided_runner_candidate_manifest",
            manifest_schema_version=startup_mapping_contract.candidate_manifest_schema_version,
            candidate_consumption_contract_version=startup_mapping_contract.candidate_consumption_contract_version,
            source_root_canonical=intent.source_root_canonical,
            source_candidate_set_digest=intent.source_snapshot_set_identity,
            source_candidate_content_digest=intent.source_snapshot_content_identity,
            candidate_files=manifest_files,
            parser_contract_digest=intent.parser_policy_identity,
            discovered_roi_ids=authority.roi_authority.complete_canonical_roi_ids,
            included_roi_ids=authority.roi_authority.selected_canonical_roi_ids,
            excluded_roi_ids=authority.roi_authority.excluded_canonical_roi_ids,
            strict_roi_inventory_digest=authority.roi_authority.canonical_roi_authority_identity,
            candidate_preflight_identity=intent.source_snapshot_identity,
            roi_preflight_identity=authority.roi_authority.canonical_roi_authority_identity,
            canonical_candidate_manifest_payload_identity="0" * 64,
        )
        manifest_id = compute_guided_runner_candidate_manifest_payload_identity(provisional_manifest)
        candidate_manifest_payload = replace_manifest_payload_identity(provisional_manifest, manifest_id)

        provisional_seed = GuidedStartupProvenanceSeed(
            provenance_schema_name="guided_startup_provenance_seed",
            provenance_schema_version=startup_mapping_contract.startup_provenance_schema_version,
            startup_mapping_contract_version=startup_mapping_contract.contract_version,
            validation_request_identity=intent.source_request_identity,
            authorization_identity=authority.canonical_authority_identity,
            production_intent_identity=intent.canonical_intent_identity,
            application_build_identity=intent.application_build_identity.canonical_identity,
            production_mapping_contract_version=intent.mapping_contract_version,
            runner_contract_version=intent.runner_contract_version,
            candidate_preflight_identity=intent.source_snapshot_identity,
            roi_preflight_identity=authority.roi_authority.canonical_roi_authority_identity,
            config_payload_identity=config_id,
            candidate_manifest_payload_identity=manifest_id,
            runner_request_identity=None,
            runnable=False,
            canonical_provenance_seed_identity="0" * 64,
        )
        seed_id = compute_guided_startup_provenance_seed_identity(provisional_seed)
        provenance_seed = replace_provenance_seed_identity(provisional_seed, seed_id)

        limiting_issue = GuidedExecutionPayloadIssue(
            category="startup_transaction_unavailable",
            section="guided_npm_startup_bridge",
            message=(
                "Exact NPM runner manifest/ROI consumption is available, but "
                "no startup transaction exists to serialize payloads, "
                "allocate output, build an invocation, or launch the runner."
            ),
        )
        return GuidedExecutionPayloadDerivationResult(
            status=GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE,
            ok=True, runnable=False,
            config_payload=config_payload, candidate_manifest_payload=candidate_manifest_payload,
            runner_request=None, provenance_seed=provenance_seed,
            config_payload_identity=config_id, candidate_manifest_payload_identity=manifest_id,
            runner_request_identity=None, provenance_seed_identity=seed_id,
            limiting_issues=(limiting_issue,), blocking_issues=(),
        )
    except Exception:
        return _unresolved("payload_internal_error", "NPM execution payload derivation failed.", "payload_derivation_exception")
