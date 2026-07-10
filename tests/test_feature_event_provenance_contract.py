"""4J16k39: the confirmed Guided feature settings must be the production base,
and every consumer must use the settings actually consumed per ROI.

Covers:
- 39a: confirmed Step 5 settings -> production base Config; sparse Custom ROI
  overrides inherit from that confirmed base; fail-closed validation.
- 39b: per-ROI consumed-settings provenance, its configuration identity, and
  the current-vs-legacy classification that governs detector-aware day plots.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.feature_event_config import FEATURE_EVENT_CONFIG_FIELDS
from photometry_pipeline.feature_event_provenance import (
    FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION,
    FEATURE_EVENT_PROVENANCE_FILENAME,
    FEATURE_EVENT_PROVENANCE_SCHEMA_V3,
    FeatureEventProvenanceError,
    PROVENANCE_MODE_CURRENT,
    PROVENANCE_MODE_LEGACY,
    PROVENANCE_MODE_UNKNOWN,
    build_feature_event_provenance_payload,
    classify_provenance_contract,
    compute_feature_config_digest,
    feature_fields_from_config,
    load_feature_event_provenance,
    resolve_roi_effective_fields,
    run_uses_current_provenance_contract,
    verify_global_default_identity,
)


# ---------------------------------------------------------------------------
# 39a: confirmed global settings become the production base
# ---------------------------------------------------------------------------


def _confirmed_intent(**feature_overrides):
    """Build a mapped production intent whose confirmed Step 5 profile carries a
    complete feature configuration, optionally with non-default values."""
    from dataclasses import replace as _r

    import photometry_pipeline.guided_backend_validation_request as vr
    import photometry_pipeline.guided_backend_validator as validator
    import photometry_pipeline.guided_production_mapping as mapping
    from tests.test_guided_backend_validator import (
        CONFIRMED_FEATURE_PROFILE_VALUES,
        _contract as _validator_contract,
        _request as _valid_request,
        _typed,
    )

    values = dict(CONFIRMED_FEATURE_PROFILE_VALUES)
    values.update(feature_overrides)

    request = _valid_request()
    request = _r(
        request,
        acquisition_dataset=_r(
            request.acquisition_dataset,
            semantic_values=request.acquisition_dataset.semantic_values
            + (_typed("target_fs_hz", 40.0),),
        ),
        feature_event=_r(
            request.feature_event,
            effective_values=tuple(_typed(k, v) for k, v in values.items()),
        ),
    )
    identity = vr.compute_guided_backend_validation_request_identity(request)
    validated = validator.validate_guided_backend_validation_request(
        request,
        canonical_request_identity=identity,
        validator_contract=_validator_contract(),
    )
    assert validated.accepted, validated.blocking_issues
    from tests.test_guided_execution_payloads import _build_app_identity

    result = mapping.map_guided_validation_request_to_execution_intent(
        request,
        canonical_request_identity=identity,
        application_build_identity=_build_app_identity(),
        mapping_contract=mapping.build_guided_production_mapping_contract(),
    )
    assert isinstance(result, mapping.GuidedProductionMappingSuccess), result
    return result.intent, values


def test_confirmed_nondefault_global_setting_reaches_production_base():
    """A confirmed non-default peak_threshold_k=7.25 must be serialized into the
    production configuration, not the baked default 2.5 (C1)."""
    from photometry_pipeline.guided_execution_payloads import (
        resolve_confirmed_feature_config_fields,
    )

    intent, expected = _confirmed_intent(peak_threshold_k=7.25)
    fields, reason = resolve_confirmed_feature_config_fields(intent)

    assert reason == ""
    assert fields is not None
    assert set(fields) == set(FEATURE_EVENT_CONFIG_FIELDS)
    assert fields["peak_threshold_k"] == 7.25
    assert fields["peak_threshold_k"] != Config().peak_threshold_k

    # The resulting base Config carries the confirmed value.
    base = replace(Config(), **fields)
    assert base.peak_threshold_k == 7.25


def test_baked_defaults_no_longer_carry_feature_fields():
    """GUIDED_CONFIG_DEFAULT_OVERRIDES must not be able to override confirmed
    feature settings -- that was the C1 root cause."""
    import photometry_pipeline.guided_execution_payloads as payloads

    confirmed = {
        k for k, v in payloads.GUIDED_CONFIG_FIELD_DISPOSITIONS.items()
        if v == payloads.CONFIG_DISPOSITION_CONFIRMED_FEATURE
    }
    assert confirmed == set(FEATURE_EVENT_CONFIG_FIELDS)
    assert not (confirmed & set(payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES))


def test_unconfirmed_or_stale_feature_settings_fail_closed():
    from dataclasses import replace as _r

    from photometry_pipeline.guided_execution_payloads import (
        resolve_confirmed_feature_config_fields,
    )

    intent, _ = _confirmed_intent()

    not_applied = _r(intent, feature_event=_r(intent.feature_event, explicitly_applied=False))
    fields, reason = resolve_confirmed_feature_config_fields(not_applied)
    assert fields is None and "never confirmed" in reason

    stale = _r(intent, feature_event=_r(intent.feature_event, current=False))
    fields, reason = resolve_confirmed_feature_config_fields(stale)
    assert fields is None and "stale" in reason


def test_incomplete_or_nonfinite_confirmed_settings_fail_closed():
    from dataclasses import replace as _r

    from photometry_pipeline.guided_execution_payloads import (
        resolve_confirmed_feature_config_fields,
    )

    intent, _ = _confirmed_intent()

    # Drop one field -> incomplete.
    trimmed = tuple(
        v for v in intent.feature_event.effective_values
        if v.field_name != "peak_min_width_sec"
    )
    incomplete = _r(intent, feature_event=_r(intent.feature_event, effective_values=trimmed))
    fields, reason = resolve_confirmed_feature_config_fields(incomplete)
    assert fields is None and "incomplete" in reason
    assert "peak_min_width_sec" in reason

    # NaN in an active numeric field -> rejected before execution.
    nan_values = tuple(
        _r(v, value=float("nan")) if v.field_name == "peak_threshold_k" else v
        for v in intent.feature_event.effective_values
    )
    nan_intent = _r(intent, feature_event=_r(intent.feature_event, effective_values=nan_values))
    fields, reason = resolve_confirmed_feature_config_fields(nan_intent)
    assert fields is None and "not finite" in reason


def test_dormant_inactive_threshold_field_does_not_block_the_base():
    """A complete profile carries peak_threshold_abs=0.0 while using mean_std.
    That dormant value must serialize into Config without failing validation."""
    from photometry_pipeline.guided_execution_payloads import (
        resolve_confirmed_feature_config_fields,
    )

    intent, _ = _confirmed_intent(peak_threshold_method="mean_std", peak_threshold_abs=0.0)
    fields, reason = resolve_confirmed_feature_config_fields(intent)
    assert reason == ""
    assert fields["peak_threshold_abs"] == 0.0

    # But an ACTIVE absolute threshold of 0.0 is still rejected.
    bad, _ = _confirmed_intent(peak_threshold_method="absolute", peak_threshold_abs=0.0)
    fields, reason = resolve_confirmed_feature_config_fields(bad)
    assert fields is None and "invalid" in reason


# ---------------------------------------------------------------------------
# 39a: sparse Custom overrides inherit from the CONFIRMED base
# ---------------------------------------------------------------------------


def _write_guided_artifact(run_dir, base_config, overrides, provenance_extra=None):
    """Write the startup artifact analyze_photometry consumes."""
    from photometry_pipeline.guided_startup_transaction import (
        GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME,
    )

    base_fields = feature_fields_from_config(base_config)
    effective = {}
    provenance = {}
    for roi, sparse in overrides.items():
        eff = dict(base_fields)
        eff.update(sparse)
        effective[roi] = eff
        provenance[roi] = {
            "source": "override",
            "feature_event_profile_id": f"custom-{roi}",
            "override_config_fields": dict(sparse),
            "effective_config_fields": eff,
        }
    for roi, entry in (provenance_extra or {}).items():
        provenance[roi] = entry

    payload = {
        "schema_name": "guided_per_roi_feature_config",
        "schema_version": "v1",
        "per_roi_override_config_fields": overrides,
        "per_roi_effective_feature_config_fields_for_overrides": effective,
        "per_roi_feature_provenance": provenance,
    }
    manifest_path = os.path.join(run_dir, "guided_candidate_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("{}")
    with open(
        os.path.join(run_dir, GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME), "w", encoding="utf-8"
    ) as f:
        json.dump(payload, f)
    return manifest_path


def test_sparse_custom_roi_inherits_from_confirmed_base(tmp_path):
    """CH1 changes only peak_threshold_k; it must inherit the CONFIRMED
    peak_min_distance_sec=2.5, not the baked execution default of 1.0."""
    from analyze_photometry import load_guided_per_roi_feature_settings

    confirmed_base = replace(Config(), peak_min_distance_sec=2.5, peak_threshold_k=2.5)
    assert Config().peak_min_distance_sec == 1.0  # the baked default we must NOT inherit

    manifest = _write_guided_artifact(
        str(tmp_path),
        confirmed_base,
        overrides={"CH1": {"peak_threshold_k": 7.25}},
        provenance_extra={
            "CH2": {
                "source": "default",
                "feature_event_profile_id": "profile-1",
                "override_config_fields": {},
                "effective_config_fields": feature_fields_from_config(confirmed_base),
            }
        },
    )

    per_roi_config, provenance = load_guided_per_roi_feature_settings(
        manifest, confirmed_base
    )

    assert set(per_roi_config) == {"CH1"}
    ch1 = per_roi_config["CH1"]
    assert ch1.peak_threshold_k == 7.25            # the user's change
    assert ch1.peak_min_distance_sec == 2.5        # inherited from CONFIRMED base
    # CH2 (Default) is absent -> Pipeline uses the confirmed base directly.
    assert "CH2" not in per_roi_config
    assert confirmed_base.peak_min_distance_sec == 2.5

    # The stored user override stays sparse.
    assert provenance["CH1"]["override_config_fields"] == {"peak_threshold_k": 7.25}


def test_unknown_override_field_fails_closed(tmp_path):
    from analyze_photometry import (
        GuidedFeatureSettingsError,
        load_guided_per_roi_feature_settings,
    )
    from photometry_pipeline.guided_startup_transaction import (
        GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME,
    )

    base = Config()
    manifest = _write_guided_artifact(str(tmp_path), base, overrides={"CH1": {"peak_threshold_k": 3.0}})
    path = os.path.join(str(tmp_path), GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME)
    payload = json.loads(open(path, encoding="utf-8").read())
    payload["per_roi_override_config_fields"]["CH1"]["not_a_real_field"] = 1
    payload["per_roi_effective_feature_config_fields_for_overrides"]["CH1"][
        "not_a_real_field"
    ] = 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    with pytest.raises(GuidedFeatureSettingsError, match="unknown"):
        load_guided_per_roi_feature_settings(manifest, base)


def test_incomplete_effective_configuration_fails_closed(tmp_path):
    from analyze_photometry import (
        GuidedFeatureSettingsError,
        load_guided_per_roi_feature_settings,
    )
    from photometry_pipeline.guided_startup_transaction import (
        GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME,
    )

    base = Config()
    manifest = _write_guided_artifact(str(tmp_path), base, overrides={"CH1": {"peak_threshold_k": 3.0}})
    path = os.path.join(str(tmp_path), GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME)
    payload = json.loads(open(path, encoding="utf-8").read())
    payload["per_roi_effective_feature_config_fields_for_overrides"]["CH1"].pop(
        "peak_min_width_sec"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    with pytest.raises(GuidedFeatureSettingsError, match="incomplete"):
        load_guided_per_roi_feature_settings(manifest, base)


def test_effective_config_not_derived_from_confirmed_base_fails_closed(tmp_path):
    """If a ROI's effective settings were resolved against a different base than
    the one Pipeline will use, refuse rather than analyze inconsistently."""
    from analyze_photometry import (
        GuidedFeatureSettingsError,
        load_guided_per_roi_feature_settings,
    )
    from photometry_pipeline.guided_startup_transaction import (
        GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME,
    )

    base = replace(Config(), peak_min_distance_sec=2.5)
    manifest = _write_guided_artifact(str(tmp_path), base, overrides={"CH1": {"peak_threshold_k": 3.0}})
    path = os.path.join(str(tmp_path), GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME)
    payload = json.loads(open(path, encoding="utf-8").read())
    # Pretend CH1 was resolved against the baked default distance (1.0).
    payload["per_roi_effective_feature_config_fields_for_overrides"]["CH1"][
        "peak_min_distance_sec"
    ] = 1.0
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    with pytest.raises(GuidedFeatureSettingsError, match="not derived from"):
        load_guided_per_roi_feature_settings(manifest, base)


def test_default_roi_disagreeing_with_confirmed_base_fails_closed(tmp_path):
    from analyze_photometry import (
        GuidedFeatureSettingsError,
        load_guided_per_roi_feature_settings,
    )
    from photometry_pipeline.guided_startup_transaction import (
        GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME,
    )

    base = Config()
    manifest = _write_guided_artifact(
        str(tmp_path), base, overrides={"CH1": {"peak_threshold_k": 3.0}},
        provenance_extra={
            "CH2": {
                "source": "default",
                "feature_event_profile_id": "p",
                "override_config_fields": {},
                "effective_config_fields": {
                    **feature_fields_from_config(base),
                    "peak_threshold_k": 99.0,  # disagrees with the base
                },
            }
        },
    )
    with pytest.raises(GuidedFeatureSettingsError, match="disagree"):
        load_guided_per_roi_feature_settings(manifest, base)


# ---------------------------------------------------------------------------
# 39b: provenance payload + configuration identity
# ---------------------------------------------------------------------------


def test_digest_is_order_independent_and_feature_only():
    base = Config()
    fields = feature_fields_from_config(base)
    shuffled = dict(reversed(list(fields.items())))
    assert compute_feature_config_digest(fields) == compute_feature_config_digest(shuffled)

    # A non-feature Config field must not change the digest.
    other = replace(base, lowpass_hz=base.lowpass_hz + 1.0)
    assert compute_feature_config_digest(feature_fields_from_config(other)) == (
        compute_feature_config_digest(fields)
    )

    # A feature field must change it.
    changed = replace(base, peak_threshold_k=base.peak_threshold_k + 1.0)
    assert compute_feature_config_digest(feature_fields_from_config(changed)) != (
        compute_feature_config_digest(fields)
    )


def test_digest_rejects_incomplete_or_unknown_fields():
    fields = feature_fields_from_config(Config())
    incomplete = dict(fields)
    incomplete.pop("peak_pre_filter")
    with pytest.raises(FeatureEventProvenanceError, match="incomplete"):
        compute_feature_config_digest(incomplete)

    unknown = dict(fields)
    unknown["bogus"] = 1
    with pytest.raises(FeatureEventProvenanceError, match="unknown"):
        compute_feature_config_digest(unknown)


def test_default_only_payload_has_one_complete_entry_per_roi():
    base = Config()
    payload = build_feature_event_provenance_payload(
        base_config=base, analyzed_rois=["CH2", "CH1"]
    )
    assert payload["schema_version"] == FEATURE_EVENT_PROVENANCE_SCHEMA_V3
    assert [e["roi"] for e in payload["rois"]] == ["CH1", "CH2"]  # deterministic order
    for entry in payload["rois"]:
        assert entry["source"] == "default"
        assert entry["override_config_fields"] == {}
        assert set(entry["effective_config_fields"]) == set(FEATURE_EVENT_CONFIG_FIELDS)
        assert entry["effective_config_digest"] == compute_feature_config_digest(
            entry["effective_config_fields"]
        )


def test_custom_payload_records_consumed_config_and_sparse_override():
    base = Config()
    custom = replace(base, peak_threshold_k=9.0)
    payload = build_feature_event_provenance_payload(
        base_config=base,
        analyzed_rois=["CH1", "CH2"],
        per_roi_feature_config={"CH1": custom},
        per_roi_source_details={
            "CH1": {
                "feature_event_profile_id": "custom-ch1",
                "override_config_fields": {"peak_threshold_k": 9.0},
            }
        },
    )
    by_roi = {e["roi"]: e for e in payload["rois"]}
    assert by_roi["CH1"]["source"] == "override"
    assert by_roi["CH1"]["effective_config_fields"]["peak_threshold_k"] == 9.0
    assert by_roi["CH1"]["override_config_fields"] == {"peak_threshold_k": 9.0}
    assert by_roi["CH2"]["source"] == "default"
    assert by_roi["CH2"]["effective_config_fields"]["peak_threshold_k"] == base.peak_threshold_k
    assert by_roi["CH1"]["effective_config_digest"] != by_roi["CH2"]["effective_config_digest"]


def _write_provenance(tmp_path, payload):
    feats = tmp_path / "features"
    feats.mkdir(parents=True, exist_ok=True)
    path = feats / FEATURE_EVENT_PROVENANCE_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_resolve_roi_effective_fields_fail_closed(tmp_path):
    base = Config()
    payload = build_feature_event_provenance_payload(base_config=base, analyzed_rois=["CH1"])
    path = _write_provenance(tmp_path, payload)
    loaded = load_feature_event_provenance(path)

    assert resolve_roi_effective_fields(loaded, "CH1")["peak_threshold_k"] == base.peak_threshold_k

    # Missing ROI entry.
    with pytest.raises(FeatureEventProvenanceError, match="no entry for ROI"):
        resolve_roi_effective_fields(loaded, "CH2")

    # Digest mismatch (settings tampered after the digest was computed).
    tampered = json.loads(json.dumps(payload))
    tampered["rois"][0]["effective_config_fields"]["peak_threshold_k"] = 12345.0
    with pytest.raises(FeatureEventProvenanceError, match="digest does not match"):
        resolve_roi_effective_fields(tampered, "CH1")

    # Incomplete entry.
    incomplete = json.loads(json.dumps(payload))
    incomplete["rois"][0]["effective_config_fields"].pop("peak_pre_filter")
    with pytest.raises(FeatureEventProvenanceError, match="incomplete"):
        resolve_roi_effective_fields(incomplete, "CH1")


def test_missing_provenance_file_fails_closed(tmp_path):
    with pytest.raises(FeatureEventProvenanceError, match="missing"):
        load_feature_event_provenance(str(tmp_path / "nope.json"))


# ---------------------------------------------------------------------------
# 39b: current-vs-legacy classification (absence is NOT Default-only)
# ---------------------------------------------------------------------------


def _write_run_report(phasic_out, *, current_contract, contract_version=None, legacy=False):
    """Write a run report.

    current_contract -> declares the exact supported contract version.
    legacy           -> POSITIVELY identifies a pre-contract run: a well-formed
                        historical report (analytical_contract + configuration)
                        with no feature_event_provenance section at all.
    neither          -> an ambiguous report that proves nothing (unknown).
    """
    os.makedirs(phasic_out, exist_ok=True)
    report = {"run_context": {"run_type": "full"}}
    if current_contract:
        report["feature_event_provenance"] = {
            "contract_version": contract_version or FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION,
            "schema_version": FEATURE_EVENT_PROVENANCE_SCHEMA_V3,
        }
    if legacy:
        report["analytical_contract"] = {"strict_mode_guarantees": []}
        report["configuration"] = {"target_fs_hz": 40.0}
    with open(os.path.join(phasic_out, "run_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f)


def test_absence_alone_does_not_classify_a_current_run_as_legacy(tmp_path):
    """A current run whose provenance file was deleted must NOT be read as
    legacy/Default-only: the contract-version signal still says current."""
    phasic_out = str(tmp_path / "phasic_out")
    _write_run_report(phasic_out, current_contract=True)

    mode, path, _reason = classify_provenance_contract(phasic_out)
    assert mode == PROVENANCE_MODE_CURRENT
    assert path and path.endswith(FEATURE_EVENT_PROVENANCE_FILENAME)
    assert not os.path.exists(path)

    with pytest.raises(FeatureEventProvenanceError):
        load_feature_event_provenance(path)


def test_missing_run_report_is_unknown_not_legacy(tmp_path):
    """A damaged current run must never be silently downgraded to legacy."""
    phasic_out = str(tmp_path / "phasic_out")
    os.makedirs(phasic_out, exist_ok=True)
    mode, path, reason = classify_provenance_contract(phasic_out)
    assert mode == PROVENANCE_MODE_UNKNOWN
    assert path is None
    assert "missing" in reason


def test_malformed_run_report_is_unknown_not_legacy(tmp_path):
    phasic_out = str(tmp_path / "phasic_out")
    os.makedirs(phasic_out, exist_ok=True)
    with open(os.path.join(phasic_out, "run_report.json"), "w", encoding="utf-8") as f:
        f.write("{not json")
    mode, path, reason = classify_provenance_contract(phasic_out)
    assert mode == PROVENANCE_MODE_UNKNOWN
    assert path is None
    assert "malformed" in reason


def test_ambiguous_run_report_without_legacy_signal_is_unknown(tmp_path):
    """A report with neither the contract signal nor the historical schema
    signals proves nothing; absence alone is not an explicit legacy signal."""
    phasic_out = str(tmp_path / "phasic_out")
    _write_run_report(phasic_out, current_contract=False, legacy=False)
    mode, path, reason = classify_provenance_contract(phasic_out)
    assert mode == PROVENANCE_MODE_UNKNOWN
    assert path is None
    assert "positively" in reason


def test_explicit_legacy_fixture_is_recognized(tmp_path):
    """A genuinely legacy run: well-formed historical report, no provenance
    section. This is the ONLY shape permitted to use the global configuration."""
    phasic_out = str(tmp_path / "phasic_out")
    _write_run_report(phasic_out, current_contract=False, legacy=True)
    mode, path, reason = classify_provenance_contract(phasic_out)
    assert mode == PROVENANCE_MODE_LEGACY
    assert path is None
    assert reason == ""


def test_unknown_contract_version_is_unknown_not_legacy(tmp_path):
    phasic_out = str(tmp_path / "phasic_out")
    _write_run_report(
        phasic_out, current_contract=True, contract_version="feature_event_provenance.v99"
    )
    mode, path, reason = classify_provenance_contract(phasic_out)
    assert mode == PROVENANCE_MODE_UNKNOWN
    assert path is None
    assert "unsupported" in reason


def test_empty_contract_version_is_not_a_valid_current_contract():
    assert run_uses_current_provenance_contract(
        {"feature_event_provenance": {"contract_version": ""}}
    ) is False
    assert run_uses_current_provenance_contract(
        {"feature_event_provenance": {"contract_version": "feature_event_provenance.v99"}}
    ) is False
    assert run_uses_current_provenance_contract(
        {"feature_event_provenance": {
            "contract_version": FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION
        }}
    ) is True
    assert run_uses_current_provenance_contract({"run_context": {}}) is False
    assert run_uses_current_provenance_contract(None) is False


# ---------------------------------------------------------------------------
# 39b follow-up: global Default identity is bound to config_used.yaml
# ---------------------------------------------------------------------------


def test_global_identity_verifies_against_matching_config_used():
    base = Config()
    payload = build_feature_event_provenance_payload(
        base_config=base, analyzed_rois=["CH1", "CH2"]
    )
    digest = verify_global_default_identity(payload, base)
    assert digest == payload["global_default_config_digest"]


def test_config_used_differing_from_recorded_global_default_fails():
    base = Config()
    payload = build_feature_event_provenance_payload(base_config=base, analyzed_rois=["CH1"])
    drifted = replace(base, peak_threshold_k=base.peak_threshold_k + 1.0)
    with pytest.raises(FeatureEventProvenanceError, match="config_used.yaml does not match"):
        verify_global_default_identity(payload, drifted)


def test_tampered_global_default_digest_fails():
    base = Config()
    payload = build_feature_event_provenance_payload(base_config=base, analyzed_rois=["CH1"])
    payload["global_default_config_digest"] = "0" * 64
    with pytest.raises(
        FeatureEventProvenanceError, match="does not match its recorded digest"
    ):
        verify_global_default_identity(payload, base)


def test_incomplete_recorded_global_default_fields_fail():
    base = Config()
    payload = build_feature_event_provenance_payload(base_config=base, analyzed_rois=["CH1"])
    payload["global_default_config_fields"].pop("peak_pre_filter")
    with pytest.raises(FeatureEventProvenanceError, match="incomplete"):
        verify_global_default_identity(payload, base)


def test_default_roi_differing_from_global_default_fails():
    base = Config()
    payload = build_feature_event_provenance_payload(base_config=base, analyzed_rois=["CH1"])
    entry = payload["rois"][0]
    assert entry["source"] == "default"
    drifted_fields = dict(entry["effective_config_fields"])
    drifted_fields["peak_threshold_k"] = drifted_fields["peak_threshold_k"] + 1.0
    entry["effective_config_fields"] = drifted_fields
    entry["effective_config_digest"] = compute_feature_config_digest(drifted_fields)

    with pytest.raises(FeatureEventProvenanceError, match="Default ROI"):
        verify_global_default_identity(payload, base)


def test_custom_roi_legitimately_differs_and_still_verifies():
    base = Config()
    custom = replace(base, peak_threshold_k=9.0)
    payload = build_feature_event_provenance_payload(
        base_config=base,
        analyzed_rois=["CH1", "CH2"],
        per_roi_feature_config={"CH1": custom},
    )
    digest = verify_global_default_identity(payload, base)

    by_roi = {e["roi"]: e for e in payload["rois"]}
    assert by_roi["CH2"]["effective_config_digest"] == digest      # Default == global
    assert by_roi["CH1"]["effective_config_digest"] != digest      # Custom may differ
    assert compute_feature_config_digest(
        by_roi["CH1"]["effective_config_fields"]
    ) == by_roi["CH1"]["effective_config_digest"]


# ---------------------------------------------------------------------------
# 39b: wrapper preflight fails closed before any ROI plot is launched
# ---------------------------------------------------------------------------


def _write_config_used(phasic_out, config):
    import yaml

    os.makedirs(phasic_out, exist_ok=True)
    with open(os.path.join(phasic_out, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config.__dict__), f)


def _wrapper_preflight(phasic_out, rois, *, require_current=True):
    from tools.run_full_pipeline_deliverables import (
        _resolve_feature_provenance_for_plots,
    )

    return _resolve_feature_provenance_for_plots(
        phasic_out, rois, require_current=require_current
    )


def _current_run_fixture(tmp_path, *, rois=("CH1", "CH2"), per_roi=None, base=None):
    base = base or Config()
    phasic_out = str(tmp_path / "phasic_out")
    _write_run_report(phasic_out, current_contract=True)
    _write_config_used(phasic_out, base)
    payload = build_feature_event_provenance_payload(
        base_config=base, analyzed_rois=list(rois), per_roi_feature_config=per_roi or {}
    )
    _write_provenance(tmp_path / "phasic_out", payload)
    return phasic_out, payload


def test_wrapper_preflight_accepts_complete_current_run(tmp_path):
    phasic_out, _ = _current_run_fixture(tmp_path)
    mode, path = _wrapper_preflight(phasic_out, ["CH1", "CH2"])
    assert mode == PROVENANCE_MODE_CURRENT
    assert os.path.isfile(path)


def test_wrapper_preflight_fails_when_current_run_lost_its_provenance(tmp_path):
    phasic_out = str(tmp_path / "phasic_out")
    _write_run_report(phasic_out, current_contract=True)
    _write_config_used(phasic_out, Config())
    with pytest.raises(RuntimeError, match="refused"):
        _wrapper_preflight(phasic_out, ["CH1"])


def test_wrapper_preflight_fails_on_missing_run_report(tmp_path):
    """A freshly produced run whose report vanished must fail, not go legacy."""
    phasic_out = str(tmp_path / "phasic_out")
    os.makedirs(phasic_out, exist_ok=True)
    _write_config_used(phasic_out, Config())
    with pytest.raises(RuntimeError, match="positively identify"):
        _wrapper_preflight(phasic_out, ["CH1"])


def test_wrapper_preflight_fails_on_malformed_run_report(tmp_path):
    phasic_out = str(tmp_path / "phasic_out")
    os.makedirs(phasic_out, exist_ok=True)
    _write_config_used(phasic_out, Config())
    with open(os.path.join(phasic_out, "run_report.json"), "w", encoding="utf-8") as f:
        f.write("{truncated")
    with pytest.raises(RuntimeError, match="positively identify"):
        _wrapper_preflight(phasic_out, ["CH1"])


def test_wrapper_preflight_fails_on_unknown_contract_version(tmp_path):
    phasic_out = str(tmp_path / "phasic_out")
    _write_run_report(
        phasic_out, current_contract=True, contract_version="feature_event_provenance.v99"
    )
    _write_config_used(phasic_out, Config())
    with pytest.raises(RuntimeError, match="positively identify"):
        _wrapper_preflight(phasic_out, ["CH1"])


def test_wrapper_preflight_rejects_legacy_run_from_current_build(tmp_path):
    """The wrapper just invoked the current analysis build, so a report that only
    identifies a pre-contract run means something is wrong -- fail closed."""
    phasic_out = str(tmp_path / "phasic_out")
    _write_run_report(phasic_out, current_contract=False, legacy=True)
    _write_config_used(phasic_out, Config())
    with pytest.raises(RuntimeError, match="does not declare the current"):
        _wrapper_preflight(phasic_out, ["CH1"], require_current=True)


def test_wrapper_preflight_fails_when_one_roi_entry_is_missing(tmp_path):
    phasic_out, _ = _current_run_fixture(tmp_path, rois=("CH1",))
    with pytest.raises(RuntimeError, match="refused"):
        _wrapper_preflight(phasic_out, ["CH1", "CH2"])


def test_wrapper_preflight_fails_on_digest_mismatch(tmp_path):
    base = Config()
    phasic_out = str(tmp_path / "phasic_out")
    _write_run_report(phasic_out, current_contract=True)
    _write_config_used(phasic_out, base)
    payload = build_feature_event_provenance_payload(base_config=base, analyzed_rois=["CH1"])
    payload["rois"][0]["effective_config_fields"]["peak_threshold_k"] = 999.0
    _write_provenance(tmp_path / "phasic_out", payload)

    with pytest.raises(RuntimeError, match="refused"):
        _wrapper_preflight(phasic_out, ["CH1"])


def test_wrapper_preflight_fails_when_config_used_disagrees_with_provenance(tmp_path):
    """config_used.yaml must hash to the recorded global Default digest."""
    base = Config()
    phasic_out = str(tmp_path / "phasic_out")
    _write_run_report(phasic_out, current_contract=True)
    payload = build_feature_event_provenance_payload(base_config=base, analyzed_rois=["CH1"])
    _write_provenance(tmp_path / "phasic_out", payload)
    _write_config_used(phasic_out, replace(base, peak_threshold_k=base.peak_threshold_k + 1.0))

    with pytest.raises(RuntimeError, match="config_used.yaml does not match"):
        _wrapper_preflight(phasic_out, ["CH1"])


def test_wrapper_preflight_unknown_never_falls_back_even_when_not_requiring_current(tmp_path):
    """Absence of metadata is not a legacy signal, regardless of require_current."""
    phasic_out = str(tmp_path / "phasic_out")
    os.makedirs(phasic_out, exist_ok=True)
    with pytest.raises(RuntimeError, match="positively identify"):
        _wrapper_preflight(phasic_out, ["CH1"], require_current=False)


def test_wrapper_preflight_explicit_legacy_run_uses_global_config_without_claims(tmp_path):
    """Only a POSITIVELY identified pre-contract run may use global settings, and
    only when the caller intentionally opts out of requiring the current contract."""
    phasic_out = str(tmp_path / "phasic_out")
    _write_run_report(phasic_out, current_contract=False, legacy=True)
    _write_config_used(phasic_out, Config())

    mode, path = _wrapper_preflight(phasic_out, ["CH1", "CH2"], require_current=False)
    assert mode == PROVENANCE_MODE_LEGACY
    assert path is None


# ---------------------------------------------------------------------------
# 39b: day-plot resolver uses the ROI's own settings
# ---------------------------------------------------------------------------


def test_dayplot_resolver_uses_roi_effective_settings(tmp_path):
    from tools.plot_phasic_dayplot_bundle import resolve_roi_feature_config

    base = Config()
    custom = replace(base, peak_threshold_k=9.0)
    payload = build_feature_event_provenance_payload(
        base_config=base,
        analyzed_rois=["CH1", "CH2"],
        per_roi_feature_config={"CH1": custom},
    )
    path = _write_provenance(tmp_path, payload)

    ch1_cfg, ch1_id = resolve_roi_feature_config(base, "CH1", PROVENANCE_MODE_CURRENT, path)
    ch2_cfg, ch2_id = resolve_roi_feature_config(base, "CH2", PROVENANCE_MODE_CURRENT, path)

    # The Custom ROI replays with ITS settings, not the global Defaults.
    assert ch1_cfg.peak_threshold_k == 9.0
    assert ch2_cfg.peak_threshold_k == base.peak_threshold_k
    assert ch1_id["source"] == "override" and ch2_id["source"] == "default"
    assert ch1_id["verified_against_roi_provenance"] is True

    # The identity recorded for the plot matches the analysis provenance entry.
    by_roi = {e["roi"]: e for e in payload["rois"]}
    assert ch1_id["effective_config_digest"] == by_roi["CH1"]["effective_config_digest"]
    assert ch2_id["effective_config_digest"] == by_roi["CH2"]["effective_config_digest"]


def test_dayplot_resolver_legacy_mode_labels_unknown(tmp_path):
    from tools.plot_phasic_dayplot_bundle import resolve_roi_feature_config

    base = Config()
    cfg, identity = resolve_roi_feature_config(base, "CH1", PROVENANCE_MODE_LEGACY, None)
    assert cfg is base
    assert identity["provenance_mode"] == "legacy"
    assert identity["source"] == "unknown_legacy"
    assert identity["verified_against_roi_provenance"] is False


def test_dayplot_resolver_missing_roi_entry_exits(tmp_path):
    from tools.plot_phasic_dayplot_bundle import resolve_roi_feature_config

    base = Config()
    payload = build_feature_event_provenance_payload(base_config=base, analyzed_rois=["CH1"])
    path = _write_provenance(tmp_path, payload)

    with pytest.raises(SystemExit):
        resolve_roi_feature_config(base, "CH2", PROVENANCE_MODE_CURRENT, path)


def test_dayplot_resolver_current_mode_without_path_exits(tmp_path):
    from tools.plot_phasic_dayplot_bundle import resolve_roi_feature_config

    with pytest.raises(SystemExit):
        resolve_roi_feature_config(Config(), "CH1", PROVENANCE_MODE_CURRENT, None)


def test_dayplot_resolver_digest_mismatch_exits(tmp_path):
    from tools.plot_phasic_dayplot_bundle import resolve_roi_feature_config

    base = Config()
    payload = build_feature_event_provenance_payload(base_config=base, analyzed_rois=["CH1"])
    payload["rois"][0]["effective_config_digest"] = "0" * 64
    path = _write_provenance(tmp_path, payload)

    with pytest.raises(SystemExit):
        resolve_roi_feature_config(base, "CH1", PROVENANCE_MODE_CURRENT, path)


# ---------------------------------------------------------------------------
# C2: strict peak-count replay must use the ROI's own settings
# ---------------------------------------------------------------------------


def _synthetic_dff_trace():
    import numpy as np

    rng = np.random.default_rng(7)
    fs = 20.0
    n = 600
    t = np.arange(n) / fs
    trace = 0.02 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 0.004, n)
    for start in (80, 200, 330, 470):
        idx = slice(start, min(start + 25, n))
        trace[idx] += 0.09 * np.hanning(len(range(*idx.indices(n))))
    return t, trace, fs


def test_strict_verification_uses_roi_settings_and_would_fail_on_global(tmp_path):
    """The C2 defect: a Custom ROI's features.csv peak_count was replayed with the
    GLOBAL config, so a correct analysis died with 'Plotting Logic Mismatch'.

    Here the Custom ROI's permissive settings genuinely detect more peaks than the
    global Defaults. Replaying with the ROI's recorded settings reproduces the
    analysis count; replaying with the global settings does not.
    """
    from photometry_pipeline.core.feature_extraction import get_peak_indices_for_trace
    from tools.plot_phasic_dayplot_bundle import (
        resolve_roi_feature_config,
        verify_peak_count_strict,
    )

    t, trace, fs = _synthetic_dff_trace()

    global_cfg = replace(
        Config(),
        peak_threshold_method="mean_std",
        peak_threshold_k=2.5,
        peak_min_prominence_k=2.0,
        peak_min_width_sec=0.3,
    )
    custom_cfg = replace(
        global_cfg,
        peak_threshold_method="percentile",
        peak_threshold_percentile=1.0,
        peak_min_prominence_k=0.0,
        peak_min_width_sec=0.0,
    )

    # What analysis would have written into features.csv for each ROI.
    expected_default = len(get_peak_indices_for_trace(trace, fs, global_cfg))
    expected_custom = len(get_peak_indices_for_trace(trace, fs, custom_cfg))
    assert expected_custom != expected_default, "settings must actually diverge"

    payload = build_feature_event_provenance_payload(
        base_config=global_cfg,
        analyzed_rois=["CH1", "CH2"],
        per_roi_feature_config={"CH1": custom_cfg},
    )
    path = _write_provenance(tmp_path, payload)

    ch1_cfg, _ = resolve_roi_feature_config(global_cfg, "CH1", PROVENANCE_MODE_CURRENT, path)
    ch2_cfg, _ = resolve_roi_feature_config(global_cfg, "CH2", PROVENANCE_MODE_CURRENT, path)

    # Detector-aware replay reproduces each ROI's own analysis count.
    peaks = verify_peak_count_strict(trace, t, fs, ch1_cfg, expected_custom, "CH1", 0, "f")
    assert len(peaks) == expected_custom
    peaks = verify_peak_count_strict(trace, t, fs, ch2_cfg, expected_default, "CH2", 0, "f")
    assert len(peaks) == expected_default

    # Replaying the Custom ROI against the GLOBAL config is exactly the old
    # defect and must hard-fail rather than silently mis-plot.
    with pytest.raises(SystemExit):
        verify_peak_count_strict(trace, t, fs, global_cfg, expected_custom, "CH1", 0, "f")


def test_wrapper_argv_for_current_run_carries_mode_and_expected_path(tmp_path):
    """The real command handed to each per-ROI day-plot process for a current run
    carries --provenance-mode current and --feature-event-provenance <real path>."""
    from tools.run_full_pipeline_deliverables import _dayplot_provenance_args

    phasic_out, _payload = _current_run_fixture(tmp_path)
    mode, path = _wrapper_preflight(phasic_out, ["CH1", "CH2"])
    assert mode == PROVENANCE_MODE_CURRENT

    args = _dayplot_provenance_args(mode, path)
    assert args[0] == "--provenance-mode"
    assert args[1] == "current"
    assert args[2] == "--feature-event-provenance"
    assert args[3] == path
    assert os.path.isfile(args[3])
    expected = os.path.join(phasic_out, "features", FEATURE_EVENT_PROVENANCE_FILENAME)
    assert os.path.normcase(args[3]) == os.path.normcase(expected)


def test_wrapper_argv_for_legacy_run_carries_mode_without_path():
    from tools.run_full_pipeline_deliverables import _dayplot_provenance_args

    assert _dayplot_provenance_args(PROVENANCE_MODE_LEGACY, None) == [
        "--provenance-mode",
        "legacy",
    ]


def test_wrapper_classifies_once_before_the_roi_plot_loop():
    """The child must never independently infer the contract."""
    import inspect

    import tools.run_full_pipeline_deliverables as wrapper

    source = inspect.getsource(wrapper)
    classify = source.index("_resolve_feature_provenance_for_plots(\n")
    roi_loop = source.index("for roi in regions:")
    cmd = source.index("cmd_bundle = [sys.executable, 'tools/plot_phasic_dayplot_bundle.py'")
    assert classify < roi_loop < cmd
    assert "_dayplot_provenance_args(" in source[cmd : cmd + 1200]
