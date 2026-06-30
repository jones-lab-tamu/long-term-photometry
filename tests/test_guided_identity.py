from __future__ import annotations

from dataclasses import replace
import math

import pytest

from photometry_pipeline.guided_identity import (
    CANONICALIZATION_ALGORITHM_VERSION,
    DEFERRED_CANONICAL_IDENTITY_FIELDS,
    DEFERRED_CANONICAL_IDENTITY_STATUS,
    GUIDED_VALIDATION_REQUEST_SCHEMA_VERSION,
    IDENTITY_SCHEMA_NAME,
    IDENTITY_SCHEMA_VERSION,
    GuidedCanonicalValidationIdentityPayload,
    GuidedIdentityError,
    build_canonical_guided_identity_payload_from_request,
    canonicalize_absolute_path,
    compute_canonical_guided_identity,
    encode_canonical_identity_payload,
    encode_canonical_value,
)
from photometry_pipeline.guided_validation_request import (
    GuidedValidationRequest,
    compute_request_identity,
)


def _request(**overrides: object) -> GuidedValidationRequest:
    values: dict[str, object] = {
        "source_path": "C:/Data/Raw",
        "source_format": "rwd",
        "acquisition_mode": "intermittent",
        "sessions_per_hour": 6,
        "session_duration_sec": 120.0,
        "exclude_incomplete_final_rwd_chunk": True,
        "timeline_anchor_mode": "civil",
        "included_roi_ids": ["ROI1", "ROI0"],
        "execution_mode": "phasic",
        "run_profile": "full",
        "traces_only": False,
        "subset_contract_version": "global_dynamic_fit_only.v1",
        "strategy_scope": "global",
        "global_correction_strategy": "dynamic_fit",
        "dynamic_fit_mode": "global_linear_regression",
        "output_base_path": "C:/Data/Output",
        "output_overwrite": False,
        "output_path_role": "output_base",
        "output_creation_timing": "future_execution_start_only",
        "run_directory_strategy": "derive_unique_run_id_under_output_base",
    }
    values.update(overrides)
    return GuidedValidationRequest(**values)


def _payload() -> GuidedCanonicalValidationIdentityPayload:
    return build_canonical_guided_identity_payload_from_request(_request())


def test_canonical_encoding_is_independent_of_dict_insertion_order():
    first = {"outer": {"b": 2, "a": 1}, "flag": True}
    second = {"flag": True, "outer": {"a": 1, "b": 2}}
    assert encode_canonical_value(first) == encode_canonical_value(second)


def test_digest_changes_when_consumed_value_changes():
    payload = _payload()
    changed = replace(payload, session_duration_sec=121.0)
    assert compute_canonical_guided_identity(payload) != compute_canonical_guided_identity(changed)


def test_identity_domain_schema_changes_alter_digest():
    payload = _payload()
    schema_changed = replace(payload, identity_schema_version="v2")
    algorithm_changed = replace(payload, canonicalization_algorithm_version="typed_json_utf8.v2")
    assert compute_canonical_guided_identity(payload) != compute_canonical_guided_identity(schema_changed)
    assert compute_canonical_guided_identity(payload) != compute_canonical_guided_identity(algorithm_changed)


def test_windows_drive_path_normalizes_separators_case_and_segments():
    result = canonicalize_absolute_path(r"C:\Data\.\Raw\Sub\..\ROI")
    assert result.canonical_path == r"c:\data\raw\roi"
    assert result.path_style == "windows_drive"
    assert canonicalize_absolute_path("c:/DATA/raw/roi") == result


def test_backslash_and_forward_slash_unc_paths_canonicalize_identically():
    backslash = canonicalize_absolute_path(
        r"\\Server\Share\Data\.\Raw\..\ROI"
    )
    forward_slash = canonicalize_absolute_path(
        "//Server/Share/Data/./Raw/../ROI"
    )
    assert backslash.canonical_path == r"\\server\share\data\roi"
    assert backslash.path_style == "windows_unc"
    assert forward_slash == backslash


def test_posix_path_preserves_case_and_resolves_segments():
    result = canonicalize_absolute_path("/Data/Raw/./Session/../ROI")
    assert result.canonical_path == "/Data/Raw/ROI"
    assert result.path_style == "posix"
    assert canonicalize_absolute_path("/data/Raw/ROI") != result


@pytest.mark.parametrize(
    "path",
    [
        "relative/path",
        r"relative\path",
        r"C:relative\path",
        r"/posix\mixed",
    ],
)
def test_relative_mixed_and_ambiguous_paths_are_rejected(path: str):
    with pytest.raises(GuidedIdentityError):
        canonicalize_absolute_path(path)


@pytest.mark.parametrize(
    "path",
    [
        r"\\server",
        "\\\\server\\",
        "//server",
        "//server/",
        r"\\\share\data",
        "///share/data",
    ],
)
def test_malformed_unc_paths_are_rejected(path: str):
    with pytest.raises(GuidedIdentityError, match="UNC paths require"):
        canonicalize_absolute_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "/../data",
        r"C:\..\data",
        r"\\server\share\..\data",
    ],
)
def test_path_traversal_above_root_is_rejected(path: str):
    with pytest.raises(GuidedIdentityError):
        canonicalize_absolute_path(path)


def test_duplicate_included_roi_ids_are_rejected():
    with pytest.raises(GuidedIdentityError, match="Duplicate"):
        build_canonical_guided_identity_payload_from_request(
            _request(included_roi_ids=["ROI0", "ROI0"])
        )


def test_included_roi_ids_are_sorted_after_duplicate_rejection():
    payload = _payload()
    assert payload.included_roi_ids == ("ROI0", "ROI1")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected(value: float):
    with pytest.raises(GuidedIdentityError, match="finite"):
        build_canonical_guided_identity_payload_from_request(
            _request(session_duration_sec=value)
        )
    with pytest.raises(GuidedIdentityError, match="Non-finite"):
        encode_canonical_value({"value": value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source_format", "auto"),
        ("acquisition_mode", "continuous"),
        ("timeline_anchor_mode", "elapsed"),
        ("execution_mode", "both"),
        ("run_profile", "tuning_prep"),
        ("traces_only", True),
        ("strategy_scope", "per_roi"),
        ("global_correction_strategy", "signal_only_f0"),
        ("dynamic_fit_mode", "unsupported"),
        ("output_path_role", "run_dir"),
        ("output_creation_timing", "preview"),
        ("run_directory_strategy", "gui_owned"),
        ("output_overwrite", True),
    ],
)
def test_unsupported_first_subset_values_are_rejected(field_name: str, value: object):
    with pytest.raises(GuidedIdentityError):
        build_canonical_guided_identity_payload_from_request(
            _request(**{field_name: value})
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "source_path",
        "sessions_per_hour",
        "session_duration_sec",
        "global_correction_strategy",
        "dynamic_fit_mode",
        "output_base_path",
    ],
)
def test_missing_required_request_values_are_rejected(field_name: str):
    with pytest.raises(GuidedIdentityError, match="required"):
        build_canonical_guided_identity_payload_from_request(
            _request(**{field_name: None})
        )


def test_payload_has_required_schema_and_explicit_deferred_fields():
    payload = _payload()
    assert payload.identity_schema_name == IDENTITY_SCHEMA_NAME
    assert payload.identity_schema_version == IDENTITY_SCHEMA_VERSION
    assert payload.canonicalization_algorithm_version == CANONICALIZATION_ALGORITHM_VERSION
    assert payload.request_schema_version == GUIDED_VALIDATION_REQUEST_SCHEMA_VERSION
    assert payload.deferred_fields == DEFERRED_CANONICAL_IDENTITY_FIELDS
    assert payload.deferred_fields_status == DEFERRED_CANONICAL_IDENTITY_STATUS
    assert payload.deferred_fields_status == "unresolved_future_identity_inputs"


def test_payload_contains_no_run_or_artifact_fields():
    forbidden = {
        "run_dir",
        "run_id",
        "production_run_id",
        "artifact_path",
        "status_path",
        "run_report_path",
        "manifest_path",
        "config_path",
        "command_path",
        "argv",
    }
    assert forbidden.isdisjoint(GuidedCanonicalValidationIdentityPayload.__dataclass_fields__)
    encoded = encode_canonical_identity_payload(_payload()).decode("utf-8")
    assert all(f'"{field}"' not in encoded for field in forbidden)


def test_build_from_request_produces_canonical_payload_and_digest_without_mutation():
    request = _request()
    original_roi_ids = list(request.included_roi_ids)
    payload = build_canonical_guided_identity_payload_from_request(request)
    digest = compute_canonical_guided_identity(payload)

    assert request.included_roi_ids == original_roi_ids
    assert payload.source_path_canonical == r"c:\data\raw"
    assert payload.output_base_path_canonical == r"c:\data\output"
    assert len(digest) == 64
    int(digest, 16)


def test_identity_foundation_remains_distinct_and_non_authorizing():
    request = _request()
    draft_fingerprint = compute_request_identity(request)
    payload = build_canonical_guided_identity_payload_from_request(request)
    canonical_identity = compute_canonical_guided_identity(payload)
    assert draft_fingerprint != canonical_identity
    assert payload.deferred_fields
    assert payload.deferred_fields_status == "unresolved_future_identity_inputs"
    assert "never current Run authorization" in (
        compute_canonical_guided_identity.__doc__ or ""
    )
