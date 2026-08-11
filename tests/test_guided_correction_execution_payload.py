import json

import pytest
from pathlib import Path
import hashlib

from photometry_pipeline.core.types import PerRoiCorrectionSpec
from photometry_pipeline.guided_correction_payload import (
    GuidedCorrectionPayloadError,
    correction_payload_identity,
    load_guided_correction_payload,
    serialize_guided_correction_payload,
)
from photometry_pipeline.guided_startup_transaction import (
    GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION,
    LEGACY_GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION,
)
from photometry_pipeline.guided_production_mapping import (
    GuidedProductionPerRoiStrategy,
    GuidedProductionTypedValue,
)


def _entry(roi, selected):
    signal = selected == "signal_only_f0"
    return GuidedProductionPerRoiStrategy(
        roi_id=roi,
        strategy_family="signal_only_f0" if signal else "dynamic_fit",
        dynamic_fit_mode=None if signal else selected,
        selected_strategy=selected,
        evidence_source_type="accepted_test_evidence",
        evidence_reference_json=json.dumps({"roi": roi}, sort_keys=True),
        explicit_user_mark=True,
        current_or_stale="current",
    )


def test_mixed_four_strategy_round_trip_is_deterministic(tmp_path):
    entries = (
        _entry("A", "robust_global_event_reject"),
        _entry("B", "signal_only_f0"),
        _entry("C", "global_linear_regression"),
        _entry("D", "adaptive_event_gated_regression"),
    )
    first = serialize_guided_correction_payload(("A", "B", "C", "D"), entries)
    second = serialize_guided_correction_payload(("D", "C", "B", "A"), tuple(reversed(entries)))
    assert json.loads(first)["canonical_correction_payload_identity"] == json.loads(second)["canonical_correction_payload_identity"]
    path = tmp_path / "correction.json"
    path.write_bytes(first)
    resolved = load_guided_correction_payload(path, ("D", "C", "B", "A"))
    assert {roi: spec.selected_strategy for roi, spec in resolved.items()} == {
        "A": "robust_global_event_reject", "B": "signal_only_f0",
        "C": "global_linear_regression", "D": "adaptive_event_gated_regression",
    }


def test_all_signal_and_heterogeneous_fit_maps_round_trip(tmp_path):
    for entries in (
        (_entry("A", "signal_only_f0"), _entry("B", "signal_only_f0")),
        (_entry("A", "robust_global_event_reject"), _entry("B", "global_linear_regression")),
    ):
        path = tmp_path / (entries[0].selected_strategy + ".json")
        path.write_bytes(serialize_guided_correction_payload(("A", "B"), entries))
        assert set(load_guided_correction_payload(path, ("A", "B"))) == {"A", "B"}


@pytest.mark.parametrize("mutation", ["schema", "coverage", "identity", "family"])
def test_malformed_current_payload_fails_closed(tmp_path, mutation):
    path = tmp_path / "correction.json"
    path.write_bytes(serialize_guided_correction_payload(("A",), (_entry("A", "global_linear_regression"),)))
    payload = json.loads(path.read_text())
    if mutation == "schema": payload["schema_version"] = "v999"
    elif mutation == "coverage": payload["included_roi_ids"] = ["A", "B"]
    elif mutation == "identity": payload["per_roi_correction"][0]["evidence_identity"] += "-changed"
    else: payload["per_roi_correction"][0]["strategy_family"] = "unknown"
    path.write_text(json.dumps(payload))
    with pytest.raises(GuidedCorrectionPayloadError):
        load_guided_correction_payload(path, ("A",))


def _write_startup_provenance(directory, contract, payload_bytes=None):
    document = {"startup_contract_version": contract}
    if payload_bytes is not None:
        payload = json.loads(payload_bytes)
        document.update({
            "serialized_native_correction_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "native_correction_payload_identity": payload["canonical_correction_payload_identity"],
        })
    (Path(directory) / "guided_startup_provenance.json").write_text(
        json.dumps(document), encoding="utf-8"
    )


def test_cli_loader_requires_positive_current_or_legacy_contract(tmp_path):
    from analyze_photometry import load_guided_per_roi_correction
    from tests.test_guided_run_per_roi_feature_execution import (
        _build_two_roi_guided_manifest,
    )

    _root, manifest = _build_two_roi_guided_manifest(tmp_path)
    correction_bytes = serialize_guided_correction_payload(
        ("ROI0", "ROI1"),
        (_entry("ROI0", "global_linear_regression"), _entry("ROI1", "robust_global_event_reject")),
    )
    correction_path = manifest.parent / "guided_per_roi_correction.json"
    correction_path.write_bytes(correction_bytes)
    _write_startup_provenance(
        manifest.parent, GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION, correction_bytes
    )
    assert set(load_guided_per_roi_correction(manifest)) == {"ROI0", "ROI1"}

    correction_path.unlink()
    with pytest.raises(GuidedCorrectionPayloadError, match="missing"):
        load_guided_per_roi_correction(manifest)

    _write_startup_provenance(
        manifest.parent, LEGACY_GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION
    )
    assert load_guided_per_roi_correction(manifest) is None


@pytest.mark.parametrize("provenance", [None, {}, {"startup_contract_version": "unknown.v9"}])
def test_cli_loader_refuses_missing_native_payload_without_positive_legacy(
    tmp_path, provenance
):
    from analyze_photometry import load_guided_per_roi_correction
    from tests.test_guided_run_per_roi_feature_execution import (
        _build_two_roi_guided_manifest,
    )

    _root, manifest = _build_two_roi_guided_manifest(tmp_path)
    if provenance is not None:
        (manifest.parent / "guided_startup_provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
    with pytest.raises(GuidedCorrectionPayloadError):
        load_guided_per_roi_correction(manifest)


def test_cli_loader_refuses_native_file_mutated_after_authorization(tmp_path):
    from analyze_photometry import load_guided_per_roi_correction
    from tests.test_guided_run_per_roi_feature_execution import (
        _build_two_roi_guided_manifest,
    )

    _root, manifest = _build_two_roi_guided_manifest(tmp_path)
    correction_bytes = serialize_guided_correction_payload(
        ("ROI0", "ROI1"),
        (_entry("ROI0", "global_linear_regression"), _entry("ROI1", "robust_global_event_reject")),
    )
    path = manifest.parent / "guided_per_roi_correction.json"
    path.write_bytes(correction_bytes)
    _write_startup_provenance(
        manifest.parent, GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION, correction_bytes
    )
    path.write_bytes(correction_bytes + b" ")
    with pytest.raises(GuidedCorrectionPayloadError, match="authorized startup provenance"):
        load_guided_per_roi_correction(manifest)


@pytest.mark.parametrize(
    ("strategy", "legacy_field", "parameter_values"),
    [
        (
            "robust_global_event_reject",
            "robust_event_reject_local_var_window_sec",
            (
                ("robust_event_reject_max_iters", "int", 3),
                ("robust_event_reject_residual_z_thresh", "float", 3.5),
                ("robust_event_reject_min_keep_fraction", "float", 0.5),
            ),
        ),
        (
            "adaptive_event_gated_regression",
            "adaptive_event_gate_local_var_window_sec",
            (
                ("adaptive_event_gate_residual_z_thresh", "float", 3.5),
                ("adaptive_event_gate_smooth_window_sec", "float", 60.0),
                ("adaptive_event_gate_min_trust_fraction", "float", 0.5),
            ),
        ),
    ],
)
def test_payload_reopens_legacy_local_variance_fields_without_retaining_them(
    tmp_path, strategy, legacy_field, parameter_values
):
    parameters = tuple(
        GuidedProductionTypedValue(
            name, kind, value, "applied_dynamic_fit_per_roi"
        )
        for name, kind, value in parameter_values
    )
    entry = _entry("ROI1", strategy)
    entry = type(entry)(**{**entry.__dict__, "effective_parameters": parameters})
    path = tmp_path / "current.json"
    path.write_bytes(serialize_guided_correction_payload(("ROI1",), (entry,)))
    current_spec = load_guided_correction_payload(path, ("ROI1",))["ROI1"]
    assert dict(current_spec.effective_parameters) == {
        name: value for name, _kind, value in parameter_values
    }
    assert legacy_field not in dict(current_spec.effective_parameters)

    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["per_roi_correction"][0]["effective_parameters"][legacy_field] = 20.0
    legacy_parameters = tuple(current_spec.effective_parameters) + (
        (legacy_field, 20.0),
    )
    legacy_spec = PerRoiCorrectionSpec(
        roi_id=current_spec.roi_id,
        strategy_family=current_spec.strategy_family,
        selected_strategy=current_spec.selected_strategy,
        dynamic_fit_mode=current_spec.dynamic_fit_mode,
        parameter_identity=current_spec.parameter_identity,
        evidence_identity=current_spec.evidence_identity,
        effective_parameters=legacy_parameters,
    )
    legacy["canonical_correction_payload_identity"] = correction_payload_identity(
        ("ROI1",), {"ROI1": legacy_spec}
    )
    legacy_path = tmp_path / "legacy_with_local_variance.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    resolved_legacy_parameters = dict(
        load_guided_correction_payload(legacy_path, ("ROI1",))["ROI1"].effective_parameters
    )
    assert resolved_legacy_parameters == dict(current_spec.effective_parameters)
    assert legacy_field not in resolved_legacy_parameters

    legacy_without_parameters = json.loads(path.read_text(encoding="utf-8"))
    legacy_entries = []
    for raw in legacy_without_parameters["per_roi_correction"]:
        legacy_entries.append(
            {
                key: raw[key]
                for key in (
                    "roi_id",
                    "strategy_family",
                    "selected_strategy",
                    "dynamic_fit_mode",
                    "parameter_identity",
                    "evidence_identity",
                )
            }
        )
    legacy_without_parameters["per_roi_correction"] = legacy_entries
    legacy_basis = dict(legacy_without_parameters)
    legacy_basis.pop("canonical_correction_payload_identity", None)
    legacy_without_parameters["canonical_correction_payload_identity"] = hashlib.sha256(
        json.dumps(legacy_basis, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    legacy_without_parameters_path = tmp_path / "legacy_without_parameters.json"
    legacy_without_parameters_path.write_text(
        json.dumps(legacy_without_parameters), encoding="utf-8"
    )
    resolved_legacy = load_guided_correction_payload(
        legacy_without_parameters_path, ("ROI1",)
    )
    assert resolved_legacy["ROI1"].effective_parameters == ()
