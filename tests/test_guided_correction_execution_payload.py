import json

import pytest
from pathlib import Path
import hashlib

from photometry_pipeline.guided_correction_payload import (
    GuidedCorrectionPayloadError,
    load_guided_correction_payload,
    serialize_guided_correction_payload,
)
from photometry_pipeline.guided_startup_transaction import (
    GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION,
    LEGACY_GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION,
)
from photometry_pipeline.guided_production_mapping import GuidedProductionPerRoiStrategy


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
