"""Terminal correction-evidence verification against real native cache output."""

from pathlib import Path
import json
import shutil

import h5py
import numpy as np
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.core.types import PerRoiCorrectionSpec
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.run_completion_contract import (
    COMPLETION_KEY,
    PROFILE_TUNING_PREP,
    TERMINAL_CORRUPTED,
    TERMINAL_SUCCESS_CURRENT,
    TERMINAL_SUCCESS_WITH_MISSING,
    build_manifest_completion_block,
    build_report_completion_block,
    build_status_completion_block,
    classify_run_terminal_state,
    correction_completion_error,
    normalize_run_mode,
    sha256_file,
)


def _write_source(path: Path) -> None:
    n = 200
    t = np.arange(n, dtype=float) / 10.0
    values = {"TimeStamp": t}
    for index in range(2):
        values[f"Region{index}-410"] = 2.0 + 0.1 * np.sin(0.2 * t + index)
        values[f"Region{index}-470"] = 5.0 + index + 0.2 * np.cos(0.3 * t + index)
    path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(values).to_csv(path, index=False)


def _mixed_map() -> dict[str, PerRoiCorrectionSpec]:
    return {
        "Region0": PerRoiCorrectionSpec(
            "Region0", "dynamic_fit", "global_linear_regression", "global_linear_regression"
        ),
        "Region1": PerRoiCorrectionSpec("Region1", "signal_only_f0", "signal_only_f0"),
    }


@pytest.fixture
def native_run(tmp_path):
    source = tmp_path / "input" / "2024_01_01-00_00_00" / "fluorescence.csv"
    _write_source(source)
    cfg = Config(
        target_fs_hz=10.0,
        chunk_duration_sec=20.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        lowpass_hz=2.0,
        filter_order=2,
        window_sec=10.0,
        min_samples_per_window=10,
        signal_only_f0_min_window_samples=21,
    )
    analysis = tmp_path / "analysis"
    Pipeline(cfg, mode="phasic", per_roi_correction=_mixed_map()).run(
        str(source.parent.parent), str(analysis), force_format="rwd", recursive=True
    )
    mode = normalize_run_mode(
        run_profile="tuning_prep",
        run_type="full",
        acquisition_mode="intermittent",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=False,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_TUNING_PREP,
        expected_rois=["Region0", "Region1"],
        chunked_input_processing=True,
        shared_input_manifest=False,
    )
    return analysis, mode


def _root_for_case(tmp_path: Path, analysis: Path, name: str) -> Path:
    root = tmp_path / name
    shutil.copytree(analysis, root / "_analysis" / "phasic_out")
    return root


def _write_terminal_set(root: Path, mode: dict, *, run_id: str = "native-run") -> None:
    """Build a minimal real current terminal set around a native analysis."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_report.json").write_text(
        json.dumps({"completion_contract": build_report_completion_block(run_id=run_id)}, indent=2),
        encoding="utf-8",
    )
    manifest = {
        COMPLETION_KEY: build_manifest_completion_block(
            str(root),
            run_id=run_id,
            run_mode=mode,
            finalized_utc="2026-07-11T00:00:00+00:00",
        )
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    status = {
        "run_id": run_id,
        "run_profile": mode["run_profile"],
        "run_type": mode["run_type"],
        "acquisition_mode": mode["acquisition_mode"],
        "traces_only": mode["traces_only"],
        "phase": "final",
        "status": "success",
        "errors": [],
        COMPLETION_KEY: build_status_completion_block(
            run_id=run_id,
            manifest_sha256=sha256_file(str(root / "MANIFEST.json")),
        ),
    }
    (root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def test_real_mixed_native_cache_passes_completion_verifier(native_run, tmp_path):
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, "positive")
    assert correction_completion_error(str(root), mode) == ""


def test_valid_equal_provenance_pair_classifies_current_success(native_run, tmp_path):
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, "coherent_success")
    _write_terminal_set(root, mode)
    assert correction_completion_error(str(root), mode) == ""
    classification = classify_run_terminal_state(str(root))
    assert classification.state == TERMINAL_SUCCESS_CURRENT


@pytest.mark.parametrize(
    "mutation",
    [
        "remove_metadata",
        "remove_report",
        "metadata_none",
        "report_list",
        "report_strategy",
        "metadata_identity",
    ],
)
def test_one_sided_or_mismatched_provenance_fails_and_classifies_corrupted(
    native_run, tmp_path, mutation
):
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, mutation)
    _write_terminal_set(root, mode)
    metadata_path = root / "_analysis" / "phasic_out" / "run_metadata.json"
    report_path = root / "_analysis" / "phasic_out" / "run_report.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "remove_metadata":
        del metadata["correction_provenance"]
    elif mutation == "remove_report":
        del report["derived_settings"]["correction_provenance"]
    elif mutation == "metadata_none":
        metadata["correction_provenance"] = None
    elif mutation == "report_list":
        report["derived_settings"]["correction_provenance"] = []
    elif mutation == "report_strategy":
        report["derived_settings"]["correction_provenance"]["requested_by_roi"][1][
            "selected_strategy"
        ] = "global_linear_regression"
    elif mutation == "metadata_identity":
        metadata["correction_provenance"]["requested_by_roi"][0][
            "parameter_identity"
        ] = "tampered-identity"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Repin the terminal set so classification exercises provenance coherence,
    # rather than merely detecting a stale artifact digest.
    _write_terminal_set(root, mode)
    error = correction_completion_error(str(root), mode)
    assert error
    assert classify_run_terminal_state(str(root)).state == TERMINAL_CORRUPTED


def test_both_provenance_copies_absent_use_pre_provenance_compatibility(native_run, tmp_path):
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, "both_absent")
    _write_terminal_set(root, mode)
    metadata_path = root / "_analysis" / "phasic_out" / "run_metadata.json"
    report_path = root / "_analysis" / "phasic_out" / "run_report.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del metadata["correction_provenance"]
    del report["derived_settings"]["correction_provenance"]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_terminal_set(root, mode)
    assert correction_completion_error(str(root), mode) == ""
    assert classify_run_terminal_state(str(root)).state == TERMINAL_SUCCESS_CURRENT


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("missing_baseline", "baseline"),
        ("missing_qc", "QC"),
        ("signal_dynamic_mode", "dynamic-fit mode"),
        ("dynamic_missing_fit", "fit_ref"),
        ("dynamic_orphan_baseline", "orphan Signal-Only"),
        ("missing_session", "cache sessions"),
        ("low_dff", "dF/F coverage"),
    ],
)
def test_completion_verifier_rejects_one_corrupt_native_component(
    native_run, tmp_path, mutation, expected
):
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, mutation)
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "r+") as handle:
        signal = handle["roi/Region1/chunk_0"]
        fit = handle["roi/Region0/chunk_0"]
        if mutation == "missing_baseline":
            del signal["signal_only_f0_baseline"]
        elif mutation == "missing_qc":
            for key in (
                "signal_only_f0_production_available",
                "signal_only_f0_production_baseline_source",
                "signal_only_f0_production_formula",
            ):
                if key in signal.attrs:
                    del signal.attrs[key]
        elif mutation == "signal_dynamic_mode":
            signal.attrs["correction_dynamic_fit_mode"] = "global_linear_regression"
        elif mutation == "dynamic_missing_fit":
            del fit["fit_ref"]
        elif mutation == "dynamic_orphan_baseline":
            fit.create_dataset("signal_only_f0_baseline", data=np.ones(200))
        elif mutation == "missing_session":
            for roi in ("Region0", "Region1"):
                del handle[f"roi/{roi}/chunk_0"]
        elif mutation == "low_dff":
            dff = signal["dff"][()]
            dff[:] = np.nan
            signal["dff"][...] = dff
    error = correction_completion_error(str(root), mode)
    assert expected.lower() in error.lower()


def test_completion_verifier_rejects_requested_consumed_mismatch(native_run, tmp_path):
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, "requested_mismatch")
    for relative in (
        Path("_analysis") / "phasic_out" / "run_metadata.json",
        Path("_analysis") / "phasic_out" / "run_report.json",
    ):
        path = root / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload["correction_provenance"] if "correction_provenance" in payload else payload["derived_settings"]["correction_provenance"]
        provenance["requested_by_roi"][1]["selected_strategy"] = "global_linear_regression"
        provenance["requested_by_roi"][1]["strategy_family"] = "dynamic_fit"
        provenance["requested_by_roi"][1]["dynamic_fit_mode"] = "global_linear_regression"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    error = correction_completion_error(str(root), mode)
    assert "strategy" in error.lower() or "provenance" in error.lower()


def test_real_native_completion_allows_authorized_missing_middle_session(tmp_path):
    input_root = tmp_path / "missing_input"
    sources = []
    for index in range(3):
        source = input_root / f"2024_01_01-00_0{index}_00" / "fluorescence.csv"
        _write_source(source)
        sources.append(source)
    sources[1].write_text("BROKEN\n", encoding="utf-8")
    cfg = Config(
        target_fs_hz=10.0,
        chunk_duration_sec=20.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        lowpass_hz=2.0,
        filter_order=2,
        signal_only_f0_min_window_samples=21,
        authorized_missing_sessions=[str(sources[1])],
    )
    analysis = tmp_path / "missing_analysis"
    Pipeline(
        cfg,
        mode="phasic",
        per_roi_correction={
            "Region0": PerRoiCorrectionSpec("Region0", "signal_only_f0", "signal_only_f0"),
            "Region1": PerRoiCorrectionSpec("Region1", "signal_only_f0", "signal_only_f0"),
        },
    ).run(str(input_root), str(analysis), force_format="rwd", recursive=True)
    root = _root_for_case(tmp_path, analysis, "missing_root")
    mode = normalize_run_mode(
        run_profile="tuning_prep",
        run_type="full",
        acquisition_mode="intermittent",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=False,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_TUNING_PREP,
        expected_rois=["Region0", "Region1"],
        chunked_input_processing=True,
        shared_input_manifest=False,
    )
    assert correction_completion_error(str(root), mode) == ""


def test_real_native_completion_allows_authorized_final_exclusion(tmp_path):
    input_root = tmp_path / "excluded_input"
    sources = []
    for index in range(3):
        source = input_root / f"2024_01_01-00_0{index}_00" / "fluorescence.csv"
        _write_source(source)
        sources.append(source)
    sources[-1].write_text("BROKEN\n", encoding="utf-8")
    cfg = Config(
        target_fs_hz=10.0,
        chunk_duration_sec=20.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        lowpass_hz=2.0,
        filter_order=2,
        signal_only_f0_min_window_samples=21,
        exclude_incomplete_final_rwd_chunk=True,
        rwd_excluded_source_files=[str(sources[-1])],
    )
    analysis = tmp_path / "excluded_analysis"
    Pipeline(cfg, mode="phasic", per_roi_correction=_mixed_map()).run(
        str(input_root), str(analysis), force_format="rwd", recursive=True
    )
    completeness = json.loads(
        (analysis / "input_processing_completeness.json").read_text(encoding="utf-8")
    )
    assert completeness["expected"][-1]["disposition"] == "authorized_exclusion"
    assert [record["cache_chunk_id"] for record in completeness["processed"]] == [0, 1]

    root = _root_for_case(tmp_path, analysis, "excluded_root")
    mode = normalize_run_mode(
        run_profile="tuning_prep",
        run_type="full",
        acquisition_mode="intermittent",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=False,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_TUNING_PREP,
        expected_rois=["Region0", "Region1"],
        chunked_input_processing=True,
        shared_input_manifest=False,
    )
    _write_terminal_set(root, mode)
    assert correction_completion_error(str(root), mode) == ""
    classification = classify_run_terminal_state(str(root))
    assert classification.state == TERMINAL_SUCCESS_WITH_MISSING
    assert classification.final_exclusion_count == 1
    with h5py.File(root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5", "r") as handle:
        assert [int(value) for value in handle["meta/chunk_ids"][()]] == [0, 1]
        for roi in ("Region0", "Region1"):
            assert sorted(handle[f"roi/{roi}"].keys()) == ["chunk_0", "chunk_1"]
