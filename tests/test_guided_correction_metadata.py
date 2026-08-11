from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.core.tonic_dff import (
    apply_global_fit,
    compute_global_iso_fit_robust,
)
from photometry_pipeline.core.types import PerRoiCorrectionSpec
from photometry_pipeline.pipeline import Pipeline


ROI_IDS = ("ROI1", "ROI2")


def _write_mapped_csv_sessions(source_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    time_sec = np.arange(1201, dtype=float) / 10.0
    for session_index in (1, 2):
        rows = [
            "ElapsedSeconds,ROI1_Signal,ROI1_Reference,ROI2_Signal,ROI2_Reference"
        ]
        for value in time_sec:
            roi1_reference = (
                2.0
                + 0.1 * np.cos(value)
                + 0.75 * (session_index - 1)
            )
            roi2_reference = (
                2.2
                + 0.1 * np.sin(value)
                + 0.5 * (session_index - 1)
            )
            rows.append(
                f"{value:.6f},{4.0 + 1.5 * roi1_reference + 0.2 * np.sin(value):.6f},"
                f"{roi1_reference:.6f},"
                f"{3.0 + 1.2 * roi2_reference + 0.25 * np.cos(value):.6f},"
                f"{roi2_reference:.6f}"
            )
        (source_root / f"session_{session_index}.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )


def _config() -> Config:
    return Config(
        target_fs_hz=10.0,
        chunk_duration_sec=100.0,
        custom_tabular_time_col="ElapsedSeconds",
        custom_tabular_time_unit="seconds",
        custom_tabular_roi_mapping_json=json.dumps(
            [
                {
                    "roi_id": "ROI1",
                    "signal_column": "ROI1_Signal",
                    "reference_column": "ROI1_Reference",
                },
                {
                    "roi_id": "ROI2",
                    "signal_column": "ROI2_Signal",
                    "reference_column": "ROI2_Reference",
                },
            ]
        ),
        lowpass_hz=2.0,
        filter_order=2,
        signal_only_f0_min_window_samples=21,
    )


def _correction_map(strategy_by_roi: dict[str, str]):
    return {
        roi_id: (
            PerRoiCorrectionSpec(roi_id, "signal_only_f0", "signal_only_f0")
            if strategy == "signal_only_f0"
            else PerRoiCorrectionSpec(
                roi_id, "dynamic_fit", strategy, strategy
            )
        )
        for roi_id, strategy in strategy_by_roi.items()
    }


def _run_metadata_case(tmp_path: Path, strategy_by_roi: dict[str, str]):
    source_root = tmp_path / "source"
    _write_mapped_csv_sessions(source_root)
    analysis_out = tmp_path / "analysis"
    Pipeline(
        _config(),
        mode="phasic",
        per_roi_correction=_correction_map(strategy_by_roi),
    ).run(
        str(source_root),
        str(analysis_out),
        force_format="custom_tabular",
        recursive=False,
    )
    metadata = json.loads(
        (analysis_out / "run_metadata.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (analysis_out / "run_report.json").read_text(encoding="utf-8")
    )
    return analysis_out, metadata, report


def _assert_summary_matches_saved_applied_fields(
    analysis_out: Path, metadata: dict
) -> None:
    summary = metadata["applied_correction_summary"]
    summary_by_roi = {
        record["roi_id"]: record for record in summary["by_roi"]
    }
    with h5py.File(analysis_out / "phasic_trace_cache.h5", "r") as cache:
        for roi_id, record in summary_by_roi.items():
            group = cache[f"roi/{roi_id}/chunk_0"]
            assert record["strategy_family"] == group.attrs[
                "correction_strategy_family"
            ]
            assert record["applied_strategy"] == group.attrs[
                "correction_applied_strategy"
            ]
            assert record["applied_correction_source"] == group.attrs[
                "correction_applied_source"
            ]


def test_all_signal_only_csv_metadata_uses_applied_signal_only_semantics(
    tmp_path: Path,
):
    analysis_out, metadata, report = _run_metadata_case(
        tmp_path,
        {"ROI1": "signal_only_f0", "ROI2": "signal_only_f0"},
    )

    summary = metadata["applied_correction_summary"]
    assert summary["authority"] == "applied_correction_records"
    assert summary["classification"] == "all_signal_only_f0"
    assert summary["applied_strategies"] == ["signal_only_f0"]
    assert summary["applied_sources"] == ["signal_only_f0_baseline"]
    assert {
        record["roi_id"]: record["applied_strategy"]
        for record in summary["by_roi"]
    } == {"ROI1": "signal_only_f0", "ROI2": "signal_only_f0"}
    assert metadata["baseline_method"] == "signal_only_f0_baseline"
    assert metadata["f0_source"] == "signal_only_f0_baseline"
    assert metadata["phasic_uv_fit_method"] == "not_applicable"
    assert metadata["regression_mode"] == "not_applicable"
    assert "correction_provenance" in metadata

    contract = report["analytical_contract"]
    assert contract["correction_semantics"]["scope"] == "per_roi"
    assert contract["signal_semantics"]["uv_fit"] == (
        "not used for Signal-Only F0 ROIs"
    )
    assert contract["baseline_semantics"]["f0_source"] == (
        "signal_only_f0_baseline"
    )
    assert contract["baseline_semantics"]["dff_formula"] == (
        "100 * (sig_raw - signal_only_f0_baseline) "
        "/ signal_only_f0_baseline"
    )
    _assert_summary_matches_saved_applied_fields(analysis_out, metadata)


def test_mixed_csv_metadata_remains_explicitly_per_roi(tmp_path: Path):
    analysis_out, metadata, report = _run_metadata_case(
        tmp_path,
        {"ROI1": "signal_only_f0", "ROI2": "global_linear_regression"},
    )

    summary = metadata["applied_correction_summary"]
    assert summary["classification"] == "mixed"
    assert summary["strategy_families"] == ["dynamic_fit", "signal_only_f0"]
    assert summary["applied_strategies"] == [
        "global_linear_regression",
        "signal_only_f0",
    ]
    assert summary["applied_sources"] == [
        "fitted_reference",
        "signal_only_f0_baseline",
    ]
    assert {
        record["roi_id"]: record["applied_strategy"]
        for record in summary["by_roi"]
    } == {"ROI1": "signal_only_f0", "ROI2": "global_linear_regression"}
    assert {
        metadata[field]
        for field in (
            "baseline_method",
            "f0_source",
            "phasic_uv_fit_method",
            "regression_mode",
        )
    } == {"per_roi"}

    contract = report["analytical_contract"]
    assert contract["correction_semantics"]["strategy_classification"] == (
        "mixed"
    )
    assert contract["baseline_semantics"]["method"] == "per_roi"
    assert contract["baseline_semantics"]["f0_source"] == "per_roi"
    assert "ROI-specific" in contract["baseline_semantics"]["interpretation_note"]
    assert "sig_raw - uv_fit" not in str(contract["signal_semantics"])
    _assert_summary_matches_saved_applied_fields(analysis_out, metadata)


@pytest.mark.parametrize(
    "strategy",
    [
        "global_linear_regression",
        "robust_global_event_reject",
        "adaptive_event_gated_regression",
    ],
)
def test_uniform_reference_csv_metadata_retains_reference_semantics(
    tmp_path: Path, strategy: str,
):
    analysis_out, metadata, report = _run_metadata_case(
        tmp_path,
        {
            "ROI1": strategy,
            "ROI2": strategy,
        },
    )

    summary = metadata["applied_correction_summary"]
    assert summary["classification"] == "all_reference_based"
    assert metadata["baseline_method"] == "guided_global_fit_reference_session_median"
    assert metadata["f0_source"] == "global_fitted_reference_session_median"
    assert metadata["phasic_uv_fit_method"] == "dynamic"
    assert metadata["regression_mode"] == "dynamic"

    provenance = metadata["guided_reference_f0_provenance"]["by_roi"]
    assert set(provenance) == set(ROI_IDS)
    for roi_id in ROI_IDS:
        assert provenance[roi_id]["method"] == "compute_global_iso_fit_robust"
        assert provenance[roi_id]["apply_method"] == "apply_global_fit"
        assert provenance[roi_id]["f0_aggregation"] == "median"
        assert provenance[roi_id]["f0_scope"] == "session"
        assert len(provenance[roi_id]["f0_values"]) == 2
        assert provenance[roi_id]["f0_values"][0]["f0"] != provenance[roi_id]["f0_values"][1]["f0"]

    with h5py.File(analysis_out / "phasic_trace_cache.h5", "r") as cache:
        for roi_id in ROI_IDS:
            groups = [cache[f"roi/{roi_id}/chunk_{chunk_id}"] for chunk_id in (0, 1)]
            uv_all = np.concatenate([group["uv_raw"][()] for group in groups])
            sig_all = np.concatenate([group["sig_raw"][()] for group in groups])
            slope, intercept, fit_ok, _n_used = compute_global_iso_fit_robust(
                uv_all, sig_all, max_points=200_000, n_iter=3, z_thresh=4.0
            )
            assert fit_ok
            for group in groups:
                expected_f0 = float(
                    np.median(apply_global_fit(group["uv_raw"][()], slope, intercept))
                )
                assert float(group.attrs["correction_f0_value"]) == expected_f0
                assert float(group.attrs["correction_global_fit_slope"]) == slope
                assert float(group.attrs["correction_global_fit_intercept"]) == intercept
                np.testing.assert_allclose(
                    group["dff"][()],
                    100.0 * group["delta_f"][()] / expected_f0,
                    rtol=1e-12,
                    atol=1e-12,
                )

    contract = report["analytical_contract"]
    assert contract["signal_semantics"]["uv_fit"] == (
        "estimated artifact component derived from uv_filt fit to sig_filt"
    )
    assert contract["baseline_semantics"]["method"] == (
        "guided_global_fit_reference_session_median"
    )
    assert contract["baseline_semantics"]["f0_source"] == (
        "global_fitted_reference_session_median"
    )
    assert contract["baseline_semantics"]["dff_formula"] == (
        "100 * delta_f / median finite apply_global_fit(uv_raw, slope, intercept) per session"
    )
    _assert_summary_matches_saved_applied_fields(analysis_out, metadata)
