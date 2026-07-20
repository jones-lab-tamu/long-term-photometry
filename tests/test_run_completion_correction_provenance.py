"""Terminal correction-evidence verification against real native cache output."""

from pathlib import Path
import json
import shutil

import h5py
import numpy as np
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.completed_run_review import (
    CompletedRunReviewError,
    classify_analysis_plot_source,
    load_completed_phasic_review,
    resolve_analysis_plot_context,
    resolve_persisted_cache_strategy,
)
from photometry_pipeline.core.types import PerRoiCorrectionSpec
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.run_completion_contract import (
    COMPLETION_KEY,
    PROFILE_TUNING_PREP,
    PROFILE_FULL_INTERMITTENT,
    TERMINAL_CORRUPTED,
    TERMINAL_SUCCESS_CURRENT,
    TERMINAL_SUCCESS_WITH_MISSING,
    build_manifest_completion_block,
    build_report_completion_block,
    build_status_completion_block,
    classify_run_terminal_state,
    correction_completion_error,
    normalize_run_mode,
    required_artifacts_for_run_mode,
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


def test_tonic_and_combined_native_branches_pass_completion_verification(
    native_run, tmp_path, monkeypatch
):
    phasic_analysis, _phasic_mode = native_run
    source_root = tmp_path / "input"
    tonic_analysis = tmp_path / "tonic_analysis"
    cfg = Config(
        target_fs_hz=10.0, chunk_duration_sec=20.0,
        rwd_time_col="TimeStamp", uv_suffix="-410", sig_suffix="-470",
        lowpass_hz=2.0, filter_order=2, window_sec=10.0,
        min_samples_per_window=10, signal_only_f0_min_window_samples=21,
    )
    Pipeline(cfg, mode="tonic", per_roi_correction=_mixed_map()).run(
        str(source_root), str(tonic_analysis), force_format="rwd", recursive=True
    )
    from tools import plot_tonic_48h
    tonic_png = tmp_path / "prefinal_tonic.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_tonic_48h.py", "--analysis-out", str(tonic_analysis),
            "--roi", "Region1", "--out", str(tonic_png),
        ],
    )
    plot_tonic_48h.main()
    assert tonic_png.is_file()
    root = tmp_path / "combined_native"
    shutil.copytree(phasic_analysis, root / "_analysis" / "phasic_out")
    shutil.copytree(tonic_analysis, root / "_analysis" / "tonic_out")
    mode = normalize_run_mode(
        run_profile="full", run_type="full", acquisition_mode="intermittent",
        traces_only=False, phasic_analysis=True, tonic_analysis=True,
        feature_extraction_ran=True, deliverable_profile=PROFILE_FULL_INTERMITTENT,
        expected_rois=["Region0", "Region1"], chunked_input_processing=True,
        shared_input_manifest=False,
    )
    assert correction_completion_error(str(root), mode) == ""
    for relative in required_artifacts_for_run_mode(mode):
        path = root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test artifact")
    _write_terminal_set(root, mode, run_id="combined-native")
    def refuse_recompute(*_args, **_kwargs):
        raise AssertionError("completed Review must use persisted correction evidence")
    monkeypatch.setattr(
        "photometry_pipeline.pipeline.compute_signal_only_f0_candidate",
        refuse_recompute,
    )
    monkeypatch.setattr(
        "photometry_pipeline.core.regression.fit_chunk_dynamic",
        refuse_recompute,
    )
    review = load_completed_phasic_review(root)
    assert review.analysis_branches == ("tonic", "phasic")
    assert review.strategy_label_for_roi("Region1") == "Signal-Only F0"
    assert review.sessions_for_roi("Region1", branch="tonic")[0].production_f0_baseline is not None
    assert review.sessions_for_roi("Region0", branch="tonic")[0].fitted_reference is not None
    assert review.sessions_for_roi("Region1", branch="phasic")[0].production_f0_baseline is not None
    from gui.run_report_viewer import RunReportViewer
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    viewer = RunReportViewer()
    assert viewer.load_report(str(root)) is True
    assert "tonic and phasic analyses" in viewer._correction_summary_label.text()
    tonic_cache = root / "_analysis" / "tonic_out" / "tonic_trace_cache.h5"
    held_cache = tonic_cache.with_suffix(".held")
    tonic_cache.rename(held_cache)
    assert viewer.load_report(str(root)) is False
    assert viewer._workspace.isHidden()
    assert viewer.phasic_review_model is None
    assert "cannot be verified" in viewer._status_label.text().lower()
    held_cache.rename(tonic_cache)
    tonic_meta = json.loads(
        (tonic_analysis / "run_metadata.json").read_text(encoding="utf-8")
    )
    phasic_meta = json.loads(
        (phasic_analysis / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert tonic_meta["correction_provenance"]["requested_by_roi"] == (
        phasic_meta["correction_provenance"]["requested_by_roi"]
    )

    with h5py.File(
        root / "_analysis" / "tonic_out" / "tonic_trace_cache.h5", "r+"
    ) as cache:
        cache["roi/Region0/chunk_0"].attrs["correction_selected_strategy"] = (
            "robust_global_event_reject"
        )
        cache["roi/Region0/chunk_0"].attrs["correction_dynamic_fit_mode"] = (
            "robust_global_event_reject"
        )
    assert "mismatches the request" in correction_completion_error(
        str(root), mode
    ).lower()
    refused_png = tmp_path / "refused_tonic.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_tonic_48h.py", "--analysis-out",
            str(root / "_analysis" / "tonic_out"), "--roi", "Region0",
            "--out", str(refused_png),
        ],
    )
    with pytest.raises(SystemExit):
        plot_tonic_48h.main()
    assert not refused_png.exists()


def test_completed_review_loader_reads_persisted_mixed_evidence_without_recompute(
    native_run, tmp_path, monkeypatch
):
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, "review_model")
    _write_terminal_set(root, mode)

    def fail(*_args, **_kwargs):
        raise AssertionError("completed Review must not recompute correction evidence")

    monkeypatch.setattr(
        "photometry_pipeline.pipeline.compute_signal_only_f0_candidate", fail
    )
    monkeypatch.setattr("photometry_pipeline.core.regression.fit_chunk_dynamic", fail)
    model = load_completed_phasic_review(root)
    assert model.rois == ("Region0", "Region1")
    assert model.heterogeneous_correction is True
    dynamic = model.sessions_for_roi("Region0")[0]
    signal_only = model.sessions_for_roi("Region1")[0]
    assert dynamic.strategy_family == "dynamic_fit"
    assert dynamic.fitted_reference is not None
    assert dynamic.production_f0_baseline is None
    assert dynamic.strategy_label == "Global linear regression"
    assert signal_only.strategy_family == "signal_only_f0"
    assert signal_only.production_f0_baseline is not None
    assert signal_only.fitted_reference is None
    assert signal_only.strategy_label == "Signal-Only F0"
    with h5py.File(analysis / "phasic_trace_cache.h5", "r") as handle:
        np.testing.assert_array_equal(
            signal_only.production_f0_baseline,
            handle["roi/Region1/chunk_0/signal_only_f0_baseline"][()],
        )
        np.testing.assert_array_equal(
            signal_only.canonical_dff,
            handle["roi/Region1/chunk_0/dff"][()],
        )


def test_completed_review_loader_reads_all_signal_only_run(tmp_path):
    source = tmp_path / "input" / "2024_01_01-00_00_00" / "fluorescence.csv"
    _write_source(source)
    analysis = tmp_path / "all_signal_analysis"
    cfg = Config(
        target_fs_hz=10.0,
        chunk_duration_sec=20.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        lowpass_hz=2.0,
        filter_order=2,
        signal_only_f0_min_window_samples=21,
    )
    all_signal = {
        roi: PerRoiCorrectionSpec(roi, "signal_only_f0", "signal_only_f0")
        for roi in ("Region0", "Region1")
    }
    Pipeline(cfg, mode="phasic", per_roi_correction=all_signal).run(
        str(source.parent.parent), str(analysis), force_format="rwd", recursive=True
    )
    root = _root_for_case(tmp_path, analysis, "all_signal_root")
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
    review = load_completed_phasic_review(root)
    assert review.heterogeneous_correction is False
    assert review.strategy_label_for_roi("Region0") == "Signal-Only F0"
    assert review.sessions_for_roi("Region0")[0].fitted_reference is None
    assert review.sessions_for_roi("Region0")[0].production_f0_baseline is not None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_family",
        "missing_selected",
        "signal_only_missing_baseline",
        "dynamic_missing_fit_ref",
        "inconsistent_sessions",
        "config_only",
    ],
)
def test_current_plot_strategy_resolution_refuses_malformed_evidence(
    native_run, tmp_path, mutation
):
    analysis, _mode = native_run
    cache_path = analysis / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "r+") as handle:
        signal = handle["roi/Region1/chunk_0"]
        dynamic = handle["roi/Region0/chunk_0"]
        if mutation == "missing_family":
            del signal.attrs["correction_strategy_family"]
        elif mutation == "missing_selected":
            del signal.attrs["correction_selected_strategy"]
        elif mutation == "signal_only_missing_baseline":
            del signal["signal_only_f0_baseline"]
        elif mutation == "dynamic_missing_fit_ref":
            del dynamic["fit_ref"]
        elif mutation == "inconsistent_sessions":
            handle.copy(signal, "roi/Region1/chunk_1")
            handle["roi/Region1/chunk_1"].attrs["correction_selected_strategy"] = (
                "signal_only_f0_inconsistent"
            )
        elif mutation == "config_only":
            for attrs in (signal.attrs, dynamic.attrs):
                for key in ("correction_strategy_family", "correction_selected_strategy"):
                    if key in attrs:
                        del attrs[key]
        chunk_ids = [0, 1] if mutation == "inconsistent_sessions" else [0]
        roi = "Region1" if mutation in {
            "missing_family",
            "missing_selected",
            "signal_only_missing_baseline",
            "inconsistent_sessions",
            "config_only",
        } else "Region0"
        with pytest.raises(CompletedRunReviewError):
            with h5py.File(cache_path, "r") as cache:
                resolve_persisted_cache_strategy(
                    cache, roi, chunk_ids, strict_current=True
                )


def test_valid_equal_provenance_pair_classifies_current_success(native_run, tmp_path):
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, "coherent_success")
    _write_terminal_set(root, mode)
    assert correction_completion_error(str(root), mode) == ""
    classification = classify_run_terminal_state(str(root))
    assert classification.state == TERMINAL_SUCCESS_CURRENT


def test_analysis_without_native_provenance_or_terminal_files_is_standalone(tmp_path):
    analysis = tmp_path / "historical_analysis"
    analysis.mkdir()
    assert classify_analysis_plot_source(analysis) == "standalone"


def test_analysis_with_unverified_terminal_evidence_fails_closed(tmp_path):
    run_dir = tmp_path / "unverified_run"
    analysis = run_dir / "_analysis" / "phasic_out"
    analysis.mkdir(parents=True)
    (run_dir / "status.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CompletedRunReviewError):
        classify_analysis_plot_source(analysis)


def test_valid_native_pair_remains_current_with_partial_terminal_evidence(native_run, tmp_path):
    source_analysis, _mode = native_run
    run_dir = tmp_path / "partial_native_run"
    analysis = run_dir / "_analysis" / "phasic_out"
    shutil.copytree(source_analysis, analysis)
    (run_dir / "status.json").write_text("{}", encoding="utf-8")
    assert classify_analysis_plot_source(analysis) == "current"


def _mutate_prefinal_native_analysis(analysis: Path, mutation: str) -> None:
    metadata_path = analysis / "run_metadata.json"
    report_path = analysis / "run_report.json"
    cache_path = analysis / "phasic_trace_cache.h5"
    if mutation == "missing_metadata_provenance":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        del metadata["correction_provenance"]
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return
    if mutation == "disagreeing_provenance":
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["derived_settings"]["correction_provenance"]["requested_by_roi"][0][
            "selected_strategy"
        ] = "robust_global_event_reject"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return
    with h5py.File(cache_path, "r+") as handle:
        if mutation == "missing_family":
            del handle["roi/Region1/chunk_0"].attrs["correction_strategy_family"]
        elif mutation == "missing_selected":
            del handle["roi/Region1/chunk_0"].attrs["correction_selected_strategy"]
        elif mutation == "missing_signal_only_baseline":
            del handle["roi/Region1/chunk_0/signal_only_f0_baseline"]
        elif mutation == "missing_dynamic_fit_ref":
            del handle["roi/Region0/chunk_0/fit_ref"]
        elif mutation == "strategy_changes_across_sessions":
            handle.copy("roi/Region1/chunk_0", "roi/Region1/chunk_1")
            handle["roi/Region1/chunk_1"].attrs["correction_selected_strategy"] = (
                "signal_only_f0_inconsistent"
            )
            meta = handle["meta"]
            chunk_ids = np.asarray(meta["chunk_ids"][()]).reshape(-1)
            del meta["chunk_ids"]
            meta.create_dataset("chunk_ids", data=np.concatenate([chunk_ids, [1]]))
            if "source_files" in meta:
                source_files = np.asarray(meta["source_files"][()]).reshape(-1)
                del meta["source_files"]
                meta.create_dataset(
                    "source_files", data=np.concatenate([source_files, source_files[:1]])
                )
        else:
            raise AssertionError(f"unknown pre-finalization mutation: {mutation}")


def _update_requested_provenance_pair(
    analysis: Path, roi: str, **updates: object
) -> None:
    metadata_path = analysis / "run_metadata.json"
    report_path = analysis / "run_report.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = (
        metadata["correction_provenance"]["requested_by_roi"],
        report["derived_settings"]["correction_provenance"]["requested_by_roi"],
    )
    for requested in records:
        record = next(item for item in requested if item.get("roi_id") == roi)
        record.update(updates)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _cache_chunks(handle, roi: str):
    group = handle[f"roi/{roi}"]
    return [
        group[name]
        for name in sorted(group.keys())
        if str(name).startswith("chunk_")
    ]


def _mutate_requested_consumed_mismatch(analysis: Path, mutation: str) -> str:
    cache_path = analysis / "phasic_trace_cache.h5"
    if mutation == "requested_robust_consumed_global":
        _update_requested_provenance_pair(
            analysis,
            "Region0",
            strategy_family="dynamic_fit",
            selected_strategy="robust_global_event_reject",
            dynamic_fit_mode="robust_global_event_reject",
        )
        return "Region0"
    if mutation == "parameter_identity_mismatch":
        _update_requested_provenance_pair(
            analysis, "Region0", parameter_identity="requested-parameter-identity"
        )
        return "Region0"
    if mutation == "evidence_identity_mismatch":
        _update_requested_provenance_pair(
            analysis, "Region0", evidence_identity="requested-evidence-identity"
        )
        return "Region0"

    with h5py.File(cache_path, "r+") as handle:
        if mutation == "requested_global_consumed_robust":
            roi = "Region0"
            for group in _cache_chunks(handle, roi):
                group.attrs["correction_strategy_family"] = "dynamic_fit"
                group.attrs["correction_selected_strategy"] = "robust_global_event_reject"
                group.attrs["correction_dynamic_fit_mode"] = "robust_global_event_reject"
                group.attrs["dynamic_fit_mode_resolved"] = "robust_global_event_reject"
                group.attrs["dynamic_fit_engine"] = "robust_global_event_reject"
            return roi
        if mutation == "requested_signal_consumed_dynamic":
            roi = "Region1"
            for group in _cache_chunks(handle, roi):
                group.attrs["correction_strategy_family"] = "dynamic_fit"
                group.attrs["correction_selected_strategy"] = "global_linear_regression"
                group.attrs["correction_dynamic_fit_mode"] = "global_linear_regression"
                group.attrs["dynamic_fit_mode_resolved"] = "global_linear_regression"
                group.attrs["dynamic_fit_engine"] = "global_linear_regression"
                if "signal_only_f0_baseline" in group:
                    del group["signal_only_f0_baseline"]
                if "fit_ref" in group:
                    del group["fit_ref"]
                group.create_dataset(
                    "fit_ref", data=np.ones_like(group["time_sec"][()], dtype=float)
                )
            return roi
        if mutation == "requested_dynamic_consumed_signal":
            roi = "Region0"
            for group in _cache_chunks(handle, roi):
                group.attrs["correction_strategy_family"] = "signal_only_f0"
                group.attrs["correction_selected_strategy"] = "signal_only_f0"
                for attr_name in (
                    "correction_dynamic_fit_mode",
                    "dynamic_fit_mode_resolved",
                    "dynamic_fit_engine",
                ):
                    if attr_name in group.attrs:
                        del group.attrs[attr_name]
                if "fit_ref" in group:
                    del group["fit_ref"]
                if "signal_only_f0_baseline" in group:
                    del group["signal_only_f0_baseline"]
                group.create_dataset(
                    "signal_only_f0_baseline",
                    data=np.ones_like(group["time_sec"][()], dtype=float),
                )
            return roi
    raise AssertionError(f"unknown requested/consumed mutation: {mutation}")


@pytest.mark.parametrize("tool", ["dayplot", "impact"])
@pytest.mark.parametrize(
    "mutation",
    [
        "requested_global_consumed_robust",
        "requested_robust_consumed_global",
        "requested_signal_consumed_dynamic",
        "requested_dynamic_consumed_signal",
        "parameter_identity_mismatch",
        "evidence_identity_mismatch",
    ],
)
def test_prefinalization_plot_refuses_requested_consumed_mismatch(
    native_run, tmp_path, monkeypatch, capsys, tool, mutation
):
    analysis, _mode = native_run
    roi = _mutate_requested_consumed_mismatch(analysis, mutation)
    if tool == "dayplot":
        from tools import plot_phasic_dayplot_bundle as plotter

        output_dir = tmp_path / f"dayplot_{mutation}"
        monkeypatch.setattr(
            plotter.sys,
            "argv",
            [
                "plot_phasic_dayplot_bundle.py",
                "--analysis-out",
                str(analysis),
                "--roi",
                roi,
                "--output-dir",
                str(output_dir),
                "--sessions-per-hour",
                "1",
                "--no-write-dff-grid",
                "--no-write-stacked",
            ],
        )
        with pytest.raises(SystemExit):
            plotter.main()
        assert not list(output_dir.glob("*.png"))
    else:
        from tools import plot_phasic_correction_impact as plotter

        output_path = tmp_path / f"impact_{mutation}.png"
        monkeypatch.setattr(
            plotter.sys,
            "argv",
            [
                "plot_phasic_correction_impact.py",
                "--analysis-out",
                str(analysis),
                "--roi",
                roi,
                "--chunk-id",
                "0",
                "--out",
                str(output_path),
            ],
        )
        with pytest.raises(SystemExit):
            plotter.main()
        assert not output_path.exists()
    captured = capsys.readouterr()
    assert "requested-versus-consumed" in (
        captured.out + captured.err
    ).lower()


@pytest.mark.parametrize("tool", ["dayplot", "impact"])
def test_current_plot_refuses_unreadable_authoritative_chunk_index(
    native_run, tmp_path, monkeypatch, tool
):
    analysis, _mode = native_run
    with h5py.File(analysis / "phasic_trace_cache.h5", "r+") as handle:
        del handle["meta/chunk_ids"]
    if tool == "dayplot":
        from tools import plot_phasic_dayplot_bundle as plotter

        output_dir = tmp_path / "dayplot_missing_chunk_index"
        monkeypatch.setattr(
            plotter.sys,
            "argv",
            [
                "plot_phasic_dayplot_bundle.py",
                "--analysis-out",
                str(analysis),
                "--roi",
                "Region0",
                "--output-dir",
                str(output_dir),
                "--sessions-per-hour",
                "1",
                "--no-write-dff-grid",
                "--no-write-stacked",
            ],
        )
        with pytest.raises(Exception):
            plotter.main()
        assert not list(output_dir.glob("*.png"))
    else:
        from tools import plot_phasic_correction_impact as plotter

        output_path = tmp_path / "impact_missing_chunk_index.png"
        monkeypatch.setattr(
            plotter.sys,
            "argv",
            [
                "plot_phasic_correction_impact.py",
                "--analysis-out",
                str(analysis),
                "--roi",
                "Region0",
                "--chunk-id",
                "0",
                "--out",
                str(output_path),
            ],
        )
        with pytest.raises(SystemExit):
            plotter.main()
        assert not output_path.exists()


@pytest.mark.parametrize("tool", ["dayplot", "impact"])
def test_prefinalization_native_plot_entry_is_current_and_writes_output(
    native_run, tmp_path, monkeypatch, tool
):
    analysis, _mode = native_run
    assert classify_analysis_plot_source(analysis) == "current"
    context = resolve_analysis_plot_context(analysis)
    assert context.source_kind == "current"
    assert context.requested_by_roi["Region0"]["selected_strategy"] == (
        "global_linear_regression"
    )
    if tool == "dayplot":
        from tools import plot_phasic_dayplot_bundle as plotter

        output_dir = tmp_path / "day_plots"
        monkeypatch.setattr(
            plotter.sys,
            "argv",
            [
                "plot_phasic_dayplot_bundle.py",
                "--analysis-out",
                str(analysis),
                "--roi",
                "Region0",
                "--output-dir",
                str(output_dir),
                "--sessions-per-hour",
                "1",
                "--no-write-dff-grid",
                "--no-write-stacked",
            ],
        )
        plotter.main()
        assert (output_dir / "phasic_dynamic_fit_day_000.png").is_file()
    else:
        from tools import plot_phasic_correction_impact as plotter

        output_path = tmp_path / "correction_impact.png"
        monkeypatch.setattr(
            plotter.sys,
            "argv",
            [
                "plot_phasic_correction_impact.py",
                "--analysis-out",
                str(analysis),
                "--roi",
                "Region0",
                "--chunk-id",
                "0",
                "--out",
                str(output_path),
            ],
        )
        plotter.main()
        assert output_path.is_file()


@pytest.mark.parametrize("tool", ["dayplot", "impact"])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_metadata_provenance",
        "disagreeing_provenance",
        "missing_family",
        "missing_selected",
        "missing_signal_only_baseline",
        "missing_dynamic_fit_ref",
        "strategy_changes_across_sessions",
    ],
)
def test_prefinalization_native_plot_entry_refuses_malformed_evidence(
    native_run, tmp_path, monkeypatch, tool, mutation
):
    analysis, _mode = native_run
    _mutate_prefinal_native_analysis(analysis, mutation)
    output_dir = tmp_path / f"{tool}_{mutation}_out"
    if tool == "dayplot":
        from tools import plot_phasic_dayplot_bundle as plotter

        monkeypatch.setattr(
            plotter.sys,
            "argv",
            [
                "plot_phasic_dayplot_bundle.py",
                "--analysis-out",
                str(analysis),
                "--roi",
                "Region1" if "signal_only" in mutation or mutation in {"missing_family", "missing_selected", "strategy_changes_across_sessions"} else "Region0",
                "--output-dir",
                str(output_dir),
                "--sessions-per-hour",
                "1",
                "--no-write-dff-grid",
                "--no-write-stacked",
            ],
        )
        with pytest.raises(SystemExit):
            plotter.main()
        assert not list(output_dir.glob("*.png"))
    else:
        from tools import plot_phasic_correction_impact as plotter

        output_path = tmp_path / f"{tool}_{mutation}.png"
        monkeypatch.setattr(
            plotter.sys,
            "argv",
            [
                "plot_phasic_correction_impact.py",
                "--analysis-out",
                str(analysis),
                "--roi",
                "Region1" if "signal_only" in mutation or mutation in {"missing_family", "missing_selected", "strategy_changes_across_sessions"} else "Region0",
                "--chunk-id",
                "0",
                "--out",
                str(output_path),
            ],
        )
        with pytest.raises(SystemExit):
            plotter.main()
        assert not output_path.exists()


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
    with pytest.raises(CompletedRunReviewError, match="no verified native correction settings"):
        load_completed_phasic_review(root)


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


def test_completion_verifier_accepts_grid_aligned_nonzero_canonical_time_origin(
    native_run, tmp_path
):
    """The real production predicate this once was
    (guided_run_20260720T181438506426Z_1edeecfc21c6): NPM's strict grid
    (io.adapters._resolve_npm_strict_grid) is anchored to the UV/signal
    overlap origin and is "intentionally not re-zeroed to inner_start:
    staggered streams may therefore produce a first output time greater
    than zero" -- the real preserved run's chunk_0 legitimately started at
    0.02s (one full sample period at its real 50 Hz target rate), not
    0.0s. A canonical time_sec that starts at a *grid-aligned*, non-negative
    offset (an exact integer multiple of 1/fs_hz) must be accepted, not
    refused -- the completion verifier's genuine invariant is grid
    alignment, not a hardcoded zero."""
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, "grid_aligned_nonzero_origin")
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "r+") as handle:
        group = handle["roi/Region0/chunk_0"]
        fs = float(group.attrs["fs_hz"])
        time_sec = group["time_sec"][()]
        time_sec = time_sec + (1.0 / fs)  # exactly one sample period, grid-aligned
        group["time_sec"][...] = time_sec
    error = correction_completion_error(str(root), mode)
    assert error == "", error


def test_completion_verifier_rejects_non_grid_aligned_canonical_time_origin(
    native_run, tmp_path
):
    """A canonical time_sec[0] that is not an exact multiple of 1/fs_hz --
    i.e. not a real sample the strict grid construction could ever produce
    -- is still correctly refused."""
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, "non_grid_aligned_origin")
    _write_terminal_set(root, mode)
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "r+") as handle:
        group = handle["roi/Region0/chunk_0"]
        fs = float(group.attrs["fs_hz"])
        time_sec = group["time_sec"][()]
        # Half a sample period -- not an integer multiple of 1/fs_hz.
        time_sec = time_sec + (0.5 / fs)
        group["time_sec"][...] = time_sec
    error = correction_completion_error(str(root), mode)
    assert "invalid canonical time identity" in error
    assert "Region0" in error
    _write_terminal_set(root, mode)
    assert classify_run_terminal_state(str(root)).state == TERMINAL_CORRUPTED


def test_completion_verifier_rejects_negative_canonical_time_origin(
    native_run, tmp_path
):
    """A negative canonical time origin is never valid for either format's
    construction algorithm and must still be refused."""
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, "negative_origin")
    _write_terminal_set(root, mode)
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "r+") as handle:
        group = handle["roi/Region0/chunk_0"]
        fs = float(group.attrs["fs_hz"])
        time_sec = group["time_sec"][()]
        time_sec = time_sec - (1.0 / fs)
        group["time_sec"][...] = time_sec
    error = correction_completion_error(str(root), mode)
    assert "invalid canonical time identity" in error
    assert "Region0" in error
    _write_terminal_set(root, mode)
    assert classify_run_terminal_state(str(root)).state == TERMINAL_CORRUPTED


def test_completion_verifier_rejects_missing_fs_hz_attr_for_time_origin_check(
    native_run, tmp_path
):
    """A cache chunk missing the fs_hz attribute the grid-alignment check
    depends on must fail closed, not silently skip verification."""
    analysis, mode = native_run
    root = _root_for_case(tmp_path, analysis, "missing_fs_hz")
    _write_terminal_set(root, mode)
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "r+") as handle:
        group = handle["roi/Region0/chunk_0"]
        del group.attrs["fs_hz"]
    error = correction_completion_error(str(root), mode)
    assert "invalid canonical time identity" in error


def _write_npm_staggered_source(path: Path, *, n_pairs: int = 201, dt: float = 0.1) -> None:
    """A real-shaped NPM source with staggered UV/SIG timestamps: SIG runs
    0.03s (less than one full sample period) ahead of the corresponding UV
    sample, mirroring the real production dataset's own stagger (SIG
    started 0.016672s before UV in guided_run_20260720T181438506426Z_...).
    This is the exact condition that makes UV define the overlap origin
    while SIG's first in-window sample lands at a small positive offset,
    producing a grid-aligned but nonzero canonical time_sec[0]."""
    rows = []
    frame = 0
    base = 1000.0
    for i in range(n_pairs):
        uv_t = base + i * dt
        sig_t = base + i * dt - 0.03
        phase = 0.2 * i
        rows.append((frame, uv_t, 1, 2.0 + 0.15 * np.sin(phase), 1.5 + 0.12 * np.cos(phase + 0.3)))
        frame += 1
        rows.append((frame, sig_t, 2, 5.0 + 1.1 * np.sin(phase + 0.1), 4.0 + 0.9 * np.cos(phase + 0.4)))
        frame += 1
    # SIG rows are timestamped earlier than their paired UV row but written
    # in acquisition (frame) order, exactly like the real interleaved NPM
    # vendor format -- sort by timestamp is not assumed by the adapter.
    lines = ["FrameCounter,Timestamp,LedState,Region0G,Region1G"]
    for frame_counter, timestamp, led_state, r0, r1 in rows:
        lines.append(f"{frame_counter},{timestamp},{led_state},{r0},{r1}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def npm_native_run(tmp_path):
    source = tmp_path / "npm_input" / "photometryData2025-03-05T15_37_44.csv"
    _write_npm_staggered_source(source)
    cfg = Config(
        target_fs_hz=10.0,
        chunk_duration_sec=20.0,
        npm_time_axis="system_timestamp",
        npm_system_ts_col="Timestamp",
        npm_computer_ts_col="ComputerTimestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
        allow_partial_final_chunk=False,
        adapter_value_nan_policy="strict",
        timestamp_cv_max=0.05,
        lowpass_hz=2.0,
        filter_order=2,
        window_sec=10.0,
        min_samples_per_window=10,
        signal_only_f0_min_window_samples=21,
    )
    analysis = tmp_path / "npm_analysis"
    Pipeline(cfg, mode="phasic", per_roi_correction=_mixed_map()).run(
        str(source.parent), str(analysis), force_format="npm", recursive=False
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
    return analysis, mode, cfg


def test_real_staggered_npm_cache_has_grid_aligned_nonzero_origin_and_is_accepted(
    npm_native_run, tmp_path
):
    """End-to-end numerical/temporal equivalence proof with a real-shaped,
    staggered NPM fixture (not RWD): the real io.adapters NPM strict-grid
    path produces a genuinely nonzero, grid-aligned canonical time_sec[0]
    for this fixture, exactly as it does for the real preserved production
    dataset, and the repaired completion verifier accepts it -- proving the
    repair and the real NPM adapter path agree end to end, not merely in a
    hand-constructed HDF5 fixture."""
    analysis, mode, cfg = npm_native_run
    root = _root_for_case(tmp_path, analysis, "npm_staggered")
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"

    fs = cfg.target_fs_hz
    with h5py.File(cache_path, "r") as handle:
        for roi in ("Region0", "Region1"):
            group = handle[f"roi/{roi}/chunk_0"]
            time_sec = np.asarray(group["time_sec"][()])
            sig_raw = np.asarray(group["sig_raw"][()])
            uv_raw = np.asarray(group["uv_raw"][()])

            # The genuine repro of the real bug: nonzero, but grid-aligned.
            assert time_sec[0] > 0.0
            samples_from_origin = time_sec[0] * fs
            assert abs(samples_from_origin - round(samples_from_origin)) < 1e-9

            # Sample spacing is exactly 1/target_fs_hz throughout.
            np.testing.assert_allclose(np.diff(time_sec), 1.0 / fs, atol=1e-9)

            # No missing/duplicated samples; shapes agree.
            assert sig_raw.shape == time_sec.shape
            assert uv_raw.shape == time_sec.shape
            assert np.all(np.isfinite(sig_raw))
            assert np.all(np.isfinite(uv_raw))

    error = correction_completion_error(str(root), mode)
    assert error == "", error


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
    _write_terminal_set(root, mode)
    review = load_completed_phasic_review(root)
    sessions = review.sessions_for_roi("Region0")
    assert [session.disposition for session in sessions] == [
        "process",
        "authorized_missing_corrupted",
        "process",
    ]
    assert [session.chunk_id for session in sessions] == [0, None, 2]
    assert sessions[2].time_sec[0] == 0.0


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
    review = load_completed_phasic_review(root)
    assert [session.disposition for session in review.sessions_for_roi("Region0")] == [
        "process",
        "process",
        "authorized_exclusion",
    ]
    assert review.sessions_for_roi("Region0")[-1].chunk_id is None
