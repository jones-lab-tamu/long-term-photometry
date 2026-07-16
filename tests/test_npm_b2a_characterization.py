import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from photometry_pipeline.completed_run_review import (
    CompletedRunReviewError,
    load_completed_phasic_review,
)
from photometry_pipeline.config import Config
from photometry_pipeline.discovery import discover_inputs
from photometry_pipeline.io.adapters import load_chunk
from photometry_pipeline.pipeline import Pipeline


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "npm_b2a"


def _npm_config(**overrides) -> Config:
    values = dict(
        allow_partial_final_chunk=True,
        target_fs_hz=2.0,
        chunk_duration_sec=1.0,
        npm_time_axis="system_timestamp",
        npm_system_ts_col="SystemTimestamp",
        npm_computer_ts_col="ComputerTimestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
    )
    values.update(overrides)
    return Config(**values)


def _write_generated_npm_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "chunk_duration_sec: 600",
                "target_fs_hz: 20",
                "baseline_method: uv_raw_percentile_session",
                "baseline_percentile: 10",
                "npm_time_axis: system_timestamp",
                "npm_frame_col: FrameCounter",
                "npm_system_ts_col: SystemTimestamp",
                "npm_computer_ts_col: ComputerTimestamp",
                "npm_led_col: LedState",
                "npm_region_prefix: Region",
                "npm_region_suffix: G",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=Path.cwd(), capture_output=True, text=True)


def test_current_npm_discovery_uses_embedded_filename_timestamp_order():
    cfg = _npm_config()
    input_dir = FIXTURE_ROOT / "basic"

    discovered = discover_inputs(str(input_dir), cfg, force_format="npm")
    pipeline = Pipeline(cfg)
    pipeline.discover_files(str(input_dir), force_format="npm")

    expected = [
        "z_vendor2025-03-05T15_38_59.csv",
        "a_vendor2025-03-05T15_39_00.csv",
        "m_vendor2025-03-05T15_39_05.csv",
    ]
    assert [Path(item["path"]).name for item in discovered["sessions"]] == expected
    assert [Path(path).name for path in pipeline.file_list] == expected
    assert [item["session_id"] for item in discovered["sessions"]] == [
        Path(name).stem for name in expected
    ]
    assert discovered["n_total_discovered"] == 3
    assert all("gap" not in item for item in discovered["sessions"])


def test_current_npm_session_identity_is_stem_and_relocation_changes_source_path(tmp_path):
    cfg = _npm_config()
    original = FIXTURE_ROOT / "basic" / "z_vendor2025-03-05T15_38_59.csv"
    relocated_dir = tmp_path / "relocated"
    relocated_dir.mkdir()
    relocated = relocated_dir / original.name
    shutil.copy2(original, relocated)

    original_session = discover_inputs(
        str(original.parent), cfg, force_format="npm"
    )["sessions"][0]
    relocated_session = discover_inputs(
        str(relocated.parent), cfg, force_format="npm"
    )["sessions"][0]

    assert original_session["session_id"] == relocated_session["session_id"]
    assert original_session["path"] != relocated_session["path"]
    # The current durable C8/cache source identity is the absolute source path,
    # so relocation preserves the stem but not the source identity.


def test_current_npm_loader_maps_led_states_to_uv_and_signal_on_one_roi_column():
    cfg = _npm_config()
    path = FIXTURE_ROOT / "basic" / "z_vendor2025-03-05T15_38_59.csv"

    chunk = load_chunk(str(path), "npm", cfg, chunk_id=0)

    assert chunk.channel_names == ["Region0"]
    np.testing.assert_allclose(chunk.uv_raw[:, 0], [10.0, 11.0])
    np.testing.assert_allclose(chunk.sig_raw[:, 0], [100.0, 101.0])
    np.testing.assert_allclose(chunk.time_sec, [0.0, 0.5])
    assert chunk.fs_hz == 2.0
    assert chunk.metadata["session_time"]["session_id"] == path.stem
    assert chunk.metadata["roi_map"] == {"Region0": {"raw_col": "Region0G"}}
    for key in (
        "resolved_time_col",
        "resolved_led_col",
        "source_fs_hz",
        "output_time_basis",
        "signal_channel_identity",
        "reference_channel_identity",
    ):
        assert key not in chunk.metadata


def test_current_npm_loader_discovers_multiple_rois_by_natural_region_order():
    cfg = _npm_config()
    path = FIXTURE_ROOT / "multi_roi" / "photometryData2025-03-05T16_00_00.csv"

    chunk = load_chunk(str(path), "npm", cfg, chunk_id=0)

    assert chunk.channel_names == ["Region0", "Region1"]
    assert chunk.metadata["roi_map"] == {
        "Region0": {"raw_col": "Region0G"},
        "Region1": {"raw_col": "Region1G"},
    }
    np.testing.assert_allclose(chunk.uv_raw[:, 0], [10.0, 11.0])
    np.testing.assert_allclose(chunk.sig_raw[:, 0], [100.0, 101.0])
    np.testing.assert_allclose(chunk.uv_raw[:, 1], [20.0, 21.0])
    np.testing.assert_allclose(chunk.sig_raw[:, 1], [200.0, 201.0])


def test_npm_preview_load_reads_only_the_selected_roi(monkeypatch):
    cfg = _npm_config()
    path = FIXTURE_ROOT / "multi_roi" / "photometryData2025-03-05T16_00_00.csv"
    from photometry_pipeline.io import adapters

    real_read_csv = adapters.pd.read_csv
    calls = []

    def recording_read_csv(*args, **kwargs):
        calls.append(dict(kwargs))
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(adapters.pd, "read_csv", recording_read_csv)

    chunk = load_chunk(
        str(path), "npm", cfg, chunk_id=0, selected_roi="Region1"
    )

    assert chunk.channel_names == ["Region1"]
    assert chunk.metadata["roi_map"] == {"Region1": {"raw_col": "Region1G"}}
    assert calls[0].get("nrows") == 0
    assert calls[1]["usecols"] == ["Timestamp", "LedState", "Region1G"]
    np.testing.assert_allclose(chunk.uv_raw[:, 0], [20.0, 21.0])
    np.testing.assert_allclose(chunk.sig_raw[:, 0], [200.0, 201.0])


def test_npm_preview_load_rejects_an_unknown_selected_roi():
    cfg = _npm_config()
    path = FIXTURE_ROOT / "multi_roi" / "photometryData2025-03-05T16_00_00.csv"

    with pytest.raises(ValueError, match="Requested ROI 'Region9' is not present"):
        load_chunk(str(path), "npm", cfg, chunk_id=0, selected_roi="Region9")


def test_current_npm_malformed_missing_signal_refuses():
    cfg = _npm_config()
    path = FIXTURE_ROOT / "malformed" / "missing_signal.csv"

    with pytest.raises(ValueError, match="NPM: Insufficient data"):
        load_chunk(str(path), "npm", cfg, chunk_id=0)


def test_current_npm_phasic_pipeline_persists_cache_and_c8_shape(tmp_path):
    cfg = _npm_config()
    output_dir = tmp_path / "phasic_out"

    Pipeline(cfg, mode="phasic").run(
        str(FIXTURE_ROOT / "basic"),
        str(output_dir),
        force_format="npm",
        traces_only=True,
    )

    completeness = json.loads(
        (output_dir / "input_processing_completeness.json").read_text(
            encoding="utf-8"
        )
    )
    assert completeness["input_format"] == "npm"
    assert len(completeness["expected"]) == 3
    assert len(completeness["processed"]) == 3
    assert completeness["missing"] == []

    with h5py.File(output_dir / "phasic_trace_cache.h5", "r") as cache:
        assert cache["meta"].attrs["mode"] == "phasic"
        attrs = cache["roi"]["Region0"]["chunk_0"].attrs
        assert attrs["fs_hz"] == 2.0
        assert str(attrs["source_file"]).endswith(
            "z_vendor2025-03-05T15_38_59.csv"
        )
        for key in (
            "resolved_time_col",
            "source_fs_hz",
            "output_time_basis",
            "signal_channel_identity",
            "reference_channel_identity",
        ):
            assert key not in attrs


def test_current_npm_combined_wrapper_runs_tonic_and_phasic_but_review_refuses(
    tmp_path,
):
    config_path = tmp_path / "npm.yaml"
    input_dir = tmp_path / "npm_data"
    output_dir = tmp_path / "combined_out"
    _write_generated_npm_config(config_path)

    generated = _run(
        [
            sys.executable,
            "tools/synth_photometry_dataset.py",
            "--out",
            str(input_dir),
            "--format",
            "npm",
            "--config",
            str(config_path),
            "--total-days",
            "0.05",
            "--recording-duration-min",
            "10",
            "--recordings-per-hour",
            "2",
            "--fs-hz",
            "20",
            "--n-rois",
            "1",
            "--start-iso",
            "2025-03-05T15:37:44",
            "--seed",
            "123",
        ]
    )
    assert generated.returncode == 0, (
        f"NPM generation failed\nSTDOUT:\n{generated.stdout}\n"
        f"STDERR:\n{generated.stderr}"
    )

    wrapped = _run(
        [
            sys.executable,
            "tools/run_full_pipeline_deliverables.py",
            "--input",
            str(input_dir),
            "--config",
            str(config_path),
            "--format",
            "npm",
            "--mode",
            "both",
            "--out",
            str(output_dir),
            "--overwrite",
            "--sessions-per-hour",
            "2",
        ]
    )
    assert wrapped.returncode == 0, (
        f"NPM combined wrapper failed\nSTDOUT:\n{wrapped.stdout}\n"
        f"STDERR:\n{wrapped.stderr}"
    )

    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert (output_dir / "_analysis" / "tonic_out" / "tonic_trace_cache.h5").is_file()
    assert (output_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").is_file()
    for mode in ("tonic_out", "phasic_out"):
        completeness = json.loads(
            (output_dir / "_analysis" / mode / "input_processing_completeness.json").read_text(
                encoding="utf-8"
            )
        )
        assert completeness["input_format"] == "npm"
        assert len(completeness["processed"]) == 2

    with pytest.raises(
        CompletedRunReviewError, match="missing canonical data for ROI/session"
    ):
        load_completed_phasic_review(output_dir)
