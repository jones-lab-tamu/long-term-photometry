from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from photometry_pipeline.config import Config
from photometry_pipeline.guided_sampling_rate import (
    GUIDED_SAMPLING_RATE_FAILURE_MESSAGE,
    GuidedSamplingRateError,
    normalize_guided_sampling_rate_hz,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


@pytest.mark.parametrize(
    ("inferred", "expected"),
    [
        (20.006401986, 20.0),
        (20.000400008, 20.0),
        (10.000300005, 10.0),
    ],
)
def test_reliable_rate_is_normalized_to_whole_hz(inferred, expected):
    assert normalize_guided_sampling_rate_hz(inferred) == expected


def test_unreliable_noninteger_rate_is_rejected():
    with pytest.raises(
        GuidedSamplingRateError,
        match="Sampling rate could not be determined reliably",
    ):
        normalize_guided_sampling_rate_hz(22.7)


def _write_npm_session(path: Path, *, rate_hz: float) -> None:
    step = 1.0 / (2.0 * rate_hz)
    rows = ["Timestamp,LedState,Region0G"]
    for index in range(240):
        rows.append(
            f"{index * step:.12f},{1 if index % 2 == 0 else 2},"
            f"{1.0 + index * 0.001:.6f}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_npm_candidate_uses_all_session_timestamps_not_50_hz_baseline(
    window, tmp_path
):
    for index in range(2):
        _write_npm_session(
            tmp_path / f"photometryData2026-01-0{index + 1}T12_00_00.csv",
            rate_hz=20.006401986,
        )
    baseline = Config.from_yaml("config/qc_universal_config.yaml")
    assert baseline.target_fs_hz == 50.0

    inferred = window._infer_npm_dataset_contract_overrides(
        "npm",
        input_path=str(tmp_path),
        baseline_config=baseline,
        whole_hz=True,
    )
    assert inferred["target_fs_hz"] == 20.0

    window._guided_input_dir_edit.setText(str(tmp_path))
    window._guided_format_combo.setCurrentText("npm")
    window._discovery_cache = {"resolved_format": "npm"}
    index = window._guided_acquisition_mode_combo.findData("intermittent")
    window._guided_acquisition_mode_combo.setCurrentIndex(index)
    window._guided_sessions_per_hour_edit.setText("1")
    window._guided_session_duration_edit.setText("6")
    candidate = window._guided_new_analysis_dataset_contract_candidate()

    assert candidate.status == "inferred"
    assert candidate.contract_values["target_fs_hz"] == 20.0
    window._guided_new_analysis_dataset_contract_snapshot = replace(
        candidate,
        status="applied",
        explicitly_applied=True,
    )
    assert window._guided_recording_target_fs_hz("npm") == 20.0
    assert window._guided_feature_preview_config_overrides(
        {"source_path": str(next(tmp_path.glob("*.csv")))},
        "npm",
    )["target_fs_hz"] == 20.0


def _write_csv_session(
    path: Path,
    *,
    rate_hz: float,
    milliseconds: bool = False,
) -> None:
    scale = 1000.0 if milliseconds else 1.0
    time_name = "ElapsedMilliseconds" if milliseconds else "ElapsedSeconds"
    rows = [f"{time_name},Signal,Reference"]
    for index in range(240):
        rows.append(
            f"{index / rate_hz * scale:.12f},"
            f"{2.0 + index * 0.001:.6f},{1.0 + index * 0.001:.6f}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.mark.parametrize("milliseconds", [False, True])
def test_csv_all_session_timestamps_produce_same_20_hz_target(
    window, tmp_path, milliseconds
):
    for index in range(2):
        _write_csv_session(
            tmp_path / f"session_{index + 1:04d}.csv",
            rate_hz=20.000400008,
            milliseconds=milliseconds,
        )
    interpretation = {
        "ordered_source_files": [
            "session_0001.csv",
            "session_0002.csv",
        ],
        "time_column": (
            "ElapsedMilliseconds" if milliseconds else "ElapsedSeconds"
        ),
        "time_scale_to_seconds": 0.001 if milliseconds else 1.0,
    }
    assert (
        window._infer_custom_tabular_dataset_target_fs_hz(
            interpretation,
            input_path=str(tmp_path),
        )
        == 20.0
    )


def test_csv_unreliable_or_inconsistent_rate_is_rejected(window, tmp_path):
    _write_csv_session(tmp_path / "session_0001.csv", rate_hz=22.7)
    interpretation = {
        "ordered_source_files": ["session_0001.csv"],
        "time_column": "ElapsedSeconds",
        "time_scale_to_seconds": 1.0,
    }
    with pytest.raises(
        GuidedSamplingRateError,
        match="Sampling rate could not be determined reliably",
    ):
        window._infer_custom_tabular_dataset_target_fs_hz(
            interpretation,
            input_path=str(tmp_path),
        )

    _write_csv_session(tmp_path / "session_0001.csv", rate_hz=20.0)
    _write_csv_session(tmp_path / "session_0002.csv", rate_hz=21.0)
    interpretation["ordered_source_files"] = [
        "session_0001.csv",
        "session_0002.csv",
    ]
    with pytest.raises(GuidedSamplingRateError) as exc_info:
        window._infer_custom_tabular_dataset_target_fs_hz(
            interpretation,
            input_path=str(tmp_path),
        )
    assert GUIDED_SAMPLING_RATE_FAILURE_MESSAGE in str(exc_info.value)


def test_rwd_feature_preview_uses_recording_rate_not_local_session_rate(
    window, monkeypatch
):
    window._guided_format_combo.setCurrentText("rwd")
    index = window._guided_acquisition_mode_combo.findData("intermittent")
    window._guided_acquisition_mode_combo.setCurrentIndex(index)
    window._rwd_contract_cache = {
        "overrides": {"target_fs_hz": 20.0},
    }
    monkeypatch.setattr(
        window,
        "_infer_rwd_chunk_contract",
        lambda _path: {
            "fs_hz": 20.040883402,
            "chunk_duration_sec": 600.0,
            "time_col": "TimeStamp",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
        },
    )

    overrides = window._guided_feature_preview_config_overrides(
        {"source_path": "session.csv"},
        "rwd",
    )
    assert overrides["target_fs_hz"] == 20.0
