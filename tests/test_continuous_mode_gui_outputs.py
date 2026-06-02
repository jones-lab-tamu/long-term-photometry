import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from gui.run_report_parser import is_successful_completed_run_dir, resolve_region_deliverables
from gui.run_report_viewer import RunReportViewer


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_custom_tabular_csv(path: Path, duration_sec: float, fs_hz: float = 10.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    iso = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 2.0 + 0.9 * iso + 0.04 * np.sin(2.0 * np.pi * 0.08 * t + 0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_sec": t, "Region0_iso": iso, "Region0_sig": sig}).to_csv(
        path,
        index=False,
    )


def _write_continuous_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "target_fs_hz: 10.0",
                "chunk_duration_sec: 600.0",
                "allow_partial_final_chunk: false",
                "acquisition_mode: continuous",
                "continuous_window_sec: 600.0",
                "continuous_step_sec: 600.0",
                "allow_partial_final_window: false",
                "custom_tabular_time_col: time_sec",
                "custom_tabular_uv_suffix: _iso",
                "custom_tabular_sig_suffix: _sig",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_continuous_wrapper(tmp_path: Path, mode: str) -> Path:
    input_dir = tmp_path / f"input_{mode}"
    out_dir = tmp_path / f"out_{mode}"
    cfg_path = tmp_path / f"continuous_{mode}.yaml"
    _write_custom_tabular_csv(input_dir / "session_000.csv", duration_sec=1200.0)
    _write_continuous_config(cfg_path)
    cmd = [
        sys.executable,
        "tools/run_full_pipeline_deliverables.py",
        "--input",
        str(input_dir),
        "--out",
        str(out_dir),
        "--config",
        str(cfg_path),
        "--format",
        "custom_tabular",
        "--mode",
        mode,
        "--overwrite",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    return out_dir


def _assert_continuous_region_deliverables(out_dir: Path) -> None:
    ok, reason = is_successful_completed_run_dir(str(out_dir))
    assert ok, reason

    regions = resolve_region_deliverables(str(out_dir))
    assert len(regions) == 1
    assert regions[0]["name"] == "Region0"
    labels = {label for label, _path, status in regions[0]["subfolders"] if status == "ok"}
    assert labels == {"Summary", "Tables"}
    assert not (out_dir / "Region0" / "day_plots").exists()


def _load_viewer_tab_map(qapp, out_dir: Path) -> dict[str, list[str]]:
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(out_dir)) is True
        assert viewer.available_regions() == ["Region0"]
        assert "No region deliverables" not in viewer._status_label.text()
        tab_map = viewer._region_tab_images["Region0"]
        return {
            tab: [Path(path).name for path in paths]
            for tab, paths in tab_map.items()
        }
    finally:
        viewer.close()


def test_gui_completed_run_parser_accepts_continuous_phasic_summary_outputs(
    tmp_path: Path,
    qapp,
):
    out_dir = _run_continuous_wrapper(tmp_path, "phasic")

    _assert_continuous_region_deliverables(out_dir)
    assert (out_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv").exists()
    assert (out_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png").exists()
    assert (out_dir / "Region0" / "summary" / "phasic_auc_timeseries.png").exists()

    tab_map = _load_viewer_tab_map(qapp, out_dir)
    assert tab_map["Phasic Summary"] == [
        "phasic_auc_timeseries.png",
        "phasic_peak_rate_timeseries.png",
    ]
    assert tab_map["Tonic"] == []
    assert tab_map["Phasic Sig/Iso"] == []
    assert tab_map["Dynamic Fit"] == []
    assert tab_map["Phasic dFF"] == []
    assert tab_map["Phasic Stacked"] == []


def test_gui_completed_run_parser_accepts_continuous_both_summary_outputs(
    tmp_path: Path,
    qapp,
):
    out_dir = _run_continuous_wrapper(tmp_path, "both")

    _assert_continuous_region_deliverables(out_dir)
    assert (out_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv").exists()
    assert (out_dir / "Region0" / "tables" / "continuous_tonic_window_summary.csv").exists()
    assert (out_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png").exists()
    assert (out_dir / "Region0" / "summary" / "phasic_auc_timeseries.png").exists()
    assert (out_dir / "Region0" / "summary" / "tonic_overview.png").exists()

    tab_map = _load_viewer_tab_map(qapp, out_dir)
    assert tab_map["Tonic"] == ["tonic_overview.png"]
    assert tab_map["Phasic Summary"] == [
        "phasic_auc_timeseries.png",
        "phasic_peak_rate_timeseries.png",
    ]
    assert tab_map["Phasic Sig/Iso"] == []
    assert tab_map["Dynamic Fit"] == []
    assert tab_map["Phasic dFF"] == []
    assert tab_map["Phasic Stacked"] == []


def test_gui_completed_run_parser_accepts_continuous_tonic_only_summary_outputs(
    tmp_path: Path,
    qapp,
):
    out_dir = _run_continuous_wrapper(tmp_path, "tonic")

    _assert_continuous_region_deliverables(out_dir)
    assert (out_dir / "Region0" / "tables" / "continuous_tonic_window_summary.csv").exists()
    assert (out_dir / "Region0" / "summary" / "tonic_overview.png").exists()
    assert not (out_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png").exists()
    assert not (out_dir / "Region0" / "summary" / "phasic_auc_timeseries.png").exists()

    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert any(
        skip["reason"] == "phasic mode not requested"
        for skip in manifest["continuous_outputs"]["plot_skips"]
    )

    tab_map = _load_viewer_tab_map(qapp, out_dir)
    assert tab_map["Tonic"] == ["tonic_overview.png"]
    assert tab_map["Phasic Summary"] == []
    assert tab_map["Phasic Sig/Iso"] == []
    assert tab_map["Dynamic Fit"] == []
    assert tab_map["Phasic dFF"] == []
    assert tab_map["Phasic Stacked"] == []
