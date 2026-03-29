import dataclasses
import hashlib
import json
import os
import sys

import h5py
import numpy as np
import pytest
import yaml
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QGroupBox, QSizePolicy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gui.main_window import MainWindow
from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def _make_completed_run_with_cache(
    tmp_path,
    *,
    with_config: bool = True,
    roi_chunk_keep: dict | None = None,
    chunk_ids: tuple[int, ...] = (0, 1, 2),
):
    run_dir = tmp_path / "run_complete"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True, exist_ok=True)

    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")

    cfg = Config()
    if with_config:
        with open(phasic_out / "config_used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=True)

    cache_path = phasic_out / "phasic_trace_cache.h5"
    with Hdf5TraceCacheWriter(str(cache_path), "phasic", cfg) as writer:
        t = np.arange(0, 240.0, 1.0)
        for cid in chunk_ids:
            sig_r0 = np.sin(0.07 * t) + 0.2 * np.sin(0.4 * t) + (cid * 0.05)
            uv_r0 = 0.4 * np.sin(0.07 * t + 0.3) + 0.2
            delta_r0 = sig_r0 - uv_r0
            dff_r0 = 100.0 * delta_r0 / 30.0

            sig_r1 = np.cos(0.05 * t) + 0.15 * np.cos(0.5 * t)
            uv_r1 = 0.3 * np.cos(0.05 * t + 0.2) + 0.25
            delta_r1 = sig_r1 - uv_r1
            dff_r1 = 100.0 * delta_r1 / 25.0

            chunk = Chunk(
                chunk_id=cid,
                source_file=f"session_{cid}.csv",
                format="cache",
                time_sec=t,
                uv_raw=np.column_stack([uv_r0, uv_r1]),
                sig_raw=np.column_stack([sig_r0, sig_r1]),
                fs_hz=1.0,
                channel_names=["Region0", "Region1"],
            )
            chunk.delta_f = np.column_stack([delta_r0, delta_r1])
            chunk.dff = np.column_stack([dff_r0, dff_r1])
            writer.add_chunk(chunk, chunk_id=cid, source_file=f"session_{cid}.csv")

    if roi_chunk_keep:
        with h5py.File(cache_path, "a") as h5:
            for roi_name, keep in roi_chunk_keep.items():
                grp = h5.get(f"roi/{roi_name}")
                if grp is None:
                    continue
                keep_set = {int(x) for x in keep}
                for key in list(grp.keys()):
                    if not str(key).startswith("chunk_"):
                        continue
                    try:
                        cid = int(str(key).split("_", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    if cid not in keep_set:
                        del grp[key]
    return run_dir


def _make_completed_run_with_quantized_preview_cache(tmp_path, *, lowpass_hz: float = 1.0):
    run_dir = tmp_path / "run_complete_quantized_preview"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True, exist_ok=True)

    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")

    cfg = Config(
        target_fs_hz=20.0,
        lowpass_hz=float(lowpass_hz),
        event_signal="dff",
        peak_threshold_method="mean_std",
        peak_threshold_k=2.0,
        peak_min_distance_sec=0.5,
    )
    with open(phasic_out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=True)

    cache_path = phasic_out / "phasic_trace_cache.h5"
    with Hdf5TraceCacheWriter(str(cache_path), "phasic", cfg) as writer:
        t_true = np.arange(0.0, 60.0, 1.0 / cfg.target_fs_hz)
        t_quantized = np.floor(t_true)
        signal = 0.35 * np.sin(2.0 * np.pi * 0.3 * t_true) + 0.85 * np.sin(2.0 * np.pi * 4.5 * t_true)
        chunk = Chunk(
            chunk_id=0,
            source_file="session_0.csv",
            format="cache",
            time_sec=t_quantized,
            uv_raw=np.zeros((len(t_quantized), 1), dtype=float),
            sig_raw=np.zeros((len(t_quantized), 1), dtype=float),
            fs_hz=cfg.target_fs_hz,
            channel_names=["Region0"],
        )
        chunk.delta_f = signal.reshape(-1, 1)
        chunk.dff = signal.reshape(-1, 1)
        writer.add_chunk(chunk, chunk_id=0, source_file="session_0.csv")
    return run_dir


def _qimage_sha256(image) -> str:
    payload = QByteArray()
    buffer = QBuffer(payload)
    assert buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    assert image.save(buffer, "PNG")
    buffer.close()
    return hashlib.sha256(bytes(payload)).hexdigest()


def _qimage_diff_counts(image_a, image_b) -> tuple[int, int]:
    """
    Return (total_diff_pixels, diff_pixels_outside_legend_zone).

    The legend occupies the lower-left corner in retune overlays; requiring
    differences outside that zone prevents legend-only regressions.
    """
    a = image_a.convertToFormat(QImage.Format.Format_RGBA8888)
    b = image_b.convertToFormat(QImage.Format.Format_RGBA8888)
    assert a.size() == b.size()
    h, w = a.height(), a.width()
    ba = np.frombuffer(a.constBits(), dtype=np.uint8).reshape((h, w, 4))
    bb = np.frombuffer(b.constBits(), dtype=np.uint8).reshape((h, w, 4))
    diff = np.any(ba != bb, axis=2)
    total = int(np.count_nonzero(diff))

    # Approximate legend zone: lower-left area where text key is rendered.
    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[int(h * 0.72):, : int(w * 0.34)] = True
    outside = int(np.count_nonzero(diff & ~legend_mask))
    return total, outside


def _write_png(path, width: int = 320, height: int = 180) -> None:
    from PySide6.QtGui import QPixmap

    pix = QPixmap(width, height)
    pix.fill()
    assert pix.save(str(path))


def _add_results_workspace_artifacts(run_dir) -> None:
    run_dir = os.fspath(run_dir)
    report_path = os.path.join(run_dir, "run_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "success",
                "phase": "final",
                "run_context": {"status": "success", "phase": "final"},
            },
            f,
        )

    for region in ("Region0", "Region1"):
        summary_dir = os.path.join(run_dir, region, "summary")
        day_dir = os.path.join(run_dir, region, "day_plots")
        tables_dir = os.path.join(run_dir, region, "tables")
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(day_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        _write_png(os.path.join(summary_dir, "phasic_correction_impact.png"), 900, 420)
        _write_png(os.path.join(summary_dir, "tonic_overview.png"), 900, 420)
        _write_png(os.path.join(summary_dir, "phasic_auc_timeseries.png"), 900, 420)
        _write_png(os.path.join(summary_dir, "phasic_peak_rate_timeseries.png"), 900, 420)
        _write_png(os.path.join(day_dir, "phasic_sig_iso_day_001.png"), 700, 1800)
        _write_png(os.path.join(day_dir, "phasic_sig_iso_day_002.png"), 700, 1800)
        _write_png(os.path.join(day_dir, "phasic_dff_day_001.png"), 700, 1800)
        _write_png(os.path.join(day_dir, "phasic_stacked_day_001.png"), 700, 1800)


def test_tuning_workspace_availability_requires_prerequisites(window, tmp_path):
    window._is_complete_workspace_active = False
    window._current_run_dir = ""
    window._refresh_tuning_workspace_availability()
    assert window._tuning_group.isHidden()

    # Complete state active, but no valid run dir yet.
    window._is_complete_workspace_active = True
    window._current_run_dir = str(tmp_path / "missing")
    window._refresh_tuning_workspace_availability()
    assert not window._tuning_workspace_available
    assert "No completed run directory is active" in window._tuning_availability_label.text()

    # Run dir exists but phasic output tree missing.
    run_dir_missing_phasic = tmp_path / "run_missing_phasic"
    run_dir_missing_phasic.mkdir(parents=True, exist_ok=True)
    (run_dir_missing_phasic / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir_missing_phasic / "MANIFEST.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    window._current_run_dir = str(run_dir_missing_phasic)
    window._refresh_tuning_workspace_availability()
    assert not window._tuning_workspace_available
    assert "_analysis/phasic_out" in window._tuning_availability_label.text()

    # phasic_out exists but cache missing.
    run_dir_missing_cache = tmp_path / "run_missing_cache"
    (run_dir_missing_cache / "_analysis" / "phasic_out").mkdir(parents=True, exist_ok=True)
    (run_dir_missing_cache / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir_missing_cache / "MANIFEST.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    window._current_run_dir = str(run_dir_missing_cache)
    window._refresh_tuning_workspace_availability()
    assert not window._tuning_workspace_available
    assert "phasic cache is missing" in window._tuning_availability_label.text()

    # cache exists but config snapshot missing.
    run_dir_missing_cfg = _make_completed_run_with_cache(tmp_path / "missing_cfg", with_config=False)
    window._current_run_dir = str(run_dir_missing_cfg)
    window._refresh_tuning_workspace_availability()
    assert not window._tuning_workspace_available
    assert "config_used.yaml" in window._tuning_availability_label.text()

    # All prerequisites present -> available.
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    assert window._tuning_workspace_available
    assert window._tuning_roi_combo.count() >= 1
    assert window._tuning_chunk_combo.count() >= 1


def test_tuning_workspace_missing_config_snapshot_blocks(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path, with_config=False)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    assert not window._tuning_workspace_available
    msg = window._tuning_availability_label.text()
    assert "missing config snapshot" in msg.lower()
    assert "config_used.yaml" in msg


def test_tuning_workspace_roi_specific_chunk_population(window, tmp_path):
    run_dir = _make_completed_run_with_cache(
        tmp_path,
        roi_chunk_keep={
            "Region0": [0, 2],
            "Region1": [1],
        },
    )
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    assert window._tuning_workspace_available

    window._tuning_roi_combo.setCurrentText("Region0")
    region0_chunks = [window._tuning_chunk_combo.itemText(i) for i in range(window._tuning_chunk_combo.count())]
    assert region0_chunks == ["0", "2"]

    window._tuning_roi_combo.setCurrentText("Region1")
    region1_chunks = [window._tuning_chunk_combo.itemText(i) for i in range(window._tuning_chunk_combo.count())]
    assert region1_chunks == ["1"]


def test_downstream_tuning_control_population_includes_hardening_defaults(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path / "downstream_defaults")
    cfg_path = run_dir / "_analysis" / "phasic_out" / "config_used.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update(
        {
            "event_signal": "delta_f",
            "peak_threshold_method": "absolute",
            "peak_threshold_k": 4.2,
            "peak_threshold_percentile": 92.0,
            "peak_threshold_abs": 0.33,
            "peak_min_distance_sec": 1.4,
            "peak_min_prominence_k": 1.75,
            "peak_min_width_sec": 0.28,
            "peak_pre_filter": "lowpass",
            "event_auc_baseline": "median",
        }
    )
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)

    # Seed a stale mixed state first, then confirm full downstream defaults re-apply.
    window._tuning_event_signal_combo.setCurrentText("dff")
    window._tuning_peak_method_combo.setCurrentText("mean_std")
    window._tuning_peak_k_spin.setValue(8.8)
    window._tuning_peak_pct_spin.setValue(22.0)
    window._tuning_peak_abs_spin.setValue(2.2)
    window._tuning_peak_dist_spin.setValue(8.8)
    window._tuning_peak_prominence_k_spin.setValue(0.11)
    window._tuning_peak_width_sec_spin.setValue(1.11)
    if window._tuning_peak_pre_filter_combo.findText("none") >= 0:
        window._tuning_peak_pre_filter_combo.setCurrentText("none")
    if window._tuning_event_auc_combo.findText("zero") >= 0:
        window._tuning_event_auc_combo.setCurrentText("zero")

    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    QApplication.processEvents()

    prefilter_options = [
        window._tuning_peak_pre_filter_combo.itemText(i)
        for i in range(window._tuning_peak_pre_filter_combo.count())
    ]
    assert "smooth" in prefilter_options
    assert "lowpass" not in prefilter_options

    assert window._tuning_event_signal_combo.currentText() == "delta_f"
    assert window._tuning_peak_method_combo.currentText() == "absolute"
    assert window._tuning_peak_k_spin.value() == pytest.approx(4.2)
    assert window._tuning_peak_pct_spin.value() == pytest.approx(92.0)
    assert window._tuning_peak_abs_spin.value() == pytest.approx(0.33)
    assert window._tuning_peak_dist_spin.value() == pytest.approx(1.4)
    assert window._tuning_peak_prominence_k_spin.value() == pytest.approx(1.75)
    assert window._tuning_peak_width_sec_spin.value() == pytest.approx(0.28)
    assert window._tuning_peak_pre_filter_combo.currentText() == "smooth"
    assert window._tuning_event_auc_combo.currentText() == "median"


def test_tuning_workspace_disclosure_collapsed_by_default(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._current_run_dir = str(run_dir)
    window._enter_complete_state_workspace()
    QApplication.processEvents()

    assert not window._tuning_disclosure_btn.isChecked()
    assert window._tuning_disclosure_btn.arrowType() == Qt.RightArrow
    assert window._tuning_content.isHidden()
    assert not window._tuning_collapsed_status_label.isHidden()
    assert window._tuning_availability_label.isHidden()


def test_tuning_hierarchy_normalized_no_nested_correction_groupbox(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._current_run_dir = str(run_dir)
    window._enter_complete_state_workspace()
    QApplication.processEvents()

    child_group_boxes = window._tuning_group.findChildren(QGroupBox)
    assert all(box.title() != "Correction-Sensitive Retune" for box in child_group_boxes)


def test_tuning_workspace_disclosure_toggle(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._current_run_dir = str(run_dir)
    window._enter_complete_state_workspace()
    QApplication.processEvents()
    assert not window._report_viewer.isHidden()

    window._tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._tuning_disclosure_btn.isChecked()
    assert window._tuning_disclosure_btn.arrowType() == Qt.DownArrow
    assert not window._tuning_content.isHidden()
    assert not window._tuning_availability_label.isHidden()
    assert window._tuning_collapsed_status_label.isHidden()
    assert window._report_viewer.isHidden()

    window._tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert not window._tuning_disclosure_btn.isChecked()
    assert window._tuning_disclosure_btn.arrowType() == Qt.RightArrow
    assert window._tuning_content.isHidden()
    assert not window._tuning_collapsed_status_label.isHidden()
    assert window._tuning_availability_label.isHidden()
    assert not window._report_viewer.isHidden()


def test_tuning_disclosure_resets_collapsed_on_complete_state_entry(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._current_run_dir = str(run_dir)
    window._enter_complete_state_workspace()
    window._set_tuning_disclosure_expanded(True)
    assert not window._tuning_content.isHidden()

    window._exit_complete_state_workspace()
    window._enter_complete_state_workspace()
    QApplication.processEvents()

    assert not window._tuning_disclosure_btn.isChecked()
    assert window._tuning_content.isHidden()


def test_tuning_workspace_internal_scroll_and_geometry_sanity(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._current_run_dir = str(run_dir)
    window._enter_complete_state_workspace()
    window._set_tuning_disclosure_expanded(True)
    window.resize(1300, 900)
    window.show()
    QApplication.processEvents()

    assert window._tuning_scroll.widgetResizable()
    assert window._tuning_scroll.widget() is window._tuning_scroll_content

    assert window._tuning_roi_combo.minimumWidth() >= 200
    assert window._tuning_chunk_combo.minimumWidth() >= 200
    assert window._tuning_peak_method_combo.minimumWidth() >= 200
    assert window._tuning_peak_dist_spin.minimumWidth() >= 120

    assert window._tuning_roi_combo.width() >= 150
    assert window._tuning_chunk_combo.width() >= 150
    assert window._tuning_peak_method_combo.width() >= 150
    assert window._tuning_peak_dist_spin.width() >= 110


def test_tuning_mode_switch_preserves_results_region_tab_and_image(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    _add_results_workspace_artifacts(run_dir)
    window._current_run_dir = str(run_dir)
    assert window._report_viewer.load_report(str(run_dir))
    window._enter_complete_state_workspace()
    window.resize(1400, 900)
    window.show()
    QApplication.processEvents()

    window._report_viewer._region_combo.setCurrentText("Region1")
    phasic_raw_idx = -1
    for i in range(window._report_viewer._tabs.count()):
        if window._report_viewer._tabs.tabText(i) == "Phasic Sig/Iso":
            phasic_raw_idx = i
            break
    assert phasic_raw_idx >= 0
    window._report_viewer._tabs.setCurrentIndex(phasic_raw_idx)
    QApplication.processEvents()

    window._report_viewer._on_next_image()
    before_region = window._report_viewer.selected_region()
    before_tab = window._report_viewer._selected_tab()
    before_key = (before_region, before_tab)
    before_idx = window._report_viewer._tab_indices.get(before_key)
    assert before_idx == 1

    window._tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._report_viewer.isHidden()
    assert window._tuning_roi_combo.currentText() == before_region

    window._tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert not window._report_viewer.isHidden()
    assert window._report_viewer.selected_region() == before_region
    assert window._report_viewer._selected_tab() == before_tab
    assert window._report_viewer._tab_indices.get(before_key) == before_idx


def test_tuning_expanded_mode_uses_meaningful_results_pane_height(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    _add_results_workspace_artifacts(run_dir)
    window._current_run_dir = str(run_dir)
    assert window._report_viewer.load_report(str(run_dir))
    window._enter_complete_state_workspace()
    window.resize(1400, 900)
    window.show()
    QApplication.processEvents()

    collapsed_height = window._tuning_group.height()
    window._tuning_disclosure_btn.click()
    QApplication.processEvents()

    parent_height = window._tuning_group.parentWidget().height()
    assert window._report_viewer.isHidden()
    assert window._tuning_group.height() > collapsed_height + 80
    assert window._tuning_group.height() >= int(parent_height * 0.55)


def test_correction_disclosure_alone_triggers_tuning_mode(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    _add_results_workspace_artifacts(run_dir)
    window._current_run_dir = str(run_dir)
    assert window._report_viewer.load_report(str(run_dir))
    window._enter_complete_state_workspace()
    window.resize(1400, 900)
    window.show()
    QApplication.processEvents()

    assert not window._tuning_disclosure_btn.isChecked()
    assert not window._correction_tuning_disclosure_btn.isChecked()
    assert not window._report_viewer.isHidden()

    collapsed_height = window._tuning_group.height()
    window._correction_tuning_disclosure_btn.click()
    QApplication.processEvents()

    parent_height = window._tuning_group.parentWidget().height()
    assert window._report_viewer.isHidden()
    assert window._correction_tuning_content.isVisible()
    assert window._tuning_group.height() > collapsed_height + 80
    assert window._tuning_group.height() >= int(parent_height * 0.55)


def test_tuning_mode_switch_uses_both_disclosures(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    _add_results_workspace_artifacts(run_dir)
    window._current_run_dir = str(run_dir)
    assert window._report_viewer.load_report(str(run_dir))
    window._enter_complete_state_workspace()
    window.resize(1400, 900)
    window.show()
    QApplication.processEvents()

    assert not window._report_viewer.isHidden()

    # Downstream only -> tuning mode.
    window._tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._report_viewer.isHidden()

    # Both expanded -> tuning mode.
    window._correction_tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._report_viewer.isHidden()

    # Collapse one while other expanded -> still tuning mode.
    window._tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._report_viewer.isHidden()

    # Collapse both -> report viewer returns.
    window._correction_tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert not window._report_viewer.isHidden()

    # Correction only -> tuning mode.
    window._correction_tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._report_viewer.isHidden()

    # Both expanded -> tuning mode.
    window._tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._report_viewer.isHidden()

    # Collapse one while other expanded -> still tuning mode.
    window._correction_tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._report_viewer.isHidden()

    # Collapse last expanded disclosure -> normal mode.
    window._tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert not window._report_viewer.isHidden()


def test_correction_mode_switch_preserves_results_region_tab_and_image(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    _add_results_workspace_artifacts(run_dir)
    window._current_run_dir = str(run_dir)
    assert window._report_viewer.load_report(str(run_dir))
    window._enter_complete_state_workspace()
    window.resize(1400, 900)
    window.show()
    QApplication.processEvents()

    window._report_viewer._region_combo.setCurrentText("Region1")
    phasic_raw_idx = -1
    for i in range(window._report_viewer._tabs.count()):
        if window._report_viewer._tabs.tabText(i) == "Phasic Sig/Iso":
            phasic_raw_idx = i
            break
    assert phasic_raw_idx >= 0
    window._report_viewer._tabs.setCurrentIndex(phasic_raw_idx)
    QApplication.processEvents()

    window._report_viewer._on_next_image()
    before_region = window._report_viewer.selected_region()
    before_tab = window._report_viewer._selected_tab()
    before_key = (before_region, before_tab)
    before_idx = window._report_viewer._tab_indices.get(before_key)
    assert before_idx == 1

    window._correction_tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._report_viewer.isHidden()
    assert window._correction_tuning_roi_combo.currentText() == before_region

    window._correction_tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert not window._report_viewer.isHidden()
    assert window._report_viewer.selected_region() == before_region
    assert window._report_viewer._selected_tab() == before_tab
    assert window._report_viewer._tab_indices.get(before_key) == before_idx


def test_tuning_workspace_wiring_and_refresh(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()

    window._tuning_roi_combo.setCurrentText("Region1")
    window._tuning_chunk_combo.setCurrentText("2")
    window._tuning_event_signal_combo.setCurrentText("dff")
    method_idx = window._tuning_peak_method_combo.findText("absolute")
    assert method_idx >= 0
    window._tuning_peak_method_combo.setCurrentIndex(method_idx)
    assert window._tuning_peak_method_combo.currentText() == "absolute"
    window._tuning_peak_abs_spin.setValue(0.15)
    window._tuning_peak_dist_spin.setValue(1.5)
    window._tuning_peak_prominence_k_spin.setValue(1.8)
    window._tuning_peak_width_sec_spin.setValue(0.35)
    if window._tuning_peak_pre_filter_combo.findText("smooth") >= 0:
        window._tuning_peak_pre_filter_combo.setCurrentText("smooth")

    overlay_path = tmp_path / "overlay_test.png"
    from PySide6.QtGui import QPixmap
    pix = QPixmap(320, 120)
    pix.fill()
    assert pix.save(str(overlay_path))

    captured = {}

    def _fake_retune(**kwargs):
        captured.update(kwargs)
        out = tmp_path / "retune_out"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "event_signal_used": kwargs["overrides"]["event_signal"],
            "artifacts": {"retuned_overlay_png": str(overlay_path)},
        }

    monkeypatch.setattr("gui.main_window.run_cache_downstream_retune", _fake_retune)

    window._on_run_tuning()

    assert captured["run_dir"] == str(run_dir)
    assert captured["roi"] == "Region1"
    assert captured["chunk_id"] == 2
    assert captured["overrides"]["peak_threshold_method"] == "absolute"
    assert captured["overrides"]["peak_threshold_abs"] == pytest.approx(0.15)
    assert captured["overrides"]["peak_min_prominence_k"] == pytest.approx(1.8)
    assert captured["overrides"]["peak_min_width_sec"] == pytest.approx(0.35)
    assert captured["overrides"]["peak_pre_filter"] == "smooth"
    assert captured["out_dir"] is None

    assert "ROI: Region1" in window._tuning_summary_label.text()
    assert "Chunk: 2" in window._tuning_summary_label.text()
    assert "retune_out" in window._tuning_summary_label.text()
    assert window._tuning_overlay_title.text() == "overlay_test.png"
    shown = window._tuning_overlay_label.pixmap()
    assert shown is not None and not shown.isNull()

    window._tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._report_viewer.isHidden()
    assert window._tuning_overlay_title.text() == "overlay_test.png"
    shown_expanded = window._tuning_overlay_label.pixmap()
    assert shown_expanded is not None and not shown_expanded.isNull()

    window._tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert not window._report_viewer.isHidden()
    window._tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._tuning_overlay_title.text() == "overlay_test.png"
    shown_restored = window._tuning_overlay_label.pixmap()
    assert shown_restored is not None and not shown_restored.isNull()


def test_tuning_same_slot_reloads_overlay_when_path_reused(window, tmp_path, monkeypatch):
    from PySide6.QtGui import QColor, QPixmap

    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._tuning_roi_combo.setCurrentText("Region0")
    window._tuning_chunk_combo.setCurrentText("0")

    overlay_path = tmp_path / "overlay_same_slot.png"

    def _write_colored_overlay(color_name: str) -> None:
        pix = QPixmap(120, 80)
        pix.fill(QColor(color_name))
        assert pix.save(str(overlay_path))

    def _fake_retune(**kwargs):
        pre_filter = kwargs["overrides"].get("peak_pre_filter", "none")
        _write_colored_overlay("red" if pre_filter == "none" else "blue")
        out = tmp_path / "retune_out_same_slot"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "event_signal_used": kwargs["overrides"]["event_signal"],
            "artifacts": {"retuned_overlay_png": str(overlay_path)},
        }

    monkeypatch.setattr("gui.main_window.run_cache_downstream_retune", _fake_retune)

    if window._tuning_peak_pre_filter_combo.findText("none") >= 0:
        window._tuning_peak_pre_filter_combo.setCurrentText("none")
    window._on_run_tuning()
    pix_none = window._tuning_active_overlay_pixmap.toImage()
    color_none = pix_none.pixelColor(5, 5)
    assert color_none.red() > color_none.blue()

    if window._tuning_peak_pre_filter_combo.findText("smooth") >= 0:
        window._tuning_peak_pre_filter_combo.setCurrentText("smooth")
    window._on_run_tuning()
    pix_low = window._tuning_active_overlay_pixmap.toImage()
    color_low = pix_low.pixelColor(5, 5)
    assert color_low.blue() > color_low.red()

    # Same overlay path was reused, but the displayed pixmap must refresh to new content.
    assert window._tuning_active_overlay_path == str(overlay_path)


def test_tuning_same_slot_display_changes_for_none_vs_smooth_real_backend(window, tmp_path):
    run_dir = _make_completed_run_with_quantized_preview_cache(tmp_path, lowpass_hz=10.0)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    assert window._tuning_workspace_available

    window._tuning_roi_combo.setCurrentText("Region0")
    window._tuning_chunk_combo.setCurrentText("0")
    window._tuning_event_signal_combo.setCurrentText("dff")
    window._tuning_peak_method_combo.setCurrentText("median_mad")
    window._tuning_peak_k_spin.setValue(4.0)

    if window._tuning_peak_pre_filter_combo.findText("none") >= 0:
        window._tuning_peak_pre_filter_combo.setCurrentText("none")
    window._on_run_tuning()
    QApplication.processEvents()

    shown_none = window._tuning_overlay_label.pixmap()
    assert shown_none is not None and not shown_none.isNull()
    image_none = shown_none.toImage()
    digest_none = _qimage_sha256(image_none)
    assert window._tuning_overlay_title.text() == "retuned_overlay_Region0_chunk_000.png"

    if window._tuning_peak_pre_filter_combo.findText("smooth") >= 0:
        window._tuning_peak_pre_filter_combo.setCurrentText("smooth")
    window._on_run_tuning()
    QApplication.processEvents()

    shown_low = window._tuning_overlay_label.pixmap()
    assert shown_low is not None and not shown_low.isNull()
    image_low = shown_low.toImage()
    digest_low = _qimage_sha256(image_low)
    assert window._tuning_overlay_title.text() == "retuned_overlay_Region0_chunk_000.png"

    # Same ROI/chunk preview slot is reused in-place; the displayed image must change.
    assert window._tuning_roi_combo.currentText() == "Region0"
    assert window._tuning_chunk_combo.currentText() == "0"
    assert digest_none != digest_low
    total_diff, trace_area_diff = _qimage_diff_counts(image_none, image_low)
    assert total_diff > 0
    # Guard against legend-only changes; the plotted trace region must also move.
    assert trace_area_diff > 0


def test_tuning_display_debug_is_off_by_default(window, tmp_path):
    run_dir = _make_completed_run_with_quantized_preview_cache(tmp_path, lowpass_hz=10.0)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._tuning_roi_combo.setCurrentText("Region0")
    window._tuning_chunk_combo.setCurrentText("0")
    window._tuning_event_signal_combo.setCurrentText("dff")
    window._tuning_peak_method_combo.setCurrentText("median_mad")
    window._tuning_peak_k_spin.setValue(4.0)
    if window._tuning_peak_pre_filter_combo.findText("none") >= 0:
        window._tuning_peak_pre_filter_combo.setCurrentText("none")
    window._on_run_tuning()
    QApplication.processEvents()
    retune_dir = str(window._tuning_last_result.get("retune_dir", ""))
    assert retune_dir
    assert not os.path.exists(os.path.join(retune_dir, "retune_preview_debug_display.json"))


def test_tuning_display_debug_records_loaded_source_when_enabled(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMETRY_RETUNE_DEBUG", "1")
    w = MainWindow()
    try:
        run_dir = _make_completed_run_with_quantized_preview_cache(tmp_path, lowpass_hz=10.0)
        w._is_complete_workspace_active = True
        w._current_run_dir = str(run_dir)
        w._refresh_tuning_workspace_availability()
        assert w._tuning_workspace_available
        w._tuning_roi_combo.setCurrentText("Region0")
        w._tuning_chunk_combo.setCurrentText("0")
        w._tuning_event_signal_combo.setCurrentText("dff")
        w._tuning_peak_method_combo.setCurrentText("median_mad")
        w._tuning_peak_k_spin.setValue(4.0)

        if w._tuning_peak_pre_filter_combo.findText("none") >= 0:
            w._tuning_peak_pre_filter_combo.setCurrentText("none")
        w._on_run_tuning()
        QApplication.processEvents()
        first = dict(w._tuning_last_result)
        first_pix = w._tuning_overlay_label.pixmap()
        assert first_pix is not None and not first_pix.isNull()
        first_image = first_pix.toImage()
        first_debug_path = os.path.join(first["retune_dir"], "retune_preview_debug_display.json")
        assert os.path.isfile(first_debug_path)
        with open(first_debug_path, "r", encoding="utf-8") as f:
            dbg_none = json.load(f)

        if w._tuning_peak_pre_filter_combo.findText("smooth") >= 0:
            w._tuning_peak_pre_filter_combo.setCurrentText("smooth")
        w._on_run_tuning()
        QApplication.processEvents()
        second = dict(w._tuning_last_result)
        second_pix = w._tuning_overlay_label.pixmap()
        assert second_pix is not None and not second_pix.isNull()
        second_image = second_pix.toImage()
        second_debug_path = os.path.join(second["retune_dir"], "retune_preview_debug_display.json")
        assert os.path.isfile(second_debug_path)
        with open(second_debug_path, "r", encoding="utf-8") as f:
            dbg_low = json.load(f)

        assert dbg_none["preview_slot_id"] == "post_run_tuning_overlay"
        assert dbg_low["preview_slot_id"] == "post_run_tuning_overlay"
        assert dbg_none["peak_pre_filter"] == "none"
        assert dbg_low["peak_pre_filter"] == "smooth"
        assert dbg_none["overlay_loaded_path"] == first["artifacts"]["retuned_overlay_png"]
        assert dbg_low["overlay_loaded_path"] == second["artifacts"]["retuned_overlay_png"]
        assert dbg_low["overlay_previous_path"] == dbg_none["overlay_loaded_path"]
        assert dbg_none["overlay_loaded_bytes_sha256"] == dbg_none["overlay_file_sha256"]
        assert dbg_low["overlay_loaded_bytes_sha256"] == dbg_low["overlay_file_sha256"]
        assert dbg_none["displayed_pixmap_sha256_png"].strip()
        assert dbg_low["displayed_pixmap_sha256_png"].strip()
        assert dbg_none["displayed_pixmap_sha256_png"] != dbg_low["displayed_pixmap_sha256_png"]
        total_diff, trace_area_diff = _qimage_diff_counts(first_image, second_image)
        assert total_diff > 0
        assert trace_area_diff > 0
    finally:
        w.close()
        w.deleteLater()


def test_tuning_overlay_fit_render_uses_smooth_transformation(window, monkeypatch):
    from PySide6.QtGui import QPixmap

    calls = []
    original_scaled = QPixmap.scaled

    def _spy_scaled(self, *args, **kwargs):
        mode = None
        if len(args) >= 3:
            mode = args[2]
        elif "mode" in kwargs:
            mode = kwargs["mode"]
        calls.append(mode)
        return original_scaled(self, *args, **kwargs)

    monkeypatch.setattr(QPixmap, "scaled", _spy_scaled)

    pix = QPixmap(400, 200)
    pix.fill()
    window._tuning_active_overlay_pixmap = pix
    window._render_tuning_overlay()

    assert calls
    assert calls[-1] == Qt.SmoothTransformation


def test_tuning_overlay_fit_render_does_not_upscale_past_native_size(window):
    from PySide6.QtGui import QPixmap

    pix = QPixmap(240, 120)
    pix.fill()
    window._tuning_active_overlay_pixmap = pix
    window._render_tuning_overlay()

    shown = window._tuning_overlay_label.pixmap()
    assert shown is not None and not shown.isNull()
    assert shown.width() <= 240
    assert shown.height() <= 120


def test_tuning_preview_click_toggles_fit_and_full_size(window, qapp, tmp_path):
    image_path = tmp_path / "tuning_zoom_toggle.png"
    _write_png(image_path, width=2000, height=1200)

    window._set_tuning_overlay_image(str(image_path))
    qapp.processEvents()

    assert window._tuning_overlay_zoom_mode is False
    assert "toggle fit/full size" in window._tuning_overlay_zoom_hint_label.text().lower()
    fit_pix = window._tuning_overlay_label.pixmap()
    assert fit_pix is not None and not fit_pix.isNull()
    viewport = window._tuning_overlay_scroll.viewport().size()
    assert fit_pix.width() <= viewport.width()
    assert fit_pix.height() <= viewport.height()

    window._tuning_overlay_label.clicked.emit()
    qapp.processEvents()
    assert window._tuning_overlay_zoom_mode is True
    assert "return to fit mode" in window._tuning_overlay_zoom_hint_label.text().lower()
    full_pix = window._tuning_overlay_label.pixmap()
    assert full_pix is not None and not full_pix.isNull()
    assert full_pix.size() == window._tuning_active_overlay_pixmap.size()

    window._tuning_overlay_label.clicked.emit()
    qapp.processEvents()
    assert window._tuning_overlay_zoom_mode is False
    assert "toggle fit/full size" in window._tuning_overlay_zoom_hint_label.text().lower()
    fit_pix_2 = window._tuning_overlay_label.pixmap()
    assert fit_pix_2 is not None and not fit_pix_2.isNull()
    assert fit_pix_2.width() <= viewport.width()
    assert fit_pix_2.height() <= viewport.height()
    assert fit_pix_2.size() != full_pix.size()


def test_tuning_preview_new_image_resets_zoom_to_fit(window, qapp, tmp_path):
    img_a = tmp_path / "tuning_zoom_reset_a.png"
    img_b = tmp_path / "tuning_zoom_reset_b.png"
    _write_png(img_a, width=1800, height=900)
    _write_png(img_b, width=1600, height=800)

    window._set_tuning_overlay_image(str(img_a))
    qapp.processEvents()
    window._tuning_overlay_label.clicked.emit()
    qapp.processEvents()
    assert window._tuning_overlay_zoom_mode is True

    window._set_tuning_overlay_image(str(img_b))
    qapp.processEvents()
    assert window._tuning_overlay_zoom_mode is False
    assert "toggle fit/full size" in window._tuning_overlay_zoom_hint_label.text().lower()


def test_tuning_same_slot_overlay_refresh_resets_zoom_and_updates_pixels(window, qapp, tmp_path):
    from PySide6.QtGui import QColor, QPixmap

    overlay_path = tmp_path / "tuning_same_slot_zoom.png"

    red = QPixmap(1600, 900)
    red.fill(QColor("red"))
    assert red.save(str(overlay_path))
    window._set_tuning_overlay_image(str(overlay_path))
    qapp.processEvents()

    window._tuning_overlay_label.clicked.emit()
    qapp.processEvents()
    assert window._tuning_overlay_zoom_mode is True

    blue = QPixmap(1600, 900)
    blue.fill(QColor("blue"))
    assert blue.save(str(overlay_path))
    window._set_tuning_overlay_image(str(overlay_path))
    qapp.processEvents()

    assert window._tuning_overlay_zoom_mode is False
    assert "toggle fit/full size" in window._tuning_overlay_zoom_hint_label.text().lower()
    shown = window._tuning_overlay_label.pixmap()
    assert shown is not None and not shown.isNull()
    img = shown.toImage()
    px = img.pixelColor(4, 4)
    assert px.blue() > px.red()


def test_correction_overlay_fit_render_uses_smooth_transformation(window, monkeypatch):
    from PySide6.QtGui import QPixmap

    calls = []
    original_scaled = QPixmap.scaled

    def _spy_scaled(self, *args, **kwargs):
        mode = None
        if len(args) >= 3:
            mode = args[2]
        elif "mode" in kwargs:
            mode = kwargs["mode"]
        calls.append(mode)
        return original_scaled(self, *args, **kwargs)

    monkeypatch.setattr(QPixmap, "scaled", _spy_scaled)

    pix = QPixmap(800, 400)
    pix.fill()
    window._correction_tuning_active_inspection_pixmap = pix
    window._render_correction_tuning_overlay()

    assert calls
    assert calls[-1] == Qt.SmoothTransformation


def test_correction_preview_container_prefers_larger_inspection_view(window):
    assert window._correction_tuning_inspection_scroll.minimumHeight() >= 220
    assert window._correction_tuning_inspection_scroll.maximumHeight() >= 680
    assert (
        window._correction_tuning_inspection_scroll.sizePolicy().verticalPolicy()
        == QSizePolicy.Expanding
    )
    layout = window._correction_tuning_scroll_content.layout()
    idx = layout.indexOf(window._correction_tuning_inspection_scroll)
    assert idx >= 0
    assert layout.stretch(idx) == 1


def test_correction_preview_click_toggles_fit_and_full_size(window, qapp, tmp_path):
    image_path = tmp_path / "correction_zoom_toggle.png"
    _write_png(image_path, width=2000, height=1200)

    window._set_correction_tuning_overlay_image(str(image_path))
    qapp.processEvents()

    assert window._correction_tuning_zoom_mode is False
    assert "toggle fit/full size" in window._correction_tuning_zoom_hint_label.text().lower()
    fit_pix = window._correction_tuning_inspection_label.pixmap()
    assert fit_pix is not None and not fit_pix.isNull()
    viewport = window._correction_tuning_inspection_scroll.viewport().size()
    assert fit_pix.width() <= viewport.width()
    assert fit_pix.height() <= viewport.height()

    window._correction_tuning_inspection_label.clicked.emit()
    qapp.processEvents()
    assert window._correction_tuning_zoom_mode is True
    assert "return to fit mode" in window._correction_tuning_zoom_hint_label.text().lower()
    full_pix = window._correction_tuning_inspection_label.pixmap()
    assert full_pix is not None and not full_pix.isNull()
    assert full_pix.size() == window._correction_tuning_active_inspection_pixmap.size()

    window._correction_tuning_inspection_label.clicked.emit()
    qapp.processEvents()
    assert window._correction_tuning_zoom_mode is False
    assert "toggle fit/full size" in window._correction_tuning_zoom_hint_label.text().lower()
    fit_pix_2 = window._correction_tuning_inspection_label.pixmap()
    assert fit_pix_2 is not None and not fit_pix_2.isNull()
    assert fit_pix_2.width() <= viewport.width()
    assert fit_pix_2.height() <= viewport.height()
    assert fit_pix_2.size() != full_pix.size()


def test_correction_preview_new_image_resets_zoom_to_fit(window, qapp, tmp_path):
    img_a = tmp_path / "correction_zoom_reset_a.png"
    img_b = tmp_path / "correction_zoom_reset_b.png"
    _write_png(img_a, width=1800, height=900)
    _write_png(img_b, width=1600, height=800)

    window._set_correction_tuning_overlay_image(str(img_a))
    qapp.processEvents()
    window._correction_tuning_inspection_label.clicked.emit()
    qapp.processEvents()
    assert window._correction_tuning_zoom_mode is True

    window._set_correction_tuning_overlay_image(str(img_b))
    qapp.processEvents()
    assert window._correction_tuning_zoom_mode is False
    assert "toggle fit/full size" in window._correction_tuning_zoom_hint_label.text().lower()


def test_tuning_apply_back_copies_only_downstream_fields(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    original_window_sec = window._window_sec_edit.text()

    if window._tuning_event_signal_combo.findText("delta_f") >= 0:
        window._tuning_event_signal_combo.setCurrentText("delta_f")
    window._tuning_peak_method_combo.setCurrentText("absolute")
    window._tuning_peak_k_spin.setValue(2.75)
    window._tuning_peak_pct_spin.setValue(91.5)
    window._tuning_peak_abs_spin.setValue(0.245)
    window._tuning_peak_dist_spin.setValue(2.5)
    window._tuning_peak_prominence_k_spin.setValue(1.65)
    window._tuning_peak_width_sec_spin.setValue(0.55)
    if window._tuning_peak_pre_filter_combo.findText("smooth") >= 0:
        window._tuning_peak_pre_filter_combo.setCurrentText("smooth")
    if window._tuning_event_auc_combo.findText("median") >= 0:
        window._tuning_event_auc_combo.setCurrentText("median")

    window._on_apply_tuning_values_to_run_settings()
    QApplication.processEvents()

    assert window._event_signal_combo.currentText() == window._tuning_event_signal_combo.currentText()
    assert window._peak_method_combo.currentText() == "absolute"
    assert float(window._peak_k_edit.text()) == pytest.approx(2.75, rel=1e-6)
    assert float(window._peak_pct_edit.text()) == pytest.approx(91.5, rel=1e-6)
    assert float(window._peak_abs_edit.text()) == pytest.approx(0.245, rel=1e-6)
    assert float(window._peak_dist_edit.text()) == pytest.approx(2.5, rel=1e-6)
    assert float(window._peak_min_prominence_k_edit.text()) == pytest.approx(1.65, rel=1e-6)
    assert float(window._peak_min_width_sec_edit.text()) == pytest.approx(0.55, rel=1e-6)
    assert window._tuning_peak_pre_filter_combo.currentText() == "smooth"
    assert window._peak_pre_filter_combo.currentText() == "lowpass"
    assert window._event_auc_combo.currentText() == window._tuning_event_auc_combo.currentText()

    # Correction-sensitive control must remain unchanged.
    assert window._window_sec_edit.text() == original_window_sec


def test_tuning_apply_back_persists_after_new_run(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    if window._tuning_event_signal_combo.findText("delta_f") >= 0:
        window._tuning_event_signal_combo.setCurrentText("delta_f")
    window._tuning_peak_method_combo.setCurrentText("percentile")
    window._tuning_peak_pct_spin.setValue(88.0)
    window._tuning_peak_dist_spin.setValue(1.75)
    window._on_apply_tuning_values_to_run_settings()
    QApplication.processEvents()

    applied_signal = window._event_signal_combo.currentText()
    applied_method = window._peak_method_combo.currentText()
    applied_pct = window._peak_pct_edit.text()
    applied_dist = window._peak_dist_edit.text()

    window._on_new_run()
    QApplication.processEvents()

    assert window._event_signal_combo.currentText() == applied_signal
    assert window._peak_method_combo.currentText() == applied_method
    assert window._peak_pct_edit.text() == applied_pct
    assert window._peak_dist_edit.text() == applied_dist


def test_tuning_apply_back_source_of_truth_mean_std_delta_f(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    if window._tuning_event_signal_combo.findText("delta_f") >= 0:
        window._tuning_event_signal_combo.setCurrentText("delta_f")
    window._tuning_peak_method_combo.setCurrentText("mean_std")
    window._tuning_peak_k_spin.setValue(9.875)

    window._on_apply_tuning_values_to_run_settings()
    QApplication.processEvents()

    assert window._event_signal_combo.currentText() == "delta_f"
    assert window._peak_method_combo.currentText() == "mean_std"
    assert float(window._peak_k_edit.text()) == pytest.approx(9.875, rel=1e-6)


def test_tuning_apply_back_invalidates_validated_state(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    window._validation_passed = True
    window._validated_run_signature = "sig-old"
    window._validated_run_dir = "old_run_dir"

    window._tuning_peak_method_combo.setCurrentText("mean_std")
    window._tuning_peak_k_spin.setValue(8.25)
    window._on_apply_tuning_values_to_run_settings()
    QApplication.processEvents()

    assert window._validation_passed is False
    assert window._validated_run_signature is None
    assert window._peak_method_combo.currentText() == "mean_std"
    assert float(window._peak_k_edit.text()) == pytest.approx(8.25, rel=1e-6)


def test_tuning_apply_back_next_run_serializes_applied_values(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path / "cache_run")
    input_dir = tmp_path / "input"
    out_base = tmp_path / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_base.mkdir(parents=True, exist_ok=True)

    window._input_dir.setText(str(input_dir))
    window._output_dir.setText(str(out_base))
    window._use_custom_config_cb.setChecked(False)
    window._update_config_source_ui()

    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    if window._tuning_event_signal_combo.findText("delta_f") >= 0:
        window._tuning_event_signal_combo.setCurrentText("delta_f")
    window._tuning_peak_method_combo.setCurrentText("mean_std")
    window._tuning_peak_k_spin.setValue(7.125)
    window._tuning_peak_dist_spin.setValue(2.25)
    window._tuning_peak_prominence_k_spin.setValue(1.2)
    window._tuning_peak_width_sec_spin.setValue(0.45)
    if window._tuning_peak_pre_filter_combo.findText("smooth") >= 0:
        window._tuning_peak_pre_filter_combo.setCurrentText("smooth")
    if window._tuning_event_auc_combo.findText("median") >= 0:
        window._tuning_event_auc_combo.setCurrentText("median")

    window._on_apply_tuning_values_to_run_settings()
    QApplication.processEvents()

    argv = window._build_argv(validate_only=False, overwrite=True)
    assert argv
    config_path = os.path.join(window._current_run_dir, "config_effective.yaml")
    spec_path = os.path.join(window._current_run_dir, "gui_run_spec.json")
    assert os.path.isfile(config_path)
    assert os.path.isfile(spec_path)

    with open(config_path, "r", encoding="utf-8") as f:
        effective_cfg = yaml.safe_load(f) or {}
    with open(spec_path, "r", encoding="utf-8") as f:
        run_spec = json.load(f)

    assert effective_cfg["peak_threshold_method"] == "mean_std"
    assert effective_cfg["peak_threshold_method"] != "median_mad"
    assert effective_cfg["event_signal"] == "delta_f"
    assert effective_cfg["peak_threshold_k"] == pytest.approx(7.125, rel=1e-6)
    assert effective_cfg["peak_min_distance_sec"] == pytest.approx(2.25, rel=1e-6)
    assert effective_cfg["peak_min_prominence_k"] == pytest.approx(1.2, rel=1e-6)
    assert effective_cfg["peak_min_width_sec"] == pytest.approx(0.45, rel=1e-6)
    assert effective_cfg["peak_pre_filter"] == "lowpass"
    assert effective_cfg["event_auc_baseline"] == "median"

    overrides = run_spec.get("config_overrides", {})
    assert overrides["peak_threshold_method"] == "mean_std"
    assert overrides["event_signal"] == "delta_f"
    assert overrides["peak_threshold_k"] == pytest.approx(7.125, rel=1e-6)
    assert overrides["peak_min_prominence_k"] == pytest.approx(1.2, rel=1e-6)
    assert overrides["peak_min_width_sec"] == pytest.approx(0.45, rel=1e-6)


def test_tuning_apply_back_method_switch_serialization(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path / "switch_cache")
    input_dir = tmp_path / "input_switch"
    out_base = tmp_path / "out_switch"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_base.mkdir(parents=True, exist_ok=True)

    window._input_dir.setText(str(input_dir))
    window._output_dir.setText(str(out_base))
    window._use_custom_config_cb.setChecked(False)
    window._update_config_source_ui()

    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    # percentile -> absolute
    window._tuning_peak_method_combo.setCurrentText("absolute")
    window._tuning_peak_abs_spin.setValue(0.333)
    window._on_apply_tuning_values_to_run_settings()
    QApplication.processEvents()

    window._build_argv(validate_only=False, overwrite=True)
    cfg_abs_path = os.path.join(window._current_run_dir, "config_effective.yaml")
    with open(cfg_abs_path, "r", encoding="utf-8") as f:
        cfg_abs = yaml.safe_load(f) or {}
    assert cfg_abs["peak_threshold_method"] == "absolute"
    assert cfg_abs["peak_threshold_abs"] == pytest.approx(0.333, rel=1e-6)

    # absolute -> mean_std
    window._tuning_peak_method_combo.setCurrentText("mean_std")
    window._tuning_peak_k_spin.setValue(6.5)
    window._on_apply_tuning_values_to_run_settings()
    QApplication.processEvents()

    window._build_argv(validate_only=False, overwrite=True)
    cfg_k_path = os.path.join(window._current_run_dir, "config_effective.yaml")
    with open(cfg_k_path, "r", encoding="utf-8") as f:
        cfg_k = yaml.safe_load(f) or {}
    assert cfg_k["peak_threshold_method"] == "mean_std"
    assert cfg_k["peak_threshold_k"] == pytest.approx(6.5, rel=1e-6)


def test_run_spec_event_fields_read_from_visible_main_controls(window, tmp_path, monkeypatch):
    input_dir = tmp_path / "input_visible"
    out_base = tmp_path / "out_visible"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_base.mkdir(parents=True, exist_ok=True)

    window._input_dir.setText(str(input_dir))
    window._output_dir.setText(str(out_base))
    window._use_custom_config_cb.setChecked(False)
    window._update_config_source_ui()

    if window._event_signal_combo.findText("delta_f") >= 0:
        window._event_signal_combo.setCurrentText("delta_f")
    window._peak_method_combo.setCurrentText("mean_std")
    window._peak_k_edit.setText("12.5")
    window._peak_pct_edit.setText("91")
    window._peak_abs_edit.setText("0.42")
    window._peak_dist_edit.setText("1.8")
    window._peak_min_prominence_k_edit.setText("1.35")
    window._peak_min_width_sec_edit.setText("0.27")
    if window._peak_pre_filter_combo.findText("lowpass") >= 0:
        window._peak_pre_filter_combo.setCurrentText("lowpass")
    if window._event_auc_combo.findText("median") >= 0:
        window._event_auc_combo.setCurrentText("median")

    captured = {}

    def _fake_parse(
        event_signal_text,
        peak_method_text,
        k_val,
        pct_val,
        abs_val,
        dist_val,
        auc_baseline_text,
        defaults,
        peak_pre_filter_text=None,
        peak_prominence_k_str=None,
        peak_width_sec_str=None,
    ):
        captured.update(
            {
                "event_signal_text": event_signal_text,
                "peak_method_text": peak_method_text,
                "k_val": k_val,
                "pct_val": pct_val,
                "abs_val": abs_val,
                "dist_val": dist_val,
                "auc_baseline_text": auc_baseline_text,
                "peak_pre_filter_text": peak_pre_filter_text,
                "peak_prominence_k_str": peak_prominence_k_str,
                "peak_width_sec_str": peak_width_sec_str,
            }
        )
        return {}, None

    monkeypatch.setattr("gui.main_window.parse_and_validate_event_feature_knobs", _fake_parse)
    spec = window._build_run_spec(validate_only=False)
    assert spec is not None

    assert captured["event_signal_text"] == window._event_signal_combo.currentText()
    assert captured["peak_method_text"] == window._peak_method_combo.currentText()
    assert captured["k_val"] == window._peak_k_edit.text()
    assert captured["pct_val"] == window._peak_pct_edit.text()
    assert captured["abs_val"] == window._peak_abs_edit.text()
    assert captured["dist_val"] == window._peak_dist_edit.text()
    assert captured["auc_baseline_text"] == window._event_auc_combo.currentText()
    assert captured["peak_pre_filter_text"] == window._peak_pre_filter_combo.currentText()
    assert captured["peak_prominence_k_str"] == window._peak_min_prominence_k_edit.text()
    assert captured["peak_width_sec_str"] == window._peak_min_width_sec_edit.text()


def test_tuning_workspace_boundary_message_and_controls(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    scope_msg = window._tuning_scope_note.text().lower()
    assert "retunes downstream event detection from cached phasic traces" in scope_msg
    assert "before deciding whether to rerun" in scope_msg
    assert window._tuning_collapsed_status_label.text().strip() != ""
    assert window._tuning_scope_note.wordWrap()
    assert window._tuning_availability_label.wordWrap()
    assert not window._tuning_scope_note.isHidden()
    assert not window._tuning_availability_label.isHidden()

    labels = [lbl.text() for lbl in window._tuning_controls_container.findChildren(type(window._status_label))]
    assert "Regression Window:" not in labels
    assert "Baseline Method:" not in labels
    assert "R-Low Threshold:" not in labels


def test_tuning_workspace_real_backend_run(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()

    window._tuning_roi_combo.setCurrentText("Region0")
    window._tuning_chunk_combo.setCurrentText("1")
    window._tuning_event_signal_combo.setCurrentText("dff")
    window._tuning_peak_method_combo.setCurrentText("mean_std")
    window._tuning_peak_k_spin.setValue(1.0)
    window._tuning_peak_dist_spin.setValue(1.0)
    window._on_run_tuning()

    assert isinstance(window._tuning_last_result, dict)
    assert window._tuning_last_result["selected_roi"] == "Region0"
    assert window._tuning_last_result["inspection_chunk_id"] == 1
    overlay = window._tuning_last_result["artifacts"]["retuned_overlay_png"]
    assert os.path.isfile(overlay)
    assert "ROI: Region0" in window._tuning_summary_label.text()
    assert "Chunk: 1" in window._tuning_summary_label.text()


def test_tuning_overlay_is_fit_to_view_on_first_display(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()

    window.resize(1200, 900)
    window.show()
    QApplication.processEvents()

    overlay_path = tmp_path / "overlay_big.png"
    from PySide6.QtGui import QPixmap
    big = QPixmap(2200, 1400)
    big.fill()
    assert big.save(str(overlay_path))

    def _fake_retune(**kwargs):
        out = tmp_path / "retune_overlay"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "event_signal_used": kwargs["overrides"]["event_signal"],
            "artifacts": {"retuned_overlay_png": str(overlay_path)},
        }

    monkeypatch.setattr("gui.main_window.run_cache_downstream_retune", _fake_retune)
    window._on_run_tuning()
    window._set_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    shown = window._tuning_overlay_label.pixmap()
    assert shown is not None
    assert not shown.isNull()
    assert window._tuning_overlay_label.text().strip() == ""
    viewport = window._tuning_overlay_scroll.viewport().size()
    assert shown.width() <= viewport.width() + 1
    assert shown.height() <= viewport.height() + 1


def test_tuning_text_labels_wrap_after_run_tuning_result(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window.resize(1250, 900)
    window.show()
    QApplication.processEvents()

    overlay_path = tmp_path / "overlay_wrap.png"
    _write_png(overlay_path, width=1800, height=1000)

    long_retune_dir = os.path.join(
        str(tmp_path),
        "retune",
        ("very_long_tuning_output_path_segment_" * 5) + "out",
    )

    def _fake_retune(**kwargs):
        return {
            "retune_dir": long_retune_dir,
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "event_signal_used": kwargs["overrides"]["event_signal"],
            "artifacts": {"retuned_overlay_png": str(overlay_path)},
        }

    monkeypatch.setattr("gui.main_window.run_cache_downstream_retune", _fake_retune)
    window._on_run_tuning()
    QApplication.processEvents()
    QApplication.processEvents()

    assert window._tuning_scope_note.wordWrap()
    assert window._tuning_availability_label.wordWrap()
    assert window._tuning_summary_label.wordWrap()

    viewport_w = window._tuning_scroll.viewport().size().width()
    tol = 2
    assert window._tuning_scope_note.width() <= viewport_w + tol
    assert window._tuning_availability_label.width() <= viewport_w + tol
    assert window._tuning_summary_label.width() <= viewport_w + tol

    assert not window._tuning_scope_note.isHidden()
    assert not window._tuning_availability_label.isHidden()
    assert not window._tuning_summary_label.isHidden()
    assert "Retune output:" in window._tuning_summary_label.text()
    assert long_retune_dir in window._tuning_summary_label.text()


def test_tuning_overlay_fits_after_run_tuning_result_layout_change(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window.resize(1280, 920)
    window.show()
    QApplication.processEvents()

    overlay_path = tmp_path / "overlay_post_result_big.png"
    _write_png(overlay_path, width=3000, height=1900)

    long_retune_dir = os.path.join(
        str(tmp_path),
        "retune",
        ("retune_layout_change_" * 5) + "out",
    )

    def _fake_retune(**kwargs):
        return {
            "retune_dir": long_retune_dir,
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "event_signal_used": kwargs["overrides"]["event_signal"],
            "artifacts": {"retuned_overlay_png": str(overlay_path)},
        }

    monkeypatch.setattr("gui.main_window.run_cache_downstream_retune", _fake_retune)
    window._on_run_tuning()
    QApplication.processEvents()
    QApplication.processEvents()

    shown = window._tuning_overlay_label.pixmap()
    assert shown is not None
    assert not shown.isNull()
    viewport = window._tuning_overlay_scroll.viewport().size()
    assert shown.width() <= viewport.width() + 1
    assert shown.height() <= viewport.height() + 1


def test_correction_tuning_availability_gating(window, tmp_path):
    window._is_complete_workspace_active = False
    window._current_run_dir = ""
    window._refresh_tuning_workspace_availability()
    assert window._tuning_group.isHidden()
    assert not window._correction_tuning_workspace_available

    window._is_complete_workspace_active = True
    window._current_run_dir = str(tmp_path / "missing")
    window._refresh_tuning_workspace_availability()
    assert not window._correction_tuning_workspace_available
    assert "No completed run directory is active" in window._correction_tuning_availability_label.text()

    run_dir_missing_cfg = _make_completed_run_with_cache(tmp_path / "corr_missing_cfg", with_config=False)
    window._current_run_dir = str(run_dir_missing_cfg)
    window._refresh_tuning_workspace_availability()
    assert not window._correction_tuning_workspace_available
    assert "config_used.yaml" in window._correction_tuning_availability_label.text()

    run_dir = _make_completed_run_with_cache(tmp_path / "corr_ok")
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    assert window._correction_tuning_workspace_available
    assert window._correction_tuning_roi_combo.count() >= 1
    assert window._correction_tuning_chunk_combo.count() >= 1
    ready_msg = window._correction_tuning_availability_label.text().lower()
    assert "all available sessions" in ready_msg
    assert "cached chunks" not in ready_msg


def test_correction_tuning_disclosure_collapsed_and_reset_on_reentry(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._current_run_dir = str(run_dir)
    window._enter_complete_state_workspace()
    QApplication.processEvents()

    assert not window._correction_tuning_disclosure_btn.isChecked()
    assert window._correction_tuning_disclosure_btn.arrowType() == Qt.RightArrow
    assert window._correction_tuning_content.isHidden()
    assert not window._correction_tuning_collapsed_status_label.isHidden()
    assert window._correction_tuning_availability_label.isHidden()

    window._correction_tuning_disclosure_btn.click()
    QApplication.processEvents()
    assert window._correction_tuning_disclosure_btn.isChecked()
    assert not window._correction_tuning_content.isHidden()
    assert window._correction_tuning_collapsed_status_label.isHidden()
    assert not window._correction_tuning_availability_label.isHidden()

    window._exit_complete_state_workspace()
    window._enter_complete_state_workspace()
    QApplication.processEvents()
    assert not window._correction_tuning_disclosure_btn.isChecked()
    assert window._correction_tuning_content.isHidden()
    assert not window._correction_tuning_collapsed_status_label.isHidden()
    assert window._correction_tuning_availability_label.isHidden()


def test_correction_tuning_control_population_and_defaults(window, tmp_path):
    run_dir = _make_completed_run_with_cache(
        tmp_path,
        roi_chunk_keep={"Region0": [0, 2], "Region1": [1]},
    )
    cfg_path = run_dir / "_analysis" / "phasic_out" / "config_used.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update(
        {
            "baseline_method": "uv_globalfit_percentile_session",
            "baseline_percentile": 17.5,
            "lowpass_hz": 1.7,
            "window_sec": 44.0,
            "step_sec": 7.0,
            "min_valid_windows": 9,
            "min_samples_per_window": 51,
            "r_low": 0.11,
            "r_high": 0.91,
            "g_min": 0.22,
        }
    )
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)

    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    QApplication.processEvents()

    assert window._correction_tuning_workspace_available
    rois = [window._correction_tuning_roi_combo.itemText(i) for i in range(window._correction_tuning_roi_combo.count())]
    assert set(rois) >= {"Region0", "Region1"}

    window._correction_tuning_roi_combo.setCurrentText("Region0")
    chunks_r0 = [window._correction_tuning_chunk_combo.itemText(i) for i in range(window._correction_tuning_chunk_combo.count())]
    assert chunks_r0 == ["0", "2"]

    window._correction_tuning_roi_combo.setCurrentText("Region1")
    chunks_r1 = [window._correction_tuning_chunk_combo.itemText(i) for i in range(window._correction_tuning_chunk_combo.count())]
    assert chunks_r1 == ["1"]

    assert window._correction_tuning_baseline_method_combo.currentText() == "uv_globalfit_percentile_session"
    assert window._correction_tuning_fit_mode_combo.currentData() == "rolling_filtered_to_raw"
    assert window._correction_tuning_baseline_pct_spin.value() == pytest.approx(17.5)
    assert window._correction_tuning_lowpass_spin.value() == pytest.approx(1.7)
    assert window._correction_tuning_window_spin.value() == pytest.approx(44.0)
    assert window._correction_tuning_min_samples_spin.value() == 51
    assert not window._correction_tuning_baseline_subtract_cb.isChecked()
    assert window._correction_tuning_lowpass_spin.isEnabled()
    assert window._correction_tuning_window_spin.isEnabled()
    assert window._correction_tuning_min_samples_spin.isEnabled()
    assert window._correction_tuning_baseline_subtract_cb.isEnabled()
    for legacy_attr in (
        "_correction_tuning_step_spin",
        "_correction_tuning_min_valid_windows_spin",
        "_correction_tuning_r_low_spin",
        "_correction_tuning_r_high_spin",
        "_correction_tuning_g_min_spin",
    ):
        assert not hasattr(window, legacy_attr)
    labels = [
        lbl.text()
        for lbl in window._correction_tuning_controls_container.findChildren(
            type(window._status_label)
        )
    ]
    assert "Preview session:" in labels
    assert "Chunk (inspection):" not in labels
    assert "Regression Step (s):" not in labels
    assert "Min Valid Windows:" not in labels
    assert "R-Low Threshold:" not in labels
    assert "R-High Threshold:" not in labels
    assert "G-Min Threshold:" not in labels


def test_correction_tuning_fit_mode_selector_exposes_expanded_modes(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    QApplication.processEvents()

    modes = [
        window._correction_tuning_fit_mode_combo.itemData(i)
        for i in range(window._correction_tuning_fit_mode_combo.count())
    ]
    assert modes[:3] == [
        "rolling_filtered_to_raw",
        "rolling_filtered_to_filtered",
        "global_linear_regression",
    ]
    assert "robust_global_event_reject" in modes
    assert "adaptive_event_gated_regression" in modes
    labels = [
        window._correction_tuning_fit_mode_combo.itemText(i)
        for i in range(window._correction_tuning_fit_mode_combo.count())
    ]
    assert labels[:3] == [
        "Rolling regression (filtered→raw)",
        "Rolling regression (filtered→filtered)",
        "Global linear regression",
    ]
    assert "Robust global fit + event rejection" in labels
    assert "Adaptive event-gated regression" in labels


def test_post_run_tuning_tooltips_cover_downstream_and_correction_controls(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    def _label(root, text: str):
        for candidate in root.findChildren(type(window._status_label)):
            if candidate.text() == text:
                return candidate
        return None

    downstream_pairs = [
        ("ROI:", window._tuning_roi_combo),
        ("Chunk:", window._tuning_chunk_combo),
        ("Event Signal:", window._tuning_event_signal_combo),
        ("Peak Threshold Method:", window._tuning_peak_method_combo),
        ("Peak Threshold K:", window._tuning_peak_k_spin),
        ("Peak Threshold Percentile:", window._tuning_peak_pct_spin),
        ("Peak Threshold Absolute:", window._tuning_peak_abs_spin),
        ("Peak Min Distance (s):", window._tuning_peak_dist_spin),
        ("Peak Min Prominence K:", window._tuning_peak_prominence_k_spin),
        ("Peak Min Width (s):", window._tuning_peak_width_sec_spin),
        ("Peak Pre-Filter:", window._tuning_peak_pre_filter_combo),
        ("Event AUC Baseline:", window._tuning_event_auc_combo),
    ]
    for label_text, control in downstream_pairs:
        label = _label(window._tuning_controls_container, label_text)
        assert label is not None, f"Missing downstream label {label_text}"
        assert label.toolTip().strip(), f"Missing downstream label tooltip {label_text}"
        assert control.toolTip().strip(), f"Missing downstream control tooltip {label_text}"

    correction_pairs = [
        ("ROI:", window._correction_tuning_roi_combo),
        ("Preview session:", window._correction_tuning_chunk_combo),
        ("Baseline Method:", window._correction_tuning_baseline_method_combo),
        ("Baseline Percentile:", window._correction_tuning_baseline_pct_spin),
        ("Lowpass Filter (Hz):", window._correction_tuning_lowpass_spin),
        ("Dynamic Fit Mode:", window._correction_tuning_fit_mode_combo),
        ("Baseline subtract before fit:", window._correction_tuning_baseline_subtract_cb),
        ("Regression Window (s):", window._correction_tuning_window_spin),
        ("Min Samples/Window:", window._correction_tuning_min_samples_spin),
        ("Adaptive residual z-threshold:", window._correction_tuning_adaptive_residual_z_spin),
        ("Adaptive local variance window (s):", window._correction_tuning_adaptive_local_var_window_spin),
        ("Adaptive local variance ratio threshold:", window._correction_tuning_adaptive_local_var_ratio_spin),
        ("Adaptive smooth window (s):", window._correction_tuning_adaptive_smooth_window_spin),
        ("Adaptive minimum trust fraction:", window._correction_tuning_adaptive_min_trust_fraction_spin),
        ("Adaptive freeze interpolation method:", window._correction_tuning_adaptive_freeze_interp_combo),
    ]
    for label_text, control in correction_pairs:
        label = _label(window._correction_tuning_controls_container, label_text)
        assert label is not None, f"Missing correction label {label_text}"
        assert label.toolTip().strip(), f"Missing correction label tooltip {label_text}"
        assert control.toolTip().strip(), f"Missing correction control tooltip {label_text}"

    removed_legacy_labels = [
        "Regression Step (s):",
        "Min Valid Windows:",
        "R-Low Threshold:",
        "R-High Threshold:",
        "G-Min Threshold:",
    ]
    for label_text in removed_legacy_labels:
        assert _label(window._correction_tuning_controls_container, label_text) is None

    assert window._tuning_disclosure_btn.toolTip().strip()
    assert window._correction_tuning_disclosure_btn.toolTip().strip()
    assert window._run_tuning_btn.toolTip().strip()
    assert window._open_tuning_dir_btn.toolTip().strip()
    assert window._apply_tuning_btn.toolTip().strip()
    assert window._run_correction_tuning_btn.toolTip().strip()
    assert window._open_correction_tuning_dir_btn.toolTip().strip()
    assert window._tuning_overlay_title.toolTip().strip()
    assert window._tuning_overlay_label.toolTip().strip()
    assert window._tuning_summary_label.toolTip().strip()
    assert window._correction_tuning_inspection_title.toolTip().strip()
    assert window._correction_tuning_inspection_label.toolTip().strip()
    assert window._correction_tuning_summary_label.toolTip().strip()


def test_tuning_preview_session_selectors_show_all_roi_sessions(window, tmp_path):
    run_dir = _make_completed_run_with_cache(
        tmp_path / "full_sessions",
        chunk_ids=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        roi_chunk_keep={"Region0": [0, 1, 2, 5, 7, 9], "Region1": [3, 4, 8]},
    )
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    QApplication.processEvents()

    window._tuning_roi_combo.setCurrentText("Region0")
    downstream_r0 = [
        window._tuning_chunk_combo.itemText(i)
        for i in range(window._tuning_chunk_combo.count())
    ]
    assert downstream_r0 == ["0", "1", "2", "5", "7", "9"]
    assert len(downstream_r0) == 6

    window._tuning_roi_combo.setCurrentText("Region1")
    downstream_r1 = [
        window._tuning_chunk_combo.itemText(i)
        for i in range(window._tuning_chunk_combo.count())
    ]
    assert downstream_r1 == ["3", "4", "8"]

    window._correction_tuning_roi_combo.setCurrentText("Region0")
    correction_r0 = [
        window._correction_tuning_chunk_combo.itemText(i)
        for i in range(window._correction_tuning_chunk_combo.count())
    ]
    assert correction_r0 == ["0", "1", "2", "5", "7", "9"]
    assert len(correction_r0) == 6

    window._correction_tuning_roi_combo.setCurrentText("Region1")
    correction_r1 = [
        window._correction_tuning_chunk_combo.itemText(i)
        for i in range(window._correction_tuning_chunk_combo.count())
    ]
    assert correction_r1 == ["3", "4", "8"]


def test_correction_tuning_backend_wiring_and_result_refresh(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    window._correction_tuning_roi_combo.setCurrentText("Region0")
    window._correction_tuning_chunk_combo.setCurrentText("2")
    window._correction_tuning_baseline_method_combo.setCurrentText("uv_raw_percentile_session")
    window._correction_tuning_baseline_pct_spin.setValue(13.5)
    window._correction_tuning_lowpass_spin.setValue(1.4)
    window._correction_tuning_baseline_subtract_cb.setChecked(True)
    window._correction_tuning_window_spin.setValue(45.0)
    window._correction_tuning_min_samples_spin.setValue(21)

    inspection_path = tmp_path / "correction_inspect.png"
    _write_png(inspection_path, width=900, height=420)
    captured = {}

    def _fake_correction_retune(**kwargs):
        captured.update(kwargs)
        out = tmp_path / "correction_retune_out"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "artifacts": {
                "retuned_correction_inspection_png": str(inspection_path),
            },
        }

    monkeypatch.setattr("gui.main_window.run_cache_correction_retune", _fake_correction_retune)
    window._on_run_correction_tuning()
    QApplication.processEvents()

    assert captured["run_dir"] == str(run_dir)
    assert captured["roi"] == "Region0"
    assert captured["chunk_id"] == 2
    assert captured["out_dir"] is None
    overrides = captured["overrides"]
    assert set(overrides.keys()) == {
        "dynamic_fit_mode",
        "baseline_method",
        "baseline_percentile",
        "lowpass_hz",
        "baseline_subtract_before_fit",
        "window_sec",
        "min_samples_per_window",
    }
    assert overrides["dynamic_fit_mode"] == "rolling_filtered_to_raw"
    assert overrides["baseline_subtract_before_fit"] is True

    assert "ROI: Region0" in window._correction_tuning_summary_label.text()
    assert "Recomputed across: all available sessions for this ROI" in window._correction_tuning_summary_label.text()
    assert "Dynamic fit mode: Rolling regression (filtered→raw)" in window._correction_tuning_summary_label.text()
    assert "Baseline subtract before fit: enabled" in window._correction_tuning_summary_label.text()
    assert "Preview session: 2" in window._correction_tuning_summary_label.text()
    assert "Inspection chunk:" not in window._correction_tuning_summary_label.text()
    assert "correction_retune_out" in window._correction_tuning_summary_label.text()
    title = window._correction_tuning_inspection_title.text()
    assert "[1/1]" in title
    assert title.endswith("correction_inspect.png")
    shown = window._correction_tuning_inspection_label.pixmap()
    assert shown is not None and not shown.isNull()
    assert window._open_correction_tuning_dir_btn.isEnabled()


def test_correction_tuning_busy_state_feedback_and_cleanup_on_success(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    inspection_path = tmp_path / "busy_state_success.png"
    _write_png(inspection_path, width=900, height=420)
    seen = {"busy": False, "disabled": False, "running_text": False}

    def _fake_correction_retune(**kwargs):
        cursor = QApplication.overrideCursor()
        seen["busy"] = cursor is not None and cursor.shape() == Qt.WaitCursor
        seen["disabled"] = not window._run_correction_tuning_btn.isEnabled()
        seen["running_text"] = window._run_correction_tuning_btn.text() == "Running..."
        out = tmp_path / "correction_retune_busy_success"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "artifacts": {
                "retuned_correction_inspection_png": str(inspection_path),
            },
        }

    monkeypatch.setattr("gui.main_window.run_cache_correction_retune", _fake_correction_retune)
    assert window._run_correction_tuning_btn.isEnabled()
    window._on_run_correction_tuning()
    QApplication.processEvents()

    assert seen["busy"]
    assert seen["disabled"]
    assert seen["running_text"]
    assert QApplication.overrideCursor() is None
    assert window._run_correction_tuning_btn.isEnabled()
    assert window._run_correction_tuning_btn.text() == "Run Correction Retune"


def test_correction_tuning_busy_state_cleanup_on_failure_and_early_exit(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    monkeypatch.setattr("gui.main_window.QMessageBox.warning", lambda *args, **kwargs: 0)
    monkeypatch.setattr("gui.main_window.QMessageBox.critical", lambda *args, **kwargs: 0)

    # Early-exit path (no preview session selected) must still restore busy state.
    window._correction_tuning_chunk_combo.clear()
    window._on_run_correction_tuning()
    QApplication.processEvents()
    assert QApplication.overrideCursor() is None
    assert window._run_correction_tuning_btn.isEnabled()
    assert window._run_correction_tuning_btn.text() == "Run Correction Retune"

    # Exception path must also restore busy state and button state.
    window._populate_correction_tuning_chunk_choices(
        window._correction_tuning_roi_combo.currentText().strip()
    )

    def _raising_retune(**_kwargs):
        raise RuntimeError("forced retune failure")

    monkeypatch.setattr("gui.main_window.run_cache_correction_retune", _raising_retune)
    window._on_run_correction_tuning()
    QApplication.processEvents()
    assert QApplication.overrideCursor() is None
    assert window._run_correction_tuning_btn.isEnabled()
    assert window._run_correction_tuning_btn.text() == "Run Correction Retune"


def test_correction_tuning_global_fit_mode_disables_rolling_knobs_and_plumbs_override(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    idx = window._correction_tuning_fit_mode_combo.findData("global_linear_regression")
    assert idx >= 0
    window._correction_tuning_fit_mode_combo.setCurrentIndex(idx)
    QApplication.processEvents()

    assert not window._correction_tuning_window_spin.isEnabled()
    assert not window._correction_tuning_min_samples_spin.isEnabled()
    assert not window._correction_tuning_baseline_subtract_cb.isEnabled()

    inspection_path = tmp_path / "correction_inspect_global.png"
    _write_png(inspection_path, width=900, height=420)
    captured = {}

    def _fake_correction_retune(**kwargs):
        captured.update(kwargs)
        out = tmp_path / "correction_retune_out_global"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "artifacts": {
                "retuned_correction_inspection_png": str(inspection_path),
            },
        }

    monkeypatch.setattr("gui.main_window.run_cache_correction_retune", _fake_correction_retune)
    window._on_run_correction_tuning()
    QApplication.processEvents()

    overrides = captured["overrides"]
    assert overrides["dynamic_fit_mode"] == "global_linear_regression"
    assert "window_sec" not in overrides
    assert "min_samples_per_window" not in overrides
    assert "baseline_subtract_before_fit" not in overrides
    assert "Dynamic fit mode: Global linear regression" in window._correction_tuning_summary_label.text()
    assert (
        "Baseline subtract before fit: inactive in global linear regression mode"
        in window._correction_tuning_summary_label.text()
    )
    assert "Adaptive diagnostics:" not in window._correction_tuning_summary_label.text()


def test_correction_tuning_filtered_to_filtered_mode_plumbs_baseline_toggle(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    idx = window._correction_tuning_fit_mode_combo.findData("rolling_filtered_to_filtered")
    assert idx >= 0
    window._correction_tuning_fit_mode_combo.setCurrentIndex(idx)
    window._correction_tuning_baseline_subtract_cb.setChecked(True)
    QApplication.processEvents()

    assert window._correction_tuning_window_spin.isEnabled()
    assert window._correction_tuning_min_samples_spin.isEnabled()
    assert window._correction_tuning_baseline_subtract_cb.isEnabled()

    inspection_path = tmp_path / "correction_inspect_filtered_to_filtered.png"
    _write_png(inspection_path, width=900, height=420)
    captured = {}

    def _fake_correction_retune(**kwargs):
        captured.update(kwargs)
        out = tmp_path / "correction_retune_out_filtered_to_filtered"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "artifacts": {
                "retuned_correction_inspection_png": str(inspection_path),
            },
        }

    monkeypatch.setattr("gui.main_window.run_cache_correction_retune", _fake_correction_retune)
    window._on_run_correction_tuning()
    QApplication.processEvents()

    overrides = captured["overrides"]
    assert overrides["dynamic_fit_mode"] == "rolling_filtered_to_filtered"
    assert overrides["baseline_subtract_before_fit"] is True
    assert "window_sec" in overrides
    assert "min_samples_per_window" in overrides
    assert "Dynamic fit mode: Rolling regression (filtered→filtered)" in window._correction_tuning_summary_label.text()
    assert "Baseline subtract before fit: enabled" in window._correction_tuning_summary_label.text()


def test_correction_tuning_robust_event_reject_mode_plumbs_overrides(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    idx = window._correction_tuning_fit_mode_combo.findData("robust_global_event_reject")
    assert idx >= 0
    window._correction_tuning_fit_mode_combo.setCurrentIndex(idx)
    window._correction_tuning_robust_max_iters_spin.setValue(4)
    window._correction_tuning_robust_residual_z_spin.setValue(3.1)
    window._correction_tuning_robust_local_var_window_spin.setValue(9.0)
    window._correction_tuning_robust_local_var_ratio_enable_cb.setChecked(True)
    window._correction_tuning_robust_local_var_ratio_spin.setValue(4.4)
    window._correction_tuning_robust_min_keep_fraction_spin.setValue(0.62)
    QApplication.processEvents()

    assert not window._correction_tuning_window_spin.isEnabled()
    assert not window._correction_tuning_min_samples_spin.isEnabled()
    assert not window._correction_tuning_baseline_subtract_cb.isEnabled()
    assert window._correction_tuning_robust_max_iters_spin.isEnabled()
    assert window._correction_tuning_robust_residual_z_spin.isEnabled()
    assert window._correction_tuning_robust_local_var_window_spin.isEnabled()
    assert window._correction_tuning_robust_local_var_ratio_enable_cb.isEnabled()
    assert window._correction_tuning_robust_local_var_ratio_spin.isEnabled()
    assert window._correction_tuning_robust_min_keep_fraction_spin.isEnabled()

    inspection_path = tmp_path / "correction_inspect_robust.png"
    _write_png(inspection_path, width=900, height=420)
    captured = {}

    def _fake_correction_retune(**kwargs):
        captured.update(kwargs)
        out = tmp_path / "correction_retune_out_robust"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "artifacts": {
                "retuned_correction_inspection_png": str(inspection_path),
            },
        }

    monkeypatch.setattr("gui.main_window.run_cache_correction_retune", _fake_correction_retune)
    window._on_run_correction_tuning()
    QApplication.processEvents()

    overrides = captured["overrides"]
    assert overrides["dynamic_fit_mode"] == "robust_global_event_reject"
    assert overrides["robust_event_reject_max_iters"] == 4
    assert overrides["robust_event_reject_residual_z_thresh"] == pytest.approx(3.1)
    assert overrides["robust_event_reject_local_var_window_sec"] == pytest.approx(9.0)
    assert overrides["robust_event_reject_local_var_ratio_thresh"] == pytest.approx(4.4)
    assert overrides["robust_event_reject_min_keep_fraction"] == pytest.approx(0.62)
    assert "window_sec" not in overrides
    assert "min_samples_per_window" not in overrides
    assert "baseline_subtract_before_fit" not in overrides
    assert "Dynamic fit mode: Robust global fit + event rejection" in window._correction_tuning_summary_label.text()
    assert (
        "Baseline subtract before fit: inactive in robust global event-reject mode"
        in window._correction_tuning_summary_label.text()
    )
    summary = window._correction_tuning_summary_label.text()
    assert "Robust event-reject settings:" in summary
    assert "max_iters=4" in summary
    assert "residual_z=3.1" in summary
    assert "local_var_window_s=9" in summary
    assert "local_var_ratio=4.4" in summary
    assert "min_keep=0.62" in summary
    assert "Adaptive diagnostics:" not in summary


def test_correction_tuning_adaptive_event_gated_mode_plumbs_overrides(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    idx = window._correction_tuning_fit_mode_combo.findData("adaptive_event_gated_regression")
    assert idx >= 0
    window._correction_tuning_fit_mode_combo.setCurrentIndex(idx)
    window._correction_tuning_adaptive_residual_z_spin.setValue(3.25)
    window._correction_tuning_adaptive_local_var_window_spin.setValue(8.5)
    window._correction_tuning_adaptive_local_var_ratio_enable_cb.setChecked(True)
    window._correction_tuning_adaptive_local_var_ratio_spin.setValue(4.3)
    window._correction_tuning_adaptive_smooth_window_spin.setValue(70.0)
    window._correction_tuning_adaptive_min_trust_fraction_spin.setValue(0.61)
    idx_interp = window._correction_tuning_adaptive_freeze_interp_combo.findData("linear_hold")
    assert idx_interp >= 0
    window._correction_tuning_adaptive_freeze_interp_combo.setCurrentIndex(idx_interp)
    QApplication.processEvents()

    assert not window._correction_tuning_window_spin.isEnabled()
    assert not window._correction_tuning_min_samples_spin.isEnabled()
    assert not window._correction_tuning_baseline_subtract_cb.isEnabled()
    assert not window._correction_tuning_robust_max_iters_spin.isEnabled()
    assert window._correction_tuning_adaptive_residual_z_spin.isEnabled()
    assert window._correction_tuning_adaptive_local_var_window_spin.isEnabled()
    assert window._correction_tuning_adaptive_local_var_ratio_enable_cb.isEnabled()
    assert window._correction_tuning_adaptive_local_var_ratio_spin.isEnabled()
    assert window._correction_tuning_adaptive_smooth_window_spin.isEnabled()
    assert window._correction_tuning_adaptive_min_trust_fraction_spin.isEnabled()
    assert window._correction_tuning_adaptive_freeze_interp_combo.isEnabled()

    inspection_path = tmp_path / "correction_inspect_adaptive.png"
    _write_png(inspection_path, width=900, height=420)
    captured = {}

    def _fake_correction_retune(**kwargs):
        captured.update(kwargs)
        out = tmp_path / "correction_retune_out_adaptive"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "artifacts": {
                "retuned_correction_inspection_png": str(inspection_path),
                "retuned_correction_inspection_adaptive_diagnostics": {
                    "fit_mode_resolved": "adaptive_event_gated_regression",
                    "trust_fraction": 0.58,
                    "gated_fraction": 0.42,
                    "fallback_mode": "none",
                    "fallback_failed": False,
                    "fallback_status": "no",
                },
            },
        }

    monkeypatch.setattr("gui.main_window.run_cache_correction_retune", _fake_correction_retune)
    window._on_run_correction_tuning()
    QApplication.processEvents()

    overrides = captured["overrides"]
    assert overrides["dynamic_fit_mode"] == "adaptive_event_gated_regression"
    assert overrides["adaptive_event_gate_residual_z_thresh"] == pytest.approx(3.25)
    assert overrides["adaptive_event_gate_local_var_window_sec"] == pytest.approx(8.5)
    assert overrides["adaptive_event_gate_local_var_ratio_thresh"] == pytest.approx(4.3)
    assert overrides["adaptive_event_gate_smooth_window_sec"] == pytest.approx(70.0)
    assert overrides["adaptive_event_gate_min_trust_fraction"] == pytest.approx(0.61)
    assert overrides["adaptive_event_gate_freeze_interp_method"] == "linear_hold"
    assert "window_sec" not in overrides
    assert "min_samples_per_window" not in overrides
    assert "baseline_subtract_before_fit" not in overrides
    assert "robust_event_reject_max_iters" not in overrides
    summary = window._correction_tuning_summary_label.text()
    assert "Dynamic fit mode: Adaptive event-gated regression" in summary
    assert "Baseline subtract before fit: inactive in adaptive event-gated mode" in summary
    assert "Adaptive event-gated settings:" in summary
    assert "residual_z=3.25" in summary
    assert "local_var_window_s=8.5" in summary
    assert "local_var_ratio=4.3" in summary
    assert "smooth_window_s=70" in summary
    assert "min_trust=0.61" in summary
    assert "freeze_interp=linear_hold" in summary
    assert "Adaptive diagnostics: trust fraction=0.580, gated fraction=0.420, fallback=no" in summary


def test_correction_tuning_robust_summary_shows_runtime_diagnostics(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    idx = window._correction_tuning_fit_mode_combo.findData("robust_global_event_reject")
    assert idx >= 0
    window._correction_tuning_fit_mode_combo.setCurrentIndex(idx)
    QApplication.processEvents()

    inspection_path = tmp_path / "correction_inspect_robust_diag.png"
    _write_png(inspection_path, width=900, height=420)

    def _fake_correction_retune(**kwargs):
        out = tmp_path / "correction_retune_out_robust_diag"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "artifacts": {
                "retuned_correction_inspection_png": str(inspection_path),
                "retuned_correction_inspection_robust_diagnostics": {
                    "fit_mode_resolved": "robust_global_event_reject",
                    "iterations_completed": 3,
                    "keep_fraction": 0.779,
                    "fallback_status": "no",
                    "fallback_to_global_linear": False,
                    "fallback_failed": False,
                    "excluded_count": 87,
                    "excluded_fraction": 0.221,
                },
            },
        }

    monkeypatch.setattr("gui.main_window.run_cache_correction_retune", _fake_correction_retune)
    window._on_run_correction_tuning()
    QApplication.processEvents()

    summary = window._correction_tuning_summary_label.text()
    assert "Dynamic fit mode: Robust global fit + event rejection" in summary
    assert "Robust diagnostics: keep fraction=0.779, iterations=3, fallback=no" in summary


def test_correction_tuning_backend_loads_four_panel_carousel(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    raw_path = tmp_path / "retuned_correction_inspection_Region0_chunk_002_raw.png"
    centered_path = tmp_path / "retuned_correction_inspection_Region0_chunk_002_centered.png"
    fit_path = tmp_path / "retuned_correction_inspection_Region0_chunk_002_fit.png"
    dff_path = tmp_path / "retuned_correction_inspection_Region0_chunk_002_dff.png"
    for p in (raw_path, centered_path, fit_path, dff_path):
        _write_png(p, width=960, height=480)

    def _fake_correction_retune(**kwargs):
        out = tmp_path / "correction_retune_out_carousel"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "artifacts": {
                "retuned_correction_inspection_pngs": [
                    str(raw_path),
                    str(centered_path),
                    str(fit_path),
                    str(dff_path),
                ],
                "retuned_correction_inspection_panel_labels": [
                    "Raw absolute sig/iso",
                    "Centered common-gain sig/iso",
                    "Dynamic fit",
                    "Final corrected dF/F",
                ],
            },
        }

    monkeypatch.setattr("gui.main_window.run_cache_correction_retune", _fake_correction_retune)
    window._on_run_correction_tuning()
    QApplication.processEvents()

    assert window._correction_tuning_inspection_counter_label.text() == "1/4"
    assert window._correction_tuning_prev_btn.isEnabled()
    assert window._correction_tuning_next_btn.isEnabled()
    assert "Dynamic fit" in window._correction_tuning_inspection_title.text()
    assert window._correction_tuning_inspection_title.text().endswith(fit_path.name)

    window._on_correction_tuning_next_image()
    QApplication.processEvents()
    assert window._correction_tuning_inspection_counter_label.text() == "2/4"
    assert "Final corrected dF/F" in window._correction_tuning_inspection_title.text()
    assert window._correction_tuning_inspection_title.text().endswith(dff_path.name)

    window._on_correction_tuning_prev_image()
    QApplication.processEvents()
    assert window._correction_tuning_inspection_counter_label.text() == "1/4"
    assert "Dynamic fit" in window._correction_tuning_inspection_title.text()

    window._on_correction_tuning_prev_image()
    QApplication.processEvents()
    assert window._correction_tuning_inspection_counter_label.text() == "4/4"
    assert "Raw absolute sig/iso" in window._correction_tuning_inspection_title.text()
    assert window._correction_tuning_inspection_title.text().endswith(raw_path.name)


def test_correction_tuning_scope_integrity_and_no_downstream_controls(window, tmp_path):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    text = window._correction_tuning_scope_note.text().lower()
    assert "across all sessions available for that roi" in text
    assert "preview session is used only for the inspection figure" in text
    assert "isolated retune directory" in text
    assert "not modified" in text

    assert not hasattr(window, "_correction_tuning_event_signal_combo")
    assert not hasattr(window, "_correction_tuning_peak_method_combo")
    assert not hasattr(window, "_correction_tuning_peak_prominence_k_spin")
    assert not hasattr(window, "_correction_tuning_peak_width_sec_spin")


def test_correction_tuning_state_resets_on_new_run(window, tmp_path, monkeypatch):
    run_dir = _make_completed_run_with_cache(tmp_path)
    window._is_complete_workspace_active = True
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_workspace_availability()
    window._set_tuning_disclosure_expanded(True)
    window._set_correction_tuning_disclosure_expanded(True)
    QApplication.processEvents()

    inspection_path = tmp_path / "correction_reset.png"
    _write_png(inspection_path, width=700, height=380)

    def _fake_correction_retune(**kwargs):
        out = tmp_path / "correction_reset_out"
        out.mkdir(exist_ok=True)
        return {
            "retune_dir": str(out),
            "selected_roi": kwargs["roi"],
            "inspection_chunk_id": kwargs["chunk_id"],
            "artifacts": {"retuned_correction_inspection_png": str(inspection_path)},
        }

    monkeypatch.setattr("gui.main_window.run_cache_correction_retune", _fake_correction_retune)
    window._on_run_correction_tuning()
    QApplication.processEvents()
    assert isinstance(window._correction_tuning_last_result, dict)
    assert window._open_correction_tuning_dir_btn.isEnabled()

    window._on_new_run()
    QApplication.processEvents()

    assert window._correction_tuning_last_result is None
    assert window._correction_tuning_summary_label.text() == "No correction retune result yet."
    assert window._correction_tuning_inspection_title.text() == "No correction inspection artifact loaded."
    assert not window._open_correction_tuning_dir_btn.isEnabled()
