import json
import os
import shutil
import sys
import csv

import pytest
import yaml
from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def _set_valid_dirs(w: MainWindow, tmp_path):
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir(parents=True, exist_ok=True)
    w._input_dir.setText(str(input_dir))
    w._output_dir.setText(str(out_dir))


def _write_vendor_style_rwd_dataset(
    root_dir,
    *,
    fs_hz: float,
    chunk_duration_sec: float,
    n_rois: int = 2,
    time_col: str = "TimeStamp",
    uv_suffix: str = "-410",
    sig_suffix: str = "-470",
    timestamp_unit: str = "seconds",
    chunk_names: list[str] | None = None,
    include_metadata_fps: bool = True,
):
    root_dir.mkdir(parents=True, exist_ok=True)
    if chunk_names is None:
        chunk_names = ["2025_01_01-00_00_00"]
    n_samples = int(round(fs_hz * chunk_duration_sec))
    header = [time_col, "Events"]
    for i in range(1, n_rois + 1):
        header.extend([f"CH{i}{uv_suffix}", f"CH{i}{sig_suffix}"])

    for chunk_name in chunk_names:
        chunk_dir = root_dir / chunk_name
        chunk_dir.mkdir(parents=True, exist_ok=True)
        csv_path = chunk_dir / "fluorescence.csv"

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if include_metadata_fps:
                metadata = '{"Fps": %.6f, "Generator": "pytest"}' % float(fs_hz)
            else:
                metadata = '{"Generator": "pytest"}'
            writer.writerow([metadata] + [""] * (len(header) - 1))
            writer.writerow(header)

            for idx in range(n_samples):
                t_seconds = idx / fs_hz
                if timestamp_unit == "seconds":
                    t_out = t_seconds
                elif timestamp_unit == "milliseconds":
                    t_out = t_seconds * 1000.0
                else:
                    raise ValueError(f"Unsupported timestamp_unit: {timestamp_unit}")
                row = [f"{t_out:.6f}".rstrip("0").rstrip("."), ""]
                for roi in range(n_rois):
                    uv = 100.0 + roi + 0.01 * idx
                    sig = uv + 5.0
                    row.extend([f"{uv:.6f}", f"{sig:.6f}"])
                writer.writerow(row)


def _write_stale_baseline_config(path):
    stale_cfg = {
        "target_fs_hz": 50.0,
        "chunk_duration_sec": 600.0,
        "rwd_time_col": "Time(s)",
        "uv_suffix": "-415",
        "sig_suffix": "-470",
        "lowpass_hz": 1.0,
        "baseline_method": "uv_raw_percentile_session",
        "baseline_percentile": 10.0,
        "f0_min_value": 1e-9,
    }
    path.write_text(yaml.safe_dump(stale_cfg, sort_keys=True), encoding="utf-8")


def test_default_baseline_path_is_used_without_custom_yaml(window, tmp_path):
    _set_valid_dirs(window, tmp_path)
    window._use_custom_config_cb.setChecked(False)
    window._config_path.setText("")
    window._update_button_states()

    assert os.path.isfile(window._lab_default_config_path)
    assert window._lab_default_config_path.replace("\\", "/").endswith(
        "config/qc_universal_config.yaml"
    )
    assert not window._use_custom_config_cb.isChecked()
    assert window._validate_gui_inputs() is None
    assert window._active_config_source_path() == window._lab_default_config_path

    spec = window._build_run_spec(validate_only=True)
    assert spec.config_source_path == window._lab_default_config_path
    assert "Baseline Config Source: Lab standard default:" in window._effective_summary_label.text()


def test_custom_yaml_mode_is_optional_and_enforced(window, tmp_path):
    _set_valid_dirs(window, tmp_path)
    window._use_custom_config_cb.setChecked(True)
    assert window._config_path.isEnabled()

    window._config_path.setText("")
    err_missing = window._validate_gui_inputs()
    assert err_missing is not None
    assert "Custom Config YAML is enabled" in err_missing

    custom_cfg = tmp_path / "custom_config.yaml"
    custom_cfg.write_text("lowpass_hz: 0.25\n", encoding="utf-8")
    window._config_path.setText(str(custom_cfg))
    assert window._validate_gui_inputs() is None

    spec = window._build_run_spec(validate_only=True)
    assert spec.config_source_path == str(custom_cfg)


def test_summary_and_discovery_spec_show_actual_baseline_source(window, tmp_path):
    _set_valid_dirs(window, tmp_path)

    window._use_custom_config_cb.setChecked(False)
    window._update_button_states()
    summary_default = window._effective_summary_label.text()
    assert "Baseline Config Source: Lab standard default:" in summary_default
    assert window._lab_default_config_path in summary_default
    default_disco_spec = window._build_discovery_spec()
    assert default_disco_spec.config_source_path == window._lab_default_config_path

    custom_cfg = tmp_path / "custom_config_2.yaml"
    custom_cfg.write_text("lowpass_hz: 0.25\n", encoding="utf-8")
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(custom_cfg))
    window._update_button_states()
    summary_custom = window._effective_summary_label.text()
    assert "Baseline Config Source: Custom YAML:" in summary_custom
    assert str(custom_cfg) in summary_custom
    custom_disco_spec = window._build_discovery_spec()
    assert custom_disco_spec.config_source_path == str(custom_cfg)


def test_gui_run_spec_records_active_config_source(window, tmp_path):
    _set_valid_dirs(window, tmp_path)

    window._use_custom_config_cb.setChecked(False)
    window._config_path.setText("")
    argv_default = window._build_argv(validate_only=True)
    assert "--config" in argv_default
    default_run_dir = window._current_run_dir
    with open(os.path.join(default_run_dir, "gui_run_spec.json"), "r", encoding="utf-8") as f:
        spec_default = json.load(f)
    assert spec_default["config_source_path"] == window._lab_default_config_path

    custom_cfg = tmp_path / "custom_full_config.yaml"
    shutil.copy2(window._lab_default_config_path, custom_cfg)
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(custom_cfg))
    argv_custom = window._build_argv(validate_only=True)
    assert "--config" in argv_custom
    custom_run_dir = window._current_run_dir
    with open(os.path.join(custom_run_dir, "gui_run_spec.json"), "r", encoding="utf-8") as f:
        spec_custom = json.load(f)
    assert spec_custom["config_source_path"] == str(custom_cfg)


def test_vendor_style_rwd_selection_overrides_stale_acquisition_contract(window, tmp_path):
    _set_valid_dirs(window, tmp_path)

    dataset_root = tmp_path / "vendor_rwd_20hz"
    _write_vendor_style_rwd_dataset(
        dataset_root,
        fs_hz=20.0,
        chunk_duration_sec=600.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
    )

    stale_cfg = tmp_path / "stale_baseline.yaml"
    _write_stale_baseline_config(stale_cfg)

    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(stale_cfg))
    window._input_dir.setText(str(dataset_root))
    window._format_combo.setCurrentText("rwd")

    spec = window._build_run_spec(validate_only=True)
    assert spec.data_contract_overrides["target_fs_hz"] == pytest.approx(20.0, abs=1e-6)
    assert spec.data_contract_overrides["chunk_duration_sec"] == pytest.approx(600.0, abs=1e-6)
    assert spec.data_contract_overrides["rwd_time_col"] == "TimeStamp"
    assert spec.data_contract_overrides["uv_suffix"] == "-410"
    assert spec.data_contract_overrides["sig_suffix"] == "-470"

    argv = window._build_argv(validate_only=True)
    assert "--config" in argv
    cfg_path = os.path.join(window._current_run_dir, "config_effective.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        effective = yaml.safe_load(f)

    assert effective["target_fs_hz"] == pytest.approx(20.0, abs=1e-6)
    assert effective["chunk_duration_sec"] == pytest.approx(600.0, abs=1e-6)
    assert effective["rwd_time_col"] == "TimeStamp"
    assert effective["uv_suffix"] == "-410"
    assert effective["sig_suffix"] == "-470"


def test_dataset_switch_recomputes_contract_without_stale_values(window, tmp_path):
    _set_valid_dirs(window, tmp_path)

    dataset_a = tmp_path / "dataset_a"
    _write_vendor_style_rwd_dataset(
        dataset_a,
        fs_hz=20.0,
        chunk_duration_sec=60.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
    )

    dataset_b = tmp_path / "dataset_b"
    _write_vendor_style_rwd_dataset(
        dataset_b,
        fs_hz=25.0,
        chunk_duration_sec=60.0,
        n_rois=2,
        time_col="Time(s)",
        uv_suffix="-415",
        sig_suffix="-470",
    )

    stale_cfg = tmp_path / "stale_baseline_switch.yaml"
    _write_stale_baseline_config(stale_cfg)
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(stale_cfg))

    window._input_dir.setText(str(dataset_a))
    window._format_combo.setCurrentText("auto")
    spec_a = window._build_discovery_spec()
    assert spec_a.data_contract_overrides["target_fs_hz"] == pytest.approx(20.0, abs=1e-6)
    assert spec_a.data_contract_overrides["rwd_time_col"] == "TimeStamp"
    assert spec_a.data_contract_overrides["uv_suffix"] == "-410"

    window._input_dir.setText(str(dataset_b))
    window._format_combo.setCurrentText("rwd")
    spec_b = window._build_discovery_spec()
    assert spec_b.data_contract_overrides["target_fs_hz"] == pytest.approx(25.0, abs=1e-6)
    assert spec_b.data_contract_overrides["rwd_time_col"] == "Time(s)"
    assert spec_b.data_contract_overrides["uv_suffix"] == "-415"

    argv = window._build_argv(validate_only=True)
    assert "--config" in argv
    cfg_path = os.path.join(window._current_run_dir, "config_effective.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        effective = yaml.safe_load(f)

    assert effective["target_fs_hz"] == pytest.approx(25.0, abs=1e-6)
    assert effective["chunk_duration_sec"] == pytest.approx(60.0, abs=1e-6)
    assert effective["rwd_time_col"] == "Time(s)"
    assert effective["uv_suffix"] == "-415"


def test_rwd_contract_inference_supports_clear_millisecond_timestamps(window, tmp_path):
    _set_valid_dirs(window, tmp_path)

    dataset_root = tmp_path / "vendor_rwd_ms"
    _write_vendor_style_rwd_dataset(
        dataset_root,
        fs_hz=20.0,
        chunk_duration_sec=60.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        timestamp_unit="milliseconds",
    )

    stale_cfg = tmp_path / "stale_baseline_ms.yaml"
    _write_stale_baseline_config(stale_cfg)

    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(stale_cfg))
    window._input_dir.setText(str(dataset_root))
    window._format_combo.setCurrentText("rwd")

    spec = window._build_run_spec(validate_only=True)
    assert spec.data_contract_overrides["target_fs_hz"] == pytest.approx(20.0, abs=1e-6)
    assert spec.data_contract_overrides["chunk_duration_sec"] == pytest.approx(60.0, abs=1e-6)


def test_rwd_contract_inference_rejects_ambiguous_timestamp_units(window, tmp_path):
    _set_valid_dirs(window, tmp_path)

    dataset_root = tmp_path / "vendor_rwd_ambiguous"
    _write_vendor_style_rwd_dataset(
        dataset_root,
        fs_hz=1.0,
        chunk_duration_sec=10.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        timestamp_unit="seconds",
        include_metadata_fps=False,
    )

    stale_cfg = tmp_path / "stale_baseline_ambiguous.yaml"
    _write_stale_baseline_config(stale_cfg)
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(stale_cfg))
    window._input_dir.setText(str(dataset_root))
    window._format_combo.setCurrentText("rwd")

    with pytest.raises(ValueError, match="Ambiguous RWD timestamp units"):
        window._build_run_spec(validate_only=True)


def test_rwd_contract_inference_rejects_inconsistent_multichunk_run(window, tmp_path):
    _set_valid_dirs(window, tmp_path)

    dataset_root = tmp_path / "vendor_rwd_inconsistent"
    _write_vendor_style_rwd_dataset(
        dataset_root,
        fs_hz=20.0,
        chunk_duration_sec=60.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        chunk_names=["2025_01_01-00_00_00"],
    )
    _write_vendor_style_rwd_dataset(
        dataset_root,
        fs_hz=20.0,
        chunk_duration_sec=60.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-415",
        sig_suffix="-470",
        chunk_names=["2025_01_01-00_10_00"],
    )

    stale_cfg = tmp_path / "stale_baseline_inconsistent.yaml"
    _write_stale_baseline_config(stale_cfg)
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(stale_cfg))
    window._input_dir.setText(str(dataset_root))
    window._format_combo.setCurrentText("rwd")

    with pytest.raises(ValueError, match="Inconsistent RWD contract across chunks"):
        window._build_run_spec(validate_only=True)


def test_effective_config_export_and_launch_tokens_use_inferred_contract(window, tmp_path):
    _set_valid_dirs(window, tmp_path)

    dataset_root = tmp_path / "vendor_rwd_launch_contract"
    _write_vendor_style_rwd_dataset(
        dataset_root,
        fs_hz=20.0,
        chunk_duration_sec=600.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        timestamp_unit="seconds",
    )

    stale_cfg = tmp_path / "stale_baseline_launch.yaml"
    _write_stale_baseline_config(stale_cfg)
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(stale_cfg))
    window._input_dir.setText(str(dataset_root))
    window._format_combo.setCurrentText("rwd")

    argv = window._build_argv(validate_only=False)
    assert "--config" in argv
    cfg_path = argv[argv.index("--config") + 1]
    assert cfg_path.endswith("config_effective.yaml")
    assert os.path.isfile(cfg_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        effective = yaml.safe_load(f)

    fs = float(effective["target_fs_hz"])
    chunk_duration = float(effective["chunk_duration_sec"])
    n_target = int(round(chunk_duration * fs))
    grid_end = (n_target - 1) / fs
    raw_end = None
    csv_path = dataset_root / "2025_01_01-00_00_00" / "fluorescence.csv"
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[1]
    time_idx = header.index("TimeStamp")
    raw_end = float(rows[-1][time_idx])
    tol = 1.0 / fs

    assert fs == pytest.approx(20.0, abs=1e-6)
    assert chunk_duration == pytest.approx(600.0, abs=1e-6)
    assert raw_end == pytest.approx(599.95, abs=1e-6)
    assert grid_end == pytest.approx(599.95, abs=1e-6)
    assert effective["rwd_time_col"] == "TimeStamp"
    assert effective["uv_suffix"] == "-410"
    assert effective["sig_suffix"] == "-470"
    # Mirrors strict reader coverage semantics:
    # require raw_end >= grid_end - 1/fs.
    assert raw_end >= (grid_end - tol)


def test_rwd_contract_cache_hit_reuses_chunk_scan(window, tmp_path, monkeypatch):
    _set_valid_dirs(window, tmp_path)

    dataset_root = tmp_path / "vendor_rwd_cache_hit"
    _write_vendor_style_rwd_dataset(
        dataset_root,
        fs_hz=20.0,
        chunk_duration_sec=60.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        chunk_names=["2025_01_01-00_00_00", "2025_01_01-00_10_00"],
    )

    stale_cfg = tmp_path / "stale_baseline_cache_hit.yaml"
    _write_stale_baseline_config(stale_cfg)
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(stale_cfg))
    window._input_dir.setText(str(dataset_root))
    window._format_combo.setCurrentText("rwd")

    orig = window._infer_rwd_chunk_contract
    calls = {"n": 0}

    def _wrapped(path):
        calls["n"] += 1
        return orig(path)

    monkeypatch.setattr(window, "_infer_rwd_chunk_contract", _wrapped)

    spec_a = window._build_run_spec(validate_only=True)
    first_calls = calls["n"]
    assert first_calls > 0

    spec_b = window._build_run_spec(validate_only=True)
    assert calls["n"] == first_calls, "Second call should reuse cached contract and skip rescanning chunks."
    assert spec_a.data_contract_overrides == spec_b.data_contract_overrides


def test_gui_timing_logs_disabled_by_default(window):
    assert window._gui_timing_enabled is False
    before = window._log_view.toPlainText()
    window._emit_gui_timing("START", "unit_test_probe")
    after = window._log_view.toPlainText()
    assert after == before


def test_gui_timing_logs_enabled_via_env(qapp, monkeypatch):
    monkeypatch.setenv("PHOTOMETRY_GUI_TIMING", "1")
    w = MainWindow()
    try:
        assert w._gui_timing_enabled is True
        w._emit_gui_timing("START", "unit_test_probe")
        text = w._log_view.toPlainText()
        assert "GUI_TIMING START action=unknown step=unit_test_probe" in text
    finally:
        w.close()
        w.deleteLater()


def test_rwd_contract_cache_invalidates_on_input_change(window, tmp_path, monkeypatch):
    _set_valid_dirs(window, tmp_path)

    dataset_a = tmp_path / "vendor_rwd_cache_input_a"
    _write_vendor_style_rwd_dataset(
        dataset_a,
        fs_hz=20.0,
        chunk_duration_sec=60.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        chunk_names=["2025_01_01-00_00_00"],
    )
    dataset_b = tmp_path / "vendor_rwd_cache_input_b"
    _write_vendor_style_rwd_dataset(
        dataset_b,
        fs_hz=20.0,
        chunk_duration_sec=60.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        chunk_names=["2025_01_01-00_00_00"],
    )

    stale_cfg = tmp_path / "stale_baseline_cache_input.yaml"
    _write_stale_baseline_config(stale_cfg)
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(stale_cfg))
    window._format_combo.setCurrentText("rwd")

    orig = window._infer_rwd_chunk_contract
    calls = {"n": 0}

    def _wrapped(path):
        calls["n"] += 1
        return orig(path)

    monkeypatch.setattr(window, "_infer_rwd_chunk_contract", _wrapped)

    window._input_dir.setText(str(dataset_a))
    window._build_run_spec(validate_only=True)
    first_calls = calls["n"]
    assert first_calls > 0

    window._input_dir.setText(str(dataset_b))
    window._build_run_spec(validate_only=True)
    assert calls["n"] > first_calls, "Input path change should invalidate cache and trigger fresh scan."


def test_rwd_contract_cache_invalidates_on_dataset_change(window, tmp_path, monkeypatch):
    _set_valid_dirs(window, tmp_path)

    dataset_root = tmp_path / "vendor_rwd_cache_mutation"
    _write_vendor_style_rwd_dataset(
        dataset_root,
        fs_hz=20.0,
        chunk_duration_sec=60.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        chunk_names=["2025_01_01-00_00_00", "2025_01_01-00_10_00"],
    )

    stale_cfg = tmp_path / "stale_baseline_cache_mutation.yaml"
    _write_stale_baseline_config(stale_cfg)
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(stale_cfg))
    window._input_dir.setText(str(dataset_root))
    window._format_combo.setCurrentText("rwd")

    orig = window._infer_rwd_chunk_contract
    calls = {"n": 0}

    def _wrapped(path):
        calls["n"] += 1
        return orig(path)

    monkeypatch.setattr(window, "_infer_rwd_chunk_contract", _wrapped)

    window._build_run_spec(validate_only=True)
    first_calls = calls["n"]
    assert first_calls > 0

    # Mutate one chunk file so mtime/size signature changes.
    csv_path = dataset_root / "2025_01_01-00_10_00" / "fluorescence.csv"
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write("\n")

    window._build_run_spec(validate_only=True)
    assert calls["n"] > first_calls, "Dataset mutation should invalidate cache and trigger fresh scan."


def test_cached_and_uncached_effective_configs_are_identical(window, tmp_path):
    _set_valid_dirs(window, tmp_path)

    dataset_root = tmp_path / "vendor_rwd_cache_equiv"
    _write_vendor_style_rwd_dataset(
        dataset_root,
        fs_hz=20.0,
        chunk_duration_sec=600.0,
        n_rois=2,
        time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        chunk_names=["2025_01_01-00_00_00", "2025_01_01-00_10_00"],
    )

    stale_cfg = tmp_path / "stale_baseline_cache_equiv.yaml"
    _write_stale_baseline_config(stale_cfg)
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(stale_cfg))
    window._input_dir.setText(str(dataset_root))
    window._format_combo.setCurrentText("rwd")

    argv_a = window._build_argv(validate_only=True)
    assert "--config" in argv_a
    cfg_a = argv_a[argv_a.index("--config") + 1]
    with open(cfg_a, "r", encoding="utf-8") as f:
        data_a = yaml.safe_load(f)

    argv_b = window._build_argv(validate_only=True)
    assert "--config" in argv_b
    cfg_b = argv_b[argv_b.index("--config") + 1]
    with open(cfg_b, "r", encoding="utf-8") as f:
        data_b = yaml.safe_load(f)

    assert data_a == data_b
