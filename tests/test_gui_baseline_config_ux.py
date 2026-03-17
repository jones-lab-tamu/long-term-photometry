import json
import os
import shutil
import sys

import pytest
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
