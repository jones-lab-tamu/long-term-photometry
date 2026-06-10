import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit

from gui.main_window import MainWindow
from gui.synthetic_demo_dialog import GenerateSyntheticDemoDatasetDialog
from gui.synthetic_demo_generator import (
    FAST_DEMO_TYPE,
    LONG_DEMO_TYPE,
    build_long_duration_demo_command,
    copy_fast_quickstart_demo,
    long_duration_tutorial_config_text,
    write_long_duration_demo_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_fast_copy_helper_copies_bundled_demo_to_selected_root(tmp_path: Path):
    destination = tmp_path / "my_fast_demo"
    result = copy_fast_quickstart_demo(destination)

    assert result.success, result.message
    assert result.input_dir == destination
    assert result.config_path == destination / "tutorial_config.yaml"
    assert result.format == "rwd"
    assert result.sessions_per_hour == 2
    assert result.mode == "both"
    assert (destination / "tutorial_config.yaml").exists()
    assert (destination / "generation_manifest.yaml").exists()
    assert (destination / "README.md").exists()
    assert sorted(destination.glob("*/fluorescence.csv"))


def test_fast_copy_succeeds_into_existing_empty_destination(tmp_path: Path):
    destination = tmp_path / "empty_demo"
    destination.mkdir()
    assert not any(destination.iterdir())

    result = copy_fast_quickstart_demo(destination)

    assert result.success, result.message
    assert result.input_dir == destination
    assert result.config_path == destination / "tutorial_config.yaml"
    assert (destination / "tutorial_config.yaml").exists()
    assert (destination / "generation_manifest.yaml").exists()
    assert (destination / "README.md").exists()
    assert sorted(destination.glob("*/fluorescence.csv"))


def test_fast_copy_refuses_non_empty_destination_unless_overwrite(tmp_path: Path):
    destination = tmp_path / "existing_demo"
    destination.mkdir()
    marker = destination / "keep_me.txt"
    marker.write_text("old", encoding="utf-8")

    refused = copy_fast_quickstart_demo(destination)
    assert refused.success is False
    assert marker.exists()

    replaced = copy_fast_quickstart_demo(destination, overwrite=True)
    assert replaced.success is True
    assert not marker.exists()
    assert (destination / "tutorial_config.yaml").exists()
    assert sorted(destination.glob("*/fluorescence.csv"))


def test_long_duration_demo_command_uses_fixed_curated_parameters(tmp_path: Path):
    destination = tmp_path / "long_demo"
    cmd = build_long_duration_demo_command(destination)
    joined = " ".join(cmd)

    for token in (
        "--format",
        "rwd",
        "--total-days",
        "2",
        "--recording-duration-min",
        "10",
        "--recordings-per-hour",
        "2",
        "--fs-hz",
        "10",
        "--n-rois",
        "2",
        "--preset",
        "biological_shared_nuisance",
        "--seed",
        "2026",
        "--artifact-enable-motion",
        "--artifact-motion-min-per-day",
        "1",
        "--artifact-motion-rate-per-day",
        "20",
    ):
        assert token in cmd or token in joined
    assert str(destination / "tutorial_config.yaml") in cmd


def test_long_duration_demo_config_uses_conservative_event_defaults(tmp_path: Path):
    text = long_duration_tutorial_config_text(recording_duration_min=1.0)
    assert "peak_threshold_method: mean_std" in text
    assert "peak_threshold_k: 2.5" in text
    assert "peak_min_distance_sec: 1.0" in text
    assert "peak_min_prominence_k: 2.0" in text
    assert "peak_min_width_sec: 0.3" in text

    destination = tmp_path / "long_demo_config"
    config_path = write_long_duration_demo_config(destination, recording_duration_min=1.0)
    written = config_path.read_text(encoding="utf-8")
    assert "peak_threshold_k: 2.5" in written
    assert "peak_min_prominence_k: 2.0" in written
    assert "peak_min_width_sec: 0.3" in written


def test_one_command_wrapper_shortened_smoke(tmp_path: Path):
    destination = tmp_path / "mini_long_demo"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "examples" / "generate_long_duration_demo.py"),
        "--out",
        str(destination),
        "--total-days",
        "0.05",
        "--recording-duration-min",
        "1",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (destination / "tutorial_config.yaml").exists()
    assert (destination / "generation_manifest.yaml").exists()
    assert sorted(destination.glob("*/fluorescence.csv"))
    assert "recommended GUI settings" in result.stdout


def test_dialog_instantiates_with_exact_two_curated_choices_and_no_advanced_fields(qapp):
    dialog = GenerateSyntheticDemoDatasetDialog()
    try:
        combo = dialog.findChild(QComboBox, "demo_choice_combo")
        assert combo is not None
        assert [combo.itemText(i) for i in range(combo.count())] == [
            FAST_DEMO_TYPE,
            LONG_DEMO_TYPE,
        ]
        line_edits = dialog.findChildren(QLineEdit)
        assert [w.objectName() for w in line_edits] == ["output_folder_edit"]
        dialog_text = " ".join(
            [combo.itemText(i) for i in range(combo.count())]
            + [w.objectName() for w in dialog.findChildren(QLineEdit)]
        ).lower()
        for forbidden in (
            "total_days",
            "recording_duration",
            "fs_hz",
            "n_rois",
            "seed",
            "artifact_motion_rate",
        ):
            assert forbidden not in dialog_text
    finally:
        dialog.close()


def test_main_window_tools_menu_contains_synthetic_demo_action(qapp):
    window = MainWindow()
    try:
        actions = [action.text() for action in window.menuBar().actions()]
        assert "Tools" in actions
        tools_menu = window._tools_menu
        assert tools_menu is not None
        assert "Generate Synthetic Demo Dataset" in [action.text() for action in tools_menu.actions()]
    finally:
        window.close()


def test_set_as_current_input_populates_main_window_fields(qapp, tmp_path: Path):
    window = MainWindow()
    try:
        destination = tmp_path / "fast_demo"
        result = copy_fast_quickstart_demo(destination)
        assert result.success
        window._apply_synthetic_demo_result_to_inputs(result)
        assert window._input_dir.text() == str(destination)
        assert window._config_path.text() == str(destination / "tutorial_config.yaml")
        assert window._format_combo.currentText() == "rwd"
        assert window._sph_edit.text() == "2"
        assert window._mode_combo.currentText() == "both"
        assert window._use_custom_config_cb.isChecked() is True
    finally:
        window.close()


def test_docs_link_gui_presets_and_cli_generator():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (REPO_ROOT / "docs" / "quickstart_gui_synthetic.md").read_text(encoding="utf-8")
    demo_docs = (REPO_ROOT / "docs" / "synthetic_demo_datasets.md").read_text(encoding="utf-8")

    assert "docs/synthetic_dataset_generator_cli.md" in readme
    assert (REPO_ROOT / "docs" / "synthetic_dataset_generator_cli.md").exists()
    assert "Tools -> Generate Synthetic Demo Dataset" in quickstart
    assert "Fast quickstart demo" in demo_docs
    assert "Long-duration intermittent demo" in demo_docs
