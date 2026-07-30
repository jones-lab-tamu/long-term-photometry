import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QPushButton

from gui.main_window import MainWindow
from gui.run_report_parser import classify_completed_run_candidate
from photometry_pipeline.completed_run_review import load_completed_review_overview
from photometry_pipeline.config import Config
from photometry_pipeline.discovery import discover_inputs
from photometry_pipeline.io.adapters import load_chunk
from gui.synthetic_demo_dialog import GenerateSyntheticDemoDatasetDialog
from gui.synthetic_demo_generator import (
    GUIDED_DEMO_FOLDER_NAME,
    GUIDED_DEMO_FS_HZ,
    GUIDED_DEMO_HEADERS,
    GUIDED_DEMO_ROWS_PER_SESSION,
    GUIDED_DEMO_SESSION_COUNT,
    build_long_duration_demo_command,
    copy_fast_quickstart_demo,
    generate_guided_csv_demo,
    guided_demo_readme_text,
    long_duration_tutorial_config_text,
    write_long_duration_demo_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_fixed_guided_demo_production_contract(tmp_path: Path):
    progress_updates: list[tuple[int, int]] = []
    result = generate_guided_csv_demo(
        tmp_path,
        progress=lambda current, total: progress_updates.append((current, total)),
    )

    assert result.success, result.message
    assert result.input_dir == tmp_path / GUIDED_DEMO_FOLDER_NAME
    assert result.format == "custom_tabular"
    assert progress_updates[0] == (1, GUIDED_DEMO_SESSION_COUNT)
    assert progress_updates[-1] == (
        GUIDED_DEMO_SESSION_COUNT,
        GUIDED_DEMO_SESSION_COUNT,
    )

    csv_files = sorted(result.input_dir.glob("session_*.csv"))
    assert [path.name for path in csv_files] == [
        f"session_{index:04d}.csv"
        for index in range(1, GUIDED_DEMO_SESSION_COUNT + 1)
    ]
    assert all(
        sum(1 for _ in path.open("r", encoding="utf-8"))
        == GUIDED_DEMO_ROWS_PER_SESSION + 1
        for path in csv_files
    )

    data = np.genfromtxt(
        csv_files[0],
        delimiter=",",
        names=True,
        dtype=np.float64,
        encoding="utf-8",
    )
    assert tuple(data.dtype.names or ()) == GUIDED_DEMO_HEADERS
    assert len(data) == GUIDED_DEMO_ROWS_PER_SESSION
    assert data["ElapsedSeconds"][0] == 0.0
    assert np.allclose(np.diff(data["ElapsedSeconds"]), 1.0 / GUIDED_DEMO_FS_HZ)
    assert data["ElapsedSeconds"][-1] == pytest.approx(599.95)
    for header in GUIDED_DEMO_HEADERS:
        assert np.isfinite(data[header]).all()
    assert not np.array_equal(data["ROI1_Signal"], data["ROI2_Signal"])
    assert np.corrcoef(data["ROI1_Signal"], data["ROI1_Reference"])[0, 1] > 0.45
    assert np.corrcoef(data["ROI2_Signal"], data["ROI2_Reference"])[0, 1] > 0.45
    roi1_residual = data["ROI1_Signal"] - 1.08 * (
        data["ROI1_Reference"] - 100.0
    ) - 118.0
    roi2_residual = data["ROI2_Signal"] - 1.08 * (
        data["ROI2_Reference"] - 107.0
    ) - 126.5
    assert np.max(roi1_residual) > 2.8
    assert np.max(roi2_residual) > 3.0
    production_config = Config(
        target_fs_hz=20.0,
        chunk_duration_sec=600.0,
        custom_tabular_time_col="ElapsedSeconds",
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
            ],
            separators=(",", ":"),
        ),
    )
    discovery = discover_inputs(
        str(result.input_dir),
        production_config,
        force_format="custom_tabular",
    )
    assert len(discovery["sessions"]) == GUIDED_DEMO_SESSION_COUNT
    assert [roi["roi_id"] for roi in discovery["rois"]] == ["ROI1", "ROI2"]


def test_guided_demo_is_reproducible_with_fixed_seed(tmp_path: Path):
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    first = generate_guided_csv_demo(
        first_parent, _session_count=2, _rows_per_session=400
    )
    second = generate_guided_csv_demo(
        second_parent, _session_count=2, _rows_per_session=400
    )
    assert first.success and second.success
    for index in range(1, 3):
        name = f"session_{index:04d}.csv"
        assert (first.input_dir / name).read_bytes() == (
            second.input_dir / name
        ).read_bytes()


def test_guided_demo_refuses_existing_final_folder(tmp_path: Path):
    final_folder = tmp_path / GUIDED_DEMO_FOLDER_NAME
    final_folder.mkdir()
    marker = final_folder / "keep_me.txt"
    marker.write_text("preserve", encoding="utf-8")

    result = generate_guided_csv_demo(
        tmp_path, _session_count=1, _rows_per_session=20
    )

    assert result.success is False
    assert "already exists" in result.message
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_guided_demo_failure_cleans_transaction_and_leaves_no_final_folder(
    tmp_path: Path,
):
    def fail_during_progress(current: int, _total: int) -> None:
        if current == 2:
            raise RuntimeError("simulated generation failure")

    result = generate_guided_csv_demo(
        tmp_path,
        progress=fail_during_progress,
        _session_count=3,
        _rows_per_session=100,
    )

    assert result.success is False
    assert "simulated generation failure" in result.message
    assert not (tmp_path / GUIDED_DEMO_FOLDER_NAME).exists()
    assert not list(tmp_path.glob(f".{GUIDED_DEMO_FOLDER_NAME}.tmp-*"))


def test_guided_demo_readme_contains_required_setup_instructions():
    text = guided_demo_readme_text()
    for required in (
        "synthetic demonstration data",
        "not real biological data",
        "CSV files, one file per session",
        "intermittent",
        "Sessions per hour: 2",
        "Session duration: 600 seconds",
        "`ElapsedSeconds`",
        "Time unit: seconds",
        "`ROI1_Signal` with `ROI1_Reference`",
        "`ROI2_Signal` with `ROI2_Reference`",
        "Confirm the displayed natural filename order",
        "Fixed daily anchor",
        "`07:00`",
        "`12:00:00`",
        "Do not draw biological conclusions",
    ):
        assert required in text


def test_dialog_has_one_fixed_guided_flow_and_no_rwd_presets(qapp):
    dialog = GenerateSyntheticDemoDatasetDialog()
    try:
        assert dialog.windowTitle() == "Generate Guided Demo Dataset"
        assert dialog.findChildren(QComboBox) == []
        assert [
            widget.objectName() for widget in dialog.findChildren(QLineEdit)
        ] == ["output_folder_edit"]
        button_text = [button.text() for button in dialog.findChildren(QPushButton)]
        assert "Generate" in button_text
        assert "Open Folder" in button_text
        assert "Set as Current Input" not in button_text
        visible_text = " ".join(
            label.text() for label in dialog.findChildren(QPushButton)
        ) + dialog._status_text.toPlainText()
        assert "RWD" not in visible_text
        assert "preset" not in visible_text.lower()
    finally:
        dialog.close()


def test_main_window_tools_menu_has_guided_demo_without_full_control_handoff(qapp):
    window = MainWindow()
    try:
        assert "Generate Guided Demo Dataset" in [
            action.text() for action in window._tools_menu.actions()
        ]
        assert not hasattr(window, "_apply_synthetic_demo_result_to_inputs")
        assert window._input_dir.text() == ""
    finally:
        window.close()


def test_docs_present_guided_demo_as_normal_gui_path():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (REPO_ROOT / "docs" / "quickstart_gui_synthetic.md").read_text(
        encoding="utf-8"
    )
    demo_docs = (REPO_ROOT / "docs" / "synthetic_demo_datasets.md").read_text(
        encoding="utf-8"
    )

    assert "docs/synthetic_dataset_generator_cli.md" in readme
    assert "Tools -> Generate Guided Demo Dataset" in quickstart
    assert "long_term_photometry_guided_demo" in quickstart
    assert "Set as Current Input" not in quickstart
    assert "one fixed, vendor-neutral CSV recording" in demo_docs
    assert "not visible GUI demo choices" in demo_docs


def test_generated_guided_csv_bounded_real_pipeline_and_completed_loading(
    tmp_path: Path,
):
    generated = generate_guided_csv_demo(
        tmp_path / "source_parent",
        _session_count=2,
    )
    assert generated.success, generated.message
    source_files = sorted(generated.input_dir.glob("session_*.csv"))
    assert [path.name for path in source_files] == [
        "session_0001.csv",
        "session_0002.csv",
    ]

    config = yaml.safe_load(
        (REPO_ROOT / "config" / "qc_universal_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    config.update(
        {
            "target_fs_hz": 20.0,
            "chunk_duration_sec": 600.0,
            "custom_tabular_time_col": "ElapsedSeconds",
            "custom_tabular_time_unit": "seconds",
            "custom_tabular_roi_mapping_json": json.dumps(
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
                ],
                separators=(",", ":"),
            ),
        }
    )
    config_path = tmp_path / "guided_demo_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    parsed_config = Config.from_yaml(str(config_path))
    discovery = discover_inputs(
        str(generated.input_dir),
        parsed_config,
        force_format="custom_tabular",
    )
    assert discovery["resolved_format"] == "CUSTOM_TABULAR"
    assert [session["session_id"] for session in discovery["sessions"]] == [
        "session_0001",
        "session_0002",
    ]
    assert [roi["roi_id"] for roi in discovery["rois"]] == ["ROI1", "ROI2"]
    first_chunk = load_chunk(
        str(source_files[0]), "custom_tabular", parsed_config, chunk_id=0
    )
    assert first_chunk.fs_hz == 20.0
    assert first_chunk.channel_names == ["ROI1", "ROI2"]

    run_dir = tmp_path / "completed_run"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_full_pipeline_deliverables.py"),
        "--input",
        str(generated.input_dir),
        "--out",
        str(run_dir),
        "--config",
        str(config_path),
        "--format",
        "custom_tabular",
        "--mode",
        "both",
        "--sessions-per-hour",
        "2",
        "--session-duration-s",
        "600",
        "--timeline-anchor-mode",
        "fixed_daily_anchor",
        "--fixed-daily-anchor-clock",
        "07:00",
        "--guided-recording-start-clock",
        "12:00:00",
        "--guided-recording-start-clock-source",
        "user_entered",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert (run_dir / "_analysis" / "tonic_out").is_dir()
    assert (run_dir / "_analysis" / "phasic_out").is_dir()
    features = run_dir / "_analysis" / "phasic_out" / "features" / "features.csv"
    assert features.stat().st_size > 0
    day_plots = [
        path
        for roi in ("ROI1", "ROI2")
        for path in (run_dir / roi / "day_plots").glob("*.png")
    ]
    assert day_plots
    assert all(path.stat().st_size > 0 for path in day_plots)
    accepted, reason = classify_completed_run_candidate(str(run_dir))
    assert accepted, reason
    overview = load_completed_review_overview(str(run_dir))
    assert overview["terminal_state"] == "success"
    assert set(overview["included_rois"]) == {"ROI1", "ROI2"}
    assert set(overview["analysis_branches"]) == {"tonic", "phasic"}


# These focused checks preserve concrete developer/test dependencies on the
# internal RWD generator helpers without exposing them in the application UI.
def test_internal_fast_rwd_copy_remains_available_for_regression_use(tmp_path: Path):
    destination = tmp_path / "fast_rwd_fixture"
    result = copy_fast_quickstart_demo(destination)
    assert result.success, result.message
    assert (destination / "tutorial_config.yaml").exists()
    assert sorted(destination.glob("*/fluorescence.csv"))


def test_internal_long_rwd_command_and_config_remain_available(tmp_path: Path):
    destination = tmp_path / "long_rwd_fixture"
    cmd = build_long_duration_demo_command(destination)
    assert "--format" in cmd and "rwd" in cmd
    assert "--seed" in cmd and "2026" in cmd
    text = long_duration_tutorial_config_text(recording_duration_min=1.0)
    assert "peak_threshold_k: 2.5" in text
    config_path = write_long_duration_demo_config(
        destination, recording_duration_min=1.0
    )
    assert config_path.exists()


def test_internal_long_rwd_wrapper_shortened_smoke(tmp_path: Path):
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
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (destination / "tutorial_config.yaml").exists()
    assert sorted(destination.glob("*/fluorescence.csv"))
