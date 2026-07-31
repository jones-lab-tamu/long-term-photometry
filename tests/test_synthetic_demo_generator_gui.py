import csv
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
from photometry_pipeline.preview.correction_preview import (
    run_guided_local_correction_preview,
)
from gui.synthetic_demo_dialog import GenerateSyntheticDemoDatasetDialog
from gui.synthetic_demo_generator import (
    GUIDED_DEMO_FOLDER_NAME,
    GUIDED_DEMO_FS_HZ,
    GUIDED_DEMO_HEADERS,
    GUIDED_DEMO_ROWS_PER_SESSION,
    GUIDED_DEMO_SESSION_COUNT,
    GUIDED_DEMO_SESSIONS_PER_DAY,
    build_long_duration_demo_command,
    copy_fast_quickstart_demo,
    generate_guided_csv_demo,
    guided_demo_readme_text,
    long_duration_tutorial_config_text,
    write_long_duration_demo_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

GUIDED_DEMO_ROI_MAPPING_JSON = json.dumps(
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
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_guided_demo_config(directory: Path) -> Path:
    """Build the ordinary Guided CSV configuration for the demo dataset."""
    config = yaml.safe_load(
        (REPO_ROOT / "config" / "qc_universal_config.yaml").read_text(encoding="utf-8")
    )
    config.update(
        {
            "target_fs_hz": float(GUIDED_DEMO_FS_HZ),
            "chunk_duration_sec": 600.0,
            "custom_tabular_time_col": "ElapsedSeconds",
            "custom_tabular_time_unit": "seconds",
            "custom_tabular_roi_mapping_json": GUIDED_DEMO_ROI_MAPPING_JSON,
        }
    )
    config_path = directory / "guided_demo_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return config_path


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _fast_component(trace: np.ndarray, *, baseline_window_sec: float = 20.0) -> np.ndarray:
    """Transient-scale component: sample noise smoothed away, slow drift removed."""
    smoothed = _moving_average(trace, 0.25 * GUIDED_DEMO_FS_HZ)
    return smoothed - _moving_average(smoothed, baseline_window_sec * GUIDED_DEMO_FS_HZ)


def _transient_peak_indices(trace: np.ndarray, *, min_amplitude: float = 1.0) -> np.ndarray:
    """Transient peaks at least one second apart. A test measurement only."""
    residual = _fast_component(trace)
    interior = residual[1:-1]
    candidates = (
        np.flatnonzero(
            (interior >= min_amplitude)
            & (interior >= residual[:-2])
            & (interior > residual[2:])
        )
        + 1
    )
    kept: list[int] = []
    for index in candidates:
        if not kept or index - kept[-1] >= GUIDED_DEMO_FS_HZ:
            kept.append(int(index))
    return np.array(kept, dtype=int)


@pytest.fixture(scope="module")
def guided_demo_sessions(tmp_path_factory) -> list[np.ndarray]:
    """The full production demo dataset, loaded once for signal-character tests."""
    parent = tmp_path_factory.mktemp("guided_demo_signal_character")
    result = generate_guided_csv_demo(parent)
    assert result.success, result.message
    return [
        np.loadtxt(path, delimiter=",", skiprows=1)
        for path in sorted(result.input_dir.glob("session_*.csv"))
    ]


# Column offsets in the generated CSV files.
_TIME, _ROI1_SIGNAL, _ROI1_REFERENCE, _ROI2_SIGNAL, _ROI2_REFERENCE = range(5)
_ROI_COLUMNS = ((_ROI1_SIGNAL, _ROI1_REFERENCE), (_ROI2_SIGNAL, _ROI2_REFERENCE))


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
    assert np.corrcoef(data["ROI1_Signal"], data["ROI1_Reference"])[0, 1] > 0.35
    assert np.corrcoef(data["ROI2_Signal"], data["ROI2_Reference"])[0, 1] > 0.35
    # Each signal carries transients well above its own sample-to-sample noise.
    for roi in ("ROI1", "ROI2"):
        signal = data[f"{roi}_Signal"]
        noise = float(np.std(np.diff(signal)))
        assert np.max(_fast_component(signal)) > 5.0 * noise
    production_config = Config(
        target_fs_hz=20.0,
        chunk_duration_sec=600.0,
        custom_tabular_time_col="ElapsedSeconds",
        custom_tabular_roi_mapping_json=GUIDED_DEMO_ROI_MAPPING_JSON,
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


def test_guided_demo_event_timing_is_irregular_and_differs_across_sessions(
    guided_demo_sessions,
):
    peak_seconds = [
        set((_transient_peak_indices(session[:, _ROI1_SIGNAL]) // GUIDED_DEMO_FS_HZ).tolist())
        for session in guided_demo_sessions
    ]
    assert all(len(seconds) > 5 for seconds in peak_seconds)
    # No transient position is repeated in every session.
    assert set.intersection(*peak_seconds) == set()
    # Neighbouring sessions share little more than chance.
    overlaps = [
        len(first & second) / max(1, min(len(first), len(second)))
        for first, second in zip(peak_seconds, peak_seconds[1:])
    ]
    assert float(np.mean(overlaps)) < 0.25
    # Spacing is irregular rather than a repeated grid.
    first_gaps = np.diff(np.sort(np.fromiter(peak_seconds[0], dtype=float)))
    assert float(np.std(first_gaps)) > 5.0


def test_guided_demo_transients_decay_more_slowly_than_they_rise(guided_demo_sessions):
    segments = []
    for session in guided_demo_sessions[:12]:
        residual = _fast_component(session[:, _ROI1_SIGNAL])
        before = int(1.5 * GUIDED_DEMO_FS_HZ)
        after = int(5.0 * GUIDED_DEMO_FS_HZ)
        for index in _transient_peak_indices(
            session[:, _ROI1_SIGNAL], min_amplitude=1.5
        ):
            if before <= index < residual.size - after:
                segments.append(residual[index - before : index + after])
    assert len(segments) > 50
    average = np.mean(np.array(segments), axis=0)
    peak = int(1.5 * GUIDED_DEMO_FS_HZ)
    half_second = int(0.5 * GUIDED_DEMO_FS_HZ)
    # Half a second after the peak the transient is still elevated; half a
    # second before it, it has barely started.
    assert average[peak + half_second] > 2.0 * average[peak - half_second]
    rising_area = float(np.sum(average[peak - before : peak]))
    falling_area = float(np.sum(average[peak + 1 : peak + 1 + before]))
    assert falling_area > 1.5 * rising_area


def test_guided_demo_event_counts_vary_across_sessions_and_between_rois(
    guided_demo_sessions,
):
    counts = {
        roi: np.array(
            [
                _transient_peak_indices(session[:, signal_column]).size
                for session in guided_demo_sessions
            ]
        )
        for roi, (signal_column, _) in zip(("ROI1", "ROI2"), _ROI_COLUMNS)
    }
    for roi, values in counts.items():
        assert values.min() > 0, roi
        # Not a fixed number of events per session.
        assert len(set(values.tolist())) > 10, roi
        assert float(values.std()) > 3.0, roi
    assert not np.array_equal(counts["ROI1"], counts["ROI2"])
    # Related but distinguishable overall activity.
    ratio = counts["ROI1"].sum() / counts["ROI2"].sum()
    assert 0.5 < ratio < 2.0
    assert abs(ratio - 1.0) > 0.02


def _session_event_counts(sessions, signal_column: int) -> np.ndarray:
    return np.array(
        [_transient_peak_indices(session[:, signal_column]).size for session in sessions]
    )


def test_guided_demo_activity_changes_within_each_scheduled_day(guided_demo_sessions):
    quarter = GUIDED_DEMO_SESSIONS_PER_DAY // 4
    for signal_column, _ in _ROI_COLUMNS:
        counts = _session_event_counts(guided_demo_sessions, signal_column)
        for day_start in range(0, counts.size, GUIDED_DEMO_SESSIONS_PER_DAY):
            day = counts[day_start : day_start + GUIDED_DEMO_SESSIONS_PER_DAY]
            quarters = [
                float(day[index : index + quarter].mean())
                for index in range(0, day.size, quarter)
            ]
            # Visibly non-flat within every scheduled day...
            assert max(quarters) > 1.4 * min(quarters), day_start
        # ...but not a clean noiseless ramp between neighbouring sessions.
        assert float(np.std(np.diff(counts))) > 3.0


def test_guided_demo_spans_two_days_without_repeating_the_first(guided_demo_sessions):
    assert len(guided_demo_sessions) == 2 * GUIDED_DEMO_SESSIONS_PER_DAY
    first_day = guided_demo_sessions[:GUIDED_DEMO_SESSIONS_PER_DAY]
    second_day = guided_demo_sessions[GUIDED_DEMO_SESSIONS_PER_DAY:]
    # Day 2 is newly generated data, not a copy of day 1.
    for day_one, day_two in zip(first_day, second_day):
        assert not np.array_equal(day_one, day_two)
        assert not np.allclose(day_one[:, _ROI1_SIGNAL], day_two[:, _ROI1_SIGNAL])
    for signal_column, _ in _ROI_COLUMNS:
        counts_one = _session_event_counts(first_day, signal_column)
        counts_two = _session_event_counts(second_day, signal_column)
        assert not np.array_equal(counts_one, counts_two)


def test_guided_demo_daily_activity_pattern_recurs_on_the_second_day(
    guided_demo_sessions,
):
    half_day = GUIDED_DEMO_SESSIONS_PER_DAY // 2
    for signal_column, _ in _ROI_COLUMNS:
        counts = _session_event_counts(guided_demo_sessions, signal_column)
        days = [
            counts[start : start + GUIDED_DEMO_SESSIONS_PER_DAY]
            for start in range(0, counts.size, GUIDED_DEMO_SESSIONS_PER_DAY)
        ]
        assert len(days) == 2
        # The same nominal daily phase is busier on both days.
        for day in days:
            assert day[:half_day].mean() > day[half_day:].mean()
        # Corresponding halves are comparable across days without matching.
        for index in (0, half_day):
            first = days[0][index : index + half_day].mean()
            second = days[1][index : index + half_day].mean()
            assert 0.5 < first / second < 2.0
            assert first != second


def test_guided_demo_session_baselines_vary_modestly(guided_demo_sessions):
    for column in (_ROI1_SIGNAL, _ROI1_REFERENCE, _ROI2_SIGNAL, _ROI2_REFERENCE):
        starts = np.array(
            [float(session[: GUIDED_DEMO_FS_HZ, column].mean()) for session in guided_demo_sessions]
        )
        # No two sessions begin at exactly the same fluorescence level.
        assert len(set(np.round(starts, 6).tolist())) == len(guided_demo_sessions)
        assert 0.1 < float(starts.std()) < 5.0
        assert float(starts.max() - starts.min()) < 0.25 * float(np.abs(starts.mean()))


def test_guided_demo_within_session_bleaching_is_present_and_varies(
    guided_demo_sessions,
):
    edge = 10 * GUIDED_DEMO_FS_HZ
    declines = np.array(
        [
            float(session[:edge, _ROI1_SIGNAL].mean() - session[-edge:, _ROI1_SIGNAL].mean())
            for session in guided_demo_sessions
        ]
    )
    assert float(np.median(declines)) > 0.0
    assert float(declines.std()) > 0.05
    # Bleaching stays small next to the transients riding on top of it.
    assert float(declines.max()) < 5.0


def test_guided_demo_signal_reference_relationship_is_positive_but_imperfect(
    guided_demo_sessions,
):
    for signal_column, reference_column in _ROI_COLUMNS:
        correlations = np.array(
            [
                float(np.corrcoef(session[:, signal_column], session[:, reference_column])[0, 1])
                for session in guided_demo_sessions
            ]
        )
        assert correlations.min() > 0.2
        assert float(np.median(correlations)) > 0.4
        # The reference is not a rescaled copy of the signal.
        assert correlations.max() < 0.97
        assert float(correlations.std()) > 0.01


def test_guided_demo_shared_disturbances_appear_in_signal_and_reference(
    guided_demo_sessions,
):
    matched = 0
    inspected = 0
    for session in guided_demo_sessions:
        for signal_column, reference_column in _ROI_COLUMNS:
            reference_fast = _fast_component(
                session[:, reference_column], baseline_window_sec=10.0
            )
            signal_fast = _fast_component(
                session[:, signal_column], baseline_window_sec=10.0
            )
            dip = int(np.argmin(reference_fast))
            if reference_fast[dip] > -0.6:
                continue
            inspected += 1
            nearby = signal_fast[max(0, dip - 10) : dip + 11]
            if float(nearby.min()) < -0.3:
                matched += 1
    assert inspected > 20
    # A disturbance strong enough to show in the reference also shows in the
    # signal, at a similar but not identical amplitude.
    assert matched >= 0.8 * inspected


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


def test_guided_demo_real_correction_preview_keeps_transients_for_both_rois(
    tmp_path: Path,
):
    generated = generate_guided_csv_demo(tmp_path / "source_parent", _session_count=1)
    assert generated.success, generated.message
    config_path = _write_guided_demo_config(tmp_path)
    source = generated.input_dir / "session_0001.csv"

    for roi in ("ROI1", "ROI2"):
        result = run_guided_local_correction_preview(
            source,
            tmp_path / f"preview_{roi}",
            roi=roi,
            chunk_index=0,
            input_format="custom_tabular",
            config_path=config_path,
        )
        assert result["status"] == "success", result.get("errors")
        assert not result.get("errors")

        trace_path = (
            Path(result["preview_output_dir"])
            / "method_robust_global_event_reject_trace.csv"
        )
        rows = list(csv.DictReader(trace_path.open(encoding="utf-8")))
        signal = np.array([float(row["sig_raw"]) for row in rows])
        reference = np.array([float(row["uv_raw"]) for row in rows])
        corrected = np.array([float(row["delta_f"]) for row in rows])
        assert np.isfinite(corrected).all()

        # Common variation is removed without flattening the transients.
        raw_coupling = abs(float(np.corrcoef(signal, reference)[0, 1]))
        corrected_coupling = abs(float(np.corrcoef(corrected, reference)[0, 1]))
        assert raw_coupling > 0.3
        assert corrected_coupling < 0.5 * raw_coupling
        assert float(corrected.std()) > 0.0
        prominence = (corrected.max() - corrected.mean()) / corrected.std()
        assert prominence > 5.0
        assert _transient_peak_indices(corrected).size > 5


def test_guided_demo_readme_contains_required_setup_instructions():
    text = guided_demo_readme_text()
    for required in (
        "synthetic demonstration data",
        "not real biological data",
        "CSV files, one file per session",
        "intermittent",
        "96 files across 48 scheduled hours",
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
        # The only choice is the acquisition structure; no format, preset, or
        # sampling-rate selectors.
        combos = dialog.findChildren(QComboBox)
        assert [combo.objectName() for combo in combos] == [
            "recording_structure_combo"
        ]
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
    assert "Intermittent recording, 48 hours" in demo_docs
    assert "Continuous recording, 48 hours" in demo_docs
    assert "one vendor-neutral CSV recording" in demo_docs
    assert "not visible GUI demo choices" in demo_docs
    assert "long_term_photometry_continuous_demo" in quickstart


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

    config_path = _write_guided_demo_config(tmp_path)
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
    feature_rows = list(csv.DictReader(features.open(encoding="utf-8")))
    assert {row["roi"] for row in feature_rows} == {"ROI1", "ROI2"}
    peak_counts = {
        roi: [
            int(float(row["peak_count"]))
            for row in sorted(
                (candidate for candidate in feature_rows if candidate["roi"] == roi),
                key=lambda candidate: int(candidate["chunk_id"]),
            )
        ]
        for roi in ("ROI1", "ROI2")
    }
    for roi, counts in peak_counts.items():
        assert len(counts) == 2, roi
        # Real Feature Detection finds transients, and not a fixed five.
        assert all(count > 5 for count in counts), (roi, counts)
        assert counts != [5, 5], roi
    assert peak_counts["ROI1"] != peak_counts["ROI2"]
    assert all(row["status"] == "valid" for row in feature_rows)
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
