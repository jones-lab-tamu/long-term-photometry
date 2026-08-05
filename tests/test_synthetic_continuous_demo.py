"""The fixed Guided continuous demo, through the real continuous authorities."""

import json

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QComboBox

from gui.synthetic_demo_dialog import (
    CONTINUOUS_CHOICE,
    DEMO_CHOICES,
    INTERMITTENT_CHOICE,
    GenerateSyntheticDemoDatasetDialog,
)
from gui.synthetic_demo_generator import (
    GUIDED_CONTINUOUS_DEMO_FILE_NAME,
    GUIDED_CONTINUOUS_DEMO_FOLDER_NAME,
    GUIDED_CONTINUOUS_DEMO_FS_HZ,
    GUIDED_CONTINUOUS_DEMO_HEADERS,
    GUIDED_CONTINUOUS_DEMO_TONIC_AMPLITUDE_AU,
    GUIDED_CONTINUOUS_DEMO_TONIC_PERIOD_HOURS,
    GUIDED_CONTINUOUS_DEMO_TONIC_PHASE_HOURS,
    GUIDED_CONTINUOUS_DEMO_TONIC_TRUTH_FILENAME,
    _continuous_event_rate_modulation,
    _guided_continuous_demo_tonic_value,
    generate_guided_continuous_demo,
    guided_continuous_demo_readme_text,
)


# Long enough to clear the production minimum-duration inspection gate while
# keeping automated runs bounded. Production stays fixed at 48 hours.
BOUNDED_DURATION_SEC = 3600.0
DAY_SEC = 86400.0


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def continuous_demo(tmp_path_factory):
    parent = tmp_path_factory.mktemp("continuous_demo")
    result = generate_guided_continuous_demo(
        parent, _duration_sec=BOUNDED_DURATION_SEC
    )
    assert result.success, result.message
    return result


@pytest.fixture(scope="module")
def continuous_samples(continuous_demo):
    """Timestamp plus the four channel columns, skipping the metadata row."""
    return np.loadtxt(
        continuous_demo.input_dir / GUIDED_CONTINUOUS_DEMO_FILE_NAME,
        delimiter=",",
        skiprows=1,
    )


_TIME, _CH1_SIGNAL, _CH1_REFERENCE, _CH2_SIGNAL, _CH2_REFERENCE = range(5)
_ROI_COLUMNS = ((_CH1_SIGNAL, _CH1_REFERENCE), (_CH2_SIGNAL, _CH2_REFERENCE))


def _moving_average(values, window):
    window = max(1, int(window))
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _fast_component(trace, baseline_window_sec=20.0):
    smoothed = _moving_average(trace, 0.3 * GUIDED_CONTINUOUS_DEMO_FS_HZ)
    return smoothed - _moving_average(
        smoothed, baseline_window_sec * GUIDED_CONTINUOUS_DEMO_FS_HZ
    )


def _transient_peak_indices(trace, min_amplitude=1.0):
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
        if not kept or index - kept[-1] >= GUIDED_CONTINUOUS_DEMO_FS_HZ:
            kept.append(int(index))
    return np.array(kept, dtype=int)


# --------------------------------------------------------------------------
# Fixed source contract
# --------------------------------------------------------------------------


def test_continuous_demo_writes_one_uninterrupted_recording(continuous_demo):
    folder = continuous_demo.input_dir
    assert folder.name == GUIDED_CONTINUOUS_DEMO_FOLDER_NAME
    assert (folder / "README.md").is_file()
    source = folder / GUIDED_CONTINUOUS_DEMO_FILE_NAME
    assert source.is_file()
    # One recording file, not a per-session folder tree.
    assert sorted(path.name for path in folder.iterdir()) == sorted(
        [
            "README.md",
            GUIDED_CONTINUOUS_DEMO_FILE_NAME,
            GUIDED_CONTINUOUS_DEMO_TONIC_TRUTH_FILENAME,
        ]
    )
    with source.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").rstrip("\r")
        first_row = handle.readline().strip()
    # One header row and then numeric data: no vendor metadata row.
    assert tuple(header.split(",")) == GUIDED_CONTINUOUS_DEMO_HEADERS
    assert header.startswith("ElapsedSeconds,")
    assert first_row.split(",")[0] == "0.000"
    for vendor in ("Fps", "TimeStamp", "Events", "CH1-410", "CH1-470", "CH2-410", "CH2-470"):
        assert vendor not in header


def test_continuous_demo_time_axis_is_regular_and_half_open(continuous_samples):
    time_sec = continuous_samples[:, _TIME]
    expected_samples = int(BOUNDED_DURATION_SEC * GUIDED_CONTINUOUS_DEMO_FS_HZ)
    assert time_sec.size == expected_samples
    assert time_sec[0] == 0.0
    steps = np.diff(time_sec)
    assert np.all(steps > 0.0)
    assert np.allclose(steps, 1.0 / GUIDED_CONTINUOUS_DEMO_FS_HZ)
    assert time_sec[-1] == pytest.approx(
        BOUNDED_DURATION_SEC - 1.0 / GUIDED_CONTINUOUS_DEMO_FS_HZ
    )
    assert np.isfinite(continuous_samples).all()


def test_continuous_demo_readme_states_the_fixed_contract():
    text = guided_continuous_demo_readme_text()
    for required in (
        "synthetic demonstration data",
        "not real biological data",
        "continuous, one uninterrupted recording",
        "Total duration: 48 hours",
        "Sampling rate: 8 Hz",
        "CSV files, one continuous recording",
        "`ElapsedSeconds`",
        "`ROI1_Signal` with `ROI1_Reference`",
        "`ROI2_Signal` with `ROI2_Reference`",
        "Fixed daily anchor",
        "`07:00`",
        "`00:00:00`",
        "`19:00`",
        "repeats on the second day",
        "Phasic activity is highest",
        "`tonic_truth.json`",
        "Do not draw biological conclusions",
    ):
        assert required in text


def test_continuous_demo_writes_tonic_truth(continuous_demo):
    truth_path = continuous_demo.input_dir / GUIDED_CONTINUOUS_DEMO_TONIC_TRUTH_FILENAME
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    assert truth["recording_start"] == "00:00:00"
    assert truth["duration_hours"] == pytest.approx(1.0)
    assert truth["sampling_rate_hz"] == GUIDED_CONTINUOUS_DEMO_FS_HZ
    assert truth["tonic_signal_only"] is True
    assert "cos" in truth["tonic_equation"]
    assert "07:00" in truth["phasic_alignment"]
    assert "19:00" in truth["phasic_alignment"]
    assert [roi["roi_id"] for roi in truth["rois"]] == ["ROI1", "ROI2"]
    for roi_index, roi in enumerate(truth["rois"]):
        assert roi["tonic_period_hours"] == pytest.approx(
            GUIDED_CONTINUOUS_DEMO_TONIC_PERIOD_HOURS[roi_index]
        )
        assert roi["tonic_amplitude_au"] == pytest.approx(
            GUIDED_CONTINUOUS_DEMO_TONIC_AMPLITUDE_AU[roi_index]
        )
        assert roi["tonic_peak_phase_hours"] == pytest.approx(
            GUIDED_CONTINUOUS_DEMO_TONIC_PHASE_HOURS[roi_index]
        )
        assert roi["expected_first_day_peak_clock"] == "07:00"
        assert roi["expected_first_day_trough_clock"] == "19:00"


def test_continuous_tonic_truth_has_two_aligned_daily_cycles():
    hours = np.arange(0.0, 48.0001, 0.25)
    for roi_index in (0, 1):
        tonic = _guided_continuous_demo_tonic_value(hours * 3600.0, roi_index)
        first_day = tonic[:96]
        second_day = tonic[96:192]
        assert hours[int(np.argmax(first_day))] == pytest.approx(7.0)
        assert hours[int(np.argmin(first_day))] == pytest.approx(19.0)
        assert hours[96 + int(np.argmax(second_day))] == pytest.approx(31.0)
        assert hours[96 + int(np.argmin(second_day))] == pytest.approx(43.0)
        assert np.allclose(first_day, second_day)

    assert GUIDED_CONTINUOUS_DEMO_TONIC_PHASE_HOURS[0] == pytest.approx(
        GUIDED_CONTINUOUS_DEMO_TONIC_PHASE_HOURS[1]
    )


def test_continuous_phasic_modulation_is_aligned_and_shared():
    hours = np.arange(0.0, 48.0001, 0.25)
    modulation = _continuous_event_rate_modulation(hours * 3600.0)
    first_day = modulation[:96]
    second_day = modulation[96:192]

    assert hours[int(np.argmax(first_day))] == pytest.approx(7.0)
    assert hours[int(np.argmin(first_day))] == pytest.approx(19.0)
    assert hours[96 + int(np.argmax(second_day))] == pytest.approx(31.0)
    assert hours[96 + int(np.argmin(second_day))] == pytest.approx(43.0)
    assert np.allclose(first_day, second_day)


def test_continuous_demo_refuses_existing_folder_and_cleans_up(tmp_path):
    final_folder = tmp_path / GUIDED_CONTINUOUS_DEMO_FOLDER_NAME
    final_folder.mkdir()
    marker = final_folder / "keep_me.txt"
    marker.write_text("preserve", encoding="utf-8")

    result = generate_guided_continuous_demo(tmp_path, _duration_sec=1200.0)

    assert result.success is False
    assert "already exists" in result.message
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob(f".{GUIDED_CONTINUOUS_DEMO_FOLDER_NAME}.tmp-*"))


def test_continuous_demo_failure_leaves_no_partial_output(tmp_path):
    def fail_midway(current: int, _total: int) -> None:
        if current == 2:
            raise RuntimeError("simulated continuous failure")

    result = generate_guided_continuous_demo(
        tmp_path, progress=fail_midway, _duration_sec=3000.0
    )

    assert result.success is False
    assert "simulated continuous failure" in result.message
    assert not (tmp_path / GUIDED_CONTINUOUS_DEMO_FOLDER_NAME).exists()
    assert not list(tmp_path.glob(f".{GUIDED_CONTINUOUS_DEMO_FOLDER_NAME}.tmp-*"))


def test_continuous_demo_is_byte_reproducible(tmp_path):
    first = generate_guided_continuous_demo(tmp_path / "a", _duration_sec=1200.0)
    second = generate_guided_continuous_demo(tmp_path / "b", _duration_sec=1200.0)
    assert first.success and second.success
    assert (first.input_dir / GUIDED_CONTINUOUS_DEMO_FILE_NAME).read_bytes() == (
        second.input_dir / GUIDED_CONTINUOUS_DEMO_FILE_NAME
    ).read_bytes()
    assert (
        first.input_dir / GUIDED_CONTINUOUS_DEMO_TONIC_TRUTH_FILENAME
    ).read_bytes() == (
        second.input_dir / GUIDED_CONTINUOUS_DEMO_TONIC_TRUTH_FILENAME
    ).read_bytes()


def test_continuous_tonic_is_added_to_signal_only(tmp_path, monkeypatch):
    import gui.synthetic_demo_generator as generator

    original_amplitudes = generator.GUIDED_CONTINUOUS_DEMO_TONIC_AMPLITUDE_AU
    monkeypatch.setattr(
        generator, "GUIDED_CONTINUOUS_DEMO_TONIC_AMPLITUDE_AU", (0.0, 0.0)
    )
    without_tonic = generate_guided_continuous_demo(
        tmp_path / "without", _duration_sec=1200.0
    )
    monkeypatch.setattr(
        generator,
        "GUIDED_CONTINUOUS_DEMO_TONIC_AMPLITUDE_AU",
        original_amplitudes,
    )
    with_tonic = generate_guided_continuous_demo(
        tmp_path / "with", _duration_sec=1200.0
    )
    assert without_tonic.success and with_tonic.success

    baseline = np.loadtxt(
        without_tonic.input_dir / GUIDED_CONTINUOUS_DEMO_FILE_NAME,
        delimiter=",",
        skiprows=1,
    )
    actual = np.loadtxt(
        with_tonic.input_dir / GUIDED_CONTINUOUS_DEMO_FILE_NAME,
        delimiter=",",
        skiprows=1,
    )
    np.testing.assert_array_equal(baseline[:, _CH1_REFERENCE], actual[:, _CH1_REFERENCE])
    np.testing.assert_array_equal(baseline[:, _CH2_REFERENCE], actual[:, _CH2_REFERENCE])
    for roi_index, signal_column in enumerate((_CH1_SIGNAL, _CH2_SIGNAL)):
        expected = _guided_continuous_demo_tonic_value(
            actual[:, _TIME], roi_index
        )
        np.testing.assert_allclose(
            actual[:, signal_column] - baseline[:, signal_column],
            expected,
            rtol=0.0,
            atol=2.1e-6,
        )


# --------------------------------------------------------------------------
# Signal character
# --------------------------------------------------------------------------


def test_continuous_demo_has_asymmetric_calcium_like_transients(continuous_samples):
    residual = _fast_component(continuous_samples[:, _CH1_SIGNAL])
    before = int(1.5 * GUIDED_CONTINUOUS_DEMO_FS_HZ)
    after = int(5.0 * GUIDED_CONTINUOUS_DEMO_FS_HZ)
    segments = [
        residual[index - before : index + after]
        for index in _transient_peak_indices(
            continuous_samples[:, _CH1_SIGNAL], min_amplitude=1.5
        )
        if before <= index < residual.size - after
    ]
    assert len(segments) > 20
    average = np.mean(np.array(segments), axis=0)
    half_second = int(0.5 * GUIDED_CONTINUOUS_DEMO_FS_HZ)
    assert average[before + half_second] > 2.0 * average[before - half_second]
    rising = float(np.sum(average[:before]))
    falling = float(np.sum(average[before + 1 : before + 1 + before]))
    assert falling > 1.5 * rising


def test_continuous_demo_event_timing_is_irregular(continuous_samples):
    peaks = _transient_peak_indices(continuous_samples[:, _CH1_SIGNAL])
    assert peaks.size > 20
    gaps = np.diff(peaks) / GUIDED_CONTINUOUS_DEMO_FS_HZ
    assert float(np.std(gaps)) > 2.0
    assert float(gaps.max()) > 3.0 * float(np.median(gaps))


def test_continuous_demo_rois_are_related_but_distinct(continuous_samples):
    first = _transient_peak_indices(continuous_samples[:, _CH1_SIGNAL])
    second = _transient_peak_indices(continuous_samples[:, _CH2_SIGNAL])
    assert first.size > 10 and second.size > 10
    assert not np.array_equal(first, second)
    assert not np.allclose(
        continuous_samples[:, _CH1_SIGNAL], continuous_samples[:, _CH2_SIGNAL]
    )


def test_continuous_demo_signal_reference_coupling_is_positive_but_imperfect(
    continuous_samples,
):
    for signal_column, reference_column in _ROI_COLUMNS:
        correlation = float(
            np.corrcoef(
                continuous_samples[:, signal_column],
                continuous_samples[:, reference_column],
            )[0, 1]
        )
        assert 0.2 < correlation < 0.97


def test_continuous_demo_shared_disturbances_reach_both_channels(continuous_samples):
    matched = 0
    inspected = 0
    for signal_column, reference_column in _ROI_COLUMNS:
        reference_fast = _fast_component(
            continuous_samples[:, reference_column], baseline_window_sec=10.0
        )
        signal_fast = _fast_component(
            continuous_samples[:, signal_column], baseline_window_sec=10.0
        )
        for index in np.argsort(reference_fast)[:5]:
            if reference_fast[index] > -0.5:
                continue
            inspected += 1
            nearby = signal_fast[max(0, index - 5) : index + 6]
            if float(nearby.min()) < -0.25:
                matched += 1
    assert inspected >= 4
    assert matched >= 0.75 * inspected


def test_continuous_demo_baseline_stays_bounded_over_the_recording(continuous_samples):
    for signal_column, reference_column in _ROI_COLUMNS:
        for column in (signal_column, reference_column):
            trace = continuous_samples[:, column]
            baseline = _moving_average(trace, 60 * GUIDED_CONTINUOUS_DEMO_FS_HZ)
            # Bleaching and drift are present but never approach zero.
            assert baseline.min() > 0.75 * float(np.median(trace))
            assert baseline.max() < 1.25 * float(np.median(trace))
            assert np.isfinite(trace).all()


def test_continuous_demo_daily_modulation_spans_two_days_without_repeating():
    """Event schedules carry a daily cycle that differs between the two days."""
    from gui.synthetic_demo_generator import _continuous_event_schedule

    rng = np.random.default_rng(2026)
    starts, amplitudes, _rise, _decay = _continuous_event_schedule(
        rng, 2 * DAY_SEC, 0
    )
    first_day = starts[starts < DAY_SEC]
    second_day = starts[starts >= DAY_SEC] - DAY_SEC
    assert first_day.size > 100 and second_day.size > 100
    # Both days carry the same nominal busy/quiet structure...
    for day in (first_day, second_day):
        quarters = [
            int(np.sum((day >= edge) & (day < edge + DAY_SEC / 4)))
            for edge in np.arange(4) * DAY_SEC / 4
        ]
        assert max(quarters) > 1.4 * min(quarters)
    # ...without day 2 repeating day 1.
    assert first_day.size != second_day.size
    assert not np.array_equal(first_day[:100], second_day[:100])
    assert amplitudes.min() >= 0.6 and amplitudes.max() <= 9.0


def test_artifact_crossing_a_block_boundary_renders_as_one_waveform():
    """An artifact spanning two write blocks must not be renormalized twice."""
    from gui.synthetic_demo_generator import (
        GUIDED_CONTINUOUS_DEMO_BLOCK_SEC,
        _render_continuous_artifacts,
    )

    fs = GUIDED_CONTINUOUS_DEMO_FS_HZ
    block_samples = int(GUIDED_CONTINUOUS_DEMO_BLOCK_SEC * fs)
    boundary_sec = float(GUIDED_CONTINUOUS_DEMO_BLOCK_SEC)
    amplitude = -0.9
    duration = 4.0
    # Starts one second before the boundary and runs three seconds past it.
    schedule = (
        np.array([boundary_sec - 1.0]),
        np.array([amplitude]),
        np.array([duration]),
    )

    first_block = np.arange(0, block_samples, dtype=np.float64) / fs
    second_block = (
        np.arange(block_samples, 2 * block_samples, dtype=np.float64) / fs
    )
    whole = np.arange(0, 2 * block_samples, dtype=np.float64) / fs

    blocked = np.concatenate(
        [
            _render_continuous_artifacts(first_block, fs, schedule),
            _render_continuous_artifacts(second_block, fs, schedule),
        ]
    )
    single = _render_continuous_artifacts(whole, fs, schedule)

    # Two adjacent blocks reproduce the single-call rendering exactly.
    assert np.array_equal(blocked, single)

    span = slice(int((boundary_sec - 1.0) * fs), int((boundary_sec + 3.0) * fs))
    rendered = blocked[span]
    assert np.count_nonzero(rendered) > 10

    # The scheduled amplitude is reached once, from the complete shape.
    assert rendered.min() == pytest.approx(amplitude)
    assert np.argmin(rendered) < block_samples - int((boundary_sec - 1.0) * fs)

    # No amplitude reset or jump at the boundary: the tail past the boundary is
    # a continuing decay, never re-scaled back to full amplitude.
    tail = blocked[block_samples : block_samples + int(3.0 * fs)]
    assert abs(tail[0]) < abs(amplitude)
    assert np.all(np.diff(tail) >= -1e-12)  # magnitude only shrinks
    steps = np.abs(np.diff(blocked[span]))
    boundary_step = abs(
        blocked[block_samples] - blocked[block_samples - 1]
    )
    assert boundary_step <= 2.0 * float(np.max(steps))
    # The tail decays toward zero rather than restarting.
    assert abs(tail[-1]) < 0.25 * abs(tail[0])


# --------------------------------------------------------------------------
# Real Guided continuous authorities
# --------------------------------------------------------------------------


def test_generated_continuous_demo_passes_real_guided_continuous_authorities(
    continuous_demo,
):
    from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
        evaluate_continuous_rwd_timestamp_continuity,
    )
    from photometry_pipeline.guided_continuous_rwd_recording import (
        build_guided_continuous_rwd_recording_description,
    )
    from photometry_pipeline.guided_continuous_rwd_target_grid import (
        build_guided_continuous_rwd_target_grid,
    )
    from photometry_pipeline.io.csv_continuous_source import (
        ContinuousCsvRoiSelection,
        inspect_continuous_csv_recording,
    )

    inspection = inspect_continuous_csv_recording(
        continuous_demo.input_dir / GUIDED_CONTINUOUS_DEMO_FILE_NAME,
        time_column="ElapsedSeconds",
        time_unit="seconds",
        roi_selections=[
            ContinuousCsvRoiSelection("ROI1", "ROI1_Signal", "ROI1_Reference"),
            ContinuousCsvRoiSelection("ROI2", "ROI2_Signal", "ROI2_Reference"),
        ],
    )
    assert inspection.status == "completed", inspection.outcome_category
    assert inspection.outcome_category == "inspection_completed"
    assert inspection.source_stable
    assert inspection.parser_facts is not None
    assert inspection.parser_facts.time_column == "ElapsedSeconds"
    assert inspection.parser_facts.header_row_index == 0
    assert inspection.parser_facts.timestamp_unit == "seconds"

    roi_pairs = inspection.channels.roi_pairs
    assert [pair.roi_id for pair in roi_pairs] == ["ROI1", "ROI2"]
    assert [pair.signal_column for pair in roi_pairs] == ["ROI1_Signal", "ROI2_Signal"]
    assert [pair.reference_column for pair in roi_pairs] == ["ROI1_Reference", "ROI2_Reference"]
    assert inspection.channels.nonfinite_selected_value_count == 0
    assert inspection.channels.malformed_row_count == 0

    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("ROI1", "ROI2")
    )
    assert recording.source_format == "custom_tabular"
    assert recording.acquisition_mode == "continuous"
    assert tuple(recording.roi.included_roi_ids) == ("ROI1", "ROI2")

    continuity = evaluate_continuous_rwd_timestamp_continuity(
        recording,
        source_path=continuous_demo.input_dir / GUIDED_CONTINUOUS_DEMO_FILE_NAME,
    )
    grid = build_guided_continuous_rwd_target_grid(recording, continuity)
    assert grid is not None


# --------------------------------------------------------------------------
# Visible selection UI
# --------------------------------------------------------------------------


def test_dialog_offers_exactly_two_scientist_facing_recording_structures(qapp):
    dialog = GenerateSyntheticDemoDatasetDialog()
    try:
        combos = dialog.findChildren(QComboBox)
        assert len(combos) == 1
        combo = combos[0]
        labels = [combo.itemText(index) for index in range(combo.count())]
        assert labels == [
            "Intermittent recording, 48 hours",
            "Continuous recording, 48 hours",
        ]
        values = [combo.itemData(index) for index in range(combo.count())]
        assert values == [INTERMITTENT_CHOICE, CONTINUOUS_CHOICE]
        # Intermittent stays the default.
        assert combo.currentIndex() == 0
        assert dialog.selected_recording_structure() == INTERMITTENT_CHOICE

        visible = " ".join(labels + [dialog._status_text.toPlainText()])
        for forbidden in ("RWD", "Custom Tabular", "custom_tabular", "CSV", "Hz", "parser"):
            assert forbidden not in visible
        # No duration or sampling-rate controls were added.
        assert [
            widget.objectName() for widget in dialog.findChildren(type(dialog._output_folder_edit))
        ] == ["output_folder_edit"]
    finally:
        dialog.close()


def test_dialog_choice_dispatches_to_the_matching_generator(qapp, tmp_path, monkeypatch):
    import gui.synthetic_demo_dialog as dialog_module

    called: list[str] = []

    def fake_continuous(destination_parent, *, progress=None):
        called.append("continuous")
        return dialog_module.DemoGenerationResult(
            success=True,
            demo_type="continuous",
            input_dir=tmp_path / "continuous",
            config_path=tmp_path / "continuous" / "README.md",
            format="rwd",
            sessions_per_hour=0,
            mode="both",
            message="ok",
        )

    def fake_intermittent(destination_parent, *, progress=None):
        called.append("intermittent")
        return dialog_module.DemoGenerationResult(
            success=True,
            demo_type="intermittent",
            input_dir=tmp_path / "intermittent",
            config_path=tmp_path / "intermittent" / "README.md",
            format="custom_tabular",
            sessions_per_hour=2,
            mode="both",
            message="ok",
        )

    monkeypatch.setattr(dialog_module, "generate_guided_continuous_demo", fake_continuous)
    monkeypatch.setattr(dialog_module, "generate_guided_csv_demo", fake_intermittent)

    for choice, expected in (
        (INTERMITTENT_CHOICE, "intermittent"),
        (CONTINUOUS_CHOICE, "continuous"),
    ):
        called.clear()
        worker = dialog_module._GuidedDemoWorker(tmp_path, choice)
        worker.run()
        assert called == [expected]


def test_dialog_status_names_the_folder_for_the_selected_structure(qapp):
    dialog = GenerateSyntheticDemoDatasetDialog()
    try:
        assert GUIDED_CONTINUOUS_DEMO_FOLDER_NAME not in dialog._status_text.toPlainText()
        dialog._structure_combo.setCurrentIndex(1)
        assert dialog.selected_recording_structure() == CONTINUOUS_CHOICE
        assert GUIDED_CONTINUOUS_DEMO_FOLDER_NAME in dialog._status_text.toPlainText()
    finally:
        dialog.close()


def test_demo_choices_are_the_only_two_supported_structures():
    assert DEMO_CHOICES == (
        ("Intermittent recording, 48 hours", INTERMITTENT_CHOICE),
        ("Continuous recording, 48 hours", CONTINUOUS_CHOICE),
    )
