"""Focused tests for the developer-only Guided demo variant utility.

Covers the two developer overrides (`_session_count`, `_tonic_scale`) and the
standalone script that drives them. The shipped demo's default behavior must be
completely unaffected.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from gui.synthetic_demo_generator import (
    GUIDED_CONTINUOUS_DEMO_DURATION_SEC,
    GUIDED_CONTINUOUS_DEMO_FS_HZ,
    GUIDED_CONTINUOUS_DEMO_TONIC_AMPLITUDE_AU,
    GUIDED_CONTINUOUS_DEMO_TONIC_PERIOD_HOURS,
    GUIDED_CONTINUOUS_DEMO_TONIC_PHASE_HOURS,
    GUIDED_DEMO_SESSION_COUNT,
    GUIDED_DEMO_SESSIONS_PER_DAY,
    GUIDED_DEMO_SESSIONS_PER_HOUR,
    GUIDED_DEMO_TONIC_AMPLITUDE_AU,
    GUIDED_DEMO_TONIC_PERIOD_HOURS,
    GUIDED_DEMO_TONIC_PHASE_HOURS,
    _guided_continuous_demo_tonic_value,
    _guided_demo_tonic_value,
    generate_guided_continuous_demo,
    generate_guided_csv_demo,
    guided_continuous_demo_readme_text,
    guided_demo_readme_text,
    guided_demo_session_filename,
    guided_demo_session_start_time,
)

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "generate_guided_demo_variant.py"
ROWS = 200  # keep generated sessions small; the tonic is a per-session constant


def _generate(parent: Path, *, sessions: int, tonic_scale: float = 1.0):
    result = generate_guided_csv_demo(
        parent,
        _session_count=sessions,
        _rows_per_session=ROWS,
        _tonic_scale=tonic_scale,
    )
    assert result.success, result.message
    return result


def _session_signal_means(folder: Path, sessions: int) -> np.ndarray:
    """Mean ROI1/ROI2 signal per session (columns 1 and 3)."""
    means = []
    for index in range(sessions):
        values = np.loadtxt(
            folder / guided_demo_session_filename(index), delimiter=",", skiprows=1
        )
        means.append([values[:, 1].mean(), values[:, 3].mean()])
    return np.asarray(means, dtype=float)


# ---------------------------------------------------------------------------
# 1. Existing/default behavior is unchanged
# ---------------------------------------------------------------------------


def test_default_tonic_scale_reproduces_the_shipped_tonic_values():
    for session_index in (0, 1, 17, 95):
        for roi_index in (0, 1):
            elapsed_hours = session_index / GUIDED_DEMO_SESSIONS_PER_HOUR
            expected = GUIDED_DEMO_TONIC_AMPLITUDE_AU[roi_index] * np.cos(
                2.0
                * np.pi
                * (elapsed_hours - GUIDED_DEMO_TONIC_PHASE_HOURS[roi_index])
                / GUIDED_DEMO_TONIC_PERIOD_HOURS[roi_index]
            )
            assert _guided_demo_tonic_value(session_index, roi_index) == pytest.approx(
                expected
            )


def test_default_readme_text_is_unchanged():
    text = guided_demo_readme_text()
    assert "- Sessions: 96 files across 48 scheduled hours" in text
    assert "- Sessions per hour: 2" in text
    assert text == guided_demo_readme_text(session_count=GUIDED_DEMO_SESSION_COUNT)


def test_default_generation_is_byte_identical_to_explicit_defaults(tmp_path):
    implicit = generate_guided_csv_demo(
        tmp_path / "implicit", _session_count=2, _rows_per_session=ROWS
    )
    explicit = _generate(tmp_path / "explicit", sessions=2, tonic_scale=1.0)
    assert implicit.success and explicit.success
    for index in range(2):
        name = guided_demo_session_filename(index)
        assert (implicit.input_dir / name).read_bytes() == (
            explicit.input_dir / name
        ).read_bytes()


# ---------------------------------------------------------------------------
# 2. Duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("days", [1, 2])
def test_session_count_scales_with_requested_days(tmp_path, days):
    sessions = days * GUIDED_DEMO_SESSIONS_PER_DAY
    result = _generate(tmp_path / f"d{days}", sessions=sessions)
    assert len(sorted(result.input_dir.glob("session_*.csv"))) == sessions
    truth = json.loads(
        (result.input_dir / "tonic_truth.json").read_text(encoding="utf-8")
    )
    assert len(truth["records"]) == sessions * 2
    assert f"- Sessions: {sessions} files" in (
        result.input_dir / "README.md"
    ).read_text(encoding="utf-8")


def test_session_timing_is_continuous_across_day_boundaries(tmp_path):
    sessions = 2 * GUIDED_DEMO_SESSIONS_PER_DAY
    result = _generate(tmp_path / "two_days", sessions=sessions)
    names = sorted(path.name for path in result.input_dir.glob("session_*.csv"))
    assert names == [guided_demo_session_filename(i) for i in range(sessions)]

    starts = [guided_demo_session_start_time(i) for i in range(sessions)]
    gaps = {
        (starts[i + 1] - starts[i]).total_seconds() for i in range(sessions - 1)
    }
    # One uniform cadence everywhere, including across midnight.
    assert gaps == {3600.0 / GUIDED_DEMO_SESSIONS_PER_HOUR}
    assert starts[-1] - starts[0] == (
        starts[1] - starts[0]
    ) * (sessions - 1)


def test_tonic_rhythm_does_not_reset_at_each_day_boundary(tmp_path):
    sessions = 2 * GUIDED_DEMO_SESSIONS_PER_DAY
    first_day = [_guided_demo_tonic_value(i, 0) for i in range(GUIDED_DEMO_SESSIONS_PER_DAY)]
    second_day = [
        _guided_demo_tonic_value(i, 0)
        for i in range(GUIDED_DEMO_SESSIONS_PER_DAY, sessions)
    ]
    # A 24 h period over a 24 h day repeats; the point is that it is one
    # continuous cosine of elapsed hours, never restarted per day.
    assert np.allclose(first_day, second_day)
    boundary = GUIDED_DEMO_SESSIONS_PER_DAY
    step_inside = abs(
        _guided_demo_tonic_value(boundary - 1, 0) - _guided_demo_tonic_value(boundary - 2, 0)
    )
    step_across = abs(
        _guided_demo_tonic_value(boundary, 0) - _guided_demo_tonic_value(boundary - 1, 0)
    )
    assert step_across == pytest.approx(step_inside, rel=0.2)


# ---------------------------------------------------------------------------
# 3. HIGH versus LOW
# ---------------------------------------------------------------------------


def test_low_tonic_is_materially_smaller_with_identical_phase_and_period(tmp_path):
    sessions = GUIDED_DEMO_SESSIONS_PER_DAY
    high = _generate(tmp_path / "high", sessions=sessions, tonic_scale=1.0)
    low = _generate(tmp_path / "low", sessions=sessions, tonic_scale=0.25)

    high_means = _session_signal_means(high.input_dir, sessions)
    low_means = _session_signal_means(low.input_dir, sessions)
    # Everything except the tonic is bit-identical, so the paired difference
    # isolates the tonic exactly and is not polluted by session-level noise.
    difference = high_means - low_means

    for roi in (0, 1):
        amplitude = GUIDED_DEMO_TONIC_AMPLITUDE_AU[roi]
        # difference = (1 - 0.25) * amplitude * cos(...), so its swing is 1.5A.
        assert difference[:, roi].ptp() == pytest.approx(1.5 * amplitude, rel=1e-3)

        analytic_high = np.array(
            [_guided_demo_tonic_value(i, roi) for i in range(sessions)]
        )
        analytic_low = np.array(
            [_guided_demo_tonic_value(i, roi, tonic_scale=0.25) for i in range(sessions)]
        )
        assert analytic_low.ptp() == pytest.approx(0.25 * analytic_high.ptp(), rel=1e-9)
        # Clearly lower, but still a large multiple of the ~0.2 AU sample noise.
        assert analytic_low.ptp() > 1.5
        # Same period and phase: peak and trough land on the same sessions.
        assert int(np.argmax(analytic_low)) == int(np.argmax(analytic_high))
        assert int(np.argmin(analytic_low)) == int(np.argmin(analytic_high))
        assert np.allclose(analytic_low, 0.25 * analytic_high)


def test_low_tonic_changes_only_the_tonic_component(tmp_path):
    sessions = 2
    high = _generate(tmp_path / "high", sessions=sessions, tonic_scale=1.0)
    low = _generate(tmp_path / "low", sessions=sessions, tonic_scale=0.25)

    for index in range(sessions):
        name = guided_demo_session_filename(index)
        high_values = np.loadtxt(high.input_dir / name, delimiter=",", skiprows=1)
        low_values = np.loadtxt(low.input_dir / name, delimiter=",", skiprows=1)
        # Timestamps and both reference channels are untouched.
        assert np.array_equal(high_values[:, 0], low_values[:, 0])
        assert np.array_equal(high_values[:, 2], low_values[:, 2])
        assert np.array_equal(high_values[:, 4], low_values[:, 4])
        for roi_index, column in enumerate((1, 3)):
            difference = high_values[:, column] - low_values[:, column]
            expected = _guided_demo_tonic_value(
                index, roi_index
            ) - _guided_demo_tonic_value(index, roi_index, tonic_scale=0.25)
            # A constant per-session offset: the phasic/noise/drift structure
            # is bit-identical between the two variants.
            assert difference.max() - difference.min() < 1e-5
            assert difference.mean() == pytest.approx(expected, abs=1e-5)


def test_tonic_truth_reports_the_amplitude_actually_generated(tmp_path):
    low = _generate(tmp_path / "low", sessions=2, tonic_scale=0.25)
    truth = json.loads((low.input_dir / "tonic_truth.json").read_text(encoding="utf-8"))
    by_roi = {record["roi_id"]: record for record in truth["records"]}
    assert by_roi["ROI1"]["tonic_amplitude_au"] == pytest.approx(
        0.25 * GUIDED_DEMO_TONIC_AMPLITUDE_AU[0]
    )
    assert by_roi["ROI2"]["tonic_amplitude_au"] == pytest.approx(
        0.25 * GUIDED_DEMO_TONIC_AMPLITUDE_AU[1]
    )
    assert by_roi["ROI1"]["tonic_period_hours"] == GUIDED_DEMO_TONIC_PERIOD_HOURS[0]
    assert by_roi["ROI1"]["tonic_phase_hours"] == GUIDED_DEMO_TONIC_PHASE_HOURS[0]


# ---------------------------------------------------------------------------
# 4. Structural validation and output safety
# ---------------------------------------------------------------------------


def test_generated_variant_passes_the_shipped_structural_validation(tmp_path):
    # generate_guided_csv_demo runs _validate_guided_demo_folder before it
    # publishes, so a successful result is that validation passing.
    result = _generate(tmp_path / "validated", sessions=3, tonic_scale=0.25)
    assert result.success
    assert (result.input_dir / "tonic_truth.json").is_file()
    assert (result.input_dir / "README.md").is_file()


def test_existing_destination_folder_is_not_overwritten(tmp_path):
    parent = tmp_path / "parent"
    first = _generate(parent, sessions=1)
    sentinel = first.input_dir / "session_0001_2025_01_01-00_00_00.csv"
    before = sentinel.read_bytes()

    second = generate_guided_csv_demo(
        parent, _session_count=1, _rows_per_session=ROWS, _tonic_scale=0.25
    )
    assert second.success is False
    assert "already exists" in second.message
    assert sentinel.read_bytes() == before


# ---------------------------------------------------------------------------
# 5. The standalone script
# ---------------------------------------------------------------------------


def _run_script(argv: list[str], monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", ["generate_guided_demo_variant.py", *argv])
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_script_generates_a_low_tonic_dataset(tmp_path, monkeypatch):
    import gui.synthetic_demo_generator as generator_module

    calls: dict = {}
    original = generator_module.generate_guided_csv_demo

    def _spy(parent, **kwargs):
        calls.update(kwargs)
        kwargs["_rows_per_session"] = ROWS
        return original(parent, **kwargs)

    monkeypatch.setattr(generator_module, "generate_guided_csv_demo", _spy)

    out = tmp_path / "script_low"
    assert (
        _run_script(
            ["--mode", "intermittent", "--days", "1", "--tonic", "low", "--output", str(out)],
            monkeypatch,
        )
        == 0
    )
    assert calls["_session_count"] == GUIDED_DEMO_SESSIONS_PER_DAY
    assert calls["_tonic_scale"] == 0.25
    dataset = out / "long_term_photometry_guided_demo"
    assert len(sorted(dataset.glob("session_*.csv"))) == GUIDED_DEMO_SESSIONS_PER_DAY


@pytest.mark.parametrize(
    ("tonic", "expected_scale"), [("high", 1.0), ("low", 0.25)]
)
def test_script_intermittent_maps_days_and_tonic(tmp_path, monkeypatch, tonic, expected_scale):
    import gui.synthetic_demo_generator as generator_module

    calls: dict = {}

    def _capture(parent, **kwargs):
        calls.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(generator_module, "generate_guided_csv_demo", _capture)
    monkeypatch.setattr(
        generator_module,
        "generate_guided_continuous_demo",
        lambda *a, **k: pytest.fail("intermittent mode must not call the continuous demo"),
    )
    _run_script(
        ["--mode", "intermittent", "--days", "12", "--tonic", tonic, "--output", str(tmp_path / "x")],
        monkeypatch,
    )
    assert calls["_tonic_scale"] == expected_scale
    assert calls["_session_count"] == 12 * GUIDED_DEMO_SESSIONS_PER_DAY
    assert "_duration_sec" not in calls


@pytest.mark.parametrize(
    ("tonic", "expected_scale"), [("high", 1.0), ("low", 0.25)]
)
def test_script_continuous_maps_days_and_tonic(tmp_path, monkeypatch, tonic, expected_scale):
    import gui.synthetic_demo_generator as generator_module

    calls: dict = {}

    def _capture(parent, **kwargs):
        calls.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(generator_module, "generate_guided_continuous_demo", _capture)
    monkeypatch.setattr(
        generator_module,
        "generate_guided_csv_demo",
        lambda *a, **k: pytest.fail("continuous mode must not call the intermittent demo"),
    )
    _run_script(
        ["--mode", "continuous", "--days", "12", "--tonic", tonic, "--output", str(tmp_path / "x")],
        monkeypatch,
    )
    assert calls["_tonic_scale"] == expected_scale
    assert calls["_duration_sec"] == pytest.approx(12 * 86400.0)
    assert "_session_count" not in calls


def test_script_requires_a_mode(tmp_path, monkeypatch):
    assert (
        _run_script(
            ["--days", "1", "--tonic", "high", "--output", str(tmp_path / "x")], monkeypatch
        )
        == 2
    )


@pytest.mark.parametrize("days", ["0", "-3", "abc"])
def test_script_rejects_non_positive_days(tmp_path, monkeypatch, days):
    assert (
        _run_script(
            ["--mode", "intermittent", "--days", days, "--tonic", "high", "--output", str(tmp_path / "x")],
            monkeypatch,
        )
        == 2
    )


def test_script_reports_failure_when_the_destination_exists(tmp_path, monkeypatch):
    import gui.synthetic_demo_generator as generator_module

    original = generator_module.generate_guided_csv_demo

    def _small(parent, **kwargs):
        kwargs["_rows_per_session"] = ROWS
        return original(parent, **kwargs)

    monkeypatch.setattr(generator_module, "generate_guided_csv_demo", _small)
    out = tmp_path / "twice"
    argv = ["--mode", "intermittent", "--days", "0.5", "--tonic", "high", "--output", str(out)]
    assert _run_script(argv, monkeypatch) == 0
    assert _run_script(argv, monkeypatch) == 1


def test_script_continuous_end_to_end_produces_a_valid_recording(tmp_path, monkeypatch):
    out = tmp_path / "script_continuous"
    assert (
        _run_script(
            ["--mode", "continuous", "--days", str(600 / 86400), "--tonic", "low", "--output", str(out)],
            monkeypatch,
        )
        == 0
    )
    dataset = out / "long_term_photometry_continuous_demo"
    assert (dataset / "continuous_recording.csv").is_file()
    truth = json.loads((dataset / "tonic_truth.json").read_text(encoding="utf-8"))
    assert truth["rois"][0]["tonic_amplitude_au"] == pytest.approx(
        0.25 * GUIDED_CONTINUOUS_DEMO_TONIC_AMPLITUDE_AU[0]
    )


# ---------------------------------------------------------------------------
# 6. Continuous demo: default behavior, duration, HIGH vs LOW
# ---------------------------------------------------------------------------

CONTINUOUS_BLOCK_SEC = 600.0  # one shipped write block; keeps these tests small


def _generate_continuous(parent: Path, *, duration_sec: float, tonic_scale: float = 1.0):
    result = generate_guided_continuous_demo(
        parent, _duration_sec=duration_sec, _tonic_scale=tonic_scale
    )
    assert result.success, result.message
    return result


def _continuous_values(folder: Path) -> np.ndarray:
    return np.loadtxt(folder / "continuous_recording.csv", delimiter=",", skiprows=1)


def test_continuous_default_tonic_scale_reproduces_the_shipped_tonic_values():
    for elapsed_sec in (0.0, 3600.0, 7 * 3600.0, 47 * 3600.0):
        for roi_index in (0, 1):
            expected = GUIDED_CONTINUOUS_DEMO_TONIC_AMPLITUDE_AU[roi_index] * np.cos(
                2.0
                * np.pi
                * (elapsed_sec / 3600.0 - GUIDED_CONTINUOUS_DEMO_TONIC_PHASE_HOURS[roi_index])
                / GUIDED_CONTINUOUS_DEMO_TONIC_PERIOD_HOURS[roi_index]
            )
            assert _guided_continuous_demo_tonic_value(
                elapsed_sec, roi_index
            ) == pytest.approx(expected)


def test_continuous_default_readme_text_is_unchanged():
    text = guided_continuous_demo_readme_text()
    assert "- Total duration: 48 hours" in text
    assert "- Sampling rate: 8 Hz" in text
    assert text == guided_continuous_demo_readme_text(
        duration_sec=GUIDED_CONTINUOUS_DEMO_DURATION_SEC
    )


def test_continuous_default_call_matches_explicit_default_private_arguments(tmp_path):
    implicit = generate_guided_continuous_demo(
        tmp_path / "implicit", _duration_sec=CONTINUOUS_BLOCK_SEC
    )
    explicit = generate_guided_continuous_demo(
        tmp_path / "explicit", _duration_sec=CONTINUOUS_BLOCK_SEC, _tonic_scale=1.0
    )
    assert implicit.success and explicit.success
    name = "continuous_recording.csv"
    assert (implicit.input_dir / name).read_bytes() == (
        explicit.input_dir / name
    ).read_bytes()


@pytest.mark.parametrize("blocks", [1, 3])
def test_continuous_sample_count_and_cadence_scale_with_duration(tmp_path, blocks):
    duration_sec = blocks * CONTINUOUS_BLOCK_SEC
    result = _generate_continuous(tmp_path / f"b{blocks}", duration_sec=duration_sec)
    values = _continuous_values(result.input_dir)

    expected_samples = int(round(duration_sec * GUIDED_CONTINUOUS_DEMO_FS_HZ))
    assert values.shape[0] == expected_samples
    steps = np.diff(values[:, 0])
    assert np.allclose(steps, 1.0 / GUIDED_CONTINUOUS_DEMO_FS_HZ, atol=1e-6)
    assert values[0, 0] == 0.0
    assert values[-1, 0] == pytest.approx(
        (expected_samples - 1) / GUIDED_CONTINUOUS_DEMO_FS_HZ, abs=1e-3
    )
    truth = json.loads(
        (result.input_dir / "tonic_truth.json").read_text(encoding="utf-8")
    )
    assert truth["duration_hours"] == pytest.approx(duration_sec / 3600.0)
    assert f"- Total duration: {duration_sec / 3600.0:g} hours" in (
        result.input_dir / "README.md"
    ).read_text(encoding="utf-8")


def test_continuous_tonic_phase_continues_across_day_boundaries():
    # One unbroken cosine of elapsed time: sampling either side of midnight
    # crosses smoothly, and a whole day later lands back on the same value.
    step = 1.0 / GUIDED_CONTINUOUS_DEMO_FS_HZ
    day = 86400.0
    before = _guided_continuous_demo_tonic_value(day - step, 0)
    at = _guided_continuous_demo_tonic_value(day, 0)
    after = _guided_continuous_demo_tonic_value(day + step, 0)
    inside = _guided_continuous_demo_tonic_value(3600.0, 0) - (
        _guided_continuous_demo_tonic_value(3600.0 - step, 0)
    )
    assert abs(at - before) == pytest.approx(abs(inside), rel=0.5)
    assert (after - at) * (at - before) > 0  # no discontinuity or reset
    assert _guided_continuous_demo_tonic_value(0.0, 0) == pytest.approx(
        _guided_continuous_demo_tonic_value(day, 0)
    )
    assert _guided_continuous_demo_tonic_value(5 * day + 1234.0, 0) == pytest.approx(
        _guided_continuous_demo_tonic_value(1234.0, 0)
    )


def test_continuous_low_tonic_is_exactly_a_quarter_with_identical_phase():
    elapsed = np.arange(0.0, 2 * 86400.0, 600.0)
    for roi_index in (0, 1):
        high = _guided_continuous_demo_tonic_value(elapsed, roi_index)
        low = _guided_continuous_demo_tonic_value(elapsed, roi_index, tonic_scale=0.25)
        assert np.allclose(low, 0.25 * high)
        assert low.ptp() == pytest.approx(0.25 * high.ptp())
        assert int(np.argmax(low)) == int(np.argmax(high))
        assert int(np.argmin(low)) == int(np.argmin(high))
        assert low.ptp() > 1.5  # still clearly detectable


def test_continuous_low_tonic_changes_only_the_signal_channels(tmp_path):
    duration_sec = CONTINUOUS_BLOCK_SEC
    high = _generate_continuous(tmp_path / "high", duration_sec=duration_sec)
    low = _generate_continuous(tmp_path / "low", duration_sec=duration_sec, tonic_scale=0.25)

    high_values = _continuous_values(high.input_dir)
    low_values = _continuous_values(low.input_dir)
    assert high_values.shape == low_values.shape

    # Timestamps and both reference channels are untouched.
    assert np.array_equal(high_values[:, 0], low_values[:, 0])
    assert np.array_equal(high_values[:, 2], low_values[:, 2])
    assert np.array_equal(high_values[:, 4], low_values[:, 4])

    for roi_index, column in enumerate((1, 3)):
        difference = high_values[:, column] - low_values[:, column]
        expected = _guided_continuous_demo_tonic_value(
            high_values[:, 0], roi_index
        ) - _guided_continuous_demo_tonic_value(
            high_values[:, 0], roi_index, tonic_scale=0.25
        )
        # Only the tonic differs; everything else is bit-identical, so the
        # residual is the tonic difference within file quantization (1e-6).
        assert np.allclose(difference, expected, atol=2e-6)


def test_continuous_existing_destination_folder_is_not_overwritten(tmp_path):
    parent = tmp_path / "parent"
    first = _generate_continuous(parent, duration_sec=CONTINUOUS_BLOCK_SEC)
    sentinel = first.input_dir / "continuous_recording.csv"
    before = sentinel.read_bytes()

    second = generate_guided_continuous_demo(
        parent, _duration_sec=CONTINUOUS_BLOCK_SEC, _tonic_scale=0.25
    )
    assert second.success is False
    assert "already exists" in second.message
    assert sentinel.read_bytes() == before
