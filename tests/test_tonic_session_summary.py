"""Focused tests for the session-level repeated-session tonic summary."""

import os

import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.core.tonic_dff import compute_global_iso_fit_robust
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.tonic_session_summary import (
    METHOD_GLOBAL_ISOSBESTIC,
    METHOD_SIGNAL_ONLY,
    MIN_FINITE_SAMPLES,
    MIN_FITTED_ISOSBESTIC,
    REASON_GLOBAL_FIT_FAILED,
    REASON_INVALID_GLOBAL_DENOMINATOR,
    REASON_NONPOSITIVE_SLOPE,
    STATUS_INSUFFICIENT,
    STATUS_INVALID_DENOMINATOR,
    STATUS_NO_FINITE,
    STATUS_UNAVAILABLE,
    STATUS_VALID,
    SUMMARY_COLUMNS,
    TONIC_PERCENTILE,
    UNITS_FRACTION_DFF,
    UNITS_RAW_AU,
    build_tonic_session_summary,
    write_tonic_session_summary,
)

HOUR = 3600.0
DAY = 86400.0
OMEGA = 2 * np.pi / DAY
NS = 600
FS = 20.0
A_TRUE = 1.5
B_TRUE = 20.0


def _write_cache(tmp_path, sessions, rois=("Region0",), fmt="rwd", subdir="tonic_out"):
    """Build a real tonic cache through the production writer.

    ``sessions`` is a list of (uv_2d, sig_2d) arrays shaped (T, n_rois).
    """
    out_dir = tmp_path / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = os.path.join(str(out_dir), "tonic_trace_cache.h5")
    with Hdf5TraceCacheWriter(cache_path, "tonic", config=None) as writer:
        for index, (uv, sig) in enumerate(sessions):
            chunk = Chunk(
                chunk_id=index,
                source_file=f"sess{index:03d}/fluorescence.csv",
                format=fmt,
                time_sec=np.arange(uv.shape[0], dtype=float) / FS,
                uv_raw=np.asarray(uv, dtype=float),
                sig_raw=np.asarray(sig, dtype=float),
                delta_f=np.zeros_like(np.asarray(sig, dtype=float)),
                fs_hz=FS,
                channel_names=list(rois),
                metadata={},
            )
            writer.add_chunk(chunk, chunk_id=index, source_file=chunk.source_file)
    return str(out_dir)


def _paired_sessions(n_sessions=48, frac_amp=0.0, seed=0, tau_days=2.0,
                     events=0, artifact=0.0, n_rois=1, roi_gain=(1.0,),
                     n_samples=NS):
    """Paired isosbestic/signal sessions with a known multiplicative rhythm."""
    rng = np.random.default_rng(seed)
    t_local = np.arange(n_samples) / FS
    out = []
    for s in range(n_sessions):
        tg = s * (HOUR / 2.0) + t_local
        uv_cols, sig_cols = [], []
        for r in range(n_rois):
            gain = roi_gain[r % len(roi_gain)]
            iso = 200.0 * np.exp(-tg / (tau_days * DAY)) + 300.0 + rng.normal(0, 0.5, n_samples)
            base = (A_TRUE * gain) * iso + B_TRUE
            ca = frac_amp * np.sin(OMEGA * (tg - 6.0 * HOUR))
            sig = base * (1.0 + ca)
            if events:
                for c in rng.choice(n_samples, events, replace=False):
                    tl = np.maximum(t_local - t_local[c], 0.0)
                    w = np.where(
                        t_local >= t_local[c],
                        (1 - np.exp(-tl / 0.05)) * np.exp(-tl / 0.5),
                        0.0,
                    )
                    sig = sig + 0.30 * float(np.mean(base)) * (w / w.max() if w.max() > 0 else w)
            if artifact:
                for c in rng.choice(n_samples, 3, replace=False):
                    tl = np.maximum(t_local - t_local[c], 0.0)
                    w = np.where(
                        t_local >= t_local[c],
                        (1 - np.exp(-tl / 0.1)) * np.exp(-tl / 2.0),
                        0.0,
                    )
                    w = w / w.max() if w.max() > 0 else w
                    sgn = 1.0 if rng.random() > 0.5 else -1.0
                    iso = iso + sgn * artifact * w
                    sig = sig + sgn * artifact * (A_TRUE * gain) * w
            uv_cols.append(iso)
            sig_cols.append(sig + rng.normal(0, 0.5, n_samples))
        out.append((np.column_stack(uv_cols), np.column_stack(sig_cols)))
    return out


def _harmonic(values, times):
    finite = np.isfinite(values)
    design = np.column_stack(
        [np.cos(OMEGA * times[finite]), np.sin(OMEGA * times[finite]), np.ones(int(finite.sum()))]
    )
    coeffs, *_ = np.linalg.lstsq(design, values[finite], rcond=None)
    amplitude = float(np.hypot(coeffs[0], coeffs[1]))
    phase = float((np.arctan2(-coeffs[0], coeffs[1]) / OMEGA / HOUR) % 24.0)
    return amplitude, phase


# ----------------------------------------------------------------- primary ---


def test_primary_method_matches_the_defined_equation_exactly(tmp_path):
    """dF/F0 = (sig - (a*uv + b)) / (a*uv + b); tonic = 2nd percentile."""
    out_dir = _write_cache(tmp_path, _paired_sessions(n_sessions=6, seed=1))
    rows = build_tonic_session_summary(out_dir)
    assert {row["tonic_method"] for row in rows} == {METHOD_GLOBAL_ISOSBESTIC}
    assert {row["units"] for row in rows} == {UNITS_FRACTION_DFF}

    import h5py

    slope = rows[0]["global_slope"]
    intercept = rows[0]["global_intercept"]
    with h5py.File(os.path.join(out_dir, "tonic_trace_cache.h5"), "r") as handle:
        for row in rows:
            group = handle["roi"]["Region0"][f"chunk_{int(row['session_index'])}"]
            sig = group["sig_raw"][()]
            uv = group["uv_raw"][()]
            fitted = slope * uv + intercept
            expected = np.percentile((sig - fitted) / fitted, TONIC_PERCENTILE)
            assert row["tonic_value"] == pytest.approx(expected)
            assert row["n_finite_samples"] == sig.size
            assert row["percentile"] == TONIC_PERCENTILE


def test_global_fit_is_estimated_once_per_roi(tmp_path):
    """Each ROI gets its own slope/intercept, shared by all of its sessions."""
    sessions = _paired_sessions(n_sessions=8, seed=2, n_rois=2, roi_gain=(1.0, 3.0))
    out_dir = _write_cache(tmp_path, sessions, rois=("Region0", "Region1"))
    frame = pd.DataFrame(build_tonic_session_summary(out_dir))

    for roi in ("Region0", "Region1"):
        roi_rows = frame[frame["roi"] == roi]
        assert roi_rows["global_slope"].nunique() == 1
        assert roi_rows["global_intercept"].nunique() == 1
    slope0 = frame[frame["roi"] == "Region0"]["global_slope"].iloc[0]
    slope1 = frame[frame["roi"] == "Region1"]["global_slope"].iloc[0]
    assert slope0 == pytest.approx(A_TRUE, rel=0.05)
    assert slope1 == pytest.approx(A_TRUE * 3.0, rel=0.05)


def test_known_tonic_rhythm_amplitude_and_phase_are_preserved(tmp_path):
    sessions = _paired_sessions(n_sessions=96, frac_amp=0.04, seed=3)
    out_dir = _write_cache(tmp_path, sessions)
    rows = build_tonic_session_summary(out_dir)
    values = np.array([row["tonic_value"] for row in rows], dtype=float)
    times = np.array([row["session_index"] for row in rows], dtype=float) * (HOUR / 2.0)

    amplitude, phase = _harmonic(values, times)
    assert amplitude == pytest.approx(0.04, rel=0.10)
    phase_error = min((phase - 6.0) % 24.0, 24.0 - (phase - 6.0) % 24.0)
    assert phase_error < 0.5


def test_positive_transients_do_not_dominate_the_session_value(tmp_path):
    """Large events on ~8% of samples must not move the 2nd-percentile value.

    Uses 200 s sessions so the event density is realistic; the default 30 s test
    session is too short for events to be anything but the majority of samples.
    """
    long_session = dict(n_sessions=48, frac_amp=0.04, seed=4, n_samples=4000)
    clean = _write_cache(tmp_path, _paired_sessions(**long_session), subdir="clean")
    spiky = _write_cache(tmp_path, _paired_sessions(events=8, **long_session), subdir="spiky")
    clean_values = np.array([r["tonic_value"] for r in build_tonic_session_summary(clean)])
    spiky_values = np.array([r["tonic_value"] for r in build_tonic_session_summary(spiky)])

    times = np.arange(clean_values.size, dtype=float) * (HOUR / 2.0)
    clean_amp, _ = _harmonic(clean_values, times)
    spiky_amp, _ = _harmonic(spiky_values, times)
    assert spiky_amp == pytest.approx(clean_amp, rel=0.10)


def test_shared_signal_reference_artifacts_are_corrected(tmp_path):
    clean = _write_cache(tmp_path, _paired_sessions(n_sessions=48, frac_amp=0.04, seed=5),
                         subdir="noartifact")
    moved = _write_cache(tmp_path,
                         _paired_sessions(n_sessions=48, frac_amp=0.04, seed=5, artifact=30.0),
                         subdir="artifact")
    clean_values = np.array([r["tonic_value"] for r in build_tonic_session_summary(clean)])
    moved_values = np.array([r["tonic_value"] for r in build_tonic_session_summary(moved)])

    times = np.arange(clean_values.size, dtype=float) * (HOUR / 2.0)
    clean_amp, _ = _harmonic(clean_values, times)
    moved_amp, _ = _harmonic(moved_values, times)
    assert moved_amp == pytest.approx(clean_amp, rel=0.10)


def test_negative_tonic_values_are_retained(tmp_path):
    out_dir = _write_cache(tmp_path, _paired_sessions(n_sessions=8, seed=6))
    values = [row["tonic_value"] for row in build_tonic_session_summary(out_dir)]
    assert all(np.isfinite(values))
    assert min(values) < 0.0


def test_tonic_output_does_not_depend_on_phasic_correction_fields(tmp_path):
    """delta_f/dff/uv_fit carry the phasic result; tonic must ignore them."""
    sessions = _paired_sessions(n_sessions=8, seed=7)
    baseline_dir = _write_cache(tmp_path, sessions, subdir="phasic_a")

    out_dir = tmp_path / "phasic_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = os.path.join(str(out_dir), "tonic_trace_cache.h5")
    with Hdf5TraceCacheWriter(cache_path, "tonic", config=None) as writer:
        for index, (uv, sig) in enumerate(sessions):
            chunk = Chunk(
                chunk_id=index,
                source_file=f"sess{index:03d}/fluorescence.csv",
                format="rwd",
                time_sec=np.arange(uv.shape[0], dtype=float) / FS,
                uv_raw=np.asarray(uv, dtype=float),
                sig_raw=np.asarray(sig, dtype=float),
                # A wildly different "phasic" result for the same raw channels.
                delta_f=np.full_like(np.asarray(sig, dtype=float), 999.0),
                fs_hz=FS,
                channel_names=["Region0"],
                metadata={},
            )
            writer.add_chunk(chunk, chunk_id=index, source_file=chunk.source_file)

    before = [r["tonic_value"] for r in build_tonic_session_summary(baseline_dir)]
    after = [r["tonic_value"] for r in build_tonic_session_summary(str(out_dir))]
    assert before == pytest.approx(after)


# ---------------------------------------------------------------- fallback ---


def test_failed_global_fit_selects_signal_only_tonic_for_the_whole_roi(tmp_path):
    """Too few paired samples for the robust global fit (needs >= 1000)."""
    rng = np.random.default_rng(8)
    sessions = []
    for s in range(12):
        iso = np.full((20, 1), 100.0) + rng.normal(0, 0.1, (20, 1))
        sig = 400.0 * np.exp(-s / 40.0) + rng.normal(0, 0.1, (20, 1))
        sessions.append((iso, sig))
    out_dir = _write_cache(tmp_path, sessions)
    rows = build_tonic_session_summary(out_dir)

    assert {row["tonic_method"] for row in rows} == {METHOD_SIGNAL_ONLY}
    assert {row["units"] for row in rows} == {UNITS_RAW_AU}
    assert all(row["fallback_reason"].startswith(REASON_GLOBAL_FIT_FAILED) for row in rows)
    assert all(np.isnan(row["global_slope"]) for row in rows)
    assert all(np.isnan(row["global_intercept"]) for row in rows)
    assert all(np.isnan(row["global_fit_n_used"]) for row in rows)
    assert all(row["status"] == STATUS_VALID for row in rows)
    assert all(np.isfinite(row["tonic_value"]) for row in rows)


def test_nonpositive_slope_selects_signal_only_tonic(tmp_path):
    """A reference anticorrelated with the signal yields slope < 0."""
    rng = np.random.default_rng(9)
    sessions = []
    for s in range(24):
        iso = np.linspace(100.0, 140.0, NS).reshape(-1, 1) + rng.normal(0, 0.2, (NS, 1))
        sig = (-2.0 * iso) + 900.0 + rng.normal(0, 0.2, (NS, 1))
        sessions.append((iso, sig))
    out_dir = _write_cache(tmp_path, sessions)
    rows = build_tonic_session_summary(out_dir)

    assert {row["tonic_method"] for row in rows} == {METHOD_SIGNAL_ONLY}
    assert all(row["fallback_reason"].startswith(REASON_NONPOSITIVE_SLOPE) for row in rows)
    assert {row["units"] for row in rows} == {UNITS_RAW_AU}


def test_fallback_applies_the_existing_offset_preserving_bleach_correction(tmp_path):
    """tonic_s == baseline_s - fitted_bleach_s + mean(fitted_bleach)."""
    from photometry_pipeline.core.tonic_output import _fit_exponential_bleach_trend

    rng = np.random.default_rng(10)
    n_sessions = 30
    sessions = []
    for s in range(n_sessions):
        iso = np.linspace(100.0, 140.0, NS).reshape(-1, 1) + rng.normal(0, 0.2, (NS, 1))
        sig = (-2.0 * iso) + 900.0 + 300.0 * np.exp(-s / 8.0) + rng.normal(0, 0.2, (NS, 1))
        sessions.append((iso, sig))
    out_dir = _write_cache(tmp_path, sessions)
    rows = build_tonic_session_summary(out_dir)
    assert {row["tonic_method"] for row in rows} == {METHOD_SIGNAL_ONLY}

    baselines = np.array(
        [np.percentile(sig[:, 0], TONIC_PERCENTILE) for _, sig in sessions], dtype=float
    )
    times = np.arange(n_sessions, dtype=float)
    trend, meta = _fit_exponential_bleach_trend(times, baselines)
    assert not meta["fallback"]
    expected = baselines - trend + float(meta["anchor"])
    assert [row["tonic_value"] for row in rows] == pytest.approx(list(expected))
    # The fallback removes the bleaching it was given.
    assert abs(np.polyfit(times, np.array([r["tonic_value"] for r in rows]), 1)[0]) < 0.5


def test_global_denominator_failure_falls_back_for_the_whole_roi(tmp_path):
    """Fit succeeds with a positive slope but yields no usable session at all.

    The reference is positive and the signal is negative, so the fitted
    isosbestic (which tracks the signal) is never safely positive anywhere.
    """
    rng = np.random.default_rng(21)
    n_sessions, n_samples = 30, 200
    sessions = []
    for s in range(n_sessions):
        uv = np.linspace(100.0, 200.0, n_samples).reshape(-1, 1) + rng.normal(
            0, 0.5, (n_samples, 1)
        )
        sig = 2.0 * uv - 800.0 - 100.0 * np.exp(-s / 8.0) + rng.normal(
            0, 0.5, (n_samples, 1)
        )
        sessions.append((uv, sig))

    # Preconditions: the global fit really does succeed with a positive slope,
    # and no session has enough safely positive fitted-isosbestic samples.
    uv_all = np.concatenate([uv[:, 0] for uv, _ in sessions])
    sig_all = np.concatenate([sig[:, 0] for _, sig in sessions])
    slope, intercept, ok, _n_used = compute_global_iso_fit_robust(uv_all, sig_all)
    assert ok is True
    assert slope > 0.0
    for uv, _sig in sessions:
        fitted = slope * uv[:, 0] + intercept
        assert int(np.sum(fitted > MIN_FITTED_ISOSBESTIC)) < MIN_FINITE_SAMPLES

    out_dir = _write_cache(tmp_path, sessions)
    rows = build_tonic_session_summary(out_dir)

    assert {row["tonic_method"] for row in rows} == {METHOD_SIGNAL_ONLY}
    assert {row["units"] for row in rows} == {UNITS_RAW_AU}
    assert all(
        row["fallback_reason"].startswith(REASON_INVALID_GLOBAL_DENOMINATOR)
        for row in rows
    )
    assert all(row["status"] == STATUS_VALID for row in rows)
    assert all(np.isfinite(row["tonic_value"]) for row in rows)
    assert all(np.isnan(row["global_slope"]) for row in rows)


def test_one_invalid_denominator_session_does_not_switch_the_roi_method(tmp_path):
    """An isolated unusable session stays NaN; the ROI keeps the primary method."""
    rng = np.random.default_rng(22)
    n_sessions, n_samples = 24, 200
    bad_index = 9
    sessions = []
    for s in range(n_sessions):
        # The bad session sits on the same signal/reference line, so it shifts
        # leverage but not the fitted slope -- only its own fitted isosbestic
        # lands below the positivity threshold.
        centre = -1000.0 if s == bad_index else 400.0
        uv = centre + rng.normal(0, 5.0, (n_samples, 1))
        sig = 2.0 * uv + 20.0 + rng.normal(0, 0.5, (n_samples, 1))
        sessions.append((uv, sig))

    out_dir = _write_cache(tmp_path, sessions)
    rows = {row["session_index"]: row for row in build_tonic_session_summary(out_dir)}

    assert {row["tonic_method"] for row in rows.values()} == {METHOD_GLOBAL_ISOSBESTIC}
    assert {row["units"] for row in rows.values()} == {UNITS_FRACTION_DFF}
    assert rows[bad_index]["status"] == STATUS_INVALID_DENOMINATOR
    assert np.isnan(rows[bad_index]["tonic_value"])
    assert rows[bad_index]["fallback_reason"] == ""
    for index, row in rows.items():
        if index != bad_index:
            assert row["status"] == STATUS_VALID
            assert np.isfinite(row["tonic_value"])


def test_methods_are_never_mixed_within_one_roi(tmp_path):
    sessions = _paired_sessions(n_sessions=16, seed=11, n_rois=2, roi_gain=(1.0, 2.0))
    out_dir = _write_cache(tmp_path, sessions, rois=("Region0", "Region1"))
    frame = pd.DataFrame(build_tonic_session_summary(out_dir))
    for roi in ("Region0", "Region1"):
        assert frame[frame["roi"] == roi]["tonic_method"].nunique() == 1
        assert frame[frame["roi"] == roi]["units"].nunique() == 1


# ------------------------------------------------- sessions, status, format ---


def test_insufficient_and_empty_sessions_get_explicit_statuses(tmp_path):
    sessions = _paired_sessions(n_sessions=12, seed=12)
    # Session 4: only 5 finite samples. Session 7: no finite samples at all.
    uv4, sig4 = sessions[4]
    sig4 = sig4.copy()
    sig4[5:, 0] = np.nan
    sessions[4] = (uv4, sig4)
    uv7, sig7 = sessions[7]
    sig7 = sig7.copy()
    sig7[:, 0] = np.nan
    sessions[7] = (uv7, sig7)

    out_dir = _write_cache(tmp_path, sessions)
    rows = {row["session_index"]: row for row in build_tonic_session_summary(out_dir)}
    assert rows[4]["status"] == STATUS_INSUFFICIENT
    assert np.isnan(rows[4]["tonic_value"])
    assert rows[7]["status"] == STATUS_NO_FINITE
    assert np.isnan(rows[7]["tonic_value"])
    assert rows[0]["status"] == STATUS_VALID


def test_nonpositive_fitted_isosbestic_samples_are_excluded_from_the_ratio():
    """Only samples with fitted_iso > threshold may enter dF/F0."""
    from photometry_pipeline.tonic_session_summary import _primary_session_value

    rng = np.random.default_rng(13)
    # Half the session has a reference that maps to exactly 0, half to +100.
    uv = np.concatenate([np.zeros(60), np.full(60, 100.0)])
    sig = np.concatenate([rng.normal(500, 1.0, 60), rng.normal(150, 1.0, 60)])
    value, status, n_finite = _primary_session_value(sig, uv, slope=1.0, intercept=0.0)

    assert status == STATUS_VALID
    assert n_finite == 60  # the zero-denominator half is excluded, not used
    expected = np.percentile((sig[60:] - 100.0) / 100.0, TONIC_PERCENTILE)
    assert value == pytest.approx(expected)


def test_invalid_denominator_status_when_too_few_positive_denominators():
    """A session left with fewer than 10 usable ratio samples is marked."""
    from photometry_pipeline.tonic_session_summary import _primary_session_value

    rng = np.random.default_rng(17)
    uv = np.concatenate([np.zeros(100), np.full(5, 100.0)])
    sig = rng.normal(200, 1.0, 105)
    value, status, n_finite = _primary_session_value(sig, uv, slope=1.0, intercept=0.0)

    assert status == STATUS_INVALID_DENOMINATOR
    assert np.isnan(value)
    assert n_finite == 5


def test_missing_middle_and_final_sessions_keep_their_timeline_positions(tmp_path, monkeypatch):
    """Authoritative missing slots stay in place with NaN tonic values."""
    import photometry_pipeline.tonic_session_summary as subject

    sessions = _paired_sessions(n_sessions=10, seed=14)
    out_dir = _write_cache(tmp_path, sessions)

    def fake_sessions(_analysis_out, _cache):
        records = []
        cache_id = 0
        for index in range(13):
            if index in (4, 11, 12):          # missing middle + missing final block
                records.append(
                    {
                        "session_index": index,
                        "cache_chunk_id": None,
                        "source_file": f"sess{index:03d}/fluorescence.csv",
                        "status": "missing_corrupted",
                        "expected_start_time": None,
                        "expected_start_time_text": "",
                    }
                )
            else:
                records.append(
                    {
                        "session_index": index,
                        "cache_chunk_id": cache_id,
                        "source_file": f"sess{index:03d}/fluorescence.csv",
                        "status": "valid",
                        "expected_start_time": None,
                        "expected_start_time_text": "",
                    }
                )
                cache_id += 1
        return records

    monkeypatch.setattr(subject, "_authoritative_sessions", fake_sessions)
    rows = build_tonic_session_summary(out_dir)

    assert [row["session_index"] for row in rows] == list(range(13))
    for index in (4, 11, 12):
        row = rows[index]
        assert np.isnan(row["tonic_value"])
        assert row["status"] == "missing_corrupted"
    assert np.isfinite(rows[3]["tonic_value"])
    assert np.isfinite(rows[10]["tonic_value"])


@pytest.mark.parametrize("fmt", ["rwd", "npm", "custom_tabular"])
def test_repeated_formats_share_one_tonic_calculation(tmp_path, fmt):
    """Post-ingestion the module reads only normalized fields: no format branch."""
    sessions = _paired_sessions(n_sessions=12, frac_amp=0.04, seed=15)
    out_dir = _write_cache(tmp_path, sessions, fmt=fmt, subdir=f"tonic_{fmt}")
    rows = build_tonic_session_summary(out_dir)
    assert len(rows) == 12
    assert {row["tonic_method"] for row in rows} == {METHOD_GLOBAL_ISOSBESTIC}
    reference = _write_cache(tmp_path, sessions, fmt="rwd", subdir="tonic_reference")
    expected = [row["tonic_value"] for row in build_tonic_session_summary(reference)]
    assert [row["tonic_value"] for row in rows] == pytest.approx(expected)


def test_saved_summary_is_written_and_reopens(tmp_path):
    out_dir = _write_cache(tmp_path, _paired_sessions(n_sessions=10, seed=16))
    target = os.path.join(str(tmp_path), "tonic_session_summary.csv")
    result = write_tonic_session_summary(out_dir, target)

    assert os.path.isfile(target)
    assert result["row_count"] == 10
    assert result["tonic_method_by_roi"] == {"Region0": METHOD_GLOBAL_ISOSBESTIC}

    frame = pd.read_csv(target)
    assert list(frame.columns) == SUMMARY_COLUMNS
    assert len(frame) == 10
    assert frame["units"].unique().tolist() == [UNITS_FRACTION_DFF]
    assert frame["percentile"].unique().tolist() == [TONIC_PERCENTILE]
    assert frame["session_index"].tolist() == sorted(frame["session_index"].tolist())


def test_missing_cache_fails_closed(tmp_path):
    from photometry_pipeline.tonic_session_summary import TonicSessionSummaryError

    with pytest.raises(TonicSessionSummaryError):
        build_tonic_session_summary(str(tmp_path / "absent"))


def test_denominator_threshold_is_positive_and_deterministic():
    assert MIN_FITTED_ISOSBESTIC > 0.0
    assert MIN_FITTED_ISOSBESTIC == 1e-9
