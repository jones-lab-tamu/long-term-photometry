from __future__ import annotations

import numpy as np
import pytest

from photometry_pipeline import guided_continuous_rwd_phasic_detection as subject
from photometry_pipeline.config import Config
from photometry_pipeline.core.feature_extraction import (
    extract_features,
    get_peak_indices_for_trace,
)
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.io.hdf5_cache_reader import open_phasic_cache

# Reuse the D1 module's synthetic-recording builder for legitimate,
# identity-bearing review_binding/target_grid authorities. Its own
# correction math is irrelevant here -- this module writes its own
# hand-crafted D1-shaped cache content directly.
from tests.test_guided_continuous_rwd_correction_pass_persistence import _build_case

FS_HZ = 10.0


@pytest.fixture(scope="module")
def authorities(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_d3b_a") / "recording"
    binding, grid, _draft, _contract, _source = _build_case(folder)
    return binding, grid


def _gaussian_bump(n: int, center: float, amplitude: float, sigma: float) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    return amplitude * np.exp(-0.5 * ((idx - center) / sigma) ** 2)


def _write_synthetic_cache(
    path: str,
    *,
    binding,
    grid,
    roi_traces: dict[str, np.ndarray],
    chunk_sample_counts: list[int],
    config: Config,
) -> None:
    """Write a phasic-mode D1-shaped cache with hand-crafted per-ROI ``dff``
    values, split across exactly the requested chunk boundaries.

    Meta identities are stamped to match ``binding``/``grid`` so
    ``detect_guided_continuous_rwd_phasic_features``'s authority validation
    passes; there is no live C4c completion object here (this is not a real
    correction pass), so ``continuous_completion_identity`` is a
    self-consistent placeholder -- the kernel only requires it to be
    non-empty, never independently re-verified in this call context (see
    module docstring).
    """
    included_roi_ids = list(binding.recording.roi.included_roi_ids)
    total = sum(chunk_sample_counts)
    assert total == grid.target_sample_count, (total, grid.target_sample_count)

    writer = Hdf5TraceCacheWriter(path, "phasic", config)
    start = 0
    for chunk_id, n in enumerate(chunk_sample_counts):
        stop = start + n
        local_time = np.arange(n, dtype=np.float64) / FS_HZ
        window_start = start / FS_HZ
        window_end = (stop - 1) / FS_HZ
        dff_cols = [np.asarray(roi_traces[roi][start:stop], dtype=np.float64) for roi in included_roi_ids]
        chunk = Chunk(
            chunk_id=chunk_id,
            source_file="synthetic-continuous-source",
            format="rwd",
            time_sec=local_time,
            uv_raw=np.zeros((n, len(included_roi_ids)), dtype=np.float64),
            sig_raw=np.zeros((n, len(included_roi_ids)), dtype=np.float64),
            dff=np.column_stack(dff_cols),
            fs_hz=FS_HZ,
            channel_names=included_roi_ids,
            metadata={
                "acquisition_mode": "continuous",
                "window_index": chunk_id,
                "window_start_sec": window_start,
                "window_end_sec": window_end,
                "window_duration_sec": window_end - window_start,
            },
        )
        writer.add_chunk(chunk, chunk_id=chunk_id, source_file="synthetic-continuous-source")
        start = stop

    writer.meta.attrs["continuous_acquisition_mode"] = "continuous"
    writer.meta.attrs["continuous_completion_identity"] = "a" * 64
    writer.meta.attrs["continuous_recording_identity"] = binding.recording.recording_identity
    writer.meta.attrs["continuous_target_grid_identity"] = grid.target_grid_identity
    writer.meta.attrs["continuous_correction_segment_plan_identity"] = "b" * 64
    writer.meta.attrs["continuous_target_sample_count"] = total
    writer.meta.attrs["continuous_corrected_segment_count"] = len(chunk_sample_counts)
    writer.finalize()


def _write_cache_with_raw_chunks(
    path: str,
    *,
    binding,
    grid,
    included_roi_ids: list[str],
    per_chunk: list[dict],
    config: Config,
) -> None:
    """Write a phasic-mode cache from fully explicit per-chunk content, for
    constructing deliberately malformed cache-reconstruction fixtures.

    Each entry in ``per_chunk`` is a dict with keys ``time_sec`` (1-D array,
    per ROI or shared), ``dff`` (dict roi_id -> 1-D array), ``fs_hz``,
    ``window_start_sec``, ``window_end_sec``, ``source_file``. Meta
    identities are stamped to match ``binding``/``grid`` exactly as
    ``_write_synthetic_cache`` does.
    """
    writer = Hdf5TraceCacheWriter(path, "phasic", config)
    total = 0
    for chunk_id, spec in enumerate(per_chunk):
        time_sec = np.asarray(spec["time_sec"], dtype=np.float64)
        n = time_sec.shape[0]
        dff_cols = [np.asarray(spec["dff"][roi], dtype=np.float64) for roi in included_roi_ids]
        chunk = Chunk(
            chunk_id=chunk_id,
            source_file=spec["source_file"],
            format="rwd",
            time_sec=time_sec,
            uv_raw=np.zeros((n, len(included_roi_ids)), dtype=np.float64),
            sig_raw=np.zeros((n, len(included_roi_ids)), dtype=np.float64),
            dff=np.column_stack(dff_cols),
            fs_hz=float(spec["fs_hz"]),
            channel_names=included_roi_ids,
            metadata={
                "acquisition_mode": "continuous",
                "window_index": chunk_id,
                "window_start_sec": float(spec["window_start_sec"]),
                "window_end_sec": float(spec["window_end_sec"]),
                "window_duration_sec": float(spec["window_end_sec"]) - float(spec["window_start_sec"]),
            },
        )
        writer.add_chunk(chunk, chunk_id=chunk_id, source_file=spec["source_file"])
        total += n

    writer.meta.attrs["continuous_acquisition_mode"] = "continuous"
    writer.meta.attrs["continuous_completion_identity"] = "a" * 64
    writer.meta.attrs["continuous_recording_identity"] = binding.recording.recording_identity
    writer.meta.attrs["continuous_target_grid_identity"] = grid.target_grid_identity
    writer.meta.attrs["continuous_correction_segment_plan_identity"] = "b" * 64
    writer.meta.attrs["continuous_target_sample_count"] = total
    writer.meta.attrs["continuous_corrected_segment_count"] = len(per_chunk)
    writer.finalize()


def _reference_detection(trace: np.ndarray, config: Config):
    """The scientific reference: the existing detector run once on one
    unbroken trace, via the exact production functions."""
    n = trace.shape[0]
    chunk = Chunk(
        chunk_id=0,
        source_file="reference",
        format="rwd",
        time_sec=np.arange(n, dtype=np.float64) / FS_HZ,
        uv_raw=np.zeros((n, 1), dtype=np.float64),
        sig_raw=np.zeros((n, 1), dtype=np.float64),
        dff=trace.reshape(-1, 1),
        fs_hz=FS_HZ,
        channel_names=["ROI"],
        metadata={},
    )
    row = extract_features(chunk, config).iloc[0]
    peak_idx, pol = get_peak_indices_for_trace(trace, FS_HZ, config, return_polarities=True)
    return row, peak_idx, pol


def _independent_chunk_peak_count(trace: np.ndarray, chunk_sample_counts: list[int], config: Config) -> int:
    """The naive (scientifically wrong) approach this kernel exists to
    avoid: run the detector independently on each storage chunk."""
    total = 0
    start = 0
    for n in chunk_sample_counts:
        stop = start + n
        idx = get_peak_indices_for_trace(trace[start:stop], FS_HZ, config)
        total += len(idx)
        start = stop
    return total


# ---------------------------------------------------------------------------
# Comprehensive multi-scenario synthetic fixture
# ---------------------------------------------------------------------------
#
# Chunk boundaries at multiples of 97 samples (chosen to be unrelated to any
# peak-shape parameter), except a deliberately short final chunk -- exactly
# mirroring one accepted short final D1 storage chunk. ROI1 carries every
# boundary scenario; ROI2 carries an independent, non-overlapping set of
# peaks to prove per-ROI isolation.


@pytest.fixture(scope="module")
def scenario(authorities, tmp_path_factory):
    binding, grid = authorities
    total = grid.target_sample_count
    nominal = 97
    chunk_sample_counts = [nominal] * (total // nominal)
    remainder = total - sum(chunk_sample_counts)
    if remainder:
        chunk_sample_counts.append(remainder)
    else:
        # Force a genuinely short final chunk regardless of total's factors.
        chunk_sample_counts[-1] -= 40
        chunk_sample_counts.append(40)
    assert sum(chunk_sample_counts) == total
    boundaries = np.cumsum(chunk_sample_counts)[:-1]  # sample index where each new chunk starts

    rng = np.random.default_rng(20260722)
    roi1 = rng.normal(0.0, 0.3, size=total)
    roi2 = rng.normal(0.0, 0.3, size=total)

    b0 = int(boundaries[0])  # boundary between chunk 0 and 1 (index 97)
    b3 = int(boundaries[3])  # a later boundary, comfortably inside the trace
    b7 = int(boundaries[7])
    tail_start = int(sum(chunk_sample_counts[:-1]))

    # ROI1 scenarios:
    peak_amp, peak_sigma = 12.0, 2.5
    roi1 += _gaussian_bump(total, b0, peak_amp, peak_sigma)  # straddles boundary 0/1 exactly
    roi1 += _gaussian_bump(total, b3 - 4, peak_amp, peak_sigma)  # refractory pair around boundary 3
    roi1 += _gaussian_bump(total, b3 + 3, peak_amp, peak_sigma)  # (7 samples apart < 10-sample min distance)
    roi1 += _gaussian_bump(total, b7 - 9, peak_amp, peak_sigma)  # near-boundary VALID pair around boundary 7
    roi1 += _gaussian_bump(total, b7 + 8, peak_amp, peak_sigma)  # (17 samples apart > 10-sample min distance)
    tail_peak_center = tail_start + min(20, chunk_sample_counts[-1] // 2)
    roi1 += _gaussian_bump(total, tail_peak_center, peak_amp, peak_sigma)  # inside the short final tail

    # ROI2: independent, non-overlapping single clean peak far from ROI1's events.
    roi2_center = int(boundaries[15]) if len(boundaries) > 15 else total // 2
    roi2 += _gaussian_bump(total, roi2_center, peak_amp, peak_sigma)

    cache_path = str(tmp_path_factory.mktemp("cr1_d3b_a_cache") / "corrected.h5")
    config = Config()
    _write_synthetic_cache(
        cache_path,
        binding=binding,
        grid=grid,
        roi_traces={"ROI1": roi1, "ROI2": roi2},
        chunk_sample_counts=chunk_sample_counts,
        config=config,
    )

    result = subject.detect_guided_continuous_rwd_phasic_features(
        cache_path, review_binding=binding, target_grid=grid, config=config
    )

    ref1_row, ref1_idx, ref1_pol = _reference_detection(roi1, config)
    ref2_row, ref2_idx, ref2_pol = _reference_detection(roi2, config)

    return {
        "binding": binding,
        "grid": grid,
        "config": config,
        "cache_path": cache_path,
        "chunk_sample_counts": chunk_sample_counts,
        "boundaries": boundaries,
        "b0": b0,
        "b3": b3,
        "b7": b7,
        "tail_start": tail_start,
        "roi1": roi1,
        "roi2": roi2,
        "result": result,
        "ref1": (ref1_row, ref1_idx, ref1_pol),
        "ref2": (ref2_row, ref2_idx, ref2_pol),
    }


# ---------------------------------------------------------------------------
# Exact reference equivalence
# ---------------------------------------------------------------------------


def test_kernel_matches_unbroken_reference_exactly_roi1(scenario):
    result = scenario["result"]
    ref_row, ref_idx, ref_pol = scenario["ref1"]
    detection = result.per_roi["ROI1"]

    assert detection.event_count == int(ref_row["peak_count"])
    assert detection.mean == pytest.approx(float(ref_row["mean"]))
    assert detection.median == pytest.approx(float(ref_row["median"]))
    assert detection.std == pytest.approx(float(ref_row["std"]))
    assert detection.mad == pytest.approx(float(ref_row["mad"]))
    assert detection.auc == pytest.approx(float(ref_row["auc"]))

    expected_times = ref_idx.astype(np.float64) / FS_HZ
    np.testing.assert_allclose(detection.peak_global_times_sec, expected_times, atol=1e-9)
    np.testing.assert_array_equal(detection.peak_polarities, ref_pol)


def test_kernel_matches_unbroken_reference_exactly_roi2(scenario):
    result = scenario["result"]
    ref_row, ref_idx, ref_pol = scenario["ref2"]
    detection = result.per_roi["ROI2"]

    assert detection.event_count == int(ref_row["peak_count"])
    assert detection.auc == pytest.approx(float(ref_row["auc"]))
    expected_times = ref_idx.astype(np.float64) / FS_HZ
    np.testing.assert_allclose(detection.peak_global_times_sec, expected_times, atol=1e-9)


# ---------------------------------------------------------------------------
# Boundary-specific scientific proofs
# ---------------------------------------------------------------------------


def test_boundary_straddling_peak_not_lost_or_duplicated(scenario):
    """The peak centered exactly on a chunk boundary must appear exactly
    once, at its true global time."""
    ref_row, ref_idx, _ = scenario["ref1"]
    detection = scenario["result"].per_roi["ROI1"]
    expected_time = scenario["b0"] / FS_HZ
    matches = np.isclose(detection.peak_global_times_sec, expected_time, atol=0.15)
    assert matches.sum() == 1
    # And the reference (ground truth) agrees it is exactly one peak there too.
    ref_matches = np.isclose(ref_idx / FS_HZ, expected_time, atol=0.15)
    assert ref_matches.sum() == 1


def test_refractory_pair_across_boundary_matches_reference(scenario):
    """Two candidate peaks 7 samples (0.7s) apart straddling a boundary are
    closer than peak_min_distance_sec=1.0s: exactly one must survive, and the
    kernel must agree with the unbroken reference on which one."""
    b3 = scenario["b3"]
    window = (b3 - 20, b3 + 20)
    ref_row, ref_idx, _ = scenario["ref1"]
    ref_local = ref_idx[(ref_idx >= window[0]) & (ref_idx < window[1])]
    assert len(ref_local) == 1, "reference detector must suppress one of the refractory pair"

    detection = scenario["result"].per_roi["ROI1"]
    kernel_times_in_window = detection.peak_global_times_sec[
        (detection.peak_global_times_sec >= window[0] / FS_HZ)
        & (detection.peak_global_times_sec < window[1] / FS_HZ)
    ]
    assert len(kernel_times_in_window) == 1
    assert kernel_times_in_window[0] == pytest.approx(ref_local[0] / FS_HZ)


def test_near_boundary_valid_pair_both_retained(scenario):
    """Two candidate peaks 17 samples (1.7s) apart straddling a boundary are
    farther apart than peak_min_distance_sec=1.0s: both must be retained."""
    b7 = scenario["b7"]
    window = (b7 - 20, b7 + 20)
    ref_row, ref_idx, _ = scenario["ref1"]
    ref_local = ref_idx[(ref_idx >= window[0]) & (ref_idx < window[1])]
    assert len(ref_local) == 2

    detection = scenario["result"].per_roi["ROI1"]
    kernel_times_in_window = detection.peak_global_times_sec[
        (detection.peak_global_times_sec >= window[0] / FS_HZ)
        & (detection.peak_global_times_sec < window[1] / FS_HZ)
    ]
    assert len(kernel_times_in_window) == 2
    np.testing.assert_allclose(sorted(kernel_times_in_window), sorted(ref_local / FS_HZ))


def test_final_short_tail_event_included(scenario):
    """The final chunk is deliberately shorter than nominal; a valid event
    placed inside it must still be detected and reported at its true global
    time, not dropped merely because it belongs to the short tail."""
    tail_start = scenario["tail_start"]
    detection = scenario["result"].per_roi["ROI1"]
    tail_times = detection.peak_global_times_sec[
        detection.peak_global_times_sec >= tail_start / FS_HZ
    ]
    assert len(tail_times) == 1
    assert scenario["chunk_sample_counts"][-1] < 97  # the tail really is short


def test_global_timing_not_segment_local(scenario):
    """Event times must be continuous recording time, never restarting at 0
    within a chunk. All ROI1 peaks after the first storage chunk must report
    a global time larger than any possible segment-local time for a chunk of
    size 97 (0.0..9.6s)."""
    detection = scenario["result"].per_roi["ROI1"]
    later_events = detection.peak_global_times_sec[detection.peak_global_times_sec > 20.0]
    assert len(later_events) >= 3
    assert np.all(np.diff(np.sort(detection.peak_global_times_sec)) >= 0.0)


def test_roi_isolation_canonical_order(scenario):
    result = scenario["result"]
    binding = scenario["binding"]
    assert result.included_roi_ids == tuple(binding.recording.roi.included_roi_ids)
    # ROI2's single peak must not appear in ROI1's results and vice versa.
    roi1_times = set(np.round(scenario["result"].per_roi["ROI1"].peak_global_times_sec, 1))
    roi2_times = set(np.round(scenario["result"].per_roi["ROI2"].peak_global_times_sec, 1))
    assert roi1_times.isdisjoint(roi2_times)
    assert scenario["result"].per_roi["ROI2"].event_count == 1


# ---------------------------------------------------------------------------
# Threshold uses recording-global statistics, not per-chunk statistics
# ---------------------------------------------------------------------------


def test_threshold_uses_global_statistics_not_per_chunk(authorities, tmp_path_factory):
    """Construct a recording with a low-noise region and a high-noise
    region. mean_std threshold estimation over the WHOLE recording differs
    materially from a threshold computed from only the low-noise region --
    prove the kernel's detection matches the global (established) reference,
    not a per-chunk-local threshold."""
    binding, grid = authorities
    total = grid.target_sample_count
    config = Config()  # peak_threshold_method='mean_std', k=2.5 (default)

    rng = np.random.default_rng(4)
    trace = np.concatenate(
        [
            rng.normal(0.0, 0.3, size=total // 2),  # low-noise half
            rng.normal(0.0, 4.0, size=total - total // 2),  # high-noise half
        ]
    )
    # A moderate bump in the low-noise half: well above a low-noise-only
    # threshold, but (by construction, verified below) not necessarily above
    # the recording-global threshold once the high-noise half is included.
    bump_center = total // 4
    trace += _gaussian_bump(total, bump_center, 3.0, 2.5)

    low_noise_only = trace[: total // 2]
    global_bounds = get_peak_indices_for_trace(trace, FS_HZ, config)
    local_bounds_idx = get_peak_indices_for_trace(low_noise_only, FS_HZ, config)

    # The two approaches must actually differ for this test to mean anything.
    assert set(local_bounds_idx.tolist()) != set(
        global_bounds[global_bounds < total // 2].tolist()
    ) or len(local_bounds_idx) != len(global_bounds[global_bounds < total // 2])

    chunk_sample_counts = [97] * (total // 97)
    remainder = total - sum(chunk_sample_counts)
    if remainder:
        chunk_sample_counts.append(remainder)
    cache_path = str(tmp_path_factory.mktemp("cr1_d3b_a_threshold") / "corrected.h5")
    _write_synthetic_cache(
        cache_path,
        binding=binding,
        grid=grid,
        roi_traces={"ROI1": trace, "ROI2": np.zeros(total)},
        chunk_sample_counts=chunk_sample_counts,
        config=config,
    )
    result = subject.detect_guided_continuous_rwd_phasic_features(
        cache_path, review_binding=binding, target_grid=grid, config=config
    )
    expected_times = global_bounds.astype(np.float64) / FS_HZ
    np.testing.assert_allclose(
        result.per_roi["ROI1"].peak_global_times_sec, expected_times, atol=1e-9
    )


# ---------------------------------------------------------------------------
# Demonstrates the defect this kernel exists to fix (section 4/16)
# ---------------------------------------------------------------------------


def test_independent_chunk_processing_differs_from_unbroken_reference(scenario):
    ref_row, ref_idx, _ = scenario["ref1"]
    independent_count = _independent_chunk_peak_count(
        scenario["roi1"], scenario["chunk_sample_counts"], scenario["config"]
    )
    assert independent_count != int(ref_row["peak_count"])
    # And the kernel (this module) matches the reference, not the naive count.
    assert scenario["result"].per_roi["ROI1"].event_count == int(ref_row["peak_count"])
    assert scenario["result"].per_roi["ROI1"].event_count != independent_count


# ---------------------------------------------------------------------------
# Uniform continuous timebase refusals
# ---------------------------------------------------------------------------


def test_irregular_within_chunk_cadence_is_rejected(authorities, tmp_path_factory):
    binding, grid = authorities
    total = grid.target_sample_count
    expected_fs_hz = 1.0 / float(grid.cadence_fraction)
    config = Config()

    time_sec = np.arange(total, dtype=np.float64) / expected_fs_hz
    # Strictly increasing, but the back half's interval is doubled -- not
    # uniformly 1/fs_hz.
    half = total // 2
    time_sec[half:] = time_sec[half - 1] + (
        np.arange(1, total - half + 1, dtype=np.float64) * (2.0 / expected_fs_hz)
    )

    cache_path = str(tmp_path_factory.mktemp("cr1_d3b_a_irregular") / "corrected.h5")
    _write_cache_with_raw_chunks(
        cache_path,
        binding=binding,
        grid=grid,
        included_roi_ids=["ROI1", "ROI2"],
        per_chunk=[
            {
                "time_sec": time_sec,
                "dff": {"ROI1": np.zeros(total), "ROI2": np.zeros(total)},
                "fs_hz": expected_fs_hz,
                "window_start_sec": 0.0,
                "window_end_sec": float(time_sec[-1]),
                "source_file": "synthetic-continuous-source",
            }
        ],
        config=config,
    )
    with pytest.raises(subject.GuidedContinuousRwdPhasicDetectionError):
        subject.detect_guided_continuous_rwd_phasic_features(
            cache_path, review_binding=binding, target_grid=grid, config=config
        )


def test_nonzero_local_time_origin_is_rejected(authorities, tmp_path_factory):
    binding, grid = authorities
    total = grid.target_sample_count
    expected_fs_hz = 1.0 / float(grid.cadence_fraction)
    config = Config()

    time_sec = np.arange(total, dtype=np.float64) / expected_fs_hz + 0.05

    cache_path = str(tmp_path_factory.mktemp("cr1_d3b_a_nonzero_origin") / "corrected.h5")
    _write_cache_with_raw_chunks(
        cache_path,
        binding=binding,
        grid=grid,
        included_roi_ids=["ROI1", "ROI2"],
        per_chunk=[
            {
                "time_sec": time_sec,
                "dff": {"ROI1": np.zeros(total), "ROI2": np.zeros(total)},
                "fs_hz": expected_fs_hz,
                "window_start_sec": 0.0,
                "window_end_sec": float(time_sec[-1]),
                "source_file": "synthetic-continuous-source",
            }
        ],
        config=config,
    )
    with pytest.raises(subject.GuidedContinuousRwdPhasicDetectionError):
        subject.detect_guided_continuous_rwd_phasic_features(
            cache_path, review_binding=binding, target_grid=grid, config=config
        )


def test_event_time_length_mismatch_is_rejected(authorities, tmp_path_factory):
    binding, grid = authorities
    total = grid.target_sample_count
    expected_fs_hz = 1.0 / float(grid.cadence_fraction)
    config = Config()

    time_sec = np.arange(total, dtype=np.float64) / expected_fs_hz
    short_dff = np.zeros(total - 1)  # deliberately one sample short, both ROIs

    cache_path = str(tmp_path_factory.mktemp("cr1_d3b_a_length_mismatch") / "corrected.h5")
    _write_cache_with_raw_chunks(
        cache_path,
        binding=binding,
        grid=grid,
        included_roi_ids=["ROI1", "ROI2"],
        per_chunk=[
            {
                "time_sec": time_sec,
                "dff": {"ROI1": short_dff, "ROI2": short_dff},
                "fs_hz": expected_fs_hz,
                "window_start_sec": 0.0,
                "window_end_sec": float(time_sec[-1]),
                "source_file": "synthetic-continuous-source",
            }
        ],
        config=config,
    )
    with pytest.raises(subject.GuidedContinuousRwdPhasicDetectionError):
        subject.detect_guided_continuous_rwd_phasic_features(
            cache_path, review_binding=binding, target_grid=grid, config=config
        )


def test_source_file_mismatch_across_chunks_is_rejected(authorities, tmp_path_factory):
    binding, grid = authorities
    total = grid.target_sample_count
    expected_fs_hz = 1.0 / float(grid.cadence_fraction)
    config = Config()
    n1 = total // 2
    n2 = total - n1

    def _chunk(n, start_index, source_file):
        local_time = np.arange(n, dtype=np.float64) / expected_fs_hz
        window_start = start_index / expected_fs_hz
        return {
            "time_sec": local_time,
            "dff": {"ROI1": np.zeros(n), "ROI2": np.zeros(n)},
            "fs_hz": expected_fs_hz,
            "window_start_sec": window_start,
            "window_end_sec": window_start + local_time[-1],
            "source_file": source_file,
        }

    cache_path = str(tmp_path_factory.mktemp("cr1_d3b_a_source_mismatch") / "corrected.h5")
    _write_cache_with_raw_chunks(
        cache_path,
        binding=binding,
        grid=grid,
        included_roi_ids=["ROI1", "ROI2"],
        per_chunk=[
            _chunk(n1, 0, "synthetic-continuous-source-A"),
            _chunk(n2, n1, "synthetic-continuous-source-B"),
        ],
        config=config,
    )
    with pytest.raises(subject.GuidedContinuousRwdPhasicDetectionError):
        subject.detect_guided_continuous_rwd_phasic_features(
            cache_path, review_binding=binding, target_grid=grid, config=config
        )


def test_cache_cadence_disagreeing_with_target_grid_is_rejected(authorities, tmp_path_factory):
    binding, grid = authorities
    total = grid.target_sample_count
    wrong_fs_hz = (1.0 / float(grid.cadence_fraction)) + 0.5  # far outside tolerance
    config = Config()

    time_sec = np.arange(total, dtype=np.float64) / wrong_fs_hz

    cache_path = str(tmp_path_factory.mktemp("cr1_d3b_a_cadence_mismatch") / "corrected.h5")
    _write_cache_with_raw_chunks(
        cache_path,
        binding=binding,
        grid=grid,
        included_roi_ids=["ROI1", "ROI2"],
        per_chunk=[
            {
                "time_sec": time_sec,
                "dff": {"ROI1": np.zeros(total), "ROI2": np.zeros(total)},
                "fs_hz": wrong_fs_hz,
                "window_start_sec": 0.0,
                "window_end_sec": float(time_sec[-1]),
                "source_file": "synthetic-continuous-source",
            }
        ],
        config=config,
    )
    with pytest.raises(subject.GuidedContinuousRwdPhasicDetectionError):
        subject.detect_guided_continuous_rwd_phasic_features(
            cache_path, review_binding=binding, target_grid=grid, config=config
        )


# ---------------------------------------------------------------------------
# Authority validation, cancellation
# ---------------------------------------------------------------------------


def test_recording_identity_mismatch_is_rejected(authorities, tmp_path_factory):
    binding, grid = authorities
    other_folder = tmp_path_factory.mktemp("cr1_d3b_a_other") / "recording"
    other_binding, _other_grid, _d, _c, _s = _build_case(other_folder, phase=1.3)
    total = grid.target_sample_count
    config = Config()
    cache_path = str(tmp_path_factory.mktemp("cr1_d3b_a_mismatch") / "corrected.h5")
    _write_synthetic_cache(
        cache_path,
        binding=binding,
        grid=grid,
        roi_traces={"ROI1": np.zeros(total), "ROI2": np.zeros(total)},
        chunk_sample_counts=[total],
        config=config,
    )
    with pytest.raises(subject.GuidedContinuousRwdPhasicDetectionError):
        subject.detect_guided_continuous_rwd_phasic_features(
            cache_path, review_binding=other_binding, target_grid=grid, config=config
        )


def test_missing_cache_is_rejected(authorities):
    binding, grid = authorities
    with pytest.raises(subject.GuidedContinuousRwdPhasicDetectionError):
        subject.detect_guided_continuous_rwd_phasic_features(
            "does-not-exist.h5", review_binding=binding, target_grid=grid, config=Config()
        )


def test_cancellation_before_roi_is_honored(authorities, tmp_path_factory):
    binding, grid = authorities
    total = grid.target_sample_count
    config = Config()
    cache_path = str(tmp_path_factory.mktemp("cr1_d3b_a_cancel") / "corrected.h5")
    _write_synthetic_cache(
        cache_path,
        binding=binding,
        grid=grid,
        roi_traces={"ROI1": np.zeros(total), "ROI2": np.zeros(total)},
        chunk_sample_counts=[total],
        config=config,
    )
    with pytest.raises(subject.GuidedContinuousRwdPhasicDetectionError):
        subject.detect_guided_continuous_rwd_phasic_features(
            cache_path,
            review_binding=binding,
            target_grid=grid,
            config=config,
            cancellation_requested=lambda: True,
        )


def test_cancellation_between_rois_stops_before_second_roi(authorities, tmp_path_factory):
    binding, grid = authorities
    total = grid.target_sample_count
    config = Config()
    cache_path = str(tmp_path_factory.mktemp("cr1_d3b_a_cancel2") / "corrected.h5")
    _write_synthetic_cache(
        cache_path,
        binding=binding,
        grid=grid,
        roi_traces={"ROI1": np.zeros(total), "ROI2": np.zeros(total)},
        chunk_sample_counts=[total],
        config=config,
    )
    calls = {"count": 0}

    def cancel_after_first_roi():
        calls["count"] += 1
        return calls["count"] > 3  # allow ROI1's 3 checkpoints, cancel before ROI2

    with pytest.raises(subject.GuidedContinuousRwdPhasicDetectionError):
        subject.detect_guided_continuous_rwd_phasic_features(
            cache_path,
            review_binding=binding,
            target_grid=grid,
            config=config,
            cancellation_requested=cancel_after_first_roi,
        )
    assert calls["count"] >= 4
