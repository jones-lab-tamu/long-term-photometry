import unittest

import numpy as np

from photometry_pipeline.core.tonic_timeline import (
    TONIC_TIMELINE_MODE_COMPRESSED,
    TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
    TONIC_TIMELINE_MODE_REAL_ELAPSED,
    build_tonic_chunk_time_axis,
    normalize_tonic_timeline_mode,
    remap_gapfree_axis_to_elapsed_span,
    tonic_timeline_axis_label,
)


class TestTonicTimelineMode(unittest.TestCase):
    def test_mode_normalization_aliases(self):
        self.assertEqual(
            normalize_tonic_timeline_mode("elapsed"),
            TONIC_TIMELINE_MODE_REAL_ELAPSED,
        )
        self.assertEqual(
            normalize_tonic_timeline_mode("compressed"),
            TONIC_TIMELINE_MODE_COMPRESSED,
        )
        self.assertEqual(
            normalize_tonic_timeline_mode("compressed_recording_time"),
            TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
        )

    def test_real_elapsed_keeps_off_gaps(self):
        t = np.linspace(0, 10.0, 201)
        prev_end = None
        prev_dt = None
        out = []
        for i in range(3):
            t_axis, state = build_tonic_chunk_time_axis(
                time_sec_local=t,
                timeline_mode_raw=TONIC_TIMELINE_MODE_REAL_ELAPSED,
                chunk_sequence_index=i,
                actual_schedule_index=i,
                stride_sec=1800.0,
                prev_chunk_end_sec=prev_end,
                prev_dt_sec=prev_dt,
            )
            out.append(t_axis)
            prev_end = state["prev_chunk_end_sec"]
            prev_dt = state["prev_dt_sec"]
        full = np.concatenate(out)
        d = np.diff(full)
        self.assertGreater(float(np.max(d)), 1000.0)

    def test_gap_free_mode_removes_off_gaps_but_preserves_elapsed_span(self):
        t = np.linspace(0, 10.0, 201)
        prev_gap_end = None
        prev_gap_dt = None
        prev_real_end = None
        prev_real_dt = None
        out_gap = []
        out_real = []
        for i in range(3):
            t_real, real_state = build_tonic_chunk_time_axis(
                time_sec_local=t,
                timeline_mode_raw=TONIC_TIMELINE_MODE_REAL_ELAPSED,
                chunk_sequence_index=i,
                actual_schedule_index=i,
                stride_sec=1800.0,
                prev_chunk_end_sec=prev_real_end,
                prev_dt_sec=prev_real_dt,
            )
            t_gap, gap_state = build_tonic_chunk_time_axis(
                time_sec_local=t,
                timeline_mode_raw=TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
                chunk_sequence_index=i,
                actual_schedule_index=i,
                stride_sec=1800.0,
                prev_chunk_end_sec=prev_gap_end,
                prev_dt_sec=prev_gap_dt,
            )
            out_real.append(t_real)
            out_gap.append(t_gap)
            prev_real_end = real_state["prev_chunk_end_sec"]
            prev_real_dt = real_state["prev_dt_sec"]
            prev_gap_end = gap_state["prev_chunk_end_sec"]
            prev_gap_dt = gap_state["prev_dt_sec"]
        full_real = np.concatenate(out_real)
        full_gap = np.concatenate(out_gap)
        remapped = remap_gapfree_axis_to_elapsed_span(
            full_gap,
            elapsed_start_sec=float(full_real[0]),
            elapsed_end_sec=float(full_real[-1]),
        )
        d = np.diff(remapped)
        self.assertTrue(np.all(d > 0))
        self.assertLess(float(np.max(d)), 20.0)
        self.assertGreater(float(np.max(d)), 1.0)
        self.assertAlmostEqual(float(remapped[0]), float(full_real[0]), places=6)
        self.assertAlmostEqual(float(remapped[-1]), float(full_real[-1]), places=6)

    def test_axis_label_is_mode_honest(self):
        self.assertEqual(
            tonic_timeline_axis_label(TONIC_TIMELINE_MODE_REAL_ELAPSED),
            "Time (hours)",
        )
        self.assertEqual(
            tonic_timeline_axis_label(TONIC_TIMELINE_MODE_COMPRESSED),
            "Gap-free elapsed time (hours)",
        )


if __name__ == "__main__":
    unittest.main()
