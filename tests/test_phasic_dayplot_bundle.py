import os
import sys
import unittest
import tempfile
import subprocess
import textwrap
import glob
import pandas as pd
import numpy as np
from unittest.mock import patch
from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.core.feature_extraction import extract_features
from photometry_pipeline.viz.display_prep import prepare_centered_common_gain
from photometry_pipeline.viz.semantic_colors import (
    DFF_COLOR,
    FITTED_REFERENCE_COLOR,
    NEUTRAL_BASELINE_COLOR,
    RAW_REFERENCE_COLOR,
    RAW_SIGNAL_COLOR,
    color_to_rgb,
)

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tools.plot_phasic_dayplot_bundle as bundle


def _image_rgb_colors(image):
    return {tuple(pixel) for row in np.asarray(image)[..., :3] for pixel in row}


def _has_rgb_color(image, color, *, tolerance=12):
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)[..., :3]
    target = np.asarray(color_to_rgb(color), dtype=np.int16)
    return bool(np.any(np.sum(np.abs(pixels - target), axis=-1) <= tolerance))


def _has_red_marker(image):
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)[..., :3]
    return bool(
        np.any(
            (pixels[..., 0] >= 150)
            & (pixels[..., 1] <= 130)
            & (pixels[..., 2] <= 130)
        )
    )


def test_lightweight_dayplot_renderers_use_shared_semantic_colors():
    t = np.linspace(0.0, 9.0, 10)
    layout = bundle._sig_iso_tile_layout(sph=1, dpi=80)
    title_font = bundle._get_font(12)
    signal_reference = {
        "t": t,
        "sig": np.linspace(0.0, 1.0, t.size),
        "uv": np.linspace(2.0, 3.0, t.size),
        "chunk_id": 0,
    }
    colors = _image_rgb_colors(
        bundle._render_sig_iso_panel_tile_lightweight(
            signal_reference, layout, title_font
        )
    )
    assert color_to_rgb(RAW_SIGNAL_COLOR) in colors
    assert color_to_rgb(RAW_REFERENCE_COLOR) in colors

    dynamic = {
        "t": t,
        "sig_fit": np.linspace(0.0, 1.0, t.size),
        "fit_ref": np.linspace(2.0, 3.0, t.size),
        "chunk_id": 0,
        "correction_strategy_family": "dynamic_fit",
    }
    colors = _image_rgb_colors(
        bundle._render_dynamic_fit_panel_tile_lightweight(
            dynamic, layout, title_font
        )
    )
    assert color_to_rgb(RAW_SIGNAL_COLOR) in colors
    assert color_to_rgb(FITTED_REFERENCE_COLOR) in colors

    signal_only = {**dynamic, "correction_strategy_family": "signal_only_f0"}
    colors = _image_rgb_colors(
        bundle._render_dynamic_fit_panel_tile_lightweight(
            signal_only, layout, title_font
        )
    )
    assert color_to_rgb(NEUTRAL_BASELINE_COLOR) in colors
    assert color_to_rgb(FITTED_REFERENCE_COLOR) not in colors

    dff_panel = {
        "t": t,
        "dff": np.sin(t),
        "chunk_id": 0,
        "peak_indices": np.array([3]),
    }
    dff_image, _stats = bundle._render_dff_panel_tile_lightweight(
        dff_panel,
        bundle._dff_tile_layout(sph=1, dpi=80),
        title_font,
        -1.2,
        1.2,
        show_peak_markers=True,
    )
    colors = _image_rgb_colors(dff_image)
    assert color_to_rgb(DFF_COLOR) in colors
    assert (220, 0, 0) in colors

    stacked = bundle._render_stacked_day_canvas_lightweight(
        day=0,
        plot_roi="Region0",
        slot_traces=[(t, np.sin(t))],
        smooth_window_s=1.0,
        dpi=80,
    )
    assert color_to_rgb(DFF_COLOR) in _image_rgb_colors(stacked)


def _assert_color_in_broad_x_regions(image, color, *, plot_x0, plot_w, buckets=8):
    pixels = np.asarray(image.convert("RGB"))
    target = np.asarray(color_to_rgb(color), dtype=np.int16)
    region = np.asarray(pixels[:, plot_x0 : plot_x0 + plot_w, :3], dtype=np.int16)
    mask = np.sum(np.abs(region - target), axis=-1) <= 12
    for bucket in range(buckets):
        left = (bucket * plot_w) // buckets
        right = ((bucket + 1) * plot_w) // buckets
        assert np.any(mask[:, left:right]), (
            f"{color} is absent from broad x-region {bucket}"
        )


def test_correction_reference_keeps_complete_x_domain_in_lightweight_renderer():
    t = np.linspace(0.0, 599.0, 1200)
    panels = [
        {
            "t": t,
            "sig_fit": -0.5 + 0.2 * np.sin(t / 19.0),
            "fit_ref": 1.5 + 0.2 * np.cos(t / 23.0),
            "chunk_id": 0,
            "correction_strategy_family": "dynamic_fit",
            "correction_reference_field": "fit_ref",
        },
        {
            "t": t,
            "sig_fit": 2.5 + 0.2 * np.cos(t / 17.0),
            "fit_ref": -1.5 + 0.2 * np.sin(t / 29.0),
            "chunk_id": 1,
            "correction_strategy_family": "dynamic_fit",
            "correction_reference_field": "fit_ref",
        },
    ]
    layout = bundle._dynamic_fit_tile_layout(sph=1, dpi=120)
    title_font = bundle._get_font(12)
    plot_x0 = max(8, int(0.02 * layout["tile_w"]))
    plot_w = layout["tile_w"] - (2 * plot_x0)
    original_paint = bundle._paint_trace_minmax

    for panel in panels:
        calls = []

        def capture_paint(
            tile_arr,
            x_idx,
            y_idx,
            plot_x0_arg,
            plot_y0,
            plot_w_arg,
            plot_h,
            color,
            stroke=1,
            dash_period=None,
        ):
            calls.append(
                {
                    "x_idx": np.asarray(x_idx).copy(),
                    "y_idx": np.asarray(y_idx).copy(),
                    "color": tuple(color),
                    "plot_x0": plot_x0_arg,
                    "plot_w": plot_w_arg,
                    "dash_period": dash_period,
                }
            )
            return original_paint(
                tile_arr,
                x_idx,
                y_idx,
                plot_x0_arg,
                plot_y0,
                plot_w_arg,
                plot_h,
                color,
                stroke,
                dash_period,
            )

        with patch.object(bundle, "_paint_trace_minmax", side_effect=capture_paint):
            image = bundle._render_dynamic_fit_panel_tile_lightweight(
                panel, layout, title_font
            )

        assert [call["color"] for call in calls] == [
            color_to_rgb(RAW_SIGNAL_COLOR),
            color_to_rgb(FITTED_REFERENCE_COLOR),
        ]
        assert len(calls[0]["x_idx"]) == len(calls[0]["y_idx"]) == len(t)
        assert len(calls[1]["x_idx"]) == len(calls[1]["y_idx"]) == len(t)
        assert np.isfinite(calls[0]["x_idx"]).all()
        assert np.isfinite(calls[0]["y_idx"]).all()
        assert np.isfinite(calls[1]["x_idx"]).all()
        assert np.isfinite(calls[1]["y_idx"]).all()
        np.testing.assert_array_equal(calls[0]["x_idx"], calls[1]["x_idx"])
        assert calls[0]["x_idx"].min() == 0
        assert calls[0]["x_idx"].max() == plot_w - 1
        assert calls[0]["dash_period"] is None
        assert calls[1]["dash_period"] is None

        _assert_color_in_broad_x_regions(
            image,
            RAW_SIGNAL_COLOR,
            plot_x0=plot_x0,
            plot_w=plot_w,
        )
        _assert_color_in_broad_x_regions(
            image,
            FITTED_REFERENCE_COLOR,
            plot_x0=plot_x0,
            plot_w=plot_w,
        )


def test_correction_reference_full_renderer_draws_complete_shared_x_domain():
    import matplotlib.pyplot as plt

    t = np.linspace(0.0, 599.0, 1200)
    panels = [
        {
            "t": t,
            "sig_fit": -0.5 + 0.2 * np.sin(t / 19.0),
            "fit_ref": 1.5 + 0.2 * np.cos(t / 23.0),
            "correction_strategy_family": "dynamic_fit",
            "correction_reference_field": "fit_ref",
        },
        {
            "t": t,
            "sig_fit": 2.5 + 0.2 * np.cos(t / 17.0),
            "fit_ref": -1.5 + 0.2 * np.sin(t / 29.0),
            "correction_strategy_family": "dynamic_fit",
            "correction_reference_field": "fit_ref",
        },
    ]
    figure, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    try:
        for axis, panel in zip(axes, panels):
            bundle._plot_dynamic_fit_panel(
                axis,
                panel,
                bundle._yrange_from_dynamic_panel(panel, float(t[0]), float(t[-1])),
            )

        for axis in axes:
            lines = axis.get_lines()
            assert [line.get_color() for line in lines] == [
                RAW_SIGNAL_COLOR,
                FITTED_REFERENCE_COLOR,
            ]
            assert len(lines[0].get_xdata()) == len(lines[0].get_ydata()) == len(t)
            assert len(lines[1].get_xdata()) == len(lines[1].get_ydata()) == len(t)
            np.testing.assert_array_equal(lines[0].get_xdata(), lines[1].get_xdata())
            assert np.isfinite(lines[0].get_xdata()).all()
            assert np.isfinite(lines[0].get_ydata()).all()
            assert np.isfinite(lines[1].get_xdata()).all()
            assert np.isfinite(lines[1].get_ydata()).all()
            assert not np.allclose(lines[0].get_ydata(), lines[1].get_ydata())
    finally:
        plt.close(figure)


def _run_cli(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def test_peak_count_annotation_hidden_when_markers_hidden():
    txt, color = bundle._build_peak_count_annotation(
        show_peak_markers=False,
        count_value=5,
        n_clipped=0,
    )
    assert txt is None
    assert color is None


def test_peak_count_annotation_shown_for_valid_marker_on_count():
    txt, color = bundle._build_peak_count_annotation(
        show_peak_markers=True,
        count_value=7,
        n_clipped=2,
    )
    assert txt == "peaks=7\n(2 clipped)"
    assert color == "blue"


def test_peak_count_annotation_omitted_for_missing_or_invalid_count():
    for val in (np.nan, None, "NaN", "bad", -1):
        txt, color = bundle._build_peak_count_annotation(
            show_peak_markers=True,
            count_value=val,
            n_clipped=0,
        )
        assert txt is None
        assert color is None


def test_correction_reference_resolution_uses_signal_only_baseline_without_fit_ref(tmp_path):
    import h5py

    cache_path = tmp_path / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "w") as handle:
        meta = handle.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray(["Region1"], dtype="S"))
        meta.create_dataset("chunk_ids", data=np.asarray([0], dtype=int))
        group = handle.create_group("roi/Region1/chunk_0")
        group.attrs["correction_strategy_family"] = "signal_only_f0"
        group.attrs["correction_selected_strategy"] = "signal_only_f0"
        group.create_dataset("signal_only_f0_baseline", data=np.ones(4))
    with bundle.open_phasic_cache(str(cache_path)) as cache:
        resolved = bundle._resolve_cached_correction_reference(cache, "Region1", [0])
    assert resolved["strategy_family"] == "signal_only_f0"
    assert resolved["field"] == "signal_only_f0_baseline"
    assert resolved["label"] == "Signal-only F0 baseline"

class TestPhasicDayplotBundle(unittest.TestCase):
    
    @patch('tools.plot_phasic_dayplot_bundle.sys.argv', ['plot_phasic_dayplot_bundle.py', '--analysis-out', '/fake', '--roi', 'Region0', '--output-dir', '/fake_out', '--sessions-per-hour', '2'])
    def test_parse_args_defaults(self):
        args = bundle.parse_args()
        self.assertEqual(args.analysis_out, '/fake')
        self.assertEqual(args.roi, 'Region0')
        self.assertEqual(args.output_dir, '/fake_out')
        self.assertEqual(args.sessions_per_hour, 2)
        self.assertEqual(args.timeline_anchor_mode, "civil")
        self.assertIsNone(args.fixed_daily_anchor_clock)
        self.assertTrue(args.write_dff_grid)
        self.assertTrue(args.write_sig_iso_grid)
        self.assertTrue(args.write_stacked)
        self.assertTrue(args.show_peak_markers)
        self.assertFalse(args.export_display_series_csv)

    @patch('tools.plot_phasic_dayplot_bundle.sys.argv', ['plot_phasic_dayplot_bundle.py', '--analysis-out', '/f', '--roi', 'R0', '--output-dir', '/o', '--sessions-per-hour', '1', '--no-write-stacked'])
    def test_parse_args_overrides(self):
        args = bundle.parse_args()
        self.assertTrue(args.write_dff_grid)
        self.assertTrue(args.write_sig_iso_grid)
        self.assertFalse(args.write_stacked)

    def test_display_series_export_off_by_default(self):
        self.create_synthetic_phasic_cache(
            cid=0,
            include_dff=True,
            include_sig=True,
            include_fit_ref=True,
        )
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--no-write-dff-grid',
            '--no-write-stacked',
            '--sig-iso-render-mode', 'full',
        ]
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main()

        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_sig_iso_day_000.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_dynamic_fit_day_000.png')))
        sig_image = bundle.Image.open(
            os.path.join(self.output_dir, 'phasic_sig_iso_day_000.png')
        )
        dynamic_image = bundle.Image.open(
            os.path.join(self.output_dir, 'phasic_dynamic_fit_day_000.png')
        )
        assert _has_rgb_color(sig_image, RAW_SIGNAL_COLOR)
        assert _has_rgb_color(sig_image, RAW_REFERENCE_COLOR)
        assert _has_rgb_color(dynamic_image, RAW_SIGNAL_COLOR)
        assert _has_rgb_color(dynamic_image, FITTED_REFERENCE_COLOR)
        self.assertEqual(glob.glob(os.path.join(self.output_dir, '*_display_series.csv')), [])

    def test_display_series_export_enabled_writes_long_format_csv(self):
        self.create_synthetic_phasic_cache(
            cid=0,
            include_dff=True,
            include_sig=True,
            include_fit_ref=True,
        )
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--no-write-dff-grid',
            '--no-write-stacked',
            '--export-display-series-csv',
            '--source-run-profile', 'full',
        ]
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main()

        sig_csv = os.path.join(self.output_dir, 'phasic_sig_iso_day_000_display_series.csv')
        dyn_csv = os.path.join(self.output_dir, 'phasic_dynamic_fit_day_000_display_series.csv')
        self.assertTrue(os.path.exists(sig_csv))
        self.assertTrue(os.path.exists(dyn_csv))

        df = pd.read_csv(sig_csv)
        required = {
            'roi', 'plot_type', 'source_run_profile', 'source_artifact',
            'trace_kind', 'x', 'y', 'display_series_export',
            'display_downsampled', 'display_downsample_rule',
            'day_index', 'slot_index', 'slot_label',
            'chunk_id', 'session_id', 'is_placeholder',
        }
        self.assertTrue(required.issubset(set(df.columns)))
        self.assertTrue(df['display_series_export'].astype(bool).all())
        self.assertEqual(set(df['plot_type'].dropna().astype(str).unique()), {'phasic_day_sig_iso'})
        self.assertEqual(set(df['source_run_profile'].dropna().astype(str).unique()), {'full'})
        self.assertEqual(set(df['source_artifact'].dropna().astype(str).unique()), {'day_plots/phasic_sig_iso_day_000.png'})
        self.assertEqual(set(df['display_downsampled'].astype(bool).unique()), {True})
        self.assertTrue(df['display_downsample_rule'].astype(str).str.contains('x-pixel', case=False).all())

        placeholder_flags = set(df['is_placeholder'].astype(str).str.lower().unique())
        self.assertIn('true', placeholder_flags)
        self.assertIn('false', placeholder_flags)
        non_placeholder = df[df['is_placeholder'].astype(str).str.lower() == 'false']
        self.assertTrue((non_placeholder['chunk_id'].astype(str).str.len() > 0).all())

    def test_check_monotonicity(self):
        self.assertTrue(bundle.check_monotonicity([0, 1, 2, 3]))
        self.assertFalse(bundle.check_monotonicity([0, 2, 1, 3]))

    def test_check_continuity(self):
        self.assertTrue(bundle.check_continuity([0, 1, 2, 3], 1.0))
        self.assertFalse(bundle.check_continuity([0, 1, 4, 5], 1.0))

    def test_duration_contract_accepts_vendor_like_shorter_admitted_chunks(self):
        # Matches real NPM pattern: admitted durations below nominal 600s.
        durations = [599.20, 594.54, 589.96, 585.17, 580.19]
        median_s, tol_s = bundle._resolve_duration_contract(durations, nominal_duration_s=600.0)
        self.assertAlmostEqual(median_s, float(np.median(durations)), places=6)
        self.assertGreaterEqual(tol_s, 2.0)

    def test_duration_contract_rejects_gross_profile_mismatch(self):
        with self.assertRaises(RuntimeError) as cm:
            bundle._resolve_duration_contract([120.0, 121.0, 122.0], nominal_duration_s=600.0)
        self.assertIn("Duration profile mismatch", str(cm.exception))

    def test_build_day_slot_maps_raises_on_collision(self):
        cached_by_day = {
            0: [
                {"hour": 9, "col": 1, "chunk_id": 10},
                {"hour": 9, "col": 1, "chunk_id": 11},
            ]
        }
        with self.assertRaises(RuntimeError) as cm:
            bundle.build_day_slot_maps(cached_by_day, sph=2)
        self.assertIn("Raw/dFF slot collision detected", str(cm.exception))

    def test_build_stacked_slot_traces_preserves_fixed_template_with_blanks(self):
        sph = 2
        slot_map = {
            (0, 1): {"chunk_id": 10},
            (1, 0): {"chunk_id": 11},
        }
        t = np.array([0.0, 1.0, 2.0])
        smoothed_data = {
            10: (t, np.array([1.0, 1.5, 2.0])),
            11: (t, np.array([0.5, 0.4, 0.3])),
        }

        slot_traces = bundle._build_stacked_slot_traces(slot_map, smoothed_data, sph=sph)
        self.assertEqual(len(slot_traces), 24 * sph)
        # Slot indexing follows the same fixed template as raw/dFF: idx = hour * sph + col.
        self.assertIsNone(slot_traces[0])      # H00 left
        self.assertIsNotNone(slot_traces[1])   # H00 right
        self.assertIsNotNone(slot_traces[2])   # H01 left
        self.assertIsNone(slot_traces[3])      # H01 right
        self.assertIsNone(slot_traces[-1])     # trailing template slot preserved

    def test_stacked_lightweight_canvas_size_depends_on_total_slots_not_occupied(self):
        t = np.linspace(0.0, 10.0, 51)
        sparse = [None] * 48
        sparse[9] = (t, np.sin(t))
        sparse[10] = (t, np.cos(t))
        dense = [(t, np.sin(t + i * 0.1)) for i in range(48)]

        img_sparse = bundle._render_stacked_day_canvas_lightweight(
            day=0,
            plot_roi="Region0",
            slot_traces=sparse,
            smooth_window_s=0.6,
            dpi=100,
            timeline_anchor_label="fixed-daily-anchor@07:00:00",
        )
        img_dense = bundle._render_stacked_day_canvas_lightweight(
            day=0,
            plot_roi="Region0",
            slot_traces=dense,
            smooth_window_s=0.6,
            dpi=100,
            timeline_anchor_label="fixed-daily-anchor@07:00:00",
        )

        self.assertEqual(img_sparse.size, img_dense.size)

    def test_stacked_slot_layout_uses_full_slot_span_for_sparse_days(self):
        t = np.linspace(0.0, 10.0, 101)
        sparse_top = [None] * 48
        for i in range(9):
            sparse_top[i] = (t, np.sin(t))

        step, data_y_min, data_y_max, y0, y1 = bundle._compute_stacked_slot_layout(sparse_top)

        # This is the core regression guard:
        # sparse occupancy in early slots must still reserve full slot span.
        self.assertLess(y0, data_y_min + step)
        self.assertGreater(y1, data_y_max + (46.0 * step))

    def test_prepare_sig_iso_centered_panel_matches_shared_helper_semantics(self):
        sig_raw = np.array([57.0, 60.0, 63.0, 60.0], dtype=float)
        uv_raw = np.array([84.5, 85.0, 85.5, 85.0], dtype=float)

        sig_expected, uv_expected = prepare_centered_common_gain(sig_raw, uv_raw)
        sig_centered, uv_centered = bundle._prepare_sig_iso_centered_panel(sig_raw, uv_raw)

        self.assertTrue(np.allclose(sig_centered, sig_expected))
        self.assertTrue(np.allclose(uv_centered, uv_expected))
        self.assertAlmostEqual(float(np.nanmedian(sig_centered)), 0.0)
        self.assertAlmostEqual(float(np.nanmedian(uv_centered)), 0.0)
        self.assertGreater(float(np.nanmax(np.abs(sig_centered))), float(np.nanmax(np.abs(uv_centered))))

    def test_prepare_sig_iso_centered_panel_falls_back_for_no_finite_trace(self):
        sig_raw = np.array([np.nan, np.nan, np.nan], dtype=float)
        uv_raw = np.array([1.0, 2.0, 3.0], dtype=float)

        sig_centered, uv_centered = bundle._prepare_sig_iso_centered_panel(sig_raw, uv_raw)

        self.assertTrue(np.all(np.isnan(sig_centered)))
        self.assertTrue(np.allclose(uv_centered, np.array([-1.0, 0.0, 1.0], dtype=float)))

    def test_prepare_sig_iso_centered_panel_falls_back_for_no_finite_trace_inverse(self):
        sig_raw = np.array([57.0, 60.0, 63.0], dtype=float)
        uv_raw = np.array([np.nan, np.nan, np.nan], dtype=float)

        sig_centered, uv_centered = bundle._prepare_sig_iso_centered_panel(sig_raw, uv_raw)

        self.assertTrue(np.allclose(sig_centered, np.array([-3.0, 0.0, 3.0], dtype=float)))
        self.assertTrue(np.all(np.isnan(uv_centered)))

    def test_prepare_sig_iso_centered_panel_both_unusable_is_stable(self):
        sig_raw = np.array([np.nan, np.nan], dtype=float)
        uv_raw = np.array([np.nan, np.nan], dtype=float)

        sig_centered, uv_centered = bundle._prepare_sig_iso_centered_panel(sig_raw, uv_raw)

        self.assertTrue(np.all(np.isnan(sig_centered)))
        self.assertTrue(np.all(np.isnan(uv_centered)))

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.analysis_out = os.path.join(self.test_dir.name, '_analysis')
        self.traces_dir = os.path.join(self.analysis_out, 'traces')
        self.features_dir = os.path.join(self.analysis_out, 'features')
        self.output_dir = os.path.join(self.test_dir.name, 'day_plots')
        
        os.makedirs(self.traces_dir)
        os.makedirs(self.output_dir)
        
        config_path = os.path.join(self.analysis_out, 'config_used.yaml')
        with open(config_path, 'w') as f:
            f.write("target_fs_hz: 10.0\n")
            f.write("peak_threshold_method: 'absolute'\n")
            f.write("peak_threshold_abs: 0.5\n")
            f.write("peak_min_distance_sec: 1.0\n")

    def create_synthetic_chunk(self, cid=0, include_dff=False, include_sig=True):
        t = np.arange(0, 600, 0.1)
        data = {'time_sec': t}
        if include_sig:
            data['Region0_sig_raw'] = np.sin(t)
            data['Region0_uv_raw'] = np.cos(t)
        if include_dff:
            data['Region0_dff'] = np.sin(t) - np.cos(t)
            
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.traces_dir, f"chunk_{cid}.csv")
        df.to_csv(csv_path, index=False)

    def create_features_csv(self, peak_count=0):
        os.makedirs(self.features_dir, exist_ok=True)
        df = pd.DataFrame([{
            'chunk_id': 0, 'roi': 'Region0', 'peak_count': peak_count, 'auc': 0.0
        }])
        df.to_csv(os.path.join(self.features_dir, 'features.csv'), index=False)

    def create_synthetic_phasic_cache(
        self,
        cid=0,
        include_dff=False,
        include_sig=True,
        include_fit_ref=True,
        include_delta_f=False,
        dff_data=None,
        fit_ref_data=None,
        delta_f_data=None,
        sig_data=None,
        uv_data=None,
        time_data=None,
    ):
        import h5py
        cache_path = os.path.join(self.analysis_out, 'phasic_trace_cache.h5')
        
        # Open in append mode so we can add multiple chunks if tests ever needed it
        with h5py.File(cache_path, 'a') as f:
            if 'meta' not in f:
                meta = f.create_group('meta')
                meta.attrs['mode'] = 'phasic'
                meta.attrs['schema_version'] = '1.0'
                dt_str = h5py.string_dtype(encoding='utf-8')
                meta.create_dataset('rois', data=np.array(['Region0'], dtype=object), dtype=dt_str)
                meta.create_dataset('chunk_ids', data=np.array([cid], dtype=int), maxshape=(None,))
            else:
                meta = f['meta']
                cids = list(meta['chunk_ids'][()])
                if cid not in cids:
                    cids.append(cid)
                    del meta['chunk_ids']
                    meta.create_dataset('chunk_ids', data=np.array(cids, dtype=int))
                
            if 'roi' not in f:
                f.create_group('roi')
            roi_grp = f.require_group('roi/Region0')
            c_grp = roi_grp.require_group(f'chunk_{cid}')
            
            t = np.arange(0, 600, 0.1) if time_data is None else np.asarray(time_data)
            
            if 'time_sec' in c_grp: del c_grp['time_sec']
            c_grp.create_dataset('time_sec', data=t)
            
            if include_sig:
                if 'sig_raw' in c_grp: del c_grp['sig_raw']
                if 'uv_raw' in c_grp: del c_grp['uv_raw']
                c_grp.create_dataset('sig_raw', data=np.sin(t) if sig_data is None else np.asarray(sig_data))
                c_grp.create_dataset('uv_raw', data=np.cos(t) if uv_data is None else np.asarray(uv_data))
                if include_fit_ref:
                    if 'fit_ref' in c_grp: del c_grp['fit_ref']
                    c_grp.create_dataset(
                        'fit_ref',
                        data=(0.8 * np.sin(t) + 0.1) if fit_ref_data is None else np.asarray(fit_ref_data),
                    )
                
            if include_dff:
                if 'dff' in c_grp: del c_grp['dff']
                # The prompt tests with some high peaks. Let's make sure test_full_dff_mode can overwrite this.
                c_grp.create_dataset('dff', data=(np.sin(t) - np.cos(t)) if dff_data is None else np.asarray(dff_data))

            if include_delta_f:
                if 'delta_f' in c_grp: del c_grp['delta_f']
                c_grp.create_dataset('delta_f', data=np.sin(t) if delta_f_data is None else np.asarray(delta_f_data))

    def test_sig_iso_only_mode_no_dff(self):
        # A. sig/iso-only mode: no features.csv, no dff column -> success
        self.create_synthetic_chunk(cid=0, include_dff=False)
        self.create_synthetic_phasic_cache(cid=0, include_dff=False)
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--no-write-dff-grid',
            '--no-write-stacked'
        ]
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main() # Should not raise
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_sig_iso_day_000.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_dynamic_fit_day_000.png')))

    def test_correction_reference_main_full_and_qc_renderers_keep_both_traces(self):
        t = np.linspace(0.0, 599.0, 1200)
        panels = [
            (
                -0.5 + 0.2 * np.sin(t / 19.0),
                1.5 + 0.2 * np.cos(t / 23.0),
            ),
            (
                2.5 + 0.2 * np.cos(t / 17.0),
                -1.5 + 0.2 * np.sin(t / 29.0),
            ),
        ]
        for cid, (sig_fit, fit_ref) in enumerate(panels):
            self.create_synthetic_chunk(cid=cid, include_dff=False)
            self.create_synthetic_phasic_cache(
                cid=cid,
                include_dff=False,
                sig_data=sig_fit,
                fit_ref_data=fit_ref,
                time_data=t,
            )

        for render_mode in ("qc", "full"):
            output_dir = os.path.join(self.test_dir.name, f"day_plots_{render_mode}")
            os.makedirs(output_dir, exist_ok=True)
            test_args = [
                "plot_phasic_dayplot_bundle.py",
                "--analysis-out",
                self.analysis_out,
                "--roi",
                "Region0",
                "--output-dir",
                output_dir,
                "--sessions-per-hour",
                "2",
                "--no-write-dff-grid",
                "--no-write-stacked",
                "--sig-iso-render-mode",
                render_mode,
            ]
            captured_axes = []
            real_save = bundle.save_png_fast

            def capture_save(fig, out_path, dpi):
                if os.path.basename(out_path).startswith("phasic_dynamic_fit_day_"):
                    captured_axes.extend(
                        axis for axis in fig.axes if len(axis.get_lines()) == 2
                    )
                return real_save(fig, out_path, dpi)

            with patch("tools.plot_phasic_dayplot_bundle.sys.argv", test_args):
                with patch.object(bundle, "save_png_fast", side_effect=capture_save):
                    bundle.main()

            image = bundle.Image.open(
                os.path.join(output_dir, "phasic_dynamic_fit_day_000.png")
            )
            if render_mode == "qc":
                layout = bundle._dynamic_fit_tile_layout(sph=2, dpi=120)
                plot_x0 = max(8, int(0.02 * layout["tile_w"]))
                plot_w = layout["tile_w"] - (2 * plot_x0)
                for column in range(2):
                    left = layout["left_label_w"] + column * (
                        layout["tile_w"] + layout["col_gap"]
                    )
                    top = layout["top_title_h"]
                    tile = image.crop(
                        (left, top, left + layout["tile_w"], top + layout["tile_h"])
                    )
                    _assert_color_in_broad_x_regions(
                        tile,
                        RAW_SIGNAL_COLOR,
                        plot_x0=plot_x0,
                        plot_w=plot_w,
                    )
                    _assert_color_in_broad_x_regions(
                        tile,
                        FITTED_REFERENCE_COLOR,
                        plot_x0=plot_x0,
                        plot_w=plot_w,
                    )
            else:
                assert len(captured_axes) == 2
                for axis in captured_axes:
                    lines = axis.get_lines()
                    assert [line.get_color() for line in lines] == [
                        RAW_SIGNAL_COLOR,
                        FITTED_REFERENCE_COLOR,
                    ]
                    np.testing.assert_array_equal(
                        lines[0].get_xdata(), lines[1].get_xdata()
                    )
                    assert len(lines[0].get_xdata()) == len(t)
                    assert len(lines[1].get_xdata()) == len(t)

    def test_stacked_only_mode_no_features(self):
        # B. stacked-only mode: no features.csv, dff necessary but feature shouldn't be read 
        self.create_synthetic_chunk(cid=0, include_dff=True)
        self.create_synthetic_phasic_cache(cid=0, include_dff=True)
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--no-write-dff-grid',
            '--no-write-sig-iso-grid',
            '--write-stacked',
            '--stacked-render-mode', 'full',
        ]
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main() # Should not raise
        stacked_path = os.path.join(self.output_dir, 'phasic_stacked_day_000.png')
        self.assertTrue(os.path.exists(stacked_path))
        assert _has_rgb_color(bundle.Image.open(stacked_path), DFF_COLOR)

    def test_stacked_emits_same_day_family_when_one_day_has_zero_occupied_traces(self):
        t = np.arange(0, 600, 0.1)
        zero_dff = np.zeros_like(t)
        nan_dff = np.full_like(t, np.nan)

        # Use fallback chunk-index timeline with sessions-per-hour=1:
        # chunk ids 0..23 => day 0, chunk id 24 => day 1.
        for cid in range(24):
            self.create_synthetic_phasic_cache(cid=cid, include_dff=True, dff_data=zero_dff, time_data=t)
        self.create_synthetic_phasic_cache(cid=24, include_dff=True, dff_data=nan_dff, time_data=t)

        os.makedirs(self.features_dir, exist_ok=True)
        pd.DataFrame(
            [{'chunk_id': cid, 'roi': 'Region0', 'peak_count': 0, 'auc': 0.0} for cid in range(25)]
        ).to_csv(os.path.join(self.features_dir, 'features.csv'), index=False)

        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
        ]
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main()

        def _days(prefix):
            days = []
            for name in os.listdir(self.output_dir):
                if name.startswith(prefix) and name.endswith('.png'):
                    days.append(int(name.rsplit('_', 1)[1].split('.')[0]))
            return sorted(days)

        dff_days = _days('phasic_dFF_day_')
        sig_days = _days('phasic_sig_iso_day_')
        dyn_days = _days('phasic_dynamic_fit_day_')
        stacked_days = _days('phasic_stacked_day_')

        self.assertEqual(dff_days, sig_days)
        self.assertEqual(sig_days, dyn_days)
        self.assertEqual(sig_days, stacked_days)
        self.assertIn(1, stacked_days)

    def test_dff_grid_mode_requires_features(self):
        # C. dFF-grid mode requires features.csv
        self.create_synthetic_chunk(cid=0, include_dff=True)
        self.create_synthetic_phasic_cache(cid=0, include_dff=True)
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--write-dff-grid',
            '--no-write-sig-iso-grid',
            '--no-write-stacked'
        ]
        
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                bundle.main()
            self.assertEqual(cm.exception.code, 1)

    def test_dff_grid_without_peak_markers_does_not_require_features(self):
        self.create_synthetic_chunk(cid=0, include_dff=True)
        self.create_synthetic_phasic_cache(cid=0, include_dff=True)
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--write-dff-grid',
            '--no-write-sig-iso-grid',
            '--no-write-stacked',
            '--dff-render-mode', 'full',
            '--hide-peak-markers',
            '--export-display-series-csv',
            '--source-run-profile', 'full',
        ]
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main()

        dff_png = os.path.join(self.output_dir, 'phasic_dFF_day_000.png')
        dff_csv = os.path.join(self.output_dir, 'phasic_dFF_day_000_display_series.csv')
        self.assertTrue(os.path.exists(dff_png))
        self.assertTrue(os.path.exists(dff_csv))
        assert _has_rgb_color(bundle.Image.open(dff_png), DFF_COLOR)

        df = pd.read_csv(dff_csv)
        self.assertEqual(set(df['plot_type'].dropna().astype(str).unique()), {'phasic_day_dff'})
        self.assertNotIn('peak_marker', set(df['trace_kind'].dropna().astype(str).unique()))

    def test_marker_on_full_render_skips_missing_peak_indices_without_crashing(self):
        t = np.arange(0, 600, 0.1)
        dff_data = np.zeros_like(t)
        dff_data[100] = 100.0
        dff_data[200] = 100.0
        self.create_synthetic_phasic_cache(cid=0, include_dff=True, dff_data=dff_data, time_data=t)
        self.create_features_csv(peak_count=2)

        orig_build_day_slot_maps = bundle.build_day_slot_maps

        def _strip_peak_indices(*args, **kwargs):
            out = orig_build_day_slot_maps(*args, **kwargs)
            for slot_map in out.values():
                for panel in slot_map.values():
                    panel.pop("peak_indices", None)
            return out

        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--write-dff-grid',
            '--no-write-sig-iso-grid',
            '--no-write-stacked',
            '--dff-render-mode', 'full',
            '--show-peak-markers',
        ]
        with patch('tools.plot_phasic_dayplot_bundle.build_day_slot_maps', side_effect=_strip_peak_indices):
            with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
                bundle.main()

        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_dFF_day_000.png')))

    def test_marker_on_display_series_exports_peak_markers_when_indices_exist(self):
        t = np.arange(0, 600, 0.1)
        dff_data = np.zeros_like(t)
        dff_data[100] = 100.0
        dff_data[200] = 100.0
        self.create_synthetic_phasic_cache(cid=0, include_dff=True, dff_data=dff_data, time_data=t)
        self.create_features_csv(peak_count=2)

        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--write-dff-grid',
            '--no-write-sig-iso-grid',
            '--no-write-stacked',
            '--show-peak-markers',
            '--export-display-series-csv',
            '--source-run-profile', 'full',
        ]
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main()

        dff_csv = os.path.join(self.output_dir, 'phasic_dFF_day_000_display_series.csv')
        self.assertTrue(os.path.exists(dff_csv))
        df = pd.read_csv(dff_csv)
        marker_rows = df[df['trace_kind'].astype(str) == 'peak_marker']
        self.assertGreaterEqual(len(marker_rows), 2)

    def test_extract_peak_indices_supports_aliases_and_filters_invalid_values(self):
        record = {
            "peak_index": [1, 2.0, 2.4, np.nan, "bad", -1, 7],
        }
        indices, skipped = bundle._extract_peak_indices(record, n_samples=5)
        np.testing.assert_array_equal(indices, np.array([1, 2], dtype=int))
        self.assertEqual(skipped, 5)

    def test_full_dff_mode(self):
        # D. full dFF mode passes when feature and dff traces match
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--write-dff-grid',
            '--write-stacked',
        ]
        
        # Minimal trace CSV required for discovery logic
        t = np.arange(0, 600, 0.1)
        df_dff = pd.DataFrame({
            'time_sec': t,
            'Region0_sig_raw': np.zeros_like(t),
            'Region0_uv_raw': np.zeros_like(t),
            'Region0_dff': np.zeros_like(t)
        })
        df_dff.to_csv(os.path.join(self.traces_dir, 'chunk_0.csv'), index=False)
        
        # Add 2 peaks into the cache instead (but zeros otherwise)
        self.create_synthetic_phasic_cache(cid=0, include_dff=True)
        import h5py
        cache_path = os.path.join(self.analysis_out, 'phasic_trace_cache.h5')
        with h5py.File(cache_path, 'a') as f:
            dff_data = np.zeros_like(t)
            dff_data[100] = 100.0
            dff_data[200] = 100.0
            del f['roi/Region0/chunk_0/dff']
            f['roi/Region0/chunk_0'].create_dataset('dff', data=dff_data)
        
        self.create_features_csv(peak_count=2)
        
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main()
            
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_dFF_day_000.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_stacked_day_000.png')))


    def test_missing_cache_fails_with_csvs_present(self):
        # Even if chunk_0.csv exists, missing HDF5 cache should hard fail
        self.create_synthetic_chunk(cid=0, include_dff=True)
        # We explicitly DO NOT create the phasic cache
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--no-write-dff-grid',
            '--no-write-stacked'
        ]
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                bundle.main()
            self.assertEqual(cm.exception.code, 1)

    @patch('tools.plot_phasic_dayplot_bundle.discover_chunks')
    def test_cache_pull_fields_all_modes(self, mock_discover_chunks):
        # We replace the old CSV usecols test with one that mocks discovery and ensures the cache reader requests exact fields
        import numpy as np
        
        # We physically create the chunk to avoid mocking read_csv at all
        self.create_synthetic_chunk(cid=0, include_dff=True)
        self.create_synthetic_phasic_cache(cid=0, include_dff=True)
        self.create_features_csv(peak_count=0) # Must exist for dFF Grid mode
        
        mock_discover_chunks.return_value = [(0, os.path.join(self.traces_dir, 'chunk_0.csv'))]
        
        cases = [
            ("sig_iso_only", ['--write-sig-iso-grid', '--no-write-dff-grid', '--no-write-stacked'], 
             ['time_sec', 'sig_raw', 'uv_raw', 'fit_ref']),
             
            ("stacked_only", ['--no-write-sig-iso-grid', '--no-write-dff-grid', '--write-stacked'], 
             ['time_sec', 'dff']),
             
            ("dff_grid_only", ['--no-write-sig-iso-grid', '--write-dff-grid', '--no-write-stacked'], 
             ['time_sec', 'dff']),
             
            ("full_mode", ['--write-sig-iso-grid', '--write-dff-grid', '--write-stacked'], 
             ['time_sec', 'sig_raw', 'uv_raw', 'dff', 'fit_ref'])
        ]
        
        with patch('tools.plot_phasic_dayplot_bundle.resolve_roi', return_value='Region0'):
            for name, flags, expected_fields in cases:
                with patch('tools.plot_phasic_dayplot_bundle.load_cache_chunk_fields') as mock_load:
                    # Provide dummy array tuples back matching requested length
                    mock_load.return_value = tuple([np.array([1, 2])] * len(expected_fields))
                    
                    test_args = [
                        'plot_phasic_dayplot_bundle.py', '--analysis-out', self.analysis_out,
                        '--roi', 'Region0', '--output-dir', self.output_dir, '--sessions-per-hour', '1',
                        '--session-duration-s', '0.1'
                    ] + flags
                    
                    with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
                        bundle.main()
                        
                    self.assertTrue(mock_load.called, f"{name}: load_cache_chunk_fields not called")
                    
                    # Check exact fields requested are strictly equal
                    fields_requested = mock_load.call_args[0][3]
                    self.assertEqual(fields_requested, expected_fields, f"{name}: requested fields mismatch")

    def test_dff_grid_verification_uses_configured_event_signal_and_ignores_display_smoothing(self):
        t = np.arange(0, 600, 0.1)
        fs_hz = 10.0
        rng = np.random.default_rng(7)

        # Detection trace (delta_f): few broad events.
        delta_f = 0.02 * np.sin(0.07 * t)
        for center in (90.0, 260.0, 430.0):
            delta_f += 2.2 * np.exp(-0.5 * ((t - center) / 0.9) ** 2)

        # Display trace (dff): many oscillatory peaks that should NOT drive authoritative count.
        dff = 0.9 * np.sin(2 * np.pi * 1.4 * t) + 0.4 * np.sin(2 * np.pi * 3.2 * t)
        dff += 0.05 * rng.standard_normal(len(t))

        with open(os.path.join(self.analysis_out, 'config_used.yaml'), 'w', encoding='utf-8') as f:
            f.write("target_fs_hz: 10.0\n")
            f.write("lowpass_hz: 1.0\n")
            f.write("filter_order: 3\n")
            f.write("event_signal: 'delta_f'\n")
            f.write("peak_threshold_method: 'mean_std'\n")
            f.write("peak_threshold_k: 1.5\n")
            f.write("peak_min_distance_sec: 0.5\n")
            f.write("peak_pre_filter: 'lowpass'\n")

        cfg = Config.from_yaml(os.path.join(self.analysis_out, 'config_used.yaml'))
        chunk = Chunk(
            chunk_id=0,
            source_file='chunk_0.csv',
            format='rwd',
            time_sec=t,
            uv_raw=np.zeros((len(t), 1), dtype=float),
            sig_raw=np.zeros((len(t), 1), dtype=float),
            fs_hz=fs_hz,
            channel_names=['Region0'],
            dff=dff.reshape(-1, 1),
            delta_f=delta_f.reshape(-1, 1),
        )
        feat_df = extract_features(chunk, cfg)
        expected_count = int(feat_df.iloc[0]['peak_count'])
        self.assertGreater(expected_count, 0)

        self.create_features_csv(peak_count=expected_count)
        self.create_synthetic_phasic_cache(
            cid=0,
            include_dff=True,
            include_sig=True,
            include_delta_f=True,
            dff_data=dff,
            delta_f_data=delta_f,
            sig_data=np.zeros_like(t),
            uv_data=np.zeros_like(t),
            time_data=t,
        )

        for smooth in ('0.2', '20.0'):
            test_args = [
                'plot_phasic_dayplot_bundle.py',
                '--analysis-out', self.analysis_out,
                '--roi', 'Region0',
                '--output-dir', self.output_dir,
                '--sessions-per-hour', '1',
                '--write-dff-grid',
                '--no-write-sig-iso-grid',
                '--write-stacked',
                '--smooth-window-s', smooth,
            ]
            with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
                bundle.main()

        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_dFF_day_000.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_stacked_day_000.png')))

def test_dayplot_bundle_uses_analysis_fs_for_strict_peak_verification(tmp_path):
    repo_root = PROJECT_ROOT
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """\
            chunk_duration_sec: 600
            target_fs_hz: 20
            baseline_method: uv_raw_percentile_session
            baseline_percentile: 10
            rwd_time_col: TimeStamp
            uv_suffix: "-410"
            sig_suffix: "-470"
            peak_threshold_method: mean_std
            peak_threshold_k: 2.0
            peak_threshold_percentile: 95.0
            peak_threshold_abs: 0.0
            peak_min_distance_sec: 0.5
            peak_pre_filter: none
            event_signal: dff
            window_sec: 20.0
            step_sec: 5.0
            """
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "input_rwd"
    phasic_out = tmp_path / "phasic_out"
    output_dir = tmp_path / "day_plots"

    synth_cmd = [
        sys.executable,
        "tools/synth_photometry_dataset.py",
        "--out",
        str(input_dir),
        "--format",
        "rwd",
        "--config",
        str(cfg_path),
        "--preset",
        "biological_shared_nuisance",
        "--total-days",
        "0.125",
        "--recording-duration-min",
        "10",
        "--recordings-per-hour",
        "2",
        "--n-rois",
        "2",
        "--start-iso",
        "2025-01-01T00:00:00",
        "--seed",
        "42",
    ]
    synth_res = _run_cli(synth_cmd, cwd=repo_root)
    assert synth_res.returncode == 0, (
        "Synthetic input generation failed.\n"
        f"STDOUT:\n{synth_res.stdout}\nSTDERR:\n{synth_res.stderr}"
    )

    analyze_cmd = [
        sys.executable,
        "analyze_photometry.py",
        "--input",
        str(input_dir),
        "--config",
        str(cfg_path),
        "--out",
        str(phasic_out),
        "--mode",
        "phasic",
        "--format",
        "rwd",
        "--recursive",
        "--overwrite",
        "--sessions-per-hour",
        "2",
    ]
    analyze_res = _run_cli(analyze_cmd, cwd=repo_root)
    assert analyze_res.returncode == 0, (
        "Phasic analysis failed.\n"
        f"STDOUT:\n{analyze_res.stdout}\nSTDERR:\n{analyze_res.stderr}"
    )
    assert (phasic_out / "phasic_trace_cache.h5").exists()
    assert (phasic_out / "features" / "features.csv").exists()

    # Regression contract: before this fix, this exact path could fail with:
    # "CRITICAL: Plotting Logic Mismatch ... (X vs Y)" due to inferred-fs drift.
    dayplot_cmd = [
        sys.executable,
        "tools/plot_phasic_dayplot_bundle.py",
        "--analysis-out",
        str(phasic_out),
        "--roi",
        "Region0",
        "--output-dir",
        str(output_dir),
        "--sessions-per-hour",
        "2",
    ]
    dayplot_res = _run_cli(dayplot_cmd, cwd=repo_root)
    combined = f"{dayplot_res.stdout}\n{dayplot_res.stderr}"
    assert dayplot_res.returncode == 0, (
        "Dayplot bundle failed unexpectedly.\n"
        f"STDOUT:\n{dayplot_res.stdout}\nSTDERR:\n{dayplot_res.stderr}"
    )
    assert "Plotting Logic Mismatch" not in combined
    assert (output_dir / "phasic_dFF_day_000.png").exists()


if __name__ == '__main__':
    unittest.main()
