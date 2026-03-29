import unittest

import numpy as np

from photometry_pipeline.core.tonic_output import (
    TONIC_OUTPUT_MODE_FLATTEN_BLEACH,
    TONIC_OUTPUT_MODE_PRESERVE_RAW,
    apply_tonic_output_mode_to_session,
    normalize_tonic_output_mode,
)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return np.nan
    coef = np.polyfit(x[m], y[m], 1)
    return float(coef[0])


class TestTonicOutputMode(unittest.TestCase):
    def test_preserve_raw_mode_is_identity(self):
        t = np.linspace(0, 600, 200)
        sig = 1000.0 - 0.2 * t + 3.0 * np.sin(t / 30.0)
        uv = 800.0 - 0.15 * t + 2.0 * np.cos(t / 40.0)
        df = sig - uv

        sig_out, uv_out, df_out, meta = apply_tonic_output_mode_to_session(
            time_sec=t,
            sig_raw=sig,
            uv_raw=uv,
            deltaf_raw=df,
            mode_raw=TONIC_OUTPUT_MODE_PRESERVE_RAW,
        )
        self.assertTrue(np.allclose(sig_out, sig))
        self.assertTrue(np.allclose(uv_out, uv))
        self.assertTrue(np.allclose(df_out, df))
        self.assertEqual(int(meta.get("fallback_count", -1)), 0)

    def test_flatten_mode_reduces_within_session_bleach_shape(self):
        t = np.linspace(0, 600, 300)
        uv = 800.0 + 110.0 * np.exp(-t / 180.0) + 1.0 * np.cos(t / 70.0)
        iso_fit = 120.0 + 1.25 * uv
        event = 2.0 * np.sin(t / 50.0)
        sig = iso_fit + event
        df = sig - iso_fit

        sig_out, uv_out, df_out, _ = apply_tonic_output_mode_to_session(
            time_sec=t,
            sig_raw=sig,
            uv_raw=uv,
            deltaf_raw=df,
            mode_raw=TONIC_OUTPUT_MODE_FLATTEN_BLEACH,
        )

        self.assertLess(abs(_slope(t, sig_out)), abs(_slope(t, sig)))
        self.assertLess(abs(_slope(t, uv_out)), abs(_slope(t, uv)))
        self.assertAlmostEqual(float(np.nanmean(sig_out)), float(np.nanmean(sig)), delta=1e-6)
        self.assertAlmostEqual(float(np.nanmean(uv_out)), float(np.nanmean(uv)), delta=1e-6)

    def test_flatten_mode_preserves_between_session_offset(self):
        t = np.linspace(0, 600, 250)
        uv_a = 700.0 + 90.0 * np.exp(-t / 220.0)
        uv_b = 900.0 + 90.0 * np.exp(-t / 220.0)
        iso_fit_a = 80.0 + 1.1 * uv_a
        iso_fit_b = 80.0 + 1.1 * uv_b
        sig_a = iso_fit_a + 0.8 * np.sin(t / 40.0)
        sig_b = iso_fit_b + 0.8 * np.sin(t / 40.0)
        df_a = sig_a - iso_fit_a
        df_b = sig_b - iso_fit_b

        sig_a_out, uv_a_out, _, _ = apply_tonic_output_mode_to_session(
            time_sec=t,
            sig_raw=sig_a,
            uv_raw=uv_a,
            deltaf_raw=df_a,
            mode_raw=TONIC_OUTPUT_MODE_FLATTEN_BLEACH,
        )
        sig_b_out, uv_b_out, _, _ = apply_tonic_output_mode_to_session(
            time_sec=t,
            sig_raw=sig_b,
            uv_raw=uv_b,
            deltaf_raw=df_b,
            mode_raw=TONIC_OUTPUT_MODE_FLATTEN_BLEACH,
        )

        self.assertAlmostEqual(
            float(np.nanmean(sig_b_out) - np.nanmean(sig_a_out)),
            float(np.nanmean(sig_b) - np.nanmean(sig_a)),
            delta=1e-6,
        )
        self.assertAlmostEqual(
            float(np.nanmean(uv_b_out) - np.nanmean(uv_a_out)),
            float(np.nanmean(uv_b) - np.nanmean(uv_a)),
            delta=1e-6,
        )

    def test_deltaf_is_recomputed_from_flattened_channels(self):
        t = np.linspace(0, 600, 400)
        uv = 500.0 + 130.0 * np.exp(-t / 170.0) + 0.4 * np.cos(t / 30.0)
        iso_fit = 60.0 + 1.4 * uv
        bio = 6.0 * np.sin(t / 25.0)
        sig = iso_fit + bio
        df = sig - iso_fit

        sig_out, uv_out, df_out, meta = apply_tonic_output_mode_to_session(
            time_sec=t,
            sig_raw=sig,
            uv_raw=uv,
            deltaf_raw=df,
            mode_raw=TONIC_OUTPUT_MODE_FLATTEN_BLEACH,
        )
        affine = meta["channels"]["deltaF"]["affine_meta"]
        self.assertTrue(bool(affine.get("success", False)))
        slope = float(affine["slope"])
        intercept = float(affine["intercept"])
        recon = sig_out - (slope * uv_out + intercept)
        m = np.isfinite(recon) & np.isfinite(df_out)
        self.assertGreater(np.sum(m), 100)
        self.assertTrue(np.allclose(df_out[m], recon[m], atol=1e-6))

    def test_guardrail_fallback_for_degenerate_time(self):
        t = np.ones(100)
        sig = np.linspace(0, 1, 100)
        uv = np.linspace(1, 2, 100)
        df = sig - uv

        sig_out, uv_out, df_out, meta = apply_tonic_output_mode_to_session(
            time_sec=t,
            sig_raw=sig,
            uv_raw=uv,
            deltaf_raw=df,
            mode_raw=TONIC_OUTPUT_MODE_FLATTEN_BLEACH,
        )
        self.assertTrue(np.allclose(sig_out, sig))
        self.assertTrue(np.allclose(uv_out, uv))
        self.assertTrue(np.allclose(df_out, df))
        self.assertEqual(int(meta.get("fallback_count", 0)), 2)

    def test_mode_normalization_aliases(self):
        self.assertEqual(
            normalize_tonic_output_mode("preserve_raw"),
            TONIC_OUTPUT_MODE_PRESERVE_RAW,
        )
        self.assertEqual(
            normalize_tonic_output_mode("flatten_session_bleach"),
            TONIC_OUTPUT_MODE_FLATTEN_BLEACH,
        )


if __name__ == "__main__":
    unittest.main()
