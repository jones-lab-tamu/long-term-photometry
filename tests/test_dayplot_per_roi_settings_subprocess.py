"""4J16k39 follow-up: a bounded, real-subprocess proof that detector-aware day
plotting uses each ROI's OWN recorded feature settings.

This deliberately builds the smallest fixture the day-plot script can consume
(one short chunk, one ROI, one day) and invokes tools/plot_phasic_dayplot_bundle.py
as an actual process. It does NOT touch the full synthetic multi-stage wrapper or
tests/test_phasic_verification_chain.py (a known pre-existing hang).

The Custom ROI's effective settings detect a different number of peaks than the
global Default settings, so:
  - running with --provenance-mode current + the real provenance file succeeds
    (strict peak-count verification replays the ROI's own settings);
  - running the same fixture against the global configuration reproduces the
    original C2 defect and fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from photometry_pipeline.config import Config
from photometry_pipeline.core.feature_extraction import get_peak_indices_for_trace
from photometry_pipeline.feature_event_provenance import (
    FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION,
    FEATURE_EVENT_PROVENANCE_FILENAME,
    FEATURE_EVENT_PROVENANCE_SCHEMA_V3,
    build_feature_event_provenance_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROI = "Region0"
FS_HZ = 10.0


def _trace(n_sec=120.0):
    t = np.arange(0.0, n_sec, 1.0 / FS_HZ)
    rng = np.random.default_rng(11)
    dff = 0.02 * np.sin(0.07 * t) + 0.01 * rng.standard_normal(len(t))
    for center in (20.0, 50.0, 80.0, 105.0):
        dff += 1.8 * np.exp(-0.5 * ((t - center) / 0.9) ** 2)
    return t, dff


def _write_cache(analysis_out: Path, t, dff):
    import h5py

    with h5py.File(analysis_out / "phasic_trace_cache.h5", "w") as f:
        meta = f.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        dt_str = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("rois", data=np.array([ROI], dtype=object), dtype=dt_str)
        meta.create_dataset("chunk_ids", data=np.array([0], dtype=int), maxshape=(None,))
        c = f.require_group(f"roi/{ROI}/chunk_0")
        c.create_dataset("time_sec", data=t)
        c.create_dataset("sig_raw", data=np.zeros_like(t))
        c.create_dataset("uv_raw", data=np.zeros_like(t))
        c.create_dataset("fit_ref", data=np.zeros_like(t))
        c.create_dataset("dff", data=dff)


def _write_config_used(analysis_out: Path, cfg: Config):
    with open(analysis_out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg.__dict__), f)


def _write_run_report(analysis_out: Path, *, current: bool):
    report = {
        "run_context": {"run_type": "full"},
        "analytical_contract": {"strict_mode_guarantees": []},
        "configuration": {"target_fs_hz": FS_HZ},
    }
    if current:
        report["feature_event_provenance"] = {
            "contract_version": FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION,
            "schema_version": FEATURE_EVENT_PROVENANCE_SCHEMA_V3,
        }
    with open(analysis_out / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f)


@pytest.fixture
def custom_roi_run(tmp_path):
    """A current-contract analysis output where ROI's Custom settings genuinely
    detect a different peak count than the global Default settings."""
    analysis_out = tmp_path / "_analysis"
    (analysis_out / "features").mkdir(parents=True)
    output_dir = tmp_path / "day_plots"
    output_dir.mkdir()

    t, dff = _trace()

    # Global Default: strict. Custom: permissive -> different peak count.
    global_cfg = replace(
        Config(),
        target_fs_hz=FS_HZ,
        chunk_duration_sec=120.0,
        event_signal="dff",
        peak_threshold_method="mean_std",
        peak_threshold_k=3.0,
        peak_min_prominence_k=2.0,
        peak_min_width_sec=0.3,
        peak_min_distance_sec=1.0,
        peak_pre_filter="none",
    )
    custom_cfg = replace(
        global_cfg,
        peak_threshold_method="percentile",
        peak_threshold_percentile=1.0,
        peak_min_prominence_k=0.0,
        peak_min_width_sec=0.0,
    )

    expected_default = len(get_peak_indices_for_trace(dff, FS_HZ, global_cfg))
    expected_custom = len(get_peak_indices_for_trace(dff, FS_HZ, custom_cfg))
    assert expected_custom != expected_default, "fixture must actually diverge"
    assert expected_custom > 0

    # features.csv records what analysis produced for this ROI: the CUSTOM count.
    pd.DataFrame(
        [{"chunk_id": 0, "roi": ROI, "peak_count": expected_custom, "auc": 0.0}]
    ).to_csv(analysis_out / "features" / "features.csv", index=False)

    _write_cache(analysis_out, t, dff)
    _write_config_used(analysis_out, global_cfg)
    _write_run_report(analysis_out, current=True)

    payload = build_feature_event_provenance_payload(
        base_config=global_cfg,
        analyzed_rois=[ROI],
        per_roi_feature_config={ROI: custom_cfg},
        per_roi_source_details={
            ROI: {
                "feature_event_profile_id": "custom-region0",
                "override_config_fields": {
                    "peak_threshold_method": "percentile",
                    "peak_threshold_percentile": 1.0,
                    "peak_min_prominence_k": 0.0,
                    "peak_min_width_sec": 0.0,
                },
            }
        },
    )
    prov_path = analysis_out / "features" / FEATURE_EVENT_PROVENANCE_FILENAME
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return {
        "analysis_out": analysis_out,
        "output_dir": output_dir,
        "provenance_path": prov_path,
        "payload": payload,
        "expected_custom": expected_custom,
        "expected_default": expected_default,
    }


def _run_dayplot(analysis_out, output_dir, extra):
    cmd = [
        sys.executable,
        "tools/plot_phasic_dayplot_bundle.py",
        "--analysis-out", str(analysis_out),
        "--roi", ROI,
        "--output-dir", str(output_dir),
        "--sessions-per-hour", "1",
        "--session-duration-s", "120",
        "--write-dff-grid",
        "--no-write-sig-iso-grid",
        "--no-write-stacked",
        *extra,
    ]
    return subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False
    )


def test_dayplot_subprocess_uses_custom_roi_settings_and_verifies(custom_roi_run):
    """Positive proof through the real process: strict peak verification succeeds
    because it replays the Custom ROI's own recorded settings."""
    res = _run_dayplot(
        custom_roi_run["analysis_out"],
        custom_roi_run["output_dir"],
        [
            "--provenance-mode", "current",
            "--feature-event-provenance", str(custom_roi_run["provenance_path"]),
        ],
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Plotting Logic Mismatch" not in res.stdout

    identity_path = custom_roi_run["output_dir"] / "dayplot_feature_config.json"
    assert identity_path.is_file()
    identity = json.loads(identity_path.read_text(encoding="utf-8"))

    entry = custom_roi_run["payload"]["rois"][0]
    assert identity["roi"] == ROI
    assert identity["provenance_mode"] == "current"
    assert identity["source"] == "override"
    assert identity["verified_against_roi_provenance"] is True
    assert identity["verified_against_config_used"] is True
    # The day-plot metadata records the SAME effective configuration identity
    # that analysis recorded for this ROI.
    assert identity["effective_config_digest"] == entry["effective_config_digest"]
    assert identity["global_default_config_digest"] == (
        custom_roi_run["payload"]["global_default_config_digest"]
    )


def test_dayplot_subprocess_against_global_config_reproduces_old_mismatch(custom_roi_run):
    """Negative control: replaying this Custom ROI with the global Default
    configuration (the pre-4J16k39 behavior) fails strict verification."""
    res = _run_dayplot(
        custom_roi_run["analysis_out"], custom_roi_run["output_dir"], ["--provenance-mode", "legacy"]
    )
    assert res.returncode != 0
    assert "Plotting Logic Mismatch" in res.stdout


def test_dayplot_subprocess_current_mode_without_provenance_fails(custom_roi_run):
    res = _run_dayplot(
        custom_roi_run["analysis_out"], custom_roi_run["output_dir"], ["--provenance-mode", "current"]
    )
    assert res.returncode != 0
    assert "did not supply" in res.stdout


def test_dayplot_subprocess_missing_roi_entry_fails(custom_roi_run):
    payload = json.loads(custom_roi_run["provenance_path"].read_text(encoding="utf-8"))
    payload["rois"] = [dict(payload["rois"][0], roi="SomeOtherRoi")]
    custom_roi_run["provenance_path"].write_text(json.dumps(payload), encoding="utf-8")

    res = _run_dayplot(
        custom_roi_run["analysis_out"],
        custom_roi_run["output_dir"],
        [
            "--provenance-mode", "current",
            "--feature-event-provenance", str(custom_roi_run["provenance_path"]),
        ],
    )
    assert res.returncode != 0
    assert "no entry for ROI" in res.stdout


def test_dayplot_subprocess_config_used_mismatch_fails(custom_roi_run):
    """config_used.yaml drifting from the recorded global Default must fail before
    any marker replay or strict verification."""
    analysis_out = custom_roi_run["analysis_out"]
    drifted = Config.from_yaml(str(analysis_out / "config_used.yaml"))
    drifted.peak_threshold_k = drifted.peak_threshold_k + 1.0
    _write_config_used(analysis_out, drifted)

    res = _run_dayplot(
        analysis_out,
        custom_roi_run["output_dir"],
        [
            "--provenance-mode", "current",
            "--feature-event-provenance", str(custom_roi_run["provenance_path"]),
        ],
    )
    assert res.returncode != 0
    assert "config_used.yaml does not match" in res.stdout
