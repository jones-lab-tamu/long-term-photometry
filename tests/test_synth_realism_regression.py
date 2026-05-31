import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gui.run_report_parser import is_successful_completed_run_dir


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)


def _assert_ok(res: subprocess.CompletedProcess[str], context: str) -> None:
    assert res.returncode == 0, (
        f"{context} failed\n"
        f"CMD: {' '.join(res.args)}\n"
        f"STDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}\n"
    )


def _write_cfg(path: Path, *, chunk_duration_sec: int = 60, target_fs_hz: int = 5) -> Path:
    cfg_path = path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"chunk_duration_sec: {chunk_duration_sec}",
                f"target_fs_hz: {target_fs_hz}",
                "baseline_method: uv_raw_percentile_session",
                "baseline_percentile: 10",
                'rwd_time_col: "TimeStamp"',
                'uv_suffix: "-410"',
                'sig_suffix: "-470"',
                'npm_frame_col: "FrameCounter"',
                'npm_system_ts_col: "Timestamp"',
                'npm_computer_ts_col: "Timestamp"',
                'npm_led_col: "LedState"',
                'npm_region_prefix: "Region"',
                'npm_region_suffix: "G"',
                'npm_time_axis: "system_timestamp"',
                "window_sec: 20.0",
                "step_sec: 5.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return cfg_path


def _generate_fixture(
    tmp_path: Path,
    *,
    name: str,
    fmt: str,
    cfg_path: Path,
    extra_args: list[str],
) -> Path:
    out_dir = tmp_path / name
    cmd = [
        sys.executable,
        "tools/synth_photometry_dataset.py",
        "--out",
        str(out_dir),
        "--format",
        fmt,
        "--config",
        str(cfg_path),
    ] + extra_args
    res = _run(cmd)
    _assert_ok(res, f"fixture generation [{name}]")
    return out_dir


def _load_manifest(out_dir: Path) -> dict:
    m = out_dir / "generation_manifest.yaml"
    assert m.is_file(), f"missing generation manifest: {m}"
    data = yaml.safe_load(m.read_text(encoding="utf-8")) or {}
    assert isinstance(data, dict)
    return data


def test_realism_fixture_matrix_generates_expected_anomaly_metadata(tmp_path: Path):
    cfg = _write_cfg(tmp_path, chunk_duration_sec=60, target_fs_hz=5)
    base_common = [
        "--total-days",
        "0.12",
        "--recording-duration-min",
        "1",
        "--recordings-per-hour",
        "3",
        "--fs-hz",
        "5",
        "--n-rois",
        "2",
        "--seed",
        "1234",
    ]

    fixtures: dict[str, tuple[str, list[str]]] = {
        "clean_rwd": (
            "rwd",
            base_common + ["--preset", "biological_shared_nuisance", "--start-iso", "2025-01-01T00:00:00"],
        ),
        "rwd_realism_stress": (
            "rwd",
            base_common
            + [
                "--preset",
                "realism_stress",
                "--start-iso",
                "2025-01-01T00:00:00",
                "--start-iso-random-offset-min",
                "37",
                "37",
            ],
        ),
        "clean_npm": (
            "npm",
            base_common + ["--preset", "biological_shared_nuisance", "--start-iso", "2025-01-01T00:00:00"],
        ),
        "npm_realism_stress": (
            "npm",
            base_common
            + [
                "--preset",
                "realism_stress",
                "--start-iso",
                "2025-01-01T00:00:00",
                "--start-iso-random-offset-min",
                "31",
                "31",
            ],
        ),
        "near_threshold_coverage": (
            "rwd",
            base_common
            + [
                "--preset",
                "biological_shared_nuisance",
                "--start-iso",
                "2025-01-01T00:00:00",
                "--near-threshold-end-coverage",
                "true",
                "--edge-truncate-samples-max",
                "2",
            ],
        ),
        "non_midnight_fixed_anchor": (
            "rwd",
            base_common
            + [
                "--preset",
                "biological_shared_nuisance",
                "--start-iso",
                "2025-01-01T13:37:11",
                "--start-iso-random-offset-min",
                "17",
                "17",
            ],
        ),
        "dropped_truncated_session": (
            "rwd",
            base_common
            + [
                "--preset",
                "biological_shared_nuisance",
                "--start-iso",
                "2025-01-01T00:00:00",
                "--session-drop-prob",
                "0.5",
                "--edge-truncate-samples-max",
                "4",
            ],
        ),
    }

    generated: dict[str, Path] = {}
    manifests: dict[str, dict] = {}
    for name, (fmt, args) in fixtures.items():
        out_dir = _generate_fixture(tmp_path, name=name, fmt=fmt, cfg_path=cfg, extra_args=args)
        generated[name] = out_dir
        manifests[name] = _load_manifest(out_dir)

    required_manifest_keys = {
        "generator",
        "command",
        "resolved_preset",
        "seed",
        "start_time",
        "sessions_requested",
        "sessions_generated",
        "sessions_dropped",
        "sessions_truncated",
        "per_session_sample_counts",
        "timestamp_jitter",
        "per_channel_support_summary",
        "npm_uv_sig_asymmetry",
        "near_threshold_coverage",
    }
    for name, manifest in manifests.items():
        assert required_manifest_keys.issubset(set(manifest.keys())), name

    clean_rwd = manifests["clean_rwd"]
    stress_rwd = manifests["rwd_realism_stress"]
    clean_npm = manifests["clean_npm"]
    stress_npm = manifests["npm_realism_stress"]
    near_thresh = manifests["near_threshold_coverage"]
    non_midnight = manifests["non_midnight_fixed_anchor"]
    dropped_trunc = manifests["dropped_truncated_session"]

    assert clean_rwd["resolved_preset"] == "biological_shared_nuisance"
    assert stress_rwd["resolved_preset"] == "realism_stress"
    assert stress_rwd["sessions_generated"] > 0
    assert stress_rwd["timestamp_jitter"]["enabled"] is True

    clean_gap = clean_npm["npm_uv_sig_asymmetry"]["observed_start_gap_sec"]["max"] or 0.0
    stress_gap = stress_npm["npm_uv_sig_asymmetry"]["observed_start_gap_sec"]["max"] or 0.0
    assert float(stress_gap) > float(clean_gap)

    assert bool(near_thresh["near_threshold_coverage"]["enabled"]) is True
    assert near_thresh["near_threshold_coverage"]["count"] >= 1

    expected_start = "2025-01-01T13:54:11"
    assert non_midnight["start_time"]["resolved_start_iso"] == expected_start

    assert dropped_trunc["sessions_dropped"]["count"] >= 1
    assert dropped_trunc["sessions_truncated"]["count"] >= 1


def test_non_midnight_fixed_anchor_validation_path(tmp_path: Path):
    cfg = _write_cfg(tmp_path, chunk_duration_sec=60, target_fs_hz=5)
    data_dir = _generate_fixture(
        tmp_path,
        name="fixed_anchor_input",
        fmt="rwd",
        cfg_path=cfg,
        extra_args=[
            "--preset",
            "biological_shared_nuisance",
            "--total-days",
            "0.08",
            "--recording-duration-min",
            "1",
            "--recordings-per-hour",
            "2",
            "--fs-hz",
            "5",
            "--n-rois",
            "2",
            "--start-iso",
            "2025-01-01T13:37:11",
            "--start-iso-random-offset-min",
            "11",
            "11",
            "--seed",
            "99",
        ],
    )
    out_dir = tmp_path / "fixed_anchor_validate_out"
    cmd_validate = [
        sys.executable,
        "tools/run_full_pipeline_deliverables.py",
        "--input",
        str(data_dir),
        "--out",
        str(out_dir),
        "--config",
        str(cfg),
        "--format",
        "rwd",
        "--validate-only",
        "--sessions-per-hour",
        "2",
        "--timeline-anchor-mode",
        "fixed_daily_anchor",
        "--fixed-daily-anchor-clock",
        "06:00:00",
    ]
    res = _run(cmd_validate)
    _assert_ok(res, "fixed-anchor validate-only")


@pytest.mark.parametrize(
    "fixture_name,preset,total_days,extra",
    [
        (
            "clean_rwd_full_workflow",
            "biological_shared_nuisance",
            "0.12",
            [],
        ),
        (
            "realism_rwd_full_workflow",
            "realism_stress",
            "0.25",
            [
                "--session-drop-prob",
                "0.1",
                "--edge-truncate-samples-max",
                "1",
                "--timestamp-jitter-ms-std",
                "2.0",
                "--near-threshold-end-coverage",
                "false",
            ],
        ),
    ],
)
def test_rwd_fixtures_run_validate_analysis_plot_and_retune(
    tmp_path: Path,
    fixture_name: str,
    preset: str,
    total_days: str,
    extra: list[str],
):
    cfg = _write_cfg(tmp_path, chunk_duration_sec=60, target_fs_hz=5)
    data_dir = _generate_fixture(
        tmp_path,
        name=fixture_name,
        fmt="rwd",
        cfg_path=cfg,
        extra_args=[
            "--preset",
            preset,
            "--total-days",
            total_days,
            "--recording-duration-min",
            "1",
            "--recordings-per-hour",
            "2",
            "--fs-hz",
            "5",
            "--n-rois",
            "2",
            "--start-iso",
            "2025-01-01T08:15:00",
            "--seed",
            "2026",
        ]
        + extra,
    )
    _load_manifest(data_dir)

    run_dir = tmp_path / f"{fixture_name}_run"
    validate_cmd = [
        sys.executable,
        "tools/run_full_pipeline_deliverables.py",
        "--input",
        str(data_dir),
        "--out",
        str(run_dir),
        "--config",
        str(cfg),
        "--format",
        "rwd",
        "--validate-only",
        "--sessions-per-hour",
        "2",
    ]
    _assert_ok(_run(validate_cmd), f"{fixture_name} validate-only")

    tonic_out = tmp_path / f"{fixture_name}_tonic"
    tonic_cmd = [
        sys.executable,
        "analyze_photometry.py",
        "--input",
        str(data_dir),
        "--config",
        str(cfg),
        "--out",
        str(tonic_out),
        "--format",
        "rwd",
        "--mode",
        "tonic",
        "--recursive",
        "--overwrite",
        "--sessions-per-hour",
        "2",
    ]
    _assert_ok(_run(tonic_cmd), f"{fixture_name} tonic")

    phasic_out = tmp_path / f"{fixture_name}_phasic"
    phasic_cmd = [
        sys.executable,
        "analyze_photometry.py",
        "--input",
        str(data_dir),
        "--config",
        str(cfg),
        "--out",
        str(phasic_out),
        "--format",
        "rwd",
        "--mode",
        "phasic",
        "--recursive",
        "--overwrite",
        "--sessions-per-hour",
        "2",
    ]
    _assert_ok(_run(phasic_cmd), f"{fixture_name} phasic")

    plot_rate = tmp_path / f"{fixture_name}_rate.png"
    plot_auc = tmp_path / f"{fixture_name}_auc.png"
    plot_cmd = [
        sys.executable,
        "tools/plot_phasic_time_series_summary.py",
        "--analysis-out",
        str(phasic_out),
        "--sessions-per-hour",
        "2",
        "--out-rate-png",
        str(plot_rate),
        "--out-auc-png",
        str(plot_auc),
    ]
    _assert_ok(_run(plot_cmd), f"{fixture_name} plotting")
    assert plot_rate.is_file()
    assert plot_auc.is_file()

    full_cmd = [
        sys.executable,
        "tools/run_full_pipeline_deliverables.py",
        "--input",
        str(data_dir),
        "--out",
        str(run_dir),
        "--config",
        str(cfg),
        "--format",
        "rwd",
        "--mode",
        "both",
        "--sessions-per-hour",
        "2",
        "--overwrite",
    ]
    _assert_ok(_run(full_cmd), f"{fixture_name} full deliverables")
    ok, reason = is_successful_completed_run_dir(str(run_dir))
    assert ok, reason

    correction_cmd = [
        sys.executable,
        "tools/run_cache_correction_retune.py",
        "--run-dir",
        str(run_dir),
        "--roi",
        "CH1",
        "--set",
        "dynamic_fit_mode=global_linear_regression",
    ]
    _assert_ok(_run(correction_cmd), f"{fixture_name} correction retune")

    downstream_cmd = [
        sys.executable,
        "tools/run_cache_downstream_retune.py",
        "--run-dir",
        str(run_dir),
        "--roi",
        "CH1",
        "--set",
        "peak_threshold_method=absolute",
        "--set",
        "peak_threshold_abs=0.05",
    ]
    _assert_ok(_run(downstream_cmd), f"{fixture_name} downstream retune")


@pytest.mark.parametrize(
    "fixture_name,preset",
    [
        ("clean_npm_pipeline", "biological_shared_nuisance"),
        ("stress_npm_pipeline", "realism_stress"),
    ],
)
def test_npm_fixtures_run_phasic_pipeline_and_manifest(tmp_path: Path, fixture_name: str, preset: str):
    cfg = _write_cfg(tmp_path, chunk_duration_sec=60, target_fs_hz=5)
    data_dir = _generate_fixture(
        tmp_path,
        name=fixture_name,
        fmt="npm",
        cfg_path=cfg,
        extra_args=[
            "--preset",
            preset,
            "--total-days",
            "0.10",
            "--recording-duration-min",
            "1",
            "--recordings-per-hour",
            "2",
            "--fs-hz",
            "5",
            "--n-rois",
            "2",
            "--start-iso",
            "2025-01-01T09:17:00",
            "--seed",
            "3030",
        ],
    )
    manifest = _load_manifest(data_dir)
    assert manifest["sessions_generated"] > 0

    phasic_out = tmp_path / f"{fixture_name}_phasic_out"
    phasic_cmd = [
        sys.executable,
        "analyze_photometry.py",
        "--input",
        str(data_dir),
        "--config",
        str(cfg),
        "--out",
        str(phasic_out),
        "--format",
        "npm",
        "--mode",
        "phasic",
        "--overwrite",
        "--sessions-per-hour",
        "2",
    ]
    _assert_ok(_run(phasic_cmd), f"{fixture_name} phasic")
    assert (phasic_out / "run_report.json").is_file()
    assert (phasic_out / "phasic_trace_cache.h5").is_file()
    assert (phasic_out / "features" / "features.csv").is_file()
