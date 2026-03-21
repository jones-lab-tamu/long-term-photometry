import os
import subprocess
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "synth_photometry_dataset_v2.py"


def _write_test_config(
    path: Path,
    *,
    chunk_duration_sec: float = 60.0,
    target_fs_hz: float = 5.0,
) -> Path:
    cfg_path = path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"chunk_duration_sec: {chunk_duration_sec}",
                f"target_fs_hz: {target_fs_hz}",
                "rwd_time_col: TimeStamp",
                "uv_suffix: '-410'",
                "sig_suffix: '-470'",
                "window_sec: 20.0",
                "step_sec: 10.0",
                "r_low: -1.0",
                "r_high: 1.0",
                "g_min: 0.0",
                "min_valid_windows: 1",
                "min_samples_per_window: 20",
                "lowpass_hz: 1.0",
                "baseline_method: uv_raw_percentile_session",
                "baseline_percentile: 10.0",
                "peak_threshold_method: percentile",
                "peak_threshold_percentile: 90.0",
                "peak_min_distance_sec: 0.5",
                "qc_max_chunk_fail_fraction: 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return cfg_path


def _base_cmd(
    out_dir: Path,
    cfg_path: Path,
    preset: str = "fixture_basic",
    *,
    total_days: float = 1.0 / 24.0,
    recording_duration_min: float = 1.0,
    recordings_per_hour: int = 1,
    n_rois: int = 3,
    start_iso: str = "2025-01-01T00:00:00",
    seed: int = 123,
    fs_hz: float | None = None,
    tonic_amp: float | None = None,
    events_per_chunk_mean: float | None = None,
    uv_noise_std: float | None = None,
    sig_noise_std: float | None = None,
    shared_nuisance_std: float | None = None,
    slow_drift_amp: float | None = None,
    phase_lock_strength: float | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--out",
        str(out_dir),
        "--format",
        "rwd",
        "--config",
        str(cfg_path),
        "--preset",
        preset,
        "--total-days",
        str(total_days),
        "--recording-duration-min",
        str(recording_duration_min),
        "--recordings-per-hour",
        str(recordings_per_hour),
        "--n-rois",
        str(n_rois),
        "--start-iso",
        start_iso,
        "--seed",
        str(seed),
    ]
    if tonic_amp is not None:
        cmd.extend(["--tonic-amp", str(tonic_amp)])
    if events_per_chunk_mean is not None:
        cmd.extend(["--events-per-chunk-mean", str(events_per_chunk_mean)])
    if uv_noise_std is not None:
        cmd.extend(["--uv-noise-std", str(uv_noise_std)])
    if sig_noise_std is not None:
        cmd.extend(["--sig-noise-std", str(sig_noise_std)])
    if shared_nuisance_std is not None:
        cmd.extend(["--shared-nuisance-std", str(shared_nuisance_std)])
    if slow_drift_amp is not None:
        cmd.extend(["--slow-drift-amp", str(slow_drift_amp)])
    if phase_lock_strength is not None:
        cmd.extend(["--phase-lock-strength", str(phase_lock_strength)])
    if fs_hz is not None:
        cmd.extend(["--fs-hz", str(fs_hz)])
    return cmd


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    repo_root = str(REPO_ROOT)
    env["PYTHONPATH"] = repo_root if not existing else f"{repo_root}{os.pathsep}{existing}"
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_without_pythonpath(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _single_chunk_dir(out_dir: Path) -> Path:
    chunk_dirs = sorted([p for p in out_dir.iterdir() if p.is_dir()])
    assert len(chunk_dirs) == 1
    return chunk_dirs[0]


def _chunk_dirs(out_dir: Path) -> list[Path]:
    return sorted([p for p in out_dir.iterdir() if p.is_dir()])


def _read_manifest(out_dir: Path) -> dict:
    return yaml.safe_load((out_dir / "generation_manifest.yaml").read_text(encoding="utf-8"))


def _chunk_csv_bytes_map(out_dir: Path) -> dict[str, bytes]:
    return {
        f"{chunk.name}/fluorescence.csv": (chunk / "fluorescence.csv").read_bytes()
        for chunk in _chunk_dirs(out_dir)
    }


def test_v2_generator_creates_small_rwd_fixture_dataset(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_basic"
    res = _run(_base_cmd(out_dir, cfg, preset="fixture_basic"))
    assert res.returncode == 0, res.stderr
    assert out_dir.exists()
    assert _single_chunk_dir(out_dir).exists()


def test_v2_direct_execution_from_repo_root_without_pythonpath(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_direct_no_pythonpath"
    cmd = _base_cmd(out_dir, cfg, preset="fixture_basic", seed=321)
    cmd[1] = "tools/synth_photometry_dataset_v2.py"
    res = _run_without_pythonpath(cmd)
    assert res.returncode == 0, (
        f"Direct execution failed without PYTHONPATH\nCMD: {' '.join(cmd)}\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
    assert (out_dir / "generation_manifest.yaml").is_file()


def test_v2_direct_execution_has_no_overflow_runtime_warnings(tmp_path: Path):
    cfg = _write_test_config(tmp_path, chunk_duration_sec=600.0, target_fs_hz=20.0)
    out_dir = tmp_path / "out_no_overflow_warning"
    cmd = _base_cmd(
        out_dir,
        cfg,
        preset="fixture_shared_nuisance",
        total_days=1.0 / 24.0,
        recording_duration_min=10.0,
        recordings_per_hour=2,
        n_rois=5,
        seed=42,
    )
    cmd[1] = "tools/synth_photometry_dataset_v2.py"
    res = _run_without_pythonpath(cmd)
    assert res.returncode == 0, (
        f"Generator failed\nCMD: {' '.join(cmd)}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
    stderr_lower = res.stderr.lower()
    assert "overflow encountered in exp" not in stderr_lower
    assert "overflow encountered in multiply" not in stderr_lower
    assert (out_dir / "generation_manifest.yaml").is_file()


def test_v2_rwd_chunk_folder_structure(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_structure"
    res = _run(_base_cmd(out_dir, cfg))
    assert res.returncode == 0, res.stderr

    chunk_dir = _single_chunk_dir(out_dir)
    assert chunk_dir.name == "2025_01_01-00_00_00"
    assert (chunk_dir / "fluorescence.csv").is_file()


def test_v2_fluorescence_header_matches_config(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_header"
    res = _run(_base_cmd(out_dir, cfg))
    assert res.returncode == 0, res.stderr

    df = pd.read_csv(_single_chunk_dir(out_dir) / "fluorescence.csv")
    expected = [
        "TimeStamp",
        "Region0-470",
        "Region0-410",
        "Region1-470",
        "Region1-410",
        "Region2-470",
        "Region2-410",
    ]
    assert list(df.columns) == expected


def test_v2_row_count_matches_duration_times_fs(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_rows"
    res = _run(_base_cmd(out_dir, cfg))
    assert res.returncode == 0, res.stderr

    df = pd.read_csv(_single_chunk_dir(out_dir) / "fluorescence.csv")
    assert len(df) == 300  # 60s * 5Hz


def test_v2_manifest_written_with_key_fields(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_manifest"
    res = _run(_base_cmd(out_dir, cfg))
    assert res.returncode == 0, res.stderr

    manifest_path = out_dir / "generation_manifest.yaml"
    assert manifest_path.is_file()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["preset"] == "fixture_basic"
    assert manifest["base_preset_name"] == "fixture_basic"
    assert manifest["preset_intended_for_e2e_pipeline_validation"] is False
    assert manifest["preset_intended_as_fixture_scale_dataset"] is True
    assert manifest["used_cli_overrides"] is False
    assert manifest["overrides_applied"] == {}
    assert manifest["resolved_generation_values"] == {
        "tonic_amp": 0.18,
        "events_per_chunk_mean": 4.0,
        "uv_noise_std": 0.01,
        "sig_noise_std": 0.015,
        "shared_nuisance_std": 0.0,
        "slow_drift_amp": 0.0,
        "phase_lock_strength": 0.0,
    }
    assert manifest["seed"] == 123
    assert manifest["config_path"] == str(cfg.resolve())
    assert manifest["output_format"] == "rwd"
    assert manifest["resolved_chunk_duration_sec"] == 60.0
    assert manifest["resolved_samples_per_chunk"] == 300
    assert manifest["recordings_per_hour"] == 1
    assert manifest["chunk_interval_sec"] == 3600.0
    assert manifest["n_rois"] == 3
    assert manifest["roi_names"] == ["Region0", "Region1", "Region2"]
    assert manifest["normalized_start_iso"] == "2025-01-01T00:00:00"
    assert manifest["expected_headers"] == [
        "TimeStamp",
        "Region0-470",
        "Region0-410",
        "Region1-470",
        "Region1-410",
        "Region2-470",
        "Region2-410",
    ]
    assert manifest["total_chunk_count"] == 1
    summary = manifest["generated_data_summary"]
    assert summary["total_chunk_count"] == 1
    assert summary["rows_per_chunk"] == 300
    assert summary["total_row_count"] == 300
    assert set(summary["per_roi_mean_signal"].keys()) == {"Region0", "Region1", "Region2"}
    assert set(summary["per_roi_mean_uv"].keys()) == {"Region0", "Region1", "Region2"}
    assert set(summary["per_roi_signal_sd"].keys()) == {"Region0", "Region1", "Region2"}
    assert set(summary["per_roi_uv_sd"].keys()) == {"Region0", "Region1", "Region2"}
    assert set(summary["per_roi_signal_uv_corr"].keys()) == {"Region0", "Region1", "Region2"}
    assert manifest["resolved_fs_hz"] == 5.0
    assert "validation_summary" in manifest
    discoverability = manifest["validation_summary"]["discoverability"]
    assert discoverability["resolved_format"] == "RWD"
    assert discoverability["n_total_discovered"] == 1
    assert discoverability["n_rois"] == 3
    assert discoverability["roi_ids"] == ["Region0", "Region1", "Region2"]
    assert discoverability["interface"]["module"] == "photometry_pipeline.discovery"
    assert discoverability["interface"]["callable"] == "discover_inputs"


def test_v2_fixture_phase_locked_generates_successfully(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_phase_locked"
    res = _run(_base_cmd(out_dir, cfg, preset="fixture_phase_locked"))
    assert res.returncode == 0, res.stderr
    assert (_single_chunk_dir(out_dir) / "fluorescence.csv").is_file()


def test_v2_format_npm_fails_clearly(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_npm"
    cmd = _base_cmd(out_dir, cfg)
    idx = cmd.index("--format")
    cmd[idx + 1] = "npm"
    res = _run(cmd)
    assert res.returncode != 0
    message = (res.stderr + "\n" + res.stdout).lower()
    assert "npm output is intentionally not implemented" in message


def test_v2_duration_mismatch_fails_clearly(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_bad_duration"
    cmd = _base_cmd(out_dir, cfg)
    idx = cmd.index("--recording-duration-min")
    cmd[idx + 1] = "2.0"  # config says 60s (1.0 minute)
    res = _run(cmd)
    assert res.returncode != 0
    message = (res.stderr + "\n" + res.stdout).lower()
    assert "recording-duration-min mismatch with config chunk_duration_sec" in message


def test_v2_multi_chunk_structure_headers_and_rows(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_multi_chunk"
    res = _run(
        _base_cmd(
            out_dir,
            cfg,
            preset="fixture_shared_nuisance",
            total_days=1.0 / 12.0,  # 2h
            recordings_per_hour=2,  # every 30 min
            n_rois=4,
            seed=777,
        )
    )
    assert res.returncode == 0, res.stderr

    chunk_dirs = _chunk_dirs(out_dir)
    assert len(chunk_dirs) == 4

    start_dt = datetime.fromisoformat("2025-01-01T00:00:00")
    expected_names = [
        (start_dt + timedelta(minutes=30 * i)).strftime("%Y_%m_%d-%H_%M_%S")
        for i in range(4)
    ]
    assert [p.name for p in chunk_dirs] == expected_names

    expected_header = [
        "TimeStamp",
        "Region0-470",
        "Region0-410",
        "Region1-470",
        "Region1-410",
        "Region2-470",
        "Region2-410",
        "Region3-470",
        "Region3-410",
    ]
    for chunk_dir in chunk_dirs:
        csv_path = chunk_dir / "fluorescence.csv"
        assert csv_path.is_file()
        df = pd.read_csv(csv_path)
        assert list(df.columns) == expected_header
        assert len(df) == 300


def test_v2_deterministic_reproducibility_with_same_seed(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_a = tmp_path / "out_det_a"
    out_b = tmp_path / "out_det_b"
    cmd_a = _base_cmd(
        out_a,
        cfg,
        preset="fixture_phase_locked",
        total_days=1.0 / 12.0,
        recordings_per_hour=2,
        n_rois=3,
        seed=4242,
    )
    cmd_b = _base_cmd(
        out_b,
        cfg,
        preset="fixture_phase_locked",
        total_days=1.0 / 12.0,
        recordings_per_hour=2,
        n_rois=3,
        seed=4242,
    )

    res_a = _run(cmd_a)
    res_b = _run(cmd_b)
    assert res_a.returncode == 0, res_a.stderr
    assert res_b.returncode == 0, res_b.stderr

    manifest_a = _read_manifest(out_a)
    manifest_b = _read_manifest(out_b)
    assert manifest_a == manifest_b

    chunks_a = _chunk_dirs(out_a)
    chunks_b = _chunk_dirs(out_b)
    assert [p.name for p in chunks_a] == [p.name for p in chunks_b]
    for da, db in zip(chunks_a, chunks_b):
        csv_a = da / "fluorescence.csv"
        csv_b = db / "fluorescence.csv"
        assert csv_a.read_bytes() == csv_b.read_bytes()


def test_v2_default_preset_uses_no_cli_overrides(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_default_no_overrides"
    res = _run(_base_cmd(out_dir, cfg, preset="fixture_basic", seed=31337))
    assert res.returncode == 0, res.stderr

    manifest = _read_manifest(out_dir)
    assert manifest["used_cli_overrides"] is False
    assert manifest["overrides_applied"] == {}


def test_v2_manifest_records_explicit_cli_overrides(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_override_manifest"
    res = _run(
        _base_cmd(
            out_dir,
            cfg,
            preset="fixture_basic",
            seed=431,
            tonic_amp=0.42,
            shared_nuisance_std=0.03,
            phase_lock_strength=0.65,
        )
    )
    assert res.returncode == 0, res.stderr

    manifest = _read_manifest(out_dir)
    assert manifest["used_cli_overrides"] is True
    assert manifest["overrides_applied"] == {
        "tonic_amp": 0.42,
        "shared_nuisance_std": 0.03,
        "phase_lock_strength": 0.65,
    }
    resolved = manifest["resolved_generation_values"]
    assert resolved["tonic_amp"] == 0.42
    assert resolved["shared_nuisance_std"] == 0.03
    assert resolved["phase_lock_strength"] == 0.65


def test_v2_deterministic_reproducibility_with_same_seed_and_overrides(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_a = tmp_path / "out_det_override_a"
    out_b = tmp_path / "out_det_override_b"
    common_kwargs = {
        "preset": "fixture_phase_locked",
        "total_days": 1.0 / 12.0,
        "recordings_per_hour": 2,
        "n_rois": 3,
        "seed": 5511,
        "tonic_amp": 0.33,
        "events_per_chunk_mean": 2.5,
        "uv_noise_std": 0.013,
        "sig_noise_std": 0.019,
        "shared_nuisance_std": 0.012,
        "slow_drift_amp": 0.031,
        "phase_lock_strength": 0.5,
    }
    res_a = _run(_base_cmd(out_a, cfg, **common_kwargs))
    res_b = _run(_base_cmd(out_b, cfg, **common_kwargs))
    assert res_a.returncode == 0, res_a.stderr
    assert res_b.returncode == 0, res_b.stderr

    assert _read_manifest(out_a) == _read_manifest(out_b)
    assert _chunk_csv_bytes_map(out_a) == _chunk_csv_bytes_map(out_b)


def test_v2_override_changes_generated_data(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    baseline_out = tmp_path / "out_override_change_baseline"
    override_out = tmp_path / "out_override_change_override"
    common_kwargs = {
        "preset": "fixture_basic",
        "seed": 9988,
        "total_days": 1.0 / 12.0,
        "recordings_per_hour": 2,
        "n_rois": 3,
    }

    baseline_res = _run(_base_cmd(baseline_out, cfg, **common_kwargs))
    override_res = _run(_base_cmd(override_out, cfg, tonic_amp=0.37, **common_kwargs))
    assert baseline_res.returncode == 0, baseline_res.stderr
    assert override_res.returncode == 0, override_res.stderr

    baseline_manifest = _read_manifest(baseline_out)
    override_manifest = _read_manifest(override_out)
    assert (
        baseline_manifest["resolved_generation_values"]["tonic_amp"]
        != override_manifest["resolved_generation_values"]["tonic_amp"]
    )
    assert _chunk_csv_bytes_map(baseline_out) != _chunk_csv_bytes_map(override_out)


def test_v2_invalid_negative_tonic_amp_fails_clearly(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_bad_tonic_amp"
    res = _run(_base_cmd(out_dir, cfg, tonic_amp=-0.1))
    assert res.returncode != 0
    message = (res.stderr + "\n" + res.stdout).lower()
    assert "--tonic-amp must be >= 0." in message


def test_v2_invalid_negative_sig_noise_std_fails_clearly(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_bad_sig_noise"
    res = _run(_base_cmd(out_dir, cfg, sig_noise_std=-0.01))
    assert res.returncode != 0
    message = (res.stderr + "\n" + res.stdout).lower()
    assert "--sig-noise-std must be >= 0." in message


def test_v2_invalid_phase_lock_strength_fails_clearly(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_bad_phase_lock"
    res = _run(_base_cmd(out_dir, cfg, phase_lock_strength=1.5))
    assert res.returncode != 0
    message = (res.stderr + "\n" + res.stdout).lower()
    assert "--phase-lock-strength must be in [0, 1]." in message


def test_v2_relational_event_margin_feasibility_fails_clearly(tmp_path: Path):
    cfg = _write_test_config(tmp_path, chunk_duration_sec=8.0, target_fs_hz=5.0)
    out_dir = tmp_path / "out_bad_event_margin"
    res = _run(
        _base_cmd(
            out_dir,
            cfg,
            preset="fixture_basic",
            recording_duration_min=8.0 / 60.0,
        )
    )
    assert res.returncode != 0
    message = res.stderr + "\n" + res.stdout
    assert "Infeasible event placement interior" in message


def test_v2_relational_event_spacing_feasibility_fails_clearly(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_bad_event_density"
    res = _run(
        _base_cmd(
            out_dir,
            cfg,
            preset="fixture_basic",
            events_per_chunk_mean=100.0,
        )
    )
    assert res.returncode != 0
    message = res.stderr + "\n" + res.stdout
    assert "Infeasible event density" in message


def test_v2_relational_waveform_sampling_feasibility_fails_clearly(tmp_path: Path):
    cfg = _write_test_config(tmp_path)
    out_dir = tmp_path / "out_bad_waveform_sampling"
    res = _run(
        _base_cmd(
            out_dir,
            cfg,
            preset="fixture_basic",
            fs_hz=0.1,
        )
    )
    assert res.returncode != 0
    message = res.stderr + "\n" + res.stdout
    assert "Infeasible event waveform sampling" in message


def test_v2_fixture_e2e_small_runs_tonic_phasic_and_full_deliverables(tmp_path: Path):
    cfg = _write_test_config(tmp_path, chunk_duration_sec=120.0, target_fs_hz=5.0)
    out_data = tmp_path / "out_fixture_e2e_small"
    gen_res = _run(
        _base_cmd(
            out_data,
            cfg,
            preset="fixture_e2e_small",
            total_days=1.0 / 16.0,  # 3 chunks with recordings_per_hour=2
            recording_duration_min=2.0,
            recordings_per_hour=2,
            n_rois=2,
            seed=2026,
        )
    )
    assert gen_res.returncode == 0, (
        f"Generator failed\nSTDOUT:\n{gen_res.stdout}\nSTDERR:\n{gen_res.stderr}"
    )

    manifest = _read_manifest(out_data)
    assert manifest["preset"] == "fixture_e2e_small"
    assert manifest["preset_intended_for_e2e_pipeline_validation"] is True
    assert manifest["preset_intended_as_fixture_scale_dataset"] is True
    assert manifest["total_chunk_count"] == 3

    tonic_out = tmp_path / "tonic_out"
    tonic_cmd = [
        sys.executable,
        "analyze_photometry.py",
        "--input",
        str(out_data),
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
    ]
    tonic_res = _run(tonic_cmd)
    assert tonic_res.returncode == 0, (
        f"Tonic analysis failed\nCMD: {' '.join(tonic_cmd)}\n"
        f"STDOUT:\n{tonic_res.stdout}\nSTDERR:\n{tonic_res.stderr}"
    )
    assert tonic_out.is_dir()
    assert (tonic_out / "run_report.json").is_file()
    assert (tonic_out / "tonic_trace_cache.h5").is_file()
    tonic_report = json.loads((tonic_out / "run_report.json").read_text(encoding="utf-8"))
    assert tonic_report["run_context"]["n_sessions_resolved"] == manifest["total_chunk_count"]
    assert len(tonic_report["roi_selection"]["selected_rois"]) == manifest["n_rois"]

    phasic_out = tmp_path / "phasic_out"
    phasic_cmd = [
        sys.executable,
        "analyze_photometry.py",
        "--input",
        str(out_data),
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
    ]
    phasic_res = _run(phasic_cmd)
    assert phasic_res.returncode == 0, (
        f"Phasic analysis failed\nCMD: {' '.join(phasic_cmd)}\n"
        f"STDOUT:\n{phasic_res.stdout}\nSTDERR:\n{phasic_res.stderr}"
    )
    assert phasic_out.is_dir()
    assert (phasic_out / "run_report.json").is_file()
    assert (phasic_out / "phasic_trace_cache.h5").is_file()
    features_csv = phasic_out / "features" / "features.csv"
    assert features_csv.is_file()
    phasic_report = json.loads((phasic_out / "run_report.json").read_text(encoding="utf-8"))
    assert phasic_report["run_context"]["n_sessions_resolved"] == manifest["total_chunk_count"]
    assert len(phasic_report["roi_selection"]["selected_rois"]) == manifest["n_rois"]
    features_df = pd.read_csv(features_csv)
    assert len(features_df) > 0

    full_out = tmp_path / "full_out"
    full_cmd = [
        sys.executable,
        "tools/run_full_pipeline_deliverables.py",
        "--input",
        str(out_data),
        "--out",
        str(full_out),
        "--config",
        str(cfg),
        "--format",
        "rwd",
        "--overwrite",
        "--sessions-per-hour",
        "2",
    ]
    full_res = _run(full_cmd)
    assert full_res.returncode == 0, (
        f"Full deliverables failed\nCMD: {' '.join(full_cmd)}\n"
        f"STDOUT:\n{full_res.stdout}\nSTDERR:\n{full_res.stderr}"
    )

    assert full_out.is_dir()
    assert (full_out / "MANIFEST.json").is_file()
    assert (full_out / "run_report.json").is_file()
    assert (full_out / "_analysis" / "tonic_out").is_dir()
    assert (full_out / "_analysis" / "phasic_out").is_dir()
    root_report = json.loads((full_out / "run_report.json").read_text(encoding="utf-8"))
    assert root_report["run_context"]["n_sessions_resolved"] == manifest["total_chunk_count"]

    roi0_summary = full_out / "Region0" / "summary"
    assert roi0_summary.is_dir()
    summary_pngs = sorted(roi0_summary.glob("*.png"))
    assert summary_pngs, "Expected at least one Region0 summary PNG deliverable."
