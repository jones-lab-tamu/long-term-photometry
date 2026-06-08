import pytest

from photometry_pipeline.config import Config


def test_config_defaults_include_intermittent_acquisition_mode():
    cfg = Config()
    assert cfg.acquisition_mode == "intermittent"
    assert float(cfg.continuous_window_sec) > 0.0
    assert float(cfg.continuous_step_sec) > 0.0


def test_config_defaults_use_artifact_aware_correction_and_nonzero_event_filters():
    cfg = Config()
    assert cfg.dynamic_fit_mode == "robust_global_event_reject"
    assert cfg.peak_min_distance_sec == pytest.approx(1.0)
    assert cfg.peak_min_prominence_k == pytest.approx(1.0)
    assert cfg.peak_min_width_sec == pytest.approx(0.2)


def test_explicit_yaml_overrides_artifact_aware_defaults(tmp_path):
    cfg_path = tmp_path / "explicit_defaults_override.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "dynamic_fit_mode: rolling_filtered_to_raw",
                "peak_min_distance_sec: 0.25",
                "peak_min_prominence_k: 0.0",
                "peak_min_width_sec: 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(str(cfg_path))
    assert cfg.dynamic_fit_mode == "rolling_filtered_to_raw"
    assert cfg.peak_min_distance_sec == pytest.approx(0.25)
    assert cfg.peak_min_prominence_k == pytest.approx(0.0)
    assert cfg.peak_min_width_sec == pytest.approx(0.0)


def test_config_accepts_continuous_mode_values(tmp_path):
    cfg_path = tmp_path / "continuous_ok.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "acquisition_mode: continuous",
                "continuous_window_sec: 900.0",
                "continuous_step_sec: 900.0",
                "allow_partial_final_window: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(str(cfg_path))
    assert cfg.acquisition_mode == "continuous"
    assert float(cfg.continuous_window_sec) == 900.0
    assert float(cfg.continuous_step_sec) == 900.0
    assert cfg.allow_partial_final_window is True


def test_config_rejects_invalid_acquisition_mode(tmp_path):
    cfg_path = tmp_path / "continuous_invalid_mode.yaml"
    cfg_path.write_text("acquisition_mode: unsupported\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid acquisition_mode"):
        Config.from_yaml(str(cfg_path))


def test_config_rejects_nonpositive_continuous_window(tmp_path):
    cfg_path = tmp_path / "continuous_invalid_window.yaml"
    cfg_path.write_text("continuous_window_sec: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="continuous_window_sec must be > 0"):
        Config.from_yaml(str(cfg_path))


def test_config_rejects_nonpositive_continuous_step(tmp_path):
    cfg_path = tmp_path / "continuous_invalid_step.yaml"
    cfg_path.write_text("continuous_step_sec: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="continuous_step_sec must be > 0"):
        Config.from_yaml(str(cfg_path))


def test_config_rejects_step_window_mismatch(tmp_path):
    cfg_path = tmp_path / "continuous_mismatch.yaml"
    cfg_path.write_text(
        "continuous_window_sec: 600.0\ncontinuous_step_sec: 300.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="continuous_step_sec must equal continuous_window_sec"):
        Config.from_yaml(str(cfg_path))


def test_existing_config_without_acquisition_mode_still_loads(tmp_path):
    cfg_path = tmp_path / "legacy_config.yaml"
    cfg_path.write_text("target_fs_hz: 40.0\n", encoding="utf-8")
    cfg = Config.from_yaml(str(cfg_path))
    assert cfg.acquisition_mode == "intermittent"
