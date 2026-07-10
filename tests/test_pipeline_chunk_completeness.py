"""Every admitted intermittent chunk reaches one terminal disposition (4J16k41 / C8).

These drive the real Pipeline in-process over a small multi-chunk RWD recording.
An admitted chunk that cannot be processed must fail the run; it is never omitted
from the outputs while the run still succeeds.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.input_processing_completeness import (
    DISPOSITION_AUTHORIZED_EXCLUSION,
    INPUT_COMPLETENESS_FILENAME,
    InputProcessingError,
    validate_input_completeness,
)

CHUNK_NAMES = [
    "2024_01_01-00_00_00",
    "2024_01_01-01_00_00",
    "2024_01_01-02_00_00",
]


def _write_rwd_chunk(chunk_dir: Path, *, n: int = 600, fs: float = 10.0, seed: int = 0, flat: bool = False):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    if flat:
        uv = np.ones(n)
        sig = np.ones(n) * 1.2
    else:
        uv = 1.0 + 0.05 * np.sin(t * 0.3) + rng.normal(0, 0.02, n)
        sig = 1.2 * uv + 0.1 * rng.normal(0, 1, n)
    pd.DataFrame({"TimeStamp": t, "Region0-470": sig, "Region0-410": uv}).to_csv(
        chunk_dir / "fluorescence.csv", index=False
    )


def _build_input(tmp_path: Path, *, malformed=(), flat=()) -> Path:
    inp = tmp_path / "input"
    for i, name in enumerate(CHUNK_NAMES):
        if i in malformed:
            (inp / name).mkdir(parents=True, exist_ok=True)
            (inp / name / "fluorescence.csv").write_text("not,valid\nrwd,chunk\n", encoding="utf-8")
        else:
            _write_rwd_chunk(inp / name, seed=i, flat=(i in flat))
    return inp


def _config(tmp_path: Path, **overrides) -> Config:
    cfg = Config()
    cfg.chunk_duration_sec = 60
    cfg.target_fs_hz = 10
    cfg.baseline_method = "uv_raw_percentile_session"
    cfg.baseline_percentile = 10
    cfg.rwd_time_col = "TimeStamp"
    cfg.uv_suffix = "-410"
    cfg.sig_suffix = "-470"
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _run(tmp_path: Path, cfg: Config, inp: Path, *, mode: str = "phasic") -> Path:
    out = tmp_path / f"out_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    Pipeline(cfg, mode=mode).run(str(inp), str(out), force_format="rwd", recursive=True)
    return out


def _record(out: Path) -> dict:
    return json.loads((out / INPUT_COMPLETENESS_FILENAME).read_text(encoding="utf-8"))


# 1 / 16. Complete input set, exact accounting, zero-event chunk still counts ---


def test_all_admitted_chunks_process_with_exact_accounting(tmp_path: Path):
    # Include a flat (zero-event) chunk to prove it still counts as processed.
    out = _run(tmp_path, _config(tmp_path), _build_input(tmp_path, flat=(1,)))

    record = _record(out)
    assert validate_input_completeness(record) == ""
    assert len(record["expected"]) == 3
    assert len(record["processed"]) == 3
    assert {p["index"] for p in record["processed"]} == {0, 1, 2}


# 2 / 12. Malformed chunk fails the run, never omitted ------------------------


def test_middle_malformed_chunk_fails_the_run(tmp_path: Path):
    with pytest.raises(Exception):
        _run(tmp_path, _config(tmp_path), _build_input(tmp_path, malformed=(1,)))
    assert not (tmp_path / "out_phasic" / INPUT_COMPLETENESS_FILENAME).exists()


def test_malformed_final_chunk_without_exclusion_fails(tmp_path: Path):
    with pytest.raises(Exception):
        _run(tmp_path, _config(tmp_path), _build_input(tmp_path, malformed=(2,)))
    assert not (tmp_path / "out_phasic" / INPUT_COMPLETENESS_FILENAME).exists()


# 5 / 6 / 7. Injected load / correction / feature exceptions fail the run ------


def test_load_exception_for_one_chunk_fails(tmp_path: Path, monkeypatch):
    inp = _build_input(tmp_path)
    import photometry_pipeline.pipeline as pipeline_mod

    real_load = pipeline_mod.load_chunk

    def _boom(path, fmt, config, **kwargs):
        if CHUNK_NAMES[1] in str(path):
            raise RuntimeError("injected load failure")
        return real_load(path, fmt, config, **kwargs)

    monkeypatch.setattr(pipeline_mod, "load_chunk", _boom)
    with pytest.raises(Exception):
        _run(tmp_path, _config(tmp_path), inp)
    assert not (tmp_path / "out_phasic" / INPUT_COMPLETENESS_FILENAME).exists()


def test_feature_extraction_exception_for_one_chunk_fails(tmp_path: Path, monkeypatch):
    inp = _build_input(tmp_path)
    from photometry_pipeline.core import feature_extraction

    real_extract = feature_extraction.extract_features
    calls = {"n": 0}

    def _boom(chunk, config, per_roi_config=None):
        calls["n"] += 1
        if calls["n"] == 2:  # fail on the second processed chunk
            raise RuntimeError("injected feature-extraction failure")
        return real_extract(chunk, config, per_roi_config=per_roi_config)

    monkeypatch.setattr(feature_extraction, "extract_features", _boom)
    with pytest.raises(Exception):
        _run(tmp_path, _config(tmp_path), inp)
    assert not (tmp_path / "out_phasic" / INPUT_COMPLETENESS_FILENAME).exists()


def test_analysis_exception_for_one_chunk_fails(tmp_path: Path, monkeypatch):
    """A per-chunk correction/analysis exception terminates the run."""
    inp = _build_input(tmp_path)

    real_apply = Pipeline._apply_standard_analysis
    calls = {"n": 0}

    def _boom(self, chunk, i):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected analysis failure")
        return real_apply(self, chunk, i)

    monkeypatch.setattr(Pipeline, "_apply_standard_analysis", _boom)
    with pytest.raises(Exception):
        _run(tmp_path, _config(tmp_path), inp)
    assert not (tmp_path / "out_phasic" / INPUT_COMPLETENESS_FILENAME).exists()


# 8 / 10 / 11. Authorized exclusion vs. non-final / multiple ------------------


def test_authorized_incomplete_final_chunk_excluded(tmp_path: Path):
    # Final chunk malformed, but explicitly excluded by the incomplete-final
    # policy: the two valid chunks process and the run succeeds.
    inp = _build_input(tmp_path, malformed=(2,))
    final_source = str(inp / CHUNK_NAMES[2] / "fluorescence.csv")
    cfg = _config(tmp_path, rwd_excluded_source_files=[final_source])

    out = _run(tmp_path, cfg, inp)
    record = _record(out)
    assert validate_input_completeness(record) == ""
    assert record["expected"][-1]["disposition"] == DISPOSITION_AUTHORIZED_EXCLUSION
    assert len(record["processed"]) == 2


def test_excluding_a_non_final_chunk_fails(tmp_path: Path):
    inp = _build_input(tmp_path)
    middle_source = str(inp / CHUNK_NAMES[1] / "fluorescence.csv")
    cfg = _config(tmp_path, rwd_excluded_source_files=[middle_source])

    with pytest.raises(ValueError, match="final chronological chunk"):
        _run(tmp_path, cfg, inp)


def test_two_recorded_exclusions_fail(tmp_path: Path):
    inp = _build_input(tmp_path)
    cfg = _config(
        tmp_path,
        rwd_excluded_source_files=[
            str(inp / CHUNK_NAMES[1] / "fluorescence.csv"),
            str(inp / CHUNK_NAMES[2] / "fluorescence.csv"),
        ],
    )
    with pytest.raises(ValueError):
        _run(tmp_path, cfg, inp)


# 3 / 4. Source drift after the admitted set is frozen ------------------------


def test_source_disappearing_after_freeze_fails(tmp_path: Path, monkeypatch):
    inp = _build_input(tmp_path)
    import photometry_pipeline.pipeline as pipeline_mod

    real_load = pipeline_mod.load_chunk
    state = {"deleted": False}

    def _delete_then_load(path, fmt, config, **kwargs):
        # After the admitted manifest is frozen, remove a still-pending chunk.
        if not state["deleted"]:
            state["deleted"] = True
            (inp / CHUNK_NAMES[2] / "fluorescence.csv").unlink()
        return real_load(path, fmt, config, **kwargs)

    monkeypatch.setattr(pipeline_mod, "load_chunk", _delete_then_load)
    with pytest.raises(InputProcessingError) as excinfo:
        _run(tmp_path, _config(tmp_path), inp)
    assert excinfo.value.category == "source_drift"


def test_source_changing_size_after_freeze_fails(tmp_path: Path, monkeypatch):
    inp = _build_input(tmp_path)
    import photometry_pipeline.pipeline as pipeline_mod

    real_load = pipeline_mod.load_chunk
    state = {"changed": False}

    def _resize_then_load(path, fmt, config, **kwargs):
        if not state["changed"]:
            state["changed"] = True
            target = inp / CHUNK_NAMES[2] / "fluorescence.csv"
            target.write_text(target.read_text(encoding="utf-8") + "\n0,0,0\n", encoding="utf-8")
        return real_load(path, fmt, config, **kwargs)

    monkeypatch.setattr(pipeline_mod, "load_chunk", _resize_then_load)
    with pytest.raises(InputProcessingError) as excinfo:
        _run(tmp_path, _config(tmp_path), inp)
    assert excinfo.value.category == "source_drift"


# Tonic mode is bound too ------------------------------------------------------


def test_tonic_mode_writes_completeness_record(tmp_path: Path):
    out = _run(tmp_path, _config(tmp_path), _build_input(tmp_path), mode="tonic")
    record = _record(out)
    assert validate_input_completeness(record) == ""
    assert len(record["processed"]) == 3
