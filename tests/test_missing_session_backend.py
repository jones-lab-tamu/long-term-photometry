"""Approved missing sessions stay in their time slot as explicit gaps (4J16k41c).

Drives the real Pipeline in-process over multi-chunk RWD. A scientist-approved
corrupted session is never removed from the time axis: it keeps its chronological
index and validated timestamp, contributes NaN (never zero) session summaries,
and is distinct from a valid zero-event session. Later sessions never shift.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.input_processing_completeness import (
    DISPOSITION_AUTHORIZED_EXCLUSION,
    DISPOSITION_AUTHORIZED_MISSING,
    DISPOSITION_PROCESS,
    INPUT_COMPLETENESS_FILENAME,
    InputProcessingError,
    validate_input_completeness,
    build_session_index,
)

NAMES = [
    "2024_01_01-00_00_00",
    "2024_01_01-01_00_00",
    "2024_01_01-02_00_00",
    "2024_01_01-03_00_00",
]


def _write_valid(chunk_dir: Path, *, seed: int, flat: bool = False, n: int = 600, fs: float = 10.0):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    if flat:
        uv = np.ones(n)
        sig = np.ones(n) * 1.2  # no events -> zero-event valid session
    else:
        uv = 1.0 + 0.03 * rng.normal(0, 1, n)
        sig = 1.2 * uv + 0.1 * rng.normal(0, 1, n)
    pd.DataFrame({"TimeStamp": t, "Region0-470": sig, "Region0-410": uv}).to_csv(
        chunk_dir / "fluorescence.csv", index=False
    )


def _write_corrupted(chunk_dir: Path):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "fluorescence.csv").write_text("garbage,not,valid\n1,2,3\n", encoding="utf-8")


def _build_input(tmp_path: Path, *, corrupted=(), flat=(), n_sessions=3) -> Path:
    inp = tmp_path / "input"
    for i in range(n_sessions):
        d = inp / NAMES[i]
        if i in corrupted:
            _write_corrupted(d)
        else:
            _write_valid(d, seed=i, flat=(i in flat))
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
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _run(tmp_path: Path, cfg: Config, inp: Path, *, mode: str = "phasic") -> Path:
    out = tmp_path / f"out_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    Pipeline(cfg, mode=mode).run(str(inp), str(out), force_format="rwd", recursive=True)
    return out


def _record(out: Path) -> dict:
    return json.loads((out / INPUT_COMPLETENESS_FILENAME).read_text(encoding="utf-8"))


def _source(inp: Path, i: int) -> str:
    return str(inp / NAMES[i] / "fluorescence.csv")


# Session index: missing session keeps index + timestamp --------------------


def test_missing_middle_session_keeps_index_and_timestamp(tmp_path: Path):
    inp = _build_input(tmp_path, corrupted=(1,))
    cfg = _config(tmp_path, authorized_missing_sessions=[_source(inp, 1)])
    out = _run(tmp_path, cfg, inp)

    record = _record(out)
    assert validate_input_completeness(record) == ""
    by_index = {e["index"]: e for e in record["expected"]}

    assert by_index[1]["disposition"] == DISPOSITION_AUTHORIZED_MISSING
    assert by_index[1]["expected_start_time"] == "2024-01-01T01:00:00"
    assert by_index[1]["expected_duration_sec"] == 60.0
    assert by_index[1]["failure_category"]
    assert by_index[1]["authorization_source"]

    # The next session keeps its original timestamp AND session number.
    assert by_index[2]["disposition"] == DISPOSITION_PROCESS
    assert by_index[2]["expected_start_time"] == "2024-01-01T02:00:00"
    assert by_index[2]["index"] == 2


def test_cache_ids_preserve_original_session_numbers_across_gap(tmp_path: Path):
    inp = _build_input(tmp_path, corrupted=(1,))
    cfg = _config(tmp_path, authorized_missing_sessions=[_source(inp, 1)])
    out = _run(tmp_path, cfg, inp)

    record = _record(out)
    processed = {p["index"]: p["cache_chunk_id"] for p in record["processed"]}
    # Session 2 processed into cache slot 1: the storage id must not be read as
    # the original session number, leaving the approved gap visible.
    assert processed == {0: 0, 2: 2}


# features.csv: NaN not zero, distinct from zero-event --------------------------


def test_missing_row_is_nan_and_distinct_from_zero_event(tmp_path: Path):
    # Session 0 valid-with-events, session 1 approved missing, session 2 flat
    # (valid zero-event).
    inp = _build_input(tmp_path, corrupted=(1,), flat=(2,))
    cfg = _config(tmp_path, authorized_missing_sessions=[_source(inp, 1)])
    out = _run(tmp_path, cfg, inp)

    feats = pd.read_csv(out / "features" / "features.csv")
    assert "status" in feats.columns and "session_index" in feats.columns

    by_session = {int(r.session_index): r for r in feats.itertuples()}

    # Zero-event valid session: status valid, peak_count 0 (a real number).
    assert by_session[2].status == "valid"
    assert int(by_session[2].peak_count) == 0

    # Missing session: status missing_corrupted, peak_count NaN (never zero),
    # no cache/storage chunk_id.
    assert by_session[1].status == "missing_corrupted"
    assert math.isnan(by_session[1].peak_count)
    assert math.isnan(by_session[1].auc)
    assert math.isnan(by_session[1].chunk_id)


def test_no_event_level_rows_are_fabricated(tmp_path: Path):
    inp = _build_input(tmp_path, corrupted=(1,))
    cfg = _config(tmp_path, authorized_missing_sessions=[_source(inp, 1)])
    out = _run(tmp_path, cfg, inp)
    # The production run has no event-level table; none is created for the gap.
    assert not list((out).glob("**/*events*.csv"))


# Authorization gating ---------------------------------------------------------


def test_unapproved_corrupted_session_still_fails(tmp_path: Path):
    inp = _build_input(tmp_path, corrupted=(1,))
    cfg = _config(tmp_path)  # no authorization
    with pytest.raises(Exception):
        _run(tmp_path, cfg, inp)
    assert not (tmp_path / "out_phasic" / INPUT_COMPLETENESS_FILENAME).exists()


def test_approving_a_session_not_discovered_fails(tmp_path: Path):
    inp = _build_input(tmp_path, corrupted=(1,))
    cfg = _config(
        tmp_path, authorized_missing_sessions=[str(inp / "nope" / "fluorescence.csv")]
    )
    with pytest.raises(ValueError, match="not found among the discovered"):
        _run(tmp_path, cfg, inp)


def test_missing_session_without_validated_timestamp_cannot_be_authorized(tmp_path: Path):
    # A corrupted session whose folder has no datetime token cannot be placed.
    inp = tmp_path / "input"
    (inp / "sessionA").mkdir(parents=True)
    _write_valid(inp / "2024_01_01-00_00_00", seed=0)
    _write_corrupted(inp / "sessionA")
    cfg = _config(tmp_path, authorized_missing_sessions=[str(inp / "sessionA" / "fluorescence.csv")])
    with pytest.raises(InputProcessingError) as excinfo:
        _run(tmp_path, cfg, inp)
    assert excinfo.value.category == "unresolvable_missing_session_time"


@pytest.mark.parametrize("input_format", ["npm", "custom_tabular"])
def test_generic_filename_datetime_cannot_authorize_missing_session(tmp_path: Path, input_format: str):
    """Non-RWD filename tokens are not yet an authoritative timing contract."""
    source = tmp_path / "session_2024-01-01_01_00_00.csv"
    source.write_text("time,signal\n0,1\n", encoding="utf-8")
    with pytest.raises(InputProcessingError) as excinfo:
        build_session_index(
            acquisition_mode="intermittent",
            input_format=input_format,
            ordered_sources=[str(source)],
            missing_sources=[str(source)],
            expected_duration_sec=60.0,
        )
    assert excinfo.value.category == "unsupported_missing_session_timing"


# Phasic and tonic share the same session index --------------------------------


def test_phasic_and_tonic_share_the_same_session_index(tmp_path: Path):
    inp = _build_input(tmp_path, corrupted=(1,))
    cfg = _config(tmp_path, authorized_missing_sessions=[_source(inp, 1)])
    phasic = _run(tmp_path, cfg, inp, mode="phasic")
    tonic = _run(tmp_path, cfg, inp, mode="tonic")

    p_rec = _record(phasic)
    t_rec = _record(tonic)
    # Same admitted set, same missing set, same identity.
    assert p_rec["frozen_manifest_digest"] == t_rec["frozen_manifest_digest"]
    assert [e["disposition"] for e in p_rec["expected"]] == [
        e["disposition"] for e in t_rec["expected"]
    ]


def test_different_missing_sets_yield_different_identity(tmp_path: Path):
    inp = _build_input(tmp_path, corrupted=(1,))
    cfg_a = _config(tmp_path, authorized_missing_sessions=[_source(inp, 1)])
    out_a = _run(tmp_path, cfg_a, inp, mode="phasic")

    inp2 = _build_input(tmp_path / "b", corrupted=())
    cfg_b = _config(tmp_path)  # no missing
    out_b = _run(tmp_path / "b", cfg_b, inp2, mode="tonic")

    assert _record(out_a)["frozen_manifest_digest"] != _record(out_b)["frozen_manifest_digest"]


# Final exclusion stays distinct from a missing middle session -----------------


def test_final_exclusion_is_distinct_from_a_missing_session(tmp_path: Path):
    inp = _build_input(tmp_path, corrupted=(1,))
    cfg = _config(
        tmp_path,
        authorized_missing_sessions=[_source(inp, 1)],
        rwd_excluded_source_files=[_source(inp, 2)],  # final chunk excluded
    )
    out = _run(tmp_path, cfg, inp)

    record = _record(out)
    by_index = {e["index"]: e for e in record["expected"]}
    assert by_index[1]["disposition"] == DISPOSITION_AUTHORIZED_MISSING
    assert by_index[2]["disposition"] == DISPOSITION_AUTHORIZED_EXCLUSION
    assert by_index[0]["disposition"] == DISPOSITION_PROCESS


# Clean runs unchanged ---------------------------------------------------------


def test_clean_run_has_no_missing_dispositions(tmp_path: Path):
    inp = _build_input(tmp_path)
    out = _run(tmp_path, _config(tmp_path), inp)

    record = _record(out)
    assert all(e["disposition"] == DISPOSITION_PROCESS for e in record["expected"])
    assert record["missing"] == []
    feats = pd.read_csv(out / "features" / "features.csv")
    assert set(feats["status"]) == {"valid"}


# Day layout preserves chronology across a gap (no backward shift) -------------

from photometry_pipeline.viz.phasic_data_prep import compute_day_layout  # noqa: E402


def test_datetime_layout_keeps_later_session_at_its_real_time(tmp_path: Path):
    # Session 1 missing: sessions 0 and 2 present with real timestamps.
    entries = [
        (0, "/x/2024_01_01-00_00_00/chunk_0.csv"),
        (2, "/x/2024_01_01-02_00_00/chunk_2.csv"),
    ]
    fmap = {
        (0, "R"): {"source_file": "/x/2024_01_01-00_00_00/fluorescence.csv"},
        (2, "R"): {"source_file": "/x/2024_01_01-02_00_00/fluorescence.csv"},
    }
    ds = compute_day_layout(entries, fmap, "R", sessions_per_hour=1)
    hours = {c.chunk_id: c.hour_idx for c in ds.chunks}
    # The surviving later session stays at hour 2 -- it does not slide into the
    # missing session's hour-1 slot.
    assert hours[0] == 0
    assert hours[2] == 2


def test_fallback_layout_anchors_by_session_id_not_row_position(tmp_path: Path):
    # No inferable datetime; ids carry the chronological session number with a gap.
    entries = [(0, "/x/a/chunk_0.csv"), (2, "/x/c/chunk_2.csv")]
    fmap = {(0, "R"): {"source_file": "a.csv"}, (2, "R"): {"source_file": "c.csv"}}
    ds = compute_day_layout(entries, fmap, "R", sessions_per_hour=1)
    hours = {c.chunk_id: c.hour_idx for c in ds.chunks}
    assert hours[0] == 0
    assert hours[2] == 2  # not 1 -- later session is not pulled backward
