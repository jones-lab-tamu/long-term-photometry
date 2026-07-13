"""Authoritative RWD chronological ordering contract (A2).

Covers the shared ordering primitive (photometry_pipeline.io.rwd_chronology)
directly, plus its two production consumers: io.adapters.discover_rwd_chunks
(the execution/preview discovery path) and io.rwd_source_snapshot (the
Guided validation/preflight candidate-snapshot path). Both consumers must
agree, since a mismatch between them is exactly the "two parallel
chronology systems" risk this task closes.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from photometry_pipeline.io.rwd_chronology import (
    RwdChronologyError,
    find_rwd_session_overlaps,
    order_rwd_session_candidates,
    parse_rwd_session_folder_timestamp,
)
from photometry_pipeline.io.adapters import discover_rwd_chunks
from photometry_pipeline.io.rwd_source_snapshot import (
    build_rwd_source_candidate_snapshot,
)


# ---------------------------------------------------------------------------
# Pure ordering primitive
# ---------------------------------------------------------------------------


def test_parse_accepts_only_full_canonical_token():
    assert parse_rwd_session_folder_timestamp("2025_03_05-14_30_00") is not None
    assert parse_rwd_session_folder_timestamp("OLD_2025_03_05-14_30_00") is None
    assert parse_rwd_session_folder_timestamp("2025_03_05-14_30_00_retry") is None
    assert parse_rwd_session_folder_timestamp("") is None
    assert parse_rwd_session_folder_timestamp("not-a-timestamp") is None


def test_order_is_independent_of_input_order():
    names = ["2025_01_01-02_00_00", "2025_01_01-00_00_00", "2025_01_01-01_00_00"]
    expected = sorted(names)
    for _ in range(20):
        shuffled = names[:]
        random.shuffle(shuffled)
        ordered = order_rwd_session_candidates(shuffled, name_of=lambda n: n)
        assert ordered == expected


def test_normal_folder_ordering_matches_expected_sequence():
    names = [
        "2025_01_01-00_00_00",
        "2025_01_01-00_10_00",
        "2025_01_01-00_20_00",
    ]
    ordered = order_rwd_session_candidates(names, name_of=lambda n: n)
    assert ordered == names


def test_malformed_timestamp_fails_closed_not_lexical_fallback():
    names = ["2025_01_01-00_00_00", "not_a_timestamp"]
    with pytest.raises(RwdChronologyError) as excinfo:
        order_rwd_session_candidates(names, name_of=lambda n: n)
    assert excinfo.value.category == "malformed_session_timestamp"
    assert "not_a_timestamp" in str(excinfo.value)


def test_duplicate_timestamp_fails_closed_no_silent_tiebreak():
    names = ["2025_01_01-00_00_00", "2025_01_01-00_00_00"]
    with pytest.raises(RwdChronologyError) as excinfo:
        order_rwd_session_candidates(names, name_of=lambda n: n)
    assert excinfo.value.category == "duplicate_session_timestamp"


def test_repeated_wall_clock_time_across_distinct_names_still_flagged():
    """Two distinct session folder names that happen to parse to the exact
    same wall-clock instant must not be silently collapsed or arbitrarily
    tie-broken by name -- this is the only scientifically honest response
    given no second timing source exists for RWD."""
    names = ["2025_01_01-00_00_00", "2025_01_01-00_00_00"]
    with pytest.raises(RwdChronologyError):
        order_rwd_session_candidates(names, name_of=lambda n: n)


def test_overlap_detection_flags_adjacent_overlapping_sessions():
    from datetime import datetime

    ordered = [
        ("a", datetime(2025, 1, 1, 0, 0, 0)),
        ("b", datetime(2025, 1, 1, 0, 5, 0)),  # starts 300s later
    ]
    overlaps = find_rwd_session_overlaps(ordered, session_duration_sec=600.0)
    assert overlaps == [("a", "b")]


def test_overlap_detection_allows_back_to_back_sessions():
    from datetime import datetime

    ordered = [
        ("a", datetime(2025, 1, 1, 0, 0, 0)),
        ("b", datetime(2025, 1, 1, 0, 10, 0)),  # starts exactly at a's end
    ]
    overlaps = find_rwd_session_overlaps(ordered, session_duration_sec=600.0)
    assert overlaps == []


# ---------------------------------------------------------------------------
# discover_rwd_chunks (production execution / preview discovery path)
# ---------------------------------------------------------------------------


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("TimeStamp,Region0-410,Region0-470\n0,1.0,2.0\n", encoding="utf-8")


def test_discover_rwd_chunks_ignores_filesystem_enumeration_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Sessions returned in random filesystem order still produce the same
    authoritative chronological sequence."""
    names = ["2025_02_01-09_00_00", "2025_02_01-08_00_00", "2025_02_01-10_00_00"]
    for name in names:
        _touch(tmp_path / name / "fluorescence.csv")

    import photometry_pipeline.io.adapters as adapters_module

    real_scandir = os.scandir

    class _ShuffledScandir:
        def __init__(self, path):
            self._entries = list(real_scandir(path))
            random.shuffle(self._entries)

        def __enter__(self):
            return iter(self._entries)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(adapters_module.os, "scandir", _ShuffledScandir)
    chunks = discover_rwd_chunks(str(tmp_path))
    got = [os.path.basename(os.path.dirname(p)) for p in chunks]
    assert got == sorted(names)


def test_discover_rwd_chunks_malformed_folder_name_fails_closed(tmp_path: Path):
    _touch(tmp_path / "2025_02_01-08_00_00" / "fluorescence.csv")
    _touch(tmp_path / "renamed_backup" / "fluorescence.csv")
    with pytest.raises(ValueError) as excinfo:
        discover_rwd_chunks(str(tmp_path))
    assert "renamed_backup" in str(excinfo.value)
    assert "does not match the expected recording-time format" in str(excinfo.value)


def test_discover_rwd_chunks_duplicate_timestamp_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Two distinct source-tree entries that resolve to the identical
    acquisition timestamp must refuse rather than pick one arbitrarily.

    Two real directories can never share one filesystem name, so this
    injects two distinct synthetic scandir entries with the same
    (duplicate) canonical name to exercise the collision path
    discover_rwd_chunks shares with real production discovery.
    """
    real_dir = tmp_path / "2025_02_01-08_00_00"
    _touch(real_dir / "fluorescence.csv")

    class _FakeEntry:
        def __init__(self, name: str, path: str):
            self.name = name
            self.path = path

        def is_dir(self):
            return True

    import photometry_pipeline.io.adapters as adapters_module

    def fake_scandir(_path):
        return [
            _FakeEntry("2025_02_01-08_00_00", str(real_dir)),
            _FakeEntry("2025_02_01-08_00_00", str(real_dir)),
        ]

    class _CM:
        def __enter__(self):
            return fake_scandir(str(tmp_path))

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(adapters_module.os, "scandir", lambda _p: _CM())
    with pytest.raises(ValueError) as excinfo:
        discover_rwd_chunks(str(tmp_path))
    assert "same start time" in str(excinfo.value)


# ---------------------------------------------------------------------------
# rwd_source_snapshot (Guided validation / preflight candidate-snapshot path)
# ---------------------------------------------------------------------------


def test_snapshot_and_discovery_agree_on_order(tmp_path: Path):
    """The order frozen by Guided validation's candidate snapshot and the
    order Pipeline execution discovers must be provably identical -- both
    consume the same authoritative chronology rule."""
    names = ["2025_03_01-11_00_00", "2025_03_01-09_00_00", "2025_03_01-10_00_00"]
    for name in names:
        _touch(tmp_path / name / "fluorescence.csv")

    chunks = discover_rwd_chunks(str(tmp_path))
    discovery_order = [os.path.basename(os.path.dirname(p)) for p in chunks]

    snapshot = build_rwd_source_candidate_snapshot(str(tmp_path))
    snapshot_order = [
        c.canonical_relative_path.split("/", 1)[0] for c in snapshot.candidates
    ]

    assert discovery_order == sorted(names)
    assert snapshot_order == sorted(names)
    assert discovery_order == snapshot_order


def test_snapshot_identity_changes_when_order_changes(tmp_path: Path):
    """A source whose chronological order changes (a session's timestamp
    moves) must change the frozen candidate-set identity, not silently
    keep validation bound to a stale order."""
    _touch(tmp_path / "2025_03_01-09_00_00" / "fluorescence.csv")
    _touch(tmp_path / "2025_03_01-10_00_00" / "fluorescence.csv")
    first = build_rwd_source_candidate_snapshot(str(tmp_path))

    # Move the earlier session later in time (a real re-ordering of
    # chronology, not merely a filename edit).
    (tmp_path / "2025_03_01-09_00_00").rename(tmp_path / "2025_03_01-11_00_00")
    second = build_rwd_source_candidate_snapshot(str(tmp_path))

    assert first.source_candidate_set_digest != second.source_candidate_set_digest
    first_order = [c.canonical_relative_path for c in first.candidates]
    second_order = [c.canonical_relative_path for c in second.candidates]
    assert first_order != second_order


# ---------------------------------------------------------------------------
# Real production-boundary lifecycle: Pipeline.run() with shuffled discovery
# ---------------------------------------------------------------------------


def _write_valid_rwd_session(chunk_dir: Path, *, seed: int, n: int = 300, fs: float = 10.0) -> None:
    import numpy as np
    import pandas as pd

    chunk_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    uv = 1.0 + 0.05 * np.sin(t * 0.3) + rng.normal(0, 0.02, n)
    sig = 1.2 * uv + 0.1 * rng.normal(0, 1, n)
    pd.DataFrame({"TimeStamp": t, "Region0-470": sig, "Region0-410": uv}).to_csv(
        chunk_dir / "fluorescence.csv", index=False
    )


def test_real_pipeline_run_consumes_authoritative_order_under_shuffled_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Crosses the real production execution boundary (Pipeline.run, not a
    helper): with filesystem enumeration order shuffled, the run's
    persisted chronological session index must still reflect true
    acquisition order, and the run must complete identically to an
    unshuffled run (deterministic output, not incidental to OS order)."""
    from photometry_pipeline.config import Config
    from photometry_pipeline.pipeline import Pipeline
    from photometry_pipeline.input_processing_completeness import (
        INPUT_COMPLETENESS_FILENAME,
    )
    import json
    import photometry_pipeline.io.adapters as adapters_module

    names = ["2025_04_01-00_00_00", "2025_04_01-00_10_00", "2025_04_01-00_20_00"]

    def _build_input(root: Path) -> None:
        for i, name in enumerate(names):
            _write_valid_rwd_session(root / name, seed=i)

    def _config() -> Config:
        cfg = Config()
        cfg.chunk_duration_sec = 30
        cfg.target_fs_hz = 10
        cfg.baseline_method = "uv_raw_percentile_session"
        cfg.baseline_percentile = 10
        cfg.rwd_time_col = "TimeStamp"
        cfg.uv_suffix = "-410"
        cfg.sig_suffix = "-470"
        return cfg

    # Baseline: normal (unshuffled) filesystem enumeration.
    normal_input = tmp_path / "normal_input"
    _build_input(normal_input)
    normal_out = tmp_path / "normal_out"
    Pipeline(_config(), mode="phasic").run(
        str(normal_input), str(normal_out), force_format="rwd", recursive=True
    )
    normal_record = json.loads(
        (normal_out / INPUT_COMPLETENESS_FILENAME).read_text(encoding="utf-8")
    )
    normal_order = [entry["source"] for entry in normal_record["expected"]]
    normal_names = [os.path.basename(os.path.dirname(p)) for p in normal_order]
    assert normal_names == names

    # Shuffled: filesystem enumeration order reversed at discovery time.
    real_scandir = os.scandir

    class _ReversedScandir:
        def __init__(self, path):
            self._entries = list(reversed(list(real_scandir(path))))

        def __enter__(self):
            return iter(self._entries)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(adapters_module.os, "scandir", _ReversedScandir)

    shuffled_input = tmp_path / "shuffled_input"
    _build_input(shuffled_input)
    shuffled_out = tmp_path / "shuffled_out"
    Pipeline(_config(), mode="phasic").run(
        str(shuffled_input), str(shuffled_out), force_format="rwd", recursive=True
    )
    shuffled_record = json.loads(
        (shuffled_out / INPUT_COMPLETENESS_FILENAME).read_text(encoding="utf-8")
    )
    shuffled_order = [entry["source"] for entry in shuffled_record["expected"]]
    shuffled_names = [os.path.basename(os.path.dirname(p)) for p in shuffled_order]

    # The persisted chronological order is identical regardless of the
    # filesystem enumeration order discovery happened to see.
    assert shuffled_names == names == normal_names
