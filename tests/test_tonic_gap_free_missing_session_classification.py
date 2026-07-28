"""Gap-free tonic time is blocked only by a compressible missing interval.

An explicitly excluded incomplete final recording session ends the recording
earlier; it never collapses an interval between two retained sessions, so it
must not force the real elapsed-time tonic view.  A missing or excluded session
that still has retained recording data after it does compress a real interval
and must keep blocking gap-free elapsed time.

These checks drive the real Pipeline and then the real ``plot_tonic_48h.py``
command the deliverables wrapper issues.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from photometry_pipeline.input_processing_completeness import (
    DISPOSITION_AUTHORIZED_EXCLUSION,
    DISPOSITION_AUTHORIZED_MISSING,
    DISPOSITION_PROCESS,
)
from photometry_pipeline.viz.phasic_data_prep import (
    last_retained_plot_session_index,
    missing_sessions_followed_by_retained_data,
)
from tests.test_missing_session_backend import (
    NAMES,
    _build_input,
    _config,
    _record,
    _run,
    _source,
)

GAP_FREE = "gap_free_elapsed_time"
REAL_ELAPSED = "real_elapsed_time"


# ======================================================================
# Classification authority
# ======================================================================


def _session(index: int, status: str, *, cache_chunk_id=None) -> dict:
    return {
        "session_index": index,
        "status": status,
        "cache_chunk_id": cache_chunk_id,
    }


def _valid(index: int) -> dict:
    return _session(index, "valid", cache_chunk_id=index)


def _missing(index: int) -> dict:
    return _session(index, "missing_corrupted")


def _final_exclusion(index: int) -> dict:
    return _session(index, "authorized_final_exclusion")


def _blocking_indices(sessions) -> list:
    return [
        int(item["session_index"])
        for item in missing_sessions_followed_by_retained_data(sessions)
    ]


def test_no_missing_sessions_never_blocks():
    sessions = [_valid(0), _valid(1), _valid(2)]
    assert _blocking_indices(sessions) == []
    assert last_retained_plot_session_index(sessions) == 2


def test_excluded_incomplete_final_session_only_never_blocks():
    sessions = [_valid(0), _valid(1), _valid(2), _final_exclusion(3)]
    assert _blocking_indices(sessions) == []


def test_terminal_exclusion_with_no_retained_session_after_it_never_blocks():
    # The excluded fragment is the highest chronological index, so nothing
    # retained follows it even though the recording is now shorter.
    sessions = [_valid(0), _final_exclusion(1)]
    assert _blocking_indices(sessions) == []
    assert last_retained_plot_session_index(sessions) == 0


def test_internal_missing_session_blocks():
    sessions = [_valid(0), _missing(1), _valid(2), _valid(3)]
    assert _blocking_indices(sessions) == [1]


def test_internal_missing_session_plus_terminal_exclusion_blocks_only_the_internal_one():
    sessions = [_valid(0), _missing(1), _valid(2), _final_exclusion(3)]
    assert _blocking_indices(sessions) == [1]


def test_multiple_internal_missing_sessions_all_block():
    sessions = [_valid(0), _missing(1), _valid(2), _missing(3), _valid(4)]
    assert _blocking_indices(sessions) == [1, 3]


def test_missing_first_session_followed_by_retained_data_blocks():
    sessions = [_missing(0), _valid(1), _valid(2)]
    assert _blocking_indices(sessions) == [0]


def test_no_retained_session_at_all_yields_no_classification():
    sessions = [_missing(0), _final_exclusion(1)]
    assert last_retained_plot_session_index(sessions) is None
    assert _blocking_indices(sessions) == []


def test_untagged_missing_records_keep_the_protective_behavior():
    from tools.plot_tonic_48h import sessions_blocking_gap_free_timeline

    # A caller that did not go through ``assemble_arrays`` cannot prove a slot
    # is terminal, so the guard must not be relaxed for it.
    assert sessions_blocking_gap_free_timeline([{"session_index": 1}]) == [
        {"session_index": 1}
    ]
    assert sessions_blocking_gap_free_timeline([]) == []
    assert sessions_blocking_gap_free_timeline(None) == []


# ======================================================================
# Natural path: the real plot_tonic_48h.py command
# ======================================================================


def _plot_tonic(out: Path, target: Path, timeline_mode: str, *, sessions_per_hour="1"):
    cmd = [
        sys.executable,
        "tools/plot_tonic_48h.py",
        "--analysis-out",
        str(out),
        "--roi",
        "Region0",
        "--out",
        str(target),
        "--tonic-output-mode",
        "flatten_session_bleach_preserve_session_baseline",
        "--tonic-timeline-mode",
        timeline_mode,
        "--format",
        "rwd",
        "--sessions-per-hour",
        sessions_per_hour,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def _terminal_exclusion_run(tmp_path: Path):
    """Valid retained sessions plus one explicitly excluded final session."""
    inp = _build_input(tmp_path, n_sessions=4)
    # The final session is a short incomplete fragment the user excluded.
    from tests.test_missing_session_backend import _write_valid

    _write_valid(inp / NAMES[3], seed=3, n=120)
    cfg = _config(tmp_path, rwd_excluded_source_files=[_source(inp, 3)])
    return inp, _run(tmp_path, cfg, inp, mode="tonic")


def _internal_missing_run(tmp_path: Path):
    inp = _build_input(tmp_path, corrupted=(1,), n_sessions=3)
    cfg = _config(tmp_path, authorized_missing_sessions=[_source(inp, 1)])
    return inp, _run(tmp_path, cfg, inp, mode="tonic")


def test_excluded_final_session_is_recorded_as_authorized_exclusion(tmp_path: Path):
    _inp, out = _terminal_exclusion_run(tmp_path)
    record = _record(out)
    by_index = {int(e["index"]): e for e in record["expected"]}
    assert [by_index[i]["disposition"] for i in sorted(by_index)] == [
        DISPOSITION_PROCESS,
        DISPOSITION_PROCESS,
        DISPOSITION_PROCESS,
        DISPOSITION_AUTHORIZED_EXCLUSION,
    ]


def test_gap_free_tonic_publication_succeeds_for_terminal_exclusion(tmp_path: Path):
    _inp, out = _terminal_exclusion_run(tmp_path)
    target = tmp_path / "tonic_overview.png"
    result = _plot_tonic(out, target, GAP_FREE)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Approved missing sessions require" not in combined
    assert "would compress" not in combined
    assert target.is_file()


def test_real_elapsed_tonic_publication_also_succeeds_for_terminal_exclusion(
    tmp_path: Path,
):
    _inp, out = _terminal_exclusion_run(tmp_path)
    target = tmp_path / "tonic_overview_real.png"
    result = _plot_tonic(out, target, REAL_ELAPSED)
    assert result.returncode == 0, result.stdout + result.stderr
    assert target.is_file()


def test_gap_free_tonic_publication_is_rejected_for_internal_missing_session(
    tmp_path: Path,
):
    _inp, out = _internal_missing_run(tmp_path)
    record = _record(out)
    by_index = {int(e["index"]): e for e in record["expected"]}
    assert by_index[1]["disposition"] == DISPOSITION_AUTHORIZED_MISSING

    target = tmp_path / "tonic_overview_gap_free.png"
    result = _plot_tonic(out, target, GAP_FREE)
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    # Scientist-facing, and explicit that retained data follow the gap.
    assert "Recording session(s) 2" in combined
    assert "missing or excluded" in combined
    assert "later recording sessions were still analyzed" in combined
    assert "real elapsed-time tonic view" in combined
    assert not target.exists()


def test_real_elapsed_tonic_publication_succeeds_for_internal_missing_session(
    tmp_path: Path,
):
    _inp, out = _internal_missing_run(tmp_path)
    target = tmp_path / "tonic_overview_real_gap.png"
    result = _plot_tonic(out, target, REAL_ELAPSED)
    assert result.returncode == 0, result.stdout + result.stderr
    assert target.is_file()


def test_assemble_arrays_tags_terminal_exclusion_as_not_compressible(tmp_path: Path):
    from types import SimpleNamespace

    from photometry_pipeline.io.hdf5_cache_reader import open_tonic_cache
    from tools.plot_tonic_48h import (
        MISSING_SESSION_COMPRESSIBLE_KEY,
        assemble_arrays,
        sessions_blocking_gap_free_timeline,
    )

    _inp, out = _terminal_exclusion_run(tmp_path)
    cache = open_tonic_cache(str(out / "tonic_trace_cache.h5"))
    try:
        args = SimpleNamespace(
            analysis_out=str(out),
            input=None,
            format="auto",
            sessions_per_hour=1,
            tonic_output_mode="preserve_raw_session_shape",
            tonic_timeline_mode=GAP_FREE,
            include_visual_separators=False,
        )
        _t, _s, _u, _d, missing = assemble_arrays(
            cache, "Region0", args, return_missing_metadata=True
        )
    finally:
        cache.close()

    # The excluded slot is still reported to the marker code unchanged; only the
    # gap-free compatibility classification treats it as non-compressible.
    assert [item["status"] for item in missing] == ["authorized_final_exclusion"]
    assert missing[0][MISSING_SESSION_COMPRESSIBLE_KEY] is False
    assert sessions_blocking_gap_free_timeline(missing) == []
