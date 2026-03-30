import inspect

from photometry_pipeline.utils import timeline


def test_schedule_mapping_preserves_first_match_and_fallback(monkeypatch):
    monkeypatch.setattr(
        timeline,
        "discover_csv_or_rwd_chunks",
        lambda _: ["C:/data/session_a.csv", "C:/data/session_b.csv", "C:/data/session_a.csv"],
    )
    monkeypatch.setattr(
        timeline,
        "sort_npm_files",
        lambda files: ["C:/data/session_a.csv", "C:/data/session_b.csv", "C:/data/session_a.csv"],
    )

    actual_positions = timeline.map_cached_sources_to_schedule_positions(
        raw_input_dir="C:/data",
        fmt="npm",
        cached_source_files=[
            "c:\\data\\session_a.csv",
            "c:\\data\\session_b.csv",
            "c:\\data\\missing.csv",
            "c:\\data\\session_a.csv",
        ],
        cids=[10, 11, 12, 13],
    )

    # Duplicate source file should map to the first discovered position (legacy list.index semantics),
    # and true misses should still fall back to cids.
    assert actual_positions == [0, 1, 12, 0]


def test_schedule_mapping_uses_sorted_npm_file_order(monkeypatch):
    monkeypatch.setattr(
        timeline,
        "discover_csv_or_rwd_chunks",
        lambda _: ["C:/data/chunk_3.csv", "C:/data/chunk_1.csv", "C:/data/chunk_2.csv"],
    )
    monkeypatch.setattr(
        timeline,
        "sort_npm_files",
        lambda files: ["C:/data/chunk_1.csv", "C:/data/chunk_2.csv", "C:/data/chunk_3.csv"],
    )

    actual_positions = timeline.map_cached_sources_to_schedule_positions(
        raw_input_dir="C:/data",
        fmt="npm",
        cached_source_files=["C:/data/chunk_2.csv", "C:/data/chunk_1.csv", "C:/data/missing.csv"],
        cids=[3, 4, 5],
    )
    assert actual_positions == [1, 0, 5]


def test_schedule_mapping_hot_path_no_longer_uses_list_index():
    src = inspect.getsource(timeline.map_cached_sources_to_schedule_positions)
    assert "normalized_file_list.index(" not in src
