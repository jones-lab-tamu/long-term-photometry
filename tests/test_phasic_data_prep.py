"""
Tests for photometry_pipeline.viz.phasic_data_prep

Verifies the shared phasic plotting data-preparation helper introduced in Part 1.
"""

import os
import sys
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure repo root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from photometry_pipeline.viz.phasic_data_prep import (
    ChunkRecord, PhasicDataSet,
    discover_chunks, infer_datetime_from_string, infer_session_folder_name,
    parse_session_folder_datetime, infer_session_datetime,
    build_feature_map, resolve_roi, compute_day_layout,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def synth_traces_dir():
    """Create a temp directory with 96 synthetic chunk CSVs (2 days × 24h × 2 sph)."""
    tmp = tempfile.mkdtemp(prefix="test_phasic_prep_")
    n_chunks = 96
    fs = 10.0
    duration = 600.0
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs

    for i in range(n_chunks):
        df = pd.DataFrame({
            'time_sec': t,
            'Region0_sig_raw': np.random.randn(n_samples) * 100 + 1000,
            'Region0_uv_raw': np.random.randn(n_samples) * 50 + 800,
            'Region0_dff': np.random.randn(n_samples) * 0.02,
            'Region1_sig_raw': np.random.randn(n_samples) * 100 + 1100,
            'Region1_uv_raw': np.random.randn(n_samples) * 50 + 900,
            'Region1_dff': np.random.randn(n_samples) * 0.03,
        })
        df.to_csv(os.path.join(tmp, f"chunk_{i:04d}.csv"), index=False)

    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def synth_features_csv(synth_traces_dir):
    """Create a matching features.csv for the synthetic traces."""
    feat_dir = os.path.join(synth_traces_dir, '..', 'features')
    os.makedirs(feat_dir, exist_ok=True)
    feats_path = os.path.join(feat_dir, 'features.csv')

    rows = []
    for i in range(96):
        for roi in ['Region0', 'Region1']:
            rows.append({
                'chunk_id': i,
                'roi': roi,
                'peak_count': np.random.randint(0, 10),
                'auc': np.random.rand() * 0.1,
                'mean': np.random.rand() * 0.01,
                'std': np.random.rand() * 0.005,
                'source_file': f'session_{i:04d}_2025_01_{(i // 48) + 1:02d}_{(i % 48) // 2:02d}_00_00.csv',
            })
    pd.DataFrame(rows).to_csv(feats_path, index=False)
    yield feats_path
    shutil.rmtree(feat_dir, ignore_errors=True)


# ======================================================================
# Tests: discover_chunks
# ======================================================================

class TestDiscoverChunks:
    def test_ordering(self, synth_traces_dir):
        """Chunks are returned sorted by chunk_id."""
        entries = discover_chunks(synth_traces_dir)
        assert len(entries) == 96
        ids = [cid for cid, _ in entries]
        assert ids == sorted(ids)
        assert ids[0] == 0
        assert ids[-1] == 95

    def test_paths_absolute(self, synth_traces_dir):
        """All returned paths are absolute."""
        entries = discover_chunks(synth_traces_dir)
        for _, path in entries:
            assert os.path.isabs(path)

    def test_empty_dir_raises(self):
        """Raises RuntimeError for an empty directory."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RuntimeError, match="No trace files"):
                discover_chunks(tmp)


# ======================================================================
# Tests: infer_datetime_from_string
# ======================================================================

class TestInferDatetime:
    def test_hyphen_underscore(self):
        dt = infer_datetime_from_string("2025-01-15_14_30_00_session.csv")
        assert dt == datetime(2025, 1, 15, 14, 30, 0)

    def test_compact(self):
        dt = infer_datetime_from_string("20250115_143000.csv")
        assert dt == datetime(2025, 1, 15, 14, 30, 0)

    def test_none_on_failure(self):
        assert infer_datetime_from_string("no_date_here.csv") is None

    def test_none_on_non_string(self):
        assert infer_datetime_from_string(12345) is None
        assert infer_datetime_from_string(None) is None

    def test_parses_datetime_from_rwd_parent_folder_path(self):
        dt = infer_datetime_from_string(
            r"C:\data\run\2026_03_10-11_33_05\fluorescence.csv"
        )
        assert dt == datetime(2026, 3, 10, 11, 33, 5)


class TestSessionFolderInference:
    def test_session_folder_from_rwd_path(self):
        label = infer_session_folder_name(
            r"C:\data\run\2026_03_10-11_33_05\fluorescence.csv"
        )
        assert label == "2026_03_10-11_33_05"

    def test_session_folder_fallback_to_basename_stem(self):
        label = infer_session_folder_name("chunk_0007.csv")
        assert label == "chunk_0007"

    def test_parse_session_folder_datetime(self):
        dt = parse_session_folder_datetime("2026_03_10-11_33_05")
        assert dt == datetime(2026, 3, 10, 11, 33, 5)

    def test_infer_session_datetime_prefers_session_folder_over_parent_timestamp(self):
        src = (
            "C:/data/2026_03_11-01_33_05/"
            "RWD/2026_03_10-16_33_05/fluorescence.csv"
        )
        dt = infer_session_datetime(src)
        assert dt == datetime(2026, 3, 10, 16, 33, 5)


# ======================================================================
# Tests: build_feature_map
# ======================================================================

class TestBuildFeatureMap:
    def test_all_rois(self, synth_features_csv):
        fm = build_feature_map(synth_features_csv)
        # 96 chunks × 2 ROIs = 192 entries
        assert len(fm) == 192
        assert (0, 'Region0') in fm
        assert (95, 'Region1') in fm

    def test_filtered_roi(self, synth_features_csv):
        fm = build_feature_map(synth_features_csv, roi='Region0')
        assert len(fm) == 96
        assert all(roi == 'Region0' for (_, roi) in fm.keys())

    def test_missing_file(self):
        fm = build_feature_map("/nonexistent/features.csv")
        assert fm == {}


# ======================================================================
# Tests: resolve_roi
# ======================================================================

class TestResolveRoi:
    def test_explicit_roi(self, synth_traces_dir):
        """Explicit ROI is returned directly."""
        first = os.path.join(synth_traces_dir, "chunk_0000.csv")
        assert resolve_roi(first, "Region0", '_dff') == "Region0"

    def test_auto_dff(self, synth_traces_dir):
        """Auto-detects ROI from _dff columns, picks first alphabetically."""
        first = os.path.join(synth_traces_dir, "chunk_0000.csv")
        roi = resolve_roi(first, None, '_dff')
        assert roi == "Region0"  # Region0 < Region1 alphabetically

    def test_auto_sig_raw(self, synth_traces_dir):
        """Auto-detects ROI from _sig_raw columns."""
        first = os.path.join(synth_traces_dir, "chunk_0000.csv")
        roi = resolve_roi(first, None, '_sig_raw')
        assert roi == "Region0"


# ======================================================================
# Tests: compute_day_layout
# ======================================================================

class TestComputeDayLayout:
    def test_sequential_layout_96_chunks_2sph(self, synth_traces_dir):
        """96 chunks with sph=2 → 2 days, 24 hours, proper rank."""
        entries = discover_chunks(synth_traces_dir)
        pds = compute_day_layout(entries, None, "Region0", sessions_per_hour=2)

        assert pds.roi == "Region0"
        assert pds.sessions_per_hour == 2
        assert len(pds.chunks) == 96

        # 2 days
        assert sorted(pds.chunks_by_day.keys()) == [0, 1]
        assert len(pds.chunks_by_day[0]) == 48
        assert len(pds.chunks_by_day[1]) == 48

    def test_day_count_48_chunks_2sph(self, synth_traces_dir):
        """48 chunks with sph=2 → 1 day."""
        entries = discover_chunks(synth_traces_dir)[:48]
        pds = compute_day_layout(entries, None, "Region0", sessions_per_hour=2)
        assert sorted(pds.chunks_by_day.keys()) == [0]
        assert len(pds.chunks_by_day[0]) == 48

    def test_hour_idx_range(self, synth_traces_dir):
        """All hour_idx values are in [0, 23]."""
        entries = discover_chunks(synth_traces_dir)
        pds = compute_day_layout(entries, None, "Region0", sessions_per_hour=2)
        hours = set(c.hour_idx for c in pds.chunks)
        assert min(hours) >= 0
        assert max(hours) <= 23

    def test_hour_rank_range(self, synth_traces_dir):
        """All hour_rank values are in [0, sph-1]."""
        entries = discover_chunks(synth_traces_dir)
        pds = compute_day_layout(entries, None, "Region0", sessions_per_hour=2)
        ranks = set(c.hour_rank for c in pds.chunks)
        assert ranks == {0, 1}

    def test_chunk_record_fields(self, synth_traces_dir):
        """ChunkRecords have expected fields filled."""
        entries = discover_chunks(synth_traces_dir)
        pds = compute_day_layout(entries, None, "Region0", sessions_per_hour=2)
        cr = pds.chunks[0]
        assert cr.chunk_id == 0
        assert os.path.exists(cr.trace_path)
        assert isinstance(cr.day_idx, int)
        assert isinstance(cr.hour_idx, int)
        assert isinstance(cr.hour_rank, int)
        assert isinstance(cr.within_hour_offset_sec, float)

    def test_inferred_sph(self, synth_traces_dir):
        """When sph=None and no datetimes, SPH is inferred from chunk count."""
        entries = discover_chunks(synth_traces_dir)
        pds = compute_day_layout(entries, None, "Region0", sessions_per_hour=None)
        # 96 chunks → n_days_est = ceil(96/48) = 2 → sph = round(96/(24*2)) = 2
        assert pds.sessions_per_hour == 2

    def test_feature_map_passthrough(self, synth_traces_dir, synth_features_csv):
        """feature_map is stored in PhasicDataSet when provided."""
        entries = discover_chunks(synth_traces_dir)
        fm = build_feature_map(synth_features_csv, roi='Region0')
        pds = compute_day_layout(entries, fm, "Region0", sessions_per_hour=2)
        assert pds.feature_map is fm
        assert len(pds.feature_map) == 96

    def test_empty_entries_raises(self):
        """Raises RuntimeError for empty chunk list."""
        with pytest.raises(RuntimeError, match="No chunk entries"):
            compute_day_layout([], None, "Region0", sessions_per_hour=2)

    def test_datetime_layout_is_used_even_when_sessions_per_hour_is_provided(self, synth_traces_dir):
        """Timestamped source paths should drive day/hour assignment regardless of provided SPH."""
        entries = discover_chunks(synth_traces_dir)
        t0 = datetime(2026, 3, 10, 0, 0, 0)
        fm = {}
        for cid, _ in entries:
            dt = t0 + timedelta(minutes=30 * cid)
            fm[(cid, "Region0")] = {
                "source_file": f"C:/real_rwd/{dt.strftime('%Y_%m_%d-%H_%M_%S')}/fluorescence.csv"
            }

        # Deliberately mismatched SPH hint should not collapse datetime-derived day/hour mapping.
        pds = compute_day_layout(entries, fm, "Region0", sessions_per_hour=5)
        assert sorted(pds.chunks_by_day.keys()) == [0, 1]
        by_cid = {c.chunk_id: c for c in pds.chunks}
        assert by_cid[0].session_folder == "2026_03_10-00_00_00"
        assert by_cid[0].elapsed_from_start_sec == 0.0
        assert by_cid[0].hour_idx == 0
        assert by_cid[1].hour_idx == 0
        assert by_cid[95].day_idx == 1
        assert by_cid[95].hour_idx == 23
        assert by_cid[95].elapsed_from_start_sec == pytest.approx(95 * 1800.0)

    def test_offset_civil_clock_hour_slot_mapping(self, synth_traces_dir):
        """
        Mandatory placement contract:
        11:33 -> hour 11 right slot
        12:03 -> hour 12 left slot
        12:33 -> hour 12 right slot
        13:03 -> hour 13 left slot
        """
        entries = discover_chunks(synth_traces_dir)[:4]
        roots = [
            "2026_03_10-11_33_05",
            "2026_03_10-12_03_05",
            "2026_03_10-12_33_05",
            "2026_03_10-13_03_05",
        ]
        fm = {}
        for (cid, _), root in zip(entries, roots):
            fm[(cid, "Region0")] = {
                "source_file": f"C:/real/{root}/fluorescence.csv"
            }

        pds = compute_day_layout(entries, fm, "Region0", sessions_per_hour=2)
        by_cid = {c.chunk_id: c for c in pds.chunks}
        assert by_cid[0].session_folder == roots[0]
        assert by_cid[0].datetime_inferred == datetime(2026, 3, 10, 11, 33, 5)
        assert by_cid[0].hour_idx == 11
        assert by_cid[0].hour_rank == 1
        assert by_cid[1].hour_idx == 12
        assert by_cid[1].hour_rank == 0
        assert by_cid[2].hour_idx == 12
        assert by_cid[2].hour_rank == 1
        assert by_cid[3].hour_idx == 13
        assert by_cid[3].hour_rank == 0
        assert by_cid[1].elapsed_from_start_sec == pytest.approx(1800.0)
        assert by_cid[2].elapsed_from_start_sec == pytest.approx(3600.0)
        assert by_cid[3].elapsed_from_start_sec == pytest.approx(5400.0)

    def test_elapsed_anchor_hour_slot_mapping(self, synth_traces_dir):
        entries = discover_chunks(synth_traces_dir)[:4]
        roots = [
            "2026_03_10-11_33_05",
            "2026_03_10-12_03_05",
            "2026_03_10-12_33_05",
            "2026_03_10-13_03_05",
        ]
        fm = {}
        for (cid, _), root in zip(entries, roots):
            fm[(cid, "Region0")] = {"source_file": f"C:/real/{root}/fluorescence.csv"}

        pds = compute_day_layout(
            entries,
            fm,
            "Region0",
            sessions_per_hour=2,
            timeline_anchor_mode="elapsed",
        )
        by_cid = {c.chunk_id: c for c in pds.chunks}
        assert by_cid[0].hour_idx == 0
        assert by_cid[0].hour_rank == 0
        assert by_cid[1].hour_idx == 0
        assert by_cid[1].hour_rank == 1
        assert by_cid[2].hour_idx == 1
        assert by_cid[2].hour_rank == 0
        assert by_cid[3].hour_idx == 1
        assert by_cid[3].hour_rank == 1

    def test_fixed_daily_anchor_hour_slot_mapping(self, synth_traces_dir):
        entries = discover_chunks(synth_traces_dir)[:4]
        roots = [
            "2026_03_10-11_33_05",
            "2026_03_10-12_03_05",
            "2026_03_10-12_33_05",
            "2026_03_10-13_03_05",
        ]
        fm = {}
        for (cid, _), root in zip(entries, roots):
            fm[(cid, "Region0")] = {"source_file": f"C:/real/{root}/fluorescence.csv"}

        pds = compute_day_layout(
            entries,
            fm,
            "Region0",
            sessions_per_hour=2,
            timeline_anchor_mode="fixed_daily_anchor",
            fixed_daily_anchor_clock="07:00",
        )
        by_cid = {c.chunk_id: c for c in pds.chunks}
        # 11:33 -> ZT 4:33 -> hour 4, right
        assert by_cid[0].hour_idx == 4
        assert by_cid[0].hour_rank == 1
        # 12:03 -> ZT 5:03 -> hour 5, left
        assert by_cid[1].hour_idx == 5
        assert by_cid[1].hour_rank == 0
        # 12:33 -> ZT 5:33 -> hour 5, right
        assert by_cid[2].hour_idx == 5
        assert by_cid[2].hour_rank == 1
        # 13:03 -> ZT 6:03 -> hour 6, left
        assert by_cid[3].hour_idx == 6
        assert by_cid[3].hour_rank == 0

    def test_fixed_anchor_uses_session_folder_datetime_not_parent_path_timestamp(self, synth_traces_dir):
        """
        Mandatory regression:
        If parent folders include an earlier timestamp token, placement must still
        use the canonical session-folder datetime token.
        """
        entries = discover_chunks(synth_traces_dir)[:6]
        roots = [
            "2026_03_10-11_33_05",
            "2026_03_10-12_03_05",
            "2026_03_10-12_33_05",
            "2026_03_10-13_03_05",
            "2026_03_10-16_03_05",
            "2026_03_10-16_33_05",
        ]
        fm = {}
        for (cid, _), root in zip(entries, roots):
            # Deliberately inject an earlier timestamp in a parent path component.
            fm[(cid, "Region0")] = {
                "source_file": f"C:/data/2026_03_11-01_33_05/RWD/{root}/fluorescence.csv"
            }

        pds = compute_day_layout(
            entries,
            fm,
            "Region0",
            sessions_per_hour=2,
            timeline_anchor_mode="fixed_daily_anchor",
            fixed_daily_anchor_clock="07:00",
        )
        by_cid = {c.chunk_id: c for c in pds.chunks}

        # Required fixed-anchor placement contract.
        assert by_cid[0].hour_idx == 4
        assert by_cid[0].hour_rank == 1
        assert by_cid[1].hour_idx == 5
        assert by_cid[1].hour_rank == 0
        assert by_cid[2].hour_idx == 5
        assert by_cid[2].hour_rank == 1
        assert by_cid[3].hour_idx == 6
        assert by_cid[3].hour_rank == 0
        assert by_cid[4].hour_idx == 9
        assert by_cid[4].hour_rank == 0
        assert by_cid[5].hour_idx == 9
        assert by_cid[5].hour_rank == 1

    def test_fixed_daily_anchor_boundary_cases(self, synth_traces_dir):
        entries = discover_chunks(synth_traces_dir)[:3]
        roots = [
            "2026_03_10-07_03_00",
            "2026_03_10-07_33_00",
            "2026_03_11-06_50_00",
        ]
        fm = {}
        for (cid, _), root in zip(entries, roots):
            fm[(cid, "Region0")] = {"source_file": f"C:/real/{root}/fluorescence.csv"}

        pds = compute_day_layout(
            entries,
            fm,
            "Region0",
            sessions_per_hour=2,
            timeline_anchor_mode="fixed_daily_anchor",
            fixed_daily_anchor_clock="07:00",
        )
        by_cid = {c.chunk_id: c for c in pds.chunks}
        assert by_cid[0].hour_idx == 0
        assert by_cid[0].hour_rank == 0
        assert by_cid[1].hour_idx == 0
        assert by_cid[1].hour_rank == 1
        # 06:50 belongs to previous anchored day, near hour 23 right slot.
        assert by_cid[2].day_idx == 0
        assert by_cid[2].hour_idx == 23
        assert by_cid[2].hour_rank == 1

    def test_two_day_offset_start_spans_full_duration_without_truncation(self, synth_traces_dir):
        """Offset-start 96-chunk timeline preserves ~48h elapsed span and calendar day progression."""
        entries = discover_chunks(synth_traces_dir)
        t0 = datetime(2026, 3, 10, 11, 33, 5)
        fm = {}
        for cid, _ in entries:
            dt = t0 + timedelta(minutes=30 * cid)
            fm[(cid, "Region0")] = {
                "source_file": f"C:/real_rwd/{dt.strftime('%Y_%m_%d-%H_%M_%S')}/fluorescence.csv"
            }

        pds = compute_day_layout(entries, fm, "Region0", sessions_per_hour=2)
        by_cid = {c.chunk_id: c for c in pds.chunks}
        assert by_cid[95].elapsed_from_start_sec == pytest.approx(95 * 1800.0)
        assert by_cid[95].elapsed_from_start_sec / 3600.0 == pytest.approx(47.5)
        # Civil-clock anchor can span three calendar dates for a 48h run starting midday.
        assert sorted(pds.chunks_by_day.keys()) == [0, 1, 2]
        assert by_cid[0].hour_idx == 11
        assert by_cid[95].hour_idx == 11

    def test_two_day_offset_start_under_fixed_daily_anchor(self, synth_traces_dir):
        entries = discover_chunks(synth_traces_dir)
        t0 = datetime(2026, 3, 10, 11, 33, 5)
        fm = {}
        for cid, _ in entries:
            dt = t0 + timedelta(minutes=30 * cid)
            fm[(cid, "Region0")] = {
                "source_file": f"C:/real_rwd/{dt.strftime('%Y_%m_%d-%H_%M_%S')}/fluorescence.csv"
            }

        pds = compute_day_layout(
            entries,
            fm,
            "Region0",
            sessions_per_hour=2,
            timeline_anchor_mode="fixed_daily_anchor",
            fixed_daily_anchor_clock="07:00",
        )
        by_cid = {c.chunk_id: c for c in pds.chunks}
        assert by_cid[95].elapsed_from_start_sec / 3600.0 == pytest.approx(47.5)
        assert sorted(pds.chunks_by_day.keys()) == [0, 1, 2]
        # First session at 11:33 should map to anchored hour 4, right slot.
        assert by_cid[0].hour_idx == 4
        assert by_cid[0].hour_rank == 1

    def test_unparseable_sources_fall_back_to_sequential_schedule(self, synth_traces_dir):
        entries = discover_chunks(synth_traces_dir)
        fm = {}
        for cid, _ in entries:
            fm[(cid, "Region0")] = {"source_file": f"session_{cid:04d}.csv"}

        pds = compute_day_layout(entries, fm, "Region0", sessions_per_hour=2)
        by_cid = {c.chunk_id: c for c in pds.chunks}
        assert by_cid[0].datetime_inferred is None
        assert by_cid[95].day_idx == 1
        assert by_cid[95].hour_idx == 23
        assert by_cid[95].hour_rank == 1
        assert by_cid[95].elapsed_from_start_sec == pytest.approx(95 * 1800.0)

    def test_slot_placement_uses_clock_offset_not_occurrence_rank(self, synth_traces_dir):
        """
        Two sessions in the same hour and same half-hour bin should map to the same slot.
        A rank-based implementation would assign 0/1 and fail this contract.
        """
        entries = discover_chunks(synth_traces_dir)[:2]
        roots = [
            "2026_03_10-12_35_00",
            "2026_03_10-12_50_00",
        ]
        fm = {}
        for (cid, _), root in zip(entries, roots):
            fm[(cid, "Region0")] = {"source_file": f"C:/real/{root}/fluorescence.csv"}

        pds = compute_day_layout(entries, fm, "Region0", sessions_per_hour=2)
        by_cid = {c.chunk_id: c for c in pds.chunks}
        assert by_cid[0].hour_idx == 12
        assert by_cid[1].hour_idx == 12
        assert by_cid[0].hour_rank == 1
        assert by_cid[1].hour_rank == 1

    def test_invalid_fixed_anchor_clock_raises(self, synth_traces_dir):
        entries = discover_chunks(synth_traces_dir)[:1]
        fm = {(0, "Region0"): {"source_file": "C:/real/2026_03_10-11_33_05/fluorescence.csv"}}
        with pytest.raises(ValueError, match="Invalid fixed_daily_anchor clock"):
            compute_day_layout(
                entries,
                fm,
                "Region0",
                sessions_per_hour=2,
                timeline_anchor_mode="fixed_daily_anchor",
                fixed_daily_anchor_clock="25:99",
            )
