"""
Tests for RunReportViewer and run_report_parser dynamic discovery.
"""

import os
import json
import tempfile
import pytest
from gui.run_report_parser import (
    resolve_region_deliverables, 
    resolve_internal_artifacts,
    resolve_primary_artifacts
)

def test_discover_region_deliverables_dynamic():
    """Verify that the parser finds arbitrary region folders with semantic subfolders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create Region A with 3 subfolders
        reg_a = os.path.join(tmpdir, "RegionA")
        os.makedirs(os.path.join(reg_a, "summary"))
        os.makedirs(os.path.join(reg_a, "day_plots"))
        os.makedirs(os.path.join(reg_a, "tables"))
        
        # Create Region B with 2 subfolders
        reg_b = os.path.join(tmpdir, "RegionB")
        os.makedirs(os.path.join(reg_b, "summary"))
        os.makedirs(os.path.join(reg_b, "tables"))
        
        # Create a non-region folder
        os.makedirs(os.path.join(tmpdir, "some_other_folder"))
        
        # Create an internal folder (should be skipped by region discovery)
        os.makedirs(os.path.join(tmpdir, "_analysis"))
        
        regions = resolve_region_deliverables(tmpdir)
        
        assert len(regions) == 2
        names = [r['name'] for r in regions]
        assert "RegionA" in names
        assert "RegionB" in names
        
        reg_a_data = next(r for r in regions if r['name'] == "RegionA")
        assert len(reg_a_data['subfolders']) == 3
        
        reg_b_data = next(r for r in regions if r['name'] == "RegionB")
        assert len(reg_b_data['subfolders']) == 2
        labels_b = [f[0] for f in reg_b_data['subfolders']]
        assert "Summary" in labels_b
        assert "Tables" in labels_b
        assert "Day Plots" not in labels_b

def test_internal_analysis_links_optional():
    """Verify that _analysis subfolders are discovered."""
    with tempfile.TemporaryDirectory() as tmpdir:
        analysis_dir = os.path.join(tmpdir, "_analysis")
        os.makedirs(os.path.join(analysis_dir, "phasic_out"))
        
        internal = resolve_internal_artifacts(tmpdir)
        assert len(internal) == 1
        assert internal[0][0] == "Phasic Analysis (Internal)"
        
        os.makedirs(os.path.join(analysis_dir, "tonic_out"))
        internal = resolve_internal_artifacts(tmpdir)
        assert len(internal) == 2

def test_no_obsolete_primary_quick_links():
    """Ensure we don't return hardcoded traces/features/etc anymore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an old-style traces folder
        os.makedirs(os.path.join(tmpdir, "traces"))
        
        # The new primary artifacts resolver should NOT find it
        primary = resolve_primary_artifacts(tmpdir, {})
        labels = [p[0] for p in primary]
        assert "Traces Folder" not in labels
        
        # And it shouldn't show up in regions if it doesn't have the subfolders
        regions = resolve_region_deliverables(tmpdir)
        assert len(regions) == 0

def test_primary_artifacts_resolver():
    """Verify root-level artifacts are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "status.json"), 'w') as f: f.write('{}')
        with open(os.path.join(tmpdir, "MANIFEST.json"), 'w') as f: f.write('{}')
        
        primary = resolve_primary_artifacts(tmpdir, {})
        assert len(primary) == 2
        labels = [p[0] for p in primary]
        assert "Run Status" in labels
        assert "Output Manifest" in labels
