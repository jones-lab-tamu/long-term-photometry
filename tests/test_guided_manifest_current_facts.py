from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_manifest_current_facts as current_facts
from photometry_pipeline.config import Config
from photometry_pipeline.guided_execution_preflight import (
    compute_guided_strict_roi_inventory_digest,
)


def _write_session(root: Path, name: str, rois=("ROI0", "ROI1")) -> Path:
    path = root / name / "fluorescence.csv"
    path.parent.mkdir(parents=True)
    columns = ["Time(s)"]
    row = ["0"]
    for roi in rois:
        columns.extend((f"{roi}-410", f"{roi}-470"))
        row.extend(("1", "2"))
    path.write_text(",".join(columns) + "\n" + ",".join(row) + "\n")
    return path


def _facts(tmp_path, included=("ROI0",)):
    root = tmp_path / "source"
    _write_session(root, "session_a")
    _write_session(root, "session_b")
    return root, current_facts.build_guided_manifest_current_facts(
        source_root=root,
        config=Config(),
        manifest_included_roi_ids=included,
    )


def test_current_facts_builds_ordered_candidates_and_roi_inventory(tmp_path):
    root, facts = _facts(tmp_path)
    assert tuple(item.canonical_relative_path for item in facts.current_candidates) == (
        "session_a/fluorescence.csv",
        "session_b/fluorescence.csv",
    )
    assert all(Path(item.absolute_path).is_file() for item in facts.current_candidates)
    inventory = facts.current_roi_inventory
    assert inventory.discovered_roi_ids == ("ROI0", "ROI1")
    assert inventory.included_roi_ids == ("ROI0",)
    assert inventory.excluded_roi_ids == ("ROI1",)
    assert len(inventory.parser_contract_digest) == 64


def test_strict_roi_digest_uses_first_subset_include_mode(tmp_path):
    _, facts = _facts(tmp_path)
    from photometry_pipeline.io.rwd_source_snapshot import (
        build_rwd_source_candidate_snapshot,
    )

    root = Path(facts.current_candidates[0].absolute_path).parents[1]
    snapshot = build_rwd_source_candidate_snapshot(str(root))
    inventory = facts.current_roi_inventory
    expected = compute_guided_strict_roi_inventory_digest(
        source_candidate_content_digest=snapshot.source_candidate_content_digest,
        parser_contract_digest=inventory.parser_contract_digest,
        discovered_roi_ids=inventory.discovered_roi_ids,
        included_roi_ids=inventory.included_roi_ids,
        excluded_roi_ids=inventory.excluded_roi_ids,
        selection_mode="include",
    )
    assert current_facts.GUIDED_FIRST_SUBSET_SELECTION_MODE == "include"
    assert inventory.strict_roi_inventory_digest == expected


def test_current_facts_refuses_missing_included_roi(tmp_path):
    root = tmp_path / "source"
    _write_session(root, "session_a")
    with pytest.raises(ValueError, match="absent"):
        current_facts.build_guided_manifest_current_facts(
            source_root=root,
            config=Config(),
            manifest_included_roi_ids=("MISSING",),
        )


def test_current_facts_performs_no_writes(tmp_path, monkeypatch):
    root = tmp_path / "source"
    _write_session(root, "session_a")

    def fail(*_args, **_kwargs):
        raise AssertionError("write prohibited")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    facts = current_facts.build_guided_manifest_current_facts(
        source_root=root,
        config=Config(),
        manifest_included_roi_ids=("ROI0",),
    )
    assert facts.current_candidates


def test_current_facts_import_boundary_and_verifier_purity():
    source = Path(current_facts.__file__).read_text(encoding="utf-8")
    imported = _imports(source)
    prohibited = (
        "photometry_pipeline.pipeline",
        "tools.run_full_pipeline_deliverables",
        "analyze_photometry",
        "subprocess",
        "gui",
        "photometry_pipeline.guided_execution_payloads",
    )
    assert not any(
        name == marker or name.startswith(f"{marker}.")
        for name in imported
        for marker in prohibited
    )

    import photometry_pipeline.guided_manifest_verification as verification

    verifier_imports = _imports(Path(verification.__file__).read_text(encoding="utf-8"))
    verifier_prohibited = prohibited + (
        "photometry_pipeline.io.adapters",
        "photometry_pipeline.guided_manifest_current_facts",
    )
    assert not any(
        name == marker or name.startswith(f"{marker}.")
        for name in verifier_imports
        for marker in verifier_prohibited
    )


def _imports(source: str) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add(node.module or "")
    return result
