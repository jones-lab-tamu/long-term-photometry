#!/usr/bin/env python3
"""Run the explicit applied_dff production chain for one ROI and strategy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_applied_dff_features import run_applied_dff_features  # noqa: E402
from tools.verify_applied_dff_cache import verify_applied_dff_cache  # noqa: E402
from tools.verify_applied_dff_feature_outputs import verify_applied_dff_feature_outputs  # noqa: E402
from tools.write_applied_dff_cache import write_applied_dff_cache  # noqa: E402

TOOL_NAME = "run_applied_dff_pipeline"
SUPPORTED_STRATEGIES = {"dynamic_fit", "signal_only_f0"}
PIPELINE_CONTRACT = "explicit-strategy verified-cache separate-output only"

SUMMARY_FIELDS = [
    "pipeline_passed",
    "failed_stage",
    "roi",
    "strategy",
    "phasic_out",
    "output_root",
    "applied_output_dir",
    "feature_output_dir",
    "pipeline_output_dir",
    "feature_config_path",
    "feature_config_hash",
    "overwrite",
    "fail_on_warning",
    "created_at_utc",
    "stage_write_applied_cache_passed",
    "stage_verify_applied_cache_passed",
    "stage_run_applied_features_passed",
    "stage_verify_feature_outputs_passed",
    "applied_trace_source",
    "applied_trace_complete",
    "n_chunks",
    "n_chunks_processed",
    "n_features",
    "feature_output_granularity",
    "one_feature_row_per_chunk",
    "one_feature_row_per_chunk_matches_detector",
    "semantic_status",
    "source_phasic_cache_sha256_before",
    "source_phasic_cache_sha256_after",
    "applied_trace_cache_sha256",
    "features_csv",
    "feature_summary_json",
    "feature_output_semantic_verification_json",
    "hdf5_modified_source_phasic_cache",
    "legacy_features_modified",
    "failure_messages",
]


class AppliedDffPipelineError(RuntimeError):
    """Raised when the explicit applied_dff pipeline cannot complete."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report or {}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        out = {}
        for key in SUMMARY_FIELDS:
            value = row.get(key, "")
            if isinstance(value, bool):
                value = str(value).lower()
            out[key] = value
        writer.writerow(out)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _legacy_features_hash(phasic_out: Path) -> str:
    path = phasic_out / "features" / "features.csv"
    return _file_sha256(path) if path.exists() else ""


def _feature_config_hash(path: Path | None) -> str:
    return _file_sha256(path) if path is not None and path.exists() else ""


def _roi_exists(cache_path: Path, roi: str) -> bool:
    try:
        with h5py.File(cache_path, "r") as h5:
            return f"roi/{roi}" in h5
    except Exception:
        return False


def _validate_roi_path_component(roi: str) -> None:
    text = str(roi or "").strip()
    if not text:
        raise AppliedDffPipelineError("ROI is empty or unsafe for output path")
    if "/" in text or "\\" in text:
        raise AppliedDffPipelineError(f"ROI is unsafe for output path: {roi}")
    if ".." in text:
        raise AppliedDffPipelineError(f"ROI is unsafe for output path: {roi}")
    if Path(text).is_absolute() or Path(text).drive:
        raise AppliedDffPipelineError(f"ROI is unsafe for output path: {roi}")


def _assert_inside_output_root(base: Path, output_root: Path) -> None:
    root_resolved = output_root.resolve()
    base_resolved = base.resolve()
    try:
        base_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise AppliedDffPipelineError(
            f"pipeline output path is outside output_root: {base_resolved}"
        ) from exc
    if base_resolved == root_resolved:
        raise AppliedDffPipelineError("pipeline output path must be strictly inside output_root")


def _paths(output_root: Path, roi: str, strategy: str, output_name: str | None = None) -> dict[str, Path]:
    base = output_root / (output_name if output_name is not None else f"{roi}_{strategy}")
    return {
        "base": base,
        "applied": base / "applied",
        "features": base / "features",
        "pipeline": base / "pipeline",
    }


def _empty_summary(
    *,
    phasic_out: Path,
    roi: str,
    strategy: str,
    output_root: Path,
    paths: dict[str, Path],
    feature_config: Path | None,
    overwrite: bool,
    fail_on_warning: bool,
    source_hash_before: str,
) -> dict[str, Any]:
    return {
        "pipeline_passed": False,
        "failed_stage": "",
        "roi": roi,
        "strategy": strategy,
        "phasic_out": str(phasic_out),
        "output_root": str(output_root),
        "applied_output_dir": str(paths["applied"]),
        "feature_output_dir": str(paths["features"]),
        "pipeline_output_dir": str(paths["pipeline"]),
        "feature_config_path": str(feature_config) if feature_config is not None else "",
        "feature_config_hash": _feature_config_hash(feature_config),
        "overwrite": bool(overwrite),
        "fail_on_warning": bool(fail_on_warning),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_write_applied_cache_passed": False,
        "stage_verify_applied_cache_passed": False,
        "stage_run_applied_features_passed": False,
        "stage_verify_feature_outputs_passed": False,
        "applied_trace_source": "",
        "applied_trace_complete": False,
        "n_chunks": 0,
        "n_chunks_processed": 0,
        "n_features": 0,
        "feature_output_granularity": "",
        "one_feature_row_per_chunk": False,
        "one_feature_row_per_chunk_matches_detector": False,
        "semantic_status": "",
        "source_phasic_cache_sha256_before": source_hash_before,
        "source_phasic_cache_sha256_after": "",
        "applied_trace_cache_sha256": "",
        "features_csv": "",
        "feature_summary_json": "",
        "feature_output_semantic_verification_json": str(paths["features"] / "feature_output_semantic_verification.json"),
        "hdf5_modified_source_phasic_cache": False,
        "legacy_features_modified": False,
        "failure_messages": "",
    }


def _write_pipeline_outputs(paths: dict[str, Path], summary: dict[str, Any], provenance: dict[str, Any]) -> None:
    _write_json(paths["pipeline"] / "applied_dff_pipeline_summary.json", summary)
    _write_csv(paths["pipeline"] / "applied_dff_pipeline_summary.csv", summary)
    _write_json(paths["pipeline"] / "applied_dff_pipeline_provenance.json", provenance)


def _fail(
    *,
    stage: str,
    exc: Exception,
    summary: dict[str, Any],
    provenance: dict[str, Any],
    paths: dict[str, Path],
    phasic_out: Path,
    source_cache: Path,
    legacy_hash_before: str,
) -> None:
    source_hash_after = _file_sha256(source_cache) if source_cache.exists() else ""
    legacy_hash_after = _legacy_features_hash(phasic_out)
    summary.update(
        {
            "failed_stage": stage,
            "source_phasic_cache_sha256_after": source_hash_after,
            "hdf5_modified_source_phasic_cache": bool(summary.get("source_phasic_cache_sha256_before") != source_hash_after),
            "legacy_features_modified": bool(legacy_hash_before != legacy_hash_after),
            "failure_messages": str(exc),
        }
    )
    provenance.update(
        {
            "source_phasic_cache_sha256_after": source_hash_after,
            "hdf5_modified_source_phasic_cache": summary["hdf5_modified_source_phasic_cache"],
            "legacy_features_modified": summary["legacy_features_modified"],
        }
    )
    _write_pipeline_outputs(paths, summary, provenance)
    raise AppliedDffPipelineError(str(exc), report=summary) from exc


def run_applied_dff_pipeline(
    phasic_out: str | os.PathLike[str],
    *,
    roi: str,
    strategy: str,
    output_root: str | os.PathLike[str],
    output_name: str | None = None,
    feature_config: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    fail_on_warning: bool = False,
) -> dict[str, Any]:
    strategy = str(strategy or "").strip()
    if strategy not in SUPPORTED_STRATEGIES:
        raise AppliedDffPipelineError(f"unsupported applied_dff pipeline strategy: {strategy}")
    _validate_roi_path_component(roi)
    phasic_path = Path(phasic_out).resolve()
    output_root_path = Path(output_root).resolve()
    feature_config_path = Path(feature_config).resolve() if feature_config is not None else None
    paths = (
        _paths(output_root_path, roi, strategy)
        if output_name is None
        else _paths(output_root_path, roi, strategy, output_name=output_name)
    )
    source_cache = phasic_path / "phasic_trace_cache.h5"

    if dry_run:
        return {
            "dry_run": True,
            "phasic_out": str(phasic_path),
            "roi": roi,
            "strategy": strategy,
            "source_phasic_cache_exists": source_cache.exists(),
            "roi_exists": _roi_exists(source_cache, roi) if source_cache.exists() else False,
            "would_write_applied_cache": True,
            "would_verify_applied_cache": True,
            "would_run_applied_features": True,
            "would_verify_feature_outputs": True,
            "applied_output_dir": str(paths["applied"]),
            "feature_output_dir": str(paths["features"]),
            "pipeline_output_dir": str(paths["pipeline"]),
        }

    if not phasic_path.exists():
        raise AppliedDffPipelineError(f"phasic_out does not exist: {phasic_path}")
    if not source_cache.exists():
        raise AppliedDffPipelineError(f"source phasic cache missing: {source_cache}")
    if not _roi_exists(source_cache, roi):
        raise AppliedDffPipelineError(f"requested ROI not found in source cache: {roi}")
    if paths["base"].exists():
        if not overwrite:
            raise AppliedDffPipelineError(f"pipeline output already exists, refusing without --overwrite: {paths['base']}")
        _assert_inside_output_root(paths["base"], output_root_path)
        shutil.rmtree(paths["base"])
    paths["pipeline"].mkdir(parents=True, exist_ok=True)

    source_hash_before = _file_sha256(source_cache)
    legacy_hash_before = _legacy_features_hash(phasic_path)
    summary = _empty_summary(
        phasic_out=phasic_path,
        roi=roi,
        strategy=strategy,
        output_root=output_root_path,
        paths=paths,
        feature_config=feature_config_path,
        overwrite=overwrite,
        fail_on_warning=fail_on_warning,
        source_hash_before=source_hash_before,
    )
    provenance: dict[str, Any] = {
        "tool_name": TOOL_NAME,
        "created_at_utc": summary["created_at_utc"],
        "phasic_out": str(phasic_path),
        "roi": roi,
        "strategy": strategy,
        "output_root": str(output_root_path),
        "applied_output_dir": str(paths["applied"]),
        "feature_output_dir": str(paths["features"]),
        "pipeline_output_dir": str(paths["pipeline"]),
        "stage_reports": {},
        "stage_output_paths": {},
        "source_phasic_cache_sha256_before": source_hash_before,
        "source_phasic_cache_sha256_after": "",
        "hdf5_modified_source_phasic_cache": False,
        "legacy_features_modified": False,
        "pipeline_contract": PIPELINE_CONTRACT,
    }

    try:
        write_report = write_applied_dff_cache(
            phasic_path,
            roi=roi,
            requested_correction_strategy=strategy,
            output_dir=paths["applied"],
            overwrite=overwrite,
        )
        summary["stage_write_applied_cache_passed"] = True
        provenance["stage_reports"]["write_applied_cache"] = write_report.get("summary", {})
        provenance["stage_output_paths"]["write_applied_cache"] = write_report
    except Exception as exc:
        _fail(stage="write_applied_cache", exc=exc, summary=summary, provenance=provenance, paths=paths, phasic_out=phasic_path, source_cache=source_cache, legacy_hash_before=legacy_hash_before)

    try:
        verify_report = verify_applied_dff_cache(
            phasic_path,
            roi=roi,
            strategy=strategy,
            applied_output_dir=paths["applied"],
            fail_on_warning=fail_on_warning,
        )
        summary["stage_verify_applied_cache_passed"] = True
        summary["applied_trace_source"] = verify_report.get("applied_trace_source", "")
        summary["applied_trace_complete"] = bool(verify_report.get("applied_trace_complete"))
        summary["n_chunks"] = int(verify_report.get("n_chunks") or 0)
        provenance["stage_reports"]["verify_applied_dff_cache"] = verify_report
    except Exception as exc:
        _fail(stage="verify_applied_dff_cache", exc=exc, summary=summary, provenance=provenance, paths=paths, phasic_out=phasic_path, source_cache=source_cache, legacy_hash_before=legacy_hash_before)

    try:
        feature_report = run_applied_dff_features(
            phasic_path,
            roi=roi,
            strategy=strategy,
            applied_output_dir=paths["applied"],
            output_dir=paths["features"],
            feature_config=feature_config_path,
            overwrite=overwrite,
            skip_verification=False,
        )
        summary["stage_run_applied_features_passed"] = True
        feature_summary = feature_report.get("summary", {})
        summary["feature_config_path"] = feature_summary.get("feature_config_path", summary["feature_config_path"])
        summary["feature_config_hash"] = feature_summary.get("feature_config_hash", summary["feature_config_hash"])
        if not summary["n_chunks"]:
            summary["n_chunks"] = int(feature_summary.get("n_chunks") or 0)
        summary["n_chunks_processed"] = int(feature_summary.get("n_chunks_processed") or 0)
        summary["n_features"] = int(feature_summary.get("n_features") or 0)
        summary["features_csv"] = feature_report.get("features_csv", "")
        summary["feature_summary_json"] = feature_report.get("feature_summary_json", "")
        provenance["stage_reports"]["run_applied_dff_features"] = feature_summary
        provenance["stage_output_paths"]["run_applied_dff_features"] = feature_report
    except Exception as exc:
        _fail(stage="run_applied_dff_features", exc=exc, summary=summary, provenance=provenance, paths=paths, phasic_out=phasic_path, source_cache=source_cache, legacy_hash_before=legacy_hash_before)

    try:
        semantic_report = verify_applied_dff_feature_outputs(
            phasic_path,
            roi=roi,
            strategy=strategy,
            applied_output_dir=paths["applied"],
            feature_output_dir=paths["features"],
            write_summary=paths["features"] / "feature_output_semantic_verification.json",
            fail_on_warning=fail_on_warning,
        )
        summary["stage_verify_feature_outputs_passed"] = True
        summary["feature_output_granularity"] = semantic_report.get("feature_output_granularity", "")
        summary["one_feature_row_per_chunk"] = bool(semantic_report.get("one_feature_row_per_chunk"))
        summary["one_feature_row_per_chunk_matches_detector"] = bool(semantic_report.get("one_feature_row_per_chunk_matches_detector"))
        summary["semantic_status"] = semantic_report.get("semantic_status", "")
        provenance["stage_reports"]["verify_applied_dff_feature_outputs"] = semantic_report
    except Exception as exc:
        _fail(stage="verify_applied_dff_feature_outputs", exc=exc, summary=summary, provenance=provenance, paths=paths, phasic_out=phasic_path, source_cache=source_cache, legacy_hash_before=legacy_hash_before)

    source_hash_after = _file_sha256(source_cache)
    legacy_hash_after = _legacy_features_hash(phasic_path)
    applied_summary_path = paths["applied"] / "applied_correction_summary.json"
    if applied_summary_path.exists():
        applied_summary = json.loads(applied_summary_path.read_text(encoding="utf-8"))
        summary["applied_trace_cache_sha256"] = applied_summary.get("applied_trace_cache_sha256", "")
        if not summary["n_chunks"]:
            summary["n_chunks"] = int(applied_summary.get("n_chunks") or 0)
    summary["source_phasic_cache_sha256_after"] = source_hash_after
    summary["hdf5_modified_source_phasic_cache"] = bool(source_hash_before != source_hash_after)
    summary["legacy_features_modified"] = bool(legacy_hash_before != legacy_hash_after)
    summary["pipeline_passed"] = bool(
        summary["stage_write_applied_cache_passed"]
        and summary["stage_verify_applied_cache_passed"]
        and summary["stage_run_applied_features_passed"]
        and summary["stage_verify_feature_outputs_passed"]
        and not summary["hdf5_modified_source_phasic_cache"]
        and not summary["legacy_features_modified"]
    )
    if summary["hdf5_modified_source_phasic_cache"]:
        summary["failed_stage"] = "read_only_guard"
        summary["failure_messages"] = "source phasic cache changed during pipeline"
    if summary["legacy_features_modified"]:
        summary["failed_stage"] = "read_only_guard"
        summary["failure_messages"] = "legacy features changed during pipeline"
    provenance.update(
        {
            "source_phasic_cache_sha256_after": source_hash_after,
            "hdf5_modified_source_phasic_cache": summary["hdf5_modified_source_phasic_cache"],
            "legacy_features_modified": summary["legacy_features_modified"],
        }
    )
    _write_pipeline_outputs(paths, summary, provenance)
    if not summary["pipeline_passed"]:
        raise AppliedDffPipelineError(summary.get("failure_messages") or "pipeline failed", report=summary)
    return {
        "dry_run": False,
        "summary": summary,
        "provenance": provenance,
        "summary_json": str(paths["pipeline"] / "applied_dff_pipeline_summary.json"),
        "summary_csv": str(paths["pipeline"] / "applied_dff_pipeline_summary.csv"),
        "provenance_json": str(paths["pipeline"] / "applied_dff_pipeline_provenance.json"),
    }


def _print_report(report: dict[str, Any]) -> None:
    if report.get("dry_run"):
        for key, value in report.items():
            print(f"{key}: {value}")
        return
    summary = report.get("summary", report)
    for key in (
        "pipeline_passed",
        "failed_stage",
        "roi",
        "strategy",
        "applied_trace_source",
        "applied_trace_complete",
        "n_chunks_processed",
        "n_features",
        "semantic_status",
        "feature_output_granularity",
        "one_feature_row_per_chunk",
        "one_feature_row_per_chunk_matches_detector",
        "hdf5_modified_source_phasic_cache",
        "legacy_features_modified",
        "failure_messages",
    ):
        print(f"{key}: {summary.get(key)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the explicit applied_dff production chain.")
    parser.add_argument("--phasic-out", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = run_applied_dff_pipeline(
            args.phasic_out,
            roi=args.roi,
            strategy=args.strategy,
            output_root=args.output_root,
            feature_config=args.feature_config,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            fail_on_warning=bool(args.fail_on_warning),
        )
    except AppliedDffPipelineError as exc:
        if exc.report:
            _print_report(exc.report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
