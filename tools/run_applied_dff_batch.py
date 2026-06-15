#!/usr/bin/env python3
"""Run explicit applied_dff pipelines from a user-provided ROI/strategy manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_applied_dff_pipeline import AppliedDffPipelineError, run_applied_dff_pipeline  # noqa: E402

TOOL_NAME = "run_applied_dff_batch"
ORCHESTRATOR_TOOL = "tools/run_applied_dff_pipeline.py"
SUPPORTED_STRATEGIES = {"dynamic_fit", "signal_only_f0"}
FORBIDDEN_STRATEGY_FIELDS = {
    "recommended_strategy",
    "chosen_strategy",
    "selected_strategy",
    "best_strategy",
}

ROW_FIELDS = [
    "row_index",
    "roi",
    "strategy",
    "output_name",
    "output_dir",
    "status",
    "pipeline_passed",
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
    "error_message",
    "pipeline_summary_json",
    "pipeline_provenance_json",
]


class AppliedDffBatchError(RuntimeError):
    """Raised when an explicit applied_dff batch cannot complete."""

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


def _contains_path(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _legacy_features_path(phasic_out: Path) -> Path:
    return phasic_out / "features" / "features.csv"


def _legacy_features_hash(phasic_out: Path) -> str:
    path = _legacy_features_path(phasic_out)
    return _file_sha256(path) if path.exists() else ""


def _assert_safe_output_root(output_root: Path, phasic_out: Path, source_cache: Path) -> None:
    output_resolved = output_root.resolve()
    phasic_resolved = phasic_out.resolve()
    legacy_features_dir = phasic_resolved / "features"
    legacy_features = legacy_features_dir / "features.csv"
    if output_resolved == phasic_resolved:
        raise AppliedDffBatchError("unsafe output_root equals phasic_out")
    if output_resolved == legacy_features_dir.resolve():
        raise AppliedDffBatchError("unsafe output_root equals legacy features directory")
    if _contains_path(output_resolved, source_cache):
        raise AppliedDffBatchError("unsafe output_root contains phasic_trace_cache.h5")
    if legacy_features.exists() and _contains_path(output_resolved, legacy_features):
        raise AppliedDffBatchError("unsafe output_root contains legacy features.csv")


def _validate_path_component(value: str, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or text == "." or text == "..":
        raise AppliedDffBatchError(f"unsafe {field_name}: {value!r}")
    if "/" in text or "\\" in text:
        raise AppliedDffBatchError(f"unsafe {field_name}: {value!r}")
    if ".." in text:
        raise AppliedDffBatchError(f"unsafe {field_name}: {value!r}")
    if Path(text).is_absolute() or Path(text).drive or re.match(r"^[A-Za-z]:", text):
        raise AppliedDffBatchError(f"unsafe {field_name}: {value!r}")
    return text


def _default_output_name(roi: str, strategy: str) -> str:
    return f"{roi}_{strategy}"


def _parse_csv_manifest(path: Path, output_root: Path, default_feature_config: Path | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise AppliedDffBatchError("manifest is empty")
        missing = {"roi", "strategy"} - set(reader.fieldnames)
        if missing:
            raise AppliedDffBatchError(f"manifest missing required column(s): {', '.join(sorted(missing))}")
        rows = list(reader)

    parsed: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_output_names: set[str] = set()
    for idx, raw in enumerate(rows, start=1):
        roi = str(raw.get("roi") or "").strip()
        strategy = str(raw.get("strategy") or "").strip()
        if not roi:
            raise AppliedDffBatchError(f"manifest row {idx} has empty roi")
        if strategy not in SUPPORTED_STRATEGIES:
            raise AppliedDffBatchError(f"manifest row {idx} has unsupported strategy: {strategy}")
        pair = (roi, strategy)
        if pair in seen_pairs:
            raise AppliedDffBatchError(f"duplicate ROI/strategy row is not supported: {roi} {strategy}")
        seen_pairs.add(pair)
        output_raw = raw.get("output_name")
        output_name = _validate_path_component(
            output_raw if output_raw is not None else _default_output_name(roi, strategy),
            field_name="output_name",
        )
        if output_name in seen_output_names:
            raise AppliedDffBatchError(f"duplicate output_name is not supported: {output_name}")
        seen_output_names.add(output_name)
        row_feature_config = raw.get("feature_config")
        feature_config = (
            Path(row_feature_config).resolve()
            if row_feature_config is not None and str(row_feature_config).strip()
            else default_feature_config
        )
        parsed.append(
            {
                "row_index": idx,
                "roi": roi,
                "strategy": strategy,
                "output_name": output_name,
                "output_dir": str((output_root / output_name).resolve()),
                "feature_config": str(feature_config) if feature_config is not None else "",
            }
        )
    return parsed


def _load_manifest(path: Path, output_root: Path, default_feature_config: Path | None) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".csv":
        raise AppliedDffBatchError("only CSV manifests are supported")
    rows = _parse_csv_manifest(path, output_root, default_feature_config)
    if not rows:
        raise AppliedDffBatchError("manifest contains no rows")
    return rows


def _empty_row(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": int(plan["row_index"]),
        "roi": plan["roi"],
        "strategy": plan["strategy"],
        "output_name": plan["output_name"],
        "output_dir": plan["output_dir"],
        "status": "pending",
        "pipeline_passed": False,
        "applied_trace_source": "",
        "applied_trace_complete": False,
        "n_chunks_processed": 0,
        "n_features": 0,
        "semantic_status": "",
        "feature_output_granularity": "",
        "one_feature_row_per_chunk": False,
        "one_feature_row_per_chunk_matches_detector": False,
        "hdf5_modified_source_phasic_cache": False,
        "legacy_features_modified": False,
        "error_message": "",
        "pipeline_summary_json": "",
        "pipeline_provenance_json": "",
    }


def _row_from_pipeline(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    summary = dict(report.get("summary") or {})
    row = _empty_row(plan)
    row.update(
        {
            "status": "completed" if bool(summary.get("pipeline_passed")) else "failed",
            "pipeline_passed": bool(summary.get("pipeline_passed")),
            "applied_trace_source": summary.get("applied_trace_source", ""),
            "applied_trace_complete": bool(summary.get("applied_trace_complete")),
            "n_chunks_processed": int(summary.get("n_chunks_processed") or 0),
            "n_features": int(summary.get("n_features") or 0),
            "semantic_status": summary.get("semantic_status", ""),
            "feature_output_granularity": summary.get("feature_output_granularity", ""),
            "one_feature_row_per_chunk": bool(summary.get("one_feature_row_per_chunk")),
            "one_feature_row_per_chunk_matches_detector": bool(summary.get("one_feature_row_per_chunk_matches_detector")),
            "hdf5_modified_source_phasic_cache": bool(summary.get("hdf5_modified_source_phasic_cache")),
            "legacy_features_modified": bool(summary.get("legacy_features_modified")),
            "error_message": summary.get("failure_messages", ""),
            "pipeline_summary_json": report.get("summary_json", ""),
            "pipeline_provenance_json": report.get("provenance_json", ""),
        }
    )
    return row


def _row_from_error(plan: dict[str, Any], exc: Exception) -> dict[str, Any]:
    row = _empty_row(plan)
    row.update({"status": "failed", "error_message": str(exc)})
    if isinstance(exc, AppliedDffPipelineError) and exc.report:
        row["pipeline_passed"] = bool(exc.report.get("pipeline_passed"))
        row["applied_trace_source"] = exc.report.get("applied_trace_source", "")
        row["applied_trace_complete"] = bool(exc.report.get("applied_trace_complete"))
        row["n_chunks_processed"] = int(exc.report.get("n_chunks_processed") or 0)
        row["n_features"] = int(exc.report.get("n_features") or 0)
        row["semantic_status"] = exc.report.get("semantic_status", "")
        row["feature_output_granularity"] = exc.report.get("feature_output_granularity", "")
        row["one_feature_row_per_chunk"] = bool(exc.report.get("one_feature_row_per_chunk"))
        row["one_feature_row_per_chunk_matches_detector"] = bool(exc.report.get("one_feature_row_per_chunk_matches_detector"))
        row["hdf5_modified_source_phasic_cache"] = bool(exc.report.get("hdf5_modified_source_phasic_cache"))
        row["legacy_features_modified"] = bool(exc.report.get("legacy_features_modified"))
    return row


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in ROW_FIELDS:
                value = row.get(key, "")
                if isinstance(value, bool):
                    value = str(value).lower()
                out[key] = value
            writer.writerow(out)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _batch_summary(
    *,
    phasic_out: Path,
    manifest: Path,
    output_root: Path,
    rows: list[dict[str, Any]],
    n_manifest_rows: int,
    continue_on_error: bool,
    source_hash_before: str,
    source_hash_after: str,
    legacy_hash_before: str,
    legacy_hash_after: str,
) -> dict[str, Any]:
    n_failed = sum(1 for row in rows if row.get("status") == "failed")
    n_completed = sum(1 for row in rows if row.get("status") == "completed")
    hdf5_modified = bool(source_hash_before and source_hash_after and source_hash_before != source_hash_after)
    legacy_modified = bool(legacy_hash_before != legacy_hash_after)
    return {
        "batch_passed": bool(n_failed == 0 and n_completed == n_manifest_rows and not hdf5_modified and not legacy_modified),
        "tool_name": TOOL_NAME,
        "phasic_out": str(phasic_out),
        "manifest": str(manifest),
        "output_root": str(output_root),
        "n_manifest_rows": int(n_manifest_rows),
        "n_rows_completed": int(n_completed),
        "n_rows_failed": int(n_failed),
        "n_rows_skipped": int(n_manifest_rows - len(rows)),
        "continue_on_error": bool(continue_on_error),
        "no_auto_selection": True,
        "no_strategy_chosen": True,
        "no_inference": True,
        "no_gui": True,
        "no_global_routing": True,
        "hdf5_modified_source_phasic_cache": hdf5_modified,
        "legacy_features_modified": legacy_modified,
        "rows": rows,
    }


def _batch_provenance(
    *,
    phasic_out: Path,
    manifest: Path,
    output_root: Path,
    manifest_hash: str,
    source_hash_before: str,
    source_hash_after: str,
    legacy_hash_before: str,
    legacy_hash_after: str,
) -> dict[str, Any]:
    return {
        "tool_name": TOOL_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phasic_out": str(phasic_out),
        "manifest": str(manifest),
        "output_root": str(output_root),
        "manifest_sha256": manifest_hash,
        "source_phasic_cache_sha256_before": source_hash_before,
        "source_phasic_cache_sha256_after": source_hash_after,
        "legacy_features_sha256_before": legacy_hash_before,
        "legacy_features_sha256_after": legacy_hash_after,
        "hdf5_modified_source_phasic_cache": bool(source_hash_before and source_hash_after and source_hash_before != source_hash_after),
        "legacy_features_modified": bool(legacy_hash_before != legacy_hash_after),
        "no_auto_selection": True,
        "no_strategy_chosen": True,
        "no_strategy_inference": True,
        "orchestrator_tool": ORCHESTRATOR_TOOL,
    }


def _write_batch_outputs(output_root: Path, summary: dict[str, Any], provenance: dict[str, Any]) -> dict[str, str]:
    batch_dir = output_root / "batch"
    summary_csv = batch_dir / "applied_dff_batch_summary.csv"
    summary_json = batch_dir / "applied_dff_batch_summary.json"
    provenance_json = batch_dir / "applied_dff_batch_provenance.json"
    _write_rows_csv(summary_csv, summary["rows"])
    _write_json(summary_json, summary)
    _write_json(provenance_json, provenance)
    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "provenance_json": str(provenance_json),
    }


def run_applied_dff_batch(
    phasic_out: str | os.PathLike[str],
    *,
    manifest: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    overwrite: bool = False,
    dry_run: bool = False,
    continue_on_error: bool = False,
    feature_config: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    phasic_path = Path(phasic_out).resolve()
    manifest_path = Path(manifest).resolve()
    output_root_path = Path(output_root).resolve()
    feature_config_path = Path(feature_config).resolve() if feature_config is not None else None
    source_cache = phasic_path / "phasic_trace_cache.h5"

    if not manifest_path.exists():
        raise AppliedDffBatchError(f"manifest does not exist: {manifest_path}")
    _assert_safe_output_root(output_root_path, phasic_path, source_cache)
    plans = _load_manifest(manifest_path, output_root_path, feature_config_path)

    if dry_run:
        planned_rows = [{**plan, "would_run": True} for plan in plans]
        return {
            "dry_run": True,
            "phasic_out": str(phasic_path),
            "manifest": str(manifest_path),
            "output_root": str(output_root_path),
            "n_manifest_rows": len(plans),
            "planned_rows": planned_rows,
            "no_auto_selection": True,
            "no_strategy_chosen": True,
            "no_inference": True,
        }

    if not phasic_path.exists():
        raise AppliedDffBatchError(f"phasic_out does not exist: {phasic_path}")
    if not source_cache.exists():
        raise AppliedDffBatchError(f"source phasic cache missing: {source_cache}")

    manifest_hash = _file_sha256(manifest_path)
    source_hash_before = _file_sha256(source_cache)
    legacy_hash_before = _legacy_features_hash(phasic_path)
    completed_rows: list[dict[str, Any]] = []
    first_error: Exception | None = None

    for plan in plans:
        try:
            report = run_applied_dff_pipeline(
                phasic_path,
                roi=plan["roi"],
                strategy=plan["strategy"],
                output_root=output_root_path,
                output_name=plan["output_name"],
                feature_config=plan["feature_config"] or None,
                overwrite=overwrite,
            )
            completed_rows.append(_row_from_pipeline(plan, report))
        except Exception as exc:
            completed_rows.append(_row_from_error(plan, exc))
            if first_error is None:
                first_error = exc
            if not continue_on_error:
                break

    source_hash_after = _file_sha256(source_cache) if source_cache.exists() else ""
    legacy_hash_after = _legacy_features_hash(phasic_path)
    summary = _batch_summary(
        phasic_out=phasic_path,
        manifest=manifest_path,
        output_root=output_root_path,
        rows=completed_rows,
        n_manifest_rows=len(plans),
        continue_on_error=continue_on_error,
        source_hash_before=source_hash_before,
        source_hash_after=source_hash_after,
        legacy_hash_before=legacy_hash_before,
        legacy_hash_after=legacy_hash_after,
    )
    provenance = _batch_provenance(
        phasic_out=phasic_path,
        manifest=manifest_path,
        output_root=output_root_path,
        manifest_hash=manifest_hash,
        source_hash_before=source_hash_before,
        source_hash_after=source_hash_after,
        legacy_hash_before=legacy_hash_before,
        legacy_hash_after=legacy_hash_after,
    )
    paths = _write_batch_outputs(output_root_path, summary, provenance)
    report = {"dry_run": False, "summary": summary, "provenance": provenance, **paths}
    if not summary["batch_passed"]:
        message = str(first_error) if first_error is not None else "batch failed"
        if summary["hdf5_modified_source_phasic_cache"]:
            message = "source phasic cache changed during batch"
        if summary["legacy_features_modified"]:
            message = "legacy features changed during batch"
        raise AppliedDffBatchError(message, report=report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    if report.get("dry_run"):
        print(f"dry_run: true")
        print(f"n_manifest_rows: {report.get('n_manifest_rows')}")
        for row in report.get("planned_rows", []):
            print(f"planned_row: {row['row_index']} {row['roi']} {row['strategy']} {row['output_dir']}")
        return
    summary = report.get("summary", {})
    print(f"batch_passed: {str(summary.get('batch_passed')).lower()}")
    print(f"n_manifest_rows: {summary.get('n_manifest_rows')}")
    print(f"n_rows_completed: {summary.get('n_rows_completed')}")
    print(f"n_rows_failed: {summary.get('n_rows_failed')}")
    print(f"n_rows_skipped: {summary.get('n_rows_skipped')}")
    print(f"hdf5_modified_source_phasic_cache: {str(summary.get('hdf5_modified_source_phasic_cache')).lower()}")
    print(f"legacy_features_modified: {str(summary.get('legacy_features_modified')).lower()}")
    if report.get("summary_json"):
        print(f"summary_json: {report['summary_json']}")
    if report.get("provenance_json"):
        print(f"provenance_json: {report['provenance_json']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run explicit applied_dff pipelines from a manifest.")
    parser.add_argument("--phasic-out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--feature-config", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = run_applied_dff_batch(
            args.phasic_out,
            manifest=args.manifest,
            output_root=args.output_root,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            continue_on_error=bool(args.continue_on_error),
            feature_config=args.feature_config,
        )
    except AppliedDffBatchError as exc:
        if exc.report:
            _print_report(exc.report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
