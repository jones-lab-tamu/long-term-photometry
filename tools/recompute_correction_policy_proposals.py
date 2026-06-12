#!/usr/bin/env python3
"""Recompute diagnostic correction-policy proposal fields for an existing phasic run."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photometry_pipeline.core.correction_policy_proposal import (  # noqa: E402
    MODE_BASELINE_LEGACY,
    SUPPORTED_CORRECTION_POLICIES,
    apply_correction_policy_proposals,
    normalize_policy_flags,
    policy_field_names,
    summarize_correction_policy_proposals,
)


POLICY_FLAG_PREFIX = "proposal_flags_"
FLAG_LIST_FIELDS = {
    "reference_comparison_flags",
    "dynamic_fit_qc_hard_flags",
    "dynamic_fit_qc_soft_flags",
    "dynamic_fit_qc_flags",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _json_safe(value.item())
    except Exception:
        pass
    return str(value)


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(x) for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return [dict(x) for x in data["records"] if isinstance(x, dict)]
    return []


def _load_csv_records(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    df = pd.read_csv(path)
    records = df.to_dict(orient="records")
    return [dict(x) for x in records], list(df.columns)


def _normalize_loaded_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for key, value in list(out.items()):
        if key.startswith(POLICY_FLAG_PREFIX) or key in {
            "reference_comparison_flags",
            "dynamic_fit_qc_hard_flags",
            "dynamic_fit_qc_soft_flags",
            "dynamic_fit_qc_flags",
        }:
            out[key] = normalize_policy_flags(value)
    return out


def _serialize_record_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for key, value in list(out.items()):
        if key in FLAG_LIST_FIELDS or key.startswith(POLICY_FLAG_PREFIX):
            out[key] = ";".join(normalize_policy_flags(value))
    return out


def _records_to_csv(records: list[dict[str, Any]], columns: list[str]) -> str:
    all_columns = list(columns)
    for field in policy_field_names():
        if field not in all_columns:
            all_columns.append(field)
    for rec in records:
        for key in rec:
            if key not in all_columns:
                all_columns.append(key)
    rows = [_serialize_record_for_csv(rec) for rec in records]
    return pd.DataFrame(rows, columns=all_columns).to_csv(index=False)


def _records_to_json(records: list[dict[str, Any]]) -> str:
    json_records = []
    for rec in records:
        out = dict(rec)
        for key, value in list(out.items()):
            if key.startswith(POLICY_FLAG_PREFIX):
                out[key] = normalize_policy_flags(value)
        json_records.append(_json_safe(out))
    return json.dumps(json_records, indent=2, allow_nan=False) + "\n"


def _load_qc_summary(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return {}, False
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return (data if isinstance(data, dict) else {}), True


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _backup_existing(paths: list[Path], timestamp: str) -> list[Path]:
    backups = []
    for path in paths:
        if path.exists():
            backup = path.with_name(f"{path.name}.bak_{timestamp}")
            shutil.copy2(path, backup)
            backups.append(backup)
    return backups


def recompute_policy_proposals(
    phasic_out: str | os.PathLike[str],
    *,
    backup: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    phasic_path = Path(phasic_out).resolve()
    qc_dir = phasic_path / "qc"
    csv_path = qc_dir / "baseline_reference_candidate_by_chunk.csv"
    json_path = qc_dir / "baseline_reference_candidate_by_chunk.json"
    summary_path = qc_dir / "qc_summary.json"

    warnings: list[str] = []
    source = None
    columns: list[str] = []
    if json_path.exists():
        records = _load_json_records(json_path)
        source = "json"
    elif csv_path.exists():
        records, columns = _load_csv_records(csv_path)
        source = "csv"
    else:
        raise FileNotFoundError(
            "Missing baseline_reference_candidate_by_chunk.json or "
            f"baseline_reference_candidate_by_chunk.csv under {qc_dir}"
        )

    if csv_path.exists() and not columns:
        _, columns = _load_csv_records(csv_path)
    if not records:
        warnings.append("No baseline-reference candidate records were found.")

    normalized = [_normalize_loaded_record(rec) for rec in records]
    updated = [apply_correction_policy_proposals(rec) for rec in normalized]
    proposal_summary = summarize_correction_policy_proposals(updated)
    stale_modes = [
        policy
        for policy in SUPPORTED_CORRECTION_POLICIES
        if MODE_BASELINE_LEGACY
        in proposal_summary[policy].get("proposed_correction_mode_counts", {})
    ]
    if stale_modes:
        warnings.append(
            "Policy summary still contains baseline_reference_candidate for: "
            + ", ".join(stale_modes)
        )

    qc_summary, summary_existed = _load_qc_summary(summary_path)
    if not summary_existed:
        warnings.append("qc_summary.json was missing and will be created.")
    qc_summary["correction_policy_proposal_summary"] = proposal_summary

    csv_text = _records_to_csv(updated, columns)
    json_text = _records_to_json(updated)
    summary_text = json.dumps(_json_safe(qc_summary), indent=2, allow_nan=False) + "\n"

    backup_paths: list[Path] = []
    if not dry_run:
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_paths = _backup_existing([csv_path, json_path, summary_path], timestamp)
        _write_text(csv_path, csv_text)
        _write_text(json_path, json_text)
        _write_text(summary_path, summary_text)

    return {
        "phasic_out": str(phasic_path),
        "source": source,
        "records_processed": int(len(updated)),
        "dry_run": bool(dry_run),
        "backups_created": [str(p) for p in backup_paths],
        "csv_updated": bool(not dry_run),
        "json_updated": bool(not dry_run),
        "qc_summary_updated": bool(not dry_run),
        "summary": proposal_summary,
        "warnings": warnings,
        "csv_text": csv_text,
        "json_text": json_text,
        "summary_text": summary_text,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"phasic_out: {report['phasic_out']}")
    print(f"records_processed: {report['records_processed']}")
    print(f"dry_run: {str(report['dry_run']).lower()}")
    print(f"backups_created: {len(report['backups_created'])}")
    print(f"csv_updated: {str(report['csv_updated']).lower()}")
    print(f"json_updated: {str(report['json_updated']).lower()}")
    print(f"qc_summary_updated: {str(report['qc_summary_updated']).lower()}")
    for warning in report.get("warnings", []):
        print(f"WARNING: {warning}")
    for policy in SUPPORTED_CORRECTION_POLICIES:
        summary = report["summary"][policy]
        print(f"\n[{policy}]")
        print(f"proposed_correction_mode_counts: {summary['proposed_correction_mode_counts']}")
        print(f"review_required_counts: {summary['review_required_counts']}")
        print(f"review_queue_candidate_counts: {summary['review_queue_candidate_counts']}")
        print(f"warning_level_counts: {summary['warning_level_counts']}")
        print(f"mandatory_review_fraction: {summary['mandatory_review_fraction']:.6f}")
        print(
            "review_queue_candidate_fraction: "
            f"{summary['review_queue_candidate_fraction']:.6f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute diagnostic correction-policy proposal fields from an "
            "existing _analysis/phasic_out folder without rerunning analysis."
        )
    )
    parser.add_argument("--phasic-out", required=True, help="Path to _analysis/phasic_out")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak_* files")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = recompute_policy_proposals(
            args.phasic_out,
            backup=not bool(args.no_backup),
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
