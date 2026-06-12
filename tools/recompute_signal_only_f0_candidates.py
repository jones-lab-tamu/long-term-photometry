#!/usr/bin/env python3
"""Recompute diagnostic signal-only F0 candidate fields for an existing phasic run."""

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

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photometry_pipeline.core.signal_only_f0_candidate import (  # noqa: E402
    DEFAULTS as SIGNAL_ONLY_F0_DEFAULTS,
    compute_signal_only_f0_candidate,
    summarize_signal_only_f0_candidates,
)
from photometry_pipeline.io.hdf5_cache_reader import (  # noqa: E402
    CacheReadError,
    load_cache_chunk_fields,
    open_phasic_cache,
)


SIGNAL_ONLY_F0_FIELD_NAMES = set(
    compute_signal_only_f0_candidate(np.arange(10.0, dtype=float)).keys()
) - {"signal_only_f0_candidate"}
FLAG_LIST_FIELDS = {
    "reference_comparison_flags",
    "dynamic_fit_qc_hard_flags",
    "dynamic_fit_qc_soft_flags",
    "dynamic_fit_qc_flags",
    "signal_state_flags",
    "signal_only_f0_flags",
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
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return str(value)


def _normalize_flag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and not math.isfinite(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if ";" in text:
            return [x.strip() for x in text.split(";") if x.strip()]
        return [text]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


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
    return [dict(x) for x in df.to_dict(orient="records")], list(df.columns)


def _load_qc_summary(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return {}, False
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return (data if isinstance(data, dict) else {}), True


def _find_phasic_cache(phasic_path: Path) -> Path:
    direct = phasic_path / "phasic_trace_cache.h5"
    if direct.exists():
        return direct
    matches = sorted(phasic_path.rglob("phasic_trace_cache.h5"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing phasic_trace_cache.h5 under {phasic_path}")


def _candidate_config_paths(phasic_path: Path) -> list[Path]:
    paths = []
    for base in [phasic_path, phasic_path.parent, phasic_path.parent.parent]:
        paths.extend([base / "config_used.yaml", base / "config_effective.yaml"])
    return paths


def _load_signal_only_f0_config(phasic_path: Path) -> tuple[dict[str, Any], bool, str | None]:
    for path in _candidate_config_paths(phasic_path):
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        config = {key: data[key] for key in SIGNAL_ONLY_F0_DEFAULTS if key in data}
        return config, False, str(path)
    return {}, True, None


def _normalize_loaded_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for key in list(out):
        if key in FLAG_LIST_FIELDS or key.startswith("proposal_flags_"):
            out[key] = _normalize_flag_list(out[key])
    return out


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    roi = str(record.get("roi", "")).strip()
    if not roi:
        raise ValueError(f"QC record is missing roi: {record}")
    try:
        chunk_id = int(record.get("chunk_id"))
    except Exception as exc:
        raise ValueError(f"QC record for ROI {roi} is missing numeric chunk_id") from exc
    return roi, chunk_id


def _merge_signal_only_f0_fields(record: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for key in SIGNAL_ONLY_F0_FIELD_NAMES:
        out.pop(key, None)
    for key, value in diagnostics.items():
        if key != "signal_only_f0_candidate":
            out[key] = value
    return out


def _serialize_record_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for key, value in list(out.items()):
        if key in FLAG_LIST_FIELDS or key.startswith("proposal_flags_"):
            out[key] = ";".join(_normalize_flag_list(value))
    return out


def _records_to_csv(records: list[dict[str, Any]], columns: list[str]) -> str:
    all_columns = list(columns)
    for field in sorted(SIGNAL_ONLY_F0_FIELD_NAMES):
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
            if key == "signal_only_f0_flags" or key.startswith("proposal_flags_"):
                out[key] = _normalize_flag_list(value)
        json_records.append(_json_safe(out))
    return json.dumps(json_records, indent=2, allow_nan=False) + "\n"


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


def recompute_signal_only_f0_candidates(
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
    cache_path = _find_phasic_cache(phasic_path)

    warnings: list[str] = [
        "HDF5 signal_only_f0_candidate trace write-back is deferred for post-hoc recompute."
    ]
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

    config, using_defaults, config_source = _load_signal_only_f0_config(phasic_path)
    updated: list[dict[str, Any]] = []
    missing: list[str] = []
    with open_phasic_cache(str(cache_path)) as cache:
        for raw_rec in records:
            rec = _normalize_loaded_record(raw_rec)
            roi, chunk_id = _record_key(rec)
            try:
                time_sec, sig_raw = load_cache_chunk_fields(
                    cache, roi, chunk_id, ["time_sec", "sig_raw"]
                )
            except CacheReadError as exc:
                missing.append(f"{roi}/chunk_{chunk_id}: {exc}")
                continue
            diagnostics = compute_signal_only_f0_candidate(
                signal=np.asarray(sig_raw, dtype=float),
                time=np.asarray(time_sec, dtype=float),
                signal_state=rec,
                config=config,
            )
            updated.append(_merge_signal_only_f0_fields(rec, diagnostics))
    if missing:
        raise RuntimeError(
            "Cannot recompute signal-only F0 candidates because raw signal/time data are "
            "missing from phasic_trace_cache.h5: " + "; ".join(missing)
        )

    qc_summary, summary_existed = _load_qc_summary(summary_path)
    if not summary_existed:
        warnings.append("qc_summary.json was missing and will be created.")
    f0_summary = summarize_signal_only_f0_candidates(updated)
    f0_summary["using_default_signal_only_f0_config"] = bool(using_defaults)
    f0_summary["signal_only_f0_config_source"] = config_source
    f0_summary["hdf5_trace_write_back"] = "deferred"
    qc_summary["signal_only_f0_candidate_summary"] = f0_summary

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
        "cache_path": str(cache_path),
        "source": source,
        "records_processed": int(len(updated)),
        "dry_run": bool(dry_run),
        "backups_created": [str(p) for p in backup_paths],
        "csv_updated": bool(not dry_run),
        "json_updated": bool(not dry_run),
        "qc_summary_updated": bool(not dry_run),
        "using_default_signal_only_f0_config": bool(using_defaults),
        "signal_only_f0_config_source": config_source,
        "warnings": warnings,
        "summary": f0_summary,
        "csv_text": csv_text,
        "json_text": json_text,
        "summary_text": summary_text,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"phasic_out: {report['phasic_out']}")
    print(f"cache_path: {report['cache_path']}")
    print(f"records_processed: {report['records_processed']}")
    print(f"dry_run: {str(report['dry_run']).lower()}")
    print(f"backups_created: {len(report['backups_created'])}")
    print(f"csv_updated: {str(report['csv_updated']).lower()}")
    print(f"json_updated: {str(report['json_updated']).lower()}")
    print(f"qc_summary_updated: {str(report['qc_summary_updated']).lower()}")
    print(
        "using_default_signal_only_f0_config: "
        f"{str(report['using_default_signal_only_f0_config']).lower()}"
    )
    if report.get("signal_only_f0_config_source"):
        print(f"signal_only_f0_config_source: {report['signal_only_f0_config_source']}")
    for warning in report.get("warnings", []):
        print(f"WARNING: {warning}")
    summary = report.get("summary", {})
    print(
        "signal_only_f0_candidate_viability_counts: "
        f"{summary.get('signal_only_f0_candidate_viability_counts', {})}"
    )
    print(f"signal_only_f0_flag_counts: {summary.get('signal_only_f0_flag_counts', {})}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute signal-only F0 diagnostic fields from an existing "
            "_analysis/phasic_out folder without rerunning analysis."
        )
    )
    parser.add_argument("--phasic-out", required=True, help="Path to _analysis/phasic_out")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak_* files")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = recompute_signal_only_f0_candidates(
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
