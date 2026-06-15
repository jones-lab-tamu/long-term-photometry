#!/usr/bin/env python3
"""Run production features from a verified applied_dff cache."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photometry_pipeline.config import Config  # noqa: E402
from photometry_pipeline.core.feature_extraction import extract_features  # noqa: E402
from tools.verify_applied_dff_cache import (  # noqa: E402
    AppliedDffCacheVerificationError,
    verify_applied_dff_cache,
)

STRATEGIES = ("dynamic_fit", "signal_only_f0")
TOOL_NAME = "run_applied_dff_features"
OUTPUT_FILES = (
    "features.csv",
    "features.json",
    "feature_summary.csv",
    "feature_summary.json",
    "feature_provenance.json",
)

FEATURE_SUMMARY_FIELDS = [
    "roi",
    "requested_correction_strategy",
    "applied_correction_strategy",
    "applied_trace_source",
    "applied_trace_units",
    "feature_detection_input",
    "applied_trace_cache_path",
    "applied_trace_cache_sha256",
    "source_phasic_cache_path",
    "source_phasic_cache_sha256",
    "applied_cache_verification_passed",
    "applied_cache_verification_skipped",
    "applied_trace_complete",
    "n_chunks",
    "n_chunks_processed",
    "n_chunks_skipped",
    "n_features",
    "feature_config_path",
    "feature_config_hash",
    "output_dir",
    "features_csv",
    "features_json",
    "created_at_utc",
    "tool_name",
    "upstream_applied_trace_warning_level",
    "upstream_applied_trace_review_required",
    "upstream_applied_trace_flags",
    "hdf5_modified_source_phasic_cache",
    "legacy_features_modified",
]

PROVENANCE_COLUMNS = [
    "requested_correction_strategy",
    "applied_correction_strategy",
    "applied_trace_source",
    "applied_trace_units",
    "feature_detection_input",
    "applied_trace_cache_path",
    "applied_trace_cache_sha256",
    "source_phasic_cache_path",
    "source_phasic_cache_sha256",
    "feature_config_hash",
    "feature_config_path",
    "upstream_warning_level",
    "upstream_review_required",
    "upstream_flags",
]


class AppliedDffFeatureRunError(RuntimeError):
    """Raised when applied_dff production feature extraction cannot proceed."""


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _legacy_features_hash(phasic_out: Path) -> str:
    path = phasic_out / "features" / "features.csv"
    return _file_sha256(path) if path.exists() else ""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "y", "on"}


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_scalar(value.item())
        return [_decode_scalar(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return _decode_scalar(value.item())
    return value


def _h5_value(group: h5py.Group, name: str) -> Any:
    return _decode_scalar(group[name][()])


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, bool):
                    value = str(value).lower()
                elif isinstance(value, float) and not math.isfinite(value):
                    value = ""
                out[key] = value
            writer.writerow(out)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _config_payload(config: Config) -> dict[str, Any]:
    payload = dataclasses.asdict(config)
    payload["event_signal"] = "dff"
    return payload


def _load_feature_config(path: Path | None) -> tuple[Config, str, str, str]:
    if path is None:
        cfg = Config(event_signal="dff")
        payload = _config_payload(cfg)
        config_json = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
        return cfg, "", config_json, hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    if not path.exists():
        raise AppliedDffFeatureRunError(f"feature config not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise AppliedDffFeatureRunError("feature config must be a mapping")
    payload = dict(payload)
    payload["event_signal"] = "dff"
    cfg = Config(**payload)
    config_json = json.dumps(_json_safe(_config_payload(cfg)), sort_keys=True, separators=(",", ":"))
    return cfg, str(path), config_json, hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise AppliedDffFeatureRunError(f"output already exists, refusing to overwrite without --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def _infer_fs_hz(time_sec: np.ndarray) -> float:
    t = np.asarray(time_sec, dtype=float).reshape(-1)
    diffs = np.diff(t[np.isfinite(t)])
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size:
        return float(1.0 / np.median(diffs))
    return float(Config().target_fs_hz)


def _load_applied_summary(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "applied_correction_summary.json").read_text(encoding="utf-8"))


def _process_chunks(
    *,
    applied_cache: Path,
    roi: str,
    strategy: str,
    cfg: Config,
    applied_summary: dict[str, Any],
    feature_config_hash: str,
    feature_config_path: str,
) -> tuple[list[dict[str, Any]], int, int]:
    expected_source = {"dynamic_fit": "dynamic_fit_dff", "signal_only_f0": "signal_only_f0_dff"}[strategy]
    rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    with h5py.File(applied_cache, "r") as h5:
        chunk_ids = [int(x) for x in np.asarray(h5["meta/chunk_ids"][()]).reshape(-1).tolist()]
        for chunk_id in chunk_ids:
            grp = h5[f"roi/{roi}/chunk_{chunk_id}"]
            available = _as_bool(_h5_value(grp, "available"))
            if not available:
                skipped += 1
                continue
            source = str(_h5_value(grp, "applied_trace_source"))
            if source != expected_source:
                raise AppliedDffFeatureRunError(f"applied_trace_source mismatch for chunk {chunk_id}")
            if "time_sec" not in grp or "applied_dff" not in grp:
                raise AppliedDffFeatureRunError(f"chunk {chunk_id} missing time_sec/applied_dff")
            time_sec = np.asarray(grp["time_sec"][()], dtype=float).reshape(-1)
            applied_dff = np.asarray(grp["applied_dff"][()], dtype=float).reshape(-1)
            if time_sec.shape != applied_dff.shape:
                raise AppliedDffFeatureRunError(f"time_sec/applied_dff length mismatch for chunk {chunk_id}")
            chunk = SimpleNamespace(
                chunk_id=int(chunk_id),
                source_file=str(_h5_value(grp, "source_file")),
                channel_names=[roi],
                time_sec=time_sec,
                fs_hz=_infer_fs_hz(time_sec),
                dff=applied_dff.reshape(-1, 1),
                metadata={},
            )
            df = extract_features(chunk, cfg)
            for row in df.to_dict(orient="records"):
                row.setdefault("roi", roi)
                row.setdefault("chunk_id", int(chunk_id))
                row.setdefault("source_file", str(_h5_value(grp, "source_file")))
                row.update(
                    {
                        "requested_correction_strategy": strategy,
                        "applied_correction_strategy": applied_summary.get("applied_correction_strategy", ""),
                        "applied_trace_source": applied_summary.get("applied_trace_source", ""),
                        "applied_trace_units": applied_summary.get("applied_trace_units", "dff"),
                        "feature_detection_input": "applied_dff",
                        "applied_trace_cache_path": str(applied_cache),
                        "applied_trace_cache_sha256": applied_summary.get("applied_trace_cache_sha256", ""),
                        "source_phasic_cache_path": applied_summary.get("source_phasic_cache_path", ""),
                        "source_phasic_cache_sha256": applied_summary.get("source_phasic_cache_sha256", ""),
                        "feature_config_hash": feature_config_hash,
                        "feature_config_path": feature_config_path,
                        "upstream_warning_level": str(_h5_value(grp, "warning_level")),
                        "upstream_review_required": _as_bool(_h5_value(grp, "review_required")),
                        "upstream_flags": str(_h5_value(grp, "flags")),
                    }
                )
                rows.append(row)
            processed += 1
    return rows, processed, skipped


def run_applied_dff_features(
    phasic_out: str | os.PathLike[str],
    *,
    roi: str,
    strategy: str,
    applied_output_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    feature_config: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    skip_verification: bool = False,
) -> dict[str, Any]:
    strategy = str(strategy or "").strip()
    if strategy not in STRATEGIES:
        raise AppliedDffFeatureRunError(f"unsupported applied_dff feature strategy: {strategy}")
    phasic_path = Path(phasic_out).resolve()
    applied_dir = Path(applied_output_dir).resolve()
    selected_output = Path(output_dir).resolve() if output_dir is not None else applied_dir / "features"
    expected_outputs = {name: str(selected_output / name) for name in OUTPUT_FILES}
    if dry_run:
        source_exists = (phasic_path / "phasic_trace_cache.h5").exists()
        applied_exists = (applied_dir / "applied_trace_cache.h5").exists()
        summary_exists = (applied_dir / "applied_correction_summary.json").exists()
        return {
            "dry_run": True,
            "would_verify_applied_cache": not bool(skip_verification),
            "would_run_feature_detection": bool(source_exists and applied_exists and summary_exists),
            "output_dir": str(selected_output),
            "expected_outputs": expected_outputs,
        }

    source_cache = phasic_path / "phasic_trace_cache.h5"
    applied_cache = applied_dir / "applied_trace_cache.h5"
    source_hash_before = _file_sha256(source_cache)
    applied_hash_before = _file_sha256(applied_cache)
    legacy_hash_before = _legacy_features_hash(phasic_path)

    verification = None
    if not skip_verification:
        try:
            verification = verify_applied_dff_cache(
                phasic_path,
                roi=roi,
                strategy=strategy,
                applied_output_dir=applied_dir,
            )
        except AppliedDffCacheVerificationError as exc:
            raise AppliedDffFeatureRunError(f"applied cache verification failed: {exc}") from exc

    applied_summary = _load_applied_summary(applied_dir)
    if not _as_bool(applied_summary.get("applied_trace_available")):
        raise AppliedDffFeatureRunError("applied trace is not available")
    if not _as_bool(applied_summary.get("applied_trace_complete")):
        raise AppliedDffFeatureRunError("applied trace is incomplete; production applied features require complete input")
    _prepare_output(selected_output, overwrite=overwrite)

    cfg, config_path, config_json, config_hash = _load_feature_config(Path(feature_config).resolve() if feature_config else None)
    feature_rows, processed, skipped = _process_chunks(
        applied_cache=applied_cache,
        roi=roi,
        strategy=strategy,
        cfg=cfg,
        applied_summary=applied_summary,
        feature_config_hash=config_hash,
        feature_config_path=config_path,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    features_csv = selected_output / "features.csv"
    features_json = selected_output / "features.json"
    summary_csv = selected_output / "feature_summary.csv"
    summary_json = selected_output / "feature_summary.json"
    provenance_json = selected_output / "feature_provenance.json"
    fields = list(feature_rows[0].keys()) if feature_rows else ["chunk_id", "source_file", "roi", *PROVENANCE_COLUMNS]
    _write_csv(features_csv, feature_rows, fields)
    _write_json(features_json, {"features": feature_rows})

    source_hash_after = _file_sha256(source_cache)
    applied_hash_after = _file_sha256(applied_cache)
    legacy_hash_after = _legacy_features_hash(phasic_path)
    summary = {
        "roi": roi,
        "requested_correction_strategy": strategy,
        "applied_correction_strategy": applied_summary.get("applied_correction_strategy", ""),
        "applied_trace_source": applied_summary.get("applied_trace_source", ""),
        "applied_trace_units": applied_summary.get("applied_trace_units", "dff"),
        "feature_detection_input": "applied_dff",
        "applied_trace_cache_path": str(applied_cache),
        "applied_trace_cache_sha256": applied_summary.get("applied_trace_cache_sha256", ""),
        "source_phasic_cache_path": str(source_cache),
        "source_phasic_cache_sha256": applied_summary.get("source_phasic_cache_sha256", ""),
        "applied_cache_verification_passed": bool(verification.get("verification_passed")) if verification else False,
        "applied_cache_verification_skipped": bool(skip_verification),
        "applied_trace_complete": _as_bool(applied_summary.get("applied_trace_complete")),
        "n_chunks": int(applied_summary.get("n_chunks", processed + skipped)),
        "n_chunks_processed": int(processed),
        "n_chunks_skipped": int(skipped),
        "n_features": int(len(feature_rows)),
        "feature_config_path": config_path,
        "feature_config_hash": config_hash,
        "output_dir": str(selected_output),
        "features_csv": str(features_csv),
        "features_json": str(features_json),
        "created_at_utc": created_at,
        "tool_name": TOOL_NAME,
        "upstream_applied_trace_warning_level": applied_summary.get("applied_trace_warning_level", ""),
        "upstream_applied_trace_review_required": _as_bool(applied_summary.get("applied_trace_review_required")),
        "upstream_applied_trace_flags": applied_summary.get("applied_trace_flags", ""),
        "hdf5_modified_source_phasic_cache": bool(source_hash_before != source_hash_after),
        "legacy_features_modified": bool(legacy_hash_before != legacy_hash_after),
    }
    _write_csv(summary_csv, [summary], FEATURE_SUMMARY_FIELDS)
    _write_json(summary_json, summary)
    provenance = {
        "tool_name": TOOL_NAME,
        "created_at_utc": created_at,
        "phasic_out": str(phasic_path),
        "roi": roi,
        "requested_correction_strategy": strategy,
        "applied_output_dir": str(applied_dir),
        "applied_trace_cache_path": str(applied_cache),
        "applied_trace_cache_sha256": applied_summary.get("applied_trace_cache_sha256", ""),
        "applied_cache_verification_passed": summary["applied_cache_verification_passed"],
        "applied_cache_verification_skipped": summary["applied_cache_verification_skipped"],
        "applied_cache_verification_summary": verification or {"skipped": True},
        "feature_config_path": config_path,
        "feature_config_json": config_json,
        "feature_config_hash": config_hash,
        "input_trace_dataset": "applied_dff",
        "input_trace_units": "dff",
        "output_dir": str(selected_output),
        "outputs_written": [str(selected_output / name) for name in OUTPUT_FILES],
        "hdf5_modified_source_phasic_cache": summary["hdf5_modified_source_phasic_cache"],
        "legacy_features_modified": summary["legacy_features_modified"],
        "source_phasic_cache_sha256_before": source_hash_before,
        "source_phasic_cache_sha256_after": source_hash_after,
        "applied_trace_cache_sha256_before": applied_hash_before,
        "applied_trace_cache_sha256_after": applied_hash_after,
        "legacy_features_sha256_before": legacy_hash_before,
        "legacy_features_sha256_after": legacy_hash_after,
    }
    _write_json(provenance_json, provenance)
    if summary["hdf5_modified_source_phasic_cache"]:
        raise AppliedDffFeatureRunError("source phasic cache changed during applied feature run")
    if applied_hash_before != applied_hash_after:
        raise AppliedDffFeatureRunError("applied trace cache changed during applied feature run")
    if summary["legacy_features_modified"]:
        raise AppliedDffFeatureRunError("legacy features.csv changed during applied feature run")
    return {
        "dry_run": False,
        "output_dir": str(selected_output),
        "features_csv": str(features_csv),
        "features_json": str(features_json),
        "feature_summary_csv": str(summary_csv),
        "feature_summary_json": str(summary_json),
        "feature_provenance_json": str(provenance_json),
        "summary": summary,
        "provenance": provenance,
    }


def _print_report(report: dict[str, Any]) -> None:
    if report.get("dry_run"):
        for key in ("dry_run", "would_verify_applied_cache", "would_run_feature_detection", "output_dir", "expected_outputs"):
            print(f"{key}: {report.get(key)}")
        return
    summary = report["summary"]
    for key in (
        "roi",
        "requested_correction_strategy",
        "applied_trace_source",
        "feature_detection_input",
        "applied_cache_verification_passed",
        "applied_cache_verification_skipped",
        "applied_trace_complete",
        "n_chunks_processed",
        "n_features",
        "hdf5_modified_source_phasic_cache",
        "legacy_features_modified",
        "output_dir",
    ):
        print(f"{key}: {summary.get(key)}")
    print(f"features_csv: {report['features_csv']}")
    print(f"feature_summary_json: {report['feature_summary_json']}")
    print(f"feature_provenance_json: {report['feature_provenance_json']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run production features from a verified applied_dff cache.")
    parser.add_argument("--phasic-out", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--applied-output-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = run_applied_dff_features(
            args.phasic_out,
            roi=args.roi,
            strategy=args.strategy,
            applied_output_dir=args.applied_output_dir,
            output_dir=args.output_dir,
            feature_config=args.feature_config,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            skip_verification=bool(args.skip_verification),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
