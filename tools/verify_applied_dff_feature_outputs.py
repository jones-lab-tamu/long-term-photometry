#!/usr/bin/env python3
"""Semantically verify production applied_dff feature outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd

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
VERIFICATION_MODE = "applied_dff_feature_output_semantic_verification"
REQUIRED_FEATURE_FILES = (
    "features.csv",
    "features.json",
    "feature_summary.csv",
    "feature_summary.json",
    "feature_provenance.json",
)
REQUIRED_PROVENANCE_COLUMNS = [
    "roi",
    "chunk_id",
    "source_file",
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
EVENT_LIKE_COLUMNS = {
    "peak_time",
    "event_time",
    "event_start",
    "event_end",
    "event_start_time_sec",
    "event_end_time_sec",
    "peak_idx",
    "onset_idx",
    "offset_idx",
    "peak_amplitude",
    "duration",
}


class AppliedDffFeatureOutputVerificationError(RuntimeError):
    """Raised when applied_dff feature output semantic verification fails."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report or {}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): _file_sha256(path) for path in paths if path.exists()}


def _legacy_features_path(phasic_out: Path) -> Path:
    return phasic_out / "features" / "features.csv"


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


def _infer_fs_hz(time_sec: np.ndarray) -> float:
    t = np.asarray(time_sec, dtype=float).reshape(-1)
    diffs = np.diff(t[np.isfinite(t)])
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size:
        return float(1.0 / np.median(diffs))
    return float(Config().target_fs_hz)


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_rows_from_json(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("features"), list):
        return payload["features"]
    if isinstance(payload, list):
        return payload
    return []


def _load_feature_config(provenance: dict[str, Any]) -> Config:
    text = str(provenance.get("feature_config_json") or "{}")
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise AppliedDffFeatureOutputVerificationError(f"invalid feature_config_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise AppliedDffFeatureOutputVerificationError("feature_config_json must decode to an object")
    payload["event_signal"] = "dff"
    return Config(**payload)


def _expected_detector_rows(applied_cache: Path, roi: str, cfg: Config) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with h5py.File(applied_cache, "r") as h5:
        chunk_ids = [int(x) for x in np.asarray(h5["meta/chunk_ids"][()]).reshape(-1).tolist()]
        for chunk_id in chunk_ids:
            grp = h5[f"roi/{roi}/chunk_{chunk_id}"]
            if not _as_bool(_h5_value(grp, "available")):
                continue
            time_sec = np.asarray(grp["time_sec"][()], dtype=float).reshape(-1)
            applied_dff = np.asarray(grp["applied_dff"][()], dtype=float).reshape(-1)
            chunk = SimpleNamespace(
                chunk_id=int(chunk_id),
                source_file=str(_h5_value(grp, "source_file")),
                channel_names=[roi],
                time_sec=time_sec,
                fs_hz=_infer_fs_hz(time_sec),
                dff=applied_dff.reshape(-1, 1),
                metadata={},
            )
            frames.append(extract_features(chunk, cfg))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _sort_detector(df: pd.DataFrame) -> pd.DataFrame:
    keys = [key for key in ("chunk_id", "roi", "source_file") if key in df.columns]
    out = df.copy()
    if keys:
        out = out.sort_values(keys).reset_index(drop=True)
    else:
        out = out.sort_index().reset_index(drop=True)
    return out


def _compare_detector(expected: pd.DataFrame, observed: pd.DataFrame) -> tuple[int, int, int]:
    row_mismatches = 0
    value_mismatches = 0
    compared = 0
    if len(expected) != len(observed):
        row_mismatches = 1
    common = [col for col in expected.columns if col in observed.columns]
    if len(expected) == 0 and len(observed) == 0:
        return row_mismatches, value_mismatches, 0
    if len(common) != len(expected.columns):
        value_mismatches += len(set(expected.columns) - set(common))
    exp = _sort_detector(expected[common]) if common else expected
    obs = _sort_detector(observed[common]) if common else observed
    n = min(len(exp), len(obs))
    compared = n
    for idx in range(n):
        for col in common:
            a = exp.iloc[idx][col]
            b = obs.iloc[idx][col]
            a_num = pd.to_numeric(pd.Series([a]), errors="coerce").iloc[0]
            b_num = pd.to_numeric(pd.Series([b]), errors="coerce").iloc[0]
            if pd.notna(a_num) or pd.notna(b_num):
                if not (pd.notna(a_num) and pd.notna(b_num) and np.isclose(float(a_num), float(b_num), rtol=1e-8, atol=1e-10, equal_nan=True)):
                    value_mismatches += 1
            elif str(a) != str(b):
                value_mismatches += 1
    return row_mismatches, value_mismatches, compared


def _infer_granularity(features: pd.DataFrame, n_chunks_processed: int, detector_columns: list[str]) -> str:
    if len(features) == 0:
        return "empty"
    event_cols = EVENT_LIKE_COLUMNS.intersection(set(features.columns))
    one_per_chunk = n_chunks_processed > 0 and len(features) == n_chunks_processed
    chunk_cols = {"mean", "median", "std", "mad", "peak_count", "auc"}.intersection(set(detector_columns))
    if one_per_chunk and chunk_cols and not event_cols:
        return "chunk_summary"
    if event_cols and not one_per_chunk:
        return "event_or_peak_rows"
    if event_cols and one_per_chunk:
        return "mixed"
    return "unknown"


def _base_report(roi: str, strategy: str, write_summary: Path | None) -> dict[str, Any]:
    return {
        "verification_passed": False,
        "verification_mode": VERIFICATION_MODE,
        "roi": roi,
        "strategy": strategy,
        "applied_trace_source": "",
        "feature_detection_input": "",
        "applied_cache_verification_passed": False,
        "applied_trace_complete": False,
        "n_chunks": 0,
        "n_chunks_processed": 0,
        "n_feature_rows_observed": 0,
        "n_detector_rows_expected": 0,
        "n_detector_rows_compared": 0,
        "n_detector_row_count_mismatches": 0,
        "n_detector_value_mismatches": 0,
        "n_missing_required_provenance_columns": 0,
        "n_provenance_value_mismatches": 0,
        "feature_output_granularity": "unknown",
        "one_feature_row_per_chunk": False,
        "one_feature_row_per_chunk_matches_detector": False,
        "semantic_status": "fail",
        "detector_columns_expected": [],
        "detector_columns_observed": [],
        "provenance_columns_observed": [],
        "hdf5_modified_source_phasic_cache": False,
        "applied_cache_modified": False,
        "legacy_features_modified": False,
        "feature_outputs_modified": False,
        "verification_output_path": str(write_summary or ""),
        "failure_messages": "",
    }


def verify_applied_dff_feature_outputs(
    phasic_out: str | os.PathLike[str],
    *,
    roi: str,
    strategy: str,
    applied_output_dir: str | os.PathLike[str],
    feature_output_dir: str | os.PathLike[str],
    write_summary: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
    fail_on_warning: bool = False,
) -> dict[str, Any]:
    strategy = str(strategy or "").strip()
    if strategy not in STRATEGIES:
        raise AppliedDffFeatureOutputVerificationError(f"unsupported strategy: {strategy}")
    phasic_path = Path(phasic_out).resolve()
    applied_dir = Path(applied_output_dir).resolve()
    feature_dir = Path(feature_output_dir).resolve()
    summary_path = Path(write_summary).resolve() if write_summary is not None else None
    report = _base_report(roi, strategy, summary_path)
    if dry_run:
        return {**report, "dry_run": True, "would_verify_feature_outputs": True}

    failures: list[str] = []
    source_cache = phasic_path / "phasic_trace_cache.h5"
    applied_cache = applied_dir / "applied_trace_cache.h5"
    feature_files = [feature_dir / name for name in REQUIRED_FEATURE_FILES]
    for path in [source_cache, applied_cache, applied_dir / "applied_correction_summary.json", *feature_files]:
        if not path.exists():
            failures.append(f"missing required file: {path}")
    if failures:
        report["failure_messages"] = ";".join(failures)
        _write_report(summary_path, report)
        raise AppliedDffFeatureOutputVerificationError(report["failure_messages"], report=report)

    source_before = _file_sha256(source_cache)
    applied_before = _file_sha256(applied_cache)
    legacy_before = _snapshot([_legacy_features_path(phasic_path)])
    feature_before = _snapshot(feature_files)

    try:
        applied_verification = verify_applied_dff_cache(
            phasic_path,
            roi=roi,
            strategy=strategy,
            applied_output_dir=applied_dir,
            fail_on_warning=fail_on_warning,
        )
    except AppliedDffCacheVerificationError as exc:
        failures.append(f"applied cache verification failed: {exc}")
        report["applied_cache_verification_passed"] = False
        report["failure_messages"] = ";".join(failures)
        _write_report(summary_path, report)
        raise AppliedDffFeatureOutputVerificationError(report["failure_messages"], report=report) from exc

    applied_summary = _load_json(applied_dir / "applied_correction_summary.json")
    feature_summary = _load_json(feature_dir / "feature_summary.json")
    provenance = _load_json(feature_dir / "feature_provenance.json")
    features = pd.read_csv(feature_dir / "features.csv")
    feature_json_rows = _feature_rows_from_json(feature_dir / "features.json")
    cfg = _load_feature_config(provenance)
    expected = _expected_detector_rows(applied_cache, roi, cfg)

    report.update(
        {
            "applied_trace_source": str(feature_summary.get("applied_trace_source", "")),
            "feature_detection_input": str(feature_summary.get("feature_detection_input", "")),
            "applied_cache_verification_passed": bool(applied_verification.get("verification_passed")),
            "applied_trace_complete": _as_bool(feature_summary.get("applied_trace_complete")),
            "n_chunks": int(feature_summary.get("n_chunks", 0)),
            "n_chunks_processed": int(feature_summary.get("n_chunks_processed", 0)),
            "n_feature_rows_observed": int(len(features)),
            "n_detector_rows_expected": int(len(expected)),
            "detector_columns_expected": list(expected.columns),
            "detector_columns_observed": [col for col in expected.columns if col in features.columns],
            "provenance_columns_observed": [col for col in REQUIRED_PROVENANCE_COLUMNS if col in features.columns],
        }
    )

    for key in (
        "requested_correction_strategy",
        "applied_correction_strategy",
        "applied_trace_source",
        "applied_trace_units",
        "applied_trace_cache_path",
        "applied_trace_cache_sha256",
        "source_phasic_cache_path",
        "source_phasic_cache_sha256",
    ):
        if str(feature_summary.get(key, "")) != str(applied_summary.get(key, "")):
            failures.append(f"feature summary/applied summary mismatch for {key}")
    if _as_bool(feature_summary.get("applied_trace_complete")) != _as_bool(applied_summary.get("applied_trace_complete")):
        failures.append("feature summary/applied summary mismatch for applied_trace_complete")
    if str(feature_summary.get("upstream_applied_trace_warning_level", "")) != str(applied_summary.get("applied_trace_warning_level", "")):
        failures.append("feature summary/applied warning level mismatch")
    if _as_bool(feature_summary.get("upstream_applied_trace_review_required")) != _as_bool(applied_summary.get("applied_trace_review_required")):
        failures.append("feature summary/applied review flag mismatch")
    if str(feature_summary.get("upstream_applied_trace_flags", "")) != str(applied_summary.get("applied_trace_flags", "")):
        failures.append("feature summary/applied flags mismatch")
    if str(provenance.get("applied_trace_cache_sha256", "")) != str(applied_summary.get("applied_trace_cache_sha256", "")):
        failures.append("feature provenance/applied cache hash mismatch")
    if str(provenance.get("feature_config_hash", "")) != str(feature_summary.get("feature_config_hash", "")):
        failures.append("feature provenance/summary config hash mismatch")
    if len(feature_json_rows) != len(features):
        failures.append("features.csv and features.json row count mismatch")
    if int(feature_summary.get("n_features", -1)) != len(features):
        failures.append("feature summary n_features mismatch")
    if str(feature_summary.get("features_csv", "")) != str(feature_dir / "features.csv"):
        failures.append("feature summary features_csv path mismatch")
    if str(feature_summary.get("features_json", "")) != str(feature_dir / "features.json"):
        failures.append("feature summary features_json path mismatch")
    if str(feature_summary.get("feature_detection_input", "")) != "applied_dff":
        failures.append("feature_detection_input is not applied_dff")

    missing_cols = [col for col in REQUIRED_PROVENANCE_COLUMNS if col not in features.columns]
    report["n_missing_required_provenance_columns"] = len(missing_cols)
    if missing_cols:
        failures.append(f"missing required provenance columns: {missing_cols}")

    expected_source = {"dynamic_fit": "dynamic_fit_dff", "signal_only_f0": "signal_only_f0_dff"}[strategy]
    for _, row in features.iterrows():
        checks = {
            "requested_correction_strategy": strategy,
            "applied_correction_strategy": strategy,
            "applied_trace_source": expected_source,
            "applied_trace_units": "dff",
            "feature_detection_input": "applied_dff",
            "applied_trace_cache_sha256": applied_summary.get("applied_trace_cache_sha256", ""),
            "source_phasic_cache_sha256": applied_summary.get("source_phasic_cache_sha256", ""),
        }
        for key, expected_value in checks.items():
            if key in features.columns and str(row.get(key, "")) != str(expected_value):
                report["n_provenance_value_mismatches"] += 1
    if report["n_provenance_value_mismatches"]:
        failures.append("feature provenance value mismatch")

    row_mismatch, value_mismatch, compared = _compare_detector(expected, features)
    report["n_detector_row_count_mismatches"] = int(row_mismatch)
    report["n_detector_value_mismatches"] = int(value_mismatch)
    report["n_detector_rows_compared"] = int(compared)
    if row_mismatch:
        failures.append("detector row count mismatch")
    if value_mismatch:
        failures.append("detector value mismatch")

    granularity = _infer_granularity(features, int(feature_summary.get("n_chunks_processed", 0)), list(expected.columns))
    one_per_chunk = int(feature_summary.get("n_chunks_processed", 0)) > 0 and len(features) == int(feature_summary.get("n_chunks_processed", 0))
    report["feature_output_granularity"] = granularity
    report["one_feature_row_per_chunk"] = bool(one_per_chunk)
    report["one_feature_row_per_chunk_matches_detector"] = bool(one_per_chunk and len(expected) == len(features))
    if one_per_chunk and len(expected) != len(features):
        failures.append("one row per chunk does not match detector output")

    source_after = _file_sha256(source_cache)
    applied_after = _file_sha256(applied_cache)
    legacy_after = _snapshot([_legacy_features_path(phasic_path)])
    feature_after = _snapshot(feature_files)
    report["hdf5_modified_source_phasic_cache"] = source_before != source_after
    report["applied_cache_modified"] = applied_before != applied_after
    report["legacy_features_modified"] = legacy_before != legacy_after
    report["feature_outputs_modified"] = feature_before != feature_after
    if report["hdf5_modified_source_phasic_cache"]:
        failures.append("source phasic cache changed during verification")
    if report["applied_cache_modified"]:
        failures.append("applied cache changed during verification")
    if report["legacy_features_modified"]:
        failures.append("legacy features changed during verification")
    if report["feature_outputs_modified"]:
        failures.append("feature outputs changed during verification")

    report["semantic_status"] = "pass" if not failures else "fail"
    report["verification_passed"] = not failures
    report["failure_messages"] = ";".join(str(x) for x in failures if str(x))
    _write_report(summary_path, report)
    if failures:
        raise AppliedDffFeatureOutputVerificationError(report["failure_messages"], report=report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    for key in (
        "verification_passed",
        "semantic_status",
        "roi",
        "strategy",
        "applied_trace_source",
        "feature_detection_input",
        "n_feature_rows_observed",
        "n_detector_rows_expected",
        "n_detector_value_mismatches",
        "feature_output_granularity",
        "one_feature_row_per_chunk",
        "one_feature_row_per_chunk_matches_detector",
        "hdf5_modified_source_phasic_cache",
        "applied_cache_modified",
        "legacy_features_modified",
        "feature_outputs_modified",
        "failure_messages",
    ):
        print(f"{key}: {report.get(key)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantically verify applied_dff feature outputs.")
    parser.add_argument("--phasic-out", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--applied-output-dir", required=True)
    parser.add_argument("--feature-output-dir", required=True)
    parser.add_argument("--write-summary", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = verify_applied_dff_feature_outputs(
            args.phasic_out,
            roi=args.roi,
            strategy=args.strategy,
            applied_output_dir=args.applied_output_dir,
            feature_output_dir=args.feature_output_dir,
            write_summary=args.write_summary,
            dry_run=bool(args.dry_run),
            fail_on_warning=bool(args.fail_on_warning),
        )
    except AppliedDffFeatureOutputVerificationError as exc:
        if exc.report:
            _print_report(exc.report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
