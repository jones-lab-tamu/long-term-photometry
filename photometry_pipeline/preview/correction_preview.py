"""Guided Workflow correction-preview backend contract helpers.

Stage 4C preview is backend-only. The helpers validate preview namespaces,
source availability, allowed reference-preview methods, and preview-only
provenance dictionaries. The preview runner reads completed-run/phasic-cache
sources read-only, runs allowed reference correction methods in memory, and
writes only preview artifacts into a validated preview namespace. It does not
write manifests, route applied-dF/F, run feature extraction, validate GUI setup,
or launch the pipeline.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from photometry_pipeline.config import Config
from photometry_pipeline.core import preprocessing, regression
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.hdf5_cache_reader import (
    CacheReadError,
    list_cache_chunk_ids,
    list_cache_rois,
    list_cache_source_files,
    load_cache_chunk_attrs,
    open_phasic_cache,
)
from photometry_pipeline.io.adapters import load_chunk
from photometry_pipeline.guided_diagnostic_cache import resolve_diagnostic_cache_source


PREVIEW_PROVENANCE_FILENAME = "preview_provenance.json"
PREVIEW_SUMMARY_FILENAME = "preview_summary.json"
PREVIEW_COMPARISON_PLOT_FILENAME = "comparison_plot.png"
METHOD_TRACE_FILENAME_TEMPLATE = "method_{method}_trace.csv"
METHOD_DIAGNOSTICS_FILENAME_TEMPLATE = "method_{method}_diagnostics.json"

PREVIEW_WARNING_TEXT = (
    "Preview-only artifact. This is not a production output, does not choose a "
    "strategy, does not write a manifest, does not route applied-dF/F, and does "
    "not run feature extraction."
)

GUIDED_REFERENCE_PREVIEW_METHODS = (
    "robust_global_event_reject",
    "adaptive_event_gated_regression",
    "global_linear_regression",
)

GUIDED_REFERENCE_PREVIEW_METHOD_SET = set(GUIDED_REFERENCE_PREVIEW_METHODS)
REJECTED_GUIDED_PREVIEW_METHODS = {
    "signal_only_f0",
    "auto",
    "needs_review",
    "no_correction",
    "rolling_local_regression",
    "rolling_filtered_to_raw",
    "rolling_filtered_to_filtered",
}
VALID_PREVIEW_SOURCE_TYPES = {"completed_run", "phasic_cache", "diagnostic_cache"}
REQUIRED_PREVIEW_CHUNK_FIELDS = ("time_sec", "sig_raw", "uv_raw")


@dataclass(frozen=True)
class PreviewPathValidationResult:
    ok: bool
    resolved_path: str
    code: str
    reason: str
    protected_namespace: str = ""


@dataclass(frozen=True)
class PreviewMethodValidationResult:
    ok: bool
    methods: tuple[str, ...]
    code: str
    reason: str
    invalid_methods: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreviewSourceValidationResult:
    ok: bool
    source_type: str
    completed_run_dir: str = ""
    phasic_out: str = ""
    phasic_trace_cache_path: str = ""
    config_path: str = ""
    diagnostic_cache_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    code: str = "ok"
    reason: str = ""


class GuidedCorrectionPreviewError(RuntimeError):
    """Raised for explicit backend preview failures."""


def _resolve_path(path: str | os.PathLike[str] | None) -> str:
    if path is None:
        return ""
    text = os.fspath(path).strip()
    if not text:
        return ""
    return os.path.realpath(os.path.abspath(text))


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _is_equal_or_inside(path: str, root: str) -> bool:
    if not path or not root:
        return False
    path_norm = _norm(path)
    root_norm = _norm(root)
    if path_norm == root_norm:
        return True
    try:
        rel = os.path.relpath(path_norm, root_norm)
    except ValueError:
        return False
    return rel != os.curdir and not rel.startswith(".." + os.sep) and rel != ".."


def _is_strictly_inside(path: str, root: str) -> bool:
    return _is_equal_or_inside(path, root) and _norm(path) != _norm(root)


def _is_preview_scoped_leaf(path: str | os.PathLike[str], preview_id: str) -> bool:
    resolved = _resolve_path(path)
    if not resolved or not _safe_preview_id_component(preview_id):
        return False
    parts = Path(resolved).parts
    if len(parts) < 3:
        return False
    return (
        parts[-1] == preview_id
        and parts[-2] == "previews"
        and parts[-3] == "_guided_workflow"
    )


def _reject(
    *,
    resolved_path: str,
    code: str,
    reason: str,
    protected_namespace: str = "",
) -> PreviewPathValidationResult:
    return PreviewPathValidationResult(
        ok=False,
        resolved_path=resolved_path,
        code=code,
        reason=reason,
        protected_namespace=protected_namespace,
    )


def _safe_preview_id_component(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if "/" in text or "\\" in text or ".." in text:
        return False
    if Path(text).is_absolute() or Path(text).drive:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text))


def make_guided_preview_id(prefix: str = "preview", *, now: datetime | None = None) -> str:
    """Return a filesystem-safe preview id without path separators."""
    prefix_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix or "preview")).strip("._-")
    if not prefix_safe:
        prefix_safe = "preview"
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(4)
    preview_id = f"{prefix_safe}_{stamp}_{suffix}"
    if not _safe_preview_id_component(preview_id):
        raise RuntimeError(f"generated unsafe preview_id: {preview_id!r}")
    return preview_id


def validate_guided_preview_output_dir(
    preview_dir: str | os.PathLike[str] | None,
    *,
    completed_run_dir: str | os.PathLike[str] | None = None,
    phasic_out: str | os.PathLike[str] | None = None,
    source_roots: Iterable[str | os.PathLike[str]] | None = None,
    applied_dff_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> PreviewPathValidationResult:
    """Validate a proposed preview output directory without creating it.

    Completed-run previews may live only at
    ``<completed_run_dir>/_guided_workflow/previews/<preview_id>``. User-selected
    preview roots outside protected namespaces are also allowed for future
    raw-input preview contracts.
    """
    resolved = _resolve_path(preview_dir)
    if not resolved:
        return _reject(
            resolved_path="",
            code="empty_preview_path",
            reason="Preview output directory is empty.",
        )
    if os.path.isfile(resolved):
        return _reject(
            resolved_path=resolved,
            code="preview_path_is_file",
            reason="Preview output path resolves to an existing file.",
            protected_namespace=resolved,
        )

    source_paths = [_resolve_path(p) for p in (source_roots or []) if _resolve_path(p)]
    applied_paths = [_resolve_path(p) for p in (applied_dff_roots or []) if _resolve_path(p)]
    completed = _resolve_path(completed_run_dir)
    phasic = _resolve_path(phasic_out)
    phasic_features = os.path.join(phasic, "features") if phasic else ""
    completed_guided_previews = (
        os.path.join(completed, "_guided_workflow", "previews") if completed else ""
    )

    if phasic_features and _is_equal_or_inside(resolved, phasic_features):
        return _reject(
            resolved_path=resolved,
            code="inside_legacy_features",
            reason="Preview output directory must be separate from legacy features.",
            protected_namespace=phasic_features,
        )
    if phasic and _is_equal_or_inside(resolved, phasic):
        return _reject(
            resolved_path=resolved,
            code="inside_phasic_out",
            reason="Preview output directory must be separate from phasic_out.",
            protected_namespace=phasic,
        )
    for source in source_paths:
        if _is_equal_or_inside(resolved, source):
            return _reject(
                resolved_path=resolved,
                code="inside_source_root",
                reason="Preview output directory must be separate from source input roots.",
                protected_namespace=source,
            )
    for applied_root in applied_paths:
        if _is_equal_or_inside(resolved, applied_root):
            return _reject(
                resolved_path=resolved,
                code="inside_applied_dff_root",
                reason="Preview output directory must be separate from applied-dF/F production outputs.",
                protected_namespace=applied_root,
            )

    if completed:
        if _norm(resolved) == _norm(completed):
            return _reject(
                resolved_path=resolved,
                code="completed_run_root",
                reason="Preview output directory cannot be the completed run root.",
                protected_namespace=completed,
            )
        if _is_strictly_inside(resolved, completed):
            if not completed_guided_previews or not _is_strictly_inside(
                resolved,
                completed_guided_previews,
            ):
                return _reject(
                    resolved_path=resolved,
                    code="inside_completed_run_outside_preview_namespace",
                    reason=(
                        "Completed-run preview output must be strictly inside "
                        "<completed_run_dir>/_guided_workflow/previews/<preview_id>."
                    ),
                    protected_namespace=completed,
                )
            rel = os.path.relpath(_norm(resolved), _norm(completed_guided_previews))
            parts = [part for part in rel.split(os.sep) if part and part != os.curdir]
            if len(parts) != 1 or not _safe_preview_id_component(parts[0]):
                return _reject(
                    resolved_path=resolved,
                    code="unsafe_preview_id",
                    reason="Completed-run preview output must end with one safe preview_id component.",
                    protected_namespace=completed_guided_previews,
                )

    return PreviewPathValidationResult(
        ok=True,
        resolved_path=resolved,
        code="ok",
        reason="Preview output directory is allowed.",
    )


def validate_preview_methods(methods: Iterable[str] | None) -> PreviewMethodValidationResult:
    """Validate Stage 4C reference-preview methods and preserve input order."""
    if methods is None:
        return PreviewMethodValidationResult(
            ok=False,
            methods=(),
            code="missing_methods",
            reason="No preview methods were provided.",
        )
    normalized: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for raw in methods:
        method = str(raw or "").strip().lower()
        if not method:
            invalid.append(str(raw))
            continue
        if method in seen:
            duplicates.append(method)
            continue
        seen.add(method)
        if method not in GUIDED_REFERENCE_PREVIEW_METHOD_SET:
            invalid.append(method)
        else:
            normalized.append(method)
    if invalid:
        return PreviewMethodValidationResult(
            ok=False,
            methods=tuple(normalized),
            code="unsupported_preview_method",
            reason=(
                "Stage 4C reference preview allows only "
                f"{', '.join(GUIDED_REFERENCE_PREVIEW_METHODS)}."
            ),
            invalid_methods=tuple(invalid),
        )
    if duplicates:
        return PreviewMethodValidationResult(
            ok=False,
            methods=tuple(normalized),
            code="duplicate_preview_method",
            reason="Duplicate preview methods are not allowed.",
            invalid_methods=tuple(duplicates),
        )
    if not normalized:
        return PreviewMethodValidationResult(
            ok=False,
            methods=(),
            code="missing_methods",
            reason="No supported preview methods were provided.",
        )
    return PreviewMethodValidationResult(
        ok=True,
        methods=tuple(normalized),
        code="ok",
        reason="Preview methods are allowed.",
    )


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.ndarray):
        if value.size > 100:
            return {
                "omitted": True,
                "reason": "array omitted from preview diagnostics",
                "shape": list(value.shape),
            }
        return [_json_safe(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(x) for x in value]
    return str(value)


def _write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def _result_failed(
    *,
    preview_id: str,
    preview_output_dir: str = "",
    errors: Iterable[str],
    warnings: Iterable[str] | None = None,
    method_statuses: dict[str, Any] | None = None,
    preview_provenance_path: str = "",
    preview_summary_path: str = "",
    generated_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "preview_id": preview_id,
        "status": "failed",
        "preview_output_dir": preview_output_dir,
        "preview_provenance_path": preview_provenance_path,
        "preview_summary_path": preview_summary_path,
        "generated_artifacts": dict(generated_artifacts or {}),
        "method_statuses": dict(method_statuses or {}),
        "warnings": [str(x) for x in (warnings or [])],
        "errors": [str(x) for x in errors],
    }


def _read_json_dict(path: str) -> tuple[dict[str, Any], str | None]:
    if not os.path.isfile(path):
        return {}, f"File missing at {path}"
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {}, f"Parse error: {exc}"
    if not isinstance(data, dict):
        return {}, f"Root of JSON file is not an object: {path}"
    return data, None


def _is_successful_completed_run_dir(run_dir: str) -> tuple[bool, str]:
    """Backend-local completed-run success check; no GUI imports."""
    if not os.path.isdir(run_dir):
        return False, f"Directory does not exist: {run_dir}"

    report_path = os.path.join(run_dir, "run_report.json")
    report, report_err = _read_json_dict(report_path)
    if report_err is None:
        status_tokens = [
            str(report.get("status", "")).strip().lower(),
            str(report.get("run_status", "")).strip().lower(),
            str(report.get("final_status", "")).strip().lower(),
            str(report.get("result", "")).strip().lower(),
        ]
        phase_tokens = [
            str(report.get("phase", "")).strip().lower(),
            str(report.get("run_phase", "")).strip().lower(),
            str(report.get("final_phase", "")).strip().lower(),
        ]
        success_tokens = {"success", "complete", "completed", "done"}
        if any(tok in success_tokens for tok in status_tokens if tok):
            if not any(phase_tokens) or any(
                tok in {"final", "complete", "completed", "done"}
                for tok in phase_tokens
                if tok
            ):
                return True, "run_report.json indicates a successful completed run."

        run_ctx = report.get("run_context", {})
        if isinstance(run_ctx, dict):
            ctx_status = str(run_ctx.get("status", "")).strip().lower()
            ctx_phase = str(run_ctx.get("phase", "")).strip().lower()
            if ctx_status in success_tokens and (
                not ctx_phase or ctx_phase in {"final", "complete", "completed", "done"}
            ):
                return True, "run_report.json run_context indicates a successful completed run."

    status_path = os.path.join(run_dir, "status.json")
    status_data, status_err = _read_json_dict(status_path)
    if status_err is None:
        schema_ok = status_data.get("schema_version") == 1
        phase_ok = str(status_data.get("phase", "")).strip().lower() == "final"
        status_ok = str(status_data.get("status", "")).strip().lower() == "success"
        if schema_ok and phase_ok and status_ok:
            return True, "status.json indicates final success."

    manifest_path = os.path.join(run_dir, "MANIFEST.json")
    manifest, manifest_err = _read_json_dict(manifest_path)
    if manifest_err is None:
        manifest_status = str(manifest.get("status", "")).strip().lower()
        if manifest_status in {"success", "complete", "completed"}:
            return True, "MANIFEST.json indicates successful completion."

    reasons = []
    if report_err:
        reasons.append(f"run_report.json: {report_err}")
    else:
        reasons.append("run_report.json present but does not explicitly report successful completion.")
    if status_err:
        reasons.append(f"status.json: {status_err}")
    else:
        reasons.append(
            "status.json present but does not match terminal success contract "
            "(schema_version=1, phase=final, status=success)."
        )
    if manifest_err:
        reasons.append(f"MANIFEST.json: {manifest_err}")
    else:
        reasons.append("MANIFEST.json present but status is not success/completed.")
    reasons.append("Select a run directory that contains final-success metadata.")
    return False, " | ".join(reasons)


def resolve_completed_run_preview_source(
    completed_run_dir: str | os.PathLike[str] | None,
) -> PreviewSourceValidationResult:
    """Resolve completed-run preview inputs without opening or mutating cache."""
    run_dir = _resolve_path(completed_run_dir)
    if not run_dir or not os.path.isdir(run_dir):
        return PreviewSourceValidationResult(
            ok=False,
            source_type="completed_run",
            completed_run_dir=run_dir,
            code="completed_run_missing",
            reason="Completed run directory does not exist.",
        )

    successful, evidence = _is_successful_completed_run_dir(run_dir)
    if not successful:
        return PreviewSourceValidationResult(
            ok=False,
            source_type="completed_run",
            completed_run_dir=run_dir,
            code="completed_run_not_successful",
            reason=evidence,
        )

    phasic = os.path.join(run_dir, "_analysis", "phasic_out")
    cache = os.path.join(phasic, "phasic_trace_cache.h5")
    config = os.path.join(phasic, "config_used.yaml")
    if not os.path.isdir(phasic):
        return PreviewSourceValidationResult(
            ok=False,
            source_type="completed_run",
            completed_run_dir=run_dir,
            phasic_out=phasic,
            code="phasic_out_missing",
            reason="Completed run is missing _analysis/phasic_out.",
        )
    if not os.path.isfile(cache):
        return PreviewSourceValidationResult(
            ok=False,
            source_type="completed_run",
            completed_run_dir=run_dir,
            phasic_out=phasic,
            phasic_trace_cache_path=cache,
            code="phasic_cache_missing",
            reason="Completed run is missing phasic_trace_cache.h5.",
        )
    if not os.path.isfile(config):
        return PreviewSourceValidationResult(
            ok=False,
            source_type="completed_run",
            completed_run_dir=run_dir,
            phasic_out=phasic,
            phasic_trace_cache_path=cache,
            config_path=config,
            code="config_snapshot_missing",
            reason="Completed run is missing phasic config_used.yaml.",
        )
    return PreviewSourceValidationResult(
        ok=True,
        source_type="completed_run",
        completed_run_dir=run_dir,
        phasic_out=phasic,
        phasic_trace_cache_path=cache,
        config_path=config,
        code="ok",
        reason="Completed-run preview source is available.",
    )


def resolve_phasic_cache_preview_source(
    phasic_out: str | os.PathLike[str] | None,
) -> PreviewSourceValidationResult:
    """Resolve direct phasic-cache preview inputs without mutating source files."""
    phasic = _resolve_path(phasic_out)
    if not phasic or not os.path.isdir(phasic):
        return PreviewSourceValidationResult(
            ok=False,
            source_type="phasic_cache",
            phasic_out=phasic,
            code="phasic_out_missing",
            reason="phasic_out directory does not exist.",
        )
    cache = os.path.join(phasic, "phasic_trace_cache.h5")
    config = os.path.join(phasic, "config_used.yaml")
    if not os.path.isfile(cache):
        return PreviewSourceValidationResult(
            ok=False,
            source_type="phasic_cache",
            phasic_out=phasic,
            phasic_trace_cache_path=cache,
            code="phasic_cache_missing",
            reason="phasic_out is missing phasic_trace_cache.h5.",
        )
    if not os.path.isfile(config):
        return PreviewSourceValidationResult(
            ok=False,
            source_type="phasic_cache",
            phasic_out=phasic,
            phasic_trace_cache_path=cache,
            config_path=config,
            code="config_snapshot_missing",
            reason="phasic_out is missing config_used.yaml.",
        )
    return PreviewSourceValidationResult(
        ok=True,
        source_type="phasic_cache",
        phasic_out=phasic,
        phasic_trace_cache_path=cache,
        config_path=config,
        code="ok",
        reason="Phasic-cache preview source is available.",
    )


def resolve_diagnostic_cache_preview_source(
    source: str | os.PathLike[str] | None,
) -> PreviewSourceValidationResult:
    """Resolve a Guided diagnostic cache without relabeling it as a completed run."""
    result = resolve_diagnostic_cache_source(source or "")
    if not result.ok or result.source is None:
        return PreviewSourceValidationResult(
            ok=False,
            source_type="diagnostic_cache",
            code=result.status.code,
            reason=result.status.message,
        )
    resolved = result.source
    phasic_out = os.path.dirname(resolved.phasic_trace_cache_path)
    return PreviewSourceValidationResult(
        ok=True,
        source_type="diagnostic_cache",
        completed_run_dir="",
        phasic_out=phasic_out,
        phasic_trace_cache_path=resolved.phasic_trace_cache_path,
        config_path=resolved.config_used_path,
        diagnostic_cache_metadata={
            "source_type": "diagnostic_cache",
            "cache_id": resolved.cache_id,
            "cache_root_path": resolved.cache_root_path,
            "phasic_trace_cache_path": resolved.phasic_trace_cache_path,
            "config_used_path": resolved.config_used_path,
            "request_json_path": resolved.request_json_path,
            "artifact_record_path": resolved.artifact_record_path,
            "provenance_path": resolved.provenance_path,
            "source_setup_signature": resolved.source_setup_signature,
            "diagnostic_scope_signature": resolved.diagnostic_scope_signature,
            "build_request_signature": resolved.build_request_signature,
            "production_analysis": False,
            "preliminary_cache": True,
        },
        code="ok",
        reason="Diagnostic-cache preview source is available.",
    )


def _resolve_preview_source(
    source: str | os.PathLike[str] | None,
    source_type: str | None,
) -> PreviewSourceValidationResult:
    requested = str(source_type or "").strip().lower()
    if requested and requested not in VALID_PREVIEW_SOURCE_TYPES:
        return PreviewSourceValidationResult(
            ok=False,
            source_type=requested,
            code="unsupported_source_type",
            reason="Correction preview supports completed_run, phasic_cache, and diagnostic_cache sources.",
        )
    if requested == "completed_run":
        return resolve_completed_run_preview_source(source)
    if requested == "phasic_cache":
        return resolve_phasic_cache_preview_source(source)
    if requested == "diagnostic_cache":
        return resolve_diagnostic_cache_preview_source(source)

    completed = resolve_completed_run_preview_source(source)
    if completed.ok:
        return completed
    phasic = resolve_phasic_cache_preview_source(source)
    if phasic.ok:
        return phasic
    return PreviewSourceValidationResult(
        ok=False,
        source_type="",
        code="source_not_recognized",
        reason=(
            "Source is neither a successful completed run nor a phasic_out cache source. "
            f"completed_run: {completed.code}: {completed.reason} | "
            f"phasic_cache: {phasic.code}: {phasic.reason}"
        ),
    )


def _compute_chunk_fs_hz(time_sec: np.ndarray, fallback: float) -> float:
    if len(time_sec) < 2:
        return float(fallback)
    dt = np.diff(np.asarray(time_sec, dtype=float).reshape(-1))
    finite = dt[np.isfinite(dt) & (dt > 0)]
    if len(finite) == 0:
        return float(fallback)
    return float(1.0 / np.median(finite))


def _source_file_for_chunk(source_files: list[str], chunk_ids: list[int], chunk_id: int) -> str:
    if len(source_files) == len(chunk_ids):
        try:
            idx = chunk_ids.index(int(chunk_id))
            return str(source_files[idx])
        except ValueError:
            pass
    return f"chunk_{int(chunk_id)}"


def _normalize_window(window: Any) -> tuple[float, float] | None:
    if window is None:
        return None
    if isinstance(window, dict):
        start = window.get("start_sec", window.get("start"))
        end = window.get("end_sec", window.get("end"))
    elif isinstance(window, (list, tuple)) and len(window) == 2:
        start, end = window
    else:
        raise GuidedCorrectionPreviewError(
            "window must be None, a two-item (start_sec, end_sec) sequence, or a mapping."
        )
    start_f = float(start)
    end_f = float(end)
    if not np.isfinite(start_f) or not np.isfinite(end_f) or end_f <= start_f:
        raise GuidedCorrectionPreviewError(
            f"Invalid preview window: start_sec={start_f}, end_sec={end_f}."
        )
    return start_f, end_f


def _load_preview_chunk_record(
    cache: h5py.File,
    *,
    roi: str,
    chunk_index: int | None,
    window: Any,
    cfg: Config,
) -> dict[str, Any]:
    rois = list_cache_rois(cache)
    if roi not in rois:
        raise GuidedCorrectionPreviewError(f"Requested ROI '{roi}' not found in phasic cache.")
    chunk_ids = list_cache_chunk_ids(cache)
    if not chunk_ids:
        raise GuidedCorrectionPreviewError("No chunks are available in phasic cache.")
    chunk_id = int(chunk_ids[0] if chunk_index is None else chunk_index)
    if chunk_id not in chunk_ids:
        raise GuidedCorrectionPreviewError(f"Requested chunk {chunk_id} not found in phasic cache.")

    grp = cache.get(f"roi/{roi}/chunk_{chunk_id}")
    if grp is None:
        raise GuidedCorrectionPreviewError(f"Missing cache group roi/{roi}/chunk_{chunk_id}.")
    missing = [name for name in REQUIRED_PREVIEW_CHUNK_FIELDS if name not in grp]
    if missing:
        raise GuidedCorrectionPreviewError(
            f"Cache chunk missing required preview dataset(s): roi={roi} chunk={chunk_id} missing={missing}"
        )

    time_sec = np.asarray(grp["time_sec"][()], dtype=float).reshape(-1)
    sig_raw = np.asarray(grp["sig_raw"][()], dtype=float).reshape(-1)
    uv_raw = np.asarray(grp["uv_raw"][()], dtype=float).reshape(-1)
    if time_sec.size == 0:
        raise GuidedCorrectionPreviewError(f"Cache chunk is empty: roi={roi} chunk={chunk_id}.")
    if time_sec.shape != sig_raw.shape or time_sec.shape != uv_raw.shape:
        raise GuidedCorrectionPreviewError(
            f"Cache chunk length mismatch for roi={roi} chunk={chunk_id}: "
            f"time_sec={time_sec.size}, sig_raw={sig_raw.size}, uv_raw={uv_raw.size}."
        )

    selected_window = _normalize_window(window)
    if selected_window is not None:
        start_sec, end_sec = selected_window
        mask = (time_sec >= start_sec) & (time_sec <= end_sec)
        if int(np.count_nonzero(mask)) < 3:
            raise GuidedCorrectionPreviewError(
                f"Preview window selects too few samples for roi={roi} chunk={chunk_id}: "
                f"start_sec={start_sec}, end_sec={end_sec}."
            )
        base_time = time_sec[mask]
        offset = float(base_time[0])
        time_sec = base_time - offset
        sig_raw = sig_raw[mask]
        uv_raw = uv_raw[mask]

    source_files = list_cache_source_files(cache)
    fs_hz = _compute_chunk_fs_hz(time_sec, cfg.target_fs_hz)
    try:
        attrs = load_cache_chunk_attrs(cache, roi, chunk_id)
    except CacheReadError:
        attrs = {}
    return {
        "roi": roi,
        "chunk_id": chunk_id,
        "source_file": _source_file_for_chunk(source_files, chunk_ids, chunk_id),
        "time_sec": time_sec,
        "sig_raw": sig_raw,
        "uv_raw": uv_raw,
        "fs_hz": fs_hz,
        "window": selected_window,
        "metadata": attrs,
    }


def _make_preview_chunk(record: dict[str, Any], roi: str) -> Chunk:
    return Chunk(
        chunk_id=int(record["chunk_id"]),
        source_file=str(record["source_file"]),
        format="cache_preview",
        time_sec=np.asarray(record["time_sec"], dtype=float).copy(),
        uv_raw=np.asarray(record["uv_raw"], dtype=float).reshape(-1, 1).copy(),
        sig_raw=np.asarray(record["sig_raw"], dtype=float).reshape(-1, 1).copy(),
        fs_hz=float(record["fs_hz"]),
        channel_names=[roi],
        metadata=dict(record.get("metadata", {})),
    )


def _config_for_method(base_cfg: Config, method: str) -> Config:
    cfg_data = dataclasses.asdict(base_cfg)
    cfg_data["dynamic_fit_mode"] = method
    return Config(**cfg_data)


def _numeric_summary(values: np.ndarray | None) -> dict[str, Any]:
    if values is None:
        return {"available": False}
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    out: dict[str, Any] = {
        "available": True,
        "n_samples": int(arr.size),
        "n_finite": int(finite.size),
    }
    if finite.size:
        out.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
            }
        )
    return out


def _method_metadata_for_roi(chunk: Chunk, roi: str) -> dict[str, Any]:
    metadata = dict(chunk.metadata or {})
    selected: dict[str, Any] = {}
    for key in (
        "dynamic_fit_mode_requested",
        "dynamic_fit_mode_resolved",
        "dynamic_fit_mode_alias_applied",
        "baseline_subtract_before_fit_requested",
        "baseline_subtract_before_fit_applied",
        "bleach_correction_mode_requested",
        "bleach_correction_mode_resolved",
        "bleach_correction_applied",
        "dynamic_fit_engine_info",
    ):
        if key in metadata:
            selected[key] = metadata[key]
    for key in (
        "dynamic_fit_event_reject",
        "dynamic_fit_adaptive_event_gated",
        "dynamic_fit_global_linear",
        "dynamic_fit_rolling_local",
        "bleach_correction",
    ):
        value = metadata.get(key)
        if isinstance(value, dict):
            selected[key] = value.get(roi, value)
    return selected


def _write_method_trace_csv(
    path: str,
    *,
    method: str,
    record: dict[str, Any],
    chunk: Chunk,
) -> None:
    time_sec = np.asarray(record["time_sec"], dtype=float).reshape(-1)
    sig_raw = np.asarray(record["sig_raw"], dtype=float).reshape(-1)
    uv_raw = np.asarray(record["uv_raw"], dtype=float).reshape(-1)
    fit_ref = (
        np.asarray(chunk.uv_fit[:, 0], dtype=float).reshape(-1)
        if chunk.uv_fit is not None
        else np.full_like(time_sec, np.nan)
    )
    delta_f = (
        np.asarray(chunk.delta_f[:, 0], dtype=float).reshape(-1)
        if chunk.delta_f is not None
        else np.full_like(time_sec, np.nan)
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "preview_only",
                "method",
                "time_sec",
                "sig_raw",
                "uv_raw",
                "fit_ref",
                "delta_f",
            ],
        )
        writer.writeheader()
        for idx in range(len(time_sec)):
            writer.writerow(
                {
                    "preview_only": True,
                    "method": method,
                    "time_sec": float(time_sec[idx]),
                    "sig_raw": float(sig_raw[idx]),
                    "uv_raw": float(uv_raw[idx]),
                    "fit_ref": float(fit_ref[idx]) if np.isfinite(fit_ref[idx]) else "",
                    "delta_f": float(delta_f[idx]) if np.isfinite(delta_f[idx]) else "",
                }
            )


def _run_preview_method(
    *,
    method: str,
    base_cfg: Config,
    record: dict[str, Any],
    roi: str,
    preview_dir: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    diagnostics_path = os.path.join(
        preview_dir,
        METHOD_DIAGNOSTICS_FILENAME_TEMPLATE.format(method=method),
    )
    trace_path = os.path.join(preview_dir, METHOD_TRACE_FILENAME_TEMPLATE.format(method=method))
    artifacts: dict[str, str] = {"diagnostics_json": diagnostics_path}
    diagnostics: dict[str, Any] = {
        "preview_only": True,
        "strategy_recommendation": None,
        "method": method,
        "status": "failed",
        "warnings": [],
        "errors": [],
    }
    try:
        cfg = _config_for_method(base_cfg, method)
        chunk = _make_preview_chunk(record, roi)
        chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, cfg)
        chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, cfg)
        uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic")
        if uv_fit is None or delta_f is None:
            raise GuidedCorrectionPreviewError(f"{method} did not produce fit_ref and delta_f.")
        chunk.uv_fit = uv_fit
        chunk.delta_f = delta_f
        _write_method_trace_csv(trace_path, method=method, record=record, chunk=chunk)
        artifacts["trace_csv"] = trace_path
        diagnostics.update(
            {
                "status": "success",
                "dynamic_fit_mode": method,
                "n_samples": int(len(chunk.time_sec)),
                "sig_raw_summary": _numeric_summary(chunk.sig_raw[:, 0]),
                "uv_raw_summary": _numeric_summary(chunk.uv_raw[:, 0]),
                "fit_ref_summary": _numeric_summary(chunk.uv_fit[:, 0]),
                "delta_f_summary": _numeric_summary(chunk.delta_f[:, 0]),
                "dff_summary": {"available": False, "reason": "dff normalization not run in Stage 4C2 preview"},
                "dynamic_fit_metadata": _method_metadata_for_roi(chunk, roi),
                "artifact_paths": dict(artifacts),
            }
        )
    except Exception as exc:
        diagnostics["errors"] = [str(exc)]
    _write_json(diagnostics_path, diagnostics)
    return diagnostics, artifacts


def build_preview_provenance(
    *,
    preview_id: str,
    source_type: str,
    preview_output_dir: str | os.PathLike[str],
    selected_roi: str,
    selected_chunk: int | str | None = None,
    selected_window: str | None = None,
    correction_methods_compared: Iterable[str] = GUIDED_REFERENCE_PREVIEW_METHODS,
    backend_method_values: dict[str, Any] | None = None,
    config_values: dict[str, Any] | None = None,
    config_source_path: str | os.PathLike[str] | None = None,
    completed_run_dir: str | os.PathLike[str] | None = None,
    phasic_out: str | os.PathLike[str] | None = None,
    phasic_trace_cache_path: str | os.PathLike[str] | None = None,
    source_paths: Iterable[str | os.PathLike[str]] | None = None,
    source_artifact_hashes: dict[str, str] | None = None,
    diagnostic_cache_metadata: dict[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build preview-only provenance in memory; does not write files."""
    if not _safe_preview_id_component(preview_id):
        raise ValueError(f"Unsafe preview_id: {preview_id!r}")
    if source_type not in VALID_PREVIEW_SOURCE_TYPES:
        raise ValueError(f"Unsupported preview source_type: {source_type}")
    method_result = validate_preview_methods(correction_methods_compared)
    if not method_result.ok:
        raise ValueError(method_result.reason)
    return {
        "preview_only": True,
        "preview_id": str(preview_id),
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "source_type": str(source_type),
        "source_paths": [_resolve_path(path) for path in (source_paths or [])],
        "completed_run_dir": _resolve_path(completed_run_dir),
        "phasic_out": _resolve_path(phasic_out),
        "phasic_trace_cache_path": _resolve_path(phasic_trace_cache_path),
        "preview_output_dir": _resolve_path(preview_output_dir),
        "selected_roi": str(selected_roi),
        "selected_chunk": None if selected_chunk is None else selected_chunk,
        "selected_window": selected_window or "",
        "correction_methods_compared": list(method_result.methods),
        "backend_method_values": dict(backend_method_values or {}),
        "config_values": dict(config_values or {}),
        "config_source_path": _resolve_path(config_source_path),
        "source_artifact_hashes": dict(source_artifact_hashes or {}),
        "diagnostic_cache": dict(diagnostic_cache_metadata or {}),
        "pipeline_run_executed": False,
        "manifest_written": False,
        "applied_dff_routed": False,
        "feature_extraction_run": False,
        "production_output": False,
        "strategy_recommendation": None,
        "warning": PREVIEW_WARNING_TEXT,
    }


def build_preview_summary(
    *,
    preview_id: str,
    status: str = "not_run",
    method_statuses: dict[str, Any] | None = None,
    warnings: Iterable[str] | None = None,
    errors: Iterable[str] | None = None,
    generated_artifact_paths: dict[str, str] | None = None,
    stale: bool = False,
) -> dict[str, Any]:
    """Build preview-only summary in memory; does not write files."""
    if not _safe_preview_id_component(preview_id):
        raise ValueError(f"Unsafe preview_id: {preview_id!r}")
    if status not in {"success", "partial", "failed", "not_run"}:
        raise ValueError(f"Unsupported preview status: {status}")
    return {
        "preview_id": str(preview_id),
        "status": str(status),
        "method_statuses": dict(method_statuses or {}),
        "warnings": [str(x) for x in (warnings or [])],
        "errors": [str(x) for x in (errors or [])],
        "generated_artifact_paths": dict(generated_artifact_paths or {}),
        "stale": bool(stale),
    }


def run_guided_correction_preview_comparison(
    source: str | os.PathLike[str],
    preview_output_dir: str | os.PathLike[str] | None = None,
    *,
    roi: str,
    chunk_index: int | None = None,
    window: Any = None,
    methods: Iterable[str] | None = None,
    preview_id: str | None = None,
    source_type: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate preview-only correction comparison artifacts for one ROI/chunk.

    Supported sources are a successful completed-run directory or a direct
    ``phasic_out`` directory. The source cache/config are opened read-only, the
    requested methods are run on cloned in-memory chunks, and only preview
    artifacts are written under a validated preview namespace.
    """
    pid = preview_id or make_guided_preview_id("correction_preview")
    if not _safe_preview_id_component(pid):
        return _result_failed(preview_id=str(pid), errors=[f"Unsafe preview_id: {pid!r}"])

    source_result = _resolve_preview_source(source, source_type)
    if not source_result.ok:
        return _result_failed(preview_id=pid, errors=[source_result.reason])

    method_result = validate_preview_methods(methods or GUIDED_REFERENCE_PREVIEW_METHODS)
    if not method_result.ok:
        return _result_failed(preview_id=pid, errors=[method_result.reason])

    if preview_output_dir is None:
        if source_result.source_type != "completed_run" or not source_result.completed_run_dir:
            return _result_failed(
                preview_id=pid,
                errors=["preview_output_dir is required for direct phasic-cache preview sources."],
            )
        preview_dir = os.path.join(
            source_result.completed_run_dir,
            "_guided_workflow",
            "previews",
            pid,
        )
    else:
        preview_dir = _resolve_path(preview_output_dir)

    applied_root = os.path.join(source_result.phasic_out, "applied_dff")
    output_result = validate_guided_preview_output_dir(
        preview_dir,
        completed_run_dir=source_result.completed_run_dir or None,
        phasic_out=source_result.phasic_out,
        applied_dff_roots=[applied_root],
    )
    if not output_result.ok:
        return _result_failed(preview_id=pid, preview_output_dir=preview_dir, errors=[output_result.reason])
    preview_dir = output_result.resolved_path

    if os.path.exists(preview_dir) and not overwrite:
        return _result_failed(
            preview_id=pid,
            preview_output_dir=preview_dir,
            errors=["Preview output directory already exists. Pass overwrite=True to replace it."],
        )
    if os.path.exists(preview_dir) and overwrite and not _is_preview_scoped_leaf(preview_dir, pid):
        return _result_failed(
            preview_id=pid,
            preview_output_dir=preview_dir,
            errors=[
                (
                    "Existing preview output directories may only be overwritten when they are "
                    "the exact safe leaf <root>/_guided_workflow/previews/<preview_id>."
                )
            ],
        )

    try:
        base_cfg = Config.from_yaml(source_result.config_path)
    except Exception as exc:
        return _result_failed(preview_id=pid, preview_output_dir=preview_dir, errors=[str(exc)])

    try:
        with open_phasic_cache(source_result.phasic_trace_cache_path) as cache:
            record = _load_preview_chunk_record(
                cache,
                roi=str(roi),
                chunk_index=chunk_index,
                window=window,
                cfg=base_cfg,
            )
    except Exception as exc:
        return _result_failed(preview_id=pid, preview_output_dir=preview_dir, errors=[str(exc)])

    if os.path.exists(preview_dir):
        shutil.rmtree(preview_dir)
    os.makedirs(preview_dir, exist_ok=False)

    generated_artifacts: dict[str, str] = {}
    method_statuses: dict[str, Any] = {}
    warnings: list[str] = []
    errors: list[str] = []

    for method in method_result.methods:
        diagnostics, artifacts = _run_preview_method(
            method=method,
            base_cfg=base_cfg,
            record=record,
            roi=str(roi),
            preview_dir=preview_dir,
        )
        method_statuses[method] = {
            "status": diagnostics.get("status", "failed"),
            "errors": diagnostics.get("errors", []),
            "warnings": diagnostics.get("warnings", []),
            "diagnostics_json": artifacts.get("diagnostics_json", ""),
            "trace_csv": artifacts.get("trace_csv", ""),
            "strategy_recommendation": None,
        }
        generated_artifacts[f"{method}_diagnostics_json"] = artifacts.get("diagnostics_json", "")
        if artifacts.get("trace_csv"):
            generated_artifacts[f"{method}_trace_csv"] = artifacts["trace_csv"]
        if diagnostics.get("status") != "success":
            errors.extend(str(x) for x in diagnostics.get("errors", []))

    success_count = sum(1 for status in method_statuses.values() if status.get("status") == "success")
    if success_count == len(method_statuses) and method_statuses:
        status = "success"
        ok = True
    elif success_count > 0:
        status = "partial"
        ok = False
    else:
        status = "failed"
        ok = False

    if window is None:
        selected_window = ""
    else:
        normalized_window = _normalize_window(window)
        selected_window = (
            ""
            if normalized_window is None
            else f"{normalized_window[0]:.9g}-{normalized_window[1]:.9g}"
        )
    source_hashes = {
        "phasic_trace_cache.h5": _file_sha256(source_result.phasic_trace_cache_path),
        "config_used.yaml": _file_sha256(source_result.config_path),
    }
    provenance = build_preview_provenance(
        preview_id=pid,
        source_type=source_result.source_type,
        completed_run_dir=source_result.completed_run_dir,
        phasic_out=source_result.phasic_out,
        phasic_trace_cache_path=source_result.phasic_trace_cache_path,
        preview_output_dir=preview_dir,
        selected_roi=str(roi),
        selected_chunk=int(record["chunk_id"]),
        selected_window=selected_window,
        correction_methods_compared=method_result.methods,
        backend_method_values={method: {"dynamic_fit_mode": method} for method in method_result.methods},
        config_values={
            "dynamic_fit_mode": base_cfg.dynamic_fit_mode,
            "lowpass_hz": base_cfg.lowpass_hz,
            "filter_order": base_cfg.filter_order,
            "target_fs_hz": base_cfg.target_fs_hz,
            "baseline_subtract_before_fit": base_cfg.baseline_subtract_before_fit,
            "bleach_correction_mode": base_cfg.bleach_correction_mode,
        },
        config_source_path=source_result.config_path,
        source_paths=[source],
        source_artifact_hashes=source_hashes,
        diagnostic_cache_metadata=source_result.diagnostic_cache_metadata,
    )
    provenance["source_file"] = str(record["source_file"])
    provenance["selected_window_tuple"] = _json_safe(record.get("window"))
    provenance_path = os.path.join(preview_dir, PREVIEW_PROVENANCE_FILENAME)
    _write_json(provenance_path, provenance)
    generated_artifacts["preview_provenance_json"] = provenance_path

    summary = build_preview_summary(
        preview_id=pid,
        status=status,
        method_statuses=method_statuses,
        warnings=warnings,
        errors=errors,
        generated_artifact_paths=generated_artifacts,
        stale=False,
    )
    summary["preview_only"] = True
    summary["strategy_recommendation"] = None
    summary["comparison_plot"] = {"implemented": False, "reason": "comparison plot deferred in Stage 4C2"}
    summary_path = os.path.join(preview_dir, PREVIEW_SUMMARY_FILENAME)
    _write_json(summary_path, summary)
    generated_artifacts["preview_summary_json"] = summary_path

    return {
        "ok": ok,
        "preview_id": pid,
        "status": status,
        "preview_output_dir": preview_dir,
        "preview_provenance_path": provenance_path,
        "preview_summary_path": summary_path,
        "generated_artifacts": generated_artifacts,
        "method_statuses": method_statuses,
        "warnings": warnings,
        "errors": errors,
    }


def run_guided_local_correction_preview(
    source_file: str | os.PathLike[str],
    preview_output_dir: str | os.PathLike[str],
    *,
    roi: str,
    chunk_index: int,
    adapter_chunk_index: int = 0,
    segment_label: str = "",
    input_format: str,
    config_path: str | os.PathLike[str],
    methods: Iterable[str] | None = None,
    preview_id: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run preview-only correction methods on one raw source segment.

    This pathway intentionally does not accept or produce a phasic cache. It
    loads one discovered source session, extracts one ROI, and writes only
    preview-scoped traces and provenance.
    """
    pid = preview_id or make_guided_preview_id("local_correction_preview")
    source_path = _resolve_path(source_file)
    output_dir = _resolve_path(preview_output_dir)
    config_source = _resolve_path(config_path)
    local_context = {
        "selected_segment_label": str(segment_label or chunk_index),
        "selected_segment_index": int(chunk_index),
        "source_path": source_path,
        "adapter_local_chunk_id": int(adapter_chunk_index),
        "input_format": str(input_format).strip().lower(),
    }

    def failed(message: str) -> dict[str, Any]:
        result = _result_failed(
            preview_id=pid,
            preview_output_dir=output_dir,
            errors=[str(message)],
        )
        result.update(
            {
                "source_type": "local_raw_segment",
                "preview_only": True,
                "production_analysis": False,
                "roi": str(roi),
                "chunk_index": int(chunk_index),
                "preview_segment_label": str(
                    segment_label or chunk_index
                ),
                "adapter_local_chunk_id": int(adapter_chunk_index),
                "source_file": source_path,
                "local_preview_diagnostics": {
                    **local_context,
                    "adapter_error": str(message),
                },
            }
        )
        return result

    if not _safe_preview_id_component(pid):
        return failed(f"Unsafe preview_id: {pid!r}")
    method_result = validate_preview_methods(methods or GUIDED_REFERENCE_PREVIEW_METHODS)
    if not method_result.ok:
        return failed(method_result.reason)
    if not os.path.isfile(source_path):
        return failed("Selected preview segment is unavailable.")
    if not os.path.isfile(config_source):
        return failed("Correction configuration is unavailable.")
    if os.path.exists(output_dir):
        return failed("Local preview output directory already exists.")

    try:
        base_cfg = Config.from_yaml(config_source)
        applied_config_overrides = {
            str(key): value
            for key, value in (config_overrides or {}).items()
            if str(key) in Config.__dataclass_fields__
        }
        if applied_config_overrides:
            config_values = dataclasses.asdict(base_cfg)
            config_values.update(applied_config_overrides)
            base_cfg = Config(**config_values)
        raw_chunk = load_chunk(
            source_path,
            str(input_format).strip().lower(),
            base_cfg,
            int(adapter_chunk_index),
        )
        if roi not in raw_chunk.channel_names:
            raise GuidedCorrectionPreviewError(
                f"Requested ROI '{roi}' is not present in the selected preview segment."
            )
        roi_index = raw_chunk.channel_names.index(roi)
        time_sec = np.asarray(raw_chunk.time_sec, dtype=float).reshape(-1)
        segment_start = float(time_sec[0])
        segment_end = segment_start + float(base_cfg.chunk_duration_sec)
        segment_mask = (time_sec >= segment_start) & (time_sec < segment_end)
        if int(np.count_nonzero(segment_mask)) < 3:
            raise GuidedCorrectionPreviewError(
                "Selected preview segment contains too few samples."
            )
        time_sec = time_sec[segment_mask] - segment_start
        record = {
            "roi": str(roi),
            "chunk_id": int(adapter_chunk_index),
            "source_file": source_path,
            "time_sec": time_sec,
            "sig_raw": np.asarray(
                raw_chunk.sig_raw[:, roi_index], dtype=float
            ).reshape(-1)[segment_mask],
            "uv_raw": np.asarray(
                raw_chunk.uv_raw[:, roi_index], dtype=float
            ).reshape(-1)[segment_mask],
            "fs_hz": float(raw_chunk.fs_hz),
            "window": None,
            "metadata": dict(raw_chunk.metadata or {}),
        }
    except Exception as exc:
        return failed(f"{type(exc).__name__}: {exc}")

    os.makedirs(output_dir, exist_ok=False)
    generated_artifacts: dict[str, str] = {}
    method_statuses: dict[str, Any] = {}
    errors: list[str] = []
    for method in method_result.methods:
        diagnostics, artifacts = _run_preview_method(
            method=method,
            base_cfg=base_cfg,
            record=record,
            roi=str(roi),
            preview_dir=output_dir,
        )
        method_statuses[method] = {
            "status": diagnostics.get("status", "failed"),
            "errors": diagnostics.get("errors", []),
            "warnings": diagnostics.get("warnings", []),
            "diagnostics_json": artifacts.get("diagnostics_json", ""),
            "trace_csv": artifacts.get("trace_csv", ""),
            "strategy_recommendation": None,
        }
        generated_artifacts[f"{method}_diagnostics_json"] = artifacts.get(
            "diagnostics_json", ""
        )
        if artifacts.get("trace_csv"):
            generated_artifacts[f"{method}_trace_csv"] = artifacts["trace_csv"]
        if diagnostics.get("status") != "success":
            errors.extend(str(value) for value in diagnostics.get("errors", []))

    success_count = sum(
        status.get("status") == "success" for status in method_statuses.values()
    )
    status = (
        "success"
        if success_count == len(method_statuses) and method_statuses
        else "partial"
        if success_count
        else "failed"
    )
    provenance = {
        "preview_only": True,
        "production_analysis": False,
        "source_type": "local_raw_segment",
        "preview_id": pid,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_preview_scope": "single_discovered_session_single_roi",
        "selected_roi": str(roi),
        "selected_chunk": int(chunk_index),
        "selected_segment_index": int(chunk_index),
        "selected_segment_label": str(segment_label or chunk_index),
        "adapter_local_chunk_id": int(adapter_chunk_index),
        "source_file": source_path,
        "source_file_sha256": _file_sha256(source_path),
        "input_format": str(input_format).strip().lower(),
        "sample_count": int(time_sec.size),
        "time_bounds_sec": [
            float(time_sec[0]) if time_sec.size else None,
            float(time_sec[-1]) if time_sec.size else None,
        ],
        "padding_sec": 0.0,
        "correction_methods_compared": list(method_result.methods),
        "config_source_path": config_source,
        "config_sha256": _file_sha256(config_source),
        "config_overrides": _json_safe(applied_config_overrides),
        "effective_config_values": {
            "target_fs_hz": float(base_cfg.target_fs_hz),
            "chunk_duration_sec": float(base_cfg.chunk_duration_sec),
            "lowpass_hz": float(base_cfg.lowpass_hz),
            "filter_order": int(base_cfg.filter_order),
        },
        "baseline_scope": "not_computed_delta_f_preview",
        "reference_fit_scope": "selected_session",
        "pipeline_run_executed": False,
        "feature_extraction_run": False,
        "strategy_recommendation": None,
        "warning": (
            "Local correction preview for decision support only. Final analysis "
            "recomputes correction using the full selected recordings."
        ),
    }
    provenance_path = os.path.join(output_dir, PREVIEW_PROVENANCE_FILENAME)
    _write_json(provenance_path, provenance)
    generated_artifacts["preview_provenance_json"] = provenance_path
    summary = build_preview_summary(
        preview_id=pid,
        status=status,
        method_statuses=method_statuses,
        errors=errors,
        generated_artifact_paths=generated_artifacts,
    )
    summary.update(
        {
            "preview_only": True,
            "production_analysis": False,
            "source_type": "local_raw_segment",
            "strategy_recommendation": None,
        }
    )
    summary_path = os.path.join(output_dir, PREVIEW_SUMMARY_FILENAME)
    _write_json(summary_path, summary)
    generated_artifacts["preview_summary_json"] = summary_path
    return {
        "ok": status == "success",
        "preview_id": pid,
        "status": status,
        "source_type": "local_raw_segment",
        "preview_only": True,
        "production_analysis": False,
        "preview_output_dir": output_dir,
        "preview_provenance_path": provenance_path,
        "preview_summary_path": summary_path,
        "generated_artifacts": generated_artifacts,
        "method_statuses": method_statuses,
        "warnings": [],
        "errors": errors,
        "roi": str(roi),
        "chunk_index": int(chunk_index),
        "preview_segment_label": str(segment_label or chunk_index),
        "adapter_local_chunk_id": int(adapter_chunk_index),
        "local_preview_diagnostics": {
            **local_context,
            "adapter_error": "",
        },
        "source_file": source_path,
    }
