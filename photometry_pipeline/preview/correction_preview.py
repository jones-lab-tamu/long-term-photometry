"""Guided Workflow correction-preview backend contract helpers.

Stage 4C1 is intentionally non-executing. These helpers validate preview
namespaces, source availability, allowed reference-preview methods, and
preview-only provenance dictionaries. They do not create directories, load HDF5
chunks, run correction fitting, write manifests, route applied-dF/F, run feature
extraction, validate GUI setup, or launch the pipeline.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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
VALID_PREVIEW_SOURCE_TYPES = {"completed_run", "phasic_cache"}


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
    code: str = "ok"
    reason: str = ""


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


def run_guided_correction_preview_comparison(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Stage 4C2 placeholder; Stage 4C1 must not run correction recomputation."""
    raise NotImplementedError(
        "Guided correction preview comparison generation is not implemented in Stage 4C1."
    )
