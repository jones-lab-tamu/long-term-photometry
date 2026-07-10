"""Backend-only Signal-Only F0 diagnostic contract helpers.

Stage 4D diagnostics are evidence-only. These helpers define source resolution,
output namespace validation, IDs, and JSON-serializable provenance/summary
skeletons for a future diagnostic writer. They do not generate diagnostics,
write manifests, route applied-dF/F, run feature extraction, validate GUI setup,
or launch the pipeline.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from photometry_pipeline.guided_diagnostic_cache import resolve_diagnostic_cache_source
from photometry_pipeline.run_completion_contract import classify_run_terminal_state


GUIDED_WORKFLOW_DIR_NAME = "_guided_workflow"
SIGNAL_ONLY_F0_DIAGNOSTIC_DIR_NAME = "signal_only_f0_diagnostics"

SIGNAL_ONLY_F0_DIAGNOSTIC_PROVENANCE_FILENAME = (
    "signal_only_f0_diagnostic_provenance.json"
)
SIGNAL_ONLY_F0_DIAGNOSTIC_SUMMARY_FILENAME = "signal_only_f0_diagnostic_summary.json"

SOURCE_TYPE_COMPLETED_RUN = "completed_run"
SOURCE_TYPE_PHASIC_OUT_BACKEND_ONLY = "phasic_out_backend_only"
SOURCE_TYPE_DIAGNOSTIC_CACHE = "diagnostic_cache"
SUPPORTED_SOURCE_TYPES = {
    SOURCE_TYPE_COMPLETED_RUN,
    SOURCE_TYPE_PHASIC_OUT_BACKEND_ONLY,
    SOURCE_TYPE_DIAGNOSTIC_CACHE,
}

STATUS_NOT_RUN = "not_run"
STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
SUPPORTED_DIAGNOSTIC_STATUSES = {
    STATUS_NOT_RUN,
    STATUS_SUCCESS,
    STATUS_PARTIAL,
    STATUS_FAILED,
}

SIGNAL_ONLY_F0_DIAGNOSTIC_WARNING = (
    "Signal-Only F0 diagnostic evidence only; not a strategy selection, not a "
    "fallback, and not production applied-dF/F routing."
)


@dataclass(frozen=True)
class SignalOnlyF0DiagnosticSourceResult:
    ok: bool
    reason: str
    source_type: str
    completed_run_dir: str = ""
    phasic_out_dir: str = ""
    phasic_trace_cache_path: str = ""
    config_source_path: str = ""
    code: str = "ok"
    diagnostic_cache_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SignalOnlyF0DiagnosticOutputValidationResult:
    ok: bool
    reason: str
    resolved_output_dir: str = ""
    code: str = "ok"
    protected_namespace: str = ""


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


def _safe_id_component(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if "/" in text or "\\" in text or ".." in text:
        return False
    if Path(text).is_absolute() or Path(text).drive:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text))


def _is_successful_completed_run_dir(run_dir: str) -> tuple[bool, str]:
    """Backend-local completed-run success check; no GUI imports.

    Delegates to the single terminal-completion contract so this gate cannot
    drift from the completed-run loader and admit a run the loader would reject.
    """
    classification = classify_run_terminal_state(run_dir)
    return classification.is_success, classification.reason


def make_signal_only_f0_diagnostic_id(
    prefix: str = "signal_only_f0",
    *,
    now: datetime | None = None,
    token: str | None = None,
) -> str:
    """Return a filesystem-safe diagnostic id without creating directories."""
    prefix_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix or "signal_only_f0")).strip(
        "._-"
    )
    if not prefix_safe:
        prefix_safe = "signal_only_f0"
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    suffix = token if token is not None else secrets.token_hex(4)
    suffix_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(suffix)).strip("._-")
    if not suffix_safe:
        suffix_safe = secrets.token_hex(4)
    diagnostic_id = f"{prefix_safe}_{stamp}_{suffix_safe}"
    if not _safe_id_component(diagnostic_id):
        raise ValueError(f"generated unsafe Signal-Only F0 diagnostic_id: {diagnostic_id!r}")
    return diagnostic_id


def resolve_completed_run_signal_only_f0_source(
    completed_run_dir: str | os.PathLike[str] | None,
) -> SignalOnlyF0DiagnosticSourceResult:
    """Resolve completed-run diagnostic inputs without opening or mutating cache."""
    run_dir = _resolve_path(completed_run_dir)
    if not run_dir or not os.path.isdir(run_dir):
        return SignalOnlyF0DiagnosticSourceResult(
            ok=False,
            reason=f"Completed run directory does not exist: {completed_run_dir}",
            source_type=SOURCE_TYPE_COMPLETED_RUN,
            completed_run_dir=run_dir,
            code="completed_run_missing",
        )

    successful, evidence = _is_successful_completed_run_dir(run_dir)
    if not successful:
        return SignalOnlyF0DiagnosticSourceResult(
            ok=False,
            reason=evidence,
            source_type=SOURCE_TYPE_COMPLETED_RUN,
            completed_run_dir=run_dir,
            code="completed_run_not_successful",
        )

    phasic = os.path.join(run_dir, "_analysis", "phasic_out")
    cache = os.path.join(phasic, "phasic_trace_cache.h5")
    config = os.path.join(phasic, "config_used.yaml")
    if not os.path.isdir(phasic):
        return SignalOnlyF0DiagnosticSourceResult(
            ok=False,
            reason=f"Completed run is missing phasic output directory: {phasic}",
            source_type=SOURCE_TYPE_COMPLETED_RUN,
            completed_run_dir=run_dir,
            phasic_out_dir=phasic,
            code="phasic_out_missing",
        )
    if not os.path.isfile(cache):
        return SignalOnlyF0DiagnosticSourceResult(
            ok=False,
            reason=f"Completed run is missing phasic trace cache: {cache}",
            source_type=SOURCE_TYPE_COMPLETED_RUN,
            completed_run_dir=run_dir,
            phasic_out_dir=phasic,
            phasic_trace_cache_path=cache,
            code="phasic_trace_cache_missing",
        )
    if not os.path.isfile(config):
        return SignalOnlyF0DiagnosticSourceResult(
            ok=False,
            reason=f"Completed run is missing phasic config snapshot: {config}",
            source_type=SOURCE_TYPE_COMPLETED_RUN,
            completed_run_dir=run_dir,
            phasic_out_dir=phasic,
            phasic_trace_cache_path=cache,
            config_source_path=config,
            code="config_snapshot_missing",
        )
    return SignalOnlyF0DiagnosticSourceResult(
        ok=True,
        reason="completed run source resolved",
        source_type=SOURCE_TYPE_COMPLETED_RUN,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        phasic_trace_cache_path=cache,
        config_source_path=config,
        code="ok",
    )


def resolve_phasic_out_signal_only_f0_source(
    phasic_out_dir: str | os.PathLike[str] | None,
) -> SignalOnlyF0DiagnosticSourceResult:
    """Resolve a direct phasic_out source for backend-only diagnostics."""
    phasic = _resolve_path(phasic_out_dir)
    if not phasic or not os.path.isdir(phasic):
        return SignalOnlyF0DiagnosticSourceResult(
            ok=False,
            reason=f"phasic_out directory does not exist: {phasic_out_dir}",
            source_type=SOURCE_TYPE_PHASIC_OUT_BACKEND_ONLY,
            phasic_out_dir=phasic,
            code="phasic_out_missing",
        )
    cache = os.path.join(phasic, "phasic_trace_cache.h5")
    config = os.path.join(phasic, "config_used.yaml")
    if not os.path.isfile(cache):
        return SignalOnlyF0DiagnosticSourceResult(
            ok=False,
            reason=f"phasic_out is missing phasic trace cache: {cache}",
            source_type=SOURCE_TYPE_PHASIC_OUT_BACKEND_ONLY,
            phasic_out_dir=phasic,
            phasic_trace_cache_path=cache,
            code="phasic_trace_cache_missing",
        )
    if not os.path.isfile(config):
        return SignalOnlyF0DiagnosticSourceResult(
            ok=False,
            reason=f"phasic_out is missing config snapshot: {config}",
            source_type=SOURCE_TYPE_PHASIC_OUT_BACKEND_ONLY,
            phasic_out_dir=phasic,
            phasic_trace_cache_path=cache,
            config_source_path=config,
            code="config_snapshot_missing",
        )
    return SignalOnlyF0DiagnosticSourceResult(
        ok=True,
        reason="backend-only phasic_out source resolved",
        source_type=SOURCE_TYPE_PHASIC_OUT_BACKEND_ONLY,
        phasic_out_dir=phasic,
        phasic_trace_cache_path=cache,
        config_source_path=config,
        code="ok",
    )


def resolve_diagnostic_cache_signal_only_f0_source(
    source: str | os.PathLike[str] | Any,
) -> SignalOnlyF0DiagnosticSourceResult:
    """Resolve a Guided diagnostic-cache source for Signal-Only F0 review."""
    result = resolve_diagnostic_cache_source(source)
    if not result.ok or result.source is None:
        return SignalOnlyF0DiagnosticSourceResult(
            ok=False,
            reason=result.status.message,
            source_type=SOURCE_TYPE_DIAGNOSTIC_CACHE,
            code=result.status.code,
        )
    resolved = result.source
    if resolved.source_type != SOURCE_TYPE_DIAGNOSTIC_CACHE:
        return SignalOnlyF0DiagnosticSourceResult(
            ok=False,
            reason="Resolved Signal-Only F0 source is not a diagnostic cache.",
            source_type=SOURCE_TYPE_DIAGNOSTIC_CACHE,
            code="invalid_source_type",
        )
    phasic_out = os.path.dirname(resolved.phasic_trace_cache_path)
    metadata = {
        "source_type": SOURCE_TYPE_DIAGNOSTIC_CACHE,
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
    }
    return SignalOnlyF0DiagnosticSourceResult(
        ok=True,
        reason="diagnostic cache source resolved",
        source_type=SOURCE_TYPE_DIAGNOSTIC_CACHE,
        completed_run_dir="",
        phasic_out_dir=phasic_out,
        phasic_trace_cache_path=resolved.phasic_trace_cache_path,
        config_source_path=resolved.config_used_path,
        code="ok",
        diagnostic_cache_metadata=metadata,
    )


def build_default_signal_only_f0_diagnostic_output_dir(
    completed_run_dir: str | os.PathLike[str],
    diagnostic_id: str,
) -> Path:
    """Build the accepted completed-run diagnostic leaf path without creating it."""
    if not _safe_id_component(diagnostic_id):
        raise ValueError(f"Unsafe Signal-Only F0 diagnostic_id: {diagnostic_id!r}")
    return (
        Path(completed_run_dir)
        / GUIDED_WORKFLOW_DIR_NAME
        / SIGNAL_ONLY_F0_DIAGNOSTIC_DIR_NAME
        / diagnostic_id
    )


def build_default_signal_only_f0_diagnostic_cache_output_dir(
    cache_root_path: str | os.PathLike[str],
    diagnostic_id: str,
) -> Path:
    """Build the accepted diagnostic-cache review leaf path without creating it."""
    if not _safe_id_component(diagnostic_id):
        raise ValueError(f"Unsafe Signal-Only F0 diagnostic_id: {diagnostic_id!r}")
    return (
        Path(cache_root_path)
        / GUIDED_WORKFLOW_DIR_NAME
        / SIGNAL_ONLY_F0_DIAGNOSTIC_DIR_NAME
        / diagnostic_id
    )


def _is_diagnostic_scoped_leaf(path: str | os.PathLike[str], diagnostic_id: str) -> bool:
    resolved = _resolve_path(path)
    if not resolved or not _safe_id_component(diagnostic_id):
        return False
    parts = Path(resolved).parts
    if len(parts) < 3:
        return False
    return (
        parts[-1] == diagnostic_id
        and parts[-2] == SIGNAL_ONLY_F0_DIAGNOSTIC_DIR_NAME
        and parts[-3] == GUIDED_WORKFLOW_DIR_NAME
    )


def _reject_output(
    *,
    resolved_output_dir: str,
    code: str,
    reason: str,
    protected_namespace: str = "",
) -> SignalOnlyF0DiagnosticOutputValidationResult:
    return SignalOnlyF0DiagnosticOutputValidationResult(
        ok=False,
        reason=reason,
        resolved_output_dir=resolved_output_dir,
        code=code,
        protected_namespace=protected_namespace,
    )


def validate_signal_only_f0_diagnostic_output_dir(
    output_dir: str | os.PathLike[str] | None,
    *,
    completed_run_dir: str | os.PathLike[str] | None = None,
    phasic_out_dir: str | os.PathLike[str] | None = None,
    diagnostic_id: str | None = None,
    allow_existing: bool = False,
) -> SignalOnlyF0DiagnosticOutputValidationResult:
    """Validate a proposed diagnostic output directory without creating it."""
    resolved = _resolve_path(output_dir)
    if not resolved:
        return _reject_output(
            resolved_output_dir="",
            code="empty_output_dir",
            reason="Signal-Only F0 diagnostic output directory is empty.",
        )
    if os.path.isfile(resolved):
        return _reject_output(
            resolved_output_dir=resolved,
            code="output_dir_is_file",
            reason="Signal-Only F0 diagnostic output path resolves to an existing file.",
            protected_namespace=resolved,
        )

    completed = _resolve_path(completed_run_dir)
    phasic = _resolve_path(phasic_out_dir)
    if completed and _norm(resolved) == _norm(completed):
        return _reject_output(
            resolved_output_dir=resolved,
            code="output_is_completed_run",
            reason="Signal-Only F0 diagnostic output_dir must be separate from the completed run root.",
            protected_namespace=completed,
        )
    if phasic and _norm(resolved) == _norm(phasic):
        return _reject_output(
            resolved_output_dir=resolved,
            code="output_is_phasic_out",
            reason="Signal-Only F0 diagnostic output_dir must be separate from source phasic_out.",
            protected_namespace=phasic,
        )

    if phasic:
        cache = os.path.join(phasic, "phasic_trace_cache.h5")
        features = os.path.join(phasic, "features")
        legacy_features = os.path.join(features, "features.csv")
        applied = os.path.join(phasic, "applied_dff")
        if _is_equal_or_inside(resolved, features):
            return _reject_output(
                resolved_output_dir=resolved,
                code="inside_legacy_features",
                reason="Signal-Only F0 diagnostic output_dir must be separate from legacy features.",
                protected_namespace=features,
            )
        if _is_equal_or_inside(resolved, applied):
            return _reject_output(
                resolved_output_dir=resolved,
                code="inside_applied_dff",
                reason="Signal-Only F0 diagnostic output_dir must be separate from applied-dF/F production outputs.",
                protected_namespace=applied,
            )
        if _is_strictly_inside(resolved, phasic):
            return _reject_output(
                resolved_output_dir=resolved,
                code="inside_phasic_out",
                reason="Signal-Only F0 diagnostic output_dir must be separate from source phasic_out.",
                protected_namespace=phasic,
            )
        if os.path.exists(cache) and _is_equal_or_inside(cache, resolved):
            return _reject_output(
                resolved_output_dir=resolved,
                code="contains_phasic_trace_cache",
                reason="Signal-Only F0 diagnostic output_dir must not contain phasic_trace_cache.h5.",
                protected_namespace=cache,
            )
        if os.path.exists(legacy_features) and _is_equal_or_inside(legacy_features, resolved):
            return _reject_output(
                resolved_output_dir=resolved,
                code="contains_legacy_features",
                reason="Signal-Only F0 diagnostic output_dir must not contain legacy features.csv.",
                protected_namespace=legacy_features,
            )

    if completed and _is_strictly_inside(resolved, completed):
        scoped = bool(diagnostic_id) and _is_diagnostic_scoped_leaf(resolved, str(diagnostic_id))
        expected_parent = os.path.join(
            completed,
            GUIDED_WORKFLOW_DIR_NAME,
            SIGNAL_ONLY_F0_DIAGNOSTIC_DIR_NAME,
        )
        if not scoped or not _is_equal_or_inside(resolved, expected_parent):
            return _reject_output(
                resolved_output_dir=resolved,
                code="inside_completed_run_not_diagnostic_leaf",
                reason=(
                    "Signal-Only F0 diagnostic output inside a completed run must be the "
                    "exact diagnostic leaf under _guided_workflow/signal_only_f0_diagnostics."
                ),
                protected_namespace=completed,
            )

    if completed:
        if not diagnostic_id or not _safe_id_component(str(diagnostic_id)):
            return _reject_output(
                resolved_output_dir=resolved,
                code="missing_or_unsafe_diagnostic_id",
                reason=(
                    "A safe diagnostic_id is required when validating a completed-run "
                    "Signal-Only F0 diagnostic output directory."
                ),
                protected_namespace=completed,
            )
        expected = _resolve_path(
            build_default_signal_only_f0_diagnostic_output_dir(completed, str(diagnostic_id))
        )
        if _norm(resolved) != _norm(expected):
            return _reject_output(
                resolved_output_dir=resolved,
                code="outside_completed_run_diagnostic_leaf",
                reason=(
                    "Signal-Only F0 diagnostic output_dir must be the exact resolved "
                    "default diagnostic leaf under the provided completed_run_dir."
                ),
                protected_namespace=expected,
            )

    exists = os.path.exists(resolved)
    if exists and not allow_existing:
        return _reject_output(
            resolved_output_dir=resolved,
            code="output_dir_exists",
            reason="Signal-Only F0 diagnostic output directory already exists.",
            protected_namespace=resolved,
        )
    if exists and allow_existing:
        if not os.path.isdir(resolved):
            return _reject_output(
                resolved_output_dir=resolved,
                code="output_path_not_directory",
                reason="Existing Signal-Only F0 diagnostic output path is not a directory.",
                protected_namespace=resolved,
            )
        if not diagnostic_id or not _is_diagnostic_scoped_leaf(resolved, str(diagnostic_id)):
            return _reject_output(
                resolved_output_dir=resolved,
                code="existing_output_not_safe_leaf",
                reason=(
                    "Existing Signal-Only F0 diagnostic output directories may only be "
                    "overwritten when they are exact diagnostic leaves."
                ),
                protected_namespace=resolved,
            )

    return SignalOnlyF0DiagnosticOutputValidationResult(
        ok=True,
        reason="Signal-Only F0 diagnostic output directory is safe.",
        resolved_output_dir=resolved,
        code="ok",
    )


def build_signal_only_f0_diagnostic_provenance(
    *,
    diagnostic_id: str,
    source_type: str,
    created_at_utc: str | None = None,
    completed_run_dir: str | os.PathLike[str] | None = None,
    phasic_out_dir: str | os.PathLike[str] | None = None,
    phasic_trace_cache_path: str | os.PathLike[str] | None = None,
    config_source_path: str | os.PathLike[str] | None = None,
    selected_rois: list[str] | tuple[str, ...] | None = None,
    selected_chunks: list[int] | tuple[int, ...] | None = None,
    selected_window: dict[str, Any] | None = None,
    source_artifact_hashes: dict[str, str] | None = None,
    diagnostic_cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable provenance skeleton; does not write it."""
    if not _safe_id_component(diagnostic_id):
        raise ValueError(f"Unsafe Signal-Only F0 diagnostic_id: {diagnostic_id!r}")
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(f"Unsupported Signal-Only F0 diagnostic source_type: {source_type!r}")
    return {
        "diagnostic_only": True,
        "signal_only_f0": True,
        "diagnostic_id": diagnostic_id,
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_type": source_type,
        "completed_run_dir": _resolve_path(completed_run_dir),
        "phasic_out_dir": _resolve_path(phasic_out_dir),
        "phasic_trace_cache_path": _resolve_path(phasic_trace_cache_path),
        "config_source_path": _resolve_path(config_source_path),
        "selected_rois": list(selected_rois or []),
        "selected_chunks": [int(chunk) for chunk in (selected_chunks or [])],
        "selected_window": dict(selected_window or {}),
        "source_artifact_hashes": dict(source_artifact_hashes or {}),
        "diagnostic_cache": dict(diagnostic_cache_metadata or {}),
        "manifest_written": False,
        "applied_dff_routed": False,
        "production_applied_dff_output_written": False,
        "production_output": False,
        "feature_extraction_run": False,
        "pipeline_run_executed": False,
        "validation_run_executed": False,
        "strategy_recommendation": None,
        "warning": SIGNAL_ONLY_F0_DIAGNOSTIC_WARNING,
    }


def build_signal_only_f0_diagnostic_summary(
    *,
    diagnostic_id: str,
    status: str = STATUS_NOT_RUN,
    roi_statuses: dict[str, Any] | None = None,
    chunk_statuses: dict[str, Any] | None = None,
    warnings: list[str] | tuple[str, ...] | None = None,
    errors: list[str] | tuple[str, ...] | None = None,
    generated_artifact_paths: list[str] | tuple[str, ...] | None = None,
    diagnostic_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable diagnostic summary skeleton; does not write it."""
    if not _safe_id_component(diagnostic_id):
        raise ValueError(f"Unsafe Signal-Only F0 diagnostic_id: {diagnostic_id!r}")
    if status not in SUPPORTED_DIAGNOSTIC_STATUSES:
        raise ValueError(f"Unsupported Signal-Only F0 diagnostic status: {status!r}")
    return {
        "diagnostic_id": diagnostic_id,
        "status": status,
        "roi_statuses": dict(roi_statuses or {}),
        "chunk_statuses": dict(chunk_statuses or {}),
        "warnings": list(warnings or []),
        "errors": list(errors or []),
        "generated_artifact_paths": list(generated_artifact_paths or []),
        "diagnostic_metrics": dict(diagnostic_metrics or {}),
        "strategy_recommendation": None,
    }
