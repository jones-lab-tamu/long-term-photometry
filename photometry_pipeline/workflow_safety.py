"""Shared workflow validation helpers for GUI and Guided planning.

These helpers are intentionally UI-free. They return structured results so
callers can decide whether to block, warn, or display additional context.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from photometry_pipeline.config import Config


CONTINUOUS_NPM_UNSUPPORTED_MESSAGE = (
    "Continuous acquisition mode is not yet implemented for NPM/interleaved inputs."
)
CONTINUOUS_AUTO_FORMAT_MESSAGE = (
    "Continuous mode with format auto is ambiguous for mixed/unknown inputs. "
    "Use format rwd or format custom_tabular."
)

WRITE_OPERATION_KINDS = {
    "production_run",
    "diagnostic_cache",
    "preview_artifact",
    "plan_export",
}


@dataclass(frozen=True)
class WorkflowValidationResult:
    ok: bool
    code: str
    message: str
    warnings: tuple[str, ...] = ()
    resolved_format: str | None = None


@dataclass(frozen=True)
class FeatureEventDefaultsResult:
    defaults: dict[str, Any]
    baseline_source_path: str
    baseline_source_kind: str
    warnings: tuple[str, ...] = ()
    fallback_reason: str = ""


def _resolve_path_text(path: str | os.PathLike[str] | None) -> str:
    if path is None:
        return ""
    text = str(path).strip()
    if not text:
        return ""
    return os.path.normcase(os.path.abspath(os.path.normpath(text)))


def _same_or_child(path: str, root: str) -> bool:
    if not path or not root:
        return False
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _display_path(path: str) -> str:
    return path or "(not set)"


def validate_output_write_safety(
    *,
    source_root: str | os.PathLike[str] | None = None,
    output_base: str | os.PathLike[str] | None = None,
    target_path: str | os.PathLike[str] | None = None,
    operation_kind: str,
    allow_existing_target: bool = False,
    overwrite: bool = False,
    protected_roots: list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...] = (),
) -> WorkflowValidationResult:
    """Validate path relationships for workflow write operations."""

    kind = str(operation_kind or "").strip()
    if kind == "read_only_review":
        return WorkflowValidationResult(
            ok=True,
            code="read_only_review",
            message="Read-only review does not require output path validation.",
        )
    if kind not in WRITE_OPERATION_KINDS:
        return WorkflowValidationResult(
            ok=False,
            code="invalid_operation_kind",
            message=f"Unsupported operation kind for output safety validation: {operation_kind!r}.",
        )

    source = _resolve_path_text(source_root)
    base = _resolve_path_text(output_base)
    target = _resolve_path_text(target_path)
    if not target:
        return WorkflowValidationResult(
            ok=False,
            code="target_missing",
            message="Output target path is required for write operations.",
        )
    if not base:
        base = _resolve_path_text(Path(target).parent)

    if source and target == source:
        return WorkflowValidationResult(
            ok=False,
            code="target_equals_source",
            message=f"Output target must not be the same as the source path: {_display_path(target)}.",
        )
    if source and _same_or_child(target, source):
        return WorkflowValidationResult(
            ok=False,
            code="target_inside_source",
            message=(
                "Output target must not be inside the source/input folder: "
                f"target={_display_path(target)} source={_display_path(source)}."
            ),
        )
    if source and base and _same_or_child(source, base):
        return WorkflowValidationResult(
            ok=False,
            code="source_inside_output_base",
            message=(
                "Source/input folder must not be inside the output base for this write operation: "
                f"source={_display_path(source)} output_base={_display_path(base)}."
            ),
        )
    if source and _same_or_child(source, target):
        return WorkflowValidationResult(
            ok=False,
            code="source_inside_target",
            message=(
                "Source/input folder must not be inside the output target: "
                f"source={_display_path(source)} target={_display_path(target)}."
            ),
        )

    for protected in protected_roots or ():
        root = _resolve_path_text(protected)
        if root and _same_or_child(target, root):
            return WorkflowValidationResult(
                ok=False,
                code="target_inside_protected_root",
                message=(
                    "Output target must not be inside a protected output/source root: "
                    f"target={_display_path(target)} protected_root={_display_path(root)}."
                ),
            )

    warnings: list[str] = []
    parent = os.path.dirname(target)
    if parent and not os.path.isdir(parent):
        if base and os.path.isdir(base) and _same_or_child(parent, base):
            warnings.append(
                f"Output target parent will need to be created under output base: {_display_path(parent)}."
            )
        else:
            return WorkflowValidationResult(
                ok=False,
                code="target_parent_missing",
                message=f"Output target parent directory does not exist: {_display_path(parent)}.",
            )
    writable_check_path = parent if parent and os.path.isdir(parent) else base
    if writable_check_path and not os.access(writable_check_path, os.W_OK):
        return WorkflowValidationResult(
            ok=False,
            code="target_parent_not_writable",
            message=f"Output target parent directory is not writable: {_display_path(writable_check_path)}.",
        )

    if os.path.exists(target) and not allow_existing_target:
        return WorkflowValidationResult(
            ok=False,
            code="target_exists",
            message=f"Output target already exists: {_display_path(target)}.",
        )

    if os.path.exists(target) and overwrite:
        warnings.append(f"Existing output target may be overwritten: {_display_path(target)}.")

    return WorkflowValidationResult(
        ok=True,
        code="ok",
        message="Output path safety checks passed.",
        warnings=tuple(warnings),
    )


def validate_format_mode_compatibility(
    *,
    input_format: str,
    acquisition_mode: str,
    resolved_format: str | None = None,
) -> WorkflowValidationResult:
    """Validate format/acquisition-mode compatibility before launching work."""

    fmt = str(input_format or "").strip().lower()
    mode = str(acquisition_mode or "").strip().lower() or "intermittent"
    if mode not in {"intermittent", "continuous"}:
        return WorkflowValidationResult(
            ok=False,
            code="invalid_acquisition_mode",
            message="Acquisition mode must be intermittent or continuous.",
        )
    if fmt not in {"auto", "rwd", "npm", "custom_tabular"}:
        return WorkflowValidationResult(
            ok=False,
            code="invalid_format",
            message=f"Invalid format: {input_format!r}.",
        )
    if mode == "intermittent":
        return WorkflowValidationResult(
            ok=True,
            code="ok",
            message="Format/acquisition mode compatibility checks passed.",
            resolved_format=fmt if fmt != "auto" else None,
        )

    candidate = str(resolved_format or fmt).strip().lower()
    if candidate == "npm":
        return WorkflowValidationResult(
            ok=False,
            code="continuous_npm_unsupported",
            message=CONTINUOUS_NPM_UNSUPPORTED_MESSAGE,
        )
    if candidate == "auto":
        return WorkflowValidationResult(
            ok=False,
            code="continuous_auto_ambiguous",
            message=CONTINUOUS_AUTO_FORMAT_MESSAGE,
        )
    if candidate in {"rwd", "custom_tabular"}:
        return WorkflowValidationResult(
            ok=True,
            code="ok",
            message="Format/acquisition mode compatibility checks passed.",
            resolved_format=candidate,
        )
    return WorkflowValidationResult(
        ok=False,
        code="continuous_format_unsupported",
        message=f"Continuous acquisition mode is unsupported for format {candidate!r}.",
    )


def feature_event_defaults_from_config(cfg: Config) -> dict[str, Any]:
    """Return feature/event defaults from an already resolved Config object."""

    return {
        "event_signal": cfg.event_signal,
        "signal_excursion_polarity": getattr(cfg, "signal_excursion_polarity", "positive"),
        "peak_threshold_method": cfg.peak_threshold_method,
        "peak_threshold_k": cfg.peak_threshold_k,
        "peak_threshold_percentile": cfg.peak_threshold_percentile,
        "peak_threshold_abs": cfg.peak_threshold_abs,
        "peak_min_distance_sec": cfg.peak_min_distance_sec,
        "peak_min_prominence_k": getattr(cfg, "peak_min_prominence_k", 0.0),
        "peak_min_width_sec": getattr(cfg, "peak_min_width_sec", 0.0),
        "peak_pre_filter": getattr(cfg, "peak_pre_filter", "none"),
        "event_auc_baseline": cfg.event_auc_baseline,
    }


def resolve_feature_event_defaults(
    *,
    config_source_path: str | os.PathLike[str] | None = None,
    baseline_source_kind: str = "unknown",
    fallback_config: Config | None = None,
    allow_fallback: bool = True,
) -> FeatureEventDefaultsResult:
    """Resolve feature/event defaults and provenance from an active baseline."""

    source_path = str(config_source_path or "").strip()
    warnings: list[str] = []
    cfg: Config | None = None
    fallback_reason = ""
    if source_path and os.path.isfile(source_path):
        try:
            cfg = Config.from_yaml(source_path)
        except Exception as exc:
            fallback_reason = f"Unable to load baseline config {source_path}: {exc}"
    elif source_path:
        fallback_reason = f"Baseline config path does not exist: {source_path}"

    if cfg is None:
        if not allow_fallback or fallback_config is None:
            reason = fallback_reason or "No baseline config source was available."
            raise ValueError(reason)
        cfg = fallback_config
        if fallback_reason:
            warnings.append(fallback_reason)
        warnings.append("Using fallback Config object for feature/event defaults.")
        kind = "fallback"
    else:
        kind = str(baseline_source_kind or "unknown").strip() or "unknown"

    return FeatureEventDefaultsResult(
        defaults=feature_event_defaults_from_config(cfg),
        baseline_source_path=os.path.abspath(source_path) if source_path else "",
        baseline_source_kind=kind,
        warnings=tuple(warnings),
        fallback_reason=fallback_reason if kind == "fallback" else "",
    )
