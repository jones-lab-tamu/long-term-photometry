"""Compatibility validation helpers for historical Guided applied-dF/F runs."""

from pathlib import Path


class GuidedAppliedDffOrchestrationError(ValueError):
    """Raised when applied-dF/F compatibility validation fails."""
    pass


def build_guided_applied_dff_manifest_rows(
    strategy_map_payload: dict,
    applied_dff_root: Path,
    per_roi_feature_config_paths: dict | None = None,
) -> list[dict]:
    """
    Build the exact batch manifest rows for run_applied_dff_batch.py.

    Enforces the exactly-one-ROI rule before returning any rows.
    Raises GuidedAppliedDffOrchestrationError if validation fails.

    per_roi_feature_config_paths, if given, maps roi_id to the path of a
    feature-config file (as consumed by run_applied_dff_batch.py's
    `feature_config` manifest column) to use for that ROI instead of the
    batch's default feature config. A ROI absent from this mapping keeps
    today's behavior: an empty `feature_config` cell, which
    run_applied_dff_batch.py resolves to its own default.
    """
    if strategy_map_payload.get("production_strategy_map_version") != "per_roi_correction_strategy_map.v1":
        raise GuidedAppliedDffOrchestrationError("production_strategy_map_version must be exactly per_roi_correction_strategy_map.v1")

    included_roi_ids = set(strategy_map_payload.get("included_roi_ids", []))
    entries = strategy_map_payload.get("per_roi_production_strategy_map", [])

    if not included_roi_ids:
        raise GuidedAppliedDffOrchestrationError("Included ROI set cannot be empty.")

    map_roi_ids = [entry.get("roi_id") for entry in entries]
    
    # Check for missing ROIs
    missing_rois = included_roi_ids - set(map_roi_ids)
    if missing_rois:
        raise GuidedAppliedDffOrchestrationError(f"Missing ROIs in strategy map: {missing_rois}")
        
    # Check for duplicate ROIs
    if len(map_roi_ids) != len(set(map_roi_ids)):
        seen = set()
        duplicates = set(x for x in map_roi_ids if x in seen or seen.add(x))
        raise GuidedAppliedDffOrchestrationError(f"Duplicate ROIs in strategy map: {duplicates}")

    # Check for extra ROIs
    extra_rois = set(map_roi_ids) - included_roi_ids
    if extra_rois:
        raise GuidedAppliedDffOrchestrationError(f"Extra ROIs in strategy map not in included_roi_ids: {extra_rois}")

    rows = []
    output_names = set()
    output_dirs = set()

    from photometry_pipeline.guided_new_analysis_plan import FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES
    SUPPORTED_DYNAMIC_FIT_MODES = FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES

    used_dynamic_fit_modes = set()

    for entry in entries:
        roi_id = entry.get("roi_id")
        strategy_family = entry.get("strategy_family")
        explicit_user_mark = entry.get("explicit_user_mark")
        current_or_stale = entry.get("current_or_stale")
        selected_strategy = entry.get("selected_strategy")
        dynamic_fit_mode = entry.get("dynamic_fit_mode")
        
        if not roi_id:
            raise GuidedAppliedDffOrchestrationError("Strategy map entry missing roi_id.")

        if strategy_family not in ("dynamic_fit", "signal_only_f0"):
            raise GuidedAppliedDffOrchestrationError(f"Unsupported strategy_family: {strategy_family}")

        if not explicit_user_mark:
            raise GuidedAppliedDffOrchestrationError(f"Non-explicit entry for ROI {roi_id}")

        if current_or_stale != "current":
            raise GuidedAppliedDffOrchestrationError(f"Stale entry for ROI {roi_id}")

        if not selected_strategy:
            raise GuidedAppliedDffOrchestrationError(f"Missing selected_strategy for ROI {roi_id}")

        if strategy_family == "dynamic_fit":
            if selected_strategy not in SUPPORTED_DYNAMIC_FIT_MODES:
                raise GuidedAppliedDffOrchestrationError(f"Unsupported dynamic_fit selected_strategy: {selected_strategy}")
            if dynamic_fit_mode not in SUPPORTED_DYNAMIC_FIT_MODES:
                raise GuidedAppliedDffOrchestrationError(f"Unsupported dynamic_fit_mode: {dynamic_fit_mode}")
            if selected_strategy != dynamic_fit_mode:
                raise GuidedAppliedDffOrchestrationError(f"Mismatch between selected_strategy ({selected_strategy}) and dynamic_fit_mode ({dynamic_fit_mode}) for ROI {roi_id}")
            used_dynamic_fit_modes.add(dynamic_fit_mode)
            batch_strategy = "dynamic_fit"

        elif strategy_family == "signal_only_f0":
            if selected_strategy != "signal_only_f0":
                raise GuidedAppliedDffOrchestrationError(f"Unsupported signal_only_f0 selected_strategy: {selected_strategy}")
            if dynamic_fit_mode is not None:
                raise GuidedAppliedDffOrchestrationError(f"signal_only_f0 entry for ROI {roi_id} must have dynamic_fit_mode=None")
            batch_strategy = "signal_only_f0"

        # Sanitize ROI name to ensure valid output name (e.g. replace spaces with underscores)
        safe_roi = str(roi_id).replace(" ", "_").replace("/", "_").replace("\\", "_")
        output_name = f"{safe_roi}_{batch_strategy}"

        if output_name in output_names:
            raise GuidedAppliedDffOrchestrationError(f"Duplicate output_name generated: {output_name}")
        output_names.add(output_name)

        output_dir = (applied_dff_root / output_name).resolve()

        # Prevent escaping applied_dff root
        try:
            output_dir.relative_to(applied_dff_root.resolve())
        except ValueError:
            raise GuidedAppliedDffOrchestrationError(f"output_dir escapes applied_dff root: {output_dir}")

        if str(output_dir) in output_dirs:
            raise GuidedAppliedDffOrchestrationError(f"Duplicate output_dir generated: {output_dir}")
        output_dirs.add(str(output_dir))

        # Only provide the columns that _parse_csv_manifest expects
        row = {
            "roi": roi_id,
            "strategy": batch_strategy,
            "output_name": output_name,
            "feature_config": (
                str(per_roi_feature_config_paths[roi_id])
                if per_roi_feature_config_paths and roi_id in per_roi_feature_config_paths
                else ""
            ),
        }
        rows.append(row)

    if len(used_dynamic_fit_modes) > 1:
        raise GuidedAppliedDffOrchestrationError(
            "Mixed dynamic_fit modes cannot be executed by the current "
            "applied-dF/F batch manifest because manifest rows do not carry "
            f"a per-ROI dynamic-fit mode. Found: {used_dynamic_fit_modes}"
        )

    return rows
