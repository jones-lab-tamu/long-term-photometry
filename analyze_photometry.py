import argparse
import sys
import os
import json
import logging
from dataclasses import replace
from photometry_pipeline.config import Config
from photometry_pipeline.feature_event_config import FEATURE_EVENT_CONFIG_FIELDS
from photometry_pipeline.guided_startup_transaction import (
    GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME,
)
from photometry_pipeline.pipeline import Pipeline


class GuidedFeatureSettingsError(RuntimeError):
    """A Guided per-ROI feature-settings artifact is missing, incomplete, or
    inconsistent with the confirmed base configuration. Never fall back to
    baked defaults or to the global settings for a Custom ROI."""


def _feature_fields_of(config):
    return {name: getattr(config, name) for name in sorted(FEATURE_EVENT_CONFIG_FIELDS)}


def load_guided_per_roi_feature_settings(guided_candidate_manifest_path, base_config):
    """Load a Guided per-ROI feature-config artifact, if one was materialized.

    The artifact is a sibling of the Guided candidate manifest (both live in
    the Guided-allocated run directory; see guided_startup_materialization.py).
    Returns (None, None) when the manifest path is not set or no sibling
    artifact exists -- a Default-only Guided run (or a plain non-Guided CLI
    invocation) writes no artifact and analyzes every ROI with base_config.

    base_config already carries the feature-detection settings the user
    confirmed in Guided Step 5 (4J16k39a: they are serialized into
    config_effective.yaml from intent.feature_event.effective_values, not from
    baked contract defaults). Each Custom ROI's Config is therefore built from
    that same confirmed base, so unchanged fields are inherited from the
    confirmed Defaults rather than from execution defaults.

    Returns (per_roi_feature_config, per_roi_feature_provenance):
    - per_roi_feature_config: dict[str, Config] for Custom ROIs only. Each is
      base_config with that ROI's COMPLETE effective feature fields applied.
    - per_roi_feature_provenance: the artifact's per_roi_feature_provenance
      dict (every resolved ROI, Default and Custom).

    Fails closed (GuidedFeatureSettingsError) on an unknown override field, an
    incomplete effective configuration, or a Default entry whose effective
    settings disagree with the confirmed base configuration.
    """
    if not guided_candidate_manifest_path:
        return None, None
    guided_run_dir = os.path.dirname(
        os.path.abspath(guided_candidate_manifest_path)
    )
    path = os.path.join(guided_run_dir, GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME)
    if not os.path.isfile(path):
        return None, None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    effective_by_roi = payload.get(
        "per_roi_effective_feature_config_fields_for_overrides"
    ) or {}
    override_fields_by_roi = payload.get("per_roi_override_config_fields") or {}
    provenance = payload.get("per_roi_feature_provenance") or {}

    if set(override_fields_by_roi) != set(effective_by_roi):
        raise GuidedFeatureSettingsError(
            "Per-ROI feature settings are inconsistent: Custom ROIs with a "
            f"sparse override {sorted(override_fields_by_roi)} do not match "
            f"those with a complete effective configuration {sorted(effective_by_roi)}."
        )

    base_feature_fields = _feature_fields_of(base_config)

    per_roi_feature_config = {}
    for roi_id, effective_fields in effective_by_roi.items():
        fields = dict(effective_fields or {})
        unknown = set(fields) - set(FEATURE_EVENT_CONFIG_FIELDS)
        if unknown:
            raise GuidedFeatureSettingsError(
                f"ROI {roi_id} effective feature settings contain unknown "
                f"fields: {sorted(unknown)}"
            )
        missing = set(FEATURE_EVENT_CONFIG_FIELDS) - set(fields)
        if missing:
            raise GuidedFeatureSettingsError(
                f"ROI {roi_id} effective feature settings are incomplete; "
                f"missing: {sorted(missing)}"
            )
        sparse = dict(override_fields_by_roi.get(roi_id) or {})
        unknown_override = set(sparse) - set(FEATURE_EVENT_CONFIG_FIELDS)
        if unknown_override:
            raise GuidedFeatureSettingsError(
                f"ROI {roi_id} override references unknown feature fields: "
                f"{sorted(unknown_override)}"
            )
        # The complete effective configuration must equal the confirmed base
        # with only the sparse override applied. Anything else means the ROI
        # was resolved against a different base than the one Pipeline will use.
        expected = dict(base_feature_fields)
        expected.update(sparse)
        if fields != expected:
            differing = sorted(k for k in expected if fields.get(k) != expected[k])
            raise GuidedFeatureSettingsError(
                f"ROI {roi_id} effective feature settings were not derived from "
                f"the confirmed base configuration; fields differ: {differing}"
            )
        per_roi_feature_config[roi_id] = replace(base_config, **fields)

    # A Default ROI must analyze with exactly the confirmed base settings.
    for roi_id, entry in provenance.items():
        if (entry or {}).get("source") != "default":
            continue
        default_effective = dict((entry or {}).get("effective_config_fields") or {})
        if not default_effective:
            continue
        differing = sorted(
            k for k, v in default_effective.items()
            if k in base_feature_fields and base_feature_fields[k] != v
        )
        if differing:
            raise GuidedFeatureSettingsError(
                f"Default ROI {roi_id} settings disagree with the confirmed "
                f"base configuration; fields differ: {differing}"
            )

    return per_roi_feature_config, (provenance or None)


def main():
    parser = argparse.ArgumentParser(description="V1 Lab-Default Photometry Pipeline")
    parser.add_argument('--input', required=True, help="Input folder or file")
    parser.add_argument('--config', required=True, help="Path to config.yaml")
    parser.add_argument('--out', required=True, help="Output directory")
    parser.add_argument(
        '--format',
        choices=['auto', 'rwd', 'npm', 'custom_tabular'],
        default='auto',
        help="Force input format",
    )
    parser.add_argument('--recursive', action='store_true', help="Search input recursively")
    parser.add_argument('--file-glob', dest='file_glob', default="*.csv", help="Glob pattern for CSV files (alias: --glob)")
    parser.add_argument('--glob', dest='file_glob', help="Alias for --file-glob")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite output directory")
    parser.add_argument('--mode', choices=['phasic', 'tonic'], default='phasic', help="Analysis mode: 'phasic' (dynamic fit) or 'tonic' (global fit).")
    parser.add_argument('--include-rois', type=str, default=None, help="Comma-separated list of ROIs to process exclusively")
    parser.add_argument('--exclude-rois', type=str, default=None, help="Comma-separated list of ROIs to ignore")
    parser.add_argument('--events-path', type=str, default=None, help="Absolute path to parent events.ndjson to append ROI selection event to")
    parser.add_argument('--traces-only', action='store_true', help="Run traces and QC, skip feature extraction (features.csv) and feature-dependent summaries. This pipeline has no separate signal event-detection stage.")
    parser.add_argument('--event-signal', type=str, choices=['dff', 'delta_f'], help="Signal to use for peak detection features (default from config: dff)")
    parser.add_argument('--representative-session-index', type=int, default=None, help="Force a specific session index for representative artifacts (0-based)")
    parser.add_argument('--preview-first-n', type=int, default=None, help="Preview mode: process only the first N discovered sessions (after discovery/sort).")
    parser.add_argument('--sessions-per-hour', type=int, default=None, help="Force sessions per hour for timing inference (overrides inference/defaults)")
    parser.add_argument('--frozen-input-manifest', dest='frozen_input_manifest', type=str, default=None, help="Internal: path to the run-wide frozen input manifest shared across analysis subprocesses.")
    parser.add_argument(
        '--guided-candidate-manifest',
        default=None,
        help="Internal/backend use only: exact Guided candidate manifest.",
    )
    parser.add_argument(
        '--acquisition-mode',
        choices=['intermittent', 'continuous'],
        default=None,
        help="Override acquisition structure from config.",
    )
    parser.add_argument(
        '--continuous-window-sec',
        type=float,
        default=None,
        help="Override continuous-mode window duration from config.",
    )
    parser.add_argument(
        '--continuous-step-sec',
        type=float,
        default=None,
        help="Override continuous-mode step duration from config.",
    )
    partial_group = parser.add_mutually_exclusive_group()
    partial_group.add_argument(
        '--allow-partial-final-window',
        dest='allow_partial_final_window',
        action='store_true',
        help="Continuous mode only: include a trailing undersized final window.",
    )
    partial_group.add_argument(
        '--no-allow-partial-final-window',
        dest='allow_partial_final_window',
        action='store_false',
        help="Continuous mode only: drop a trailing undersized final window.",
    )
    parser.set_defaults(allow_partial_final_window=None)
    
    args = parser.parse_args()

    if args.guided_candidate_manifest and (
        args.mode != "phasic"
        or args.format != "rwd"
        or args.overwrite
        or args.traces_only
        or args.preview_first_n is not None
        or args.include_rois is not None
        or args.exclude_rois is not None
        or args.acquisition_mode == "continuous"
    ):
        print("Error: unsupported internal Guided manifest execution state.")
        sys.exit(1)
    
    # No active_glob logic needed, argparse handles dest='file_glob'
    
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Check output
    if os.path.exists(args.out) and not args.overwrite:
        print(f"Error: Output directory {args.out} exists. Use --overwrite.")
        sys.exit(1)
        
    try:
        # Load Config
        config = Config.from_yaml(args.config)
        
        # Apply CLI override for event_signal
        if args.event_signal is not None:
            config.event_signal = args.event_signal
        if args.representative_session_index is not None:
            config.representative_session_index = args.representative_session_index
        if args.preview_first_n is not None:
            config.preview_first_n = args.preview_first_n
        if args.acquisition_mode is not None:
            config.acquisition_mode = args.acquisition_mode
        if args.continuous_window_sec is not None:
            config.continuous_window_sec = float(args.continuous_window_sec)
        if args.continuous_step_sec is not None:
            config.continuous_step_sec = float(args.continuous_step_sec)
        if args.allow_partial_final_window is not None:
            config.allow_partial_final_window = bool(args.allow_partial_final_window)
        
        # Init Pipeline
        per_roi_feature_config, per_roi_feature_provenance = (
            load_guided_per_roi_feature_settings(
                args.guided_candidate_manifest, config
            )
        )
        pipeline = Pipeline(
            config,
            mode=args.mode,
            per_roi_feature_config=per_roi_feature_config,
            per_roi_feature_provenance=per_roi_feature_provenance,
        )
        
        inc_rois = [r.strip() for r in args.include_rois.split(',') if r.strip()] if args.include_rois else None
        exc_rois = [r.strip() for r in args.exclude_rois.split(',') if r.strip()] if args.exclude_rois else None
        
        # Prepare emitter if requested
        emitter = None
        if args.events_path:
            from photometry_pipeline.core.events import EventEmitter
            run_id = os.path.basename(args.out)
            emitter = EventEmitter(args.events_path, run_id, args.out, file_mode="a", allow_makedirs=False)
            # Emit engine:context BEFORE pipeline.run (preview counts not yet known)
            emitter.emit("engine", "context", "Run context initialized", payload={
                "run_type": "preview" if config.preview_first_n is not None else "full",
                "preview": {"selector": "first_n", "first_n": config.preview_first_n} if config.preview_first_n is not None else None,
                "traces_only": args.traces_only,
                "event_signal": config.event_signal,
                "representative_session_index": config.representative_session_index
            })

        # Run pipeline (ROI selection and representative session resolved inside pipeline.run)
        pipeline.run(
            args.input, args.out, args.format, args.recursive, args.file_glob,
            include_rois=inc_rois, exclude_rois=exc_rois,
            traces_only=args.traces_only,
            emitter=emitter,
            sessions_per_hour=args.sessions_per_hour,
            guided_manifest_path=args.guided_candidate_manifest,
            frozen_input_manifest_path=args.frozen_input_manifest,
        )
        
        # Emit post-run audit events
        if emitter:
            if pipeline.roi_selection is not None:
                emitter.emit("inputs", "roi_selection", "ROI selection resolved",
                             payload=pipeline.roi_selection)
            
            emitter.close()
        
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        from photometry_pipeline.input_processing_completeness import InputProcessingError
        if isinstance(e, InputProcessingError):
            try:
                os.makedirs(args.out, exist_ok=True)
                error_file = os.path.join(args.out, "input_processing_error.json")
                with open(error_file, "w", encoding="utf-8") as handle:
                    json.dump({
                        "category": e.category,
                        "phase": e.phase,
                        "source": os.path.normpath(str(e.source)).replace("\\", "/"),
                        "session_index": e.chunk_index,
                        "reason": str(e.reason),
                    }, handle, indent=2)
            except Exception as write_err:
                print(f"ERROR: Could not write input_processing_error.json: {write_err}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
