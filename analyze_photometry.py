import argparse
import sys
import os
import logging
from photometry_pipeline.config import Config
from photometry_pipeline.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description="V1 Lab-Default Photometry Pipeline")
    parser.add_argument('--input', required=True, help="Input folder or file")
    parser.add_argument('--config', required=True, help="Path to config.yaml")
    parser.add_argument('--out', required=True, help="Output directory")
    parser.add_argument('--format', choices=['auto', 'rwd', 'npm'], default='auto', help="Force input format")
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
    
    args = parser.parse_args()
    
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
        
        # Init Pipeline
        pipeline = Pipeline(config, mode=args.mode)
        
        inc_rois = [r.strip() for r in args.include_rois.split(',') if r.strip()] if args.include_rois else None
        exc_rois = [r.strip() for r in args.exclude_rois.split(',') if r.strip()] if args.exclude_rois else None
        
        # Prepare emitter if requested
        emitter = None
        if args.events_path:
            from photometry_pipeline.core.events import EventEmitter
            run_id = os.path.basename(args.out)
            emitter = EventEmitter(args.events_path, run_id, args.out, file_mode="a", allow_makedirs=False)

        # Run pipeline (ROI selection and representative session resolved inside pipeline.run)
        pipeline.run(
            args.input, args.out, args.format, args.recursive, args.file_glob,
            include_rois=inc_rois, exclude_rois=exc_rois,
            traces_only=args.traces_only,
            emitter=emitter
        )
        
        # Emit inputs:roi_selection event via EventEmitter if emitter exists
        if emitter:
            if pipeline.roi_selection is not None:
                emitter.emit("inputs", "roi_selection", "ROI selection resolved",
                             payload=pipeline.roi_selection)
            
            emitter.close()
        
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        # traceback?
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
