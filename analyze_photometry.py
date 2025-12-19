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
    parser.add_argument('--file-glob', default="*.csv", help="Glob pattern for CSV files")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite output directory")
    
    args = parser.parse_args()
    
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Check output
    if os.path.exists(args.out) and not args.overwrite:
        print(f"Error: Output directory {args.out} exists. Use --overwrite.")
        sys.exit(1)
        
    try:
        # Load Config
        config = Config.from_yaml(args.config)
        
        # Init Pipeline
        pipeline = Pipeline(config)
        
        # Run
        pipeline.run(args.input, args.out, args.format, args.recursive, args.file_glob)
        
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        # traceback?
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
