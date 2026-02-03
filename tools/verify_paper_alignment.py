import argparse
import sys
import os
import json
import glob
import pandas as pd
from typing import Dict, Any, List

# To allow importing from parent directories when run as script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.config import Config

"""
verify_paper_alignment.py

Runs the photometry pipeline and performs deterministic checks to ensure compliance 
with the core requirements of the linked paper:
1. Signal decomposition (Tonic/Phasic separation)
2. Analytical strategies for Tonic Extraction
3. Analytical strategies for Phasic Extraction
4. Core QC and Reporting requirements

Usage:
  python tools/verify_paper_alignment.py --input <path> --out <path> --config <path>
"""

def fail(reason: str):
    print(f"FAIL: {reason}")
    sys.exit(1)

def run_checks(output_dir: str) -> Dict[str, Any]:
    checks = []
    
    # ---------------------------------------------------------
    # B1 & B2: Tonic Extraction & Explicit Separation
    # ---------------------------------------------------------
    meta_path = os.path.join(output_dir, 'run_metadata.json')
    if not os.path.exists(meta_path):
        fail(f"Metadata file missing: {meta_path}")
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    # Check B1: Baseline fields existence
    required_baseline_fields = ['baseline_method', 'f0_values']
    for f in required_baseline_fields:
        if f not in meta:
            fail(f"Missing baseline metadata field: {f}")
            
    # Check B1: Valid Method
    method = meta.get('baseline_method')
    valid_methods = ["uv_raw_percentile_session", "uv_globalfit_percentile_session"]
    checks.append({
        "name": "B1_ValidBaselineMethod",
        "pass": method in valid_methods,
        "details": {"method": method, "allowed": valid_methods}
    })
    if method not in valid_methods:
        fail(f"Invalid baseline method '{method}'. Must be one of {valid_methods}")

    # Check B2: Explicit Separation
    # f0_is_from_uv_fit must be False
    is_from_fit = meta.get('f0_is_from_uv_fit')
    checks.append({
        "name": "B2_BaselineSeparatedFromArtifact",
        "pass": is_from_fit is False,
        "details": {"f0_is_from_uv_fit": is_from_fit}
    })
    if is_from_fit is not False:
        fail("Baseline F0 must NOT be derived from the phasic artifact fit model (f0_is_from_uv_fit must be False).")
        
    if meta.get('f0_source') != method:
         fail(f"Metadata inconsistency: f0_source ({meta.get('f0_source')}) != baseline_method ({method})")

    # ---------------------------------------------------------
    # B3: Dynamic Fitting Exercised
    # ---------------------------------------------------------
    reg_mode = meta.get('regression_mode')
    checks.append({
        "name": "B3_DynamicRegressionUsed",
        "pass": reg_mode == 'dynamic',
        "details": {
            "mode": reg_mode,
            "window": meta.get('regression_window_sec'),
            "step": meta.get('regression_step_sec')
        }
    })
    if reg_mode != 'dynamic':
        fail("Long-term analysis requires dynamic regression (regression_mode='dynamic').")
        
    if not meta.get('regression_window_sec') or not meta.get('regression_step_sec'):
        fail("Dynamic regression metadata missing window/step sizes.")

    # ---------------------------------------------------------
    # B4: Phasic Metrics (Event & Integrated)
    # ---------------------------------------------------------
    feat_path = os.path.join(output_dir, 'features', 'features.csv')
    if not os.path.exists(feat_path):
        fail("Features file missing.")
        
    df = pd.read_csv(feat_path)
    cols = df.columns.tolist()
    
    has_peak = 'peak_count' in cols
    has_auc = 'auc' in cols
    
    checks.append({
        "name": "B4_PhasicMetricsExist",
        "pass": has_peak and has_auc,
        "details": {"columns_found": cols}
    })
    
    if not has_peak: fail("Missing required phasic metric: 'peak_count'")
    if not has_auc: fail("Missing required phasic metric: 'auc'")

    # ---------------------------------------------------------
    # B5: QC & Visualization
    # ---------------------------------------------------------
    qc_path = os.path.join(output_dir, 'qc', 'qc_summary.json')
    traces_dir = os.path.join(output_dir, 'traces')
    viz_dir = os.path.join(output_dir, 'viz')
    
    has_qc = os.path.exists(qc_path)
    has_traces = os.path.isdir(traces_dir) and len(os.listdir(traces_dir)) > 0
    
    # We require at least one plot file in viz/
    has_viz = False
    if os.path.isdir(viz_dir):
        files = glob.glob(os.path.join(viz_dir, "*"))
        if len(files) > 0:
            has_viz = True
            
    checks.append({
        "name": "B5_QCAndVizOutputs",
        "pass": has_qc and has_traces and has_viz,
        "details": {"has_qc": has_qc, "has_traces": has_traces, "has_viz": has_viz}
    })
    
    if not has_qc: fail("QC Summary output missing.")
    if not has_traces: fail("Trace outputs missing.")
    if not has_viz: fail("Visualization outputs missing.")

    return {"pass": True, "checks": checks}

def main():
    parser = argparse.ArgumentParser(description="Verify Paper Alignment")
    parser.add_argument('--input', required=True, help='Input directory or file')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--format', default='auto', help='Input format')
    parser.add_argument('--recursive', action='store_true', help='Recursive search')
    parser.add_argument('--file-glob', default='*.csv', help='File pattern')
    
    args = parser.parse_args()
    
    # 1. Run Pipeline
    print(">>> Running Pipeline...")
    try:
        cfg = Config.from_yaml(args.config)
        pipeline = Pipeline(cfg)
        pipeline.run(
            input_dir=args.input,
            output_dir=args.out,
            force_format=args.format,
            recursive=args.recursive,
            glob_pattern=args.file_glob
        )
    except Exception as e:
        print(f"Pipeline Run Failed: {e}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 2. Verify Outputs
    print(">>> Verifying Outputs...")
    result = run_checks(args.out)
    
    # 3. Write Reports
    report_dir = os.path.join(args.out, 'paper_alignment')
    os.makedirs(report_dir, exist_ok=True)
    
    json_path = os.path.join(report_dir, 'paper_alignment_report.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
        
    md_path = os.path.join(report_dir, 'paper_alignment_report.md')
    with open(md_path, 'w') as f:
        f.write("# Paper Alignment Verification Report\n\n")
        f.write(f"**Overall Status**: {'PASS' if result['pass'] else 'FAIL'}\n\n")
        f.write("## Checks\n\n")
        f.write("| Check | Status | Details |\n")
        f.write("| --- | --- | --- |\n")
        for c in result['checks']:
            status = "PASS" if c['pass'] else "FAIL"
            details = str(c['details'])
            f.write(f"| {c['name']} | {status} | {details} |\n")
            
    print("VERIFICATION PASSED")

if __name__ == "__main__":
    main()
