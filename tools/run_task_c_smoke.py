#!/usr/bin/env python3
"""
Task C Smoke Run: High Biological Shared Nuisance
Final reconciliation and hardening for robust path discovery and auditable evidence.
"""

import os
import sys
import argparse
import subprocess
import shutil
import time
import json
import glob
import re
import string
from pathlib import Path

# Simple Repo Root Bootstrap
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

def decode_output(output):
    """Safely decode output that may be str or bytes (Python 3.12 Compat)."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        return output.decode('utf-8', errors='replace')
    except Exception:
        return str(output)

def run_command(cmd, timeout=None, env=None, label="", log_dir=None):
    print(f"Executing {label}: {' '.join(cmd)}")
    current_env = os.environ.copy()
    current_env["PYTHONWARNINGS"] = "error::RuntimeWarning"
    if env:
        current_env.update(env)
        
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            env=current_env, 
            timeout=timeout
        )
        elapsed = time.time() - start_time
        if result.returncode != 0:
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, f"fail_{label}_stdout.txt"), "w", encoding="utf-8") as f: 
                    f.write(decode_output(result.stdout))
                with open(os.path.join(log_dir, f"fail_{label}_stderr.txt"), "w", encoding="utf-8") as f: 
                    f.write(decode_output(result.stderr))
            print(f"STDOUT:\n{decode_output(result.stdout)}")
            print(f"STDERR:\n{decode_output(result.stderr)}")
            raise RuntimeError(f"Command '{label}' failed with exit code {result.returncode}")
        return elapsed
    except subprocess.TimeoutExpired as e:
        stdout_text = decode_output(e.stdout)
        stderr_text = decode_output(e.stderr)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"timeout_{label}_stdout.txt"), "w", encoding="utf-8") as f: f.write(stdout_text)
            with open(os.path.join(log_dir, f"timeout_{label}_stderr.txt"), "w", encoding="utf-8") as f: f.write(stderr_text)
        print(f"TIMEOUT: {label} after {timeout} seconds")
        if stdout_text: print(f"STDOUT SO FAR:\n{stdout_text}")
        if stderr_text: print(f"STDERR SO FAR:\n{stderr_text}")
        raise RuntimeError(f"Command '{label}' timed out after {timeout} seconds")

def check_sentinel(path, is_json=False, is_csv=False):
    """Deep check of a sentinel file."""
    if not os.path.exists(path):
        raise RuntimeError(f"Sentinel missing: {path}")
    
    size = os.path.getsize(path)
    if size == 0:
        raise RuntimeError(f"Sentinel is empty (0 bytes): {path}")
    
    if is_json:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                json.load(f)
        except Exception as e:
            raise RuntimeError(f"Sentinel JSON corruption at {path}: {e}")
            
    if is_csv:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                line = f.readline()
                if not line or len(line.strip()) == 0:
                    raise RuntimeError(f"Sentinel CSV missing header: {path}")
        except Exception as e:
            raise RuntimeError(f"Sentinel CSV read error at {path}: {e}")
            
    return size

def find_analysis_dirs(out_dir):
    """Robustly discover phasic_out and tonic_out."""
    out_dir_abs = os.path.abspath(out_dir)
    candidates = [
        os.path.join(out_dir_abs, '_analysis'),
        out_dir_abs
    ]
    
    # Also check immediate first-level subdirs
    if os.path.isdir(out_dir_abs):
        for d in os.listdir(out_dir_abs):
            p = os.path.join(out_dir_abs, d)
            if os.path.isdir(p):
                candidates.append(p)

    checked_details = []
    for base in candidates:
        p_dir = os.path.join(base, 'phasic_out')
        t_dir = os.path.join(base, 'tonic_out')
        p_meta = os.path.join(p_dir, 'run_metadata.json')
        t_meta = os.path.join(t_dir, 'run_metadata.json')
        
        # Check both exist and are non-empty
        p_exists = os.path.exists(p_meta) and os.path.getsize(p_meta) > 0
        t_exists = os.path.exists(t_meta) and os.path.getsize(t_meta) > 0
        
        checked_details.append(f"Candidate: {base} (PhasicMeta={p_exists}, TonicMeta={t_exists})")
        if p_exists and t_exists:
            return os.path.abspath(p_dir), os.path.abspath(t_dir)
            
    err_msg = f"Could not discover analysis outputs. Checked under {out_dir_abs}:\n" + "\n".join(checked_details)
    raise RuntimeError(err_msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='tests/out_task_c_smoke', help="Output root directory")
    parser.add_argument('--seed', type=int, default=42, help="Seed for determinism")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    log_dir = os.path.join(out_dir, 'logs')
    # Generate dataset in a separate parent directory to survive --overwrite
    template_dataset_dir = os.path.join(os.path.dirname(out_dir), 'smoke_dataset_temp')
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    os.makedirs(log_dir)

    dataset_dir = os.path.join(out_dir, 'dataset')
    figures_dir = os.path.join(out_dir, 'figures')
    os.makedirs(figures_dir)

    config_path = os.path.join(_repo_root, 'tests', 'qc_universal_config.yaml')
    deliverables_script = os.path.join(_repo_root, 'tools', 'run_full_pipeline_deliverables.py')

    # 1. Generate Dataset
    if os.path.exists(template_dataset_dir):
        shutil.rmtree(template_dataset_dir)
        
    print("--- Step 1: Generating Dataset ---")
    cmd_gen = [
        sys.executable, 'tools/synth_photometry_dataset.py',
        '--out', template_dataset_dir,
        '--format', 'rwd',
        '--config', config_path,
        '--total-days', '0.05',
        '--recordings-per-hour', '6',
        '--recording-duration-min', '10',
        '--n-rois', '1',
        '--seed', str(args.seed),
        '--preset', 'biological_shared_nuisance',
        '--phasic-mode', 'high_phasic'
    ]
    t_gen = run_command(cmd_gen, timeout=600, label="generator", log_dir=log_dir)

    # 2. Run Pipeline (Standardized Design 1)
    print("--- Step 2: Running Pipeline (Combined) ---")
    cmd_pipeline = [
        sys.executable, deliverables_script,
        '--input', template_dataset_dir,
        '--out', out_dir,
        '--config', config_path,
        '--format', 'rwd',
        '--sessions-per-hour', '6',
        '--overwrite'
    ]
    t_pipe = run_command(cmd_pipeline, timeout=1200, label="pipeline", log_dir=log_dir)

    # Move dataset into final location (survive overwrite)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    shutil.move(template_dataset_dir, dataset_dir)

    # Recreate figures directory (in case wiped by pipeline)
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # 3. Discovery & ROI Resolution
    phasic_out, tonic_out = find_analysis_dirs(out_dir)
    print(f"PHASIC_OUT={phasic_out}")
    print(f"TONIC_OUT={tonic_out}")

    meta_path = os.path.join(phasic_out, 'run_metadata.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    roi_map = meta.get('roi_map', {})
    if not roi_map:
        raise RuntimeError(f"roi_map is missing or empty in {meta_path}")

    # Explicit evidence of metadata structure
    rois = sorted(roi_map.keys())
    sample_key = rois[0]
    sample_val = roi_map[sample_key]
    roi_map_summary = f"n_rois={len(roi_map)}, sample='{sample_key}':{type(sample_val).__name__}"
    print(f"ROI_MAP_SAMPLE={roi_map_summary}")

    # Determine map structure and derive ROI name/index
    if isinstance(sample_key, str) and not sample_key.isdigit():
        # Case A or C: Name is the key
        roi_name = sample_key
        if isinstance(sample_val, int):
            # Structure A: { Name: Index }
            roi_idx = sample_val
        else:
            # Structure C: Name -> Dict. Derive index from rank.
            roi_idx = 0
    elif (isinstance(sample_key, (int, str)) and str(sample_key).isdigit()):
        # Case B: Index is the key { Index: Name }
        int_keys = sorted([int(k) for k in roi_map.keys()])
        roi_idx = int_keys[0]
        roi_name = roi_map.get(str(roi_idx)) or roi_map.get(roi_idx)
    else:
        raise RuntimeError(f"Unsupported roi_map structure. Key={sample_key}({type(sample_key)}), Val={type(sample_val)}")
    
    print(f"Selected ROI Name: {roi_name}")
    print(f"Selected ROI Index: {roi_idx}")

    # 4. Produce Figures
    print("--- Step 3: Consolidating Artifacts ---")
    
    # Artifact 1: Regression Fit
    reg_glob = os.path.join(phasic_out, 'viz', f'plot_D_correction_impact_{roi_name}*.png')
    reg_matches = sorted(glob.glob(reg_glob), key=os.path.getmtime, reverse=True)
    if not reg_matches:
        raise RuntimeError(f"No regression plot found for {roi_name} at {reg_glob}")
    
    reg_src = reg_matches[0]
    reg_target = os.path.join(figures_dir, 'artifact_1_regression.png')
    shutil.copy2(reg_src, reg_target)
    check_sentinel(reg_target)
    
    # Artifact 2: Event Detection
    run_command([
        sys.executable, 'tools/plot_phasic_qc_grid.py',
        '--analysis-out', phasic_out,
        '--roi', roi_name,
        '--mode', 'dff',
        '--output-dir', os.path.join(phasic_out, 'phasic_qc')
    ], timeout=300, label="phasic_qc_grid", log_dir=log_dir)
    
    grid_src = os.path.join(phasic_out, 'phasic_qc', 'day_000.png')
    grid_target = os.path.join(figures_dir, 'artifact_2_event_detection.png')
    shutil.copy2(grid_src, grid_target)
    check_sentinel(grid_target)

    # Artifact 3: Continuity
    run_command([
        sys.executable, 'tools/plot_raw_stitched.py',
        '--input', dataset_dir,
        '--format', 'rwd',
        '--config', config_path,
        '--out', figures_dir,
        '--roi', str(roi_idx),
        '--axis', 'experimental'
    ], timeout=300, label="stitched_plot", log_dir=log_dir)
    
    stitch_src = os.path.join(figures_dir, f'raw_stitched_rwd_experimental_roi{roi_idx}.png')
    stitch_target = os.path.join(figures_dir, 'artifact_3_continuity.png')
    os.rename(stitch_src, stitch_target)
    check_sentinel(stitch_target)

    # 5. Full Sentinel Audit
    print("--- Step 4: Sentinel Audit ---")
    
    checklist = [
        (os.path.join(phasic_out, 'run_metadata.json'), True, False),
        (os.path.join(phasic_out, 'qc', 'qc_summary.json'), True, False),
        (os.path.join(phasic_out, 'features', 'features.csv'), False, True),
        (os.path.join(tonic_out, 'run_metadata.json'), True, False),
        (os.path.join(tonic_out, 'qc', 'qc_summary.json'), True, False),
        (os.path.join(tonic_out, 'features', 'features.csv'), False, True),
        (reg_target, False, False),
        (grid_target, False, False),
        (stitch_target, False, False)
    ]
    
    sentinel_results = []
    for path, is_json, is_csv in checklist:
        size = check_sentinel(path, is_json, is_csv)
        sentinel_results.append((path, size))

    # 6. Final Logging (smoke_run_log.txt in UTF-8)
    log_lines = [
        "SMOKE RUN COMPLETE.",
        f"OUT_DIR={out_dir}",
        f"DATASET_DIR={dataset_dir}",
        f"PHASIC_OUT={phasic_out}",
        f"TONIC_OUT={tonic_out}",
        f"FIGURES_DIR={figures_dir}",
        f"Generator Command: {' '.join(cmd_gen)}",
        f"Pipeline Command: {' '.join(cmd_pipeline)}",
        f"ROI_MAP_SAMPLE={roi_map_summary}",
        f"Selected ROI Name: {roi_name}",
        f"Selected ROI Index: {roi_idx}",
        f"Wall-clock runtimes: Generator={t_gen:.1f}s, Pipeline={t_pipe:.1f}s",
        f"Artifact 1: {reg_target}",
        f"Artifact 2: {grid_target}",
        f"Artifact 3: {stitch_target}",
        "\nSENTINEL CHECKLIST:",
        f"{'Sentinel Path':<80} | {'Size (bytes)':>12}"
    ]
    log_lines.append("-" * 95)
    for p, s in sentinel_results:
        log_lines.append(f"{p:<80} | {s:>12,}")

    log_path = os.path.join(out_dir, 'smoke_run_log.txt')
    with open(log_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(log_lines))

    for line in log_lines:
        print(line)

if __name__ == '__main__':
    main()
