
import subprocess
import os
import sys
import shutil
from pathlib import Path

def run_command(cmd, desc):
    print(f"\n--- {desc} ---")
    print(f"Command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAILED: {res.stderr}")
        raise RuntimeError(f"Command failed: {desc}")
    else:
        print("OK")

def main():
    root = Path(__file__).resolve().parents[1]
    script_synth = str(root / 'tools' / 'synth_photometry_dataset.py')
    script_pipeline = str(root / 'analyze_photometry.py')
    script_viz = str(root / 'tools' / 'plot_tonic_dff_panel.py')
    
    demo_dir = root / 'outputs' / 'tonic_dff_demo'
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(parents=True)
    
    config_yaml = demo_dir / 'config.yaml'
    with open(config_yaml, 'w') as f:
        f.write("""
target_fs_hz: 20
chunk_duration_sec: 600
baseline_method: uv_raw_percentile_session
baseline_percentile: 10
rwd_time_col: TimeStamp
uv_suffix: "-410"
sig_suffix: "-470"
peak_threshold_method: mean_std
window_sec: 20.0
step_sec: 5.0
min_samples_per_window: 100
min_valid_windows: 5
f0_min_value: 10.0
qc_max_chunk_fail_fraction: 1.0
seed: 42
""")
    
    common_args = [
        sys.executable, script_synth,
        '--config', str(config_yaml),
        '--format', 'rwd',
        '--total-days', '1.0',
        '--recording-duration-min', '10',
        '--recordings-per-hour', '6',
        '--fs-hz', '20',
        '--n-rois', '1',
        '--seed', '101',
        '--tonic-phase-ct', '6', 
        '--tonic-phase-jitter-hr', '0.0',
        '--noise-sig-std', '0.2',
        '--noise-uv-std', '0.1'
    ]
    
    regimes = [
        ('high_tonic', ['--tonic-amplitude', '1.0']),
        ('low_tonic', ['--tonic-amplitude', '0.2'])
    ]
    
    for name, extra_args in regimes:
        raw_dir = demo_dir / name / 'raw'
        cmd_gen = common_args + ['--out', str(raw_dir)] + extra_args
        run_command(cmd_gen, f"Generating {name}")
        
        analysis_dir = demo_dir / name / 'analysis'
        cmd_pipe = [
            sys.executable, script_pipeline,
            '--input', str(raw_dir),
            '--out', str(analysis_dir),
            '--config', str(config_yaml),
            '--format', 'rwd',
            '--recursive',
            '--overwrite'
        ]
        run_command(cmd_pipe, f"Running Pipeline for {name}")
        
        cmd_viz = [
            sys.executable, script_viz,
            '--input-dir', str(analysis_dir),
            '--output-dir', str(analysis_dir),
            '--roi', 'Region0'
        ]
        run_command(cmd_viz, f"Running Tonic Viz for {name}")
        
    print(f"\nSUCCESS. Outputs in {demo_dir}")

if __name__ == '__main__':
    main()
