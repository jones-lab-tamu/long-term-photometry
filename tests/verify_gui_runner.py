import sys
import os

# Ensure project root in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import shutil
import tempfile
import json
import logging
from PySide6.QtCore import QCoreApplication
from gui.process_runner import PipelineRunner, RunnerState

def main():
    app = QCoreApplication(sys.argv)
    
    # Setup temp dir
    temp_dir = tempfile.mkdtemp(prefix="gui_runner_test_")
    print(f"Test Run Dir: {temp_dir}")
    
    # Create input data (small synth) - reuse success path from contract test logic
    # We call synth_photometry_dataset.py directly first to allow pipeline to run
    input_dir = os.path.join(temp_dir, "input")
    os.makedirs(input_dir)
    
    # Create config
    config_path = os.path.join(temp_dir, "config.yaml")
    # Copy from tests/qc_universal_config.yaml and patch
    src_config = os.path.join(PROJECT_ROOT, "tests", "qc_universal_config.yaml")
    shutil.copy2(src_config, config_path)
    with open(config_path, "a") as f:
        f.write("\nchunk_duration_sec: 60.0\n")
        f.write("window_sec: 30.0\n")
        
    # Generate data
    cmd_gen = [
        sys.executable, "tools/synth_photometry_dataset.py",
        "--out", input_dir,
        "--format", "rwd",
        "--config", config_path,
        "--total-days", "0.1",
        "--recordings-per-hour", "2",
        "--recording-duration-min", "1.0",
        "--n-rois", "1",
        "--preset", "biological_shared_nuisance"
    ]
    import subprocess
    subprocess.check_call(cmd_gen)
    
    # Setup Runner
    runner = PipelineRunner()
    run_dir = os.path.join(temp_dir, "test_run_01")
    runner.set_run_dir(run_dir)
    
    # Define slots
    def on_finished(exit_code):
        print(f"Runner finished with exit code {exit_code}")
        print(f"Final State: {runner.state}")
        
        # Verify status.json
        run_dir = os.path.join(temp_dir, "test_run_01")
        status_path = os.path.join(run_dir, "status.json")
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                s = json.load(f)
            print(f"Status JSON: status={s.get('status')}")
            if s.get('status') == 'success' and runner.state == RunnerState.SUCCESS:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print("TEST FAILED: Status mismatch")
                sys.exit(1)
        else:
            print("TEST FAILED: No status.json")
            sys.exit(1)

    runner.finished.connect(on_finished)
    
    # Start Pipeline
    # tools/run_full_pipeline_deliverables.py --input ... --out ... --config ...
    # tools/run_full_pipeline_deliverables.py --input ... --out-base ... --run-id ...
    argv = [
        sys.executable, "tools/run_full_pipeline_deliverables.py",
        "--input", input_dir,
        "--out-base", temp_dir,
        "--run-id", "test_run_01",
        "--config", config_path,
        "--format", "rwd",
        "--sessions-per-hour", "2",
        "--events", "auto",
        "--cancel-flag", "auto"
    ]
    
    print("Starting pipeline...")
    runner.start(argv, state=RunnerState.RUNNING)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
