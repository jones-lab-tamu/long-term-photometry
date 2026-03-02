import unittest
import os
import shutil
import tempfile
import json
import pandas as pd
import numpy as np

# Adjust path to import tools/pipeline
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.verify_paper_alignment import run_checks
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.config import Config

class TestPaperAlignment(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, 'input')
        self.output_dir = os.path.join(self.test_dir, 'output')
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        
        # Create synthetic chunks
        self._create_synthetic_data()
        
        # Create Config
        self.config_path = os.path.join(self.test_dir, 'config.yaml')
        self._create_config()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def _create_synthetic_data(self):
        # Create 2 chunks
        for i in range(2):
            chunk_dir = os.path.join(self.input_dir, f'chunk_{i}')
            os.makedirs(chunk_dir)
            
            # Simple synthetic data: 405 (UV) and 465 (Signal)
            # 10 Hz, 60 seconds = 600 samples
            time = np.linspace(0, 60, 600)
            # UV: slight drift
            uv = 100 + 0.01 * time + np.random.normal(0, 0.1, 600)
            # Sig: correlated with UV + some events
            sig = 200 + 2.0 * uv + np.random.normal(0, 0.5, 600)
            # Add an event
            sig[300:320] += 50
            
            df = pd.DataFrame({
                'Time(s)': time,
                'Region1-470': sig,
                'Region1-410': uv 
            })
            
            # We need to match RWD format expectations or Generic
            # Let's say we use generic format with specific columns for simplicity, 
            # OR we mock RWD. The request said "minimal synthetic RWD-style fixture".
            # RWD normally expects fluorescence.csv in subfolders.
            df.to_csv(os.path.join(chunk_dir, 'fluorescence.csv'), index=False)

    def _create_config(self):
        # YAML content
        # Must enable feature extraction and use dynamic regression
        yaml_content = f"""
target_fs_hz: 10
chunk_duration_sec: 60
window_sec: 10
step_sec: 5
baseline_method: uv_raw_percentile_session
baseline_percentile: 10
f0_min_value: 1.0
peak_threshold_method: percentile
peak_threshold_percentile: 95
peak_min_distance_sec: 1.0
"""
        with open(self.config_path, 'w') as f:
            f.write(yaml_content)

    def test_pipeline_alignment(self):
        # 1. Run Pipeline using RWD discovery logic (since we made subfolders)
        # However, our generic adapter doesn't auto-discover "fluorescence.csv" inside folders 
        # unless we tell it to look recursively or use RWD mode.
        # Let's use recursive glob for "fluorescence.csv"
        
        cfg = Config.from_yaml(self.config_path)
        pipeline = Pipeline(cfg)
        
        # Manually invoke with appropriate args
        try:
            pipeline.run(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                recursive=True,
                glob_pattern="fluorescence.csv"
            )
        except Exception as e:
            self.fail(f"Pipeline run failed: {e}")
            
        # 2. Run Verification Checks
        # We call the logic from tools/verify_paper_alignment
        try:
            result = run_checks(self.output_dir)
        except SystemExit:
            self.fail("Verification script triggered system exit.")
        except Exception as e:
            self.fail(f"Verification checks raised exception: {e}")
            
        # 3. Assertions
        self.assertTrue(result['pass'], "Paper alignment checks failed")
        
        # Check individual components
        checks_map = {c['name']: c['pass'] for c in result['checks']}
        self.assertTrue(checks_map.get('B1_ValidBaselineMethod'), "B1 Failed")
        self.assertTrue(checks_map.get('B2_BaselineSeparatedFromArtifact'), "B2 Failed")
        self.assertTrue(checks_map.get('B3_DynamicRegressionUsed'), "B3 Failed")
        self.assertTrue(checks_map.get('B4_PhasicMetricsExist'), "B4 Failed")
        self.assertTrue(checks_map.get('B5_QCAndVizOutputs'), "B5 Failed")

if __name__ == '__main__':
    unittest.main()
