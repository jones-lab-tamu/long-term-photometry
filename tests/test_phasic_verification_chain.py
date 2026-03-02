
import unittest
import subprocess
import os
import shutil
import sys
import glob
import json

# Ensure we can run modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestPhasicVerificationChain(unittest.TestCase):
    """
    Step 5 Automatic Verification:
    Executes the full 4-step Verification Protocol.
    """
    
    @staticmethod
    def _robust_cleanup(path):
        import time
        if not os.path.exists(path):
            return
        
        # Try standard removal with retries
        for i in range(5):
            try:
                shutil.rmtree(path)
                return
            except PermissionError:
                time.sleep(0.2)
            except Exception:
                # Other errors, fall through to ignore_errors
                break
        
        # Final best-effort with ignore_errors
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to cleanup {path}: {e}\n")

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.out_dir = tempfile.mkdtemp(prefix="out_verify_chain_")
        cls.yaml_path = os.path.join(PROJECT_ROOT, "tests", "qc_universal_config.yaml")
        
        # Clean start not needed with unique temp dir
        # cls._robust_cleanup(cls.out_dir)
        # os.makedirs(cls.out_dir, exist_ok=True)
        
        # Ensure config exists (should be there from previous steps, but safe to verify)
        if not os.path.exists(cls.yaml_path):
            raise RuntimeError(f"Config file not found: {cls.yaml_path}")

    def run_cmd(self, cmd_list):
        print(f"RUNNING: {' '.join(cmd_list)}")
        res = subprocess.run(
            cmd_list, 
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False
        )
        if res.returncode != 0:
            print("STDOUT:", res.stdout)
            print("STDERR:", res.stderr)
        self.assertEqual(res.returncode, 0, f"Command failed: {cmd_list[0]}")
        return res

    def test_full_chain(self):
        """
        Executes Steps 1-4 of the Strict Verification Protocol.
        Generates RUNTHROUGH_REPORT.md as a self-auditing step.
        """
        report_lines = ["# RUNTHROUGH_REPORT.md\n"]
        report_lines.append(f"## Environment\n- **OS**: {sys.platform}\n- **Python**: {sys.version.split()[0]}\n- **Repo Root**: `{PROJECT_ROOT}`\n- **Output Directory**: `{self.out_dir}`\n")
        report_lines.append("## Execution Summary\nThe runthrough was completed successfully after updating the verification chain and AUC logic.\n")
        report_lines.append("### Commands Run\n")

        # Step 1: Generate (High Tonic, Gated Phasic)
        gen_cmd = [
            sys.executable, "tools/synth_photometry_dataset.py",
            "--out", self.out_dir,
            "--format", "rwd",
            "--config", self.yaml_path,
            "--total-days", "2.0",
            "--recordings-per-hour", "2",
            "--n-rois", "2",
            "--phasic-mode", "phase_locked_to_tonic",
            "--seed", "999",
            "--preset", "biological_shared_nuisance"
        ]
        
        # Capture output for audit
        res_gen = self.run_cmd(gen_cmd)
        
        # Audit Preset
        if "Applied Preset: biological_shared_nuisance" not in res_gen.stdout:
            self.fail("Synthetic data generation did not confirm application of 'biological_shared_nuisance' preset.")
            
        # Extract the literal line(s)
        preset_lines = [line for line in res_gen.stdout.splitlines() if "Applied Preset: biological_shared_nuisance" in line]

        report_lines.append(f"1. **Generate Synthetic Data**\n   ```powershell\n   {' '.join(gen_cmd)}\n   ```\n")
        report_lines.append(f"   **Preset Audit**: VERIFIED\n   **Preset Audit Line(s)**:\n   ```text\n" + "\n".join(preset_lines) + "\n   ```\n")

        # Step 2: Tonic Pipeline & Anchor
        tonic_out = os.path.join(self.out_dir, "tonic_out")
        analyze_tonic_cmd = [
            sys.executable, "analyze_photometry.py",
            "--input", self.out_dir,
            "--out", tonic_out,
            "--config", self.yaml_path,
            "--mode", "tonic",
            "--overwrite", "--recursive", "--format", "rwd"
        ]
        self.run_cmd(analyze_tonic_cmd)
        report_lines.append(f"2. **Tonic Analysis**\n   ```powershell\n   {' '.join(analyze_tonic_cmd)}\n   ```\n")
        
        plot_tonic_cmd = [
            sys.executable, "tools/plot_tonic_48h.py",
            "--analysis-out", tonic_out
        ]
        self.run_cmd(plot_tonic_cmd)
        
        # Verify Output
        self.assertTrue(os.path.exists(os.path.join(tonic_out, "tonic_qc", "tonic_48h_overview_Region0.png")))

        # Step 3: Phasic Pipeline & Session Grid
        phasic_out = os.path.join(self.out_dir, "phasic_out")
        analyze_phasic_cmd = [
            sys.executable, "analyze_photometry.py",
            "--input", self.out_dir,
            "--out", phasic_out,
            "--config", self.yaml_path,
            "--mode", "phasic",
            "--overwrite", "--recursive", "--format", "rwd"
        ]
        self.run_cmd(analyze_phasic_cmd)
        report_lines.append(f"3. **Phasic Analysis**\n   ```powershell\n   {' '.join(analyze_phasic_cmd)}\n   ```\n")
        
        item_grid_cmd = [
            sys.executable, "tools/plot_session_grid.py",
            "--analysis-out", phasic_out
        ]
        self.run_cmd(item_grid_cmd)
        
        # Verify Session QC Output
        self.assertTrue(os.path.exists(os.path.join(phasic_out, "session_qc", "day_000_raw_iso_Region0.png")))
        
        print("Step 4a: Strict Chain Audit (Tier 1 - Default)...")
        chain_cmd_default = [
            sys.executable, "tools/plot_phasic_intermediate_chain.py",
            "--analysis-out", phasic_out
        ]
        self.run_cmd(chain_cmd_default)
        report_lines.append(f"4. **Chain Audit**\n   ```powershell\n   {' '.join(chain_cmd_default)}\n   ```\n")
        
        print("Step 4b: Strict Chain Audit (Tier 2 - Synth Gating)...")
        chain_cmd_gated = [
            sys.executable, "tools/plot_phasic_intermediate_chain.py",
            "--analysis-out", phasic_out,
            "--enable-synth-gating-check"
        ]
        self.run_cmd(chain_cmd_gated)
        
        # Verify Chain Output
        chain_dir = os.path.join(phasic_out, "phasic_chain_qc")
        pngs = glob.glob(os.path.join(chain_dir, "*.png"))
        self.assertTrue(len(pngs) > 10, "Should generate chain plots for passed chunks")
        
        # Step 5: Visualization Grid
        viz_cmd = [
            sys.executable, "tools/plot_phasic_qc_grid.py",
            "--analysis-out", phasic_out
        ]
        self.run_cmd(viz_cmd)
        self.assertTrue(os.path.exists(os.path.join(phasic_out, "phasic_qc", "day_000.png")))
        
        # Write Report
        report_path = os.path.join(PROJECT_ROOT, "RUNTHROUGH_REPORT.md")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
            
    @classmethod
    def tearDownClass(cls):
        # Robust cleanup to prevent Windows PermissionError
        cls._robust_cleanup(cls.out_dir)

if __name__ == '__main__':
    unittest.main()
