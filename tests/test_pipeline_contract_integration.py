import os
import re
import sys
import json
import shutil
import tempfile
import subprocess
import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.core.utils import natural_sort_key


def test_pipeline_contract_integration():
    """
    Integration test locking 9 critical pipeline contracts:
    1) Natural sort ordering
    2) Manifest discipline
    3) Metadata namespace
    4) NaN / degenerate data policy
    5) Regression fallback behavior
    6) AUC baseline policy
    7) Event warnings (DEGENERATE tokens)
    8) Output structure (path pattern invariants)
    9) No silent warning leakage
    """
    out_dir = tempfile.mkdtemp(prefix="contract_integ_")

    try:
        input_dir = os.path.join(out_dir, "input")
        os.makedirs(input_dir, exist_ok=True)

        out1_dir = os.path.join(out_dir, "out1")
        out2_dir = os.path.join(out_dir, "out2")

        config_path = os.path.join(out_dir, "config.yaml")

        # Build config — deterministic, triggers contracts
        cfg = Config()
        cfg.chunk_duration_sec = 300.0
        # Contract 5: window > chunk length forces regression fallback
        cfg.window_sec = 600.0
        cfg.step_sec = 60.0
        # Contract 4/7: median_mad on flatline -> DD4 zero robust variance
        cfg.peak_threshold_method = 'median_mad'
        # Contract 6: explicit AUC baseline
        cfg.event_auc_baseline = 'median'

        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(cfg.__dict__, f)

        # Generate synthetic data: 0.25 days = 6 hours, 2 rec/hr = 12 chunks
        gen_cmd = [
            sys.executable, "tools/synth_photometry_dataset.py",
            "--out", input_dir,
            "--format", "rwd",
            "--config", config_path,
            "--total-days", "0.25",
            "--recordings-per-hour", "2",
            "--recording-duration-min", "5.0",
            "--n-rois", "1",
            "--seed", "42"
        ]
        subprocess.check_call(gen_cmd)

        # Inject degenerate chunk: pure flatline (constant signal & UV)
        # This triggers DD4 (zero robust variance in peak detection)
        # without injecting NaN into the baseline reservoir.
        import glob
        csv_files = sorted(glob.glob(os.path.join(input_dir, '**', '*.csv'),
                                     recursive=True))
        target_csv = csv_files[2]  # 3rd recording

        with open(target_csv, 'r') as f:
            lines = f.readlines()

        for i in range(1, len(lines)):
            parts = lines[i].strip().split(',')
            if len(parts) >= 3:
                lines[i] = f"{parts[0]},100.0,100.0\n"

        with open(target_csv, 'w') as f:
            f.writelines(lines)

        # Pipeline runner — self-contained PYTHONPATH, no conftest
        def run_pipeline(output_dir):
            cmd = [
                sys.executable, "tools/run_full_pipeline_deliverables.py",
                "--input", input_dir,
                "--out", output_dir,
                "--config", config_path,
                "--format", "rwd",
                "--overwrite",
                "--sessions-per-hour", "2"
            ]
            env = dict(os.environ)
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..'))
            existing = env.get("PYTHONPATH", "")
            if not existing:
                env["PYTHONPATH"] = repo_root
            else:
                if repo_root not in existing.split(os.pathsep):
                    env["PYTHONPATH"] = repo_root + os.pathsep + existing

            # Contract 9: no silent warning leakage
            env["PYTHONWARNINGS"] = "error::RuntimeWarning"
            res = subprocess.run(cmd, env=env, capture_output=True, text=True)

            assert res.returncode == 0, (
                f"Pipeline failed (rc={res.returncode}).\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

        # Run pipeline twice for determinism check
        run_pipeline(out1_dir)
        run_pipeline(out2_dir)

        # -------------------------------------------------------
        # Locate the phasic output directory
        # -------------------------------------------------------
        phasic_dir = os.path.join(out1_dir, "_analysis", "phasic_out")
        run_meta_path = os.path.join(phasic_dir, "run_metadata.json")

        with open(run_meta_path, 'r') as f:
            run_meta = json.load(f)

        # -------------------------------------------------------
        # Contract 1: Natural sort ordering
        # -------------------------------------------------------
        feats1_path = os.path.join(phasic_dir, "features", "features.csv")
        feats2_path = os.path.join(
            out2_dir, "_analysis", "phasic_out", "features", "features.csv")
        df1 = pd.read_csv(feats1_path)
        df2 = pd.read_csv(feats2_path)

        out_chunk_ids = list(df1['chunk_id'].unique())
        assert out_chunk_ids == sorted(out_chunk_ids), (
            "Features CSV chunk_id ordering violates natural sort cascade"
        )

        # -------------------------------------------------------
        # Contract 2: Manifest discipline
        # -------------------------------------------------------
        assert 'chunk_metadata_outputs' in run_meta, (
            "run_metadata.json missing chunk_metadata_outputs declaration"
        )
        rel_paths = run_meta['chunk_metadata_outputs']
        assert isinstance(rel_paths, list) and len(rel_paths) > 0, (
            "chunk_metadata_outputs must be a non-empty list"
        )
        for p in rel_paths:
            assert isinstance(p, str), (
                f"chunk_metadata_outputs items must be strings, got {type(p)}"
            )
        assert len(rel_paths) >= 12, (
            f"Expected >=12 chunk meta outputs for 12 chunks, got {len(rel_paths)}"
        )
        # Natural ordering invariant on manifest list
        assert rel_paths == sorted(rel_paths, key=natural_sort_key), (
            "Manifest chunk_metadata_outputs list is not naturally ordered"
        )

        # -------------------------------------------------------
        # Contract 3: Metadata namespace
        # -------------------------------------------------------
        chunk_jsons = [os.path.join(phasic_dir, p) for p in rel_paths]
        # Sample first, middle, last
        sample_indices = sorted(set([0, len(chunk_jsons) // 2,
                                     len(chunk_jsons) - 1]))

        for idx in sample_indices:
            with open(chunk_jsons[idx], 'r') as f:
                c_meta = json.load(f)

            # Top-level stable keys
            assert 'fs_hz' in c_meta, f"Missing fs_hz in {rel_paths[idx]}"
            assert 'window_sec' in c_meta, (
                f"Missing window_sec in {rel_paths[idx]}"
            )
            # Namespace boundary
            if 'chunk_metadata' in c_meta:
                assert isinstance(c_meta['chunk_metadata'], dict), (
                    "chunk_metadata must be a dict"
                )
                warnings_list = c_meta['chunk_metadata'].get(
                    'qc_warnings', [])
                assert isinstance(warnings_list, list), (
                    "qc_warnings must be a list"
                )
                for w in warnings_list:
                    assert isinstance(w, str), (
                        "qc_warning items must be strings"
                    )

        # -------------------------------------------------------
        # Contract 4: NaN / degenerate data policy
        # Contract 7: Event warnings (DEGENERATE tokens)
        # -------------------------------------------------------
        # Scan for DEGENERATE warnings across chunk meta files
        dd_tokens_found = set()
        # Check the injected chunk (index 2) plus neighbors
        scan_indices = sorted(set([0, 1, 2, 3, len(chunk_jsons) // 2,
                                   len(chunk_jsons) - 1]))
        for idx in scan_indices:
            if idx >= len(chunk_jsons):
                continue
            with open(chunk_jsons[idx], 'r') as f:
                c_meta = json.load(f)
            cm = c_meta.get('chunk_metadata', {})
            for w in cm.get('qc_warnings', []):
                if "DEGENERATE[DD" in w:
                    tok = w.split("]")[0].split("[")[1]
                    dd_tokens_found.add(tok)

        assert len(dd_tokens_found) > 0, (
            "No DEGENERATE QC warnings emitted — degenerate data policy "
            "not enforced"
        )
        # Specifically expect DD4 (zero robust variance from flatline)
        assert 'DD4' in dd_tokens_found or 'DD2' in dd_tokens_found, (
            f"Expected DD4 or DD2 for flatline scenario, got {dd_tokens_found}"
        )

        # -------------------------------------------------------
        # Contract 5: Regression fallback behavior
        # -------------------------------------------------------
        # The injected chunk (index 2) should have window_fallback_global
        with open(chunk_jsons[2], 'r') as f:
            t_meta = json.load(f)
        assert 'chunk_metadata' in t_meta, (
            "Namespaced chunk_metadata missing from chunk meta JSON"
        )
        chunk_md = t_meta['chunk_metadata']
        assert chunk_md.get('window_fallback_global') is True, (
            "Regression fallback not invoked despite window_sec > chunk length"
        )

        # -------------------------------------------------------
        # Contract 6: AUC baseline policy
        # -------------------------------------------------------
        assert 'chunk_id' in df1.columns
        assert 'roi' in df1.columns
        assert 'auc' in df1.columns
        assert 'peak_count' in df1.columns
        # Verify AUC is present and finite for a valid (non-degenerate) chunk
        valid_chunk_df = df1[df1['chunk_id'] == 0]
        assert len(valid_chunk_df) == 1, "Expected exactly one row for chunk 0"
        val_auc = valid_chunk_df.iloc[0]['auc']
        assert np.isfinite(val_auc), "AUC missing for valid chunk"
        assert val_auc >= 0.0, "AUC must be non-negative"
        assert valid_chunk_df.iloc[0]['peak_count'] >= 0, (
            "Invalid peak_count"
        )

        # -------------------------------------------------------
        # Contract 8: Output structure (path pattern invariants)
        # -------------------------------------------------------
        for rp in rel_paths:
            # Forward slashes
            assert '\\' not in rp, (
                f"Backslash in manifest path: {rp}"
            )
            # Correct subdirectory
            assert 'phasic_intermediates/' in rp, (
                f"Manifest path not in phasic_intermediates/: {rp}"
            )
            # Filename pattern: chunk_XXXX_<ROI>_meta.json
            basename = rp.split('/')[-1]
            assert re.match(r'chunk_\d{4}_\w+_meta\.json$', basename), (
                f"Manifest filename does not match expected pattern: {basename}"
            )

        # -------------------------------------------------------
        # Contract 8 (cont): Determinism across runs
        # -------------------------------------------------------
        assert not df1.duplicated(subset=['chunk_id', 'roi']).any(), (
            "Duplicate chunk_id+roi in features"
        )
        pd.testing.assert_frame_equal(
            df1, df2, check_exact=False, rtol=1e-5, atol=1e-5
        )

        # -------------------------------------------------------
        # Contract 9: No silent warnings leakage
        # -------------------------------------------------------
        # Already enforced by PYTHONWARNINGS=error::RuntimeWarning
        # and the returncode==0 assertion in run_pipeline().

    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
