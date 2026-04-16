import pytest
import sys
import os
import json
import shutil
import tempfile
import yaml
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Add root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gui.run_spec import RunSpec

def test_gui_run_spec_includes_numeric_sessions_per_hour_only():
    """Verifies --sessions-per-hour is in argv but source label is NOT."""
    spec = RunSpec(
        input_dir="C:/data/input",
        run_dir="C:/data/run_1",
        format="npm",
        sessions_per_hour=2
    )
    argv = spec.build_runner_argv()
    assert "--sessions-per-hour" in argv
    assert argv[argv.index("--sessions-per-hour") + 1] == "2"
    assert "--sessions-per-hour-source" not in argv

def test_analyze_photometry_argparse_numeric_only():
    """Verifies analyze_photometry.py accepts numeric --sessions-per-hour."""
    from analyze_photometry import main as analyze_main
    
    test_argv = [
        "analyze_photometry.py",
        "--input", ".",
        "--config", "config.yaml",
        "--out", "out",
        "--sessions-per-hour", "3"
    ]
    
    with patch("sys.argv", test_argv), \
         patch("photometry_pipeline.config.Config.from_yaml") as mock_cfg, \
         patch("photometry_pipeline.pipeline.Pipeline.run") as mock_run:
        
        mock_cfg.return_value = MagicMock()
        
        try:
            analyze_main()
        except SystemExit:
            pass
            
        args, kwargs = mock_run.call_args
        assert kwargs.get("sessions_per_hour") == 3
        assert "sessions_per_hour_source" not in kwargs

@patch("tools.run_full_pipeline_deliverables.run_cmd")
@patch("tools.run_full_pipeline_deliverables.Config.from_yaml")
def test_wrapper_derives_provenance_from_numeric_arg(mock_cfg_load, mock_run_cmd):
    """Verifies wrapper derives 'user-provided' from numeric arg and removes source from downstream."""
    from tools.run_full_pipeline_deliverables import main as wrapper_main
    
    cfg = MagicMock()
    cfg.sessions_per_hour = 1
    mock_cfg_load.return_value = cfg
    
    test_argv = [
        "run_full_pipeline_deliverables.py",
        "--input", ".",
        "--out", "out_dir",
        "--config", "config.yaml",
        "--format", "npm",
        "--sessions-per-hour", "2",
        "--overwrite"
    ]
    
    with patch("sys.argv", test_argv), \
         patch("os.makedirs"), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", return_value=True), \
         patch("os.path.isfile", return_value=True), \
         patch("tools.run_full_pipeline_deliverables._atomic_write_json") as mock_atomic, \
         patch("tools.run_full_pipeline_deliverables.EventEmitter"), \
         patch("tools.run_full_pipeline_deliverables.check_cancel"), \
         patch("tools.run_full_pipeline_deliverables.glob.glob", return_value=["trace.csv"]), \
         patch("tools.run_full_pipeline_deliverables.pd.read_csv") as mock_read_csv, \
         patch("tools.run_full_pipeline_deliverables._ensure_root_run_report"):
        
        df = MagicMock()
        df.columns = ["time_sec"]
        df.__len__.return_value = 10
        df["time_sec"].values = [0, 600]
        mock_read_csv.return_value = df
        
        try:
            wrapper_main()
        except SystemExit:
            pass
            
        # Verify call to analyze_photometry.py has numeric but NO source
        found_call = False
        for call in mock_run_cmd.call_args_list:
            cmd = call[0][0]
            if "--sessions-per-hour" in cmd:
                assert "--sessions-per-hour-source" not in cmd
                if cmd[cmd.index("--sessions-per-hour")+1] == "2":
                    found_call = True
        assert found_call

        # Verify status.json derive source correctly
        status_written = False
        for call in mock_atomic.call_args_list:
            data = call[0][1]
            if isinstance(data, dict) and data.get("sessions_per_hour") == 2:
                assert data.get("sessions_per_hour_source") == "user-provided"
                status_written = True
        assert status_written

def test_wrapper_stamps_run_report():
    """Verifies that _ensure_root_run_report stamps the report with authoritative metadata."""
    from tools.run_full_pipeline_deliverables import _ensure_root_run_report
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        report = {
            "run_context": {
                "sessions_per_hour": None,
                "sessions_per_hour_source": None,
                "timeline_anchor_mode": None,
                "fixed_daily_anchor_clock": None,
            },
            "derived_settings": {
                "sessions_per_hour": None,
                "sessions_per_hour_source": None,
                "timeline_anchor_mode": None,
                "fixed_daily_anchor_clock": None,
            }
        }
        report_path = os.path.join(tmp_dir, "run_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f)
            
        _ensure_root_run_report(tmp_dir, None, None, None, 
                                run_type="tuning_prep",
                                run_profile="tuning_prep",
                                artifact_contract={"required_for_post_run_tuning": ["phasic_trace_cache.h5", "config_used.yaml"]},
                                intentional_skips={"skipped_outputs": ["day_plots/phasic_sig_iso_day_*.png"]},
                                sessions_per_hour=2, 
                                sessions_per_hour_source="auth-stamp",
                                timeline_anchor_mode="fixed_daily_anchor",
                                fixed_daily_anchor_clock="07:00")
        
        with open(report_path, "r") as f:
            stamped = json.load(f)
            
        assert stamped["run_context"]["sessions_per_hour"] == 2
        assert stamped["run_context"]["sessions_per_hour_source"] == "auth-stamp"
        assert stamped["run_context"]["run_type"] == "tuning_prep"
        assert stamped["run_context"]["run_profile"] == "tuning_prep"
        assert stamped["run_context"]["timeline_anchor_mode"] == "fixed_daily_anchor"
        assert stamped["run_context"]["fixed_daily_anchor_clock"] == "07:00"
        assert stamped["run_context"]["artifact_contract"]["required_for_post_run_tuning"] == [
            "phasic_trace_cache.h5",
            "config_used.yaml",
        ]
        assert stamped["run_context"]["intentional_skips"]["skipped_outputs"] == [
            "day_plots/phasic_sig_iso_day_*.png"
        ]
        assert stamped["run_mode_contract"]["run_type"] == "tuning_prep"
        assert stamped["run_mode_contract"]["intentional_skips"]["skipped_outputs"] == [
            "day_plots/phasic_sig_iso_day_*.png"
        ]
        assert stamped["derived_settings"]["timeline_anchor_mode"] == "fixed_daily_anchor"
        assert stamped["derived_settings"]["fixed_daily_anchor_clock"] == "07:00"


def test_wrapper_run_type_resolution_prefers_tuning_prep_over_preview():
    from tools.run_full_pipeline_deliverables import _resolve_effective_run_type

    assert _resolve_effective_run_type(run_profile="full", preview_first_n=None) == "full"
    assert _resolve_effective_run_type(run_profile="full", preview_first_n=5) == "preview"
    assert _resolve_effective_run_type(run_profile="tuning_prep", preview_first_n=None) == "tuning_prep"
    assert _resolve_effective_run_type(run_profile="tuning_prep", preview_first_n=5) == "tuning_prep"


def test_tuning_prep_skip_plan_reports_nonessential_outputs():
    from tools.run_full_pipeline_deliverables import _skip_plan_for_profile

    plan = _skip_plan_for_profile("tuning_prep", run_tonic_mode=True, run_phasic_mode=True)
    assert plan is not None
    assert plan["forced_traces_only"] is True
    assert "analysis.tonic_analysis" in plan["skipped_phases"]
    assert "plots.phasic_dayplot_bundle" in plan["skipped_phases"]
    assert "plots.tonic_df_timeseries_table" in plan["skipped_phases"]
    assert "_analysis/tonic_out/tonic_trace_cache.h5" in plan["skipped_outputs"]
    assert "summary/tonic_overview.png" in plan["skipped_outputs"]
    assert "day_plots/phasic_sig_iso_day_*.png" in plan["skipped_outputs"]
    assert _skip_plan_for_profile("full", run_tonic_mode=True, run_phasic_mode=True) is None


def test_validate_inputs_rejects_tuning_prep_tonic_only(tmp_path):
    from tools.run_full_pipeline_deliverables import validate_inputs

    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("target_fs_hz: 20.0\n", encoding="utf-8")
    args = SimpleNamespace(
        input=str(input_dir),
        config=str(cfg_path),
        format="npm",
        mode="tonic",
        run_type="tuning_prep",
        sessions_per_hour=2,
        session_duration_s=10.0,
        timeline_anchor_mode="civil",
        fixed_daily_anchor_clock=None,
    )
    with pytest.raises(RuntimeError, match="requires phasic-capable mode"):
        validate_inputs(args)
