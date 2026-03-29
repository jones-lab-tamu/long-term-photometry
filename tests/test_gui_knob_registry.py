"""
Tests for GUI knob registry (schema introspection, allowlists, metadata).
"""

import unittest
from dataclasses import MISSING

from gui.knobs_schema import get_config_field_specs
from gui.knobs_registry import (
    GUI_KNOBS_NORMAL,
    GUI_KNOBS_ADVANCED,
    GUI_KNOBS_DEVELOPER,
    KNOB_META,
    validate_registry_against_schema,
    filter_config_overrides,
)


class TestGuiKnobRegistry(unittest.TestCase):

    def test_config_schema_introspection_contains_expected_fields(self):
        """Introspection successfully reads Config and finds expected fields."""
        specs = get_config_field_specs()
        
        # Must be non-empty
        self.assertTrue(specs, "Schema introspection returned empty dict")
        
        # Must contain stable known keys
        known_keys = [
            "target_fs_hz", 
            "chunk_duration_sec", 
            "baseline_method",
            "peak_threshold_method",
            "event_signal"
        ]
        for k in known_keys:
            self.assertIn(k, specs, f"Expected key {k!r} missing from schema")
            
        # Verify structure of a specific key
        spec = specs["target_fs_hz"]
        self.assertEqual(spec["name"], "target_fs_hz")
        self.assertEqual(spec["type"], float)
        self.assertTrue(spec["has_default"])
        self.assertFalse(spec["optional"])
        
        # Verify Optional handling
        spec_opt = specs.get("preview_first_n")
        if spec_opt:  # It should exist
            self.assertTrue(spec_opt["optional"], "preview_first_n should be optional")

    def test_normalize_type_handles_enum(self):
        """normalize_type correctly normalizes Enum types to value choices."""
        from enum import Enum
        from gui.knobs_schema import normalize_type
        
        class TestEnum(Enum):
            A = "alpha"
            B = "beta"
            C = 42

        norm = normalize_type(TestEnum)
        self.assertEqual(norm["kind"], "enum")
        # Ensure we got the values out
        self.assertCountEqual(norm["choices"], ["alpha", "beta", 42])

        class TestEnumNames(Enum):
            A = object()
            B = object()
            
        # If values aren't string/int scalars, it should fall back to names
        norm_names = normalize_type(TestEnumNames)
        self.assertEqual(norm_names["kind"], "enum")
        self.assertCountEqual(norm_names["choices"], ["A", "B"])

    def test_normalize_type_handles_union(self):
        """normalize_type correctly normalizes non-Optional Unions."""
        from typing import Union
        from gui.knobs_schema import normalize_type
        
        norm = normalize_type(Union[int, str])
        self.assertEqual(norm["kind"], "union")
        self.assertEqual(len(norm["choices"]), 2)
        
        kinds = [c["kind"] for c in norm["choices"]]
        self.assertCountEqual(kinds, ["int", "str"])

    def test_registry_keys_exist_and_do_not_overlap(self):
        """Registry validation passes (all keys exist, no overlap, have meta)."""
        # Should not raise any assertion errors
        validate_registry_against_schema()

    def test_registry_is_constrained_to_scope(self):
        """Registry strictly excludes general preprocessing/resampling knobs."""
        out_of_scope = {
            "target_fs_hz", 
            "chunk_duration_sec", 
            "baseline_method", 
            "baseline_percentile", 
            "lowpass_hz", 
            "filter_order",
            "f0_min_value"
        }
        for knob in out_of_scope:
            self.assertNotIn(knob, GUI_KNOBS_NORMAL, f"Out of scope knob {knob} found in NORMAL")
            self.assertNotIn(knob, GUI_KNOBS_ADVANCED, f"Out of scope knob {knob} found in ADVANCED")

    def test_filter_config_overrides_rejects_unknown_keys(self):
        """Unknown keys (not in Config schema) raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            filter_config_overrides({"fake_made_up_key_123": 42})
        self.assertIn("Unknown config keys", str(cm.exception))
        self.assertIn("fake_made_up_key_123", str(cm.exception))

    def test_filter_config_overrides_blocks_developer_keys_by_default(self):
        """Developer keys are blocked unless allow_developer=True."""
        # Pick a known developer key
        dev_key = list(GUI_KNOBS_DEVELOPER)[0]
        
        # Should fail by default
        with self.assertRaises(ValueError) as cm:
            filter_config_overrides({dev_key: "some_value"})
        self.assertIn("Developer-only config keys not allowed", str(cm.exception))
        self.assertIn(dev_key, str(cm.exception))
        
        # Should succeed with allow_developer=True
        filtered = filter_config_overrides({dev_key: "some_value"}, allow_developer=True)
        self.assertIn(dev_key, filtered)

    def test_filter_config_overrides_rejects_unregistered_keys(self):
        """Valid Config keys that aren't in the GUI registry are rejected."""
        # Find a key in Config that is NOT in the GUI registry right now 
        # (e.g. preview_first_n or rwd_time_col)
        schema = get_config_field_specs()
        all_knobs = GUI_KNOBS_NORMAL | GUI_KNOBS_ADVANCED | GUI_KNOBS_DEVELOPER
        unregistered = set(schema.keys()) - all_knobs
        
        if unregistered:
            unreg_key = list(unregistered)[0]
            with self.assertRaises(ValueError) as cm:
                filter_config_overrides({unreg_key: "val"})
            self.assertIn("Config keys not in GUI allowlist", str(cm.exception))
            self.assertIn(unreg_key, str(cm.exception))

    def test_filter_config_overrides_passes_normal_keys(self):
        """Allowlisted NORMAL and ADVANCED keys pass through intact."""
        n_key = list(GUI_KNOBS_NORMAL)[0]
        a_key = list(GUI_KNOBS_ADVANCED)[0]
        
        overrides = {n_key: "val1", a_key: "val2"}
        filtered = filter_config_overrides(overrides)
        
        self.assertEqual(filtered, overrides)
        self.assertIsNot(filtered, overrides)

    def test_filter_config_overrides_accepts_baseline_subtract_before_fit(self):
        """baseline_subtract_before_fit is allowlisted and must pass filtering."""
        filtered = filter_config_overrides({"baseline_subtract_before_fit": True})
        self.assertEqual(filtered, {"baseline_subtract_before_fit": True})

    def test_filter_config_overrides_accepts_robust_event_reject_keys(self):
        filtered = filter_config_overrides(
            {
                "robust_event_reject_max_iters": 4,
                "robust_event_reject_residual_z_thresh": 3.1,
            }
        )
        self.assertEqual(
            filtered,
            {
                "robust_event_reject_max_iters": 4,
                "robust_event_reject_residual_z_thresh": 3.1,
            },
        )

    def test_run_spec_generate_derived_config_uses_filtered_overrides(self):
        """generate_derived_config filters overrides and raises on unknown."""
        import os
        import tempfile
        import yaml
        from gui.run_spec import RunSpec
        
        tmp_dir = tempfile.mkdtemp()
        try:
            # Create a valid base config
            base_config = {"event_signal": "dff"}
            config_path = os.path.join(tmp_dir, "base_config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(base_config, f)
                
            run_dir = os.path.join(tmp_dir, "run_dir")
            
            # 1. Test with unknown override -> ValueError
            spec_bad = RunSpec(
                config_source_path=config_path,
                config_overrides={"event_signal": "delta_f", "fake_key": 1}
            )
            with self.assertRaises(ValueError) as cm:
                spec_bad.generate_derived_config(run_dir)
            self.assertIn("fake_key", str(cm.exception))
            
            # 2. Test with allowed override -> Success
            spec_good = RunSpec(
                config_source_path=config_path,
                config_overrides={"event_signal": "delta_f"}
            )
            out_path = spec_good.generate_derived_config(run_dir)
            self.assertTrue(os.path.exists(out_path))
            
            # Verify the override was applied
            with open(out_path, "r") as f:
                loaded = yaml.safe_load(f)
            self.assertEqual(loaded["event_signal"], "delta_f")
            
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
