import sys
import pytest
import subprocess

def test_no_gui_modules_in_core():
    """
    Enforces that importing core modules does not trigger GUI (Qt/Matplotlib) imports.
    Runs in a subprocess to isolate from pytest collection side-effects (e.g. other tests importing pipeline).
    """
    code = """
import sys
# 1. Clean verify of core modules
import photometry_pipeline.core.baseline
import photometry_pipeline.core.normalization
import photometry_pipeline.core.types
import photometry_pipeline.core.preprocessing
import photometry_pipeline.core.feature_extraction

# Assert Qt is absent
offenders = []
for mod in ["PySide6", "shiboken6", "PyQt5", "matplotlib.pyplot"]:
    if mod in sys.modules:
        offenders.append(mod)

if offenders:
    print(f"FAILURE: Found GUI modules in sys.modules: {offenders}")
    sys.exit(1)
"""
    subprocess.check_call([sys.executable, "-c", code])

