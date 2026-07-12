from photometry_pipeline.guided_execution_capabilities import (
    GuidedExecutionCapabilities,
    PRODUCTION_GUIDED_EXECUTION_CAPABILITIES,
)
import pytest
from dataclasses import FrozenInstanceError
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    from gui.main_window import MainWindow

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    value = MainWindow(settings=settings)
    yield value
    value.close()


def test_production_guided_construction_keeps_signal_only_gate_closed(window):
    assert PRODUCTION_GUIDED_EXECUTION_CAPABILITIES.allow_signal_only_f0_execution is False
    assert window._guided_execution_capabilities.allow_signal_only_f0_execution is False


def test_signal_only_lifecycle_override_is_explicit_constructor_dependency(qapp):
    from gui.main_window import MainWindow

    capability = GuidedExecutionCapabilities(allow_signal_only_f0_execution=True)
    window = MainWindow(guided_execution_capabilities=capability)
    try:
        assert window._guided_execution_capabilities is capability
    finally:
        window.close()


def test_capability_is_frozen_after_construction(qapp):
    from gui.main_window import MainWindow

    capability = GuidedExecutionCapabilities(allow_signal_only_f0_execution=True)
    window = MainWindow(guided_execution_capabilities=capability)
    try:
        with pytest.raises(FrozenInstanceError):
            capability.allow_signal_only_f0_execution = False
        assert window._guided_execution_capabilities is capability
    finally:
        window.close()


def test_run_button_has_one_authoritative_guarded_entry_point():
    from pathlib import Path
    import gui.main_window as main_window

    source = Path(main_window.__file__).read_text(encoding="utf-8")
    assert source.count(
        "self._guided_run_btn.clicked.connect(\n"
        "            self._on_guided_run_clicked_backend_guarded\n"
        "        )"
    ) == 1
