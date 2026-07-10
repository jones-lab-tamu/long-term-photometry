from pathlib import Path
import sys

import pytest


_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_MODAL_METHODS = ("information", "warning", "critical", "question", "about")


@pytest.fixture
def no_real_modals(monkeypatch):
    """Fail instead of opening a real QMessageBox during an unattended test run.

    A GUI test that reaches an unstubbed modal blocks the whole session on a
    dialog nobody is there to dismiss. Tests that legitimately exercise a dialog
    still monkeypatch it themselves; their patch is applied after this one and
    wins.
    """
    from PySide6.QtWidgets import QMessageBox

    def _blocked(name):
        def _raise(*args, **_kwargs):
            title = args[1] if len(args) > 1 else "<no title>"
            text = args[2] if len(args) > 2 else "<no text>"
            raise AssertionError(
                f"Test opened a real QMessageBox.{name} dialog: {title!r}: {text!r}. "
                "Stub the dialog, or give the test a fixture the code under test accepts."
            )

        return _raise

    for name in _MODAL_METHODS:
        monkeypatch.setattr(QMessageBox, name, staticmethod(_blocked(name)), raising=False)
