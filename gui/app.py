"""
Photometry Pipeline Deliverables, GUI Application

Usage:
    python -m gui.app
"""

import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("JonesLab")
    app.setApplicationName("Photometry Pipeline Deliverables")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
