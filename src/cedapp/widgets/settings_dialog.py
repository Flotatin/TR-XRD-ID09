"""Settings dialog widget."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from cedapp.utils import paths

CONFIG_DIR = paths.get_config_dir(require=False)


class SettingsDialog(QDialog):
    """Simple dialog to choose theme and configuration file."""

    def __init__(self, current_theme="Light", config_path="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        if current_theme.capitalize() in ["Light", "Dark"]:
            self.theme_combo.setCurrentText(current_theme.capitalize())

        self.path_edit = QLineEdit(config_path)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)

        form_layout = QVBoxLayout()
        form_layout.addWidget(QLabel("Theme"))
        form_layout.addWidget(self.theme_combo)
        form_layout.addWidget(QLabel("Configuration file"))
        form_layout.addLayout(path_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self.result = (current_theme, config_path)

    def browse_file(self):
        current_path = Path(self.path_edit.text()).expanduser() if self.path_edit.text() else CONFIG_DIR
        if current_path.is_file():
            start_dir = current_path.parent
        else:
            start_dir = current_path if current_path.exists() else CONFIG_DIR

        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Configuration file",
            str(start_dir),
            "Text Files (*.txt);;All Files (*)",
        )
        if fname:
            self.path_edit.setText(fname)

    def accept(self):
        self.result = (self.theme_combo.currentText(), self.path_edit.text())
        super().accept()
