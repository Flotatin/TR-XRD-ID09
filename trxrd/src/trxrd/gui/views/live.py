"""Dialog "Mode Live": the folder to watch, where the analyses go, what is done with each new scan."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from trxrd.services.live import LiveSettings


def _text(path: Path) -> str:
    return "" if str(path) in ("", ".") else str(path)


class LiveDialog(QDialog):
    def __init__(self, suggestion: LiveSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mode Live")
        self.folder, self.output = QLineEdit(_text(suggestion.folder)), QLineEdit(_text(suggestion.output))
        self.folder.setToolTip("Un jeu de données (…/Water05/Water05_0001), un échantillon, ou tout RAW_DATA")
        self.analyse = QCheckBox("Analyser chaque scan (phases et réglages de ce run)")
        self.analyse.setChecked(suggestion.analyse)
        self.analyse.setToolTip("Analyse automatique avec suivi des pics, sinon intégration seule")
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #8a5a00")
        form = QFormLayout()
        form.addRow("Dossier surveillé", self._browsed(self.folder))
        form.addRow("Analyses dans", self._browsed(self.output))
        note = QLabel("Chaque nouveau scan reprend la calibration, les phases et le traitement du run affiché. Il est "
                      "enregistré dès sa fin, et s'ouvre quand le run affiché n'a plus de modification à enregistrer.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #606060")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addLayout(form)
        layout.addWidget(self.analyse)
        layout.addWidget(self.message)
        layout.addWidget(buttons)
        self.resize(560, self.sizeHint().height())

    def _browsed(self, field: QLineEdit) -> QHBoxLayout:
        def choose() -> None:
            folder = QFileDialog.getExistingDirectory(self, "Choisir un dossier", field.text())
            if folder:
                field.setText(folder)

        button = QPushButton("Parcourir…")
        button.clicked.connect(choose)
        row = QHBoxLayout()
        row.addWidget(field, 1)
        row.addWidget(button)
        return row

    def answer(self) -> LiveSettings:
        return LiveSettings(Path(self.folder.text().strip()), Path(self.output.text().strip()), self.analyse.isChecked())

    def accept(self) -> None:
        if not self.folder.text().strip() or not Path(self.folder.text().strip()).is_dir():
            self.message.setText("Ce dossier n'existe pas : choisir le dossier où arrivent les scans.")
        elif not self.output.text().strip():
            self.message.setText("Choisir où enregistrer les analyses.")
        else:
            super().accept()
