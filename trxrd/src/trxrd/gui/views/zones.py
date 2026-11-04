"""Tab "Zones": fit zone of the current spectrum and excluded zones of the run, both drawn on the spectrum."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from trxrd.gui.views.panels import RowsTable
from trxrd.gui.views.treatment import _note, _spin

COLUMNS = ("début (°)", "fin (°)")


class ZonesPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_edit: Optional[Callable[[bool], None]] = None
        self.on_fit_zone: Optional[Callable[[float, float], None]] = None
        self.on_whole: Optional[Callable[[], None]] = None
        self.on_propagate: Optional[Callable[[str], None]] = None
        self.on_row: Optional[Callable[[int], None]] = None
        self.on_remove: Optional[Callable[[], None]] = None
        self._showing = False
        self.edit = QCheckBox("Modifier à la souris (Z)")
        self.edit.setToolTip("Tirer les deux traits de la zone d'ajustement, ou une zone exclue.\n"
                             "Maj + glisser : tracer une zone exclue. Ctrl + Maj + glisser : tracer la zone d'ajustement.\n"
                             "Clic sur une zone exclue, puis Suppr : la retirer.")
        self.low, self.high = _spin(3, 0.0, 180.0, 0.05), _spin(3, 0.0, 180.0, 0.05)
        for box in (self.low, self.high):
            box.setSuffix(" °")
            box.valueChanged.connect(self._fit_zone_typed)
        self.whole = QPushButton("Tout le spectre")
        self.selected = QPushButton("Appliquer à la sélection")
        self.selected.setToolTip("Spectres choisis dans l'onglet Liste, ou zone de spectres des graphes dDAC")
        self.all = QPushButton("Appliquer à tous les spectres")
        self.all.setToolTip("Tous les spectres, et la plage de l'analyse automatique")
        self.table = RowsTable()
        self.remove = QPushButton("Retirer la zone choisie")
        self.remove.setEnabled(False)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #8a5a00")
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #8a5a00")
        self.edit.toggled.connect(lambda on: self.on_edit and self.on_edit(on))
        self.whole.clicked.connect(lambda: self.on_whole and self.on_whole())
        self.selected.clicked.connect(lambda: self.on_propagate and self.on_propagate("selection"))
        self.all.clicked.connect(lambda: self.on_propagate and self.on_propagate("all"))
        self.remove.clicked.connect(lambda: self.on_remove and self.on_remove())
        self.table.itemSelectionChanged.connect(self._row_chosen)

        limits = QHBoxLayout()
        for widget in (QLabel("de"), self.low, QLabel("à"), self.high, self.whole):
            limits.addWidget(widget)
        limits.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.selected)
        buttons.addWidget(self.all)
        layout = QVBoxLayout(self)
        layout.addWidget(self.edit)
        layout.addWidget(_note("Maj + glisser sur le spectre : zone exclue. Ctrl + Maj + glisser : zone d'ajustement."))
        layout.addWidget(QLabel("<b>Zone d'ajustement</b> (ce spectre)"))
        layout.addWidget(_note("Hors de la zone, les pics ne sont pas ajustés et ne comptent pas dans la pression."))
        layout.addLayout(limits)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("<b>Zones exclues</b> (tout le run)"))
        layout.addWidget(_note("Joint, diamant… : aucune raie n'y est attribuée, à la prochaine analyse ou au "
                               "prochain placement d'une phase. Leurs pics restent ajustés comme pics libres."))
        layout.addWidget(self.table, 1)
        layout.addWidget(self.remove)
        layout.addWidget(self.note)
        layout.addWidget(self.message)
        self.setEnabled(False)

    # ------------------------------------------------------------------ gestures

    def _fit_zone_typed(self, *_) -> None:
        if not self._showing and self.on_fit_zone is not None:
            self.on_fit_zone(self.low.value(), self.high.value())

    def _row_chosen(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows and self.on_row is not None:
            self.on_row(rows[0].row())

    # ------------------------------------------------------------------ presenter

    def show_zones(self, choices) -> None:
        self._showing = True
        self.setEnabled(True)
        for box, value in zip((self.low, self.high), choices.fit):
            box.setValue(value)
        self._showing = False
        self.table.blockSignals(True)
        self.table.show_rows(COLUMNS, [[f"{low:.3f}", f"{high:.3f}"] for low, high in choices.excluded])
        if choices.chosen is None:
            self.table.clearSelection()
        else:
            self.table.selectRow(choices.chosen)
        self.table.blockSignals(False)
        self.remove.setEnabled(choices.chosen is not None)
        self.selected.setText(f"Appliquer à la sélection ({max(choices.selected, 1)})")
        self.all.setText(f"Appliquer à tous les spectres ({choices.count})")
        self.note.setText(choices.note)

    def show_message(self, text: str) -> None:
        self.message.setText(text)

    def clear(self) -> None:
        self.table.blockSignals(True)
        self.table.show_rows(COLUMNS, [])
        self.table.blockSignals(False)
        self.note.setText("")
        self.message.setText("")
        self.setEnabled(False)
