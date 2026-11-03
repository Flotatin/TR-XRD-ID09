"""Tab "Phases": the phases of the current spectrum and the commands of the manual work."""

from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from trxrd.gui.views.panels import RowsTable


def _row(*widgets) -> QHBoxLayout:
    row = QHBoxLayout()
    for widget in widgets:
        row.addWidget(QLabel(widget) if isinstance(widget, str) else widget)
    return row


class PhasesPanel(QWidget):
    """Choose a phase of the run and its pressure: on the spectrum, its lines follow the pressure at once."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_add: Optional[Callable[[str, float], None]] = None
        self.on_remove: Optional[Callable[[str], None]] = None
        self.on_pressure: Optional[Callable[[str, float], None]] = None
        self.on_fit: Optional[Callable[[], None]] = None
        self.on_fit_series: Optional[Callable[[str, bool], None]] = None
        self.on_copy: Optional[Callable[[], None]] = None
        self.on_paste: Optional[Callable[[], None]] = None
        self.on_clear: Optional[Callable[[], None]] = None
        self._on_spectrum = {}
        self._showing = False

        self.table = RowsTable()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._row_chosen)
        self.card = QComboBox()
        self.card.currentIndexChanged.connect(self._card_chosen)
        self.pressure = QDoubleSpinBox()
        self.pressure.setKeyboardTracking(False)
        self.pressure.setRange(-5.0, 1000.0)
        self.pressure.setDecimals(2)
        self.pressure.setSingleStep(0.05)
        self.pressure.setSuffix(" GPa")
        self.pressure.setToolTip("Sur une phase du spectre, ses raies suivent la pression (flèches, molette)")
        self.pressure.valueChanged.connect(self._pressure_changed)
        self.add = QPushButton("Ajouter")
        self.remove = QPushButton("Retirer")
        self.fit = QPushButton("Ajuster ce spectre")
        self.fit_selection = QPushButton("Ajuster la sélection")
        self.fit_all = QPushButton("Ajuster tous")
        self.follow = QCheckBox("suivre : à partir du spectre courant, chacun part de son voisin")
        self.follow.setChecked(True)
        self.follow.setToolTip("Décoché : chaque spectre part de son propre modèle")
        self.copy = QPushButton("Copier le modèle")
        self.paste = QPushButton("Coller sur la sélection")
        self.clear_model = QPushButton("Vider la sélection")
        self.clear_model.setToolTip("Retirer les phases et les pics libres de la sélection (ou du spectre courant)")
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #8a5a00")

        self.add.clicked.connect(lambda: self.on_add and self.on_add(self.card.currentData(), self.pressure.value()))
        self.remove.clicked.connect(lambda: self.on_remove and self.on_remove(self.card.currentData()))
        self.fit.clicked.connect(lambda: self.on_fit and self.on_fit())
        self.fit_selection.clicked.connect(self._series("selection"))
        self.fit_all.clicked.connect(self._series("all"))
        self.copy.clicked.connect(lambda: self.on_copy and self.on_copy())
        self.paste.clicked.connect(lambda: self.on_paste and self.on_paste())
        self.clear_model.clicked.connect(lambda: self.on_clear and self.on_clear())

        layout = QVBoxLayout(self)
        layout.addWidget(self.table, 1)
        layout.addLayout(_row(self.card, self.pressure))
        layout.addLayout(_row(self.add, self.remove, self.fit))
        layout.addLayout(_row(self.fit_selection, self.fit_all))
        layout.addWidget(self.follow)
        layout.addLayout(_row(self.copy, self.paste, self.clear_model))
        layout.addWidget(self.message)
        self.setEnabled(False)

    def _series(self, scope: str) -> Callable[[], None]:
        def clicked() -> None:
            if self.on_fit_series is not None:
                self.on_fit_series(scope, self.follow.isChecked())
        return clicked

    # ------------------------------------------------------------------ gestures

    def _row_chosen(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        names: List[str] = list(self._on_spectrum)
        if rows and not self._showing and rows[0].row() < len(names):
            self.card.setCurrentIndex(max(self.card.findData(names[rows[0].row()]), 0))

    def _card_chosen(self, *_) -> None:
        name = self.card.currentData()
        here = name in self._on_spectrum
        if here:
            self._set_pressure(self._on_spectrum[name])
        self.remove.setEnabled(here)
        self.add.setEnabled(name is not None and not here)

    def _set_pressure(self, value: float) -> None:
        self.pressure.blockSignals(True)
        self.pressure.setValue(value)
        self.pressure.blockSignals(False)

    def _pressure_changed(self, value: float) -> None:
        name = self.card.currentData()
        if not self._showing and name in self._on_spectrum and self.on_pressure is not None:
            self.on_pressure(name, float(value))

    # ------------------------------------------------------------------ presenters

    def show_rows(self, columns, rows) -> None:
        self._showing = True
        try:
            self.table.show_rows(columns, rows)
        finally:
            self._showing = False

    def show_model(self, choices) -> None:
        self._showing = True
        try:
            self.setEnabled(True)
            self._on_spectrum = dict(choices.on_spectrum)
            chosen = self.card.currentData()
            names = [self.card.itemData(i) for i in range(self.card.count())]
            if names != [name for name, _ in choices.cards]:
                self.card.blockSignals(True)
                self.card.clear()
                for name, label in choices.cards:
                    self.card.addItem(label, name)
                self.card.setCurrentIndex(max(self.card.findData(chosen), 0))
                self.card.blockSignals(False)
            self._card_chosen()
            self.fit_selection.setText(f"Ajuster la sélection ({choices.selected})")
            self.fit_all.setText(f"Ajuster tous ({choices.count})")
            self.paste.setEnabled(choices.copied_from is not None)
            self.copy.setEnabled(bool(choices.on_spectrum))
        finally:
            self._showing = False

    def show_message(self, text: str) -> None:
        self.message.setText(text)

    def clear(self) -> None:
        self.table.show_rows([], [])
        self.card.clear()
        self._on_spectrum = {}
        self.message.setText("")
        self.setEnabled(False)
