"""Tabs "Phases" and "Pics": the phases and the peaks of the current spectrum and the commands of the manual work."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from trxrd.gui.presenters.display import PEAK_MODELS
from trxrd.gui.views.images import AXIS_WIDTH, limit_view, span_of
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
        self.on_phase: Optional[Callable[[], None]] = None
        self.on_line: Optional[Callable[[str, tuple, bool], None]] = None
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
        self.lines = QListWidget()
        self.lines.setToolTip("Raies de la phase choisie : cocher pour lui donner un pic, décocher pour le retirer")
        self.lines.itemChanged.connect(self._line_changed)
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

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(self.lines)
        split.setSizes([150, 150])
        layout = QVBoxLayout(self)
        layout.addWidget(split, 1)
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
        if not self._showing and self.on_phase is not None:
            self.on_phase()

    def _show_lines(self, lines) -> None:
        self.lines.blockSignals(True)
        self.lines.clear()
        for line in lines:
            item = QListWidgetItem(f"{line.label}   {line.two_theta:.3f}°   {line.intensity:.0f} %")
            item.setData(Qt.ItemDataRole.UserRole, line.hkl)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if line.used else Qt.CheckState.Unchecked)
            self.lines.addItem(item)
        self.lines.blockSignals(False)

    def _line_changed(self, item) -> None:
        if self.on_line is not None:
            used = item.checkState() == Qt.CheckState.Checked
            hkl = tuple(int(i) for i in item.data(Qt.ItemDataRole.UserRole))
            self.on_line(self.card.currentData(), hkl, used)

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
            self._show_lines(choices.lines)
            self.fit_selection.setText(f"Ajuster la sélection ({choices.selected})")
            self.fit_all.setText(f"Ajuster tous ({choices.count})")
            self.paste.setEnabled(choices.copied_from is not None)
            self.copy.setEnabled(bool(choices.on_spectrum))
        finally:
            self._showing = False

    def show_message(self, text: str) -> None:
        self.message.setText(text)

    def chosen_phase(self) -> Optional[str]:
        return self.card.currentData()

    def clear(self) -> None:
        self.table.show_rows([], [])
        self._show_lines([])
        self.card.clear()
        self._on_spectrum = {}
        self.message.setText("")
        self.setEnabled(False)


def _value_box(decimals: int, low: float, high: float, step: float, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setKeyboardTracking(False)
    box.setDecimals(decimals)
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setSuffix(suffix)
    return box


class PeaksPanel(QWidget):
    """Peaks of the current spectrum: the chosen one, its parameters, its fit alone; editing with the mouse."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_edit: Optional[Callable[[bool], None]] = None
        self.on_row: Optional[Callable[[int], None]] = None
        self.on_remove: Optional[Callable[[], None]] = None
        self.on_toggle: Optional[Callable[[], None]] = None
        self.on_model: Optional[Callable[[str], None]] = None
        self.on_value: Optional[Callable[[str, float], None]] = None
        self.on_shape: Optional[Callable[[int, float], None]] = None
        self.on_fit_peak: Optional[Callable[[], None]] = None
        self.on_fit_settings: Optional[Callable[..., None]] = None
        self._showing = False

        self.table = RowsTable()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._row_chosen)

        self.zoom = pg.PlotWidget()  # the chosen peak alone
        self.zoom.getAxis("left").setWidth(AXIS_WIDTH)
        self.zoom.showGrid(x=True, y=True, alpha=0.15)
        self.zoom.setMenuEnabled(False)
        self.zoom.hideButtons()
        self.zoom_data = self.zoom.plot(pen=pg.mkPen("#404040", width=1))
        self.zoom_curve = self.zoom.plot(pen=pg.mkPen("#d62728", width=1.5), fillLevel=0, fillBrush=pg.mkBrush(214, 39, 40, 50))
        self.zoom_left = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(128, 128, 128, 50), pen=pg.mkPen(None))
        self.zoom_right = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(128, 128, 128, 50), pen=pg.mkPen(None))
        for zone in (self.zoom_left, self.zoom_right):
            zone.setZValue(-10)
            self.zoom.addItem(zone, ignoreBounds=True)

        self.title = QLabel("Aucun pic choisi")
        self.model = QComboBox()
        self.model.addItems(list(PEAK_MODELS))
        self.model.setToolTip("Profil du pic choisi ; sans pic choisi, celui des prochains pics")
        self.center = _value_box(3, 0.0, 180.0, 0.01, " °")
        self.height_box = _value_box(1, 0.0, 1e9, 10.0)
        self.width_box = _value_box(4, 0.0001, 20.0, 0.005, " °")
        self.width_box.setToolTip("σ du profil. La largeur à mi-hauteur est affichée à côté du titre.")
        self.shapes = [_value_box(3, -1000.0, 1000.0, 0.05) for _ in range(2)]
        self.shape_labels = [QLabel("") for _ in range(2)]
        self.fit_window = _value_box(1, 1.0, 50.0, 1.0, " σ")
        self.fit_window.setToolTip("Ajustement d'un seul pic : points à moins de ce nombre de σ de son centre")
        self.variation = _value_box(0, 1.0, 1000.0, 10.0, " %")
        self.variation.setToolTip("Liberté de la largeur dans les ajustements : de w / (1 + Var) à w × (1 + Var)")
        self.fit = QPushButton("Ajuster ce pic (Maj+R)")
        self.remove = QPushButton("Supprimer (Suppr)")
        self.toggle = QPushButton("Activer / désactiver")
        self.toggle.setToolTip("Un pic désactivé reste dans la liste, mais sort de la courbe, de l'ajustement et de la maille")
        self.edit = QCheckBox("Modifier les pics à la souris (Ctrl+E)")
        self.edit.setToolTip("Tirer une poignée : 2θ et hauteur. Clic : poser le curseur. Ctrl + clic ou C : ajouter un pic,\n"
                             "sur la raie libre de la phase choisie (onglet Phases) s'il y en a une près, sinon pic libre.")
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #8a5a00")

        self.edit.toggled.connect(lambda on: self.on_edit and self.on_edit(on))
        self.remove.clicked.connect(lambda: self.on_remove and self.on_remove())
        self.toggle.clicked.connect(lambda: self.on_toggle and self.on_toggle())
        self.fit.clicked.connect(lambda: self.on_fit_peak and self.on_fit_peak())
        self.model.currentTextChanged.connect(self._model_chosen)
        for name, box in (("center", self.center), ("height", self.height_box), ("width", self.width_box)):
            box.valueChanged.connect(lambda value, name=name: self._value(name, value))
        for position, box in enumerate(self.shapes):
            box.valueChanged.connect(lambda value, position=position: self._shape(position, value))
        self.fit_window.valueChanged.connect(lambda value: self._fit_setting(fit_window=float(value)))
        self.variation.valueChanged.connect(lambda value: self._fit_setting(fit_variation=float(value) / 100))

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel("Modèle"), 0, 0)
        grid.addWidget(self.model, 0, 1, 1, 3)
        grid.addWidget(QLabel("2θ"), 1, 0)
        grid.addWidget(self.center, 1, 1)
        grid.addWidget(QLabel("Hauteur"), 1, 2)
        grid.addWidget(self.height_box, 1, 3)
        grid.addWidget(QLabel("σ"), 2, 0)
        grid.addWidget(self.width_box, 2, 1)
        grid.addWidget(self.shape_labels[0], 3, 0)
        grid.addWidget(self.shapes[0], 3, 1)
        grid.addWidget(self.shape_labels[1], 3, 2)
        grid.addWidget(self.shapes[1], 3, 3)
        grid.addWidget(QLabel("Fenêtre"), 4, 0)
        grid.addWidget(self.fit_window, 4, 1)
        grid.addWidget(QLabel("Var"), 4, 2)
        grid.addWidget(self.variation, 4, 3)
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addWidget(self.title)
        settings_layout.addLayout(grid)
        settings_layout.addLayout(_row(self.fit, self.remove, self.toggle))
        settings_layout.addWidget(self.edit)
        settings_layout.addWidget(self.message)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(self.zoom)
        split.addWidget(settings)
        split.setSizes([130, 130, 240])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(split)
        self._show_shape_labels((), ())

    # ------------------------------------------------------------------ gestures

    def _row_chosen(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows and self.on_row is not None:
            self.on_row(rows[0].row())

    def _model_chosen(self, model: str) -> None:
        if not self._showing and self.on_model is not None:
            self.on_model(model)

    def _value(self, name: str, value: float) -> None:
        if not self._showing and self.on_value is not None:
            self.on_value(name, float(value))

    def _shape(self, position: int, value: float) -> None:
        if not self._showing and self.on_shape is not None:
            self.on_shape(position, float(value))

    def _fit_setting(self, **change) -> None:
        if not self._showing and self.on_fit_settings is not None:
            self.on_fit_settings(**change)

    # ------------------------------------------------------------------ presenter

    def show_rows(self, columns, rows) -> None:
        self.table.blockSignals(True)
        self.table.show_rows(columns, rows)
        self.table.blockSignals(False)

    def select_row(self, row: Optional[int]) -> None:
        self.table.blockSignals(True)
        if row is None:
            self.table.clearSelection()
        else:
            self.table.selectRow(row)
        self.table.blockSignals(False)

    def _show_shape_labels(self, labels, bounds) -> None:
        for k, (label, box) in enumerate(zip(self.shape_labels, self.shapes)):
            shown = k < len(labels)
            label.setText(labels[k] if shown else "")
            label.setVisible(shown)
            box.setVisible(shown)
            if shown:
                box.setRange(*bounds[k])

    def show_peak(self, detail, settings) -> None:
        """The chosen peak (display.PeakDetail) or None, and the settings of the fits."""
        self._showing = True
        try:
            if settings is not None:
                self.fit_window.setValue(settings.fit_window)
                self.variation.setValue(100 * (settings.fit_variation if settings.fit_variation is not None else 3.0))
            for box in (self.center, self.height_box, self.width_box, *self.shapes):
                box.setEnabled(detail is not None)
            for button in (self.fit, self.remove, self.toggle):
                button.setEnabled(detail is not None)
            if detail is None:
                self.title.setText("Aucun pic choisi")
                self._show_shape_labels((), ())
                if settings is not None:
                    self.model.setCurrentText(settings.peak_model)
                return
            fwhm = f"  (largeur à mi-hauteur {detail.fwhm:.3f}°)" if np.isfinite(detail.fwhm) else ""
            self.title.setText(f"{detail.title}{fwhm}")
            self.model.setCurrentText(detail.model)
            self.center.setValue(detail.center)
            self.height_box.setValue(detail.height if np.isfinite(detail.height) else 0.0)
            self.width_box.setValue(detail.width if np.isfinite(detail.width) else 0.0001)
            self._show_shape_labels(detail.shape_labels, detail.shape_bounds)
            for box, value in zip(self.shapes, detail.shape):
                box.setValue(value)
        finally:
            self._showing = False

    def show_zoom(self, zoom) -> None:
        """The chosen peak alone (display.PeakZoom), or None."""
        if zoom is None:
            self.zoom_data.setData([], [])
            self.zoom_curve.setData([], [])
            for zone in (self.zoom_left, self.zoom_right):
                zone.hide()
            limit_view(self.zoom)
            return
        low, high = float(zoom.x[0]), float(zoom.x[-1])
        limit_view(self.zoom, (low, high), span_of(zoom.data, zoom.curve, [0.0]))
        self.zoom_data.setData(zoom.x, zoom.data)
        self.zoom_curve.setData(zoom.x, zoom.curve)
        self.zoom_left.setRegion((low, zoom.window[0]))
        self.zoom_right.setRegion((zoom.window[1], high))
        for zone in (self.zoom_left, self.zoom_right):
            zone.show()
        self.zoom.getPlotItem().enableAutoRange()

    def show_message(self, text: str) -> None:
        self.message.setText(text)
