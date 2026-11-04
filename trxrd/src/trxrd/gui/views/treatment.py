"""Panels "Traitement" (background and filter of the spectra) and "Détection" (peaks found, to check them)."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

from trxrd.domain import Processing
from trxrd.gui.views.panels import RowsTable

# (key, label, (first parameter, second parameter)): (label, decimals, low, high, step)
FILTERS = (
    ("svg", "Savitzky-Golay", (("Fenêtre (points)", 0, 3, 201, 2), ("Ordre", 0, 0, 10, 1))),
    ("fft", "FFT (coupe une bande de fréquences)", (("Coupure basse (1/°)", 2, 0, 1000, 0.5),
                                                    ("Coupure haute (1/°)", 2, 0, 1000, 0.5))),
    ("none", "aucun", (("—", 0, 0, 1000, 1), ("—", 0, 0, 1000, 1))),
)
# parameters given when a filter is chosen; FFT: frequencies above 15 /deg removed (peaks of 0.05 deg are kept)
FILTER_DEFAULTS = {"svg": (9.0, 2.0), "fft": (15.0, 1000.0), "none": (0.0, 0.0)}


def _spin(decimals: int, low: float, high: float, step: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setKeyboardTracking(False)  # applied when the value is complete, not at every key
    box.setDecimals(decimals)
    box.setRange(low, high)
    box.setSingleStep(step)
    return box


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #606060")
    return label


class ProcessingPanel(QWidget):
    """Background (SNIP) and filter of the current spectrum, applied at once; propagated on demand."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_change: Optional[Callable[[Processing], None]] = None
        self.on_propagate: Optional[Callable[[str], None]] = None
        self._showing = False
        self.degree = QSpinBox()
        self.degree.setKeyboardTracking(False)
        self.degree.setRange(0, 200)
        self.degree.setToolTip("Lissage du fond SNIP : fenêtre de 5 × n points (0 : pas de lissage)")
        self.window = _spin(2, 0.0, 20.0, 0.25)
        self.window.setSuffix(" °")
        self.window.setSpecialValueText("auto")
        self.window.setToolTip("Le fond suit tout ce qui est plus large que cette fenêtre.\n"
                               "auto : estimée par pybaselines ; elle échoue avec des pics larges (liquide,\n"
                               "amorphe) : mettre alors un peu plus que leur largeur (1 à 2°).")
        self.filter = QComboBox()
        for key, label, _ in FILTERS:
            self.filter.addItem(label, key)
        self.p1, self.p2 = _spin(0, 3, 201, 2), _spin(0, 0, 10, 1)
        self.p1_label, self.p2_label = QLabel(), QLabel()
        self.all = QPushButton("Appliquer à tous les spectres")
        self.selected = QPushButton("Appliquer à la sélection")
        self.selected.setToolTip("Spectres choisis dans l'onglet Liste, ou zone de spectres des graphes dDAC")
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #8a5a00")
        for box in (self.degree, self.window, self.p1, self.p2):
            box.valueChanged.connect(self._changed)
        self.filter.currentIndexChanged.connect(self._filter_chosen)
        self.all.clicked.connect(lambda: self.on_propagate and self.on_propagate("all"))
        self.selected.clicked.connect(lambda: self.on_propagate and self.on_propagate("selection"))

        form = QFormLayout()
        form.addRow("Fond (SNIP) : lissage", self.degree)
        form.addRow("Fond : fenêtre", self.window)
        form.addRow("Filtre", self.filter)
        form.addRow(self.p1_label, self.p1)
        form.addRow(self.p2_label, self.p2)
        buttons = QHBoxLayout()
        buttons.addWidget(self.all)
        buttons.addWidget(self.selected)
        layout = QVBoxLayout(self)
        layout.addWidget(_note("Réglé pour le spectre courant, puis propagé. Le spectre est montré filtré, "
                               "fond soustrait (Affichage > Signal brut pour le brut). L'ajustement utilise "
                               "toujours le signal brut moins le fond."))
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.message)
        layout.addStretch(1)
        self._labels("svg")
        self.setEnabled(False)

    def _labels(self, kind: str) -> None:
        parameters = next(p for key, _, p in FILTERS if key == kind)
        for box, label, (text, decimals, low, high, step) in zip((self.p1, self.p2), (self.p1_label, self.p2_label),
                                                                  parameters):
            box.blockSignals(True)
            box.setDecimals(decimals)
            box.setRange(low, high)
            box.setSingleStep(step)
            box.blockSignals(False)
            label.setText(text)
            box.setEnabled(kind != "none")

    def processing(self) -> Processing:
        return Processing(int(self.degree.value()), self.filter.currentData(), (self.p1.value(), self.p2.value()),
                          self.window.value() or None)

    def _filter_chosen(self, *_) -> None:
        kind = self.filter.currentData()
        self._labels(kind)
        for box, value in zip((self.p1, self.p2), FILTER_DEFAULTS[kind]):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self._changed()

    def _changed(self, *_) -> None:
        if not self._showing and self.on_change is not None:
            self.on_change(self.processing())

    def show_processing(self, processing: Processing, selected: int, count: int) -> None:
        self._showing = True
        try:
            self.setEnabled(True)
            self.filter.setCurrentIndex(max(self.filter.findData(processing.filter_type), 0))
            self._labels(self.filter.currentData())
            self.degree.setValue(int(processing.baseline_degree))
            self.window.setValue(processing.baseline_window or 0.0)
            self.p1.setValue(float(processing.filter_params[0]))
            self.p2.setValue(float(processing.filter_params[1]))
            self.all.setText(f"Appliquer à tous les spectres ({count})")
            self.selected.setText(f"Appliquer à la sélection ({selected})")
        finally:
            self._showing = False

    def show_message(self, text: str) -> None:
        self.message.setText(text)

    def clear(self) -> None:
        self.message.setText("")
        self.setEnabled(False)


class DetectionPanel(QWidget):
    """Settings of the detection (those of the automatic analysis) and the peaks found in the current spectrum."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_options: Optional[Callable[..., None]] = None
        self.on_visible: Optional[Callable[[bool], None]] = None
        self._showing = False
        self.visible = QCheckBox("Montrer les pics détectés sur le spectre")
        self.visible.setChecked(True)
        self.visible.toggled.connect(lambda shown: self.on_visible and self.on_visible(shown))
        self.min_snr = _spin(1, 1.0, 1000.0, 0.5)
        self.min_snr.setToolTip("Hauteur au-dessus du niveau local, en écarts-types du bruit local")
        self.max_fwhm = _spin(2, 0.05, 10.0, 0.1)
        self.max_fwhm.setSuffix(" °")
        self.noise_window = _spin(2, 0.2, 20.0, 0.25)
        self.noise_window.setSuffix(" °")
        self.noise_window.setToolTip("Fenêtre du bruit local et du niveau local (médiane glissante)")
        self.broad_fwhm = _spin(2, 0.0, 10.0, 0.1)
        self.broad_fwhm.setSuffix(" °")
        self.broad_fwhm.setSpecialValueText("jamais")
        self.broad_fwhm.setToolTip("Les pics plus larges n'expliquent rien et ne coûtent rien dans la recherche des\n"
                                   "phases (bosse d'un liquide à côté des raies d'un cristal).\n"
                                   "« jamais » : tous comptent, pour identifier un liquide par ses pics larges.")
        self.shoulders = QCheckBox("épaulements (pics sans maximum propre)")
        self.spectrum_filter = QCheckBox("sur le signal filtré du spectre (sinon Savitzky-Golay de 7 points)")
        for key, box in (("min_snr", self.min_snr), ("max_fwhm", self.max_fwhm), ("noise_window", self.noise_window)):
            box.valueChanged.connect(lambda value, key=key: self._option(key, float(value)))
        self.broad_fwhm.valueChanged.connect(lambda value: self._option("broad_fwhm", float(value) or None))
        for key, box in (("shoulders", self.shoulders), ("spectrum_filter", self.spectrum_filter)):
            box.toggled.connect(lambda checked, key=key: self._option(key, bool(checked)))
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.table = RowsTable()

        form = QFormLayout()
        form.addRow("Seuil (σ du bruit)", self.min_snr)
        form.addRow("Largeur max", self.max_fwhm)
        form.addRow("Fenêtre du bruit", self.noise_window)
        form.addRow("Pics larges ignorés au-delà de", self.broad_fwhm)
        layout = QVBoxLayout(self)
        layout.addWidget(self.visible)
        layout.addLayout(form)
        layout.addWidget(self.shoulders)
        layout.addWidget(self.spectrum_filter)
        layout.addWidget(_note("Réglages de l'analyse automatique (Ctrl+R), gardés dans le run. Sur le spectre : "
                               "vert, pic utilisé ; orange, trop large, ignoré ; gris, zone exclue ; "
                               "triangle vide, épaulement ; pointillés, seuil. Liquide ou amorphe : relever la "
                               "largeur max, « jamais » pour les pics larges, et régler la fenêtre du fond "
                               "(onglet Traitement)."))
        layout.addWidget(self.summary)
        layout.addWidget(self.table, 1)

    def _option(self, key: str, value) -> None:
        if not self._showing and self.on_options is not None:
            self.on_options(**{key: value})

    def show_options(self, settings) -> None:
        self._showing = True
        try:
            self.min_snr.setValue(settings.min_snr)
            self.max_fwhm.setValue(settings.max_fwhm)
            self.noise_window.setValue(settings.noise_window)
            self.broad_fwhm.setValue(settings.broad_fwhm or 0.0)
            self.shoulders.setChecked(settings.shoulders)
            self.spectrum_filter.setChecked(settings.spectrum_filter)
        finally:
            self._showing = False

    def show_detection(self, columns, rows, summary: str) -> None:
        self.table.show_rows(columns, rows)
        self.summary.setText(summary)

    def clear(self) -> None:
        self.table.show_rows([], [])
        self.summary.setText("")
