"""Dialog "automatic analysis": candidate phases, spectra, pressure range, excluded zones, advanced settings."""

from __future__ import annotations

import dataclasses
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QRadioButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from trxrd.gui.presenters.analysis import AnalysisChoices, AnalysisRequest, parse_zones, phase_label, zones_text
from trxrd.physics.eos import T_REF


def _spin(low: float, high: float, value: float, decimals: int = 2, suffix: str = "", step: float = 1.0):
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setSuffix(suffix)
    box.setValue(value)
    return box


def _row(*widgets) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(QLabel(widget) if isinstance(widget, str) else widget)
    layout.addStretch(1)
    return row


class AnalysisDialog(QDialog):
    def __init__(self, choices: AnalysisChoices, library_phases, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analyse automatique")
        self.resize(720, 640)
        self.choices, self.library_phases = choices, library_phases
        self.answer: Optional[AnalysisRequest] = None
        settings = choices.settings
        self._library: List[str] = list(settings.library)

        # candidate phases
        self.phases = QListWidget()
        for name, label, checked in choices.phases:
            self._add_phase(name, label, checked)
        add = QPushButton("Ajouter des fiches…")
        add.setToolTip("Dossier de fiches .jcpds : leurs phases deviennent candidates (et sont ajoutées au run)")
        add.clicked.connect(self._add_folder)
        every, none = QPushButton("Tout cocher"), QPushButton("Tout décocher")
        every.clicked.connect(lambda: self._check_all(True))
        none.clicked.connect(lambda: self._check_all(False))
        phases = QGroupBox("Phases candidates")
        phases_layout = QVBoxLayout(phases)
        phases_layout.addWidget(self.phases)
        phases_layout.addWidget(_row(add, every, none))

        # spectra
        last = max(choices.count - 1, 0)
        self.every = QRadioButton(f"tous ({choices.count})")
        self.current = QRadioButton(f"le spectre courant (n° {choices.current})")
        self.between = QRadioButton("du n°")
        self.first, self.last = QSpinBox(), QSpinBox()
        self.selection = QRadioButton(f"la sélection ({len(choices.selection)})")
        self.selection.setVisible(bool(choices.selection))
        for box, value in ((self.first, 0), (self.last, last)):
            box.setRange(0, last)
            box.setValue(value)
            box.valueChanged.connect(lambda *_: self.between.setChecked(True))
        group = QButtonGroup(self)
        for button in (self.every, self.current, self.selection, self.between):
            group.addButton(button)
        (self.selection if choices.selection else self.every).setChecked(True)  # as the batch of the v1
        spectra = QGroupBox("Spectres")
        spectra_layout = QVBoxLayout(spectra)
        spectra_layout.addWidget(_row(self.every, self.current, self.selection))
        spectra_layout.addWidget(_row(self.between, self.first, "au n°", self.last))
        if choices.analysed:
            note = QLabel(f"{choices.analysed} spectre(s) ont déjà des phases : l'analyse les remplace.")
            note.setStyleSheet("color: #8a5a00")
            spectra_layout.addWidget(note)

        # pressure, temperature, excluded zones
        self.pmin = _spin(-5.0, 1000.0, settings.pressure_range[0], suffix=" GPa")
        self.pmax = _spin(-5.0, 1000.0, settings.pressure_range[1], suffix=" GPa")
        self.temperature = _spin(1.0, 5000.0, settings.temperature or T_REF, decimals=1, suffix=" K", step=10.0)
        self.zones = QLineEdit(zones_text(settings.exclude))
        self.zones.setPlaceholderText("par exemple 18.48:18.68; 21.1:21.3")
        self.zones.setToolTip("Pics du joint ou du diamant : ajustés comme pics libres, attribués à aucune phase.\n"
                              "Après l'analyse, les pics qui ne bougent pas sont proposés.")
        main = QGroupBox("Recherche")
        form = QFormLayout(main)
        form.addRow("Pression", _row("de", self.pmin, "à", self.pmax))
        form.addRow("Température", _row(self.temperature))
        form.addRow("Zones exclues (2θ, °)", self.zones)

        # advanced
        low, high = choices.two_theta
        window = settings.window or (low, high)
        self.limit = QCheckBox("limiter l'analyse de")
        self.limit.setChecked(settings.window is not None)
        self.tth_min = _spin(0.0, 180.0, window[0], suffix=" °", step=0.5)
        self.tth_max = _spin(0.0, 180.0, window[1], suffix=" °", step=0.5)
        self.tolerance = _spin(0.005, 1.0, settings.tolerance, decimals=3, suffix=" °", step=0.01)
        self.max_phases = QSpinBox()
        self.max_phases.setRange(1, 8)
        self.max_phases.setValue(settings.max_phases)
        self.min_snr = _spin(1.0, 100.0, settings.min_snr, decimals=1, step=0.5)
        self.max_spread = _spin(0.0, 100.0, settings.max_spread, suffix=" GPa", step=0.5)
        self.max_step = _spin(0.0, 100.0, settings.max_step or 0.0, suffix=" GPa", step=0.5)
        self.max_step.setSpecialValueText("sans limite")
        self.track = QCheckBox("suivi des pics : tant qu'aucun pic n'apparaît ni ne disparaît, garder les phases")
        self.track.setChecked(settings.track)
        self.show_advanced = QCheckBox("Réglages avancés")
        self.advanced = QWidget()
        self.advanced.setVisible(False)
        self.show_advanced.toggled.connect(self.advanced.setVisible)
        grid = QGridLayout(self.advanced)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(2, 1)  # the fields keep their natural width
        rows = (("Plage 2θ", _row(self.limit, self.tth_min, "à", self.tth_max)),
                ("Tolérance en 2θ", self.tolerance), ("Phases au plus", self.max_phases),
                ("Seuil de détection (σ du bruit)", self.min_snr),
                ("Écart de P entre phases (souple)", self.max_spread),
                ("Saut de P entre spectres", self.max_step))
        for k, (label, widget) in enumerate(rows):
            grid.addWidget(QLabel(label), k, 0)
            grid.addWidget(widget, k, 1)
        grid.addWidget(self.track, len(rows), 0, 1, 3)

        self.message = QLabel("")
        self.message.setStyleSheet("color: #b00020")
        self.message.setWordWrap(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Lancer")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(phases, 1)
        layout.addWidget(spectra)
        layout.addWidget(main)
        layout.addWidget(self.show_advanced)
        layout.addWidget(self.advanced)
        layout.addWidget(self.message)
        layout.addWidget(self.buttons)

    # ------------------------------------------------------------------ phases

    def _add_phase(self, name: str, label: str, checked: bool) -> None:
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.phases.addItem(item)

    def phase_names(self, checked_only: bool = True) -> List[str]:
        items = [self.phases.item(i) for i in range(self.phases.count())]
        return [item.data(Qt.ItemDataRole.UserRole) for item in items
                if not checked_only or item.checkState() == Qt.CheckState.Checked]

    def _check_all(self, checked: bool) -> None:
        for i in range(self.phases.count()):
            self.phases.item(i).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def add_folder(self, folder: str) -> None:
        try:
            found = self.library_phases([folder])
        except (OSError, ValueError) as exc:
            self.message.setText(f"Lecture des fiches impossible : {exc}")
            return
        items = {self.phases.item(i).data(Qt.ItemDataRole.UserRole): self.phases.item(i)
                 for i in range(self.phases.count())}
        for name, p_range in found:
            if name not in items:
                self._add_phase(name, phase_label(name, p_range), True)
            elif p_range is not None:  # the card of the run is kept, but the search takes the P range of the library
                items[name].setText(phase_label(name, p_range))
        if folder not in self._library:
            self._library.append(folder)
        self.message.setText("" if found else f"Aucune fiche .jcpds dans {folder}")

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dossier de fiches .jcpds", "")
        if folder:
            self.add_folder(folder)

    # ------------------------------------------------------------------ result

    def indices(self) -> List[int]:
        if self.current.isChecked():
            return [self.choices.current]
        if self.selection.isChecked():
            return list(self.choices.selection)
        if self.between.isChecked():
            first, last = sorted((self.first.value(), self.last.value()))
            return list(range(first, last + 1))
        return list(range(self.choices.count))

    def request(self) -> AnalysisRequest:
        """The user's choices; ValueError when they cannot be used (no phase, unreadable zone)."""
        phases = self.phase_names()
        if not phases:
            raise ValueError("Cocher au moins une phase candidate.")
        every = len(phases) == self.phases.count()
        settings = dataclasses.replace(  # the settings the dialog does not show (detection tab) are kept
            self.choices.settings,
            phases=None if every else phases,
            library=list(self._library),
            pressure_range=tuple(sorted((self.pmin.value(), self.pmax.value()))),
            temperature=self.temperature.value(),
            tolerance=self.tolerance.value(),
            max_phases=self.max_phases.value(),
            min_snr=self.min_snr.value(),
            max_step=self.max_step.value() or None,
            max_spread=self.max_spread.value(),
            exclude=parse_zones(self.zones.text()),
            window=tuple(sorted((self.tth_min.value(), self.tth_max.value()))) if self.limit.isChecked() else None,
            track=self.track.isChecked(),
        )
        return AnalysisRequest(settings, self.indices())

    def accept(self) -> None:
        try:
            self.answer = self.request()
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        super().accept()
