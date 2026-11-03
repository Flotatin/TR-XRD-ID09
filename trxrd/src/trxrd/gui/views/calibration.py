"""Calibration dialog, in the spirit of Dioptas: calibrant image, expected rings, clicked rings, refinement."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from trxrd.gui.presenters.calibration import CalibrationPresenter
from trxrd.gui.views.plots import PALETTE, _levels, _scene_click

IMAGE_FILTER = "Images (*.h5 *.hdf5 *.tif *.tiff *.edf *.cbf *.img);;Tous les fichiers (*)"
FIXABLE = (("dist", "distance"), ("poni1", "poni 1"), ("poni2", "poni 2"), ("rot1", "rot1"), ("rot2", "rot2"),
           ("rot3", "rot3"), ("wavelength", "λ"))


def _spin(low: float, high: float, decimals: int, suffix: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    return box


class CalibrationDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None, settings: Optional[QSettings] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration du détecteur (LaB6 / CeO2)")
        self.resize(1400, 900)
        self.settings = settings or QSettings()
        self.on_click = None  # set by the presenter
        self.presenter: Optional[CalibrationPresenter] = None

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setColorMap(pg.colormap.get("viridis"))
        self.rings = pg.ImageItem(axisOrder="row-major")
        self.points = pg.ScatterPlotItem(size=5, pen=None)
        for item in (self.image, self.rings, self.points):
            self.plot.addItem(item)
        _scene_click(self.plot, lambda x, y: self.on_click and self.on_click(y, x))  # (row, col) = (y, x)

        self.calibrant = QComboBox()
        self.calibrant.addItems(["LaB6", "CeO2"])
        self.energy = _spin(1.0, 200.0, 4, " keV")
        self.distance = _spin(10.0, 2000.0, 3, " mm")
        self.center_x, self.center_y = _spin(-5000, 10000, 1, " px"), _spin(-5000, 10000, 1, " px")
        self.ring = QComboBox()
        self.fixed = {key: QCheckBox(label) for key, label in FIXABLE}
        for key in ("wavelength", "rot3"):
            self.fixed[key].setChecked(True)
        self.result = QLabel("")
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.message = QLabel("Charger l'image du calibrant (pile HDF5 : moyenne de toutes les images).")
        self.message.setWordWrap(True)

        def button(text, slot):
            widget = QPushButton(text)
            widget.clicked.connect(slot)
            return widget

        files = QGroupBox("Fichiers")
        files_layout = QVBoxLayout(files)
        for widget in (button("Charger l'image du calibrant…", self._load_image),
                       button("Charger un .poni de départ…", self._load_poni),
                       button("Charger le masque…", self._load_mask)):
            files_layout.addWidget(widget)
        start = QGroupBox("Géométrie de départ")
        start_form = QFormLayout(start)
        start_form.addRow("Calibrant", self.calibrant)
        start_form.addRow("Énergie", self.energy)
        start_form.addRow("Distance", self.distance)
        start_form.addRow("Centre x", self.center_x)
        start_form.addRow("Centre y", self.center_y)
        start_form.addRow("", button("Appliquer cette estimation", self._guess))
        picking = QGroupBox("Pointer les anneaux")
        picking_layout = QFormLayout(picking)
        picking_layout.addRow("Anneau n°", self.ring)
        picking_layout.addRow(QLabel("Cliquer sur cet anneau dans l'image : ses points sont suivis le long de l'anneau."))
        picking_layout.addRow(button("Effacer les points", lambda: self.presenter.clear_points()))
        fit = QGroupBox("Affinement")
        fit_layout = QVBoxLayout(fit)
        fixed_row = QHBoxLayout()
        fixed_row.addWidget(QLabel("Fixés :"))
        for box in self.fixed.values():
            fixed_row.addWidget(box)
        fit_layout.addLayout(fixed_row)
        fit_layout.addWidget(button("Affiner sur les points pointés", self._refine))
        fit_layout.addWidget(button("Calibrer (tous les anneaux, plusieurs passes)", lambda: self.presenter.calibrate()))
        fit_layout.addWidget(self.result)
        end = QHBoxLayout()
        end.addWidget(button("Enregistrer le .poni…", lambda: self.presenter.save()))
        self.use = button("Utiliser cette calibration", self._use)
        end.addWidget(self.use)
        end.addWidget(button("Fermer", self.reject))

        side = QWidget()
        side_layout = QVBoxLayout(side)
        for widget in (files, start, picking, fit):
            side_layout.addWidget(widget)
        side_layout.addStretch(1)
        side_layout.addWidget(self.message)
        side_layout.addLayout(end)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.plot)
        split.addWidget(side)
        split.setSizes([950, 450])
        layout = QVBoxLayout(self)
        layout.addWidget(split)

        self.calibrant.currentTextChanged.connect(lambda name: self.presenter and self.presenter.set_calibrant(name))
        self.ring.currentIndexChanged.connect(lambda k: self.presenter and k >= 0 and self.presenter.set_ring(k))
        self.energy.editingFinished.connect(lambda: self.presenter and self.presenter.set_energy(self.energy.value()))
        for box in self.fixed.values():
            box.toggled.connect(self._fixed_changed)

    # ------------------------------------------------------------------ user gestures

    def _folder(self) -> str:
        return str(self.settings.value("calibration/folder", ""))

    def _pick(self, title: str, filters: str) -> Optional[str]:
        path, _ = QFileDialog.getOpenFileName(self, title, self._folder(), filters)
        if path:
            self.settings.setValue("calibration/folder", str(path).rsplit("/", 1)[0])
        return path or None

    def _load_image(self) -> None:
        path = self._pick("Image du calibrant", IMAGE_FILTER)
        if path:
            self.presenter.load_image(path)

    def _load_poni(self) -> None:
        path = self._pick("Calibration de départ", "Calibration (*.poni)")
        if path:
            self.presenter.load_poni(path)

    def _load_mask(self) -> None:
        path = self._pick("Masque", "Masque (*.mask *.tif *.edf *.npy);;Tous (*)")
        if path:
            self.presenter.load_mask(path)

    def _guess(self) -> None:
        self.presenter.set_guess(self.distance.value(), self.center_x.value(), self.center_y.value(), self.energy.value())

    def _fixed_changed(self) -> None:
        if self.presenter is not None:
            self.presenter.set_fixed([key for key, box in self.fixed.items() if box.isChecked()])

    def _refine(self) -> None:
        self._fixed_changed()
        self.presenter.refine()

    def _use(self) -> None:
        if self.presenter.saved_poni is None and self.presenter.save() is None:
            return
        self.accept()

    # ------------------------------------------------------------------ view interface of CalibrationPresenter

    def show_image(self, image: np.ndarray) -> None:
        shown, levels = _levels(image, True)
        self.image.setImage(np.nan_to_num(shown, nan=levels[0]), levels=levels)
        self.plot.enableAutoRange()

    def show_rings(self, overlay: np.ndarray) -> None:
        rgba = np.zeros(overlay.shape + (4,), dtype=np.uint8)
        rgba[overlay > 0] = (255, 60, 60, 170)
        self.rings.setImage(rgba)

    def show_points(self, points: np.ndarray, rings: np.ndarray) -> None:
        brushes = [pg.mkBrush(PALETTE[int(k) % len(PALETTE)]) for k in rings]
        self.points.setData(x=points[:, 1] if len(points) else [], y=points[:, 0] if len(points) else [], brush=brushes)

    def show_ring_choices(self, labels) -> None:
        current = max(self.ring.currentIndex(), 0)
        self.ring.blockSignals(True)
        self.ring.clear()
        self.ring.addItems(list(labels))
        self.ring.setCurrentIndex(min(current, len(labels) - 1))
        self.ring.blockSignals(False)

    def show_geometry(self, text: str) -> None:
        self.result.setText(text)

    def show_start(self, distance_mm: float, center_col: float, center_row: float, energy_kev: float) -> None:
        for box, value in ((self.distance, distance_mm), (self.center_x, center_col), (self.center_y, center_row),
                           (self.energy, energy_kev)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)

    def show_message(self, text: str) -> None:
        self._last_message = text
        self.message.setText(text)

    def set_busy(self, text: Optional[str]) -> None:
        if text:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.message.setText(text)  # temporary: the last message comes back afterwards
        else:
            while QGuiApplication.overrideCursor() is not None:
                QGuiApplication.restoreOverrideCursor()
            self.message.setText(getattr(self, "_last_message", ""))

    def ask_save_path(self, suggested: str) -> Optional[str]:
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer la calibration", suggested, "Calibration (*.poni)")
        if path and not path.endswith(".poni"):
            path += ".poni"
        return path or None


def calibrate(parent: Optional[QWidget] = None, poni_file: Optional[str] = None, mask_file: Optional[str] = None,
              runner=None) -> Optional[Tuple[str, str]]:
    """Open the dialog; (saved .poni, mask file) if the user chose to use the calibration."""
    from trxrd.gui.tasks import ThreadRunner

    dialog = CalibrationDialog(parent)
    dialog.presenter = CalibrationPresenter(dialog, runner or ThreadRunner(), poni_file or None, mask_file or None)
    if dialog.exec() == QDialog.DialogCode.Accepted and dialog.presenter.saved_poni is not None:
        return str(dialog.presenter.saved_poni), dialog.presenter.mask_file
    return None
