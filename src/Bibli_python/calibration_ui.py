from __future__ import annotations

import os
from typing import Iterable, Optional

import fabio
import numpy as np
import pyFAI
from PIL import Image

from .drx import ENERGY_CONSTANT_KEV_M, Integrate_DRX

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )
    import pyqtgraph as pg
except Exception:  # pragma: no cover - optional UI
    Qt = None
    QDialog = None
    QLabel = None
    QPushButton = None
    QFileDialog = None
    QVBoxLayout = None
    QHBoxLayout = None
    QDoubleSpinBox = None
    pg = None


if QDialog is not None:
    class CalibDialog(QDialog):
        def __init__(
            self,
            file_img: str,
            mask: Optional[np.ndarray] = None,
            ai: Optional[pyFAI.AzimuthalIntegrator] = None,
            theta_range: Optional[Iterable[float]] = None,
            folder_start: Optional[str] = None,
            energy: Optional[float] = None,
        ):
            if pg is None:
                raise ImportError("PyQtGraph indisponible: CalibDialog ne peut pas être utilisé.")
            super().__init__()
            self.setWindowTitle("Choix de calibration DRX")
            self.setMinimumSize(1000, 800)

            self.folder_start = folder_start
            self.file_mask = None
            self.file_poni = None

            self.mask = mask
            self.ai = ai
            self.theta_range = list(theta_range) if theta_range is not None else [0.0, 40.0]
            self.energy = float(energy) if energy is not None else 0.0

            self.img_data = fabio.open(file_img).data if os.path.isfile(file_img) else np.asarray(file_img)

            self.label = QLabel("Sélectionnez un .mask et un .poni")
            self.label.setAlignment(Qt.AlignCenter)

            self.btn_mask = QPushButton("Charger fichier .mask")
            self.btn_mask.clicked.connect(self.load_mask)

            self.btn_poni = QPushButton("Charger fichier .poni")
            self.btn_poni.clicked.connect(self.load_poni)

            self.btn_update = QPushButton("Update image")
            self.btn_update.clicked.connect(self.update_image)

            self.btn_ok = QPushButton("Valider")
            self.btn_ok.clicked.connect(self.accept)

            self.energy_input = QDoubleSpinBox()
            self.energy_input.setDecimals(2)
            self.energy_input.setRange(0.0, 200000.0)
            self.energy_input.setSingleStep(10.0)
            self.energy_input.setValue(self.energy)

            self.view = pg.GraphicsLayoutWidget()
            self.img_view = self.view.addViewBox()
            self.img_item = pg.ImageItem()
            self.img_view.addItem(self.img_item)
            self.mask_item = None

            self.theta_range_label = QLabel()
            self.plot_1D = pg.PlotWidget(title="Intégration radiale (2θ)")
            self.plot_1D.setLabel("bottom", "2θ (°)")
            self.plot_1D.setLabel("left", "Intensité (a.u.)")
            self.curve_1D = self.plot_1D.plot([], [])

            self.region = pg.LinearRegionItem(values=self.theta_range, orientation=pg.LinearRegionItem.Vertical)
            self.region.setBrush(pg.mkBrush(0, 255, 0, 40))
            self.region.setZValue(10)
            self.region.sigRegionChanged.connect(self.update_theta_range)
            self.plot_1D.addItem(self.region)
            self.update_theta_range()

            layout = QVBoxLayout()
            layout.addWidget(self.label)
            layout.addWidget(self.btn_mask)
            layout.addWidget(self.btn_poni)
            layout.addWidget(self.view)
            layout.addWidget(self.plot_1D)

            e_layout = QHBoxLayout()
            e_layout.addWidget(QLabel("E (keV):"))
            e_layout.addWidget(self.energy_input)
            layout.addLayout(e_layout)

            layout.addWidget(self.theta_range_label)
            layout.addWidget(self.btn_update)
            layout.addWidget(self.btn_ok)
            self.setLayout(layout)

            self.update_image()
            self._update_energy_from_ai()

        def accept(self):  # type: ignore[override]
            self.energy = float(self.energy_input.value())
            super().accept()

        def _update_energy_from_ai(self):
            if self.ai is None:
                return
            wl = getattr(self.ai, "wavelength", None)
            if not wl:
                return
            try:
                self.energy = float(ENERGY_CONSTANT_KEV_M / wl)
            except Exception:
                return
            blocked = self.energy_input.blockSignals(True)
            self.energy_input.setValue(self.energy)
            self.energy_input.blockSignals(blocked)

        def update_theta_range(self):
            self.theta_range = list(self.region.getRegion())
            self.theta_range_label.setText(
                f"Theta range : [{self.theta_range[0]:.2f} - {self.theta_range[1]:.2f}] °"
            )

        def load_mask(self):
            fname, _ = QFileDialog.getOpenFileName(self, "Fichier .mask", directory=self.folder_start)
            if not fname:
                return
            self.file_mask = fname
            try:
                mask = np.array(Image.open(fname))
                self.mask = (mask > 0).astype(np.uint8)
                self.label.setText("Mask chargé.")
                self.update_image()
            except Exception as e:
                self.label.setText(f"Erreur lecture mask : {e}")

        def load_poni(self):
            fname, _ = QFileDialog.getOpenFileName(self, "Fichier .poni", directory=self.folder_start)
            if not fname:
                return
            self.file_poni = fname
            try:
                self.ai = pyFAI.load(fname)
                self.label.setText("PONI chargé.")
                self._update_energy_from_ai()
                self.update_image()
            except Exception as e:
                self.label.setText(f"Erreur lecture poni : {e}")

        def update_image(self):
            if self.img_data is None:
                return

            img = np.flipud(self.img_data)
            self.img_item.setImage(img, autoLevels=True)

            if self.mask_item is not None:
                self.img_view.removeItem(self.mask_item)
                self.mask_item = None

            if self.mask is not None:
                m = np.flipud(self.mask)
                rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
                rgba[..., 0] = 255
                rgba[..., 3] = 100
                rgba[m == 0, 3] = 0

                self.mask_item = pg.ImageItem(rgba)
                self.mask_item.setZValue(1)
                self.img_view.addItem(self.mask_item)

            if self.mask is not None and self.ai is not None:
                try:
                    tth, intens = Integrate_DRX(self.img_data, self.mask, self.ai, theta_range=self.theta_range)
                    self.curve_1D.setData(tth, intens)
                except Exception as e:
                    self.label.setText(f"Erreur intégration : {e}")
else:
    class CalibDialog:  # pragma: no cover - fallback without Qt
        def __init__(self, *args, **kwargs):
            raise ImportError("Qt/PyQtGraph indisponible: CalibDialog ne peut pas être utilisé.")
