"""Preview of a raw scan in the "new run" dialog: raw image or cake over the whole detector, its pattern,
and the 2θ range of the integration, which the user drags on the pattern (or on the cake)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSettings, Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QSplitter, QVBoxLayout, QWidget

from trxrd.gui.views.images import AXIS_WIDTH, ImagePanel, spacer
from trxrd.gui.views.plots import RIGHT_AXIS_WIDTH, MovableRegion

RANGE_BRUSH, RANGE_PEN = (40, 160, 70, 40), "#2a8a3e"  # green: what is integrated
SHADE = (255, 255, 255, 150)  # pixels outside the range, washed out


class ScanPreviewView(QWidget):
    def __init__(self, settings: Optional[QSettings] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_frame: Optional[Callable[[Optional[int]], None]] = None
        self.on_mode: Optional[Callable[[str], None]] = None
        self.on_range: Optional[Callable[[float, float], None]] = None
        self._usable: Optional[Tuple[float, float]] = None
        self._kind = ""  # "image" or "cake": the view range is reset when it changes
        self._message = ""

        self.which = QComboBox()
        self.which.addItem("maximum du scan", "max")
        self.which.addItem("une image", "one")
        self.which.setToolTip("Maximum pixel par pixel sur le scan : toutes les raies vues pendant la compression")
        self.frame = QSpinBox()
        self.frame.setPrefix("n° ")
        self.frame.setEnabled(False)
        self.mode = QComboBox()
        self.mode.addItem("image brute", "image")
        self.mode.addItem("cake (2θ × azimut)", "cake")
        self.whole = QPushButton("Tout le détecteur")
        self.whole.setToolTip("Plage 2θ = tous les pixels utilisables (hors masque)")
        self.whole.setEnabled(False)
        self.busy = QLabel("")
        self.busy.setStyleSheet("color: #8a5a00")
        self.which.currentIndexChanged.connect(self._frame_changed)
        self.frame.valueChanged.connect(self._frame_changed)
        self.mode.currentIndexChanged.connect(lambda i: self.on_mode and self.on_mode(self.mode.itemData(i)))
        self.whole.clicked.connect(self._whole)
        controls = QHBoxLayout()
        for widget in (QLabel("Image :"), self.which, self.frame, QLabel("  Vue :"), self.mode, self.whole):
            controls.addWidget(widget)
        controls.addStretch(1)
        controls.addWidget(self.busy)

        self.panel = ImagePanel("preview", scale="log", histogram=False, settings=settings)
        self.title = QLabel("")
        self.panel.bar.addWidget(self.title)
        image_item = self.panel.plot.getPlotItem()
        image_item.showAxis("right")  # empty, as wide as the coverage axis below: the 2θ axes of the cake line up
        image_item.getAxis("right").setWidth(RIGHT_AXIS_WIDTH)
        image_item.getAxis("right").setStyle(showValues=False)
        self.shade = pg.ImageItem(axisOrder="row-major")
        self.shade.setZValue(5)
        self.panel.plot.addItem(self.shade)
        self.cake_range = MovableRegion(self.panel.plot, self._dragged, RANGE_BRUSH, RANGE_PEN)

        self.pattern = pg.PlotWidget()
        item = self.pattern.getPlotItem()
        item.getAxis("left").setWidth(AXIS_WIDTH)
        item.setLabel("bottom", "2θ (°)")
        item.setLabel("left", "Intensité")
        item.showGrid(x=True, y=True, alpha=0.15)
        self.curve = self.pattern.plot(pen=pg.mkPen("#404040", width=1), connect="finite")
        self.seen = pg.ViewBox()  # azimuthal coverage of the rings, on the right axis
        item.showAxis("right")
        item.getAxis("right").setWidth(RIGHT_AXIS_WIDTH)
        item.setLabel("right", "azimut vu (°)")
        item.scene().addItem(self.seen)
        item.getAxis("right").linkToView(self.seen)
        self.seen.setXLink(item)
        item.vb.sigResized.connect(lambda: self.seen.setGeometry(item.vb.sceneBoundingRect()))
        self.coverage = pg.PlotDataItem(pen=pg.mkPen("#8a8a8a", width=1, style=Qt.PenStyle.DashLine))
        self.seen.addItem(self.coverage)
        self.range = MovableRegion(self.pattern, self._dragged, RANGE_BRUSH, RANGE_PEN)
        margin = spacer()  # stays aligned with the image when its histogram is shown
        margin.setVisible(self.panel.show_histogram.isChecked())
        self.panel.on_histogram = margin.setVisible
        bottom = QWidget()
        row = QHBoxLayout(bottom)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.pattern)
        row.addWidget(margin)
        self.text = QLabel("")
        self.text.setWordWrap(True)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.panel)
        split.addWidget(bottom)
        split.setSizes([430, 260])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(split, 1)
        layout.addWidget(self.text)

    # ------------------------------------------------------------------ gestures

    def _frame_changed(self, *_) -> None:
        one = self.which.currentData() == "one"
        self.frame.setEnabled(one)
        if self.on_frame is not None:
            self.on_frame(self.frame.value() if one else None)

    def _dragged(self, low: float, high: float) -> None:
        if self.on_range is not None:
            self.on_range(low, high)

    def _whole(self) -> None:
        if self._usable is not None and self.on_range is not None:
            self.on_range(*self._usable)

    # ------------------------------------------------------------------ presenter

    def set_busy(self, text: Optional[str]) -> None:
        self.busy.setText(text or "")

    def show_message(self, text: str) -> None:
        self.panel.clear()
        self.shade.clear()
        self.curve.clear()
        self.coverage.clear()
        self.range.show(None)
        self.cake_range.show(None)
        self.title.setText("")
        self._usable, self._kind, self._message = None, "", text
        self.whole.setEnabled(False)
        self.text.setText(text)

    def show_preview(self, data) -> None:
        self._usable, self._message = data.usable, data.message
        self.whole.setEnabled(data.usable is not None)
        self.frame.blockSignals(True)
        self.frame.setRange(0, max(data.frames - 1, 0))
        self.frame.blockSignals(False)
        self.title.setText(data.title)
        item = self.panel.plot.getPlotItem()
        kind = "cake" if data.cake is not None else "image"
        if kind == "cake":
            cake = data.cake
            step = (cake.x[-1] - cake.x[0]) / max(len(cake.x) - 1, 1)
            az_step = (cake.azimuth[-1] - cake.azimuth[0]) / max(len(cake.azimuth) - 1, 1)
            self.panel.set_image(cake.image, QRectF(cake.x[0] - step / 2, cake.azimuth[0] - az_step / 2,
                                                    cake.x[-1] - cake.x[0] + step,
                                                    cake.azimuth[-1] - cake.azimuth[0] + az_step))
            self.shade.hide()
            if kind != self._kind:
                item.setLabel("bottom", cake.x_label)
                item.setLabel("left", "azimut (°)")
                item.getViewBox().setAspectLocked(False)
                item.invertY(False)
                item.setXLink(self.pattern.getPlotItem())  # the rings line up with the peaks of the pattern
                item.enableAutoRange(axis=pg.ViewBox.YAxis)
        else:
            rows, columns = data.image.shape
            self.panel.set_image(data.image, QRectF(0, 0, columns, rows))
            self.cake_range.show(None)
            if kind != self._kind:
                item.setXLink(None)
                item.setLabel("bottom", "pixel")
                item.setLabel("left", "pixel")
                item.getViewBox().setAspectLocked(True)
                item.invertY(True)
                item.enableAutoRange()
        self._kind = kind
        if data.x is not None:
            self.curve.setData(data.x, data.intensity)
            self.coverage.setData(data.x, data.coverage)
            self.seen.setYRange(0, max(float(np.nanmax(data.coverage)) * 1.1, 1.0))
            self.pattern.getPlotItem().enableAutoRange()
        else:
            self.curve.clear()
            self.coverage.clear()
            self.range.show(None)
        self.text.setText(data.message)

    def show_range(self, low: float, high: float, outside: Optional[np.ndarray], text: str) -> None:
        if self._usable is not None:
            self.range.show((low, high))
        self.cake_range.show((low, high) if self._kind == "cake" else None)
        if outside is not None:
            shade = np.zeros(outside.shape + (4,), dtype=np.uint8)
            shade[outside] = SHADE
            self.shade.setImage(shade, autoLevels=False, levels=(0, 255))
            self.shade.setRect(QRectF(0, 0, outside.shape[1], outside.shape[0]))
            self.shade.show()
        else:
            self.shade.hide()
        self.text.setText(text or self._message)
