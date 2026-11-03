"""Plots of the viewer (pyqtgraph): current spectrum, P(t) with the oscilloscope, intensity map."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

pg.setConfigOptions(antialias=True, background="w", foreground="k")

# one colour per phase, in the order of display.phase_order (colour-blind friendly)
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f")


def phase_color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _scene_click(plot: pg.PlotWidget, callback: Callable[[float, float], None]) -> None:
    """callback(x, y) in data coordinates on a left click inside the plot (a drag pans, no click)."""
    def clicked(event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or event.double():
            return
        box = plot.getPlotItem().vb
        if box.sceneBoundingRect().contains(event.scenePos()):
            point = box.mapSceneToView(event.scenePos())
            callback(point.x(), point.y())
    plot.scene().sigMouseClicked.connect(clicked)


class SpectrumView(QWidget):
    """Data, background, fit, and one row of ticks per phase under the pattern (height ~ intensity)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        self.plot.addLegend(offset=(-10, 10))
        self.plot.setLabel("left", "Intensité")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.data = self.plot.plot(pen=pg.mkPen("#404040", width=1), name="données")
        self.baseline = self.plot.plot(pen=pg.mkPen("#1f77b4", width=1, style=Qt.PenStyle.DashLine), name="fond")
        self.fit = self.plot.plot(pen=pg.mkPen("#d62728", width=1.5), name="ajustement")
        self._ticks: List[pg.GraphicsObject] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

    def clear(self) -> None:
        for curve in (self.data, self.baseline, self.fit):
            curve.setData([], [])
        self._clear_ticks()
        self.plot.setTitle("")

    def _clear_ticks(self) -> None:
        for item in self._ticks:
            self.plot.removeItem(item)
        self._ticks = []

    def show_spectrum(self, curves, reset: bool) -> None:
        self.plot.setTitle(curves.title)
        self.plot.setLabel("bottom", curves.x_label)
        self.data.setData(curves.x, curves.intensity)
        for item, y in ((self.baseline, curves.baseline), (self.fit, curves.fit)):
            item.setData(curves.x, y) if y is not None else item.setData([], [])
        self._clear_ticks()
        finite = curves.intensity[np.isfinite(curves.intensity)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        step = 0.06 * max(high - low, 1e-9)
        for row, lines in enumerate(curves.lines):
            if not len(lines.x):
                continue
            color = phase_color(lines.color)
            bottom = low - (row + 1) * step
            heights = step * 0.8 * (0.3 + 0.7 * np.clip(lines.intensity, 0, 100) / 100)
            x = np.repeat(lines.x, 2)
            y = np.column_stack([np.full(len(lines.x), bottom), bottom + heights]).ravel()
            ticks = pg.PlotDataItem(x, y, connect="pairs", pen=pg.mkPen(color, width=2),
                                    name=f"{lines.phase}  {lines.pressure:.2f} GPa")
            tips = pg.ScatterPlotItem(
                lines.x, bottom + heights, size=6, brush=pg.mkBrush(color), pen=None, hoverable=True,
                data=[f"{lines.phase} ({label})  I = {i:.0f} %" for label, i in zip(lines.labels, lines.intensity)],
                tip=lambda x, y, data: f"{data}\nx = {x:.4f}",
            )
            for item in (ticks, tips):
                self.plot.addItem(item)
                self._ticks.append(item)
        if reset:
            self.plot.enableAutoRange()


class PressureView(QWidget):
    """P(t) of every phase with error bars; oscilloscope channel on the right axis (DDAC view of the v1)."""

    NO_TRACE = "aucun"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_pick: Optional[Callable[[float], None]] = None
        self.channel = QComboBox()
        self.channel.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.channel.currentTextChanged.connect(self._show_trace)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Oscilloscope :"))
        bar.addWidget(self.channel)
        bar.addStretch(1)
        self.plot = pg.PlotWidget()
        self.plot.addLegend(offset=(10, 10))
        self.plot.setLabel("left", "P (GPa)")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#808080", style=Qt.PenStyle.DotLine))
        self.plot.addItem(self.marker, ignoreBounds=True)
        # second view box sharing x, for the oscilloscope (right axis)
        item = self.plot.getPlotItem()
        self.scope = pg.ViewBox()
        item.showAxis("right")
        item.scene().addItem(self.scope)
        item.getAxis("right").linkToView(self.scope)
        self.scope.setXLink(item)
        item.vb.sigResized.connect(lambda: self.scope.setGeometry(item.vb.sceneBoundingRect()))
        self.trace = pg.PlotDataItem(pen=pg.mkPen("#a0a0a0", width=1))
        self.trace.setDownsampling(auto=True, method="peak")
        self.trace.setClipToView(True)
        self.scope.addItem(self.trace)
        self._items: List[pg.GraphicsObject] = []
        self._traces: Dict[str, object] = {}
        _scene_click(self.plot, lambda x, y: self.on_pick and self.on_pick(x))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.plot)

    def clear(self) -> None:
        for item in self._items:
            self.plot.removeItem(item)
        self._items = []
        self._traces = {}
        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.blockSignals(False)
        self.trace.setData([], [])

    def show_pressure(self, data) -> None:
        previous = self.channel.currentText()
        self.clear()
        self.plot.setLabel("bottom", data.x_label)
        for series in data.series:
            color = phase_color(series.color)
            bars = pg.ErrorBarItem(x=series.x, y=series.pressure, height=2 * series.error, beam=0,
                                   pen=pg.mkPen(color, width=1))
            points = pg.ScatterPlotItem(series.x, series.pressure, size=7, brush=pg.mkBrush(color),
                                        pen=pg.mkPen("w", width=0.5), name=series.phase)
            self.plot.addItem(bars, ignoreBounds=True)  # a few huge error bars must not set the scale
            self.plot.addItem(points)
            self._items += [bars, points]
        self._traces = {t.name: t for t in data.traces}
        names = [t.name for t in data.traces if not t.trigger] + [t.name for t in data.traces if t.trigger]
        self.channel.blockSignals(True)
        self.channel.addItems([self.NO_TRACE] + names)
        self.channel.blockSignals(False)
        choice = previous if previous in self._traces else (names[0] if names else self.NO_TRACE)
        self.channel.setCurrentText(choice)
        self._show_trace(choice)
        self.plot.enableAutoRange()

    def _show_trace(self, name: str) -> None:
        trace = self._traces.get(name)
        if trace is None:
            self.trace.setData([], [])
            self.plot.getPlotItem().getAxis("right").setLabel("")
            return
        self.trace.setData(trace.x, trace.y)
        self.plot.getPlotItem().getAxis("right").setLabel(trace.name)
        self.scope.enableAutoRange(axis=pg.ViewBox.YAxis)

    def set_marker(self, x: float) -> None:
        self.marker.setPos(x)


class MapView(QWidget):
    """Intensity (background subtracted) versus 2θ / Q / d and spectrum index; click to select."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_pick: Optional[Callable[[int], None]] = None
        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "spectre")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setColorMap(pg.colormap.get("viridis"))
        self.plot.addItem(self.image)
        self.marker = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.marker, ignoreBounds=True)
        self.log = QCheckBox("échelle log")  # a gasket peak 100 times stronger hides the sample otherwise
        self.log.setChecked(True)
        self.log.toggled.connect(lambda checked: self._data is not None and self.show_map(self._data))
        self._count = 0
        self._data = None
        _scene_click(self.plot, self._clicked)
        bar = QHBoxLayout()
        bar.addWidget(self.log)
        bar.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.plot)

    def _clicked(self, x: float, y: float) -> None:
        index = int(round(y))
        if self.on_pick is not None and 0 <= index < self._count:
            self.on_pick(index)

    def clear(self) -> None:
        self.image.clear()
        self._count = 0
        self._data = None

    def show_map(self, data) -> None:
        self._data = data
        image = np.nan_to_num(data.image)
        if self.log.isChecked():
            positive = image[image > 0]
            floor = max(float(np.percentile(positive, 5)), 1e-6) if positive.size else 1.0
            image = np.log10(np.clip(image, floor, None))
        levels = np.percentile(image, [1, 99.8]) if image.size else (0, 1)
        if levels[1] <= levels[0]:
            levels = (levels[0], levels[0] + 1)
        self.image.setImage(image, levels=levels)
        step = (data.x[-1] - data.x[0]) / max(len(data.x) - 1, 1)
        self.image.setRect(QRectF(data.x[0] - step / 2, -0.5, data.x[-1] - data.x[0] + step, image.shape[0]))
        self._count = image.shape[0]
        self.plot.setLabel("bottom", data.x_label)
        self.plot.enableAutoRange()

    def set_marker(self, index: int) -> None:
        self.marker.setPos(index)
