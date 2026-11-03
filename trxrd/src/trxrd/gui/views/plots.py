"""Plots of the viewer (pyqtgraph): current spectrum and its cake, quantities versus time, map of the patterns."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from trxrd.gui.views.images import AXIS_WIDTH, ImagePanel, spacer

pg.setConfigOptions(antialias=True, background="w", foreground="k")

# one colour per phase, in the order of display.phase_order (colour-blind friendly)
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f")
RIGHT_AXIS_WIDTH = 56  # oscilloscope axis of the evolution plot; the map below keeps the same margin


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


def _row(plot: QWidget, margin: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(0)
    row.addWidget(plot)
    row.addWidget(margin)
    return row


class _Interval:
    """Movable time interval drawn on a plot (mean dP/dt). Reports the user's drags, not the program's moves."""

    def __init__(self, plot: pg.PlotWidget, on_change: Callable[[float, float], None]) -> None:
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(255, 170, 0, 45), pen=pg.mkPen("#c07000", width=1.5))
        self.region.setZValue(-10)
        self.region.hide()
        plot.addItem(self.region, ignoreBounds=True)
        self.on_change = on_change
        self._setting = False
        self.region.sigRegionChanged.connect(self._moved)

    def _moved(self) -> None:
        if not self._setting and self.region.isVisible():
            start, stop = self.region.getRegion()
            self.on_change(float(start), float(stop))

    def show(self, region) -> None:
        self._setting = True
        try:
            if region is None:
                self.region.hide()
            else:
                self.region.setRegion(region)
                self.region.show()
        finally:
            self._setting = False


class SpectrumView(QWidget):
    """Data, background, fit, and one row of ticks per phase under the pattern (height ~ intensity)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        self.plot.getAxis("left").setWidth(AXIS_WIDTH)  # aligned with the cake below
        self.plot.addLegend(offset=(-10, 10))
        self.plot.setLabel("left", "Intensité")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.data = self.plot.plot(pen=pg.mkPen("#404040", width=1), name="données")
        self.baseline = self.plot.plot(pen=pg.mkPen("#1f77b4", width=1, style=Qt.PenStyle.DashLine), name="fond")
        self.fit = self.plot.plot(pen=pg.mkPen("#d62728", width=1.5), name="ajustement")
        self._ticks: List[pg.GraphicsObject] = []
        self.spacer = spacer()  # shown with the histogram of the cake, so that both x axes line up
        self.spacer.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(_row(self.plot, self.spacer))

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


class EvolutionView(QWidget):
    """P, V, a, c/a... of every phase versus time with error bars; an oscilloscope channel on the right axis.

    The dDAC view of the v1. The grey band marks the time range of the
    spectra; "Plage" switches between that range and the whole oscilloscope
    record (to see the piezo ramp around the spectra).
    """

    NO_TRACE = "aucun"
    RANGES = (("spectres", "spectra"), ("oscilloscope entier", "trace"))

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_pick: Optional[Callable[[float], None]] = None
        self.on_quantity: Optional[Callable[[str, bool], None]] = None
        self.on_interval: Optional[Callable[[float, float], None]] = None
        self.quantity = QComboBox()
        self.quantity.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.quantity.setToolTip("Grandeur tracée pour chaque phase")
        self.relative = QCheckBox("÷ fiche (X/X₀)")
        self.relative.setToolTip("Diviser par la valeur de la fiche à pression ambiante : V/V₀, a/a₀, (c/a)/(c/a)₀…\n"
                                 "Permet de comparer des phases différentes sur le même axe.")
        self.channel = QComboBox()
        self.channel.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.span = QComboBox()
        for label, key in self.RANGES:
            self.span.addItem(label, key)
        self.quantity.currentIndexChanged.connect(self._quantity_chosen)
        self.relative.toggled.connect(self._quantity_chosen)
        self.channel.currentTextChanged.connect(self._show_trace)
        self.span.currentIndexChanged.connect(lambda _: self._set_range())
        bar = QHBoxLayout()
        for widget in (QLabel("Grandeur :"), self.quantity, self.relative, QLabel("  Oscilloscope :"), self.channel,
                       QLabel("  Plage :"), self.span):
            bar.addWidget(widget)
        bar.addStretch(1)

        self.plot = pg.PlotWidget()
        item = self.plot.getPlotItem()
        item.getAxis("left").setWidth(AXIS_WIDTH)
        self.plot.addLegend(offset=(10, 10))
        self.plot.setLabel("left", "P (GPa)")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.band = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(128, 128, 128, 25), pen=pg.mkPen(None))
        self.band.setZValue(-20)
        self.band.hide()
        self.plot.addItem(self.band, ignoreBounds=True)
        self.marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#808080", style=Qt.PenStyle.DotLine))
        self.plot.addItem(self.marker, ignoreBounds=True)
        # second view box sharing x, for the oscilloscope (right axis)
        self.scope = pg.ViewBox()
        item.showAxis("right")
        item.getAxis("right").setWidth(RIGHT_AXIS_WIDTH)
        item.scene().addItem(self.scope)
        item.getAxis("right").linkToView(self.scope)
        self.scope.setXLink(item)
        item.vb.sigResized.connect(lambda: self.scope.setGeometry(item.vb.sceneBoundingRect()))
        self.trace = pg.PlotDataItem(pen=pg.mkPen("#4a6fa5", width=1))
        self.trace.setDownsampling(auto=True, method="peak")
        self.trace.setClipToView(True)
        self.scope.addItem(self.trace)
        self._items: List[pg.GraphicsObject] = []
        self._segments: List[pg.GraphicsObject] = []
        self._traces: Dict[str, object] = {}
        self._data = None
        self.interval = _Interval(self.plot, lambda start, stop: self.on_interval and self.on_interval(start, stop))
        _scene_click(self.plot, lambda x, y: self.on_pick and self.on_pick(x))
        self.spacer = spacer()  # same right margin as the map below when its histogram is shown
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addLayout(_row(self.plot, self.spacer))

    def _quantity_chosen(self, *_) -> None:
        key = self.quantity.currentData()
        if key and self.on_quantity is not None:
            self.on_quantity(key, self.relative.isChecked())

    def clear(self) -> None:
        self._remove_series()
        self.show_interval(None, [])
        self._traces = {}
        self._data = None
        self.band.hide()
        for box in (self.channel, self.quantity):
            box.blockSignals(True)
            box.clear()
            box.blockSignals(False)
        self.trace.setData([], [])

    def _remove_series(self) -> None:
        for item in self._items:
            self.plot.removeItem(item)
        self._items = []

    def show_evolution(self, data) -> None:
        previous = self.channel.currentText()
        self._remove_series()
        self._data = data
        self.plot.setLabel("bottom", data.x_label)
        self.plot.setLabel("left", data.y_label)
        for series in data.series:
            color = phase_color(series.color)
            bars = pg.ErrorBarItem(x=series.x, y=series.y, height=2 * series.error, beam=0, pen=pg.mkPen(color, width=1))
            points = pg.ScatterPlotItem(series.x, series.y, size=7, brush=pg.mkBrush(color),
                                        pen=pg.mkPen("w", width=0.5), name=series.phase)
            self.plot.addItem(bars, ignoreBounds=True)  # a few huge error bars must not set the scale
            self.plot.addItem(points)
            self._items += [bars, points]
        self.quantity.blockSignals(True)
        self.quantity.clear()
        for key, label in data.quantities:
            self.quantity.addItem(label, key)
        self.quantity.setCurrentIndex(max(self.quantity.findData(data.quantity), 0))
        self.quantity.blockSignals(False)
        self.relative.setEnabled(data.can_be_relative)  # the box keeps its state for the next quantity

        self._traces = {t.name: t for t in data.traces}
        names = [t.name for t in data.traces if not t.trigger] + [t.name for t in data.traces if t.trigger]
        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItems([self.NO_TRACE] + names)
        self.channel.blockSignals(False)
        choice = previous if previous in self._traces else (names[0] if names else self.NO_TRACE)
        self.channel.setCurrentText(choice)
        self._show_trace(choice)
        low, high = data.span
        self.band.setRegion((low, high))
        self.band.setVisible(len(data.x) > 1)
        self._set_range()

    def show_interval(self, region, segments) -> None:
        """The interval of the mean dP/dt (None: hidden) and the straight line of each curve over it."""
        self.interval.show(region)
        for item in self._segments:
            self.plot.removeItem(item)
        self._segments = []
        for start, stop, p_start, p_stop, color in segments:
            line = pg.PlotDataItem([start, stop], [p_start, p_stop],
                                   pen=pg.mkPen(phase_color(color), width=2, style=Qt.PenStyle.DashLine))
            self.plot.addItem(line, ignoreBounds=True)
            self._segments.append(line)

    def _set_range(self) -> None:
        if self._data is None:
            return
        span = self._data.trace_span if self.span.currentData() == "trace" else None
        low, high = span or self._data.span
        pad = 0.03 * (high - low) if high > low else 0.5
        self.plot.setXRange(low - pad, high + pad, padding=0)
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis)

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


class RateView(QWidget):
    """dP/dt of every sample (or phase) versus time, on the time axis of the evolution plot (as in the v1).

    "Moyenne sur un intervalle" shows an interval to drag, here or on the
    evolution plot: the mean dP/dt of every curve over it is written under the
    plot (and copied as a table with "Copier").
    """

    GROUPINGS = (("par échantillon", "family"), ("par phase", "phase"))

    def __init__(self, link_to: Optional[pg.PlotWidget] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_pick: Optional[Callable[[float], None]] = None
        self.on_options: Optional[Callable[[str, int], None]] = None
        self.on_interval: Optional[Callable[[float, float], None]] = None
        self.on_interval_toggle: Optional[Callable[[bool], None]] = None
        self.by = QComboBox()
        for label, key in self.GROUPINGS:
            self.by.addItem(label, key)
        self.by.setToolTip("Par échantillon : les phases d'une même substance (glaces VII, VI, Ih...) forment une\n"
                           "seule courbe, continue à travers les transitions. Par phase : une courbe par phase.")
        self.window = QSpinBox()
        self.window.setRange(1, 50)
        self.window.setPrefix("± ")
        self.window.setSuffix(" spectres")
        self.window.setToolTip("dP/dt = pente d'une droite ajustée sur le spectre et ses voisins de chaque côté :\n"
                               "plus de spectres, courbe plus lisse mais moins résolue en temps.")
        self.toggle = QPushButton("Moyenne sur un intervalle")
        self.toggle.setCheckable(True)
        self.toggle.setToolTip("dP/dt moyen = pente de la droite ajustée sur tous les spectres de l'intervalle,\n"
                               "à déplacer ici ou sur le graphe du dessus.")
        self.copy = QPushButton("Copier")
        self.copy.setToolTip("Copier le dP/dt moyen de chaque courbe (tableau pour un tableur)")
        self.copy.setEnabled(False)
        self.message = QLabel("")
        self.message.setStyleSheet("color: #8a5a00")
        self.by.currentIndexChanged.connect(self._options_chosen)
        self.window.valueChanged.connect(self._options_chosen)
        self.toggle.toggled.connect(lambda on: self.on_interval_toggle and self.on_interval_toggle(on))
        self.copy.clicked.connect(self._copy)
        bar = QHBoxLayout()
        for widget in (QLabel("dP/dt :"), self.by, QLabel("  lissage :"), self.window, self.toggle, self.copy,
                       self.message):
            bar.addWidget(widget)
        bar.addStretch(1)

        self.plot = pg.PlotWidget()
        item = self.plot.getPlotItem()
        item.getAxis("left").setWidth(AXIS_WIDTH)
        item.showAxis("right")  # empty: the time axes of the dDAC plots line up
        item.getAxis("right").setWidth(RIGHT_AXIS_WIDTH)
        item.getAxis("right").setStyle(showValues=False)
        self.plot.addLegend(offset=(10, 10))
        self.plot.setLabel("left", "dP/dt (GPa/ms)")
        self.plot.setLabel("bottom", "t (ms)")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#b0b0b0")), ignoreBounds=True)
        if link_to is not None:
            item.setXLink(link_to.getPlotItem())
        self.marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#808080", style=Qt.PenStyle.DotLine))
        self.plot.addItem(self.marker, ignoreBounds=True)
        self.interval = _Interval(self.plot, lambda start, stop: self.on_interval and self.on_interval(start, stop))
        self._items: List[pg.GraphicsObject] = []
        self._means: List[pg.GraphicsObject] = []
        self._table = ""
        _scene_click(self.plot, lambda x, y: self.on_pick and self.on_pick(x))
        self.result = QLabel("")
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.result.setWordWrap(True)
        self.spacer = spacer()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addLayout(_row(self.plot, self.spacer))
        layout.addWidget(self.result)

    def _options_chosen(self, *_) -> None:
        if self.on_options is not None:
            self.on_options(self.by.currentData(), self.window.value())

    def _copy(self) -> None:
        if self._table:
            QGuiApplication.clipboard().setText(self._table)

    def clear(self) -> None:
        self._remove(self._items)
        self.show_interval(None)
        self.message.setText("")

    def _remove(self, items: List[pg.GraphicsObject]) -> None:
        for item in items:
            self.plot.removeItem(item)
        items.clear()

    def show_rates(self, data) -> None:
        self._remove(self._items)
        for box in (self.by, self.window):
            box.blockSignals(True)
        self.by.setCurrentIndex(max(self.by.findData(data.by), 0))
        self.window.setValue(data.half_window)
        for box in (self.by, self.window):
            box.blockSignals(False)
        self.message.setText(data.message)
        self.plot.setLabel("bottom", data.x_label)
        for series in data.series:
            color = phase_color(series.color)
            bars = pg.ErrorBarItem(x=series.x, y=series.y, height=2 * series.error, beam=0, pen=pg.mkPen(color, width=1))
            curve = pg.PlotDataItem(series.x, series.y, pen=pg.mkPen(color, width=1.5), symbol="o", symbolSize=5,
                                    symbolBrush=pg.mkBrush(color), symbolPen=None, name=series.name)
            self.plot.addItem(bars, ignoreBounds=True)
            self.plot.addItem(curve)
            self._items += [bars, curve]
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis)  # x follows the evolution plot

    def show_interval(self, found) -> None:
        """Mean dP/dt over the interval (display.IntervalData), or None when the interval is off."""
        self.interval.show(None if found is None else (found.start, found.stop))
        self.toggle.blockSignals(True)
        self.toggle.setChecked(found is not None)
        self.toggle.blockSignals(False)
        self._remove(self._means)
        self._table = found.table if found is not None and found.means else ""
        self.copy.setEnabled(bool(self._table))
        if found is None:
            self.result.setText("")
            return
        for mean, color in zip(found.means, found.colors):
            line = pg.PlotDataItem([mean.start, mean.stop], [mean.rate, mean.rate],
                                   pen=pg.mkPen(phase_color(color), width=2, style=Qt.PenStyle.DashLine))
            self.plot.addItem(line, ignoreBounds=True)
            self._means.append(line)
        lines = [f'<span style="color:{phase_color(c)}">■</span> {text}' for text, c in zip(found.lines, found.colors)]
        head = f"<b>Moyenne de {found.start:.4g} à {found.stop:.4g} ms</b>"
        self.result.setText("<br>".join([head] + (lines or ["pas assez de spectres dans l'intervalle"])))

    def set_marker(self, x: float) -> None:
        self.marker.setPos(x)


class TimeMapView(QWidget):
    """Integrated patterns side by side: time along x (shared with the evolution plot), 2θ / Q / d along y."""

    def __init__(self, link_to: Optional[pg.PlotWidget] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_pick: Optional[Callable[[float], None]] = None
        self.panel = ImagePanel("map", scale="log")  # a gasket peak 100 times stronger hides the sample otherwise
        self.panel.add_control(QLabel("Diffractogrammes intégrés —"))
        self.plot, self.image = self.panel.plot, self.panel.image
        item = self.plot.getPlotItem()
        item.showAxis("right")  # empty, as wide as the oscilloscope axis above: the time axes line up
        item.getAxis("right").setWidth(RIGHT_AXIS_WIDTH)
        item.getAxis("right").setStyle(showValues=False)
        if link_to is not None:
            item.setXLink(link_to.getPlotItem())  # its x autorange is off: the evolution plot sets the range
        self.marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.marker, ignoreBounds=True)
        _scene_click(self.plot, lambda x, y: self.on_pick and self.on_pick(x))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.panel)

    def clear(self) -> None:
        self.panel.clear()

    def show_map(self, data) -> None:
        x_step = (data.x[-1] - data.x[0]) / (len(data.x) - 1) if len(data.x) > 1 else 1.0
        y_step = (data.y[-1] - data.y[0]) / max(len(data.y) - 1, 1)
        rect = QRectF(data.x[0] - x_step / 2, data.y[0] - y_step / 2, len(data.x) * x_step, len(data.y) * y_step)
        self.panel.set_image(data.image, rect)
        self.plot.setLabel("bottom", data.x_label)
        self.plot.setLabel("left", data.y_label)
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis)  # x follows the evolution plot

    def set_marker(self, x: float) -> None:
        self.marker.setPos(x)


class CakeView(QWidget):
    """Cake (2θ × azimuth) of the current spectrum, on the same x axis as the spectrum; or the raw image."""

    def __init__(self, link_to: Optional[pg.PlotWidget] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_mode: Optional[Callable[[str], None]] = None
        self.link_to = link_to
        self.mode = QComboBox()
        self.mode.addItem("cake (2θ × azimut)", "cake")
        self.mode.addItem("image brute", "image")
        self.mode.currentIndexChanged.connect(lambda i: self.on_mode and self.on_mode(self.mode.itemData(i)))
        self.message = QLabel("")
        self.message.setStyleSheet("color: #8a5a00")
        self.panel = ImagePanel("cake", scale="log", histogram=False)  # narrow: the histogram is optional
        for widget in (QLabel("Intégration :"), self.mode):
            self.panel.add_control(widget)
        self.panel.bar.addWidget(self.message)
        self.plot, self.image = self.panel.plot, self.panel.image
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.panel)

    def clear(self) -> None:
        self.panel.clear()
        self.message.setText("")

    def show_message(self, text: str) -> None:
        self.panel.clear()
        self.message.setText(text)

    def show_cake(self, data) -> None:
        self.message.setText("")
        step = (data.x[-1] - data.x[0]) / max(len(data.x) - 1, 1)
        az_step = (data.azimuth[-1] - data.azimuth[0]) / max(len(data.azimuth) - 1, 1)
        self.panel.set_image(data.image, QRectF(data.x[0] - step / 2, data.azimuth[0] - az_step / 2,
                                                data.x[-1] - data.x[0] + step,
                                                data.azimuth[-1] - data.azimuth[0] + az_step))
        item = self.plot.getPlotItem()
        item.setLabel("bottom", data.x_label)
        item.setLabel("left", "azimut (°)")
        item.getViewBox().setAspectLocked(False)
        item.invertY(False)
        if self.link_to is not None:
            item.setXLink(self.link_to.getPlotItem())  # the rings line up with the phase ticks
        item.enableAutoRange(axis=pg.ViewBox.YAxis)

    def show_image(self, image) -> None:
        self.message.setText("")
        image = np.asarray(image, dtype=float)
        self.panel.set_image(image, QRectF(0, 0, image.shape[1], image.shape[0]))
        item = self.plot.getPlotItem()
        item.setXLink(None)
        item.setLabel("bottom", "pixel")
        item.setLabel("left", "pixel")
        item.getViewBox().setAspectLocked(True)
        item.invertY(True)
        item.enableAutoRange()
