"""Plots of the viewer (pyqtgraph): current spectrum and its cake, quantities versus time, map of the patterns."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from trxrd.gui.views.images import AXIS_WIDTH, ImagePanel, keep_in_limits, limit_view, span_of, spacer

pg.setConfigOptions(antialias=True, background="w", foreground="k")

# one colour per phase, in the order of display.phase_order (colour-blind friendly)
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f")
RIGHT_AXIS_WIDTH = 56  # oscilloscope axis of the evolution plot; the map below keeps the same margin
# detected peaks: used by the phase search, too broad (ignored), in an excluded zone (free peak)
DETECTION_COLORS = {"used": "#2ca02c", "broad": "#ff7f0e", "excluded": "#7f7f7f"}


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


def drag_draws(plot: pg.PlotWidget, on_draw: Callable[..., None], active: Optional[Callable[[], bool]] = None) -> None:
    """Shift + left drag draws an interval on `plot` instead of moving the view.

    on_draw(start, stop, finished, modifiers), in data coordinates, the keys being those held at the start of the
    drag. `active`: when False the view moves as usual.
    """
    box = plot.getPlotItem().vb
    move = box.mouseDragEvent
    held = None  # modifiers of the drag in progress; None: it moves the view

    def drag(event, axis=None) -> None:
        nonlocal held
        if event.isStart():
            wanted = (axis is None and event.button() == Qt.MouseButton.LeftButton
                      and event.modifiers() & Qt.KeyboardModifier.ShiftModifier and (active is None or active()))
            held = event.modifiers() if wanted else None
        if held is None:
            move(event, axis=axis)
            return
        event.accept()
        start = box.mapSceneToView(event.buttonDownScenePos()).x()
        on_draw(start, box.mapSceneToView(event.scenePos()).x(), event.isFinish(), held)

    box.mouseDragEvent = drag


def _row(plot: QWidget, margin: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(0)
    row.addWidget(plot)
    row.addWidget(margin)
    return row


class MovableRegion:
    """Interval on a plot that the user drags (zone of spectra, 2θ range). Reports the user's drags, not the program's moves."""

    def __init__(self, plot: pg.PlotWidget, on_change: Callable[[float, float], None],
                 brush=(255, 170, 0, 45), pen: str = "#c07000", z: float = -10) -> None:
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(*brush), pen=pg.mkPen(pen, width=1.5))
        self.region.setZValue(z)
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


class _PeakHandle(pg.TargetItem):
    """Marker of a peak of the model: dragged to move it, clicked to choose it."""

    def __init__(self, ref, on_click: Callable[[object], None], **options) -> None:
        super().__init__(**options)
        self.ref, self.on_click = ref, on_click

    def mouseClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            self.on_click(self.ref)
        else:
            super().mouseClickEvent(event)


class _Zone(pg.LinearRegionItem):
    """Excluded zone on the spectrum: dragged or clicked while the zones are edited."""

    def __init__(self, row: int, on_click: Callable[[int], None], **options) -> None:
        super().__init__(**options)
        self.row, self.on_click = row, on_click

    def mouseClickEvent(self, event) -> None:
        if self.movable and event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            self.on_click(self.row)
        else:
            super().mouseClickEvent(event)


class SpectrumView(QWidget):
    """Data, background and fit; under them a band with one row of ticks per phase (height ~ intensity).

    The band has its own vertical scale and only shares the x axis: zooming on
    a small peak never moves the ticks out of sight. Faint vertical lines show
    the expected lines across the pattern too, at any zoom.
    """

    ROW_HEIGHT = 16  # px, one row of the band
    AXIS_HEIGHT = 46  # px, the x axis and its label, under the band
    RESIDUAL_HEIGHT = 80  # px
    GUIDE_MIN_INTENSITY = 5.0  # % of the card: weaker lines get no vertical line
    OUTSIDE_BRUSH = (0, 0, 0, 28)  # outside the fit zone
    EXCLUDED_BRUSH = (214, 39, 40, 45)
    CHOSEN_BRUSH = (214, 39, 40, 95)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_add_peak: Optional[Callable[[float, float], None]] = None
        self.on_move_peak: Optional[Callable[[object, float, float], None]] = None
        self.on_pick_peak: Optional[Callable[[object], None]] = None
        self.on_cursor: Optional[Callable[[float, float], None]] = None
        self.on_zone_drawn: Optional[Callable[[float, float, bool], None]] = None
        self.on_fit_zone: Optional[Callable[[float, float], None]] = None
        self.on_zone_moved: Optional[Callable[[int, float, float], None]] = None
        self.on_zone_picked: Optional[Callable[[int], None]] = None
        self.handles: Optional[List[_PeakHandle]] = None  # None: not editing the peaks
        self.plot = pg.PlotWidget()
        cross = pg.mkPen("#303030", width=1, style=Qt.PenStyle.DotLine)
        self.cross_x = pg.InfiniteLine(angle=90, movable=False, pen=cross)  # cursor of the keys C and R
        self.cross_y = pg.InfiniteLine(angle=0, movable=False, pen=cross)
        for line in (self.cross_x, self.cross_y):
            line.hide()
            self.plot.addItem(line, ignoreBounds=True)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self.plot.getAxis("left").setWidth(AXIS_WIDTH)  # aligned with the cake below
        self.plot.getAxis("bottom").setStyle(showValues=False)  # the x axis is under the band
        self.plot.getAxis("bottom").setHeight(2)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.plot.setLabel("left", "Intensité")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.data = self.plot.plot(pen=pg.mkPen("#404040", width=1), name="filtré − fond")
        self.baseline = self.plot.plot(pen=pg.mkPen("#1f77b4", width=1, style=Qt.PenStyle.DashLine), name="fond")
        self.fit = self.plot.plot(pen=pg.mkPen("#d62728", width=1.5), name="ajustement")
        self.residual = pg.PlotWidget()  # raw - background - model, on the same x axis
        self.residual.getAxis("left").setWidth(AXIS_WIDTH)
        self.residual.getAxis("bottom").setStyle(showValues=False)
        self.residual.getAxis("bottom").setHeight(2)
        self.residual.setLabel("left", "Résidu")
        self.residual.showGrid(x=True, y=True, alpha=0.15)
        self.residual.setMenuEnabled(False)
        self.residual.hideButtons()
        self.residual.setFixedHeight(self.RESIDUAL_HEIGHT)
        self.residual.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#b0b0b0")), ignoreBounds=True)
        self.residual_curve = self.residual.plot(pen=pg.mkPen("#606060", width=1), connect="finite")
        self.residual.getPlotItem().setXLink(self.plot.getPlotItem())
        keep_in_limits(self.residual, self.plot)
        self.residual_shown = True
        self.residual.hide()
        self.fills_shown = True
        self._fills: List[pg.GraphicsObject] = []
        self.band = pg.PlotWidget()
        self.band.getAxis("left").setWidth(AXIS_WIDTH)
        self.band.getAxis("left").setStyle(tickLength=0)
        self.band.showGrid(x=True, alpha=0.15)
        self.band.setMouseEnabled(x=True, y=False)
        self.band.setMenuEnabled(False)
        self.band.hideButtons()
        self.band.getPlotItem().setXLink(self.plot.getPlotItem())
        keep_in_limits(self.band, self.plot)
        self.band.setYRange(0, 1, padding=0)
        self.band.setFixedHeight(self.AXIS_HEIGHT + self.ROW_HEIGHT)
        self.guides_shown = True
        self._ticks: List[pg.GraphicsObject] = []  # in the band
        self._guides: List[pg.GraphicsObject] = []  # vertical lines in the plot
        self._samples: List[pg.GraphicsObject] = []  # legend entries of the phases
        # detected peaks (Détection tab): markers, threshold, excluded zones; not in the legend
        self.threshold = pg.PlotDataItem(pen=pg.mkPen("#8a8a8a", width=1, style=Qt.PenStyle.DotLine))
        self.detected = pg.ScatterPlotItem(hoverable=True, tip=lambda x, y, data: data)
        self.detected.setZValue(10)
        for item in (self.threshold, self.detected):
            self.plot.addItem(item)
        # zones: what is outside the fit zone is grey; excluded zones are red; drawn with Shift while they are edited
        self.zones_edited = False
        self._limits = (0.0, 0.0)
        self.outside = [pg.LinearRegionItem(movable=False, brush=pg.mkBrush(*self.OUTSIDE_BRUSH), pen=pg.mkPen(None))
                        for _ in range(2)]
        self.fit_edges = [pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#303030", width=2),
                                          hoverPen=pg.mkPen("#ff9000", width=3)) for _ in range(2)]
        self.sketch = pg.LinearRegionItem(movable=False, pen=pg.mkPen(None))  # the zone being drawn
        for item in (*self.outside, self.sketch):
            item.setZValue(-20)
            item.hide()
            self.plot.addItem(item, ignoreBounds=True)
        for line in self.fit_edges:
            line.setZValue(15)
            line.hide()
            line.sigPositionChanged.connect(self._fit_edge_moving)
            line.sigPositionChangeFinished.connect(self._fit_edge_moved)
            self.plot.addItem(line, ignoreBounds=True)
        self._excluded: List[_Zone] = []
        drag_draws(self.plot, self._zone_drawing, lambda: self.zones_edited)
        self.spacer = spacer()  # shown with the histogram of the cake, so that both x axes line up
        self.spacer.hide()
        plots = QWidget()
        stacked = QVBoxLayout(plots)
        stacked.setContentsMargins(0, 0, 0, 0)
        stacked.setSpacing(0)
        stacked.addWidget(self.plot, 1)
        stacked.addWidget(self.residual)
        stacked.addWidget(self.band)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(_row(plots, self.spacer))

    def clear(self) -> None:
        for curve in (self.data, self.baseline, self.fit, self.residual_curve):
            curve.setData([], [])
        self.residual.hide()
        self.show_peak_fills([], None)
        self._clear_ticks()
        self.show_detection(None)
        self.show_zones(None)
        self.plot.setTitle("")
        limit_view(self.plot)

    def _clicked(self, event) -> None:
        """While editing the peaks: a click puts the cursor there, Ctrl + click adds a peak there."""
        box = self.plot.getPlotItem().vb
        if self.handles is None or event.button() != Qt.MouseButton.LeftButton:
            return
        if not box.sceneBoundingRect().contains(event.scenePos()):
            return
        point = box.mapSceneToView(event.scenePos())
        self.cross_x.setPos(point.x())
        self.cross_y.setPos(point.y())
        for line in (self.cross_x, self.cross_y):
            line.show()
        if self.on_cursor is not None:
            self.on_cursor(point.x(), point.y())
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and self.on_add_peak is not None:
            self.on_add_peak(point.x(), point.y())

    def _zone_drawing(self, start: float, stop: float, finished: bool, modifiers) -> None:
        fit_zone = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        self.sketch.setBrush(pg.mkBrush(*(self.OUTSIDE_BRUSH if fit_zone else self.CHOSEN_BRUSH)))
        self.sketch.setRegion((min(start, stop), max(start, stop)))
        self.sketch.setVisible(not finished)
        if finished and self.on_zone_drawn is not None:
            QTimer.singleShot(0, lambda: self.on_zone_drawn(start, stop, fit_zone))

    def _fit_edge_moving(self) -> None:
        self._show_outside(sorted(line.value() for line in self.fit_edges))

    def _fit_edge_moved(self) -> None:
        start, stop = sorted(line.value() for line in self.fit_edges)
        if self.on_fit_zone is not None:
            QTimer.singleShot(0, lambda: self.on_fit_zone(start, stop))

    def _zone_moved(self, zone: _Zone) -> None:
        start, stop = zone.getRegion()
        if self.on_zone_moved is not None:  # later: the zones are redrawn, this one is still in its mouse event
            QTimer.singleShot(0, lambda: self.on_zone_moved(zone.row, start, stop))

    def _zone_clicked(self, row: int) -> None:
        if self.on_zone_picked is not None:
            QTimer.singleShot(0, lambda: self.on_zone_picked(row))

    def _show_outside(self, fit) -> None:
        low, high = self._limits
        for item, region in zip(self.outside, ((low, fit[0]), (fit[1], high))):
            item.setRegion(region)
            item.setVisible(region[1] > region[0])

    def show_zones(self, marks, edited: bool = False, chosen: Optional[int] = None) -> None:
        """Fit zone and excluded zones (display.ZoneMarks), or None; `edited`: the mouse moves and draws them."""
        for item in self._excluded:
            self.plot.removeItem(item)
        self._excluded = []
        self.zones_edited = marks is not None and edited
        if marks is None:
            for item in (*self.outside, *self.fit_edges):
                item.hide()
            return
        self._limits = marks.limits
        self._show_outside(marks.fit)
        for line, value in zip(self.fit_edges, marks.fit):
            line.blockSignals(True)
            line.setBounds(marks.limits)
            line.setValue(value)
            line.blockSignals(False)
            line.setVisible(edited)
        for row, zone in enumerate(marks.excluded):
            brush = self.CHOSEN_BRUSH if row == chosen else self.EXCLUDED_BRUSH
            item = _Zone(row, self._zone_clicked, values=zone, movable=edited, brush=pg.mkBrush(*brush),
                         pen=pg.mkPen("#d62728", width=1))
            item.setZValue(-15)
            item.sigRegionChangeFinished.connect(self._zone_moved)
            self.plot.addItem(item, ignoreBounds=True)
            self._excluded.append(item)

    def _handle_clicked(self, ref) -> None:
        # later: the handles are redrawn, and this one must not disappear inside its own mouse event
        if self.on_pick_peak is not None:
            QTimer.singleShot(0, lambda: self.on_pick_peak(ref))

    def _handle_moved(self, handle: _PeakHandle) -> None:
        position = handle.pos()
        if self.on_move_peak is not None:
            QTimer.singleShot(0, lambda: self.on_move_peak(handle.ref, position.x(), position.y()))

    def show_handles(self, handles, selected) -> None:
        """Handles of the peaks of the model (display.PeakHandle), or None when the peaks are not edited."""
        for item in self.handles or ():
            self.plot.removeItem(item)
        if handles is None:
            self.handles = None
            for line in (self.cross_x, self.cross_y):
                line.hide()
            return
        self.handles = []
        for handle in handles:
            color = phase_color(handle.color) if handle.color is not None else "#7f7f7f"
            chosen = handle.ref == selected
            item = _PeakHandle(handle.ref, self._handle_clicked, pos=(handle.x, handle.y), size=15 if chosen else 10, symbol="d",
                               pen=pg.mkPen("k" if chosen else color, width=2 if chosen else 1),
                               brush=pg.mkBrush(color) if handle.enabled else pg.mkBrush(None),
                               hoverBrush=pg.mkBrush("#ffd000"))
            item.setZValue(20)
            item.setToolTip(handle.tip)
            item.sigPositionChangeFinished.connect(self._handle_moved)
            self.plot.addItem(item, ignoreBounds=True)
            self.handles.append(item)

    def show_peak_fills(self, curves, selected) -> None:
        """Each peak of the model filled in the colour of its phase (display.PeakCurve), the chosen one stronger."""
        for item in self._fills:
            self.plot.removeItem(item)
        self._fills = []
        for curve in curves if self.fills_shown else ():
            color = pg.mkColor(phase_color(curve.color) if curve.color is not None else "#7f7f7f")
            color.setAlpha(150 if curve.ref == selected else 60)
            fill = pg.FillBetweenItem(pg.PlotCurveItem(curve.x, curve.top), pg.PlotCurveItem(curve.x, curve.bottom),
                                      brush=pg.mkBrush(color))
            fill.setZValue(-1)
            self.plot.addItem(fill, ignoreBounds=True)
            self._fills.append(fill)

    def set_fills(self, shown: bool) -> None:
        self.fills_shown = bool(shown)
        for item in self._fills:
            item.setVisible(self.fills_shown)

    def set_residual(self, shown: bool) -> None:
        self.residual_shown = bool(shown)
        self.residual.setVisible(self.residual_shown and self.residual_curve.getData()[0] is not None
                                 and len(self.residual_curve.getData()[0]) > 0)

    def set_guides(self, shown: bool) -> None:
        """Vertical lines of the expected reflections across the pattern."""
        self.guides_shown = bool(shown)
        for guide in self._guides:
            guide.setVisible(self.guides_shown)

    def _set_data_label(self, text: str) -> None:
        for sample, label in self.legend.items:
            if sample.item is self.data and label.text != text:
                label.setText(text)

    def _list_baseline(self, shown: bool) -> None:
        """The background is only in the legend when it is drawn (raw signal)."""
        listed = any(sample.item is self.baseline for sample, _ in self.legend.items)
        if shown and not listed:
            self.legend.addItem(self.baseline, "fond")
        elif listed and not shown:
            self.legend.removeItem(self.baseline)

    def show_detection(self, marks) -> None:
        """Detected peaks (display.DetectionMarks), or None to hide them."""
        if marks is None:
            self.detected.setData([])
            self.threshold.setData([], [])
            return
        spots = []
        for x, y, kind, shoulder, tip in zip(marks.x, marks.y, marks.kinds, marks.shoulder, marks.tips):
            color = DETECTION_COLORS[kind]
            spots.append({"pos": (x, y), "symbol": "t", "size": 11, "data": tip, "pen": pg.mkPen(color, width=1.5),
                          "brush": pg.mkBrush(None) if shoulder else pg.mkBrush(color)})
        self.detected.setData(spots)
        self.threshold.setData(marks.threshold_x, marks.threshold_y)

    def _clear_ticks(self) -> None:
        for item in self._ticks:
            self.band.removeItem(item)
        for item in self._guides:
            self.plot.removeItem(item)
        for item in self._samples:
            self.legend.removeItem(item)
        self._ticks, self._guides, self._samples = [], [], []
        self.band.getAxis("left").setTicks([[]])

    def _add_guides(self, lines, color: str) -> None:
        faint = pg.mkColor(color)
        faint.setAlpha(90)
        pen = pg.mkPen(faint, width=1, style=Qt.PenStyle.DashLine)
        for position, intensity in zip(lines.x, lines.intensity):
            if intensity >= self.GUIDE_MIN_INTENSITY:
                guide = pg.InfiniteLine(pos=float(position), angle=90, movable=False, pen=pen)
                guide.setZValue(-5)
                guide.setVisible(self.guides_shown)
                self.plot.addItem(guide, ignoreBounds=True)
                self._guides.append(guide)

    def show_spectrum(self, curves, reset: bool) -> None:
        self.plot.setTitle(curves.title)
        self.band.setLabel("bottom", curves.x_label)
        self._set_data_label(curves.data_label)
        self._list_baseline(curves.baseline is not None)
        limit_view(self.plot, span_of(curves.x), span_of(curves.intensity, curves.baseline, curves.fit))
        self.data.setData(curves.x, curves.intensity)
        for item, y in ((self.baseline, curves.baseline), (self.fit, curves.fit)):
            item.setData(curves.x, y) if y is not None else item.setData([], [])
        if curves.residual is not None:
            self.residual_curve.setData(curves.x, curves.residual)
            limit_view(self.residual, y=span_of(curves.residual, [0.0]))
        else:
            self.residual_curve.setData([], [])
        self.residual.setVisible(self.residual_shown and curves.residual is not None)
        self._clear_ticks()
        rows = [lines for lines in curves.lines if len(lines.x)]
        names = []
        for row, lines in enumerate(rows):
            color = phase_color(lines.color)
            bottom = len(rows) - 1 - row + 0.1  # first phase on top
            heights = 0.8 * (0.3 + 0.7 * np.clip(lines.intensity, 0, 100) / 100)
            x = np.repeat(lines.x, 2)
            y = np.column_stack([np.full(len(lines.x), bottom), bottom + heights]).ravel()
            ticks = pg.PlotDataItem(x, y, connect="pairs", pen=pg.mkPen(color, width=2))
            tips = pg.ScatterPlotItem(
                lines.x, bottom + heights, size=6, brush=pg.mkBrush(color), pen=None, hoverable=True,
                data=[f"{lines.phase} ({label})  I = {i:.0f} %" for label, i in zip(lines.labels, lines.intensity)],
                tip=lambda x, y, data: f"{data}\nx = {x:.4f}",
            )
            for item in (ticks, tips):
                self.band.addItem(item)
                self._ticks.append(item)
            sample = pg.PlotDataItem(pen=pg.mkPen(color, width=2))
            self.legend.addItem(sample, f"{lines.phase}  {lines.pressure:.2f} GPa")
            self._samples.append(sample)
            names.append((bottom + 0.4, lines.short))
            self._add_guides(lines, color)
        self.band.getAxis("left").setTicks([names])
        limit_view(self.band, y=(0, max(len(rows), 1)), margin=0.0)  # on x it follows the spectrum
        self.band.setYRange(0, max(len(rows), 1), padding=0)
        self.band.setFixedHeight(self.AXIS_HEIGHT + self.ROW_HEIGHT * max(len(rows), 1))
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
        self.wanted_channel: Optional[str] = None  # the user's choice: it stays from one refresh and one run to the next
        self.channel.currentTextChanged.connect(self._channel_chosen)
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
        keep_in_limits(self.scope, self.plot)
        item.vb.sigResized.connect(lambda: self.scope.setGeometry(item.vb.sceneBoundingRect()))
        self.trace = pg.PlotDataItem(pen=pg.mkPen("#4a6fa5", width=1))
        self.trace.setDownsampling(auto=True, method="peak")
        self.trace.setClipToView(True)
        self.scope.addItem(self.trace)
        self._items: List[pg.GraphicsObject] = []
        self._segments: List[pg.GraphicsObject] = []
        self._traces: Dict[str, object] = {}
        self._data = None
        self.interval = MovableRegion(self.plot, lambda start, stop: self.on_interval and self.on_interval(start, stop))
        drag_draws(self.plot, lambda start, stop, *_: self.on_interval and start != stop and self.on_interval(start, stop))
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

    def _channel_chosen(self, name: str) -> None:
        self.wanted_channel = name
        self._show_trace(name)

    def show_evolution(self, data) -> None:
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
        wanted = self.wanted_channel
        choice = wanted if wanted == self.NO_TRACE or wanted in self._traces else (names[0] if names else self.NO_TRACE)
        self.channel.blockSignals(True)  # not the user's choice: a run without that channel does not forget it
        self.channel.clear()
        self.channel.addItems([self.NO_TRACE] + names)
        self.channel.setCurrentText(choice)
        self.channel.blockSignals(False)
        self._show_trace(choice)
        low, high = data.span
        self.band.setRegion((low, high))
        self.band.setVisible(len(data.x) > 1)
        self._set_range()

    def show_interval(self, region, segments) -> None:
        """The zone of spectra (None: hidden) and the straight line of the mean dP/dt of each curve over it."""
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
        values = [a for s in self._data.series for a in (s.y - np.nan_to_num(s.error), s.y + np.nan_to_num(s.error))]
        limit_view(self.plot, (low, high), span_of(*values))  # the plots below follow on x
        self.plot.setXRange(low - pad, high + pad, padding=0)
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis)

    def _show_trace(self, name: str) -> None:
        trace = self._traces.get(name)
        if trace is None:
            self.trace.setData([], [])
            self.plot.getPlotItem().getAxis("right").setLabel("")
            return
        self.trace.setData(trace.x, trace.y)
        limit_view(self.scope, y=span_of(trace.y), margin=0.05)
        self.plot.getPlotItem().getAxis("right").setLabel(trace.name)
        self.scope.enableAutoRange(axis=pg.ViewBox.YAxis)

    def set_marker(self, x: float) -> None:
        self.marker.setPos(x)


class RateView(QWidget):
    """dP/dt of every sample (or phase) versus time, on the time axis of the evolution plot (as in the v1).

    "Zone de spectres" shows a time zone, the same on the three dDAC plots, to
    drag or to draw with Shift + drag. It chooses the spectra, and the mean
    dP/dt of every curve over it is written under the plot (and copied as a
    table with "Copier").
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
        self.toggle = QPushButton("Zone de spectres")
        self.toggle.setCheckable(True)
        self.toggle.setToolTip("Zone de temps, à tirer sur l'un des trois graphes, ou à tracer avec Maj + glisser.\n"
                               "Elle choisit les spectres (comme l'onglet Spectres : ajuster, coller, traiter, analyser…)\n"
                               "et donne le dP/dt moyen : pente de la droite ajustée sur tous ses spectres.")
        self.copy = QPushButton("Copier")
        self.copy.setToolTip("Copier le dP/dt moyen de chaque courbe (tableau pour un tableur)")
        self.copy.setEnabled(False)
        self.message = QLabel("")
        self.message.setWordWrap(True)  # a long message must not widen the window
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
            keep_in_limits(self.plot, link_to)
        self.marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#808080", style=Qt.PenStyle.DotLine))
        self.plot.addItem(self.marker, ignoreBounds=True)
        self.interval = MovableRegion(self.plot, lambda start, stop: self.on_interval and self.on_interval(start, stop))
        drag_draws(self.plot, lambda start, stop, *_: self.on_interval and start != stop and self.on_interval(start, stop))
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
        values = [a for s in data.series for a in (s.y - np.nan_to_num(s.error), s.y + np.nan_to_num(s.error), [0.0])]
        limit_view(self.plot, y=span_of(*values))
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis)  # x follows the evolution plot

    def show_interval(self, found, region=None) -> None:
        """The zone of spectra (None: off) and the mean dP/dt over it (display.IntervalData; None without times)."""
        self.interval.show(region)
        self.toggle.blockSignals(True)
        self.toggle.setChecked(region is not None)
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
        self.on_interval: Optional[Callable[[float, float], None]] = None
        self.panel = ImagePanel("map", scale="log")  # a gasket peak 100 times stronger hides the sample otherwise
        self.panel.add_control(QLabel("Diffractogrammes intégrés —"))
        self.plot, self.image = self.panel.plot, self.panel.image
        item = self.plot.getPlotItem()
        item.showAxis("right")  # empty, as wide as the oscilloscope axis above: the time axes line up
        item.getAxis("right").setWidth(RIGHT_AXIS_WIDTH)
        item.getAxis("right").setStyle(showValues=False)
        if link_to is not None:
            item.setXLink(link_to.getPlotItem())  # its x autorange is off: the evolution plot sets the range
            keep_in_limits(self.plot, link_to)
        self.marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.marker, ignoreBounds=True)
        self.interval = MovableRegion(self.plot, lambda start, stop: self.on_interval and self.on_interval(start, stop),
                                      z=10)  # over the image
        drag_draws(self.plot, lambda start, stop, *_: self.on_interval and start != stop and self.on_interval(start, stop))
        _scene_click(self.plot, lambda x, y: self.on_pick and self.on_pick(x))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.panel)

    def clear(self) -> None:
        self.panel.clear()

    def show_interval(self, region) -> None:
        self.interval.show(region)

    def show_map(self, data) -> None:
        x_step = (data.x[-1] - data.x[0]) / (len(data.x) - 1) if len(data.x) > 1 else 1.0
        y_step = (data.y[-1] - data.y[0]) / max(len(data.y) - 1, 1)
        rect = QRectF(data.x[0] - x_step / 2, data.y[0] - y_step / 2, len(data.x) * x_step, len(data.y) * y_step)
        self.panel.set_image(data.image, rect, limit_x=False)  # x follows the evolution plot
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
        self.message.setWordWrap(True)  # a long message must not widen the window
        self.message.setStyleSheet("color: #8a5a00")
        self.panel = ImagePanel("cake", scale="log", histogram=False)  # narrow: the histogram is optional
        if link_to is not None:
            keep_in_limits(self.panel.plot, link_to)
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
        self.plot.getViewBox().setAspectLocked(False)
        self.panel.set_image(data.image, QRectF(data.x[0] - step / 2, data.azimuth[0] - az_step / 2,
                                                data.x[-1] - data.x[0] + step,
                                                data.azimuth[-1] - data.azimuth[0] + az_step),
                             limit_x=self.link_to is None)  # linked: the spectrum holds the limits
        item = self.plot.getPlotItem()
        item.setLabel("bottom", data.x_label)
        item.setLabel("left", "azimut (°)")
        item.invertY(False)
        if self.link_to is not None:
            item.setXLink(self.link_to.getPlotItem())  # the rings line up with the phase ticks
        item.enableAutoRange(axis=pg.ViewBox.YAxis)

    def show_image(self, image) -> None:
        self.message.setText("")
        image = np.asarray(image, dtype=float)
        item = self.plot.getPlotItem()
        item.setXLink(None)
        item.getViewBox().setAspectLocked(True)
        self.panel.set_image(image, QRectF(0, 0, image.shape[1], image.shape[0]))
        item.setLabel("bottom", "pixel")
        item.setLabel("left", "pixel")
        item.invertY(True)
        item.enableAutoRange()
