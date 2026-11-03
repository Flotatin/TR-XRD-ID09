"""Image with adjustable scale, levels and colour map (cake, map of the patterns, calibration image).

As in Dioptas: a histogram beside the image whose levels can be dragged,
a choice of scale (linear, log, square root) and of colour map. The choices
are remembered per image (QSettings).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from trxrd.gui.presenters import display

COLORMAPS = (("viridis", "viridis"), ("inferno", "inferno"), ("magma", "magma"), ("plasma", "plasma"),
             ("cividis", "cividis"), ("turbo", "turbo"), ("gris", "gray"), ("gris inversé", "gray_r"))
SCALES = (("log", "log"), ("linéaire", "linear"), ("racine", "sqrt"))
AXIS_WIDTH = 64  # fixed width of the left axes: plots stacked on a shared x axis stay aligned


def colormap(name: str) -> pg.ColorMap:
    if name == "gray":
        return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
    if name == "gray_r":
        return pg.ColorMap([0.0, 1.0], [(255, 255, 255), (0, 0, 0)])
    return pg.colormap.get(name)


def _box(view) -> pg.ViewBox:
    return view if isinstance(view, pg.ViewBox) else view.getViewBox()


def limit_view(view, x: Optional[Tuple[float, float]] = None, y: Optional[Tuple[float, float]] = None,
               margin: float = 0.03) -> None:
    """Zooming out stops at these ranges (data coordinates), plus `margin` of their width; None: no limit.

    `view`: a plot or a view box. A plot linked to another on x has no x limit of its own: see keep_in_limits.
    """
    limits = {}
    for axis, span in (("x", x), ("y", y)):
        low = high = None
        if span is not None and np.isfinite(span).all():
            low, high = float(min(span)), float(max(span))
            pad = margin * (high - low) if high > low else 0.5
            low, high = low - pad, high + pad
        limits[f"{axis}Min"], limits[f"{axis}Max"] = low, high
    _box(view).setLimits(**limits)


def keep_in_limits(follower, main) -> None:
    """`follower`, while linked to `main` on x, stays inside the x limits of `main`.

    A view already at its limit does not move, and then nothing brings back the plots linked to it.
    """
    box, target = _box(follower), _box(main)

    def clamp(*_) -> None:
        if box.linkedView(pg.ViewBox.XAxis) is not target:
            return
        low, high = target.state["limits"]["xLimits"]
        start, stop = box.viewRange()[0]
        if (low is not None and start < low) or (high is not None and stop > high):
            box.setXRange(start if low is None else max(start, low), stop if high is None else min(stop, high), padding=0)

    box.sigXRangeChanged.connect(clamp)


def span_of(*arrays) -> Optional[Tuple[float, float]]:
    """(min, max) of the finite values of `arrays`; None when there are none."""
    values = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays if a is not None] or [np.array([])])
    values = values[np.isfinite(values)]
    return (float(values.min()), float(values.max())) if values.size else None


def _combo(choices) -> QComboBox:
    box = QComboBox()
    for label, key in choices:
        box.addItem(label, key)
    return box


class ImagePanel(QWidget):
    """Plot with one image, its histogram (levels) and a bar: scale, colours, automatic levels.

    Owners add their own items to `plot` (rings, markers) and their own
    controls to the bar (`add_control`). `on_histogram(shown)` lets an owner
    keep a plot stacked above aligned with this one (same right margin).
    """

    HISTOGRAM_WIDTH = 115

    def __init__(self, key: str, scale: str = "log", cmap: str = "viridis", histogram: bool = True,
                 settings: Optional[QSettings] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.key = key
        self.settings = settings or QSettings()
        self.on_histogram: Optional[Callable[[bool], None]] = None
        self._raw: Optional[np.ndarray] = None
        self._rect: Optional[QRectF] = None
        self._extent: Optional[QRectF] = None  # what the image covers, in data coordinates
        self._limit_x = True
        self._levels: Optional[Tuple[float, float]] = None
        self._drawing = False  # levels set by the program, not dragged by the user

        self.plot = pg.PlotWidget()
        self.plot.getAxis("left").setWidth(AXIS_WIDTH)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.plot.getViewBox().sigResized.connect(lambda *_: self.update_limits())
        self.histogram = pg.HistogramLUTWidget(image=self.image)
        self.histogram.setFixedWidth(self.HISTOGRAM_WIDTH)
        self.histogram.item.sigLevelChangeFinished.connect(self._levels_dragged)

        self.scale = _combo(SCALES)
        self.colors = _combo(COLORMAPS)
        self.auto = QCheckBox("niveaux auto")
        self.auto.setToolTip("Niveaux recalculés pour chaque image (percentiles 1 et 99,8 %).\n"
                             "Décocher pour garder les niveaux réglés sur l'histogramme.")
        self.auto.setChecked(True)
        reset = QToolButton()
        reset.setText("Auto")
        reset.setToolTip("Recalculer les niveaux maintenant")
        reset.clicked.connect(self.reset_levels)
        self.show_histogram = QCheckBox("histogramme")

        self.scale.setCurrentIndex(max(self.scale.findData(self._setting("scale", scale)), 0))
        self.colors.setCurrentIndex(max(self.colors.findData(self._setting("colormap", cmap)), 0))
        self.show_histogram.setChecked(str(self._setting("histogram", histogram)).lower() in ("true", "1"))
        self.histogram.setVisible(self.show_histogram.isChecked())
        self._apply_colormap()
        self.scale.currentIndexChanged.connect(self._scale_changed)
        self.colors.currentIndexChanged.connect(self._colormap_changed)
        self.show_histogram.toggled.connect(self._histogram_toggled)

        self.bar = QHBoxLayout()
        self._custom = 0
        for widget in (QLabel("Échelle :"), self.scale, QLabel("Couleurs :"), self.colors, self.auto, reset,
                       self.show_histogram):
            self.bar.addWidget(widget)
        self.bar.addStretch(1)
        row = QHBoxLayout()
        row.setSpacing(0)
        row.addWidget(self.plot)
        row.addWidget(self.histogram)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.bar)
        layout.addLayout(row)

    # ------------------------------------------------------------------ owners

    def add_control(self, widget: QWidget) -> None:
        """Put an owner's control at the start of the bar (before the image settings)."""
        self.bar.insertWidget(self._custom, widget)
        self._custom += 1

    def set_image(self, image: np.ndarray, rect: Optional[QRectF] = None, limit_x: bool = True) -> None:
        """Show `image` ([rows, columns], NaN allowed) over `rect` in data coordinates (pixels if None).

        `limit_x`: False when the x axis of the plot is linked to another plot, which holds the limits.
        """
        self._raw = np.asarray(image, dtype=float)
        self._rect = rect
        self._extent = rect if rect is not None else QRectF(0, 0, self._raw.shape[1], self._raw.shape[0])
        self._limit_x = limit_x
        self._draw(self.auto.isChecked() or self._levels is None)
        self.update_limits()

    def clear(self) -> None:
        self._raw = None
        self._extent = None
        self.image.clear()
        self.update_limits()

    def update_limits(self) -> None:
        """Zooming out stops when the whole image is seen. With locked proportions the view is wider, or
        taller, than the image: strict limits would crop it."""
        if self._extent is None:
            limit_view(self.plot)
            return
        box, rect = self.plot.getViewBox(), self._extent
        extra_x = extra_y = 0.0
        if box.state["aspectLocked"]:
            ratio = box.width() / max(box.height(), 1.0)
            extra_x = max(rect.height() * ratio - rect.width(), 0.0) / 2
            extra_y = max(rect.width() / max(ratio, 1e-6) - rect.height(), 0.0) / 2
        x = (rect.left() - extra_x, rect.right() + extra_x) if self._limit_x else None
        limit_view(self.plot, x, (rect.top() - extra_y, rect.bottom() + extra_y), margin=0.0)

    @property
    def levels(self) -> Optional[Tuple[float, float]]:
        """Colour levels in the scaled values (log10 of the intensity in log scale)."""
        return self._levels

    def set_levels(self, low: float, high: float) -> None:
        """Fixed levels (scaled values); turns the automatic levels off."""
        self.auto.setChecked(False)
        self._levels = (float(low), float(high))
        if self._raw is not None:
            self._draw(False)

    def reset_levels(self) -> None:
        if self._raw is not None:
            self._draw(True)

    # ------------------------------------------------------------------ drawing

    def _draw(self, new_levels: bool) -> None:
        shown = display.scaled_image(self._raw, self.scale.currentData())
        if new_levels or self._levels is None:
            self._levels = display.robust_levels(shown)
        low = self._levels[0]
        self._drawing = True
        try:
            self.image.setImage(np.nan_to_num(shown, nan=low, posinf=low, neginf=low), levels=self._levels,
                                autoLevels=False)
        finally:
            self._drawing = False
        if self._rect is not None:
            self.image.setRect(self._rect)

    def _levels_dragged(self, *_) -> None:
        if self._drawing:
            return
        low, high = self.histogram.item.getLevels()
        self._levels = (float(low), float(high))
        self.auto.setChecked(False)  # the user chose these levels: keep them for the next images

    def _scale_changed(self) -> None:
        self.settings.setValue(f"images/{self.key}/scale", self.scale.currentData())
        if self._raw is not None:
            self._draw(True)  # the old levels mean nothing in the new scale

    def _apply_colormap(self) -> None:
        self.histogram.item.gradient.setColorMap(colormap(self.colors.currentData()))

    def _colormap_changed(self) -> None:
        self.settings.setValue(f"images/{self.key}/colormap", self.colors.currentData())
        self._apply_colormap()

    def _histogram_toggled(self, shown: bool) -> None:
        self.settings.setValue(f"images/{self.key}/histogram", bool(shown))
        self.histogram.setVisible(shown)
        if self.on_histogram is not None:
            self.on_histogram(shown)

    def _setting(self, name: str, default):
        return self.settings.value(f"images/{self.key}/{name}", default)


def spacer(width: int = ImagePanel.HISTOGRAM_WIDTH) -> QWidget:
    """Empty right margin for a plot stacked above an ImagePanel whose histogram is shown."""
    widget = QWidget()
    widget.setFixedWidth(width)
    return widget
