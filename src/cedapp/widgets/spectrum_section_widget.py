"""Widget helpers for the spectrum section."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QStackedWidget,
    QWidget,
)


class SpectrumSectionWidget:
    """Builder for the main spectrum plot section."""

    def __init__(self, main) -> None:
        self.main = main
        self.spectrum_box = QGroupBox("Spectrum")
        main.spectra_layout = QVBoxLayout()
        self.spectrum_box.setLayout(main.spectra_layout)

        self.spectrum_container = QWidget()
        container_layout = QVBoxLayout(self.spectrum_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        main.top_layout = QVBoxLayout()
        container_layout.addLayout(main.top_layout)
        main.spectra_layout.addWidget(self.spectrum_container)

        main.graph_layout = GraphicsLayoutWidget()
        main.graph_layout.ci.layout.setRowStretchFactor(0, 5)
        main.graph_layout.ci.layout.setRowStretchFactor(1, 1)
        main.graph_layout.ci.layout.setColumnStretchFactor(0, 1)
        size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(3)
        size_policy.setVerticalStretch(1)
        main.graph_layout.setSizePolicy(size_policy)
        main.top_layout.addWidget(main.graph_layout, 2)

        main.ax_spectrum = main.graph_layout.addPlot(row=0, col=0)
        main.ax_spectrum.scene().sigMouseClicked.connect(main.f_cross_spectrum)

        main.ax_dy = main.graph_layout.addPlot(row=1, col=0)
        main.ax_dy.setXLink(main.ax_spectrum)

        main.graph_layout.ci.setContentsMargins(0, 0, 0, 0)
        main.graph_layout.ci.layout.setSpacing(2)

        self.right_splitter = QSplitter(Qt.Horizontal)
        self.right_splitter.setChildrenCollapsible(False)
        main.top_layout.addWidget(self.right_splitter, 1)

        self.right_widget_container = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(2)
        self.right_widget_container.setMinimumWidth(220)
        self.right_widget_container.setMaximumWidth(500)
        self.right_widget_container.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.right_splitter.addWidget(self.right_widget_container)

        self.zoom_stack = QStackedWidget()
        self.right_splitter.addWidget(self.zoom_stack)
        self._zoom_stack_base_min_width = 240
        self.zoom_stack.setMinimumWidth(self._zoom_stack_base_min_width)

        main.zoom_widget = pg.PlotWidget()
        main.ax_zoom = main.zoom_widget.getPlotItem()
        main.ax_zoom.hideAxis("bottom")
        main.ax_zoom.hideAxis("left")
        main.ax_zoom.scene().sigMouseClicked.connect(main.f_cross_zoom)
        self.right_splitter.addWidget(main.zoom_widget)
        self._zoom_default_index = self.zoom_stack.indexOf(main.zoom_widget)

        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 2)


        main.plot_pic_fit = []
        main.plot_pic_fit_curves = []

        main.index_spec = 0
        if main.RUN is not None:
            main.Spectrum = main.RUN.Spectra[main.index_spec]
            main.list_name_gauges = [gauge.name for gauge in main.Spectrum.Gauges]
        else:
            main.list_name_gauges = []

        main.plot_blfit = None
        main.list_y_fit_start = []

        main.plot_raw_spectrum = pg.PlotDataItem([], [], pen=pg.mkPen("dimgray", width=1))
        main.ax_spectrum.addItem(main.plot_raw_spectrum)

        main.plot_filtered_spectrum = pg.PlotDataItem([], [], pen=pg.mkPen("darkorange", width=1))
        main.ax_spectrum.addItem(main.plot_filtered_spectrum)

        main.plot_baseline_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen("green", width=1, style=Qt.DashLine)
        )
        main.ax_spectrum.addItem(main.plot_baseline_curve)

        main.plot_curv_fit = pg.PlotDataItem([], [], pen=pg.mkPen("m", width=2))
        main.ax_spectrum.addItem(main.plot_curv_fit)

        main.plot_curv_dY = pg.PlotDataItem([], [], pen=pg.mkPen("m", width=2))
        main.ax_dy.addItem(main.plot_curv_dY)

        main.plot_ax4 = pg.PlotDataItem([], [], pen=pg.mkPen("g", width=2))
        main.ax_dy.addItem(main.plot_ax4)

        main.plot_fit_start = pg.PlotDataItem([], [], pen=pg.mkPen("g", width=2))
        main.ax_spectrum.addItem(main.plot_fit_start)

        main.plot_data_fit = pg.PlotDataItem([], [], pen=pg.mkPen("k", width=2), name="Fit")
        main.ax_spectrum.addItem(main.plot_data_fit)

        main.plot_zoom = pg.PlotDataItem([], [], pen=pg.mkPen("k", width=2), name="Zoom")
        main.ax_zoom.addItem(main.plot_zoom)

        main.baseline = pg.PlotCurveItem([], [])
        main.plot_ax3 = pg.FillBetweenItem(main.baseline, pg.PlotCurveItem([], []), brush=(100, 100, 255, 100))
        main.ax_zoom.addItem(main.plot_ax3)

        main.plot_pic_select = pg.FillBetweenItem(
            main.baseline,
            pg.PlotCurveItem([], []),
            brush=pg.mkBrush("#00ff00"),
        )
        main.ax_spectrum.addItem(main.plot_pic_select)
        main.plot_data_pic_solo = pg.PlotDataItem([], [], pen=pg.mkPen("r", width=2))
        main.ax_zoom.addItem(main.plot_data_pic_solo)

        main.axV = pg.InfiniteLine(angle=90, movable=False, pen="g")
        main.axH = pg.InfiniteLine(angle=0, movable=False, pen="g")
        main.ax_spectrum.addItem(main.axV)
        main.ax_spectrum.addItem(main.axH)

        main.cross_zoom = pg.ScatterPlotItem([0], [0], symbol="+", size=10, pen="r")
        main.ax_zoom.addItem(main.cross_zoom)

        exclusion_brush = pg.mkBrush(255, 0, 0, 80)
        exclusion_pen = pg.mkPen((255, 0, 0, 160))
        main.zoom_exclusion_left = pg.LinearRegionItem(
            values=(0, 0),
            movable=False,
            brush=exclusion_brush,
            pen=exclusion_pen,
        )
        main.zoom_exclusion_left.setVisible(False)
        main.zoom_exclusion_left.setZValue(20)
        main.ax_zoom.addItem(main.zoom_exclusion_left)

        main.zoom_exclusion_right = pg.LinearRegionItem(
            values=(0, 0),
            movable=False,
            brush=exclusion_brush,
            pen=exclusion_pen,
        )
        main.zoom_exclusion_right.setVisible(False)
        main.zoom_exclusion_right.setZValue(20)
        main.ax_zoom.addItem(main.zoom_exclusion_right)

        main.selected_file = None
        main.index_pic_select = 0
        main.index_jauge = -1
        main.y_fit_start = None
        main.save_value = 0

        main.init_session_vars()

        layout_check = QHBoxLayout()
        main.select_clic_box = QCheckBox("Select clic pic (q)", main)
        main.select_clic_box.setChecked(True)
        layout_check.addWidget(main.select_clic_box)

        main.zone_spectrum_box = QCheckBox("Zone Fit Spectrum (Z)", main)
        main.zone_spectrum_box.setChecked(True)
        main.zone_spectrum_box.toggled.connect(main._refresh_fit_context_cache)
        layout_check.addWidget(main.zone_spectrum_box)

        main.vslmfit = QCheckBox("vslmfit", main)
        main.vslmfit.setChecked(False)
        main.vslmfit.toggled.connect(main._refresh_fit_context_cache)
        layout_check.addWidget(main.vslmfit)

        main.spectra_layout.addLayout(layout_check)

        self.dhkl_box = QGroupBox("dhkl")
        main.layout_dhkl = QVBoxLayout()
        self.dhkl_box.setLayout(main.layout_dhkl)

    def add_to_layout(self, grid_layout) -> None:
        """Add the spectrum widgets to the provided grid layout."""

        grid_layout.addWidget(self.spectrum_box, 0, 2, 4, 1)
        grid_layout.addWidget(self.dhkl_box, 0, 1, 3, 1)
    
    def set_zoom_replacement_widget(self, widget) -> None:
        """Register an alternate widget displayed instead of the zoom plot."""

        if widget is None:
            return
        if self.zoom_stack.indexOf(widget) == -1:
            self.zoom_stack.addWidget(widget)

    def show_zoom_replacement(self, widget, enabled: bool) -> None:
        """Toggle display of a replacement widget in the zoom area."""

        if widget is None:
            return
        self.set_zoom_replacement_widget(widget)
        if enabled:
            self.zoom_stack.setCurrentWidget(widget)
            widget.show()
            self.main.zoom_widget.hide()
        else:
            self.zoom_stack.setCurrentIndex(self._zoom_default_index)
            self.main.zoom_widget.show()
            widget.hide()

    def set_plot_window_ratio(self, ratio: float | None) -> None:
        """Resize preference for the right plot window instead of stretching image content."""

        if ratio is None or not np.isfinite(ratio) or ratio <= 0:
            self.zoom_stack.setMinimumWidth(self._zoom_stack_base_min_width)
            return

        splitter_height = max(1, self.right_splitter.height())
        target_width = int(np.clip(splitter_height * float(ratio), self._zoom_stack_base_min_width, 2200))
        self.zoom_stack.setMinimumWidth(target_width)

        sizes = self.right_splitter.sizes()
        if len(sizes) >= 2:
            left_width = max(220, sizes[0])
            self.right_splitter.setSizes([left_width, target_width])

    def undock_right_panel(self, host_layout) -> None:
        """Move the right splitter (gauges + zoom plot) into another layout."""

        if self.right_splitter.parent() is self.spectrum_container:
            self.main.top_layout.removeWidget(self.right_splitter)
        self.right_splitter.setParent(None)
        host_layout.addWidget(self.right_splitter)

    def is_right_panel_docked(self) -> bool:
        """Return whether the right splitter is currently in the main spectrum layout."""

        return self.right_splitter.parent() is self.spectrum_container

    def dock_right_panel(self) -> None:
        """Restore the right splitter next to the main spectrum plot."""

        if self.right_splitter.parent() is self.spectrum_container:
            return
        self.right_splitter.setParent(None)
        self.main.top_layout.addWidget(self.right_splitter, 1)


    def add_right_widget(self, widget) -> None:
        """Insert a widget on the right side of the spectrum section."""

        if widget is None:
            return
        self.right_layout.insertWidget(0, widget)
