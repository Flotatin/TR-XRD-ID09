"""Widget helpers for the spectrum section."""

from __future__ import annotations

import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class SpectrumSectionWidget:
    """Builder for the main spectrum plot section."""

    def __init__(self, main) -> None:
        self.main = main
        self.spectrum_box = QGroupBox("Spectrum")
        spectra_layout = QVBoxLayout()
        self.spectrum_box.setLayout(spectra_layout)

        self.spectrum_container = QWidget()
        container_layout = QVBoxLayout(self.spectrum_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        top_layout = QVBoxLayout()
        container_layout.addLayout(top_layout)
        spectra_layout.addWidget(self.spectrum_container)

        main.graph_layout = GraphicsLayoutWidget()
        main.graph_layout.ci.layout.setRowStretchFactor(0, 4)
        main.graph_layout.ci.layout.setRowStretchFactor(1, 1)
        main.graph_layout.ci.layout.setColumnStretchFactor(0, 1)
        size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(3)
        size_policy.setVerticalStretch(1)
        main.graph_layout.setSizePolicy(size_policy)
        top_layout.addWidget(main.graph_layout, 2)

        main.ax_spectrum = main.graph_layout.addPlot(row=0, col=0)
        main.ax_spectrum.scene().sigMouseClicked.connect(main.f_cross_spectrum)

        main.ax_dy = main.graph_layout.addPlot(row=1, col=0)
        main.ax_dy.setXLink(main.ax_spectrum)

        main.graph_layout.ci.setContentsMargins(0, 0, 0, 0)
        main.graph_layout.ci.layout.setSpacing(2)

        right_container = QWidget()
        self.right_layout = QHBoxLayout(right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(2)
        top_layout.addWidget(right_container, 1)

        main.zoom_widget = pg.PlotWidget()
        main.ax_zoom = main.zoom_widget.getPlotItem()
        main.ax_zoom.hideAxis("bottom")
        main.ax_zoom.hideAxis("left")
        main.ax_zoom.scene().sigMouseClicked.connect(main.f_cross_zoom)
        self.right_layout.addWidget(main.zoom_widget)

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

        spectra_layout.addLayout(layout_check)

        self.dhkl_box = QGroupBox("dhkl")
        main.layout_dhkl = QVBoxLayout()
        self.dhkl_box.setLayout(main.layout_dhkl)

    def add_to_layout(self, grid_layout) -> None:
        """Add the spectrum widgets to the provided grid layout."""

        grid_layout.addWidget(self.spectrum_box, 0, 2, 4, 1)
        grid_layout.addWidget(self.dhkl_box, 0, 1, 3, 1)

    def add_right_widget(self, widget) -> None:
        """Insert a widget on the right side of the spectrum section."""

        if widget is None:
            return
        self.right_layout.insertWidget(0, widget)
