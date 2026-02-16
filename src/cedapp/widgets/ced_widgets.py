"""Widgets related to the CED/dDAC UI."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QSpinBox,
)


class DdacWidget:
    """Widget containing the dDAC plots and controls."""

    def __init__(self, main) -> None:
        self._drx_container = main
        self._create_drx_plots()
        self._create_drx_persistent_items()
        self._create_drx_selection_items()
        self._init_drx_state()
        self._build_drx_controls()
        self._connect_drx_events()
        self._configure_drx_layout()


    def _create_drx_plots(self) -> None:
        host = self._drx_container
        host.fig_DRX_dynamic = GraphicsLayoutWidget()
        host.ax_P = host.fig_DRX_dynamic.addPlot(row=0, col=0)#, title="Pressure (GPa)")
        host.ax_P.setLabel("left", "Pressure", units="GPa")
        host.ax_P.showAxis("right")
        host.ax_P.getAxis("right").setLabel("Piezo", units="V")
        host.ax_P.getAxis("right").setPen(pg.mkPen("b"))
        host.ax_P_piezo = pg.ViewBox()
        host.ax_P.scene().addItem(host.ax_P_piezo)
        host.ax_P.getAxis("right").linkToView(host.ax_P_piezo)
        host.ax_P_piezo.setXLink(host.ax_P)
        host.ax_P_piezo.enableAutoRange(axis="y", enable=True)
        host.ax_P.vb.sigResized.connect(host._update_piezo_view_geometry)
        host._update_piezo_view_geometry()

        host.ax_dPdt = host.fig_DRX_dynamic.addPlot(row=1, col=0)
        host.ax_dPdt.setLabel("left", "dP/dt ", units="GPa/ms")
        host.ax_dPdt.addLegend()
        host.ax_diff_int = host.fig_DRX_dynamic.addPlot(row=2, col=0)
        host.ax_diff_int.setLabel("left", '2<font>&theta<font>', units="°")
        host.ax_diff_int.setLabel("bottom", "Time", units="ms")
        #host.fig_DRX_dynamic.ci.layout.setRowStretchFactor(0, 2)
        #host.fig_DRX_dynamic.ci.layout.setRowStretchFactor(1, 1)
        #host.fig_DRX_dynamic.ci.layout.setRowStretchFactor(2, 2)

    def _create_drx_persistent_items(self) -> None:
        host = self._drx_container
        host.img_diff_int_item = pg.ImageItem(np.zeros((1, 1), dtype=float))
        host.ax_diff_int.addItem(host.img_diff_int_item)

        host.line_P = pg.InfiniteLine(angle=90, movable=False, pen="r")
        host.line_dPdt = pg.InfiniteLine(angle=90, movable=False, pen="r")
        host.line_theta_diff = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("g", width=2))
        host.ax_P.addItem(host.line_P)
        host.ax_dPdt.addItem(host.line_dPdt)
        host.ax_diff_int.addItem(host.line_theta_diff)

    def _create_drx_selection_items(self) -> None:
        host = self._drx_container
        host.zone_diff_int = pg.LinearRegionItem(
            values=[0, 0],
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(255, 0, 0, 40),
            movable=False,
        )
        host.ax_diff_int.addItem(host.zone_diff_int)

        host.zone_multi_P = pg.LinearRegionItem(
            values=[0, 0],
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(0, 120, 255, 40),
            movable=True,
        )
        host.ax_P.addItem(host.zone_multi_P)
        host.zone_multi_dPdt = pg.LinearRegionItem(
            values=[0, 0],
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(0, 120, 255, 40),
            movable=True,
        )
        host.ax_dPdt.addItem(host.zone_multi_dPdt)
        host.zone_multi_diff_int = pg.LinearRegionItem(
            values=[0, 0],
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(0, 120, 255, 40),
            movable=True,
        )
        host.ax_diff_int.addItem(host.zone_multi_diff_int)
        host.zone_multi_P.setVisible(False)
        host.zone_multi_dPdt.setVisible(False)
        host.zone_multi_diff_int.setVisible(False)

    def _init_drx_state(self) -> None:
        host = self._drx_container
        host.list_index_file_CED_Load = []
        host.Pstart, host.Pend, host.tstart, host.tend, host.bit_dP = 2, 1, 2, 1, 0
        host.x1 = host.y1 = host.x3 = host.y3 = host.x5 = host.y5 = host.x_clic = host.y_clic = 0
        host.time = []
        host.spectre_number = []
        host.plot_P = []
        host.plot_piezo = None
        host.plot_spe = []
        host._cedx_gauge_items = {}
        host._cedx_gauge_meta = {}
        host._cedx_symbol_index = 0
        host._cedx_mean_curve_item = None
        host._cedx_mean_curve_data = (np.asarray([]), np.asarray([]))
        host._cedx_gauge_series = {}
        host._cedx_image_cache = None
        host._cedx_image_row_map = {}
        host._cedx_spectrum_theta_range = {}
        host._cedx_image_trimmed_length = 0
        host._cedx_theta_bounds = None
        host._cedx_levels = None
        host._cedx_interval_dpdt_text_items = {}
        host._runtime_gauge_elements = {}
        host._gauge_library_dirty = False
        host._current_spectrum_gauges = set()
        host._drx_in_refresh = False

    def _build_drx_controls(self) -> None:
        host = self._drx_container
        host.label_CED = QLabel("CEDd file select:", self._drx_container )
        host.label_CED.setFont(QFont("Arial", 6))
        host.label_CED.setMaximumHeight(100)

        host.spectrum_select_box = QCheckBox("clic spectrum (h)", self._drx_container )
        host.spectrum_select_box.setChecked(True)
        
        host.analysis_toggle = None

        host.label_dpdt_points = QLabel("Pts glissants dP/dt :", self._drx_container)
        host.spinbox_dpdt_points = QSpinBox(self._drx_container)
        host.spinbox_dpdt_points.setRange(1, 100)
        host.spinbox_dpdt_points.setSingleStep(1)
        host.spinbox_dpdt_points.setValue(2)
        host.spinbox_dpdt_points.setToolTip("n_points-+ dP/dt ")
        
        host.label_dpdt_smooth = QLabel("smooth dP/dt :", self._drx_container)
        host.spinbox_dpdt_smooth = QSpinBox(self._drx_container)
        host.spinbox_dpdt_smooth.setRange(0, 50)
        host.spinbox_dpdt_smooth.setSingleStep(1)
        host.spinbox_dpdt_smooth.setValue(1)
        host.spinbox_dpdt_smooth.setToolTip(
            "UnivariateSpline(time,pressures,s=smooth) pour lisser la courbe moyenne dP/dt."
        )

        host.btn_zone_dpdt = QPushButton("Zone dP/dt", self._drx_container)
        host.btn_zone_dpdt.setCheckable(True)
        host.btn_zone_dpdt.setChecked(False)

        host.label_interval_dpdt = QLabel("dP/dt zone: —", self._drx_container)
        host.label_interval_dpdt.setFont(QFont("Arial", 7))


    def _connect_drx_events(self) -> None:
        host = self._drx_container
        host.ax_P.scene().sigMouseClicked.connect(
            lambda evt: host._on_drx_plot_clicked(evt, axis_role="pressure")
        )
        host.ax_dPdt.scene().sigMouseClicked.connect(
            lambda evt: host._on_drx_plot_clicked(evt, axis_role="derivative")
        )
        host.ax_diff_int.scene().sigMouseClicked.connect(
            lambda evt: host._on_drx_plot_clicked(evt, axis_role="image")
        )
        host.spinbox_dpdt_points.valueChanged.connect(host._on_dpdt_points_changed)
        host.spinbox_dpdt_smooth.valueChanged.connect(host._on_dpdt_points_changed)
        if hasattr(host, "set_ddac_multi_zone_visibility"):
            host.btn_zone_dpdt.toggled.connect(host.set_ddac_multi_zone_visibility)


    def _configure_drx_layout(self) -> None:
        host = self._drx_container
        host.fig_DRX_dynamic.ci.setContentsMargins(0, 0, 0, 0)
        host.fig_DRX_dynamic.ci.layout.setSpacing(4)

        if host is None:
            return

        host.ddac_box = QGroupBox("dDAC")
        ddac_layout = QVBoxLayout()
        host.ddac_box.setLayout(ddac_layout)

        
        ddac_layout.addWidget(host.fig_DRX_dynamic)

        layhrun = QHBoxLayout()
        layhrun.addWidget(host.label_CED)
        layhrun.addWidget(host.spectrum_select_box)
        layhrun.addWidget(host.label_dpdt_points)
        layhrun.addWidget(host.spinbox_dpdt_points)
        layhrun.addWidget(host.label_dpdt_smooth)
        layhrun.addWidget(host.spinbox_dpdt_smooth)
        layhrun.addWidget(host.btn_zone_dpdt)
        layhrun.addWidget(host.label_interval_dpdt)
        if getattr(host, "analysis_toggle", None) is not None:
            layhrun.addWidget(host.analysis_toggle)
        ddac_layout.addLayout(layhrun)
        self._controls_layout = layhrun

        host.setLayout(ddac_layout)

    def add_to_layout(self, grid_layout) -> None:
        """Add the ddac widgets to the provided grid layout."""
        grid_layout.addWidget(self._drx_container.ddac_box, 0, 3, 4, 2)

    def add_control_widget(self, widget) -> None:
        """Append a control widget to the ddac control row."""
        if widget is None:
            return
        layout = getattr(self, "_controls_layout", None)
        if layout is None:
            return
        layout.addWidget(widget)
