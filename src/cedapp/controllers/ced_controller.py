"""Spectrum, gauge, and CED controllers for the DRX UI."""

from __future__ import annotations

import copy
import logging
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QCheckBox, QFileDialog, QMessageBox
from pynverse import inversefunc

from cedapp.drx import CL_FD_Update as CL


logger = logging.getLogger(__name__)



# ----------------------------------------------------------------------
# Gauge / dDAC controllers
# ----------------------------------------------------------------------
class DdacController:
    """Maintain dDAC-related state shared across the main window."""

    def __init__(self, host) -> None:
        self.host = host
        self._init_state()

    def _init_state(self) -> None:
        host = self.host
        init_method = getattr(host, "_init_drx_state", None)
        if callable(init_method):
            init_method()
            return
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
        host._runtime_gauge_elements = {}
        host._gauge_library_dirty = False
        host._current_spectrum_gauges = set()

