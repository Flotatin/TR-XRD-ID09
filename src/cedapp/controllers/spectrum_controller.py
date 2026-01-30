
from __future__ import annotations

import copy
import logging
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QCheckBox, QFileDialog, QMessageBox
from pynverse import inversefunc

from cedapp.drx import CL_FD_Update as CL


logger = logging.getLogger(__name__)
# ----------------------------------------------------------------------
# Spectrum controllers
# ----------------------------------------------------------------------


class SpectrumController:
    """Orchestrates spectrum related interactions.

    The controller centralises the management of interactive regions drawn on
    the main spectrum view together with higher level treatments such as batch
    fitting.  All GUI items required to perform these actions (plots, buttons)
    and the data sources (``Spectrum`` objects as well as the ``RUN`` holder)
    are injected during construction so the controller stays independent from
    the :class:`~PyQt5.QtWidgets.QMainWindow` subclass.

    Parameters
    ----------
    spectrum_getter:
        Callable returning the currently displayed ``Spectrum`` instance or
        ``None`` when no spectrum is loaded.
    run_getter:
        Callable returning the current ``RUN`` data object used for batch
        processing of spectra.
    ax_spectrum:
        :class:`pyqtgraph.PlotItem` used to display the main spectrum where
        interactive regions live.
    remove_button:
        Button toggled when regions are selected/removed.  The controller only
        toggles its enabled state; the widget remains owned by the main window.
    """

    def __init__(
        self,
        spectrum_getter: Callable[[], Optional[object]],
        run_getter: Callable[[], Optional[object]],
        ax_spectrum: pg.PlotItem,
        remove_button,
    ) -> None:
        self._get_spectrum = spectrum_getter
        self._get_run = run_getter
        self.ax_spectrum = ax_spectrum
        self.remove_button = remove_button

        self.zones: List[pg.LinearRegionItem] = []
        self.zone_selectionnee: Optional[pg.LinearRegionItem] = None
        self.theta2_range: List[Sequence[float]] = []

    # ------------------------------------------------------------------
    # Region handling
    # ------------------------------------------------------------------
    def toggle_regions_visibility(self,visible: bool) -> None:
        """Toggle the visibility of every registered region."""

        for region in self.zones:
            spectrum = self._get_spectrum()
            max_level = max(spectrum.y_corr) if spectrum else 1
            region.setZValue(max_level)
            region.setVisible(visible)
            
    def set_regions_visible(self, visible: bool) -> None:
        """Show or hide every registered region without altering their ranges."""

        for region in self.zones:
            spectrum = self._get_spectrum()
            max_level = max(spectrum.y_corr) if spectrum else 1
            region.setZValue(max_level)
            region.setVisible(visible)

    def add_zone(self, region: Optional[pg.LinearRegionItem] = None) -> pg.LinearRegionItem:
        """Register a new region on the spectrum plot.
        When *region* is ``None`` a new :class:`LinearRegionItem` spanning the
        current spectrum range is created.
        """

        spectrum = self._get_spectrum()
        if region is None:
            if spectrum is not None and getattr(spectrum, "wnb", None) is not None:
                x_values = spectrum.wnb
                start, end = float(x_values[0]), float(x_values[-1])
            else:
                start, end = 0.0, 90.0
            region = pg.LinearRegionItem([start, end])
        max_level = max(spectrum.y_corr) if spectrum else 1
        region.setZValue(max_level)
        region.mouseClickEvent = lambda ev, reg=region: self.select_zone(reg)
        region.sigRegionChangeFinished.connect(self.update_theta2_range)
        self.ax_spectrum.addItem(region)
        self.zones.append(region)
        self.update_theta2_range()
        return region

    def select_zone(self, region: pg.LinearRegionItem) -> None:
        """Visually mark *region* as the selected one."""

        if self.zone_selectionnee is not None:
            self.zone_selectionnee.setBrush(None)
        self.zone_selectionnee = region
        region.setBrush((255, 0, 0, 50))
        if self.remove_button is not None:
            self.remove_button.setEnabled(True)

    def remove_selected_zone(self) -> None:
        """Remove the currently selected region from the plot."""

        if self.zone_selectionnee is None:
            return
        self.ax_spectrum.removeItem(self.zone_selectionnee)
        if self.zone_selectionnee in self.zones:
            self.zones.remove(self.zone_selectionnee)
        self.zone_selectionnee = None
        if self.remove_button is not None:
            self.remove_button.setEnabled(False)
        self.update_theta2_range()

    def update_theta2_range(self) -> List[Sequence[float]]:
        """Recompute the active angular ranges from the regions."""

        if self.zones:
            self.theta2_range = [tuple(map(float, region.getRegion())) for region in self.zones]
        else:
            spectrum = self._get_spectrum()
            if spectrum is not None and getattr(spectrum, "wnb", None) is not None:
                self.theta2_range = [(float(spectrum.wnb[0]), float(spectrum.wnb[-1]))]
            else:
                self.theta2_range = [(0.0, 90.0)]
        return self.theta2_range

    # ------------------------------------------------------------------
    # Batch fit helpers
    # ------------------------------------------------------------------
    def run_fit_selected_spectra(
        self,
        index_start: int,
        index_stop: int,
        ngen: float,
        mutpb: float,
        cxpb: float,
        popinit: float,
        pressure_range: Sequence[float],
        max_ecart_pressure: float,
        max_elements: int,
        tolerance: float,
        custom_peak_params: Optional[dict] = None,
    ) -> bool:
        """Delegate the fit of multiple spectra to the active ``RUN`` object.

        Returns ``True`` when a run object was available and the request was
        forwarded, ``False`` otherwise.
        """

        run = self._get_run()
        if run is None:
            return False
        run.fit_selected_spectra(
            index_start=index_start,
            index_stop=index_stop,
            NGEN=ngen,
            MUTPB=mutpb,
            CXPB=cxpb,
            POPINIT=popinit,
            pressure_range=list(pressure_range),
            max_ecart_pressure=max_ecart_pressure,
            max_elements=max_elements,
            tolerance=tolerance,
            custom_peak_params=custom_peak_params or {},
        )
        return True

    # ------------------------------------------------------------------
    # Utility API for configuration reloads
    # ------------------------------------------------------------------
    def extend_with_regions(self, regions: Iterable[pg.LinearRegionItem]) -> None:
        """Register pre-created *regions* (used when reloading settings)."""

        for region in regions:
            self.add_zone(region)
