
"""Spectrum, gauge, and CED controllers for the DRX UI."""

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



class GaugeLibraryMixin:
    """Encapsulate routines manipulating the gauge library table."""

    # ------------------------------------------------------------------
    # Gauge table helpers
    # ------------------------------------------------------------------
    def _get_selected_gauge_from_table(self) -> Tuple[int, str]:
        """Return ``(row, name)`` for the currently selected gauge table entry."""

        if not hasattr(self, "gauge_table"):
            return -1, ""
        row = self.gauge_table.currentRow()
        if row < 0:
            return row, ""
        item = self.gauge_table.item(row, 0)
        return row, item.text() if item is not None else ""

    def _select_gauge_in_table(self, gauge_name: str) -> bool:
        """Select *gauge_name* inside the gauge table if present."""

        if not gauge_name or not hasattr(self, "gauge_table"):
            return False

        row_count = self.gauge_table.rowCount()
        for row in range(row_count):
            item = self.gauge_table.item(row, 0)
            if item and item.text() == gauge_name:
                self.gauge_table.setCurrentCell(row, 1)
                return True
        return False

    def _sync_library_state(self) -> None:
        """Refresh the combobox listing available gauges."""

        bibli = getattr(self.ClassDRX, "Bibli_elements", {}) if hasattr(self, "ClassDRX") else {}
        self.liste_type_Gauge = list(bibli.keys())
        selector = getattr(self, "Gauge_type_selector", None)
        if selector is None:
            return
        blocker = getattr(selector, "blockSignals", None)
        if callable(blocker):
            blocker(True)
        try:
            if hasattr(selector, "clear"):
                selector.clear()
            if hasattr(selector, "addItems"):
                selector.addItems(self.liste_type_Gauge)
            model = getattr(selector, "model", lambda: None)()
            if model is not None:
                for ind, gauge in enumerate(self.liste_type_Gauge):
                    item = model.item(ind)
                    if item is not None:
                        item.setBackground(QColor("orange"))
        finally:
            if callable(blocker):
                blocker(False)

    # ------------------------------------------------------------------
    # Slots manipulating the library content
    # ------------------------------------------------------------------
    def _on_gauge_status_changed(self, item) -> None:
        """React to toggles performed on the gauge table."""

        col = item.column()
        if col != 1:
            return

        row = item.row()
        gauge_name = self.gauge_table.item(row, 0).text()

        if item.checkState() == Qt.Checked:
            if gauge_name not in self.ClassDRX.Bibli_elements:
                chemin_fichier = None
                for f in self.ClassDRX.list_file:
                    if os.path.basename(f) == gauge_name:
                        chemin_fichier = f
                        break
                if chemin_fichier and os.path.isfile(chemin_fichier):
                    file = pd.read_csv(chemin_fichier, sep=":", header=None, engine="python")
                    new_element = CL.Element_Bibli(file=file, E=self.ClassDRX.E)
                    self.ClassDRX.Bibli_elements[new_element.name] = new_element
                else:
                    runtime_elements = getattr(self, "_runtime_gauge_elements", None)
                    if isinstance(runtime_elements, dict):
                        element = runtime_elements.get(gauge_name)
                        if element is not None:
                            self.ClassDRX.Bibli_elements[gauge_name] = copy.deepcopy(element)
        else:
            if gauge_name in self.ClassDRX.Bibli_elements:
                del self.ClassDRX.Bibli_elements[gauge_name]

        self._sync_library_state()

    def f_select_bib_gauge(self) -> None:
        """Open a file dialog allowing the user to import gauge files."""

        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Sélectionner un ou plusieurs fichiers",
            self.folder_bibDRX,
            "Gauge files (*.jcpds *.txt);;All Files (*)",
            options=options,
        )
        if not files:
            return

        self.folder_bib_gauge.setText(",".join([os.path.basename(f) for f in files]))

        borne = (
            list(self.calib.theta_range)
            if self.calib and getattr(self.calib, "theta_range", None)
            else list(getattr(self, "DEFAULT_THETA_RANGE", [0, 90]))
        )
        nouvelle_class = CL.DRX(folder=files, Borne=borne, E=self.get_energy_value())
        nouvelle_biblio = nouvelle_class.Bibli_elements
        nouveaux_fichiers = nouvelle_class.list_file

        if not hasattr(self, "ClassDRX") or self.ClassDRX is None:
            self.ClassDRX = CL.DRX(folder=files, Borne=borne, E=self.get_energy_value())
        else:
            for nom_gauge, valeurs in nouvelle_biblio.items():
                if nom_gauge not in self.ClassDRX.Bibli_elements:
                    self.ClassDRX.Bibli_elements[nom_gauge] = valeurs
            for f in nouveaux_fichiers:
                if f not in self.ClassDRX.list_file:
                    self.ClassDRX.list_file.append(f)

        self._sync_library_state()
        self.update_gauge_table()

    def f_dell_bib_gauge(self) -> None:
        """Remove the currently selected gauge from the library."""

        _, gauge_type = self._get_selected_gauge_from_table()
        if not gauge_type:
            return

        reply = QMessageBox.question(
            self,
            "Confirmer suppression",
            f"Supprimer la jauge '{gauge_type}' ?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        if gauge_type in self.ClassDRX.Bibli_elements:
            names = list(self.ClassDRX.Bibli_elements.keys())
            index = names.index(gauge_type)
            del self.ClassDRX.Bibli_elements[gauge_type]

            if 0 <= index < len(self.ClassDRX.list_file):
                del self.ClassDRX.list_file[index]

        self._sync_library_state()
        self.update_gauge_table()

    def f_out_bib_gauge(self) -> None:
        """Toggle the inclusion of the selected gauge in the library."""

        _, gauge_name = self._get_selected_gauge_from_table()

        if not gauge_name:
            return

        if gauge_name in self.ClassDRX.Bibli_elements:
            del self.ClassDRX.Bibli_elements[gauge_name]
        else:
            chemin_fichier = None
            for f in self.ClassDRX.list_file:
                if os.path.basename(f) == gauge_name:
                    chemin_fichier = f
                    break

            if chemin_fichier is None or not os.path.isfile(chemin_fichier):
                chemin_fichier = os.path.join(self.folder_bibDRX, gauge_name)

            element = None
            if os.path.isfile(chemin_fichier):
                file = pd.read_csv(chemin_fichier, sep=":", header=None, engine="python")
                element = CL.Element_Bibli(file=file, E=self.ClassDRX.E)
            else:
                runtime_elements = getattr(self, "_runtime_gauge_elements", None)
                if isinstance(runtime_elements, dict):
                    runtime_element = runtime_elements.get(gauge_name)
                    if runtime_element is not None:
                        element = copy.deepcopy(runtime_element)
                if element is None:
                    logger.error("Fichier introuvable : %s", chemin_fichier)
                    return

            element_name = getattr(element, "name", gauge_name)
            self.ClassDRX.Bibli_elements[element_name] = element

        self._sync_library_state()


class GaugeController:
    """Manage pressure/temperature adjustments and gauge visualisation.

    Parameters
    ----------
    spectrum_getter:
        Callable returning the currently selected ``Spectrum`` instance or
        ``None``.
    gauge_getter, gauge_setter:
        Callables used to consult and update the currently selected gauge.
    ax_spectrum:
        Plot item used to draw helper lines for the active gauge.
    layout_dhkl:
        Layout hosting the list of :class:`QCheckBox` used to toggle d-spacing
        lines.
    lamb0_entry, name_spe_entry:
        Line edits reflecting the current gauge parameters.
    spinbox_p, spinbox_t:
        Spin boxes controlling pressure and temperature.
    get_bit_modif_PTlambda, set_bit_modif_PTlambda:
        Callbacks used to consult/update the flag guarding concurrent
        modifications of pressure/temperature/lambda.
    get_bit_load_jauge, get_bit_modif_jauge:
        Callbacks returning the state of the different edition modes.
    get_index_jauge:
        Callable returning the index of the gauge being edited inside the
        current ``Spectrum``.
    set_save_value:
        Callback used to mirror the last value applied on the calling widget.
    gauge_color_getter:
        Optional callable returning the color used for helper lines associated
        with the provided gauge name.
    cl_module:
        Module (``cedapp.drx.CL_FD_Update``) exposing the helper functions to
        convert between ruby fluorescence and pressure.
    """

    def __init__(
        self,
        spectrum_getter: Callable[[], Optional[object]],
        gauge_getter: Callable[[], Optional[object]],
        gauge_setter: Callable[[Optional[object]], None],
        ax_spectrum: pg.PlotItem,
        ax_dy: pg.PlotItem,
        layout_dhkl,
        lamb0_entry,
        name_spe_entry,
        spinbox_p,
        spinbox_t,
        get_bit_modif_PTlambda: Callable[[], bool],
        set_bit_modif_PTlambda: Callable[[bool], None],
        get_bit_load_jauge: Callable[[], bool],
        get_bit_modif_jauge: Callable[[], bool],
        get_index_jauge: Callable[[], int],
        set_save_value: Callable[[float], None],
        gauge_color_getter: Optional[Callable[[str], Optional[Sequence[int]]]] = None,
        cl_module=None,
    ) -> None:
        self._get_spectrum = spectrum_getter
        self._get_gauge = gauge_getter
        self._set_gauge = gauge_setter
        self.ax_spectrum = ax_spectrum
        self.ax_dy = ax_dy
        self.layout_dhkl = layout_dhkl
        self.lamb0_entry = lamb0_entry
        self.name_spe_entry = name_spe_entry
        self.spinbox_p = spinbox_p
        self.spinbox_t = spinbox_t
        self._get_bit_modif_PTlambda = get_bit_modif_PTlambda
        self._set_bit_modif_PTlambda = set_bit_modif_PTlambda
        self._get_bit_load_jauge = get_bit_load_jauge
        self._get_bit_modif_jauge = get_bit_modif_jauge
        self._get_index_jauge = get_index_jauge
        self._set_save_value = set_save_value
        if gauge_color_getter is None:
            self._get_gauge_color = lambda _name: None
        else:
            self._get_gauge_color = gauge_color_getter
        self.CL = cl_module

        self.lines: List[Sequence[pg.GraphicsObject]] = []
        self.fixed_lines: Dict[str, List[Optional[pg.InfiniteLine]]] = {}
        self.save_var: List[bool] = []
        self.var_checkboxes: List[QCheckBox] = []
        self.deltalambdaP = 0.0
        self.deltalambdaT = 0.0

    # ------------------------------------------------------------------
    # Utilities shared by several actions
    # ------------------------------------------------------------------
    def clear_lines(self, clear_fixed: bool = False) -> None:
        """Remove the helper lines from the spectrum view.

        Parameters
        ----------
        clear_fixed:
            When ``True`` also remove the fixed reference lines associated with
            non active gauges.
        """
        while self.lines:
            line = self.lines.pop()
            if isinstance(line, (list, tuple)):
                for item in line:
                    if isinstance(item, pg.InfiniteLine):
                        self.ax_dy.removeItem(item)
                    else:
                        self.ax_spectrum.removeItem(item)
            elif isinstance(line, pg.InfiniteLine):
                self.ax_dy.removeItem(line)
            else:
                self.ax_spectrum.removeItem(line)
        if clear_fixed:
            self._clear_fixed_lines()

    def _clear_fixed_lines(self) -> None:
        """Remove helper lines corresponding to non selected gauges."""
        for lines in self.fixed_lines.values():
            for line in lines:
                if isinstance(line, pg.InfiniteLine):
                    self.ax_dy.removeItem(line)
        self.fixed_lines.clear()

    def refresh_fixed_lines(self, active_element) -> None:
        """Redraw fixed reference lines for gauges other than *active_element*."""

        self._clear_fixed_lines()

        spectrum = self._get_spectrum()
        if spectrum is None:
            return

        active_name = getattr(active_element, "name", None)
        if active_name is None and hasattr(active_element, "Element_ref"):
            active_name = getattr(active_element.Element_ref, "name", None)

        current_visibility = self.save_var if self.save_var else []

        for gauge in getattr(spectrum, "Gauges", []):
            element = getattr(gauge, "Element_ref", None)
            if element is None:
                continue
            element_name = getattr(element, "name", None)
            gauge_name = getattr(gauge, "name", element_name)
            if active_name is not None and (
                gauge_name == active_name or element_name == active_name
            ):
                continue

            save_flags = list(getattr(element, "save_var", []) or [])
            thetas = list(getattr(element, "thetas_PV", []) or [])
            if not thetas:
                continue

            stored_lines: List[Optional[pg.InfiniteLine]] = []
            for idx, theta in enumerate(thetas):
                line_item: Optional[pg.InfiniteLine] = None
                if idx < len(save_flags) and save_flags[idx]:
                    color = None
                    if gauge_name is not None:
                        color = self._get_gauge_color(gauge_name)
                    if color is None and element_name is not None:
                        color = self._get_gauge_color(element_name)
                    if color is None:
                        color = (150, 150, 150)
                    pen = pg.mkPen(color=color, width=1)
                    line_item = pg.InfiniteLine(
                        pos=theta[0],
                        angle=90,
                        pen=pen,
                    )
                    visible = True
                    if idx < len(current_visibility):
                        visible = current_visibility[idx]
                    line_item.setVisible(bool(visible))
                    self.ax_dy.addItem(line_item)
                stored_lines.append(line_item)

            if any(line is not None for line in stored_lines):
                if gauge_name is None:
                    gauge_name = str(id(gauge))
                self.fixed_lines[gauge_name] = stored_lines
    # ------------------------------------------------------------------
    # Pressure handling
    # ------------------------------------------------------------------
    def f_p_move(self, gauge, value: float):
        """Update *gauge* according to a pressure change and redraw guides."""

        gauge.P = round(value, 3)
        spectrum = self._get_spectrum()
        max_level = max(spectrum.y_corr) if spectrum else 1

        gauge.Eos_Pdhkl(value)
        self.deltalambdaP = 0

        thetas: Sequence[Sequence[float]] = gauge.thetas_PV
        while len(self.lines) > len(thetas):
            line = self.lines.pop()
            self.ax_spectrum.removeItem(line[0])
            self.ax_dy.removeItem(line[1])

        for i, peak in enumerate(thetas):
            x = peak[0]
            height = max_level * peak[1] / 100
            if i < len(self.lines):
                self.lines[i][0].setData([x, x], [0, height])
                self.lines[i][1].setValue(x)
            else:
                line_item = pg.PlotDataItem(
                    [x, x],
                    [0, height],
                    pen=pg.mkPen("b", width=1),
                    symbol=None,
                )
                line_item_dy = pg.InfiniteLine(
                    pos=x,
                    angle=90,
                    pen=pg.mkPen("b", width=1)
                    )
                self.ax_spectrum.addItem(line_item)
                self.ax_dy.addItem(line_item_dy)
                self.lines.append([line_item, line_item_dy])
        return gauge

    def spinbox_p_move(self, value: float) -> None:
        """Qt slot linked to the pressure spin box."""

        if self._get_bit_modif_PTlambda():
            return
        try:
            gauge = self._get_gauge()
            if self._get_bit_load_jauge() and gauge is not None:
                gauge.lamb_fit = 0
                gauge = self.f_p_move(gauge, value)
                self._set_gauge(gauge)
            if self._get_bit_modif_jauge():
                spectrum = self._get_spectrum()
                index = self._get_index_jauge()
                if spectrum is not None and 0 <= index < len(spectrum.Gauges):
                    gauge = spectrum.Gauges[index]
                    gauge.lamb_fit = 0
                    gauge.Element_ref = self.f_p_move(gauge.Element_ref, value)
            self._set_save_value(value)
        except Exception:  # pragma: no cover - defensive UI feedback
            logger.exception("Erreur dans spinbox_p_move")

    # ------------------------------------------------------------------
    # Temperature handling
    # ------------------------------------------------------------------
    def f_t_move(self, gauge, value: float):
        """Update *gauge* according to a temperature change and redraw guides."""

        gauge.T = round(value, 3)
        spectrum = self._get_spectrum()
        max_level = max(spectrum.y_corr) if spectrum else 1
        try:
            x_value = round(
                float(
                    inversefunc(
                        lambda x: self.CL.T_Ruby_by_P(x, P=gauge.P, lamb0R=gauge.lamb0),
                        value,
                    )
                ),
                3,
            )
            self.deltalambdaT = x_value - gauge.lamb0
        except Exception:  # pragma: no cover - keep UI usable on failure
            x_value = 0
            self.deltalambdaT = 0
        for i, delta in enumerate(gauge.deltaP0i):
            ctr = x_value + delta[0]
            line_item = self.ax_spectrum.plot(
                [ctr, ctr],
                [0, max_level * delta[1]],
                " ",
                c=gauge.color_print[0],
                marker="|",
                markersize=70,
                markeredgewidth=2,
            )[0]
            line_item_dy = pg.InfiniteLine(
                pos=ctr,
                angle=90,
                pen=pg.mkPen(gauge.color_print[0], width=1),
            )
            self.ax_dy.addItem(line_item_dy)
            self.lines.append([line_item, line_item_dy])
        self._set_bit_modif_PTlambda(False)
        return gauge

    def spinbox_t_move(self, value: float) -> None:
        """Qt slot linked to the temperature spin box."""

        if self._get_bit_modif_PTlambda():
            return
        try:
            gauge = self._get_gauge()
            if self._get_bit_load_jauge() and gauge is not None:
                conv = inversefunc(
                    lambda x: self.CL.T_Ruby_by_P(x, P=gauge.P, lamb0R=gauge.lamb0),
                    value,
                )
                gauge.lamb_fit = round(float(conv), 3)
                self.clear_lines()
                gauge = self.f_t_move(gauge, value)
                self._set_gauge(gauge)
                if self._get_spectrum() is None and gauge is not None:
                    l0 = gauge.lamb0
                    self.ax_spectrum.setXRange(l0 * 0.95, l0 * 1.1)
            if self._get_bit_modif_jauge():
                spectrum = self._get_spectrum()
                index = self._get_index_jauge()
                if spectrum is not None and 0 <= index < len(spectrum.Gauges):
                    gauge = spectrum.Gauges[index]
                    conv = inversefunc(
                        lambda x: self.CL.T_Ruby_by_P(x, P=gauge.P, lamb0R=gauge.lamb0),
                        value,
                    )
                    gauge.lamb_fit = round(float(conv), 3)
                    self.clear_lines()
                    spectrum.Gauges[index] = self.f_t_move(gauge, value)
            self._set_save_value(value)
        except Exception:  # pragma: no cover - defensive UI feedback
            logger.exception("Erreur dans spinbox_t_move")

    # ------------------------------------------------------------------
    # Loading and display of gauge d-hkl information
    # ------------------------------------------------------------------
    def clear_var_widgets(self) -> None:
        """Remove every checkbox linked to the active gauge."""

        while self.layout_dhkl.count():
            item = self.layout_dhkl.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for checkbox in self.var_checkboxes:
            checkbox.deleteLater()
        self.var_checkboxes = []
        self.save_var = []

    def f_Gauge_Load(self, gauge_select):
        """Prepare widgets and helpers for the provided gauge."""

        self.clear_var_widgets()
        QApplication.processEvents()

        if self.lamb0_entry is not None:
            self.lamb0_entry.setText("ACODER")
        if self.name_spe_entry is not None:
            self.name_spe_entry.setText("ACODER")

        self.spinbox_p.setValue(gauge_select.P_start)
        self.f_p_move(gauge_select, value=gauge_select.P_start)

        init = False
        if gauge_select.save_var is None:
            gauge_select.save_var = []
            init = True

        spectrum = self._get_spectrum()
        if spectrum is None:
            theta_min, theta_max = 0, 45
        else:
            theta_min, theta_max = spectrum.wnb[0], spectrum.wnb[-1]

        self.var_checkboxes = []
        for i, l_dhkl in enumerate(gauge_select.thetas_PV):
            checkbox = QCheckBox("d" + "".join(map(str, l_dhkl[2:])))
            if init:
                visible = theta_min < l_dhkl[0] < theta_max and l_dhkl[1] >40
                gauge_select.save_var.append(visible)
            elif len(gauge_select.save_var) - 1 < i:
                visible = False
                gauge_select.save_var.append(visible)
            else:
                visible = gauge_select.save_var[i]
            checkbox.setChecked(bool(visible))
            checkbox.stateChanged.connect(self.f_print_dhkl)
            self.var_checkboxes.append(checkbox)
            self.layout_dhkl.addWidget(checkbox)

        self.save_var = copy.deepcopy(gauge_select.save_var)
        self.refresh_fixed_lines(gauge_select)
        return gauge_select

    def f_print_dhkl(self, _state: Optional[int] = None) -> None:
        """Toggle the visibility of each helper line according to checkboxes."""

        active_gauge = self._get_gauge()
        for i, line in enumerate(self.lines):
            state = self.var_checkboxes[i].isChecked()
            if active_gauge is not None and active_gauge.save_var is not None:
                active_gauge.save_var[i] = state
            if i < len(self.save_var):
                self.save_var[i] = state
            for l in line:
                l.setVisible(state)
            for fixed in self.fixed_lines.values():
                if i < len(fixed):
                    line_item = fixed[i]
                    if isinstance(line_item, pg.InfiniteLine):
                        line_item.setVisible(state)
