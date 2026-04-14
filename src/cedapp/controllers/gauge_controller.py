
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
        run_getter: Optional[Callable[[], Optional[object]]] = None,
        library_getter: Optional[Callable[[], Optional[dict]]] = None,
        get_apply_temperature_to_all: Optional[Callable[[], bool]] = None,
        get_use_fixed_pressure_solver: Optional[Callable[[], bool]] = None,
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
        self._get_run = run_getter if run_getter is not None else (lambda: None)
        self._get_library = library_getter if library_getter is not None else (lambda: None)
        self._get_apply_temperature_to_all = (
            get_apply_temperature_to_all if get_apply_temperature_to_all is not None else (lambda: False)
        )
        self._get_use_fixed_pressure_solver = (
            get_use_fixed_pressure_solver if get_use_fixed_pressure_solver is not None else (lambda: False)
        )
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

    def sync_temperature_spinbox(self, element, gauge_temp: Optional[float] = None) -> None:
        """Enable and update temperature spinbox from the selected gauge element."""

        has_thermal_eos = getattr(element, "ALPHAKT", None) is not None
        self.spinbox_t.blockSignals(True)
        self.spinbox_t.setEnabled(has_thermal_eos)
        if has_thermal_eos:
            selected_temp = gauge_temp if gauge_temp is not None else getattr(element, "T", 293)
            self.spinbox_t.setValue(float(selected_temp))
        else:
            self.spinbox_t.setValue(293)
            self.deltalambdaT = 0
        self.spinbox_t.blockSignals(False)
        self.update_pt_mode_spinbox_colors()

    def _propagate_temperature_to_library_and_run(self, temperature: float) -> None:
        """Propagate selected element temperature to the gauge library and loaded run."""

        selected = self._get_gauge()
        element_name = getattr(selected, "name", None)
        if element_name is None:
            return

        library = self._get_library() or {}
        if element_name in library:
            library[element_name].T = round(float(temperature), 3)

        if not self._get_apply_temperature_to_all():
            return

        run = self._get_run()
        if run is None:
            return

        for spec in getattr(run, "Spectra", []) or []:
            for gauge_item in getattr(spec, "Gauges", []) or []:
                ref = getattr(gauge_item, "Element_ref", None)
                if getattr(ref, "name", None) == element_name:
                    gauge_item.T = round(float(temperature), 3)
                    ref.T = round(float(temperature), 3)

    def _current_fixe_mode(self) -> str:
        return "T" if self._get_use_fixed_pressure_solver() else "P"

    def update_pt_mode_spinbox_colors(self) -> None:
        """Color spinboxes: green for free variable, red for fixed one."""

        p_is_fixed = bool(self._get_use_fixed_pressure_solver())
        free_style = "QDoubleSpinBox { background-color: #d9f7d9; }"
        fixed_style = "QDoubleSpinBox { background-color: #ffd9d9; }"
        self.spinbox_p.setStyleSheet(fixed_style if p_is_fixed else free_style)
        self.spinbox_t.setStyleSheet(free_style if p_is_fixed else fixed_style)

    def handle_spinbox_pt_changed(self, _value: Optional[float] = None) -> None:
        """Single slot handling both pressure and temperature spinbox changes."""

        if self._get_bit_modif_PTlambda():
            return
        try:
            pressure = float(self.spinbox_p.value())
            temperature = float(self.spinbox_t.value())

            gauge = self._get_gauge()
            if self._get_bit_load_jauge() and gauge is not None:
                gauge.lamb_fit = 0
                gauge = self._apply_gauge_state_update(
                    gauge,
                    pressure=pressure,
                    temperature=temperature,
                    clear_lines=True,
                )
                self._set_gauge(gauge)

            if self._get_bit_modif_jauge():
                spectrum = self._get_spectrum()
                index = self._get_index_jauge()
                if spectrum is not None and 0 <= index < len(spectrum.Gauges):
                    selected_gauge = spectrum.Gauges[index]
                    selected_gauge.lamb_fit = 0
                    selected_gauge.P = round(float(pressure), 3)
                    selected_gauge.T = round(float(temperature), 3)
                    selected_gauge.Element_ref = self._apply_gauge_state_update(
                        selected_gauge.Element_ref,
                        pressure=pressure,
                        temperature=temperature,
                    )
                    spectrum.fixe_mode = self._current_fixe_mode()

            self._propagate_temperature_to_library_and_run(temperature)
            self._set_save_value(pressure)
            self.update_pt_mode_spinbox_colors()
        except Exception:  # pragma: no cover - defensive UI feedback
            logger.exception("Erreur dans handle_spinbox_pt_changed")

    def handle_temperature_spinbox_changed(self, value: float) -> None:
        """Backward-compatible alias for temperature-only wiring."""

        self.handle_spinbox_pt_changed(value)

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


        if getattr(gauge, "ALPHAKT", None) is not None and getattr(gauge, "T", None) is not None:
            gauge.Eos_Pdhkl(value, T=gauge.T)
        else:
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

    def _apply_gauge_state_update(
        self,
        gauge,
        *,
        pressure: Optional[float] = None,
        temperature: Optional[float] = None,
        clear_lines: bool = False,
    ):
        """Apply pressure/temperature changes to *gauge* using the pressure redraw path."""

        if gauge is None:
            return None
        if temperature is not None:
            gauge.T = round(float(temperature), 3)
        if pressure is None:
            pressure = getattr(gauge, "P", getattr(gauge, "P_start", 0))
        pressure = round(float(pressure), 3)
        if hasattr(gauge, "P_start"):
            gauge.P_start = pressure
        if clear_lines:
            self.clear_lines()
        return self.f_p_move(gauge, pressure)
    
    def spinbox_p_move(self, value: float) -> None:
        """Qt slot linked to the pressure spin box."""
        self.handle_spinbox_pt_changed(value)

    # ------------------------------------------------------------------
    # Temperature handling
    # ------------------------------------------------------------------
    def spinbox_t_move(self, value: float) -> None:
        self.handle_spinbox_pt_changed(value)

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
