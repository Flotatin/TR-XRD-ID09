# BANC CED Dynamic 
from collections import defaultdict
import logging
import os
import sys
import dill
import traceback
import json
import numpy as np
import ast
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from PyQt5.QtWidgets import (QApplication,
                             QSizePolicy,
                             QMainWindow,
                             QLabel,
                             QPushButton,
                             QFileDialog,
                             QGridLayout,
                             QWidget,
                             QTableWidget,
                             QComboBox,
                             QVBoxLayout,
                             QHBoxLayout,
                             QTableWidgetItem,
                             QDoubleSpinBox,
                             QGroupBox,
                             QTableView,
                             QStyledItemDelegate,
                             QTabWidget,
                             QLineEdit,
                             QMessageBox,
                             QCheckBox,
                             QListWidget,
                             QTextEdit,
                             QSpinBox,
                             QDialog,
                             QDesktopWidget,
                             QToolButton,
                             QAbstractItemView,
                             )

try:
    from PyQt5.QtWidgets import QProgressBar  # type: ignore
except ImportError:  # pragma: no cover - fallback for headless tests
    QProgressBar = None  # type: ignore


from PyQt5 import QtWidgets
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex, QTimer, pyqtSignal

from PyQt5.QtGui import QColor, QFont

if hasattr(QtWidgets, "QTabWidget"):
    QTabWidget = QtWidgets.QTabWidget
else:  # pragma: no cover - fallback for test stubs lacking QTabWidget
    class QTabWidget(QWidget):
        """Minimal fallback tab widget used when Qt is not fully available."""

        West = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tabs = []
            self._current_index = -1

        def setTabPosition(self, *args, **kwargs):
            pass

        def addTab(self, widget, label):
            self._tabs.append((widget, label))
            if self._current_index == -1:
                self._current_index = 0
            return len(self._tabs) - 1

        def setCurrentWidget(self, widget):
            index = self.indexOf(widget)
            if index >= 0:
                self._current_index = index

        def indexOf(self, widget):
            for idx, (tab_widget, _label) in enumerate(self._tabs):
                if tab_widget is widget:
                    return idx
            return -1

        def removeTab(self, index):
            if 0 <= index < len(self._tabs):
                self._tabs.pop(index)
                if not self._tabs:
                    self._current_index = -1
                elif self._current_index >= len(self._tabs):
                    self._current_index = len(self._tabs) - 1

        def count(self):
            return len(self._tabs)

        def setCurrentIndex(self, index):
            if 0 <= index < len(self._tabs):
                self._current_index = index

        def setTabVisible(self, index, visible):
            # Visibility has no effect in the simplified fallback.
            pass

from datetime import datetime
from pyqtgraph import PlotItem
import pyqtgraph as pg
import copy
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import io
from math import isnan
import pandas as pd
import matplotlib.colors as mcolors
from pynverse import inversefunc
import re



import pyFAI
import fabio
import warnings

from cedapp.ui import qt_styles, ui_sections
from cedapp.utils import paths
from cedapp.utils.logging_config import setup_logging


import silx

from silx.io.h5py_utils import File

claire = True

style = qt_styles.qt_stylesheet(light=claire)
qt_styles.configure_pyqtgraph(light=claire)

Setup_mode = True

DEFAULT_THETA_RANGE = [8, 25]
DEFAULT_ENERGY = 19000

_DEBUG_ENV = os.getenv("DRX_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
setup_logging(debug=_DEBUG_ENV)
logger = logging.getLogger(__name__)

folder_start = ""
file_config = r'config_H2O.txt'#paths.get_default_config_path()
text_dir = paths.get_text_dir(require=False)

file_help = text_dir / "Help.txt"
file_command = text_dir / "Command.txt"
file_variables = text_dir / "Variables.txt"

folder_Bib_DRX = paths.get_bibdrx_dir(require=False)

from cedapp.drx import Calibration

from cedapp.drx import CL_FD_Update as CL
from cedapp.drx.pic import Pics
from cedapp.drx.gauge import Gauge, Element

from cedapp.drx import Oscilloscope_LeCroy_QTinterface as Oscilo

from cedapp.controllers import (
    ConfigurationMixin,
    DdacController,
    FileSelectionController,
    GaugeController,
    GaugeLibraryMixin,
    SpectrumController,
)

from cedapp.widgets import (
    JcpdsEditor,
    KeyboardWindow,
    SettingsDialog,
    load_help_file,
    ProgressDialog
)

UI_STATE_FIELDS = set(ui_sections.UIState.__dataclass_fields__.keys())

_RELEVANT_MODIFIERS = Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier


def _method_action(method_name: str) -> Callable[["MainWindow"], None]:
    """Return a callable invoking ``method_name`` on ``self``."""

    def _caller(self: "MainWindow") -> None:
        getattr(self, method_name)()

    return _caller


def _toggle_checkbox_action(attribute: str) -> Callable[["MainWindow"], None]:
    """Return a callable toggling the checkbox stored in ``attribute``."""

    def _caller(self: "MainWindow") -> None:
        checkbox = getattr(self, attribute)
        checkbox.setChecked(not checkbox.isChecked())

    return _caller


def _direct_action(handler: Callable[["MainWindow"], None]) -> Dict[str, object]:
    """Helper used for shortcuts that should exit early from the handler."""

    return {"handler": handler, "name": None, "requires_box": False, "direct_return": True}


def _setattr_and_call(method_name: str, attribute: str, value) -> Callable[["MainWindow"], None]:
    """Return a callable setting ``attribute`` before invoking ``method_name``."""

    def _caller(self: "MainWindow") -> None:
        setattr(self, attribute, value)
        getattr(self, method_name)()

    return _caller


KEYBOARD_SHORTCUTS: Dict[Tuple[int, int], Dict[str, object]] = {
    (Qt.Key_Return, Qt.ControlModifier): _direct_action(lambda self: self.execute_code()),
    (Qt.Key_C, Qt.ControlModifier): _direct_action(_method_action("copy_gauge_models_to_clipboard")),
    (Qt.Key_R, Qt.ShiftModifier): {
        "handler": _method_action("Replace_pic_fit"),
        "name": "Replace pic",
        "requires_box": False,
        "allow_extra_modifiers": True,
    },
    (Qt.Key_R, Qt.NoModifier): {
        "handler": _method_action("Replace_pic"),
        "name": "Replace pic",
        "requires_box": False,
    },
    (Qt.Key_U, Qt.ShiftModifier): {
        "handler": _method_action("Undo_pic"),
        "name": "Undo pic",
        "requires_box": False,
        "allow_extra_modifiers": True,
    },
    (Qt.Key_U, Qt.NoModifier): {
        "handler": _method_action("Undo_pic_select"),
        "name": "Undo pic",
        "requires_box": False,
    },
    (Qt.Key_C, Qt.NoModifier): {
        "handler": _method_action("Click_Confirme"),
        "name": "Confirm pic",
        "requires_box": False,
    },
    (Qt.Key_Y, Qt.NoModifier): {
        "handler": _method_action("Auto_pic"),
        "name": "Auto pic",
        "requires_box": False,
    },
    (Qt.Key_I, Qt.NoModifier): {
        "handler": _method_action("f_out_bib_gauge"),
        "name": "Output bib gauge",
        "requires_box": False,
    },
    (Qt.Key_S, Qt.ShiftModifier): {
        "handler": _method_action("CREAT_new_Spectrum"),
        "name": "New Spectrum",
        "requires_box": True,
        "allow_extra_modifiers": True,
    },
    (Qt.Key_F, Qt.ShiftModifier): {
        "handler": _method_action("FIT_lmfitVScurvfit"),
        "name": "Fit total",
        "requires_box": True,
        "allow_extra_modifiers": True,
    },
    (Qt.Key_B, Qt.ShiftModifier): {
        "handler": _method_action("Baseline_spectrum"),
        "name": "Baseline",
        "requires_box": False,
    },
    (Qt.Key_X, Qt.NoModifier): _direct_action(
        lambda self: self.Spectrum.Calcul_study(mini=True)
    ),
    (Qt.Key_A, Qt.ShiftModifier): {
        "handler": _method_action("f_Gauge_Add_in_Spectrum"),
        "name": "add Gauge",
        "requires_box": False,
        "allow_extra_modifiers": True,
    },
    (Qt.Key_D, Qt.ShiftModifier): {
        "handler": _method_action("f_dell_bib_gauge"),
        "name": "Delete last pic",
        "requires_box": False,
        "allow_extra_modifiers": True,
    },
    (Qt.Key_D, Qt.NoModifier): {
        "handler": _method_action("Dell_Jauge"),
        "name": "dell Gauge",
        "requires_box": False,
    },
    (Qt.Key_E, Qt.ShiftModifier): {
        "handler": _method_action("CREAT_empty_CEDd_from_loaded_files"),
        "name": "New CEDd",
        "requires_box": True,
        "allow_extra_modifiers": True,
    },
    (Qt.Key_F5, Qt.NoModifier): {
        "handler": _setattr_and_call("CLEAR_CEDd", "bit_bypass", True),
        "name": "Clear CEDd",
        "requires_box": True,
    },
    (Qt.Key_F3, Qt.NoModifier): {
        "handler": _method_action("SAVE_CEDd"),
        "name": "Save CEDd",
        "requires_box": True,
    },
    (Qt.Key_F4, Qt.NoModifier): {
        "handler": _method_action("REFRESH"),
        "name": "Refresh data CEDd",
        "requires_box": True,
    },
    (Qt.Key_T, Qt.ShiftModifier): {
        "handler": _method_action("CEDX_Fit_all"),
        "name": "CEDX_Fit_all",
        "requires_box": True,
        "allow_extra_modifiers": True,
    },
    (Qt.Key_L, Qt.ShiftModifier): {
        **_direct_action(lambda self: self._open_oscilloscope_viewer()),
        "allow_extra_modifiers": True,
    },
    (Qt.Key_P, Qt.ShiftModifier): {
        "handler": _method_action("toggle_colonne"),
        "name": "Toggle column",
        "requires_box": False,
        "allow_extra_modifiers": True,
    },
    (Qt.Key_M, Qt.NoModifier): {
        "handler": _method_action("try_find_peak"),
        "name": "Try find peak",
        "requires_box": False,
    },
    (Qt.Key_Q, Qt.NoModifier): _direct_action(_toggle_checkbox_action("select_clic_box")),
    (Qt.Key_H, Qt.NoModifier): _direct_action(_toggle_checkbox_action("spectrum_select_box")),
    (Qt.Key_F2, Qt.NoModifier): _direct_action(_method_action("afficher_clavier_utilise")),
    (Qt.Key_S, Qt.ControlModifier): _direct_action(_method_action("show_gauge_selection_zone")),
    (Qt.Key_V, Qt.ControlModifier): _direct_action(_method_action("paste_gauge_models_from_clipboard")),
    (Qt.Key_Delete, Qt.NoModifier): _direct_action(_method_action("delete_gauges_in_zone")),
    (Qt.Key_Escape, Qt.NoModifier): _direct_action(_method_action("hide_gauge_selection_zone")),
    (Qt.Key_Z, Qt.NoModifier): {
        "handler": _method_action("f_print_region"),
        "name": "f_print_region",
        "requires_box": False,
    },
}




class MainWindow(ConfigurationMixin, GaugeLibraryMixin, QMainWindow):
    def __init__(self, folder_start="", config_file=file_config):
        super().__init__()
        self.ui_state = ui_sections.UIState()
        self.Spectrum=None
        self.zone_spectrum=[0,90]
        self.gauge_select: Optional[Element] = None
        self.spectro_data: Optional[pd.DataFrame] = None
        self.spec_fit: List[Optional[object]] = []
        self.RUN=None
        self._gauge_clipboard: Optional[List[Gauge]] = None
        self._gauge_clipboard_zone: Optional[pg.LinearRegionItem] = None
        self.config_file = paths.resolve_config_path(config_file).expanduser()
        self.folder_start = folder_start or ""
        self.DEFAULT_THETA_RANGE = DEFAULT_THETA_RANGE
        self.DEFAULT_ENERGY = DEFAULT_ENERGY
        self.energy_value = float(self.DEFAULT_ENERGY)
        self.file_command = str(file_command)
        self.file_help = str(file_help)
        self.file_variables = str(file_variables)
        
        self.file_controller = FileSelectionController(self)
        self.loaded_drx_scan_name = ""
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(100)
        self.live_timer.timeout.connect(self._check_live_folder)
        self.live_mode_active = False
        self.jungfrau_mode="burst"
#########################################################################################################################################################################################
#? Setup Main window parameters

        self._configure_window()
        self._build_command_panel()

#? Param treatement section

        self._build_tools_panel()

#? text box section
        self._build_message_label()

#? PAram FIT section
        self._build_model_peak_section()

#? Plot Spectrum section section
        self._init_plot_widgets()

#? Setup file loading section

        self._build_file_section()

#? Interface dDAC
        self._build_ddac_section()

#? Gauge information
        self._build_gauge_section()
        self._init_controllers()

#?Mise en fomre

        self._finalize_layout()

#?Mise en fomre
        
#########################################################################################################################################################################################
#?Mise en fomre
        self.viewer=None
        self.clavier_visuel = None


        self.folder_bibDRX = str(folder_Bib_DRX)  # os.path.join(folder_Bib_DRX,os.listdir(folder_Bib_DRX)[0])
        self.calib=None
        self.ClassDRX=CL.DRX(folder=None,Borne=DEFAULT_THETA_RANGE,E=DEFAULT_ENERGY)
        self.list_file_bibli_drx=self.ClassDRX.list_file
        self.set_energy_value(self.ClassDRX.E)
        

        self.load_paths_from_txt()
        self.f_change_file_type()
        self._sync_library_state()
        self._warn_missing_resources()

    def _read_setup_config(self, config_path: Path, repo_root: Path) -> Dict[str, object]:
        if not config_path.exists():
            raise ValueError(f"Setup config not found: {config_path}")

        config_entries: Dict[str, object] = {}
        bib_files: List[str] = []

        with config_path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "bib_files":
                    try:
                        bib_files = json.loads(value)
                    except Exception as exc:
                        raise ValueError("Invalid bib_files entry in setup config.") from exc
                    continue
                if key == "zone":
                    try:
                        config_entries[key] = ast.literal_eval(value)
                    except Exception as exc:
                        raise ValueError("Invalid zone entry in setup config.") from exc
                    continue
                config_entries[key] = value

        if bib_files:
            config_entries["bib_files"] = bib_files

        required_fields = (
            "folder_DRX",
            "folder_OSC",
            "folder_CED",
            "loaded_file_DRX",
            "loaded_file_OSC",
            "calib_file_mask",
            "calib_file_poni",
            "energie_DRX",
            "bib_files",
        )

        missing = []
        for field in required_fields:
            value = config_entries.get(field)
            if value is None or (isinstance(value, str) and not self._sanitize_config_value(value)):
                missing.append(field)
        if missing:
            raise ValueError(f"Missing setup config fields: {', '.join(missing)}")

        def resolve_path(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate if candidate.is_absolute() else repo_root / candidate

        resolved = {
            "folder_DRX": resolve_path(self._sanitize_config_value(config_entries["folder_DRX"])),
            "folder_OSC": resolve_path(self._sanitize_config_value(config_entries["folder_OSC"])),
            "folder_CED": resolve_path(self._sanitize_config_value(config_entries["folder_CED"])),
            "loaded_file_DRX": resolve_path(self._sanitize_config_value(config_entries["loaded_file_DRX"])),
            "loaded_file_OSC": resolve_path(self._sanitize_config_value(config_entries["loaded_file_OSC"])),
            "calib_file_mask": resolve_path(self._sanitize_config_value(config_entries["calib_file_mask"])),
            "calib_file_poni": resolve_path(self._sanitize_config_value(config_entries["calib_file_poni"])),
            "energie_DRX": float(self._sanitize_config_value(config_entries["energie_DRX"])),
            "bib_files": [
                resolve_path(self._sanitize_config_value(entry))
                for entry in config_entries.get("bib_files", [])
                if self._sanitize_config_value(entry)
            ],
            "zone": config_entries.get("zone", []),
        }

        file_fields = (
            "loaded_file_DRX",
            "loaded_file_OSC",
            "calib_file_mask",
            "calib_file_poni",
        )
        for field in file_fields:
            path = resolved[field]
            if not isinstance(path, Path) or not path.exists():
                raise ValueError(f"Setup config path not found for {field}: {path}")

        if not resolved["bib_files"]:
            raise ValueError("Setup config bib_files entry is empty.")

        for bib_path in resolved["bib_files"]:
            if not bib_path.exists():
                raise ValueError(f"Setup config bib file not found: {bib_path}")

        return resolved

    def _add_setup_bib_elements(self, bib_files: List[Path]) -> List[str]:
        theta_range = (
            list(self.calib.theta_range)
            if self.calib and getattr(self.calib, "theta_range", None)
            else list(DEFAULT_THETA_RANGE)
        )
        energy = self.get_energy_value()
        library_source = CL.DRX(folder=[str(path) for path in bib_files], Borne=theta_range, E=energy)
        added_names: List[str] = []

        if getattr(self, "ClassDRX", None) is None:
            self.ClassDRX = CL.DRX(folder=None, Borne=theta_range, E=energy)

        for name, element in library_source.Bibli_elements.items():
            if name not in self.ClassDRX.Bibli_elements:
                self.ClassDRX.Bibli_elements[name] = element
                added_names.append(name)

        for file_path in library_source.list_file:
            if file_path not in self.ClassDRX.list_file:
                self.ClassDRX.list_file.append(file_path)

        self.liste_type_Gauge = list(self.ClassDRX.Bibli_elements.keys())
        self._sync_library_state()
        self.update_gauge_table()
        return added_names

    def _run_setup_mode(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "TEST" / "config_TEST.txt"
        config = self._read_setup_config(config_path, repo_root)

        self.config_file = config_path
        self.dict_folders["DRX"] = str(config["folder_DRX"])
        self.dict_folders["Oscilloscope"] = str(config["folder_OSC"])
        self.dict_folders["CED"] = str(config["folder_CED"])
        self.loaded_file_DRX = str(config["loaded_file_DRX"])
        self.loaded_file_OSC = str(config["loaded_file_OSC"])

        self.file_label_spectro.setText(
            f"DRX: {os.path.basename(self.loaded_file_DRX) if self.loaded_file_DRX else 'None'}"
        )
        self.file_label_oscilo.setText(
            f"Oscillo: {os.path.basename(self.loaded_file_OSC) if self.loaded_file_OSC else 'None'}"
        )

        self.calib = Calibration.Calib_DRX(
            file_mask=str(config["calib_file_mask"]),
            file_poni=str(config["calib_file_poni"]),
            theta_range=list(getattr(self, "DEFAULT_THETA_RANGE", DEFAULT_THETA_RANGE)),
            energy=float(config["energie_DRX"]),
        )

        self.set_energy_value(float(config["energie_DRX"]))
        added_gauges = self._add_setup_bib_elements(config["bib_files"])

        self.CREAT_empty_CEDd_from_loaded_files()

        gauge_name = added_gauges[0] if added_gauges else None
        if gauge_name is None and self.liste_type_Gauge:
            gauge_name = self.liste_type_Gauge[0]
        if gauge_name is None:
            raise ValueError("Setup mode could not determine a gauge to add.")

        self.f_gauge_select(gauge_name)
        self.f_Gauge_Add_in_Spectrum()

        if hasattr(self, "index_start_entry") and hasattr(self, "index_stop_entry"):
            self.index_start_entry.setValue(0)
            self.index_stop_entry.setValue(10)

        self._CEDX_auto_compo()
        self.REFRESH()

    def _warn_missing_resources(self) -> None:
        missing_messages = []
        if not self.config_file.exists():
            missing_messages.append(f"Configuration file not found: {self.config_file}")
        if not Path(self.file_help).exists():
            missing_messages.append(f"Help file not found: {self.file_help}")
        if not Path(self.file_command).exists():
            missing_messages.append(f"Command file not found: {self.file_command}")
        if not Path(self.file_variables).exists():
            missing_messages.append(f"Variables file not found: {self.file_variables}")
        if not Path(self.folder_bibDRX).exists():
            missing_messages.append(f"BibDRX directory not found: {self.folder_bibDRX}")
        if missing_messages:
            QMessageBox.warning(self, "Missing resources", "\n".join(missing_messages))

    def __getattr__(self, name: str):
        ui_state = self.__dict__.get("ui_state")
        if ui_state is not None and name in UI_STATE_FIELDS:
            return getattr(ui_state, name)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        if name != "ui_state" and "ui_state" in self.__dict__ and name in UI_STATE_FIELDS:
            setattr(self.__dict__["ui_state"], name, value)
            return
        super().__setattr__(name, value)




#########################################################################################################################################################################################
#? CALVIER COMMANDE CONTROLE

    def _configure_window(self) -> None:
        """Configure the base window layout."""

        self.setWindowTitle("0.1 vXRD © F.Dembele")
        self.grid_layout = QGridLayout()
        central_widget = QWidget()
        central_widget.setLayout(self.grid_layout)
        self.setCentralWidget(central_widget)

    def _build_command_panel(self) -> None:
        """Create the settings panel together with the help/command widgets."""
        ui_sections.build_command_panel(self)

    def _build_file_section(self) -> None:
        """Create the file loading controls and python tooling panel."""
        ui_sections.build_file_section(self)

    def _build_tools_panel(self) -> None:
        """Create widgets controlling spectrum processing parameters."""
        ui_sections.build_tools_panel(self)

    def _build_message_label(self) -> None:
        """Display a status message label."""
        ui_sections.build_message_label(self)

    def _build_model_peak_section(self) -> None:
        """Configure the model peak parameter widgets."""
        ui_sections.build_model_peak_section(self)

    def _init_plot_widgets(self) -> None:
        """Initialise the main spectrum plot area and related items."""
        ui_sections.init_plot_widgets(self)

    def _update_fit_window(self, indexX: Optional[Sequence[int]] = None) -> None:
        """Update the excluded zoom regions according to the current fit window."""

        left_region = getattr(self, "zoom_exclusion_left", None)
        right_region = getattr(self, "zoom_exclusion_right", None)

        if left_region is None or right_region is None:
            return

        def _hide_regions() -> None:
            left_region.setVisible(False)
            right_region.setVisible(False)

        gauge_index = getattr(self, "index_jauge", None)

        if (
            self.Spectrum is None
            or self.index_pic_select is None
            or gauge_index is None
            or gauge_index < 0
        ):
            _hide_regions()
            return

        try:
            params = self.Param0[gauge_index][self.index_pic_select]
        except (IndexError, TypeError, AttributeError):
            _hide_regions()
            return

        if not params:
            _hide_regions()
            return

        n_sigma_widget = getattr(self, "sigma_pic_fit_entry", None)
        if n_sigma_widget is None:
            _hide_regions()
            return

        center = float(params[0])
        sigma = float(params[2])
        n_sigma = float(n_sigma_widget.value())

        if not np.isfinite(center) or not np.isfinite(sigma) or not np.isfinite(n_sigma):
            _hide_regions()
            return

        spectrum_x = getattr(self.Spectrum, "wnb", None)
        if spectrum_x is None:
            _hide_regions()
            return

        x_data = np.asarray(spectrum_x, dtype=float)
        if x_data.size == 0:
            _hide_regions()
            return

        window_left = center - n_sigma * sigma
        window_right = center + n_sigma * sigma
        fit_left = float(min(window_left, window_right))
        fit_right = float(max(window_left, window_right))

        if indexX is None:
            mask = (x_data >= fit_left) & (x_data <= fit_right)
            index_array = np.nonzero(mask)[0]
        else:
            index_array = np.asarray(indexX)

        if index_array.size == 0:
            _hide_regions()
            return

        x_min = float(np.min(x_data))
        x_max = float(np.max(x_data))

        left_region_end = min(max(fit_left, x_min), x_max)
        right_region_start = max(min(fit_right, x_max), x_min)

        if left_region_end > x_min:
            left_region.setRegion((x_min, left_region_end))
            left_region.setVisible(True)
        else:
            left_region.setVisible(False)

        if right_region_start < x_max:
            right_region.setRegion((right_region_start, x_max))
            right_region.setVisible(True)
        else:
            right_region.setVisible(False)

    def _build_ddac_section(self) -> None:
        """Create the dDAC plots widget."""
        ui_sections.build_ddac_section(self)

    def _build_gauge_section(self) -> None:
        """Initialise widgets related to gauge information."""
        ui_sections.build_gauge_section(self)

    def _init_controllers(self) -> None:
        """Instantiate domain controllers bound to the active widgets."""

        self.ddac_controller = DdacController(self)
        self.spectrum_controller = SpectrumController(
            spectrum_getter=lambda: self.Spectrum,
            run_getter=lambda: self.RUN,
            ax_spectrum=self.ax_spectrum,
            remove_button=getattr(self, "remove_btn", None),
        )
        self.gauge_controller = GaugeController(
            spectrum_getter=lambda: self.Spectrum,
            gauge_getter=lambda: self.gauge_select,
            gauge_setter=lambda value: setattr(self, "gauge_select", value),
            ax_spectrum=self.ax_spectrum,
            ax_dy=self.ax_dy,
            layout_dhkl=self.layout_dhkl,
            lamb0_entry=getattr(self, "lamb0_entry", None),
            name_spe_entry=getattr(self, "name_spe_entry", None),
            spinbox_p=self.spinbox_P,
            spinbox_t=self.spinbox_T,
            get_bit_modif_PTlambda=lambda: self.bit_modif_PTlambda,
            set_bit_modif_PTlambda=lambda value: setattr(self, "bit_modif_PTlambda", value),
            get_bit_load_jauge=lambda: self.bit_load_jauge,
            get_bit_modif_jauge=lambda: self.bit_modif_jauge,
            get_index_jauge=lambda: self.index_jauge,
            set_save_value=lambda value: setattr(self, "save_value", value),
            gauge_color_getter=lambda name: self._get_gauge_color(name),
            cl_module=CL,
        )
        self.spinbox_P.valueChanged.connect(self.gauge_controller.spinbox_p_move)
        self.spinbox_T.valueChanged.connect(self.gauge_controller.spinbox_t_move)
        self.bit_load_jauge = False
        self.bit_modif_jauge = False

    def _finalize_layout(self) -> None:
        """Apply stretch factors to the main grid layout."""

        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setColumnStretch(2, 6)
        self.grid_layout.setColumnStretch(4, 3)
        self.grid_layout.setRowStretch(0, 5)
        self.grid_layout.setRowStretch(1, 1)
        self.grid_layout.setRowStretch(2, 1)
        self.grid_layout.setRowStretch(3, 2)
        self.grid_layout.setRowStretch(4, 1)
        self.bit_c_reduite = True

    def apply_theme(self, light=True):
        """Apply the selected theme to the application."""
        global claire, style
        claire = light
        style = qt_styles.qt_stylesheet(light=claire)
        QApplication.instance().setStyleSheet(style)
        qt_styles.configure_pyqtgraph(light=claire)

    def open_settings_dialog(self):
        dialog = SettingsDialog("Light" if claire else "Dark", str(self.config_file), self)
        if dialog.exec_() == QDialog.Accepted:
            theme, path = dialog.result
            self.apply_theme(theme == 'Light')
            if path:
                self.config_file = Path(path).expanduser()
                self.load_paths_from_txt()

    def _open_oscilloscope_viewer(self) -> None:
        base_folder = self.folder_start or ""
        folder = os.path.join(base_folder, "Aquisition_LECROY_Banc_CEDd")
        self.viewer = Oscilo.OscilloscopeViewer(folder=folder)
        self.viewer.show()

    def _find_shortcut_action(self, key: int, modifiers: Qt.KeyboardModifiers) -> Optional[Dict[str, object]]:
        normalized = modifiers & _RELEVANT_MODIFIERS
        action = KEYBOARD_SHORTCUTS.get((key, normalized))
        if action is not None:
            return action

        for (shortcut_key, mask), candidate in KEYBOARD_SHORTCUTS.items():
            if shortcut_key != key or mask == Qt.NoModifier:
                continue
            if not candidate.get("allow_extra_modifiers"):
                continue
            if normalized & mask == mask:
                return candidate
        if normalized == Qt.NoModifier:
            return KEYBOARD_SHORTCUTS.get((key, Qt.NoModifier))
        return None

    def keyPressEvent(self, event):  # - - - COMMANDE CLAVIER - - - #
        key = event.key()
        modifiers = event.modifiers()

        if self.viewer is not None and self.focusWidget() == self.viewer:
            logger.debug("Focus in Lecroy viewer; keyboard shortcut ignored.")
            return

        if Setup_mode is True:
            logger.debug("Touche pressée: %s", key)

        action = self._find_shortcut_action(key, modifiers)
        if action is None:
            super().keyPressEvent(event)
            return

        handler = action["handler"]

        if action.get("direct_return"):
            try:
                handler(self)
            except Exception:
                e = traceback.format_exc()
                self.Print_error(e)
            return

        try:
            if action.get("requires_box"):
                self.Box_loading(lambda: handler(self), action.get("name") or "")
            else:
                handler(self)
                name = action.get("name")
                if name:
                    self.text_box_msg.setText(f"{name} SUCCES idTouche: {key}")
        except Exception:
            e = traceback.format_exc()
            self.Print_error(e)
        finally:
            self.bit_bypass = False

    def Box_loading(self,fonction,name_f):
        # Création de la QMessageBox
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowTitle(name_f+"\n En cours")
        self.msg_box.setText("ATTENTION \n Sur un malentendu ça peu planter")
        self.msg_box.setStandardButtons(QMessageBox.NoButton)
        self.msg_box.show()
        try:
            fonction()
            text=name_f+"\n Terminée"
        except Exception:
            e = traceback.format_exc()
            text='ERROR:'+ str(e)+'  \n check in console variable'
            """Actions lorsque la tâche est terminée."""
        self.msg_box.setText(text+"\n Press Entrée for quit")
        ok_button = self.msg_box.addButton(QMessageBox.Ok)
        ok_button.setText("OK (Entrée)")
        ok_button.setShortcut("Return")

    def Print_error(self,error):
        if sys.exc_info()[0] is not None:
            logger.exception("UI error displayed to user")
        else:
            logger.error("UI error displayed to user: %s", error)
        error_box = QMessageBox(self)
        error_box.setWindowTitle("Warning error")

        text='ERROR:'+ str(error)+'  \n check in console variable \n Press X for quit'
        error_box.setText(text)

        x_button = error_box.addButton("Quit (X)", QMessageBox.AcceptRole)
        error_box.setDefaultButton(x_button)

        def on_key_press(event):
            if event.key() == Qt.Key_X:
                x_button.click()

        error_box.keyPressEvent = on_key_press
        
        error_box.exec_()    


#########################################################################################################################################################################################
#? COMMANDE

    def execute_code(self,code=None):
        if code is None:
            # Récupérer le code de la zone de texte
            code = self.text_edit.toPlainText()
        # Créer un flux de texte pour capturer la sortie de print()
        stdout_capture = io.StringIO()
        sys.stdout = stdout_capture 
        if Setup_mode==True:
            # Compiler et exécuter le code
            exec_locals = {'self': self ,'CL': CL}
            exec(code, {}, exec_locals)
            
            # Récupérer la sortie capturée et restaurer stdout
            sys.stdout = sys.__stdout__  # Restaurer stdout
            output = stdout_capture.getvalue()  # Récupérer la sortie du flux
            output += exec_locals.get('result', 'Exécution réussie sans sortie spécifique.')
            self.output_display.setPlainText(output)
        else:
            try:
                # Compiler et exécuter le code
                exec_locals = {'self': self ,'CL': CL}
                exec(code, {}, exec_locals)
                
                # Récupérer la sortie capturée et restaurer stdout
                sys.stdout = sys.__stdout__  # Restaurer stdout
                output = stdout_capture.getvalue()  # Récupérer la sortie du flux
                output += exec_locals.get('result', 'Exécution réussie sans sortie spécifique.')
                self.output_display.setPlainText(output)
            except Exception as e:
                # Restaurer stdout en cas d'erreur et afficher l'erreur dans la zone de sortie
                sys.stdout = sys.__stdout__
                self.output_display.setPlainText(f"Erreur : {e}")
    
    def toggle_colonne(self):
        if self.bit_c_reduite:
            self.grid_layout.setColumnStretch(2, 5)
        else:
            self.grid_layout.setColumnStretch(2, 0)
        self.bit_c_reduite = not self.bit_c_reduite  # Inverser l'état

    def _update_help_button_color(self, visible: bool) -> None:
        color = 'lightgreen' if visible else 'lightcoral'
        self.help_toggle_btn.setStyleSheet(f"background-color: {color};")
        
        
    def _update_live_button_color(self, active: bool) -> None:
        color = "lightgreen" if active else "lightcoral"
        self.live_toggle_btn.setStyleSheet(f"background-color: {color};")


    def _ensure_help_entries_loaded(self) -> None:
        """Populate the help cache for the keyboard window."""
        if self.help_entries:
            return
        self.helpLabel.clear()
        try:
            load_help_file(self.helpLabel, self.file_help)
            self.help_entries = [self.helpLabel.item(i).text() for i in range(self.helpLabel.count())]
        except Exception as exc:
            self.help_entries = [f"Error loading file: {exc}"]

    def get_fit_variation(self) -> float:
        """Return the fit variation ratio derived from the percentage spin box."""
        if not hasattr(self, "inter_entry"):
            return 1.0
        return float(self.inter_entry.value()) / 100.0

    def _selection_highlight_color(self, alpha: int = 100) -> QColor:
        """Return the highlight colour used when a peak is selected."""
        color = QColor("lightgreen")
        color.setAlpha(alpha)
        return color

    def toggle_help_box(self):
        """Show or hide the contextual help panel on the right side."""
        visible = not self.help_widget.isVisible()
        self.help_widget.setVisible(visible)
        self.help_tab_visible = visible
        self._update_help_button_color(visible)

    def toggle_live_mode(self, active: Optional[bool] = None) -> None:
        """Toggle the live DRX folder monitoring."""
        if active is None:
            active = bool(self.live_toggle_btn.isChecked())

        self.live_mode_active = active
        self._update_live_button_color(active)

        if active:
            self.live_timer.start()
            self._check_live_folder()
        else:
            self.live_timer.stop()

    def set_jungfrau_mode(self, mode: str) -> None:
        self.jungfrau_mode = mode.lower()
    


    def _get_latest_drx_folder(self) -> Optional[Path]:
        drx_root = Path(self.dict_folders.get("DRX", ""))
        if not drx_root.exists():
            return None

        folders = [entry for entry in drx_root.iterdir() if entry.is_dir()]
        if not folders:
            return None

        return max(folders, key=lambda entry: entry.stat().st_ctime)

    def _get_current_drx_folder(self) -> Optional[Path]:
        if not self.loaded_file_DRX:
            return None

        drx_root = Path(self.dict_folders.get("DRX", ""))
        if not drx_root.exists():
            return None

        try:
            relative_path = Path(self.loaded_file_DRX).resolve().relative_to(drx_root.resolve())
        except Exception:
            return None

        if not relative_path.parts:
            return None

        return drx_root / relative_path.parts[0]

    def _find_latest_drx_file(self, folder: Path) -> Optional[Path]:
        candidates = list(folder.rglob("scan_jf1m_0000.h5"))
        if not candidates:
            return None
        return max(candidates, key=lambda entry: entry.stat().st_mtime)

    def _set_loaded_drx_file(self, file_path: Path) -> bool:
        if not file_path.exists():
            self.text_box_msg.setText("File not found")
            return False

        self.loaded_file_DRX = str(file_path)
        self.file_label_spectro.setText(f"DRX: {file_path.name}")
        self.DRX_selector.clear()

        try:
            data_drx = fabio.open(str(file_path))
        except Exception as exc:
            self.text_box_msg.setText(f"Erreur lecture DRX: {exc}")
            return False
        
        num_spec=min(self.DRX_selector.currentIndex(),data_drx.nframes-1)
        self.DRX_selector.addItems([f"DRX{i}" for i in range(data_drx.nframes)])
        self.DRX_selector.setCurrentIndex(0)
        return True

    def _create_spectrum_from_loaded_drx(self, save_gauges: Optional[List[Element]] = None) -> bool:
        if not os.path.exists(self.loaded_file_DRX):
            self.text_box_msg.setText("File not found")
            return False

        if self.calib is None or getattr(self.calib, "mask", None) is None or getattr(self.calib, "ai", None) is None:
            self.text_box_msg.setText("err: calibration not available for integration")
            return False

        save_gauges = list(save_gauges or [])
        try:
            img_data = fabio.open(self.loaded_file_DRX)
            frame = img_data.getframe(min(self.DRX_selector.currentIndex(),img_data.nframes-1)).data
        except Exception as exc:
            self.text_box_msg.setText(f"Erreur chargement DRX: {exc}")
            return False

        tth, intens = Calibration.Integrate_DRX(
            frame,
            self.calib.mask,
            self.calib.ai,
            theta_range=self.calib.theta_range,
        )

        self.text_box_msg.setText("New integration")
        self.bit_bypass = True
        try:
            self.f_Spectrum_Load(
                Spectrum=CL.Spectre(np.array(tth), np.array(intens), Gauges=save_gauges)
            )
        finally:
            self.bit_bypass = False
        return True

    def _check_live_folder(self) -> None:
        if not self.live_mode_active:
            return
        
        latest_folder = self._get_latest_drx_folder()
        if latest_folder is None:
            return

        current_folder = self._get_current_drx_folder()
        if current_folder is not None and latest_folder.resolve() == current_folder.resolve():
            return

        latest_file = self._find_latest_drx_file(latest_folder)
        if latest_file is None:
            return

        if self.loaded_file_DRX and Path(self.loaded_file_DRX).resolve() == latest_file.resolve():
            return

        previous_spectrum = self.Spectrum
        save_gauges = []
        has_fit_gauges = False
        if previous_spectrum is not None:
            gauges = getattr(previous_spectrum, "Gauges", None) or []
            if gauges:
                save_gauges = copy.deepcopy(gauges)
                has_fit_gauges = bool(
                    getattr(previous_spectrum, "bit_fit", False)
                    or any(getattr(gauge, "bit_fit", False) for gauge in gauges)
                )

        if not self._set_loaded_drx_file(latest_file):
            return

        if not self._create_spectrum_from_loaded_drx(save_gauges=save_gauges):
            return

        if has_fit_gauges and save_gauges:
            previous_bypass = self.bit_bypass
            self.bit_bypass = True
            try:
                self.FIT_lmfitVScurvfit()
            finally:
                self.bit_bypass = previous_bypass



    def try_command(self,item):
        print("à coder")
        #command=item.text()
        #self.execute_code(code=text[1])

    def display_command(self, item):
        previous_command = self.text_edit.toPlainText()

        index = self.list_Commande.row(item)
        if 0 <= index < len(self.list_Commande_python):
            commande = self.list_Commande_python[index]
            self.text_edit.setText(previous_command + commande)

    def code_print(self):
        self.execute_code(code="print("+self.text_edit.toPlainText()+')')
    
    def code_len(self):
        self.execute_code(code="print(len("+self.text_edit.toPlainText()+'))')

    def code_clear(self):
        self.text_edit.setText("")

    def select_folder_dict(self):
        file_type = self.type_selector.currentText()
        # Fonction pour parcourir un dossier et afficher ses fichiers
        options = QFileDialog.Options()
        self.dict_folders[file_type] = QFileDialog.getExistingDirectory(self, f"Sélectionner un dossier pour {file_type}", options=options,directory=self.dict_folders[file_type])
        if self.dict_folders[file_type] :
            self.f_change_file_type()

    def _refresh_drx_view(self) -> None:
        if getattr(self, "_drx_in_refresh", False):
            return
        self._drx_in_refresh = True
        try:
            line_value = getattr(self, "x_clic", 0)
            if hasattr(self, "line_P"):
                self.line_P.setValue(line_value)
            if hasattr(self, "line_dPdt"):
                self.line_dPdt.setValue(line_value)

            zone = getattr(self, "zone_diff_int", None)
            zone_range = getattr(self, "zone_diff_int_range", None)
            if zone is not None and zone_range:
                zone.setRegion(zone_range)

            image = getattr(self, "_cedx_image_cache", None)
            if image is not None and hasattr(self, "img_diff_int_item"):
                self._update_image_safe(
                    self.img_diff_int_item,
                    image,
                    levels=getattr(self, "_cedx_levels", None),
                )
        finally:
            self._drx_in_refresh = False

    def _update_curve_safe(self, item, x, y) -> None:
        if item is None:
            return
        x_values = np.asarray(x, dtype=float)
        y_values = np.asarray(y, dtype=float)
        if x_values.shape != y_values.shape:
            return
        last = getattr(item, "_last_xy", None)
        if last is not None:
            last_x, last_y = last
            if last_x.shape == x_values.shape and last_y.shape == y_values.shape:
                if np.allclose(last_x, x_values) and np.allclose(last_y, y_values):
                    return
        item.setData(x_values, y_values)
        item._last_xy = (x_values.copy(), y_values.copy())

    def _update_image_safe(self, item, image, levels=None) -> None:
        if item is None or image is None:
            return
        img = np.asarray(image)
        item.setImage(img, autoLevels=False)
        if levels is not None:
            try:
                item.setLevels(levels)
            except Exception:
                pass

    def _run_drx_task(self, func, *args, **kwargs) -> None:
        def _runner() -> None:
            try:
                func(*args, **kwargs)
            finally:
                self._refresh_drx_view()

        QTimer.singleShot(0, _runner)

    def _axis_from_role(self, axis_role: str):
        return {
            "pressure": getattr(self, "ax_P", None),
            "derivative": getattr(self, "ax_dPdt", None),
            "image": getattr(self, "ax_diff_int", None),
        }.get(axis_role, getattr(self, "ax_P", None))

    def _on_drx_plot_clicked(self, event, *, axis_role: str) -> None:
        axis = self._axis_from_role(axis_role)
        self._run_drx_task(self.on_move, event, axis, axis_role == "image", axis_role != "image")

    def _update_piezo_view_geometry(self) -> None:
        """Keep the auxiliary piezo Y axis aligned with the pressure view box."""
        if not hasattr(self, "ax_P") or not hasattr(self, "ax_P_piezo"):
            return
        self.ax_P_piezo.setGeometry(self.ax_P.vb.sceneBoundingRect())
        self.ax_P_piezo.linkedViewChanged(self.ax_P.vb, self.ax_P_piezo.XAxis)

#########################################################################################################################################################################################
#? COMMANDE print1
    def on_move(self, event,ax,dt=False,argmin=False):  # Fonction appelée lorsqu'on clique dans un graphique
        self.setFocus()
        pos = event.scenePos()
        if ax.sceneBoundingRect().contains(pos):
            mouse_point = ax.vb.mapSceneToView(pos)
        else:
            return    
        x = mouse_point.x()
        self.x_clic = x
        if self.spectrum_select_box.isChecked() is True:
                if dt:
                    # Trouve le plus grand i tel que time[i] <= x_clic
                    i = np.searchsorted(self.time, self.x_clic, side="right")-1

                    # Vérification pour éviter d'être hors limites
                    if 0 <= i < len(self.time)-1:
                        new_spec_nb = self.spectre_number[i]
                    else:
                        new_spec_nb = self.index_spec  # Ou une gestion d'erreur/log

                elif argmin:
                    new_spec_nb = self.spectre_number[np.argmin(abs(self.time - self.x_clic))]

                if new_spec_nb != self.index_spec:
                    # Déplace les lignes verticales
                    t_s = copy.deepcopy(self.time[new_spec_nb])
                    try:
                        t_e = copy.deepcopy(self.time[new_spec_nb + 1])
                    except Exception as e:
                        t_e = t_s + 0.05
                    t_range = [t_s, t_e]
                    self.line_P.setValue(t_s)
                    self.line_dPdt.setValue(t_s)
                    self.zone_diff_int.setRegion(t_range)

                    self.RUN.Spectra[self.index_spec] = copy.deepcopy(self.Spectrum)
                    self.index_spec = new_spec_nb
                    self.DRX_selector.setCurrentIndex(self.index_spec)
                    self.bit_bypass = True
                    self.f_Spectrum_Load()
                    self.bit_bypass = False


    def REFRESH(self):
        if self.RUN is None:
            return
        self.RUN.Spectra[self.index_spec] = self.Spectrum
        self.RUN.Corr_Summary()
        self._update_cedx_plots_from_run(reset_legend=False)
        self.bit_bypass = True
        self.f_Spectrum_Load(Spectrum=self.RUN.Spectra[self.index_spec])
        self.bit_bypass = False
        self._refresh_drx_view()
#########################################################################################################################################################################################
#? COMMANDE FILE
   
    def f_edit_gauge(self):
        element = self.gauge_select
        if element is None:
            QMessageBox.information(self, "Info", "No gauge selected")
            return
        dlg = JcpdsEditor(element, self)
        if dlg.exec_() == QDialog.Accepted:
            self.gauge_controller.f_Gauge_Load(element)

    def f_select_directory(self,file_name,file_label,name,type_file=".asc"):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
         # Créer une instance de QFileDialog
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)  # Permet de sélectionner un seul fichier
        dialog.setNameFilter(f"Text Files (*{type_file});;All Files (*)")
        dialog.setViewMode(QFileDialog.Detail)  # Affiche les fichiers avec des détails comme la date, la taille, etc.

        # Définir le répertoire initial
        if file_name is None:
            dialog.setDirectory(self.folder_start or "")
        else:
            dialog.setDirectory(os.path.dirname(file_name))

         # Afficher la boîte de dialogue et récupérer le fichier sélectionné
        if dialog.exec_():
            file_name_bis = dialog.selectedFiles()[0]  # Récupère le premier fichier sélectionné
            file_name = file_name_bis  # Met à jour la variable locale
            file_label.setText(f"Selected folder: {os.path.basename(file_name)}")
            return file_name  # Retourn

    def f_data_spectro(self):
        self.spectro_data = None
        if self.dict_folders["DRX"]:
            try:
                self.spectro_data=pd.read_csv( self.dict_folders["DRX"], sep='\s+',header=None,skiprows=43,engine='python')
            except Exception as e:
                self.text_box_msg.setText("ERRO FILE")
                return

        if self.spectro_data is not None and len(self.spectro_data.columns) ==2:
            wave=self.spectro_data.iloc[:,0]
            Iua=self.spectro_data.iloc[:,1]
            wave_unique=np.unique(wave)
            num_spec = len(wave)//len(wave_unique)
            if num_spec >=1:
                Iua=Iua.values.reshape(num_spec,len(wave_unique)).T
                self.spectro_data=pd.DataFrame(np.column_stack([wave_unique,Iua]),columns=[0]+ [i+1 for i in range(num_spec)])

        self.DRX_selector.clear()
        if self.spectro_data is None:
            return
        for i in range(1,len(self.spectro_data.columns)):
            self.DRX_selector.addItem(str(f"Spec n°{i}"))
        self.DRX_selector.setCurrentIndex(0)

    def f_change_file_type(self):
        self.file_controller.change_file_type()

    def f_filter_files(self):
        self.file_controller.filter_files()
    
    def f_select_file(self):
        self.file_controller.select_file()


#########################################################################################################################################################################################
#? COMMANDE LIST EXP
    def get_energy_value(self) -> float:
        """Return the current X-ray energy value in keV."""

        return float(self.energy_value)

    def set_energy_value(self, value: float) -> None:
        """Update the global energy value and propagate it to dependent objects."""
        try:
            energy = float(value)
        except (TypeError, ValueError):
            return

        self.energy_value = energy

        if getattr(self, "ClassDRX", None) is not None:
            self.ClassDRX.set_E(energy)

            bibli = getattr(self.ClassDRX, "Bibli_elements", None)
            if isinstance(bibli, dict):
                for element in bibli.values():
                    if element is None:
                        continue

                    setattr(element, "E", energy)

                    refresh_method = getattr(element, "Eos_Pdhkl", None)
                    if callable(refresh_method):
                        try:
                            refresh_method(getattr(element, "P_start", 0))
                        except Exception:
                            # Keep the interface resilient: ignore elements that
                            # cannot be refreshed with the current signature.
                            pass

        if getattr(self, "calib", None) is not None:
            setattr(self.calib, "energy", energy)

        if hasattr(self, "energy_label"):
            self.energy_label.setText(self._format_energy_label())

    def _format_energy_label(self) -> str:
        """Format the energy label displayed in the tools panel."""

        return f"Energy: {int(self.energy_value*1e-3)} keV"

    def select_folder_DRX(self):
        options = QFileDialog.Options()
        self.dict_folders["DRX"] = QFileDialog.getExistingDirectory(self, "F_DRX", options=options,directory=os.path.dirname(self.dict_folders["DRX"]))
        
    def select_folder_oscilo(self):
        options = QFileDialog.Options()
        self.dict_folders["Oscilloscope"] = QFileDialog.getExistingDirectory(self, "F_Oscilo", options=options, directory=os.path.dirname(self.dict_folders["Oscilloscope"]))

    def select_file_DRX(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "F_DRX",
            options=options,
            directory=os.path.dirname(self.loaded_file_DRX),
        )
        if file_path:
            self.set_loaded_drx_file(file_path)

                
    def select_file_oscilo(self):
        options = QFileDialog.Options()
        self.loaded_file_OSC,_ = QFileDialog.getOpenFileName(self, "F_Oscilo", options=options,directory=self.loaded_file_OSC)
        self.file_label_oscilo.setText(f"Oscillo: {os.path.basename(self.loaded_file_OSC) if self.loaded_file_OSC else 'None'}")

    def Calibration_DRX(self):
        if self.RUN is None:
            return

        self.calib = self.RUN.calib
        new_energy = 1.239841984e-6 / self.calib.ai.wavelength

        if self.energy_value != new_energy and new_energy is not None:
            self.set_energy_value(new_energy)

    def _open_calibration_dialog(self):
        if self.RUN is not None and getattr(self.RUN, "calib", None) is not None :
            self.calib = self.RUN.calib

        file_img = None
        if os.path.isfile(self.loaded_file_DRX):
            try:
                data_drx = fabio.open(self.loaded_file_DRX)
                file_img = data_drx.get_frame(self.DRX_selector.currentIndex()).data
            except Exception:
                print("no file DRX load for calib")

        if self.calib is None:
            self.text_box_msg.setText("warn: no calibration loaded")
            return

        new_energy = self.calib.Change_calib(file=file_img, energy=self.get_energy_value())

        if self.energy_value != new_energy and new_energy is not None:
            self.set_energy_value(new_energy)

    def set_loaded_drx_file(self, file_path: str, scan_name: Optional[str] = None) -> None:
        self.loaded_file_DRX = file_path
        self.loaded_drx_scan_name = scan_name or self._infer_drx_scan_name(file_path)
        self.file_label_spectro.setText(f"DRX: {os.path.basename(self.loaded_file_DRX) if self.loaded_file_DRX else 'None'}")
        if self.loaded_file_DRX:
            try:
                self.DRX_selector.clear()
                data_drx = fabio.open(self.loaded_file_DRX)
                self.DRX_selector.addItems([f"DRX{i}" for i in range(data_drx.nframes)])
            except Exception as exc:
                print(f"Erreur lors du chargement DRX : {exc}")


    def _infer_drx_scan_name(self, file_path: str) -> str:
        if not file_path:
            return ""
        scan_folder = os.path.basename(os.path.dirname(file_path))
        if scan_folder.lower().startswith("scan"):
            root_folder = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            match = re.match(r"(.*)_\d+$", root_folder)
            prefix = match.group(1) if match else root_folder
            return f"{prefix}_{scan_folder}"
        return os.path.splitext(os.path.basename(file_path))[0]

    def _get_cedx_base_name(self) -> str:
        if self.loaded_drx_scan_name:
            return self.loaded_drx_scan_name
        if self.loaded_file_OSC:
            return os.path.basename(self.loaded_file_OSC)
        if self.loaded_file_DRX:
            return os.path.splitext(os.path.basename(self.loaded_file_DRX))[0]
        return "CEDX"


#########################################################################################################################################################################################
#? COMMANDE AUTO DIF
    def f_print_region(self):
        self.spectrum_controller.toggle_regions_visibility()

    def set_find_peaks_zones_visibility(self, checked: bool) -> None:
        if hasattr(self, "spectrum_controller"):
            self.spectrum_controller.set_regions_visible(bool(checked))

    def run_fit_selected_spectra(self):
        index_start = self.index_start_entry.value()
        index_stop = self.index_stop_entry.value()
        
        success = self.spectrum_controller.run_fit_selected_spectra(
            index_start=index_start,
            index_stop=index_stop,
            ngen=self.NGEN_entry.value(),
            mutpb=self.MUTPB_entry.value(),
            cxpb=self.CXPB_entry.value(),
            popinit=self.POPINIT_entry.value(),
            pressure_range=[0, self.p_range_entry.value()],
            max_ecart_pressure=1,
            max_elements=self.nb_max_element_entry.value(),
            tolerance=self.tolerance_entry.value(),
            custom_peak_params={
                "height": self.height_entry.value(),
                "distance": self.distance_entry.value(),
                "prominence": self.prominence_entry.value(),
                "width": self.width_entry.value(),
                "number_peak_max": self.nb_peak_entry.value(),
            },
        )
        if not success:
            self.text_box_msg.setText("No RUN loaded.")

    def ajouter_zone(self):
        self.spectrum_controller.add_zone()

    def selectionner_zone(self, region):
        self.spectrum_controller.select_zone(region)

    def supprimer_zone(self):
        self.spectrum_controller.remove_selected_zone()

    def f_update_theta2_range(self):
        if hasattr(self, "spectrum_controller"):
            self.theta2_range = [list(map(float, region)) for region in self.spectrum_controller.update_theta2_range()]
        return getattr(self, "theta2_range", [])

#########################################################################################################################################################################################
#? COMMANDE PRINT
    def _compute_spectrum_bounds(self, *, include_overlays: bool = True):
        spectrum = getattr(self, "Spectrum", None)
        if spectrum is None:
            return None

        x_values = np.asarray(getattr(spectrum, "wnb", []), dtype=float)
        if x_values.size == 0:
            return None

        finite_x = x_values[np.isfinite(x_values)]
        if finite_x.size == 0:
            return None

        x_min = float(finite_x.min())
        x_max = float(finite_x.max())
        y_candidates = []

        def _append(values):
            if values is None:
                return
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                return
            finite = arr[np.isfinite(arr)]
            if finite.size:
                y_candidates.append(finite)

        _append(getattr(spectrum, "y_corr", None))
        _append(getattr(self, "y_fit_start", None))

        if include_overlays:
            if getattr(self, "act_show_raw", None) is not None and self.act_show_raw.isChecked():
                _append(getattr(spectrum, "spec", None))
            if getattr(self, "act_show_filtered", None) is not None and self.act_show_filtered.isChecked():
                _append(getattr(spectrum, "y_filtre", None))
            if getattr(self, "act_show_baseline", None) is not None and self.act_show_baseline.isChecked():
                _append(getattr(spectrum, "blfit", None))

        if not y_candidates:
            return None

        y_values = np.concatenate(y_candidates)
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return None
        if y_min == y_max:
            y_min -= 1
            y_max += 1

        return x_min, x_max, y_min, y_max

    def _apply_spectrum_limits(self, *, include_overlays: bool = True) -> None:
        bounds = self._compute_spectrum_bounds(include_overlays=include_overlays)
        if bounds is None:
            return

        x_min, x_max, y_min, y_max = bounds
        padding = (y_max - y_min) * 0.05
        y_max_limit = y_max + padding

        self.ax_spectrum.disableAutoRange()
        self.ax_spectrum.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max_limit)
        self.ax_spectrum.setXRange(x_min, x_max)
        self.ax_spectrum.setYRange(y_min, y_max_limit)

    
    def plot_spectrum(self):# INUTILE ,
        self.ax_spectrum.set_xlabel('X (U.A)')
        self.ax_spectrum.set_ylabel('Y (U.A)')
        self.plot_zoom =self.ax_zoom.plot(self.Spectrum.wnb,self.Spectrum.y_corr,'-',color='dimgray',markersize=4)[0]
        self.cross_zoom.setPos(self.Spectrum.wnb[0],self.Spectrum.y_corr[0])  

        self.plot_data_fit.setData(self.Spectrum.wnb, self.Spectrum.y_corr)

        
        
        self._apply_spectrum_limits(include_overlays=True)
        self.ax_dy.setXRange(min(self.Spectrum.wnb),max(self.Spectrum.wnb))
        #self.ax_zoom.plot(self.Spectrum.wnb,self.Spectrum.y_corr,'-',color='dimgray',markersize=4) #,label='Data brut'
        self._update_spectrum_overlay_data()
        self.canvas_spec.draw()
        
#########################################################################################################################################################################################
#? COMMANDE self.update 
    def f_gauge_select(self, source=None):
        gauge_name = None
        # Si on reçoit directement un nom
        if isinstance(source, str):
            gauge_name = source
        # Sinon on prend la sélection courante de la table
        if not gauge_name:
            _, gauge_name = self._get_selected_gauge_from_table()
        if not gauge_name:
            self.f_print_dhkl()
            return

        # Surligner la ligne dans la table
        self._select_gauge_in_table(gauge_name)

        in_spectrum = self.Spectrum is not None and gauge_name in [G.name for G in self.Spectrum.Gauges]
        if not in_spectrum:
            if gauge_name in getattr(self.ClassDRX, "Bibli_elements", {}):
                self.bit_load_jauge = True
                self.bit_modif_jauge = False
                self.gauge_select = self.gauge_controller.f_Gauge_Load(
                    copy.deepcopy(self.ClassDRX.Bibli_elements[gauge_name])
                )
            else:
                self.text_box_msg.setText("Gauge out bibli \n pres i to add")
                return
        else:
            self.index_jauge = [ga.name for ga in self.Spectrum.Gauges].index(gauge_name)
            self.bit_load_jauge = False
            self.bit_modif_jauge = True

            self.gauge_select = self.gauge_controller.f_Gauge_Load(
                copy.deepcopy(self.Spectrum.Gauges[self.index_jauge].Element_ref)
            )
            self.listbox_pic.clear()
            for name in self.list_text_pic[self.index_jauge]:
                self.listbox_pic.addItem(name)
        self.f_print_dhkl()
        
    def f_filtre_select(self):
        col1 = self.filtre_type_selector.model().item(self.filtre_type_selector.currentIndex()).background().color().getRgb()
        self.filtre_type_selector.setStyleSheet("background-color: rgba{};	selection-background-color: gray;".format(col1))
        if self.filtre_type_selector.currentText() == "svg":
            self.param_filtre_1_name.setText("w_length")
            self.param_filtre_2_name.setText("degpoly")
        elif self.filtre_type_selector.currentText() == "fft":
            self.param_filtre_1_name.setText("f_c l")
            self.param_filtre_2_name.setText("f_c h")
        else:
            self.param_filtre_1_name.setText("filtre p1")
            self.param_filtre_2_name.setText("filtre p2")

    def f_model_pic_type(self):# - - - SELECT MODEL PIC - - -#
        col1 = self.model_pic_type_selector.model().item(self.model_pic_type_selector.currentIndex()).background().color().getRgb()
        self.model_pic_type_selector.setStyleSheet("background-color: rgba{};	selection-background-color: gray;".format(col1))



        # Clear existing dynamic widgets
        while self.coef_dynamic_spinbox and self.coef_dynamic_spinbox is not []:
            widget = self.coef_dynamic_spinbox.pop()
            self.grid_layout.removeWidget(widget)
            widget.deleteLater()

        while self.coef_dynamic_label and self.coef_dynamic_label is not []:
            widget = self.coef_dynamic_label.pop()
            self.grid_layout.removeWidget(widget)
            widget.deleteLater()

        self.model_pic_fit=self.model_pic_type_selector.currentText()

        pic_exemple=Pics(model_fit=self.model_pic_fit)
        
        for i, coef in enumerate(pic_exemple.name_coef_spe):
            layh=QHBoxLayout()
            lim_min,lim_max=-10,10
            v_start=1
            if coef == "fraction":            
                text="\u03B1"
                v_start=0.5
                lim_min,lim_max=0,1
                
            elif coef == "beta":
                text="\u03B2"
                v_start=0.5
            elif coef == "sigma_r":
                text="\u03C3 <sub>right<\sub>"
                v_start=0.15
            elif coef=="expon":
                text="m"
                v_start=1.1
                lim_min=0.5
            elif coef=="skew":
                text="\u03C5"
                v_start=0
            coef_label = QLabel(text+":", self)
            layh.addWidget(coef_label)
            spinbox_coef = QDoubleSpinBox(self)
            spinbox_coef.valueChanged.connect(self.setFocus)
            spinbox_coef.setRange(lim_min, lim_max)
            spinbox_coef.setSingleStep(0.01)
            spinbox_coef.setValue(v_start)
            layh.addWidget(spinbox_coef)
            self.ParampicLayout.addLayout(layh)
            self.coef_dynamic_label.append(coef_label)
            self.coef_dynamic_spinbox.append(spinbox_coef)
    
    def f_cross_spectrum(self,event):
        pos = event.scenePos()
        if self.ax_spectrum.sceneBoundingRect().contains(pos):
            mouse_point = self.ax_spectrum.vb.mapSceneToView(pos)
            self.f_click_cross(mouse_point)
            if self.select_clic_box.isChecked():
                self.found_pic()

    def f_cross_zoom(self,event):
        pos = event.scenePos()
        if self.ax_zoom.sceneBoundingRect().contains(pos):
            mouse_point = self.ax_zoom.vb.mapSceneToView(pos)
            self.f_click_cross(mouse_point)

    def f_click_cross(self, mouse_point):
            X0, Y0 = mouse_point.x(), mouse_point.y()
            self.setFocus()
            self.X0
            # Mise à jour des croix (InfiniteLine)
            self.axV.setPos(X0)
            self.axH.setPos(Y0)
            self.cross_zoom.setPos(X0, Y0)  # Si c’est une croix dans la vue zoomée

            self.X0, self.Y0 = round(X0, 3), round(Y0, 3)

    def found_pic(self):
        if not self.Param0:
            return

        rng = (self.Spectrum.wnb[-1] - self.Spectrum.wnb[0]) / 50
        candidates = [
            (abs(peak[0] - self.X0), gauge_idx, peak_idx)
            for gauge_idx, peaks in enumerate(self.Param0)
            for peak_idx, peak in enumerate(peaks)
        ]

        if not candidates:
            return

        distance, gauge_idx, peak_idx = min(candidates, key=lambda entry: entry[0])
        if distance >= rng:
            return

        if gauge_idx != self.index_jauge:
            self.index_jauge = gauge_idx
            self._select_gauge_in_table(self.Spectrum.Gauges[self.index_jauge].name)
            self.f_Gauge_Load()

        self.index_pic_select = peak_idx
        self.bit_bypass = True
        self.select_pic()
        self.bit_bypass = False

    def f_lambda0(self):
        lambda0=str(self.Spectrum.Gauges[self.index_jauge].lamb0)
        try:
            self.Spectrum.Gauges[self.index_jauge].lamb0=float(self.lamb0_entry.text())
        except Exception as e:
            self.lamb0_entry.setText(lambda0)
            print("ERROR:",e,"in lambda0")
        
    def f_name_spe(self):
        name_spe=str(self.Spectrum.Gauges[self.index_jauge].name_spe)
        try:
            self.Spectrum.Gauges[self.index_jauge].spe=float(self.name_spe_entry.text())
           
        except Exception as e:
            self.name_spe_entry.setText(name_spe)
            print("ERROR:",e,"in lambda0")

    def f_print_dhkl(self):
        self.gauge_controller.f_print_dhkl()

#########################################################################################################################################################################################
#? COMMANDE Treateamnt Spectrummfit
    def FIT_lmfitVScurvfitOLD(self): # Fonction pour fit
        save_jauge=self.index_jauge
        save_pic=self.index_pic_select

        self.Param_FIT=[]
        list_F=[]
        initial_guess=[]
        bounds_min,bounds_max=[],[]
        #self.nb_jauges=len(self.Spectrum.Gauges)
        x_min, x_max=float(self.Param0[0][0][0]),float(self.Param0[0][0][0])
        for j in range(len(self.Spectrum.Gauges)):
            for i in range(self.J[j]):
                x_min,x_max=min(x_min,float(self.Param0[j][i][0])-float(self.Param0[j][i][2])*5),max(x_max,float(self.Param0[j][i][0])+float(self.Param0[j][i][2])*5)
                self.Spectrum.Gauges[j].pics[i].Update(ctr=float(self.Param0[j][i][0]),ampH=float(self.Param0[j][i][1]),coef_spe=self.Param0[j][i][3],sigma=float(self.Param0[j][i][2]),model_fit=self.Param0[j][i][4],inter=self.get_fit_variation())
                params_f=self.Spectrum.Gauges[j].pics[i].model.make_params()
                list_F.append(self.Spectrum.Gauges[j].pics[i].f_model)
                initial_guess+= self.Param0[j][i][:3]
                for c in self.Param0[j][i][3]:
                    initial_guess+=[c]
                bounds_min+=[self.Spectrum.Gauges[j].pics[i].ctr[1][0],self.Spectrum.Gauges[j].pics[i].ampH[1][0],self.Spectrum.Gauges[j].pics[i].sigma[1][0]]
                bounds_max+=[self.Spectrum.Gauges[j].pics[i].ctr[1][1],self.Spectrum.Gauges[j].pics[i].ampH[1][1],self.Spectrum.Gauges[j].pics[i].sigma[1][1]]
                
                for c in self.Spectrum.Gauges[j].pics[i].coef_spe:
                    bounds_min+=[c[1][0]]
                    bounds_max+=[c[1][1]]
            self.Spectrum.Gauges[j].Update_model()
        bounds=[bounds_min,bounds_max]
        print( [[guest,sig] for guest,sig in zip(initial_guess, np.array(bounds_max)-np.array(bounds_min))])
 
        for j in range(len(self.Spectrum.Gauges)):
            if j ==0:
                self.Spectrum.model = self.Spectrum.Gauges[0].model
            else:
                self.Spectrum.model+=self.Spectrum.Gauges[j].model
       
        self.Spectrum.Data_treatement()
        if self.zone_spectrum_box.isChecked():
            self.Spectrum.indexX=np.where((self.Spectrum.wnb >= x_min) & (self.Spectrum.wnb <= x_max))[0]
            x_sub=self.Spectrum.wnb[self.Spectrum.indexX]
            y_sub = self.Spectrum.y_corr[self.Spectrum.indexX]
            blfit = self.Spectrum.blfit[self.Spectrum.indexX]
        else:
            if self.X_e[0]!=None and self.X_s[0]!=None and self.Spectrum.indexX is not None:
                self.Zone_fit[0] = np.where((self.Spectrum.wnb >= self.X_s[0]) & (self.Spectrum.wnb <= self.X_e[0]))[0]
                x_sub=self.Spectrum.wnb[self.Zone_fit[0]]
                self.Spectrum.indexX=self.Zone_fit[0]
                for J in self.Spectrum.Gauges:
                    J.indexX=self.Zone_fit[0]
                y_sub = self.Spectrum.y_corr[self.Spectrum.indexX]
                blfit = self.Spectrum.blfit[self.Spectrum.indexX]
            else:
                y_sub=self.Spectrum.y_corr
                blfit=self.Spectrum.blfit


        if self.vslmfit.isChecked():
            self.Spectrum.FIT()
            for i, J in enumerate(self.Spectrum.Gauges):
                for j , p in enumerate(J.pics):
                    params_f=p.model.make_params()
                    y_plot=p.model.eval(params_f,x=self.Spectrum.wnb)
                    self.list_y_fit_start[i][j]=y_plot
            


        sum_function = CL.Gen_sum_F(list_F)

        try :
            params , params_covar = curve_fit(sum_function,x_sub,y_sub,p0=initial_guess,bounds=bounds)
            params_sigma = np.sqrt(np.diag(params_covar))
            param_print=[[str(round(init,3))+f"+-"+str(round(init-bound,3))] for init,bound in zip(initial_guess,bounds_min)]
        except Exception as e:
            self.Spectrum.bit_fit=True
            self.bit_fit_T=True 
            self.text_box_msg.setText('FIT ERROR'+str(e))
           
            return
        fit=sum_function(x_sub,*params)

        if sum((fit-y_sub)**2) < sum(self.Spectrum.dY**2):
            text_fit= rf"Curve_fit BEST you can Validate \n REPORT: {[ p for p in param_print]}"
        else:
            text_fit= "Curve_fit LESS GOOD you can Cancel"
        #self.plot_start=self.ax_spectrum.plot(self.Spectrum.wnb,sum_function(self.Spectrum.wnb,*initial_guess),'r')
        self.plot_curv_fit.setData(x_sub,fit)
        self.plot_curv_dY.setData(x_sub,y_sub-fit)
       
        self.Print_fit_start()
       

        if self.bit_bypass is False :
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("CURVE FIT DONE")
            text=text_fit+'\n Save fit Press "v" Cancel Press "c"' #\n Launch lmfit Press "l"
            msg_box.setText(text)

            v_button = msg_box.addButton("Validate", QMessageBox.AcceptRole)

            a_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

            msg_box.setDefaultButton(v_button)

            def on_key_press(event):
                if event.key() == Qt.Key_V:
                    v_button.click()
                elif event.key() == Qt.Key_C:
                    a_button.click()

            msg_box.keyPressEvent = on_key_press
            msg_box.exec_()
            

        if msg_box.clickedButton() == v_button or (self.bit_bypass is True and text_fit=="Curve_fit BEST you can Validate" ):             
            self.plot_curv_fit.setData([],[])
            self.plot_curv_dY.setData([],[])

            self.Spectrum.Y= fit +  blfit
            self.Spectrum.X=x_sub
            self.Spectrum.dY= y_sub-fit 
            self.Spectrum.lamb_fit =params[0]
            ij_3,ij_4,ij_5=0,0,0
            params_list=list(params)

            for i, J in enumerate(self.Spectrum.Gauges):
                for j , p in enumerate(J.pics):
                    n_c=len(self.Param0[i][j][3])
                    start_idx = 3 * ij_3 + 4 * ij_4 + 5 * ij_5
                    end_idx = start_idx + 3
                    l_sigma=params_covar[start_idx:end_idx]
                    if n_c == 0:
                        self.Param0[i][j][:4] = params_list[start_idx:end_idx] #list(params[start_idx:end_idx]) 
                        ij_3 += 1
                    elif n_c == 1:
                        self.Param0[i][j][:4] = params_list[start_idx:end_idx] + [np.array([params_list[end_idx]])]#list(params[start_idx:end_idx]) + list(np.array(params[end_idx]))
                        ij_4 += 1  
                    elif n_c == 2:
                        self.Param0[i][j][:4] = params_list[start_idx:end_idx] + [np.array(params_list[end_idx:end_idx+2])] #list(params[start_idx:end_idx]) + list(np.array(params[end_idx:end_idx+2]))
                        ij_5 += 1
                    p.Update(ctr=float(self.Param0[i][j][0]),ampH=float(self.Param0[i][j][1]),coef_spe=self.Param0[i][j][3],sigma=float(self.Param0[i][j][2]),inter=self.get_fit_variation())
                    p.ctr=[p.ctr,[p.ctr-l_sigma[0],p.ctr+l_sigma[0]]]

                    params_f = p.model.make_params()
                    y_plot = p.model.eval(params_f, x=self.Spectrum.wnb)
                    self._update_pic_display(i, j, y_plot)

                J.lamb_fit=self.Param0[i][0][0]
                J.bit_fit=True
                
            
            
            

            self.Spectrum.bit_fit=True
            self.Spectrum.Calcul_study(mini=True)
            self.text_box_msg.setText('FIT TOTAL \n DONE')
            self.bit_fit_T=True    
            self.index_jauge=save_jauge
            self.index_pic_select=save_pic
            self.f_Gauge_Load()
            self.Print_fit_start()


        else:
            self.plot_curv_fit.setData([],[])
            self.plot_curv_dY.setData([],[])
            self.bit_fit_T=True 
            for i, J in enumerate(self.Spectrum.Gauges):         
                J.bit_fit=True
                if self.vslmfit.isChecked():
                    for j, p in enumerate(J.pics):
                        y_plot=self.list_y_fit_start[i][j]
                        self._update_pic_fill_data(
                            i,
                            j,
                            self.Spectrum.wnb,
                            y_plot,
                            np.zeros_like(self.Spectrum.blfit),
                        )
                        new_P0,_=p.Out_model()                       
                        self.Param0[i][j][:4]=new_P0
                        new_name= self.Nom_pic[i][j] + "   X0:"+str(self.Param0[i][j][0])+"   Y0:"+ str(self.Param0[i][j][1]) + "   sigma:" + str(self.Param0[i][j][2]) + "   Coef:" + str(self.Param0[i][j][3]) + " ; Modele:" + str(self.Param0[i][j][4])
                        self.list_text_pic[i][j]=str(new_name)
                        if i ==save_jauge:
                            self.listbox_pic.takeItem(j)
                            self.listbox_pic.insertItem(j,str(new_name))
                J.lamb_fit=self.Param0[i][0][0]
            self.Spectrum.bit_fit=True
            self.Spectrum.Calcul_study(mini=True)
            self.text_box_msg.setText('BAD FIT r^2 INCREAS')

    def FIT_lmfitVScurvfit(self,skip_ui=False):
        """Fit avec comparaison lmfit vs curve_fit"""
        save_jauge, save_pic = self.index_jauge, self.index_pic_select
        self.Param_FIT, list_F, initial_guess = [], [], []
        bounds_min, bounds_max = [], []

        # Détermination zone globale x_min / x_max + préparation des paramètres initiaux
        x_min = x_max = float(self.Param0[0][0][0])

        for j, gauge in enumerate(self.Spectrum.Gauges):
            for i in range(self.J[j]):
                ctr, ampH, sigma, coef_spe, model_fit = self.Param0[j][i][:5]
                x_min = min(x_min, float(ctr) - float(sigma) * 5)
                x_max = max(x_max, float(ctr) + float(sigma) * 5)

                gauge.pics[i].Update(
                    ctr=float(ctr), ampH=float(ampH), coef_spe=coef_spe,
                    sigma=float(sigma), model_fit=model_fit,
                    inter=self.get_fit_variation()
                )

                list_F.append(gauge.pics[i].f_model)
                initial_guess += [ctr, ampH, sigma] + list(coef_spe)

                bounds_min += [
                    gauge.pics[i].ctr[1][0],
                    gauge.pics[i].ampH[1][0],
                    gauge.pics[i].sigma[1][0]
                ] + [c[1][0] for c in gauge.pics[i].coef_spe]

                bounds_max += [
                    gauge.pics[i].ctr[1][1],
                    gauge.pics[i].ampH[1][1],
                    gauge.pics[i].sigma[1][1]
                ] + [c[1][1] for c in gauge.pics[i].coef_spe]

            gauge.Update_model()

        bounds = [bounds_min, bounds_max]

        # Construction modèle complet
        self.Spectrum.model = sum((g.model for g in self.Spectrum.Gauges[1:]), self.Spectrum.Gauges[0].model)

        # Sélection zone spectrale
        self.Spectrum.Data_treatement()
        if self.zone_spectrum_box.isChecked():
            self.Spectrum.indexX = np.where((self.Spectrum.wnb >= x_min) & (self.Spectrum.wnb <= x_max))[0]
            x_sub = self.Spectrum.wnb[self.Spectrum.indexX]
            y_sub, blfit = self.Spectrum.y_corr[self.Spectrum.indexX], self.Spectrum.blfit[self.Spectrum.indexX]
        elif self.X_e[0] is not None and self.X_s[0] is not None and self.Spectrum.indexX is not None:
            self.Zone_fit[0] = np.where((self.Spectrum.wnb >= self.X_s[0]) & (self.Spectrum.wnb <= self.X_e[0]))[0]
            x_sub, self.Spectrum.indexX = self.Spectrum.wnb[self.Zone_fit[0]], self.Zone_fit[0]
            for gauge in self.Spectrum.Gauges:
                gauge.indexX = self.Zone_fit[0]
            y_sub, blfit = self.Spectrum.y_corr[self.Spectrum.indexX], self.Spectrum.blfit[self.Spectrum.indexX]
        else:
            x_sub, y_sub, blfit = self.Spectrum.wnb, self.Spectrum.y_corr, self.Spectrum.blfit

        # Si lmfit demandé -> préfit
        if self.vslmfit.isChecked():
            self.Spectrum.FIT()
            for i, gauge in enumerate(self.Spectrum.Gauges):
                for j, p in enumerate(gauge.pics):
                    params_f = p.model.make_params()
                    self.list_y_fit_start[i][j] = p.model.eval(params_f, x=self.Spectrum.wnb)

        # Fit curve_fit
        sum_function = CL.Gen_sum_F(list_F)
        try:
            params, params_covar = curve_fit(sum_function, x_sub, y_sub, p0=initial_guess, bounds=bounds)
            params_sigma = np.sqrt(np.diag(params_covar))
        except Exception as e:
            self.Spectrum.bit_fit = self.bit_fit_T = True
            self.text_box_msg.setText(f"FIT ERROR {e}")
            return

        fit = sum_function(x_sub, *params)
        best = (np.sum((fit - y_sub) ** 2) < np.sum(self.Spectrum.dY ** 2))
        text_fit = "Curve_fit BEST you can Validate" if best else "Curve_fit LESS GOOD you can Cancel"

        if not skip_ui:
            # Affichage fit
            self.plot_curv_fit.setData(x_sub, fit)
            self.plot_curv_dY.setData(x_sub, y_sub - fit)
            self.Print_fit_start(skip_ui=skip_ui)

        # Interaction validation
        go = True
        if not self.bit_bypass:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("CURVE FIT DONE")
            msg_box.setText(text_fit + '\n Save fit Press "v" Cancel Press "c"')

            v_button = msg_box.addButton("Validate", QMessageBox.AcceptRole)
            c_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)
            msg_box.setDefaultButton(v_button)

            def on_key_press(event):
                if event.key() == Qt.Key_V: v_button.click()
                elif event.key() == Qt.Key_C: c_button.click()
            msg_box.keyPressEvent = on_key_press
            msg_box.exec_()

            go = (msg_box.clickedButton() == v_button)

        # Post-traitement
        if go and best:
            self._apply_best_fit(params, params_sigma, fit, x_sub, y_sub, blfit, save_jauge, save_pic,skip_ui=skip_ui)
        else:
            self._apply_bad_fit(save_jauge,skip_ui=skip_ui)

    def _apply_best_fit(self, params, params_sigma, fit, x_sub, y_sub, blfit, save_jauge, save_pic,skip_ui=False):
        """Applique les résultats quand le fit est validé et meilleur que l'existant"""
        if not skip_ui:
            self.plot_curv_fit.setData([], [])
            self.plot_curv_dY.setData([], [])

        # Mise à jour spectre global
        self.Spectrum.Y = fit + blfit
        self.Spectrum.X = x_sub
        self.Spectrum.dY = y_sub - fit
        self.Spectrum.lamb_fit = params[0]

        params_list = list(params)
        ij_3 = ij_4 = ij_5 = 0

        for i, gauge in enumerate(self.Spectrum.Gauges):
            for j, p in enumerate(gauge.pics):
                n_c = len(self.Param0[i][j][3])
                start_idx = 3 * ij_3 + 4 * ij_4 + 5 * ij_5
                end_idx = start_idx + 3

                l_sigma = params_sigma[start_idx:end_idx]
                if n_c == 0:
                    self.Param0[i][j][:4] = params_list[start_idx:end_idx]
                    ij_3 += 1
                elif n_c == 1:
                    self.Param0[i][j][:4] = params_list[start_idx:end_idx] + [np.array([params_list[end_idx]])]
                    ij_4 += 1
                elif n_c == 2:
                    self.Param0[i][j][:4] = params_list[start_idx:end_idx] + [np.array(params_list[end_idx:end_idx + 2])]
                    ij_5 += 1

                # Mise à jour du pic
                p.Update(
                    ctr=float(self.Param0[i][j][0]),
                    ampH=float(self.Param0[i][j][1]),
                    coef_spe=self.Param0[i][j][3],
                    sigma=float(self.Param0[i][j][2]),
                    inter=self.get_fit_variation()
                )
                params_f = p.model.make_params()
                y_plot = p.model.eval(params_f, x=self.Spectrum.wnb)
                self.list_y_fit_start[i][j] = y_plot

                # Nom + affichage
                new_name = (
                    f"{self.Nom_pic[i][j]}   X0:{self.Param0[i][j][0]}   "
                    f"Y0:{self.Param0[i][j][1]}   sigma:{self.Param0[i][j][2]}   "
                    f"Coef:{self.Param0[i][j][3]} ; Modele:{self.Param0[i][j][4]}"
                )
                self.list_text_pic[i][j] = str(new_name)
                if not skip_ui:
                    self._update_pic_fill_data(
                        i,
                        j,
                        self.Spectrum.wnb,
                        y_plot,
                        np.zeros_like(self.Spectrum.blfit),
                    )

            gauge.lamb_fit = self.Param0[i][0][0]
            gauge.bit_fit = True

        self.Spectrum.estimate_all_sigma_noise()
        self.Spectrum.bit_fit = True
        self.bit_fit_T = True

        if not skip_ui:
            self.text_box_msg.setText("FIT TOTAL \n DONE")
            self.Spectrum.Calcul_study(mini=True)
            # Restauration indices
            self.index_jauge = save_jauge
            self.index_pic_select = save_pic
            self.f_Gauge_Load()
            self.Print_fit_start(skip_ui=skip_ui)

    def _apply_bad_fit(self, save_jauge,skip_ui=False):
        """Applique les résultats si le fit est rejeté ou moins bon"""
        if not skip_ui:
            self.plot_curv_fit.setData([], [])
            self.plot_curv_dY.setData([], [])
        self.bit_fit_T = True
        for i, gauge in enumerate(self.Spectrum.Gauges):
            gauge.bit_fit = True
            if self.vslmfit.isChecked():
                for j, p in enumerate(gauge.pics):
                    y_plot = self.list_y_fit_start[i][j]
                    new_P0, _ = p.Out_model()
                    self.Param0[i][j][:4] = new_P0
                    if not skip_ui:
                        self._update_pic_display(i, j, y_plot, update_listbox=i == save_jauge)
            gauge.lamb_fit = self.Param0[i][0][0]

        self.Spectrum.estimate_all_sigma_noise()
        self.Spectrum.bit_fit = True
        if not skip_ui:
            self.Spectrum.Calcul_study(mini=True)
            self.text_box_msg.setText("BAD FIT r^2 INCREAS")    
    
    def Baseline_spectrum(self):
        param=[float(self.param_filtre_1_entry.text()),float(self.param_filtre_2_entry.text())]

        if self.filtre_type_selector.currentText() == "svg":
            param[0],param[1]=int(param[0]),int(param[1])
        self.Spectrum.Data_treatement(
            deg_baseline=int(self.deg_baseline_entry.value()),
            type_filtre=self.filtre_type_selector.currentText(),
            param_f=param,
            print_data_QT=False,)
        
        self.plot_data_fit.setData(self.Spectrum.wnb, self.Spectrum.y_corr)
        self.plot_zoom.setData(self.Spectrum.wnb, self.Spectrum.y_corr)
        self._update_spectrum_overlay_data()
        self._apply_spectrum_limits(include_overlays=True)
        self.ax_dy.setXRange(min(self.Spectrum.wnb),max(self.Spectrum.wnb))

    def _update_spectrum_overlay_data(self):
        if getattr(self, "Spectrum", None) is None:
            self.plot_raw_spectrum.setData([], [])
            self.plot_filtered_spectrum.setData([], [])
            self.plot_baseline_curve.setData([], [])
            self.update_spectrum_overlays()
            return

        x_values = getattr(self.Spectrum, "wnb", None)
        if x_values is None:
            self.plot_raw_spectrum.setData([], [])
            self.plot_filtered_spectrum.setData([], [])
            self.plot_baseline_curve.setData([], [])
            self.update_spectrum_overlays()
            return

        raw_values = getattr(self.Spectrum, "spec", None)
        filtered_values = getattr(self.Spectrum, "y_filtre", None)
        baseline_values = getattr(self.Spectrum, "blfit", None)

        self.plot_raw_spectrum.setData(
            x_values, np.asarray(raw_values) if raw_values is not None else []
        )
        self.plot_filtered_spectrum.setData(
            x_values, np.asarray(filtered_values) if filtered_values is not None else []
        )
        self.plot_baseline_curve.setData(
            x_values, np.asarray(baseline_values) if baseline_values is not None else []
        )
        self.update_spectrum_overlays()

    def update_spectrum_overlays(self):
        if not hasattr(self, "act_show_raw"):
            return

        spectrum = getattr(self, "Spectrum", None)
        raw_values = getattr(spectrum, "spec", None) if spectrum is not None else None
        filtered_values = getattr(spectrum, "y_filtre", None) if spectrum is not None else None
        baseline_values = getattr(spectrum, "blfit", None) if spectrum is not None else None

        raw_available = raw_values is not None and len(raw_values) > 0
        filtered_available = filtered_values is not None and len(filtered_values) > 0
        baseline_available = baseline_values is not None and len(baseline_values) > 0

        self.act_show_raw.setEnabled(raw_available)
        if not raw_available:
            self.act_show_raw.setChecked(False)
        self.plot_raw_spectrum.setVisible(raw_available and self.act_show_raw.isChecked())

        self.act_show_filtered.setEnabled(filtered_available)
        if not filtered_available:
            self.act_show_filtered.setChecked(False)
        self.plot_filtered_spectrum.setVisible(
            filtered_available and self.act_show_filtered.isChecked()
        )

        self.act_show_baseline.setEnabled(baseline_available)
        if not baseline_available:
            self.act_show_baseline.setChecked(False)
        self.plot_baseline_curve.setVisible(
            baseline_available and self.act_show_baseline.isChecked()
        )
        if spectrum is not None:
            self._apply_spectrum_limits(include_overlays=True)

    
    def _CEDX_multi_fit(self):
        if self.RUN is None:
            return print("no CED X LOAD")
        self.RUN.Spectra[self.index_spec] = self.Spectrum

        index_start = self.index_start_entry.value()
        index_stop = self.index_stop_entry.value()

        # Vérification et ajustement des indices
        index_start = max(0, index_start)
        index_stop = min(len(self.RUN.Spectra) - 1, index_stop)

        if index_start > index_stop:
            print("Index start must be less than or equal to index stop.")
            return
        skip_ui=self.skip_ui_update_checkbox.isChecked()
        total_steps = max(0, index_stop - index_start)
        progress_dialog = None
        if total_steps > 0:
            progress_dialog = ProgressDialog(
                "Ajustement des spectres en cours...",
                "STOP",
                0,
                total_steps,
                self,
            )
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setAutoClose(True)
            progress_dialog.show()
            QApplication.processEvents()
        try:
            for step, i in enumerate(range(index_start, index_stop), start=1):
                if progress_dialog is not None:
                    if progress_dialog.wasCanceled():
                        break
                    progress_dialog.setLabelText(
                        f"Spectre {i + 1} ({step}/{total_steps})"
                    )
                    QApplication.processEvents()

                self.bit_bypass = False
                if skip_ui:
                    self.Spectrum = self.RUN.Spectra[i]
                    self._last_spectrum_load_bypass = True
                    self._rebuild_fit_state(skip_ui=skip_ui)
                else:
                    self.f_Spectrum_Load(Spectrum=self.RUN.Spectra[i])
                try:
                    self.bit_bypass = True
                    self.FIT_lmfitVScurvfit(skip_ui=skip_ui)
                except Exception as e:
                    print("Error_fit:", e)
                self.bit_bypass = False
                self.RUN.Spectra[i] = self.Spectrum


                if progress_dialog is not None:
                    progress_dialog.setValue(step)
                    QApplication.processEvents()
        finally:
            if progress_dialog is not None:
                if not progress_dialog.wasCanceled():
                    progress_dialog.setValue(total_steps)
                progress_dialog.close()

        self.Spectrum=self.RUN.Spectra[self.index_spec]
        self.REFRESH()

    def _ensure_gauge_clipboard_zone(self) -> Optional[pg.LinearRegionItem]:
        if not hasattr(self, "ax_diff_int"):
            return None
        if self._gauge_clipboard_zone is None:
            zone = pg.LinearRegionItem(
                values=[0, 0],
                orientation=pg.LinearRegionItem.Vertical,
                brush=pg.mkBrush(0, 120, 255, 40),
                movable=True,
            )
            zone.setZValue(10)
            self.ax_diff_int.addItem(zone)
            zone.setVisible(False)
            self._gauge_clipboard_zone = zone
        return self._gauge_clipboard_zone

    def _default_gauge_zone_range(self) -> Optional[Tuple[float, float]]:
        if self.RUN is None:
            return None
        time_values = getattr(self, "time", None)
        if time_values is None or len(time_values) == 0:
            start = float(getattr(self, "index_spec", 0))
            dt = 1.0
        else:
            start = float(self._get_cedx_time_value(getattr(self, "index_spec", 0)))
            dt = self._get_cedx_dt(getattr(self, "index_spec", 0))
            if dt is None and len(time_values) > 1:
                dt = float(np.nanmean(np.diff(np.asarray(time_values, dtype=float))))
            if dt is None or not np.isfinite(dt) or dt <= 0:
                dt = 1.0
        end = start + dt
        return (start, end) if start <= end else (end, start)

    def _get_gauge_selection_indices(self) -> List[int]:
        if self.RUN is None:
            return []
        zone = self._gauge_clipboard_zone
        if zone is None or not zone.isVisible():
            return []
        start, end = sorted(map(float, zone.getRegion()))
        time_values = getattr(self, "time", None)
        if time_values is None or len(time_values) == 0:
            time_values = np.arange(len(self.RUN.Spectra), dtype=float)
        indices = []
        max_len = min(len(time_values), len(self.RUN.Spectra))
        for idx in range(max_len):
            value = time_values[idx]
            try:
                time_value = float(value)
            except Exception:
                continue
            if start <= time_value <= end:
                indices.append(idx)
        return indices

    def _format_spectra_indices(self, indices: Sequence[int], per_line: int = 12) -> str:
        if not indices:
            return "-"
        lines = []
        for i in range(0, len(indices), per_line):
            lines.append(", ".join(str(idx) for idx in indices[i : i + per_line]))
        return "\n".join(lines)

    def copy_gauge_models_to_clipboard(self) -> None:
        spectrum = getattr(self, "Spectrum", None)
        gauges = getattr(spectrum, "Gauges", None) if spectrum is not None else None
        if not gauges:
            self.text_box_msg.setText("No gauge model to copy.")
            return
        self._gauge_clipboard = copy.deepcopy(gauges)
        zone = self._ensure_gauge_clipboard_zone()
        if zone is not None:
            default_range = self._default_gauge_zone_range()
            if default_range is not None:
                zone.setRegion(default_range)
            zone.setVisible(True)
        self.text_box_msg.setText(f"Gauge model copied ({len(gauges)}).")

    def show_gauge_selection_zone(self) -> None:
        zone = self._ensure_gauge_clipboard_zone()
        if zone is None:
            return
        if not zone.isVisible():
            default_range = self._default_gauge_zone_range()
            if default_range is not None:
                zone.setRegion(default_range)
        zone.setVisible(True)
        self.text_box_msg.setText("Zone de sélection active.")

    def hide_gauge_selection_zone(self) -> None:
        zone = self._gauge_clipboard_zone
        if zone is not None:
            zone.setVisible(False)
            self.text_box_msg.setText("Zone de sélection masquée.")

    def paste_gauge_models_from_clipboard(self) -> None:
        if self.RUN is None:
            self.text_box_msg.setText("No RUN loaded.")
            return
        if not self._gauge_clipboard:
            self.text_box_msg.setText("No gauge model copied.")
            return
        indices = self._get_gauge_selection_indices()
        if not indices:
            self.text_box_msg.setText("No spectrum in selection zone.")
            return

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Paste gauge models")
        selection_text = self._format_spectra_indices(indices)
        msg_box.setText(
            "Spectres dans la zone:\n"
            f"{selection_text}\n\n"
            "Confirmer le collage des jauges ?"
        )
        v_button = msg_box.addButton("Validate", QMessageBox.AcceptRole)
        a_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)
        msg_box.setDefaultButton(v_button)

        def on_key_press(event):
            if event.key() == Qt.Key_V:
                v_button.click()
            elif event.key() == Qt.Key_C:
                a_button.click()

        msg_box.keyPressEvent = on_key_press
        msg_box.exec_()

        if msg_box.clickedButton() != v_button:
            return

        for idx in indices:
            spectrum = self.RUN.Spectra[idx]
            spectrum.Gauges = copy.deepcopy(self._gauge_clipboard)
            spectrum.bit_fit = False
            for gauge in spectrum.Gauges:
                self._register_gauge_from_run(gauge)

        if getattr(self, "index_spec", None) in indices:
            self.f_Spectrum_Load(Spectrum=self.RUN.Spectra[self.index_spec])

        self._update_cedx_plots_from_run(reset_legend=True)
        self.text_box_msg.setText(
            f"Gauges collées sur {len(indices)} spectres."
        )

    def delete_gauges_in_zone(self) -> None:
        if self.RUN is None:
            self.text_box_msg.setText("No RUN loaded.")
            return
        indices = self._get_gauge_selection_indices()
        if not indices:
            self.text_box_msg.setText("No spectrum in selection zone.")
            return
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Dell gauge models")
        selection_text = self._format_spectra_indices(indices)
        msg_box.setText(
            "Spectres dans la zone:\n"
            f"{selection_text}\n\n"
            "Confirmer la suppression des jauges ?"
        )
        v_button = msg_box.addButton("Validate", QMessageBox.AcceptRole)
        a_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)
        msg_box.setDefaultButton(v_button)

        def on_key_press(event):
            if event.key() == Qt.Key_V:
                v_button.click()
            elif event.key() == Qt.Key_C:
                a_button.click()

        msg_box.keyPressEvent = on_key_press
        msg_box.exec_()

        if msg_box.clickedButton() != v_button:
            return

        for idx in indices:
            spectrum = self.RUN.Spectra[idx]
            spectrum.Gauges = []
            spectrum.bit_fit = False

        if getattr(self, "index_spec", None) in indices:
            self.f_Spectrum_Load(Spectrum=self.RUN.Spectra[self.index_spec])

        self._update_cedx_plots_from_run(reset_legend=True)
        self.text_box_msg.setText(
            f"Gauges supprimées sur {len(indices)} spectres."
        )
 
#########################################################################################################################################################################################
#? COMMANDE LOAD IT
    
    def _cedx_symbol_palette(self):
        return ["o", "s", "t", "+", "p", "x", "d", "^", "star", "triangle"]

    def _assign_cedx_symbol(self):
        palette = self._cedx_symbol_palette()
        symbol = palette[self._cedx_symbol_index % len(palette)]
        self._cedx_symbol_index += 1
        return symbol

    def _get_gauge_color(self, gauge_name):
        if not gauge_name:
            return (200, 200, 200)
        index = self._ensure_gauge_known(gauge_name)
        if index < 0:
            return (200, 200, 200)
        try:
            return self.gauge_colors[index]
        except (IndexError, TypeError):
            return (200, 200, 200)

    def _ensure_gauge_known(self, gauge_name, element_ref=None):
        if not gauge_name:
            return -1
        if not hasattr(self, "liste_type_Gauge") or self.liste_type_Gauge is None:
            self.liste_type_Gauge = []
        if gauge_name not in self.liste_type_Gauge:
            self.liste_type_Gauge.append(gauge_name)
            self._gauge_library_dirty = True
        index = self.liste_type_Gauge.index(gauge_name)
        self._ensure_gauge_color_capacity(index)
        if element_ref is not None:
            bibli = getattr(self.ClassDRX, "Bibli_elements", None) if hasattr(self, "ClassDRX") else None
            stored_ref = copy.deepcopy(element_ref)
            if bibli is not None and gauge_name not in bibli:
                bibli[gauge_name] = copy.deepcopy(stored_ref)
                self._gauge_library_dirty = True
            if getattr(self, "_runtime_gauge_elements", None) is None:
                self._runtime_gauge_elements = {}
            self._runtime_gauge_elements[gauge_name] = stored_ref
        return index

    def _ensure_gauge_color_capacity(self, index):
        if not hasattr(self, "gauge_colors") or self.gauge_colors is None:
            self.gauge_colors = []
        palette = getattr(self, "_gauge_color_palette", None)
        if palette is None:
            palette = list(mcolors.TABLEAU_COLORS.values()) if hasattr(mcolors, "TABLEAU_COLORS") else []
            if not palette:
                palette = [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                    "#bcbd22",
                    "#17becf",
                ]
            self._gauge_color_palette = palette
        while len(self.gauge_colors) <= index:
            color = palette[len(self.gauge_colors) % len(palette)]
            self.gauge_colors.append(color)

    def _register_gauge_from_run(self, gauge):
        gauge_name = getattr(gauge, "name", None)
        element_ref = getattr(gauge, "Element_ref", None)
        if element_ref is None:
            element_ref = getattr(gauge, "element_ref", None)
        return self._ensure_gauge_known(gauge_name, element_ref)

    def _clear_pic_fills(self):
        for lp in getattr(self, "plot_pic_fit", []):
            for pic in lp:
                try:
                    self.ax_spectrum.removeItem(pic)
                except Exception:
                    pass
        self.plot_pic_fit = []
        self.plot_pic_fit_curves = []

    def _format_pic_label(self, gauge_index: int, pic_index: int) -> str:
        name = self.Nom_pic[gauge_index][pic_index]
        params = self.Param0[gauge_index][pic_index]
        return (
            f"{name}   X0:{params[0]}   Y0:{params[1]}   "
            f"sigma:{params[2]}   Coef:{params[3]} ; Modele:{params[4]}"
        )

    def _create_pic_fill(
        self,
        x_data=None,
        bottom_data=None,
        top_data=None,
        brush=None,
    ):
        x_values = [] if x_data is None else x_data
        bottom_values = [] if bottom_data is None else bottom_data
        top_values = [] if top_data is None else top_data

        bottom_curve = pg.PlotCurveItem(x_values, bottom_values)
        top_curve = pg.PlotCurveItem(x_values, top_values)
        fill_item = pg.FillBetweenItem(bottom_curve, top_curve)
        if brush is not None:
            fill_item.setBrush(brush)
        return fill_item, bottom_curve, top_curve

    def _get_pic_curves(self, gauge_index: int, pic_index: int):
        curves_per_gauge = getattr(self, "plot_pic_fit_curves", [])
        if 0 <= gauge_index < len(curves_per_gauge):
            gauge_curves = curves_per_gauge[gauge_index]
            if 0 <= pic_index < len(gauge_curves):
                return gauge_curves[pic_index]
        return None, None

    def _update_pic_fill_data(
        self,
        gauge_index: int,
        pic_index: int,
        x_data,
        top_data,
        bottom_data=None,
    ) -> None:
        bottom_curve, top_curve = self._get_pic_curves(gauge_index, pic_index)
        if bottom_curve is None or top_curve is None:
            return

        x_values = [] if x_data is None else x_data
        top_values = [] if top_data is None else top_data
        if bottom_data is None:
            bottom_values = np.zeros_like(np.asarray(top_values))
        else:
            bottom_values = bottom_data

        bottom_curve.setData(x_values, bottom_values)
        top_curve.setData(x_values, top_values)

    def _replace_listbox_item(self, index: int, text: str):
        if not hasattr(self, "listbox_pic"):
            return
        if index < 0:
            return
        count = self.listbox_pic.count()
        if index < count:
            self.listbox_pic.takeItem(index)
            self.listbox_pic.insertItem(index, text)
        else:
            self.listbox_pic.insertItem(count, text)

    def _update_pic_display(
        self,
        gauge_index: int,
        pic_index: int,
        y_plot,
        *,
        store_fit: bool = True,
        update_listbox: bool = False,
    ):
        if store_fit:
            self.list_y_fit_start[gauge_index][pic_index] = y_plot
        self._update_pic_fill_data(
            gauge_index,
            pic_index,
            self.Spectrum.wnb,
            y_plot,
            np.zeros_like(self.Spectrum.blfit),
        )
        label = self._format_pic_label(gauge_index, pic_index)
        self.list_text_pic[gauge_index][pic_index] = label
        if update_listbox and gauge_index == self.index_jauge:
            self._replace_listbox_item(pic_index, label)

    def _refresh_gauge_views(self, spectrum=None):
        spectrum = spectrum or self.Spectrum
        gauges = getattr(spectrum, "Gauges", []) or []

        nb_j = len(gauges)
        self.Nom_pic = [[] for _ in range(nb_j)]
        self.list_text_pic = [[] for _ in range(nb_j)]
        self.Param0 = [[] for _ in range(nb_j)]
        self.J = [0 for _ in range(nb_j)]
        self.list_y_fit_start = [[] for _ in range(nb_j)]
        self.list_name_gauges = []

        self._clear_pic_fills()
        self.plot_pic_fit = [[] for _ in range(nb_j)]
        self.plot_pic_fit_curves = [[] for _ in range(nb_j)]

        if nb_j == 0:
            self._current_spectrum_gauges = set()
            self._apply_gauge_presence_colors()
            return

        baseline_values = np.zeros_like(spectrum.wnb)

        for i, gauge in enumerate(gauges):
            self.list_name_gauges.append(gauge.name)
            color_index = self._register_gauge_from_run(gauge)
            if 0 <= color_index < len(self.gauge_colors):
                gauge_color_value = self.gauge_colors[color_index]
            else:
                gauge_color_value = (200, 200, 200)

            for j, pic in enumerate(gauge.pics):
                self.Nom_pic[i].append(pic.name)
                new_P0, param = pic.Out_model()
                self.Param0[i].append(new_P0 + [pic.model_fit])
                y_plot = pic.model.eval(param, x=spectrum.wnb)
                self.list_y_fit_start[i].append(y_plot)
                self.J[i] += 1
                pic_index = len(self.Param0[i]) - 1
                self.list_text_pic[i].append(self._format_pic_label(i, pic_index))

                brush = QColor(gauge_color_value)
                brush.setAlpha(100)
                pic_fill, bottom_curve, top_curve = self._create_pic_fill(
                    spectrum.wnb,
                    baseline_values,
                    y_plot,
                    brush,
                )

                self.plot_pic_fit[i].append(pic_fill)
                self.plot_pic_fit_curves[i].append((bottom_curve, top_curve))
                self.ax_spectrum.addItem(pic_fill)

        self._current_spectrum_gauges = set(self.list_name_gauges)
        self._apply_gauge_presence_colors()

    def _refresh_gauge_fit_cache(self, spectrum=None):
        spectrum = spectrum or self.Spectrum
        gauges = getattr(spectrum, "Gauges", []) or []

        nb_j = len(gauges)
        self.Nom_pic = [[] for _ in range(nb_j)]
        self.list_text_pic = [[] for _ in range(nb_j)]
        self.Param0 = [[] for _ in range(nb_j)]
        self.J = [0 for _ in range(nb_j)]
        self.list_y_fit_start = [[] for _ in range(nb_j)]
        self.list_name_gauges = []

        if nb_j == 0:
            return

        for i, gauge in enumerate(gauges):
            self.list_name_gauges.append(gauge.name)
            for pic in gauge.pics:
                self.Nom_pic[i].append(pic.name)
                new_P0, param = pic.Out_model()
                self.Param0[i].append(new_P0 + [pic.model_fit])
                y_plot = pic.model.eval(param, x=spectrum.wnb)
                self.list_y_fit_start[i].append(y_plot)
                self.J[i] += 1
                pic_index = len(self.Param0[i]) - 1
                self.list_text_pic[i].append(self._format_pic_label(i, pic_index))

    def _refresh_gauge_library_if_needed(self):
        if getattr(self, "_gauge_library_dirty", False):
            self._sync_library_state()
            self.update_gauge_table()
            self._gauge_library_dirty = False

    def _apply_gauge_presence_colors(self):
        if not hasattr(self, "gauge_table"):
            return
        current = getattr(self, "_current_spectrum_gauges", set())
        row_count = self.gauge_table.rowCount() if hasattr(self.gauge_table, "rowCount") else 0
        use_presence_colors = bool(current)
        for row in range(row_count):
            status_item = self.gauge_table.item(row, 1)
            if status_item is None:
                continue
            gauge_item = self.gauge_table.item(row, 0)
            gauge_name = gauge_item.text() if gauge_item is not None else ""
            if use_presence_colors:
                color = "lightgreen" if gauge_name in current else "orange"
            else:
                color = "white"
            status_item.setBackground(QColor(color))

    def _get_data_CEDX(self,RUN):
        l_P, l_dP, l_t, l_sigma_P = [], [], [], []
        l_T, l_sigma_T = [], []
        n_J = []
        gauge_lookup = {}
        gauge_indices = []
        spectra = getattr(RUN, "Spectra", []) or []
        for i, spec in enumerate(spectra):
            gauges = getattr(spec, "Gauges", []) or []
            for gauge in gauges:
                name = gauge.name
                if not getattr(spec, "bit_fit", False):
                    continue
                if name not in gauge_lookup:
                    symbol, color = self._cedx_gauge_meta.get(name, (None, None))
                    if symbol is None:
                        symbol = self._assign_cedx_symbol()
                    if color is None:
                        color = self._get_gauge_color(name)
                    gauge_lookup[name] = len(l_P)
                    n_J.append([name, symbol, color])
                    l_P.append([])
                    l_dP.append([])
                    l_t.append([])
                    l_sigma_P.append([])
                    gauge_indices.append([])
                    self._cedx_gauge_meta[name] = (symbol, color)
                index = gauge_lookup[name]
                study = getattr(spec, "study", pd.DataFrame())
                column_name = "P_" + name
                if column_name in study.columns:
                    p = float(study[column_name])
                    l_sigma_P[index].append(abs(p * 0.1 + 0.1))
                else:
                    p = 0
                    l_sigma_P[index].append(0)
                if l_P[index] and i > 0:
                    try:
                        dt = (RUN.Time_spectrum[i] - RUN.Time_spectrum[i - 1]) * 1e3
                    except Exception:
                        dt = None
                    if dt:
                        l_dP[index].append((p - l_P[index][-1]) / dt)
                    else:
                        l_dP[index].append(np.nan)
                l_P[index].append(p)
                if RUN.Time_spectrum is not None and i < len(RUN.Time_spectrum):
                    l_t[index].append(RUN.Time_spectrum[i] * 1e3)
                else:
                    l_t[index].append(float(i))
                gauge_indices[index].append(i)
       

        if RUN.Time_spectrum is not None:
            
            dt_n= len(RUN.Time_spectrum)-len(RUN.Spectra)
            if dt_n>0:
                Time = [RUN.Time_spectrum[i]*1e3 for i in range(len(RUN.Spectra))]
            elif dt_n<0:
                dt=np.mean(np.diff(RUN.Time_spectrum))*1e3
                Time=np.array(list(np.array(RUN.Time_spectrum)*1e3)+[RUN.Time_spectrum[-1] + dt*(1+i) for i in range(abs(dt_n))])
            else:
                Time=np.array(RUN.Time_spectrum)*1e3

        else:
            Time=[i for i in range(len(RUN.Spectra))]  # Si Time_spectrum est None, on utilise l'index des fichiers DRX

        spectre_number=[i for i in range(len(RUN.Spectra))]


        if RUN.data_oscillo is not None:

            time_amp=np.array(RUN.data_oscillo['Time']*1e3)
            b=np.array(RUN.data_oscillo['Channel3'])
            amp=CL.savgol_filter(b,101,2)
        else:
            time_amp,amp=[],[]

        return l_P,l_sigma_P,l_t,l_sigma_T,Time,spectre_number,time_amp,amp,n_J,l_dP,gauge_indices

    def _update_cedx_image(self, run, time_axis):
        spectra = getattr(run, "Spectra", []) or []
        valid_entries = []
        theta_ranges = {}
        for idx, spec in enumerate(spectra):
            theta = getattr(spec, "wnb", None)
            intensity = getattr(spec, "spec", None)
            if theta is None or intensity is None or not len(theta):
                continue
            valid_entries.append((idx, theta, intensity))
            theta_ranges[idx] = (float(theta[0]), float(theta[-1]))

        if not valid_entries:
            self._cedx_image_cache = None
            self._cedx_image_row_map = {}
            self._cedx_spectrum_theta_range = {}
            self._cedx_image_trimmed_length = 0
            self._cedx_theta_bounds = None
            self._cedx_levels = None
            return None

        theta_min = max(entry[1][0] for entry in valid_entries)
        theta_max = min(entry[1][-1] for entry in valid_entries)
        if theta_min >= theta_max:
            self._cedx_image_cache = None
            self._cedx_image_row_map = {}
            self._cedx_spectrum_theta_range = {}
            self._cedx_image_trimmed_length = 0
            self._cedx_theta_bounds = None
            self._cedx_levels = None
            return None

        previous_shape = None if self._cedx_image_cache is None else self._cedx_image_cache.shape
        previous_levels = self._cedx_levels

        spectra_trimmed = []
        row_map = {}
        for spec_idx, theta, intensity in valid_entries:
            theta_arr = np.asarray(theta, dtype=float)
            intensity_arr = np.asarray(intensity, dtype=float)
            mask = (theta_arr > theta_min) & (theta_arr < theta_max)
            trimmed = intensity_arr[mask]
            if trimmed.size == 0:
                trimmed = np.full((1,), np.nan, dtype=float)
            row_map[spec_idx] = len(spectra_trimmed)
            spectra_trimmed.append(trimmed)

        min_length = min(col.size for col in spectra_trimmed)
        if min_length == 0:
            self._cedx_image_cache = None
            self._cedx_image_row_map = {}
            self._cedx_spectrum_theta_range = {}
            self._cedx_image_trimmed_length = 0
            self._cedx_theta_bounds = None
            self._cedx_levels = None
            return None

        trimmed_columns = [col[:min_length] for col in spectra_trimmed]
        image_data = np.stack(trimmed_columns, axis=0).astype(float) + 5.0
        self._cedx_image_cache = image_data
        self._cedx_image_row_map = row_map
        self._cedx_spectrum_theta_range = theta_ranges
        self._cedx_image_trimmed_length = int(min_length)
        self._cedx_theta_bounds = (float(theta_min), float(theta_max))
        self._update_image_safe(self.img_diff_int_item, image_data, levels=previous_levels)

        level_min = 6.0
        if previous_levels is None or previous_shape != image_data.shape:
            level_max = float(np.nanmax(image_data)) if image_data.size else level_min
            level_max = max(level_max, level_min)
            self._cedx_levels = (level_min, level_max)
        elif self._cedx_levels is not None:
            current_max = float(np.nanmax(image_data)) if image_data.size else level_min
            if current_max > self._cedx_levels[1]:
                self._cedx_levels = (level_min, current_max)
        if self._cedx_levels is not None:
            self.img_diff_int_item.setLevels(self._cedx_levels)

        time_array = np.asarray(time_axis, dtype=float) if time_axis is not None else np.asarray([])
        if time_array.size:
            x_min = float(np.nanmin(time_array))
            x_max = float(np.nanmax(time_array))
        else:
            x_min = 0.0
            x_max = float(len(spectra_trimmed) - 1) if spectra_trimmed else 1.0

        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if theta_max < theta_min:
            theta_min, theta_max = theta_max, theta_min

        rect = pg.QtCore.QRectF(x_min, theta_min, x_max - x_min, theta_max - theta_min)
        self.img_diff_int_item.setRect(rect)

        if self._cedx_levels is not None:
            self.img_diff_int_item.setLevels(self._cedx_levels)

        self.ax_diff_int.setLimits(xMin=x_min, xMax=x_max)
        self.ax_diff_int.setXRange(x_min, x_max)
        self.ax_diff_int.invertY(True)
        return theta_min, theta_max

    def _update_cedx_image_incremental(self, spectrum_index):
        if self.RUN is None:
            return
        if self._cedx_image_cache is None or self._cedx_theta_bounds is None:
            self._update_cedx_image(self.RUN, getattr(self, "time", None)*1e3)
            return

        spectra = getattr(self.RUN, "Spectra", []) or []
        if spectrum_index < 0 or spectrum_index >= len(spectra):
            return

        spectrum = spectra[spectrum_index]
        theta = getattr(spectrum, "wnb", None)
        intensity = getattr(spectrum, "spec", None)
        bit_fit = getattr(spectrum, "bit_fit", False)
        row_map = self._cedx_image_row_map
        has_row = spectrum_index in row_map

        if not bit_fit or theta is None or intensity is None or not len(theta):
            if has_row:
                self._update_cedx_image(self.RUN, getattr(self, "time", None)*1e3)
            return

        theta_min, theta_max = self._cedx_theta_bounds
        self._cedx_spectrum_theta_range[spectrum_index] = (float(theta[0]), float(theta[-1]))

        global_min = max(val[0] for val in self._cedx_spectrum_theta_range.values())
        global_max = min(val[1] for val in self._cedx_spectrum_theta_range.values())
        if global_min > theta_min or global_max < theta_max:
            self._update_cedx_image(self.RUN, getattr(self, "time", None)*1e3)
            return

        theta_arr = np.asarray(theta, dtype=float)
        intensity_arr = np.asarray(intensity, dtype=float)
        mask = (theta_arr > theta_min) & (theta_arr < theta_max)
        trimmed = intensity_arr[mask]
        if trimmed.size == 0:
            trimmed = np.full((1,), np.nan, dtype=float)

        min_length = self._cedx_image_trimmed_length
        if min_length <= 0:
            self._update_cedx_image(self.RUN, getattr(self, "time", None)*1e3)
            return
        if trimmed.size < min_length or not has_row:
            self._update_cedx_image(self.RUN, getattr(self, "time", None)*1e3)
            return

        row_index = row_map[spectrum_index]
        trimmed = trimmed[:min_length].astype(float) + 5.0
        self._cedx_image_cache[row_index, :] = trimmed

        if self._cedx_levels is not None:
            row_max = float(np.nanmax(trimmed)) if trimmed.size else self._cedx_levels[0]
            if row_max > self._cedx_levels[1]:
                self._cedx_levels = (self._cedx_levels[0], row_max)
                self.img_diff_int_item.setLevels(self._cedx_levels)

        self.img_diff_int_item.updateImage()

    def _get_cedx_time_value(self, spectrum_index):
        time_array = getattr(self, "time", None)
        if time_array is None or spectrum_index < 0:
            return float(spectrum_index)
        try:
            return float(time_array[spectrum_index])
        except Exception:
            return float(spectrum_index)

    def _get_cedx_dt(self, spectrum_index):
        if spectrum_index <= 0:
            return None
        time_array = getattr(self, "time", None)
        if time_array is None:
            return None
        if spectrum_index >= len(time_array):
            return None
        try:
            dt = float(time_array[spectrum_index]) - float(time_array[spectrum_index - 1])
        except Exception:
            return None
        if not np.isfinite(dt) or abs(dt) < 1e-12:
            return None
        return dt

    def _recompute_cedx_derivatives_for_gauge(self, name):
        series = self._cedx_gauge_series.get(name)
        if not series:
            return
        pressures = series.get("pressure", [])
        indices = series.get("spectra_indices", [])
        derivs = []
        for pos in range(1, len(pressures)):
            spec_index = indices[pos]
            dt = self._get_cedx_dt(spec_index)
            if dt is None:
                derivs.append(np.nan)
                continue
            prev_val = pressures[pos - 1]
            curr_val = pressures[pos]
            derivs.append((curr_val - prev_val) / dt)
        series["deriv"] = derivs

    def _update_cedx_plot_item_for_gauge(self, name):
        items = self._cedx_gauge_items.get(name)
        series = self._cedx_gauge_series.get(name)
        if items is None or series is None:
            return

        pressure_x = np.asarray(series.get("time", []), dtype=float)
        pressure_y = np.asarray(series.get("pressure", []), dtype=float)
        deriv_y = np.asarray(series.get("deriv", []), dtype=float)
        deriv_x = pressure_x[1:] if pressure_x.size else np.asarray([])

        pressure_item = items.get("pressure")
        deriv_item = items.get("deriv")
        if pressure_item is not None:
            self._update_curve_safe(pressure_item, pressure_x, pressure_y)
        if deriv_item is not None:
            self._update_curve_safe(deriv_item, deriv_x, deriv_y)

    def update_cedx_from_spectrum(self, spectrum_index):
        if self.RUN is None:
            return
        spectra = getattr(self.RUN, "Spectra", []) or []
        if spectrum_index < 0 or spectrum_index >= len(spectra):
            return

        if not getattr(self, "_cedx_gauge_series", None) or not getattr(self, "_cedx_gauge_items", None):
            self._update_cedx_plots_from_run()
            return

        spectrum = spectra[spectrum_index]
        time_value = self._get_cedx_time_value(spectrum_index)
        new_gauges = {}
        if getattr(spectrum, "bit_fit", False):
            study = getattr(spectrum, "study", pd.DataFrame())
            gauges = getattr(spectrum, "Gauges", []) or []
            for gauge in gauges:
                name = gauge.name
                column = f"P_{name}"
                if column in study.columns:
                    pressure = float(study[column])
                else:
                    pressure = 0.0
                new_gauges[name] = pressure

        # Update or add gauges present in the spectrum
        for name, pressure in new_gauges.items():
            series = self._cedx_gauge_series.get(name)
            if series is None:
                symbol, color = self._cedx_gauge_meta.get(name, (None, None))
                if symbol is None:
                    symbol = self._assign_cedx_symbol()
                if color is None:
                    color = self._get_gauge_color(name)
                pressure_item = pg.ScatterPlotItem(
                    x=np.asarray([time_value], dtype=float),
                    y=np.asarray([pressure], dtype=float),
                    symbol=symbol,
                    pen=pg.mkPen(color),
                    brush=pg.mkBrush(color),
                    size=10,
                    name=name,
                )
                deriv_item = pg.ScatterPlotItem(
                    x=np.asarray([], dtype=float),
                    y=np.asarray([], dtype=float),
                    symbol=symbol,
                    pen=pg.mkPen(color),
                    brush=pg.mkBrush(color),
                    size=10,
                    name=name + " dP",
                )
                self.ax_P.addItem(pressure_item)
                self.ax_dPdt.addItem(deriv_item)
                self._cedx_gauge_items[name] = {"pressure": pressure_item, "deriv": deriv_item, "symbol": symbol, "color": color}
                self._cedx_gauge_meta[name] = (symbol, color)
                self._cedx_gauge_series[name] = {
                    "pressure": [pressure],
                    "time": [time_value],
                    "deriv": [],
                    "spectra_indices": [spectrum_index],
                }
                continue

            indices = series.setdefault("spectra_indices", [])
            pressures = series.setdefault("pressure", [])
            times = series.setdefault("time", [])

            if spectrum_index in indices:
                pos = indices.index(spectrum_index)
                pressures[pos] = pressure
                times[pos] = time_value
            else:
                insert_pos = np.searchsorted(indices, spectrum_index)
                indices.insert(insert_pos, spectrum_index)
                pressures.insert(insert_pos, pressure)
                times.insert(insert_pos, time_value)

            self._recompute_cedx_derivatives_for_gauge(name)
            self._update_cedx_plot_item_for_gauge(name)

        # Remove gauges that are no longer present
        gauges_to_remove = []
        for name, series in self._cedx_gauge_series.items():
            indices = series.get("spectra_indices", [])
            if spectrum_index in indices and name not in new_gauges:
                pos = indices.index(spectrum_index)
                indices.pop(pos)
                series.get("pressure", []).pop(pos)
                series.get("time", []).pop(pos)
                self._recompute_cedx_derivatives_for_gauge(name)
                if not indices:
                    gauges_to_remove.append(name)
                else:
                    self._update_cedx_plot_item_for_gauge(name)

        for name in gauges_to_remove:
            items = self._cedx_gauge_items.pop(name, {})
            self._safe_remove_plot_item(self.ax_P, items.get("pressure"))
            self._safe_remove_plot_item(self.ax_dPdt, items.get("deriv"))
            self._cedx_gauge_series.pop(name, None)

        self.plot_P = [[items.get("pressure"), items.get("deriv")] for items in self._cedx_gauge_items.values()]

        pressure_series = [np.asarray(series.get("pressure", []), dtype=float) for series in self._cedx_gauge_series.values()]
        deriv_series = [np.asarray(series.get("deriv", []), dtype=float) for series in self._cedx_gauge_series.values()]
        self._update_cedx_mean_curve()

        time_amp = []
        amp_arr = None
        if getattr(self.RUN, "data_oscillo", None) is not None:
            time_amp = np.asarray(self.RUN.data_oscillo.get("Time", []), dtype=float) * 1e3
            channel = np.asarray(self.RUN.data_oscillo.get("Channel3", []), dtype=float)
            if time_amp.size and channel.size:
                try:
                    amp_arr = CL.savgol_filter(channel, 101, 2)
                except Exception:
                    amp_arr = channel
                if self.plot_piezo is None:
                    self.plot_piezo = pg.PlotCurveItem(x=time_amp, y=amp_arr, pen="b")
                    getattr(self, "ax_P_piezo", self.ax_P).addItem(self.plot_piezo)
                else:
                    self.plot_piezo.setData(time_amp, amp_arr)
            elif self.plot_piezo is not None:
                self.plot_piezo.setData([], [])
        elif self.plot_piezo is not None:
            self.plot_piezo.setData([], [])

        time_attr = getattr(self, "time", None)
        time_display = np.asarray(time_attr, dtype=float) if time_attr is not None else np.asarray([])
        time_limits = time_display
        if isinstance(time_amp, np.ndarray) and time_amp.size:
            time_limits = np.concatenate((time_limits, time_amp)) if time_limits.size else time_amp

        self._reset_cedx_plot_limits(
            time_display,
            time_limits,
            pressure_series,
            deriv_series,
            amp=amp_arr,
            theta_range=self._cedx_theta_bounds,
            mean_curve=self._cedx_mean_curve_data,
        )

        self._update_cedx_image_incremental(spectrum_index)
        self._refresh_gauge_library_if_needed()
    
    def _update_cedx_plot_items(
        self,
        run,
        Time,
        time_amp,
        amp,
        n_J,
        l_P,
        l_t,
        l_dP,
        gauge_indices,
        reset_legend=False,
    ):
        if reset_legend and self.ax_dPdt.legend is not None:
            try:
                self.ax_dPdt.legend.scene().removeItem(self.ax_dPdt.legend)
            except Exception:
                pass
            self.ax_dPdt.legend = None

        if self.ax_dPdt.legend is None:
            self.ax_dPdt.addLegend()

        new_items = {}
        order = []
        for idx, (name, symbol, color) in enumerate(n_J):
            color_pen = pg.mkPen(color)
            color_brush = pg.mkBrush(color)
            pressure_x = np.asarray(l_t[idx], dtype=float)
            pressure_y = np.asarray(l_P[idx], dtype=float)
            deriv_x = pressure_x[1:] if pressure_x.size else np.asarray([])
            deriv_y = np.asarray(l_dP[idx], dtype=float)

            if name in self._cedx_gauge_items:
                pressure_item = self._cedx_gauge_items[name]["pressure"]
                deriv_item = self._cedx_gauge_items[name]["deriv"]
                pressure_item.setData(x=pressure_x, y=pressure_y)
                pressure_item.setSymbol(symbol)
                pressure_item.setPen(color_pen)
                pressure_item.setBrush(color_brush)
                deriv_item.setData(x=deriv_x, y=deriv_y)
                deriv_item.setSymbol(symbol)
                deriv_item.setPen(color_pen)
                deriv_item.setBrush(color_brush)
            else:
                pressure_item = pg.ScatterPlotItem(
                    x=pressure_x,
                    y=pressure_y,
                    symbol=symbol,
                    pen=color_pen,
                    brush=color_brush,
                    size=10,
                    name=name,
                )
                deriv_item = pg.ScatterPlotItem(
                    x=deriv_x,
                    y=deriv_y,
                    symbol=symbol,
                    pen=color_pen,
                    brush=color_brush,
                    size=10,
                    name=name + " dP",
                )
                self.ax_P.addItem(pressure_item)
                self.ax_dPdt.addItem(deriv_item)

            new_items[name] = {"pressure": pressure_item, "deriv": deriv_item, "symbol": symbol, "color": color}
            order.append(name)

        for name, items in list(self._cedx_gauge_items.items()):
            if name not in new_items:
                self._safe_remove_plot_item(self.ax_P, items.get("pressure"))
                self._safe_remove_plot_item(self.ax_dPdt, items.get("deriv"))

        self._cedx_gauge_items = new_items
        self.plot_P = [[new_items[name]["pressure"], new_items[name]["deriv"]] for name in order]

        gauge_series = {}
        for idx, (name, _symbol, _color) in enumerate(n_J):
            gauge_series[name] = {
                "pressure": list(np.asarray(l_P[idx], dtype=float)),
                "time": list(np.asarray(l_t[idx], dtype=float)),
                "deriv": list(np.asarray(l_dP[idx], dtype=float)),
                "spectra_indices": list(gauge_indices[idx]),
            }
        self._cedx_gauge_series = gauge_series
        self._update_cedx_mean_curve()

        time_amp_arr = np.asarray(time_amp, dtype=float) if time_amp is not None else None
        amp_arr = np.asarray(amp, dtype=float) if amp is not None else None
        piezo_axis = getattr(self, "ax_P_piezo", self.ax_P)

        if (
            time_amp_arr is not None
            and amp_arr is not None
            and time_amp_arr.size > 0
            and amp_arr.size > 0
        ):
            if self.plot_piezo is None:
                self.plot_piezo = pg.PlotCurveItem(x=time_amp_arr, y=amp_arr, pen="b")
                piezo_axis.addItem(self.plot_piezo)
            else:
                self.plot_piezo.setData(time_amp_arr, amp_arr)
        elif self.plot_piezo is not None:
            self.plot_piezo.setData([], [])

        theta_range = self._update_cedx_image(run, Time)

        pressure_series = [np.asarray(series, dtype=float) for series in l_P]
        deriv_series = [np.asarray(series, dtype=float) for series in l_dP]
        time_display = np.asarray(Time, dtype=float) if Time is not None else np.asarray([])
        time_limits = time_display
        if time_amp is not None and len(time_amp):
            time_amp_array = np.asarray(time_amp, dtype=float)
            if time_amp_array.size:
                time_limits = np.concatenate((time_limits, time_amp_array)) if time_limits.size else time_amp_array
        amp_array = None
        if amp is not None and len(amp):
            amp_array = np.asarray(amp, dtype=float)
        self._reset_cedx_plot_limits(
            time_display,
            time_limits,
            pressure_series,
            deriv_series,
            amp=amp_array,
            theta_range=theta_range,
            mean_curve=self._cedx_mean_curve_data,
        )
        self._refresh_gauge_library_if_needed()

    def _update_cedx_plots_from_run(self, reset_legend=False):
        if self.RUN is None:
            return
        (
            l_P,
            l_sigma_P,
            l_t,
            l_sigma_T,
            Time,
            spectre_number,
            time_amp,
            amp,
            n_J,
            l_dP,
            gauge_indices,
        ) = self._get_data_CEDX(self.RUN)
        Time = np.asarray(Time, dtype=float)
        self.time = Time
        self.spectre_number = spectre_number
        self._update_cedx_plot_items(
            run=self.RUN,
            Time=Time,
            time_amp=time_amp,
            amp=amp,
            n_J=n_J,
            l_P=l_P,
            l_t=l_t,
            l_dP=l_dP,
            gauge_indices=gauge_indices,
            reset_legend=reset_legend,
        )

    def f_CEDX_Load(self,item,objet_run=None):
        name_select=None
        if objet_run is None:
            # Fonction pour charger l'objet Python à partir du fichier sélectionné
            chemin_fichier = os.path.join(self.dict_folders["CED"], item.text())
            #self.liste_objets_widget.setCurrentRow(item)
            index = self.listbox_file.row(item)
            objet_run = CL.LOAD_CEDd(chemin_fichier)
            name_select=item.text()

        else:
            index=0

        if type(objet_run) is CL.CED_DRX:
            print("PRINT CEDd objet_run", objet_run)
            self.RUN = objet_run
            try:
                name_select = os.path.basename(self.RUN.CEDd_path)
            except:
                pass
            self._clear_cedx_plot_items()
            self._cedx_gauge_items = {}
            self._cedx_gauge_meta = {}
            self._cedx_symbol_index = 0

            (
                l_P,
                l_sigma_P,
                l_t,
                l_sigma_T,
                Time,
                spectre_number,
                time_amp,
                amp,
                n_J,
                l_dP,
                gauge_indices,
            ) = self._get_data_CEDX(self.RUN)
            Time = np.asarray(Time, dtype=float)
            self.time = Time
            self.spectre_number = spectre_number
            self._update_cedx_plot_items(
                run=self.RUN,
                Time=Time,
                time_amp=time_amp,
                amp=amp,
                n_J=n_J,
                l_P=l_P,
                l_t=l_t,
                l_dP=l_dP,
                gauge_indices=gauge_indices,
                reset_legend=True,
            )

            #print("LOAD id_select:",index,"list_id",self.list_index_file_CED_Load)
            #self.axc2l2.clear()
            #self.axc2l2=self.RUN.Spectra[0].Print(ax=#self.axc2l2)
            #self.CLEAR_ALL()
            #self.RUN=objet_run
            self.DRX_selector.clear()
            for i in range(len(self.RUN.Spectra)):
                self.DRX_selector.addItem(f"drx_{i}")
            self.index_spec=0
            self.DRX_selector.setCurrentIndex(self.index_spec)
            self.bit_bypass=True
            self.f_Spectrum_Load()
            self.bit_bypass=False
            self._refresh_drx_view()

        self.label_CED.setText( "CEDd "+name_select+" select")
        #print("LOAD id_select:",index,"list_id",self.list_index_file_CED_Load)   

    def _save_current_spectrum_to_run(self):
        if getattr(self, "bit_bypass", False):
            logger.debug("Bypass actif : pas de sauvegarde du spectre courant")
            return True

        if getattr(self, "Spectrum", None) is None:
            logger.debug("Aucun spectre courant à sauvegarder")
            return True

        if not getattr(self, "RUN", None):
            logger.debug("Aucun RUN chargé : impossible de sauvegarder le spectre courant")
            return False

        try:
            self.RUN.Spectra[self.index_spec] = self.Spectrum
            logger.debug(
                "Spectre courant sauvegardé dans RUN à l'index %s", self.index_spec
            )
            self.update_cedx_from_spectrum(self.index_spec)
        except Exception:
            self.text_box_msg.setText("ERROR Load Spec")
            logger.exception(
                "Impossible de sauvegarder le spectre courant dans le RUN"
            )
            return False

        return True

    def _load_spectrum_from_run(self, index, previous_spectrum):
        if not getattr(self, "RUN", None):
            logger.debug("Aucun RUN disponible pour charger un spectre")
            return False

        try:
            current_spec = self.RUN.Spectra[index]
        except Exception:
            logger.exception("Impossible de récupérer le spectre RUN[%s]", index)
            self.text_box_msg.setText("ERROR Load Spec")
            raise

        logger.debug("bit_fit RUN[%s] : %s", index, getattr(current_spec, "bit_fit", None))
        logger.debug(
            "bit_fit précédent : %s", getattr(previous_spectrum, "bit_fit", None)
        )

        bypass = False
        light_clone = getattr(current_spec, "light_copy", None)
        if getattr(current_spec, "bit_fit", False):
            logger.info("→ Référence directe du spectre RUN (bit_fit=True)")
            self.Spectrum = current_spec
        else:
            cloned_spec = light_clone() if callable(light_clone) else copy.copy(current_spec)
            if getattr(previous_spectrum, "bit_fit", False):
                logger.info(
                    "→ Fusion des jauges/modèles depuis le spectre précédent vers le RUN"
                )
                self.Spectrum = cloned_spec
                self.Spectrum.Gauges = list(getattr(previous_spectrum, "Gauges", []))
                self.Spectrum.model = getattr(previous_spectrum, "model", None)
                bypass = True
            else:
                logger.info("→ Remplacement du spectre courant par celui du RUN")
                self.Spectrum = cloned_spec

        return bypass

    def _rebuild_fit_state(self,skip_ui=False):
        if getattr(self, "Spectrum", None) is None:
            logger.debug("Aucun spectre chargé : reconstruction de l'état fit ignorée")
            return

        bypass = getattr(self, "_last_spectrum_load_bypass", False)
        gauges = getattr(self.Spectrum, "Gauges", None) or []

        if not (
            getattr(self.Spectrum, "bit_fit", False)
            or bypass
            or getattr(self, "bit_bypass", False)
            or any(getattr(gauge, "bit_fit", False) for gauge in gauges)
        ):
            logger.debug("Aucun fit à reconstruire pour ce spectre")
            return

        nb_j = len(gauges)
        self.Zone_fit = [None] * nb_j
        self.X_s = [None] * nb_j
        self.X_e = [None] * nb_j
        self.bit_fit = [False] * nb_j

        self.Zspec1, self.Zspec2 = None, None

        for i in range(nb_j):
            self.Zone_fit[i] = gauges[i].indexX
            if i == 0:
                self.Zone_fit[i] = self.Spectrum.indexX

        if self.Spectrum.indexX is not None:
            ym = min(self.Spectrum.spec)
            yM = max(self.Spectrum.spec)

            for i in range(nb_j):
                if gauges[i].indexX is not None:
                    self.X_s[i] = self.Spectrum.wnb[gauges[i].indexX][0]
                    self.X_e[i] = self.Spectrum.wnb[gauges[i].indexX][-1]
                    if i == 0 and not skip_ui:
                        self.X_s[0] = self.Spectrum.wnb[self.Spectrum.indexX][0]
                        self.X_e[0] = self.Spectrum.wnb[self.Spectrum.indexX][-1]

                        left = pg.PlotCurveItem([self.Spectrum.wnb[0], self.X_s[0]], [yM, yM])
                        right = pg.PlotCurveItem([self.X_e[0], self.Spectrum.wnb[-1]], [yM, yM])
                        baseline = pg.PlotCurveItem(
                            self.Spectrum.wnb, np.full_like(self.Spectrum.wnb, ym)
                        )
                        self.Zspec1 = pg.FillBetweenItem(
                            baseline, left, brush=pg.mkBrush(0, 1, 2, 50)
                        )
                        self.Zspec2 = pg.FillBetweenItem(
                            baseline, right, brush=pg.mkBrush(0, 1, 2, 50)
                        )
                        self.ax_spectrum.addItem(self.Zspec1)
                        self.ax_spectrum.addItem(self.Zspec2)

                    pg.PlotCurveItem([self.Spectrum.wnb[0], self.X_s[i]], [yM, yM])
                    pg.PlotCurveItem([self.X_e[i], self.Spectrum.wnb[-1]], [yM, yM])
                    pg.PlotCurveItem(
                        self.Spectrum.wnb, np.full_like(self.Spectrum.wnb, ym)
                    )

        if self.Spectrum.model is None and gauges:
            for j, gauge in enumerate(gauges):
                gauge.Update_model()
                if j == 0:
                    self.Spectrum.model = gauge.model
                else:
                    self.Spectrum.model += gauge.model
        if not skip_ui:
            self._refresh_gauge_views()
            self.Print_fit_start(skip_ui=skip_ui)

        if hasattr(self, "plot_raw_spectrum"):
            self._update_spectrum_overlay_data()

    def _sync_filter_controls(self):
        if getattr(self, "Spectrum", None) is None:
            logger.debug("Aucun spectre disponible pour synchroniser les filtres")
            return

        index = self.filtre_type_selector.findText(self.Spectrum.type_filtre)
        if index != -1:
            self.filtre_type_selector.setCurrentIndex(index)

        self.param_filtre_1_entry.setText(str(self.Spectrum.param_f[0]))
        self.param_filtre_2_entry.setText(str(self.Spectrum.param_f[1]))
        self.deg_baseline_entry.setValue(self.Spectrum.deg_baseline)
        self.Baseline_spectrum()

    def _select_first_gauge(self):
        if getattr(self, "Spectrum", None) is None:
            logger.debug("Aucun spectre disponible pour sélectionner une jauge")
            self.index_jauge = -1
            self._refresh_gauge_library_if_needed()
            return

        gauges = getattr(self.Spectrum, "Gauges", None) or []
        if not gauges:
            logger.debug("Aucune jauge dans le spectre")
            self.index_jauge = -1
            self._refresh_gauge_library_if_needed()
            return

        self.index_jauge = 0
        self._select_gauge_in_table(gauges[self.index_jauge].name)
        self.f_Gauge_Load()
        if hasattr(self, "plot_pic_select") and hasattr(self, "baseline"):
            self.plot_pic_select.setCurves(
                self.baseline, pg.PlotCurveItem([], [])
            )
        self._refresh_gauge_library_if_needed()

    def f_Spectrum_Load(self, item=None, Spectrum=None):  # - - - LOAD SPEC (r) - - - #
        logger.info("DEBUT chargement du spectre")

        previous_spectrum = self.Spectrum
        nb_j_old = (
            len(previous_spectrum.Gauges)
            if getattr(previous_spectrum, "Gauges", None)
            else 0
        )
        logger.info("Nombre de jauges précédentes : %s", nb_j_old)

        bypass = False
        if Spectrum is None:
            logger.debug("Aucun spectre passé en argument")
            if not self._save_current_spectrum_to_run():
                return

            if not getattr(self, "bit_bypass", False):
                selected_index = self.DRX_selector.currentIndex()
                logger.debug("Index sélectionné dans DRX_selector : %s", selected_index)
                if selected_index != -1 and selected_index != self.index_spec:
                    logger.debug(
                        "Changement index_spec : %s → %s", self.index_spec, selected_index
                    )
                    self.index_spec = selected_index
                    self.DRX_selector.setCurrentIndex(self.index_spec)

            try:
                bypass = self._load_spectrum_from_run(
                    self.index_spec, previous_spectrum
                )
            except Exception:
                return
        else:
            logger.info("Un spectre a été passé en argument → mise à jour directe")
            self.Spectrum = Spectrum
            bypass = True
            self.bit_bypass = False

        self._last_spectrum_load_bypass = bypass
        self._rebuild_fit_state()
        self._sync_filter_controls()
        self._select_first_gauge()

        logger.info("FIN chargement du spectre")

    def f_Gauge_Load(self): # - - - SELECET JAUGE - - -#
        required_attrs = (
            "spinbox_T",
            "lamb0_entry",
            "name_spe_entry",
            "spinbox_P",
            "listbox_pic",
            "list_text_pic",
            "gauge_controller",
        )
        if any(not hasattr(self, attr) for attr in required_attrs):
            return

        self.bit_load_jauge=False
        self.bit_modif_jauge=True

        self.spinbox_T.setEnabled(False)
        self.spinbox_T.setValue(293)
        self.deltalambdaT=0
        self.lamb0_entry.setText(str(self.Spectrum.Gauges[self.index_jauge].lamb0))
        self.name_spe_entry.setText(str(self.Spectrum.Gauges[self.index_jauge].name_spe))
        p=self.Spectrum.Gauges[self.index_jauge].P
        self.spinbox_P.setValue(p) 

        self.listbox_pic.clear()
        for name in self.list_text_pic[self.index_jauge]:
            self.listbox_pic.addItem(name)
        self._select_gauge_in_table(self.Spectrum.Gauges[self.index_jauge].name)
        self.Spectrum.Gauges[self.index_jauge].Element_ref.P_start=p
        self.Spectrum.Gauges[self.index_jauge].Element_ref=self.gauge_controller.f_Gauge_Load(copy.deepcopy(self.Spectrum.Gauges[self.index_jauge].Element_ref))
  
    def f_Gauge_Add_in_Spectrum(self): # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ADD JAUGE - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        row = self.gauge_table.currentRow()
        new_g = self.gauge_table.item(row, 0).text() if row >= 0 else None
        if new_g not in self.list_name_gauges:
            self.bit_modif_jauge =False
        if self.bit_modif_jauge is True :
            return print("if you want this gauges DELL AND RELOAD")
        self.bit_load_jauge=False
        new_g=self.gauge_select #self.ClassDRX.Bibli_elements[new_g]
        print(self.gauge_select.save_var)
        new_element = Element(new_g,name=new_g.name)
        new_element.P=self.spinbox_P.value()

        #new_Jauge.lamb_fit=new_Jauge.inv_f_P(new_Jauge.P)
        self.Spectrum.Gauges.append(new_element)

        self.index_jauge=len(self.Spectrum.Gauges)-1
        self.Update_var(new_element.name) #self.list_name_gauges.append(new_g) in update var
        self.f_Gauge_Load()
        self.gauge_controller.refresh_fixed_lines(self.gauge_select)

        # récupérer l'item de la 2ᵉ colonne
        status_item = self.gauge_table.item(row, 1)
        if status_item is not None:
            status_item.setBackground(QColor("lightgreen"))  # ta couleur custom
        self.Auto_pic()

#########################################################################################################################################################################################
#? COMMANDE UNLOAD
    def save_summary_CED(self):
        self.RUN.Summary["Time"]=self.RUN.Time_spectrum
        self.RUN.Summary.to_csv(os.path.basename(self.RUN.CEDd_path), index=False)
        #self.RUN.Summary.to_csv(os.path.join(self.folder_outpout,CL.os.path.basename(self.RUN.CEDd_path)), index=False)
        
    

    def CLEAR_CEDd(self):
        self._clear_cedx_plot_items()
        self._cedx_gauge_items = {}
        self._cedx_gauge_meta = {}
        self._cedx_symbol_index = 0

        for plot in self.plot_spe:
            plot.clear()
        self.plot_spe = []

        if self.plot_piezo is not None:
            self.plot_piezo.setData([],[])

        self.time = []
        self.spectre_number = []
        self.CLEAR_ALL(empty=True)

    def Dell_Jauge(self):# - - - DELL JAUGE- - -#
        if self.index_jauge == -1:
            return print("jauge not select")
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Warning dell Jauge")
        text='You going to dell '+ self.Spectrum.Gauges[self.index_jauge].name+'\n Press "v" for Validate "c" for Cancel'
        msg_box.setText(text)

        v_button = msg_box.addButton("Validate", QMessageBox.AcceptRole)
        a_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

        msg_box.setDefaultButton(v_button)

        def on_key_press(event):
            if event.key() == Qt.Key_V:
                v_button.click()
            elif event.key() == Qt.Key_C:
                a_button.click()

        msg_box.keyPressEvent = on_key_press
        msg_box.exec_()
        

        if msg_box.clickedButton() == v_button:

            if self.index_jauge==0 and len(self.Param0)>2:
                self.X_s[self.index_jauge+1],self.X_e[self.index_jauge+1],self.Zone_fit[self.index_jauge+1]=self.X_s[self.index_jauge],self.X_e[self.index_jauge],self.Zone_fit[self.index_jauge]

            self.Click_Clear()
            del(self.Nom_pic[self.index_jauge])
            del(self.Param0[self.index_jauge])
            del(self.list_text_pic[self.index_jauge])
            del(self.J[self.index_jauge])
            for idx, pic in enumerate(self.plot_pic_fit[self.index_jauge]):
                bottom_curve, top_curve = self._get_pic_curves(self.index_jauge, idx)
                if bottom_curve is not None:
                    bottom_curve.setData([], [])
                if top_curve is not None:
                    top_curve.setData([], [])
                try:
                    self.ax_spectrum.removeItem(pic)
                except Exception:
                    pass
            del(self.plot_pic_fit[self.index_jauge])
            del(self.plot_pic_fit_curves[self.index_jauge])
            del(self.list_y_fit_start[self.index_jauge])
            del(self.list_name_gauges[self.index_jauge])
            try:
                del(self.Param_FIT[self.index_jauge]) 
            except Exception as e:
                print("del(Param_FIT[J])",e)
            del(self.X_s[self.index_jauge])
            del(self.X_e[self.index_jauge])
            del(self.Zone_fit[self.index_jauge])
            #del(self.bit_plot_fit[self.index_jauge])

            #del(self.bit_fit[self.index_jauge])
            del(self.Spectrum.Gauges[self.index_jauge])
           
            self.text_box_msg.setText('JAUGE DELL')
            # récupérer l'item de la 2ᵉ colonne
            row = self.gauge_table.currentRow()
            status_item = self.gauge_table.item(row, 1)
            if status_item is not None:
                status_item.setBackground(QColor("lightred"))  # ta couleur custom
            self.Print_fit_start()
            self.index_jauge-=1
            if len(self.Param0) != 0:
                gauge_name = self.list_name_gauges[self.index_jauge]
                self._select_gauge_in_table(gauge_name)
                self.f_Gauge_Load()
                self.gauge_controller.refresh_fixed_lines(self.gauge_select)
            else:
                self.f_gauge_select()
                self.Spectrum.bit_fit=False
                self.gauge_controller.refresh_fixed_lines(self.gauge_select)
            self.update_gauge_table()
        else:
            print("Function stopped.")
    
    def init_session_vars(self):
        self.selected_file = None
        self.index_jauge = -1
        self.Nom_pic = []
        self.spec_fit = []
        self.Param0 = []
        self.Param_FIT = []
        self.y_fit_start = None
        self.save_value = 0
        self.J = []
        self.X_s = []
        self.X_e = []
        self.Zone_fit = []
        self.list_text_pic = []
        self._current_spectrum_gauges = set()
        self.X0 = 0
        self.Y0 = 0
        self.Zspec1 = None
        self.Zspec2 = None
        self.bit_print_fit_T = False
        self.bit_fit_T = False
        self.bit_fit = []
        self.bit_modif_jauge = False
        self.bit_load_jauge = False
        self.bit_filtre = False
        self.bit_plot_fit = []
        self.plot_pic_fit = []
        self.plot_pic_fit_curves = []

    def CLEAR_ALL(self,empty=False): 
        self.gauge_controller.clear_var_widgets()
        self.index_spec=0
        self.DRX_selector.setCurrentIndex(self.index_spec)
       

        if (type(self.Spectrum) is CL.Spectre) and (empty==False):
            #self.plot_spectrum()
           #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - OBJET - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            #self.nb_jauges=len(self.Spectrum.Gauges)
            self.list_name_gauges=[jauge.name for jauge in self.Spectrum.Gauges]
            self._current_spectrum_gauges = set(self.list_name_gauges)
            print("gauges save")
        else:
            print("gauges dell")
            #self.nb_jauges=0
            self.list_name_gauges=[]
            self._current_spectrum_gauges = set()
            self.listbox_pic.clear()

        for gauge_index, l_pic in enumerate(self.plot_pic_fit):
            for pic_index, pic in enumerate(l_pic):
                bottom_curve, top_curve = self._get_pic_curves(gauge_index, pic_index)
                if bottom_curve is not None:
                    bottom_curve.setData([], [])
                if top_curve is not None:
                    top_curve.setData([], [])
                try:
                    self.ax_spectrum.removeItem(pic)
                except Exception:
                    pass
        self.plot_pic_fit = []
        self.plot_pic_fit_curves = []
        self.plot_blfit= None
        self.list_y_fit_start=[]
        self.gauge_controller.clear_lines(clear_fixed=True)
        self.lines=[]
        self.plot_raw_spectrum.setData([],[])
        self.plot_filtered_spectrum.setData([],[])
        self.plot_baseline_curve.setData([],[])
        self.update_spectrum_overlays()
        self._apply_gauge_presence_colors()
        self.plot_ax4.setData([],[]) 
        self.plot_fit_start.setData([],[])
        self.plot_data_fit.setData([],[]) 
        self.plot_zoom.setData([],[])  
        self.baseline.setData([],[])  
        self.plot_ax3.setCurves(pg.PlotCurveItem([],[]),pg.PlotCurveItem([],[]))
        self.plot_data_pic_solo.setData([],[])
        self.axV.setPos(self.X0)        
        self.axH.setPos(self.Y0)
        # Zoom area line (optionnel)
        self.cross_zoom.setPos(0,0) 
        self._clear_cedx_plot_items()
    
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - FILE- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        self.init_session_vars()


    def DEBUG_SPECTRUM(self):
        self.CLEAR_ALL()
        self.bit_bypass=True
        self.f_Spectrum_Load()
        self.bit_bypass=False
#########################################################################################################################################################################################
#? COMMANDE SAVE
    def try_find_peak(self):
        x = self.Spectrum.wnb
        y = copy.deepcopy(self.Spectrum.y_corr)

        # Lire les valeurs des entrées Qt
        height = self.height_entry.value()
        distance = self.distance_entry.value()
        prominence = self.prominence_entry.value()
        width = self.width_entry.value()
        number_peak_max = self.nb_peak_entry.value() if hasattr(self, "nb_peak_entry") else 10


        theta2_range = self.spectrum_controller.update_theta2_range()
        y_mask = []
        last_valid = None
        for yi, xi in zip(y, x):
            if any(a <= xi <= b for a, b in theta2_range):
                last_valid = yi
                y_mask.append(yi)
            else:
                y_mask.append(last_valid if last_valid is not None else 0)
        
        # Appel de la fonction de détection
        _, peak_x, result = self.ClassDRX.F_Find_peaks(
            x, y_mask,
            height=height*max(y_mask),
            distance=distance,
            prominence=prominence*max(y_mask),
            width=width,
            number_peak_max=number_peak_max
        )

        # Supprimer l'affichage précédent s'il existe
        if hasattr(self, "peak_plot_item") and self.peak_plot_item is not None:
            self.ax_spectrum.removeItem(self.peak_plot_item)
            self.peak_plot_item = None

        # Ajouter les nouveaux pics détectés
        peak_y = [y[np.argmin(np.abs(x - px))] for px in peak_x]  # approx ordonnée
        self.peak_plot_item = self.ax_spectrum.plot(
            peak_x,
            peak_y,
            pen=None,
            symbol='o',
            symbolSize=8,
            symbolBrush='r',
            name="Peaks"
        )

        print(f"{len(peak_x)} pics détectés et affichés. \n {result}")

    def clear_gauges_range(self):
        if not hasattr(self, "RUN") or self.RUN is None:
            self.text_box_msg.setText("No RUN loaded.")
            return

        spectra = getattr(self.RUN, "Spectra", []) or []
        if not spectra:
            self.text_box_msg.setText("No spectra loaded.")
            return

        index_start = max(0, self.index_start_entry.value())
        index_stop = min(len(spectra) - 1, self.index_stop_entry.value())

        if index_start > index_stop:
            QMessageBox.warning(
                self,
                "Index invalides",
                "Index start doit être inférieur ou égal à index stop.",
            )
            return

        message = (
            f"Effacer toutes les jauges pour les spectres {index_start} à {index_stop} ?"
        )
        response = QMessageBox.question(
            self,
            "Confirmation",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return

        for i in range(index_start, index_stop + 1):
            spectrum = spectra[i]
            if spectrum is None:
                continue
            spectrum.Gauges = []
            spectrum.study = pd.DataFrame()
            spectrum.bit_fit = False
            spectrum.fit = "Fit Non effectué"

        summary = getattr(self.RUN, "Summary", None)
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            if "n°Spec" in summary.columns:
                self.RUN.Summary = (
                    summary.loc[
                        ~summary["n°Spec"].between(index_start, index_stop)
                    ].reset_index(drop=True)
                )

        if index_start <= self.index_spec <= index_stop:
            self.f_Spectrum_Load(Spectrum=self.RUN.Spectra[self.index_spec])
            self.REFRESH()
    
    def CREAT_empty_CEDd_from_loaded_files(self):
        if not hasattr(self, "loaded_file_DRX") or self.loaded_file_DRX is None:
            print("Aucun fichier DRX chargé.")
            return
        if self.calib is None:
            print("Calibration absente : impossible de créer un CEDd.")
            return
        data_oscillo=None
        if self.jungfrau_mode !="oscillo":
            try:
                scan_f_name=self.listbox_file.currentItem().text()
                h5name=os.path.dirname(self.loaded_file_DRX).replace('/'+scan_f_name,"").split("/")[-1] + '.h5'
                path_file_h5=os.path.dirname(self.loaded_file_DRX).replace(scan_f_name,"") +h5name 
                
                scan_lbl = str(int(scan_f_name.replace("scan", ""))) + ".1"
                print(scan_lbl,'->',path_file_h5)
                with File(path_file_h5) as f:
                    t2 = f[scan_lbl]["measurement"]["ch6_time"][:].ravel()
                    y2 = f[scan_lbl]["measurement"]["ch6"][:].ravel()
                    t1 = f[scan_lbl]["measurement"]["ch5_time"][:].ravel()
                    y1 = f[scan_lbl]["measurement"]["ch5"][:].ravel()
                
                if not (len(t1) == len(y1) == len(y2)):
                    raise ValueError(
                    f"Tailles incohérentes : "
                    f"t1={len(t1)}, y1={len(y1)}, y2={len(y2)}"
                )

                data_oscillo = pd.DataFrame(
                    {
                    "Time": np.array(t1),
                    "Channel2": np.array(y1),
                    "Channel3": np.array(y2),
                    }
                    )
            except Exception as e:
                print("ERROR h5:",e)

        self.CLEAR_CEDd()

        CEDX = CL.CED_DRX(
            self.loaded_file_DRX,
            self.calib,
            self.ClassDRX.E,
            data_oscillo=data_oscillo,
            time_index="Channel2",
            deg_baseline=int(self.deg_baseline_entry.value())
        )
        if self.jungfrau_mode =="continue":
            CEDX.Time_spectrum= np.array([i*1e-3 for i in range(len(CEDX.Spectra))])
            CEDX.time_index=np.array([i for i in range(len(CEDX.Spectra))])
        self.RUN = CEDX
        self.RUN.CEDd_path = os.path.join(
            self.dict_folders["CED"],
            f"{self._get_cedx_base_name()}.CEDX",
        )

        print(f"CED vide créé avec {len(CEDX.Spectra)} spectres. Pas de recherche de compos effectué.")
        self.f_CEDX_Load(objet_run=self.RUN, item=None)

    def _CEDX_auto_compo(self):
        if not hasattr(self, "RUN") or self.RUN is None:
            print("Aucun RUN chargé.")
            return

        skip_ui=self.skip_ui_update_checkbox.isChecked()
        try:
            index_start = self.index_start_entry.value()
            index_stop = self.index_stop_entry.value()

            # Vérification et ajustement des indices
            index_start = max(0, index_start)
            index_stop = min(len(self.RUN.Spectra) - 1, index_stop)

            if index_start > index_stop:
                print("Index start must be less than or equal to index stop.")
                return

            # Paramètres find_peaks
            height = self.height_entry.value()
            distance = self.distance_entry.value()
            prominence = self.prominence_entry.value()
            width = self.width_entry.value()
            number_peak_max = self.nb_peak_entry.value()

            # Paramètres find_compo
            ngen = self.NGEN_entry.value()
            mutpb = self.MUTPB_entry.value()
            cxpb = self.CXPB_entry.value()
            popinit = self.POPINIT_entry.value()
            tolerance = self.tolerance_entry.value()
            p_range = self.p_range_entry.value()
            nb_max_element = self.nb_max_element_entry.value()

        except Exception as e:
            print("Erreur de lecture des paramètres GUI :", e)
            return

        self.RUN.Spectra[self.index_spec] = self.Spectrum

        self.RUN.Summary = pd.DataFrame()  # Réinitialisation du résumé si nécessaire
        # Récupération des Gauges précédents si disponibles
        theta_limits = list(self.calib.theta_range) if self.calib and getattr(self.calib, "theta_range", None) else list(DEFAULT_THETA_RANGE)

        if index_start > 0 and self.RUN.Spectra[index_start - 1].Gauges:
            best_ind = [(G.name,G.P) for G in self.RUN.Spectra[index_start - 1].Gauges]#[(G.name,G.P,[pic for pic, _, _, _, _ in G.Element_ref.Eos_Pdhkl(G.P, extract=True) if theta_limits[0] < pic < theta_limits[-1]]) for G in self.RUN.Spectra[index_start - 1].Gauges]
            last_pressure=np.mean([g.P for g in self.RUN.Spectra[index_start - 1].Gauges])
        else:
            best_ind = None
            last_pressure= None

        spectra_indices = list(range(index_start, index_stop + 1))
        progress_dialog = ProgressDialog(
            "Recherche automatique des compositions...",
            "Annuler",
            0,
            len(spectra_indices),
            self,
        ) if spectra_indices else None
        if progress_dialog is not None:
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setAutoClose(True)
            progress_dialog.show()
            QApplication.processEvents()

        try:
            for step, i in enumerate(spectra_indices, start=1):
                if progress_dialog is not None:
                    if progress_dialog.wasCanceled():
                        print("Annulé par l'utilisateur !")
                        break
                    progress_dialog.setLabelText(
                        f"bibDRX:{[elem for elem in self.ClassDRX.Bibli_elements]} \n Spectre {i + 1} ({step}/{len(spectra_indices)}) \n Compo de départ : {[(ind,p) for ind,p in best_ind] if best_ind is not None else None} \n P{[last_pressure * 0.75 - 2, last_pressure * 1.25 + 2] if last_pressure is not None else [-0.5, p_range] }"
                    )
                    QApplication.processEvents()

                X = self.RUN.Spectra[i]
                if skip_ui:
                    self.Spectrum = X
                else:
                    self.CLEAR_ALL(empty=False)
                    self.index_spec = i
                    self.DRX_selector.setCurrentIndex(self.index_spec)
                    self.bit_bypass = True
                    self.f_Spectrum_Load(Spectrum=X)
                    self.bit_bypass = False
                theta2_range = self.spectrum_controller.update_theta2_range()
                # Détection des pics
                x=X.wnb
                y=copy.deepcopy(X.y_corr)
                y_mask = []
                last_valid = None
                for yi, xi in zip(y, x):
                    if any(a <= xi <= b for a, b in theta2_range):
                        last_valid = yi
                        y_mask.append(yi)
                    else:
                        y_mask.append(last_valid if last_valid is not None else 0)

                try:
                    _, detected_peaks, result = self.ClassDRX.F_Find_peaks(
                        X.wnb, y_mask,
                        height=height*max(y_mask),
                        distance=distance,
                        prominence=prominence*max(y_mask),
                        width=width,
                        number_peak_max=number_peak_max
                    )
                    #print(f"[DRX {i}] Peaks found: {[f'th2:{pic} I:{i}' for pic,i in zip(detected_peaks,result['peak_heights'])]}")

                except Exception as e:
                    print("ERROR find peak :",e)
                    break
                if not skip_ui:
                    if hasattr(self, "peak_plot_item") and self.peak_plot_item is not None:
                        self.ax_spectrum.removeItem(self.peak_plot_item)
                        self.peak_plot_item = None

                    peak_y = [
                        X.y_corr[np.argmin(np.abs(X.wnb - px))] for px in detected_peaks
                    ]
                    self.peak_plot_item = self.ax_spectrum.plot(
                        detected_peaks,
                        peak_y,
                        pen=None,
                        symbol='o',
                        symbolSize=8,
                        symbolBrush='r',
                        name="Peaks"
                    )
                
                    QApplication.processEvents()
                
                best_ind, _, Gauges = self.ClassDRX.F_Find_compo(
                    detected_peaks,
                    NGEN=ngen,
                    MUTPB=mutpb,
                    CXPB=cxpb,
                    POPINIT=popinit,
                    pressure_range=[-0.5, p_range] if last_pressure == None else[last_pressure * 0.75 - 2, last_pressure * 1.25 + 2],
                    max_ecart_pressure=2,
                    max_elements=nb_max_element,
                    theta2_range=theta2_range,
                    tolerance=tolerance,
                    indiv_start=best_ind,
                    print_process=True
                )
                # Mise à jour du spectre
                X.Gauges = Gauges
                X.bit_fit = True
                
                
                
                for x,g in enumerate(X.Gauges):
                    pi=0
                    for ps,save in enumerate(g.Element_ref.save_var):
                        if save:
                            x0=g.Element_ref.thetas_PV[ps][0]
                            m=np.argmin(np.abs(X.wnb -x0 ))
                            g.pics[pi].ampH[0]=round(X.y_corr[m],3)
                            g.pics[pi].Update()
                            pi+=1
                    g.Update_model()
                    g.bit_fit=True
                X.Calcul_study(mini=True)
                
                last_pressure = np.mean([g.P for g in Gauges])
                
                # Ajout au résumé
                self.RUN.Summary = pd.concat([
                    self.RUN.Summary,
                    pd.concat([pd.DataFrame({"n°Spec": [int(i)]}), X.study], axis=1)
                ], ignore_index=True)
                if progress_dialog is not None:
                    progress_dialog.setValue(step)
                    QApplication.processEvents()

        # ... (fin de ta méthode) ...
        finally:
            if progress_dialog is not None:
                if not progress_dialog.wasCanceled():
                    progress_dialog.setValue(len(spectra_indices))
                progress_dialog.close()
        #print(f"RUN mis à jour pour les spectres {index_start} à {index_stop}.")
        self.CLEAR_CEDd()
        self.f_CEDX_Load(objet_run=self.RUN, item=None)

    def CREAT_new_Spectrum(self):
        save_gauges=[]
        if type(self.Spectrum) is CL.Spectre :
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Warning save gauges")
            text='Do you want save gauges \n Press "v" foCLEAR_ALLr save "c" delete'
            msg_box.setText(text)

            v_button = msg_box.addButton("Save", QMessageBox.AcceptRole)
            a_button = msg_box.addButton("Delete", QMessageBox.RejectRole)

            msg_box.setDefaultButton(v_button)

            def on_key_press(event):
                if event.key() == Qt.Key_V:
                    v_button.click()
                elif event.key() == Qt.Key_C:
                    a_button.click()

            msg_box.keyPressEvent = on_key_press
            msg_box.exec_()
            

            if msg_box.clickedButton() == v_button:
                save_gauges=copy.deepcopy(self.Spectrum.Gauges)
        nb_img=self.DRX_selector.currentIndex()
        self.CLEAR_ALL(empty=False)

        #file_DXR=self.list_exp[self.exp_selector.currentIndex()][1][self.run_selector.currentIndex()][2][self.DRX_selector.currentIndex()]

        if not os.path.exists(self.loaded_file_DRX):
            self.text_box_msg.setText("File not found")
            return

        img_data=fabio.open(self.loaded_file_DRX)

        if self.calib is None or getattr(self.calib, "mask", None) is None or getattr(self.calib, "ai", None) is None:
            self.text_box_msg.setText("err: calibration not available for integration")
            return

        tth, intens = Calibration.Integrate_DRX(img_data.getframe(nb_img).data, self.calib.mask, self.calib.ai, theta_range=self.calib.theta_range)

        self.text_box_msg.setText("New integration")
        self.bit_bypass=True
        self.f_Spectrum_Load(Spectrum=CL.Spectre(np.array(tth),np.array(intens),Gauges=save_gauges,deg_baseline=int(self.deg_baseline_entry.value())))
        self.bit_bypass=False

    def SAVE_CEDd(self):
        self.RUN.Spectra[self.index_spec]=self.Spectrum
        self.RUN.data_drx=None
        CL.SAVE_CEDd(self.RUN)
#########################################################################################################################################################################################
#? COMMANDE PIC

    def Update_var(self,name=None):
        self.list_name_gauges.append(name)
        self.Nom_pic.append([])
        self.spec_fit.append(None)
        self.Param0.append([])
        self.Param_FIT.append([])
        self.J.append(0) #Compteur du nombre d epic selectioné
        self.X0=0 #cordonnée X du clique
        self.Y0=0 #cordonnée Y du clique [None)
        self.X_s.append(None) #X de départ de donné selectioné
        self.X_e.append(None) #X de fin de ddonné selectionéinter_entryself.
        self.Zone_fit.append(None)
        self.bit_plot_fit.append(False) # True les fit son affiché sinon False
        self.list_text_pic.append([]) # texte qui donne les different pic enregistré
        self.bit_fit.append(True)
        self.plot_pic_fit.append([])
        self.plot_pic_fit_curves.append([])
        self.list_y_fit_start.append([])
    
    def Auto_pic_T(self):
        save_index=self.index_jauge
        for i in range(len(self.Spectrum.Gauges)):
            self.index_jauge=i
            self.Auto_pic()
        self.index_jauge=save_index

    def Auto_pic(self):
        if self.index_jauge == -1:
            print("Auto_pic bug resolve : if self.index_jauge==-1")
            return

        gauge = self.Spectrum.Gauges[self.index_jauge]
        color_index = self._register_gauge_from_run(gauge)
        gauge_color_value = (
            self.gauge_colors[color_index]
            if 0 <= color_index < len(self.gauge_colors)
            else (200, 200, 200)
        )
        elem_ref = gauge.Element_ref

        try:
            name_dhkl = [f"D{int(h)}{int(k)}{int(l)}" for (h,k,l),theta in zip(elem_ref.name_dhkl,elem_ref.thetas_PV) if self.Spectrum.x_corr[0]<theta[0]<self.Spectrum.x_corr[-1]]
        except Exception as e:
            elem_ref._debug()
            print("_debug",elem_ref.name)
            name_dhkl = [f"D{int(h)}{int(k)}{int(l)}" for (h,k,l),theta in zip(elem_ref.name_dhkl,elem_ref.thetas_PV) if self.Spectrum.x_corr[0]<theta[0]<self.Spectrum.x_corr[-1]]

        old_save_var = list(elem_ref.save_var)
        new_save_var = list(self.gauge_controller.save_var)
        n = len(name_dhkl)

        # k = index dans la liste effective des pics présents !
        k = 0

        for j in range(n):
            before = old_save_var[j]
            after = new_save_var[j]
            print(f"j={j} | old={before} | new={after} | k={k}")

            # AJOUT
            if not before and after:
                X0 = elem_ref.thetas_PV[j][0]
                m = np.argmin(np.abs(self.Spectrum.wnb - X0))
                Y0 = round(self.Spectrum.y_corr[m], 3)
                self.X0, self.Y0 = X0, Y0

                self.Nom_pic[self.index_jauge].insert(k, self.list_name_gauges[self.index_jauge] + '_' + name_dhkl[j] + '_')
                self.Param0[self.index_jauge].insert(k, j)
                self.list_text_pic[self.index_jauge].insert(k, j)
                self.list_y_fit_start[self.index_jauge].insert(k, j)
                gauge.pics.insert(k, j)
                brush = QColor(gauge_color_value)
                brush.setAlpha(100)
                base_x, base_y = (self.baseline.getData() if hasattr(self, "baseline") else ([], []))
                if base_x is None:
                    base_x = []
                if base_y is None:
                    base_y = []
                pic_fill, bottom_curve, top_curve = self._create_pic_fill(
                    base_x,
                    base_y,
                    [],
                    brush,
                )
                self.plot_pic_fit[self.index_jauge].insert(k, pic_fill)
                self.plot_pic_fit_curves[self.index_jauge].insert(k, (bottom_curve, top_curve))
                self.ax_spectrum.addItem(pic_fill)
                self.listbox_pic.insertItem(self.J[self.index_jauge] - 1, str(k))
                self.J[self.index_jauge] += 1
                self.index_pic_select = k
                self.Replace_pic()
                print("ref:",j,"pics:", k, "add in")
                k += 1

            # SUPPRESSION
            elif before and not after:
                self.index_pic_select = k
                self.Undo_pic_select()
                print("ref:",j,"pics:", k, "del")
                # NE PAS INCREMENTER k car le pic vient d'être retiré

            # 3. MODIFICATION d'un pic existant
            elif before and after:
                # Si le pic n'existe pas encore dans la liste cible, c'est un ajout déguisé
                if k >= len(self.Param0[self.index_jauge]):
                    X0 = elem_ref.thetas_PV[j][0]
                    m = np.argmin(np.abs(self.Spectrum.wnb - X0))
                    Y0 = round(self.Spectrum.y_corr[m], 3)
                    self.X0, self.Y0 = X0, Y0
                    self.Nom_pic[self.index_jauge].insert(k, self.list_name_gauges[self.index_jauge] + '_' + name_dhkl[j] + '_')
                    self.Param0[self.index_jauge].insert(k, j)
                    self.list_text_pic[self.index_jauge].insert(k, j)
                    self.list_y_fit_start[self.index_jauge].insert(k, j)
                    gauge.pics.insert(k, j)
                    brush = QColor(gauge_color_value)
                    brush.setAlpha(100)
                    base_x, base_y = (self.baseline.getData() if hasattr(self, "baseline") else ([], []))
                    if base_x is None:
                        base_x = []
                    if base_y is None:
                        base_y = []
                    pic_fill, bottom_curve, top_curve = self._create_pic_fill(
                        base_x,
                        base_y,
                        [],
                        brush,
                    )
                    self.plot_pic_fit[self.index_jauge].insert(k, pic_fill)
                    self.plot_pic_fit_curves[self.index_jauge].insert(k, (bottom_curve, top_curve))
                    self.ax_spectrum.addItem(pic_fill)
                    self.listbox_pic.insertItem(self.J[self.index_jauge] - 1, str(k))
                    self.J[self.index_jauge] += 1
                    self.index_pic_select = k
                    self.Replace_pic()
                    print("ref:",j,"pics:", k, "add (from empty)")
                    k += 1
                else:
                    X0 = elem_ref.thetas_PV[j][0]
                    m = np.argmin(np.abs(self.Spectrum.wnb - X0))
                    Y0 = round(self.Spectrum.y_corr[m], 3)
                    self.X0, self.Y0 = X0, Y0
                    self.index_pic_select = k
                    self.bit_bypass = True
                    self.Replace_pic()
                    self.bit_bypass = False
                    print("ref:",j,"pics:", k, "chg")
                    k += 1

        # Mettre à jour l'état de référence et recalculer
        elem_ref.save_var = copy.deepcopy(self.gauge_controller.save_var)
        gauge.init_ref()
        self.Spectrum.bit_fit = False
        gauge.bit_fit = False
        gauge.nb_pic = sum(self.gauge_controller.save_var)

    def Click_Confirme(self): # Fonction qui confirme le choix du pic et qui passe au suivant
        self.Nom_pic[self.index_jauge].append(self.list_name_gauges[self.index_jauge] +'_p'+str(self.J[self.index_jauge])+'_')

        self.Param0[self.index_jauge].append([self.X0,self.Y0,float(self.spinbox_sigma.value()),np.array([float(spin.value()) for spin in self.coef_dynamic_spinbox]),str(self.model_pic_fit)])
        new_name= str(self.Nom_pic[self.index_jauge][-1]) + "   X0:"+str(self.Param0[self.index_jauge][-1][0])+"   Y0:"+ str(self.Param0[self.index_jauge][-1][1]) + "   sigma:" + str(self.Param0[self.index_jauge][-1][2]) + "   Coef:" + str(self.Param0[self.index_jauge][-1][3]) +" ; Modele:" + str(self.Param0[self.index_jauge][-1][4])
        self.J[self.index_jauge]+=1
        self.text_box_msg.setText('P0 pic'+str(self.J[self.index_jauge])+': \n VIDE')
        self.list_text_pic[self.index_jauge].append(str(new_name))
        self.listbox_pic.insertItem(self.J[self.index_jauge]-1,new_name)                    
        
        X_pic=Pics(name=self.Nom_pic[self.index_jauge][-1],ctr=self.Param0[self.index_jauge][-1][0],ampH=self.Param0[self.index_jauge][-1][1],coef_spe=self.Param0[self.index_jauge][-1][3],sigma=self.Param0[self.index_jauge][-1][2],model_fit=self.Param0[self.index_jauge][-1][4])
        params=X_pic.model.make_params()
        y_plot=X_pic.model.eval(params,x=self.Spectrum.wnb)#+self.Spectrum.blfit

        self.list_y_fit_start[self.index_jauge].append(y_plot)

        self.Print_fit_start()

        pic_fill, bottom_curve, top_curve = self._create_pic_fill(
            self.Spectrum.wnb,
            np.zeros_like(self.Spectrum.blfit),
            y_plot,
            (100, 100, 255, 100),
        )
        self.plot_pic_fit[self.index_jauge].append(pic_fill)
        self.plot_pic_fit_curves[self.index_jauge].append((bottom_curve, top_curve))
        self.ax_spectrum.addItem(pic_fill)
        self.Spectrum.Gauges[self.index_jauge].pics.append(X_pic)
       
    def Click_Zone(self): # fonction qui mais en rouge la zone qu'on exclu du fit
        if self.index_jauge==-1:
            return print("Zone bug resolve : if self.index_jauge==-1")
        m=min(np.array(self.Spectrum.spec))
        M=max(np.array(self.Spectrum.spec))
        ym,yM= m,M #m-0.1*abs(m) , m
        if (type(self.Zone_fit[self.index_jauge]) == type(None) and type(self.X_s[self.index_jauge]) == type(None)) :
            self.X_s[self.index_jauge]=float(self.X0)
            self.text_box_msg.setText('Zone 1')
            if self.index_jauge==0 and self.zone_spectrum_box.isChecked():
                self.Zspec1=self.ax_spectrum.fill_between([self.Spectrum.wnb[0],self.X_s[self.index_jauge]],ym,yM, color="k", alpha=0.2)

        else:
            if (type(self.Zone_fit[self.index_jauge]) == type(None) and type(self.X_e[self.index_jauge])== type(None)):
                if float(self.X0) < self.X_s[self.index_jauge]:
                    return print("X_end < X_start")
                self.X_e[self.index_jauge]=float(self.X0)
                self.text_box_msg.setText('Zone 2')
                if self.index_jauge==0 and self.zone_spectrum_box.isChecked():
                    self.Zspec2=self.ax_spectrum.fill_between([self.X_e[self.index_jauge],self.Spectrum.wnb[-1]],ym,yM,  color="k", alpha=0.2)
                    self.Zone_fit[0] = np.where((self.Spectrum.wnb >= self.X_s[0]) & (self.Spectrum.wnb <= self.X_e[0]))[0]
                    self.Spectrum.indexX=self.Zone_fit[0]
                    #self.Spectrum.wnb= self.Spectrum.wnb[self.Zone_fit[0]]
                    self.Baseline_spectrum()
                self.Spectrum.Gauges[self.index_jauge].indexX=self.Zone_fit[self.index_jauge]

            else:
                if self.index_jauge==0 and self.zone_spectrum_box.isChecked():
                    self.Zspec1.remove()
                    self.Zspec1=None
                    self.Zspec2.remove()
                    self.Zspec2=None
                    self.Spectrum.indexX=None
                    self.Spectrum.x_corr=self.Spectrum.wnb
                    self.Zspec1, self.Zspec2 = None, None

                self.X_s[self.index_jauge],self.X_e[self.index_jauge],self.Zone_fit[self.index_jauge]=None,None,None
                self.Spectrum.Gauges[self.index_jauge].indexX=None
                self.text_box_msg.setText('Zone Clear')
           
    def Click_Clear(self): #efface tout ce quia  était fais
        if self.index_jauge==-1:
            return print(" Clear resolve : pad de jauge")
        self.Nom_pic[self.index_jauge]=[]
        self.J[self.index_jauge],self.X0, self.Y0 =0,0,0
        self.list_text_pic[self.index_jauge]=[]   
            
        if len(self.plot_pic_fit[self.index_jauge]) !=0:
            for pic_index, fill_item in enumerate(self.plot_pic_fit[self.index_jauge]):
                bottom_curve, top_curve = self._get_pic_curves(self.index_jauge, pic_index)
                if bottom_curve is not None:
                    bottom_curve.setData([], [])
                if top_curve is not None:
                    top_curve.setData([], [])
                try:
                    self.ax_spectrum.removeItem(fill_item)
                except Exception:
                    pass
        
        if self.bit_filtre== True:
            self.filtre_OFF()
    
        self.X_s[self.index_jauge],self.X_e[self.index_jauge],self.Zone_fit[self.index_jauge]=None,None,None
        self.Param0[self.index_jauge]=[]
        self.Spectrum.Gauges[self.index_jauge].pics=[]
        try:
            self.Param_FIT[self.index_jauge]=[]
        except Exception as e:
            print("Param_FIT[J]=[]",e)

        self.plot_pic_fit[self.index_jauge]=[]
        self.plot_pic_fit_curves[self.index_jauge]=[]
        self.list_y_fit_start[self.index_jauge]=[]
        self.listbox_pic.clear()
        self.text_box_msg.setText(f"GL&HF")

        self.Print_fit_start()

    def Print_fit_start(self):
        """Update fit preview plots if spectrum data are available."""
        if not hasattr(self, "Spectrum") or self.Spectrum is None:
            return

        y_corr = getattr(self.Spectrum, "y_corr", None)
        wnb = getattr(self.Spectrum, "wnb", None)
        if y_corr is None or wnb is None:
            return

        fit_checkbox = getattr(self, 'fit_start_box', None)
        fit_enabled = bool(fit_checkbox.isChecked()) if fit_checkbox is not None else True

        if fit_enabled and any(len(lst) != 0 for lst in self.list_y_fit_start):
            self.y_fit_start = None
            for i, l in enumerate(self.list_y_fit_start):
                for j, y in enumerate(l):
                    if self.y_fit_start is None:
                        self.y_fit_start = y
                    else:
                        self.y_fit_start = self.y_fit_start + y
            self.plot_fit_start.setData(self.Spectrum.wnb, self.y_fit_start)
        else:
            self.y_fit_start = None
            if hasattr(self, 'plot_fit_start'):
                self.plot_fit_start.setData([], [])

        self.ax_spectrum.disableAutoRange()         # indispensable
        self.ax_spectrum.setLimits(yMin=None, yMax=None)
        self.ax_spectrum.setYRange(min(self.Spectrum.y_corr) ,max(self.Spectrum.y_corr)*1.05)
        try:
            y_min = np.nanmin(self.Spectrum.y_corr)
            y_max = np.nanmax(self.Spectrum.y_corr)
        except ValueError:
            print("Le spectre contient des valeurs invalides (NaN); impossible d'ajuster l'axe Y.")
        else:
            if np.isnan(y_min) or np.isnan(y_max):
                print("Le spectre contient des valeurs invalides (NaN); impossible d'ajuster l'axe Y.")
            else:
                self.ax_spectrum.setYRange(y_min, y_max * 1.05)

        if self.y_fit_start is not None:
            if self.Spectrum.indexX is not None:
                
                self.Spectrum.dY=self.Spectrum.y_corr[self.Spectrum.indexX]-self.y_fit_start[self.Spectrum.indexX]
                self.plot_ax4.setData(self.Spectrum.wnb[self.Spectrum.indexX],self.Spectrum.dY)
            else:
                self.Spectrum.dY=self.Spectrum.y_corr-self.y_fit_start
                self.plot_ax4.setData(self.Spectrum.wnb,self.Spectrum.dY)

            
            self.ax_dy.setYRange(min(self.Spectrum.dY)*1.05 ,max(self.Spectrum.dY)*1.05)
        self.ax_dy.addItem(pg.InfiniteLine(angle=0, movable=False, pen='k'))

    def _safe_remove_plot_item(self, axis, item):
        """Remove *item* from *axis* without raising if it is missing."""
        if axis is None or item is None:
            return
        try:
            axis.removeItem(item)
        except Exception:
            pass

    def _clear_cedx_plot_items(self):
        """Remove previously drawn CEDX plots so they do not overlap."""
        if not hasattr(self, "ax_P"):
            return

        for plots in getattr(self, "plot_P", []):
            if not plots:
                continue
            pressure_plot = plots[0] if len(plots) > 0 else None
            deriv_plot = plots[1] if len(plots) > 1 else None
            self._safe_remove_plot_item(self.ax_P, pressure_plot)
            self._safe_remove_plot_item(self.ax_dPdt, deriv_plot)

        self.plot_P = []
        self._cedx_gauge_items = {}
        self._cedx_gauge_series = {}
        if self._cedx_mean_curve_item is not None:
            if self.ax_dPdt.legend is not None:
                try:
                    self.ax_dPdt.legend.removeItem(self._cedx_mean_curve_item.name())
                except Exception:
                    pass
            self._safe_remove_plot_item(self.ax_dPdt, self._cedx_mean_curve_item)
            self._cedx_mean_curve_item = None
        self._cedx_mean_curve_data = (np.asarray([]), np.asarray([]))
        self._cedx_image_cache = None
        self._cedx_image_row_map = {}
        self._cedx_spectrum_theta_range = {}
        self._cedx_image_trimmed_length = 0
        self._cedx_theta_bounds = None
        self._cedx_levels = None

        if hasattr(self, "plot_piezo"):
            piezo_axis = getattr(self, "ax_P_piezo", self.ax_P)
            self._safe_remove_plot_item(piezo_axis, self.plot_piezo)
            self.plot_piezo = None



    def _compute_mean_dp_curve(self, gauge_series, smooth=1, n_points=500):
        """Compute dP/dt by extrapolating the mean pressure path with a spline fit."""
        if not gauge_series:
            return np.asarray([]), np.asarray([])

        aggregated_pressures: Dict[int, List[float]] = defaultdict(list)
        time_by_index: Dict[int, float] = {}

        # Collecte des pressions et temps par index de spectre
        for series in gauge_series.values():
            pressures = series.get("pressure", []) or []
            times = series.get("time", []) or []
            indices = series.get("spectra_indices", []) or []
            for pos, pressure in enumerate(pressures):
                if pressure is None or np.isnan(pressure):
                    continue
                if pos >= len(indices):
                    continue
                spec_index = indices[pos]
                aggregated_pressures[spec_index].append(float(pressure))
                if pos < len(times):
                    time_value = times[pos]
                else:
                    time_value = spec_index
                if time_value is None or (isinstance(time_value, float) and np.isnan(time_value)):
                    time_value = spec_index
                time_by_index[spec_index] = float(time_value)

        if not aggregated_pressures:
            return np.asarray([]), np.asarray([])

        time_values: List[float] = []
        mean_pressures: List[float] = []
        for spec_index in sorted(aggregated_pressures):
            vals = [v for v in aggregated_pressures[spec_index] if not np.isnan(v)]
            if not vals:
                continue
            time_values.append(time_by_index.get(spec_index, float(spec_index)))
            mean_pressures.append(np.mean(vals))

        # Sécurité : trop peu de points pour spline
        if len(mean_pressures) < 3:
            return np.asarray(time_values, dtype=float), np.asarray([])

        time_array = np.asarray(time_values, dtype=float)
        mean_array = np.asarray(mean_pressures, dtype=float)

        # Fit spline lissée (smooth=0 => interpolation exacte)
        spline = UnivariateSpline(time_array, mean_array, s=smooth)

        # Génération d'une grille régulière de temps
        t_new = np.linspace(time_array.min(), time_array.max(), n_points)
        dpdt = spline.derivative()(t_new)

        return t_new, dpdt

    def _update_cedx_mean_curve(self):
        """Update the average dP/dt curve plotted on the derivative axis."""
        gauge_series = getattr(self, "_cedx_gauge_series", {}) or {}
        x_mean, y_mean = self._compute_mean_dp_curve(gauge_series)

        has_data = x_mean.size and y_mean.size
        if has_data:
            self._cedx_mean_curve_data = (np.asarray(x_mean, dtype=float), np.asarray(y_mean, dtype=float))
            if self._cedx_mean_curve_item is None:
                pen = pg.mkPen(color="k", width=2, style=Qt.DashLine)
                self._cedx_mean_curve_item = pg.PlotCurveItem(
                    x_mean,
                    y_mean,
                    pen=pen,
                    name="Moyenne dP/dt",
                )
                self.ax_dPdt.addItem(self._cedx_mean_curve_item)
        else:
            self._cedx_mean_curve_data = (np.asarray([]), np.asarray([]))
            if self._cedx_mean_curve_item is not None:
                if self.ax_dPdt.legend is not None:
                    try:
                        self.ax_dPdt.legend.removeItem(self._cedx_mean_curve_item.name())
                    except Exception:
                        pass
                self._safe_remove_plot_item(self.ax_dPdt, self._cedx_mean_curve_item)
                self._cedx_mean_curve_item = None

    def _reset_cedx_plot_limits(
        self,
        display_time,
        limit_time,
        pressure_series,
        deriv_series,
        amp=None,
        theta_range=None,
        mean_curve=None,
    ):
        """Rescale the CEDX axes according to the provided data."""
        if not hasattr(self, "ax_P"):
            return

        display_time = np.asarray(display_time) if display_time is not None else np.asarray([])
        limit_time = np.asarray(limit_time) if limit_time is not None else np.asarray([])

        if mean_curve is not None and len(mean_curve) == 2:
            mean_x = np.asarray(mean_curve[0], dtype=float)
            if mean_x.size:
                if limit_time.size:
                    limit_time = np.concatenate((limit_time, mean_x))
                else:
                    limit_time = mean_x
                if display_time.size:
                    display_time = np.concatenate((display_time, mean_x))
                else:
                    display_time = mean_x

        if not limit_time.size and display_time.size:
            limit_time = display_time
        if not display_time.size and limit_time.size:
            display_time = limit_time

        if limit_time.size:
            limit_min = np.nanmin(limit_time)
            limit_max = np.nanmax(limit_time)
            if limit_min == limit_max:
                t_pad = 1 if limit_min == 0 else abs(limit_min) * 0.05
                limit_min -= t_pad
                limit_max += t_pad
            self.ax_P.setLimits(xMin=limit_min, xMax=limit_max)
            self.ax_dPdt.setLimits(xMin=limit_min, xMax=limit_max)
            self.ax_diff_int.setLimits(xMin=limit_min, xMax=limit_max)

        if display_time.size:
            display_min = np.nanmin(display_time)
            display_max = np.nanmax(display_time)
            if display_min == display_max:
                t_pad = 1 if display_min == 0 else abs(display_min) * 0.05
                display_min -= t_pad
                display_max += t_pad
            self.ax_P.setXRange(display_min, display_max)
            self.ax_dPdt.setXRange(display_min, display_max)
            self.ax_diff_int.setXRange(display_min, display_max)

        def _collect(values):
            collected = []
            for series in values:
                if series is None:
                    continue
                collected.extend([v for v in series if v is not None and not np.isnan(v)])
            return collected

        pressure_values = _collect(pressure_series)
        if amp is not None:
            pressure_values.extend([v for v in amp if v is not None and not np.isnan(v)])

        if pressure_values:
            p_min = min(pressure_values)
            p_max = max(pressure_values)
            if p_min == p_max:
                pad = 1 if p_min == 0 else abs(p_min) * 0.05
                p_min -= pad
                p_max += pad
            else:
                pad = (p_max - p_min) * 0.05
                p_min -= pad
                p_max += pad
            self.ax_P.setLimits(yMin=p_min, yMax=p_max)
            self.ax_P.setYRange(p_min, p_max)

        deriv_values = _collect(deriv_series)
        if mean_curve is not None and len(mean_curve) == 2:
            mean_y = np.asarray(mean_curve[1], dtype=float)
            deriv_values.extend([v for v in mean_y if not np.isnan(v)])
        if deriv_values:
            d_min = min(deriv_values)
            d_max = max(deriv_values)
            if d_min == d_max:
                pad = 1 if d_min == 0 else abs(d_min) * 0.1
                d_min -= pad
                d_max += pad
            else:
                pad = (d_max - d_min) * 0.05
                d_min -= pad
                d_max += pad
            self.ax_dPdt.setLimits(yMin=d_min, yMax=d_max)
            self.ax_dPdt.setYRange(d_min, d_max)

        if theta_range is not None and len(theta_range) == 2:
            theta_min, theta_max = theta_range
            if theta_min is not None and theta_max is not None:
                if theta_min == theta_max:
                    pad = 1 if theta_min == 0 else abs(theta_min) * 0.05
                    theta_min -= pad
                    theta_max += pad
                self.ax_diff_int.setLimits(yMin=min(theta_min, theta_max), yMax=max(theta_min, theta_max))
                self.ax_diff_int.setYRange(min(theta_min, theta_max), max(theta_min, theta_max))

    def update_gauge_table(self):
        """Met à jour la table des gauges avec le statut inclus/exclus."""
        if not hasattr(self, "gauge_table"):
            return

        gauges = list(self.liste_type_Gauge)
        print("Update table, gauges =", gauges)  # debug
        self.gauge_table.setRowCount(len(gauges))

        for row, gauge_name in enumerate(gauges):
            print("Ajout ligne", row, gauge_name)  # debug

            # Colonne 0 : nom
            item_name = QTableWidgetItem(gauge_name)
            item_name.setFlags(item_name.flags() ^ Qt.ItemIsEditable)
            # appliquer la couleur définie dans 
            color = self.gauge_colors[row] if row < len(self.gauge_colors) else "#cccccc"
            item_name.setBackground(QColor(color))
            self.gauge_table.setItem(row, 0, item_name)

            # Colonne 1 : statut (case à cocher)
            item_status = QTableWidgetItem()
            item_status.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            if gauge_name in getattr(self.ClassDRX, "Bibli_elements", {}):
                item_status.setCheckState(Qt.Checked)
            else:
                item_status.setCheckState(Qt.Unchecked)
            self.gauge_table.setItem(row, 1, item_status)

        self._apply_gauge_presence_colors()
        self.gauge_table.resizeColumnsToContents()
        self.gauge_table.repaint()  # force le rendu
        self.gauge_table.itemChanged.connect(self._on_gauge_status_changed)

    def on_gauge_table_selection_changed(self):
        """Synchronize gauge combo box when a table row is selected."""
        if not hasattr(self, "gauge_table"):
            return
        row, gauge_name = self._get_selected_gauge_from_table()
        if not gauge_name:
            return
        self.f_gauge_select(gauge_name)

    def on_gauge_table_double_clicked(self, row, col):
        """Open the gauge selected in the overview table."""
        if self.Spectrum is None:
            return
        if row >= len(self.Spectrum.Gauges):
            return
        gauge_name = self.Spectrum.Gauges[row].name
        self._select_gauge_in_table(gauge_name)
        self.f_gauge_select(gauge_name)
        self.index_jauge = row
        self.f_Gauge_Load()
        if hasattr(self, "LOAD_Gauge"):
            self.LOAD_Gauge()

    def Undo_pic(self):
        if self.J[self.index_jauge] >0:
            del(self.Nom_pic[self.index_jauge][-1])
            del(self.Param0[self.index_jauge][-1])
            del(self.Spectrum.Gauges[self.index_jauge].pics[-1])
            self.text_box_msg.setText('P0 pic'+str(self.J[self.index_jauge])+': \n VIDE')
            del(self.list_text_pic[self.index_jauge][-1])
            self.J[self.index_jauge]-=1
            self.listbox_pic.takeItem(self.J[self.index_jauge])

            fill_item = self.plot_pic_fit[self.index_jauge].pop()
            bottom_curve, top_curve = self.plot_pic_fit_curves[self.index_jauge].pop()
            if bottom_curve is not None:
                bottom_curve.setData([], [])
            if top_curve is not None:
                top_curve.setData([], [])
            try:
                self.ax_spectrum.removeItem(fill_item)
            except Exception:
                pass
           
            self.text_box_msg.setText('UNDO PIC')
            del(self.list_y_fit_start[self.index_jauge][-1])
            self.Print_fit_start()
    
    def Undo_pic_select(self):
        if self.index_pic_select is not None and self.J[self.index_jauge] >0 :
            self.text_box_msg.setText('PIC'+ self.Nom_pic[self.index_jauge][self.index_pic_select] +' DELETED')
            del(self.Nom_pic[self.index_jauge][self.index_pic_select])
            del(self.Param0[self.index_jauge][self.index_pic_select])
            self.text_box_msg.setText('PIC'+str(self.J[self.index_jauge])+': \n VIDE')
            del(self.list_text_pic[self.index_jauge][self.index_pic_select])
            self.J[self.index_jauge]-=1
            self.listbox_pic.takeItem(self.index_pic_select)
            fill_item = self.plot_pic_fit[self.index_jauge].pop(self.index_pic_select)
            bottom_curve, top_curve = self.plot_pic_fit_curves[self.index_jauge].pop(self.index_pic_select)
            if bottom_curve is not None:
                bottom_curve.setData([], [])
            if top_curve is not None:
                top_curve.setData([], [])
            try:
                self.ax_spectrum.removeItem(fill_item)
            except Exception:
                pass
            del(self.list_y_fit_start[self.index_jauge][self.index_pic_select])
            del(self.Spectrum.Gauges[self.index_jauge].pics[self.index_pic_select])
            self.Print_fit_start()
                            
    def select_pic(self):
        
        if self.bit_bypass ==False:
            self.index_pic_select=self.listbox_pic.currentRow()
            self.X0, self.Y0=self.Param0[self.index_jauge][self.index_pic_select][0],self.Param0[self.index_jauge][self.index_pic_select][1]
            self.axV.setPos(self.X0)        
            self.axH.setPos(self.Y0)
            self.cross_zoom.setPos(self.X0,self.Y0)

        y_plot=self.list_y_fit_start[self.index_jauge][self.index_pic_select]

        brush = self._selection_highlight_color()

        self.plot_ax3.setCurves(pg.PlotCurveItem(self.Spectrum.wnb,np.zeros_like(self.Spectrum.wnb)),pg.PlotCurveItem(self.Spectrum.wnb,y_plot))
        self.plot_ax3.setBrush(brush)
        # Overlay on the main spectrum to highlight the selected peak
        baseline_curve = pg.PlotCurveItem(self.Spectrum.wnb, np.zeros_like(self.Spectrum.blfit))
        selected_curve = pg.PlotCurveItem(self.Spectrum.wnb, y_plot)
        self.plot_pic_select.setCurves(baseline_curve, selected_curve)
        self.plot_data_pic_solo.setData(self.Spectrum.wnb,np.array(self.Spectrum.spec)-(self.y_fit_start-y_plot)-self.Spectrum.blfit)
        self.ax_zoom.disableAutoRange()
        self.ax_zoom.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self.ax_zoom.setYRange(min(y_plot),max(y_plot)*1.25)
        self.ax_zoom.setXRange(self.Param0[self.index_jauge][self.index_pic_select][0]-5*self.Param0[self.index_jauge][self.index_pic_select][2],self.Param0[self.index_jauge][self.index_pic_select][0]+5*self.Param0[self.index_jauge][self.index_pic_select][2])
        n_sigma = int(self.sigma_pic_fit_entry.value())
        if self.Spectrum is not None and getattr(self.Spectrum, "wnb", None) is not None:
            wnb_array = np.asarray(self.Spectrum.wnb)
            indexX = np.where((wnb_array > self.Param0[self.index_jauge][self.index_pic_select][0]-self.Param0[self.index_jauge][self.index_pic_select][2]*n_sigma) & (wnb_array < self.Param0[self.index_jauge][self.index_pic_select][0]+self.Param0[self.index_jauge][self.index_pic_select][2]*n_sigma))[0]
        else:
            indexX = np.array([])
        self._update_fit_window(indexX=indexX)
        self.spinbox_sigma.setValue(self.Param0[self.index_jauge][self.index_pic_select][2])
        self.model_pic_fit=self.Param0[self.index_jauge][self.index_pic_select][4]
        for i,model in enumerate(self.liste_type_model_pic):
            if model == self.model_pic_fit:
                self.model_pic_type_selector.setCurrentIndex(i)
        
        self.bit_bypass=True
        self.f_model_pic_type()
        self.bit_bypass=False

        self.text_box_msg.setText('PIC SELECT'+ self.Nom_pic[self.index_jauge][self.index_pic_select])
        
        for i, spin in enumerate(self.coef_dynamic_spinbox):
            spin.setValue(self.Param0[self.index_jauge][self.index_pic_select][3][i])
          
    def f_pic_fit(self):
        n_sigma=int(self.sigma_pic_fit_entry.value())
        indexX=np.where((self.Spectrum.wnb > self.Param0[self.index_jauge][self.index_pic_select][0]-self.Param0[self.index_jauge][self.index_pic_select][2]*n_sigma) & (self.Spectrum.wnb < self.Param0[self.index_jauge][self.index_pic_select][0]+self.Param0[self.index_jauge][self.index_pic_select][2]*n_sigma))[0]
        x_sub =np.array(self.Spectrum.wnb[indexX])
        self.y_fit_start=self.y_fit_start-self.list_y_fit_start[self.index_jauge][self.index_pic_select]

        y_sub =(self.Spectrum.y_corr-self.y_fit_start)[indexX]
        inter=self.get_fit_variation()

        X_pic=Pics(name=self.Nom_pic[self.index_jauge][self.index_pic_select],ctr=self.Param0[self.index_jauge][self.index_pic_select][0],ampH=self.Param0[self.index_jauge][self.index_pic_select][1],coef_spe=self.Param0[self.index_jauge][self.index_pic_select][3],sigma=self.Param0[self.index_jauge][self.index_pic_select][2],inter=inter,Delta_ctr=self.Param0[self.index_jauge][self.index_pic_select][0]*inter/10,model_fit=self.Param0[self.index_jauge][self.index_pic_select][4])
        try:
            out = X_pic.model.fit(y_sub , x=x_sub)
            self.Param0[self.index_jauge][self.index_pic_select][:4]=X_pic.Out_model(out)
            self._update_fit_window(indexX=indexX)
            return out, X_pic ,indexX , True
        except Exception as e:
            print("f_pic_fit ERROR pic not change:",e)
            self._update_fit_window(indexX=indexX)
            return X_pic.model , X_pic,indexX , False
   
    def Replace_pic_fit(self):# - - - REPLACE PIQUE (r + maj) - - -#
        out , X_pic, indexX ,bit =self.f_pic_fit()
        self._update_fit_window(indexX=indexX)
        if bit is False :
            return self.text_box_msg.setText('PIC FIT'+ self.Nom_pic[self.index_jauge][self.index_pic_select] + ' PARAME ERROR')
        new_name= self.Nom_pic[self.index_jauge][self.index_pic_select] + "   X0:"+str(self.Param0[self.index_jauge][self.index_pic_select][0])+"   Y0:"+ str(self.Param0[self.index_jauge][self.index_pic_select][1]) + "   sigma:" + str(self.Param0[self.index_jauge][self.index_pic_select][2]) + "   Coef:" + str(self.Param0[self.index_jauge][self.index_pic_select][3]) + " ; Modele:" + str(self.Param0[self.index_jauge][self.index_pic_select][4])
        self.list_text_pic[self.index_jauge][self.index_pic_select]=str(new_name)
        
        self.listbox_pic.takeItem(self.index_pic_select)
        self.listbox_pic.insertItem(self.index_pic_select,str(new_name))
        self.text_box_msg.setText('PIC FIT'+ self.Nom_pic[self.index_jauge][self.index_pic_select] + ' REPLACE')

   
        if bit is True:
            y_plot=out.best_fit
            p=out.params
        else:
            p=out.make_params()
            y_plot=X_pic.model.eval(p,x=self.Spectrum.wnb)
        

        self.plot_pic_select.setCurves(self.baseline, pg.PlotCurveItem([],[]))
    

        brush = self._selection_highlight_color()
        self.plot_ax3.setCurves(pg.PlotCurveItem(self.Spectrum.wnb[indexX],self.Spectrum.blfit[indexX]),pg.PlotCurveItem(self.Spectrum.wnb[indexX],y_plot))
        
        self.ax_zoom.disableAutoRange()
        self.ax_zoom.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self.ax_zoom.setYRange(min(y_plot),max(y_plot)*1.25)
        self.ax_zoom.setXRange(self.Param0[self.index_jauge][self.index_pic_select][0]-5*self.Param0[self.index_jauge][self.index_pic_select][2],self.Param0[self.index_jauge][self.index_pic_select][0]+5*self.Param0[self.index_jauge][self.index_pic_select][2])
        
        y_plot=X_pic.model.eval(p,x=self.Spectrum.wnb)
        self.list_y_fit_start[self.index_jauge][self.index_pic_select]=y_plot
        self.Spectrum.Gauges[self.index_jauge].pics[self.index_pic_select]=X_pic


        self._update_pic_fill_data(
            self.index_jauge,
            self.index_pic_select,
            self.Spectrum.wnb,
            y_plot,
            np.zeros_like(self.Spectrum.blfit),
        )

        self.Print_fit_start()

        self.bit_bypass=True
        self.select_pic()
        self.bit_bypass=False
        self.plot_ax3.setBrush(brush)
        #self.plot_pic_fit[self.index_jauge][self.index_pic_select].setBrush(brush)
             
    def Replace_pic(self):# - - - REPLACE PIQUE (r) - - -#
        if self.index_pic_select is not None:
            #name=self.listbox_pic.item(self.index_pic_select).text()
            #motif = r'_p(\d+)_'
            #matches = re.findall(motif, name)
            #index_pic=int(matches[0])
            #if index_pic != self.index_pic_select:
            #print(index_pic,self.index_pic_select,"Different betewen list and select")qqqqqqq

            if self.bit_bypass == False:
                self.Param0[self.index_jauge][self.index_pic_select]=[self.X0,self.Y0,float(self.spinbox_sigma.value()),np.array([float(spin.value()) for spin in self.coef_dynamic_spinbox]),str(self.model_pic_fit) ]
            else:
                self.Param0[self.index_jauge][self.index_pic_select][0],self.Param0[self.index_jauge][self.index_pic_select][1]=self.X0,self.Y0 
            new_name= self.Nom_pic[self.index_jauge][self.index_pic_select] + "   X0:"+str(self.Param0[self.index_jauge][self.index_pic_select][0])+"   Y0:"+ str(self.Param0[self.index_jauge][self.index_pic_select][1]) + "   sigma:" + str(self.Param0[self.index_jauge][self.index_pic_select][2]) + "   Coef:" + str(self.Param0[self.index_jauge][self.index_pic_select][3]) + " ; Modele:" + str(self.Param0[self.index_jauge][self.index_pic_select][4])
            self.list_text_pic[self.index_jauge][self.index_pic_select]=str(new_name)
            self.listbox_pic.takeItem(self.index_pic_select)
            self.listbox_pic.insertItem(self.index_pic_select,str(new_name))
            self.text_box_msg.setText('PIC'+ self.Nom_pic[self.index_jauge][self.index_pic_select] + ' REPLACE')

            X_pic=Pics(name=self.Nom_pic[self.index_jauge][self.index_pic_select],ctr=self.Param0[self.index_jauge][self.index_pic_select][0],ampH=self.Param0[self.index_jauge][self.index_pic_select][1],coef_spe=self.Param0[self.index_jauge][self.index_pic_select][3],sigma=self.Param0[self.index_jauge][self.index_pic_select][2],model_fit=self.Param0[self.index_jauge][self.index_pic_select][4])
            params=X_pic.model.make_params()
            y_plot=X_pic.model.eval(params,x=self.Spectrum.wnb)
            self._update_pic_fill_data(
                self.index_jauge,
                self.index_pic_select,
                self.Spectrum.wnb,
                y_plot,
                np.zeros_like(self.Spectrum.blfit),
            )
      

            self.plot_ax3.setCurves(pg.PlotCurveItem(self.Spectrum.wnb,np.zeros_like(self.Spectrum.blfit)),pg.PlotCurveItem(self.Spectrum.wnb,y_plot))
            self.plot_ax3.setBrush(self._selection_highlight_color())
            self.ax_zoom.disableAutoRange()
            self.ax_zoom.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
            self.ax_zoom.setYRange(min(y_plot),max(y_plot)*1.25)
            self.ax_zoom.setXRange(self.Param0[self.index_jauge][self.index_pic_select][0]-5*self.Param0[self.index_jauge][self.index_pic_select][2],self.Param0[self.index_jauge][self.index_pic_select][0]+5*self.Param0[self.index_jauge][self.index_pic_select][2])
            self.plot_pic_select.setCurves(self.baseline, pg.PlotCurveItem([],[]))


            self.list_y_fit_start[self.index_jauge][self.index_pic_select]=y_plot

            self.Spectrum.Gauges[self.index_jauge].pics[self.index_pic_select]=X_pic

            self.Print_fit_start()
            n_sigma = int(self.sigma_pic_fit_entry.value())
            if self.Spectrum is not None and getattr(self.Spectrum, "wnb", None) is not None:
                wnb_array = np.asarray(self.Spectrum.wnb)
                center = self.Param0[self.index_jauge][self.index_pic_select][0]
                sigma_current = self.Param0[self.index_jauge][self.index_pic_select][2]
                indexX = np.where((wnb_array > center - sigma_current * n_sigma) & (wnb_array < center + sigma_current * n_sigma))[0]
            else:
                indexX = np.array([])
            self._update_fit_window(indexX=indexX)
            #self.ax_spectrum.legend()

    def afficher_clavier_utilise(self):

        self._ensure_help_entries_loaded()
        key_to_description: Dict[str, str] = {}

        active_entries = [self.helpLabel.item(i).text() for i in range(self.helpLabel.count())]
        if active_entries:
            self.help_entries = active_entries

        for raw_entry in self.help_entries:
            item_text = raw_entry.strip()

            if not item_text or item_text.startswith("#") or item_text.startswith("*"):
                continue

            match = re.match(r"([^\s:]+)\s*:\s*(.+)", item_text)
            if not match:
                continue
            raw_keys = match.group(1)
            description = match.group(2).strip()

            desc_parts = re.match(r"^(.*?)\s*(\([^)]*\))?$", description)
            clean_description = desc_parts.group(1).strip() if desc_parts else description
            extra = desc_parts.group(2).strip() if desc_parts and desc_parts.group(2) else ""

            text_block = clean_description
            if extra:
                text_block += f"\n{extra}"

            if "+" in raw_keys or re.match(r"F\d+", raw_keys, re.IGNORECASE):
                special_key = raw_keys.upper().replace("*", "")
                if special_key not in key_to_description:
                    key_to_description[special_key] = text_block
                elif text_block not in key_to_description[special_key]:
                    key_to_description[special_key] += f"{text_block}"
            else:
                keys = re.findall(r"[a-zA-Z]", raw_keys)
                for key in keys:
                    k = key.upper()
                    if k not in key_to_description:
                        key_to_description[k] = text_block
                    elif text_block not in key_to_description[k]:
                        key_to_description[k] += f"{text_block}"

        help_entries = list(self.help_entries)
        if self.clavier_visuel is None:
            self.clavier_visuel = KeyboardWindow(key_to_description, help_entries)#, parent=self)
        else:
            self.clavier_visuel.update_content(key_to_description, help_entries)

        self.clavier_visuel.show()
        self.clavier_visuel.raise_()
        self.clavier_visuel.activateWindow()




def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file to load")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="activer le niveau de log DEBUG (équivaut à DRX_DEBUG=1)",
    )
    args = parser.parse_args()

    debug_enabled = args.debug or _DEBUG_ENV
    setup_logging(debug=debug_enabled)
    logger.info("Mode debug %s", "activé" if debug_enabled else "désactivé")

    if args.config:
        config_path = paths.resolve_config_path(args.config)
    else:
        config_path = file_config

    app = QApplication(sys.argv)
    app.setStyleSheet(style)

    window = MainWindow(folder_start, config_path)
    window.showMaximized()
    #window.setFixedSize(window.sizeHint())
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
