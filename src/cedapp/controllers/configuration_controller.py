"""Configuration persistence helpers for the DRX application."""

from __future__ import annotations

import ast
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import fabio
import pyqtgraph as pg

from cedapp.drx import Calibration
from cedapp.drx import CL_FD_Update as CL
from cedapp.utils import paths


class ConfigurationMixin:
    """Provide reusable helpers to persist window configuration.

    The mixin is designed for widgets orchestrating several domain
    controllers.  Consumers are expected to expose a subset of attributes used
    by the persistence logic (``config_file``, ``dict_folders``, ``calib``,
    ``spectrum_controller`` ...).  The :class:`MainWindow` class is the
    canonical implementation and serves as documentation for the required
    attributes.
    """

    def save_paths_to_txt(self) -> None:
        """Serialise the active configuration to :attr:`config_file`."""

        config_path = Path(self.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w", encoding="utf-8") as file:
            file.write(f"folder_DRX={self.dict_folders['DRX']}\n")
            file.write(f"folder_OSC={self.dict_folders['Oscilloscope']}\n")
            file.write(f"folder_CED={self.dict_folders['CED']}\n")
            file.write(f"loaded_file_DRX={self.loaded_file_DRX}\n")
            file.write(f"loaded_file_OSC={self.loaded_file_OSC}\n")
            calib_mask = getattr(self.calib, "file_mask", "") if self.calib else ""
            calib_poni = getattr(self.calib, "file_poni", "") if self.calib else ""
            file.write(f"calib_file_mask={calib_mask}\n")
            file.write(f"calib_file_poni={calib_poni}\n")
            zones = [list(map(float, r.getRegion())) for r in self.spectrum_controller.zones]
            file.write(f"zone={zones if zones else None}\n")

            if hasattr(self.ClassDRX, "E"):
                file.write(f"energie_DRX={self.ClassDRX.E}\n")

            if hasattr(self.ClassDRX, "list_file"):
                file.write(f"bib_files={json.dumps(self.ClassDRX.list_file)}\n")

            save_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"date={save_date}\n")

        self.text_box_msg.setText(f"save \n {config_path}")

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _sanitize_config_value(self, value: Any) -> str:
        """Normalise empty values read from the configuration file."""

        if value is None:
            return ""
        value_str = str(value).strip()
        return "" if value_str.lower() == "none" else value_str

    def _reset_missing_configuration(self) -> None:
        """Reset runtime attributes when the configuration file is absent."""

        self.loaded_file_DRX = ""
        self.loaded_file_OSC = ""
        self.calib = None
        if hasattr(self, "spectrum_controller"):
            for region in list(self.spectrum_controller.zones):
                self.ax_spectrum.removeItem(region)
            self.spectrum_controller.zones.clear()
            self.spectrum_controller.theta2_range = []
        self.zones = []
        self.theta2_range = [[0, 90]]
        if hasattr(self, "DRX_selector"):
            self.DRX_selector.clear()
        self.file_label_spectro.setText("DRX: None")
        self.file_label_oscilo.setText("Oscillo: None")

    def load_paths_from_txt(self) -> None:
        """Restore the configuration stored in :attr:`config_file`."""

        config_path = Path(self.config_file)
        messages: List[str] = [f"load: {config_path}"]

        if not config_path.exists():
            self._reset_missing_configuration()
            messages.append("err: no config file")
            self.text_box_msg.setText("".join(messages))
            return

        config_entries: Dict[str, str] = {}
        list_files: List[str] = []
        list_files_line: str | None = None

        with config_path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("bib_files="):
                    list_files_line = line.split("=", 1)[1]
                    continue
                if line.startswith("bib_file="):
                    list_files.append(line.split("=", 1)[1])
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "zone":
                    try:
                        config_entries[key] = ast.literal_eval(value)
                    except Exception:
                        messages.append("err: invalid zone format")
                    continue
                config_entries[key] = value

        if list_files_line:
            try:
                list_files = json.loads(list_files_line)
            except Exception:
                messages.append("err: invalid bib_files entry")
                list_files = []

        resolved_bib_files = paths.resolve_bibdrx_paths(list_files)
        missing_bib_files = [path for path in resolved_bib_files if path and not Path(path).exists()]
        if missing_bib_files:
            messages.append(f"err: missing bibdrx files ({len(missing_bib_files)})")
        list_files = [path for path in resolved_bib_files if path and Path(path).exists()]

        self.dict_folders["DRX"] = self._sanitize_config_value(config_entries.get("folder_DRX", ""))
        self.dict_folders["CED"] = self._sanitize_config_value(config_entries.get("folder_CED", ""))
        self.dict_folders["Oscilloscope"] = self._sanitize_config_value(
            config_entries.get("folder_OSC", "")
        )
        self.loaded_file_DRX = self._sanitize_config_value(config_entries.get("loaded_file_DRX", ""))
        self.loaded_file_OSC = self._sanitize_config_value(config_entries.get("loaded_file_OSC", ""))

        if hasattr(self, "spectrum_controller"):
            for region in list(self.spectrum_controller.zones):
                self.ax_spectrum.removeItem(region)
            self.spectrum_controller.zones.clear()
            self.spectrum_controller.theta2_range = []
        self.zones = []

        zone_entries = config_entries.get("zone") or []
        loaded_regions: List[pg.LinearRegionItem] = []
        if isinstance(zone_entries, (list, tuple)):
            for entry in zone_entries:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    continue
                try:
                    start, end = float(entry[0]), float(entry[1])
                except Exception:
                    continue
                region = pg.LinearRegionItem([start, end])
                region.setVisible(False)
                loaded_regions.append(region)

        if loaded_regions:
            if hasattr(self, "spectrum_controller"):
                self.spectrum_controller.extend_with_regions(loaded_regions)
                for region in self.spectrum_controller.zones:
                    region.setVisible(False)
                self.zones = list(self.spectrum_controller.zones)
            else:
                for region in loaded_regions:
                    region.mouseClickEvent = lambda ev, reg=region: self.selectionner_zone(reg)
                    region.sigRegionChangeFinished.connect(self.f_update_theta2_range)
                    self.ax_spectrum.addItem(region)
                self.zones = loaded_regions

        if hasattr(self, "spectrum_controller") and self.spectrum_controller.zones:
            self.theta2_range = [list(map(float, region.getRegion())) for region in self.spectrum_controller.zones]
        elif self.zones:
            self.theta2_range = [list(map(float, region.getRegion())) for region in self.zones]
        elif hasattr(self, "Spectrum") and self.Spectrum is not None and getattr(self.Spectrum, "wnb", None) is not None:
            self.theta2_range = [[float(self.Spectrum.wnb[0]), float(self.Spectrum.wnb[-1])]]
        else:
            self.theta2_range = [[0, 90]]
            if not hasattr(self, "Spectrum"):
                messages.append("err: Spectrum unavailable, defaulting theta2_range")

        calib_mask = self._sanitize_config_value(config_entries.get("calib_file_mask", ""))
        calib_poni = self._sanitize_config_value(config_entries.get("calib_file_poni", ""))
        default_theta_range: List[float] = list(getattr(self, "DEFAULT_THETA_RANGE", [0, 90]))
        preferred_theta_range = (
            list(self.calib.theta_range) if self.calib and getattr(self.calib, "theta_range", None) else default_theta_range
        )
        calib_loaded = False

        if calib_mask and calib_poni:
            try:
                if self.calib is None:
                    self.calib = Calibration.Calib_DRX(
                        file_mask=calib_mask,
                        file_poni=calib_poni,
                        theta_range=preferred_theta_range,
                    )
                else:
                    self.calib.theta_range = preferred_theta_range
                    self.calib.Load_calib(file_mask=calib_mask, file_poni=calib_poni)
                calib_loaded = getattr(self.calib, "ai", None) is not None and getattr(self.calib, "mask", None) is not None
            except Exception:
                calib_loaded = False
            if not calib_loaded:
                messages.append("err: calib load failed")
                self.calib = None
        else:
            if calib_mask or calib_poni:
                messages.append("err: calib config incomplete")
            else:
                messages.append("warn: no calib files")
            self.calib = None

        theta_range = list(getattr(self, "DEFAULT_THETA_RANGE", [0, 90]))
        if self.calib and getattr(self.calib, "theta_range", None):
            theta_range = list(self.calib.theta_range)

        energy_value = self._sanitize_config_value(config_entries.get("energie_DRX", ""))
        if energy_value:
            try:
                energy = float(energy_value)
            except Exception:
                messages.append("err: invalid energie_DRX value")
                energy = float(getattr(self, "DEFAULT_ENERGY", 19000))
        else:
            class_drx = getattr(self, "ClassDRX", None)
            if class_drx is None:
                messages.append("err: ClassDRX unavailable, using default energy")
            energy = float(getattr(class_drx, "E", getattr(self, "DEFAULT_ENERGY", 19000)))

        if self.calib and getattr(self.calib, "ai", None) is not None:
            try:
                energy = round(1239.8e-9 / self.calib.ai.wavelength, 0)
            except Exception:
                messages.append("err: unable to compute energy from calib")

        try:
            self.ClassDRX = CL.DRX(folder=list_files if list_files else None, Borne=theta_range, E=energy)
        except Exception:
            messages.append("err: bibliotheque DRX load failed")
            self.ClassDRX = CL.DRX(folder=None, Borne=theta_range, E=energy)

        if hasattr(self, "set_energy_value"):
            self.set_energy_value(energy)

        self.liste_type_Gauge = list(self.ClassDRX.Bibli_elements.keys())

        if hasattr(self, "DRX_selector"):
            self.DRX_selector.clear()
            if self.loaded_file_DRX:
                if os.path.isfile(self.loaded_file_DRX):
                    try:
                        data_drx = fabio.open(self.loaded_file_DRX)
                        self.DRX_selector.addItems([f"DRX{i}" for i in range(data_drx.nframes)])
                    except Exception:
                        messages.append("err: unable to load DRX file")
                else:
                    messages.append("warn: DRX file path invalid")

        self.file_label_spectro.setText(
            f"DRX: {os.path.basename(self.loaded_file_DRX) if self.loaded_file_DRX else 'None'}"
        )
        self.file_label_oscilo.setText(
            f"Oscillo: {os.path.basename(self.loaded_file_OSC) if self.loaded_file_OSC else 'None'}"
        )

        self.text_box_msg.setText("\n".join(messages))
        self.update_gauge_table()
