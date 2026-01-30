"""Non-UI services for the DRX application."""

from __future__ import annotations

import os
import re


class FileSelectionController:
    """Handle file list scanning and selection logic."""

    def __init__(self, host) -> None:
        self.host = host

    def change_file_type(self) -> None:
        file_type = self.host.type_selector.currentText()
        folder = self.host.dict_folders[file_type]

        try:
            files_brute = os.listdir(folder)
            files = sorted(
                [f for f in files_brute],
                key=lambda x: os.path.getctime(os.path.join(folder, x)),
                reverse=True,
            )
        except Exception as exc:
            print(f"Erreur lors de la lecture du dossier {folder} : {exc}")
            files = []

        self.host.current_file_list = files
        self.host.full_path_list = [os.path.join(folder, f) for f in files]

        self.host.search_bar.setText("")
        self.host.search_bar.setPlaceholderText(f"Search {file_type}...")
        self.host.listbox_file.clear()
        self.host.listbox_file.addItems(files)

    def filter_files(self) -> None:
        filter_text = self.host.search_bar.text().lower()
        filtered_files = [f for f in self.host.current_file_list if filter_text in f.lower()]
        self.host.listbox_file.clear()
        self.host.listbox_file.addItems(filtered_files)

    def select_file(self) -> None:
        file_type = self.host.type_selector.currentText()
        file_item = self.host.listbox_file.currentItem()

        if file_item is None:
            return

        if file_type == self.host.type_folder[1]:
            self.host.loaded_file_OSC = os.path.join(self.host.dict_folders[file_type], file_item.text())

            match = re.match(r"([^_]+)_(\d+)_scan(\d+)", file_item.text())
            if match:
                first_part = match.group(1).lower()
                number_after_underscore = match.group(2)
                number_after_scan = match.group(3)
                dos = "_0001"
                if first_part + number_after_underscore == "water01":
                    dos = "_0002"

                drx_path = os.path.join(
                    self.host.dict_folders["DRX"],
                    rf"{first_part}{number_after_underscore}\{first_part}{number_after_underscore}{dos}\scan{number_after_scan.zfill(4) }\scan_jf1m_0000.h5",
                )
                if os.path.isfile(drx_path):
                    self.host.set_loaded_drx_file(drx_path)
                else:
                    self.host.text_box_msg.setText(f"no file DRX for : \n {file_item.text()} \n oscilloscope")
            else:
                self.host.text_box_msg.setText(f"no file DRX for : \n {file_item.text()} \n oscilloscope")

        elif file_type == self.host.type_folder[2]:
            selected_path = os.path.join(self.host.dict_folders[file_type], file_item.text())
            drx_path = ""
            if os.path.isdir(selected_path):
                drx_path = self._find_scan_h5(selected_path)
            elif selected_path.lower().endswith(".h5"):
                drx_path = selected_path

            if drx_path:
                self.host.set_loaded_drx_file(drx_path)
            else:
                self.host.text_box_msg.setText(f"no h5 file found in : \n {selected_path}")

        elif file_type == self.host.type_folder[0]:
            self.host.bit_bypass = True
            self.host.CLEAR_CEDd()
            self.host.bit_bypass = False
            self.host.f_CEDX_Load(item=file_item)

        self._update_file_labels()

    def _update_file_labels(self) -> None:
        self.host.file_label_spectro.setText(
            f"DRX: {os.path.basename(self.host.loaded_file_DRX) if self.host.loaded_file_DRX else 'None'}"
        )
        self.host.file_label_oscilo.setText(
            f"OSC: {os.path.basename(self.host.loaded_file_OSC) if self.host.loaded_file_OSC else 'None'}"
        )
    def _find_scan_h5(self, scan_folder: str) -> str:
        try:
            entries = os.listdir(scan_folder)
        except OSError:
            return ""

        preferred = "scan_jf1m_0000.h5"
        if preferred in entries:
            return os.path.join(scan_folder, preferred)

        candidates = sorted(
            [name for name in entries if name.lower().endswith(".h5")],
            key=str.lower,
        )
        if candidates:
            return os.path.join(scan_folder, candidates[0])
        return ""
