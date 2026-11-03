"""Dialog "new run from RAW_DATA": dataset, scans, calibration, phase cards, output folder.

The dialog browses with the functions it is given (services.raw_data,
passed by the presenter) and remembers its fields between sessions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from trxrd.gui.presenters.viewer import NewRunRequest

SCAN_COLUMNS = ("Scan", "Images", "Oscillo", "Commande", "Début")


def _path_row(edit: QLineEdit, browse) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    button = QPushButton("…")
    button.setFixedWidth(32)
    button.clicked.connect(browse)
    layout.addWidget(button)
    return row


class NewRunDialog(QDialog):
    def __init__(self, browse, parent: Optional[QWidget] = None, settings: Optional[QSettings] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nouveau run depuis RAW_DATA")
        self.resize(1100, 720)
        self.browse = browse
        self.settings = settings or QSettings()
        self._datasets = []
        self._scans = []

        self.root = QLineEdit(self._get("raw_root", ""))
        self.root.editingFinished.connect(self.load_datasets)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Échantillon / jeu de données"])
        self.tree.itemSelectionChanged.connect(self.load_scans)
        self.images_only = QCheckBox("seulement les scans avec images")
        self.images_only.setChecked(True)
        self.images_only.toggled.connect(self.load_scans)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._update_ok)
        scans_box = QWidget()
        scans_layout = QVBoxLayout(scans_box)
        scans_layout.setContentsMargins(0, 0, 0, 0)
        scans_layout.addWidget(self.images_only)
        scans_layout.addWidget(self.table)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.tree)
        split.addWidget(scans_box)
        split.setSizes([300, 800])

        self.poni = QLineEdit(self._get("poni", ""))
        self.mask = QLineEdit(self._get("mask", ""))
        self.tth_min, self.tth_max = QDoubleSpinBox(), QDoubleSpinBox()
        for box, key, default in ((self.tth_min, "tth_min", 8.0), (self.tth_max, "tth_max", 25.0)):
            box.setRange(0.0, 180.0)
            box.setDecimals(2)
            box.setSuffix(" °")
            box.setValue(float(self._get(key, default)))
        self.calibrate = QPushButton("Calibrer…")
        self.calibrate.setToolTip("Calibration sur une image de LaB6 / CeO2 (à venir)")
        self.calibrate.setEnabled(False)
        tth = QWidget()
        tth_layout = QHBoxLayout(tth)
        tth_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (QLabel("de"), self.tth_min, QLabel("à"), self.tth_max):
            tth_layout.addWidget(widget)
        tth_layout.addStretch(1)
        calibration = QGroupBox("Calibration")
        form = QFormLayout(calibration)
        form.addRow("Fichier .poni", _path_row(self.poni, lambda: self._pick_file(self.poni, "Calibration (*.poni)")))
        form.addRow("Masque", _path_row(self.mask, lambda: self._pick_file(self.mask, "Masque (*.mask *.tif *.edf *.npy);;Tous (*)")))
        form.addRow("Plage 2θ", tth)
        form.addRow("", self.calibrate)

        self.phases = QListWidget()
        self.phases.addItems(self._phase_defaults())
        add, remove = QPushButton("Ajouter un dossier…"), QPushButton("Retirer")
        add.clicked.connect(self._add_phase_folder)
        remove.clicked.connect(lambda: [self.phases.takeItem(self.phases.row(i)) for i in self.phases.selectedItems()])
        phases = QGroupBox("Fiches de phase (candidates pour l'analyse)")
        phases_layout = QVBoxLayout(phases)
        phases_layout.addWidget(self.phases)
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(add)
        buttons_row.addWidget(remove)
        buttons_row.addStretch(1)
        phases_layout.addLayout(buttons_row)

        self.folder = QLineEdit(self._get("folder", ""))
        out = QGroupBox("Enregistrement")
        out_form = QFormLayout(out)
        out_form.addRow("Dossier des .trxrd.h5", _path_row(self.folder, self._pick_folder))

        bottom = QHBoxLayout()
        bottom.addWidget(calibration, 3)
        bottom.addWidget(phases, 2)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Créer")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.message = QLabel("")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Dossier RAW_DATA de la campagne"))
        layout.addWidget(_path_row(self.root, self._pick_root))
        layout.addWidget(split, 1)
        layout.addLayout(bottom)
        layout.addWidget(out)
        layout.addWidget(self.message)
        layout.addWidget(self.buttons)
        for edit in (self.poni, self.folder):
            edit.textChanged.connect(self._update_ok)
        self.load_datasets()
        self._update_ok()

    # ------------------------------------------------------------------ settings

    def _get(self, key: str, default):
        return self.settings.value(f"new_run/{key}", default)

    def _phase_defaults(self) -> List[str]:
        saved = self.settings.value("new_run/phases", None)
        if saved:
            return [saved] if isinstance(saved, str) else list(saved)
        library = self.browse.default_phase_library()
        return [str(library / name) for name in ("H2O", "Sn") if library and (library / name).is_dir()]

    def _save(self) -> None:
        values = {"raw_root": self.root.text(), "poni": self.poni.text(), "mask": self.mask.text(),
                  "tth_min": self.tth_min.value(), "tth_max": self.tth_max.value(), "folder": self.folder.text(),
                  "phases": [self.phases.item(i).text() for i in range(self.phases.count())]}
        for key, value in values.items():
            self.settings.setValue(f"new_run/{key}", value)

    # ------------------------------------------------------------------ browsing

    def _pick_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dossier RAW_DATA", self.root.text())
        if folder:
            self.root.setText(folder)
            if not self.folder.text():
                self.folder.setText(str(Path(folder).parent / "analyse_v2"))
            self.load_datasets()

    def _pick_file(self, edit: QLineEdit, filters: str) -> None:
        start = edit.text() or self.root.text()
        path, _ = QFileDialog.getOpenFileName(self, "Choisir un fichier", start, filters)
        if path:
            edit.setText(path)
            self._update_ok()

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dossier des runs", self.folder.text() or self.root.text())
        if folder:
            self.folder.setText(folder)
            self._update_ok()

    def _add_phase_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dossier de fiches .jcpds", "")
        if folder:
            self.phases.addItem(folder)

    def load_datasets(self) -> None:
        self.tree.clear()
        self._datasets = []
        root = Path(self.root.text())
        if not self.root.text() or not root.is_dir():
            self.message.setText("Choisir le dossier RAW_DATA de la campagne.")
            return
        try:
            self._datasets = self.browse.find_datasets(root)
        except OSError as exc:
            self.message.setText(f"Lecture impossible : {exc}")
            return
        samples = {}
        for k, dataset in enumerate(self._datasets):
            parent = samples.get(dataset.sample)
            if parent is None:
                parent = samples[dataset.sample] = QTreeWidgetItem(self.tree, [dataset.sample])
            item = QTreeWidgetItem(parent, [dataset.name])
            item.setData(0, Qt.ItemDataRole.UserRole, k)
        self.tree.expandAll()
        self.message.setText(f"{len(self._datasets)} jeux de données")

    def selected_dataset(self):
        items = self.tree.selectedItems()
        index = items[0].data(0, Qt.ItemDataRole.UserRole) if items else None
        return self._datasets[index] if index is not None else None

    def load_scans(self) -> None:
        dataset = self.selected_dataset()
        self._scans = self.browse.list_scans(dataset, images_only=self.images_only.isChecked()) if dataset else []
        self.table.setColumnCount(len(SCAN_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(SCAN_COLUMNS))
        self.table.setRowCount(len(self._scans))
        for i, scan in enumerate(self._scans):
            row = (scan.label, str(scan.frames or ("?" if scan.image_file else "")), "oui" if scan.oscilloscope else "",
                   scan.title, scan.start_time[:19].replace("T", " "))
            for j, text in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(text))
        self.table.horizontalHeader().resizeSections(QHeaderView.ResizeMode.ResizeToContents)
        self._update_ok()

    def selected_scans(self) -> List[Path]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()}) if self.table.selectionModel() else []
        return [self._scans[r].image_file for r in rows if self._scans[r].image_file is not None]

    def _update_ok(self) -> None:
        ready = bool(self.selected_scans()) and bool(self.poni.text()) and bool(self.folder.text())
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ready)

    # ------------------------------------------------------------------ result

    def request(self) -> NewRunRequest:
        return NewRunRequest(
            scans=self.selected_scans(),
            folder=Path(self.folder.text()),
            poni_file=self.poni.text(),
            mask_file=self.mask.text() or None,
            two_theta_range=(self.tth_min.value(), self.tth_max.value()),
            phase_paths=[self.phases.item(i).text() for i in range(self.phases.count())],
        )

    def accept(self) -> None:
        self._save()
        super().accept()
