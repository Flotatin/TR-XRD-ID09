"""Main window of the viewer: layout of the views, menus and toolbar from the action registry, dialogs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QLabel, QMainWindow, QMessageBox, QProgressDialog, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

from trxrd.gui.actions import ActionRegistry
from trxrd.gui.views.panels import Navigator, RowsTable
from trxrd.gui.views.plots import CakeView, MapView, PressureView, SpectrumView
from trxrd.physics.lattice import AXIS_LABELS

FILE_FILTER = "Runs (*.trxrd.h5 *.h5 *.CEDX *.CEDXUp *.cedx);;Tous les fichiers (*)"
SAVE_FILTER = "Run TR-XRD v2 (*.trxrd.h5)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.resize(1500, 900)
        self.on_close_request: Optional[Callable[[], bool]] = None  # set by the application
        self.on_unit: Optional[Callable[[str], None]] = None
        self.actions = ActionRegistry(self)
        self._folder = ""

        self.navigator = Navigator()
        self.spectrum = SpectrumView()
        self.cake = CakeView(link_to=self.spectrum.plot)  # under the spectrum, as in the v1
        self.pressure = PressureView()
        self.map = MapView()
        self.phases = RowsTable()
        self.peaks = RowsTable()

        tabs = QTabWidget()
        tabs.addTab(self.pressure, "P(t)")
        tabs.addTab(self.map, "Carte")
        center = QSplitter(Qt.Orientation.Vertical)
        center.addWidget(self.spectrum)
        center.addWidget(self.cake)
        center.addWidget(tabs)
        center.setSizes([420, 260, 300])
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(_titled("Phases du spectre", self.phases))
        right.addWidget(_titled("Pics du spectre", self.peaks))
        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(self.navigator)
        main.addWidget(center)
        main.addWidget(right)
        main.setSizes([300, 820, 380])
        self.setCentralWidget(main)

        self.units = QComboBox()
        self.units.currentIndexChanged.connect(self._unit_chosen)
        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.status_label, 1)

    # ------------------------------------------------------------------ built once the actions exist

    def build_menus(self, toolbar: Sequence[str] = ()) -> None:
        for menu in self.actions.menus():
            target = self.menuBar().addMenu(menu)
            for action in self.actions.in_menu(menu):
                target.addAction(action)
        bar = self.addToolBar("Outils")
        bar.setMovable(False)
        for key in toolbar:
            bar.addAction(self.actions.get(key)) if key != "|" else bar.addSeparator()
        bar.addSeparator()
        bar.addWidget(QLabel(" Axe : "))
        bar.addWidget(self.units)

    def _unit_chosen(self, index: int) -> None:
        unit = self.units.itemData(index)
        if unit and self.on_unit is not None:
            self.on_unit(unit)

    # ------------------------------------------------------------------ window interface of MainPresenter

    def set_title(self, text: str) -> None:
        self.setWindowTitle(text)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_units(self, available: Sequence[str], current: str) -> None:
        self.units.blockSignals(True)
        self.units.clear()
        for unit in available:
            self.units.addItem(AXIS_LABELS[unit], unit)
        self.units.setCurrentIndex(max(self.units.findData(current), 0))
        self.units.blockSignals(False)
        for unit, key in (("2th_deg", "unit_2th"), ("q_A^-1", "unit_q"), ("d_A", "unit_d")):
            if key in self.actions.actions:
                action = self.actions.get(key)
                action.setEnabled(unit in available)
                action.setChecked(unit == current)

    def set_busy(self, text: Optional[str]) -> None:
        if text:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.statusBar().showMessage(text)
        else:
            while QGuiApplication.overrideCursor() is not None:
                QGuiApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()

    def ask_open_path(self) -> Optional[str]:
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir un run", self._folder, FILE_FILTER)
        if path:
            self._folder = str(Path(path).parent)
        return path or None

    def ask_save_path(self, suggested: str) -> Optional[str]:
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le run", suggested, SAVE_FILTER)
        if path and not path.endswith(".h5"):
            path += ".trxrd.h5"
        return path or None

    def confirm_discard(self) -> bool:
        answer = QMessageBox.question(
            self, "Modifications non enregistrées",
            "Le run a été modifié et n'est pas enregistré. Abandonner les modifications ?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def show_error(self, title: str, text: str) -> None:
        QMessageBox.critical(self, title, text)

    def show_info(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def ask_new_run(self, browse):
        from trxrd.gui.views.new_run import NewRunDialog

        dialog = NewRunDialog(browse, self)
        return dialog.request() if dialog.exec() == QDialog.DialogCode.Accepted else None

    def start_progress(self, text: str, on_cancel: Callable[[], None]) -> None:
        self._progress = QProgressDialog(text, "Arrêter", 0, 1000, self)
        self._progress.setWindowTitle("TR-XRD v2")
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.canceled.connect(on_cancel)
        self._progress.setValue(0)

    def set_progress(self, done: int, total: int) -> None:
        if getattr(self, "_progress", None) is not None:
            self._progress.setMaximum(max(int(total), 1))
            self._progress.setValue(int(done))

    def end_progress(self) -> None:
        if getattr(self, "_progress", None) is not None:
            self._progress.canceled.disconnect()
            self._progress.close()
            self._progress = None

    # ------------------------------------------------------------------ help, closing

    def show_shortcuts(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Raccourcis clavier")
        table = RowsTable()
        table.show_rows(("Menu", "Action", "Raccourci"), self.actions.shortcut_table())
        layout = QVBoxLayout(dialog)
        layout.addWidget(table)
        dialog.resize(520, 420)
        dialog.exec()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.on_close_request is None or self.on_close_request():
            event.accept()
        else:
            event.ignore()


def _titled(title: str, widget: QWidget) -> QWidget:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    label = QLabel(f"<b>{title}</b>")
    layout.addWidget(label)
    layout.addWidget(widget)
    return box


def application(argv: Sequence[str]) -> QApplication:
    app = QApplication.instance() or QApplication(list(argv))
    app.setApplicationName("TR-XRD v2")
    app.setOrganizationName("trxrd")
    return app
