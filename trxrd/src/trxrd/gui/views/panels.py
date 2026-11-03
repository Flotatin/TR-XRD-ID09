"""Tables and the spectrum navigator (list + slider)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QLabel, QSlider, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


class RowsTable(QTableWidget):
    """Read-only table filled with rows of text."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)

    def show_rows(self, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        self.setUpdatesEnabled(False)
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(list(columns))
        self.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, text in enumerate(row):
                item = QTableWidgetItem(text)
                if j and text[:1] in "-0123456789":
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.setItem(i, j, item)
        self.setUpdatesEnabled(True)


class Navigator(QWidget):
    """Spectra of the run (index, time, phases) and a slider; reports the chosen spectrum."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_select: Optional[Callable[[int], None]] = None
        self.table = RowsTable()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.itemSelectionChanged.connect(self._row_selected)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._report)
        self.label = QLabel("")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addWidget(self.slider)
        layout.addWidget(self.label)

    def _report(self, index: int) -> None:
        if self.on_select is not None:
            self.on_select(int(index))

    def _row_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self._report(rows[0].row())

    def show_spectra(self, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        self.table.blockSignals(True)
        self.table.show_rows(columns, rows)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(rows) - 1, 0))
        self.slider.setEnabled(bool(rows))
        self.slider.blockSignals(False)

    def set_current(self, index: int, count: int) -> None:
        for widget in (self.table, self.slider):
            widget.blockSignals(True)
        if count:
            self.table.selectRow(index)
            self.table.scrollToItem(self.table.item(index, 0), QAbstractItemView.ScrollHint.EnsureVisible)
            self.slider.setValue(index)
            self.label.setText(f"Spectre {index} / {count - 1}")
        else:
            self.table.clearSelection()
            self.label.setText("")
        for widget in (self.table, self.slider):
            widget.blockSignals(False)
