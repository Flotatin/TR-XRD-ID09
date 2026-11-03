"""Tables and the spectrum navigator (list + slider)."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt
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


def _ranges(rows: Sequence[int]) -> List[tuple]:
    """[2, 3, 4, 7] -> [(2, 4), (7, 7)]"""
    out: List[list] = []
    for row in sorted(rows):
        if out and row == out[-1][1] + 1:
            out[-1][1] = row
        else:
            out.append([row, row])
    return [tuple(r) for r in out]


class Navigator(QWidget):
    """Spectra of the run (index, time, phases) and a slider; reports the current spectrum and the selection
    (several spectra chosen with Shift or Ctrl + click)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_select: Optional[Callable[[int], None]] = None
        self.on_selection: Optional[Callable[[List[int]], None]] = None
        self.table = RowsTable()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
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

    def selected_rows(self) -> List[int]:
        return sorted(index.row() for index in self.table.selectionModel().selectedRows())

    def _row_selected(self) -> None:
        rows = self.selected_rows()
        if self.on_selection is not None:
            self.on_selection(rows)
        if rows:
            current = self.table.currentRow()
            self._report(current if current in rows else rows[0])

    def show_spectra(self, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        self.table.blockSignals(True)
        self.table.show_rows(columns, rows)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(rows) - 1, 0))
        self.slider.setEnabled(bool(rows))
        self.slider.blockSignals(False)

    def set_current(self, index: int, count: int, selection: Sequence[int] = ()) -> None:
        """Current spectrum `index`, inside the selection; elsewhere (keys, slider, plots) it is selected alone."""
        for widget in (self.table, self.slider):
            widget.blockSignals(True)
        rows = list(selection) if index in selection else [index]
        if count:
            if rows != self.selected_rows():
                chosen = QItemSelection()
                last = self.table.columnCount() - 1
                for first, end in _ranges(rows):
                    chosen.select(self.table.model().index(first, 0), self.table.model().index(end, last))
                self.table.selectionModel().select(chosen, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            self.table.selectionModel().setCurrentIndex(self.table.model().index(index, 0),
                                                        QItemSelectionModel.SelectionFlag.NoUpdate)
            self.table.scrollToItem(self.table.item(index, 0), QAbstractItemView.ScrollHint.EnsureVisible)
            self.slider.setValue(index)
            self.label.setText(f"Spectre {index} / {count - 1}")
        else:
            self.table.clearSelection()
            self.label.setText("")
        for widget in (self.table, self.slider):
            widget.blockSignals(False)
        if count and rows != list(selection) and self.on_selection is not None:
            self.on_selection(rows)  # moved outside the selection: it is now the current spectrum alone
