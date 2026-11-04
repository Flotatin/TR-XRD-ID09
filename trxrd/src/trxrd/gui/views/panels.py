"""Tables, the slider of the spectra (between two arrows) and their list (tab "Spectres")."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QSlider, QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout, QWidget,
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


class SpectraList(QWidget):
    """Spectra of the run (index, time, phases): a click goes to a spectrum, Shift or Ctrl + click chooses several."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_select: Optional[Callable[[int], None]] = None
        self.on_selection: Optional[Callable[[List[int]], None]] = None
        self.table = RowsTable()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.itemSelectionChanged.connect(self._rows_chosen)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)

    def selected_rows(self) -> List[int]:
        return sorted(index.row() for index in self.table.selectionModel().selectedRows())

    def _rows_chosen(self) -> None:
        rows = self.selected_rows()
        if self.on_selection is not None:
            self.on_selection(rows)
        if rows and self.on_select is not None:
            current = self.table.currentRow()
            self.on_select(current if current in rows else rows[0])

    def show_spectra(self, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        self.table.blockSignals(True)
        self.table.show_rows(columns, rows)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def set_current(self, index: int, selection: Sequence[int] = ()) -> None:
        """The spectra chosen are highlighted, or else the current spectrum `index` alone."""
        count = self.table.rowCount()
        self.table.blockSignals(True)
        if count:
            rows = list(selection) or [index]
            if rows != self.selected_rows():
                chosen = QItemSelection()
                last = self.table.columnCount() - 1
                for first, end in _ranges(rows):
                    chosen.select(self.table.model().index(first, 0), self.table.model().index(end, last))
                self.table.selectionModel().select(chosen, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            self.table.selectionModel().setCurrentIndex(self.table.model().index(index, 0),
                                                        QItemSelectionModel.SelectionFlag.NoUpdate)
            self.table.scrollToItem(self.table.item(index, 0), QAbstractItemView.ScrollHint.EnsureVisible)
        else:
            self.table.clearSelection()
        self.table.blockSignals(False)


class Navigator(QWidget):
    """Slider of the spectra between two arrows, and where we are under it; reports the current spectrum."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_select: Optional[Callable[[int], None]] = None
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._report)
        self.previous = self._arrow(Qt.ArrowType.LeftArrow, "Spectre précédent", -1)
        self.following = self._arrow(Qt.ArrowType.RightArrow, "Spectre suivant", +1)
        self.label = QLabel("")
        row = QHBoxLayout()
        row.addWidget(self.previous)
        row.addWidget(self.slider, 1)
        row.addWidget(self.following)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.label)
        self.set_current(0, 0)

    def _arrow(self, kind: Qt.ArrowType, tip: str, step: int) -> QToolButton:
        button = QToolButton()
        button.setArrowType(kind)
        button.setAutoRepeat(True)  # held down, it runs through the spectra
        button.setToolTip(tip)
        button.clicked.connect(lambda: self.slider.setValue(self.slider.value() + step))
        return button

    def _report(self, index: int) -> None:
        if self.on_select is not None:
            self.on_select(int(index))

    def set_current(self, index: int, count: int, text: str = "") -> None:
        """Current spectrum `index` among `count`; `text`: where we are."""
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(count - 1, 0))
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.slider.setEnabled(count > 0)
        self.previous.setEnabled(index > 0)
        self.following.setEnabled(index < count - 1)
        self.label.setText(text)
