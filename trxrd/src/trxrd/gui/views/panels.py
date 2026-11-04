"""Tables and the spectrum navigator (a slider between two arrows)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from PySide6.QtCore import Qt
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
