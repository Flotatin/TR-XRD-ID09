"""Shared widget helper functions."""

from __future__ import annotations

from typing import List

from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem,QVBoxLayout ,QProgressBar,QPushButton,QDialog


def creat_spin_label(spinbox, label_text, label_unit=None):
    layout = QHBoxLayout()
    label = QLabel(label_text)
    layout.addWidget(label)
    layout.addWidget(spinbox)
    if label_unit:
        label_unit = QLabel(label_unit)
        layout.addWidget(label_unit)
    return layout


def load_help_file(widget: QListWidget, path: str) -> None:
    """Populate *widget* with commands from the given help file."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                command = line.strip()
                if not command:
                    continue
                item = QListWidgetItem(command)
                text = command
                if command.startswith("#"):
                    font = QFont("Courier New", 10, QFont.Bold)
                    item.setFont(font)
                    item.setForeground(QColor("royalblue"))
                    text = command[1:]
                else:
                    font = QFont("Arial", 8, QFont.Bold)
                    item.setFont(font)
                    item.setForeground(QColor("white"))
                item.setText(text)
                widget.addItem(item)
    except Exception as exc:
        widget.addItem(f"Error loading file: {exc}")


def load_command_file(list_widget: QListWidget, python_commands: List[str], path: str) -> None:
    """Fill the visible list and cache command strings loaded from *path*."""
    list_widget.clear()
    python_commands.clear()
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                command = line.strip()
                if not command:
                    continue
                item = QListWidgetItem(command)
                text = command
                if command.startswith("#"):
                    font = QFont("Courier New", 11, QFont.Bold)
                    item.setFont(font)
                    item.setForeground(QColor("royalblue"))
                    text = command[1:]
                elif command.startswith("self"):
                    font = QFont("Arial", 10, QFont.Bold)
                    item.setFont(font)
                    item.setForeground(QColor("k"))
                    text = command[5:]
                elif command.startswith("."):
                    text = command[1:]
                    if command.endswith(")"):
                        font = QFont("Arial", 8, QFont.Bold)
                        item.setFont(font)
                        item.setForeground(QColor("lightgreen"))
                    else:
                        font = QFont("Arial", 9, QFont.Bold)
                        item.setFont(font)
                        item.setForeground(QColor("tomato"))
                else:
                    font = QFont("Arial", 10, QFont.Bold)
                    item.setFont(font)
                    item.setForeground(QColor("k"))
                item.setText(text.split("(")[0])
                python_commands.append(command)
                list_widget.addItem(item)
    except FileNotFoundError:
        list_widget.addItem(f"Error: File '{path}' not found")
        python_commands.clear()


class ProgressDialog(QDialog):
    """Simple progress dialog compatible with headless tests."""

    def __init__(self, label_text: str, cancel_text: str, minimum: int, maximum: int, parent=None) -> None:
        super().__init__(parent)
        self._canceled = False
        self._auto_close = False
        self._minimum = minimum
        self._maximum = maximum
        self._value = minimum

        self.setWindowTitle("Progression")
        layout = QVBoxLayout(self)

        self.label = QLabel(label_text)
        layout.addWidget(self.label)

        if QProgressBar is not None:
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(minimum, maximum)
            layout.addWidget(self.progress_bar)
            self._progress_label = None
        else:  # pragma: no cover - executed only in environments without QProgressBar
            self.progress_bar = None
            self._progress_label = QLabel(self._format_progress(minimum))
            layout.addWidget(self._progress_label)

        self.cancel_button = QPushButton(cancel_text)
        layout.addWidget(self.cancel_button)
        self.cancel_button.clicked.connect(self._on_cancel)

    def _on_cancel(self) -> None:
        self._canceled = True
        self.reject()

    def setWindowModality(self, modality) -> None:  # type: ignore[override]
        super().setWindowModality(modality)

    def setMinimumDuration(self, duration: int) -> None:
        # Compatibility method; no behaviour needed for this simple dialog.
        pass

    def setAutoClose(self, auto_close: bool) -> None:
        self._auto_close = auto_close

    def setLabelText(self, text: str) -> None:
        self.label.setText(text)

    def setValue(self, value: int) -> None:
        self._value = value
        if self.progress_bar is not None:
            self.progress_bar.setValue(value)
            maximum = self.progress_bar.maximum()
        else:
            maximum = self._maximum
            if self._progress_label is not None:
                self._progress_label.setText(self._format_progress(value))
        if self._auto_close and value >= maximum and not self._canceled:
            self.accept()

    def wasCanceled(self) -> bool:
        return self._canceled

    def close(self) -> None:  # type: ignore[override]
        super().close()

    def _format_progress(self, value: int) -> str:
        span = max(self._maximum - self._minimum, 1)
        progress = max(min(value - self._minimum, span), 0)
        percent = int(progress * 100 / span)
        return f"{percent}%"
