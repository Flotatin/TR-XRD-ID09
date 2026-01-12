"""Keyboard helper widget."""

from __future__ import annotations

from typing import Dict, Iterable, List

from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class KeyboardWindow(QWidget):
    """Simple window displaying a virtual keyboard with mapped commands."""

    KEYBOARD_LAYOUT: List[List[str]] = [
        ["Esc", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"],
        ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "Backspace"],
        ["Tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]", "\\"],
        ["Caps", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'", "Enter"],
        ["Shift", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "Shift"],
        ["Ctrl", "Win", "Alt", "Space", "Alt", "Fn", "Menu", "Ctrl"],
    ]

    def __init__(self, key_to_description, help_entries=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clavier - Touches utilisées")
        self.key_to_description = dict(key_to_description)
        self.help_entries = list(help_entries or [])

        self._button_base_labels: Dict[QPushButton, str] = {}
        self._keyboard_buttons: Dict[str, List[QPushButton]] = {}

        self._build_ui()
        self.update_content(self.key_to_description, self.help_entries)

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout()

        self.help_list = QListWidget()
        self.help_list.setMinimumWidth(220)
        main_layout.addWidget(self.help_list)

        keyboard_layout = QVBoxLayout()
        self.label = QLabel("Touches utilisées (en vert clair) :")
        self.label.setFont(QFont("Arial", 10))
        keyboard_layout.addWidget(self.label)

        grid = QGridLayout()
        for row_idx, row in enumerate(self.KEYBOARD_LAYOUT):
            for col_idx, key in enumerate(row):
                btn = QPushButton(key)
                btn.setFont(QFont("Arial", 9))
                btn.setFixedSize(120, 65)
                key_upper = key.upper()
                btn.clicked.connect(lambda checked, key=key_upper: self.show_command(key))
                self._button_base_labels[btn] = key
                self._keyboard_buttons.setdefault(key_upper, []).append(btn)
                grid.addWidget(btn, row_idx, col_idx)

        keyboard_layout.addLayout(grid)
        main_layout.addLayout(keyboard_layout)
        self.setLayout(main_layout)

    def _populate_help_entries(self, entries: Iterable[str]) -> None:
        self.help_list.clear()
        has_entries = False
        for entry in entries:
            item = QListWidgetItem(entry)
            text = entry
            if entry.startswith("-"):
                font = QFont("Courier New", 10, QFont.Bold)
                item.setFont(font)
                item.setForeground(QColor("royalblue"))
                text = entry[1:]
            else:
                font = QFont("Arial", 8, QFont.Bold)
                item.setFont(font)
                item.setForeground(QColor("white"))
            item.setText(text)
            self.help_list.addItem(item)
            has_entries = True

        self.help_list.setVisible(has_entries)

    def update_content(self, key_to_description, help_entries=None):
        self.key_to_description = dict(key_to_description)
        self.help_entries = list(help_entries or [])

        self._populate_help_entries(self.help_entries)

        default_font = QFont("Arial", 9)
        command_font = QFont("Courier", 7)

        for key_upper, buttons in self._keyboard_buttons.items():
            description = self.key_to_description.get(key_upper)
            for btn in buttons:
                base_text = self._button_base_labels[btn]
                if description:
                    btn.setStyleSheet("background-color: lightgreen;")
                    btn.setText(f"{base_text}\n{description}")
                    btn.setFont(command_font)
                else:
                    btn.setStyleSheet("background-color: gray;")
                    btn.setText(base_text)
                    btn.setFont(default_font)

        self.label.setText("Touches utilisées (en vert clair) :")

    def show_command(self, key):
        if key in self.key_to_description:
            command = self.key_to_description[key]
            self.label.setText(f"Commande: {command}")

    def closeEvent(self, event):
        self.hide()
        event.ignore()
