"""Shared helpers for compact popup-menu based controls."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction, QMenu, QToolButton


def make_menu_button(title: str, parent=None) -> QToolButton:
    """Create a compact text-only tool button that opens a popup menu."""

    button = QToolButton(parent)
    button.setText(title)
    button.setPopupMode(QToolButton.InstantPopup)
    button.setToolButtonStyle(Qt.ToolButtonTextOnly)
    button.setMenu(QMenu(button))
    return button


def add_menu_action(menu: QMenu, text: str, slot) -> QAction:
    """Append a regular action to *menu* and connect it to *slot*."""

    action = QAction(text, menu)
    action.triggered.connect(lambda _checked=False: slot())
    menu.addAction(action)
    return action


def add_check_menu_action(menu: QMenu, text: str, checked: bool, slot) -> QAction:
    """Append a checkable action to *menu* and connect its toggled state."""

    action = QAction(text, menu)
    action.setCheckable(True)
    action.setChecked(checked)
    action.toggled.connect(slot)
    menu.addAction(action)
    return action
