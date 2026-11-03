"""Single registry of the QActions: menus, toolbar and keyboard shortcuts come from here.

Single letters (R, U, C, Y, M, D...) are left free for the interactive fit,
which will take the v1 shortcuts; F3 saves, as in the v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QWidget


@dataclass
class ActionInfo:
    key: str
    menu: str
    text: str
    shortcuts: Sequence[str]
    tip: str


class ActionRegistry:
    def __init__(self, parent: QWidget) -> None:
        self.parent = parent
        self.actions: Dict[str, QAction] = {}
        self.info: Dict[str, ActionInfo] = {}

    def add(self, key: str, menu: str, text: str, callback: Callable[[], None],
            shortcuts: Sequence[str] = (), tip: str = "", checkable: bool = False) -> QAction:
        if key in self.actions:
            raise ValueError(f"action {key!r} already registered")
        action = QAction(text, self.parent)
        action.setShortcuts([QKeySequence(s) for s in shortcuts])
        action.setCheckable(checkable)
        if tip:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        action.triggered.connect(lambda checked=False: callback())
        self.parent.addAction(action)  # shortcuts work wherever the focus is in the window
        self.actions[key] = action
        self.info[key] = ActionInfo(key, menu, text.replace("&", ""), tuple(shortcuts), tip)
        return action

    def get(self, key: str) -> QAction:
        return self.actions[key]

    def menus(self) -> List[str]:
        return list(dict.fromkeys(info.menu for info in self.info.values()))

    def in_menu(self, menu: str) -> List[QAction]:
        return [self.actions[k] for k, info in self.info.items() if info.menu == menu]

    def shortcut_table(self) -> List[tuple]:
        """(menu, action, shortcuts) for the help window."""
        return [(i.menu.replace("&", ""), i.text, ", ".join(QKeySequence(s).toString(QKeySequence.NativeText) for s in i.shortcuts))
                for i in self.info.values() if i.shortcuts]

    def find(self, shortcut: str) -> Optional[str]:
        target = QKeySequence(shortcut)
        return next((k for k, i in self.info.items() if any(QKeySequence(s) == target for s in i.shortcuts)), None)
