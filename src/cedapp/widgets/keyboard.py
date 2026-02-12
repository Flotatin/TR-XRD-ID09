from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton, QSizePolicy
)

# --- Layout clavier (comme avant) ---
KEYBOARD_LAYOUT: List[List[str]] = [
    ["Esc", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"],
    ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "Backspace"],
    ["Tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]", "\\"],
    ["Caps", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'", "Enter"],
    ["Shift", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "Shift"],
    ["Ctrl", "Win", "Alt", "Space", "Alt", "Fn", "Menu", "Ctrl"],
]

# Couleurs par catégorie (optionnel)
CATEGORY_COLORS = {
    "fit": "#7ec8e3",
    "file": "#f7c873",
    "analysis": "#d5a6ff",
    "ui": "#c9c9c9",
    "editing": "#8fd694",
    "other": "#bdbdbd",
}

# Mapping "label clavier" -> Qt.Key
# (tu peux compléter selon tes besoins)
LABEL_TO_QTKEY = {
    "Esc": Qt.Key_Escape,
    "Tab": Qt.Key_Tab,
    "Caps": Qt.Key_CapsLock,
    "Shift": Qt.Key_Shift,
    "Ctrl": Qt.Key_Control,
    "Alt": Qt.Key_Alt,
    "Win": Qt.Key_Meta,
    "Fn": Qt.Key_unknown,
    "Menu": Qt.Key_Menu,
    "Space": Qt.Key_Space,
    "Enter": Qt.Key_Return,
    "Backspace": Qt.Key_Backspace,
    "Delete": Qt.Key_Delete,
}

def _label_to_keycode(label: str) -> Optional[int]:
    # F-keys
    if label.upper().startswith("F") and label[1:].isdigit():
        n = int(label[1:])
        return getattr(Qt, f"Key_F{n}", None)

    # lettres
    if len(label) == 1 and label.isalpha():
        return getattr(Qt, f"Key_{label.upper()}", None)

    # chiffres
    if len(label) == 1 and label.isdigit():
        return getattr(Qt, f"Key_{label}", None)

    # ponctuation utile
    punct_map = {
        "`": Qt.Key_QuoteLeft,
        "-": Qt.Key_Minus,
        "=": Qt.Key_Equal,
        "[": Qt.Key_BracketLeft,
        "]": Qt.Key_BracketRight,
        "\\": Qt.Key_Backslash,
        ";": Qt.Key_Semicolon,
        "'": Qt.Key_Apostrophe,
        ",": Qt.Key_Comma,
        ".": Qt.Key_Period,
        "/": Qt.Key_Slash,
    }
    if label in punct_map:
        return punct_map[label]

    return LABEL_TO_QTKEY.get(label)


def _mod_label(mod: int) -> str:
    if mod == Qt.NoModifier:
        return ""
    parts = []
    if mod & Qt.ControlModifier:
        parts.append("Ctrl")
    if mod & Qt.ShiftModifier:
        parts.append("Shift")
    if mod & Qt.AltModifier:
        parts.append("Alt")
    return "+".join(parts)


class KeyboardWindow(QWidget):
    """
    Vrai clavier (grille) + couches Normal/Shift/Ctrl.
    Chaque touche affiche l'action correspondant à la couche active.
    Tooltip au survol.
    """

    def __init__(self, shortcuts: Dict[Tuple[int, int], Dict[str, object]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clavier - Raccourcis")
        self.shortcuts = shortcuts

        # couches proposées (tu peux ajouter Alt si tu veux)
        self.layers: List[Tuple[str, int]] = [
            ("Normal", Qt.NoModifier),
            ("Shift", Qt.ShiftModifier),
            ("Ctrl", Qt.ControlModifier),
        ]
        self._active_modifier = Qt.NoModifier

        # boutons clavier par label
        self._buttons: Dict[str, QPushButton] = {}

        self._build_ui()
        self._refresh_keys()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # barre couches
        top = QHBoxLayout()
        top.addWidget(QLabel("Couche :"))

        self._layer_group = QButtonGroup(self)
        self._layer_group.setExclusive(True)

        for i, (name, mod) in enumerate(self.layers):
            rb = QRadioButton(name)
            rb.setChecked(i == 0)
            rb.toggled.connect(lambda checked, m=mod: self._on_layer_changed(checked, m))
            self._layer_group.addButton(rb)
            top.addWidget(rb)

        top.addStretch(1)
        self._info = QLabel("")
        self._info.setStyleSheet("color: #dddddd;")
        top.addWidget(self._info)
        root.addLayout(top)

        # grille clavier
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)

        font_base = QFont("Arial", 8)
        for r, row in enumerate(KEYBOARD_LAYOUT):
            c = 0
            for label in row:
                btn = QPushButton()
                btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                btn.setMinimumSize(85, 55)
                btn.setFont(font_base)
                btn.setObjectName(f"key_{label}")

                # stocke le label “physique”
                btn._kbd_label = label  # type: ignore[attr-defined]

                btn.clicked.connect(lambda _=False, lab=label: self._on_key_clicked(lab))

                self._buttons[label] = btn
                grid.addWidget(btn, r, c)
                c += 1

        root.addLayout(grid)
        self.resize(1200, 420)

    def _on_layer_changed(self, checked: bool, modifier: int):
        if not checked:
            return
        self._active_modifier = modifier
        self._refresh_keys()

    def _on_key_clicked(self, label: str):
        # juste informatif (pas d’exécution)
        keycode = _label_to_keycode(label)
        if keycode is None:
            self._info.setText(f"{label} : (pas de mapping Qt.Key)")
            return

        action = self._get_action_for(keycode, self._active_modifier)
        if action is None:
            self._info.setText(f"{_mod_label(self._active_modifier)} {label} : aucune action")
            return

        name = action.get("name") or "Unnamed"
        self._info.setText(f"{_mod_label(self._active_modifier)} {label} → {name}")

    def _get_action_for(self, keycode: int, modifier: int) -> Optional[Dict[str, object]]:
        # match exact
        act = self.shortcuts.get((keycode, modifier))
        if act is not None:
            return act

        # support allow_extra_modifiers (comme ton keyPressEvent)
        if modifier != Qt.NoModifier:
            for (k, mask), cand in self.shortcuts.items():
                if k != keycode or mask == Qt.NoModifier:
                    continue
                if not cand.get("allow_extra_modifiers"):
                    continue
                if (modifier & mask) == mask:
                    return cand

        # fallback NoModifier
        return self.shortcuts.get((keycode, Qt.NoModifier))

    def _all_modifiers_present_for_key(self, keycode: int) -> List[int]:
        mods = []
        for _, mod in self.layers:
            if self._get_action_for(keycode, mod) is not None:
                mods.append(mod)
        return mods

    def _refresh_keys(self):
        for label, btn in self._buttons.items():
            keycode = _label_to_keycode(label)

            # base text (toujours)
            base = label

            if keycode is None:
                btn.setText(base)
                btn.setToolTip("")
                btn.setStyleSheet("background-color: #666; color: white;")
                continue

            action = self._get_action_for(keycode, self._active_modifier)
            mods_present = self._all_modifiers_present_for_key(keycode)

            # Indication si action existe dans d'autres couches
            layers_badge = ""
            if mods_present:
                tags = []
                for m in mods_present:
                    if m == Qt.NoModifier:
                        tags.append("N")
                    elif m == Qt.ShiftModifier:
                        tags.append("S")
                    elif m == Qt.ControlModifier:
                        tags.append("C")
                    elif m == Qt.AltModifier:
                        tags.append("A")
                layers_badge = " [" + "".join(tags) + "]"

            if action is None:
                btn.setText(base + layers_badge)
                btn.setToolTip(f"{base} : aucune action dans la couche actuelle")
                btn.setStyleSheet("background-color: #444; color: #ddd;")
                continue

            name = action.get("name") or "Unnamed"
            desc = action.get("description") or ""
            cat = action.get("category") or "other"

            color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])

            # Texte sur 2 lignes
            btn.setText(f"{base}{layers_badge}\n{name}")

            # Tooltip riche
            modtxt = _mod_label(self._active_modifier)
            combo = f"{modtxt+'+' if modtxt else ''}{base}"
            handler = action.get("handler")
            handler_txt = ""
            if handler is not None:
                handler_txt = f"\nHandler: {getattr(handler, '__name__', str(handler))}"

            tooltip = f"{combo}\n{name}"
            if desc:
                tooltip += f"\n\n{desc}"
            tooltip += handler_txt
            btn.setToolTip(tooltip)

            # style
            btn.setStyleSheet(
                f"background-color: {color}; color: black; border-radius: 6px;"
            )
