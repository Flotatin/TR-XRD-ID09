import pyqtgraph as pg

LIGHT_STYLESHEET = """
/* Police scientifique */
* {
    font-family: 'Bitstream Vera Sans Mono', monospace;
    font-size: 12pt;
}

/* Fond principal */
QMainWindow {
    background-color: #f3f7fb;
}

/* Style g\xC3\xA9n\xC3\xA9ral des widgets */
QWidget {
    color: #22303a;
    background-color: #f8fafc;
}

/* Combobox */
QComboBox {
    border: 1px solid #50b8de;
    border-radius: 4px;
    padding: 5px;
    background-color: #eef5fa;
    color: #22303a;
}
QComboBox QAbstractItemView {
    selection-background-color: #c6e6f7;
    selection-color: #17202a;
}

/* Boutons */
QPushButton {
    background-color: #50b8de;
    color: #ffffff;
    border-radius: 5px;
    padding: 8px;
    font-size: 14px;
}
QPushButton:hover {
    background-color: #3299c5;
}
QPushButton:pressed {
    background-color: #217ca3;
}

/* Champs de saisie */
QLineEdit, QSpinBox, QTextEdit, QDoubleSpinBox {
    background-color: #eef5fa;
    color: #22303a;
    border: 1px solid #50b8de;
    border-radius: 4px;
    padding: 5px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #ffa94d;
}

/* Menus et barres de menu */
QMenuBar {
    background-color: #e6eff6;
    color: #22303a;
}
QMenu {
    background-color: #e6eff6;
    color: #22303a;
}
QMenu::item:selected {
    background-color: #bae3fa;
    color: #ef7100;
}

/* Barres de d\xC3\xA9filement */
QScrollBar:vertical, QScrollBar:horizontal {
    background: #e0eaf2;
    width: 12px;
}
QScrollBar::handle {
    background: #50b8de;
    border-radius: 6px;
}
QScrollBar::handle:hover {
    background: #3299c5;
}

/* Cases \xC3\xA0 cocher et boutons radio */
QCheckBox, QRadioButton {
    color: #22303a;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #50b8de;
}

/* Barres d'onglets */
QTabBar::tab {
    background-color: #eef5fa;
    color: #22303a;
    padding: 6px;
    border-radius: 5px;
}
QTabBar::tab:selected {
    background-color: #50b8de;
    color: white;
}
"""

DARK_STYLESHEET = """
/* Appliquer la police scientifique */
* {
    font-family: 'Bitstream Vera Sans Mono', monospace;
    font-size: 12pt;
}

/* Fond principal */
QMainWindow {
    background-color: #2b2b2b;
}

/* Style g\xC3\xA9n\xC3\xA9ral des widgets */
QWidget {
    color: #e0e0e0;
    background-color: #333333;
}

/* Combobox */
QComboBox {
    border: 1px solid #0099cc;
    border-radius: 4px;
    padding: 5px;
}
QComboBox QAbstractItemView {
    selection-background-color: grey;
    selection-color: white;
}

/* Boutons */
QPushButton {
    background-color: #0099cc;
    color: #ffffff;
    border-radius: 5px;
    padding: 8px;
    font-size: 14px;
}
QPushButton:hover {
    background-color: #0077aa;
}
QPushButton:pressed {
    background-color: #005577;
}

/* Champs de saisie */
QLineEdit, QSpinBox, QTextEdit ,QDoubleSpinBox{
    background-color: #444444;
    color: #e0e0e0;
    border: 1px solid #0099cc;
    border-radius: 4px;
    padding: 5px;
}
QLineEdit:focus, QSpinBox:focus {
    border: 1px solid #ffaa55;
}
QLineEdit:focus, QDoubleSpinBox:focus {
    border: 1px solid #ffaa55;
}

/* Menus et barres de menu */
QMenuBar {
    background-color: #222222;
    color: #e0e0e0;
}
QMenu {
    background-color: #222222;
    color: #e0e0e0;
}
QMenu::item:selected {
    background-color: #555555;
    color: #ffaa55;
}

/* Barres de d\xC3\xA9filement */
QScrollBar:vertical, QScrollBar:horizontal {
    background: #444444;
    width: 12px;
}
QScrollBar::handle {
    background: #0099cc;
    border-radius: 6px;
}
QScrollBar::handle:hover {
    background: #0077aa;
}

/* Cases \xC3\xA0 cocher et boutons radio */
QCheckBox, QRadioButton {
    color: #e0e0e0;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #0099cc;
}

/* Barres d onglets */
QTabBar::tab {
    background-color: #444444;
    color: #e0e0e0;
    padding: 6px;
    border-radius: 5px;
}
QTabBar::tab:selected {
    background-color: #0099cc;
    color: white;
}
"""

def qt_stylesheet(light=True):
    """Return the stylesheet for the selected theme."""
    return LIGHT_STYLESHEET if light else DARK_STYLESHEET


def configure_pyqtgraph(light=True):
    """Configure global pyqtgraph options to match the theme."""
    if light:
        pg.setConfigOption('background', '#e6eefa')
        pg.setConfigOption('foreground', '#22303a')
    else:
        pg.setConfigOption('background', '#2b2b2b')
        pg.setConfigOption('foreground', '#e0e0e0')
    pg.setConfigOption('antialias', True)


def configure_axes(ax, light=True):
    """Apply axis styling to a :class:`~pyqtgraph.PlotItem`."""
    if light:
        axis_color = '#50b8de'
        label_color = '#22303a'
        grid_alpha = 0.25
        ax.getViewBox().setBackgroundColor('#f8fafc')
    else:
        axis_color = '#e0e0e0'
        label_color = '#e0e0e0'
        grid_alpha = 0.3
    ax.getAxis('bottom').setPen(pg.mkPen(color=axis_color, width=2))
    ax.getAxis('left').setPen(pg.mkPen(color=axis_color, width=2))
    ax.getAxis('bottom').setTextPen(pg.mkPen(color=label_color))
    ax.getAxis('left').setTextPen(pg.mkPen(color=label_color))
    if hasattr(ax, 'titleLabel') and ax.titleLabel is not None:
        ax.setTitle(ax.titleLabel.text, color=label_color)
    ax.showGrid(x=True, y=True, alpha=grid_alpha)
