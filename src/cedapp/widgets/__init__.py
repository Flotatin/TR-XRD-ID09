"""Widgets package for the DRX application."""

from cedapp.widgets.ced_widgets import DdacWidget
from cedapp.widgets.editors import JcpdsEditor
from cedapp.widgets.helpers import (
    ProgressDialog,
    creat_spin_label,
    load_command_file,
    load_help_file,
)
from cedapp.widgets.keyboard import KeyboardWindow
from cedapp.widgets.settings_dialog import SettingsDialog
from cedapp.widgets.spectrum_section_widget import SpectrumSectionWidget
from cedapp.widgets.tab_section_widget import TabSectionWidget ,FitParamWidget


__all__ = [
    "KeyboardWindow",
    "DdacWidget",
    "JcpdsEditor",
    "SettingsDialog",
    "SpectrumSectionWidget",
    "TabSectionWidget",
    "creat_spin_label",
    "load_help_file",
    "load_command_file",
    "ProgressDialog",
    "FitParamWidget",
]
