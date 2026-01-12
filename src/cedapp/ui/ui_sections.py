"""UI section builders for the DRX main window."""

from __future__ import annotations

from dataclasses import dataclass, field
import matplotlib.colors as mcolors

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cedapp.widgets import (
    DdacWidget,
    SpectrumSectionWidget,
    TabSectionWidget,
    creat_spin_label,
    load_command_file,
    FitParamWidget,
)


@dataclass
class UIState:
    """Container for UI widget references."""

    settings_button: QToolButton | None = None
    help_toggle_btn: QPushButton | None = None
    clear_btn: QPushButton | None = None
    help_widget: QWidget | None = None
    CommandeLayout: QVBoxLayout | None = None
    help_entries: list[str] = field(default_factory=list)
    helpLabel: QListWidget | None = None
    list_Commande: QListWidget | None = None
    list_Commande_python: list[str] = field(default_factory=list)
    widget_python: QWidget | None = None
    text_edit: QTextEdit | None = None
    execute_button: QPushButton | None = None
    output_display: QTextEdit | None = None
    ButtonPrint: QPushButton | None = None
    ButtonLen: QPushButton | None = None
    ButtonClearcode: QPushButton | None = None
    help_tab_index: int | None = None
    help_tab_visible: bool = False

    select_file_DRX_button: QPushButton | None = None
    file_label_spectro: QLabel | None = None
    select_file_oscilo_button: QPushButton | None = None
    file_label_oscilo: QLabel | None = None
    Calibration_DRX_button: QPushButton | None = None
    plot_fit_toggle: QPushButton | None = None
    setup_mode_button: QPushButton | None = None
    DRX_selector: QComboBox | None = None
    type_selector: QComboBox | None = None
    listbox_file: QListWidget | None = None
    search_bar: QLineEdit | None = None

    gauge_table: QTableWidget | None = None
    name_gauge: QLabel | None = None
    energy_label: QLabel | None = None
    tab_section_widget: TabSectionWidget | None = None

    text_box_msg: QLabel | None = None

    ParampicLayout: QVBoxLayout | None = None
    fit_param_widget: FitParamWidget | None = None
    coef_dynamic_spinbox: list[QDoubleSpinBox] = field(default_factory=list)
    coef_dynamic_label: list[QLabel] = field(default_factory=list)
    model_pic_type_selector: QComboBox | None = None
    liste_type_model_pic: list[str] = field(default_factory=list)
    model_pic_fit: str | None = None
    spinbox_sigma: QDoubleSpinBox | None = None

    spectrum_section_widget: SpectrumSectionWidget | None = None
    ddac_widget: DdacWidget | None = None
    spinbox_P: QDoubleSpinBox | None = None
    spinbox_T: QDoubleSpinBox | None = None
    listbox_pic: QListWidget | None = None


class CalibrationButton(QPushButton):
    """Push button that emits a double-click signal."""

    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


def build_command_panel(window) -> None:
    """Create the settings panel together with the help/command widgets."""

    state = window.ui_state
    box = QGroupBox("⬩")
    layout = QVBoxLayout()
    state.settings_button = QToolButton()
    state.settings_button.setText("⚙")
    state.settings_button.clicked.connect(window.open_settings_dialog)
    layout.addWidget(state.settings_button)

    state.help_toggle_btn = QPushButton("Help", window)
    state.help_toggle_btn.clicked.connect(window.toggle_help_box)
    layout.addWidget(state.help_toggle_btn)

    state.clear_btn = QPushButton("SAVE Summary")
    state.clear_btn.clicked.connect(window.save_summary_CED)
    layout.addWidget(state.clear_btn)

    state.help_widget = QWidget()
    state.CommandeLayout = QVBoxLayout()
    state.help_entries = []

    # Help entries kept for the keyboard window only
    state.helpLabel = QListWidget()
    state.helpLabel.itemDoubleClicked.connect(window.try_command)
    state.helpLabel.hide()

    commands_title = QLabel("Commandes")
    state.CommandeLayout.addWidget(commands_title)

    state.list_Commande = QListWidget()
    state.CommandeLayout.addWidget(state.list_Commande)
    state.list_Commande.itemClicked.connect(window.display_command)

    state.list_Commande_python = []
    load_command_file(state.list_Commande, state.list_Commande_python, window.file_command)
    state.widget_python = QWidget()
    prompt_layout = QVBoxLayout()
    state.text_edit = QTextEdit(window)
    state.text_edit.setPlaceholderText(
        "Enter your python code here, to use libraries start with CL., example: np.pi -> CL.np.pi..."
    )
    prompt_layout.addWidget(state.text_edit)

    state.execute_button = QPushButton("Click to run code (Shift + Entry)", window)
    state.execute_button.clicked.connect(window.execute_code)
    prompt_layout.addWidget(state.execute_button)

    state.output_display = QTextEdit(window)
    state.output_display.setReadOnly(True)
    state.output_display.setPlaceholderText("Output print...")
    prompt_layout.addWidget(state.output_display)

    state.widget_python.setLayout(prompt_layout)
    state.CommandeLayout.addWidget(state.widget_python)

    state.ButtonPrint = QPushButton("print(...)")
    state.ButtonPrint.clicked.connect(window.code_print)
    state.CommandeLayout.addWidget(state.ButtonPrint)

    state.ButtonLen = QPushButton("len(...)")
    state.ButtonLen.clicked.connect(window.code_len)
    state.CommandeLayout.addWidget(state.ButtonLen)

    state.ButtonClearcode = QPushButton("Clear")
    state.ButtonClearcode.clicked.connect(window.code_clear)
    state.CommandeLayout.addWidget(state.ButtonClearcode)

    state.help_widget.setLayout(state.CommandeLayout)
    state.help_widget.setVisible(False)
    layout.addWidget(state.help_widget)

    box.setLayout(layout)
    window.grid_layout.addWidget(box, 0, 5, 5, 1)

    state.help_tab_index = None
    state.help_tab_visible = False
    window._update_help_button_color(state.help_tab_visible)


def build_file_section(window) -> None:
    """Create the file loading controls and python tooling panel."""

    state = window.ui_state
    file_box = QGroupBox("File loading")
    file_layout = QVBoxLayout()
    row_layout = QHBoxLayout()

    state.select_file_DRX_button = QPushButton("f_DRX ", window)
    state.select_file_DRX_button.clicked.connect(window.select_file_DRX)
    row_layout.addWidget(state.select_file_DRX_button)

    state.file_label_spectro = QLabel("init", window)
    row_layout.addWidget(state.file_label_spectro)

    state.select_file_oscilo_button = QPushButton("f_Oscillo", window)
    state.select_file_oscilo_button.clicked.connect(window.select_file_oscilo)
    row_layout.addWidget(state.select_file_oscilo_button)

    state.file_label_oscilo = QLabel("init", window)
    row_layout.addWidget(state.file_label_oscilo)

    state.Calibration_DRX_button = CalibrationButton("Calibration", window)
    state.Calibration_DRX_button.clicked.connect(window.Calibration_DRX)
    state.Calibration_DRX_button.doubleClicked.connect(window._open_calibration_dialog)
    row_layout.addWidget(state.Calibration_DRX_button)

    bouton_configue = QPushButton("Save config")
    bouton_configue.clicked.connect(window.save_paths_to_txt)
    row_layout.addWidget(bouton_configue)

    state.setup_mode_button = QPushButton("Setup mode", window)
    state.setup_mode_button.clicked.connect(window._run_setup_mode)
    row_layout.addWidget(state.setup_mode_button)

    state.plot_fit_toggle = QPushButton("Plot fit", window)
    state.plot_fit_toggle.setCheckable(True)
    initial_plot_visibility = getattr(window.plot_fit_start, "isVisible", lambda: True)()
    state.plot_fit_toggle.setChecked(initial_plot_visibility)

    def update_plot_fit(checked: bool) -> None:
        window.plot_fit_start.setVisible(checked)
        color = "lightgreen" if checked else "lightcoral"
        state.plot_fit_toggle.setStyleSheet(f"background-color: {color};")
        if checked:
            window.Print_fit_start()

    state.plot_fit_toggle.toggled.connect(update_plot_fit)
    update_plot_fit(state.plot_fit_toggle.isChecked())
    row_layout.addWidget(state.plot_fit_toggle)

    file_layout.addLayout(row_layout)
    file_layout.addWidget(state.text_box_msg)
    file_box.setLayout(file_layout)
    window.grid_layout.addWidget(file_box, 4, 2, 1, 3)

    group_fichiers = QGroupBox("File gestion")
    layout_fichiers = QVBoxLayout()
    state.DRX_selector = QComboBox(window)
    if window.RUN is not None:
        for i in range(len(window.RUN.Spectra)):
            state.DRX_selector.addItem(f"drx_{i}")
    layout_fichiers.addWidget(state.DRX_selector)

    state.type_selector = QComboBox()
    window.type_folder = ["CED", "Oscilloscope", "DRX"]
    state.type_selector.addItems(window.type_folder)
    state.type_selector.currentIndexChanged.connect(window.f_change_file_type)
    layout_fichiers.addWidget(state.type_selector)

    state.listbox_file = QListWidget()
    state.listbox_file.doubleClicked.connect(window.f_select_file)
    layout_fichiers.addWidget(state.listbox_file)

    state.search_bar = QLineEdit()
    state.search_bar.setPlaceholderText("Search...")
    state.search_bar.textChanged.connect(window.f_filter_files)
    layout_fichiers.addWidget(state.search_bar)

    window.dict_folders = {"CED": "", "Oscilloscope": "", "DRX": ""}
    window.loaded_file_DRX = ""
    window.loaded_file_OSC = ""
    window.zones = []
    window.current_file_list = []

    bouton_dossier = QPushButton("Select folder")
    bouton_dossier.clicked.connect(window.select_folder_dict)
    layout_fichiers.addWidget(bouton_dossier)

    group_fichiers.setLayout(layout_fichiers)
    window.grid_layout.addWidget(group_fichiers, 3, 0, 2, 2)


def build_tools_panel(window) -> None:
    """Create widgets controlling spectrum processing parameters."""

    state = window.ui_state
    param_box = QGroupBox("Tools")
    layout = QVBoxLayout()

    gauge_box = QGroupBox("Gauges overview")
    gauge_layout = QVBoxLayout()

    state.gauge_table = QTableWidget(0, 2, window)
    state.gauge_table.setHorizontalHeaderLabels(["Gauge", "State"])
    state.gauge_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    state.gauge_table.horizontalHeader().setStretchLastSection(True)
    state.gauge_table.verticalHeader().setVisible(False)
    state.gauge_table.cellDoubleClicked.connect(window.on_gauge_table_double_clicked)
    state.gauge_table.itemSelectionChanged.connect(window.on_gauge_table_selection_changed)
    gauge_layout.addWidget(state.gauge_table)

    state.name_gauge = QLabel("Add ?")
    gauge_layout.addWidget(state.name_gauge)

    state.energy_label = QLabel(window._format_energy_label())
    gauge_layout.addWidget(state.energy_label)

    gauge_box.setLayout(gauge_layout)
    layout.addWidget(gauge_box)

    state.tab_section_widget = TabSectionWidget(window)
    layout.addWidget(state.tab_section_widget)

    param_box.setLayout(layout)
    window.grid_layout.addWidget(param_box, 0, 0, 2, 2)


def build_message_label(window) -> None:
    """Display a status message label."""

    window.ui_state.text_box_msg = QLabel("Good Luck and Have Fun")


def build_model_peak_section(window) -> None:
    """Configure the model peak parameter widgets."""

    state = window.ui_state
    parampic_box = QGroupBox("Model peak")
    state.ParampicLayout = QVBoxLayout()
    state.fit_param_widget = FitParamWidget(window)
    state.ParampicLayout.addWidget(state.fit_param_widget)

    state.coef_dynamic_spinbox, state.coef_dynamic_label = [], []

    state.model_pic_type_selector = QComboBox(window)
    state.liste_type_model_pic = ["PearsonIV", "PseudoVoigt", "Moffat", "SplitLorentzian", "Gaussian"]
    state.model_pic_type_selector.addItems(state.liste_type_model_pic)
    tableau_colors = list(mcolors.TABLEAU_COLORS.values())
    for ind in range(state.model_pic_type_selector.count()):
        color = tableau_colors[ind % len(tableau_colors)]
        item = state.model_pic_type_selector.model().item(ind)
        if item is not None:
            item.setBackground(QColor(color))
    state.model_pic_type_selector.currentIndexChanged.connect(window.f_model_pic_type)
    state.ParampicLayout.addWidget(state.model_pic_type_selector)
    state.model_pic_fit = state.model_pic_type_selector.currentText()

    state.spinbox_sigma = QDoubleSpinBox()
    state.spinbox_sigma.valueChanged.connect(window.setFocus)
    state.spinbox_sigma.valueChanged.connect(
        lambda _value: window._update_fit_window() if getattr(window, "index_pic_select", None) is not None else None
    )
    state.spinbox_sigma.setRange(0.01, 10)
    state.spinbox_sigma.setSingleStep(0.01)
    state.spinbox_sigma.setValue(0.15)
    state.ParampicLayout.addLayout(creat_spin_label(state.spinbox_sigma, "σ :"))

    parampic_box.setLayout(state.ParampicLayout)
    window.grid_layout.addWidget(parampic_box, 2, 0, 1, 2)

    window.bit_bypass = True
    window.f_model_pic_type()
    window.bit_bypass = False


def init_plot_widgets(window) -> None:
    """Initialise the main spectrum plot area and related items."""
    state = window.ui_state
    state.spectrum_section_widget = SpectrumSectionWidget(window)
    state.spectrum_section_widget.add_to_layout(window.grid_layout)


def build_ddac_section(window) -> None:
    """Create the dDAC plots widget."""
    state = window.ui_state
    state.ddac_widget = DdacWidget(window)
    state.ddac_widget.add_to_layout(window.grid_layout)


def build_gauge_section(window) -> None:
    """Initialise widgets related to gauge information."""

    state = window.ui_state
    add_box = QGroupBox("Gauge information")
    layout = QVBoxLayout()

    layh4 = QHBoxLayout()
    layh4.addWidget(QLabel("P="), alignment=Qt.AlignRight)
    state.spinbox_P = QDoubleSpinBox()
    state.spinbox_P.setRange(-10.0, 1000.0)
    state.spinbox_P.setSingleStep(0.1)
    state.spinbox_P.setValue(0.0)
    layh4.addWidget(state.spinbox_P)
    layh4.addWidget(QLabel("GPa"))
    window.deltalambdaP = 0

    layh4.addWidget(QLabel("T="), alignment=Qt.AlignRight)
    state.spinbox_T = QDoubleSpinBox()
    state.spinbox_T.setRange(0, 3000)
    state.spinbox_T.setSingleStep(1)
    state.spinbox_T.setValue(293)
    state.spinbox_T.setEnabled(False)
    layh4.addWidget(state.spinbox_T)
    layh4.addWidget(QLabel("K"))
    window.deltalambdaT = 0
    layout.addLayout(layh4)

    state.listbox_pic = QListWidget()
    state.listbox_pic.doubleClicked.connect(window.select_pic)
    layout.addWidget(state.listbox_pic)

    add_box.setLayout(layout)
    window.AddBox = add_box
    if getattr(window.ui_state, "spectrum_section_widget", None) is not None:
        window.ui_state.spectrum_section_widget.add_right_widget(add_box)
    window.bit_modif_PTlambda = False
