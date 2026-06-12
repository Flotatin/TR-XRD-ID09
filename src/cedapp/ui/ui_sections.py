"""UI section builders for the DRX main window."""

from __future__ import annotations

from dataclasses import dataclass, field
import matplotlib.colors as mcolors
import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAction,
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QCheckBox,
    QDialog,
    QGridLayout,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QSpinBox,
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
from cedapp.widgets.menu_helpers import (
    add_check_menu_action,
    add_menu_action,
    make_menu_button,
)


@dataclass
class UIState:
    """Container for UI widget references."""

    settings_button: QToolButton | None = None
    help_toggle_btn: QToolButton | None = None
    live_toggle_btn: QPushButton | None = None
    clear_btn: QAction | None = None
    help_widget: QWidget | None = None
    CommandeLayout: QVBoxLayout | None = None
    help_entries: list[str] = field(default_factory=list)
    helpLabel: QListWidget | None = None
    list_Commande: QListWidget | None = None
    list_Commande_python: list[str] = field(default_factory=list)
    widget_python: QWidget | None = None
    text_edit: QTextEdit | None = None
    execute_button: QAction | None = None
    output_display: QTextEdit | None = None
    ButtonPrint: QAction | None = None
    ButtonLen: QAction | None = None
    ButtonClearcode: QAction | None = None
    help_tab_index: int | None = None
    help_tab_visible: bool = False

    select_file_DRX_button: QAction | None = None
    file_label_spectro: QLabel | None = None
    select_file_oscilo_button: QAction | None = None
    file_label_oscilo: QLabel | None = None
    Calibration_DRX_button: QAction | None = None
    detector_distance_button: QAction | None = None
    plot_fit_toggle: QAction | None = None
    setup_mode_button: QPushButton | None = None
    DRX_selector: QComboBox | None = None
    type_selector: QComboBox | None = None
    listbox_file: QListWidget | None = None
    search_bar: QLineEdit | None = None
    main_controls_layout: QHBoxLayout | None = None
    ddac_menu_button: QToolButton | None = None
    jungfrau_mode_box: QComboBox | None = None

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
    label_CED: QLabel | None = None
    spectrum_select_box: QAction | None = None
    analysis_toggle: QCheckBox | None = None
    cb_piezo_consigne: QCheckBox | None = None
    label_plot_metric: QLabel | None = None
    cedx_metric_combo: QComboBox | None = None
    label_dpdt_points: QLabel | None = None
    spinbox_dpdt_points: QSpinBox | None = None
    label_dpdt_smooth: QLabel | None = None
    spinbox_dpdt_smooth: QSpinBox | None = None
    btn_zone_dpdt: QAction | None = None
    btn_time_arg: QAction | None = None
    spinbox_P: QDoubleSpinBox | None = None
    spinbox_T: QDoubleSpinBox | None = None
    apply_temp_all_gauges_checkbox: QCheckBox | None = None
    pt_solver_pfix_checkbox: QCheckBox | None = None
    listbox_pic: QListWidget | None = None
    right_view_tabs: QTabWidget | None = None
    right_view_zoom_action: QPushButton | None = None
    right_view_print_action: QPushButton | None = None
    undock_panel_button: QPushButton | None = None


def configure_main_window(window) -> None:
    """Configure the base main-window container and grid layout."""

    window.setWindowTitle("0.1 vXRD © F.Dembele")
    window.grid_layout = QGridLayout()
    window.setMinimumSize(1200, 780)
    window.grid_layout.setContentsMargins(6, 6, 6, 6)
    window.grid_layout.setSpacing(6)
    central_widget = QWidget()
    central_widget.setLayout(window.grid_layout)
    window.setCentralWidget(central_widget)


def create_piezo_consigne_checkbox(window) -> QCheckBox:
    """Create the piezo setpoint toggle used by the dDAC menu."""

    state = window.ui_state
    state.cb_piezo_consigne = QCheckBox("piézo")
    state.cb_piezo_consigne.setChecked(True)
    state.cb_piezo_consigne.toggled.connect(window._set_piezo_setpoint_visible)
    return state.cb_piezo_consigne


def add_model_peak_coefficient_control(window, text: str, minimum: float, maximum: float, value: float) -> None:
    """Add one dynamic coefficient spinbox to the model-peak parameter panel."""

    row_layout = QHBoxLayout()
    coef_label = QLabel(f"{text}:", window)
    row_layout.addWidget(coef_label)

    spinbox_coef = QDoubleSpinBox(window)
    spinbox_coef.valueChanged.connect(window.setFocus)
    spinbox_coef.setRange(minimum, maximum)
    spinbox_coef.setSingleStep(0.01)
    spinbox_coef.setValue(value)
    row_layout.addWidget(spinbox_coef)

    window.ParampicLayout.addLayout(row_layout)
    window.coef_dynamic_label.append(coef_label)
    window.coef_dynamic_spinbox.append(spinbox_coef)


def create_gauge_panel_dialog(window):
    """Create the floating dialog that receives the right gauge/print panel."""

    dialog = QDialog(window)
    dialog.setWindowTitle("Zoom / Print détaché")
    dialog_layout = QVBoxLayout(dialog)
    dialog_layout.setContentsMargins(8, 8, 8, 8)
    dialog.finished.connect(lambda _result: window._on_gauge_panel_dialog_closed())
    return dialog


def build_print_plate_widget(window) -> None:
    """Create the print-plate replacement widget once during UI initialization."""

    spectrum_section = getattr(window.ui_state, "spectrum_section_widget", None)
    if spectrum_section is None or window.print_plate_window is not None:
        return

    window.print_plate_window = QWidget(window)
    layout = QVBoxLayout(window.print_plate_window)
    controls_layout = QHBoxLayout()
    window._print_plate_mode_button = QPushButton("Image", window.print_plate_window)
    window._print_plate_mode_button.setCheckable(True)
    window._print_plate_mode_button.clicked.connect(window._on_print_plate_mode_toggled)
    controls_layout.addWidget(window._print_plate_mode_button)
    layout.addLayout(controls_layout)

    hist_widget = pg.PlotWidget()
    hist_plot = hist_widget.getPlotItem()
    hist_plot.setLabel("left", "Counts")
    hist_plot.setLabel("bottom", "Pixel value")
    window._print_plate_hist_curve = pg.PlotDataItem([], [], pen=pg.mkPen("k", width=1))
    hist_plot.addItem(window._print_plate_hist_curve)
    window._print_plate_hist_region = pg.LinearRegionItem(
        values=(0, 1), orientation=pg.LinearRegionItem.Vertical
    )
    window._print_plate_hist_region.sigRegionChanged.connect(window._on_print_plate_levels_changed)
    hist_plot.addItem(window._print_plate_hist_region)
    layout.addWidget(hist_widget, 1)

    plate_widget = pg.PlotWidget()
    window.print_plate_plot = plate_widget.getPlotItem()
    window.print_plate_plot.setLabel("left", "pixel Y")
    window.print_plate_plot.setLabel("bottom", "pixel X")
    window.print_plate_plot.invertY(True)
    window.print_plate_image_item = pg.ImageItem(np.zeros((1, 1), dtype=float))
    window.print_plate_plot.addItem(window.print_plate_image_item)
    window.print_plate_plot.scene().sigMouseClicked.connect(window._on_print_plate_clicked)
    layout.addWidget(plate_widget, 4)

    spectrum_section.set_zoom_replacement_widget(window.print_plate_window)


def build_file_section(window) -> None:
    """Create the file loading controls and python tooling panel."""

    state = window.ui_state
    file_box = QGroupBox("0.1 vXRD F.Dembele")
    file_layout = QVBoxLayout()
    row_layout = QHBoxLayout()

    state.live_toggle_btn = QPushButton("Live", window)
    state.live_toggle_btn.setCheckable(True)
    state.live_toggle_btn.toggled.connect(window.toggle_live_mode)
    row_layout.addWidget(state.live_toggle_btn)

    state.jungfrau_mode_box = QComboBox(window)
    state.jungfrau_mode_box.addItems(["Burst","Continue", "Oscillo"])
    state.jungfrau_mode_box.currentTextChanged.connect(window.set_jungfrau_mode)
    row_layout.addWidget(state.jungfrau_mode_box)


    file_menu_button = make_menu_button("Fichier ▾", window)
    file_menu = file_menu_button.menu()
    state.select_file_DRX_button = add_menu_action(
        file_menu, "Ouvrir un fichier DRX…", window.select_file_DRX
    )
    state.select_file_oscilo_button = add_menu_action(
        file_menu, "Ouvrir un fichier oscillo…", window.select_file_oscilo
    )
    file_menu.addSeparator()
    add_menu_action(file_menu, "Sélectionner le dossier affiché…", window.select_folder_dict)
    add_menu_action(file_menu, "Sélectionner le dossier DRX…", window.select_folder_DRX)
    add_menu_action(file_menu, "Sélectionner le dossier oscillo…", window.select_folder_oscilo)
    row_layout.addWidget(file_menu_button)

    state.file_label_spectro = QLabel("init", window)
    row_layout.addWidget(state.file_label_spectro)

    state.file_label_oscilo = QLabel("init", window)
    row_layout.addWidget(state.file_label_oscilo)


    calibration_menu_button = make_menu_button("Calibration ▾", window)
    calibration_menu = calibration_menu_button.menu()
    state.Calibration_DRX_button = add_menu_action(
        calibration_menu, "Calibration rapide", window.Calibration_DRX
    )
    add_menu_action(calibration_menu, "Dialogue calibration avancé…", window._open_calibration_dialog)
    state.detector_distance_button = add_menu_action(
        calibration_menu, "Mettre à jour distance détecteur", window.update_detector_distance_without_integration
    )
    row_layout.addWidget(calibration_menu_button)

    display_menu_button = make_menu_button("Affichage ▾", window)
    display_menu = display_menu_button.menu()
    for label, source_action in (
        ("Spectre brut", getattr(window, "act_show_raw", None)),
        ("Spectre filtré", getattr(window, "act_show_filtered", None)),
        ("Baseline", getattr(window, "act_show_baseline", None)),
    ):
        if source_action is None:
            continue
        mirror_action = add_check_menu_action(
            display_menu, label, source_action.isChecked(), source_action.setChecked
        )
        source_action.toggled.connect(mirror_action.setChecked)
    display_menu.addSeparator()
    for label, checkbox in (
        ("Sélection pic au clic", getattr(window, "select_clic_box", None)),
        ("Zone Fit Spectrum", getattr(window, "zone_spectrum_box", None)),
        ("vslmfit", getattr(window, "vslmfit", None)),
    ):
        if checkbox is None:
            continue
        mirror_action = add_check_menu_action(
            display_menu, label, checkbox.isChecked(), checkbox.setChecked
        )
        checkbox.toggled.connect(mirror_action.setChecked)
    display_menu.addSeparator()
    initial_plot_visibility = getattr(window.plot_fit_start, "isVisible", lambda: True)()

    def update_plot_fit(checked: bool) -> None:
        window.plot_fit_start.setVisible(checked)
        if checked:
            window.Print_fit_start()

    state.plot_fit_toggle = add_check_menu_action(
        display_menu, "Afficher le plot fit", initial_plot_visibility, update_plot_fit
    )
    update_plot_fit(state.plot_fit_toggle.isChecked())
    row_layout.addWidget(display_menu_button)

    state.ddac_menu_button = make_menu_button("dDAC ▾", window)
    row_layout.addWidget(state.ddac_menu_button)
    
    state.help_toggle_btn = make_menu_button(chr(0x2699), window)
    help_menu = state.help_toggle_btn.menu()
    add_menu_action(help_menu, "Module debug", window.toggle_help_box)
    add_menu_action(help_menu, "Settings", window.open_settings_dialog)
    
    add_menu_action(help_menu, "Setup mode", window._run_setup_mode)
    row_layout.addWidget(state.help_toggle_btn)

    advanced_button = make_menu_button(chr(0x1F4BE), window)
    advanced_menu = advanced_button.menu()
    state.clear_btn = add_menu_action(advanced_menu, "Exporter / Save Summary", window.save_summary_CED)
    add_menu_action(advanced_menu, "Sauver la configuration", window.save_paths_to_txt)
    advanced_menu.addSeparator()
    row_layout.addWidget(advanced_button)

    state.analysis_toggle = None

    state.main_controls_layout = row_layout
    file_layout.addLayout(row_layout)
    file_layout.addWidget(state.text_box_msg)

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

    console_button = make_menu_button("Console Python ▾", window)
    console_menu = console_button.menu()
    state.execute_button = add_menu_action(
        console_menu, "Exécuter le code (Shift + Entrée)", window.execute_code
    )
    console_menu.addSeparator()
    state.ButtonPrint = add_menu_action(console_menu, "Insérer print(...)", window.code_print)
    state.ButtonLen = add_menu_action(console_menu, "Insérer len(...)", window.code_len)
    state.ButtonClearcode = add_menu_action(console_menu, "Vider la console", window.code_clear)
    prompt_layout.addWidget(console_button)

    state.output_display = QTextEdit(window)
    state.output_display.setReadOnly(True)
    state.output_display.setPlaceholderText("Output print...")
    prompt_layout.addWidget(state.output_display)

    state.widget_python.setLayout(prompt_layout)
    state.CommandeLayout.addWidget(state.widget_python)

    state.help_widget.setLayout(state.CommandeLayout)
    state.help_widget.setVisible(False)
    file_layout.addWidget(state.help_widget)

    
    file_box.setLayout(file_layout)
    window.grid_layout.addWidget(file_box, 4, 2, 1, 3)

    state.help_tab_index = None
    state.help_tab_visible = False







    group_fichiers = QGroupBox("File gestion")
    group_fichiers.setMaximumWidth(420)
    layout_fichiers = QVBoxLayout()
    state.DRX_selector = QComboBox(window)
    if window.RUN is not None:
        for i in range(len(window.RUN.Spectra)):
            state.DRX_selector.addItem(f"drx_{i}")
    layout_fichiers.addWidget(state.DRX_selector)
    state.DRX_selector.currentIndexChanged.connect(window._update_print_plate_from_selector)

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

    group_fichiers.setLayout(layout_fichiers)
    window.grid_layout.addWidget(group_fichiers, 3, 0, 2, 2)


def populate_ddac_menu(window) -> None:
    """Populate the top dDAC menu once the dDAC controls exist."""

    state = window.ui_state
    button = state.ddac_menu_button
    if button is None:
        return

    menu = button.menu()
    menu.clear()

    spectrum_box = getattr(window, "spectrum_select_box", None)
    if spectrum_box is not None:
        spectrum_action = add_check_menu_action(
            menu, "Clic spectrum", spectrum_box.isChecked(), spectrum_box.setChecked
        )
        spectrum_box.toggled.connect(spectrum_action.setChecked)

    menu.addSeparator()

    zone_button = getattr(window, "btn_zone_dpdt", None)
    if zone_button is not None:
        zone_action = add_check_menu_action(
            menu, "Zone dP/dt", zone_button.isChecked(), zone_button.setChecked
        )
        zone_button.toggled.connect(zone_action.setChecked)

    time_button = getattr(window, "btn_time_arg", None)
    if time_button is not None:
        time_action = add_check_menu_action(
            menu, "Plage temps", time_button.isChecked(), time_button.setChecked
        )
        time_button.toggled.connect(time_action.setChecked)


def build_tools_panel(window) -> None:
    """Create widgets controlling spectrum processing parameters."""

    state = window.ui_state
    param_box = QGroupBox("Tools")
    param_box.setMaximumWidth(260)
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
    window.grid_layout.addWidget(param_box, 0, 0, 3, 1)


def build_message_label(window) -> None:
    """Display a status message label."""

    window.ui_state.text_box_msg = QLabel("Good Luck and Have Fun")


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
    add_box = QGroupBox("Sample section")
    layout = QHBoxLayout()

    gauge_layout = QVBoxLayout()
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

    state.apply_temp_all_gauges_checkbox = QCheckBox(r"$T_{All}$")
    state.apply_temp_all_gauges_checkbox.setChecked(False)
    layh4.addWidget(state.apply_temp_all_gauges_checkbox)

    state.pt_solver_pfix_checkbox = QCheckBox("$P_{fix}$")
    state.pt_solver_pfix_checkbox.setChecked(False)
    layh4.addWidget(state.pt_solver_pfix_checkbox)
    gauge_layout.addLayout(layh4)

    state.listbox_pic = QListWidget()
    state.listbox_pic.doubleClicked.connect(window.select_pic)
    gauge_layout.addWidget(state.listbox_pic)
    layout.addLayout(gauge_layout, 2)

    control_layout = QVBoxLayout()
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
    control_layout.addWidget(parampic_box)

    view_buttons_layout = QHBoxLayout()
    state.right_view_zoom_action = QPushButton("Zoom", window)
    state.right_view_zoom_action.setCheckable(True)
    state.right_view_zoom_action.setChecked(True)
    state.right_view_zoom_action.clicked.connect(
        lambda checked: window.show_print_plate(False) if checked else window.set_zoom_print_overlay_visible(False)
    )
    view_buttons_layout.addWidget(state.right_view_zoom_action)

    state.right_view_print_action = QPushButton("Print", window)
    state.right_view_print_action.setCheckable(True)
    state.right_view_print_action.setChecked(False)
    state.right_view_print_action.clicked.connect(
        lambda checked: window.show_print_plate(True) if checked else window.set_zoom_print_overlay_visible(False)
    )
    view_buttons_layout.addWidget(state.right_view_print_action)

    state.undock_panel_button = QPushButton("Undock", window)
    state.undock_panel_button.setCheckable(True)
    state.undock_panel_button.toggled.connect(window.toggle_gauge_panel_dock)
    view_buttons_layout.addWidget(state.undock_panel_button)
    control_layout.addLayout(view_buttons_layout)
    layout.addLayout(control_layout, 1)

    add_box.setLayout(layout)

    window.ensure_print_plate_widget()

    window.AddBox = add_box
    if getattr(window.ui_state, "spectrum_section_widget", None) is not None:
        window.ui_state.spectrum_section_widget.add_right_widget(add_box)
    window.bit_modif_PTlambda = False
    window.bit_modif_PTlambda = False

    window.bit_bypass = True
    window.f_model_pic_type()
    window.bit_bypass = False
