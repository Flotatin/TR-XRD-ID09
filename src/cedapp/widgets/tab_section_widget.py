"""Widgets related to the processing tabs section."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .helpers import creat_spin_label

try:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
except ModuleNotFoundError:  # pragma: no cover - fallback when matplotlib unavailable
    cm = None
    mcolors = None


class SampleWidget(QWidget):
    """Widget used to manage sample gauge information."""

    def __init__(self, main):
        super().__init__()
        layout = QVBoxLayout()
        #name = QLabel("Gauge section - - -")
        #layout.addWidget(name)

        main.folder_bib_gauge = QPushButton("Add element", main)
        main.folder_bib_gauge.clicked.connect(main.f_select_bib_gauge)
        layout.addWidget(main.folder_bib_gauge)

        main.edit_bib_gauge = QPushButton("Edit element", main)
        main.edit_bib_gauge.clicked.connect(main.f_edit_gauge)
        layout.addWidget(main.edit_bib_gauge)

        if cm and mcolors:
            main.gauge_colors = [mcolors.to_hex(cm.get_cmap("tab10")(i)) for i in range(10)]
        else:
            main.gauge_colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]

        main.lamb0_entry = QLineEdit()
        layout.addLayout(creat_spin_label(main.lamb0_entry, "λ<sub>0</sub>:"))
        main.name_spe_entry = QLineEdit()
        main.name_spe_entry.editingFinished.connect(main.f_name_spe)
        layout.addLayout(creat_spin_label(main.name_spe_entry, "G.spe:"))

        self.setLayout(layout)
        self.setVisible(True)


class BaselineWidget(QWidget):
    """Widget containing baseline parameters."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        layout = QVBoxLayout()

        main.deg_baseline_entry = QSpinBox()
        main.deg_baseline_entry.valueChanged.connect(main.setFocus)
        main.deg_baseline_entry.setSingleStep(1)
        main.deg_baseline_entry.setRange(0, 50)
        main.deg_baseline_entry.setValue(0)
        layout.addLayout(creat_spin_label(main.deg_baseline_entry, "°bline"))

        self.setLayout(layout)
        self.setVisible(False)


class FiltreWidget(QWidget):
    """Widget for data filter parameters."""

    def __init__(self, main):
        super().__init__()
        layout = QVBoxLayout()
        #name = QLabel("filtre data section - - -")
        #layout.addWidget(name)

        main.filtre_type_selector = QComboBox(main)
        liste_type_filtre = ["svg", "fft", "No filtre"]
        main.filtre_type_selector.addItems(liste_type_filtre)
        filtre_colors = ["darkblue", "darkred", "darkgrey"]
        for ind in range(len(liste_type_filtre)):
            main.filtre_type_selector.model().item(ind).setBackground(QColor(filtre_colors[ind]))
        main.filtre_type_selector.currentIndexChanged.connect(main.f_filtre_select)
        layout.addLayout(creat_spin_label(main.filtre_type_selector, "Filtre:"))

        layh1 = QHBoxLayout()
        main.param_filtre_1_name = QLabel("")
        layh1.addWidget(main.param_filtre_1_name)
        main.param_filtre_1_entry = QLineEdit()
        main.param_filtre_1_entry.setText("10")
        layh1.addWidget(main.param_filtre_1_entry)
        layout.addLayout(layh1)

        layh2 = QHBoxLayout()
        main.param_filtre_2_name = QLabel("")
        layh2.addWidget(main.param_filtre_2_name)
        main.param_filtre_2_entry = QLineEdit()
        main.param_filtre_2_entry.setText("1")
        layh2.addWidget(main.param_filtre_2_entry)
        layout.addLayout(layh2)

        self.setLayout(layout)
        self.setVisible(False)


class FindCompoWidget(QWidget):
    """Widget for auto composition search parameters."""

    def __init__(self, main):
        super().__init__()
        layout = QVBoxLayout()
        #name = QLabel("find_compo section - - -")
        #layout.addWidget(name)

        main.NGEN_entry = QSpinBox()
        main.NGEN_entry.setSingleStep(1)
        main.NGEN_entry.setRange(0, 1000)
        main.NGEN_entry.setValue(100)
        main.NGEN_entry.valueChanged.connect(main.setFocus)
        main.NGEN_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.NGEN_entry, "NGEN"))

        main.MUTPB_entry = QDoubleSpinBox()
        main.MUTPB_entry.setSingleStep(0.01)
        main.MUTPB_entry.setRange(0, 1)
        main.MUTPB_entry.setValue(0.5)
        main.MUTPB_entry.valueChanged.connect(main.setFocus)
        main.MUTPB_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.MUTPB_entry, "MUTPB"))

        main.CXPB_entry = QDoubleSpinBox()
        main.CXPB_entry.setSingleStep(0.01)
        main.CXPB_entry.setRange(0, 1)
        main.CXPB_entry.setValue(0.5)
        main.CXPB_entry.valueChanged.connect(main.setFocus)
        main.CXPB_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.CXPB_entry, "CXPB"))

        main.POPINIT_entry = QSpinBox()
        main.POPINIT_entry.setSingleStep(1)
        main.POPINIT_entry.setRange(0, 500)
        main.POPINIT_entry.setValue(100)
        main.POPINIT_entry.valueChanged.connect(main.setFocus)
        main.POPINIT_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.POPINIT_entry, "POPINIT"))

        main.tolerance_entry = QDoubleSpinBox()
        main.tolerance_entry.setRange(0, 1)
        main.tolerance_entry.setValue(0.1)
        main.tolerance_entry.setSingleStep(0.05)
        main.tolerance_entry.valueChanged.connect(main.setFocus)
        main.tolerance_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.tolerance_entry, "%"))

        main.p_range_entry = QSpinBox()
        main.p_range_entry.setRange(1, 100)
        main.p_range_entry.setValue(15)
        main.p_range_entry.setSingleStep(1)
        main.p_range_entry.valueChanged.connect(main.setFocus)
        main.p_range_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.p_range_entry, "P rg"))

        main.nb_max_element_entry = QSpinBox()
        main.nb_max_element_entry.setRange(1, 100)
        main.nb_max_element_entry.setValue(3)
        main.nb_max_element_entry.setSingleStep(1)
        main.nb_max_element_entry.valueChanged.connect(main.setFocus)
        main.nb_max_element_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.nb_max_element_entry, "nb max"))

        self.setLayout(layout)
        self.setVisible(False)


class FindPeaksWidget(QWidget):
    """Widget containing parameters for scipy.signal.find_peaks."""

    def __init__(self, main):
        super().__init__()
        layout = QVBoxLayout()
        #name = QLabel("find_peaks section - - -")
        #layout.addWidget(name)
        
        name = QLabel("height")
        layout.addWidget(name)
        main.height_entry = QDoubleSpinBox()
        main.height_entry.setDecimals(2)
        main.height_entry.setSingleStep(0.1)
        main.height_entry.setRange(0, 100)
        main.height_entry.setValue(15)
        main.height_entry.setSuffix(" %")
        main.height_entry.valueChanged.connect(main.setFocus)
        main.height_entry.valueChanged.connect(main._update_find_peaks_exclusion_region)
        main.height_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.height_entry, "", "% Imax"))

        main.print_exclusion_checkbox = QCheckBox("exclusion line")
        main.print_exclusion_checkbox.setChecked(False)
        main.print_exclusion_checkbox.toggled.connect(
            main.toggle_find_peaks_exclusion_region
        )
        layout.addWidget(main.print_exclusion_checkbox)
        
        
        name = QLabel("distance")
        layout.addWidget(name)
        main.distance_entry = QDoubleSpinBox()
        main.distance_entry.setDecimals(1)
        main.distance_entry.setSingleStep(0.1)
        main.distance_entry.setRange(1, 10)
        main.distance_entry.setValue(1)
        main.distance_entry.valueChanged.connect(main.setFocus)
        main.distance_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.distance_entry, ""))

        name = QLabel("prominence")
        layout.addWidget(name)
        main.prominence_entry = QDoubleSpinBox()
        main.prominence_entry.setDecimals(2)
        main.prominence_entry.setSingleStep(0.1)
        main.prominence_entry.setRange(0, 100)
        main.prominence_entry.setValue(5)
        main.prominence_entry.setSuffix(" %")
        main.prominence_entry.valueChanged.connect(main.setFocus)
        main.prominence_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.prominence_entry, "", "% Imax"))

        name = QLabel("width")
        layout.addWidget(name)
        main.width_entry = QDoubleSpinBox()
        main.width_entry.setDecimals(1)
        main.width_entry.setSingleStep(0.1)
        main.width_entry.setRange(0.0, 1000)
        main.width_entry.setValue(5)
        main.width_entry.valueChanged.connect(main.setFocus)
        main.width_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.width_entry, ""))

        name = QLabel("Max nb peaks")
        layout.addWidget(name)
        main.nb_peak_entry = QSpinBox()
        main.nb_peak_entry.setRange(1, 100)
        main.nb_peak_entry.setValue(10)
        main.nb_peak_entry.setSingleStep(1)
        main.nb_peak_entry.valueChanged.connect(main.setFocus)
        main.nb_peak_entry.valueChanged.connect(main._refresh_auto_compo_settings_cache)
        layout.addLayout(creat_spin_label(main.nb_peak_entry, ""))

        self.setLayout(layout)
        self.setVisible(False)


class FitParamWidget(QWidget):
    """Widget containing parameters for peak fitting."""

    def __init__(self, main):
        super().__init__()
        layout = QVBoxLayout()

        main.spinbox_cycle = QSpinBox()
        main.spinbox_cycle.valueChanged.connect(main.setFocus)
        main.spinbox_cycle.setRange(0, 10)
        main.spinbox_cycle.setSingleStep(1)
        main.spinbox_cycle.setValue(1)
        layout.addLayout(creat_spin_label(main.spinbox_cycle, "?(Y)"))

        main.sigma_pic_fit_entry = QSpinBox()
        main.sigma_pic_fit_entry.valueChanged.connect(main.setFocus)
        main.sigma_pic_fit_entry.valueChanged.connect(
            lambda _value: main._update_fit_window() if getattr(main, "index_pic_select", None) is not None else None
        )
        main.sigma_pic_fit_entry.setRange(1, 20)
        main.sigma_pic_fit_entry.setSingleStep(1)
        main.sigma_pic_fit_entry.setValue(5)
        layout.addLayout(creat_spin_label(main.sigma_pic_fit_entry, "nσ(R)"))

        main.inter_entry = QDoubleSpinBox()
        main.inter_entry.setDecimals(1)
        main.inter_entry.setSuffix(" %")
        main.inter_entry.valueChanged.connect(main.setFocus)
        main.inter_entry.valueChanged.connect(main._refresh_fit_context_cache)
        main.inter_entry.setRange(1.0, 500.0)
        main.inter_entry.setSingleStep(1.0)
        main.inter_entry.setValue(120.0)
        layout.addLayout(creat_spin_label(main.inter_entry, "Var"))
        self.setLayout(layout)
        self.setVisible(True)

class FitSelectWidget(QWidget):
    """Widget for selecting ranges to fit spectra."""

    def __init__(self, main):
        super().__init__()
        layout = QVBoxLayout()
        
        main.add_btn = QPushButton("add")
        main.add_btn.clicked.connect(main.ajouter_zone)
        layout.addWidget(main.add_btn)

        main.remove_btn = QPushButton("dell")
        main.remove_btn.clicked.connect(main.supprimer_zone)
        main.remove_btn.setEnabled(False)
        layout.addWidget(main.remove_btn)
        
        main.multi_spec_toolbar = QToolBar()
        main.multi_spec_action = main.multi_spec_toolbar.addAction("Multi")
        main.multi_spec_action.setCheckable(True)
        main.multi_spec_action.setChecked(False)
        main.multi_spec_action.toggled.connect(main.set_ddac_multi_zone_visibility)
        main.spec_zone_action = main.multi_spec_toolbar.addAction("Spec")
        main.spec_zone_action.setCheckable(True)
        main.spec_zone_action.setChecked(False)
        main.spec_zone_action.toggled.connect(main.set_find_peaks_zones_visibility)
        layout.addWidget(main.multi_spec_toolbar)


        main.index_start_entry = QSpinBox()
        main.index_start_entry.setRange(0, 2000)
        main.index_start_entry.setValue(1)
        main.index_start_entry.valueChanged.connect(main._refresh_batch_range_cache)
        layout.addLayout(creat_spin_label(main.index_start_entry, "idx start"))

        main.index_stop_entry = QSpinBox()
        main.index_stop_entry.setRange(0, 2000)
        main.index_stop_entry.setValue(10)
        main.index_stop_entry.valueChanged.connect(main._refresh_batch_range_cache)
        layout.addLayout(creat_spin_label(main.index_stop_entry, "idx stop"))

        main.fit_select_button = QPushButton("Find Compo")
        main.fit_select_button.clicked.connect(main._CEDX_auto_compo)
        layout.addWidget(main.fit_select_button)

        main.multi_fit_button = QPushButton("Multi fit")
        main.multi_fit_button.clicked.connect(main._CEDX_multi_fit)
        layout.addWidget(main.multi_fit_button)

        main.skip_ui_update_checkbox = QCheckBox(
            "no refresh"
        )
        main.skip_ui_update_checkbox.setToolTip(
            "Exécute l'action sur la CED sans recharger ni afficher chaque spectre."
        )
        main.skip_ui_update_checkbox.toggled.connect(main._set_skip_ui_update)
        layout.addWidget(main.skip_ui_update_checkbox)

        main.clear_gauges_button = QPushButton("remove gauges")
        main.clear_gauges_button.clicked.connect(main.clear_gauges_range)
        layout.addWidget(main.clear_gauges_button)


        self.setLayout(layout)
        self.setVisible(True)

class TabSectionWidget(QWidget):
    """Widget that builds the processing tabs used in the tools panel."""

    def __init__(self, main) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        main.spectrum_toolbar = QToolBar()
        main.act_show_raw = main.spectrum_toolbar.addAction("brut")
        main.act_show_raw.setCheckable(True)
        main.act_show_raw.setChecked(False)
        main.act_show_filtered = main.spectrum_toolbar.addAction("filtré")
        main.act_show_filtered.setCheckable(True)
        main.act_show_filtered.setChecked(False)
        main.act_show_baseline = main.spectrum_toolbar.addAction("baseline")
        main.act_show_baseline.setCheckable(True)
        main.act_show_baseline.setChecked(False)
        main.act_show_raw.triggered.connect(main.update_spectrum_overlays)
        main.act_show_filtered.triggered.connect(main.update_spectrum_overlays)
        main.act_show_baseline.triggered.connect(main.update_spectrum_overlays)

        main.widget_baseline = BaselineWidget(main)
        main.widget_baseline.setVisible(True)
        main.widget_filtre = FiltreWidget(main)
        main.widget_filtre.setVisible(True)
        main.f_filtre_select()

        main.data_processing_tab = QWidget()
        data_processing_layout = QVBoxLayout()
        data_processing_layout.setContentsMargins(0, 0, 0, 0)
        data_processing_layout.addWidget(main.spectrum_toolbar)
        data_processing_layout.addWidget(main.widget_baseline)
        data_processing_layout.addWidget(main.widget_filtre)
        main.data_processing_tab.setLayout(data_processing_layout)

        main.widget_sample = SampleWidget(main)
        main.widget_find_peaks = FindPeaksWidget(main)
        main.widget_find_compo = FindCompoWidget(main)
        main.widget_fit_select = FitSelectWidget(main)

        main.processing_tabs = QTabWidget()
        main.processing_tabs.setTabPosition(QTabWidget.West)
        main.processing_tabs.addTab(main.data_processing_tab, "Data traitement")
        main.processing_tabs.addTab(main.widget_sample, "Sample")
        main.processing_tabs.addTab(main.widget_find_peaks, "Peaks")
        main.processing_tabs.addTab(main.widget_find_compo, "Compo")
        main.processing_tabs.addTab(main.widget_fit_select, "Fit sel.")
        main.processing_tabs.setCurrentWidget(main.data_processing_tab)
        layout.addWidget(main.processing_tabs)

        self.setLayout(layout)
