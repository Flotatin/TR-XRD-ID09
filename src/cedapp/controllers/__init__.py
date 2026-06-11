"""Controller package for the DRX application."""

from cedapp.controllers.ced_controller import DdacController
from cedapp.controllers.analysis_controller import (
    ANALYSE_COLUMNS,
    AnalysisController,
    ensure_analyse_dataframe,
)
from cedapp.controllers.configuration_controller import ConfigurationMixin
from cedapp.controllers.gauge_controller import GaugeController, GaugeLibraryMixin
from cedapp.controllers.services import FileSelectionController
from cedapp.controllers.spectrum_controller import SpectrumController

__all__ = [
    "ConfigurationMixin",
    "AnalysisController",
    "ANALYSE_COLUMNS",
    "DdacController",
    "ensure_analyse_dataframe",
    "FileSelectionController",
    "GaugeController",
    "GaugeLibraryMixin",
    "SpectrumController",
    "initialize_file_selection_controller",
    "initialize_main_controllers",
]


def initialize_file_selection_controller(window) -> None:
    """Attach the file-selection controller to ``window``."""

    window.file_controller = FileSelectionController(window)


def initialize_main_controllers(window, cl_module, logger) -> None:
    """Instantiate and connect controllers bound to already-created widgets."""

    window.ddac_controller = DdacController(window)
    window.spectrum_controller = SpectrumController(
        spectrum_getter=lambda: window.Spectrum,
        run_getter=lambda: window.RUN,
        ax_spectrum=window.ax_spectrum,
        remove_button=getattr(window, "remove_btn", None),
    )
    window._connect_ddac_multi_zone_signals()
    window._connect_time_arg_zone_signals()
    window.gauge_controller = GaugeController(
        spectrum_getter=lambda: window.Spectrum,
        gauge_getter=lambda: window.gauge_select,
        gauge_setter=lambda value: setattr(window, "gauge_select", value),
        ax_spectrum=window.ax_spectrum,
        ax_dy=window.ax_dy,
        layout_dhkl=window.layout_dhkl,
        lamb0_entry=getattr(window, "lamb0_entry", None),
        name_spe_entry=getattr(window, "name_spe_entry", None),
        spinbox_p=window.spinbox_P,
        spinbox_t=window.spinbox_T,
        get_bit_modif_PTlambda=lambda: window.bit_modif_PTlambda,
        set_bit_modif_PTlambda=lambda value: setattr(window, "bit_modif_PTlambda", value),
        get_bit_load_jauge=lambda: window.bit_load_jauge,
        get_bit_modif_jauge=lambda: window.bit_modif_jauge,
        get_index_jauge=lambda: window.index_jauge,
        set_save_value=lambda value: setattr(window, "save_value", value),
        run_getter=lambda: window.RUN,
        library_getter=lambda: getattr(window.ClassDRX, "Bibli_elements", {}),
        get_apply_temperature_to_all=lambda: bool(
            getattr(window, "apply_temp_all_gauges_checkbox", None)
            and window.apply_temp_all_gauges_checkbox.isChecked()
        ),
        get_use_fixed_pressure_solver=lambda: bool(
            getattr(window, "pt_solver_pfix_checkbox", None)
            and window.pt_solver_pfix_checkbox.isChecked()
        ),
        gauge_color_getter=lambda name: window._get_gauge_color(name),
        cl_module=cl_module,
    )
    window.spinbox_P.valueChanged.connect(window.gauge_controller.handle_spinbox_pt_changed)
    window.spinbox_T.valueChanged.connect(window.gauge_controller.handle_spinbox_pt_changed)
    if getattr(window, "pt_solver_pfix_checkbox", None) is not None:
        window.pt_solver_pfix_checkbox.toggled.connect(
            window.gauge_controller.update_pt_mode_spinbox_colors
        )
        window.pt_solver_pfix_checkbox.toggled.connect(window._on_pt_solver_mode_toggled)
    window.bit_load_jauge = False
    window.bit_modif_jauge = False

    window._analysis_overlays = []
    window.analysis_ctl = None
    window.cb_piezo_consigne = None
    if getattr(window, "ax_P", None) is not None:
        window.analysis_ctl = AnalysisController(
            window,
            window.ax_P,
            dpdt_plot=getattr(window, "ax_dPdt", None),
            x_axis="time_ms",
        )
        from cedapp.ui import ui_sections

        window.cb_piezo_consigne = ui_sections.create_piezo_consigne_checkbox(window)
        ddac_widget = getattr(window.ui_state, "ddac_widget", None)
        if ddac_widget is not None:
            ddac_widget.add_control_widget(window.analysis_ctl.cb_analysis)
            ddac_widget.add_control_widget(window.cb_piezo_consigne)
        else:
            logger.debug("dDAC widget unavailable for analysis toggle placement.")
    else:
        logger.debug("Analysis controller not created: ax_P not available.")
