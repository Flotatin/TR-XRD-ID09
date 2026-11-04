"""Builds the window, the presenters and the actions, and runs them (started from app.py).

    trxrd-gui [run.trxrd.h5 | old.CEDX]
    python -m trxrd.gui [file]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

from trxrd import __version__
from trxrd.gui.presenters.analysis import AnalysisPresenter
from trxrd.gui.presenters.editing import EditingPresenter, PeaksPresenter
from trxrd.gui.presenters.live import LivePresenter
from trxrd.gui.presenters.treatment import DetectionPresenter, ProcessingPresenter
from trxrd.gui.presenters.viewer import (
    CakePresenter, DisplaySettings, EvolutionPresenter, MainPresenter, MapPresenter, NavigatorPresenter,
    RatePresenter, RateSettings, SpectraListPresenter, SpectrumPresenter, TablesPresenter,
)
from trxrd.gui.presenters.zones import SpectraZonePresenter, ZonesPresenter
from trxrd.gui.tasks import ThreadRunner
from trxrd.gui.views.main_window import MainWindow, application
from trxrd.services.session import Session

FILE, EDIT, NAVIGATION, VIEW, ANALYSIS, TOOLS, HELP = ("&Fichier", "&Édition", "&Navigation", "&Affichage", "A&nalyse",
                                                       "&Outils", "&Aide")


@dataclass
class Viewer:
    session: Session
    settings: DisplaySettings
    window: MainWindow
    main: MainPresenter
    analysis: AnalysisPresenter
    editing: EditingPresenter
    peaks: PeaksPresenter
    zones: ZonesPresenter
    live: LivePresenter
    presenters: List[object]


def create_viewer(session: Optional[Session] = None, runner=None) -> Viewer:
    """The whole viewer; `runner` runs the long tasks (default: a worker thread)."""
    session = session or Session()
    settings = DisplaySettings()
    window = MainWindow()
    runner = runner or ThreadRunner()
    cake = CakePresenter(session, settings, window.cake, runner, window.spectrum)
    rate_settings = RateSettings()  # shared by the dDAC plots and the list of the spectra (zone of spectra)
    presenters = [
        SpectrumPresenter(session, settings, window.spectrum),
        cake,
        EvolutionPresenter(session, window.evolution, rate_settings),
        RatePresenter(session, window.dpdt, rate_settings),
        MapPresenter(session, settings, window.map, rate_settings),
        TablesPresenter(session, window.phases, window.peaks),
        NavigatorPresenter(session, window.navigator),
        SpectraListPresenter(session, window.spectra, rate_settings),
        SpectraZonePresenter(session, rate_settings),
        ProcessingPresenter(session, window.processing),
        DetectionPresenter(session, settings, window.detection, window.spectrum),
    ]
    main = MainPresenter(session, settings, window, runner, version=__version__)
    analysis = AnalysisPresenter(session, window, runner)
    editing = EditingPresenter(session, window.phases, window, runner)
    peaks = PeaksPresenter(session, settings, window.peaks, window.spectrum, window.phases)
    zones = ZonesPresenter(session, settings, window.zones, window.spectrum)
    live = LivePresenter(session, window, runner)
    window.on_close_request = main.may_discard
    window.on_closed = lambda: live.toggle(False)
    window.on_unit = main.set_unit

    add = window.actions.add
    add("new_run", FILE, "&Nouveau run depuis RAW_DATA…", main.new_run, ["Ctrl+N"],
        "Intégrer des scans bruts (images Jungfrau + oscilloscope du fichier maître) en nouveaux runs")
    add("open", FILE, "&Ouvrir…", main.open, ["Ctrl+O"], "Ouvrir un run .trxrd.h5 ou un ancien fichier .CEDX")
    add("save", FILE, "&Enregistrer", main.save, ["Ctrl+S", "F3"], "Enregistrer le run (.trxrd.h5)")
    add("save_as", FILE, "Enregistrer &sous…", main.save_as, ["Ctrl+Shift+S"])
    add("close", FILE, "&Fermer le run", main.close, ["Ctrl+W"])
    add("quit", FILE, "&Quitter", window.close, ["Ctrl+Q"])
    add("undo", EDIT, "&Annuler", main.undo, ["Ctrl+Z"])
    add("redo", EDIT, "&Rétablir", main.redo, ["Ctrl+Y", "Ctrl+Shift+Z"])
    add("copy_model", EDIT, "&Copier le modèle du spectre", editing.copy, ["Ctrl+C", "Ctrl+Shift+C"],
        "Phases et pics libres du spectre courant, à coller sur d'autres spectres")
    add("paste_model", EDIT, "C&oller le modèle sur la sélection", editing.paste, ["Ctrl+V", "Ctrl+Shift+V"])
    add("edit_peaks", EDIT, "Modifier les &pics à la souris", lambda: window.peaks.edit.setChecked(window.actions.get("edit_peaks").isChecked()),
        ["Ctrl+E"], checkable=True, tip="Poignées sur les pics du modèle ; Ctrl + clic pour ajouter un pic")
    add("confirm_peak", EDIT, "Ajouter un pic au curseur", peaks.confirm, ["C"], "Curseur : un clic sur le spectre")
    add("replace_peak", EDIT, "Replacer le pic choisi au curseur", peaks.replace, ["R"])
    add("edit_zones", EDIT, "Modifier les &zones à la souris", lambda: window.zones.edit.setChecked(window.actions.get("edit_zones").isChecked()),
        ["Z"], checkable=True, tip="Zone d'ajustement et zones exclues ; Maj + glisser pour tracer une zone exclue")
    add("delete_peak", EDIT, "&Supprimer le pic ou la zone choisie",
        lambda: zones.remove() if window.zones.edit.isChecked() and zones.chosen is not None else peaks.remove(),
        ["Del", "U"])

    def edit_modes_changed() -> None:
        window.actions.get("edit_peaks").setChecked(window.peaks.edit.isChecked())
        window.actions.get("edit_zones").setChecked(window.zones.edit.isChecked())
        # the key stays free for the fields otherwise
        window.actions.get("delete_peak").setEnabled(window.peaks.edit.isChecked() or window.zones.edit.isChecked())

    def zones_toggled(on: bool) -> None:
        edit_modes_changed()
        if on:
            window.tabs.setCurrentWidget(window.zones)

    edit_modes_changed()
    window.peaks.edit.toggled.connect(lambda _: edit_modes_changed())
    window.zones.edit.toggled.connect(zones_toggled)
    add("previous", NAVIGATION, "Spectre &précédent", lambda: main.step(-1), ["Ctrl+Left", "PgUp"])
    add("next", NAVIGATION, "Spectre &suivant", lambda: main.step(+1), ["Ctrl+Right", "PgDown"])
    add("first", NAVIGATION, "P&remier spectre", main.first, ["Ctrl+Home"])
    add("last", NAVIGATION, "&Dernier spectre", main.last, ["Ctrl+End"])
    add("unit_2th", VIEW, "Axe en &2θ", lambda: main.set_unit("2th_deg"), ["Ctrl+1"], checkable=True)
    add("unit_q", VIEW, "Axe en &Q (Å⁻¹)", lambda: main.set_unit("q_A^-1"), ["Ctrl+2"], checkable=True)
    add("unit_d", VIEW, "Axe en &d (Å)", lambda: main.set_unit("d_A"), ["Ctrl+3"], checkable=True)

    def toggle_cake() -> None:
        shown = window.actions.get("cake").isChecked()
        window.cake.setVisible(shown)
        window.align_plots()
        cake.set_visible(shown)  # nothing is computed while it is hidden

    add("raw", VIEW, "Signal &brut (avec le fond)", lambda: settings.set_raw(window.actions.get("raw").isChecked()),
        ["Ctrl+B"], checkable=True,
        tip="Par défaut le spectre est montré filtré, fond soustrait : c'est sur ce signal qu'on cherche les pics")
    def toggle_guides() -> None:
        shown = window.actions.get("guides").isChecked()
        window.spectrum.set_guides(shown)
        window.cake.set_guides(shown)

    add("guides", VIEW, "Raies sur le spec&tre et le cake", toggle_guides, ["Ctrl+G"], checkable=True,
        tip="Traits des raies attendues à travers le spectre et le cake, à tout zoom")
    window.actions.get("guides").setChecked(True)
    add("fills", VIEW, "Pics &remplis", lambda: window.spectrum.set_fills(window.actions.get("fills").isChecked()),
        ["Ctrl+P"], checkable=True, tip="Chaque pic du modèle rempli à la couleur de sa phase")
    window.actions.get("fills").setChecked(True)
    add("residual", VIEW, "Ré&sidu", lambda: window.spectrum.set_residual(window.actions.get("residual").isChecked()),
        ["Ctrl+D"], checkable=True, tip="Signal brut moins le fond et l'ajustement, sous le spectre")
    window.actions.get("residual").setChecked(True)
    add("cake", VIEW, "&Intégration sous le spectre (cake)", toggle_cake, ["Ctrl+K"], checkable=True,
        tip="Cake (2θ × azimut) ou image brute du spectre courant, calculé depuis les images de RAW_DATA")
    window.actions.get("cake").setChecked(True)
    add("clear_theta", VIEW, "Effacer le repère 2θ", lambda: cake.set_theta(None), ["Esc"],
        "Un clic sur le spectre, le cake ou l'image pose un repère : trait, ou arc de cercle sur l'image")
    add("auto", ANALYSIS, "Analyse &automatique…", analysis.start, ["Ctrl+R"],
        "Détecter les pics, trouver les phases et leurs pressions, ajuster (Pearson IV), avec suivi des pics")
    add("fit", ANALYSIS, "A&juster le spectre", editing.fit, ["Ctrl+F", "Shift+F"],
        "Ajuster les pics du spectre courant, puis sa maille et sa pression")
    add("fit_peak", ANALYSIS, "Ajuster le pic choisi", peaks.fit_peak, ["Shift+R"],
        "Le pic choisi seul, les autres étant soustraits (onglet Pics)")
    add("fit_selection", ANALYSIS, "Ajuster la &sélection", lambda: editing.fit_series("selection", window.phases.follow.isChecked()),
        ["Ctrl+Shift+F", "Shift+T"], "Ajuster les spectres choisis, à partir du spectre courant")
    add("add_phase", ANALYSIS, "Ajouter la phase choisie", window.phases.add.click, ["Shift+A"],
        "La phase et la pression de l'onglet Phases, sur le spectre courant")
    add("remove_phase", ANALYSIS, "Retirer la phase choisie", window.phases.remove.click, ["D"])
    add("show_detection", ANALYSIS, "Voir la détection des pics", lambda: window.tabs.setCurrentWidget(window.detection), ["M"])
    def toggle_live() -> None:
        action = window.actions.get("live")
        action.setChecked(live.toggle(action.isChecked()))  # the user may have cancelled

    add("live", TOOLS, "Mode &Live…", toggle_live, checkable=True,
        tip="Pendant l'expérience : chaque nouveau scan est intégré, analysé, enregistré et ouvert")
    add("calibration", TOOLS, "&Calibration du détecteur…", window.show_calibration, ["Ctrl+L"],
        "Calibrer la géométrie sur une image de LaB6 / CeO2 et enregistrer le .poni")
    add("shortcuts", HELP, "&Raccourcis clavier", window.show_shortcuts, ["F1"])
    window.build_menus(toolbar=("new_run", "open", "save", "|", "raw", "auto", "live"))
    main.update()  # the unit actions exist now
    return Viewer(session, settings, window, main, analysis, editing, peaks, zones, live, presenters)


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="trxrd-gui", description=f"TR-XRD v2 ({__version__}): viewer")
    parser.add_argument("run", nargs="?", help=".trxrd.h5 run file or old .CEDX file")
    args, qt_args = parser.parse_known_args(argv)
    app = application([sys.argv[0], *qt_args])
    viewer = create_viewer()
    viewer.window.show()
    if args.run:
        viewer.main.open(args.run)
    return app.exec()
