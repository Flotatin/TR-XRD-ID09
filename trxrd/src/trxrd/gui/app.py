"""Entry point of the GUI: builds the window, the presenters and the actions.

    trxrd-gui [run.trxrd.h5 | old.CEDX]
    python -m trxrd.gui [file]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

from trxrd import __version__
from trxrd.gui.presenters.viewer import (
    CakePresenter, DisplaySettings, EvolutionPresenter, MainPresenter, MapPresenter, NavigatorPresenter,
    RatePresenter, RateSettings, SpectrumPresenter, TablesPresenter,
)
from trxrd.gui.tasks import ThreadRunner
from trxrd.gui.views.main_window import MainWindow, application
from trxrd.services.session import Session

FILE, NAVIGATION, VIEW, TOOLS, HELP = "&Fichier", "&Navigation", "&Affichage", "&Outils", "&Aide"


@dataclass
class Viewer:
    session: Session
    settings: DisplaySettings
    window: MainWindow
    main: MainPresenter
    presenters: List[object]


def create_viewer(session: Optional[Session] = None, runner=None) -> Viewer:
    """The whole viewer; `runner` runs the long tasks (default: a worker thread)."""
    session = session or Session()
    settings = DisplaySettings()
    window = MainWindow()
    runner = runner or ThreadRunner()
    cake = CakePresenter(session, settings, window.cake, runner)
    rate_settings = RateSettings()  # dP/dt: shared by the evolution and dP/dt plots (interval of the mean)
    presenters = [
        SpectrumPresenter(session, settings, window.spectrum),
        cake,
        EvolutionPresenter(session, window.evolution, rate_settings),
        RatePresenter(session, window.dpdt, rate_settings),
        MapPresenter(session, settings, window.map),
        TablesPresenter(session, window.phases, window.peaks),
        NavigatorPresenter(session, window.navigator),
    ]
    main = MainPresenter(session, settings, window, runner, version=__version__)
    window.on_close_request = main.may_discard
    window.on_unit = main.set_unit

    add = window.actions.add
    add("new_run", FILE, "&Nouveau run depuis RAW_DATA…", main.new_run, ["Ctrl+N"],
        "Intégrer des scans bruts (images Jungfrau + oscilloscope du fichier maître) en nouveaux runs")
    add("open", FILE, "&Ouvrir…", main.open, ["Ctrl+O"], "Ouvrir un run .trxrd.h5 ou un ancien fichier .CEDX")
    add("save", FILE, "&Enregistrer", main.save, ["Ctrl+S", "F3"], "Enregistrer le run (.trxrd.h5)")
    add("save_as", FILE, "Enregistrer &sous…", main.save_as, ["Ctrl+Shift+S"])
    add("close", FILE, "&Fermer le run", main.close, ["Ctrl+W"])
    add("quit", FILE, "&Quitter", window.close, ["Ctrl+Q"])
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

    add("cake", VIEW, "&Intégration sous le spectre (cake)", toggle_cake, ["Ctrl+K"], checkable=True,
        tip="Cake (2θ × azimut) ou image brute du spectre courant, calculé depuis les images de RAW_DATA")
    window.actions.get("cake").setChecked(True)
    add("calibration", TOOLS, "&Calibration du détecteur…", window.show_calibration, ["Ctrl+L"],
        "Calibrer la géométrie sur une image de LaB6 / CeO2 et enregistrer le .poni")
    add("shortcuts", HELP, "&Raccourcis clavier", window.show_shortcuts, ["F1"])
    window.build_menus(toolbar=("new_run", "open", "save", "|", "previous", "next"))
    main.update()  # the unit actions exist now
    return Viewer(session, settings, window, main, presenters)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="trxrd-gui", description=f"TR-XRD v2 ({__version__}): viewer")
    parser.add_argument("run", nargs="?", help=".trxrd.h5 run file or old .CEDX file")
    args, qt_args = parser.parse_known_args(argv)
    app = application([sys.argv[0], *qt_args])
    viewer = create_viewer()
    viewer.window.show()
    if args.run:
        viewer.main.open(args.run)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
