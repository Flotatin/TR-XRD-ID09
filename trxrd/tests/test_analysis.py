"""Automatic analysis as the user asks for it (services.analysis): settings, work on a copy, static peaks."""

import json

import pytest

from conftest import LIBRARY
from synthetic import COMPRESSION as TRUTH
from synthetic import GASKET
from trxrd.physics import eos
from trxrd.services import analysis
from trxrd.services.analysis import AnalysisSettings


@pytest.fixture
def run(compression_run):
    return compression_run


def test_settings_become_the_options_and_survive_json():
    settings = AnalysisSettings(phases=["H2O_Ice_VII"], pressure_range=(1.0, 12.0), temperature=400.0, max_step=0.5,
                                exclude=[(18.48, 18.68)], window=(8.0, 24.0), track=False)
    options = settings.options()
    assert options.search.pressure_range == (1.0, 12.0) and options.search.max_pressure_step == 0.5
    assert options.search.temperature == options.fit.temperature == 400.0
    assert options.exclude == [(18.48, 18.68)] and options.window == (8.0, 24.0) and not options.track
    default = AnalysisSettings().options()
    assert default.fit.temperature == eos.T_REF and default.exclude is None and default.window is None
    assert AnalysisSettings.from_dict(json.loads(json.dumps(settings.to_dict()))) == settings  # as in Run.ui_state
    assert AnalysisSettings.from_dict({"pmax": 3, "track": False}) == AnalysisSettings(track=False)  # other versions
    assert AnalysisSettings.from_dict(None) == AnalysisSettings()


def test_analysis_works_on_a_copy_then_goes_into_the_run(run, tmp_path):
    from trxrd.io.h5store import load_run, save_run

    settings = AnalysisSettings(pressure_range=(0.0, 10.0))
    steps = []
    result = analysis.analyse(run, range(len(run)), settings, lambda done, total: steps.append(done) or True)
    assert steps == list(range(1, len(TRUTH) + 1)) and not result.stopped
    assert all(not s.phases and s.model is None for s in run.spectra) and run.ui_state == {}  # run untouched
    assert result.indices == list(range(len(TRUTH)))
    assert result.tracked == 0  # the strong gasket peak, explained by no phase, stops the peak tracking
    assert result.phase_counts() == {"H2O_Ice_VI": 3, "H2O_Ice_VII": 4}
    assert [round(p.position, 1) for p in result.static] == [round(GASKET, 1)]  # proposed as an excluded zone

    result.apply_to(run)
    for spectrum, truth in zip(run.spectra, TRUTH):
        assert {f.phase for f in spectrum.phases} == set(truth)
    saved = load_run(save_run(run, tmp_path / "run.trxrd.h5"))
    again = analysis.saved_settings(saved)  # the next analysis starts from these settings
    assert again.pressure_range == (0.0, 10.0) and again.temperature == pytest.approx(eos.T_REF)  # T of the fits

    excluded = analysis.analyse(run, range(len(run)), AnalysisSettings(pressure_range=(0.0, 10.0),
                                                                       exclude=[result.static[0].zone()]))
    assert excluded.static == []  # an excluded zone is not proposed again
    assert excluded.tracked >= 1 and excluded.phase_counts() == result.phase_counts()  # tracking works again


def test_analysis_stopped_keeps_what_is_done(run):
    result = analysis.analyse(run, range(len(run)), AnalysisSettings(pressure_range=(0.0, 10.0)),
                              lambda done, total: done < 2)
    assert result.stopped and result.indices == [0, 1] and set(result.spectra) == {0, 1} and result.static == []
    result.apply_to(run)
    assert run.spectra[1].phases and not run.spectra[2].phases


def test_candidates_and_library(run):
    with_sn = analysis.analyse(run, [0], AnalysisSettings(phases=["H2O_Ice_VI"], library=[str(LIBRARY / "Sn")]))
    assert "Sn_Beta" in with_sn.phases and "Sn_Beta" not in run.phases  # added to the copy only...
    with_sn.apply_to(run)
    assert "Sn_Beta" in run.phases  # ...then to the run with the analysed spectra

    ice = run.phases["H2O_Ice_VII"]
    assert sorted(analysis.add_library(run, [LIBRARY / "H2O"]))[:2] == ["H2O_Ice_II", "H2O_Ice_III"]
    assert run.phases["H2O_Ice_VII"] is ice  # the cards of the run win
    assert sorted(name for name, _ in analysis.library_phases([LIBRARY / "Sn"])) ==         ["Sn_Beta", "Sn_Gamma", "Sn_alpha", "Sn_bcc"]
    assert analysis.library_phases([]) == []

    only_vii = analysis.analyse(run, [5], AnalysisSettings(phases=["H2O_Ice_VII", "missing"], pressure_range=(0, 10)))
    assert {f.phase for f in only_vii.spectra[5].phases} == {"H2O_Ice_VII"}  # unknown names are skipped
    with pytest.raises(ValueError, match="candidate"):
        analysis.analyse(run, [5], AnalysisSettings(phases=["missing"]))


def test_saved_temperature_of_old_analyses(run):
    from trxrd.domain import PhaseFit

    run.spectra[0].phases = [PhaseFit("H2O_Ice_VI", pressure=1.0, temperature=400.0)]
    run.spectra[1].phases = [PhaseFit("H2O_Ice_VI", pressure=1.4, temperature=420.0),
                             PhaseFit("H2O_Ice_VII", pressure=2.0)]
    assert analysis.saved_settings(run).temperature == pytest.approx(410.0)  # median of the fits that have one
    run.ui_state[analysis.UI_KEY] = AnalysisSettings(temperature=350.0).to_dict()
    assert analysis.saved_settings(run).temperature == 350.0
