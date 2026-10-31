import numpy as np
import pytest

from conftest import LIBRARY
from synthetic import ENERGY, phase_pattern
from trxrd.domain import Calibration, Run, Spectrum
from trxrd.io.h5store import save_run
from trxrd.io.jcpds import read_library
from trxrd.processing.phase_search import SearchOptions
from trxrd.services.identification import AutoOptions, auto_analyse

# ice VI -> ice VII transition during a compression
TRUTH = [{"H2O_Ice_VI": 1.0}, {"H2O_Ice_VI": 1.4}, {"H2O_Ice_VI": 1.8, "H2O_Ice_VII": 2.2},
         {"H2O_Ice_VII": 2.6}, {"H2O_Ice_VII": 3.1}, {"H2O_Ice_VII": 3.7}]


@pytest.fixture
def run():
    library = read_library([LIBRARY / "H2O", LIBRARY / "Sn"])
    spectra = []
    for i, truth in enumerate(TRUTH):
        x, y = phase_pattern([library[n] for n in truth], list(truth.values()), seed=i, texture=0.4)
        spectra.append(Spectrum(x, y, time=i * 1e-3, frame=i))
    return Run("compression", spectra, Calibration(energy_ev=ENERGY, poni="Distance: 0.164\n"), library)


def test_auto_analysis_follows_a_phase_transition(run):
    steps = []
    results = auto_analyse(run, range(len(run)), options=AutoOptions(search=SearchOptions(pressure_range=(0, 10))),
                           progress=lambda k, n: steps.append(k) or True)
    assert steps == list(range(1, len(TRUTH) + 1))
    for spectrum, truth, result in zip(run.spectra, TRUTH, results):
        assert result.applied is not None and not result.applied.unexplained
        found = {f.phase: f for f in spectrum.phases}
        assert set(found) == set(truth)
        for name, pressure in truth.items():
            assert found[name].pressure == pytest.approx(pressure, abs=0.02)  # after the Pearson IV fit
            assert all(p.model == "PearsonIV" and p.hkl is not None for p in found[name].peaks)
        assert spectrum.model is not None


def test_cli_auto(tmp_path, run, capsys):
    from trxrd import XRDRun
    from trxrd.cli import main

    path = save_run(run, tmp_path / "compression.trxrd.h5")
    assert main(["auto", str(path), "--pmax", "10", "--frames", "3:6", "--phases", "H2O_Ice_VI", "H2O_Ice_VII"]) == 0
    assert "H2O_Ice_VII" in capsys.readouterr().out
    pressure = XRDRun(path).summary["P_H2O_Ice_VII"]
    assert np.isnan(pressure[0]) and pressure[5] == pytest.approx(3.7, abs=0.02)


def test_static_peaks_are_reported(run):
    from synthetic import X, pearson_peaks
    from trxrd.services.identification import find_static_peaks

    gasket = pearson_peaks(X, [(18.58, 3000, 0.04, 1.5, 0.3)], background=0.0, seed=99)
    for spectrum in run.spectra:
        spectrum.intensity = spectrum.intensity + gasket
    static = find_static_peaks(run, range(len(run)))
    assert [round(p.position, 1) for p in static] == [18.6]
    assert static[0].fraction == 1.0 and static[0].zone()[0] < 18.58 < static[0].zone()[1]


def test_excluded_strong_peak_does_not_drag_the_fit(run):
    """A strong peak of no phase (gasket) next to an ice VII line, left out of the analysis."""
    from synthetic import X, pearson_peaks

    card = run.phases["H2O_Ice_VII"]
    line = card.two_theta_lines(3.7, ENERGY)[0][0]
    run.spectra[5].intensity = run.spectra[5].intensity + pearson_peaks(X, [(line + 0.25, 20000, 0.04, 1.5, 0.3)], background=0.0)
    options = AutoOptions(search=SearchOptions(pressure_range=(0, 10)), exclude=[(line + 0.15, line + 0.35)])
    auto_analyse(run, [5], options=options)
    found = {f.phase: f.pressure for f in run.spectra[5].phases}
    assert found["H2O_Ice_VII"] == pytest.approx(3.7, abs=0.03)
