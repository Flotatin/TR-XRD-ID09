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
    # the search runs on the first spectrum and where peaks appear (transition), tracking elsewhere
    assert [r.tracked for r in results] == [False, True, False, False, True, True]
    for spectrum, truth, result in zip(run.spectra, TRUTH, results):
        assert result.applied is not None
        assert all(result.peaks[j].snr < 10 for j in result.applied.unexplained)  # only weak ones
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


def test_tracking_avoids_the_phase_search(run, monkeypatch):
    from trxrd.services import identification

    calls = []
    search = identification.search_composition
    monkeypatch.setattr(identification, "search_composition", lambda *a, **k: calls.append(1) or search(*a, **k))
    auto_analyse(run, range(len(run)), options=AutoOptions(search=SearchOptions(pressure_range=(0, 10))))
    assert len(calls) == 3
    calls.clear()
    auto_analyse(run, range(len(run)), options=AutoOptions(search=SearchOptions(pressure_range=(0, 10)), track=False))
    assert len(calls) == len(TRUTH)


def test_track_peaks_rules():
    from trxrd.processing.peaks import DetectedPeak
    from trxrd.processing.phase_search import Match, Solution
    from trxrd.services.identification import track_peaks

    def peaks(*items):
        return [DetectedPeak(x, 100.0, 100.0, 0.08, snr) for x, snr in items]

    before = peaks((13.0, 50), (14.0, 80), (16.0, 6))
    previous = Solution({"Sn_Beta": 2.0}, 1.0, [Match(0, "Sn_Beta", 0, 13.0), Match(1, "Sn_Beta", 1, 14.0)], [2])
    moved = track_peaks(before, previous, {"Sn_Beta": 2.1}, peaks((12.95, 50), (13.93, 80)))
    assert moved is not None and moved.pressures == {"Sn_Beta": 2.1}
    assert [(m.peak, m.reflection, m.two_theta) for m in moved.matches] == [(0, 0, 12.95), (1, 1, 13.93)]
    # a weak peak may vanish (16.0 above) or appear: still tracked, the newcomer is unexplained
    assert track_peaks(before, previous, {}, peaks((13.0, 50), (14.0, 80), (17.5, 6))).unexplained == [2]
    # a significant peak appears, vanishes, or jumps too far: full search needed
    assert track_peaks(before, previous, {}, peaks((13.0, 50), (14.0, 80), (17.5, 40))) is None
    assert track_peaks(before, previous, {}, peaks((13.0, 50))) is None
    assert track_peaks(before, previous, {}, peaks((13.0, 50), (14.4, 80))) is None


def test_tracking_stops_on_a_strong_unexplained_peak():
    """Water12: a Sn line (SNR 160) the carried solution never explained kept the tracking for hundreds of spectra."""
    from trxrd.processing.peaks import DetectedPeak
    from trxrd.processing.phase_search import Match, Solution
    from trxrd.services.identification import track_peaks

    before = [DetectedPeak(16.7, 900, 900, 0.25, 60.0), DetectedPeak(15.0, 900, 900, 0.25, 160.0)]
    now = [DetectedPeak(16.72, 900, 900, 0.25, 60.0), DetectedPeak(15.0, 900, 900, 0.25, 160.0)]
    previous = Solution({"H2O_Ice_VII": 9.0}, 1.0, [Match(0, "H2O_Ice_VII", 0, 16.7)], [1])
    assert track_peaks(before, previous, {}, now) is not None  # old behaviour: tracked, 15.0 never explained
    assert track_peaks(before, previous, {}, now, unexplained_snr=10.0) is None
    # a broad hump (liquid, amorphous) left unexplained does not break the tracking
    humped = [now[0], DetectedPeak(15.0, 900, 900, 0.8, 160.0)]
    assert track_peaks(before, previous, {}, humped, unexplained_snr=10.0, max_fwhm=0.5) is not None


def test_auto_analysis_searches_again_while_a_strong_peak_is_unexplained(run):
    from synthetic import X, pearson_peaks

    stray = pearson_peaks(X, [(11.0, 3000, 0.04, 1.5, 0.3)], background=0.0, seed=7)  # no candidate line there
    for spectrum in run.spectra:
        spectrum.intensity = spectrum.intensity + stray
    search = SearchOptions(pressure_range=(0, 10))
    tracked = [r.tracked for r in auto_analyse(run, range(len(run)), options=AutoOptions(search=search))]
    assert not any(tracked)
    old = AutoOptions(search=search, track_unexplained_snr=None)
    assert any(r.tracked for r in auto_analyse(run, range(len(run)), options=old))
    # an excluded peak is no unexplained peak, whichever option holds the zone
    for options in (AutoOptions(search=search, exclude=[(10.9, 11.1)]),
                    AutoOptions(search=SearchOptions(pressure_range=(0, 10), exclude=[(10.9, 11.1)]))):
        assert any(r.tracked for r in auto_analyse(run, range(len(run)), options=options))


def test_static_lines_of_the_sample_are_not_suggested():
    """At nearly constant pressure the Sn lines do not move: they are no gasket peaks."""
    from synthetic import X, pearson_peaks
    from trxrd.services.identification import find_static_peaks, screen_static_peaks

    library = read_library([LIBRARY / "H2O", LIBRARY / "Sn"])
    gasket = pearson_peaks(X, [(18.58, 3000, 0.04, 1.5, 0.3)], background=0.0, seed=99)
    spectra = []
    for i, p_ice in enumerate([2.0, 2.4, 2.8, 3.2, 3.6, 4.0]):  # ice moves, Sn stays at 2.5 GPa
        x, y = phase_pattern([library["H2O_Ice_VII"], library["Sn_Beta"]], [p_ice, 2.5], seed=i)
        spectra.append(Spectrum(x, y + gasket, time=i * 1e-3, frame=i))
    run = Run("static_sn", spectra, Calibration(energy_ev=ENERGY, poni="Distance: 0.164\n"), library)
    options = AutoOptions(search=SearchOptions(pressure_range=(0, 10), max_pressure_spread=None))
    auto_analyse(run, range(len(run)), ["H2O_Ice_VII", "Sn_Beta"], options)
    sn_lines = [x for x, r in library["Sn_Beta"].two_theta_lines(2.5, ENERGY) if r.intensity >= 20 and X[0] + 0.3 < x < X[-1] - 0.3]
    static = find_static_peaks(run, range(len(run)))
    assert sum(any(abs(p.position - x) < 0.1 for p in static) for x in sn_lines) >= 2  # Sn lines look static
    kept = screen_static_peaks(run, static, range(len(run)))
    assert [round(p.position, 1) for p in kept] == [18.6]  # only the gasket is suggested


def test_cli_temperature_goes_to_the_fit():
    from trxrd.cli import _auto_options, build_parser

    args = build_parser().parse_args(["compare", "ref.CEDX"])
    assert _auto_options(args).fit.temperature == 298.0
    options = _auto_options(args, 400.0)  # compare: temperature of the reference
    assert options.fit.temperature == options.search.temperature == 400.0
    args = build_parser().parse_args(["compare", "ref.CEDX", "--temperature", "450"])
    assert _auto_options(args, 400.0).search.temperature == 450.0
