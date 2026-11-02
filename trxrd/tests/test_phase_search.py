import numpy as np
import pytest

from conftest import LIBRARY
from synthetic import ENERGY, X, pearson_peaks, phase_pattern
from trxrd.io.jcpds import read_library
from trxrd.processing.background import snip_baseline
from trxrd.processing.peaks import detect_peaks
from trxrd.processing.phase_search import LineTables, SearchOptions, search_composition


@pytest.fixture(scope="module")
def library():
    return read_library([LIBRARY / "H2O", LIBRARY / "Sn"])  # 14 phases


@pytest.fixture(scope="module")
def tables(library):
    return LineTables(library, ENERGY, (X[0], X[-1]), SearchOptions(pressure_range=(0, 20)))


def detected(library, truth, seed=0, **kwargs):
    x, y = phase_pattern([library[n] for n in truth], list(truth.values()), seed=seed, **kwargs)
    return detect_peaks(x, y - snip_baseline(y, 1))


@pytest.mark.parametrize("truth", [
    {"H2O_Ice_VII": 2.5, "H2O_Ice_VI": 1.8},
    {"H2O_Ice_VII": 4.2},
    {"H2O_Ice_VII": 3.0, "Sn_Beta": 3.2},
])
def test_finds_phases_and_pressures(library, tables, truth):
    peaks = detected(library, truth, spurious=[(11.0, 300)], texture=0.5)
    best = search_composition(peaks, tables)[0]
    assert set(best.pressures) == set(truth)
    for name, pressure in truth.items():
        assert best.pressures[name] == pytest.approx(pressure, abs=0.1)
    spurious = [i for i, p in enumerate(peaks) if abs(p.position - 11.0) < 0.05]
    assert spurious and set(spurious) <= set(best.unexplained)


def test_deterministic_and_without_duplicates(library, tables):
    peaks = detected(library, {"H2O_Ice_VII": 2.5, "H2O_Ice_VI": 1.8})
    first, second = search_composition(peaks, tables), search_composition(peaks, tables)
    assert [s.pressures for s in first] == [s.pressures for s in second]
    keys = [tuple(sorted((n, round(p, 3)) for n, p in s.pressures.items())) for s in first]
    assert len(keys) == len(set(keys))
    assert [s.cost for s in first] == sorted(s.cost for s in first)


def test_previous_spectrum_restricts_the_pressure(library):
    options = SearchOptions(pressure_range=(0, 20), max_pressure_step=0.3)
    tables = LineTables(library, ENERGY, (X[0], X[-1]), options)
    peaks = detected(library, {"H2O_Ice_VII": 3.1})
    best = search_composition(peaks, tables, previous={"H2O_Ice_VII": 3.0})[0]
    assert 2.7 <= best.pressures["H2O_Ice_VII"] <= 3.3


def test_no_peak_no_solution(tables):
    assert search_composition([], tables) == []


def test_phase_with_a_forest_of_lines_is_not_chosen_by_chance(library, tables):
    # ice XXI has ~70 lines in the window: it "explains" anything, but only by chance
    for truth in ({"H2O_Ice_VII": 1.4, "Sn_Beta": 2.0}, {"H2O_Ice_VI": 1.0, "Sn_Beta": 1.4}):
        best = search_composition(detected(library, truth, texture=0.3), tables)[0]
        assert set(best.pressures) == set(truth)


def test_continuity_is_counted_per_family(library):
    tables = LineTables(library, ENERGY, (X[0], X[-1]), SearchOptions(pressure_range=(0, 20)))
    peaks = detected(library, {"H2O_Ice_VI": 1.0, "Sn_Beta": 1.4})
    # ice VII -> ice VI is a transformation of the same sample: cheap
    best = search_composition(peaks, tables, previous={"H2O_Ice_VII": 1.5, "Sn_Beta": 1.6})[0]
    assert set(best.pressures) == {"H2O_Ice_VI", "Sn_Beta"}


def test_family_names_ignore_case():
    from trxrd.processing.phase_search import family

    assert family("KCL_B1") == family("KCl_B2") == "kcl" and family("H2O_Ice_VII") == "h2o"


def test_marker_seen_on_two_lines_may_appear(library):
    """Real runs: Sn shows only (200) and (101) (texture); after a spectrum with ice alone, it must come back."""
    lines = []
    for name, scale, keep in (("H2O_Ice_VII", 4000, None), ("Sn_Beta", 3000, {(2, 0, 0), (1, 0, 1)})):
        for two_theta, reflection in library[name].two_theta_lines(3.0, ENERGY, max_lines=len(library[name].reflections)):
            if keep is None or reflection.hkl in keep:
                lines.append((two_theta, scale * reflection.intensity / 100, 0.035, 1.5, 0.6))
    y = pearson_peaks(X, lines)
    peaks = detect_peaks(X, y - snip_baseline(y, 1))
    found = {}
    for appear in (1.0, 0.0):  # old behaviour: appearing cost as much as vanishing
        tables = LineTables(library, ENERGY, (X[0], X[-1]), SearchOptions(pressure_range=(0, 20), cost_family_appear=appear))
        found[appear] = search_composition(peaks, tables, previous={"H2O_Ice_VII": 2.9})[0].pressures
    assert set(found[1.0]) == {"H2O_Ice_VII"}
    assert set(found[0.0]) == {"H2O_Ice_VII", "Sn_Beta"} and found[0.0]["Sn_Beta"] == pytest.approx(3.0, abs=0.1)
    # a family that vanishes still costs: the marker does not disappear for nothing
    assert SearchOptions().cost_family_change > SearchOptions().cost_family_appear


def test_broad_hump_neither_explains_nor_costs(library):
    """Water12: a liquid/amorphous hump on the ice Ic (311) line made ice Ic replace ice VII."""
    from trxrd.processing.peaks import DetectedPeak

    vii = [x for x, _ in library["H2O_Ice_VII"].two_theta_lines(9.0, ENERGY)[:2]]  # (110), (200)
    hump = [x for x, r in library["H2O_Ice_Ic"].two_theta_lines(0.05, ENERGY) if r.hkl == (3, 1, 1)][0]
    peaks = [DetectedPeak(vii[0], 900, 900, 0.25, 40.0), DetectedPeak(hump, 500, 500, 0.75, 40.0),
             DetectedPeak(vii[1], 300, 300, 0.25, 15.0)]
    best = {}
    for max_fwhm in (None, 0.5):
        tables = LineTables(library, ENERGY, (X[0], X[-1]), SearchOptions(pressure_range=(0, 20), max_fwhm=max_fwhm))
        best[max_fwhm] = search_composition(peaks, tables)[0].pressures
    assert set(best[None]) == {"H2O_Ice_Ic"}  # old behaviour
    assert set(best[0.5]) == {"H2O_Ice_VII"} and best[0.5]["H2O_Ice_VII"] == pytest.approx(9.0, abs=0.1)


def test_lines_in_excluded_zones_are_neither_seen_nor_missed(library):
    card = library["H2O_Ice_VII"]
    line_110 = card.two_theta_lines(3.0, ENERGY)[0][0]
    options = SearchOptions(pressure_range=(0, 20), exclude=[(line_110 - 0.2, line_110 + 0.2)])
    tables = LineTables({"H2O_Ice_VII": card}, ENERGY, (X[0], X[-1]), options)
    phase = tables.phases["H2O_Ice_VII"]
    row = np.argmin(np.abs(phase.pressures - 3.0))
    assert np.isnan(phase.lines[row]).sum() >= 1  # the (110) line is masked at 3 GPa
