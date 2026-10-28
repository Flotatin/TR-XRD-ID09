import pytest

from conftest import LIBRARY
from synthetic import ENERGY, X, phase_pattern
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
