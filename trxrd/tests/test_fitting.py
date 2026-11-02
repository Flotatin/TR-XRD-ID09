import numpy as np
import pytest

from conftest import LIBRARY
from trxrd.domain import Calibration, Peak, Processing, Run, Spectrum
from trxrd.io.jcpds import read_jcpds
from trxrd.physics import profiles
from trxrd.processing.cell import Reflex, refine_cell
from trxrd.processing.peakfit import fit_peaks
from trxrd.services.fitting import FitOptions, fit_spectra, fit_spectrum

ENERGY = 19000.0


def synthetic_spectrum(cards, pressures, rng, x=np.linspace(8, 30, 1101), width=0.04):
    """Pattern with every reflection of `cards` at `pressures`, flat background and noise."""
    y = np.full_like(x, 50.0)
    for card, pressure in zip(cards, pressures):
        for two_theta, reflection in card.two_theta_lines(pressure, ENERGY, max_lines=len(card.reflections)):
            y += profiles.pseudo_voigt(x, two_theta, 10 * reflection.intensity, width, 0.4)
    return x, y + rng.normal(0, 1.0, x.size)


def make_run(spectra, cards):
    return Run(
        name="synthetic",
        spectra=[Spectrum(x, y, processing=Processing(baseline_degree=1, filter_type="none")) for x, y in spectra],
        calibration=Calibration(energy_ev=ENERGY, poni="Distance: 0.164\n"),
        phases={card.name: card for card in cards},
    )


@pytest.mark.parametrize("model", sorted(profiles.MODELS))
def test_profiles_peak_at_center_with_height(model):
    x = np.linspace(-1, 1, 4001)
    y = profiles.evaluate(model, x, 0.1, 3.0, 0.05)
    assert x[np.argmax(y)] == pytest.approx(0.1, abs=1e-3)
    assert y.max() == pytest.approx(3.0, rel=1e-3)


def test_fit_peaks_recovers_parameters():
    rng = np.random.default_rng(1)
    x = np.linspace(10, 12, 400)
    y = profiles.pearson_iv(x, 10.8, 100, 0.05, 1.5, 0.8) + profiles.pseudo_voigt(x, 11.1, 40, 0.04, 0.3)
    y += rng.normal(0, 0.5, x.size)
    start = [Peak("a", "PearsonIV", center=10.75, width=0.03), Peak("b", "PseudoVoigt", center=11.15, width=0.03)]
    result = fit_peaks(x, y, start)
    assert result.success
    a, b = result.peaks
    assert a.center == pytest.approx(10.8, abs=3 * a.center_err + 1e-4)
    assert b.center == pytest.approx(11.1, abs=3 * b.center_err + 1e-4)
    assert a.height == pytest.approx(100, rel=0.02) and b.height == pytest.approx(40, rel=0.05)
    assert 0 < a.center_err < 0.01
    assert result.reduced_chi2 == pytest.approx(0.25, rel=0.3)  # noise variance


def test_refine_cell_tetragonal_and_isotropic_fallback():
    card = read_jcpds(LIBRARY / "Sn" / "Sn_Beta_modif.jcpds")
    wavelength = 6.525e-11
    lines = card.two_theta_lines(3.0, ENERGY)[:6]
    reflexes = [Reflex(ref.hkl, tth, 1e-3) for tth, ref in lines]
    full = refine_cell(card, reflexes, 1.239841984e-6 / ENERGY)
    assert full.mode == "free"
    assert full.volume == pytest.approx(card.volume_from_pressure(3.0), rel=1e-6)
    assert 0 < full.volume_err < 0.1
    single = refine_cell(card, reflexes[:1], wavelength)
    assert single.mode == "isotropic" and single.n_reflexes == 1
    assert single.c / single.a == pytest.approx(card.c / card.a)


def test_fit_spectrum_finds_pressure_of_two_phases():
    rng = np.random.default_rng(2)
    ice7 = read_jcpds(LIBRARY / "H2O" / "Ice-VII.jcpds")
    ice6 = read_jcpds(LIBRARY / "H2O" / "Ice-VI.jcpds")
    run = make_run([synthetic_spectrum([ice7, ice6], [2.5, 1.8], rng)], [ice7, ice6])
    fit_spectrum(run, 0, {"H2O_Ice_VII": 2.2, "H2O_Ice_VI": 1.6}, options=FitOptions(model="PseudoVoigt", min_intensity=10))
    fits = {f.phase: f for f in run.spectra[0].phases}
    assert fits["H2O_Ice_VII"].pressure == pytest.approx(2.5, abs=0.05)
    assert fits["H2O_Ice_VI"].pressure == pytest.approx(1.8, abs=0.05)
    assert 0 < fits["H2O_Ice_VII"].pressure_err < 0.05
    assert run.spectra[0].model is not None and np.isfinite(run.spectra[0].model).all()


def test_fit_spectra_follows_a_compression():
    rng = np.random.default_rng(3)
    ice7 = read_jcpds(LIBRARY / "H2O" / "Ice-VII.jcpds")
    pressures = [3.0, 3.4, 3.8, 4.2, 4.6]
    run = make_run([synthetic_spectrum([ice7], [p], rng) for p in pressures], [ice7])
    steps = []
    done = fit_spectra(run, range(5), {"H2O_Ice_VII": 3.0}, options=FitOptions(model="PseudoVoigt"),
                       progress=lambda k, n: steps.append(k) or True)
    assert done == [0, 1, 2, 3, 4] and steps == [1, 2, 3, 4, 5]
    found = [s.phase_fit("H2O_Ice_VII").pressure for s in run.spectra]
    np.testing.assert_allclose(found, pressures, atol=0.05)


def test_two_stage_fit_frees_the_shape_only_if_it_helps():
    rng = np.random.default_rng(4)
    x = np.linspace(10, 11, 200)
    y = profiles.pearson_iv(x, 10.5, 200, 0.03, 1.4, 1.2) + rng.normal(0, 1.0, x.size)
    start = [Peak("a", "PearsonIV", center=10.48, width=0.02)]
    fixed = fit_peaks(x, y, start, free_shape=False)
    free = fit_peaks(x, y, start)
    assert fixed.peaks[0].shape == profiles.MODELS["PearsonIV"].shape_default
    assert free.reduced_chi2 < fixed.reduced_chi2
    assert free.peaks[0].center == pytest.approx(10.5, abs=0.003)
