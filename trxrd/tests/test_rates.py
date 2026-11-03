"""dP/dt: sliding straight line, mean rate over an interval, curves per sample or per phase."""

import numpy as np
import pytest

from trxrd.processing.rates import linear_fit, local_slopes
from trxrd.services import rates


def test_linear_fit():
    t = np.arange(10.0)
    fit = linear_fit(t, 3.0 - 2.0 * t)
    assert fit.slope == pytest.approx(-2.0) and fit.intercept == pytest.approx(3.0) and fit.n == 10
    assert fit.slope_err == pytest.approx(0.0, abs=1e-12) and fit.at(5.0) == pytest.approx(-7.0)
    noise = np.random.default_rng(1).normal(0, 0.1, 10)
    noisy = linear_fit(t, 3.0 - 2.0 * t + noise)
    expected = np.sqrt(np.sum((noisy.at(t) - (3.0 - 2.0 * t + noise)) ** 2) / 8 / np.sum((t - t.mean()) ** 2))
    assert noisy.slope_err == pytest.approx(expected) and noisy.rms == pytest.approx(0.1, rel=0.5)
    assert np.isnan(linear_fit([0.0, 1.0], [1.0, 2.0]).slope_err)  # 2 points: a slope, no error
    assert np.isnan(linear_fit([1.0], [1.0]).slope) and np.isnan(linear_fit([1.0, 1.0], [1.0, 2.0]).slope)
    assert linear_fit([0.0, np.nan, 2.0, 3.0], [0.0, 5.0, 2.0, 3.0]).n == 3  # NaN pairs left out


def test_local_slopes_are_sliding_straight_lines():
    t = np.linspace(0, 1, 21)
    slope, error = local_slopes(t, t**2, 2)
    np.testing.assert_allclose(slope[2:-2], 2 * t[2:-2], atol=1e-9)  # exact for a parabola, centred window
    assert np.isfinite(slope).all()  # 3 points at least at the ends
    edge, _ = local_slopes(t, t**2, 1)
    assert np.isnan(edge[0]) and np.isnan(edge[-1]) and np.isfinite(edge[1:-1]).all()  # 2 points: no slope

    rng = np.random.default_rng(3)
    times = np.sort(rng.uniform(0, 5, 40))  # uneven times
    values = rng.normal(0, 1, 40)
    slope, error = local_slopes(times, values, 3)
    for i in (0, 7, 20, 39):
        window = slice(max(i - 3, 0), i + 4)
        direct = linear_fit(times[window], values[window])
        assert slope[i] == pytest.approx(direct.slope) and error[i] == pytest.approx(direct.slope_err)
    assert np.isnan(local_slopes([0.0, 1.0], [0.0, 1.0], 2)[0]).all()


def _entries():
    """Ice VII then ice VI (coexisting at 4 ms) and Sn, decompression at -2 GPa/ms."""
    out = []
    for i, t in enumerate(np.arange(9.0)):
        if t <= 4:
            out.append((i, t, "H2O_Ice_VII", 10.0 - 2 * t))
        if t >= 4:
            out.append((i, t, "H2O_Ice_VI", 10.0 - 2 * t + (0.2 if t == 4 else 0.0)))
        out.append((i, t, "Sn_Beta", 11.0 - 2 * t))
    return out


def test_curves_per_sample_or_phase():
    by_sample = rates.pressure_curves(_entries(), "family")
    assert [(c.name, c.phases) for c in by_sample] == [("H2O", ["H2O_Ice_VII", "H2O_Ice_VI"]), ("Sn", ["Sn_Beta"])]
    water = by_sample[0]
    assert list(water.index) == list(range(9)) and water.pressure[4] == pytest.approx(2.1)  # coexisting: averaged
    assert [c.name for c in rates.pressure_curves(_entries(), "phase")] == ["H2O_Ice_VII", "Sn_Beta", "H2O_Ice_VI"]
    with pytest.raises(ValueError):
        rates.pressure_curves(_entries(), "sample")

    curves = rates.dpdt(_entries(), half_window=2)
    np.testing.assert_allclose(curves[1].rate, -2.0)  # Sn: a straight line
    assert np.nanmedian(curves[0].rate) == pytest.approx(-2.0, abs=0.05)  # continuous across the transition


def test_mean_rate_over_an_interval():
    means = rates.mean_dpdt(_entries(), 7.0, 1.0)  # bounds in any order
    water, tin = means
    assert water.name == "H2O" and water.n == 7 and water.start == 1.0 and water.stop == 7.0
    assert water.rate == pytest.approx(-2.0, abs=0.02) and np.isfinite(water.rate_err)
    assert tin.rate == pytest.approx(-2.0) and tin.p_start == pytest.approx(9.0) and tin.delta_p == pytest.approx(-12.0)
    assert tin.delta_t == 6.0 and tin.rms == pytest.approx(0.0, abs=1e-9)
    assert rates.mean_dpdt(_entries(), 1.2, 1.8) == []  # no spectrum inside
    table = rates.rates_table(means).splitlines()
    assert len(table) == 3 and table[0].startswith("courbe\t") and table[1].split("\t")[:2] == ["H2O", "H2O_Ice_VII, H2O_Ice_VI"]


def test_rates_of_a_run(sample_run, tmp_path):
    from trxrd import XRDRun
    from trxrd.io.h5store import save_run

    entries = rates.run_entries(sample_run)
    assert entries[0] == (0, 0.0, "H2O_Ice_VII", 2.0) and len(entries) == 4  # times in ms
    curve = rates.dpdt(sample_run, 1)[0]
    assert curve.name == "H2O" and list(curve.index) == [0, 1, 2] and curve.pressure[2] == pytest.approx(2.75)
    assert curve.rate[1] == pytest.approx(0.375)  # straight line through (0, 2), (1, 3), (2, 2.75)
    mean = rates.mean_dpdt(sample_run, 0.0, 2.0, by="phase")
    assert [m.name for m in mean] == ["H2O_Ice_VII"] and mean[0].rate == pytest.approx(1.0)
    saved = XRDRun(save_run(sample_run, tmp_path / "run.trxrd.h5")).to_run()  # in a notebook
    assert rates.dpdt(saved, 1)[0].rate[1] == pytest.approx(0.375)
    for spectrum in sample_run.spectra:
        spectrum.time = np.nan
    assert rates.run_entries(sample_run) == [] and rates.dpdt(sample_run) == []


def test_slope_and_secant_differ_over_a_curved_pressure():
    t = np.round(np.arange(0.0, 5.01, 0.1), 10)
    entries = [(i, ti, "Sn_Beta", -ti**3) for i, ti in enumerate(t)]
    mean, = rates.mean_dpdt(entries, 1.0, 3.0, half_window=2)
    assert (mean.first, mean.last, mean.n) == (1.0, 3.0, 21)
    assert mean.secant == pytest.approx(-13.0, abs=0.1)  # (P(3) - P(1)) / 2, on the smoothed pressure
    assert mean.rate == pytest.approx(-12.7, abs=0.1)  # the straight line through the curve is less steep
    assert mean.rms > 0.1  # and the points stray from it
    t_lin = np.arange(10.0)
    straight, = rates.mean_dpdt([(i, ti, "Sn_Beta", 5 - 2 * ti) for i, ti in enumerate(t_lin)], 2.0, 7.0)
    assert straight.rate == pytest.approx(-2.0) and straight.secant == pytest.approx(-2.0)  # they agree when straight
