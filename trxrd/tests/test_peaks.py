import numpy as np
import pytest

from synthetic import X, pearson_peaks
from trxrd.processing.background import snip_baseline
from trxrd.processing.peaks import DetectionOptions, detect_peaks, local_noise


def corrected(y):
    return y - snip_baseline(y, 1)


def test_local_noise_follows_poisson_statistics():
    y = np.random.default_rng(0).poisson(np.full(X.size, 400.0)).astype(float)
    assert np.median(local_noise(y, X[1] - X[0])) == pytest.approx(20.0, rel=0.1)


def test_weak_peak_next_to_a_very_strong_one():
    # old thresholds were relative to the strongest peak: the weak one was lost
    y = pearson_peaks(X, [(15.0, 30000, 0.04, 1.6, 0.8), (16.0, 300, 0.04, 1.5, 0.6)])
    found = [p.position for p in detect_peaks(X, corrected(y))]
    assert len(found) == 2
    np.testing.assert_allclose(found, [15.0, 16.0], atol=0.01)


def test_no_false_peak_in_pure_noise():
    false = sum(len(detect_peaks(X, corrected(pearson_peaks(X, [], seed=seed)))) for seed in range(10))
    assert false <= 1


def test_unresolved_shoulder_is_found():
    hits = 0
    for seed in range(20):
        y = pearson_peaks(X, [(14.30, 900, 0.04, 1.5, 0.3), (14.39, 350, 0.04, 1.5, 0.3)], seed=seed)
        peaks = detect_peaks(X, corrected(y))
        hits += any(abs(p.position - 14.39) < 0.04 and p.shoulder for p in peaks)
    assert hits >= 12  # 14/20 at the time of writing; the second peak has no maximum of its own


def test_width_in_degrees_and_options():
    y = pearson_peaks(X, [(10.0, 2000, 0.05, 1.5, 0.0), (20.0, 2000, 0.05, 1.5, 0.0), (24.95, 2000, 0.05, 1.5, 0.0)])
    peaks = detect_peaks(X, corrected(y))
    assert [round(p.position, 1) for p in peaks] == [10.0, 20.0]  # 24.95 is within the edge margin
    expected_fwhm = 2 * 0.05 * np.sqrt(2 ** (1 / 1.5) - 1)
    assert peaks[0].fwhm == pytest.approx(expected_fwhm, rel=0.35)  # 3 points per FWHM: coarse
    only = detect_peaks(X, corrected(y), DetectionOptions(windows=[(15, 21)]))
    assert [round(p.position, 1) for p in only] == [20.0]
    without = detect_peaks(X, corrected(y), DetectionOptions(exclude=[(19.8, 20.2)]))
    assert [round(p.position, 1) for p in without] == [10.0]
