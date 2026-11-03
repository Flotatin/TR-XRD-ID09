"""Treatment of the spectra (background, filter) and the peak detection as the user checks it."""

import numpy as np
import pytest
from scipy.signal import savgol_coeffs, savgol_filter

from synthetic import GASKET, X
from trxrd.domain import Processing
from trxrd.processing import background
from trxrd.processing.peaks import DetectionOptions, detect_peaks, detection_levels
from trxrd.services import treatment
from trxrd.services.analysis import AnalysisSettings


def test_noise_gain_of_the_filters():
    x = np.arange(8, 25, 1 / 48)
    assert background.noise_gain(x, Processing(1, "svg", (9, 2))) == pytest.approx(np.sqrt(np.sum(savgol_coeffs(9, 2) ** 2)))
    assert background.noise_gain(x, Processing(1, "none", (0, 0))) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, (200, x.size))
    fft = Processing(1, "fft", (5.0, 30.0))  # cuts 5-30 /deg: the noise above 5 /deg (Nyquist 24 /deg)
    measured = np.std([background.smooth(x, row, fft) for row in noise])
    assert background.noise_gain(x, fft) == pytest.approx(measured, rel=0.05)


def test_filters_that_cannot_be_applied():
    background.check(Processing(2, "svg", (9, 2)), 800)
    for wrong, message in ((Processing(1, "svg", (8, 2)), "impair"), (Processing(1, "svg", (9, 9)), "ordre"),
                           (Processing(1, "svg", (901, 2)), "plus grande"), (Processing(1, "fft", (10, 2)), "FFT"),
                           (Processing(1, "median", (3, 0)), "inconnu"), (Processing(-1, "none", (0, 0)), "fond")):
        with pytest.raises(ValueError, match=message):
            background.check(wrong, 800)


def test_detection_on_a_given_filtered_signal():
    rng = np.random.default_rng(3)
    y = 400 * np.exp(-0.5 * ((X - 15.0) / 0.03) ** 2) + rng.normal(0, 10, X.size)
    default = detect_peaks(X, y)
    sg = savgol_filter(y, 7, 3)
    gain = float(np.sqrt(np.sum(savgol_coeffs(7, 3) ** 2)))
    same = detect_peaks(X, y, smoothed=sg, noise_gain=gain)  # what the default does inside
    assert [(p.position, p.snr) for p in same] == [(p.position, p.snr) for p in default]
    assert [round(p.position, 2) for p in default] == [15.0]
    levels = detection_levels(X, y, DetectionOptions(min_snr=4.0), sg, gain)
    np.testing.assert_allclose(levels.threshold, 4.0 * levels.smooth_noise)
    assert levels.noise.mean() == pytest.approx(10, rel=0.2) and levels.smooth_noise.mean() < levels.noise.mean()


def test_processing_case_by_case_or_propagated(compression_run):
    run = compression_run
    old = [s.baseline for s in run.spectra]
    wide = Processing(3, "svg", (15, 2))
    assert treatment.set_processing(run, [2], wide) == [2]
    assert run.spectra[2].processing == wide and run.spectra[2].processing is not wide  # its own copy
    np.testing.assert_allclose(run.spectra[2].baseline, background.snip_baseline(run.spectra[2].intensity, 3))
    assert run.spectra[1].processing == Processing() and run.spectra[1].baseline is old[1]
    filtered, raw = treatment.corrected(run.spectra[2])
    np.testing.assert_allclose(filtered, savgol_filter(run.spectra[2].intensity, 15, 2) - run.spectra[2].baseline)
    np.testing.assert_allclose(raw, run.spectra[2].intensity - run.spectra[2].baseline)

    assert treatment.set_processing(run, range(len(run)), run.spectra[2].processing) == list(range(len(run)))
    assert all(s.processing == wide for s in run.spectra)
    with pytest.raises(ValueError, match="impair"):
        treatment.set_processing(run, [0, 1], Processing(1, "svg", (10, 2)))
    assert run.spectra[0].processing == wide  # nothing changed


def test_detection_shows_what_the_analysis_detects(compression_run):
    from trxrd.services.identification import analysis_window, detect

    run = compression_run
    hump = 1500 * np.exp(-0.5 * ((X - 21.0) / 0.35) ** 2)  # FWHM 0.8 deg: liquid or amorphous
    run.spectra[5].intensity = run.spectra[5].intensity + hump
    run.spectra[5].baseline = None
    settings = AnalysisSettings(exclude=[(GASKET - 0.1, GASKET + 0.1)])
    found = treatment.detection(run, 5, settings)
    options = settings.options()
    expected = detect(run, 5, analysis_window(run, options), options.detection)
    assert [p.position for p in found.peaks] == [p.position for p in expected]  # the same detection
    positions = [round(p.position, 1) for p in found.peaks]
    assert positions[found.excluded.index(True)] == round(GASKET, 1) and sum(found.excluded) == 1
    assert positions[found.broad.index(True)] == 21.0 and sum(found.broad) == 1
    assert found.used == len(found.peaks) - 2 and found.broad_fwhm == 0.5
    assert found.x.size == found.smoothed.size == found.threshold.size == found.baseline.size
    assert np.all(found.threshold > 0)

    assert AnalysisSettings().spectrum_filter  # the maxima are searched on the filtered signal by default
    detection = AnalysisSettings(min_snr=8.0, max_fwhm=1.0, noise_window=2.0, shoulders=False).options().detection
    assert (detection.min_snr, detection.max_fwhm, detection.noise_window, detection.shoulders) == (8.0, 1.0, 2.0, False)
    assert detection.spectrum_filter
    without_hump = treatment.detection(run, 5, AnalysisSettings(max_fwhm=0.6))
    assert 21.0 not in [round(p.position, 1) for p in without_hump.peaks]  # too broad to be detected at all


def test_selection_of_several_spectra(compression_run):
    from trxrd.services.session import Event, Session

    session, heard = Session(), []
    session.subscribe(Event.SELECTION_CHANGED, lambda *_: heard.append(list(session.selection)))
    session.set_selection([1])  # no run
    assert session.selection == [] and heard == []
    session.attach(compression_run, None)
    session.set_selection([4, 2, 2, 99, -1])
    assert session.selection == [2, 4] and heard == [[2, 4]]
    session.set_selection([4, 2])
    assert heard == [[2, 4]]  # no change, no event
    session.attach(compression_run, None)
    assert session.selection == []
