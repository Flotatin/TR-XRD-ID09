"""Integration, inter-frame and frequency of an acquisition: measured, read from the raw file, shown."""

import numpy as np
import pytest

from trxrd.domain import Oscilloscope, Run, Source, Spectrum
from trxrd.gui.presenters import display
from trxrd.gui.presenters.viewer import AcquisitionPresenter, ImmediateRunner
from trxrd.services.acquisition import acquisition_of, duration_text, frequency_text
from trxrd.services.session import Session

STEP, START, WIDTH, PERIOD = 160e-9, 1250, 66, 107  # Water05 scan 9: 10.56 µs pulses every 17.12 µs, sampled at 160 ns


def burst(pulses=16, frames=16, exposure=1e-5) -> Run:
    t = np.arange(6250) * STEP  # 1 ms of oscilloscope
    trigger = np.zeros(t.size)
    for k in range(pulses):
        trigger[START + k * PERIOD:START + k * PERIOD + WIDTH] = 2.0
    spectra = [Spectrum(two_theta=np.linspace(5, 6, 4), intensity=np.ones(4), frame=i) for i in range(frames)]
    return Run("Water05_0001_scan0009", spectra=spectra, source=Source(exposure_time=exposure),
               oscilloscope=Oscilloscope(time=t, channels={"Channel2": trigger}, trigger_channel="Channel2"))


def test_integration_inter_frame_and_frequency_of_a_burst():
    """Water05 scan 9: 10 µs of integration, one frame every 17.12 µs; the trigger pulse lasts 10.56 µs."""
    acq = acquisition_of(burst())
    assert acq.exposure == 1e-5 and not acq.exposure_from_raw
    assert acq.period == pytest.approx(PERIOD * STEP) and acq.gap == pytest.approx((PERIOD - WIDTH) * STEP)
    assert acq.inter_frame == pytest.approx(PERIOD * STEP - 1e-5)  # between two integrations, not two pulses
    assert acq.exposure + acq.inter_frame == pytest.approx(1 / acq.frequency)  # the sum gives the frequency
    assert acq.pulse == pytest.approx(WIDTH * STEP) and acq.pulses == 16 and acq.regular
    text, tip = display.acquisition_text(acq)
    assert text == "Intégration 10 µs · inter-frame 7.12 µs · 58.41 kHz"
    assert "Inter-frame : 7.12 µs, temps entre deux intégrations" in tip
    assert "Fréquence d'acquisition : 58.41 kHz, une image toutes les 17.12 µs" in tip
    assert "Impulsion de déclenchement : 10.56 µs ; entre deux impulsions : 6.56 µs" in tip and "s'arrête" not in tip
    assert "voie Channel2 de l'oscilloscope, au pas de 0.16 µs" in tip  # the precision of these times


def test_without_the_integration_the_inter_frame_is_the_gap_between_pulses():
    acq = acquisition_of(burst(exposure=np.nan))
    assert acq.inter_frame == pytest.approx((PERIOD - WIDTH) * STEP)
    text, tip = display.acquisition_text(acq)
    assert text == "Intégration ? · inter-frame 6.56 µs · 58.41 kHz" and "(l'intégration est inconnue)" in tip


def test_the_tooltip_says_when_the_oscilloscope_stops_before_the_spectra():
    """Jungfrau continuous mode: 98 triggers on the oscilloscope for 1600 frames."""
    _, tip = display.acquisition_text(acquisition_of(burst(pulses=4)))
    assert "L'oscilloscope s'arrête après 4 impulsions, pour 16 spectres" in tip


def test_noise_on_the_trigger_channel_gives_no_inter_frame():
    """D2O_01 scans 1-20, Water01 scans 7-41: ch5 holds only noise, thousands of "pulses" for 50 to 5000 frames."""
    run = burst()
    run.oscilloscope.channels["Channel2"] = np.random.default_rng(1).normal(size=6250)
    acq = acquisition_of(run)
    assert acq.period is not None and not acq.regular and not acq.triggers
    assert acq.inter_frame is None and acq.frequency is None
    text, tip = display.acquisition_text(acq)
    assert text == "Intégration 10 µs · inter-frame ? · ?"
    assert "ne porte pas de déclenchements d'images" in tip and "irrégulières, pour 16 spectres" in tip
    run.source.exposure_time = np.nan  # the integration unknown, the noise is still not an inter-frame
    assert not acquisition_of(run).triggers


def test_pulses_faster_than_the_integration_are_not_frame_triggers():
    acq = acquisition_of(burst(exposure=2e-5))  # 20 µs images every 17.12 µs: impossible
    assert acq.regular and not acq.triggers
    text, tip = display.acquisition_text(acq)
    assert text == "Intégration 20 µs · inter-frame ? · ?" and "toutes les 17.12 µs, plus vite que l'intégration" in tip


def test_exposure_read_from_the_raw_file_of_an_older_run(tuto_scan):
    run = Run("x", source=Source(raw_file=str(tuto_scan)))  # format 1.2: no exposure stored
    acq = acquisition_of(run)
    assert acq.exposure == 1e-5 and acq.exposure_from_raw and acq.inter_frame is None
    text, tip = display.acquisition_text(acq)
    assert text == "Intégration 10 µs · inter-frame ? · ?"
    assert "(lu dans le fichier brut)" in tip and "Inter-frame et fréquence : inconnus" in tip


def test_a_run_without_oscilloscope_reads_the_master_file(raw_root):
    """The .CEDX of the v1 continuous mode have no oscilloscope (Water01_scan0074): the BLISS master file still has it."""
    old_file = raw_root.parent / "Fichier_CED_DRX" / "water02_scan0003.CEDX"
    acq = acquisition_of(Run("water02_scan0003"), [old_file])
    assert acq.pulses_from_raw and acq.pulses == 3 and acq.channel == "Channel2"
    assert acq.period == pytest.approx(3e-3, abs=1e-5) and acq.exposure is None
    text, tip = display.acquisition_text(acq)
    assert text.startswith("Intégration ? · inter-frame 2.") and text.endswith(" Hz") and "(lu dans le fichier maître : le run n'en a pas)" in tip
    assert not acquisition_of(Run("water02_scan0003")).known  # raw files not found


def test_nothing_shown_when_nothing_is_known():
    acq = acquisition_of(Run("x"))
    assert not acq.known and display.acquisition_text(acq) == ("", "")
    assert display.acquisition_text(None) == ("", "")


def test_durations():
    assert [duration_text(v) for v in (1e-5, 6.56e-6, 1.013468e-3, 1.62155, None, np.nan)] == [
        "10 µs", "6.56 µs", "1.013 ms", "1.622 s", "?", "?"]
    assert [frequency_text(v) for v in (1 / 17.12e-6, 1 / 1.013468e-3, 2.5e6, None)] == [
        "58.41 kHz", "986.7 Hz", "2.5 MHz", "?"]


class Deferred:
    """Runner that keeps the tasks: they end when the test says, in the order it wants."""

    def __init__(self):
        self.tasks = []

    def run(self, task, on_done, on_error, on_progress=None):
        self.tasks.append(lambda: on_done(task(lambda done, total: True)))


class Window:
    def __init__(self):
        self.shown = []

    def set_acquisition(self, text, tooltip):
        self.shown.append(text)


def test_status_bar_follows_the_run_and_ignores_a_late_measure():
    session, window, runner = Session(), Window(), Deferred()
    AcquisitionPresenter(session, window, runner)
    session.attach(burst(), None)
    session.attach(burst(exposure=5e-6), None)
    runner.tasks[1]()
    assert window.shown[-1] == "Intégration 5 µs · inter-frame 12.12 µs · 58.41 kHz"
    runner.tasks[0]()  # the measure of the first run ends last: it is not shown
    assert window.shown[-1] == "Intégration 5 µs · inter-frame 12.12 µs · 58.41 kHz"
    session.close()
    assert window.shown[-1] == ""


def test_status_bar_at_once(sample_run):
    session, window = Session(), Window()
    AcquisitionPresenter(session, window, ImmediateRunner())
    session.attach(sample_run, None)
    assert window.shown[-1] == "Intégration 10 µs · inter-frame ? · ?"  # flat trigger channel in the sample run
