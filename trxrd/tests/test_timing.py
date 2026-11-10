import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from trxrd.processing.timing import detect_trigger_times, find_pulses, frame_times, pulse_timing


def pulses(n=4, stop=0.8, end=1.0):
    t = np.linspace(0, end, int(5000 * end))
    y = 0.05 * t  # slow drift
    centers = np.linspace(0.1, stop, n)
    for c in centers:
        y = y + ((t > c - 0.01) & (t < c + 0.01))
    return t, y, centers


def test_detects_pulse_centres():
    t, y, centers = pulses()
    np.testing.assert_allclose(detect_trigger_times(t, y), centers, atol=1e-3)
    assert detect_trigger_times(t, y, max_triggers=2).size == 2


def test_pulse_width_period_and_gap():
    """Water05 scan 9: pulses of 10.56 µs every 17.12 µs; the v1 called the 6.56 µs between them the inter-frame time."""
    t, y, centers = pulses()
    rhythm = pulse_timing(find_pulses(t, y))
    step = t[1] - t[0]
    assert rhythm.count == 4 and rhythm.regular
    assert abs(rhythm.width - 0.02) <= step and abs(rhythm.period - (centers[1] - centers[0])) <= step
    assert rhythm.gap == pytest.approx(rhythm.period - rhythm.width, abs=step)


def test_periods_rounded_by_the_sampling_are_regular():
    """Bi01: triggers 17.12 µs apart, sampled every 1 µs: periods of 17 and 18 µs, 6 % apart, but from one clock."""
    t = np.arange(400) * 1e-6
    y = np.zeros(t.size)
    for k in range(16):
        start = 20e-6 + k * 17.12e-6
        y[(t >= start) & (t < start + 10.56e-6)] = 1.0
    found = find_pulses(t, y, baseline_degree=0)
    assert set(np.round(np.diff(found.start) * 1e6)) == {17.0, 18.0}
    rhythm = pulse_timing(found)
    assert rhythm.regular and rhythm.step == pytest.approx(1e-6) and found.step == pytest.approx(1e-6)
    assert rhythm.period == pytest.approx(17.12e-6, abs=0.1e-6)  # the median gives 17 µs


def test_a_pulse_cut_by_the_trace_is_not_measured():
    t, y, _ = pulses()
    y = y + (t < 0.005)  # the trace starts in the middle of a pulse, 5 ms long
    found = find_pulses(t, y)
    assert found.start.size == 5 and list(found.complete) == [False, True, True, True, True]
    rhythm = pulse_timing(found)
    assert rhythm.count == 4 and rhythm.regular  # counted, it would add a period of 0.1 s among the 0.23 s ones


def test_a_flat_channel_has_no_pulse():
    t = np.linspace(0, 1, 5000)
    for flat in (np.ones(t.size), np.zeros(t.size), 2 * t):  # the baseline takes nearly all of the ramp
        assert find_pulses(t, flat).start.size == 0 and detect_trigger_times(t, flat).size == 0
    assert pulse_timing(find_pulses(t, np.ones(t.size))) is None


def found(t, y):
    """Pulses without the baseline of degree 50, which bends at the start of these short synthetic traces."""
    return find_pulses(t, y, baseline_degree=0)


def test_frames_after_the_end_of_the_trace_continue_at_the_period():
    """Jungfrau continuous mode: 100 ms of oscilloscope (98 triggers) for 1600 frames at 1.0135 ms."""
    t, y, centers = pulses(n=4, stop=0.85, end=0.9)  # the trace stops while the triggers go on
    period = centers[1] - centers[0]
    times = frame_times(found(t, y), 7, t[-1])
    np.testing.assert_allclose(times, 0.1 + period * np.arange(7), atol=1e-3)
    assert frame_times(found(t, y), 3, t[-1]).size == 3


def test_triggers_that_stopped_or_were_irregular_are_not_continued():
    t, y, _ = pulses(n=4, stop=0.5)  # the last trigger 0.5 s before the end of the trace: the acquisition ended
    assert frame_times(found(t, y), 7, t[-1]).size == 4
    t, y, _ = pulses(n=4, stop=0.85, end=0.9)
    y = y - ((t > 0.59) & (t < 0.61))  # a missing trigger: one period doubled
    assert pulse_timing(found(t, y)).regular is False
    assert frame_times(found(t, y), 7, t[-1]).size == 3


def test_same_times_as_legacy(legacy):
    t, y, _ = pulses()
    old = legacy.CED_DRX.__new__(legacy.CED_DRX)
    old.Spectra = [None] * 4
    old.data_oscillo = pd.DataFrame({"Time": t, "Channel2": y})
    old.time_index = "Channel2"
    with contextlib.redirect_stdout(io.StringIO()):
        old.Temps_Pression()
    np.testing.assert_allclose(detect_trigger_times(t, y, max_triggers=4), old.Time_spectrum)
