import contextlib
import io

import numpy as np
import pandas as pd

from trxrd.processing.timing import detect_trigger_times


def pulses(n=4):
    t = np.linspace(0, 1, 5000)
    y = 0.05 * t  # slow drift
    centers = np.linspace(0.1, 0.8, n)
    for c in centers:
        y = y + ((t > c - 0.01) & (t < c + 0.01))
    return t, y, centers


def test_detects_pulse_centres():
    t, y, centers = pulses()
    np.testing.assert_allclose(detect_trigger_times(t, y), centers, atol=1e-3)
    assert detect_trigger_times(t, y, max_triggers=2).size == 2


def test_same_times_as_legacy(legacy):
    t, y, _ = pulses()
    old = legacy.CED_DRX.__new__(legacy.CED_DRX)
    old.Spectra = [None] * 4
    old.data_oscillo = pd.DataFrame({"Time": t, "Channel2": y})
    old.time_index = "Channel2"
    with contextlib.redirect_stdout(io.StringIO()):
        old.Temps_Pression()
    np.testing.assert_allclose(detect_trigger_times(t, y, max_triggers=4), old.Time_spectrum)
