"""Synthetic ID09-like patterns: asymmetric Pearson IV peaks, Poisson noise."""

import numpy as np

from trxrd.physics import profiles

ENERGY = 19000.0
X = np.arange(8, 25, 1 / 48)  # 48 points/degree, as integrated from the Jungfrau 1M


def pearson_peaks(x, peaks, background=300.0, seed=0):
    """peaks: (center, height, width, m, skew) tuples."""
    rng = np.random.default_rng(seed)
    signal = sum((profiles.pearson_iv(x, *p) for p in peaks), np.zeros_like(x))
    return rng.poisson(signal + background).astype(float)


def phase_pattern(cards, pressures, seed=0, scale=4000.0, width=0.035, spurious=(), texture=0.0, x=X):
    """All lines of `cards` at `pressures`; `texture` randomises the intensities (log-normal sigma)."""
    rng = np.random.default_rng(1000 + seed)
    peaks = []
    for card, pressure in zip(cards, pressures):
        for two_theta, reflection in card.two_theta_lines(pressure, ENERGY, max_lines=len(card.reflections)):
            factor = np.exp(rng.normal(0, texture)) if texture else 1.0
            peaks.append((two_theta, scale * reflection.intensity / 100 * factor, width, 1.5, 0.6))
    peaks.extend((t, h, width, 1.5, 0.6) for t, h in spurious)
    return x, pearson_peaks(x, peaks, seed=seed)


# ice VI -> ice VII transition during a compression (GPa), and a gasket line (deg) that does not move
COMPRESSION = [{"H2O_Ice_VI": 1.0}, {"H2O_Ice_VI": 1.4}, {"H2O_Ice_VI": 1.8, "H2O_Ice_VII": 2.2},
               {"H2O_Ice_VII": 2.6}, {"H2O_Ice_VII": 3.1}, {"H2O_Ice_VII": 3.7}]
GASKET = 18.58
