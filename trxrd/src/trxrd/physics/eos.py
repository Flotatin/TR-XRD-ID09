"""Equations of state (pressures in GPa, volumes in Angstrom^3)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import brentq

T_REF = 298.0  # K, reference temperature of the phase cards


def birch_murnaghan(volume, v0: float, k0: float, k0p: float):
    """Third-order Birch-Murnaghan P(V)."""
    x = (v0 / np.asarray(volume, dtype=float)) ** (1.0 / 3.0)
    return 1.5 * k0 * (x**7 - x**5) * (1 + 0.75 * (k0p - 4) * (x**2 - 1))


def v0_at_temperature(v0: float, alpha_t: Optional[float], temperature: float) -> float:
    """V0(T) with a constant volumetric thermal expansion coefficient."""
    if alpha_t is None:
        return float(v0)
    return float(v0) * float(np.exp(alpha_t * (float(temperature) - T_REF)))


def volume_from_pressure(
    pressure: float, v0: float, k0: float, k0p: float, v_min_ratio: float = 0.6, n_grid: int = 2000
) -> float:
    """Invert P(V) on the physical branch, i.e. the root closest to V0.

    Some cards (e.g. K0'=0) give a non-monotonic P(V) at strong compression,
    so the root is bracketed on a grid scanned from V0 downwards.
    """
    volumes = np.linspace(v0 * 1.05, v0 * v_min_ratio, n_grid)
    residual = birch_murnaghan(volumes, v0, k0, k0p) - float(pressure)
    crossings = np.flatnonzero(np.sign(residual[:-1]) != np.sign(residual[1:]))
    if residual[0] == 0:
        return float(volumes[0])
    if crossings.size == 0:
        raise ValueError(
            f"P={pressure} GPa outside the EoS range [{v_min_ratio}*V0, 1.05*V0] "
            f"(V0={v0}, K0={k0}, K0'={k0p})"
        )
    i = int(crossings[0])
    return float(brentq(lambda v: birch_murnaghan(v, v0, k0, k0p) - pressure, volumes[i + 1], volumes[i]))
