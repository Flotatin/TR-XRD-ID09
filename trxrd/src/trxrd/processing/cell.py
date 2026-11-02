"""From fitted peak positions to cell parameters, volume and pressure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from trxrd.domain import PhaseCard
from trxrd.physics import eos, lattice

# Cell lengths refined for each crystal system (angles are kept from the card).
FREE_LENGTHS: Dict[str, Tuple[str, ...]] = {
    "CUBIC": ("a",),
    "TETRAGONAL": ("a", "c"),
    "HEXAGONAL": ("a", "c"),
    "RHOMBOHEDRAL": ("a",),
    "ORTHORHOMBIC": ("a", "b", "c"),
    "MONOCLINIC": ("a", "b", "c"),
    "TRICLINIC": ("a", "b", "c"),
}
MIN_SIGMA_TWO_THETA = 1e-4  # deg, floor for peaks without a usable error


@dataclass
class Reflex:
    """One measured reflection: Miller indices, 2θ and its 1σ error (deg)."""

    hkl: Tuple[int, int, int]
    two_theta: float
    sigma: float


@dataclass
class CellResult:
    a: float
    a_err: float
    b: float
    b_err: float
    c: float
    c_err: float
    volume: float
    volume_err: float
    mode: str  # "free": lengths refined independently, "isotropic": one scale factor
    n_reflexes: int


def sigma_two_theta_from_distance(two_theta_deg: float, distance_m: float, sigma_distance_m: float) -> float:
    """Error on 2θ (deg) due to the sample-detector distance error."""
    if not sigma_distance_m or not distance_m:
        return 0.0
    derivative = -0.5 * np.sin(2 * np.radians(two_theta_deg)) / distance_m
    return float(np.degrees(abs(derivative) * sigma_distance_m))


def _free_lengths(symmetry: str) -> Tuple[str, ...]:
    for key, names in FREE_LENGTHS.items():
        if key in symmetry:
            return names
    return ("a", "b", "c")


def _cell(card: PhaseCard, names: Sequence[str], values: Sequence[float], isotropic: bool) -> lattice.Cell:
    base = lattice.constrain(card.symmetry, card.cell)
    if isotropic:
        k = values[0]
        return lattice.Cell(base.a * k, base.b * k, base.c * k, base.alpha, base.beta, base.gamma)
    lengths = dict(zip(names, values))
    cell = base._replace(**lengths)
    return lattice.constrain(card.symmetry, cell)


def refine_cell(card: PhaseCard, reflexes: Sequence[Reflex], wavelength_m: float) -> CellResult:
    """Weighted least squares on d-spacings: residuals (d_obs - d_calc) / σ_d.

    With fewer reflexes than free lengths, only an isotropic scale of the
    reference cell is refined (axial ratios of the card are kept).
    """
    if not reflexes:
        raise ValueError(f"No reflection to refine {card.name}")
    names = _free_lengths(card.symmetry)
    isotropic = len(reflexes) < len(names)
    tth = np.array([r.two_theta for r in reflexes], dtype=float)
    sigma_tth = np.maximum(np.nan_to_num(np.array([r.sigma for r in reflexes], dtype=float)), MIN_SIGMA_TWO_THETA)
    d_obs = np.array([lattice.d_from_two_theta(t, wavelength_m) for t in tth])
    # d = λ / (2 sin θ)  =>  σ_d = d cot(θ) σ_θ, σ_θ = σ_2θ / 2
    sigma_d = d_obs / np.tan(np.radians(tth) / 2) * np.radians(sigma_tth) / 2

    def residuals(values):
        cell = _cell(card, names, values, isotropic)
        d_calc = np.array([lattice.d_spacing(card.symmetry, cell, r.hkl) for r in reflexes])
        return (d_obs - d_calc) / sigma_d

    start = [1.0] if isotropic else [getattr(card, n) for n in names]
    solution = least_squares(residuals, start, x_scale="jac")
    jac = solution.jac
    try:
        cov = np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        cov = np.full((len(start), len(start)), np.nan)
    dof = len(reflexes) - len(start)
    if dof > 0:
        cov *= max(1.0, 2 * solution.cost / dof)  # never smaller than the measurement errors

    cell = _cell(card, names, solution.x, isotropic)
    volume = lattice.cell_volume(card.symmetry, cell)

    def derived(values):
        c = _cell(card, names, values, isotropic)
        return np.array([c.a, c.b, c.c, lattice.cell_volume(card.symmetry, c)])

    grad = np.empty((4, len(start)))
    for j in range(len(start)):
        step = 1e-6 * max(abs(solution.x[j]), 1.0)
        shifted = solution.x.copy()
        shifted[j] += step
        grad[:, j] = (derived(shifted) - derived(solution.x)) / step
    errors = np.sqrt(np.clip(np.diag(grad @ cov @ grad.T), 0, None))
    return CellResult(
        a=cell.a, a_err=float(errors[0]), b=cell.b, b_err=float(errors[1]), c=cell.c, c_err=float(errors[2]),
        volume=volume, volume_err=float(errors[3]),
        mode="isotropic" if isotropic else "free", n_reflexes=len(reflexes),
    )


def pressure_with_error(card: PhaseCard, volume: float, volume_err: float, temperature: float = eos.T_REF):
    pressure = card.pressure_from_volume(volume, temperature)
    if not np.isfinite(volume_err) or volume_err <= 0:
        return pressure, np.nan
    step = 1e-4 * volume
    slope = (card.pressure_from_volume(volume + step, temperature) - card.pressure_from_volume(volume - step, temperature)) / (2 * step)
    return pressure, float(abs(slope) * volume_err)
