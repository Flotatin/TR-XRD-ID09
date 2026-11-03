"""Crystal lattice geometry: cell volume, d-spacings, Bragg angles.

All symmetries go through the general (triclinic) metric tensor, after the
cell has been constrained to its crystal system. Lengths in Angstrom,
angles in degrees, wavelengths in metres.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Sequence, Tuple

import numpy as np

SYMMETRIES = ("CUBIC", "TETRAGONAL", "HEXAGONAL", "ORTHORHOMBIC", "RHOMBOHEDRAL", "MONOCLINIC", "TRICLINIC")


class Cell(NamedTuple):
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0


def constrain(symmetry: str, cell: Cell) -> Cell:
    """Enforce the relations imposed by the crystal system."""
    sym = symmetry.upper()
    a, b, c, al, be, ga = cell
    if "CUBIC" in sym:
        return Cell(a, a, a, 90.0, 90.0, 90.0)
    if "TETRAGONAL" in sym:
        return Cell(a, a, c, 90.0, 90.0, 90.0)
    if "HEXAGONAL" in sym:
        return Cell(a, a, c, 90.0, 90.0, 120.0)
    if "ORTHORHOMBIC" in sym:
        return Cell(a, b, c, 90.0, 90.0, 90.0)
    if "RHOMBOHEDRAL" in sym:
        return Cell(a, a, a, al, al, al)
    if "MONOCLINIC" in sym:
        return Cell(a, b, c, 90.0, be, 90.0)
    return cell


# Independent cell lengths of each crystal system (the others follow from `constrain`).
FREE_LENGTHS: Dict[str, Tuple[str, ...]] = {
    "CUBIC": ("a",),
    "TETRAGONAL": ("a", "c"),
    "HEXAGONAL": ("a", "c"),
    "RHOMBOHEDRAL": ("a",),
    "ORTHORHOMBIC": ("a", "b", "c"),
    "MONOCLINIC": ("a", "b", "c"),
    "TRICLINIC": ("a", "b", "c"),
}


def free_lengths(symmetry: str) -> Tuple[str, ...]:
    """Cell lengths that are independent in this crystal system: ("a",) for cubic, ("a", "c") for hexagonal..."""
    sym = symmetry.upper()
    for key, names in FREE_LENGTHS.items():
        if key in sym:
            return names
    return ("a", "b", "c")


def cell_volume(symmetry: str, cell: Cell) -> float:
    a, b, c, al, be, ga = constrain(symmetry, cell)
    ca, cb, cg = np.cos(np.radians([al, be, ga]))
    return float(a * b * c * np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg))


def scale_cell_to_volume(symmetry: str, cell: Cell, volume: float) -> Cell:
    """Isotropic scaling (axial ratios and angles kept) to reach `volume`."""
    cell = constrain(symmetry, cell)
    k = (volume / cell_volume(symmetry, cell)) ** (1.0 / 3.0)
    return Cell(cell.a * k, cell.b * k, cell.c * k, cell.alpha, cell.beta, cell.gamma)


def reciprocal_metric(symmetry: str, cell: Cell) -> np.ndarray:
    a, b, c, al, be, ga = constrain(symmetry, cell)
    ca, cb, cg = np.cos(np.radians([al, be, ga]))
    g = np.array(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ]
    )
    return np.linalg.inv(g)


def d_spacing(symmetry: str, cell: Cell, hkl: Sequence[int]) -> float:
    h = np.asarray(hkl, dtype=float)
    return float(1.0 / np.sqrt(h @ reciprocal_metric(symmetry, cell) @ h))


def two_theta_from_d(d_angstrom: float, wavelength_m: float) -> float:
    return float(np.degrees(2 * np.arcsin(wavelength_m * 1e10 / (2 * d_angstrom))))


def d_from_two_theta(two_theta_deg: float, wavelength_m: float) -> float:
    return float(wavelength_m * 1e10 / (2 * np.sin(np.radians(two_theta_deg) / 2)))


# ----------------------------------------------------------------------------- axis units

AXIS_UNITS = ("2th_deg", "q_A^-1", "d_A")
AXIS_LABELS = {"2th_deg": "2θ (°)", "q_A^-1": "Q (Å⁻¹)", "d_A": "d (Å)"}


def q_from_two_theta(two_theta_deg, wavelength_m):
    """Momentum transfer Q = 4π sin(θ) / λ in Å⁻¹."""
    return 4 * np.pi * np.sin(np.radians(np.asarray(two_theta_deg, dtype=float)) / 2) / (wavelength_m * 1e10)


def two_theta_from_q(q, wavelength_m):
    return np.degrees(2 * np.arcsin(np.asarray(q, dtype=float) * wavelength_m * 1e10 / (4 * np.pi)))


def convert_two_theta(two_theta_deg, unit: str, wavelength_m: float):
    """2θ (deg) -> `unit` among AXIS_UNITS (vectorised)."""
    if unit == "2th_deg":
        return np.asarray(two_theta_deg, dtype=float)
    q = q_from_two_theta(two_theta_deg, wavelength_m)
    if unit == "q_A^-1":
        return q
    if unit == "d_A":
        with np.errstate(divide="ignore"):
            return 2 * np.pi / q
    raise ValueError(f"Unknown axis unit {unit!r}, expected one of {AXIS_UNITS}")
