"""Build a Calibration from the files produced by pyFAI/Dioptas (.poni + mask)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from trxrd.domain import Calibration, wavelength_to_energy

PathLike = Union[str, Path]


def read_mask(path: PathLike) -> np.ndarray:
    """Dioptas/pyFAI mask (tif, edf, npy...) as uint8, non-zero = masked."""
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    else:
        try:
            import fabio

            data = fabio.open(str(path)).data
        except Exception:
            from PIL import Image

            data = np.array(Image.open(path))
    return (np.asarray(data) > 0).astype(np.uint8)


def poni_wavelength(poni_text: str) -> Optional[float]:
    match = re.search(r"^\s*Wavelength\s*:\s*([-+0-9.eE]+)", poni_text, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def calibration_from_files(
    poni_file: PathLike,
    mask_file: Optional[PathLike] = None,
    *,
    two_theta_range: Tuple[float, float] = (0.0, 40.0),
    points_per_degree: int = 50,
    energy_ev: Optional[float] = None,
    flip_axis: Optional[int] = None,
) -> Calibration:
    """The energy defaults to the wavelength stored in the .poni file."""
    poni = Path(poni_file).read_text(encoding="utf-8", errors="replace")
    if energy_ev is None:
        wavelength = poni_wavelength(poni)
        if wavelength is None:
            raise ValueError(f"No wavelength in {poni_file}: give energy_ev explicitly")
        energy_ev = wavelength_to_energy(wavelength)
    return Calibration(
        energy_ev=float(energy_ev),
        poni=poni,
        mask=read_mask(mask_file) if mask_file else None,
        two_theta_range=(float(two_theta_range[0]), float(two_theta_range[1])),
        points_per_degree=int(points_per_degree),
        flip_axis=flip_axis,
        poni_file=str(poni_file),
        mask_file=str(mask_file or ""),
    )
