from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AutoCompoSettings:
    height: float
    distance: float
    prominence: float
    width: float
    number_peak_max: int
    ngen: int
    mutpb: float
    cxpb: float
    popinit: int
    tolerance: float
    p_range: float
    nb_max_element: int


@dataclass(frozen=True)
class BatchRange:
    start: int
    stop: int
    indices: List[int]

    @property
    def total_steps(self) -> int:
        return len(self.indices)


def build_batch_range(
    index_start: int,
    index_stop: int,
    total_spectra: int,
) -> Optional[BatchRange]:
    if total_spectra <= 0:
        return None
    start = max(0, int(index_start))
    stop = min(total_spectra - 1, int(index_stop))
    if start > stop:
        return None
    indices = list(range(start, stop + 1))
    return BatchRange(start=start, stop=stop, indices=indices)


def resolve_theta2_range(
    theta2_range: Optional[Sequence[Sequence[float]]],
    spectrum: Optional[object],
) -> List[Tuple[float, float]]:
    if theta2_range:
        return [tuple(map(float, values)) for values in theta2_range]
    if spectrum is not None and getattr(spectrum, "wnb", None) is not None:
        wnb = spectrum.wnb
        if len(wnb) > 0:
            return [(float(wnb[0]), float(wnb[-1]))]
    return [(0.0, 90.0)]


def mask_spectrum_values(
    x_values: Sequence[float],
    y_values: Sequence[float],
    theta2_range: Sequence[Sequence[float]],
) -> np.ndarray:
    y_mask: List[float] = []
    last_valid = None
    for yi, xi in zip(y_values, x_values):
        if any(a <= xi <= b for a, b in theta2_range):
            last_valid = float(yi)
            y_mask.append(last_valid)
        else:
            y_mask.append(last_valid if last_valid is not None else 0.0)
    return np.asarray(y_mask, dtype=float)
