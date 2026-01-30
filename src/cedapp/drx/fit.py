from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FitContext:
    skip_ui_update: bool = False
    use_zone_spectrum: bool = False
    use_lmfit_prefit: bool = False
    fit_variation: float = 1.0


def select_fit_region(
    spectrum,
    gauges: Sequence[object],
    use_zone_spectrum: bool,
    x_min: float,
    x_max: float,
    x_s: Optional[float],
    x_e: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if use_zone_spectrum:
        index = np.where((spectrum.wnb >= x_min) & (spectrum.wnb <= x_max))[0]
        spectrum.indexX = index
        return (
            spectrum.wnb[index],
            spectrum.y_corr[index],
            spectrum.blfit[index],
            None,
        )

    if x_s is not None and x_e is not None and spectrum.indexX is not None:
        zone_fit = np.where((spectrum.wnb >= x_s) & (spectrum.wnb <= x_e))[0]
        spectrum.indexX = zone_fit
        for gauge in gauges:
            gauge.indexX = zone_fit
        return (
            spectrum.wnb[zone_fit],
            spectrum.y_corr[zone_fit],
            spectrum.blfit[zone_fit],
            zone_fit,
        )

    return spectrum.wnb, spectrum.y_corr, spectrum.blfit, None
