"""Peak profiles, all parametrised the same way for curve_fit:

    f(x, center, height, width, *shape)

`center` is the position of the maximum and `height` its value, whatever
the model. `width` is the model's natural scale:

    Gaussian         standard deviation                     shape: ()
    PseudoVoigt      half width at half maximum (HWHM)      shape: (fraction,)   0 = Gauss, 1 = Lorentz
    PearsonIV        scale a                                shape: (m, skew)     m > 1/2
    Moffat           scale                                  shape: (beta,)
    SplitLorentzian  left HWHM                              shape: (right HWHM,)

Those are the conventions of the legacy lmfit models, so converted peaks
are evaluated with the same curves.
"""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Sequence, Tuple

import numpy as np

LN2 = np.log(2.0)


def gaussian(x, center, height, width):
    return height * np.exp(-((x - center) ** 2) / (2 * width**2))


def pseudo_voigt(x, center, height, width, fraction):
    u2 = ((x - center) / width) ** 2
    return height * ((1 - fraction) * np.exp(-LN2 * u2) + fraction / (1 + u2))


def pearson_iv(x, center, height, width, m, skew):
    """Pearson IV shifted so that its mode sits at `center`."""
    lam = center + width * skew / (2 * m)  # location parameter of the usual form
    u = (x - lam) / width
    mode_value = (1 + (skew / (2 * m)) ** 2) ** -m * np.exp(-skew * np.arctan(-skew / (2 * m)))
    return height / mode_value * (1 + u**2) ** -m * np.exp(-skew * np.arctan(u))


def moffat(x, center, height, width, beta):
    return height * (1 + ((x - center) / width) ** 2) ** -beta


def split_lorentzian(x, center, height, width, width_right):
    hw = np.where(x < center, width, width_right)
    return height * hw**2 / ((x - center) ** 2 + hw**2)


class ProfileModel(NamedTuple):
    function: Callable
    shape_names: Tuple[str, ...]
    shape_default: Tuple[float, ...]
    shape_bounds: Tuple[Tuple[float, float], ...]


MODELS: Dict[str, ProfileModel] = {
    "Gaussian": ProfileModel(gaussian, (), (), ()),
    "PseudoVoigt": ProfileModel(pseudo_voigt, ("fraction",), (0.5,), ((0.0, 1.0),)),
    "PearsonIV": ProfileModel(pearson_iv, ("m", "skew"), (1.1, 0.0), ((0.501, 20.0), (-5.0, 5.0))),
    "Moffat": ProfileModel(moffat, ("beta",), (1.0,), ((0.1, 20.0),)),
    "SplitLorentzian": ProfileModel(split_lorentzian, ("width_right",), (0.05,), ((1e-5, 10.0),)),
}


def model(name: str) -> ProfileModel:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown peak model {name!r}, expected one of {sorted(MODELS)}") from None


def evaluate(name: str, x, center: float, height: float, width: float, shape: Sequence[float] = ()) -> np.ndarray:
    profile = model(name)
    shape = tuple(shape) or profile.shape_default
    return profile.function(np.asarray(x, dtype=float), center, height, width, *shape)


def fwhm(name: str, width: float, shape: Sequence[float] = ()) -> float:
    """Full width at half maximum, computed numerically (valid for every model)."""
    x = np.linspace(-50 * width, 50 * width, 200001)
    y = evaluate(name, x, 0.0, 1.0, width, shape)
    above = x[y >= 0.5]
    return float(above[-1] - above[0])
