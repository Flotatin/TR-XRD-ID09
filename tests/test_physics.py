import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from conftest import LIBRARY
from trxrd.io.jcpds import read_jcpds
from trxrd.physics import eos, lattice
from trxrd.physics.lattice import Cell


def test_cubic_and_hexagonal_d_spacing():
    assert lattice.d_spacing("CUBIC", Cell(4.0, 4.0, 4.0), (1, 1, 1)) == pytest.approx(4 / np.sqrt(3))
    a, c, (h, k, l) = 4.5, 7.3, (1, 0, 2)
    expected = 1 / np.sqrt(4 / 3 * (h * h + h * k + k * k) / a**2 + l * l / c**2)
    assert lattice.d_spacing("HEXAGONAL", Cell(a, a, c), (h, k, l)) == pytest.approx(expected)


def test_monoclinic_d_spacing_textbook_formula():
    a, b, c, beta = 6.67, 6.11, 3.30, 110.37
    h, k, l = 1, 1, -1
    sb, cb = np.sin(np.radians(beta)), np.cos(np.radians(beta))
    inv_d2 = (h * h / a**2 + k * k * sb**2 / b**2 + l * l / c**2 - 2 * h * l * cb / (a * c)) / sb**2
    assert lattice.d_spacing("MONOCLINIC", Cell(a, b, c, 90, beta, 90), (h, k, l)) == pytest.approx(inv_d2**-0.5)


def test_rhombohedral_volume_ice_iv():
    card = read_jcpds(LIBRARY / "H2O" / "Ice-IV.jcpds")  # no VOLUME in the card: computed from a, alpha
    assert card.v0 == pytest.approx(375.4, abs=0.1)


def test_volume_from_pressure_inverts_eos():
    v = eos.volume_from_pressure(7.3, 41.136, 21.23, 4.0)
    assert eos.birch_murnaghan(v, 41.136, 21.23, 4.0) == pytest.approx(7.3)


def test_volume_from_pressure_takes_physical_branch_when_non_monotonic():
    # K0' = 0 (Ga-II card): P(V) peaks near 0.9 GPa, keep the root closest to V0
    v = eos.volume_from_pressure(0.5, 276.0, 4.9, 0.0)
    assert 0.85 * 276 < v < 276
    assert eos.birch_murnaghan(v, 276.0, 4.9, 0.0) == pytest.approx(0.5)


def test_eos_out_of_range_raises():
    with pytest.raises(ValueError):
        eos.volume_from_pressure(1e4, 41.136, 21.23, 4.0)


@pytest.mark.parametrize(
    "card", ["H2O/Ice-VII.jcpds", "H2O/Ice-VI.jcpds", "H2O/Ice-Ih.jcpds", "H2O/Ice-II.jcpds", "Sn/Sn_Beta_modif.jcpds", "Ga/Ga-I.jcpds"]
)
def test_two_theta_lines_match_legacy(legacy, card):
    path = LIBRARY / card
    if not path.exists():
        pytest.skip(f"{card} not in library")
    new = read_jcpds(path)
    frame = pd.read_csv(path, sep=":", header=None, engine="python")
    with contextlib.redirect_stdout(io.StringIO()):
        old = legacy.Element_Bibli(file=frame, E=19000).Eos_Pdhkl(1.0, extract=True)
    new_lines = [t for t, _ in new.two_theta_lines(1.0, 19000)]
    # the legacy code uses hc = 1239.8 eV.nm instead of 1239.84: ~3e-5 relative
    np.testing.assert_allclose(new_lines[: len(old)], [o[0] for o in old], atol=5e-3)
