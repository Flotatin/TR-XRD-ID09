import pytest

from conftest import LIBRARY
from trxrd.io.jcpds import name_from_comment, parse_jcpds, read_jcpds, read_library


def test_read_ice_vii():
    card = read_jcpds(LIBRARY / "H2O" / "Ice-VII.jcpds")
    assert card.name == "H2O_Ice_VII"
    assert card.symmetry == "CUBIC"
    assert (card.k0, card.k0p, card.v0) == (21.23, 4.0, 41.136)
    assert card.alpha_t == pytest.approx(1.16e-4)
    assert card.p_range == (-1.0, 70.0)
    assert card.z_max == 18 and card.fu == 2
    assert [r.hkl for r in card.reflections[:2]] == [(1, 1, 0), (2, 0, 0)]
    assert "DIHKL" in card.source_text


def test_whole_library_parses():
    cards = read_library(sorted(p for p in LIBRARY.iterdir() if p.is_dir()))
    assert len(cards) > 20
    assert all(card.v0 and card.v0 > 0 for card in cards.values())


def test_name_from_comment_keeps_legacy_naming():
    assert name_from_comment("Sn Beta /T[0,600]/P[-1,9]/Z=51") == "Sn_Beta"
    assert name_from_comment("Ga I /T[150,350]/P[-0.2,1]/Z=18 #, Cmce") == "Ga_I"


def test_incomplete_card_is_rejected():
    with pytest.raises(ValueError):
        parse_jcpds("COMMENT: nothing useful\n")
