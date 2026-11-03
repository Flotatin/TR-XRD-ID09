"""Detector calibration on a synthetic LaB6 image (Jungfrau 1M, beam centre off the detector as at ID09)."""

import dataclasses

import h5py
import numpy as np
import pytest

pytest.importorskip("pyFAI")

from trxrd.processing.geometry import ControlPoints, Geometry, angle_maps, calibrant_rings, ring_overlay  # noqa: E402
from trxrd.services.calibration import (  # noqa: E402
    CalibrationState, calibration_from_geometry, load_calibration_image, save_poni,
)

# the ID09 geometry of January 2026 (ddac2): direct beam at x = 1210 px, outside the 1030 px wide detector
TRUE = Geometry(dist=0.165214, poni1=0.0385888, poni2=0.0931875, rot1=0.0145045, rot2=-0.0060769, rot3=0.0,
                wavelength=6.525e-11)


def make_lab6():
    """Rings of LaB6 (Gaussian, 0.03°), spotty like a coarse powder, Poisson noise, a masked module gap."""
    rng = np.random.default_rng(3)
    two_theta, chi = angle_maps(TRUE, (1064, 1030))
    image = np.full(two_theta.shape, 200.0)
    for k, ring in enumerate(calibrant_rings("LaB6", TRUE.wavelength, 30.0)):
        grains = 1 + 0.8 * np.sin(np.radians(chi) * (7 + k) + k)  # intensity along the ring
        image += 3000 / (1 + k) * grains * np.exp(-0.5 * ((two_theta - ring) / 0.03) ** 2)
    mask = np.zeros(two_theta.shape, np.uint8)
    mask[512:556] = 1  # gap between the modules
    image = rng.poisson(image).astype(float)
    image[mask > 0] = -1e5  # what the Jungfrau writes there
    return image, mask, two_theta


@pytest.fixture(scope="module")
def lab6():
    return make_lab6()


def _click(two_theta, mask, ring_index):
    ring = calibrant_rings("LaB6", TRUE.wavelength)[ring_index]
    rows, cols = np.nonzero((np.abs(two_theta - ring) < 0.01) & (mask == 0))
    return rows[len(rows) // 2] + 2, cols[len(cols) // 2] + 3  # a few pixels off: users do not click exactly


@pytest.mark.parametrize("start", [
    dataclasses.replace(TRUE, dist=TRUE.dist * 1.05, poni1=TRUE.poni1 + 10 * 75e-6, poni2=TRUE.poni2 + 10 * 75e-6,
                        rot1=0.0, rot2=0.0),
    Geometry.from_guess(0.160, 1200, 500, TRUE.wavelength),  # distance and beam centre read on the setup
], ids=["perturbed poni", "guess"])
def test_click_two_rings_then_calibrate(lab6, start):
    image, mask, two_theta = lab6
    state = CalibrationState(image, start, "LaB6", mask)
    for ring in (0, 2):
        assert state.pick(*_click(two_theta, mask, ring), ring) >= 10
    result = state.calibrate()
    assert result.rms_deg < 0.01 and result.n_rings >= 5
    found = angle_maps(result.geometry, image.shape)[0]
    band = (mask == 0) & (two_theta > 8.5) & (two_theta < 26)
    assert np.abs(found - two_theta)[band].max() < 0.01  # deg, everywhere the rings are seen
    assert result.geometry.dist == pytest.approx(TRUE.dist, rel=5e-3)


def test_refine_from_the_start_geometry_alone(lab6):
    image, mask, two_theta = lab6
    start = dataclasses.replace(TRUE, dist=TRUE.dist * 1.01, poni2=TRUE.poni2 - 3 * 75e-6)
    result = CalibrationState(image, start, "LaB6", mask).calibrate()
    assert result.rms_deg < 0.01
    with pytest.raises(ValueError):  # no ring at all on the image
        CalibrationState(np.random.default_rng(0).poisson(200.0, image.shape).astype(float), start, "LaB6").calibrate()


def test_poni_round_trip_and_calibration(tmp_path, lab6):
    import pyFAI
    from trxrd.processing.integration import azimuthal_integrator

    text = TRUE.to_poni_text()
    assert "Jungfrau" in text and "Distance: 0.165214" in text
    back = Geometry.from_poni_text(text)
    assert back.dist == pytest.approx(TRUE.dist) and back.rot1 == pytest.approx(TRUE.rot1)
    path = save_poni(TRUE, tmp_path / "calib.poni")
    reference = pyFAI.load(str(path)).center_array((1064, 1030), unit="2th_deg")
    np.testing.assert_allclose(reference, lab6[2], atol=1e-9)
    calibration = calibration_from_geometry(TRUE, lab6[1], "det.mask", (8.0, 25.0), str(path))
    assert calibration.energy_ev == pytest.approx(19001.4, abs=0.5) and calibration.mask.sum() == lab6[1].sum()
    np.testing.assert_allclose(azimuthal_integrator(calibration).center_array((1064, 1030), unit="2th_deg"), reference, atol=1e-9)


def test_beam_centre_and_rings():
    col, row = TRUE.beam_center()
    assert col == pytest.approx(1210.5, abs=0.5) and row == pytest.approx(501.1, abs=0.5)
    guess = Geometry.from_guess(0.165, 1210.5, 501.1, TRUE.wavelength)
    assert guess.beam_center() == pytest.approx((1210.5, 501.1), abs=1e-6) and guess.rot1 == 0.0
    assert calibrant_rings("LaB6", TRUE.wavelength)[:3] == pytest.approx([9.003, 12.745, 15.626], abs=2e-3)
    assert calibrant_rings("CeO2", TRUE.wavelength)[0] == pytest.approx(11.988, abs=2e-3)  # (111), a = 5.411 Å
    with pytest.raises(ValueError):
        calibrant_rings("Si", TRUE.wavelength)
    overlay = ring_overlay(np.array([[9.0, 9.003, 12.745], [5.0, 5.0, 5.0]]), np.array([9.003, 12.745]), width=0.002)
    assert overlay.tolist() == [[0, 1, 2], [0, 0, 0]]


def test_control_points():
    points = ControlPoints()
    points.add(np.zeros((3, 2)), 0)
    points.add(np.ones((2, 2)), 4)
    assert len(points) == 5 and points.ring_counts() == {0: 3, 4: 2}
    assert points.as_pyfai_array().shape == (5, 3)
    points.remove_ring(0)
    assert points.ring_counts() == {4: 2}


def test_calibration_image_from_a_stack(tmp_path):
    with h5py.File(tmp_path / "scan_jf1m_0000.h5", "w") as f:
        f["entry_0000/measurement/data"] = np.stack([np.full((4, 5), v, dtype=np.int32) for v in (1, 2, 6)])
    np.testing.assert_allclose(load_calibration_image(tmp_path / "scan_jf1m_0000.h5"), np.full((4, 5), 3.0))
    np.testing.assert_allclose(load_calibration_image(tmp_path / "scan_jf1m_0000.h5", frames=[2]), np.full((4, 5), 6.0))
