"""Preview of a raw scan before its integration ("new run" dialog): service and display data, on the TUTO scan."""

import numpy as np
import pytest
from conftest import DATA

pytest.importorskip("pyFAI")

SCAN, PONI, MASK = DATA / "scan_jf1m_0000.h5", DATA / "calib_THT8_correct_samedi.poni", DATA / "scan_jf1m_0000.mask"


@pytest.fixture(scope="module")
def preview():
    from trxrd.services.images import ScanPreview

    return ScanPreview(SCAN, PONI, MASK)


@pytest.fixture(scope="module")
def frames():
    from trxrd.io import esrf

    return [np.asarray(frame, dtype=np.float32) for _, frame in esrf.ImageStack.open(SCAN).frames()]


def test_one_frame_or_the_maximum_over_the_scan(preview, frames):
    assert preview.frames == len(frames) == 16 and preview.maximum_frames() == list(range(16))
    np.testing.assert_array_equal(preview.image(), np.max(frames, axis=0))  # every line seen at any time
    np.testing.assert_array_equal(preview.image(1), frames[1])
    np.testing.assert_array_equal(preview.image(99), frames[-1])
    assert preview.image(progress=lambda done, total: done < 2) is None  # stopped by a newer request


def test_maximum_of_a_long_scan_reads_evenly_spaced_frames(monkeypatch, preview, frames):
    monkeypatch.setattr(preview, "MAX_FRAMES", 2)
    assert preview.maximum_frames() == [0, 15]
    np.testing.assert_array_equal(preview.image(), np.maximum(frames[0], frames[15]))


def test_two_theta_of_the_pixels_and_pattern_over_the_whole_detector(preview, frames):
    from trxrd.io.calibration import calibration_from_files
    from trxrd.processing.integration import Integrator

    image = preview.image()
    masked = preview.mask > 0
    shown = preview.shown(image)
    assert np.isnan(shown[masked]).all() and np.isfinite(shown[~masked]).all()
    two_theta, azimuth = preview.angle_maps(image.shape)
    assert np.isnan(two_theta[masked]).all() and np.isfinite(azimuth[~masked]).all()
    low, high = preview.usable_range(image.shape)
    assert low <= np.nanmin(two_theta) < low + 0.01 and high - 0.01 < np.nanmax(two_theta) <= high
    assert low < 8 and high > 25  # wider than the range of the TUTO runs

    pattern = preview.pattern(image)
    assert pattern.two_theta[0] >= low and pattern.two_theta[-1] <= high
    assert np.array_equal(np.isfinite(pattern.intensity), pattern.coverage > 0)
    assert 0 < np.max(pattern.coverage) <= 360
    calibration = calibration_from_files(PONI, MASK, two_theta_range=(8.0, 25.0))
    x, y = Integrator(calibration).integrate(image)  # what the run will hold
    inside = (pattern.two_theta > 8) & (pattern.two_theta < 25)
    strongest = pattern.two_theta[inside][np.nanargmax(pattern.intensity[inside])]
    assert abs(strongest - x[np.argmax(y)]) < 0.03

    cake = preview.cake(image)
    assert cake.intensity.shape == (360, pattern.two_theta.size)
    np.testing.assert_allclose(cake.two_theta, pattern.two_theta)


def test_azimuthal_coverage_counts_the_degrees_seen():
    """Each 1° azimuth bin holding a usable pixel counts once in its 2θ bin; masked pixels (NaN) do not count."""
    from trxrd.services.images import ScanPreview

    preview = ScanPreview.__new__(ScanPreview)
    preview._coverage = None
    tth = np.array([[10.05, 10.05, 10.15, np.nan]])
    chi = np.array([[0.5, 89.5, 45.0, 10.0]])
    preview.angle_maps = lambda shape: (tth, chi)
    coverage = preview._coverage_of(np.array([10.05, 10.15, 10.25]), tth.shape)
    np.testing.assert_array_equal(coverage, [2, 1, 0])


def test_without_calibration_only_the_images(frames):
    from trxrd.services.images import ScanPreview

    bare = ScanPreview(SCAN)
    assert bare.angle_maps(frames[0].shape) is None and bare.usable_range(frames[0].shape) is None
    assert np.isfinite(bare.shown(frames[0])).all()  # no mask either
    with pytest.raises(ValueError, match="poni"):
        bare.pattern(frames[0])


def test_preview_display_data(preview, frames):
    from trxrd.gui.presenters import display
    from trxrd.services.images import ScanPreview

    image = preview.image()
    data = display.preview_data(preview, image, None)
    assert data.title == "maximum de 16 images sur 16" and data.frames == 16 and data.cake is None
    assert data.x.size == data.coverage.size == data.intensity.size and data.usable == preview.usable_range(image.shape)
    assert display.preview_data(preview, frames[1], 1).title == "image n° 1 (de 0 à 15)"
    cake = display.preview_data(preview, image, None, "cake").cake
    assert cake.x_label == "2θ (°)" and cake.image.shape[1] == data.x.size

    outside = display.outside_range(data.two_theta_map, 10.0, 20.0)
    tth = data.two_theta_map
    assert outside[tth < 10].all() and outside[tth > 20].all() and not outside[(tth > 10) & (tth < 20)].any()
    assert not outside[np.isnan(tth)].any()  # masked pixels are not "outside": they are not used at all
    text = display.range_text(data, 10.0, 20.0)
    assert text.startswith("Plage intégrée : 10.00 → 20.00° (10.00°)") and "d'azimut" in text and "au-delà" not in text
    assert "au-delà des pixels utilisables" in display.range_text(data, 2.0, 20.0)

    bare = display.preview_data(ScanPreview(SCAN), image, None)
    assert bare.x is None and bare.two_theta_map is None and ".poni" in bare.message
    assert display.range_text(bare, 10.0, 20.0) == ""
