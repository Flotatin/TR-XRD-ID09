"""Over the cake and the image: the lines of the phases, the 2θ chosen with a click (a line, or an arc on the image)."""

import numpy as np
import pytest

from trxrd.domain import PhaseFit
from trxrd.gui.presenters import display
from trxrd.gui.presenters.viewer import CakePresenter, DisplaySettings
from trxrd.processing.geometry import iso_curve
from trxrd.services.images import FrameSource
from trxrd.services.session import Session

ICE_VII = "H2O_Ice_VII"


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def method(*args):
            self.calls.append((name, args))
        return method

    def last(self, name):
        return next(args for n, args in reversed(self.calls) if n == name)

    def count(self, name):
        return sum(n == name for n, _ in self.calls)


def maps(shape, center, stretch=1.0):
    """Radius and azimuth of each pixel around `center` (x, y): `stretch` < 1 makes ellipses, like a tilted detector."""
    rows, columns = np.mgrid[:shape[0], :shape[1]] + 0.5
    dx, dy = columns - center[0], (rows - center[1]) / stretch
    return np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx))


# ------------------------------------------------------------------ the curve


def test_a_ring_with_the_beam_outside_the_detector():
    """As with the Jungfrau 1M at ID09: only an arc is seen."""
    center = (-300.0, 150.0)
    radius, chi = maps((300, 400), center)
    x, y = iso_curve(radius, chi, 450.0)
    assert np.isfinite(x).all() and x.size > 300  # one piece
    np.testing.assert_allclose(np.hypot(x - center[0], y - center[1]), 450.0, atol=0.02)
    assert (x >= 0).all() and (x <= 400).all() and (y >= 0).all() and (y <= 300).all()


def test_a_tilted_detector_gives_an_ellipse_not_a_circle():
    center, stretch = (-100.0, 120.0), 0.7
    radius, chi = maps((300, 400), center, stretch)
    x, y = iso_curve(radius, chi, 260.0)
    np.testing.assert_allclose(np.hypot(x - center[0], (y - center[1]) / stretch), 260.0, atol=0.03)


def test_the_curve_runs_along_the_ring():
    """Sorted badly, it would be drawn as a scribble."""
    radius, chi = maps((300, 400), (-300.0, 150.0))
    x, y = iso_curve(radius, chi, 450.0)
    assert np.hypot(np.diff(x), np.diff(y)).max() < 2.0


def test_a_ring_that_leaves_the_detector_comes_back_in_another_piece():
    radius, chi = maps((200, 200), (100.0, 100.0))
    x, y = iso_curve(radius, chi, 120.0)  # wider than the detector: only its four corners are crossed
    pieces = np.split(np.arange(x.size), np.flatnonzero(np.isnan(x)))
    assert np.isnan(x).sum() >= 3 and len(pieces) >= 4
    keep = np.isfinite(x)
    np.testing.assert_allclose(np.hypot(x[keep] - 100, y[keep] - 100), 120.0, atol=0.03)


def test_no_point_for_a_two_theta_off_the_detector():
    radius, chi = maps((50, 50), (25.0, 25.0))
    x, y = iso_curve(radius, chi, 500.0)
    assert x.size == 0 and y.size == 0


def test_a_ring_round_the_beam_crosses_the_azimuth_limit():
    radius, chi = maps((300, 300), (150.0, 150.0))
    x, y = iso_curve(radius, chi, 100.0)
    np.testing.assert_allclose(np.hypot(x - 150, y - 150), 100.0, atol=0.03)
    assert np.isfinite(x).all() and np.hypot(np.diff(x), np.diff(y)).max() < 2.0


def test_a_crossing_right_on_the_azimuth_limit_stays_in_its_place():
    """Two pixels on either side of ±180°: without care, their azimuth is averaged to 0, and the point flies across."""
    center = (150.0, 150.2)  # not symmetric, so that the ring passes between the two pixels of column 50
    radius, chi = maps((300, 300), center)
    above, below = radius[149, 50], radius[150, 50]
    assert chi[149, 50] < -179 and chi[150, 50] > 179 and above != below
    x, y = iso_curve(radius, chi, (above + below) / 2)
    assert np.isfinite(x).all() and np.hypot(np.diff(x), np.diff(y)).max() < 2.0


# ------------------------------------------------------------------ on the real geometry


@pytest.fixture
def built(tmp_path, raw_root):
    from conftest import DATA, LIBRARY
    from trxrd.services.run_builder import build_and_save

    scan = raw_root / "water02" / "water02_0001" / "scan0003" / "scan_jf1m_0000.h5"
    return build_and_save(scan, tmp_path / "analyse", DATA / "calib_THT8_correct_samedi.poni",
                          DATA / "scan_jf1m_0000.mask", two_theta_range=(8, 25), phase_paths=[LIBRARY / "H2O"])


def test_every_point_of_the_ring_is_at_that_two_theta(built):
    from trxrd.io.h5store import load_run

    source = FrameSource(load_run(built))
    x, y = source.ring(15.0)
    keep = np.isfinite(x)
    assert keep.sum() > 200
    seen = [source.two_theta_at(a, b) for a, b in zip(x[keep], y[keep])]
    np.testing.assert_allclose(seen, 15.0, atol=0.02)  # a pixel is about 0.02°


def test_two_theta_under_a_pixel(built):
    from trxrd.io.h5store import load_run

    source = FrameSource(load_run(built))
    two_theta = source._angles()[0]
    assert source.two_theta_at(20.5, 10.5) == pytest.approx(two_theta[10, 20])
    assert source.two_theta_at(-1.0, 10.0) is None and source.two_theta_at(10.0, 1e6) is None


def test_the_image_and_its_angles_have_the_same_shape(built):
    from trxrd.io.h5store import load_run

    source = FrameSource(load_run(built))
    assert source._angles()[0].shape == source.frame(0).shape


# ------------------------------------------------------------------ the presenter


@pytest.fixture
def cake(built):
    session, settings, view, spectrum = Session(), DisplaySettings(), Recorder(), Recorder()
    presenter = CakePresenter(session, settings, view, None, spectrum)
    session.open(built)
    session.run.spectra[0].phases = [PhaseFit(phase=ICE_VII, pressure=3.0)]
    session.modified()
    presenter.spectrum_view = spectrum
    return presenter


def test_the_lines_of_the_phases_are_on_the_cake(cake):
    lines, = cake.view.last("show_lines")
    assert [line.phase for line in lines] == [ICE_VII] and lines[0].x.size >= 1
    expected = display.expected_lines(cake.session.run, cake.session.spectrum, "2th_deg")[0]
    np.testing.assert_allclose(lines[0].x, expected.x)


def test_the_lines_follow_the_pressure_without_a_new_cake(cake):
    before = cake.view.last("show_lines")[0][0].x.copy()
    cakes = cake.view.count("show_cake")
    cake.session.spectrum.phases[0].pressure = 6.0
    cake.session.modified()
    assert (cake.view.last("show_lines")[0][0].x > before).all() and cake.view.count("show_cake") == cakes


def test_no_line_on_the_image_nor_on_a_spectrum_without_phases(cake):
    cake.session.select(1)
    assert cake.view.last("show_lines") == ([],)
    cake.session.select(0)
    assert cake.view.last("show_lines")[0]
    cake.set_mode("image")
    assert cake.view.last("show_lines") == ([],)


def test_a_click_on_the_cake_chooses_a_two_theta(cake):
    cake.clicked(15.5, 200.0)
    assert cake.theta == 15.5
    assert cake.view.last("show_theta") == (15.5,) and cake.spectrum_view.last("show_theta") == (15.5,)
    assert cake.view.last("show_ring") == (None,)


def test_a_click_on_the_spectrum_chooses_it_too(cake):
    cake.spectrum_view.on_theta(16.0)
    assert cake.theta == 16.0 and cake.view.last("show_theta") == (16.0,)


def test_a_click_on_the_image_gives_the_two_theta_of_the_pixel_and_an_arc(cake):
    cake.set_mode("image")
    source = cake._source
    two_theta = source._angles()[0]
    cake.clicked(300.5, 200.5)
    assert cake.theta == pytest.approx(two_theta[200, 300])
    assert cake.spectrum_view.last("show_theta")[0] == pytest.approx(two_theta[200, 300])  # and the line on the spectrum
    assert cake.view.last("show_theta") == (None,)  # no straight line on the image
    x, y = cake.view.last("show_ring")[0]
    keep = np.isfinite(x)
    assert keep.sum() > 100 and np.min(np.hypot(x[keep] - 300.5, y[keep] - 200.5)) < 1.5  # it goes through the click


def test_a_click_outside_the_image_changes_nothing(cake):
    cake.set_mode("image")
    cake.clicked(300.5, 200.5)
    chosen = cake.theta
    cake.clicked(-50.0, 200.0)
    assert cake.theta == chosen


def test_a_click_on_the_spectrum_draws_the_arc_on_the_image(cake):
    cake.set_mode("image")
    cake.spectrum_view.on_theta(15.0)
    x, y = cake.view.last("show_ring")[0]
    assert np.isfinite(x).sum() > 100


def test_the_two_theta_stays_from_one_spectrum_to_the_next(cake):
    cake.clicked(15.5, 200.0)
    cake.session.select(2)
    assert cake.theta == 15.5 and cake.view.last("show_theta") == (15.5,)


def test_the_two_theta_goes_from_the_cake_to_the_image_and_back(cake):
    cake.clicked(15.5, 200.0)
    cake.set_mode("image")
    assert cake.view.last("show_theta") == (None,) and cake.view.last("show_ring")[0] is not None
    cake.set_mode("cake")
    assert cake.view.last("show_theta") == (15.5,) and cake.view.last("show_ring") == (None,)


def test_a_click_in_d_is_kept_in_two_theta(cake):
    cake.settings.set_unit("d_A")
    run = cake.session.run
    d = float(display.to_unit(15.0, "d_A", run))
    cake.clicked(d, 200.0)
    assert cake.theta == pytest.approx(15.0)
    assert cake.view.last("show_theta")[0] == pytest.approx(d)
    cake.settings.set_unit("2th_deg")
    assert cake.view.last("show_theta")[0] == pytest.approx(15.0)
    assert cake.spectrum_view.last("show_theta")[0] == pytest.approx(15.0)


@pytest.mark.filterwarnings("ignore:invalid value encountered in arcsin")
def test_a_position_impossible_in_q_is_no_two_theta(cake):
    cake.settings.set_unit("q_A^-1")
    cake.clicked(1e6, 200.0)
    assert cake.theta is None and cake.view.last("show_theta") == (None,)


def test_the_two_theta_can_be_erased(cake):
    cake.clicked(15.5, 200.0)
    cake.set_theta(None)
    assert cake.view.last("show_theta") == (None,) and cake.spectrum_view.last("show_theta") == (None,)


def test_a_new_run_has_no_two_theta(cake, built):
    cake.clicked(15.5, 200.0)
    cake.session.open(built)
    assert cake.theta is None and cake.spectrum_view.last("show_theta") == (None,)


def test_an_older_ring_is_dropped():
    """The newest click wins, whatever the order in which the worker threads end."""
    from trxrd.gui.presenters.viewer import CakePresenter as Presenter

    class Waiting:
        def __init__(self):
            self.jobs = []

        def run(self, task, on_done, on_error=None, on_progress=None):
            self.jobs.append((task, on_done))

    class Source:
        run = None

        def ring(self, value):
            return value

    session, view, runner = Session(), Recorder(), Waiting()
    presenter = Presenter(session, DisplaySettings(), view, runner)
    presenter._shown = "image"
    presenter._frames = lambda run, near: Source()
    presenter.theta = 15.0
    presenter._show_ring()
    presenter.theta = 16.0
    presenter._show_ring()
    older, newer = runner.jobs
    newer[1](newer[0](None))
    older[1](older[0](None))
    assert view.last("show_ring") == (16.0,)


def test_a_ring_that_cannot_be_drawn_is_said_and_keeps_the_image(cake):
    cake.set_mode("image")
    cake.session.run.calibration.poni = ""  # an old file without its .poni
    cake._source = None
    cake.clicked(300.5, 200.5)
    assert "calibration" in cake.view.last("show_note")[0]
    assert cake.view.count("show_message") == 0
