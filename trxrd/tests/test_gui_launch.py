"""The entry point of the executable: without a console, errors must reach a log file and a message box."""

import sys

import pytest

from trxrd.gui import app as launch


@pytest.fixture
def started(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "trxrd-gui.log"
    shown = []
    monkeypatch.setattr(launch, "log_path", lambda: path)
    monkeypatch.setattr(launch, "show_error", shown.append)
    monkeypatch.setattr(launch, "_shown", set())
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # main() replaces it
    yield path, shown
    launch.stop_log()


def test_a_normal_run_leaves_its_start_and_end_in_the_log(started, monkeypatch):
    path, shown = started
    monkeypatch.setattr("trxrd.gui.builder.run", lambda argv: 0)
    assert launch.main([]) == 0
    text = path.read_text(encoding="utf-8")
    assert "start:" in text and "end: exit code 0" in text and shown == []
    assert sys.excepthook is launch.report


def test_an_error_at_start_is_logged_and_shown(started, monkeypatch):
    path, shown = started
    monkeypatch.setattr("trxrd.gui.builder.run", lambda argv: 1 / 0)
    assert launch.main([]) == 1
    assert "ZeroDivisionError" in path.read_text(encoding="utf-8")
    assert len(shown) == 1 and shown[0].startswith("ZeroDivisionError") and str(path) in shown[0]


def test_an_import_error_is_logged_and_shown(started, monkeypatch):
    path, shown = started
    monkeypatch.setitem(sys.modules, "trxrd.gui.builder", None)  # what a broken module gives
    assert launch.main([]) == 1
    assert "ModuleNotFoundError" in path.read_text(encoding="utf-8")
    assert len(shown) == 1


def test_an_error_repeated_at_every_repaint_is_shown_once(started):
    path, shown = started
    launch.start_log()
    for _ in range(5):
        try:
            raise ValueError("in a slot")
        except ValueError:
            launch.report(*sys.exc_info())
    assert len(shown) == 1 and path.read_text(encoding="utf-8").count("ValueError: in a slot") == 5


def test_an_unwritable_log_does_not_stop_the_program(started, monkeypatch, tmp_path):
    path, shown = started
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(launch, "log_path", lambda: blocker / "sub" / "trxrd-gui.log")  # its folder cannot exist
    monkeypatch.setattr("trxrd.gui.builder.run", lambda argv: 0)
    assert launch.main([]) == 0
