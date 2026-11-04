"""Live mode: follow the raw data folder, analyse each new scan, never lose or overwrite anything."""

import dataclasses
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from trxrd.domain import PhaseFit, Processing
from trxrd.gui.presenters.live import LivePresenter
from trxrd.io.h5store import load_run, save_run
from trxrd.services import analysis, live
from trxrd.services.live import LiveSettings, Watch, image_files
from trxrd.services.run_builder import Cancelled, build_and_save
from trxrd.services.session import Session

ICE_VII = "H2O_Ice_VII"


class Window:
    """Fake window: records what the presenter says."""

    def __init__(self, answer=None):
        self.answer, self.is_busy = answer, False
        self.texts, self.errors, self.period = [], [], None
        self.suggested = None

    def ask_live(self, suggestion):
        self.suggested = suggestion
        return self.answer

    def start_live(self, period):
        self.period = period

    def stop_live(self):
        self.period = None

    def set_live(self, text):
        self.texts.append(text)

    def busy(self):
        return self.is_busy

    def show_error(self, title, text):
        self.errors.append(text)

    @property
    def text(self):
        return self.texts[-1]


def new_scan(dataset: Path, number: int, age: float = 0.0) -> Path:
    """A scan landing in the dataset: a copy of scan 3, newer than everything there."""
    source = dataset / "scan0003" / "scan_jf1m_0000.h5"
    target = dataset / f"scan{number:04d}" / "scan_jf1m_0000.h5"
    target.parent.mkdir(exist_ok=True)
    shutil.copy(source, target)
    stamp = source.stat().st_mtime + 100 * number - age
    os.utime(target, (stamp, stamp))
    return target


@pytest.fixture
def dataset(raw_root):
    return raw_root / "water02" / "water02_0001"


@pytest.fixture
def built(tmp_path, dataset):
    from conftest import DATA, LIBRARY

    return build_and_save(dataset / "scan0003" / "scan_jf1m_0000.h5", tmp_path / "analyse",
                          DATA / "calib_THT8_correct_samedi.poni", DATA / "scan_jf1m_0000.mask",
                          two_theta_range=(8, 25), phase_paths=[LIBRARY / "H2O"])


@pytest.fixture
def template(built):
    return live.template_of(load_run(built))


# ------------------------------------------------------------------ where the scans are


def test_scans_of_a_dataset_a_sample_or_raw_data(raw_root, dataset):
    wanted = {dataset / "scan0003" / "scan_jf1m_0000.h5", new_scan(dataset, 4)}
    for folder in (dataset, dataset.parent, raw_root):
        assert set(image_files(folder)) == wanted


def test_no_scan_in_an_empty_or_missing_folder(tmp_path):
    assert image_files(tmp_path) == [] and image_files(tmp_path / "nowhere") == []


def test_the_dataset_of_a_run(built, dataset):
    assert live.dataset_folder(load_run(built)) == dataset


# ------------------------------------------------------------------ when a scan is ready


def test_what_is_there_at_the_start_is_not_new(dataset):
    watch = Watch(dataset)
    assert watch.poll() is None and watch.poll() is None


def test_a_new_scan_comes_once_it_stops_growing(dataset):
    watch = Watch(dataset)
    scan = new_scan(dataset, 4)
    assert watch.poll() is None  # it may still be growing
    with open(scan, "ab") as file:
        file.write(b"\0" * 10)  # it was
    assert watch.poll() is None
    assert watch.poll() == scan
    assert watch.poll() is None and watch.poll() is None  # once


def test_a_scan_rewritten_without_changing_its_size_is_still_growing(dataset):
    watch = Watch(dataset)
    scan = new_scan(dataset, 4)
    assert watch.poll() is None
    stamp = scan.stat().st_mtime + 5
    os.utime(scan, (stamp, stamp))
    assert watch.poll() is None


def test_an_unreadable_scan_is_tried_again_then_left(dataset):
    watch = Watch(dataset)
    scan = new_scan(dataset, 4)
    watch.poll()
    assert watch.poll() == scan
    assert watch.failed(scan)
    watch.poll()
    assert watch.poll() == scan  # again
    assert watch.failed(scan)
    watch.poll()
    assert watch.poll() == scan
    assert not watch.failed(scan)  # third time: left
    assert watch.poll() is None and watch.poll() is None


def test_no_scan_is_skipped(dataset):
    """Two shots between two looks, or during the long analysis of the one before."""
    watch = Watch(dataset)
    first, second = new_scan(dataset, 4), new_scan(dataset, 5)
    watch.poll()
    assert watch.poll() == first
    watch.poll()
    assert watch.poll() == second
    assert watch.poll() is None


def test_the_scans_come_in_the_order_they_landed(dataset):
    watch = Watch(dataset)
    later, earlier = new_scan(dataset, 4), new_scan(dataset, 5, age=1000)  # whatever their numbers
    watch.poll()
    assert watch.poll() == earlier
    watch.poll()
    assert watch.poll() == later


def test_a_scan_still_being_written_does_not_hold_back_the_one_before(dataset):
    watch = Watch(dataset)
    complete, growing = new_scan(dataset, 4), new_scan(dataset, 5)
    stamp = growing.stat().st_mtime
    watch.poll()
    with open(growing, "ab") as file:
        file.write(b"\0" * 10)
    os.utime(growing, (stamp + 1, stamp + 1))  # written now: the dates of these scans are ahead of the clock
    assert watch.poll() == complete


def test_a_newer_scan_passes_a_scan_left_behind(dataset):
    watch = Watch(dataset)
    scan = new_scan(dataset, 4)
    watch.poll()
    watch.poll()
    for _ in range(live.MAX_TRIES):
        watch.failed(scan)
    newer = new_scan(dataset, 5)
    watch.poll()
    assert watch.poll() == newer


# ------------------------------------------------------------------ what a new scan takes


def test_a_new_scan_takes_the_calibration_the_phases_and_the_treatment(built, dataset, tmp_path):
    from_run = load_run(built)
    for spectrum in from_run.spectra:
        spectrum.processing = Processing(4, "svg", (11, 3), baseline_window=1.0)
    from_run.ui_state["kept"] = 1
    scan = new_scan(dataset, 4)
    run, path = live.process(scan, live.template_of(from_run), LiveSettings(dataset, tmp_path / "out", analyse=False))
    assert run.name == "water02_0001_scan0004" and path == tmp_path / "out" / "water02_0001_scan0004.trxrd.h5"
    assert path.is_file() and len(load_run(path)) == 3
    assert run.calibration.poni == from_run.calibration.poni
    assert np.array_equal(run.calibration.mask, from_run.calibration.mask)
    assert set(run.phases) == set(from_run.phases) and run.ui_state["kept"] == 1
    assert run.spectra[0].processing == Processing(4, "svg", (11, 3), baseline_window=1.0)
    assert not any(s.phases for s in run.spectra)  # not analysed: integrated only


def test_the_template_is_a_copy(built):
    run = load_run(built)
    template = live.template_of(run)
    run.spectra[0].processing.baseline_degree = 40
    run.ui_state["later"] = 1
    run.phases.pop(next(iter(run.phases)))
    assert template.processing.baseline_degree != 40 and "later" not in template.ui_state
    assert len(template.phases) == len(run.phases) + 1


def test_the_phases_worked_with_are_the_candidates(built):
    run = load_run(built)
    assert live.template_of(run).ui_state[analysis.UI_KEY]["phases"] is None  # none yet: all those of the run
    run.spectra[1].phases = [PhaseFit(phase=ICE_VII, pressure=3.0)]
    assert live.template_of(run).ui_state[analysis.UI_KEY]["phases"] == [ICE_VII]


def test_the_phases_chosen_for_the_analysis_stay_the_candidates(built):
    run = load_run(built)
    run.ui_state[analysis.UI_KEY] = analysis.AnalysisSettings(phases=["H2O_Ice_VI", ICE_VII]).to_dict()
    run.spectra[1].phases = [PhaseFit(phase=ICE_VII, pressure=3.0)]
    assert live.template_of(run).ui_state[analysis.UI_KEY]["phases"] == ["H2O_Ice_VI", ICE_VII]


def test_each_scan_is_analysed_with_the_settings_of_the_run(template, dataset, tmp_path, monkeypatch):
    template.ui_state[analysis.UI_KEY]["exclude"] = [[18.4, 18.8]]
    template.ui_state[analysis.UI_KEY]["pressure_range"] = [1.0, 12.0]
    seen = {}
    real = analysis.analyse

    def spy(run, indices, settings, progress=None):
        seen.update(indices=list(indices), settings=settings)
        return real(run, indices, settings, progress)

    monkeypatch.setattr(analysis, "analyse", spy)
    run, path = live.process(new_scan(dataset, 4), template, LiveSettings(dataset, tmp_path / "out"))
    assert seen["indices"] == [0, 1, 2]
    assert seen["settings"].exclude == [(18.4, 18.8)] and seen["settings"].pressure_range == (1.0, 12.0)
    assert load_run(path).ui_state[analysis.UI_KEY]["exclude"] == [[18.4, 18.8]]


def test_a_run_without_phases_is_only_integrated(template, dataset, tmp_path):
    empty = dataclasses.replace(template, phases={})
    run, _ = live.process(new_scan(dataset, 4), empty, LiveSettings(dataset, tmp_path / "out"))
    assert len(run) == 3 and not any(s.phases for s in run.spectra)


def test_an_analysis_already_saved_is_opened_not_overwritten(template, dataset, tmp_path):
    settings = LiveSettings(dataset, tmp_path / "out", analyse=False)
    scan = new_scan(dataset, 4)
    first, path = live.process(scan, template, settings)
    first.note = "analyse faite à la main"
    first.spectra[0].phases = [PhaseFit(phase=ICE_VII, pressure=3.0)]
    save_run(first, path)
    again, same = live.process(scan, template, settings)
    assert same == path and again.note == "analyse faite à la main" and again.spectra[0].phases


@pytest.mark.parametrize("analyse, stop_at", [(False, 1), (True, 4)], ids=["integrating", "analysing"])
def test_a_stopped_scan_saves_nothing(template, dataset, tmp_path, analyse, stop_at):
    settings = LiveSettings(dataset, tmp_path / "out", analyse=analyse)
    with pytest.raises(Cancelled):
        live.process(new_scan(dataset, 4), template, settings, lambda done, total: done < stop_at)
    assert not (tmp_path / "out" / "water02_0001_scan0004.trxrd.h5").exists()


def test_the_progress_covers_both_stages(template, dataset, tmp_path):
    steps = []
    live.process(new_scan(dataset, 4), template, LiveSettings(dataset, tmp_path / "out"),
                 lambda done, total: steps.append((done, total)) or True)
    assert steps[0] == (1, 6) and steps[2] == (3, 6) and steps[-1] == (6, 6)


def test_the_raw_data_are_only_read(template, dataset, tmp_path):
    scan = new_scan(dataset, 4)
    before = sorted((p, p.stat().st_mtime_ns, p.stat().st_size) for p in dataset.rglob("*") if p.is_file())
    live.process(scan, template, LiveSettings(dataset, tmp_path / "out"))
    assert sorted((p, p.stat().st_mtime_ns, p.stat().st_size) for p in dataset.rglob("*") if p.is_file()) == before


# ------------------------------------------------------------------ the presenter


@pytest.fixture
def running(built, dataset, tmp_path):
    session, window = Session(), Window(LiveSettings(dataset, tmp_path / "out", analyse=False))
    session.open(built)
    presenter = LivePresenter(session, window)
    assert presenter.toggle(True)
    return presenter


def look(presenter, times=2):
    for _ in range(times):
        presenter.tick()


def test_it_needs_a_run_with_a_calibration(built):
    session, window = Session(), Window()
    presenter = LivePresenter(session, window)
    assert not presenter.toggle(True) and "Ouvrir d'abord un run" in window.errors[0]
    session.open(built)
    session.run.calibration.poni = ""
    assert not presenter.toggle(True) and "calibration" in window.errors[1] and window.period is None


def test_the_dialog_suggests_the_dataset_and_the_folder_of_the_run(built, dataset):
    session, window = Session(), Window(None)
    session.open(built)
    assert not LivePresenter(session, window).toggle(True)  # cancelled
    assert window.suggested.folder == dataset and window.suggested.output == built.parent and window.period is None


def test_it_looks_at_regular_times_and_says_so(running):
    assert running.window.period == LivePresenter.PERIOD and running.running
    assert running.window.text.startswith("Live : en attente du scan suivant")
    assert running.toggle(True)  # already on
    assert not running.toggle(False)
    assert running.window.period is None and running.window.text is None and not running.running


def test_a_new_scan_is_opened(running, dataset):
    session, window = running.session, running.window
    look(running)
    assert session.run.name == "water02_0001_scan0003"
    new_scan(dataset, 4)
    look(running)
    assert session.run.name == "water02_0001_scan0004" and not session.dirty
    assert session.path.is_file() and "ouvert à" in window.text and window.text.endswith("en attente du scan suivant")
    look(running, 3)
    assert session.run.name == "water02_0001_scan0004"  # once


def test_the_next_scan_takes_what_was_just_changed(running, dataset):
    running.session.run.ui_state["changed"] = 1  # after the start of the mode
    new_scan(dataset, 4)
    look(running)
    assert running.session.run.ui_state["changed"] == 1


def test_unsaved_changes_are_never_swept_away(running, dataset, built):
    session, window = running.session, running.window
    session.run.spectra[0].phases = [PhaseFit(phase=ICE_VII, pressure=3.0)]
    session.modified()
    new_scan(dataset, 4)
    look(running)
    assert session.run.name == "water02_0001_scan0003" and session.dirty and session.run.spectra[0].phases
    assert "s'ouvrira dès que le run affiché sera enregistré" in window.text
    assert (running.settings.output / "water02_0001_scan0004.trxrd.h5").is_file()  # saved all the same
    look(running, 3)
    assert session.run.name == "water02_0001_scan0003"
    session.save()
    running.tick()
    assert session.run.name == "water02_0001_scan0004"
    assert load_run(built).spectra[0].phases  # what was edited is in its file


def test_a_dialog_or_a_long_task_holds_the_switch(running, dataset):
    session, window = running.session, running.window
    window.is_busy = True
    new_scan(dataset, 4)
    look(running)
    assert session.run.name == "water02_0001_scan0003" and "fin de l'opération" in window.text
    window.is_busy = False
    running.tick()
    assert session.run.name == "water02_0001_scan0004"


def test_the_latest_scan_shows_after_a_long_wait(running, dataset):
    session = running.session
    session.modified()
    new_scan(dataset, 4)
    look(running)
    new_scan(dataset, 5)
    look(running)
    session.save()
    running.tick()
    assert session.run.name == "water02_0001_scan0005"
    assert (running.settings.output / "water02_0001_scan0004.trxrd.h5").is_file()


def test_a_scan_that_cannot_be_read_yet_is_tried_again(running, dataset, monkeypatch):
    real = live.process
    calls = []

    def failing_once(scan, template, settings, progress=None):
        calls.append(scan)
        if len(calls) == 1:
            raise OSError("unable to lock file")
        return real(scan, template, settings, progress)

    monkeypatch.setattr(live, "process", failing_once)
    new_scan(dataset, 4)
    look(running)
    assert "nouvel essai" in running.window.text and running.session.run.name == "water02_0001_scan0003"
    look(running)
    assert len(calls) == 2 and running.session.run.name == "water02_0001_scan0004"


def test_a_scan_that_stays_unreadable_is_left_and_the_mode_goes_on(running, dataset, monkeypatch):
    def failing(scan, template, settings, progress=None):
        raise OSError("corrupted")

    monkeypatch.setattr(live, "process", failing)
    new_scan(dataset, 4)
    look(running, 2 * live.MAX_TRIES)
    assert "ignoré" in running.window.text and "corrupted" in running.window.text and running.running
    monkeypatch.undo()
    new_scan(dataset, 5)
    look(running)
    assert running.session.run.name == "water02_0001_scan0005"


def test_a_folder_that_cannot_be_read_is_said_and_the_mode_goes_on(running, monkeypatch):
    def failing(self):
        raise OSError("network lost")

    monkeypatch.setattr(Watch, "poll", failing)
    running.tick()
    assert "dossier illisible" in running.window.text and "network lost" in running.window.text and running.running
    monkeypatch.undo()
    running.tick()
    assert not running._looking


def test_the_run_can_be_closed_meanwhile(running, dataset):
    running.session.close()
    new_scan(dataset, 4)
    look(running)
    assert running.session.run.name == "water02_0001_scan0004"


class Waiting:
    """Runner that keeps the tasks: they end when the test says so."""

    def __init__(self):
        self.jobs = []

    def run(self, task, on_done, on_error, on_progress=None):
        job = type("Job", (), {"cancelled": False})()
        self.jobs.append((task, on_done, on_error, on_progress, job))
        return job

    def finish(self, index=0):
        task, on_done, on_error, on_progress, job = self.jobs.pop(index)
        try:
            result = task(lambda done, total: not job.cancelled)
        except Exception as exc:
            on_error(f"{type(exc).__name__}: {exc}")
        else:
            on_done(result)


@pytest.fixture
def threaded(running):
    running.runner = Waiting()
    return running


def test_looks_never_overlap(threaded):
    threaded.tick()
    threaded.tick()
    assert len(threaded.runner.jobs) == 1


def test_no_look_while_a_scan_is_built(threaded, dataset):
    new_scan(dataset, 4)
    for _ in range(2):
        threaded.tick()
        threaded.runner.finish()
    assert threaded._building is not None and len(threaded.runner.jobs) == 1  # the build
    threaded.tick()
    assert len(threaded.runner.jobs) == 1


def test_stopping_stops_the_scan_being_built_and_opens_nothing(threaded, dataset):
    new_scan(dataset, 4)
    for _ in range(2):
        threaded.tick()
        threaded.runner.finish()
    job = threaded.runner.jobs[0][-1]
    written = threaded.settings.output / "water02_0001_scan0004.trxrd.h5"
    threaded.toggle(False)
    assert job.cancelled
    threaded.runner.finish()
    assert threaded.session.run.name == "water02_0001_scan0003" and threaded.window.text is None
    assert not written.exists()


def test_a_scan_that_ends_just_after_the_stop_is_not_opened(threaded, dataset):
    """Too late to stop it: its last step was over. Its file stays, but the user said stop."""
    new_scan(dataset, 4)
    for _ in range(2):
        threaded.tick()
        threaded.runner.finish()
    task, on_done, *_ = threaded.runner.jobs.pop(0)
    result = task(lambda done, total: True)
    threaded.toggle(False)
    on_done(result)
    assert threaded.session.run.name == "water02_0001_scan0003" and threaded._ready is None
    assert threaded.window.text is None


def test_a_look_that_ends_after_the_stop_starts_nothing(threaded, dataset):
    new_scan(dataset, 4)
    threaded.tick()
    threaded.runner.finish()
    threaded.tick()
    threaded.toggle(False)
    threaded.runner.finish()
    assert not threaded.runner.jobs and threaded._building is None


def test_a_new_start_does_not_take_the_end_of_the_old_build(threaded, dataset, tmp_path):
    new_scan(dataset, 4)
    for _ in range(2):
        threaded.tick()
        threaded.runner.finish()
    threaded.toggle(False)
    threaded.window.answer = LiveSettings(dataset, tmp_path / "other", analyse=False)
    threaded.toggle(True)
    threaded.runner.finish()  # the old build ends, stopped
    assert threaded.session.run.name == "water02_0001_scan0003" and threaded.running
    assert threaded.window.text.startswith("Live : en attente")


def test_two_scans_landing_together_are_both_analysed_and_the_last_is_shown(running, dataset):
    new_scan(dataset, 4)
    new_scan(dataset, 5)
    look(running, 4)
    assert running.session.run.name == "water02_0001_scan0005"
    for number in (4, 5):
        assert (running.settings.output / f"water02_0001_scan{number:04d}.trxrd.h5").is_file()


def test_a_scan_landing_during_a_long_build_is_not_skipped(threaded, dataset):
    new_scan(dataset, 4)
    for _ in range(2):
        threaded.tick()
        threaded.runner.finish()
    new_scan(dataset, 5)  # scan 4 is still being built
    threaded.tick()
    threaded.runner.finish()  # its end
    assert threaded.session.run.name == "water02_0001_scan0004"
    for _ in range(2):
        threaded.tick()
        threaded.runner.finish()
    threaded.runner.finish()
    assert threaded.session.run.name == "water02_0001_scan0005"


def test_the_status_follows_the_build(running, dataset):
    new_scan(dataset, 4)
    look(running)
    assert any(text and text.startswith("Live : scan0004…") and text.endswith("%") for text in running.window.texts)
