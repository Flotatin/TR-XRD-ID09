from trxrd.io.h5store import save_run
from trxrd.services.session import Event, Session


def test_open_navigate_modify_save(tmp_path, sample_run):
    path = save_run(sample_run, tmp_path / "run.h5")
    session = Session()
    events = []
    for event in (Event.RUN_CHANGED, Event.SPECTRUM_CHANGED, Event.DATA_CHANGED, Event.SAVED):
        session.subscribe(event, lambda s, e=event: events.append(e))

    session.open(path)
    assert events == [Event.RUN_CHANGED, Event.SPECTRUM_CHANGED] and not session.dirty
    session.select(99)
    assert session.index == len(sample_run.spectra) - 1
    session.select(99)  # no change, no event
    assert events.count(Event.SPECTRUM_CHANGED) == 2

    session.run.note = "edited"
    session.modified()
    assert session.dirty
    session.save()
    assert not session.dirty and events[-1] == Event.SAVED
    assert Session().open(path).note == "edited"


def test_legacy_file_is_never_overwritten(tmp_path, monkeypatch, sample_run):
    from trxrd.io import legacy

    old = tmp_path / "run24.CEDXUp"
    old.write_bytes(b"legacy bytes")
    monkeypatch.setattr(legacy, "load_legacy", lambda path: object())
    monkeypatch.setattr(legacy, "legacy_to_run", lambda obj, name: sample_run)
    session = Session()
    session.open(old)
    assert session.dirty and session.path == tmp_path / "run24.trxrd.h5"
    session.save()
    assert old.read_bytes() == b"legacy bytes"
