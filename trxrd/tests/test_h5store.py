import dataclasses

import h5py
import numpy as np
import pytest

from trxrd.domain import Run, Spectrum
from trxrd.io.h5store import FORMAT_VERSION, FormatError, is_run_file, load_run, save_run


def assert_same(a, b, path="run"):
    """Deep equality for dataclasses holding numpy arrays (NaN == NaN)."""
    if dataclasses.is_dataclass(a):
        assert type(a) is type(b), path
        for field in dataclasses.fields(a):
            assert_same(getattr(a, field.name), getattr(b, field.name), f"{path}.{field.name}")
    elif isinstance(a, dict):
        assert list(a) == list(b), path
        for key in a:
            assert_same(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=path)
    elif isinstance(a, float) and np.isnan(a):
        assert np.isnan(b), path
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_round_trip_is_lossless(tmp_path, sample_run):
    path = save_run(sample_run, tmp_path / "run.h5")
    assert_same(sample_run, load_run(path))


def test_ragged_spectra_round_trip(tmp_path, sample_run):
    sample_run.spectra.append(Spectrum(two_theta=np.linspace(5, 6, 7), intensity=np.arange(7.0)))
    loaded = load_run(save_run(sample_run, tmp_path / "run.h5"))
    assert_same(sample_run.spectra, loaded.spectra)


def test_empty_run_round_trip(tmp_path):
    run = Run(name="empty")
    assert_same(run, load_run(save_run(run, tmp_path / "empty.h5")))


def test_file_is_self_describing(tmp_path, sample_run):
    path = save_run(sample_run, tmp_path / "run.h5")
    with h5py.File(path, "r") as f:
        assert f.attrs["file_format"] == "trxrd-run"
        assert f.attrs["format_version"] == FORMAT_VERSION
        entry = f["entry"]
        assert entry.attrs["default"] == "results"
        results = entry["results"]
        assert results.attrs["signal"] == "pressure_H2O_Ice_VII"
        np.testing.assert_allclose(results["pressure_H2O_Ice_VII"][:3], [2.0, 3.0, 4.0])
        assert "DIHKL" in entry["phases/H2O_Ice_VII/jcpds"].asstr()[()]
        assert entry["spectra"].attrs["signal"] == "intensity"


def test_save_is_atomic(tmp_path, sample_run):
    path = tmp_path / "run.h5"
    save_run(sample_run, path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.h5"]


def test_raw_data_external_link(tmp_path, sample_run):
    raw = tmp_path / "scan_jf1m_0000.h5"
    with h5py.File(raw, "w") as f:
        f["entry_0000/data"] = np.zeros((2, 4, 4))
    sample_run.source.raw_file, sample_run.source.raw_dataset = str(raw), "/entry_0000/data"
    path = save_run(sample_run, tmp_path / "run.h5")
    with h5py.File(path, "r") as f:
        assert f["entry/source/raw_data"].shape == (2, 4, 4)
    raw.unlink()  # shared without the raw data: the file still loads
    assert load_run(path).source.raw_file == str(raw)


def test_rejects_foreign_or_future_files(tmp_path):
    foreign = tmp_path / "foreign.h5"
    with h5py.File(foreign, "w") as f:
        f["x"] = 1
    assert not is_run_file(foreign)
    with pytest.raises(FormatError):
        load_run(foreign)
    with h5py.File(foreign, "a") as f:
        f.attrs["file_format"] = "trxrd-run"
        f.attrs["format_version"] = "99.0"
    with pytest.raises(FormatError):
        load_run(foreign)


def test_numpy_string_names_are_written_as_plain_strings(tmp_path, sample_run):
    """Old files hold some names as numpy.str_, which HDF5 attributes refuse."""
    card = sample_run.phases.pop("H2O_Ice_VI")
    card.name = np.str_("H2O_Ice_VI")
    sample_run.phases[np.str_("H2O_Ice_VI")] = card
    loaded = load_run(save_run(sample_run, tmp_path / "run.h5"))
    assert list(loaded.phases) == ["H2O_Ice_VII", "H2O_Ice_VI"]
    assert type(loaded.phases["H2O_Ice_VI"].name) is str
