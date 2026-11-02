"""Small h5py helpers shared by the run store and the notebook reader."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import h5py
import numpy as np

STR = h5py.string_dtype("utf-8")
COMPRESSION = dict(compression="gzip", compression_opts=4, shuffle=True)


def nx_group(parent: h5py.Group, name: str, nx_class: str, **attrs: Any) -> h5py.Group:
    group = parent.require_group(name)
    group.attrs["NX_class"] = nx_class
    for key, value in attrs.items():
        group.attrs[key] = value
    return group


def write_str(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=value or "", dtype=STR)


def write_strings(group: h5py.Group, name: str, values) -> h5py.Dataset:
    values = [str(v) for v in values]
    if not values:
        return group.create_dataset(name, shape=(0,), dtype=STR)
    return group.create_dataset(name, data=np.array(values, dtype=object), dtype=STR)


def read_str(group: h5py.Group, name: str, default: str = "") -> str:
    if name not in group:
        return default
    value = group[name].asstr()[()]
    return value if isinstance(value, str) else default


def write_array(group: h5py.Group, name: str, data, **kwargs) -> h5py.Dataset:
    data = np.asarray(data)
    if data.ndim >= 2 and data.size > 1024:
        kwargs = {**COMPRESSION, **kwargs}
    return group.create_dataset(name, data=data, **kwargs)


def read_scalar(group: h5py.Group, name: str, default: Any = None) -> Any:
    if name not in group:
        return default
    value = group[name][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_table(parent: h5py.Group, name: str, columns: Mapping[str, Any]) -> h5py.Group:
    """Store a table as a group of equally long 1D datasets (readable from any HDF5 tool)."""
    group = nx_group(parent, name, "NXcollection")
    group.attrs["columns"] = list(columns)
    for column, values in columns.items():
        if isinstance(values, np.ndarray) and values.dtype != object:
            group.create_dataset(column, data=values)
        else:
            write_strings(group, column, values)
    return group


def read_table(group: h5py.Group) -> Dict[str, np.ndarray]:
    columns = [str(c) for c in group.attrs.get("columns", list(group))]
    table: Dict[str, np.ndarray] = {}
    for column in columns:
        dataset = group[column]
        if h5py.check_string_dtype(dataset.dtype) is not None:
            table[column] = np.array(dataset.asstr()[()], dtype=object)
        else:
            table[column] = dataset[()]
    return table


def nan_if_none(value: Optional[float]) -> float:
    return np.nan if value is None else float(value)


def none_if_nan(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else value
