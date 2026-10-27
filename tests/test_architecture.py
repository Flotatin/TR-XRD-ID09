"""Dependency rules between layers (see docs/ARCHITECTURE.md)."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "trxrd"
GUI_LIBS = {"PyQt5", "PyQt6", "PySide6", "pyqtgraph", "qtpy"}

# layer -> internal layers it may import
ALLOWED = {
    "physics": {"physics"},
    "domain": {"domain", "physics"},
    "io": {"io", "domain", "physics"},
    "processing": {"processing", "domain", "physics"},
    "services": {"services", "io", "processing", "domain", "physics"},
    "notebook": {"io", "domain", "physics"},
    "cli": {"services", "io", "processing", "domain", "physics", "notebook"},
}
# third-party packages each layer may use
THIRD_PARTY = {
    "physics": {"numpy", "scipy"},
    "domain": {"numpy"},
}


def modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE)
        layer = rel.parts[0].removesuffix(".py")
        if layer != "__init__":
            yield path, layer


def imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


@pytest.mark.parametrize("path,layer", list(modules()), ids=lambda v: str(v).split("trxrd")[-1])
def test_layer_dependencies(path, layer):
    for name in imports(path):
        top = name.split(".")[0]
        if top == "trxrd":
            target = name.split(".")[1] if "." in name else None
            if target is not None:
                assert target in ALLOWED.get(layer, set()), f"{layer} must not import trxrd.{target}"
        elif top in GUI_LIBS:
            assert layer == "gui", f"{path.name}: only trxrd.gui may import {top}"
        elif top in {"cedapp", "Bibli_python", "dill"}:
            assert path.name == "legacy.py", f"{path.name}: only io/legacy.py may touch the old code"
        elif layer in THIRD_PARTY and top not in __import__("sys").stdlib_module_names:
            assert top in THIRD_PARTY[layer], f"{layer} must stay pure, found {top}"
