"""Dependency rules between layers and with third-party packages (see docs/ARCHITECTURE.md)."""

import ast
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "trxrd"

# layer -> internal layers it may import
ALLOWED = {
    "physics": {"physics"},
    "domain": {"domain", "physics"},
    "io": {"io", "domain", "physics"},
    "processing": {"processing", "domain", "physics"},
    "services": {"services", "io", "processing", "domain", "physics"},
    "notebook": {"io", "domain", "physics"},
    "cli": {"services", "io", "processing", "domain", "physics", "notebook"},
    "gui": {"gui", "services", "domain", "physics"},
}
# layer -> third-party packages it may use (None: any declared dependency)
PURE = {"physics": {"numpy", "scipy"}, "domain": {"numpy"}}
GUI_ONLY = {"PySide6", "pyqtgraph"}
# import name -> distribution name in pyproject.toml
DISTRIBUTIONS = {"pyFAI": "pyfai", "PySide6": "pyside6"}


def modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        layer = path.relative_to(PACKAGE).parts[0].removesuffix(".py")
        if layer != "__init__":
            yield path, layer


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def third_party(name):
    top = name.split(".")[0]
    return None if top == "trxrd" or top in sys.stdlib_module_names else top


def declared():
    tomllib = pytest.importorskip("tomllib")  # Python >= 3.11
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    def names(requirements):
        return {r.split(">")[0].split("=")[0].split("<")[0].strip().lower() for r in requirements}
    extras = project.get("optional-dependencies", {})
    return names(project["dependencies"]), names(extras.get("notebook", [])), names(extras.get("gui", []))


@pytest.mark.parametrize("path,layer", list(modules()), ids=lambda v: str(v).split("trxrd")[-1])
def test_layer_dependencies(path, layer):
    for name in imports(path):
        parts = name.split(".")
        if parts[0] == "trxrd" and len(parts) > 1:
            assert parts[1] in ALLOWED[layer], f"{layer} must not import trxrd.{parts[1]}"
        top = third_party(name)
        if top in GUI_ONLY:
            assert layer == "gui", f"{path.name}: only trxrd.gui may import {top}"
        if top and layer in PURE:
            assert top in PURE[layer], f"{layer} must stay pure, found {top}"


def test_third_party_imports_match_pyproject():
    core, notebook, gui = declared()
    used = defaultdict(set)
    for path, layer in modules():
        for name in imports(path):
            top = third_party(name)
            if top:
                used[DISTRIBUTIONS.get(top, top).lower()].add(path.name)
    undeclared = set(used) - core - notebook - gui
    assert not undeclared, f"imported but not declared in pyproject.toml: {undeclared}"
    unused = core - set(used)
    assert not unused, f"declared but never imported: {unused}"
    for dist in notebook:
        assert used.get(dist, set()) <= {"notebook.py"}, f"{dist} is optional: only notebook.py may import it"
