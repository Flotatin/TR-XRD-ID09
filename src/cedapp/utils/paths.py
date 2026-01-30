"""Centralised path resolution for CED resources."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RESOURCES_DIR = _REPO_ROOT / "resources"


def _env_path(var_name: str) -> Path | None:
    value = os.getenv(var_name)
    if not value:
        return None
    return Path(value).expanduser()


def _ensure_path(path: Path, label: str, env_var: str | None, require: bool) -> Path:
    if require and not path.exists():
        hint = f"Set {env_var} to override." if env_var else ""
        raise FileNotFoundError(f"{label} not found at '{path}'. {hint}".strip())
    return path


def get_resources_dir(require: bool = True) -> Path:
    """Return the base resources directory."""

    path = _env_path("CEDAPP_RESOURCES_DIR") or _DEFAULT_RESOURCES_DIR
    return _ensure_path(path, "Resources directory", "CEDAPP_RESOURCES_DIR", require)


def get_config_dir(require: bool = True) -> Path:
    """Return the configuration directory."""

    path = _env_path("CEDAPP_CONFIG_DIR") or get_resources_dir(require=require) / "config"
    return _ensure_path(path, "Config directory", "CEDAPP_CONFIG_DIR", require)


def get_text_dir(require: bool = True) -> Path:
    """Return the text/help resources directory."""

    path = _env_path("CEDAPP_TEXT_DIR") or get_resources_dir(require=require) / "text"
    return _ensure_path(path, "Text resources directory", "CEDAPP_TEXT_DIR", require)


def get_bibdrx_dir(require: bool = True) -> Path:
    """Return the BibDRX library directory."""

    path = _env_path("CEDAPP_BIBDRX_DIR") or get_resources_dir(require=require) / "bibdrx"
    return _ensure_path(path, "BibDRX directory", "CEDAPP_BIBDRX_DIR", require)


def get_default_config_path() -> Path:
    """Return the default configuration file path."""

    return get_config_dir(require=False) / "config_21012025.txt"


def resolve_config_path(config_path: str | Path) -> Path:
    """Resolve a config path relative to the config directory when needed."""

    candidate = Path(config_path).expanduser()
    if candidate.is_absolute():
        return candidate
    config_dir = get_config_dir(require=False)
    parts = candidate.parts
    if parts and parts[0] == config_dir.name:
        candidate = Path(*parts[1:])
    return config_dir / candidate


def _strip_bibdrx_prefix(path: Path) -> Path | None:
    parts = [part.lower() for part in path.parts]
    if "bibdrx" in parts:
        index = parts.index("bibdrx")
        return Path(*path.parts[index + 1 :])
    return None


def resolve_bibdrx_paths(entries: Iterable[str]) -> List[str]:
    """Resolve a list of BibDRX file paths, handling legacy absolute paths."""

    bib_dir = get_bibdrx_dir(require=False)
    resolved: List[str] = []

    for entry in entries:
        if not entry:
            continue
        raw_path = Path(entry).expanduser()
        if raw_path.is_absolute() and raw_path.exists():
            resolved.append(str(raw_path))
            continue
        stripped = _strip_bibdrx_prefix(raw_path)
        if stripped is not None:
            candidate = bib_dir / stripped
            resolved.append(str(candidate))
            continue
        if not raw_path.is_absolute():
            candidate = bib_dir / raw_path
            resolved.append(str(candidate))
            continue
        resolved.append(str(raw_path))

    return resolved
