"""Reader for the .jcpds phase cards used as phase library."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

from trxrd.domain import PhaseCard, Reflection

PathLike = Union[str, Path]


def _range(pattern: str, text: str) -> Optional[Tuple[float, float]]:
    match = re.search(pattern + r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", text)
    return (float(match.group(1)), float(match.group(2))) if match else None


def _int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern + r"\s*=\s*(\d+)", text)
    return int(match.group(1)) if match else None


def name_from_comment(comment: str) -> str:
    """'H2O Ice VII /T[100,1000]/P[-1,70]' -> 'H2O_Ice_VII' (legacy naming kept)."""
    name = comment.split("/")[0].split(",")[0].strip()
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def parse_jcpds(text: str, source_file: str = "") -> PhaseCard:
    values: Dict[str, str] = {}
    reflections = []
    comment = ""
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = (part.strip() for part in raw_line.split(":", 1))
        key = key.upper()
        if key == "DIHKL":
            parts = value.split()
            if len(parts) >= 5:
                d, intensity, h, k, l = (float(p) for p in parts[:5])
                reflections.append(Reflection(d, intensity, int(h), int(k), int(l)))
        elif key == "COMMENT" and not comment:
            comment = value
        else:
            values.setdefault(key, value)

    def num(*keys: str) -> Optional[float]:
        for key in keys:
            if key in values and values[key] != "":
                return float(values[key])
        return None

    name = name_from_comment(comment) if comment else Path(source_file).stem
    if "SYMMETRY" not in values or num("A") is None:
        raise ValueError(f"Incomplete jcpds card {source_file or name!r}: SYMMETRY and A are required")
    return PhaseCard(
        name=name,
        symmetry=values["SYMMETRY"],
        a=num("A"),
        b=num("B"),
        c=num("C"),
        alpha=num("ALPHA") or 90.0,
        beta=num("BETA") or 90.0,
        gamma=num("GAMMA") or 90.0,
        v0=num("VOLUME", "V0"),
        k0=num("K0") or 0.0,
        k0p=num("K0P") if num("K0P") is not None else 4.0,
        alpha_t=num("ALPHAT"),
        reflections=reflections,
        t_range=_range(r"/T", comment),
        p_range=_range(r"/P", comment),
        v_min_ratio=num("VMIN") or 0.6,
        z_max=_int(r"/Z", comment),
        fu=_int(r"/fu", comment),
        source_text=text,
        source_file=str(source_file),
    )


def read_jcpds(path: PathLike) -> PhaseCard:
    path = Path(path)
    return parse_jcpds(path.read_text(encoding="utf-8", errors="replace"), source_file=str(path))


def read_library(paths: Iterable[PathLike]) -> Dict[str, PhaseCard]:
    """Read every card of the given files/folders, keyed by phase name."""
    cards: Dict[str, PhaseCard] = {}
    for entry in paths:
        entry = Path(entry)
        files = sorted(entry.glob("*.jcpds")) if entry.is_dir() else [entry]
        for file in files:
            card = read_jcpds(file)
            cards[card.name] = card
    return cards
