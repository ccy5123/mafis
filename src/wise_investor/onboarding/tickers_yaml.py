"""Read/write helpers for config/tickers.yaml (the 3-Tier registry).

The file is a simple tier_1 / tier_2 / tier_3 mapping of `symbol + notes`
entries. This module supports:

    load_registry_yaml(path) -> dict                # preserves tier order
    add_ticker_to_registry(path, symbol, tier, notes)
    ticker_in_registry(data, symbol) -> int | None  # returns the tier

The file is small (under 200 lines in practice) so round-tripping with
`ruamel.yaml` would be overkill — we use PyYAML and preserve the
leading-comment block manually.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_TIERS = (1, 2, 3)


@dataclass
class TickerEntry:
    symbol: str
    tier: int
    notes: str = ""


class RegistryError(RuntimeError):
    pass


def load_registry_yaml(path: Path | str) -> dict[str, Any]:
    """Return the parsed YAML mapping. Missing file → empty registry."""
    p = Path(path)
    if not p.exists():
        return {"tier_1": [], "tier_2": [], "tier_3": []}
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    # Normalize keys to the three tiers we expect.
    for key in ("tier_1", "tier_2", "tier_3"):
        data.setdefault(key, [])
    return data


def _extract_leading_comments(text: str) -> str:
    """Return the leading comment block (and any blank lines up to the
    first key) so we can preserve it on save.
    """
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped == "":
            lines.append(line)
            continue
        break
    return "\n".join(lines) + ("\n" if lines else "")


def ticker_in_registry(data: dict[str, Any], symbol: str) -> int | None:
    """Return the tier number if the symbol exists anywhere in the
    registry, else None.
    """
    sym = symbol.upper()
    for key in ("tier_1", "tier_2", "tier_3"):
        for entry in data.get(key, []) or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("symbol", "")).upper() == sym:
                return int(key.split("_")[1])
    return None


def add_ticker_to_registry(
    path: Path | str,
    symbol: str,
    tier: int,
    notes: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Insert or update a ticker entry in config/tickers.yaml.

    If the symbol already exists in a different tier and overwrite is
    False, raises RegistryError. Pass overwrite=True to move tiers.

    Returns the updated registry dict.
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"tier must be one of {VALID_TIERS}, got {tier}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Preserve any leading comment banner the file already has.
    leading = ""
    if p.exists():
        leading = _extract_leading_comments(p.read_text(encoding="utf-8"))
    data = load_registry_yaml(p)

    sym = symbol.upper()
    existing_tier = ticker_in_registry(data, sym)
    if existing_tier is not None and existing_tier != tier and not overwrite:
        raise RegistryError(
            f"{sym} already registered in tier_{existing_tier}; pass "
            f"overwrite=True to move it to tier_{tier}."
        )

    # Remove any prior occurrences (under the assumption that overwrite
    # semantics mean "canonicalize to exactly this tier").
    for key in ("tier_1", "tier_2", "tier_3"):
        data[key] = [
            e
            for e in (data.get(key) or [])
            if isinstance(e, dict)
            and str(e.get("symbol", "")).upper() != sym
        ]

    tier_key = f"tier_{tier}"
    entry = {"symbol": sym}
    if notes:
        entry["notes"] = notes
    data[tier_key] = list(data.get(tier_key) or [])
    data[tier_key].append(entry)

    # YAML dump — keep tier order stable.
    ordered = {
        "tier_1": data["tier_1"],
        "tier_2": data["tier_2"],
        "tier_3": data["tier_3"],
    }
    body = yaml.safe_dump(
        ordered,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    p.write_text(leading + body if leading else body, encoding="utf-8")
    return data


__all__ = [
    "RegistryError",
    "TickerEntry",
    "VALID_TIERS",
    "add_ticker_to_registry",
    "load_registry_yaml",
    "ticker_in_registry",
]
