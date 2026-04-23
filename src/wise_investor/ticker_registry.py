"""3-Tier ticker registry (design-v2.2 §6).

Loads `config/tickers.yaml` and exposes structured queries by tier. The
registry is read-only at runtime; promotions (Tier 2 → Tier 1) are
performed by editing the YAML file and re-running.

Validates:
- Tier 1 entries exist and each has a matching value-chain brief at
  docs/value_chains/<SYMBOL>.md (the full crew needs it).
- No ticker appears in more than one tier.
- Ticker strings are uppercase; the loader silently uppercases to be
  tolerant of hand-edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TICKERS_PATH = REPO_ROOT / "config" / "tickers.yaml"
DEFAULT_VALUE_CHAINS_DIR = REPO_ROOT / "docs" / "value_chains"


Tier = Literal["tier_1", "tier_2", "tier_3"]


@dataclass
class TickerEntry:
    symbol: str
    tier: Tier
    notes: str | None = None


@dataclass
class TickerRegistry:
    """In-memory representation of the tickers.yaml contents."""

    entries: list[TickerEntry]

    def by_tier(self, tier: Tier) -> list[TickerEntry]:
        return [e for e in self.entries if e.tier == tier]

    def symbols_by_tier(self, tier: Tier) -> list[str]:
        return [e.symbol for e in self.by_tier(tier)]

    def find(self, symbol: str) -> TickerEntry | None:
        sym = symbol.upper()
        for e in self.entries:
            if e.symbol == sym:
                return e
        return None


class RegistryError(RuntimeError):
    """Raised when the registry fails validation."""


def _parse_entry(raw: object, tier: Tier) -> TickerEntry:
    """Accept either a bare 'SYM' string or a `{symbol, notes}` mapping."""
    if isinstance(raw, str):
        return TickerEntry(symbol=raw.upper().strip(), tier=tier, notes=None)
    if isinstance(raw, dict):
        sym = raw.get("symbol")
        if not isinstance(sym, str) or not sym.strip():
            raise RegistryError(
                f"entry in {tier} missing a 'symbol' string: {raw!r}"
            )
        notes = raw.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise RegistryError(
                f"'notes' for {sym} must be a string if present: {notes!r}"
            )
        return TickerEntry(symbol=sym.upper().strip(), tier=tier, notes=notes)
    raise RegistryError(
        f"entry in {tier} is neither a string nor a mapping: {raw!r}"
    )


def load_registry(
    path: Path | str | None = None,
    value_chains_dir: Path | None = None,
    strict: bool = True,
) -> TickerRegistry:
    """Load and validate the ticker registry.

    `strict` turns value-chain-missing into a hard error (default). Set
    to False for tests or dry-runs where the full Tier 1 brief set is
    not required.
    """
    p = Path(path) if path is not None else DEFAULT_TICKERS_PATH
    if not p.exists():
        raise RegistryError(f"tickers file not found: {p}")

    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise RegistryError(
            f"{p} must have a top-level mapping; got {type(data).__name__}"
        )

    entries: list[TickerEntry] = []
    seen: set[str] = set()
    for tier in ("tier_1", "tier_2", "tier_3"):
        raw_list = data.get(tier, []) or []
        if not isinstance(raw_list, list):
            raise RegistryError(f"'{tier}' in {p} must be a list; got {type(raw_list).__name__}")
        for raw in raw_list:
            entry = _parse_entry(raw, tier)  # type: ignore[arg-type]
            if entry.symbol in seen:
                raise RegistryError(
                    f"duplicate ticker across tiers: {entry.symbol}"
                )
            seen.add(entry.symbol)
            entries.append(entry)

    registry = TickerRegistry(entries=entries)

    # Validate Tier 1 coverage: every Tier 1 ticker needs a value chain brief.
    if strict:
        vc_dir = value_chains_dir if value_chains_dir is not None else DEFAULT_VALUE_CHAINS_DIR
        missing: list[str] = []
        for entry in registry.by_tier("tier_1"):
            brief = vc_dir / f"{entry.symbol}.md"
            if not brief.exists():
                missing.append(entry.symbol)
        if missing:
            raise RegistryError(
                "Tier 1 tickers without a value chain brief "
                f"(docs/value_chains/<SYMBOL>.md): {missing}"
            )

    return registry
