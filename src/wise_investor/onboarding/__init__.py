"""Phase 2 automatic ticker onboarding (design-v2.2 §5.1 Phase 2).

Closes the #1 pain point from the Phase 1 MVP: adding a new ticker
required writing a 60-line value chain brief by hand before the crew
could run. This package auto-drafts that brief from Finnhub profile +
peers + the indexed 10-K Business/Risk-Factors sections, then
registers the ticker in config/tickers.yaml.

Output is a `<SYMBOL>.draft.md` file. The human reviews (2-3 min for
the Vulnerable Links section especially) and renames to `<SYMBOL>.md`
to activate. A `.draft.md` file is NOT picked up by the crew — only
`<SYMBOL>.md`.
"""

from wise_investor.onboarding.brief_generator import (
    RawMaterial,
    build_brief_prompt,
    gather_raw_material,
    generate_value_chain_draft,
)
from wise_investor.onboarding.tickers_yaml import (
    TickerEntry,
    add_ticker_to_registry,
    load_registry_yaml,
)

__all__ = [
    "RawMaterial",
    "TickerEntry",
    "add_ticker_to_registry",
    "build_brief_prompt",
    "gather_raw_material",
    "generate_value_chain_draft",
    "load_registry_yaml",
]
