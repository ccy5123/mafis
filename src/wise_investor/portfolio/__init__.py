"""Phase 2 portfolio state — SQLite-backed position ledger.

Design v2.2 §5.3 specifies a local SQLite store for Tier 1 holdings so
the Steward's percent-of-portfolio sizing recommendations can be
compared against actual current weights. Phase 1 ran without this
(Steward produced a sizing band and the human mentally tracked fill);
Phase 2 closes the loop so the crew can say "you're already at 4.2%;
the BUY signal recommends 3-5%, so no action needed".

Minimal schema: one positions table keyed on symbol, holding cash cost
and current shares. Market value + weight are derived on-the-fly from
the latest quote rather than stored, so the store doesn't age.
"""

from wise_investor.portfolio.store import (
    PortfolioStore,
    Position,
    WeightSnapshot,
)

__all__ = ["PortfolioStore", "Position", "WeightSnapshot"]
