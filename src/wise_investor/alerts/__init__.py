"""Phase 3 chain alerts — design-v2.2 §5.1 "가장 강력한 기능".

When a node in the value chain graph (supplier, peer, infrastructure)
has a significant news event, every target ticker (Tier 1) whose
graph reaches that node within N hops gets an alert. This is what
makes the graph more than a static document — the Skeptic agent gets
real-time feed of "which of your attack vectors just became active".

Pipeline:

    1. Load value chain graph          (NetworkX, persisted JSON)
    2. Load geopolitics snapshot       (GDELT + Google News headlines)
    3. Match news items → graph nodes  (substring / alias match)
    4. For each match, find path to a target node (BFS within N hops)
    5. Compose alert message
    6. (Optional) push to Telegram
"""

from wise_investor.alerts.chain_alerts import (
    ChainAlert,
    NewsItemLike,
    compose_alert_markdown,
    find_matching_nodes,
    find_target_paths,
    scan_for_alerts,
)

__all__ = [
    "ChainAlert",
    "NewsItemLike",
    "compose_alert_markdown",
    "find_matching_nodes",
    "find_target_paths",
    "scan_for_alerts",
]
