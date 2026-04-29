"""P0b sanity check: audit Finnhub /stock/peers matches for the
calibration manifest.

Background: peer_aggregator computes industry_median ROIC across the
top-5 Finnhub peers. If Finnhub returns peers in a different industry
than the focal ticker, the median is meaningless (e.g., a software
focal getting a peer set of "Industrial" companies). This script
flags such mismatches so they can be reviewed before the next
calibration v2 run.

Method:
  1. Read the calibration manifest.
  2. For each US / US-ADR ticker, fetch profile (focal industry)
     + peers (top-5).
  3. For each peer, fetch profile (peer industry).
  4. Flag a peer as "suspicious" if its industry differs from the
     focal industry. Empty / unknown industries are also flagged.
  5. Write `data/calibration/peer_audit_2026-04.yaml` with the
     focal/peer industry table per ticker + suspicious-count summary.

Korean tickers (peer aggregation not implemented) and SPY (ETF
self-exclusion) are skipped.

API budget: ~25 tickers × ~6 calls = ~150 calls. Finnhub free tier is
60/min, so a 1.1s sleep keeps us under the rate cap.

Run:
  uv run python scripts/probe_peer_audit.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.data.finnhub import FinnhubClient, FinnhubError  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "data" / "calibration" / "manifest.yaml"
OUTPUT_PATH = REPO_ROOT / "data" / "calibration" / "peer_audit_2026-04.yaml"

# Skip these markets (peer aggregation N/A).
SKIP_MARKETS = {"KR"}
# Skip these symbols (ETFs / known coverage gaps).
SKIP_SYMBOLS = {"SPY"}

PEER_LIMIT = 5  # mirrors peer_aggregator.DEFAULT_PEER_LIMIT
SLEEP_SEC = 1.1  # under 60/min Finnhub free-tier cap


def _load_manifest() -> list[dict]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("tickers", [])


def _safe_industry(client: FinnhubClient, symbol: str) -> str | None:
    """Return finnhub_industry or None on any error."""
    try:
        prof = client.profile(symbol)
        return prof.finnhub_industry
    except FinnhubError:
        return None
    except Exception as exc:  # network / pydantic / etc.
        print(f"  [warn] profile({symbol}) failed: {exc}", file=sys.stderr)
        return None


def _safe_peers(client: FinnhubClient, symbol: str) -> list[str]:
    try:
        return client.peers(symbol)[:PEER_LIMIT]
    except FinnhubError:
        return []
    except Exception as exc:
        print(f"  [warn] peers({symbol}) failed: {exc}", file=sys.stderr)
        return []


def main() -> int:
    tickers = _load_manifest()
    targets = [
        t for t in tickers
        if t.get("market") not in SKIP_MARKETS
        and t.get("symbol") not in SKIP_SYMBOLS
    ]
    print(f"Auditing {len(targets)} tickers (skipped {len(tickers) - len(targets)})")

    client = FinnhubClient()
    audit: list[dict] = []
    suspicious_total = 0

    for i, t in enumerate(targets, 1):
        symbol = t["symbol"]
        market = t.get("market", "?")
        print(f"[{i}/{len(targets)}] {symbol} ({market})")

        focal_industry = _safe_industry(client, symbol)
        time.sleep(SLEEP_SEC)
        peers = _safe_peers(client, symbol)
        time.sleep(SLEEP_SEC)

        if not peers:
            print("  no peers returned")
            audit.append({
                "symbol": symbol,
                "focal_industry": focal_industry,
                "peers": [],
                "suspicious": [],
                "status": "no_peers_returned",
            })
            continue

        peer_rows: list[dict] = []
        suspicious: list[str] = []
        for peer in peers:
            if peer.upper() == symbol.upper():
                continue  # self
            peer_industry = _safe_industry(client, peer)
            time.sleep(SLEEP_SEC)
            mismatch = (
                focal_industry is None
                or peer_industry is None
                or peer_industry != focal_industry
            )
            peer_rows.append({
                "symbol": peer,
                "industry": peer_industry,
                "mismatch": mismatch,
            })
            if mismatch:
                suspicious.append(peer)
                suspicious_total += 1

        audit.append({
            "symbol": symbol,
            "focal_industry": focal_industry,
            "peers": peer_rows,
            "suspicious": suspicious,
            "status": (
                "ok" if not suspicious
                else f"{len(suspicious)}/{len(peer_rows)}_mismatched"
            ),
        })

    out = {
        "metadata": {
            "audit_date": "2026-04-29",
            "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
            "peer_limit": PEER_LIMIT,
            "skipped_markets": sorted(SKIP_MARKETS),
            "skipped_symbols": sorted(SKIP_SYMBOLS),
            "tickers_audited": len(targets),
            "tickers_with_suspicious_peers": sum(
                1 for r in audit if r.get("suspicious")
            ),
            "total_suspicious_peer_count": suspicious_total,
            "rule": (
                "peer flagged 'mismatch' when its finnhub_industry differs "
                "from focal's, or either is missing — manual review needed "
                "to decide if Finnhub's peer list is acceptable for "
                "industry_median ROIC computation"
            ),
        },
        "results": audit,
    }
    OUTPUT_PATH.write_text(
        yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print()
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(
        f"Summary: {out['metadata']['tickers_with_suspicious_peers']}/"
        f"{len(targets)} tickers have ≥1 suspicious peer "
        f"(total {suspicious_total} flagged)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
