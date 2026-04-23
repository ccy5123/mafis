"""Probe Finnhub for Korean-exchange ticker formats.

Tries every plausible symbol format for Samsung Electronics to see which
(if any) returns real data on our free tier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import settings  # noqa: E402


CANDIDATES = [
    # Samsung Electronics in various formats
    "005930",
    "005930.KS",
    "005930.KRX",
    "KRX:005930",
    "KOSPI:005930",
    "SSNLF",  # OTC ADR
    # Hyundai Motor
    "005380.KS",
    "005380",
    # SK Hynix
    "000660.KS",
    "000660",
    # Naver
    "035420.KS",
    "035420",
]


def probe(symbol: str) -> None:
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol, "token": settings.finnhub_api_key}
    try:
        r = httpx.get(url, params=params, timeout=5.0)
    except Exception as e:
        print(f"  {symbol:20s} ERR {e}")
        return
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:80]
    if isinstance(body, dict) and body.get("c") not in (0, None):
        print(f"  {symbol:20s} status={r.status_code}  [green] c={body['c']}")
    elif isinstance(body, dict) and body.get("c") == 0:
        print(f"  {symbol:20s} status={r.status_code}  [empty] c=0 → symbol not recognized")
    else:
        print(f"  {symbol:20s} status={r.status_code}  body={body}")


print("Finnhub /quote probe for Korean tickers:")
for s in CANDIDATES:
    probe(s)

# Also try search to see if Finnhub knows Samsung
print("\nFinnhub /search?q=samsung electronics:")
r = httpx.get(
    "https://finnhub.io/api/v1/search",
    params={"q": "samsung electronics", "token": settings.finnhub_api_key},
    timeout=5.0,
)
data = r.json() if r.status_code == 200 else {"error": r.text}
if isinstance(data, dict) and "result" in data:
    for row in data["result"][:10]:
        print(f"  {row.get('symbol', '?'):20s} {row.get('description', '')[:50]}  type={row.get('type', '')}")
else:
    print(f"  {data}")
