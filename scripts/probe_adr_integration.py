"""End-to-end probe: live_adapter.fetch_live_fundamentals_us on ADRs.

Confirms P1b integration: when Finnhub returns 0 entries the EDGAR
fallback kicks in transparently and we get a populated TickerFundamentals
with multi-year annual data and industry_classification.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.screening.live_adapter import fetch_live_fundamentals_us  # noqa: E402

ADRS = ["TSM", "ASML", "NVO"]


def main() -> int:
    for symbol in ADRS:
        print(f"=== {symbol} ===")
        try:
            funds = fetch_live_fundamentals_us(symbol)
        except Exception as e:
            print(f"  FAILED: {e}\n")
            continue

        print(f"  industry: {funds.industry_classification}")
        print(f"  annual count: {len(funds.annual)}")
        if funds.annual:
            print("  Recent annuals (most recent last):")
            for ann in funds.annual[-5:]:
                ic = ann.invested_capital
                nopat = ann.nopat
                roic = (nopat / ic) if (nopat is not None and ic is not None and ic > 0) else None
                roic_str = f"{roic:.3%}" if roic is not None else "N/A"
                print(
                    f"    FY{ann.fiscal_year}: NOPAT={nopat}, "
                    f"IC={ic}, ROIC={roic_str}"
                )
        print(f"  quarterly margins: {len(funds.quarterly_margins)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
