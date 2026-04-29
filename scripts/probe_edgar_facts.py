"""End-to-end probe: fetch_financials_via_edgar on the 3 ADRs."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.data.edgar_facts import fetch_financials_via_edgar  # noqa: E402
from wise_investor.data.finnhub import extract_field  # noqa: E402

ADRS = ["TSM", "ASML", "NVO"]


def main() -> int:
    for symbol in ADRS:
        print(f"=== {symbol} ===")
        try:
            resp = fetch_financials_via_edgar(symbol)
        except Exception as e:
            print(f"  FAILED: {e}\n")
            continue
        print(f"  cik={resp.cik}, n_entries={len(resp.data)}")
        for entry in resp.data:
            rev = extract_field(entry, "revenue")
            gp = extract_field(entry, "gross_profit")
            opi = extract_field(entry, "operating_income")
            ta = extract_field(entry, "total_assets")
            cash = extract_field(entry, "cash_and_cash_equivalents")
            ic = ta - (cash or 0.0) if ta is not None else None
            roic = None
            if opi is not None and ic is not None and ic > 0:
                nopat = opi * (1.0 - 0.21)
                roic = nopat / ic
            print(
                f"  FY{entry.year} ({entry.form} filed {entry.filed_date}): "
                f"rev={rev}, gp={gp}, opi={opi}, ta={ta}, cash={cash}, "
                f"IC={ic}, ROIC={roic:.3%}" if roic is not None else
                f"  FY{entry.year} ({entry.form} filed {entry.filed_date}): "
                f"rev={rev}, gp={gp}, opi={opi}, ta={ta}, cash={cash}, "
                f"IC={ic}, ROIC=N/A"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
