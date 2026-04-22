"""External data source clients.

Phase 1A:
- fmp: Financial Modeling Prep client (primary financial data)
- yf: yfinance wrapper (backup / cross-validation)
- edgar: SEC EDGAR text section extraction (Risk Factors, MD&A)

Phase 3: EDGAR XBRL parsing for numerical ground-truth (design-v2.2 §3.2).
"""
