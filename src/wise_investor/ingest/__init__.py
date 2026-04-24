"""Tip ingestion package — Telegram polling + user-submitted tips.

Design doc: design-v2.2.md §3.1 (originally labelled "OpenClaw" after
the conceptual role, implemented here as a first-party Python layer).

Flow:
  1. User forwards a message from a stock group chat into the bot.
  2. Telegram long-poll (`TelegramReceiver.poll_updates`) delivers it.
  3. `intent_parser` classifies command vs free-text tip.
  4. `ticker_extractor` resolves Korean names → tickers.
  5. `TipStore` persists the tip in the shared portfolio.sqlite.
  6. Phase 2: `data.tip_feed` surfaces recent tips to the Analyst
     context during the next crew run for matching tickers.
"""
