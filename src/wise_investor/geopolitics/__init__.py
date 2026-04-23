"""Phase 3E geopolitical / event-based macro context.

Two public data sources, both free and key-less:

  gdelt.py         GDELT Project's DOC 2.0 API — structured event and article
                   retrieval over every major global news source, indexed by
                   GKG 2.0 themes (ECON_TRADE_SANCTIONS, TRADE_WAR, etc.)

  google_news.py   Google News RSS — keyword-based headline feed, broader
                   but less structured than GDELT

snapshot.py assembles a per-symbol `GeopoliticsSnapshot` that the Economist
agent can read alongside the FRED macro snapshot. Integration into
pre_gather_facts is intentionally deferred to Phase 3E-2 so we can
validate data quality via `scripts/probe_geopolitics.py` first.
"""
