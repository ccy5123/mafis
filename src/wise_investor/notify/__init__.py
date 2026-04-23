"""Phase 3B notification package.

Turns English crew reports into brief Korean push messages and sends them
via Telegram. This is the one place in the code where English -> Korean
translation happens (memory/feedback_language_strategy.md); the rest of
the pipeline stays in English for LLM accuracy.
"""
