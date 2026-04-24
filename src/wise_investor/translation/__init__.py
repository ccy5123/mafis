"""Report translation package.

Provides `translate_report()` for rendering English crew-report
markdown into the user's preferred language (ko/ja/zh). English
(`en`) short-circuits to a no-op.
"""

from wise_investor.translation.translator import translate_report

__all__ = ["translate_report"]
