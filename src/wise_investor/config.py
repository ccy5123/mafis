"""Centralized configuration loaded from environment variables.

See design-v2.2.md §7 for the "LLM is judgment, Python is calculation"
principle. The `llm_temperature` and `llm_seed` fields below are
retained for backwards compatibility — they no longer drive the
default sampling path (Phase 5 routes everything through
`config/agent_models.yaml`), but tests and legacy deployments still
read them, and users who explicitly want a deterministic profile
can re-introduce them via the YAML's `sampling:` block.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Primary data source (Phase 1B migration): Finnhub.
    # FMP retained for legacy/optional use only; prefer Finnhub going forward.
    finnhub_api_key: str = Field(default="", description="Finnhub API key (primary)")
    fmp_api_key: str = Field(default="", description="Financial Modeling Prep API key (legacy/optional)")
    fred_api_key: str = Field(
        default="", description="FRED API key (macro context for Economist agent)"
    )
    dart_api_key: str = Field(
        default="",
        description=(
            "OpenDART API key (Korean financials, opendart.fss.or.kr). "
            "Register for free at opendart.fss.or.kr/mngInfo/mngInfoMain.do"
        ),
    )

    # Telegram notification (Phase 3B) — optional. If either is empty
    # the notifier no-ops silently so developers running without a bot
    # get no spurious errors.
    telegram_bot_token: str = Field(
        default="", description="Telegram bot token from @BotFather"
    )
    telegram_chat_id: str = Field(
        default="", description="Telegram chat_id (yours, or a group's)"
    )

    # User-facing language for the Telegram summary + attached .md
    # report. Supported: ko / en / ja / zh. The summary renderer uses
    # a deterministic locale pack; the attached .md is translated via
    # the Ollama-based translator (wise_investor.translation).
    # Invalid values silently fall back to the Korean renderer so a
    # typo in .env doesn't break a crew run.
    user_language: str = Field(
        default="ko",
        description="User-facing language (ko/en/ja/zh). Default ko.",
    )

    ollama_host: str = Field(default="http://localhost:11434")
    analyst_model: str = Field(default="llama3.1:8b")
    valuer_model: str = Field(default="llama3.1:8b")
    skeptic_model: str = Field(default="qwen2.5:7b")
    steward_model: str = Field(default="qwen2.5:7b")

    # Phase 5: these fields are retained for backwards compatibility
    # only. The active sampling path goes through config/agent_models.yaml
    # via wise_investor.llm.config; see docs/llm_backends.md for how
    # to re-enable deterministic output (per-agent override there).
    llm_temperature: float = Field(
        default=0.0,
        description=(
            "Legacy / unused on the default path. Per-agent sampling "
            "now resolves via config/agent_models.yaml + the model-"
            "family recommendation. Override there for deterministic mode."
        ),
    )
    llm_seed: int = Field(
        default=42,
        description="Legacy / unused on the default path. See llm_temperature.",
    )

    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma")
    sqlite_path: Path = Field(default=PROJECT_ROOT / "data" / "portfolio.sqlite")


settings = Settings()
