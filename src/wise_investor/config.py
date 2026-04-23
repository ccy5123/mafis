"""Centralized configuration loaded from environment variables.

See design-v2.2.md §7 for the "LLM is judgment, Python is calculation" principle
and the re-review Critical #1 (reproducibility) that motivates temperature=0 + fixed seed.
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

    # Telegram notification (Phase 3B) — optional. If either is empty
    # the notifier no-ops silently so developers running without a bot
    # get no spurious errors.
    telegram_bot_token: str = Field(
        default="", description="Telegram bot token from @BotFather"
    )
    telegram_chat_id: str = Field(
        default="", description="Telegram chat_id (yours, or a group's)"
    )

    ollama_host: str = Field(default="http://localhost:11434")
    analyst_model: str = Field(default="llama3.1:8b")
    valuer_model: str = Field(default="llama3.1:8b")
    skeptic_model: str = Field(default="qwen2.5:7b")
    steward_model: str = Field(default="qwen2.5:7b")

    llm_temperature: float = Field(
        default=0.0,
        description="Fixed at 0 for reproducibility. See design-v2.2 re-review Critical #1.",
    )
    llm_seed: int = Field(default=42, description="Fixed seed for LLM reproducibility.")

    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma")
    sqlite_path: Path = Field(default=PROJECT_ROOT / "data" / "portfolio.sqlite")


settings = Settings()
