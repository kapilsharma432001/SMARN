from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SMARN"
    app_env: str = "local"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://smarn:smarn@localhost:5432/smarn"
    telegram_bot_token: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_llm_model: str = "gpt-4o-mini"
    openai_transcription_model: str = "gpt-4o-mini-transcribe"
    embedding_dimensions: int = Field(default=1536, ge=8)
    memory_search_limit: int = Field(default=5, ge=1, le=20)
    memory_relevance_max_distance: float = Field(default=0.75, ge=0.0, le=2.0)
    review_timezone: str = "Asia/Kolkata"


@lru_cache
def get_settings() -> Settings:
    return Settings()
