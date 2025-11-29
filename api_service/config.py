# api_service/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OpenAI / Azure OpenAI
    openai_api_key: str
    openai_api_base: str | None = None
    openai_model: str = "gpt-4.1-mini"

    cohere_api_key: str | None = None # Set to None for optional use

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # App
    app_env: str = "local"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings instance so we don't re-read env on every request.
    """
    return Settings() # type: ignore[arg-type]