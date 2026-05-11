"""Application configuration loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = "*"

    PROVIDER_ORDER: str = (
        "gemini,groq,openrouter,mistral,cohere,huggingface,claude,openai,grok"
    )
    DEFAULT_COOLDOWN_SECONDS: int = 60
    REQUEST_TIMEOUT: int = 120

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_VISION_MODEL: str = "gemini-2.0-flash"

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"
    OPENROUTER_VISION_MODEL: str = "google/gemini-2.0-flash-exp:free"
    OPENROUTER_MODELS: str = ""
    OPENROUTER_VISION_MODELS: str = ""

    @property
    def openrouter_text_models(self) -> List[str]:
        items = [m.strip() for m in self.OPENROUTER_MODELS.split(",") if m.strip()]
        return items or [self.OPENROUTER_MODEL]

    @property
    def openrouter_vision_models(self) -> List[str]:
        items = [m.strip() for m in self.OPENROUTER_VISION_MODELS.split(",") if m.strip()]
        return items or [self.OPENROUTER_VISION_MODEL or self.OPENROUTER_MODEL]

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_VISION_MODEL: str = "llama-3.2-90b-vision-preview"

    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL: str = "mistral-small-latest"
    MISTRAL_VISION_MODEL: str = "pixtral-12b-2409"

    COHERE_API_KEY: str = ""
    COHERE_MODEL: str = "command-r-plus"

    HUGGINGFACE_API_KEY: str = ""
    HUGGINGFACE_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct"

    CLAUDE_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    GROK_API_KEY: str = ""
    GROK_MODEL: str = "grok-2-latest"
    GROK_VISION_MODEL: str = "grok-2-vision-latest"

    @property
    def provider_order_list(self) -> List[str]:
        return [p.strip().lower() for p in self.PROVIDER_ORDER.split(",") if p.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
