"""OpenRouter provider - rotates through a list of free models on failure."""
from __future__ import annotations

from typing import List, Optional

from ..media import ProcessedMedia
from ..schemas import ChatMessage
from ..utils.errors import (
    AuthError,
    BadRequestError,
    ProviderError,
    ProviderUnavailable,
    RateLimitError,
)
from ..utils.logger import logger
from .base import BaseProvider
from .openai_compat import call_openai_compat


class OpenRouterProvider(BaseProvider):
    """OpenRouter has dozens of free models that share quotas independently.

    Instead of a single model, this provider walks through a configured list
    and tries the next one on rate-limit / not-found / 5xx errors. Only when
    all models in the list are exhausted does it raise upward so the
    pipeline can fall back to a different provider.
    """

    name = "openrouter"
    supports_vision = True

    @property
    def api_key(self) -> str:
        return self.settings.OPENROUTER_API_KEY

    @property
    def text_models(self) -> List[str]:
        return self.settings.openrouter_text_models

    @property
    def vision_models(self) -> List[str]:
        return self.settings.openrouter_vision_models

    @property
    def model(self) -> str:
        models = self.text_models
        return models[0] if models else ""

    async def generate(
        self,
        prompt: str,
        media: ProcessedMedia,
        history: Optional[List[ChatMessage]] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        use_vision = media.has_images
        models = self.vision_models if use_vision else self.text_models
        if not models:
            raise ProviderError(f"{self.name}: no models configured")

        last_exc: Optional[Exception] = None
        for idx, model in enumerate(models, start=1):
            try:
                logger.info(
                    f"[{self.name}] attempt {idx}/{len(models)} - model={model}"
                )
                return await call_openai_compat(
                    self,
                    base_url="https://openrouter.ai/api/v1/chat/completions",
                    model=model,
                    api_key=self.api_key,
                    prompt=prompt,
                    media=media,
                    history=history,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_vision=use_vision,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/multillm-backend",
                        "X-Title": "Multi-LLM Backend",
                    },
                )
            except AuthError:
                raise
            except (RateLimitError, BadRequestError, ProviderUnavailable, ProviderError) as e:
                last_exc = e
                logger.warning(
                    f"[{self.name}] model {model} failed "
                    f"({type(e).__name__}): {str(e)[:200]}"
                )
                continue

        raise last_exc or ProviderError(
            f"{self.name}: all {len(models)} models exhausted"
        )
