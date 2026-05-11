"""Provider registry: discovery + ordered lookup."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from ..config import get_settings
from ..utils.logger import logger
from .base import BaseProvider
from .claude import ClaudeProvider
from .cohere import CohereProvider
from .gemini import GeminiProvider
from .grok import GrokProvider
from .groq import GroqProvider
from .huggingface import HuggingFaceProvider
from .mistral import MistralProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider


PROVIDER_CLASSES: Dict[str, type[BaseProvider]] = {
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "groq": GroqProvider,
    "mistral": MistralProvider,
    "cohere": CohereProvider,
    "huggingface": HuggingFaceProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
    "grok": GrokProvider,
}


class ProviderRegistry:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._providers: Dict[str, BaseProvider] = {
            name: cls() for name, cls in PROVIDER_CLASSES.items()
        }
        configured = [p.name for p in self._providers.values() if p.is_configured()]
        if configured:
            logger.info(f"Configured providers: {', '.join(configured)}")
        else:
            logger.warning(
                "No providers are configured! Set at least one *_API_KEY in your .env file."
            )

    def get(self, name: str) -> Optional[BaseProvider]:
        return self._providers.get(name.lower())

    def all(self) -> List[BaseProvider]:
        return list(self._providers.values())

    def ordered(
        self,
        *,
        require_vision: bool = False,
        preferred: Optional[str] = None,
    ) -> List[BaseProvider]:
        """Return configured providers in priority order.

        - `preferred` is moved to the front when configured.
        - When `require_vision` is True, only vision-capable providers are returned.
        """
        order = list(self.settings.provider_order_list)
        for name in PROVIDER_CLASSES:
            if name not in order:
                order.append(name)

        if preferred:
            preferred = preferred.lower()
            if preferred in order:
                order.remove(preferred)
                order.insert(0, preferred)

        result: List[BaseProvider] = []
        for name in order:
            p = self._providers.get(name)
            if not p or not p.is_configured():
                continue
            if require_vision and not p.supports_vision:
                continue
            result.append(p)
        return result


@lru_cache
def get_registry() -> ProviderRegistry:
    return ProviderRegistry()
