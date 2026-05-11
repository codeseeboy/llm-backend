"""OpenAI ChatGPT provider."""
from __future__ import annotations

from typing import List, Optional

from ..media import ProcessedMedia
from ..schemas import ChatMessage
from .base import BaseProvider
from .openai_compat import call_openai_compat


class OpenAIProvider(BaseProvider):
    name = "openai"
    supports_vision = True

    @property
    def api_key(self) -> str:
        return self.settings.OPENAI_API_KEY

    @property
    def model(self) -> str:
        return self.settings.OPENAI_MODEL

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
        return await call_openai_compat(
            self,
            base_url="https://api.openai.com/v1/chat/completions",
            model=self.model,
            api_key=self.api_key,
            prompt=prompt,
            media=media,
            history=history,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            use_vision=use_vision,
        )
