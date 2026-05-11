"""HuggingFace Inference API (Router) - OpenAI-compatible chat completions."""
from __future__ import annotations

from typing import List, Optional

from ..media import ProcessedMedia
from ..schemas import ChatMessage
from .base import BaseProvider
from .openai_compat import call_openai_compat


class HuggingFaceProvider(BaseProvider):
    name = "huggingface"
    supports_vision = False

    @property
    def api_key(self) -> str:
        return self.settings.HUGGINGFACE_API_KEY

    @property
    def model(self) -> str:
        return self.settings.HUGGINGFACE_MODEL

    async def generate(
        self,
        prompt: str,
        media: ProcessedMedia,
        history: Optional[List[ChatMessage]] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        return await call_openai_compat(
            self,
            base_url="https://router.huggingface.co/v1/chat/completions",
            model=self.model,
            api_key=self.api_key,
            prompt=prompt,
            media=media,
            history=history,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            use_vision=False,
        )
