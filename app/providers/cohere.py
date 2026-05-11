"""Cohere provider (text only - free trial tier)."""
from __future__ import annotations

from typing import List, Optional

from ..media import ProcessedMedia
from ..schemas import ChatMessage
from ..utils.errors import ProviderError
from .base import BaseProvider


class CohereProvider(BaseProvider):
    name = "cohere"
    supports_vision = False

    @property
    def api_key(self) -> str:
        return self.settings.COHERE_API_KEY

    @property
    def model(self) -> str:
        return self.settings.COHERE_MODEL

    async def generate(
        self,
        prompt: str,
        media: ProcessedMedia,
        history: Optional[List[ChatMessage]] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if not self.api_key:
            raise ProviderError(f"{self.name} API key not set")

        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for msg in history:
                role = "assistant" if msg.role == "assistant" else "user" if msg.role == "user" else "system"
                messages.append({"role": role, "content": msg.content})

        full_text = self.build_text_prompt(prompt, media)
        if media.has_images:
            full_text += "\n\n[Note: images were attached but this provider is text-only.]"
        messages.append({"role": "user", "content": full_text})

        self._last_model_used = self.model
        body: dict = {"model": self.model, "messages": messages}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = await self._post_json(
            "https://api.cohere.com/v2/chat",
            headers=headers,
            json=body,
        )
        msg = data.get("message") or {}
        parts = msg.get("content") or []
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
        if not text:
            raise ProviderError(f"{self.name} empty response: {data}")
        return text
