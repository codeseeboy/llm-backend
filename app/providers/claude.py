"""Anthropic Claude provider."""
from __future__ import annotations

from typing import List, Optional

from ..media import ProcessedMedia
from ..schemas import ChatMessage
from ..utils.errors import ProviderError
from .base import BaseProvider


class ClaudeProvider(BaseProvider):
    name = "claude"
    supports_vision = True

    @property
    def api_key(self) -> str:
        return self.settings.CLAUDE_API_KEY

    @property
    def model(self) -> str:
        return self.settings.CLAUDE_MODEL

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
        if history:
            for msg in history:
                if msg.role == "system":
                    continue
                messages.append({"role": msg.role, "content": msg.content})

        full_text = self.build_text_prompt(prompt, media)
        if media.has_images:
            content: List[dict] = [{"type": "text", "text": full_text}]
            for img in media.images:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.mime_type,
                            "data": img.data_base64,
                        },
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": full_text})

        self._last_model_used = self.model
        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or 4096,
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = await self._post_json(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
        )
        parts = data.get("content") or []
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
        if not text:
            raise ProviderError(f"{self.name} empty response: {data}")
        return text
