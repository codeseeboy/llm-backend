"""Google Gemini provider (free tier via AI Studio)."""
from __future__ import annotations

from typing import List, Optional

from ..media import ProcessedMedia
from ..schemas import ChatMessage
from ..utils.errors import ProviderError
from .base import BaseProvider


class GeminiProvider(BaseProvider):
    name = "gemini"
    supports_vision = True

    @property
    def api_key(self) -> str:
        return self.settings.GEMINI_API_KEY

    @property
    def model(self) -> str:
        return self.settings.GEMINI_VISION_MODEL or self.settings.GEMINI_MODEL

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

        contents: List[dict] = []
        if history:
            for msg in history:
                role = "user" if msg.role == "user" else "model"
                if msg.role == "system":
                    continue
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        user_parts: List[dict] = []
        full_text = self.build_text_prompt(prompt, media)
        user_parts.append({"text": full_text})
        for img in media.images:
            user_parts.append(
                {
                    "inline_data": {
                        "mime_type": img.mime_type,
                        "data": img.data_base64,
                    }
                }
            )
        contents.append({"role": "user", "parts": user_parts})

        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        gen_cfg: dict = {}
        if max_tokens is not None:
            gen_cfg["maxOutputTokens"] = max_tokens
        if temperature is not None:
            gen_cfg["temperature"] = temperature
        if gen_cfg:
            body["generationConfig"] = gen_cfg

        model_name = self.model
        self._last_model_used = model_name
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={self.api_key}"
        )
        data = await self._post_json(url, headers={"Content-Type": "application/json"}, json=body)

        candidates = data.get("candidates") or []
        if not candidates:
            err = data.get("error") or data.get("promptFeedback") or data
            raise ProviderError(f"{self.name} returned no candidates: {err}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            finish = candidates[0].get("finishReason")
            raise ProviderError(f"{self.name} empty response (finishReason={finish})")
        return text
