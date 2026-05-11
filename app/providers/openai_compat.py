"""Shared helper for OpenAI-compatible chat completions APIs.

Used by OpenRouter, Groq, OpenAI, xAI Grok, and Mistral (which is also
OpenAI-compatible for chat completions).
"""
from __future__ import annotations

from typing import List, Optional

from ..media import ProcessedMedia
from ..schemas import ChatMessage
from ..utils.errors import ProviderError
from .base import BaseProvider


def build_openai_messages(
    prompt: str,
    media: ProcessedMedia,
    *,
    history: Optional[List[ChatMessage]],
    system: Optional[str],
    use_vision: bool,
) -> List[dict]:
    messages: List[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

    if use_vision and media.has_images:
        text_part = (
            f"Use the attached document content as context.\n\n"
            f"{media.combined_document_text}\n\n"
            f"----- USER PROMPT -----\n{prompt}"
        ) if media.combined_document_text else prompt
        content: List[dict] = [{"type": "text", "text": text_part}]
        for img in media.images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.mime_type};base64,{img.data_base64}"
                    },
                }
            )
        messages.append({"role": "user", "content": content})
    else:
        full_text = (
            f"Use the attached document content as context.\n\n"
            f"{media.combined_document_text}\n\n"
            f"----- USER PROMPT -----\n{prompt}"
        ) if media.combined_document_text else prompt
        messages.append({"role": "user", "content": full_text})
    return messages


async def call_openai_compat(
    provider: BaseProvider,
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    media: ProcessedMedia,
    history: Optional[List[ChatMessage]],
    system: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
    use_vision: bool,
    extra_headers: Optional[dict] = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    messages = build_openai_messages(
        prompt, media, history=history, system=system, use_vision=use_vision
    )
    body: dict = {"model": model, "messages": messages}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature

    provider._last_model_used = model
    data = await provider._post_json(base_url, headers=headers, json=body)
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(f"{provider.name} returned no choices: {data}")
    text = (choices[0].get("message") or {}).get("content") or ""
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    text = (text or "").strip()
    if not text:
        raise ProviderError(f"{provider.name} returned empty content")
    return text
