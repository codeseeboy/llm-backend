"""Base class for LLM providers."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx

from ..config import get_settings
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


class BaseProvider(ABC):
    """Base class with cooldown handling and a shared HTTP helper."""

    name: str = "base"
    supports_vision: bool = False
    default_model: str = ""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._cooldown_until: float = 0.0
        self._last_error: Optional[str] = None
        self._last_model_used: Optional[str] = None

    @property
    def last_model_used(self) -> Optional[str]:
        return self._last_model_used

    @property
    @abstractmethod
    def api_key(self) -> str:
        """Return the API key from settings (empty string if not set)."""

    @property
    def model(self) -> str:
        return self.default_model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def is_available(self) -> bool:
        return self.is_configured() and time.time() >= self._cooldown_until

    def cooldown_remaining(self) -> int:
        return max(0, int(self._cooldown_until - time.time()))

    def mark_cooldown(self, seconds: Optional[int] = None) -> None:
        seconds = seconds or self.settings.DEFAULT_COOLDOWN_SECONDS
        self._cooldown_until = time.time() + seconds
        logger.warning(
            f"[{self.name}] cooldown for {seconds}s (until "
            f"{time.strftime('%H:%M:%S', time.localtime(self._cooldown_until))})"
        )

    def set_last_error(self, message: str) -> None:
        self._last_error = message

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _classify_http_error(
        self, status_code: int, body: str, retry_after: Optional[int] = None
    ) -> ProviderError:
        snippet = body[:300]
        if status_code == 429:
            return RateLimitError(f"{self.name} rate-limited: {snippet}", retry_after=retry_after)
        if status_code in (401, 403):
            return AuthError(f"{self.name} auth error ({status_code}): {snippet}")
        if status_code in (400, 404, 422):
            return BadRequestError(f"{self.name} bad request ({status_code}): {snippet}")
        if 500 <= status_code < 600:
            return ProviderUnavailable(f"{self.name} server error ({status_code}): {snippet}")
        return ProviderError(f"{self.name} HTTP {status_code}: {snippet}")

    async def _post_json(
        self,
        url: str,
        *,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> dict:
        timeout = self.settings.REQUEST_TIMEOUT
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=json)
        except httpx.TimeoutException as e:
            raise ProviderUnavailable(f"{self.name} timeout: {e}") from e
        except httpx.HTTPError as e:
            raise ProviderUnavailable(f"{self.name} network error: {e}") from e

        if resp.status_code >= 400:
            retry_after_hdr = resp.headers.get("Retry-After")
            retry_after = None
            if retry_after_hdr and retry_after_hdr.isdigit():
                retry_after = int(retry_after_hdr)
            raise self._classify_http_error(resp.status_code, resp.text, retry_after)

        try:
            return resp.json()
        except ValueError as e:
            raise ProviderUnavailable(f"{self.name} bad JSON response: {e}") from e

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        media: ProcessedMedia,
        history: Optional[List[ChatMessage]] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Send the request and return the generated text."""

    def build_text_prompt(self, prompt: str, media: ProcessedMedia) -> str:
        """Combine the user prompt with extracted document text (for non-vision use)."""
        doc_text = media.combined_document_text
        if not doc_text:
            return prompt
        return (
            "Use the following attached document content as context.\n\n"
            f"{doc_text}\n\n"
            "----- USER PROMPT -----\n"
            f"{prompt}"
        )
