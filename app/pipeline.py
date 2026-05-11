"""Pipeline that orchestrates provider attempts with automatic fallback."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .media import ProcessedMedia
from .providers.base import BaseProvider
from .providers.registry import get_registry
from .schemas import ChatMessage
from .utils.errors import (
    AuthError,
    BadRequestError,
    NoProvidersAvailable,
    ProviderError,
    ProviderUnavailable,
    RateLimitError,
)
from .utils.logger import logger


@dataclass
class PipelineResult:
    response: str
    provider: str
    model: str
    tried: List[str] = field(default_factory=list)
    elapsed_ms: int = 0


async def run_pipeline(
    prompt: str,
    media: ProcessedMedia,
    *,
    history: Optional[List[ChatMessage]] = None,
    system: Optional[str] = None,
    preferred_provider: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> PipelineResult:
    registry = get_registry()
    require_vision = media.has_images

    candidates: List[BaseProvider] = registry.ordered(
        require_vision=require_vision,
        preferred=preferred_provider,
    )

    if not candidates:
        if require_vision:
            non_vision = registry.ordered(require_vision=False, preferred=preferred_provider)
            if non_vision:
                logger.warning(
                    "Images attached but no vision-capable provider configured; "
                    "falling back to text-only providers (images will be ignored)."
                )
                candidates = non_vision
                media.images.clear()
        if not candidates:
            raise NoProvidersAvailable(
                "No configured providers. Set at least one *_API_KEY in your .env file."
            )

    started = time.time()
    tried: List[str] = []
    last_exc: Optional[Exception] = None

    for provider in candidates:
        if not provider.is_available():
            cooldown = provider.cooldown_remaining()
            logger.info(
                f"Skipping {provider.name} (cooldown {cooldown}s remaining)"
            )
            continue
        tried.append(provider.name)
        logger.info(f"Trying provider: {provider.name} (model={provider.model})")
        try:
            text = await provider.generate(
                prompt,
                media,
                history=history,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed_ms = int((time.time() - started) * 1000)
            actual_model = provider.last_model_used or provider.model
            logger.success(
                f"[{provider.name}] success in {elapsed_ms}ms (model={actual_model})"
            )
            return PipelineResult(
                response=text,
                provider=provider.name,
                model=actual_model,
                tried=tried,
                elapsed_ms=elapsed_ms,
            )
        except RateLimitError as e:
            logger.warning(f"[{provider.name}] rate-limited: {e}")
            provider.set_last_error(str(e))
            provider.mark_cooldown(e.retry_after)
            last_exc = e
        except AuthError as e:
            logger.error(f"[{provider.name}] auth failure (will not retry): {e}")
            provider.set_last_error(str(e))
            provider.mark_cooldown(60 * 30)
            last_exc = e
        except BadRequestError as e:
            logger.error(f"[{provider.name}] bad request: {e}")
            provider.set_last_error(str(e))
            last_exc = e
        except ProviderUnavailable as e:
            logger.warning(f"[{provider.name}] unavailable: {e}")
            provider.set_last_error(str(e))
            provider.mark_cooldown(30)
            last_exc = e
        except ProviderError as e:
            logger.warning(f"[{provider.name}] error: {e}")
            provider.set_last_error(str(e))
            last_exc = e
        except Exception as e:
            logger.exception(f"[{provider.name}] unexpected error")
            provider.set_last_error(str(e))
            last_exc = e

    raise NoProvidersAvailable(
        f"All providers failed. Tried: {', '.join(tried) or 'none'}. "
        f"Last error: {last_exc}"
    )
