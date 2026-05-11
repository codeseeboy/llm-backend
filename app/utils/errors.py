"""Custom exception types used by providers and the pipeline."""
from __future__ import annotations


class ProviderError(Exception):
    """Base provider error."""

    def __init__(self, message: str, *, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitError(ProviderError):
    """Provider returned 429 / quota exceeded."""


class AuthError(ProviderError):
    """Provider returned 401/403 - missing or invalid key."""


class ProviderUnavailable(ProviderError):
    """Provider returned 5xx or network error."""


class BadRequestError(ProviderError):
    """Provider rejected the request shape (won't help to retry same provider)."""


class NoProvidersAvailable(Exception):
    """No configured provider could fulfil the request."""


__all__ = [
    "ProviderError",
    "RateLimitError",
    "AuthError",
    "ProviderUnavailable",
    "BadRequestError",
    "NoProvidersAvailable",
]
