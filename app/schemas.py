"""Pydantic models for API requests and responses."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class GenerateJSONRequest(BaseModel):
    """JSON body for the text-only /api/generate endpoint."""

    prompt: str = Field(..., min_length=1, description="User prompt (required).")
    history: Optional[List[ChatMessage]] = Field(
        default=None, description="Optional prior chat history."
    )
    system: Optional[str] = Field(
        default=None, description="Optional system instruction."
    )
    preferred_provider: Optional[str] = Field(
        default=None,
        description="Force a specific provider first (e.g. 'gemini'). "
        "Falls back to others on failure.",
    )
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class AttachmentInfo(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    kind: Literal["image", "document", "text", "unknown"]
    extracted_chars: Optional[int] = None


class GenerateResponse(BaseModel):
    success: bool
    provider: Optional[str] = None
    model: Optional[str] = None
    response: Optional[str] = None
    attachments: List[AttachmentInfo] = Field(default_factory=list)
    tried_providers: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: Optional[int] = None


class ProviderStatus(BaseModel):
    name: str
    enabled: bool
    available: bool
    cooldown_remaining_s: int
    supports_vision: bool
    model: Optional[str] = None
    last_error: Optional[str] = None


class ProvidersResponse(BaseModel):
    providers: List[ProviderStatus]
