"""FastAPI application entrypoint."""
from __future__ import annotations

import json
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .media import process_uploads
from .pipeline import run_pipeline
from .providers.registry import get_registry
from .schemas import (
    ChatMessage,
    GenerateJSONRequest,
    GenerateResponse,
    ProvidersResponse,
    ProviderStatus,
)
from .utils.errors import NoProvidersAvailable
from .utils.logger import configure_logger, logger


def create_app() -> FastAPI:
    configure_logger()
    settings = get_settings()

    app = FastAPI(
        title="Multi-LLM Backend",
        version="0.1.0",
        description=(
            "Unified API that accepts a text prompt with optional media "
            "(images, PDFs, DOCX, etc.) and routes it to the first available "
            "free LLM provider, with automatic fallback on rate-limits."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        get_registry()
        logger.info(f"Server starting on {settings.HOST}:{settings.PORT}")

    @app.get("/")
    async def root() -> dict:
        return {
            "name": "Multi-LLM Backend",
            "version": "0.1.0",
            "endpoints": {
                "POST /api/generate": "Send text + optional files (multipart/form-data)",
                "POST /api/generate/json": "Send text-only prompts (application/json)",
                "GET /api/providers": "Provider status and availability",
                "GET /health": "Health check",
                "GET /docs": "OpenAPI / Swagger UI",
            },
        }

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/providers", response_model=ProvidersResponse)
    async def providers() -> ProvidersResponse:
        registry = get_registry()
        items: List[ProviderStatus] = []
        for p in registry.all():
            items.append(
                ProviderStatus(
                    name=p.name,
                    enabled=p.is_configured(),
                    available=p.is_available(),
                    cooldown_remaining_s=p.cooldown_remaining(),
                    supports_vision=p.supports_vision,
                    model=p.model if p.is_configured() else None,
                    last_error=p.last_error,
                )
            )
        return ProvidersResponse(providers=items)

    @app.post("/api/generate/json", response_model=GenerateResponse)
    async def generate_json(req: GenerateJSONRequest) -> GenerateResponse:
        return await _do_generate(
            prompt=req.prompt,
            files=None,
            history=req.history,
            system=req.system,
            preferred_provider=req.preferred_provider,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate(
        prompt: str = Form(..., description="User prompt (required)."),
        files: Optional[List[UploadFile]] = File(default=None),
        history: Optional[str] = Form(
            default=None,
            description="Optional JSON-encoded list of {role, content} chat history.",
        ),
        system: Optional[str] = Form(default=None),
        preferred_provider: Optional[str] = Form(default=None),
        max_tokens: Optional[int] = Form(default=None),
        temperature: Optional[float] = Form(default=None),
    ) -> GenerateResponse:
        parsed_history: Optional[List[ChatMessage]] = None
        if history:
            try:
                raw = json.loads(history)
                parsed_history = [ChatMessage(**m) for m in raw]
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid `history` JSON: {e}"
                )

        return await _do_generate(
            prompt=prompt,
            files=files,
            history=parsed_history,
            system=system,
            preferred_provider=preferred_provider,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def _do_generate(
        *,
        prompt: str,
        files: Optional[List[UploadFile]],
        history: Optional[List[ChatMessage]],
        system: Optional[str],
        preferred_provider: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> GenerateResponse:
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="`prompt` is required")
        media = await process_uploads(files)
        try:
            result = await run_pipeline(
                prompt=prompt,
                media=media,
                history=history,
                system=system,
                preferred_provider=preferred_provider,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except NoProvidersAvailable as e:
            return JSONResponse(
                status_code=503,
                content=GenerateResponse(
                    success=False,
                    error=str(e),
                    attachments=media.info,
                ).model_dump(),
            )
        return GenerateResponse(
            success=True,
            provider=result.provider,
            model=result.model,
            response=result.response,
            attachments=media.info,
            tried_providers=result.tried,
            elapsed_ms=result.elapsed_ms,
        )

    return app


app = create_app()
