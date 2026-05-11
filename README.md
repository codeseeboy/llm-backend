# Multi-LLM Backend

A FastAPI backend that exposes **one unified API** for sending a text prompt
plus optional media (images, PDFs, DOCX, XLSX, HTML, MD, TXT, etc.) and
returns an LLM response. Internally it tries a chain of LLM providers in
priority order and **automatically falls back** to the next one when a
provider is rate-limited, expired, or unavailable.

## Supported providers

| Provider     | Free tier | Vision | Notes                                  |
|--------------|-----------|--------|----------------------------------------|
| Gemini       | yes       | yes    | Recommended free default               |
| OpenRouter   | yes       | yes    | Many free models available             |
| Groq         | yes       | yes    | Very fast inference                    |
| Mistral      | yes       | yes    | `pixtral-12b` for vision               |
| Cohere       | trial     | no     | Text only, falls back gracefully       |
| HuggingFace  | yes       | no     | Inference Router                        |
| Claude       | paid      | yes    | Anthropic                              |
| OpenAI       | paid      | yes    | ChatGPT                                |
| xAI Grok     | trial     | yes    | OpenAI-compatible                      |

You only need **one** API key to start; the more you add, the more resilient
the fallback becomes.

## Quick start

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
copy .env.example .env            # Windows
# cp .env.example .env            # macOS / Linux
# ...edit .env and add at least one *_API_KEY...
python run.py
```

The server starts at `http://localhost:8000` with interactive docs at
`http://localhost:8000/docs`.

## API

### `POST /api/generate` (multipart/form-data)

Send a prompt with optional file attachments.

**Form fields**

| Field                | Type      | Required | Description                                           |
|----------------------|-----------|----------|-------------------------------------------------------|
| `prompt`             | string    | yes      | The user prompt                                       |
| `files`              | file[]    | no       | One or more uploads (images, PDFs, docs, text)        |
| `system`             | string    | no       | System instruction                                    |
| `history`            | string    | no       | JSON-encoded array of `{role, content}` messages      |
| `preferred_provider` | string    | no       | e.g. `gemini`. Tried first, falls back on failure.    |
| `max_tokens`         | int       | no       |                                                       |
| `temperature`        | float     | no       |                                                       |

**Example (curl)**

```bash
curl -X POST http://localhost:8000/api/generate ^
  -F "prompt=What is in this image?" ^
  -F "files=@photo.jpg" ^
  -F "files=@notes.pdf"
```

**Example (Python)**

```python
import requests
r = requests.post(
    "http://localhost:8000/api/generate",
    data={"prompt": "Summarise this document"},
    files=[("files", open("report.pdf", "rb"))],
)
print(r.json())
```

### `POST /api/generate/json` (application/json)

Text-only convenience endpoint.

```json
{
  "prompt": "Explain quantum entanglement in 3 bullets",
  "system": "You are a helpful physics tutor.",
  "preferred_provider": "gemini",
  "temperature": 0.4
}
```

### `GET /api/providers`

Returns each provider's enabled / availability / cooldown status:

```json
{
  "providers": [
    {
      "name": "gemini",
      "enabled": true,
      "available": true,
      "cooldown_remaining_s": 0,
      "supports_vision": true,
      "model": "gemini-2.0-flash"
    }
  ]
}
```

## Response format

```json
{
  "success": true,
  "provider": "gemini",
  "model": "gemini-2.0-flash",
  "response": "...",
  "attachments": [
    { "filename": "photo.jpg", "kind": "image", "size_bytes": 12345 }
  ],
  "tried_providers": ["gemini"],
  "elapsed_ms": 842
}
```

## How fallback works

1. The pipeline filters configured providers using `PROVIDER_ORDER`.
2. If images are attached, only vision-capable providers are tried.
3. Each provider has a per-instance cooldown timer.
4. On HTTP `429` -> cooldown set from `Retry-After` (or `DEFAULT_COOLDOWN_SECONDS`).
5. On `401/403` -> 30-minute cooldown (likely invalid key).
6. On `5xx` / network -> 30-second cooldown.
7. Pipeline moves to the next configured provider immediately.
8. If every provider fails, returns HTTP 503 with diagnostics.

## File handling

- **Images** (`png`, `jpg`, `webp`, `gif`, ...) -> sent as base64 to vision-capable providers.
- **PDF** -> text extracted with `pypdf`.
- **DOCX** -> text + tables extracted with `python-docx`.
- **XLSX / XLSM** -> rendered to text with `openpyxl`.
- **HTML / MD / TXT / CSV / JSON / YAML / XML** -> read as text.
- **Other** -> attempted UTF-8 decode, otherwise reported as binary.

Document text is appended to the prompt as context. Each document is capped
at ~60k characters to keep request bodies reasonable.

## Project layout

```
backend/
├── app/
│   ├── main.py              FastAPI routes
│   ├── pipeline.py          Provider orchestration + fallback
│   ├── media.py             File parsing (image/PDF/DOCX/...)
│   ├── config.py            Settings from .env
│   ├── schemas.py           Pydantic models
│   ├── providers/
│   │   ├── base.py          BaseProvider with cooldown handling
│   │   ├── registry.py      Discovery + ordered lookup
│   │   ├── openai_compat.py Shared helper for OpenAI-style APIs
│   │   ├── gemini.py        Google Gemini
│   │   ├── openrouter.py    OpenRouter (free models)
│   │   ├── groq.py          Groq
│   │   ├── mistral.py       Mistral
│   │   ├── cohere.py        Cohere
│   │   ├── huggingface.py   HF Inference Router
│   │   ├── claude.py        Anthropic Claude
│   │   ├── openai.py        OpenAI
│   │   └── grok.py          xAI Grok
│   └── utils/               logger + custom errors
├── requirements.txt
├── .env.example
├── run.py
└── README.md
```
