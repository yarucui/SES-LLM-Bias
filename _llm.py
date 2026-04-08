"""Thin OpenRouter chat wrapper used by every LLM-calling pipeline step.

The pipeline was originally written to call the Gemini API directly via the
google-generativeai SDK, but this research environment only has an OpenRouter
key — and OpenRouter hosts Google's Gemini models under `google/...` paths.
All Gemini calls therefore route through this single helper.

Steps still refer to the logical constant `config.GEMINI_MODEL`; the value of
that constant is just an OpenRouter path now.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_HEADERS = {
    "Authorization": f"Bearer {_OPENROUTER_KEY}" if _OPENROUTER_KEY else "",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/ses-llm-bias-research",
    "X-Title": "SES-LLM-bias dissertation",
}


class LLMError(Exception):
    """Any failure when calling the LLM — network, HTTP, or parse."""


def openrouter_chat(
    model: str,
    prompt: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float = 60.0,
) -> str:
    """Single-turn chat completion. Returns the assistant message text.

    Raises ``LLMError`` on any failure so callers can catch a single exception
    type regardless of how the call failed.
    """
    if not _OPENROUTER_KEY:
        raise LLMError("OPENROUTER_API_KEY missing from .env")

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    try:
        resp = requests.post(
            f"{_OPENROUTER_BASE}/chat/completions",
            headers=_HEADERS,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise LLMError(f"network: {e}") from e

    if resp.status_code != 200:
        body = (resp.text or "")[:500]
        raise LLMError(f"http_{resp.status_code}: {body}")

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as e:
        raise LLMError(f"parse: {e}") from e

    return content or ""
