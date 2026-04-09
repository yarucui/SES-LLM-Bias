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
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
_OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")

_DEFAULT_HEADERS: dict[str, str] = {
    "Content-Type": "application/json",
    # OpenRouter asks clients to identify themselves via these headers.
    # Keep defaults here so every step script inherits them automatically.
    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com/ses-llm-bias"),
    "X-Title": os.getenv("OPENROUTER_X_TITLE", "SES-LLM-bias dissertation"),
}


class LLMError(Exception):
    """Any failure when calling the LLM — network, HTTP, or parse."""


def _build_headers(extra_headers: dict[str, str] | None) -> dict[str, str]:
    headers = dict(_DEFAULT_HEADERS)
    if _OPENROUTER_KEY:
        headers["Authorization"] = f"Bearer {_OPENROUTER_KEY}"
    if extra_headers:
        for k, v in extra_headers.items():
            if v is None:
                continue
            headers[k] = str(v)
    return headers


def _parse_openrouter_error(resp: requests.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        body = (resp.text or "").strip()
        return body[:500] if body else "no_body"

    err = data.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("error") or err.get("type") or "unknown_error"
        code = err.get("code")
        if code:
            return f"{msg} (code={code})"
        return str(msg)
    return str(err)[:500] if err else "unknown_error"


def openrouter_chat(
    model: str,
    prompt: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float = 60.0,
    extra_headers: dict[str, str] | None = None,
    max_retries: int = 3,
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

    url = f"{_OPENROUTER_BASE}/chat/completions"
    headers = _build_headers(extra_headers)

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last_err = e
            if attempt >= max_retries:
                raise LLMError(f"network: {e}") from e
        else:
            if resp.status_code == 200:
                break

            # Retry on rate-limit and transient upstream errors.
            if resp.status_code in (408, 409, 425, 429, 500, 502, 503, 504) and attempt < max_retries:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = max(0.0, float(retry_after))
                    except ValueError:
                        sleep_s = 0.0
                else:
                    # Exponential backoff + jitter
                    sleep_s = min(20.0, (2.0**attempt) + random.random())
                time.sleep(sleep_s)
                continue

            msg = _parse_openrouter_error(resp)
            raise LLMError(f"http_{resp.status_code}: {msg}")

        # Network exception path: backoff then retry
        sleep_s = min(20.0, (2.0**attempt) + random.random())
        time.sleep(sleep_s)
    else:
        # Should be unreachable due to raises above.
        raise LLMError(f"network: {last_err}") from last_err

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as e:
        raise LLMError(f"parse: {e}") from e

    return content or ""
