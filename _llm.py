"""Thin OpenAI-compatible chat wrapper used by every LLM-calling pipeline step.

Originally written against OpenRouter only, this module now also handles the
OpenAI native API (added for step5, which routes GPT-5 directly to OpenAI
to save cost). Both providers speak the same /chat/completions schema, so
the only differences between call sites are api_key, base_url, and the
OpenRouter-specific HTTP-Referer / X-Title headers.

Two public entry points:

  chat_completion(...)    -- provider-agnostic; takes api_key, base_url, model
                             explicitly. Used by step5_experiment.
  openrouter_chat(...)    -- backward-compat wrapper. Used by step2_score
                             (which predates the refactor).
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

_OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY")
_OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")


class LLMError(Exception):
    """Any failure when calling the LLM -- network, HTTP, or parse."""


def _parse_error_body(resp: requests.Response) -> str:
    """Best-effort extraction of a useful error message from an error response."""
    try:
        data = resp.json()
    except ValueError:
        body = (resp.text or "").strip()
        return body[:500] if body else "no_body"

    err = data.get("error")
    if isinstance(err, dict):
        msg  = err.get("message") or err.get("error") or err.get("type") or "unknown_error"
        code = err.get("code")
        if code:
            return f"{msg} (code={code})"
        return str(msg)
    return str(err)[:500] if err else "unknown_error"


def chat_completion(
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float = 60.0,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict | None = None,
    max_retries: int = 3,
) -> str:
    """Single-turn chat completion against any OpenAI-compatible endpoint.

    extra_body keys are merged into the request body alongside model /
    messages / max_tokens / temperature. Use it to pass provider-specific
    fields like reasoning_effort (OpenAI reasoning models, ignored by
    providers that do not implement it).

    Raises LLMError on any failure so callers can catch a single exception
    type. Retries on 408/409/425/429/5xx with exponential backoff + jitter
    and respects Retry-After when the server provides it.
    """
    if not api_key:
        raise LLMError("api_key missing")

    payload: dict = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if extra_body:
        for k, v in extra_body.items():
            payload[k] = v

    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    if extra_headers:
        for k, v in extra_headers.items():
            if v is None or v == "":
                continue
            headers[k] = str(v)

    url = f"{base_url.rstrip('/')}/chat/completions"

    data = _chat_completion_json(
        url=url,
        headers=headers,
        payload=payload,
        timeout=timeout,
        max_retries=max_retries,
    )
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"parse: {e}") from e
    return content or ""


def _chat_completion_json(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout: float,
    max_retries: int,
) -> dict:
    """Shared retrying POST helper returning the parsed JSON body."""
    last_err: Exception | None = None
    resp: requests.Response | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last_err = e
            if attempt >= max_retries:
                raise LLMError(f"network: {e}") from e
            sleep_s = min(20.0, (2.0 ** attempt) + random.random())
            time.sleep(sleep_s)
            continue

        if resp.status_code == 200:
            break

        if resp.status_code in (408, 409, 425, 429, 500, 502, 503, 504) and attempt < max_retries:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_s = max(0.0, float(retry_after))
                except ValueError:
                    sleep_s = 0.0
            else:
                sleep_s = min(20.0, (2.0 ** attempt) + random.random())
            time.sleep(sleep_s)
            continue

        msg = _parse_error_body(resp)
        raise LLMError(f"http_{resp.status_code}: {msg}")
    else:
        raise LLMError(f"network: {last_err}") from last_err

    try:
        return resp.json()
    except ValueError as e:
        raise LLMError(f"parse: {e}") from e


def chat_completion_with_usage(
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float = 60.0,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict | None = None,
    max_retries: int = 3,
) -> dict:
    """Same transport as chat_completion, but keeps provider usage fields."""
    if not api_key:
        raise LLMError("api_key missing")

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if extra_body:
        for k, v in extra_body.items():
            payload[k] = v

    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        for k, v in extra_headers.items():
            if v is None or v == "":
                continue
            headers[k] = str(v)

    url = f"{base_url.rstrip('/')}/chat/completions"
    data = _chat_completion_json(
        url=url,
        headers=headers,
        payload=payload,
        timeout=timeout,
        max_retries=max_retries,
    )

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"parse: {e}") from e

    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    return {
        "content": content or "",
        "usage": usage,
        "raw": data,
    }


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
    """Backward-compat wrapper around chat_completion for OpenRouter.

    Pre-fills the OpenRouter base URL, API key, and the HTTP-Referer /
    X-Title headers that OpenRouter asks clients to send. step2_score.py
    still calls this; new code should call chat_completion directly.
    """
    full_extra: dict[str, str] = {
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com/ses-llm-bias"),
        "X-Title":      os.getenv("OPENROUTER_X_TITLE",      "SES-LLM-bias dissertation"),
    }
    if extra_headers:
        full_extra.update({k: v for k, v in extra_headers.items() if v is not None})

    return chat_completion(
        api_key=_OPENROUTER_KEY,
        base_url=_OPENROUTER_BASE,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        extra_headers=full_extra,
        max_retries=max_retries,
    )
