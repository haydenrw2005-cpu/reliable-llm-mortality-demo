"""NL → JSON extraction via OpenAI-compatible cloud APIs (Groq, OpenAI, OpenRouter, etc.)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from .paths import EXTRACTOR_SYSTEM

# Defaults tuned for free Groq tier; override with env vars.
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def parse_first_json_object(text: str) -> Dict[str, Any]:
    payload = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", payload)
    if fenced:
        payload = fenced.group(1)
    else:
        m = re.search(r"\{[\s\S]*\}", payload)
        if not m:
            raise ValueError("No JSON object found in model output.")
        payload = m.group(0)
    obj = json.loads(payload)
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object.")
    return obj


def enrich_forecast_target_year_from_query(
    payload: Dict[str, Any], query: str
) -> Dict[str, Any]:
    """If the model omitted forecast_target_year, recover digits from the prompt."""
    if payload.get("status") == "need_clarification":
        return payload
    if payload.get("forecast_target_year") not in (None, ""):
        return payload
    patterns = (
        r"(?i)forecast\s+for\s+the\s+year\s+(20[0-9]{2})\b",
        r"(?i)forecast\s+(?:to|until|through)\s+(20[0-9]{2})\b",
        r"(?i)mortality\s+(?:in|for)\s+(20[0-9]{2})\b",
    )
    for pat in patterns:
        m = re.search(pat, query)
        if not m:
            continue
        y = int(m.group(1))
        if 2000 <= y <= 2100:
            out = dict(payload)
            out["forecast_target_year"] = y
            return out
    return payload


def _resolve_credentials() -> Tuple[str, str, str]:
    api_key = (
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("GROQ_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError(
            "Missing LLM API key. Set LLM_API_KEY (or GROQ_API_KEY / OPENAI_API_KEY)."
        )
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return api_key, base_url, model


def extract_user_request(
    query: str,
    *,
    system_prompt_path: Optional[Path] = None,
    temperature: float = 0.1,
    max_tokens: int = 768,
) -> Dict[str, Any]:
    """Call chat completions and return parsed request JSON."""
    system_path = system_prompt_path or EXTRACTOR_SYSTEM
    if not system_path.is_file():
        raise FileNotFoundError(f"System prompt not found: {system_path}")
    system_text = system_path.read_text(encoding="utf-8")

    api_key, base_url, model = _resolve_credentials()
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": query},
        ],
    }
    resp = requests.post(url, headers=headers, json=body, timeout=90)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"LLM API error {resp.status_code}: {resp.text[:800]}"
        )
    data = resp.json()
    try:
        raw = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as ex:
        raise RuntimeError(f"Unexpected LLM response shape: {data!r}") from ex

    parsed = parse_first_json_object(raw)
    return enrich_forecast_target_year_from_query(parsed, query)


def llm_status() -> Dict[str, str]:
    """Non-secret status for the UI sidebar."""
    has_key = bool(
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("GROQ_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    return {
        "configured": "yes" if has_key else "no",
        "base_url": os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        "provider_hint": "Groq (free tier) by default; any OpenAI-compatible API works",
    }
