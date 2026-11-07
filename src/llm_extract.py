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
DEFAULT_MODEL = "openai/gpt-oss-20b"


def _strip_noise(text: str) -> str:
    """Remove common model chatter around JSON (fences, thinking tags)."""
    payload = text.strip()
    payload = re.sub(r"<think>[\s\S]*?</think>", "", payload, flags=re.I)
    payload = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", payload, flags=re.I)
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", payload)
    if fenced:
        payload = fenced.group(1).strip()
    return payload.strip()


def _extract_balanced_object(text: str) -> Optional[str]:
    """Return the first top-level {...} using brace counting (not greedy regex)."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair_json_text(payload: str) -> str:
    """Light repairs for common LLM JSON mistakes."""
    s = payload.strip()
    # Smart quotes → ASCII
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # Trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Python-ish literals
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    return s


def parse_first_json_object(text: str) -> Dict[str, Any]:
    cleaned = _strip_noise(text)
    candidates: list[str] = []
    balanced = _extract_balanced_object(cleaned)
    if balanced:
        candidates.append(balanced)
    # Fallback: greedy outermost braces
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        candidates.append(m.group(0))
    candidates.append(cleaned)

    last_err: Optional[Exception] = None
    seen: set[str] = set()
    for cand in candidates:
        for variant in (cand, _repair_json_text(cand)):
            if variant in seen:
                continue
            seen.add(variant)
            try:
                obj = json.loads(variant)
            except json.JSONDecodeError as ex:
                last_err = ex
                continue
            if isinstance(obj, dict):
                return obj
            last_err = ValueError("Top-level JSON must be an object.")
    raise ValueError(
        f"Could not parse JSON from model output ({last_err}). "
        f"Raw excerpt: {cleaned[:400]!r}"
    )


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


def _chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    api_key, base_url, model = _resolve_credentials()
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    # Prefer JSON mode when the provider supports it (Groq does for many models).
    body["response_format"] = {"type": "json_object"}
    resp = requests.post(url, headers=headers, json=body, timeout=90)
    # Some models reject response_format — retry without it.
    if resp.status_code >= 400 and "response_format" in (resp.text or "").lower():
        body.pop("response_format", None)
        resp = requests.post(url, headers=headers, json=body, timeout=90)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"LLM API error {resp.status_code}: {resp.text[:800]}"
        )
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as ex:
        raise RuntimeError(f"Unexpected LLM response shape: {data!r}") from ex
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Empty LLM content: {data!r}")
    return content


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
    system_text += (
        "\n\nCRITICAL: Respond with a single valid JSON object only. "
        "No markdown, no commentary, no trailing commas."
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": query},
    ]
    raw = _chat_completion(
        messages, temperature=temperature, max_tokens=max_tokens
    )
    try:
        parsed = parse_first_json_object(raw)
    except (ValueError, json.JSONDecodeError):
        # One repair retry with the broken output as context.
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Fix the following into ONE valid JSON object for a mortality "
                    "request schema (status, country, gender, ages, fit_period, "
                    "test_period, models, outcomes, optional forecast_target_year). "
                    "Output JSON only."
                ),
            },
            {
                "role": "user",
                "content": f"User request:\n{query}\n\nBroken model output:\n{raw[:3000]}",
            },
        ]
        raw2 = _chat_completion(
            repair_messages, temperature=0.0, max_tokens=max_tokens
        )
        parsed = parse_first_json_object(raw2)

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
