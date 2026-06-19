#!/usr/bin/env python3
"""Uniform interface for calling Anthropic, OpenAI, Google Gemini, and Ollama.

Each provider has its own SDK surface and JSON-output conventions. This module
wraps all of them behind a single `classify(provider, model, system, user, ...)`
function that returns a parsed dict (or raises on hard failure).

Provider identifiers:
    "anthropic"  → claude-haiku-4-5 (or any Claude model)
    "openai"     → gpt-5-nano (or any OpenAI chat-completions model)
    "gemini"     → gemini-2.5-flash-lite (or any Gemini model)
    "ollama"     → any model pulled on the local Ollama server, e.g.
                   "qwen2.5:32b-instruct-q4_K_M". Uses urllib — no extra SDK.

Each adapter handles its own retry on transient errors. Hard failures raise
ProviderError so the caller can record them as per-model errors without losing
scores from other providers.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Common output handling
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Raised when a provider call fails irrecoverably (after retries)."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"{provider}: {message}")


def _strip_fences(text: str) -> str:
    """Strip markdown code fences if a model added them despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text


def _normalize_output(data: dict) -> dict:
    """Normalize quirks that show up across providers (GPT-5 and Gemini both
    treat the schema's section headers — TOPIC, FRAMING_DIRECTION, etc. — as
    literal UPPERCASE JSON keys, while Claude correctly uses lowercase).
    Also handles:
      - Collapsed scalar values where we expected a dict (GPT-5 sometimes
        flattens framing_tone: 6 instead of returning a dict of 5 dims).
      - Confidence returned as a 0-1 float instead of a 0-10 int.

    Applied to every provider's parsed output so the downstream Score parser
    sees a consistent shape regardless of which model produced it.
    """
    if not isinstance(data, dict):
        return data
    expected_dict_sections = {"topic", "framing_direction", "framing_tone", "voice"}
    out = {}
    for k, v in data.items():
        lk = k.lower()
        if lk in expected_dict_sections and not isinstance(v, dict):
            v = {}
        if lk == "confidence" and isinstance(v, (int, float)) and 0 < v <= 1:
            v = v * 10
        out[lk] = v
    return out


def parse_json_strict(text: str) -> dict:
    """Parse model output as JSON, with a fallback that extracts the first {...}
    block if the model wrapped its JSON in prose. Always normalizes the result
    so per-provider schema quirks don't reach the downstream Score parser."""
    text = _strip_fences(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                raise ValueError(f"json parse: {e}")
        else:
            raise ValueError(f"json parse: {e}")
    return _normalize_output(data)


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

def _classify_anthropic(client, model: str, system: str, user: str,
                        max_tokens: int, timeout: int) -> dict:
    try:
        from anthropic import APIConnectionError, APIStatusError, RateLimitError
    except ImportError:
        raise ProviderError("anthropic", "anthropic SDK not installed")

    last_err = None
    for backoff in (0, 4, 12):
        if backoff:
            time.sleep(backoff)
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
                timeout=timeout,
            )
            text = "".join(
                block.text for block in resp.content
                if getattr(block, "type", None) == "text"
            )
            return parse_json_strict(text)
        except RateLimitError as e:
            last_err = e
            continue
        except (APIConnectionError, APIStatusError) as e:
            last_err = e
            continue
        except ValueError as e:
            raise ProviderError("anthropic", str(e))
        except Exception as e:
            raise ProviderError("anthropic", f"{type(e).__name__}: {e}")
    raise ProviderError("anthropic", f"after retries: {last_err}")


# ---------------------------------------------------------------------------
# OpenAI adapter (Responses API)
# ---------------------------------------------------------------------------

def _classify_openai(client, model: str, system: str, user: str,
                     max_tokens: int, timeout: int) -> dict:
    """Use chat.completions for GPT-5 series — more reliable than the
    Responses API for parallel reasoning-model calls. Forces JSON-object
    output and asks for minimal reasoning effort so the model doesn't burn
    the entire output budget thinking.
    """
    try:
        import openai as openai_pkg
    except ImportError:
        raise ProviderError("openai", "openai SDK not installed")

    is_gpt5 = model.startswith("gpt-5")
    # 3x token cap on GPT-5 to leave room if any reasoning leaks through
    effective_max = max_tokens * 3 if is_gpt5 else max_tokens

    last_err = None
    for backoff in (0, 4, 12):
        if backoff:
            time.sleep(backoff)
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
                "timeout": timeout,
            }
            # GPT-5 uses max_completion_tokens; older models use max_tokens
            if is_gpt5:
                kwargs["max_completion_tokens"] = effective_max
                kwargs["reasoning_effort"] = "minimal"
            else:
                kwargs["max_tokens"] = effective_max
                kwargs["temperature"] = 0.0

            resp = client.chat.completions.create(**kwargs)
            choice = (resp.choices or [None])[0]
            text = (choice.message.content if choice and choice.message else "") or ""
            if not text:
                # If completion truncated mid-stream or hit safety filter, surface why
                finish = getattr(choice, "finish_reason", None) if choice else "no_choice"
                raise ProviderError(
                    "openai",
                    f"empty response (finish_reason={finish})"
                )
            return parse_json_strict(text)
        except openai_pkg.RateLimitError as e:
            last_err = e
            continue
        except (openai_pkg.APIConnectionError, openai_pkg.APIStatusError,
                openai_pkg.APITimeoutError) as e:
            last_err = e
            continue
        except ValueError as e:
            raise ProviderError("openai", str(e))
        except Exception as e:
            raise ProviderError("openai", f"{type(e).__name__}: {e}")
    raise ProviderError("openai", f"after retries: {last_err}")


# ---------------------------------------------------------------------------
# Gemini adapter (google-genai)
# ---------------------------------------------------------------------------

def _classify_gemini(client, model: str, system: str, user: str,
                     max_tokens: int, timeout: int) -> dict:
    try:
        from google.genai import types as genai_types
        from google.genai import errors as genai_errors
    except ImportError:
        raise ProviderError("gemini", "google-genai SDK not installed")

    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.0,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",  # forces JSON output
        # NOTE: deliberately NOT setting safety_settings — defaults are fine
        # for our analysis content but we may need to relax if Gemini blocks
        # articles about violence against trans people.
    )

    last_err = None
    for backoff in (0, 4, 12):
        if backoff:
            time.sleep(backoff)
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=config,
            )
            text = (resp.text or "").strip()
            if not text:
                # Some safety blocks return empty text but populate
                # candidates[0].finish_reason. Surface that as the error.
                cands = getattr(resp, "candidates", None) or []
                reason = (cands[0].finish_reason if cands else "unknown")
                raise ProviderError("gemini", f"empty response (finish_reason={reason})")
            return parse_json_strict(text)
        except genai_errors.ClientError as e:
            # 4xx errors — generally not retryable except rate limits
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                last_err = e
                continue
            raise ProviderError("gemini", msg)
        except genai_errors.ServerError as e:
            last_err = e
            continue
        except ValueError as e:
            raise ProviderError("gemini", str(e))
        except Exception as e:
            raise ProviderError("gemini", f"{type(e).__name__}: {e}")
    raise ProviderError("gemini", f"after retries: {last_err}")


# ---------------------------------------------------------------------------
# Ollama adapter (local HTTP — no SDK, uses urllib so nothing extra needs
# to be bundled). Talks to /api/chat with format=json which constrains the
# model to valid JSON output.
# ---------------------------------------------------------------------------

def _classify_ollama(base_url: str, model: str, system: str, user: str,
                     max_tokens: int, timeout: int) -> dict:
    """Call a local Ollama server's /api/chat endpoint.

    base_url: e.g. "http://localhost:11434" or "http://192.168.1.42:11434"
    model:    Ollama model id, e.g. "qwen2.5:32b-instruct-q4_K_M"

    Retries on connection errors and 5xx responses; raises ProviderError on
    4xx, empty content, or JSON parse failure.
    """
    import urllib.request
    import urllib.error

    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",  # constrain output to valid JSON
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
            # Cap the context window. The model's default (32K for Qwen 2.5)
            # blows up the KV cache and forces layer offload to CPU on a 24GB
            # card, which is ~100x slower. Our prompts (system + 12K-char body
            # + output) fit comfortably in 8K, so this keeps a 32B model fully
            # resident on a single 24GB GPU (e.g. RTX 4090).
            "num_ctx": 8192,
        },
    }
    data = json.dumps(payload).encode("utf-8")

    last_err = None
    for backoff in (0, 4, 12):
        if backoff:
            time.sleep(backoff)
        try:
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            obj = json.loads(body)
            text = ((obj.get("message") or {}).get("content") or "").strip()
            if not text:
                # surface why — some models truncate or return empty under
                # bad config; the done_reason field tells us
                reason = obj.get("done_reason") or "unknown"
                raise ProviderError(
                    "ollama", f"empty response (done_reason={reason})"
                )
            return parse_json_strict(text)
        except urllib.error.HTTPError as e:
            # 5xx retryable; 4xx terminal
            if 500 <= e.code < 600:
                last_err = e
                continue
            raise ProviderError("ollama", f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            # connection refused / timeout / DNS — retryable
            last_err = e
            continue
        except ValueError as e:
            raise ProviderError("ollama", str(e))
        except Exception as e:
            raise ProviderError("ollama", f"{type(e).__name__}: {e}")
    raise ProviderError("ollama", f"after retries: {last_err}")


def probe_ollama(base_url: str, timeout: float = 3.0) -> tuple[bool, str]:
    """Reachability check for a local/remote Ollama server.

    Returns (ok, info). On success, `info` is a comma-separated list of
    installed model names. On failure, `info` is the error text.
    """
    import urllib.request

    try:
        url = base_url.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        obj = json.loads(body)
        models = [m.get("name", "") for m in (obj.get("models") or [])]
        return True, ", ".join(m for m in models if m)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Client construction (one-time per process)
# ---------------------------------------------------------------------------

@dataclass
class Clients:
    """Holds initialized SDK clients. Build once, reuse across calls."""
    anthropic_client: Optional[object] = None
    openai_client: Optional[object] = None
    gemini_client: Optional[object] = None
    # Ollama doesn't have a stateful SDK client — we just stash the base URL
    # and let _classify_ollama use stdlib urllib at call time.
    ollama_url: Optional[str] = None


def build_clients(anthropic_key: str, openai_key: str, gemini_key: str,
                  ollama_url: Optional[str] = None) -> Clients:
    out = Clients()
    if anthropic_key:
        from anthropic import Anthropic
        out.anthropic_client = Anthropic(api_key=anthropic_key)
    if openai_key:
        from openai import OpenAI
        out.openai_client = OpenAI(api_key=openai_key)
    if gemini_key:
        from google import genai
        out.gemini_client = genai.Client(api_key=gemini_key)
    if ollama_url:
        out.ollama_url = ollama_url.rstrip("/")
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify(provider: str, clients: Clients, model: str, system: str,
             user: str, max_tokens: int = 1200, timeout: int = 60) -> dict:
    """Call the named provider with a system + user message, return parsed JSON.

    Raises ProviderError on irrecoverable failure.
    """
    if provider == "anthropic":
        if not clients.anthropic_client:
            raise ProviderError("anthropic", "no API key configured")
        return _classify_anthropic(clients.anthropic_client, model, system, user,
                                   max_tokens, timeout)
    if provider == "openai":
        if not clients.openai_client:
            raise ProviderError("openai", "no API key configured")
        return _classify_openai(clients.openai_client, model, system, user,
                                max_tokens, timeout)
    if provider == "gemini":
        if not clients.gemini_client:
            raise ProviderError("gemini", "no API key configured")
        return _classify_gemini(clients.gemini_client, model, system, user,
                                max_tokens, timeout)
    if provider == "ollama":
        if not clients.ollama_url:
            raise ProviderError("ollama", "no Ollama URL configured")
        return _classify_ollama(clients.ollama_url, model, system, user,
                                max_tokens, timeout)
    raise ProviderError(provider, "unknown provider")
