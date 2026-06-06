"""
LLM provider with automatic fallback.

Priority:
  1. Anthropic (Claude Haiku) — primary
  2. Groq (Llama 3.3 70B)    — fallback when Anthropic credits exhausted

Fallback triggers on:
  - anthropic.AuthenticationError  (invalid / expired key)
  - anthropic.PermissionDeniedError
  - anthropic.RateLimitError       (quota exceeded, not just rate limit)
  - Any error whose message contains "credit", "quota", "billing", "overload"
"""

import httpx
import json
import logging
from core.config import get_settings

logger = logging.getLogger(__name__)

# Errors that should trigger Groq fallback
FALLBACK_TRIGGERS = (
    "credit", "quota", "billing", "insufficient_quota",
    "overloaded", "capacity", "529",
)

def _should_fallback(error: Exception) -> bool:
    msg = str(error).lower()
    return any(t in msg for t in FALLBACK_TRIGGERS)


async def _call_anthropic(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
    model: str = "claude-haiku-4-5",
) -> str:
    """Call Anthropic Claude API."""
    import anthropic
    settings = get_settings()

    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return response.content[0].text


async def _call_groq(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Call Groq API (OpenAI-compatible endpoint).
    Free tier: 14,400 requests/day, 500,000 tokens/minute.
    """
    settings = get_settings()

    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY not set — no fallback available")

    # Build messages in OpenAI format
    groq_messages = [{"role": "system", "content": system}] + messages

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":      model,
                "messages":   groq_messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def llm_call(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
    prefer_json: bool = False,
) -> str:
    """
    Call LLM with automatic Anthropic → Groq fallback.

    Args:
        system:      System prompt string
        messages:    List of {"role": "user"|"assistant", "content": "..."}
        max_tokens:  Maximum tokens to generate
        prefer_json: If True, append JSON instruction to system prompt

    Returns:
        Response text string
    """
    if prefer_json:
        system += "\n\nReturn ONLY valid JSON. No markdown, no explanation."

    settings = get_settings()

    # ── Try Anthropic first ──────────────────────────────────────
    if settings.anthropic_api_key:
        try:
            text = await _call_anthropic(system, messages, max_tokens)
            logger.debug("LLM: Anthropic OK")
            return text
        except Exception as e:
            if _should_fallback(e):
                logger.warning(f"Anthropic quota/credit issue — falling back to Groq. Error: {e}")
            else:
                # Re-raise non-quota errors (e.g. bad prompt, network error)
                raise

    # ── Fallback to Groq ─────────────────────────────────────────
    if settings.groq_api_key:
        try:
            text = await _call_groq(system, messages, max_tokens)
            logger.info("LLM: Groq fallback OK")
            return text
        except Exception as e:
            raise RuntimeError(f"Both Anthropic and Groq failed. Groq error: {e}")

    raise RuntimeError(
        "No LLM available. Set ANTHROPIC_API_KEY and/or GROQ_API_KEY."
    )
