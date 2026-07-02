"""
CatalogMind — Layer 11: Groq LLM Client.

Design decisions:
  - Raw HTTP via `requests`, not the Groq SDK. One fewer dependency,
    and every request/response detail is visible for interview defense.
  - Never raises. Any failure (timeout, network error, bad status,
    malformed JSON, empty content) returns None. The caller ALWAYS
    has a deterministic fallback string ready — the schema must
    never break because Groq is down or slow.
  - Hard timeout well under the evaluator's 30s/turn budget. We'd
    rather fail fast and use a fallback than hang near the limit.
  - Low max_tokens + low temperature: we only want a short, grounded
    natural-language blurb, not creative writing. Low temperature
    also reduces (never eliminates) the chance of the model inventing
    details, which is why prompts.py must ever supply only catalog
    metadata and comparator/recommender ground truth as the LLM's
    "world" — this client has no idea what's true, prompts.py does.
"""


import os
import logging
import requests

logger = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = "llama-3.3-70b-versatile"
_TIMEOUT_SECONDS = 8.0
_MAX_TOKENS = 300
_TEMPERATURE = 0.3


def call_groq(system_prompt: str, user_prompt: str) -> str | None:
    """
    Call the Groq chat completion API.

    Args:
        system_prompt: Instructions + grounding context for the model.
        user_prompt:   The user-facing task for this specific turn.

    Returns:
        The generated reply text, or None if anything went wrong.
        Callers MUST handle None with a deterministic fallback —
        this function intentionally never raises.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set — skipping LLM call.")
        return None

    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": _MAX_TOKENS,
        "temperature": _TEMPERATURE,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            _GROQ_URL,
            json=payload,
            headers=headers,
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        logger.error("Groq call timed out after %.1fs", _TIMEOUT_SECONDS)
        return None
    except requests.exceptions.RequestException as e:
        logger.error("Groq call failed: %s", e)
        return None

    if resp.status_code != 200:
        logger.error(
            "Groq returned status %s: %s", resp.status_code, resp.text[:200]
        )
        return None

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        logger.error("Groq response malformed: %s", e)
        return None

    content = content.strip()
    if not content:
        logger.warning("Groq returned empty content.")
        return None

    return content