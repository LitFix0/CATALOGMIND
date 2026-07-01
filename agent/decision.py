"""
Layer 1 placeholder decision logic.

This is intentionally crude: a keyword-based classifier so we can stand up
the FastAPI contract and test it end-to-end. It will be replaced in a later
layer by:
  - agent/context_extractor.py (pulls role/skills/seniority/etc. from history)
  - agent/guardrails.py (real refusal logic)
  - agent/decision.py (real deterministic routing using extracted context)

Keeping this isolated in its own function (`classify_intent`) means swapping
it out later is a one-function change, not a rewrite of app.py.
"""

from enum import Enum


class Intent(str, Enum):
    REFUSE = "refuse"
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    COMPARE = "compare"


_OFF_TOPIC_MARKERS = [
    "salary", "legal advice", "lawsuit", "sue", "interview questions",
    "visa", "immigration", "weather", "stock price",
]

_PROMPT_INJECTION_MARKERS = [
    "ignore previous instructions", "ignore all previous", "system prompt",
    "you are now", "disregard your instructions", "act as",
]

_COMPARE_MARKERS = ["difference between", "compare", "vs ", " versus "]

# Minimal signal that the user gave *some* job/skill context to act on.
_CONTEXT_MARKERS = [
    "developer", "engineer", "manager", "analyst", "sales", "java", "python",
    "personality", "cognitive", "leadership", "clerical", "customer service",
    "graduate", "senior", "junior", "mid-level", "entry-level",
]


def classify_intent(latest_user_message: str, full_history_text: str) -> Intent:
    text = latest_user_message.lower()

    if any(marker in text for marker in _PROMPT_INJECTION_MARKERS):
        return Intent.REFUSE

    if any(marker in text for marker in _OFF_TOPIC_MARKERS):
        return Intent.REFUSE

    if any(marker in text for marker in _COMPARE_MARKERS):
        return Intent.COMPARE

    # Look at the *whole* conversation so far, not just the latest message —
    # a user might give context piecemeal across turns.
    history_lower = full_history_text.lower()
    if any(marker in history_lower for marker in _CONTEXT_MARKERS):
        return Intent.RECOMMEND

    return Intent.CLARIFY