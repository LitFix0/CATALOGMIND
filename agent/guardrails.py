"""
CatalogMind — Layer 5: Guardrails.

Design decisions:
  - All guardrail checks are DETERMINISTIC (no LLM involved).
    This is intentional: LLM-based content moderation adds latency,
    can be jailbroken, and is overkill for a scoped domain tool.
    Simple keyword + pattern matching is fast, predictable, and
    fully defensible in a code review.

  - Guard checks run BEFORE any LLM or FAISS call in the pipeline.
    Failing fast saves tokens and keeps p99 latency well under the
    30-second evaluator timeout.

  - We return a GuardResult dataclass (not just bool) so the caller
    gets both the verdict AND a ready-made reply string. This keeps
    decision.py clean — it just checks result.is_blocked.

  - Off-topic detection uses two layers:
      1. Hard keyword blocklist — fast exact match
      2. Topic heuristic — checks if message contains ANY SHL/
         assessment signal; if not, treats as off-topic

  - Prompt injection detection looks for instruction-override
    patterns. We intentionally keep this broad — false positives
    (refusing a legit message) are safer than false negatives
    (executing an injection).
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class GuardResult:
    """
    Result of a guardrail check.

    Attributes:
        is_blocked: True if the message should be refused.
        reason:     Short reason code (for logging/testing).
        reply:      Ready-made user-facing reply if blocked.
    """
    is_blocked: bool
    reason: str = ""
    reply: str = ""


# --------------------------------------------------------------------------- #
# Keyword lists                                                                #
# --------------------------------------------------------------------------- #

_OFF_TOPIC_KEYWORDS: list[str] = [
    # Legal
    "legal advice", "lawsuit", "sue ", "suing", "litigation",
    "discrimination", "wrongful termination", "labor law", "employment law",
    "gdpr", "data protection law", "privacy law",
    # Salary / compensation
    "salary", "compensation", "pay range", "wage", "bonus", "benefits package",
    "equity", "stock options",
    # General hiring advice
    "how to write a job description", "write a job description",
    "job description template", "how to hire", "interview questions",
    "interview tips", "onboarding", "background check", "reference check",
    "how to fire", "termination letter", "performance improvement plan",
    # Completely unrelated
    "weather", "stock price", "crypto", "recipe", "sports",
    "movie", "restaurant", "travel", "visa application",
]

_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(your\s+)?(previous\s+)?instructions",
    r"forget\s+(everything|all|your\s+instructions)",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(a\s+)?(?!an?\s+shl)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"your\s+(new\s+)?role\s+is",
    r"system\s*:\s*you",
    r"<\s*system\s*>",
    r"\[system\]",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"override\s+(safety|instructions|guidelines)",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|prompt)",
    r"print\s+(your\s+)?(system\s+prompt|instructions)",
]

_SHL_SIGNALS: list[str] = [
    "assessment", "test", "shl", "opq", "verify", "personality",
    "cognitive", "ability", "aptitude", "competency", "competencies",
    "behavioral", "situational judgment", "simulation", "hiring",
    "candidate", "recruit", "role", "position", "job", "skill",
    "developer", "engineer", "manager", "analyst", "graduate",
    "senior", "junior", "mid-level", "entry", "leader", "executive",
    "measure", "evaluate", "shortlist", "recommend", "catalog",
    "psychometric", "adaptive", "remote testing", "irt",
    "knowledge", "technical", "sales", "customer service", "clerical",
    "finance", "accounting", "data", "python", "java", "sql",
]

# Messages shorter than this skip the no-signal topic check
# (greetings like "Hi" or "Hello, I need help" should not be blocked)
_MIN_LENGTH_FOR_TOPIC_CHECK = 25


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _contains_keyword(text: str, keywords: list[str]) -> str | None:
    for kw in keywords:
        if kw in text:
            return kw
    return None


def _matches_injection_pattern(text: str) -> str | None:
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return pattern
    return None


def _has_shl_signal(text: str) -> bool:
    return any(signal in text for signal in _SHL_SIGNALS)


# --------------------------------------------------------------------------- #
# Public guard functions                                                       #
# --------------------------------------------------------------------------- #

def check_prompt_injection(text: str) -> GuardResult:
    """Detect instruction-override / prompt injection attempts."""
    normalised = _normalise(text)
    pattern = _matches_injection_pattern(normalised)
    if pattern:
        logger.warning("Prompt injection detected. Pattern: %s", pattern)
        return GuardResult(
            is_blocked=True,
            reason="prompt_injection",
            reply=(
                "I'm not able to process that request. "
                "I'm CatalogMind, and I only help with selecting "
                "SHL assessments. How can I help you find the right "
                "assessment for your role?"
            ),
        )
    return GuardResult(is_blocked=False)


def check_off_topic(text: str) -> GuardResult:
    """
    Detect messages outside the SHL assessment domain.

    Two-stage check:
      1. Hard keyword match against known off-topic topics.
      2. If message is long enough and has NO SHL domain signal → off-topic.
    """
    normalised = _normalise(text)

    matched_kw = _contains_keyword(normalised, _OFF_TOPIC_KEYWORDS)
    if matched_kw:
        logger.info("Off-topic keyword matched: '%s'", matched_kw)
        return GuardResult(
            is_blocked=True,
            reason="off_topic_keyword",
            reply=(
                "That's outside what I can help with. I'm focused on "
                "recommending SHL assessments for specific roles and skills. "
                "Feel free to describe a role you're hiring for and I'll "
                "suggest the right assessments."
            ),
        )

    if len(normalised) >= _MIN_LENGTH_FOR_TOPIC_CHECK:
        if not _has_shl_signal(normalised):
            logger.info("No SHL signal found in message: '%s'", normalised[:80])
            return GuardResult(
                is_blocked=True,
                reason="off_topic_no_signal",
                reply=(
                    "I can only help with SHL assessment recommendations. "
                    "Could you tell me about the role you're hiring for or "
                    "the skills you'd like to assess?"
                ),
            )

    return GuardResult(is_blocked=False)


def check_message(text: str) -> GuardResult:
    """
    Run all guardrail checks in priority order:
      1. Prompt injection (highest priority)
      2. Off-topic content

    Args:
        text: The latest user message to check.

    Returns:
        GuardResult — check .is_blocked to decide whether to refuse.
    """
    result = check_prompt_injection(text)
    if result.is_blocked:
        return result

    result = check_off_topic(text)
    if result.is_blocked:
        return result

    return GuardResult(is_blocked=False, reason="pass")


def check_conversation(messages: list[dict]) -> GuardResult:
    """
    Run guardrails on the latest user message from conversation history.

    Design decision: only check the LATEST user message, not the whole
    history. Earlier messages were already checked when they were current.
    Checking full history risks false positives from the assistant's own
    refusal replies containing blocked keywords.

    Args:
        messages: Full conversation history (list of role/content dicts).

    Returns:
        GuardResult from checking the latest user message.
    """
    latest_user_msg = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )

    if not latest_user_msg:
        return GuardResult(is_blocked=False, reason="no_user_message")

    return check_message(latest_user_msg)