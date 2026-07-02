"""
CatalogMind — Layer 11: Orchestrator.

Single entry point for /chat. Routes through all layers and
returns a strict-schema dict every time.

Flow:
  messages
    → goodbye check   → eoc=True if thank-you/bye
    → decide()        → REFUSE / CLARIFY / COMPARE / REFINE / RECOMMEND
    → REFUSE          → deterministic reply, no LLM
    → CLARIFY         → deterministic reply, no LLM
    → COMPARE         → comparator() + LLM narration (fallback if None)
    → REFINE          → recommender() + LLM narration (fallback if None)
    → RECOMMEND       → recommender() + LLM narration (fallback if None)

Design decisions:
  - CLARIFY and REFUSE never call the LLM — saves quota, faster,
    deterministic text cannot be hallucinated.
  - call_groq() returning None → fallback text, recommendations
    still returned normally. Schema never breaks.
  - end_of_conversation=True only on goodbye/thank-you. Never set
    on recommendation turns — evaluator expects refinement to work.
  - Top-level try/except ensures schema is always valid even on
    completely unexpected errors.
"""

import logging
from typing import Any

from agent.decision import decide, Action, DecisionResult
from agent.recommender import recommend
from agent.comparator import compare_from_targets
from agent.prompts import build_recommend_prompt, build_compare_prompt
from agent.llm_client import call_groq
from agent.context_extractor import ConversationContext
from schemas.chat_schema import Recommendation

logger = logging.getLogger(__name__)

_GOODBYE_SIGNALS: list[str] = [
    "thank you", "thanks", "that's all", "that is all",
    "goodbye", "bye", "done", "perfect", "great, thanks",
    "no more", "i'm good", "i am good", "all good",
    "that's helpful", "that helps", "got it, thanks",
    "no further", "nothing else", "that'll do",
]


def _is_goodbye(text: str) -> bool:
    lower = text.lower().strip()
    return any(sig in lower for sig in _GOODBYE_SIGNALS)


def _fallback_recommend_reply(ctx: ConversationContext, count: int, is_refinement: bool) -> str:
    role_str = f" for the {ctx.role} role" if ctx.role else ""
    action = "updated shortlist" if is_refinement else "shortlist"
    return (
        f"Here is your {action}{role_str} — "
        f"{count} assessment{'s' if count != 1 else ''} from the SHL catalog. "
        f"Let me know if you'd like to refine this further."
    )


def _fallback_compare_reply(name_a: str, name_b: str) -> str:
    return (
        f"Here is a grounded comparison of {name_a} and {name_b} "
        f"based on the SHL catalog data. "
        f"Let me know if you have further questions."
    )


def _handle_refuse(result: DecisionResult) -> dict[str, Any]:
    return {
        "reply": result.guard_result.reply,
        "recommendations": [],
        "end_of_conversation": False,
    }


def _handle_clarify(result: DecisionResult) -> dict[str, Any]:
    return {
        "reply": result.clarifying_question,
        "recommendations": [],
        "end_of_conversation": False,
    }


def _handle_recommend(
    result: DecisionResult,
    messages: list[dict],
    is_refinement: bool = False,
) -> dict[str, Any]:
    ctx = result.context
    recommendations: list[Recommendation] = recommend(ctx)

    if not recommendations:
        return {
            "reply": (
                "I wasn't able to find matching assessments right now. "
                "Please try rephrasing your request or check back shortly."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    try:
        system_prompt, user_prompt = build_recommend_prompt(
            ctx=ctx,
            recommendations=recommendations,
            is_refinement=is_refinement,
        )
        llm_reply = call_groq(system=system_prompt, user=user_prompt)
    except Exception as e:
        logger.warning("Prompt/LLM call failed: %s", e)
        llm_reply = None

    reply = llm_reply or _fallback_recommend_reply(ctx, len(recommendations), is_refinement)

    return {
        "reply": reply,
        "recommendations": [r.model_dump() for r in recommendations],
        "end_of_conversation": False,
    }


def _handle_compare(result: DecisionResult, messages: list[dict]) -> dict[str, Any]:
    ctx = result.context
    comparison = compare_from_targets(ctx.compare_targets)

    if not comparison.found:
        return {
            "reply": comparison.fallback_msg,
            "recommendations": [],
            "end_of_conversation": False,
        }

    try:
        system_prompt, user_prompt = build_compare_prompt(comparison=comparison)
        llm_reply = call_groq(system=system_prompt, user=user_prompt)
    except Exception as e:
        logger.warning("Compare prompt/LLM call failed: %s", e)
        llm_reply = None

    reply = llm_reply or _fallback_compare_reply(comparison.name_a, comparison.name_b)

    return {
        "reply": reply,
        "recommendations": [],  # per spec: empty unless shortlist also requested
        "end_of_conversation": False,
    }


def handle(messages: list[dict]) -> dict[str, Any]:
    """
    Process conversation history and return strict-schema response dict.

    Args:
        messages: Full conversation history from /chat request.

    Returns:
        {"reply": str, "recommendations": list, "end_of_conversation": bool}
        Schema is always valid — never raises.
    """
    latest_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    if _is_goodbye(latest_user):
        return {
            "reply": (
                "You're welcome! Good luck with your hiring. "
                "Feel free to come back anytime you need assessment recommendations."
            ),
            "recommendations": [],
            "end_of_conversation": True,
        }

    try:
        result: DecisionResult = decide(messages)

        if result.action == Action.REFUSE:
            return _handle_refuse(result)
        if result.action == Action.CLARIFY:
            return _handle_clarify(result)
        if result.action == Action.COMPARE:
            return _handle_compare(result, messages)
        if result.action == Action.REFINE:
            return _handle_recommend(result, messages, is_refinement=True)
        if result.action == Action.RECOMMEND:
            return _handle_recommend(result, messages, is_refinement=False)

        logger.error("Unknown action: %s", result.action)
        return {
            "reply": "Could you tell me about the role you're hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    except Exception as e:
        logger.error("Orchestrator error: %s", type(e).__name__, exc_info=True)
        return {
            "reply": "Something went wrong on my end. Please try rephrasing your request.",
            "recommendations": [],
            "end_of_conversation": False,
        }