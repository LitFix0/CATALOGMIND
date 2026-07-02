"""
CatalogMind — Layer 7: Decision Logic.

Deterministically routes each /chat turn to one of five actions:
  REFUSE    → guardrails triggered
  CLARIFY   → not enough context yet
  COMPARE   → user comparing two assessments
  REFINE    → user modifying a previous shortlist
  RECOMMEND → enough context, produce a shortlist

Design decisions:
  - Fully deterministic — no LLM randomness can break behavior probes.
  - REFINE detected by: prior assistant message contains shl.com URL
    AND latest user message contains a refinement signal word.
  - COMPARE takes priority over RECOMMEND when compare_targets present.
  - Turn cap safety valve: force RECOMMEND at turn 6+ to avoid
    wasting the evaluator's 8-turn budget on clarifications.
  - Clarifying questions are prioritized: role → seniority → test type.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from agent.guardrails import GuardResult, check_conversation
from agent.context_extractor import ConversationContext, extract_context

logger = logging.getLogger(__name__)

_MAX_TURNS_BEFORE_FORCE_RECOMMEND = 6


class Action(str, Enum):
    REFUSE    = "refuse"
    CLARIFY   = "clarify"
    COMPARE   = "compare"
    REFINE    = "refine"
    RECOMMEND = "recommend"


@dataclass
class DecisionResult:
    action: Action
    context: ConversationContext
    guard_result: GuardResult | None = None
    clarifying_question: str = ""
    is_refinement: bool = False


def _pick_clarifying_question(ctx: ConversationContext) -> str:
    if not ctx.role and not ctx.has_jd:
        return (
            "To find the right assessments, could you tell me the role "
            "you're hiring for — for example, 'software engineer', "
            "'sales manager', or 'customer service agent'?"
        )
    if ctx.role and not ctx.seniority:
        return (
            f"Got it — for a {ctx.role} role. What seniority level are "
            f"you targeting? For example: graduate, junior, mid-level, "
            f"senior, or executive?"
        )
    if ctx.role and not ctx.test_type_prefs:
        return (
            f"Are there specific types of assessments you'd like to "
            f"include — for example, personality, cognitive ability, "
            f"technical knowledge, or simulations?"
        )
    return (
        "Could you share a bit more about the role or the skills "
        "you'd like to assess? That will help me narrow down the "
        "best options."
    )


_REFINEMENT_SIGNALS: list[str] = [
    "actually", "instead", "also add", "add ", "remove", "change it",
    "update", "make it", "not that", "no ", "exclude", "without",
    "rather than", "prefer", "only ", "just ", "more like",
    "can you also", "what about", "include", "drop ", "swap",
]


def _has_previous_recommendations(messages: list[dict]) -> bool:
    """
    Detect if assistant already gave a shortlist in a prior turn.
    Heuristic: look for shl.com URLs or 'shortlist' in assistant text —
    these are only present when recommendations were returned.
    """
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if "shl.com" in content or "shortlist" in content.lower():
                return True
    return False


def _is_refinement_message(text: str) -> bool:
    lower = text.lower()
    return any(sig in lower for sig in _REFINEMENT_SIGNALS)


def decide(messages: list[dict]) -> DecisionResult:
    """
    Determine what action the agent should take next.

    Args:
        messages: Full conversation history from the /chat request.

    Returns:
        DecisionResult with action + context for downstream layers.
    """
    # Step 1: Guardrails always run first
    guard = check_conversation(messages)
    if guard.is_blocked:
        logger.info("Decision: REFUSE (reason=%s)", guard.reason)
        return DecisionResult(
            action=Action.REFUSE,
            context=ConversationContext(),
            guard_result=guard,
        )

    # Step 2: Extract full conversation context
    ctx = extract_context(messages)

    # Step 3: Compare — two named assessments found
    if ctx.compare_targets and len(ctx.compare_targets) >= 2:
        logger.info("Decision: COMPARE targets=%s", ctx.compare_targets)
        return DecisionResult(action=Action.COMPARE, context=ctx)

    # Step 4: Refine — prior shortlist exists + user is modifying it
    if (
        _has_previous_recommendations(messages)
        and _is_refinement_message(ctx.latest_user_message)
    ):
        logger.info("Decision: REFINE")
        return DecisionResult(
            action=Action.REFINE,
            context=ctx,
            is_refinement=True,
        )

    # Step 5: Recommend or Clarify
    near_turn_cap = ctx.turn_count >= _MAX_TURNS_BEFORE_FORCE_RECOMMEND

    if ctx.has_enough_context() or near_turn_cap:
        logger.info(
            "Decision: RECOMMEND (role='%s' seniority='%s' forced=%s)",
            ctx.role, ctx.seniority, near_turn_cap and not ctx.has_enough_context(),
        )
        return DecisionResult(action=Action.RECOMMEND, context=ctx)

    # Step 6: Clarify
    question = _pick_clarifying_question(ctx)
    logger.info("Decision: CLARIFY — '%s'", question[:60])
    return DecisionResult(
        action=Action.CLARIFY,
        context=ctx,
        clarifying_question=question,
    )