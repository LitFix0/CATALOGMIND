"""
CatalogMind — Layer 8: Recommender + Reranker.

Pipeline:
  1. Build a rich search query from ConversationContext (Layer 6)
  2. Retrieve top 20 candidates from FAISS (Layer 4)
  3. Rerank using a multi-signal scoring function
  4. Return top 1-10 as Recommendation objects (schema Layer 1)

Reranking signals:
  +0.30  test_type match   — user explicitly asked for this type
  +0.20  skill match       — item name/text mentions a user skill
  +0.15  seniority match   — item job_levels matches user seniority
  +0.10  remote match      — user asked remote, item supports it
  +0.05  has_description   — prefer items with richer metadata
  base   FAISS cosine score — semantic similarity (0–1 range)

Design decisions:
  - Two-stage: FAISS for recall, reranker for business-logic precision.
  - Hard filters only for explicit user requirements (remote only).
    test_type preference handled by scoring, not exclusion.
  - All URLs come directly from shl_catalog.json — never from LLM.
  - Cap at 10, minimum 1 per the assignment schema contract.
"""

import logging
from dataclasses import dataclass

from agent.context_extractor import ConversationContext
from retrieval.retriever import retrieve, RetrievalFilters, RetrievalResult
from schemas.chat_schema import Recommendation

logger = logging.getLogger(__name__)

_FAISS_RETRIEVE_K    = 20
_MAX_RECOMMENDATIONS = 10
_MIN_RECOMMENDATIONS = 1

_W_TEST_TYPE   = 0.30
_W_SKILL       = 0.20
_W_SENIORITY   = 0.15
_W_REMOTE      = 0.10
_W_DESCRIPTION = 0.05

_SENIORITY_TO_LEVEL: dict[str, list[str]] = {
    "graduate":  ["graduate", "entry-level", "general population"],
    "junior":    ["entry-level", "general population", "mid-professional"],
    "mid":       ["mid-professional", "professional individual contributor",
                  "general population"],
    "senior":    ["manager", "mid-professional", "professional individual contributor",
                  "front line manager", "supervisor"],
    "director":  ["director", "manager"],
    "executive": ["executive", "director"],
}


@dataclass
class ScoredResult:
    item: dict
    faiss_score: float
    rerank_score: float

    @property
    def total_score(self) -> float:
        return self.faiss_score + self.rerank_score


def _score_test_type(item: dict, ctx: ConversationContext) -> float:
    if not ctx.test_type_prefs:
        return 0.0
    item_codes = {c.strip() for c in item.get("test_type", "").split(",")}
    matches = sum(1 for code in ctx.test_type_prefs if code in item_codes)
    return _W_TEST_TYPE * (matches / max(len(ctx.test_type_prefs), 1))


def _score_skill(item: dict, ctx: ConversationContext) -> float:
    if not ctx.skills:
        return 0.0
    full_text = item.get("full_text_for_embedding", "").lower()
    name = item.get("name", "").lower()
    matches = sum(
        1 for skill in ctx.skills
        if skill in full_text or skill in name
    )
    return _W_SKILL * min(matches / max(len(ctx.skills), 1), 1.0)


def _score_seniority(item: dict, ctx: ConversationContext) -> float:
    if not ctx.seniority:
        return 0.0
    target_levels = _SENIORITY_TO_LEVEL.get(ctx.seniority, [])
    if not target_levels:
        return 0.0
    item_levels = [lvl.lower() for lvl in item.get("job_levels", [])]
    if any(tgt in lvl for tgt in target_levels for lvl in item_levels):
        return _W_SENIORITY
    return 0.0


def _score_remote(item: dict, ctx: ConversationContext) -> float:
    if ctx.remote_required and item.get("remote_testing", False):
        return _W_REMOTE
    return 0.0


def _score_description(item: dict) -> float:
    return _W_DESCRIPTION if item.get("description", "").strip() else 0.0


def _rerank_score(item: dict, ctx: ConversationContext) -> float:
    return (
        _score_test_type(item, ctx)
        + _score_skill(item, ctx)
        + _score_seniority(item, ctx)
        + _score_remote(item, ctx)
        + _score_description(item)
    )


def _build_filters(ctx: ConversationContext) -> RetrievalFilters:
    """Only hard-filter on explicit user requirements."""
    return RetrievalFilters(
        remote_only=ctx.remote_required,
        test_types=[],
        job_level="",
    )


def recommend(
    ctx: ConversationContext,
    top_k: int = _MAX_RECOMMENDATIONS,
) -> list[Recommendation]:
    """
    Produce a ranked shortlist of SHL assessments for a given context.

    Args:
        ctx:   ConversationContext from the context extractor.
        top_k: Max recommendations to return (capped at 10).

    Returns:
        List of Recommendation (name, url, test_type), len 1-10.
        Empty list only if FAISS is unavailable.
    """
    query = ctx.build_search_query()
    logger.info("Recommending for query: '%s'", query[:100])

    try:
        filters = _build_filters(ctx)
        results: list[RetrievalResult] = retrieve(
            query=query,
            top_k=_FAISS_RETRIEVE_K,
            filters=filters,
        )
    except RuntimeError as e:
        logger.error("FAISS retrieval failed: %s", e)
        return []

    if not results:
        logger.warning("No FAISS results for query: '%s'", query)
        return []

    scored: list[ScoredResult] = [
        ScoredResult(
            item=r.item,
            faiss_score=r.score,
            rerank_score=_rerank_score(r.item, ctx),
        )
        for r in results
    ]
    scored.sort(key=lambda x: x.total_score, reverse=True)

    for s in scored[:5]:
        logger.debug(
            "  [%.3f + %.3f = %.3f] %s (type=%s)",
            s.faiss_score, s.rerank_score, s.total_score,
            s.item.get("name", "?"), s.item.get("test_type", "?"),
        )

    capped_k = max(min(top_k, _MAX_RECOMMENDATIONS), _MIN_RECOMMENDATIONS)
    top = scored[:capped_k]

    recommendations = [
        Recommendation(
            name=s.item["name"],
            url=s.item["url"],
            test_type=s.item["test_type"],
        )
        for s in top
    ]

    logger.info(
        "Returning %d recommendations (top: %s)",
        len(recommendations),
        recommendations[0].name if recommendations else "none",
    )
    return recommendations