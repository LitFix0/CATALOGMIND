"""
CatalogMind — Layer 4: Retriever.

Retrieval pipeline:
  1. Embed the query string
  2. Search FAISS for top_k * OVERQUERY_FACTOR candidates
  3. Map FAISS positions → catalog items via entity_lookup
  4. Apply optional metadata filters (test_type, remote, job_level)
  5. Return top_k items with similarity scores

Design decisions:
  - OVERQUERY_FACTOR=3: retrieve 3x more before filtering so filters
    don't starve the result set
  - Filters are post-retrieval — with 377 items exact search is <1ms
  - retrieve_by_name() used by comparator (Layer 10) to fetch specific
    items without going through semantic search
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from ingestion.embeddings import embed_query
from retrieval.faiss_store import store

logger = logging.getLogger(__name__)

OVERQUERY_FACTOR = 3


@dataclass
class RetrievalResult:
    """A single retrieved catalog item with its similarity score."""
    item: dict
    score: float


@dataclass
class RetrievalFilters:
    """
    Optional metadata filters applied after FAISS retrieval.
    All filters are AND logic.

    test_types:  keep items whose test_type contains ANY of these codes
    remote_only: keep only remote_testing == True items
    job_level:   case-insensitive substring match against job_levels list
    """
    test_types: list[str] = field(default_factory=list)
    remote_only: bool = False
    job_level: str = ""


def _ensure_loaded() -> None:
    if not store.is_ready:
        store.load()


def _apply_filters(
    results: list[RetrievalResult],
    filters: RetrievalFilters,
) -> list[RetrievalResult]:
    filtered = []
    for result in results:
        item = result.item

        if filters.test_types:
            item_codes = [c.strip() for c in item.get("test_type", "").split(",")]
            if not any(code in item_codes for code in filters.test_types):
                continue

        if filters.remote_only and not item.get("remote_testing", False):
            continue

        if filters.job_level:
            levels_lower = [lvl.lower() for lvl in item.get("job_levels", [])]
            if not any(filters.job_level.lower() in lvl for lvl in levels_lower):
                continue

        filtered.append(result)
    return filtered


def retrieve(
    query: str,
    top_k: int = 20,
    filters: RetrievalFilters | None = None,
) -> list[RetrievalResult]:
    """
    Retrieve top_k most relevant catalog items for a query.

    Args:
        query:   Natural language query string.
        top_k:   Number of results to return after filtering.
        filters: Optional metadata filters.

    Returns:
        List of RetrievalResult sorted by descending score, len <= top_k.
    """
    _ensure_loaded()

    if not store.is_ready:
        raise RuntimeError(
            "FAISS index not loaded. Run: python -m ingestion.build_index"
        )

    fetch_k = min(top_k * OVERQUERY_FACTOR, store.index.ntotal)

    q_vec: np.ndarray = embed_query(query)
    scores, indices = store.index.search(q_vec, fetch_k)

    results: list[RetrievalResult] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        entity_id = store.id_map.get(str(idx))
        if entity_id is None:
            continue
        item = store.entity_lookup.get(entity_id)
        if item is None:
            continue
        results.append(RetrievalResult(item=item, score=float(score)))

    if filters:
        results = _apply_filters(results, filters)

    return results[:top_k]


def retrieve_by_name(name: str) -> dict | None:
    """
    Fetch a catalog item by exact or fuzzy name match.
    Used by the comparator (Layer 10).

    Pass 1: exact match (case-insensitive)
    Pass 2: substring match (case-insensitive)
    """
    _ensure_loaded()
    name_lower = name.lower().strip()

    for item in store.catalog:
        if item["name"].lower() == name_lower:
            return item

    for item in store.catalog:
        if name_lower in item["name"].lower():
            return item

    logger.warning("No catalog item found for name: '%s'", name)
    return None


def retrieve_by_names(names: list[str]) -> list[dict | None]:
    """Fetch multiple items by name. Used by comparator."""
    return [retrieve_by_name(name) for name in names]


def get_all_items() -> list[dict]:
    """Return all catalog items."""
    _ensure_loaded()
    return store.catalog