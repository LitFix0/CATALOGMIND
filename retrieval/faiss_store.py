"""
CatalogMind — Layer 4: FAISS store (singleton index loader).

Design decision: load the index, id_map, and full catalog metadata
ONCE at module import time and hold them in memory.

Why a singleton store instead of loading per-request?
  - FAISS index for 377 items is ~600KB in memory — negligible
  - shl_catalog.json is ~400KB — also negligible
  - Loading from disk on every /chat request would add 50-200ms
    latency, which matters given the 30-second evaluator timeout
  - A module-level singleton is the simplest correct solution here;
    more complex caching (Redis, etc.) would be over-engineering
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

CATALOG_PATH  = PROJECT_ROOT / "data" / "shl_catalog.json"
INDEX_PATH    = PROJECT_ROOT / "data" / "faiss_index" / "catalog.index"
ID_MAP_PATH   = PROJECT_ROOT / "data" / "faiss_index" / "id_map.json"


class FAISSStore:
    """
    Holds the FAISS index, id_map, and catalog metadata in memory.

    Attributes:
        index:         faiss.IndexFlatIP — the vector index
        id_map:        dict[str, str] — FAISS position → entity_id
        catalog:       list[dict] — all 377 normalized catalog items
        entity_lookup: dict[str, dict] — entity_id → catalog item
    """

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.id_map: dict[str, str] = {}
        self.catalog: list[dict] = []
        self.entity_lookup: dict[str, dict] = {}
        self._loaded = False

    def load(
        self,
        index_path: Path = INDEX_PATH,
        id_map_path: Path = ID_MAP_PATH,
        catalog_path: Path = CATALOG_PATH,
    ) -> None:
        """Load index + catalog from disk. Safe to call multiple times."""
        if self._loaded:
            return

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}.\n"
                "Run Layer 3 first: python -m ingestion.build_index"
            )

        logger.info("Loading FAISS index from %s", index_path)
        self.index = faiss.read_index(str(index_path))
        logger.info("FAISS index loaded: %d vectors", self.index.ntotal)

        with open(id_map_path, encoding="utf-8") as f:
            self.id_map = json.load(f)

        with open(catalog_path, encoding="utf-8") as f:
            self.catalog = json.load(f)

        self.entity_lookup = {
            item["entity_id"]: item for item in self.catalog
        }

        self._loaded = True
        logger.info(
            "FAISSStore ready: %d items, %d vectors",
            len(self.catalog), self.index.ntotal,
        )

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.index is not None


# Module-level singleton — imported by retriever.py
store = FAISSStore()