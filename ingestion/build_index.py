"""
CatalogMind — Layer 3: FAISS index builder.

What this script does:
  1. Loads data/shl_catalog.json (output of Layer 2)
  2. Embeds every item's full_text_for_embedding
  3. Builds a FAISS IndexFlatIP index (inner product = cosine sim
     because embeddings are L2-normalized in embeddings.py)
  4. Saves the index to data/faiss_index/catalog.index
  5. Saves an ID map to data/faiss_index/id_map.json so we can
     map FAISS result positions back to catalog items at query time

Design decisions:
  - IndexFlatIP over IndexIVFFlat:
      * 377 items is tiny — exact search is fast enough (~1ms)
      * IVF requires training and adds complexity for no real gain
      * If catalog grows to 10k+ items, swap to IVF then
  - id_map.json stores entity_id per FAISS position so retrieval
    layer can join back to full metadata without storing it in FAISS
  - Run this once at setup; re-run whenever shl_catalog.json changes
"""

import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.embeddings import embed_texts, EMBEDDING_DIM

logger = logging.getLogger(__name__)

CATALOG_PATH = PROJECT_ROOT / "data" / "shl_catalog.json"
INDEX_DIR = PROJECT_ROOT / "data" / "faiss_index"
INDEX_PATH = INDEX_DIR / "catalog.index"
ID_MAP_PATH = INDEX_DIR / "id_map.json"


def build_faiss_index(
    catalog_path: Path = CATALOG_PATH,
    index_path: Path = INDEX_PATH,
    id_map_path: Path = ID_MAP_PATH,
) -> None:
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Clean catalog not found at {catalog_path}.\n"
            "Run Layer 2 first: python -m scraper.clean_catalog"
        )

    logger.info("Loading catalog from %s", catalog_path)
    with open(catalog_path, encoding="utf-8") as f:
        catalog: list[dict] = json.load(f)
    logger.info("Loaded %d catalog items", len(catalog))

    texts = [item["full_text_for_embedding"] for item in catalog]
    id_map = {str(i): item["entity_id"] for i, item in enumerate(catalog)}

    logger.info("Generating embeddings...")
    embeddings: np.ndarray = embed_texts(texts)

    assert embeddings.shape == (len(catalog), EMBEDDING_DIM)

    logger.info("Building FAISS IndexFlatIP (dim=%d)...", EMBEDDING_DIM)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    logger.info("FAISS index built: %d vectors", index.ntotal)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("Saved FAISS index to %s", index_path)

    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2)
    logger.info("Saved id map to %s", id_map_path)

    print(f"\n✓ FAISS index built: {index.ntotal} vectors")
    print(f"  Index saved to: {index_path}")
    print(f"  ID map saved to: {id_map_path}")


def verify_index(
    index_path: Path = INDEX_PATH,
    id_map_path: Path = ID_MAP_PATH,
    catalog_path: Path = CATALOG_PATH,
) -> None:
    from ingestion.embeddings import embed_query

    index = faiss.read_index(str(index_path))
    with open(id_map_path, encoding="utf-8") as f:
        id_map = json.load(f)
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    entity_lookup = {item["entity_id"]: item for item in catalog}

    test_queries = [
        "personality test for sales manager",
        "cognitive ability test for graduate",
        "Java programming knowledge assessment",
        "leadership assessment for senior executive",
    ]

    print("\n--- Index verification (top 3 per query) ---")
    for query in test_queries:
        q_vec = embed_query(query)
        scores, indices = index.search(q_vec, 3)
        print(f"\nQuery: '{query}'")
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            entity_id = id_map[str(idx)]
            item = entity_lookup.get(entity_id, {})
            print(f"  {rank}. [{score:.3f}] {item.get('name', '?')} "
                  f"(type={item.get('test_type', '?')})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_faiss_index()
    verify_index()