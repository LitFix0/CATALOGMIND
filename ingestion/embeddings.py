"""
CatalogMind — Layer 3: Embedding generator.

Design decisions:
  - Model: sentence-transformers/all-MiniLM-L6-v2
      * 384-dimensional embeddings — small enough to be fast on CPU
      * Strong semantic similarity performance for short-to-medium text
      * Fits within Render's free tier memory limits
  - We embed full_text_for_embedding (built in Layer 2), not just name,
    so semantic queries like "sales personality test" can match items
    where 'sales' only appears in job_levels or description.
  - Model is loaded once as a module-level singleton — this avoids
    reloading the 80MB model on every /chat request at runtime.
  - batch_size=64 keeps memory usage flat regardless of catalog size.
"""

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384   # fixed for all-MiniLM-L6-v2


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.

    lru_cache(maxsize=1) ensures the model is loaded exactly once per
    process — safe for both the build script and the FastAPI runtime.
    """
    logger.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded (dim=%d)", EMBEDDING_DIM)
    return model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of strings and return a float32 numpy array
    of shape (len(texts), EMBEDDING_DIM).

    Args:
        texts:      List of strings to embed.
        batch_size: Number of texts per encoding batch. 64 is a safe
                    default for CPU — reduce if you hit OOM errors.

    Returns:
        np.ndarray of shape (N, 384), dtype float32.
    """
    model = get_model()
    logger.info("Embedding %d texts (batch_size=%d)", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
    )

    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string for retrieval.

    Returns shape (1, 384) float32 array — FAISS expects 2D input.
    """
    model = get_model()
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.astype(np.float32)