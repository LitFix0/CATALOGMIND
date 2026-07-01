"""
CatalogMind — Layer 2: Catalog loader and normalizer.

Design decision: SHL provides the catalog as a ready-made JSON at
    https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json
so we do NOT need a live scraper. Instead this module:
  1. Loads the raw SHL-provided JSON
  2. Normalizes every entry into our internal CatalogItem schema
  3. Derives test_type codes from the 'keys' field
  4. Builds full_text_for_embedding (used by FAISS in later layers)
  5. Writes the cleaned output to data/shl_catalog.json

Why keep a separate clean step instead of using raw JSON directly?
  - The raw JSON has redundant fields (job_levels_raw, languages_raw)
    we don't want to embed noise into FAISS vectors.
  - We need derived fields (test_type, full_text_for_embedding) that
    don't exist in the raw data.
  - Decoupling load/clean from retrieval means the FAISS layer never
    touches raw data — only our well-defined schema.

test_type code mapping (derived from sample_conversations analysis):
  A → Ability & Aptitude
  P → Personality & Behavior
  K → Knowledge & Skills
  B → Biodata & Situational Judgment
  S → Simulations
  C → Competencies
  D → Development & 360
  E → Assessment Exercises
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# test_type code mapping — derived from all 10 sample conversations            #
# --------------------------------------------------------------------------- #

_KEY_TO_CODE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Knowledge & Skills": "K",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

# Default paths — callers can override
RAW_CATALOG_PATH = Path(__file__).parent.parent / "data" / "shl_product_catalog_raw.json"
CLEAN_CATALOG_PATH = Path(__file__).parent.parent / "data" / "shl_catalog.json"


# --------------------------------------------------------------------------- #
# Core normalization                                                            #
# --------------------------------------------------------------------------- #

def derive_test_type(keys: list[str]) -> str:
    """
    Convert the raw 'keys' list to a comma-separated test_type code string.

    Examples:
      ["Personality & Behavior"]                              → "P"
      ["Knowledge & Skills", "Simulations"]                  → "K,S"
      ["Biodata & Situational Judgment", "Simulations"]      → "B,S"
      []                                                      → "K"   (safe default)
    """
    codes = [_KEY_TO_CODE[k] for k in keys if k in _KEY_TO_CODE]
    return ",".join(codes) if codes else "K"


def build_full_text(item: dict[str, Any]) -> str:
    """
    Build a single rich text blob for embedding.

    Design decision: we concatenate every semantically useful field so
    the embedding captures name, description, job levels, keys, and
    language availability in one vector. This gives better semantic
    recall than embedding name alone — a query for "sales manager
    personality test" should surface items even when the word 'sales'
    only appears in job_levels or description.
    """
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        "Job levels: " + ", ".join(item.get("job_levels", [])),
        "Keys: " + ", ".join(item.get("keys", [])),
        "Languages: " + ", ".join(item.get("languages", [])),
        "Remote testing: " + str(item.get("remote_testing", "")),
        "Adaptive/IRT: " + str(item.get("adaptive_irt", "")),
        "Duration: " + item.get("duration", ""),
    ]
    return " | ".join(p for p in parts if p.strip(" |"))


def normalize_item(raw: dict[str, Any]) -> dict[str, Any] | None:
    """
    Normalize one raw catalog entry into our CatalogItem schema.
    Returns None if the entry is unusable (missing name or link).
    """
    name = raw.get("name", "").strip()
    link = raw.get("link", "").strip()

    if not name or not link:
        logger.warning("Skipping entry with missing name or link: %s", raw.get("entity_id"))
        return None

    # Skip entries that didn't scrape cleanly
    if raw.get("status", "ok") != "ok":
        logger.warning("Skipping entry with non-ok status: %s", name)
        return None

    keys = raw.get("keys") or []
    job_levels = raw.get("job_levels") or []
    languages = raw.get("languages") or []

    normalized = {
        "entity_id": str(raw.get("entity_id", "")),
        "name": name,
        "url": link,
        "test_type": derive_test_type(keys),
        "description": raw.get("description", "").strip(),
        "duration": raw.get("duration", "").strip(),
        "remote_testing": raw.get("remote", "").strip().lower() == "yes",
        "adaptive_irt": raw.get("adaptive", "").strip().lower() == "yes",
        "job_levels": job_levels,
        "languages": languages,
        "keys": keys,
        "full_text_for_embedding": "",   # filled below
    }

    normalized["full_text_for_embedding"] = build_full_text(normalized)
    return normalized


def load_and_normalize(
    raw_path: Path = RAW_CATALOG_PATH,
    output_path: Path = CLEAN_CATALOG_PATH,
) -> list[dict[str, Any]]:
    """
    Load the raw SHL catalog JSON, normalize every entry, write
    shl_catalog.json, and return the normalized list.
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw catalog not found at {raw_path}.\n"
            "Download it from:\n"
            "  https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json\n"
            "and save it as data/shl_product_catalog_raw.json"
        )

    logger.info("Loading raw catalog from %s", raw_path)
    with open(raw_path, encoding="utf-8") as f:
        raw_items: list[dict] = json.load(f, strict=False)

    logger.info("Raw catalog has %d entries", len(raw_items))

    normalized: list[dict[str, Any]] = []
    skipped = 0

    for raw in raw_items:
        item = normalize_item(raw)
        if item is None:
            skipped += 1
        else:
            normalized.append(item)

    logger.info("Normalized %d entries, skipped %d", len(normalized), skipped)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    logger.info("Wrote clean catalog to %s", output_path)
    return normalized


# --------------------------------------------------------------------------- #
# CLI entry point: python3 -m scraper.clean_catalog                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    items = load_and_normalize()
    print(f"\n✓ {len(items)} assessments written to {CLEAN_CATALOG_PATH}")

    # Quick sanity stats
    from collections import Counter
    type_counts = Counter()
    for item in items:
        for code in item["test_type"].split(","):
            type_counts[code.strip()] += 1

    print("\ntest_type distribution:")
    for code, count in sorted(type_counts.items()):
        print(f"  {code}: {count}")

    remote_count = sum(1 for i in items if i["remote_testing"])
    adaptive_count = sum(1 for i in items if i["adaptive_irt"])
    print(f"\nRemote testing available: {remote_count}/{len(items)}")
    print(f"Adaptive/IRT:             {adaptive_count}/{len(items)}")