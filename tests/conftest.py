"""
CatalogMind — pytest conftest.py
Shared fixtures with FAISS + embeddings + LLM fully mocked.
"""

import json
import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_CATALOG = [
    {
        "entity_id": "1001",
        "name": "Occupational Personality Questionnaire OPQ32r",
        "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "test_type": "P",
        "description": "Measures 32 personality dimensions for leadership and talent assessment.",
        "duration": "25",
        "remote_testing": True,
        "adaptive_irt": False,
        "job_levels": ["Director", "Executive", "Manager", "Mid-Professional"],
        "languages": ["English International", "French"],
        "keys": ["Personality & Behavior"],
        "full_text_for_embedding": "OPQ32r personality behavior manager director executive",
    },
    {
        "entity_id": "1002",
        "name": "Java 8 (New)",
        "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
        "test_type": "K",
        "description": "Measures knowledge of Java 8 programming.",
        "duration": "20",
        "remote_testing": True,
        "adaptive_irt": False,
        "job_levels": ["Professional Individual Contributor", "Mid-Professional"],
        "languages": ["English (USA)"],
        "keys": ["Knowledge & Skills"],
        "full_text_for_embedding": "Java 8 knowledge skills developer programmer",
    },
    {
        "entity_id": "1003",
        "name": "Verify - G+",
        "url": "https://www.shl.com/products/product-catalog/view/verify-g-plus/",
        "test_type": "A",
        "description": "General cognitive ability test measuring verbal and numerical reasoning.",
        "duration": "36",
        "remote_testing": True,
        "adaptive_irt": True,
        "job_levels": ["Graduate", "Entry-Level", "General Population"],
        "languages": ["English International"],
        "keys": ["Ability & Aptitude"],
        "full_text_for_embedding": "verify cognitive ability aptitude reasoning graduate",
    },
    {
        "entity_id": "1004",
        "name": "Global Skills Development Report",
        "url": "https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
        "test_type": "A,E,B,C,D,P",
        "description": "Comprehensive skills report covering the Great 8 Domains.",
        "duration": "",
        "remote_testing": True,
        "adaptive_irt": False,
        "job_levels": ["Director", "Manager", "Graduate", "Entry-Level"],
        "languages": [],
        "keys": ["Ability & Aptitude", "Personality & Behavior"],
        "full_text_for_embedding": "global skills development report competencies",
    },
    {
        "entity_id": "1005",
        "name": "Contact Center Call Simulation (New)",
        "url": "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/",
        "test_type": "S",
        "description": "Simulates contact center calls for customer service screening.",
        "duration": "15",
        "remote_testing": True,
        "adaptive_irt": False,
        "job_levels": ["Entry-Level", "General Population"],
        "languages": ["English (USA)"],
        "keys": ["Simulations"],
        "full_text_for_embedding": "contact center call simulation customer service agent",
    },
]


@pytest.fixture(scope="session")
def sample_catalog():
    return SAMPLE_CATALOG


@pytest.fixture(scope="session")
def catalog_urls(sample_catalog):
    return {item["url"] for item in sample_catalog}


@pytest.fixture(scope="session")
def client(sample_catalog):
    from retrieval.faiss_store import store

    store.catalog = sample_catalog
    store.entity_lookup = {item["entity_id"]: item for item in sample_catalog}
    store.id_map = {str(i): item["entity_id"] for i, item in enumerate(sample_catalog)}
    store._loaded = True

    import faiss
    dim = 384
    index = faiss.IndexFlatIP(dim)
    vecs = np.random.rand(len(sample_catalog), dim).astype(np.float32)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    store.index = index

    import ingestion.embeddings as emb_mod
    emb_mod.embed_query = lambda q: (
        lambda v: (faiss.normalize_L2(v), v)[1]
    )(np.random.rand(1, 384).astype(np.float32))
    emb_mod.get_model = lambda: MagicMock()

    # Handles both agent.llm_client and agent.llm naming
    try:
        import agent.llm_client as llm_mod
    except ModuleNotFoundError:
        import agent.llm as llm_mod
    llm_mod.call_groq = lambda system, user, **kw: (
        "Here are the recommended SHL assessments based on your requirements."
    )

    import agent.prompts as prompts_mod
    prompts_mod.build_recommend_prompt = lambda ctx, recommendations, is_refinement=False: (
        "system", f"recommend {len(recommendations)} items"
    )
    prompts_mod.build_compare_prompt = lambda comparison: (
        "system", "compare"
    )

    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app, raise_server_exceptions=False)


def make_messages(*user_msgs):
    msgs = []
    for i, content in enumerate(user_msgs):
        msgs.append({"role": "user", "content": content})
        if i < len(user_msgs) - 1:
            msgs.append({"role": "assistant", "content": "Got it, tell me more."})
    return msgs


def post_chat(client, messages):
    return client.post("/chat", json={"messages": messages}).json()