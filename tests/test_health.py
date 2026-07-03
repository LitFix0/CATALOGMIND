"""CatalogMind — tests/test_health.py"""
import pytest


class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_key(self, client):
        r = client.get("/health").json()
        assert "status" in r

    def test_health_status_is_ok_when_store_ready(self, client):
        r = client.get("/health").json()
        assert r["status"] == "ok"

    def test_health_returns_no_extra_fields(self, client):
        r = client.get("/health").json()
        assert set(r.keys()) == {"status"}

    def test_health_not_ready_when_store_unloaded(self, client):
        from retrieval.faiss_store import store
        original = store._loaded
        store._loaded = False
        try:
            r = client.get("/health").json()
            assert r["status"] == "not_ready"
        finally:
            store._loaded = original