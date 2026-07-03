"""CatalogMind — tests/test_recommendation.py"""
import pytest
from tests.conftest import make_messages, post_chat, SAMPLE_CATALOG

CATALOG_URLS = {item["url"] for item in SAMPLE_CATALOG}


class TestRecommendationURLs:
    def test_all_urls_from_catalog(self, client):
        r = post_chat(client, make_messages("I need assessments for a senior Java developer"))
        for rec in r["recommendations"]:
            assert rec["url"] in CATALOG_URLS, f"Hallucinated URL: {rec['url']}"

    def test_urls_are_shl_domain(self, client):
        r = post_chat(client, make_messages("I need personality tests for managers"))
        for rec in r["recommendations"]:
            assert "shl.com" in rec["url"]

    def test_urls_are_https(self, client):
        r = post_chat(client, make_messages("I need assessments for a Python developer"))
        for rec in r["recommendations"]:
            assert rec["url"].startswith("https://")

    def test_no_placeholder_urls(self, client):
        r = post_chat(client, make_messages("I need assessments for a senior Java developer"))
        for rec in r["recommendations"]:
            assert "placeholder" not in rec["url"].lower()


class TestRecommendationCount:
    def test_recommendations_between_1_and_10(self, client):
        r = post_chat(client, make_messages("I need assessments for a senior Java developer"))
        assert 1 <= len(r["recommendations"]) <= 10

    def test_recommendations_empty_when_clarifying(self, client):
        assert post_chat(client, make_messages("I need an assessment"))["recommendations"] == []

    def test_recommendations_empty_when_refusing(self, client):
        assert post_chat(client, make_messages("What salary should I offer?"))["recommendations"] == []

    def test_recommendations_empty_on_compare(self, client):
        r = post_chat(client, make_messages("What is the difference between OPQ and Java 8?"))
        assert r["recommendations"] == []

    def test_never_more_than_10(self, client):
        r = post_chat(client, make_messages("I need every assessment you have for software developers"))
        assert len(r["recommendations"]) <= 10


class TestRecommendationFields:
    def test_each_rec_has_name(self, client):
        r = post_chat(client, make_messages("I need assessments for a Java developer"))
        for rec in r["recommendations"]:
            assert "name" in rec and rec["name"]

    def test_each_rec_has_url(self, client):
        r = post_chat(client, make_messages("I need assessments for a Java developer"))
        for rec in r["recommendations"]:
            assert "url" in rec and rec["url"]

    def test_each_rec_has_test_type(self, client):
        r = post_chat(client, make_messages("I need assessments for a Java developer"))
        for rec in r["recommendations"]:
            assert "test_type" in rec and rec["test_type"]

    def test_rec_has_no_extra_fields(self, client):
        r = post_chat(client, make_messages("I need assessments for a Java developer"))
        for rec in r["recommendations"]:
            assert set(rec.keys()) == {"name", "url", "test_type"}


class TestConversationalBehavior:
    def test_refinement_returns_updated_shortlist(self, client):
        msgs = [
            {"role": "user", "content": "I need assessments for a senior Java developer"},
            {"role": "assistant", "content": "Here is your shortlist from shl.com with Java tests."},
            {"role": "user", "content": "Actually, also add personality tests"},
        ]
        r = post_chat(client, msgs)
        assert isinstance(r["recommendations"], list)
        assert len(r["recommendations"]) <= 10

    def test_end_of_conversation_false_after_recommendation(self, client):
        r = post_chat(client, make_messages("I need assessments for a senior Java developer"))
        assert r["end_of_conversation"] == False

    def test_end_of_conversation_true_on_goodbye(self, client):
        assert post_chat(client, make_messages("Thank you, that is all!"))["end_of_conversation"] == True

    def test_end_of_conversation_true_on_thanks(self, client):
        assert post_chat(client, make_messages("Thanks, perfect!"))["end_of_conversation"] == True

    def test_end_of_conversation_false_on_question(self, client):
        assert post_chat(client, make_messages("I need assessments for a Python developer"))["end_of_conversation"] == False

    def test_turn_cap_respected(self, client):
        msgs = []
        for _ in range(3):
            msgs.append({"role": "user", "content": "I need an assessment for a candidate"})
            msgs.append({"role": "assistant", "content": "Could you tell me more about the role?"})
        msgs.append({"role": "user", "content": "I need an assessment for a candidate"})
        r = post_chat(client, msgs)
        assert isinstance(r["reply"], str) and r["reply"]
        assert len(r["recommendations"]) <= 10


class TestAntiHallucination:
    def test_compare_uses_catalog_data_only(self, client):
        r = post_chat(client, make_messages("What is the difference between OPQ and Java 8?"))
        assert r["recommendations"] == []

    def test_unknown_assessment_compare_gives_not_found(self, client):
        r = post_chat(client, make_messages("What is the difference between FakeTestX and FakeTestY?"))
        assert r["recommendations"] == []
        assert any(w in r["reply"].lower() for w in ["find", "catalog", "not", "couldn't", "help"])

    def test_groq_failure_still_returns_recs(self, client, monkeypatch):
        try:
            import agent.llm_client as llm_mod
        except ModuleNotFoundError:
            import agent.llm as llm_mod
        monkeypatch.setattr(llm_mod, "call_groq", lambda *a, **kw: None)
        r = post_chat(client, make_messages("I need assessments for a Python developer"))
        assert isinstance(r["reply"], str) and r["reply"]
        assert len(r["recommendations"]) <= 10