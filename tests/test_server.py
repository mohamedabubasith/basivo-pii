"""The REST service with the rules only (no model), so it runs without torch weights."""
import pytest
from fastapi.testclient import TestClient

from basivo_pii.detectors.patterns import PatternDetector
from basivo_pii.resolve import resolve
from basivo_pii.server import create_app

KEY = {"X-API-Key": "test-key"}
TEXT = "Mail ann.lee@example.org or ann.lee@example.org, card 4111 1111 1111 1111."


class RulesOnly:
    def analyze_batch(self, texts, labels=None, score_threshold=None, allow_list=None):
        th = {"default": 0.5 if score_threshold is None else score_threshold}
        return [resolve(t, PatternDetector().detect(t), labels, th, allow_list) for t in texts]


@pytest.fixture
def client():
    app = create_app(RulesOnly(), cfg={"model": {"name": "rules-only"}}, api_key="test-key", secret="s")
    with TestClient(app) as c:
        yield c


def test_needs_the_key(client):
    assert client.post("/analyze", json={"text": TEXT}).status_code == 401
    assert client.post("/analyze", json={"text": TEXT}, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.get("/health").json() == {"status": "ok", "model": "rules-only"}


def test_no_key_no_service():
    with pytest.raises(RuntimeError, match="BASIVO_PII_API_KEY"):
        create_app(RulesOnly(), cfg={}, api_key="")


def test_analyze_one_and_many(client):
    one = client.post("/analyze", json={"text": TEXT}, headers=KEY).json()
    assert [f["label"] for f in one["findings"]] == ["EMAIL", "EMAIL", "CARD_NUMBER"]
    assert one["has_pii"] and one["has_direct"]
    many = client.post("/analyze", json={"text": [TEXT, "nothing here"]}, headers=KEY).json()
    assert len(many) == 2 and many[1] == {"findings": [], "has_pii": False, "has_direct": False}
    only = client.post("/analyze", json={"text": TEXT, "labels": ["CARD_NUMBER"]}, headers=KEY).json()
    assert [f["label"] for f in only["findings"]] == ["CARD_NUMBER"]


def test_unknown_label_and_operator_are_errors(client):
    assert client.post("/analyze", json={"text": TEXT, "labels": ["NAME"]}, headers=KEY).status_code == 422
    r = client.post("/redact", json={"text": TEXT, "operators": {"DEFAULT": {"type": "nope"}}}, headers=KEY)
    assert r.status_code == 422 and "unknown operators" in r.json()["detail"]


def test_redact_default_and_pseudonymize_round_trip(client):
    r = client.post("/redact", json={"text": TEXT}, headers=KEY).json()
    assert r["text"] == "Mail [EMAIL] or [EMAIL], card [CARD_NUMBER]."
    ops = {"EMAIL": {"type": "pseudonymize"}, "CARD_NUMBER": {"type": "encrypt"}}
    r = client.post("/redact", json={"text": TEXT, "operators": ops}, headers=KEY).json()
    assert r["text"].startswith("Mail <EMAIL_1> or <EMAIL_1>, card ")
    assert r["mapping"] == {"<EMAIL_1>": "ann.lee@example.org"}
    back = client.post("/deanonymize", json={"text": r["text"], "items": r["items"], "mapping": r["mapping"]},
                       headers=KEY).json()
    assert back == {"text": TEXT}


def test_redact_batch_shares_the_mapping(client):
    ops = {"DEFAULT": {"type": "pseudonymize"}}
    r = client.post("/redact", json={"text": ["a@b.io", "c@d.io and a@b.io"], "operators": ops}, headers=KEY).json()
    assert [x["text"] for x in r["results"]] == ["<EMAIL_1>", "<EMAIL_2> and <EMAIL_1>"]


def test_anonymize_with_client_findings(client):
    body = {"text": "Hi Ann", "findings": [{"label": "PERSON_NAME", "start": 3, "end": 6}],
            "operators": {"PERSON_NAME": {"type": "mask"}}}
    assert client.post("/anonymize", json=body, headers=KEY).json()["text"] == "Hi ***"
    body["findings"][0]["end"] = 60
    assert client.post("/anonymize", json=body, headers=KEY).status_code == 422


def test_too_big(client):
    assert client.post("/analyze", json={"text": "x" * 200_001}, headers=KEY).status_code == 413
