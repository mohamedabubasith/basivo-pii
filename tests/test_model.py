"""End to end with the real model. Runs only when the model is in the local HF cache (no download in tests)."""
import os

import pytest

MODEL = os.environ.get("BASIVO_PII_MODEL", "abubasith86/pii-nemotron-mmbert-base-v1")


@pytest.fixture(scope="module")
def analyzer():
    os.environ["HF_HUB_OFFLINE"] = "1"
    from basivo_pii import Analyzer
    try:
        return Analyzer(MODEL, thresholds={"default": 0.5})
    except OSError:
        pytest.skip(f"{MODEL} not in the local HF cache")


def test_finds_name_email_phone(analyzer):
    from basivo_pii import Anonymizer
    text = "Hi, I am Priya Sharma, email priya.sharma@example.com, phone +91 98450 12345."
    found = analyzer.analyze(text)
    assert [(f.label, text[f.start:f.end]) for f in found] == [
        ("PERSON_NAME", "Priya Sharma"), ("EMAIL", "priya.sharma@example.com"), ("PHONE", "+91 98450 12345")]
    assert all(0.5 <= f.score <= 1 for f in found)
    assert Anonymizer().anonymize(text, found).text == "Hi, I am [PERSON_NAME], email [EMAIL], phone [PHONE]."


def test_long_text_and_batch(analyzer):
    long = "Nothing here. " * 400 + "Mail me at ann.lee@example.org please."  # far past one 1024-token window
    short = "No PII in this one."
    found_long, found_short = analyzer.analyze_batch([long, short])
    assert [long[f.start:f.end] for f in found_long] == ["ann.lee@example.org"]
    assert found_short == []
