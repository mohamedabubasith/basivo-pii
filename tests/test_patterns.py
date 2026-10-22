import pytest

from basivo_pii.detectors.patterns import PatternDetector, iban, luhn
from basivo_pii.resolve import resolve


def raw(text):
    return [(f.label, text[f.start:f.end], f.score) for f in PatternDetector().detect(text)]


def found(text):
    """What a user gets from the rules: after resolve with the default threshold 0.5."""
    return [(f.label, text[f.start:f.end], f.score)
            for f in resolve(text, PatternDetector().detect(text), thresholds={"default": 0.5})]


@pytest.mark.parametrize("text, want", [
    ("mail ann.lee@example.org now", [("EMAIL", "ann.lee@example.org", 0.95)]),
    ("card 4111 1111 1111 1111 ok", [("CARD_NUMBER", "4111 1111 1111 1111", 1.0)]),    # Luhn ok + "card"
    ("ref 4111 1111 1111 1112 ok", []),                                                # Luhn fails
    ("IBAN DE89 3704 0044 0532 0130 00 paid", [("BANK_ACCOUNT", "DE89 3704 0044 0532 0130 00", 1.0)]),
    ("send to GB82WEST12345698765432", [("BANK_ACCOUNT", "GB82WEST12345698765432", 1.0)]),
    ("send to GB82WEST12345698765431", []),                                            # mod-97 fails
    ("host 192.168.4.134 down", [("IP_ADDRESS", "192.168.4.134", 0.95)]),
    ("version 999.1.2.3", []),                                                         # not an IP
    ("at 10:30:00 we met", []),
    ("ip fe80::1ff:fe23:4567:890a here", [("IP_ADDRESS", "fe80::1ff:fe23:4567:890a", 0.95)]),
    ("see https://x.io/a?b=1.", [("URL", "https://x.io/a?b=1", 0.6)]),
    ("call +91 98450 12345 now", [("PHONE", "+91 98450 12345", 0.95)]),
    ("ssn 123-45-6789", [("NATIONAL_ID", "123-45-6789", 0.75)]),
    ("code 123-45-6789", []),                                                          # 0.4: needs context
    ("PAN ABCPE1234F", [("TAX_ID", "ABCPE1234F", 1.0)]),
    ("passport no. K1234567", [("PASSPORT", "K1234567", 0.6)]),
    ("Order K1234567", []),                                                            # 0.2: needs context
    ("my password: hunter2", [("PASSWORD", "hunter2", 0.7)]),
    ("**Temporary Password:** b9M!k7Lp#5nXz3", [("PASSWORD", "b9M!k7Lp#5nXz3", 0.7)]),
    ("- **PIN:** 2277", [("PASSWORD", "2277", 0.7)]),
    ('the password is "River@2025".', [("PASSWORD", "River@2025", 0.7)]),
    ("This password is essential for security", []),
    ("key AKIAIOSFODNN7EXAMPLE", [("API_KEY", "AKIAIOSFODNN7EXAMPLE", 0.9)]),
])
def test_rules(text, want):
    assert found(text) == want


def test_driving_licence_needs_context():
    assert raw("ref MH12 20110012345") == [("DRIVING_LICENCE", "MH12 20110012345", 0.15)]
    assert found("ref MH12 20110012345") == []
    assert found("driving licence MH12 20110012345") == [("DRIVING_LICENCE", "MH12 20110012345", 0.6)]


def test_context_word_must_be_a_whole_word():
    assert raw("lessn 123-45-6789")[0][2] == 0.4      # "ssn" inside another word: no boost
    assert raw("SSN: 123-45-6789")[0][2] == 0.75


def test_validators():
    assert luhn("4111111111111111") and not luhn("4111111111111112") and not luhn("1234")
    assert iban("DE89 3704 0044 0532 0130 00") and not iban("DE89 3704 0044 0532 0130 01")
