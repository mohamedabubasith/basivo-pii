import pytest

from basivo_pii import Anonymizer, Finding as F, Item, OperatorConfig as Op

TEXT = "Hi, I am Priya Sharma, card 4111111111111111."
FOUND = [F("PERSON_NAME", 9, 21, .99), F("CARD_NUMBER", 28, 44, .98)]


def test_default_is_replace_with_label():
    r = Anonymizer().anonymize(TEXT, FOUND)
    assert r.text == "Hi, I am [PERSON_NAME], card [CARD_NUMBER]."
    assert r.items == [Item("PERSON_NAME", 9, 22, "[PERSON_NAME]", "replace", .99),
                       Item("CARD_NUMBER", 29, 42, "[CARD_NUMBER]", "replace", .98)]
    assert all(r.text[i.start:i.end] == i.text for i in r.items)


def test_operator_per_label_and_default():
    ops = {"DEFAULT": Op("redact"), "CARD_NUMBER": Op("mask", chars=12)}
    r = Anonymizer().anonymize(TEXT, FOUND, ops)
    assert r.text == "Hi, I am , card ************1111."
    assert set(ops) == {"DEFAULT", "CARD_NUMBER"}  # the caller's dict is not changed


@pytest.mark.parametrize("op, want", [
    (Op("replace", new_value="<NAME>"), "<NAME>"),
    (Op("mask"), "************"),
    (Op("mask", char="#", chars=5, from_end=True), "Priya S#####"),
    (Op("keep"), "Priya Sharma"),
])
def test_operators(op, want):
    r = Anonymizer().anonymize(TEXT, FOUND[:1], {"PERSON_NAME": op})
    assert r.items[0].text == want and r.text == f"Hi, I am {want}, card 4111111111111111."


def test_custom_operator():
    a = Anonymizer()
    a.add_operator("upper", lambda value, label: value.upper())
    assert a.anonymize(TEXT, FOUND[:1], {"DEFAULT": Op("upper")}).text.startswith("Hi, I am PRIYA SHARMA,")


@pytest.mark.parametrize("found", [[F("EMAIL", 40, 99, 1.0)], [F("EMAIL", 0, 5, 1.0), F("EMAIL", 3, 8, 1.0)]])
def test_bad_findings_raise(found):
    with pytest.raises(ValueError):
        Anonymizer().anonymize(TEXT, found)


def test_unknown_operator_raises():
    with pytest.raises(ValueError, match="unknown operator"):
        Anonymizer().anonymize(TEXT, FOUND, {"DEFAULT": Op("nope")})


def test_pseudonymize_same_value_same_placeholder_left_to_right():
    text = "Ann met Bob. Ann left."
    found = [F("PERSON_NAME", 0, 3, .9), F("PERSON_NAME", 8, 11, .9), F("PERSON_NAME", 13, 16, .9)]
    a = Anonymizer()
    r = a.anonymize(text, found, {"DEFAULT": Op("pseudonymize")})
    assert r.text == "<PERSON_NAME_1> met <PERSON_NAME_2>. <PERSON_NAME_1> left."
    assert r.mapping == {"<PERSON_NAME_1>": "Ann", "<PERSON_NAME_2>": "Bob"}
    assert a.deanonymize(r.text, r.items, r.mapping) == text
    # a second text with the same mapping keeps the numbers
    r2 = a.anonymize("Bob and Cy", [F("PERSON_NAME", 0, 3, .9), F("PERSON_NAME", 8, 10, .9)],
                     {"DEFAULT": Op("pseudonymize")}, r.mapping)
    assert r2.text == "<PERSON_NAME_2> and <PERSON_NAME_3>"
    assert r.mapping == {"<PERSON_NAME_1>": "Ann", "<PERSON_NAME_2>": "Bob"}  # the given mapping is not changed


def test_hash_is_stable_and_keyed():
    ops = {"DEFAULT": Op("hash")}
    one = Anonymizer("k1").anonymize(TEXT, FOUND, ops).items[0].text
    assert one == Anonymizer("k1").anonymize(TEXT, FOUND, ops).items[0].text and len(one) == 16
    assert one != Anonymizer("k2").anonymize(TEXT, FOUND, ops).items[0].text
    with pytest.raises(ValueError, match="need a key"):
        Anonymizer().anonymize(TEXT, FOUND, ops)


def test_encrypt_round_trip_and_wrong_key():
    from cryptography.exceptions import InvalidTag
    a = Anonymizer("secret")
    r = a.anonymize(TEXT, FOUND, {"DEFAULT": Op("encrypt")})
    assert "Priya" not in r.text
    assert a.deanonymize(r.text, r.items) == TEXT
    with pytest.raises(InvalidTag):
        Anonymizer("other").deanonymize(r.text, r.items)


def test_operator_config_from_dict():
    assert Op.from_dict({"type": "mask", "chars": 4}) == Op("mask", chars=4)
