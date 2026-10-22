import pytest

from basivo_pii.detectors.model import decode
from basivo_pii.finding import Finding as F
from basivo_pii.resolve import resolve


def test_decode_joins_bio_and_averages_scores():
    text = "I am Mi Patel."
    tokens = [(0, 1, "O", .9), (2, 4, "O", .9), (5, 7, "B-PERSON_NAME", .9), (7, 13, "I-PERSON_NAME", .7),
              (13, 14, "O", .9)]
    assert decode(text, tokens) == [F("PERSON_NAME", 5, 13, 0.8)]


def test_decode_b_inside_a_word_continues_and_spaces_are_trimmed():
    text = "mail a.b@x.io now"
    tokens = [(4, 6, "B-EMAIL", 1.0), (6, 13, "B-EMAIL", 1.0), (13, 14, "O", 1.0)]
    assert decode(text, tokens) == [F("EMAIL", 5, 13, 1.0)]


@pytest.mark.parametrize("text, found, want", [
    # address parts across a comma and a new line: one ADDRESS
    ("Leeds, UK\nEU", [F("ADDRESS", 0, 5, .9), F("ADDRESS", 7, 9, .8), F("ADDRESS", 10, 12, .7)],
     [F("ADDRESS", 0, 12, .9)]),
    # names across a comma stay apart
    ("Stone, Ryan", [F("PERSON_NAME", 0, 5, .9), F("PERSON_NAME", 7, 11, .9)],
     [F("PERSON_NAME", 0, 5, .9), F("PERSON_NAME", 7, 11, .9)]),
    # different labels never join
    ("Ann Leeds", [F("PERSON_NAME", 0, 3, .9), F("ADDRESS", 4, 9, .9)],
     [F("PERSON_NAME", 0, 3, .9), F("ADDRESS", 4, 9, .9)]),
    # overlap: the higher score wins
    ("12345678", [F("PHONE", 0, 8, .6), F("CUSTOMER_ID", 2, 8, .9)], [F("CUSTOMER_ID", 2, 8, .9)]),
    # overlap with the same score: the longer wins
    ("12345678", [F("PHONE", 0, 8, .9), F("CUSTOMER_ID", 2, 8, .9)], [F("PHONE", 0, 8, .9)]),
])
def test_join_and_overlaps(text, found, want):
    assert resolve(text, found) == want


def test_thresholds_labels_and_allow_list():
    text = "Ann pw hunter2 at Leeds"
    found = [F("PERSON_NAME", 0, 3, .9), F("PASSWORD", 7, 14, .35), F("ADDRESS", 18, 23, .9)]
    assert resolve(text, found, thresholds={"default": .5}) == [found[0], found[2]]
    assert resolve(text, found, thresholds={"default": .5, "PASSWORD": .3}) == found
    assert resolve(text, found, labels={"ADDRESS"}) == [found[2]]
    assert resolve(text, found, allow_list={"Leeds"}) == found[:2]


def test_register_masks_every_mention_of_a_direct_identifier():
    text = "Ann Lee wrote. Ann Lee lives in Leeds. Ann Leeway is someone else."
    found = [F("PERSON_NAME", 0, 7, .9)]
    assert resolve(text, found) == [F("PERSON_NAME", 0, 7, .9), F("PERSON_NAME", 15, 22, .9, "register")]


def test_register_skips_dates_and_short_values():
    text = "On 2024 and 2024. Al and Al."
    assert resolve(text, [F("DATE", 3, 7, .9), F("PERSON_NAME", 18, 20, .9)]) == \
        [F("DATE", 3, 7, .9), F("PERSON_NAME", 18, 20, .9)]


def test_rule_beats_model_on_score_and_model_wins_a_tie():
    text = "DE89370400440532013000"
    model, rule = F("CARD_NUMBER", 0, 22, .97), F("BANK_ACCOUNT", 0, 22, 1.0, "rule:iban")
    assert resolve(text, [model, rule]) == [rule]
    assert resolve(text, [F("EMAIL", 0, 22, .95, "rule:email"), F("EMAIL", 0, 22, .95)]) == [F("EMAIL", 0, 22, .95)]


def test_register_copies_street_addresses_not_place_names():
    text = "Ships to USA. Athletica USA. 14 MG Road. Office: 14 MG Road."
    found = [F("ADDRESS", 9, 12, .9), F("ADDRESS", 29, 39, .9)]
    assert resolve(text, found) == found + [F("ADDRESS", 49, 59, .9, "register")]
