"""Rules: regex + validator (check digits) + context words, for PII with a fixed format (v1 doc, figure 2: email,
card, IBAN, IP address, URL) and for labels the model has no training data for (passport, driving licence).

A match gets `score`; with a context word (whole word, any case) near it (60 characters before, 30 after) it gets
`context_score`. A
validator returns False to drop a match. Low scores (below the default threshold 0.5) mean "only with context".
"""

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Callable

from basivo_pii.finding import Finding

BEFORE, AFTER = 60, 30


def luhn(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = sum(
        d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2)
        for i, d in enumerate(reversed(digits))
    )
    return total % 10 == 0


def iban(value: str) -> bool:
    v = re.sub(r"[ -]", "", value).upper()
    if not 15 <= len(v) <= 34:
        return False
    return int("".join(str(int(c, 36)) for c in v[4:] + v[:4])) % 97 == 1


def ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def phone_digits(value: str) -> bool:
    return 8 <= sum(c.isdigit() for c in value) <= 15


@dataclass(frozen=True)
class Pattern:
    name: str
    label: str
    regex: str
    score: float
    context: tuple[str, ...] = ()
    context_score: float | None = (
        None  # None: min(1, max(score + 0.35, 0.4)), as Presidio
    )
    validator: Callable[[str], bool] | None = None
    group: int = 0  # which regex group is the value (0 = the whole match)
    flags: int = 0
    compiled: re.Pattern = field(init=False, repr=False, compare=False)
    context_re: re.Pattern | None = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "compiled", re.compile(self.regex, self.flags))
        words = "|".join(re.escape(w) for w in self.context)
        object.__setattr__(
            self, "context_re", re.compile(rf"\b(?:{words})\b") if words else None
        )


DEFAULT_PATTERNS = [
    Pattern(
        "email",
        "EMAIL",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b",
        0.95,
    ),
    Pattern(
        "card",
        "CARD_NUMBER",
        r"\b(?:\d[ -]?){12,18}\d\b",
        0.9,  # only Luhn-valid numbers get here
        ("card", "visa", "mastercard", "amex"),
        validator=luhn,
    ),
    Pattern(
        "iban",
        "BANK_ACCOUNT",
        r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]{4}){2,7}(?:[ -]?[A-Z0-9]{1,4})?\b",
        1.0,  # only mod-97-valid IBANs get here: beats the model, which has no IBAN data
        ("iban", "account", "bank"),
        validator=iban,
    ),
    Pattern(
        "ipv4",
        "IP_ADDRESS",
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        0.6,
        ("ip", "address", "host", "server"),
        validator=ip,
    ),
    Pattern(
        "ipv6",
        "IP_ADDRESS",
        r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])",
        0.6,
        ("ip", "ipv6", "address", "host"),
        validator=ip,
    ),
    Pattern("url", "URL", r"\b(?:https?://|www\.)[^\s<>\"']*[^\s<>\"'.,;:!?)\]]", 0.6),
    Pattern(
        "phone_international",
        "PHONE",
        r"(?<![\w+])\+\d{1,3}[\s.-]?(?:\(?\d{1,5}\)?[\s.-]?){1,5}\d{2,5}(?!\w)",
        0.6,
        ("phone", "tel", "mobile", "call", "whatsapp", "contact"),
        validator=phone_digits,
    ),
    Pattern(
        "us_ssn",
        "NATIONAL_ID",
        r"\b\d{3}-\d{2}-\d{4}\b",
        0.4,
        ("ssn", "social security"),
    ),
    Pattern(
        "aadhaar",
        "NATIONAL_ID",
        r"\b\d{4} \d{4} \d{4}\b",
        0.3,
        ("aadhaar", "aadhar", "uidai"),
    ),
    Pattern(
        "india_pan", "TAX_ID", r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b", 0.7, ("pan",)
    ),
    Pattern(
        "passport",
        "PASSPORT",
        r"\b(?=[A-Z0-9]*\d)[A-Z][A-Z0-9]{6,8}\b|\b\d{9}\b",
        0.2,
        ("passport",),
        0.6,
    ),
    Pattern(
        "driving_licence",
        "DRIVING_LICENCE",
        r"\b[A-Z]{2}\d{2}[ -]?\d{11}\b|\b(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{5,17}\b",  # India: MH12 20110012345
        0.15,
        (
            "driving licence",
            "driving license",
            "driver's license",
            "driver license",
            "drivers license",
            "licence no",
            "license no",
            "dl no",
        ),
        0.6,
    ),
    Pattern(
        "password",
        "PASSWORD",
        # markdown or quotes after the colon are skipped; the value needs a digit or a symbol inside it ("password is
        # essential" is not a password); a full stop or comma at the end is not part of it
        r"\b(?:password|passwd|pwd|passcode|pin)\b[*_]*\s*(?:is|:|=)[\s*_\"'`]*(?=[^\s*\"'`]*\d|[^\s*\"'`]*[^\sA-Za-z\d*\"'`][^\s*\"'`])([^\s*\"'`]*[^\s*\"'`.,;:!?)])",
        0.7,
        group=1,
        flags=re.IGNORECASE,
    ),
    Pattern(
        "api_key",
        "API_KEY",
        r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}|\bgh[pousr]_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b"
        r"|\bAKIA[0-9A-Z]{16}\b|\bxox[abprs]-[A-Za-z0-9-]{10,}|\bAIza[0-9A-Za-z_-]{35}\b|\bhf_[A-Za-z0-9]{30,}\b"
        r"|\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        0.9,
    ),
]


class PatternDetector:
    def __init__(self, patterns: list[Pattern] = DEFAULT_PATTERNS):
        self.patterns = patterns

    def detect(self, text: str) -> list[Finding]:
        lower, out = text.lower(), []
        for p in self.patterns:
            for m in p.compiled.finditer(text):
                start, end = m.span(p.group)
                if start == end or (p.validator and not p.validator(m.group(p.group))):
                    continue
                score = p.score
                around = (
                    lower[max(0, start - BEFORE) : start]
                    + " "
                    + lower[end : end + AFTER]
                )
                if p.context_re and p.context_re.search(around):
                    score = (
                        p.context_score
                        if p.context_score is not None
                        else min(1.0, max(p.score + 0.35, 0.4))
                    )
                out.append(
                    Finding(p.label, start, end, round(score, 4), f"rule:{p.name}")
                )
        return out
