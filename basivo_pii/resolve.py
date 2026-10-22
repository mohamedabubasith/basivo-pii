"""Turn raw findings into the final list: one label per character at most.

Order: labels asked for -> score threshold (per label) -> allow list -> join parts of one thing -> overlaps
-> entity register (every other mention of a direct identifier that was found).
"""
import re

from basivo_pii.finding import Finding
from basivo_pii.labels import DIRECT

# One thing = one span (the same rule as the training data): address parts with only spaces, commas or new lines
# between them are one ADDRESS; names with only spaces between them are one PERSON_NAME.
JOIN = {"ADDRESS": re.compile(r"[\s,]*"), "PERSON_NAME": re.compile(r" +")}


def resolve(text: str, findings: list[Finding], labels: set[str] | None = None,
            thresholds: dict[str, float] | None = None, allow_list: set[str] | None = None) -> list[Finding]:
    thresholds = thresholds or {}
    kept = [f for f in findings
            if (labels is None or f.label in labels)
            and f.score >= thresholds.get(f.label, thresholds.get("default", 0.0))
            and not (allow_list and text[f.start:f.end] in allow_list)]
    return register(text, remove_overlaps(join(text, kept)))


def join(text: str, findings: list[Finding]) -> list[Finding]:
    out = []
    for f in sorted(findings, key=lambda f: (f.start, f.end)):
        prev = out[-1] if out else None
        if (prev and prev.label == f.label and f.label in JOIN and prev.end <= f.start
                and JOIN[f.label].fullmatch(text[prev.end:f.start])):
            out[-1] = Finding(f.label, prev.start, f.end, max(prev.score, f.score), prev.source)
        else:
            out.append(f)
    return out


def remove_overlaps(findings: list[Finding]) -> list[Finding]:
    """Higher score wins, then the longer span, then the model over a rule. A finding that overlaps a kept one is
    dropped."""
    kept = []
    for f in sorted(findings, key=lambda f: (-f.score, f.start - f.end, f.source != "model")):
        if all(f.end <= k.start or k.end <= f.start for k in kept):
            kept.append(f)
    return sorted(kept, key=lambda f: f.start)


def register(text: str, findings: list[Finding], min_len: int = 3) -> list[Finding]:
    """A direct identifier found once is masked everywhere: every other exact mention of the same value (as a whole
    word) is added with the same label and score, source "register". The person is protected only when every mention
    is masked (v1 doc, section 8).
    ponytail: exact value only; "Priya" alone after "Priya Sharma" relies on the model. Add name parts if leaks show."""
    out = list(findings)
    values = {text[f.start:f.end]: f for f in findings if f.label in DIRECT and f.end - f.start >= min_len}
    # a place name alone ("USA") is often part of other words ("Athletica USA"): only street addresses (with a digit)
    values = {v: f for v, f in values.items() if f.label != "ADDRESS" or any(c.isdigit() for c in v)}
    for value, f in values.items():
        for m in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)", text):
            if all(m.end() <= o.start or o.end <= m.start() for o in out):
                out.append(Finding(f.label, m.start(), m.end(), f.score, "register"))
    return sorted(out, key=lambda f: f.start)
