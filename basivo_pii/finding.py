from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    """One piece of PII: character offsets into the text, our label, a score in [0, 1] and who found it."""
    label: str
    start: int
    end: int
    score: float
    source: str = "model"
