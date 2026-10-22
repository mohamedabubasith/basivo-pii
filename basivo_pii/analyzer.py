from basivo_pii.detectors.model import ModelDetector
from basivo_pii.detectors.patterns import DEFAULT_PATTERNS, Pattern, PatternDetector
from basivo_pii.finding import Finding
from basivo_pii.resolve import resolve


class Analyzer:
    """Finds PII. Loads the model once; analyze() returns positions, labels and scores, and never changes the text.

    Flow (v1 doc, figure 2): rules and validators + the model -> resolve (threshold per label, allow list, join,
    overlaps, entity register).
    thresholds: {"default": 0.5, "PASSWORD": 0.3}; a finding below its label's threshold is dropped.
    patterns: the rules; [] for the model alone."""

    def __init__(self, model: str, revision: str | None = None, thresholds: dict[str, float] | None = None,
                 patterns: list[Pattern] = DEFAULT_PATTERNS, **model_args):
        self.detector = ModelDetector(model, revision=revision, **model_args)
        self.rules = PatternDetector(patterns)
        self.thresholds = thresholds or {}

    def analyze(self, text: str, **kwargs) -> list[Finding]:
        return self.analyze_batch([text], **kwargs)[0]

    def analyze_batch(self, texts: list[str], labels: set[str] | None = None, score_threshold: float | None = None,
                      allow_list: set[str] | None = None) -> list[list[Finding]]:
        """labels: only these labels; score_threshold: one threshold for every label, instead of self.thresholds;
        allow_list: exact values that are never PII."""
        thresholds = {"default": score_threshold} if score_threshold is not None else self.thresholds
        return [resolve(t, found + self.rules.detect(t), labels, thresholds, allow_list)
                for t, found in zip(texts, self.detector.detect(texts))]
