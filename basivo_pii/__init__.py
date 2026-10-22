"""basivo-pii: find PII with our fine-tuned model (Analyzer) and change the text (Anonymizer)."""
from basivo_pii.anonymizer import Anonymizer, Item, OperatorConfig, Result
from basivo_pii.finding import Finding
from basivo_pii.labels import LABELS


def __getattr__(name):  # Analyzer imports torch; load it only when asked for
    if name == "Analyzer":
        from basivo_pii.analyzer import Analyzer
        return Analyzer
    raise AttributeError(name)


__all__ = ["Analyzer", "Anonymizer", "Finding", "Item", "LABELS", "OperatorConfig", "Result"]
