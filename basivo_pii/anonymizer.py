from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from basivo_pii.finding import Finding
from basivo_pii.operators import OPERATORS, decrypt


class OperatorConfig:
    """OperatorConfig("mask", char="*", chars=12): which operator, with which params."""

    def __init__(self, name: str, **params):
        self.name, self.params = name, params

    @classmethod
    def from_dict(cls, d: dict) -> "OperatorConfig":
        """{"type": "mask", "chars": 12} (the config file and REST form)."""
        d = dict(d)
        return cls(d.pop("type"), **d)

    def __eq__(self, other):
        return isinstance(other, OperatorConfig) and (self.name, self.params) == (other.name, other.params)

    def __repr__(self):
        return f"OperatorConfig({self.name!r}, **{self.params})"


DEFAULT = OperatorConfig("replace")


@dataclass(frozen=True)
class Item:
    """What happened to one finding. start/end point into the NEW text."""
    label: str
    start: int
    end: int
    text: str
    operator: str
    score: float
    source: str = "model"


@dataclass(frozen=True)
class Result:
    text: str
    items: list[Item]
    mapping: dict[str, str] = field(default_factory=dict)  # pseudonymize: {placeholder: original}


class Anonymizer:
    """Changes the text at the findings. operators: {label: OperatorConfig, "DEFAULT": OperatorConfig}; a label with
    no entry uses "DEFAULT", and with no "DEFAULT" it is replaced with [LABEL].

    secret: the key for hash and encrypt when their config has no `key`."""

    def __init__(self, secret: str | bytes | None = None):
        self.operators: dict[str, Callable[..., str]] = dict(OPERATORS)
        self.secret = secret

    def add_operator(self, name: str, fn: Callable[..., str]):
        """fn(value, label, **params) -> new value."""
        self.operators[name] = fn

    def anonymize(self, text: str, findings: list[Finding], operators: dict[str, OperatorConfig] | None = None,
                  mapping: dict[str, str] | None = None) -> Result:
        """mapping: an earlier Result.mapping, so the same person keeps the same placeholder across texts. It is
        copied, not changed; the new one comes back in Result.mapping."""
        operators, mapping = operators or {}, dict(mapping or {})
        parts, items, pos, length = [], [], 0, 0
        for f in sorted(findings, key=lambda f: f.start):
            if not 0 <= pos <= f.start < f.end <= len(text):
                raise ValueError(f"finding outside the text or overlapping the one before: {f}")
            config = operators.get(f.label) or operators.get("DEFAULT") or DEFAULT
            new = self._operator(config, mapping)(text[f.start:f.end], f.label)
            parts += [text[pos:f.start], new]
            length += f.start - pos
            items.append(Item(f.label, length, length + len(new), new, config.name, f.score, f.source))
            length += len(new)
            pos = f.end
        parts.append(text[pos:])
        return Result("".join(parts), items, mapping)

    def _operator(self, config: OperatorConfig, mapping: dict[str, str]) -> Callable[[str, str], str]:
        if config.name not in self.operators:
            raise ValueError(f"unknown operator {config.name!r}; known: {sorted(self.operators)}")
        params = dict(config.params)
        if config.name == "pseudonymize":
            params["mapping"] = mapping
        if config.name in ("hash", "encrypt"):
            params.setdefault("key", self.secret)
        return partial(self.operators[config.name], **params)

    def deanonymize(self, text: str, items: list[Item], mapping: dict[str, str] | None = None,
                    key: str | bytes | None = None) -> str:
        """Undo pseudonymize (with the mapping) and encrypt (with the key, default the secret). Other operators
        cannot be undone and stay as they are."""
        for i in sorted(items, key=lambda i: i.start, reverse=True):
            if i.operator == "pseudonymize":
                original = (mapping or {})[i.text]
            elif i.operator == "encrypt":
                original = decrypt(i.text, key or self.secret)
            else:
                continue
            text = text[:i.start] + original + text[i.end:]
        return text
