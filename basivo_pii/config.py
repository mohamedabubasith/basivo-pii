"""The YAML config (see config.yaml at the project root) -> Analyzer and operators."""
from pathlib import Path

import yaml

from basivo_pii.anonymizer import OperatorConfig
from basivo_pii.detectors import patterns as p

VALIDATORS = {"luhn": p.luhn, "iban": p.iban, "ip": p.ip, "phone_digits": p.phone_digits}


def load(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def rules(cfg: dict) -> list[p.Pattern]:
    r = cfg.get("rules") or {}
    if r.get("enabled", True) is False:
        return []
    off = set(r.get("disable") or [])
    own = [p.Pattern(d["name"], d["label"], d["regex"], d["score"], tuple(d.get("context") or ()),
                     d.get("context_score"), VALIDATORS[d["validator"]] if d.get("validator") else None,
                     d.get("group", 0)) for d in r.get("add") or []]
    return [x for x in p.DEFAULT_PATTERNS if x.name not in off] + own


def analyzer(cfg: dict):
    from basivo_pii.analyzer import Analyzer  # imports torch
    m = dict(cfg["model"])
    return Analyzer(m.pop("name"), thresholds=cfg.get("thresholds"), patterns=rules(cfg), **m)


def operators(cfg: dict) -> dict[str, OperatorConfig]:
    return {label: OperatorConfig.from_dict(d) for label, d in (cfg.get("operators") or {}).items()}
