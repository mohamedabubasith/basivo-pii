from basivo_pii import OperatorConfig as Op
from basivo_pii import config


def test_project_config_loads():
    cfg = config.load("config.yaml")
    assert cfg["model"]["name"] == "abubasith86/pii-nemotron-mmbert-base-v1"
    assert config.operators(cfg) == {"DEFAULT": Op("replace")}
    assert len(config.rules(cfg)) == 14


def test_rules_disable_add_and_off():
    cfg = {"rules": {"disable": ["driving_licence"], "add": [
        {"name": "emp", "label": "CUSTOMER_ID", "regex": r"\bEMP-\d{6}\b", "score": 0.9, "context": ["employee"]}]}}
    names = [p.name for p in config.rules(cfg)]
    assert "driving_licence" not in names and names[-1] == "emp"
    assert config.rules({"rules": {"enabled": False}}) == []
