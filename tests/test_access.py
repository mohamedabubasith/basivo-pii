import pytest
from huggingface_hub.errors import GatedRepoError

from basivo_pii.detectors import model as m


def test_gated_repo_gives_request_access_steps(monkeypatch):
    def refuse(*a, **k):
        hub = GatedRepoError.__new__(GatedRepoError)  # skip __init__: it needs a live HTTP response
        raise OSError("You are trying to access a gated repo.") from hub
    monkeypatch.setattr(m.AutoTokenizer, "from_pretrained", refuse)
    with pytest.raises(m.ModelAccessError, match="request access at https://huggingface.co/owner/model"):
        m.ModelDetector("owner/model", device="cpu")


def test_other_errors_pass_through(monkeypatch):
    def broken(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(m.AutoTokenizer, "from_pretrained", broken)
    with pytest.raises(OSError, match="disk full"):
        m.ModelDetector("owner/model", device="cpu")
