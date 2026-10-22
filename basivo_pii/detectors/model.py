"""Our fine-tuned token-classification model (BIO tags) as a detector.

Long texts are split into overlapping windows; each token keeps the prediction from the window where it sits farthest
from the edge (the same as pii/eval/infer.py). A span's score is the mean probability of its tokens' tags.
"""
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from basivo_pii.finding import Finding


def decode(text: str, tokens: list[tuple[int, int, str, float]]) -> list[Finding]:
    """tokens: (start, end, tag, probability) sorted by start. B- starts a span, I- of the same label continues it;
    a B- inside the same word (no gap) continues it too. Spaces at the span edges are trimmed."""
    spans, cur = [], None
    for a, b, tag, p in tokens:
        prefix, _, label = tag.partition("-")
        if not label:
            cur = None
            continue
        same_word = cur and cur["label"] == label and a == cur["end"]
        if cur is None or cur["label"] != label or (prefix == "B" and not same_word):
            cur = {"label": label, "start": a, "end": b, "probs": [p]}
            spans.append(cur)
        else:
            cur["end"] = b
            cur["probs"].append(p)
    out = []
    for s in spans:
        piece = text[s["start"]:s["end"]]
        start = s["start"] + len(piece) - len(piece.lstrip())
        end = s["end"] - (len(piece) - len(piece.rstrip()))
        if start < end:
            out.append(Finding(s["label"], start, end, round(sum(s["probs"]) / len(s["probs"]), 4)))
    return out


class ModelAccessError(RuntimeError):
    """The fine-tuned model is gated on Hugging Face: each user requests access and runs with their own token."""

    def __init__(self, model: str):
        super().__init__(
            f"No access to the model {model!r}. It is gated: each user needs the owner's approval.\n"
            f"  1. Sign in to Hugging Face and request access at https://huggingface.co/{model}\n"
            f"  2. After you are approved, create a read token at https://huggingface.co/settings/tokens\n"
            f"  3. Set HF_TOKEN=<token> (or run `hf auth login`) and start again.")


def no_access(e: BaseException | None) -> bool:
    """True when the error chain says the Hub refused the repo: gated, private or missing for this token.
    transformers wraps Hub errors in OSError, so walk the causes."""
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError  # GatedRepoError is a subclass
    while e is not None:
        if isinstance(e, RepositoryNotFoundError):
            return True
        if isinstance(e, HfHubHTTPError) and getattr(e.response, "status_code", None) in (401, 403):
            return True
        e = e.__cause__ or e.__context__
    return False


class ModelDetector:
    def __init__(self, model: str, revision: str | None = None, max_len: int = 1024, stride: int = 256,
                 device: str = "auto", fp16: bool = True, batch_size: int = 16):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if fp16 and device == "cuda" else torch.float32
        try:
            self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
            self.model = AutoModelForTokenClassification.from_pretrained(model, revision=revision, dtype=dtype)
        except Exception as e:
            if no_access(e):
                raise ModelAccessError(model) from e
            raise
        self.model.to(device).eval()
        self.device, self.max_len, self.stride, self.batch_size = device, max_len, stride, batch_size
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def detect(self, texts: list[str]) -> list[list[Finding]]:
        """All windows of all texts go through the model in batches of batch_size."""
        enc = self.tok(texts, truncation=True, max_length=self.max_len, stride=self.stride,
                       return_overflowing_tokens=True, return_offsets_mapping=True, padding=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        owner = enc.pop("overflow_to_sample_mapping").tolist()
        lengths = enc["attention_mask"].sum(1).tolist()
        best = [{} for _ in texts]  # per text: (start, end) -> (distance from window edge, tag, probability)
        for i in range(0, len(offsets), self.batch_size):
            inputs = {k: v[i:i + self.batch_size].to(self.device) for k, v in enc.items()}
            probs, ids = self.model(**inputs).logits.float().softmax(-1).max(-1)
            for w, (p_row, id_row) in enumerate(zip(probs.tolist(), ids.tolist()), start=i):
                for j, ((a, b), p, t) in enumerate(zip(offsets[w], p_row, id_row)):
                    if a == b:  # special or padding token
                        continue
                    d = min(j, lengths[w] - j)
                    if (a, b) not in best[owner[w]] or d > best[owner[w]][(a, b)][0]:
                        best[owner[w]][(a, b)] = (d, self.id2label[t], p)
        return [decode(text, sorted((a, b, tag, p) for (a, b), (_, tag, p) in tokens.items()))
                for text, tokens in zip(texts, best)]
