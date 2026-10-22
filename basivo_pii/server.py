"""REST service. Run: BASIVO_PII_API_KEY=... uv run --extra server uvicorn basivo_pii.server:app --port 8000

Env: BASIVO_PII_API_KEY (required; clients send it as X-API-Key), BASIVO_PII_CONFIG (default ./config.yaml),
BASIVO_PII_SECRET (key for hash and encrypt). Request text is never logged.
"""
import hmac
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict

from cryptography.exceptions import InvalidTag
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from basivo_pii import config
from basivo_pii.anonymizer import Anonymizer, Item, OperatorConfig
from basivo_pii.finding import Finding
from basivo_pii.labels import DIRECT, LABELS

MAX_CHARS, MAX_TEXTS = 200_000, 256


class Options(BaseModel):
    labels: list[str] | None = None
    score_threshold: float | None = Field(None, ge=0, le=1)
    allow_list: list[str] | None = None


class AnalyzeRequest(Options):
    text: str | list[str]


class RedactRequest(AnalyzeRequest):
    operators: dict[str, dict] | None = None
    mapping: dict[str, str] | None = None


class FindingIn(BaseModel):
    label: str
    start: int
    end: int
    score: float = 1.0
    source: str = "client"


class AnonymizeRequest(BaseModel):
    text: str
    findings: list[FindingIn]
    operators: dict[str, dict] | None = None
    mapping: dict[str, str] | None = None


class ItemIn(BaseModel):
    label: str
    start: int
    end: int
    text: str
    operator: str
    score: float = 1.0
    source: str = "model"


class DeanonymizeRequest(BaseModel):
    text: str
    items: list[ItemIn]
    mapping: dict[str, str] | None = None


def create_app(analyzer=None, cfg: dict | None = None, api_key: str | None = None,
               secret: str | None = None) -> FastAPI:
    """Arguments are for tests; the server takes them from the env and the config file."""
    api_key = api_key or os.environ.get("BASIVO_PII_API_KEY")
    if not api_key:
        raise RuntimeError("set BASIVO_PII_API_KEY: the service does not run without a key")
    cfg = cfg if cfg is not None else config.load(os.environ.get("BASIVO_PII_CONFIG", "config.yaml"))
    anonymizer = Anonymizer(secret or os.environ.get("BASIVO_PII_SECRET"))
    default_ops = config.operators(cfg)
    state = {"analyzer": analyzer}
    lock = threading.Lock()  # ponytail: one request on the model at a time; send batches for throughput

    @asynccontextmanager
    async def lifespan(_):
        if state["analyzer"] is None:
            state["analyzer"] = config.analyzer(cfg)
        yield

    app = FastAPI(title="basivo-pii", lifespan=lifespan)

    def auth(x_api_key: str = Header("")):
        if not hmac.compare_digest(x_api_key.encode(), api_key.encode()):
            raise HTTPException(401, "wrong or missing X-API-Key")

    def check_labels(labels):
        unknown = sorted(set(labels or ()) - set(LABELS))
        if unknown:
            raise HTTPException(422, f"unknown labels {unknown}; known: {LABELS}")

    def texts_of(req: AnalyzeRequest) -> list[str]:
        texts = [req.text] if isinstance(req.text, str) else req.text
        if len(texts) > MAX_TEXTS or any(len(t) > MAX_CHARS for t in texts):
            raise HTTPException(413, f"at most {MAX_TEXTS} texts of {MAX_CHARS:,} characters each")
        return texts

    def analyze(req: AnalyzeRequest) -> list[list[Finding]]:
        check_labels(req.labels)
        texts = texts_of(req)
        with lock:
            return state["analyzer"].analyze_batch(
                texts, labels=set(req.labels) if req.labels else None, score_threshold=req.score_threshold,
                allow_list=set(req.allow_list) if req.allow_list else None)

    def ops(given: dict | None) -> dict[str, OperatorConfig]:
        if given is None:
            return default_ops
        check_labels(set(given) - {"DEFAULT"})
        try:
            out = {label: OperatorConfig.from_dict(d) for label, d in given.items()}
        except KeyError:
            raise HTTPException(422, 'each operator needs "type"')
        unknown = sorted({o.name for o in out.values()} - set(anonymizer.operators))
        if unknown:
            raise HTTPException(422, f"unknown operators {unknown}; known: {sorted(anonymizer.operators)}")
        return out

    def run_anonymize(text, findings, operators, mapping):
        try:
            return anonymizer.anonymize(text, findings, operators, mapping)
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e))

    def flags(findings: list[Finding]) -> dict:
        return {"has_pii": bool(findings), "has_direct": any(f.label in DIRECT for f in findings)}

    def one_or_many(req, results):
        return results[0] if isinstance(req.text, str) else results

    @app.get("/health")
    def health():
        return {"status": "ok" if state["analyzer"] is not None else "loading",
                "model": (cfg.get("model") or {}).get("name")}

    @app.get("/labels", dependencies=[Depends(auth)])
    def labels():
        return LABELS

    @app.get("/operators", dependencies=[Depends(auth)])
    def operators():
        return sorted(anonymizer.operators)

    @app.post("/analyze", dependencies=[Depends(auth)])
    def analyze_route(req: AnalyzeRequest):
        return one_or_many(req, [{"findings": [asdict(f) for f in found], **flags(found)} for found in analyze(req)])

    @app.post("/anonymize", dependencies=[Depends(auth)])
    def anonymize_route(req: AnonymizeRequest):
        check_labels({f.label for f in req.findings})
        findings = [Finding(**f.model_dump()) for f in req.findings]
        r = run_anonymize(req.text, findings, ops(req.operators), req.mapping)
        return {"text": r.text, "items": [asdict(i) for i in r.items], "mapping": r.mapping}

    @app.post("/redact", dependencies=[Depends(auth)])
    def redact_route(req: RedactRequest):
        """analyze + anonymize. In a batch the mapping carries over, so one person keeps one placeholder."""
        operators, mapping, out = ops(req.operators), req.mapping, []
        for text, found in zip(texts_of(req), analyze(req)):
            r = run_anonymize(text, found, operators, mapping)
            mapping = r.mapping
            out.append({"text": r.text, "items": [asdict(i) for i in r.items], **flags(found)})
        return {"results": out, "mapping": mapping or {}} if isinstance(req.text, list) else \
            {**out[0], "mapping": mapping or {}}

    @app.post("/deanonymize", dependencies=[Depends(auth)])
    def deanonymize_route(req: DeanonymizeRequest):
        try:
            return {"text": anonymizer.deanonymize(req.text, [Item(**i.model_dump()) for i in req.items],
                                                   req.mapping)}
        except (KeyError, ValueError, InvalidTag) as e:
            raise HTTPException(422, f"cannot undo: {e!r}")

    return app


def __getattr__(name):  # `uvicorn basivo_pii.server:app` builds the app on first access
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
