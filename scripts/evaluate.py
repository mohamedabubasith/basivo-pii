"""Score basivo-pii end to end (model + rules + resolve + register) on a test file, next to the model alone, with the
same scoring as our model card and the Presidio baselines (pii/eval/evaluate.py score).

    uv run --frozen python scripts/evaluate.py <test.jsonl> [--limit N] [--out ../reports/basivo_pii/<name>]
    uv run --frozen python scripts/evaluate.py ../data/benchmarks/nemotron_our_clean_v1_test.jsonl   # GPU, all rows
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root, for pii.eval.evaluate
from pii.eval.evaluate import score  # noqa: E402

from basivo_pii import config  # noqa: E402
from basivo_pii.analyzer import Analyzer  # noqa: E402


def metrics(docs, preds):
    d, t = score(docs, preds, typed=False), score(docs, preds, typed=True)
    return {"docs": d["docs"], "recall": d["overlap"]["recall"], "precision": d["overlap"]["precision"],
            "f2": d["overlap"]["f2"], "typed_f1": t["strict"]["f1"], "leak": d["doc_leak_rate"],
            "overmasking": d["overmasking_rate"], "recall_per_label": d["recall_per_label"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    ap.add_argument("--batch", type=int, default=32, help="texts per analyze_batch call")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "reports" / "basivo_pii"))
    args = ap.parse_args()

    docs = [json.loads(line) for line in open(args.test, encoding="utf-8")][:args.limit]
    cfg = config.load(args.config)
    full: Analyzer = config.analyzer(cfg)
    runs = {"model only": [], "basivo-pii (model + rules + register)": []}
    sources = collections.Counter()
    t0 = time.time()
    for i in range(0, len(docs), args.batch):
        texts = [d["text"] for d in docs[i:i + args.batch]]
        raw = full.detector.detect(texts)
        runs["model only"] += [[{"start": f.start, "end": f.end, "label": f.label} for f in r
                                if f.score >= full.thresholds.get(f.label, full.thresholds.get("default", 0))]
                               for r in raw]
        found = full.analyze_batch(texts)
        sources.update(f.source for r in found for f in r)
        runs["basivo-pii (model + rules + register)"] += [
            [{"start": f.start, "end": f.end, "label": f.label} for f in r] for r in found]
        print(f"\r{i + len(texts):,}/{len(docs):,}", end="", flush=True)
    seconds = time.time() - t0

    result = {"test": args.test, "docs": len(docs), "model": cfg["model"], "seconds": round(seconds, 1),
              "finding_sources": dict(sources.most_common()),
              "runs": {name: metrics(docs, preds) for name, preds in runs.items()}}
    out = Path(args.out) / f"{Path(args.test).stem}_{len(docs)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\n{'run':<42} recall  prec    F2      typedF1 leak    overmask")
    for name, m in result["runs"].items():
        print(f"{name:<42} {m['recall']:<7} {m['precision']:<7} {m['f2']:<7} {m['typed_f1']:<7} {m['leak']:<7} "
              f"{m['overmasking']}")
    print("finding sources:", result["finding_sources"], f"| {seconds:.0f}s for both runs | {out}")


if __name__ == "__main__":
    main()
