"""Call a running basivo-pii server end to end and print what comes back.

    BASIVO_PII_API_KEY=... uv run --frozen python scripts/smoke.py [http://localhost:8000]
"""
import json
import os
import sys

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
H = {"X-API-Key": os.environ["BASIVO_PII_API_KEY"]}
TEXT = ("Hi, I am Priya Sharma (PAN ABCPE1234F). Call +91 98450 12345 or mail priya.s@example.com. "
        "IBAN DE89 3704 0044 0532 0130 00. Priya Sharma lives at 14 MG Road, Bengaluru, India.")


def post(path, body):
    r = httpx.post(URL + path, json=body, headers=H, timeout=120)
    r.raise_for_status()
    return r.json()


print("health:", httpx.get(URL + "/health").json())
print("no key:", httpx.post(URL + "/analyze", json={"text": "x"}).status_code)
print("\ntext:    ", TEXT)
r = post("/redact", {"text": TEXT})
print("redact:  ", r["text"], "| has_direct:", r["has_direct"])
for i in r["items"]:
    print(f"   {i['label']:<13} {i['score']:<6} {i['source']}")
ops = {"PERSON_NAME": {"type": "pseudonymize"}, "PHONE": {"type": "mask", "chars": 10}, "EMAIL": {"type": "hash"},
       "BANK_ACCOUNT": {"type": "encrypt"}}
r = post("/redact", {"text": TEXT, "operators": ops})
print("\noperators:", json.dumps(ops))
print("redact:  ", r["text"])
print("mapping: ", r["mapping"])
back = post("/deanonymize", {"text": r["text"], "items": r["items"], "mapping": r["mapping"]})["text"]
print("undo:    ", back)
print("names and IBAN back:", all(v in back for v in r["mapping"].values())
      and "DE89 3704 0044 0532 0130 00" in back)
