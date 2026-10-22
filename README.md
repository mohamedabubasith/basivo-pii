# basivo-pii

Find PII in text and mask it: our fine-tuned model plus rules, our own code (no Presidio, no spaCy).

## Model access (required)

You can run basivo-pii anywhere: as a library, on your own servers, or as a service you host for your users. The
fine-tuned model [`abubasith86/pii-nemotron-mmbert-base-v1`](https://huggingface.co/abubasith86/pii-nemotron-mmbert-base-v1)
is **gated**: every person or organization that runs it needs the owner's approval first.

1. Sign in to Hugging Face and click **Request access** on the model page.
2. After the request is approved, create a **read** token at https://huggingface.co/settings/tokens.
3. Make the token available to basivo-pii: `export HF_TOKEN=hf_...`, or `hf auth login`, or `-e HF_TOKEN=...` for Docker.

Without approval, startup stops with `ModelAccessError` and these same steps. Don't share your token: each deployment
uses its own approved account.

## Flow (v1 doc, figure 2)

```
text ─► rules + validators ─┐
     └► model (1024-token windows, 256 overlap) ─┴─► resolve ─► findings ─► anonymize ─► masked text + items
```

1. **Model** (`detectors/model.py`): BIO tags become spans; a span's score is the mean probability of its tokens. Many
   texts go through the model in one batch.
2. **Rules** (`detectors/patterns.py`): regex + check digits + context words. A low score means "only with a context
   word near it" (below the 0.5 threshold otherwise).

   | Rule | Label | Check | Score (with context) |
   |---|---|---|---|
   | email | EMAIL | – | 0.95 |
   | card | CARD_NUMBER | Luhn | 0.9 (1.0) |
   | iban | BANK_ACCOUNT | mod-97 | 1.0 |
   | ipv4, ipv6 | IP_ADDRESS | valid address | 0.6 (0.95) |
   | url | URL | – | 0.6 |
   | phone_international | PHONE | 8–15 digits, starts with + | 0.6 (0.95) |
   | us_ssn | NATIONAL_ID | – | 0.4 (0.75) |
   | aadhaar | NATIONAL_ID | – | 0.3 (0.65) |
   | india_pan | TAX_ID | – | 0.7 (1.0) |
   | passport | PASSPORT | – | 0.2 (0.6) |
   | driving_licence | DRIVING_LICENCE | – | 0.15 (0.6) |
   | password | PASSWORD | value after password/pin, with a digit or symbol | 0.7 |
   | api_key | API_KEY | known prefixes (sk-, ghp_, AKIA, xox, AIza, hf_, JWT) | 0.9 |

3. **Resolve** (`resolve.py`): labels asked for → threshold per label → allow list → join parts of one thing (ADDRESS
   across spaces, commas, new lines; PERSON_NAME across spaces) → overlaps (higher score, then longer, then model) →
   **entity register**: every other mention of a direct identifier that was found is masked too (an ADDRESS only if it
   has a digit).
4. **Anonymize** (`anonymizer.py`, `operators.py`): an operator per label, `DEFAULT` for the rest, `[LABEL]` when
   nothing is set.

   | Operator | Params | Example | Undo |
   |---|---|---|---|
   | `replace` | `new_value` | `[PERSON_NAME]` | no |
   | `redact` | – | removed | no |
   | `mask` | `char` (`*`), `chars` (all), `from_end` | `**********12345` | no |
   | `keep` | – | unchanged, listed in items | – |
   | `pseudonymize` | `format` (`<{label}_{n}>`) | `<PERSON_NAME_1>`, same value same number | with the mapping |
   | `hash` | `key` (secret), `length` (16) | HMAC-SHA256, same value same hash | no |
   | `encrypt` | `key` (secret) | AES-256-GCM, base64 | with the key |

## Python

```python
from basivo_pii import Anonymizer, OperatorConfig, config

analyzer = config.analyzer(config.load("config.yaml"))     # or Analyzer(model, thresholds=..., patterns=...)
text = "Hi, I am Priya Sharma, card 4111 1111 1111 1111."
findings = analyzer.analyze(text)                           # [Finding(label, start, end, score, source)]
r = Anonymizer(secret="...").anonymize(text, findings, {"PERSON_NAME": OperatorConfig("pseudonymize")})
r.text      # Hi, I am <PERSON_NAME_1>, card [CARD_NUMBER].
r.items     # [Item(label, start, end, text, operator, score, source)], positions in the new text
r.mapping   # {"<PERSON_NAME_1>": "Priya Sharma"}; pass it to the next anonymize() to keep the numbers
Anonymizer(secret="...").deanonymize(r.text, r.items, r.mapping)
```

## REST

```
BASIVO_PII_API_KEY=... BASIVO_PII_SECRET=... uv run --extra server uvicorn basivo_pii.server:app --port 8000
```

Every call except `/health` needs the header `X-API-Key`. Request text is never logged.

| Endpoint | Body | Returns |
|---|---|---|
| `POST /analyze` | `{text or [texts], labels?, score_threshold?, allow_list?}` | `{findings, has_pii, has_direct}` (a list for a list) |
| `POST /redact` | the same + `operators?`, `mapping?` | `{text, items, has_pii, has_direct, mapping}`; a list gives `{results, mapping}`, one mapping for the whole batch |
| `POST /anonymize` | `{text, findings, operators?, mapping?}` | `{text, items, mapping}` |
| `POST /deanonymize` | `{text, items, mapping?}` | `{text}` |
| `GET /labels`, `/operators`, `/health` | – | lists, status |

Operators in a body: `{"DEFAULT": {"type": "replace"}, "CARD_NUMBER": {"type": "mask", "chars": 12}}`. An unknown
label or operator is an error (422), never dropped without a word. Limits: 256 texts, 200,000 characters each.

Docker: see the comment at the top of `Dockerfile`.

## Config

`config.yaml`: model (repo, pinned revision, window, stride, device, fp16, batch), thresholds per label, rules (switch
built-in ones off, add your own), operators per label.

## Checks

```
uv sync --extra server
uv run pytest -q                                   # 64 tests; test_model.py runs only if the model is in the HF cache
BASIVO_PII_API_KEY=... uv run python scripts/smoke.py http://localhost:8000      # a running server, end to end
uv run python scripts/evaluate.py <test.jsonl> [--limit N]   # model alone vs the whole wrapper, our scoring
```
