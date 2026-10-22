"""Operators: each one gets the original value and the label (plus its params) and returns the new value.

hash and encrypt need a `key` (the Anonymizer's secret when the config does not give one). pseudonymize gets the
call's `mapping` from the Anonymizer.
"""
import base64
import hashlib
import hmac
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def replace(value: str, label: str, new_value: str | None = None) -> str:
    return f"[{label}]" if new_value is None else new_value


def redact(value: str, label: str) -> str:
    return ""


def mask(value: str, label: str, char: str = "*", chars: int | None = None, from_end: bool = False) -> str:
    """Mask `chars` characters (all when None) from the start, or from the end when from_end."""
    if len(char) != 1:
        raise ValueError("mask: char must be one character")
    n = len(value) if chars is None else min(chars, len(value))
    return value[:len(value) - n] + char * n if from_end else char * n + value[n:]


def keep(value: str, label: str) -> str:
    return value


def pseudonymize(value: str, label: str, mapping: dict[str, str], format: str = "<{label}_{n}>") -> str:
    """The same value gets the same placeholder; numbers count from 1 per label, left to right. mapping is
    {placeholder: original} and grows; pass it back in to keep numbers stable across texts."""
    n = 1
    while (placeholder := format.format(label=label, n=n)) in mapping:
        if mapping[placeholder] == value:
            return placeholder
        n += 1
    mapping[placeholder] = value
    return placeholder


def _key(key: str | bytes | None) -> bytes:
    if not key:
        raise ValueError("hash and encrypt need a key (operator param 'key' or the Anonymizer's secret)")
    return key.encode() if isinstance(key, str) else key


def hmac_hash(value: str, label: str, key: str | bytes | None = None, length: int = 16) -> str:
    """HMAC-SHA256 with a secret key: the same value always gives the same hash (records can be linked), and without
    the key nobody can test guesses."""
    return hmac.new(_key(key), value.encode(), hashlib.sha256).hexdigest()[:length]


def encrypt(value: str, label: str, key: str | bytes | None = None) -> str:
    """AES-256-GCM (the key is hashed to 32 bytes), random nonce; url-safe base64 of nonce + ciphertext + tag."""
    nonce = os.urandom(12)
    sealed = AESGCM(hashlib.sha256(_key(key)).digest()).encrypt(nonce, value.encode(), None)
    return base64.urlsafe_b64encode(nonce + sealed).decode()


def decrypt(value: str, key: str | bytes | None = None) -> str:
    raw = base64.urlsafe_b64decode(value)
    return AESGCM(hashlib.sha256(_key(key)).digest()).decrypt(raw[:12], raw[12:], None).decode()


OPERATORS = {"replace": replace, "redact": redact, "mask": mask, "keep": keep, "pseudonymize": pseudonymize,
             "hash": hmac_hash, "encrypt": encrypt}
REVERSIBLE = {"pseudonymize", "encrypt"}
