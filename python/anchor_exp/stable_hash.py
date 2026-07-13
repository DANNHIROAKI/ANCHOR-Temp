"""Stable hashing primitives used by every dataset and experiment component.

The research specification deliberately leaves the exact canonical-JSON dialect
open.  This repository freezes ``anchor-canonical-json-v1``: dictionaries have
string keys, strings are NFC-normalized, tuples are arrays, finite ``-0.0`` is
normalized to ``0.0``, and JSON is emitted as sorted UTF-8 without whitespace.
The version tag is included in every digest so a future encoding cannot silently
change existing workload identities.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any


CANONICAL_JSON_VERSION = "anchor-canonical-json-v1"


def _normalize(value: Any) -> Any:
    """Convert supported values to the frozen JSON data model."""

    # NumPy support without making NumPy a dependency of this module.
    if value.__class__.__module__.startswith("numpy"):
        if getattr(value, "ndim", 0) > 0 and hasattr(value, "tolist"):
            value = value.tolist()
        elif hasattr(value, "item"):
            value = value.item()
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, pathlib.Path):
        value = str(value)
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("stable JSON does not permit NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("stable JSON dictionary keys must be strings")
            nkey = unicodedata.normalize("NFC", key)
            if nkey in normalized:
                raise ValueError(f"keys collide after Unicode normalization: {key!r}")
            normalized[nkey] = _normalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise TypeError(f"unsupported stable JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical UTF-8 encoding of *value*."""

    normalized = _normalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_hash(tag: str, *fields: Any) -> bytes:
    """Return the full SHA-256 digest for a tagged ordered field tuple."""

    envelope = {
        "fields": list(fields),
        "tag": tag,
        "version": CANONICAL_JSON_VERSION,
    }
    return hashlib.sha256(canonical_json_bytes(envelope)).digest()


def stable_hash_hex(tag: str, *fields: Any) -> str:
    return stable_hash(tag, *fields).hex()


def stable_hash128(tag: str, *fields: Any) -> int:
    """Interpret the first 128 digest bits as an unsigned big-endian integer."""

    return int.from_bytes(stable_hash(tag, *fields)[:16], "big", signed=False)


def stable_id64(tag: str, *fields: Any) -> int:
    """Derive a uint64 object id; callers constructing sets must detect collisions."""

    return int.from_bytes(stable_hash(tag, *fields)[:8], "big", signed=False)


def stable_ids64(tag: str, records: Iterable[Any]) -> list[int]:
    """Derive IDs and reject the astronomically unlikely truncation collision."""

    result: list[int] = []
    seen: dict[int, bytes] = {}
    for record in records:
        digest = stable_hash(tag, record)
        identifier = int.from_bytes(digest[:8], "big", signed=False)
        previous = seen.get(identifier)
        if previous is not None and previous != digest:
            raise RuntimeError(
                "64-bit stable-id collision; change the id mapping explicitly "
                "instead of silently merging objects"
            )
        seen[identifier] = digest
        result.append(identifier)
    return result


def hash_file(path: str | pathlib.Path, chunk_bytes: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
