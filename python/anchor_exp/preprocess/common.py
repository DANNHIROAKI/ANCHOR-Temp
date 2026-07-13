"""Shared deterministic preprocessing utilities."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np

from anchor_exp.stable_hash import canonical_json_bytes, hash_file, stable_hash


class DatasetConstructionError(RuntimeError):
    """The frozen rules cannot construct the requested workload."""


class OptionalDependencyError(RuntimeError):
    """A deliberately optional, heavy dependency is unavailable."""


class FrozenMetadataError(RuntimeError):
    """Frozen metadata or a checksum disagrees with the local artifact."""


T = TypeVar("T")
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def require_module(name: str, *, extra: str, purpose: str) -> Any:
    """Import an optional dependency or fail with an actionable message."""

    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - exercised without extras
        raise OptionalDependencyError(
            f"{purpose} requires optional module {name!r}; install the {extra!r} "
            "dependency set. No approximate fallback is used."
        ) from exc


def stable_sort_key(tag: str, *fields: Any) -> tuple[bytes, bytes]:
    """Full-digest ordering with canonical fields as the collision tie-break."""

    return stable_hash(tag, *fields), canonical_json_bytes(list(fields))


def largest_remainder_counts(
    sizes: Mapping[T, int],
    total: int,
    *,
    tie_tag: str,
    tie_prefix: Sequence[Any] = (),
) -> dict[T, int]:
    """Allocate *total* proportionally using exact integer remainders."""

    if total < 0:
        raise ValueError("total must be non-negative")
    denominator = sum(int(value) for value in sizes.values())
    if denominator < total:
        raise DatasetConstructionError(
            f"requested {total} objects from a pool of only {denominator}"
        )
    if denominator == 0:
        if total:
            raise DatasetConstructionError("cannot allocate from an empty pool")
        return {key: 0 for key in sizes}
    allocation = {
        key: (total * int(size)) // denominator for key, size in sizes.items()
    }
    missing = total - sum(allocation.values())
    ranked = sorted(
        sizes,
        key=lambda key: (
            -((total * int(sizes[key])) % denominator),
            stable_sort_key(tie_tag, *tie_prefix, key),
        ),
    )
    for key in ranked[:missing]:
        allocation[key] += 1
    if sum(allocation.values()) != total:
        raise AssertionError("largest-remainder allocation failed conservation")
    return allocation


def ensure_unique_uint64(ids: Iterable[int], *, context: str) -> np.ndarray:
    array = np.asarray(list(ids), dtype=np.uint64)
    if np.unique(array).size != array.size:
        raise DatasetConstructionError(f"64-bit object-id collision in {context}")
    return array


def checked_int64(value: int, *, context: str) -> np.int64:
    info = np.iinfo(np.int64)
    integer = int(value)
    if integer < info.min or integer > info.max:
        raise DatasetConstructionError(f"int64 overflow while constructing {context}")
    return np.int64(integer)


def validate_finite_box(lower: Sequence[float], upper: Sequence[float], *, context: str) -> None:
    if len(lower) != len(upper) or not lower:
        raise DatasetConstructionError(f"invalid dimension in {context}")
    if not all(math.isfinite(float(value)) for value in (*lower, *upper)):
        raise DatasetConstructionError(f"non-finite endpoint in {context}")
    if not all(float(lo) < float(hi) for lo, hi in zip(lower, upper, strict=True)):
        raise DatasetConstructionError(f"empty or inverted box in {context}")


def sha256_tree(paths: Iterable[pathlib.Path], *, root: pathlib.Path) -> str:
    """Hash relative names and file digests in stable order."""

    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=lambda item: str(item)):
        relative = path.relative_to(root.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(hash_file(path)))
    return digest.hexdigest()


def write_json(path: str | pathlib.Path, value: Mapping[str, Any]) -> None:
    destination = pathlib.Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(canonical_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: str | pathlib.Path) -> Any:
    with pathlib.Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_checksum(path: str | pathlib.Path, expected: str, *, label: str) -> str:
    actual = hash_file(path)
    return validate_sha256_digest(actual, expected, label=label)


def validate_sha256_digest(actual: str, expected: str, *, label: str) -> str:
    """Validate a frozen SHA-256 value and return its normalized form.

    Source-tree digests are configuration inputs, just like individual file
    digests.  Keeping this check separate from :func:`validate_checksum` lets
    streaming preprocessors verify a digest that was computed over many files
    without re-reading the tree.
    """

    actual_normalized = str(actual).strip().lower()
    expected_normalized = str(expected).strip().lower()
    if SHA256_PATTERN.fullmatch(expected_normalized) is None:
        raise FrozenMetadataError(f"{label} expected SHA-256 is not 64 hexadecimal digits")
    if SHA256_PATTERN.fullmatch(actual_normalized) is None:
        raise FrozenMetadataError(f"{label} computed an invalid SHA-256 value")
    if actual_normalized != expected_normalized:
        raise FrozenMetadataError(
            f"{label} SHA-256 mismatch: expected {expected_normalized}, got {actual_normalized}"
        )
    return actual_normalized


def require_frozen_text(value: str, *, label: str) -> str:
    """Reject missing and placeholder provenance identifiers."""

    text = str(value).strip()
    if not text or "REPLACE" in text.upper():
        raise FrozenMetadataError(f"{label} must be a non-placeholder frozen identifier")
    return text
