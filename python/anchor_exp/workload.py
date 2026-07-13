"""Canonical uncompressed workload format.

The file is structure-of-arrays so the benchmark loader can read each component
directly into its final anonymous allocation.  Every array begins at a 64-byte
boundary.  The 128-byte little-endian header is::

    magic[8], version:u16, endian:u8, endpoint:u8,
    dimension:u32, flags:u32, header_size:u32,
    n_R:u64, n_S:u64,
    offsets for R ids/lower/upper and S ids/lower/upper (six u64),
    file_size:u64, logical_payload_sha256[32].

Coordinates are row-major ``(object, dimension)``.  The logical payload digest
covers the six arrays in header order and excludes alignment padding.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

from .stable_hash import canonical_json_bytes, hash_file


MAGIC = b"ANCHORW\0"
FORMAT_VERSION = 1
ENDIAN_LITTLE = 1
HEADER_SIZE = 128
ALIGNMENT = 64
HEADER = struct.Struct("<8sHBBIII" + "Q" * 9 + "32s")
assert HEADER.size == HEADER_SIZE

ENDPOINT_CODES = {"float64": 1, "int64": 2}
CODE_ENDPOINTS = {value: key for key, value in ENDPOINT_CODES.items()}
ENDPOINT_DTYPES = {"float64": np.dtype("<f8"), "int64": np.dtype("<i8")}
IDS_DTYPE = np.dtype("<u8")


class WorkloadFormatError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Workload:
    path: pathlib.Path
    endpoint_type: str
    dimension: int
    r_ids: np.ndarray
    r_lower: np.ndarray
    r_upper: np.ndarray
    s_ids: np.ndarray
    s_lower: np.ndarray
    s_upper: np.ndarray
    payload_sha256: str
    file_sha256: str | None = None

    @property
    def n_r(self) -> int:
        return int(self.r_ids.size)

    @property
    def n_s(self) -> int:
        return int(self.s_ids.size)


def _align(offset: int) -> int:
    return (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _as_ids(name: str, values: Any) -> np.ndarray:
    # Object dtype avoids NumPy promoting a mixed small/large Python-int list to
    # float64 before we have checked the full uint64 domain.
    original = np.asarray(values, dtype=object)
    if original.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    integers = original.tolist()
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        or int(value) >= 2**64
        for value in integers
    ):
        raise ValueError(f"{name} values must lie in uint64")
    array = np.ascontiguousarray(integers, dtype=IDS_DTYPE)
    if np.unique(array).size != array.size:
        raise ValueError(f"{name} contains duplicate object ids")
    return array


def _as_coordinates(name: str, values: Any, dtype: np.dtype) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (objects, dimensions)")
    return array


def _validate_arrays(
    *,
    r_ids: Any,
    r_lower: Any,
    r_upper: Any,
    s_ids: Any,
    s_lower: Any,
    s_upper: Any,
    endpoint_type: str,
) -> tuple[np.ndarray, ...]:
    if endpoint_type not in ENDPOINT_DTYPES:
        raise ValueError("endpoint_type must be 'float64' or 'int64'")
    dtype = ENDPOINT_DTYPES[endpoint_type]
    rid = _as_ids("r_ids", r_ids)
    sid = _as_ids("s_ids", s_ids)
    rlo = _as_coordinates("r_lower", r_lower, dtype)
    rhi = _as_coordinates("r_upper", r_upper, dtype)
    slo = _as_coordinates("s_lower", s_lower, dtype)
    shi = _as_coordinates("s_upper", s_upper, dtype)
    if rlo.shape != rhi.shape or slo.shape != shi.shape:
        raise ValueError("lower and upper arrays must have identical shapes")
    if rlo.shape[0] != rid.size or slo.shape[0] != sid.size:
        raise ValueError("id and coordinate object counts differ")
    if rlo.shape[1] != slo.shape[1] or rlo.shape[1] < 1:
        raise ValueError("both sides must use the same positive dimension")
    if endpoint_type == "float64":
        if not all(np.isfinite(a).all() for a in (rlo, rhi, slo, shi)):
            raise ValueError("floating endpoint arrays contain NaN or infinity")
    if not np.all(rlo < rhi) or not np.all(slo < shi):
        raise ValueError("canonical performance workloads cannot contain empty boxes")
    return rid, rlo, rhi, sid, slo, shi


def _array_bytes(array: np.ndarray) -> memoryview:
    return memoryview(array).cast("B")


def _summary(array: np.ndarray) -> list[int | float]:
    if array.shape[0] == 0:
        return []
    values = np.min(array, axis=0)
    return [item.item() for item in values]


def _summary_max(array: np.ndarray) -> list[int | float]:
    if array.shape[0] == 0:
        return []
    values = np.max(array, axis=0)
    return [item.item() for item in values]


def write_workload(
    path: str | pathlib.Path,
    *,
    r_ids: Any,
    r_lower: Any,
    r_upper: Any,
    s_ids: Any,
    s_lower: Any,
    s_upper: Any,
    endpoint_type: str,
    metadata: Mapping[str, Any] | None = None,
    manifest_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Validate and atomically write a canonical workload and its manifest."""

    arrays = _validate_arrays(
        r_ids=r_ids,
        r_lower=r_lower,
        r_upper=r_upper,
        s_ids=s_ids,
        s_lower=s_lower,
        s_upper=s_upper,
        endpoint_type=endpoint_type,
    )
    rid, rlo, rhi, sid, slo, shi = arrays
    dimension = int(rlo.shape[1])

    offsets: list[int] = []
    cursor = HEADER_SIZE
    for array in arrays:
        cursor = _align(cursor)
        offsets.append(cursor)
        cursor += array.nbytes
    file_size = cursor

    payload_digest = hashlib.sha256()
    for array in arrays:
        payload_digest.update(_array_bytes(array))
    payload_hash = payload_digest.digest()
    header = HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        ENDIAN_LITTLE,
        ENDPOINT_CODES[endpoint_type],
        dimension,
        0,
        HEADER_SIZE,
        int(rid.size),
        int(sid.size),
        *offsets,
        file_size,
        payload_hash,
    )

    destination = pathlib.Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb", buffering=0) as stream:
            stream.write(header)
            position = HEADER_SIZE
            for offset, array in zip(offsets, arrays, strict=True):
                if offset > position:
                    stream.write(b"\0" * (offset - position))
                stream.write(_array_bytes(array))
                position = offset + array.nbytes
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    file_hash = hash_file(destination)
    manifest: dict[str, Any] = {
        "schema_version": "anchor-workload-manifest-v1",
        "format": {
            "magic": MAGIC.rstrip(b"\0").decode("ascii"),
            "version": FORMAT_VERSION,
            "endianness": "little",
            "layout": "soa-row-major-aligned64",
        },
        "workload": {
            "file_name": destination.name,
            "file_size_bytes": file_size,
            "sha256": file_hash,
            "payload_sha256": payload_hash.hex(),
            "endpoint_type": endpoint_type,
            "dimension": dimension,
            "N_total": int(rid.size + sid.size),
            "n_R": int(rid.size),
            "n_S": int(sid.size),
            "R_ids_sha256": hashlib.sha256(_array_bytes(rid)).hexdigest(),
            "S_ids_sha256": hashlib.sha256(_array_bytes(sid)).hexdigest(),
            "coordinate_min": {
                "R": _summary(rlo),
                "S": _summary(slo),
            },
            "coordinate_max": {
                "R": _summary_max(rhi),
                "S": _summary_max(shi),
            },
        },
        "metadata": dict(metadata or {}),
    }
    mpath = (
        pathlib.Path(manifest_path).resolve()
        if manifest_path is not None
        else pathlib.Path(str(destination) + ".manifest.json")
    )
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mtemporary = mpath.with_name(f".{mpath.name}.tmp-{os.getpid()}")
    try:
        with mtemporary.open("wb") as stream:
            stream.write(canonical_json_bytes(manifest) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(mtemporary, mpath)
    finally:
        if mtemporary.exists():
            mtemporary.unlink()
    return manifest


def _read_header(path: pathlib.Path) -> tuple[Any, ...]:
    with path.open("rb") as stream:
        data = stream.read(HEADER_SIZE)
    if len(data) != HEADER_SIZE:
        raise WorkloadFormatError("truncated workload header")
    fields = HEADER.unpack(data)
    magic, version, endian, endpoint_code, dimension, flags, header_size = fields[:7]
    if magic != MAGIC:
        raise WorkloadFormatError("invalid workload magic")
    if version != FORMAT_VERSION or header_size != HEADER_SIZE:
        raise WorkloadFormatError("unsupported workload format version")
    if endian != ENDIAN_LITTLE:
        raise WorkloadFormatError("only canonical little-endian workloads are supported")
    if endpoint_code not in CODE_ENDPOINTS:
        raise WorkloadFormatError("unknown endpoint type code")
    if dimension < 1 or flags != 0:
        raise WorkloadFormatError("invalid dimension or unsupported header flags")
    return fields


def read_workload(
    path: str | pathlib.Path,
    *,
    verify_payload: bool = True,
    verify_file: bool = False,
) -> Workload:
    """Memory-map a workload; no coordinate copy is made."""

    source = pathlib.Path(path).resolve()
    fields = _read_header(source)
    endpoint_code = fields[3]
    dimension = int(fields[4])
    n_r, n_s = map(int, fields[7:9])
    offsets = tuple(map(int, fields[9:15]))
    declared_size = int(fields[15])
    payload_hash = fields[16]
    actual_size = source.stat().st_size
    if declared_size != actual_size:
        raise WorkloadFormatError("file size differs from header")
    endpoint_type = CODE_ENDPOINTS[endpoint_code]
    dtype = ENDPOINT_DTYPES[endpoint_type]
    shapes = ((n_r,), (n_r, dimension), (n_r, dimension), (n_s,), (n_s, dimension), (n_s, dimension))
    dtypes = (IDS_DTYPE, dtype, dtype, IDS_DTYPE, dtype, dtype)
    arrays: list[np.ndarray] = []
    previous_end = HEADER_SIZE
    for offset, shape, item_dtype in zip(offsets, shapes, dtypes, strict=True):
        if offset % ALIGNMENT or offset < previous_end:
            raise WorkloadFormatError("array offset is unaligned or overlaps its predecessor")
        count = int(np.prod(shape, dtype=np.int64))
        end = offset + count * item_dtype.itemsize
        if end > declared_size:
            raise WorkloadFormatError("array extends beyond end of file")
        arrays.append(np.memmap(source, mode="r", dtype=item_dtype, offset=offset, shape=shape, order="C"))
        previous_end = end
    if previous_end != declared_size:
        raise WorkloadFormatError("unexpected trailing bytes")
    if verify_payload:
        digest = hashlib.sha256()
        for array in arrays:
            digest.update(_array_bytes(array))
        if digest.digest() != payload_hash:
            raise WorkloadFormatError("logical payload SHA-256 mismatch")
    rid, rlo, rhi, sid, slo, shi = arrays
    # Header validation is intentionally repeated here so corrupted files never
    # reach algorithm code even when payload verification is disabled.
    if endpoint_type == "float64" and not all(
        np.isfinite(a).all() for a in (rlo, rhi, slo, shi)
    ):
        raise WorkloadFormatError("non-finite floating endpoint")
    if not np.all(rlo < rhi) or not np.all(slo < shi):
        raise WorkloadFormatError("empty or inverted box in canonical workload")
    return Workload(
        path=source,
        endpoint_type=endpoint_type,
        dimension=dimension,
        r_ids=rid,
        r_lower=rlo,
        r_upper=rhi,
        s_ids=sid,
        s_lower=slo,
        s_upper=shi,
        payload_sha256=payload_hash.hex(),
        file_sha256=hash_file(source) if verify_file else None,
    )


def read_manifest(path: str | pathlib.Path) -> dict[str, Any]:
    manifest_path = pathlib.Path(path)
    if not manifest_path.name.endswith(".json"):
        manifest_path = pathlib.Path(str(manifest_path) + ".manifest.json")
    with manifest_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != "anchor-workload-manifest-v1":
        raise WorkloadFormatError("unsupported workload manifest schema")
    return value


def pretouch(workload: Workload) -> int:
    """Read one byte per resident page and return a checksum-like accumulator."""

    page = 4096
    accumulator = 0
    for array in (
        workload.r_ids,
        workload.r_lower,
        workload.r_upper,
        workload.s_ids,
        workload.s_lower,
        workload.s_upper,
    ):
        raw = _array_bytes(array)
        for offset in range(0, len(raw), page):
            accumulator ^= int(raw[offset])
        if raw:
            accumulator ^= int(raw[-1])
    return accumulator
