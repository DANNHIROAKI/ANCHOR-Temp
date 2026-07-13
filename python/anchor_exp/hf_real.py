"""Acquire frozen Hugging Face datasets and materialize real workloads.

This module deliberately starts from the three published Parquet datasets.  It
contains no GIS ingestion, GeoLife ``.plt`` parser, image decoder, detector, or
RPN inference path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import pathlib
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from anchor_exp.stable_hash import (
    canonical_json_bytes,
    hash_file,
    stable_hash,
    stable_hash_hex,
)
from anchor_exp.workload import read_manifest, read_workload, write_workload


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "real_data.json"
DEFAULT_LOCK = REPOSITORY_ROOT / "data_sources.lock.json"
IMPORTER_ID = "anchor-hf-real-import-v2"
SHA256_HEX = frozenset("0123456789abcdef")


class DataPreparationError(RuntimeError):
    """A source or output failed a frozen data-preparation invariant."""


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in SHA256_HEX for character in text)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise DataPreparationError(f"JSON root must be an object: {path}")
    return value


def load_real_config(path: str | pathlib.Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = pathlib.Path(path).resolve()
    value = _load_json(source)
    if value.get("schema_version") != "anchor-hf-real-data-config-v2":
        raise DataPreparationError(f"unsupported real-data configuration: {source}")
    for name in ("cmab", "geolife", "coco"):
        if not isinstance(value.get(name), Mapping):
            raise DataPreparationError(f"real-data configuration lacks {name}")
    value["_path"] = str(source)
    return value


def load_source_lock(path: str | pathlib.Path = DEFAULT_LOCK) -> dict[str, Any]:
    source = pathlib.Path(path).resolve()
    value = _load_json(source)
    if value.get("schema_version") != "anchor-hf-data-sources-lock-v2":
        raise DataPreparationError(f"unsupported source lock: {source}")
    sources = value.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {"cmab", "geolife", "coco"}:
        raise DataPreparationError(
            "source lock must contain exactly cmab, geolife, and coco"
        )
    for name, source_spec in sources.items():
        if not isinstance(source_spec, Mapping):
            raise DataPreparationError(f"invalid source entry: {name}")
        repo_id = str(source_spec.get("repo_id", ""))
        revision = str(source_spec.get("revision", ""))
        if (
            repo_id.count("/") != 1
            or len(revision) != 40
            or any(c not in SHA256_HEX for c in revision)
        ):
            raise DataPreparationError(f"unfrozen Hugging Face identity for {name}")
        assets = source_spec.get("assets")
        if not isinstance(assets, list) or not assets:
            raise DataPreparationError(f"source {name} has no locked assets")
        seen: set[str] = set()
        for asset in assets:
            if not isinstance(asset, Mapping):
                raise DataPreparationError(f"invalid asset in source {name}")
            relative = pathlib.PurePosixPath(str(asset.get("path", "")))
            digest = str(asset.get("sha256", "")).lower()
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise DataPreparationError(f"unsafe source path: {relative}")
            if relative.as_posix() in seen or not _is_sha256(digest):
                raise DataPreparationError(
                    f"invalid or duplicate locked asset: {relative}"
                )
            if int(asset.get("size", 0)) <= 0:
                raise DataPreparationError(f"invalid locked size for {relative}")
            seen.add(relative.as_posix())
    value["_path"] = str(source)
    return value


def resolve_configuration_paths(
    config_path: str | pathlib.Path = DEFAULT_CONFIG,
    lock_path: str | pathlib.Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pathlib.Path, pathlib.Path]:
    config_file = pathlib.Path(config_path).resolve()
    config = load_real_config(config_file)
    if lock_path is None:
        configured = pathlib.Path(str(config["source_lock"]))
        lock_file = (
            configured
            if configured.is_absolute()
            else config_file.parent.parent / configured
        )
    else:
        lock_file = pathlib.Path(lock_path)
    lock_file = lock_file.resolve()
    return config, load_source_lock(lock_file), config_file, lock_file


@dataclasses.dataclass(frozen=True, slots=True)
class HubAsset:
    path: str
    size: int
    sha256: str
    rows: int | None = None
    role: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HubAsset":
        return cls(
            path=str(value["path"]),
            size=int(value["size"]),
            sha256=str(value["sha256"]).lower(),
            rows=int(value["rows"]) if value.get("rows") is not None else None,
            role=str(value["role"]) if value.get("role") is not None else None,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedRemote:
    endpoint: str
    request_url: str
    content_url: str
    size: int
    linked_sha256: str | None


class HubClient:
    """Small pinned-revision Hub client with an official-to-mirror fallback."""

    def __init__(
        self, endpoint: str | None = None, *, timeout_seconds: int = 30
    ) -> None:
        configured = endpoint or os.environ.get("HF_ENDPOINT")
        candidates = [configured] if configured else []
        candidates.extend(("https://huggingface.co", "https://hf-mirror.com"))
        self.endpoints: list[str] = []
        for candidate in candidates:
            if candidate and candidate.rstrip("/") not in self.endpoints:
                self.endpoints.append(candidate.rstrip("/"))
        self.timeout_seconds = int(timeout_seconds)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"{IMPORTER_ID}/1"})
        self._active_endpoint: str | None = None

    @staticmethod
    def _url(endpoint: str, repo_id: str, revision: str, path: str) -> str:
        quoted_path = "/".join(
            requests.utils.quote(part, safe="") for part in path.split("/")
        )
        return f"{endpoint}/datasets/{repo_id}/resolve/{revision}/{quoted_path}"

    def resolve(
        self,
        repo_id: str,
        revision: str,
        path: str,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> ResolvedRemote:
        endpoints = list(self.endpoints)
        if self._active_endpoint in endpoints:
            endpoints.remove(self._active_endpoint)
            endpoints.insert(0, self._active_endpoint)
        failures: list[str] = []
        for endpoint in endpoints:
            url = self._url(endpoint, repo_id, revision, path)
            try:
                response = self.session.head(
                    url,
                    allow_redirects=False,
                    timeout=(self.timeout_seconds, self.timeout_seconds),
                )
                if response.status_code not in {200, 302, 303, 307, 308}:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code}", response=response
                    )
                actual_revision = response.headers.get("X-Repo-Commit")
                if actual_revision and actual_revision != revision:
                    raise DataPreparationError(
                        f"Hub resolved {repo_id}/{path} to {actual_revision}, expected {revision}"
                    )
                size_text = response.headers.get("X-Linked-Size")
                if size_text is None and response.status_code == 200:
                    size_text = response.headers.get("Content-Length")
                if size_text is None and expected_size is not None:
                    # Git-backed small files redirect through a cache response whose
                    # Content-Length describes the redirect body, not the file.
                    size_text = str(expected_size)
                if size_text is None:
                    raise DataPreparationError(
                        f"Hub did not report a size for {repo_id}/{path}"
                    )
                size = int(size_text)
                if expected_size is not None and size != int(expected_size):
                    raise DataPreparationError(
                        f"Hub size mismatch for {repo_id}/{path}: {size} != {expected_size}"
                    )
                linked = response.headers.get("X-Linked-Etag")
                linked_sha = linked.strip('"').lower() if linked else None
                if linked_sha is not None and not _is_sha256(linked_sha):
                    linked_sha = None
                if (
                    expected_sha256
                    and linked_sha
                    and linked_sha != expected_sha256.lower()
                ):
                    raise DataPreparationError(
                        f"Hub linked SHA-256 mismatch for {repo_id}/{path}"
                    )
                content_url = response.headers.get("Location") or url
                self._active_endpoint = endpoint
                return ResolvedRemote(endpoint, url, content_url, size, linked_sha)
            except (
                requests.RequestException,
                DataPreparationError,
                ValueError,
            ) as error:
                failures.append(f"{endpoint}: {error}")
        raise DataPreparationError(
            f"cannot resolve pinned Hub asset {repo_id}/{path}: " + " | ".join(failures)
        )

    def download(
        self,
        repo_id: str,
        revision: str,
        asset: HubAsset,
        destination: pathlib.Path,
    ) -> pathlib.Path:
        destination = destination.resolve()
        if destination.is_file():
            if (
                destination.stat().st_size == asset.size
                and hash_file(destination) == asset.sha256
            ):
                return destination
            raise DataPreparationError(
                f"existing source asset failed its lock: {destination}"
            )
        remote = self.resolve(
            repo_id,
            revision,
            asset.path,
            expected_size=asset.size,
            expected_sha256=asset.sha256,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f".{destination.name}.partial")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > asset.size:
            partial.unlink()
            offset = 0
        elif offset == asset.size:
            if hash_file(partial) == asset.sha256:
                os.replace(partial, destination)
                return destination
            partial.unlink()
            offset = 0
        digest = hashlib.sha256()
        if offset:
            with partial.open("rb") as stream:
                while chunk := stream.read(4 << 20):
                    digest.update(chunk)
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            response = self.session.get(
                remote.request_url,
                headers=headers,
                allow_redirects=True,
                stream=True,
                timeout=(self.timeout_seconds, 120),
            )
            response.raise_for_status()
            if offset and response.status_code != 206:
                offset = 0
                digest = hashlib.sha256()
                mode = "wb"
            else:
                mode = "ab" if offset else "wb"
            with partial.open(mode) as stream:
                for chunk in response.iter_content(chunk_size=4 << 20):
                    if chunk:
                        stream.write(chunk)
                        digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        except requests.RequestException as error:
            raise DataPreparationError(
                f"download failed for {repo_id}/{asset.path}: {error}"
            ) from error
        if partial.stat().st_size != asset.size or digest.hexdigest() != asset.sha256:
            raise DataPreparationError(
                f"downloaded asset failed size/SHA-256: {asset.path}"
            )
        os.replace(partial, destination)
        return destination


class HubRangeReader(io.RawIOBase):
    """Seekable HTTP byte-range reader validated against a pinned Hub revision."""

    def __init__(
        self,
        client: HubClient,
        repo_id: str,
        revision: str,
        path: str,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.repo_id = repo_id
        self.revision = revision
        self.path = path
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self.remote = client.resolve(
            repo_id,
            revision,
            path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        self.position = 0
        self.request_count = 0
        self.bytes_transferred = 0

    @property
    def size(self) -> int:
        return self.remote.size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = int(offset)
        elif whence == io.SEEK_CUR:
            position = self.position + int(offset)
        elif whence == io.SEEK_END:
            position = self.size + int(offset)
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def _refresh(self) -> None:
        self.remote = self.client.resolve(
            self.repo_id,
            self.revision,
            self.path,
            expected_size=self.expected_size,
            expected_sha256=self.expected_sha256,
        )

    def readinto(self, buffer: Any) -> int:
        count = min(len(buffer), max(0, self.size - self.position))
        if count <= 0:
            return 0
        start = self.position
        end = start + count - 1
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.session.get(
                    self.remote.content_url,
                    headers={"Range": f"bytes={start}-{end}"},
                    allow_redirects=True,
                    timeout=(self.client.timeout_seconds, 120),
                )
                if response.status_code in {401, 403} and attempt < 2:
                    self._refresh()
                    continue
                response.raise_for_status()
                content = response.content
                content_range = response.headers.get("Content-Range")
                expected_range = f"bytes {start}-{end}/{self.size}"
                if (
                    response.status_code != 206
                    or content_range != expected_range
                    or len(content) != count
                ):
                    raise DataPreparationError(
                        f"invalid range response for {self.path}: status={response.status_code}, "
                        f"Content-Range={content_range!r}, bytes={len(content)}, "
                        f"expected={expected_range!r}/{count}"
                    )
                buffer[:count] = content
                self.position += count
                self.request_count += 1
                self.bytes_transferred += count
                return count
            except (requests.RequestException, DataPreparationError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    self._refresh()
        raise DataPreparationError(f"range read failed for {self.path}: {last_error}")


def _source_assets(source: Mapping[str, Any]) -> list[HubAsset]:
    return [HubAsset.from_mapping(item) for item in source["assets"]]


def _asset_by_role(source: Mapping[str, Any], role: str) -> HubAsset:
    matches = [asset for asset in _source_assets(source) if asset.role == role]
    if len(matches) != 1:
        raise DataPreparationError(
            f"expected exactly one source asset with role {role!r}"
        )
    return matches[0]


def _source_directory(
    data_root: pathlib.Path, name: str, source: Mapping[str, Any]
) -> pathlib.Path:
    slug = str(source["repo_id"]).replace("/", "--")
    return data_root / "sources" / "huggingface" / slug / str(source["revision"])


def download_static_source(
    name: str,
    *,
    data_root: pathlib.Path,
    lock: Mapping[str, Any],
    client: HubClient,
) -> dict[str, pathlib.Path]:
    source = lock["sources"][name]
    root = _source_directory(data_root, name, source)
    result: dict[str, pathlib.Path] = {}
    for asset in _source_assets(source):
        destination = root.joinpath(*pathlib.PurePosixPath(asset.path).parts)
        result[asset.path] = client.download(
            str(source["repo_id"]), str(source["revision"]), asset, destination
        )
    return result


def _atomic_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(canonical_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_metadata(
    name: str,
    source: Mapping[str, Any],
    *,
    lock_sha256: str,
    assets: Sequence[HubAsset] | None = None,
) -> dict[str, Any]:
    selected = list(assets if assets is not None else _source_assets(source))
    return {
        "kind": "huggingface_dataset",
        "name": name,
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "dataset_url": source["dataset_url"],
        "license": source["license"],
        "source_lock_sha256": lock_sha256,
        "assets": [dataclasses.asdict(asset) for asset in selected],
    }


def _common_metadata(
    dataset_id: str,
    source_name: str,
    source: Mapping[str, Any],
    *,
    lock_sha256: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "real_dataset": dataset_id,
        "source_identifier": f"{source['repo_id']}@{source['revision']}",
        "source_repository": source["repo_id"],
        "source_revision": source["revision"],
        "source": _source_metadata(source_name, source, lock_sha256=lock_sha256),
        "importer_id": IMPORTER_ID,
        "importer_sha256": hash_file(pathlib.Path(__file__)),
        "preprocessing_config_sha256": stable_hash_hex(
            "hf-real-preprocessing-config-v2", config
        ),
    }


def write_collection_manifest(
    directory: pathlib.Path,
    *,
    dataset_id: str,
    source_lock_sha256: str,
    workloads: Sequence[Mapping[str, Any]],
    extra_files: Sequence[str] = (),
) -> None:
    records: list[dict[str, Any]] = []
    for item in workloads:
        path = directory / str(item["file_name"])
        adjacent = pathlib.Path(str(path) + ".manifest.json")
        if not path.is_file() or not adjacent.is_file():
            raise DataPreparationError(f"missing workload artifact: {path}")
        records.append(
            {
                **dict(item),
                "size": path.stat().st_size,
                "sha256": hash_file(path),
                "adjacent_manifest": adjacent.name,
                "adjacent_manifest_sha256": hash_file(adjacent),
            }
        )
    extras: list[dict[str, Any]] = []
    for name in extra_files:
        path = directory / name
        if not path.is_file():
            raise DataPreparationError(f"missing collection sidecar: {path}")
        extras.append(
            {"file_name": name, "size": path.stat().st_size, "sha256": hash_file(path)}
        )
    manifest = {
        "schema_version": "anchor-workload-collection-v2",
        "dataset_id": dataset_id,
        "source_lock_sha256": source_lock_sha256,
        "workloads": records,
        "extra_files": extras,
    }
    _atomic_json(directory / "manifest.json", manifest)
    paths = [directory / "manifest.json"]
    for row in records:
        paths.extend(
            (directory / row["file_name"], directory / row["adjacent_manifest"])
        )
    paths.extend(directory / row["file_name"] for row in extras)
    lines = [
        f"{hash_file(path)}  {path.relative_to(directory).as_posix()}"
        for path in sorted(paths)
    ]
    temporary = directory / ".checksums.sha256.partial"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, directory / "checksums.sha256")


def verify_collection(
    directory: str | pathlib.Path,
    *,
    dataset_id: str,
    source_lock_path: str | pathlib.Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    root = pathlib.Path(directory).resolve()
    manifest_path = root / "manifest.json"
    checksums_path = root / "checksums.sha256"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise DataPreparationError(f"incomplete workload collection: {root}")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != "anchor-workload-collection-v2"
        or manifest.get("dataset_id") != dataset_id
    ):
        raise DataPreparationError(f"workload collection identity mismatch: {root}")
    lock_file = pathlib.Path(source_lock_path).resolve()
    expected_lock = hash_file(lock_file)
    if manifest.get("source_lock_sha256") != expected_lock:
        raise DataPreparationError(
            f"workload collection uses a different source lock: {root}"
        )
    source_name = {
        "CMAB-1M": "cmab",
        "GeoLife-3D-1M": "geolife",
        "GeoLife-4D-1M": "geolife",
        "COCO-1M": "coco",
    }.get(dataset_id)
    if source_name is None:
        raise DataPreparationError(f"unknown real workload collection: {dataset_id}")
    expected_source = load_source_lock(lock_file)["sources"][source_name]
    observed: dict[str, tuple[int, str]] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        pure = pathlib.PurePosixPath(relative)
        if (
            not separator
            or not _is_sha256(digest)
            or pure.is_absolute()
            or ".." in pure.parts
        ):
            raise DataPreparationError(f"malformed checksum entry in {checksums_path}")
        path = root.joinpath(*pure.parts)
        if relative in observed or not path.is_file() or hash_file(path) != digest:
            raise DataPreparationError(f"collection checksum mismatch: {relative}")
        observed[relative] = (path.stat().st_size, digest)
    expected = {"manifest.json"}
    for row in manifest.get("workloads", []):
        expected.update((str(row["file_name"]), str(row["adjacent_manifest"])))
    for row in manifest.get("extra_files", []):
        expected.add(str(row["file_name"]))
    if set(observed) != expected:
        raise DataPreparationError(f"collection checksum coverage mismatch: {root}")
    object_identities: set[tuple[str, str]] = set()
    for row in manifest.get("workloads", []):
        binary = str(row["file_name"])
        adjacent = str(row["adjacent_manifest"])
        if observed[binary] != (int(row["size"]), str(row["sha256"])):
            raise DataPreparationError(
                f"collection workload metadata mismatch: {binary}"
            )
        if observed[adjacent][1] != str(row["adjacent_manifest_sha256"]):
            raise DataPreparationError(
                f"collection adjacent manifest mismatch: {adjacent}"
            )
        workload_manifest = read_manifest(root / adjacent)
        metadata = workload_manifest.get("metadata")
        source_metadata = (
            metadata.get("source") if isinstance(metadata, Mapping) else None
        )
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("dataset_id") != dataset_id
            or metadata.get("real_dataset") != dataset_id
            or metadata.get("importer_id") != IMPORTER_ID
            or metadata.get("importer_sha256") != hash_file(pathlib.Path(__file__))
            or not _is_sha256(metadata.get("preprocessing_config_sha256"))
            or not isinstance(source_metadata, Mapping)
            or source_metadata.get("kind") != "huggingface_dataset"
            or not isinstance(source_metadata.get("assets"), list)
            or not source_metadata.get("assets")
            or source_metadata.get("repo_id") != expected_source["repo_id"]
            or source_metadata.get("revision") != expected_source["revision"]
            or source_metadata.get("source_lock_sha256") != expected_lock
        ):
            raise DataPreparationError(f"workload source provenance mismatch: {binary}")
        if metadata.get("source_identifier") != (
            f"{expected_source['repo_id']}@{expected_source['revision']}"
        ):
            raise DataPreparationError(f"workload source identifier mismatch: {binary}")
        if workload_manifest["workload"]["sha256"] != observed[binary][1]:
            raise DataPreparationError(f"adjacent workload SHA mismatch: {binary}")
        frozen = workload_manifest["workload"]
        workload = read_workload(root / binary, verify_payload=True, verify_file=False)
        expected_fields = {
            "dimension": workload.dimension,
            "n_R": workload.n_r,
            "n_S": workload.n_s,
            "N_total": workload.n_r + workload.n_s,
            "endpoint_type": workload.endpoint_type,
            "payload_sha256": workload.payload_sha256,
        }
        for field, actual in expected_fields.items():
            if frozen.get(field) != actual:
                raise DataPreparationError(
                    f"adjacent workload {field} mismatch for {binary}: "
                    f"{frozen.get(field)!r} != {actual!r}"
                )
        if (
            np.unique(workload.r_ids).size != workload.n_r
            or np.unique(workload.s_ids).size != workload.n_s
        ):
            raise DataPreparationError(
                f"duplicate object id in canonical workload: {binary}"
            )
        object_identities.add(
            (str(frozen.get("R_ids_sha256")), str(frozen.get("S_ids_sha256")))
        )
    if len(object_identities) != 1:
        raise DataPreparationError(
            f"collection workloads do not share object identities: {root}"
        )
    for row in manifest.get("extra_files", []):
        name = str(row["file_name"])
        if observed[name] != (int(row["size"]), str(row["sha256"])):
            raise DataPreparationError(f"collection sidecar metadata mismatch: {name}")
    return manifest


def _new_stage(target: pathlib.Path) -> pathlib.Path:
    if target.exists():
        raise DataPreparationError(
            f"target exists but is not an admitted collection: {target}; inspect it and use a new data root"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".{target.name}.building-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    return stage


def _as_numpy(table: pa.Table, name: str, dtype: Any | None = None) -> np.ndarray:
    array = table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
    return np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)


def cmab_canonical_identity(
    source_file: np.ndarray,
    source_fid: np.ndarray,
    published_uid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive collision-free CMAB ids from the published source record key."""

    files = np.asarray(source_file, dtype=object)
    fids = np.asarray(source_fid, dtype=np.int64)
    uids = np.asarray(published_uid, dtype=np.uint64)
    if not (
        files.ndim == fids.ndim == uids.ndim == 1
        and files.size == fids.size == uids.size
    ):
        raise ValueError("CMAB identity inputs must be equally sized vectors")
    if not files.size:
        raise ValueError("CMAB identity inputs must not be empty")
    if any(not isinstance(value, str) or not value for value in files):
        raise DataPreparationError(
            "CMAB source_file contains an empty or non-text value"
        )
    if np.any(fids < 0):
        raise DataPreparationError("CMAB source_fid contains a negative value")

    order = np.lexsort((fids, files))
    sorted_files = files[order]
    sorted_fids = fids[order]
    duplicate_pair = (sorted_files[1:] == sorted_files[:-1]) & (
        sorted_fids[1:] == sorted_fids[:-1]
    )
    if np.any(duplicate_pair):
        raise DataPreparationError("CMAB (source_file, source_fid) is not unique")

    digest = hashlib.sha256(b"anchor-cmab-source-identity-v1\0")
    for file_name, fid in zip(sorted_files, sorted_fids, strict=True):
        encoded = file_name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(int(fid).to_bytes(8, "little", signed=False))

    _, uid_counts = np.unique(uids, return_counts=True)
    duplicate_uid_counts = uid_counts[uid_counts > 1]
    diagnostics = {
        "object_identity_fields": ["source_file", "source_fid"],
        "object_id_method": "uint64_lexicographic_rank_v1",
        "source_identity_sha256": digest.hexdigest(),
        "published_building_uid_unique_count": int(uid_counts.size),
        "published_building_uid_duplicate_groups": int(duplicate_uid_counts.size),
        "published_building_uid_excess_rows": int(uids.size - uid_counts.size),
        "published_building_uid_max_multiplicity": int(uid_counts.max()),
    }
    canonical_ids = np.arange(files.size, dtype=np.uint64)
    return order.astype(np.int64, copy=False), canonical_ids, diagnostics


def cmab_split_indices(
    object_id: np.ndarray,
    func_code: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    target_r: int,
    tile_size_m: int = 10_000,
    stratum_hash_tag: str = "cmab-hf-stratum-v1",
    object_hash_tag: str = "cmab-hf-building-split-v1",
) -> tuple[np.ndarray, np.ndarray]:
    """Class/tile stratified half split over canonical published CMAB rows."""

    uid = np.asarray(object_id, dtype=np.uint64)
    functions = np.asarray(func_code)
    center_x = np.asarray(cx, dtype=np.float64)
    center_y = np.asarray(cy, dtype=np.float64)
    if not (uid.ndim == functions.ndim == center_x.ndim == center_y.ndim == 1):
        raise ValueError("CMAB split inputs must be one-dimensional")
    if not (uid.size == functions.size == center_x.size == center_y.size):
        raise ValueError("CMAB split inputs have different lengths")
    if np.unique(uid).size != uid.size:
        raise DataPreparationError("CMAB canonical object id is not unique")
    if target_r < 0 or target_r > uid.size:
        raise ValueError("invalid CMAB R target")
    strata: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index in range(uid.size):
        key = (
            int(functions[index]),
            math.floor(float(center_x[index]) / tile_size_m),
            math.floor(float(center_y[index]) / tile_size_m),
        )
        strata[key].append(index)
    quotas = {key: len(indices) // 2 for key, indices in strata.items()}
    missing = target_r - sum(quotas.values())
    odd = [key for key, indices in strata.items() if len(indices) % 2]
    if missing < 0 or missing > len(odd):
        raise DataPreparationError(
            "CMAB target is incompatible with half-per-stratum split"
        )
    odd.sort(
        key=lambda key: (stable_hash(stratum_hash_tag, key), canonical_json_bytes(key))
    )
    for key in odd[:missing]:
        quotas[key] += 1
    priorities = [stable_hash(object_hash_tag, int(value)) for value in uid]
    r_values: list[int] = []
    s_values: list[int] = []
    for key in sorted(
        strata, key=lambda item: (stable_hash(stratum_hash_tag, item), item)
    ):
        ranked = sorted(
            strata[key], key=lambda index: (priorities[index], int(uid[index]))
        )
        quota = quotas[key]
        r_values.extend(ranked[:quota])
        s_values.extend(ranked[quota:])
    r = np.asarray(sorted(r_values, key=lambda index: int(uid[index])), dtype=np.int64)
    s = np.asarray(sorted(s_values, key=lambda index: int(uid[index])), dtype=np.int64)
    if r.size != target_r or s.size != uid.size - target_r:
        raise AssertionError("CMAB split size invariant failed")
    return r, s


def _closed_interval_overlap_count(
    a_lower: np.ndarray, a_upper: np.ndarray, b_lower: np.ndarray, b_upper: np.ndarray
) -> int:
    if not a_lower.size or not b_lower.size:
        return 0
    lower = np.sort(b_lower)
    upper = np.sort(b_upper)
    before = np.searchsorted(upper, a_lower, side="left")
    after = b_lower.size - np.searchsorted(lower, a_upper, side="right")
    return int(a_lower.size * b_lower.size - before.sum() - after.sum())


def _touch_orientation_count(
    r_touch: np.ndarray,
    r_other_lower: np.ndarray,
    r_other_upper: np.ndarray,
    s_touch: np.ndarray,
    s_other_lower: np.ndarray,
    s_other_upper: np.ndarray,
) -> int:
    r_order = np.argsort(r_touch, kind="stable")
    s_order = np.argsort(s_touch, kind="stable")
    r_position = s_position = total = 0
    while r_position < r_order.size and s_position < s_order.size:
        r_value = r_touch[r_order[r_position]]
        s_value = s_touch[s_order[s_position]]
        if r_value < s_value:
            r_position += 1
            continue
        if s_value < r_value:
            s_position += 1
            continue
        r_end = r_position + 1
        s_end = s_position + 1
        while r_end < r_order.size and r_touch[r_order[r_end]] == r_value:
            r_end += 1
        while s_end < s_order.size and s_touch[s_order[s_end]] == s_value:
            s_end += 1
        ri = r_order[r_position:r_end]
        si = s_order[s_position:s_end]
        total += _closed_interval_overlap_count(
            r_other_lower[ri], r_other_upper[ri], s_other_lower[si], s_other_upper[si]
        )
        r_position, s_position = r_end, s_end
    return total


def _equal_corner_count(
    r_x: np.ndarray, r_y: np.ndarray, s_x: np.ndarray, s_y: np.ndarray
) -> int:
    dtype = np.dtype([("x", np.float64), ("y", np.float64)])
    r = np.empty(r_x.size, dtype=dtype)
    s = np.empty(s_x.size, dtype=dtype)
    r["x"], r["y"] = r_x, r_y
    s["x"], s["y"] = s_x, s_y
    ru, rc = np.unique(r, return_counts=True)
    su, sc = np.unique(s, return_counts=True)
    _, ri, si = np.intersect1d(ru, su, assume_unique=True, return_indices=True)
    return int(np.sum(rc[ri] * sc[si], dtype=np.int64))


def boundary_touching_diagnostic(
    r_lower: np.ndarray,
    r_upper: np.ndarray,
    s_lower: np.ndarray,
    s_upper: np.ndarray,
) -> dict[str, Any]:
    x_ul = _touch_orientation_count(
        r_upper[:, 0],
        r_lower[:, 1],
        r_upper[:, 1],
        s_lower[:, 0],
        s_lower[:, 1],
        s_upper[:, 1],
    )
    x_lu = _touch_orientation_count(
        r_lower[:, 0],
        r_lower[:, 1],
        r_upper[:, 1],
        s_upper[:, 0],
        s_lower[:, 1],
        s_upper[:, 1],
    )
    y_ul = _touch_orientation_count(
        r_upper[:, 1],
        r_lower[:, 0],
        r_upper[:, 0],
        s_lower[:, 1],
        s_lower[:, 0],
        s_upper[:, 0],
    )
    y_lu = _touch_orientation_count(
        r_lower[:, 1],
        r_lower[:, 0],
        r_upper[:, 0],
        s_upper[:, 1],
        s_lower[:, 0],
        s_upper[:, 0],
    )
    corners = 0
    for rx, sx in ((r_upper[:, 0], s_lower[:, 0]), (r_lower[:, 0], s_upper[:, 0])):
        for ry, sy in ((r_upper[:, 1], s_lower[:, 1]), (r_lower[:, 1], s_upper[:, 1])):
            corners += _equal_corner_count(rx, ry, sx, sy)
    return {
        "definition": "closed-contact pairs excluded by strict half-open comparison",
        "x_touch_Rupper_Slower_pairs": x_ul,
        "x_touch_Rlower_Supper_pairs": x_lu,
        "y_touch_Rupper_Slower_pairs": y_ul,
        "y_touch_Rlower_Supper_pairs": y_lu,
        "corner_pairs_counted_on_both_axes": corners,
        "excluded_boundary_touching_pairs": x_ul + x_lu + y_ul + y_lu - corners,
    }


def _read_cmab_table(path: pathlib.Path) -> dict[str, np.ndarray]:
    columns = [
        "building_uid",
        "shape_id",
        "func_code",
        "level",
        "province",
        "cx",
        "cy",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "exmin",
        "eymin",
        "exmax",
        "eymax",
        "source_file",
        "source_fid",
    ]
    table = pq.ParquetFile(path).read(columns=columns)
    values = {name: _as_numpy(table, name) for name in columns}
    return values


def prepare_cmab(
    *,
    data_root: pathlib.Path,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    lock_path: pathlib.Path,
    client: HubClient,
) -> pathlib.Path:
    target = data_root / "workloads" / "cmab_1m"
    try:
        verify_collection(target, dataset_id="CMAB-1M", source_lock_path=lock_path)
        return target
    except DataPreparationError:
        if target.exists():
            raise
    source = lock["sources"][str(config["source"])]
    downloaded = download_static_source(
        "cmab", data_root=data_root, lock=lock, client=client
    )
    lock_sha = hash_file(lock_path)
    stage = _new_stage(target)
    try:
        by_level = {
            int(asset.role.split("-")[1]): asset
            for asset in _source_assets(source)
            if asset.role and asset.role.startswith("level-")
        }
        configured_levels = {int(value) for value in config["levels"]}
        if set(by_level) != configured_levels:
            raise DataPreparationError(
                f"CMAB lock levels {sorted(by_level)} differ from config "
                f"{sorted(configured_levels)}"
            )
        first = _read_cmab_table(downloaded[by_level[1].path])
        expected = int(config["expected_buildings"])
        if first["building_uid"].size != expected:
            raise DataPreparationError(
                "CMAB Guangdong row count differs from the frozen target"
            )
        order, canonical_ids, identity_diagnostics = cmab_canonical_identity(
            first["source_file"],
            first["source_fid"],
            first["building_uid"],
        )
        first = {name: values[order] for name, values in first.items()}
        if set(map(str, np.unique(first["province"]))) != {str(config["province"])}:
            raise DataPreparationError(
                "CMAB source is not exclusively the Guangdong partition"
            )
        for key in (
            "published_building_uid_unique_count",
            "published_building_uid_duplicate_groups",
            "published_building_uid_excess_rows",
            "published_building_uid_max_multiplicity",
        ):
            if identity_diagnostics[key] != int(config[f"expected_{key}"]):
                raise DataPreparationError(
                    f"CMAB published identity diagnostic mismatch for {key}"
                )

        r_index, s_index = cmab_split_indices(
            canonical_ids,
            first["func_code"],
            first["cx"],
            first["cy"],
            target_r=int(config["target_R"]),
            tile_size_m=int(config["stratification"]["tile_size_m"]),
            stratum_hash_tag=str(config["stratification"]["stratum_hash_tag"]),
            object_hash_tag=str(config["stratification"]["object_hash_tag"]),
        )
        r_ids = canonical_ids[r_index]
        s_ids = canonical_ids[s_index]
        base_lower = np.column_stack((first["xmin"], first["ymin"])).astype(np.float64)
        base_upper = np.column_stack((first["xmax"], first["ymax"])).astype(np.float64)
        rows: list[dict[str, Any]] = []
        common = _common_metadata(
            "CMAB-1M", "cmab", source, lock_sha256=lock_sha, config=config
        )
        for level in map(int, config["levels"]):
            values = _read_cmab_table(downloaded[by_level[level].path])
            level_order, level_ids, level_identity = cmab_canonical_identity(
                values["source_file"],
                values["source_fid"],
                values["building_uid"],
            )
            values = {name: data[level_order] for name, data in values.items()}
            if (
                not np.array_equal(level_ids, canonical_ids)
                or level_identity != identity_diagnostics
            ):
                raise DataPreparationError(
                    f"CMAB canonical source identity differs at level {level}"
                )
            if values["building_uid"].size != expected:
                raise DataPreparationError(f"CMAB row count differs at level {level}")
            if set(map(int, np.unique(values["level"]))) != {level}:
                raise DataPreparationError(
                    f"CMAB level column is inconsistent at level {level}"
                )
            if set(map(str, np.unique(values["province"]))) != {
                str(config["province"])
            }:
                raise DataPreparationError(
                    f"CMAB province column is inconsistent at level {level}"
                )
            for name in (
                "source_file",
                "source_fid",
                "building_uid",
                "shape_id",
                "func_code",
                "cx",
                "cy",
            ):
                if not np.array_equal(values[name], first[name]):
                    raise DataPreparationError(
                        f"CMAB published {name} differs at level {level}"
                    )
            level_base_lower = np.column_stack((values["xmin"], values["ymin"])).astype(
                np.float64
            )
            level_base_upper = np.column_stack((values["xmax"], values["ymax"])).astype(
                np.float64
            )
            if not np.array_equal(level_base_lower, base_lower) or not np.array_equal(
                level_base_upper, base_upper
            ):
                raise DataPreparationError(f"CMAB base AABBs differ at level {level}")
            expanded_lower = np.column_stack((values["exmin"], values["eymin"])).astype(
                np.float64
            )
            expanded_upper = np.column_stack((values["exmax"], values["eymax"])).astype(
                np.float64
            )
            r_lower, r_upper = expanded_lower[r_index], expanded_upper[r_index]
            s_lower, s_upper = base_lower[s_index], base_upper[s_index]
            metadata = {
                **common,
                "level": level,
                **identity_diagnostics,
                "dimension": 2,
                "crs_id": config["crs_id"],
                "projection_unit": "meter",
                "cmab_crs": "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m",
                "split_method": "cmab_hf_stratified_hash_tile_10km_v1",
                "stratification_id": "func_code-x10km-y10km-in-published-albers",
                "boundary_touching_diagnostic": boundary_touching_diagnostic(
                    r_lower, r_upper, s_lower, s_upper
                ),
                "source_asset": dataclasses.asdict(by_level[level]),
            }
            file_name = f"cmab-1m-level-{level}.bin"
            write_workload(
                stage / file_name,
                r_ids=r_ids,
                r_lower=r_lower,
                r_upper=r_upper,
                s_ids=s_ids,
                s_lower=s_lower,
                s_upper=s_upper,
                endpoint_type="float64",
                metadata=metadata,
            )
            rows.append(
                {"file_name": file_name, "level": level, "dimension": 2, "N": expected}
            )
        write_collection_manifest(
            stage,
            dataset_id="CMAB-1M",
            source_lock_sha256=lock_sha,
            workloads=rows,
        )
        verify_collection(stage, dataset_id="CMAB-1M", source_lock_path=lock_path)
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target


def split_geolife_users(
    users: Sequence[int] | np.ndarray,
    *,
    hash_tag: str = "geolife-hf-user-split-v1",
) -> dict[int, str]:
    unique = sorted({int(value) for value in users})
    ranked = sorted(unique, key=lambda user: (stable_hash(hash_tag, user), user))
    return {user: ("R" if rank % 2 == 0 else "S") for rank, user in enumerate(ranked)}


def _splitmix64(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        data += np.uint64(0x9E3779B97F4A7C15)
        data = (data ^ (data >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        data = (data ^ (data >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        data ^= data >> np.uint64(31)
    return data


def geolife_point_priorities(
    traj_id: np.ndarray,
    point_idx: np.ndarray,
    *,
    tag: str = "geolife-hf-point-priority-v1",
) -> np.ndarray:
    trajectory = np.asarray(traj_id, dtype=np.int64).view(np.uint64)
    point = np.asarray(point_idx, dtype=np.int64).astype(np.uint64)
    rotated = (point << np.uint64(32)) | (point >> np.uint64(32))
    seed = np.uint64(int.from_bytes(stable_hash(tag)[:8], "little"))
    return _splitmix64(trajectory ^ rotated ^ seed)


def _largest_remainder_for_groups(
    sizes: np.ndarray,
    total: int,
    keys: Sequence[tuple[int, int, int]],
    side: str,
) -> np.ndarray:
    denominator = int(np.sum(sizes, dtype=np.int64))
    if denominator < total:
        raise DataPreparationError(
            f"GeoLife side {side} has only {denominator} candidates"
        )
    quotas = (sizes.astype(object) * int(total) // denominator).astype(np.int64)
    remainders = np.asarray(
        [(int(total) * int(size)) % denominator for size in sizes], dtype=np.int64
    )
    missing = int(total - np.sum(quotas, dtype=np.int64))
    ranked = sorted(
        range(len(keys)),
        key=lambda index: (
            -int(remainders[index]),
            stable_hash("geolife-hf-stratum-quota-v1", side, keys[index]),
            keys[index],
        ),
    )
    quotas[np.asarray(ranked[:missing], dtype=np.int64)] += 1
    return quotas


def select_geolife_indices(
    user_id: np.ndarray,
    traj_id: np.ndarray,
    point_idx: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    epoch_ms: np.ndarray,
    *,
    target_per_side: int,
    user_hash_tag: str = "geolife-hf-user-split-v1",
    point_priority_tag: str = "geolife-hf-point-priority-v1",
    tile_size_cm: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    users = np.asarray(user_id, dtype=np.int32)
    trajectories = np.asarray(traj_id, dtype=np.int64)
    points = np.asarray(point_idx, dtype=np.int32)
    x = np.asarray(x_cm, dtype=np.int64)
    y = np.asarray(y_cm, dtype=np.int64)
    timestamps = np.asarray(epoch_ms, dtype=np.int64)
    length = users.size
    if target_per_side <= 0:
        raise ValueError("GeoLife target_per_side must be positive")
    if length == 0:
        raise DataPreparationError("GeoLife candidate relation is empty")
    if any(array.size != length for array in (trajectories, points, x, y, timestamps)):
        raise ValueError("GeoLife selection arrays have different lengths")
    user_side = split_geolife_users(users, hash_tag=user_hash_tag)
    side_codes = np.fromiter(
        (0 if user_side[int(user)] == "R" else 1 for user in users),
        dtype=np.int8,
        count=length,
    )
    months = (
        timestamps.astype("datetime64[ms]").astype("datetime64[M]").astype(np.int64)
        % 12
        + 1
    ).astype(np.int16)
    tile_x = np.floor_divide(x, int(tile_size_cm))
    tile_y = np.floor_divide(y, int(tile_size_cm))
    priorities = geolife_point_priorities(trajectories, points, tag=point_priority_tag)
    selected: dict[str, np.ndarray] = {}
    diagnostic: dict[str, Any] = {
        "candidate_rows": int(length),
        "candidate_users": len(user_side),
        "priority_algorithm": "anchor-keyed-splitmix64-v1",
        "sides": {},
    }
    for code, side in enumerate(("R", "S")):
        pool = np.flatnonzero(side_codes == code)
        if pool.size < target_per_side:
            raise DataPreparationError(
                f"GeoLife side {side} has {pool.size} candidates; {target_per_side} required"
            )
        order = np.lexsort(
            (
                points[pool],
                trajectories[pool],
                priorities[pool],
                tile_y[pool],
                tile_x[pool],
                months[pool],
            )
        )
        ranked = pool[order]
        smonth = months[ranked]
        sx = tile_x[ranked]
        sy = tile_y[ranked]
        change = np.empty(ranked.size, dtype=bool)
        change[0] = True
        change[1:] = (
            (smonth[1:] != smonth[:-1]) | (sx[1:] != sx[:-1]) | (sy[1:] != sy[:-1])
        )
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], ranked.size)
        sizes = ends - starts
        keys = [
            (int(smonth[start]), int(sx[start]), int(sy[start])) for start in starts
        ]
        quotas = _largest_remainder_for_groups(sizes, int(target_per_side), keys, side)
        pieces = [
            ranked[start : start + int(quota)]
            for start, quota in zip(starts, quotas, strict=True)
            if quota
        ]
        chosen = np.concatenate(pieces)
        chosen = chosen[np.lexsort((points[chosen], trajectories[chosen]))]
        if chosen.size != target_per_side:
            raise AssertionError("GeoLife selected side has the wrong size")
        selected[side] = chosen
        diagnostic["sides"][side] = {
            "candidate_rows": int(pool.size),
            "candidate_users": int(np.unique(users[pool]).size),
            "selected_rows": int(chosen.size),
            "selected_users": int(np.unique(users[chosen]).size),
            "strata": len(keys),
        }
    if set(users[selected["R"]]) & set(users[selected["S"]]):
        raise AssertionError("GeoLife user split leaked across sides")
    return selected["R"], selected["S"], diagnostic


def geolife_boxes(
    centers_xyzt: np.ndarray,
    dimension: int,
    *,
    spatial_radius_cm: int,
    time_radius_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers_xyzt, dtype=np.int64)
    if centers.ndim != 2 or centers.shape[1] != 4:
        raise ValueError("GeoLife centers must have columns (x,y,z,t)")
    if dimension == 3:
        selected = centers[:, (0, 1, 3)]
        radii = np.asarray(
            (spatial_radius_cm, spatial_radius_cm, time_radius_ms), dtype=np.int64
        )
    elif dimension == 4:
        selected = centers[:, (0, 1, 2, 3)]
        radii = np.asarray(
            (spatial_radius_cm, spatial_radius_cm, spatial_radius_cm, time_radius_ms),
            dtype=np.int64,
        )
    else:
        raise ValueError("GeoLife dimension must be 3 or 4")
    limits = np.iinfo(np.int64)
    if np.any(selected < limits.min + radii) or np.any(
        selected > limits.max - radii - np.int64(1)
    ):
        raise DataPreparationError("GeoLife box construction would overflow int64")
    lower = selected - radii
    upper = selected + radii + np.int64(1)
    if not np.all(lower < upper):
        raise DataPreparationError("invalid GeoLife half-open conversion")
    return lower, upper


def _load_geolife_candidates(paths: Sequence[pathlib.Path]) -> dict[str, np.ndarray]:
    columns = [
        "traj_id",
        "user_id",
        "point_idx",
        "x_min_cm",
        "x_max_cm",
        "y_min_cm",
        "y_max_cm",
        "t_min_ms",
        "t_max_ms",
        "z_min_cm",
        "z_max_cm",
    ]
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in paths:
        table = pq.ParquetFile(path).read(columns=columns)
        for name in columns:
            chunks[name].append(_as_numpy(table, name))
    values = {name: np.concatenate(parts) for name, parts in chunks.items()}
    result = {
        "traj_id": values["traj_id"].astype(np.int64),
        "user_id": values["user_id"].astype(np.int32),
        "point_idx": values["point_idx"].astype(np.int32),
    }
    for axis, lower, upper in (
        ("x_cm", "x_min_cm", "x_max_cm"),
        ("y_cm", "y_min_cm", "y_max_cm"),
        ("z_cm", "z_min_cm", "z_max_cm"),
        ("epoch_ms", "t_min_ms", "t_max_ms"),
    ):
        lo = values[lower].astype(np.int64)
        hi = values[upper].astype(np.int64)
        delta = hi - lo
        if np.any(delta < 0) or np.any(delta % 2):
            raise DataPreparationError(
                f"GeoLife published {axis} bounds have no integer center"
            )
        result[axis] = lo + delta // 2
    return result


def _dense_shared_ids(
    traj_id: np.ndarray,
    point_idx: np.ndarray,
    r_index: np.ndarray,
    s_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.concatenate((r_index, s_index))
    trajectories = traj_id[selected]
    points = point_idx[selected]
    order = np.lexsort((points, trajectories))
    if np.any(
        (trajectories[order][1:] == trajectories[order][:-1])
        & (points[order][1:] == points[order][:-1])
    ):
        raise DataPreparationError(
            "duplicate GeoLife (traj_id, point_idx) in selected set"
        )
    identifiers = np.empty(selected.size, dtype=np.uint64)
    identifiers[order] = np.arange(selected.size, dtype=np.uint64)
    return identifiers[: r_index.size], identifiers[r_index.size :]


def _write_geolife_sidecar(
    path: pathlib.Path,
    *,
    object_ids: np.ndarray,
    values: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> None:
    table = pa.table(
        {
            "object_id": pa.array(object_ids, type=pa.uint64()),
            "traj_id": pa.array(values["traj_id"][indices], type=pa.int64()),
            "point_idx": pa.array(values["point_idx"][indices], type=pa.int32()),
            "user_id": pa.array(values["user_id"][indices], type=pa.int32()),
        }
    )
    pq.write_table(table, path, compression="zstd", row_group_size=100_000)


def prepare_geolife(
    *,
    data_root: pathlib.Path,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    lock_path: pathlib.Path,
    client: HubClient,
) -> tuple[pathlib.Path, pathlib.Path]:
    target3 = data_root / "workloads" / "geolife_3d_1m"
    target4 = data_root / "workloads" / "geolife_4d_1m"
    try:
        verify_collection(
            target3, dataset_id="GeoLife-3D-1M", source_lock_path=lock_path
        )
        verify_collection(
            target4, dataset_id="GeoLife-4D-1M", source_lock_path=lock_path
        )
        return target3, target4
    except DataPreparationError:
        if target3.exists() or target4.exists():
            raise
    source = lock["sources"][str(config["source"])]
    downloaded = download_static_source(
        "geolife", data_root=data_root, lock=lock, client=client
    )
    parts = [
        downloaded[asset.path]
        for asset in _source_assets(source)
        if asset.role == "shared-altitude-valid-points"
    ]
    values = _load_geolife_candidates(parts)
    expected = int(config["expected_candidates"])
    if values["traj_id"].size != expected:
        raise DataPreparationError(f"GeoLife candidate count is not {expected}")
    r_index, s_index, coverage = select_geolife_indices(
        values["user_id"],
        values["traj_id"],
        values["point_idx"],
        values["x_cm"],
        values["y_cm"],
        values["epoch_ms"],
        target_per_side=int(config["target_per_side"]),
        user_hash_tag=str(config["user_split"]["hash_tag"]),
        point_priority_tag=str(config["point_priority"]["tag"]),
        tile_size_cm=int(config["stratification"]["tile_size_cm"]),
    )
    r_ids, s_ids = _dense_shared_ids(
        values["traj_id"], values["point_idx"], r_index, s_index
    )
    centers = np.column_stack(
        (values["x_cm"], values["y_cm"], values["z_cm"], values["epoch_ms"])
    ).astype(np.int64)
    lock_sha = hash_file(lock_path)
    stage3, stage4 = _new_stage(target3), _new_stage(target4)
    try:
        sidecars = {
            "R": "geolife-selected-points-R.parquet",
            "S": "geolife-selected-points-S.parquet",
        }
        _write_geolife_sidecar(
            stage3 / sidecars["R"], object_ids=r_ids, values=values, indices=r_index
        )
        _write_geolife_sidecar(
            stage3 / sidecars["S"], object_ids=s_ids, values=values, indices=s_index
        )
        for name in sidecars.values():
            shutil.copyfile(stage3 / name, stage4 / name)
        sidecar_manifest = {
            side: {
                "file_name": name,
                "count": int(config["target_per_side"]),
                "sha256": hash_file(stage3 / name),
            }
            for side, name in sidecars.items()
        }
        common = _common_metadata(
            "GeoLife-3D-1M", "geolife", source, lock_sha256=lock_sha, config=config
        )
        rows: dict[int, list[dict[str, Any]]] = {3: [], 4: []}
        for dimension, stage, dataset_id in (
            (3, stage3, "GeoLife-3D-1M"),
            (4, stage4, "GeoLife-4D-1M"),
        ):
            dimension_common = {
                **common,
                "dataset_id": dataset_id,
                "real_dataset": dataset_id,
                "dimension": dimension,
                "crs_id": config["crs_id"],
                "horizontal_crs": "EPSG:3857",
                "projection_unit": "centimeter; epoch-millisecond",
                "split_method": "geolife_hf_user_hash_then_month_tile_largest_remainder_v1",
                "stratification_id": "utc-month-x1km-y1km",
                "point_priority_algorithm": config["point_priority"]["algorithm"],
                "spatial_temporal_coverage_summary": coverage,
                "selected_point_manifest_R": sidecar_manifest["R"],
                "selected_point_manifest_S": sidecar_manifest["S"],
            }
            for level_text, radii in config["levels"].items():
                level = int(level_text)
                r_lower, r_upper = geolife_boxes(
                    centers[r_index],
                    dimension,
                    spatial_radius_cm=int(radii["spatial_radius_cm"]),
                    time_radius_ms=int(radii["time_radius_ms"]),
                )
                s_lower, s_upper = geolife_boxes(
                    centers[s_index],
                    dimension,
                    spatial_radius_cm=int(radii["spatial_radius_cm"]),
                    time_radius_ms=int(radii["time_radius_ms"]),
                )
                file_name = f"geolife-{dimension}d-1m-level-{level}.bin"
                write_workload(
                    stage / file_name,
                    r_ids=r_ids,
                    r_lower=r_lower,
                    r_upper=r_upper,
                    s_ids=s_ids,
                    s_lower=s_lower,
                    s_upper=s_upper,
                    endpoint_type="int64",
                    metadata={
                        **dimension_common,
                        "level": level,
                        "level_radii": dict(radii),
                    },
                )
                rows[dimension].append(
                    {
                        "file_name": file_name,
                        "level": level,
                        "dimension": dimension,
                        "N": 1_000_000,
                    }
                )
        for stage, dataset_id, dimension in (
            (stage3, "GeoLife-3D-1M", 3),
            (stage4, "GeoLife-4D-1M", 4),
        ):
            write_collection_manifest(
                stage,
                dataset_id=dataset_id,
                source_lock_sha256=lock_sha,
                workloads=rows[dimension],
                extra_files=list(sidecars.values()),
            )
            verify_collection(stage, dataset_id=dataset_id, source_lock_path=lock_path)
        published3 = False
        try:
            os.replace(stage3, target3)
            published3 = True
            os.replace(stage4, target4)
        except Exception:
            if published3 and target3.exists():
                shutil.rmtree(target3)
            raise
    finally:
        for stage in (stage3, stage4):
            if stage.exists():
                shutil.rmtree(stage)
    return target3, target4


def select_coco_images(
    images: Sequence[Mapping[str, Any]],
    *,
    count: int,
    hash_tag: str = "coco-image-subset",
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for image in images:
        split_value = image["split"]
        if isinstance(split_value, (int, np.integer)):
            split = {0: "train2017", 1: "val2017"}.get(int(split_value))
        else:
            split = str(split_value)
        if split not in {"train2017", "val2017"}:
            raise DataPreparationError(f"unknown COCO split: {split_value}")
        normalized.append(
            {
                **dict(image),
                "split": split,
                "coco_image_id": int(image["coco_image_id"]),
                "z_idx": int(image["z_idx"]),
            }
        )
    ranked = sorted(
        normalized,
        key=lambda image: (
            stable_hash(hash_tag, image["split"], image["coco_image_id"]),
            image["split"],
            image["coco_image_id"],
        ),
    )
    if len(ranked) < count:
        raise DataPreparationError(
            f"only {len(ranked)} eligible COCO images; {count} required"
        )
    return ranked[:count]


def split_coco_proposal_indices(
    split: str,
    image_id: int,
    ranks: np.ndarray,
    rect_ids: np.ndarray,
    *,
    side_count: int,
    hash_tag: str = "coco-hf-proposal-split-v1",
) -> tuple[np.ndarray, np.ndarray]:
    rank_values = np.asarray(ranks, dtype=np.int64)
    identifiers = np.asarray(rect_ids, dtype=np.int64)
    if rank_values.size != identifiers.size or rank_values.size != 2 * side_count:
        raise DataPreparationError("COCO proposal side sizes are not balanced")
    if (
        np.unique(rank_values).size != rank_values.size
        or np.unique(identifiers).size != identifiers.size
    ):
        raise DataPreparationError(
            "COCO proposal identity is not unique within an image"
        )
    ordered = sorted(
        range(rank_values.size),
        key=lambda index: (
            stable_hash(
                hash_tag,
                split,
                int(image_id),
                int(rank_values[index]),
                int(identifiers[index]),
            ),
            int(rank_values[index]),
            int(identifiers[index]),
        ),
    )
    values = np.asarray(ordered, dtype=np.int64)
    return values[:side_count], values[side_count:]


def _coco_image_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    table = pq.ParquetFile(path).read()
    columns = {name: _as_numpy(table, name) for name in table.column_names}
    return [
        {
            name: columns[name][index].item()
            if hasattr(columns[name][index], "item")
            else columns[name][index]
            for name in columns
        }
        for index in range(table.num_rows)
    ]


def _find_coco_shard(
    shards: Sequence[Mapping[str, Any]], z_index: int
) -> Mapping[str, Any]:
    matches = [
        shard
        for shard in shards
        if int(shard["start_z_idx"]) <= z_index < int(shard["end_z_idx"])
    ]
    if len(matches) != 1:
        raise DataPreparationError(
            f"COCO z_idx {z_index} maps to {len(matches)} shards"
        )
    return matches[0]


def _row_groups_for_z(parquet: pq.ParquetFile, z_values: set[int]) -> list[int]:
    column_index = parquet.schema_arrow.names.index("z_min")
    result: list[int] = []
    for index in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(index).column(column_index).statistics
        if statistics is None:
            result.append(index)
            continue
        lower, upper = int(statistics.min), int(statistics.max)
        if any(lower <= value <= upper for value in z_values):
            result.append(index)
    return result


def _proposal_digest(table: pa.Table) -> str:
    digest = hashlib.sha256()
    for name in ("rect_id", "rank", "min_x", "min_y", "max_x", "max_y"):
        array = np.ascontiguousarray(_as_numpy(table, name))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def prepare_coco(
    *,
    data_root: pathlib.Path,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    lock_path: pathlib.Path,
    client: HubClient,
) -> pathlib.Path:
    target = data_root / "workloads" / "coco_1m"
    try:
        verify_collection(target, dataset_id="COCO-1M", source_lock_path=lock_path)
        return target
    except DataPreparationError:
        if target.exists():
            raise
    source = lock["sources"][str(config["source"])]
    downloaded = download_static_source(
        "coco", data_root=data_root, lock=lock, client=client
    )
    images_asset = _asset_by_role(source, "published-image-index")
    manifest_asset = _asset_by_role(source, "published-build-manifest")
    image_rows = _coco_image_rows(downloaded[images_asset.path])
    if len(image_rows) != int(config["expected_eligible_images"]):
        raise DataPreparationError(
            "COCO published image count differs from the frozen value"
        )
    selected = select_coco_images(
        image_rows,
        count=int(config["selected_images"]),
        hash_tag=str(config["image_hash_tag"]),
    )
    upstream = _load_json(downloaded[manifest_asset.path])
    shards = upstream.get("shards")
    if not isinstance(shards, list):
        raise DataPreparationError("COCO build manifest has no shard list")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for workload_z, image in enumerate(selected):
        image["workload_z_idx"] = workload_z
        shard = _find_coco_shard(shards, int(image["z_idx"]))
        image["shard_file"] = str(shard["shard_file"])
        grouped[image["shard_file"]].append(image)
    columns = list(source["rectangle_access"]["projection"])
    proposals: dict[int, pa.Table] = {}
    shard_provenance: list[dict[str, Any]] = []
    for path in sorted(grouped):
        wanted = {int(image["z_idx"]) for image in grouped[path]}
        raw = HubRangeReader(
            client, str(source["repo_id"]), str(source["revision"]), path
        )
        if raw.size <= 0 or not _is_sha256(raw.remote.linked_sha256):
            raise DataPreparationError(
                f"COCO shard lacks pinned Hub LFS identity: {path}"
            )
        buffered = io.BufferedReader(raw, buffer_size=1 << 20)
        try:
            parquet = pq.ParquetFile(buffered)
            row_groups = _row_groups_for_z(parquet, wanted)
            table = parquet.read_row_groups(row_groups, columns=columns)
        finally:
            buffered.close()
        z = _as_numpy(table, "z_min", np.int32)
        record_type = _as_numpy(table, "type", np.int8)
        mask = (record_type == 1) & np.isin(
            z, np.asarray(sorted(wanted), dtype=np.int32)
        )
        table = table.take(pa.array(np.flatnonzero(mask), type=pa.int64()))
        z = _as_numpy(table, "z_min", np.int32)
        for value in wanted:
            proposals[value] = table.take(
                pa.array(np.flatnonzero(z == value), type=pa.int64())
            )
        shard_provenance.append(
            {
                "path": path,
                "size": raw.size,
                "linked_sha256": raw.remote.linked_sha256,
                "row_groups": row_groups,
                "requests": raw.request_count,
                "bytes_transferred": raw.bytes_transferred,
            }
        )
    side_ids: dict[str, list[np.ndarray]] = {"R": [], "S": []}
    side_lower: dict[str, list[np.ndarray]] = {"R": [], "S": []}
    side_upper: dict[str, list[np.ndarray]] = {"R": [], "S": []}
    image_manifest: list[dict[str, Any]] = []
    rank_summary: dict[str, list[float]] = {"R": [], "S": []}
    score_summary: dict[str, list[float]] = {"R": [], "S": []}
    expected_proposals = int(config["proposals_per_image"])
    side_count = int(config["proposals_per_side_per_image"])
    for workload_z, image in enumerate(selected):
        source_z = int(image["z_idx"])
        table = proposals.get(source_z)
        if table is None or table.num_rows != expected_proposals:
            raise DataPreparationError(
                f"COCO image {image['coco_image_id']} has {0 if table is None else table.num_rows} proposals"
            )
        image_ids = _as_numpy(table, "coco_image_id", np.int32)
        ranks = _as_numpy(table, "rank", np.int16)
        rect_ids = _as_numpy(table, "rect_id", np.int64)
        z_lower = _as_numpy(table, "z_min", np.int32)
        z_upper = _as_numpy(table, "z_max", np.int32)
        if set(map(int, np.unique(image_ids))) != {int(image["coco_image_id"])}:
            raise DataPreparationError("COCO proposal image identity mismatch")
        if set(map(int, np.unique(z_lower))) != {source_z} or set(
            map(int, np.unique(z_upper))
        ) != {source_z + 1}:
            raise DataPreparationError(
                "COCO proposal z slice disagrees with the image index"
            )
        if not np.array_equal(np.sort(ranks), np.arange(1, expected_proposals + 1)):
            raise DataPreparationError("COCO published ranks are not exactly 1..10000")
        for coordinate in ("min_x", "min_y", "max_x", "max_y"):
            if table.schema.field(coordinate).type != pa.float32():
                raise DataPreparationError(
                    "COCO published proposal coordinates are not float32"
                )
        r_index, s_index = split_coco_proposal_indices(
            str(image["split"]),
            int(image["coco_image_id"]),
            ranks,
            rect_ids,
            side_count=side_count,
            hash_tag=str(config["proposal_hash_tag"]),
        )
        xy_lower = np.column_stack(
            (_as_numpy(table, "min_x"), _as_numpy(table, "min_y"))
        ).astype(np.float64)
        xy_upper = np.column_stack(
            (_as_numpy(table, "max_x"), _as_numpy(table, "max_y"))
        ).astype(np.float64)
        scores = _as_numpy(table, "score", np.float32)
        if (
            not np.isfinite(xy_lower).all()
            or not np.isfinite(xy_upper).all()
            or not np.all(xy_lower < xy_upper)
        ):
            raise DataPreparationError(
                "COCO published proposal contains an invalid XY box"
            )
        for side, indices in (("R", r_index), ("S", s_index)):
            side_ids[side].append(rect_ids[indices].astype(np.uint64))
            side_lower[side].append(
                np.column_stack(
                    (
                        xy_lower[indices],
                        np.full(indices.size, workload_z, dtype=np.float64),
                    )
                )
            )
            side_upper[side].append(
                np.column_stack(
                    (
                        xy_upper[indices],
                        np.full(indices.size, workload_z + 1, dtype=np.float64),
                    )
                )
            )
            rank_summary[side].extend(
                np.quantile(ranks[indices], (0.0, 0.5, 0.9, 1.0)).tolist()
            )
            score_summary[side].extend(
                np.quantile(scores[indices], (0.0, 0.5, 0.9, 1.0)).tolist()
            )
        image_manifest.append(
            {
                "split": image["split"],
                "image_id": int(image["coco_image_id"]),
                "source_z_idx": source_z,
                "workload_z_idx": workload_z,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "source_shard": image["shard_file"],
                "proposal_count": expected_proposals,
                "proposal_rows_sha256": _proposal_digest(table),
            }
        )
    arrays = {
        key: np.concatenate(values)
        for key, values in {
            "r_ids": side_ids["R"],
            "s_ids": side_ids["S"],
            "r_lower": side_lower["R"],
            "r_upper": side_upper["R"],
            "s_lower": side_lower["S"],
            "s_upper": side_upper["S"],
        }.items()
    }
    if np.unique(np.concatenate((arrays["r_ids"], arrays["s_ids"]))).size != 1_000_000:
        raise DataPreparationError(
            "COCO selected rect_id values are not globally unique"
        )
    lock_sha = hash_file(lock_path)
    stage = _new_stage(target)
    try:
        selected_sidecar = "coco-selected-images.json"
        _atomic_json(
            stage / selected_sidecar,
            {
                "schema_version": "anchor-coco-selected-images-v2",
                "source_repository": source["repo_id"],
                "source_revision": source["revision"],
                "image_subset_id": config["image_subset_id"],
                "images": image_manifest,
            },
        )
        metadata = {
            **_common_metadata(
                "COCO-1M", "coco", source, lock_sha256=lock_sha, config=config
            ),
            "crs_id": config["crs_id"],
            "dimension": 3,
            "image_subset_id": config["image_subset_id"],
            "image_subset_sha256": hashlib.sha256(
                canonical_json_bytes(
                    [(row["split"], row["image_id"]) for row in image_manifest]
                )
            ).hexdigest(),
            "eligible_image_count": len(image_rows),
            "selected_images": image_manifest,
            "selected_images_sidecar": {
                "file_name": selected_sidecar,
                "sha256": hash_file(stage / selected_sidecar),
            },
            "proposal_stage": config["proposal_stage"],
            "proposal_split_method": config["proposal_split_method"],
            "proposal_identity_fields": config["proposal_identity"],
            "proposal_pipeline_id": stable_hash_hex(
                "coco-hf-published-proposal-pipeline-v1",
                source["revision"],
                hash_file(downloaded[manifest_asset.path]),
            ),
            "coordinate_source_type": config["coordinate_source_type"],
            "coordinate_conversion": "exact float32-to-float64 promotion",
            "upstream_builder_manifest_sha256": hash_file(
                downloaded[manifest_asset.path]
            ),
            "model_config_id": upstream.get("model_config_id"),
            "checkpoint_sha256": upstream.get("checkpoint_sha256"),
            "rank_or_score_summary": {
                side: {
                    "rank_quantile_samples_per_image": rank_summary[side],
                    "score_quantile_samples_per_image": score_summary[side],
                }
                for side in ("R", "S")
            },
            "source_shards": shard_provenance,
        }
        file_name = "coco-1m.bin"
        write_workload(
            stage / file_name,
            r_ids=arrays["r_ids"],
            r_lower=arrays["r_lower"],
            r_upper=arrays["r_upper"],
            s_ids=arrays["s_ids"],
            s_lower=arrays["s_lower"],
            s_upper=arrays["s_upper"],
            endpoint_type="float64",
            metadata=metadata,
        )
        write_collection_manifest(
            stage,
            dataset_id="COCO-1M",
            source_lock_sha256=lock_sha,
            workloads=[{"file_name": file_name, "dimension": 3, "N": 1_000_000}],
            extra_files=[selected_sidecar],
        )
        verify_collection(stage, dataset_id="COCO-1M", source_lock_path=lock_path)
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target


COLLECTIONS: dict[str, tuple[str, str]] = {
    "cmab": ("cmab_1m", "CMAB-1M"),
    "geolife-3d": ("geolife_3d_1m", "GeoLife-3D-1M"),
    "geolife-4d": ("geolife_4d_1m", "GeoLife-4D-1M"),
    "coco": ("coco_1m", "COCO-1M"),
}


def verify_real_collections(
    data_root: pathlib.Path,
    datasets: Sequence[str],
    *,
    lock_path: pathlib.Path,
) -> dict[str, Any]:
    requested: list[tuple[str, str]] = []
    for dataset in datasets:
        keys = ("geolife-3d", "geolife-4d") if dataset == "geolife" else (dataset,)
        for key in keys:
            if key not in COLLECTIONS:
                raise ValueError(f"unknown real dataset: {dataset}")
            requested.append(COLLECTIONS[key])
    verified = []
    admitted: dict[str, tuple[pathlib.Path, dict[str, Any]]] = {}
    for relative, dataset_id in requested:
        root = data_root / "workloads" / relative
        manifest = verify_collection(
            root,
            dataset_id=dataset_id,
            source_lock_path=lock_path,
        )
        admitted[dataset_id] = (root, manifest)
        verified.append(dataset_id)
    geo_ids = ("GeoLife-3D-1M", "GeoLife-4D-1M")
    if all(dataset_id in admitted for dataset_id in geo_ids):
        identities: list[tuple[Any, ...]] = []
        for dataset_id in geo_ids:
            root, manifest = admitted[dataset_id]
            first = manifest["workloads"][0]
            adjacent = read_manifest(root / str(first["adjacent_manifest"]))
            workload = adjacent["workload"]
            metadata = adjacent["metadata"]
            identities.append(
                (
                    workload["R_ids_sha256"],
                    workload["S_ids_sha256"],
                    metadata["selected_point_manifest_R"]["sha256"],
                    metadata["selected_point_manifest_S"]["sha256"],
                    metadata["preprocessing_config_sha256"],
                )
            )
        if identities[0] != identities[1]:
            raise DataPreparationError(
                "GeoLife 3D/4D collections do not share point identities"
            )
    return {"verified": verified}


def check_prerequisites(
    data_root: pathlib.Path,
    datasets: Sequence[str],
    *,
    config_path: pathlib.Path = DEFAULT_CONFIG,
    lock_path: pathlib.Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    config, lock, _, _ = resolve_configuration_paths(config_path, lock_path)
    unknown = set(datasets) - {"cmab", "geolife", "coco"}
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")
    disk_probe = data_root.resolve()
    while not disk_probe.exists():
        disk_probe = disk_probe.parent
    free = shutil.disk_usage(disk_probe).free
    locked_static = sum(
        int(asset["size"])
        for name in datasets
        for asset in lock["sources"][name]["assets"]
    )
    minimum = locked_static + 3_000_000_000
    if free < minimum:
        raise DataPreparationError(
            f"insufficient free space below {data_root}: {free} bytes; need at least {minimum}"
        )
    return {
        "config_schema": config["schema_version"],
        "lock_schema": lock["schema_version"],
        "datasets": list(datasets),
        "static_download_bytes": locked_static,
        "free_bytes": free,
        "dependencies": {
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "requests": requests.__version__,
        },
    }


def prepare_real_datasets(
    data_root: str | pathlib.Path,
    datasets: Sequence[str],
    *,
    config_path: str | pathlib.Path = DEFAULT_CONFIG,
    lock_path: str | pathlib.Path | None = None,
    endpoint: str | None = None,
    download_only: bool = False,
) -> dict[str, Any]:
    root = pathlib.Path(data_root).resolve()
    config, lock, config_file, lock_file = resolve_configuration_paths(
        config_path, lock_path
    )
    ordered = [name for name in ("cmab", "geolife", "coco") if name in set(datasets)]
    if not ordered or set(datasets) - set(ordered):
        raise ValueError("select one or more of cmab, geolife, coco")
    check_prerequisites(root, ordered, config_path=config_file, lock_path=lock_file)
    client = HubClient(endpoint)
    outputs: dict[str, Any] = {}
    if download_only:
        for name in ordered:
            paths = download_static_source(
                name, data_root=root, lock=lock, client=client
            )
            outputs[name] = [str(path) for path in paths.values()]
        return {"mode": "download-only", "outputs": outputs}
    if "cmab" in ordered:
        outputs["cmab"] = str(
            prepare_cmab(
                data_root=root,
                config=config["cmab"],
                lock=lock,
                lock_path=lock_file,
                client=client,
            )
        )
    if "geolife" in ordered:
        geo3, geo4 = prepare_geolife(
            data_root=root,
            config=config["geolife"],
            lock=lock,
            lock_path=lock_file,
            client=client,
        )
        outputs["geolife"] = [str(geo3), str(geo4)]
    if "coco" in ordered:
        outputs["coco"] = str(
            prepare_coco(
                data_root=root,
                config=config["coco"],
                lock=lock,
                lock_path=lock_file,
                client=client,
            )
        )
    verify_real_collections(root, ordered, lock_path=lock_file)
    return {"mode": "prepare", "outputs": outputs}
