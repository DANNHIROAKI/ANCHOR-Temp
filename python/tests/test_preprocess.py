from __future__ import annotations

import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests

from anchor_exp.alacarte import AlacarteConfig
from anchor_exp.cli.prepare_synthetic_suite import _epsilon
from anchor_exp.experiments import (
    REAL_DATASET_RULES,
    _validate_real_cross_workload_identities,
    _validate_real_manifest,
    expand_experiments,
    load_config,
)
from anchor_exp.hf_real import (
    DataPreparationError,
    HubRangeReader,
    ResolvedRemote,
    _load_geolife_candidates,
    boundary_touching_diagnostic,
    cmab_canonical_identity,
    cmab_split_indices,
    geolife_boxes,
    geolife_point_priorities,
    load_real_config,
    load_source_lock,
    select_coco_images,
    select_geolife_indices,
    split_coco_proposal_indices,
    split_geolife_users,
)
from anchor_exp.preprocess.common import (
    FrozenMetadataError,
    OptionalDependencyError,
    largest_remainder_counts,
    require_frozen_text,
    require_module,
    validate_sha256_digest,
)


def test_real_source_lock_contains_only_finished_hf_datasets() -> None:
    repository = pathlib.Path(__file__).resolve().parents[2]
    lock = load_source_lock(repository / "data_sources.lock.json")
    assert {name: source["repo_id"] for name, source in lock["sources"].items()} == {
        "cmab": "DannHiroaki/CMAB-Spatial-Join-0.08B",
        "geolife": "DannHiroaki/Geolife-Spatial-Join-0.15B",
        "coco": "DannHiroaki/COCO-Spatial-Join-1.23B",
    }
    assert all(len(source["revision"]) == 40 for source in lock["sources"].values())
    paths = [
        asset["path"]
        for source in lock["sources"].values()
        for asset in source["assets"]
    ]
    assert all(path.endswith((".parquet", ".json")) for path in paths)
    assert not any(path.endswith((".zip", ".jpg", ".plt", ".shp")) for path in paths)
    config = load_real_config(repository / "configs" / "real_data.json")
    assert config["coco"]["proposal_stage"] == "hf_published_rpn_top10000"
    dataset_source = {
        "CMAB-1M": "cmab",
        "GeoLife-3D-1M": "geolife",
        "GeoLife-4D-1M": "geolife",
        "COCO-1M": "coco",
    }
    for dataset_id, rule in REAL_DATASET_RULES.items():
        source_name = dataset_source[dataset_id]
        assert rule["repo_id"] == lock["sources"][source_name]["repo_id"]
        assert rule["revision"] == lock["sources"][source_name]["revision"]
        assert rule["crs_id"] == config[source_name]["crs_id"]


def test_range_reader_requires_exact_content_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        timeout_seconds = 1

        def __init__(self, content_range: str) -> None:
            self.content_range = content_range
            self.session = self

        def resolve(
            self,
            repo_id: str,
            revision: str,
            path: str,
            *,
            expected_size: int | None = None,
            expected_sha256: str | None = None,
        ) -> ResolvedRemote:
            assert expected_size == 10
            return ResolvedRemote(
                "https://example.invalid",
                "https://example.invalid/request",
                "https://example.invalid/content",
                10,
                expected_sha256,
            )

        def get(self, *args: object, **kwargs: object) -> requests.Response:
            response = requests.Response()
            response.status_code = 206
            response._content = b"abcd"
            response.headers["Content-Range"] = self.content_range
            return response

    valid = HubRangeReader(
        FakeClient("bytes 0-3/10"),
        "owner/repo",
        "a" * 40,
        "asset.parquet",
        expected_size=10,
        expected_sha256="b" * 64,
    )
    buffer = bytearray(4)
    assert valid.readinto(buffer) == 4
    assert bytes(buffer) == b"abcd"

    monkeypatch.setattr("anchor_exp.hf_real.time.sleep", lambda _: None)
    invalid = HubRangeReader(
        FakeClient("bytes 1-4/10"),
        "owner/repo",
        "a" * 40,
        "asset.parquet",
        expected_size=10,
        expected_sha256="b" * 64,
    )
    with pytest.raises(DataPreparationError, match="Content-Range"):
        invalid.readinto(bytearray(4))


def test_suite_epsilon_policy_has_certifiable_low_density_floor() -> None:
    repository = pathlib.Path(__file__).resolve().parents[2]
    suite = load_config(repository / "configs" / "experiments.json")
    policy = suite["synthetic_generation"]["epsilon_alpha_policy"]
    assert _epsilon(AlacarteConfig(alpha_target=0.1), policy) == 0.05
    assert _epsilon(AlacarteConfig(alpha_target=0.5), policy) == 0.1
    assert _epsilon(AlacarteConfig(alpha_target=10.0), policy) == 2.0
    assert _epsilon(AlacarteConfig(alpha_target=500.0), policy) == 5.0
    assert _epsilon(AlacarteConfig(alpha_target=1000.0), policy) == 5.0


def test_largest_remainder_is_exact_and_deterministic() -> None:
    sizes = {"a": 3, "b": 2}
    assert largest_remainder_counts(sizes, 4, tie_tag="test") == {"a": 2, "b": 2}
    assert sum(largest_remainder_counts(sizes, 3, tie_tag="test").values()) == 3


def test_missing_optional_dependency_has_no_fallback() -> None:
    with pytest.raises(OptionalDependencyError, match="No approximate fallback"):
        require_module(
            "anchor_dependency_that_does_not_exist_42",
            extra="test-extra",
            purpose="test operation",
        )


def test_frozen_digest_is_required_and_exact() -> None:
    digest = "ab" * 32
    assert validate_sha256_digest(digest.upper(), digest, label="asset") == digest
    with pytest.raises(FrozenMetadataError, match="not 64 hexadecimal"):
        validate_sha256_digest(digest, "REPLACE", label="asset")
    with pytest.raises(FrozenMetadataError, match="mismatch"):
        validate_sha256_digest(digest, "cd" * 32, label="asset")
    with pytest.raises(FrozenMetadataError, match="non-placeholder"):
        require_frozen_text("REPLACE_WITH_DOWNLOAD_ID", label="source")


def test_cmab_canonical_identity_uses_source_pair_not_colliding_uid() -> None:
    files = np.array(["b/source.shp", "a/source.shp", "a/source.shp"], dtype=object)
    fids = np.array([1, 2, 1], dtype=np.int64)
    published_uid = np.array([7, 7, 8], dtype=np.uint64)
    order, ids, diagnostics = cmab_canonical_identity(files, fids, published_uid)
    assert order.tolist() == [2, 1, 0]
    assert ids.tolist() == [0, 1, 2]
    assert diagnostics["published_building_uid_unique_count"] == 2
    assert diagnostics["published_building_uid_duplicate_groups"] == 1
    assert diagnostics["published_building_uid_excess_rows"] == 1

    permutation = np.array([2, 0, 1])
    other_order, other_ids, other_diagnostics = cmab_canonical_identity(
        files[permutation],
        fids[permutation],
        published_uid[permutation],
    )
    assert files[permutation][other_order].tolist() == files[order].tolist()
    assert fids[permutation][other_order].tolist() == fids[order].tolist()
    assert other_ids.tolist() == ids.tolist()
    assert (
        other_diagnostics["source_identity_sha256"]
        == diagnostics["source_identity_sha256"]
    )

    with pytest.raises(DataPreparationError, match="source_file, source_fid"):
        cmab_canonical_identity(
            np.array(["same", "same"], dtype=object),
            np.array([1, 1], dtype=np.int64),
            np.array([1, 2], dtype=np.uint64),
        )


def test_cmab_hf_split_balances_strata_and_is_order_independent() -> None:
    uid = np.arange(9, dtype=np.uint64)
    function = np.asarray([0] * 5 + [1] * 4, dtype=np.uint8)
    cx = np.asarray([0, 1, 2, 3, 4, 20_000, 20_001, 20_002, 20_003], dtype=np.float64)
    cy = np.zeros(uid.size, dtype=np.float64)
    r_a, s_a = cmab_split_indices(uid, function, cx, cy, target_r=5)
    permutation = np.arange(uid.size - 1, -1, -1)
    r_b, s_b = cmab_split_indices(
        uid[permutation],
        function[permutation],
        cx[permutation],
        cy[permutation],
        target_r=5,
    )
    assert set(map(int, uid[r_a])) == set(map(int, uid[permutation][r_b]))
    assert set(map(int, uid[s_a])) == set(map(int, uid[permutation][s_b]))
    assert len(r_a) == 5 and len(s_a) == 4
    for code, tile in ((0, 0), (1, 2)):
        member = (function == code) & (np.floor(cx / 10_000).astype(int) == tile)
        assert (
            abs(
                np.isin(np.flatnonzero(member), r_a).sum()
                - np.isin(np.flatnonzero(member), s_a).sum()
            )
            <= 1
        )


def test_cmab_boundary_diagnostic_counts_edges_and_deduplicates_corners() -> None:
    r_lower = np.asarray([[0.0, 0.0]])
    r_upper = np.asarray([[1.0, 1.0]])
    s_lower = np.asarray([[1.0, 0.5], [1.0, 1.0], [2.0, 2.0]])
    s_upper = np.asarray([[2.0, 1.5], [2.0, 2.0], [3.0, 3.0]])
    diagnostic = boundary_touching_diagnostic(r_lower, r_upper, s_lower, s_upper)
    assert diagnostic["x_touch_Rupper_Slower_pairs"] == 2
    assert diagnostic["y_touch_Rupper_Slower_pairs"] == 1
    assert diagnostic["corner_pairs_counted_on_both_axes"] == 1
    assert diagnostic["excluded_boundary_touching_pairs"] == 2


def _geolife_fixture() -> dict[str, np.ndarray]:
    users = np.repeat(np.arange(8, dtype=np.int32), 20)
    length = users.size
    return {
        "user": users,
        "traj": users.astype(np.int64) * 100,
        "point": np.tile(np.arange(20, dtype=np.int32), 8),
        "x": np.arange(length, dtype=np.int64) * 1000,
        "y": np.arange(length, dtype=np.int64) * 1000 + 50,
        "t": np.full(length, 1_609_459_200_000, dtype=np.int64)
        + np.arange(length) * 1000,
    }


def test_geolife_hf_user_and_point_selection_are_order_independent() -> None:
    values = _geolife_fixture()
    split = split_geolife_users(values["user"])
    assert sorted(split.values()).count("R") == sorted(split.values()).count("S") == 4
    r_a, s_a, diagnostic = select_geolife_indices(
        values["user"],
        values["traj"],
        values["point"],
        values["x"],
        values["y"],
        values["t"],
        target_per_side=20,
    )
    permutation = np.arange(values["user"].size - 1, -1, -1)
    r_b, s_b, _ = select_geolife_indices(
        *(
            values[name][permutation]
            for name in ("user", "traj", "point", "x", "y", "t")
        ),
        target_per_side=20,
    )
    keys_a_r = set(zip(values["traj"][r_a], values["point"][r_a], strict=True))
    keys_a_s = set(zip(values["traj"][s_a], values["point"][s_a], strict=True))
    keys_b_r = set(
        zip(
            values["traj"][permutation][r_b],
            values["point"][permutation][r_b],
            strict=True,
        )
    )
    keys_b_s = set(
        zip(
            values["traj"][permutation][s_b],
            values["point"][permutation][s_b],
            strict=True,
        )
    )
    assert keys_a_r == keys_b_r and keys_a_s == keys_b_s
    assert set(values["user"][r_a]).isdisjoint(set(values["user"][s_a]))
    assert diagnostic["priority_algorithm"] == "anchor-keyed-splitmix64-v1"
    np.testing.assert_array_equal(
        geolife_point_priorities(values["traj"], values["point"]),
        geolife_point_priorities(values["traj"], values["point"]),
    )


def test_geolife_published_physical_column_order_is_reordered_by_name(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "part.parquet"
    table = pa.table(
        {
            "traj_id": pa.array([7], type=pa.int64()),
            "user_id": pa.array([3], type=pa.int32()),
            "point_idx": pa.array([9], type=pa.int32()),
            "x_min_cm": pa.array([90], type=pa.int64()),
            "x_max_cm": pa.array([110], type=pa.int64()),
            "y_min_cm": pa.array([180], type=pa.int64()),
            "y_max_cm": pa.array([220], type=pa.int64()),
            "t_min_ms": pa.array([970], type=pa.int64()),
            "t_max_ms": pa.array([1030], type=pa.int64()),
            "z_min_cm": pa.array([270], type=pa.int64()),
            "z_max_cm": pa.array([330], type=pa.int64()),
        }
    )
    pq.write_table(table, path)
    values = _load_geolife_candidates([path])
    assert (
        values["x_cm"][0],
        values["y_cm"][0],
        values["z_cm"][0],
        values["epoch_ms"][0],
    ) == (
        100,
        200,
        300,
        1000,
    )


def test_geolife_closed_boxes_convert_to_half_open_exactly() -> None:
    centers = np.asarray([[10_000, 20_000, 30_000, 1_000_000]], dtype=np.int64)
    lower3, upper3 = geolife_boxes(
        centers, 3, spatial_radius_cm=1000, time_radius_ms=30_000
    )
    np.testing.assert_array_equal(lower3, [[9_000, 19_000, 970_000]])
    np.testing.assert_array_equal(upper3, [[11_001, 21_001, 1_030_001]])
    lower4, upper4 = geolife_boxes(
        centers, 4, spatial_radius_cm=1000, time_radius_ms=30_000
    )
    np.testing.assert_array_equal(lower4, [[9_000, 19_000, 29_000, 970_000]])
    np.testing.assert_array_equal(upper4, [[11_001, 21_001, 31_001, 1_030_001]])


def test_coco_hf_image_and_proposal_splits_are_order_independent() -> None:
    images = [
        {"split": 0, "coco_image_id": index, "z_idx": index} for index in range(20)
    ]
    a = select_coco_images(images, count=5)
    b = select_coco_images(list(reversed(images)), count=5)
    assert [(row["split"], row["coco_image_id"]) for row in a] == [
        (row["split"], row["coco_image_id"]) for row in b
    ]
    ranks = np.arange(1, 11, dtype=np.int16)
    rect_ids = np.arange(100, 110, dtype=np.int64)
    r_a, s_a = split_coco_proposal_indices(
        "train2017", 7, ranks, rect_ids, side_count=5
    )
    permutation = np.arange(9, -1, -1)
    r_b, s_b = split_coco_proposal_indices(
        "train2017", 7, ranks[permutation], rect_ids[permutation], side_count=5
    )
    assert set(map(int, rect_ids[r_a])) == set(map(int, rect_ids[permutation][r_b]))
    assert set(map(int, rect_ids[s_a])) == set(map(int, rect_ids[permutation][s_b]))


def _real_manifest(
    dataset_id: str,
    *,
    dimension: int,
    level: int | None,
    r_ids_sha256: str = "11" * 32,
    s_ids_sha256: str = "22" * 32,
) -> dict:
    counts = {
        "CMAB-1M": (471_643, 471_642, "float64"),
        "GeoLife-3D-1M": (500_000, 500_000, "int64"),
        "GeoLife-4D-1M": (500_000, 500_000, "int64"),
        "COCO-1M": (500_000, 500_000, "float64"),
    }
    repositories = {
        "CMAB-1M": "DannHiroaki/CMAB-Spatial-Join-0.08B",
        "GeoLife-3D-1M": "DannHiroaki/Geolife-Spatial-Join-0.15B",
        "GeoLife-4D-1M": "DannHiroaki/Geolife-Spatial-Join-0.15B",
        "COCO-1M": "DannHiroaki/COCO-Spatial-Join-1.23B",
    }
    revisions = {
        "CMAB-1M": "41e3c90fa42fc8eede910404fe3db29ad3897b81",
        "GeoLife-3D-1M": "a9b8439beb16de106f6ff3f54c73c6b6964d77af",
        "GeoLife-4D-1M": "a9b8439beb16de106f6ff3f54c73c6b6964d77af",
        "COCO-1M": "2e5f2a1ba741ba1148f0b2f42209a9da4635a6cb",
    }
    crs_ids = {
        "CMAB-1M": "CMAB-HF-Albers-Equal-Area-m",
        "GeoLife-3D-1M": "EPSG:3857-centimeter-plus-epoch-millisecond",
        "GeoLife-4D-1M": "EPSG:3857-centimeter-plus-epoch-millisecond",
        "COCO-1M": "COCO-original-image-pixels-plus-selected-image-index",
    }
    n_r, n_s, endpoint = counts[dataset_id]
    repository = repositories[dataset_id]
    revision = revisions[dataset_id]
    metadata: dict = {
        "dataset_id": dataset_id,
        "real_dataset": dataset_id,
        "crs_id": crs_ids[dataset_id],
        "source_identifier": f"{repository}@{revision}",
        "source_repository": repository,
        "source_revision": revision,
        "source": {
            "kind": "huggingface_dataset",
            "repo_id": repository,
            "revision": revision,
            "source_lock_sha256": "aa" * 32,
            "assets": [{"path": "published.parquet", "size": 1, "sha256": "bb" * 32}],
        },
        "importer_id": "anchor-hf-real-import-v2",
        "importer_sha256": "cc" * 32,
        "preprocessing_config_sha256": "dd" * 32,
    }
    if level is not None:
        metadata["level"] = level
    if dataset_id == "CMAB-1M":
        metadata["split_method"] = "cmab_hf_stratified_hash_tile_10km_v1"
        metadata["boundary_touching_diagnostic"] = {}
        metadata.update(
            {
                "object_identity_fields": ["source_file", "source_fid"],
                "object_id_method": "uint64_lexicographic_rank_v1",
                "source_identity_sha256": "ee" * 32,
                "published_building_uid_unique_count": 938_387,
                "published_building_uid_duplicate_groups": 4_898,
                "published_building_uid_excess_rows": 4_898,
                "published_building_uid_max_multiplicity": 2,
            }
        )
    elif dataset_id.startswith("GeoLife-"):
        metadata["dimension"] = dimension
        metadata[
            "split_method"
        ] = "geolife_hf_user_hash_then_month_tile_largest_remainder_v1"
        metadata["spatial_temporal_coverage_summary"] = {}
        for side, digest in (("R", "33" * 32), ("S", "44" * 32)):
            metadata[f"selected_point_manifest_{side}"] = {
                "file_name": f"selected-{side}.parquet",
                "count": 500_000,
                "sha256": digest,
            }
    else:
        metadata.update(
            {
                "image_subset_id": "coco_hash_subset_0",
                "image_subset_sha256": "55" * 32,
                "proposal_stage": "hf_published_rpn_top10000",
                "proposal_split_method": "coco_hf_published_hash_v1_balanced_5000_5000",
                "model_config_id": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                "checkpoint_sha256": "66" * 32,
                "upstream_builder_manifest_sha256": "77" * 32,
                "coordinate_source_type": "float32",
                "coordinate_conversion": "exact float32-to-float64 promotion",
                "eligible_image_count": 123_287,
                "proposal_identity_fields": [
                    "canonical_split",
                    "coco_image_id",
                    "rank",
                    "rect_id",
                ],
                "rank_or_score_summary": {},
                "selected_images_sidecar": {
                    "file_name": "coco-selected-images.json",
                    "sha256": "99" * 32,
                },
                "source_shards": [
                    {
                        "path": "data/rects/train2017/shard-000000.parquet",
                        "size": 1,
                        "linked_sha256": "aa" * 32,
                        "row_groups": [0],
                        "requests": 1,
                        "bytes_transferred": 1,
                    }
                ],
                "selected_images": [
                    {
                        "split": "train2017",
                        "image_id": index,
                        "source_z_idx": index,
                        "workload_z_idx": index,
                        "width": 640,
                        "height": 480,
                        "proposal_count": 10_000,
                        "proposal_rows_sha256": "88" * 32,
                    }
                    for index in range(100)
                ],
            }
        )
    return {
        "workload": {
            "file_name": f"{dataset_id}.bin",
            "dimension": dimension,
            "n_R": n_r,
            "n_S": n_s,
            "N_total": n_r + n_s,
            "endpoint_type": endpoint,
            "R_ids_sha256": r_ids_sha256,
            "S_ids_sha256": s_ids_sha256,
        },
        "metadata": metadata,
    }


def test_real_manifest_admission_checks_hf_source_level_and_subset() -> None:
    cmab = _real_manifest("CMAB-1M", dimension=2, level=2)
    _validate_real_manifest(
        dataset_id="CMAB-1M",
        dimension=2,
        level=2,
        manifest=cmab,
        path=pathlib.Path("cmab.bin"),
    )
    cmab["metadata"]["level"] = 1
    with pytest.raises(ValueError, match="level mismatch"):
        _validate_real_manifest(
            dataset_id="CMAB-1M",
            dimension=2,
            level=2,
            manifest=cmab,
            path=pathlib.Path("cmab.bin"),
        )
    coco = _real_manifest("COCO-1M", dimension=3, level=None)
    _validate_real_manifest(
        dataset_id="COCO-1M",
        dimension=3,
        level=None,
        manifest=coco,
        path=pathlib.Path("coco.bin"),
    )
    coco["metadata"]["image_subset_id"] = "wrong-subset"
    with pytest.raises(ValueError, match="subset mismatch"):
        _validate_real_manifest(
            dataset_id="COCO-1M",
            dimension=3,
            level=None,
            manifest=coco,
            path=pathlib.Path("coco.bin"),
        )


def test_cross_workload_admission_requires_shared_real_object_ids() -> None:
    first = _real_manifest("CMAB-1M", dimension=2, level=1)
    second = _real_manifest("CMAB-1M", dimension=2, level=2, r_ids_sha256="99" * 32)
    with pytest.raises(ValueError, match="R object ids"):
        _validate_real_cross_workload_identities(
            {pathlib.Path("level-1.bin"): first, pathlib.Path("level-2.bin"): second}
        )


def test_experiment_config_expands_all_twelve_sweeps() -> None:
    repository = pathlib.Path(__file__).resolve().parents[2]
    config = load_config(repository / "configs" / "experiments.json")
    cases = expand_experiments(config, data_root=repository / "data")
    assert len({case.experiment_id for case in cases}) == 12
    assert len(cases) == 55
    assert {case.data_seed for case in cases if case.dataset_type == "synthetic"} == {0}
    t_cases = [
        case
        for case in cases
        if case.experiment_id == "Alacarte-G2-t" and case.data_seed == 0
    ]
    assert len({case.workload_id for case in t_cases}) == 1
    assert {case.t for case in t_cases} == {
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
    }
