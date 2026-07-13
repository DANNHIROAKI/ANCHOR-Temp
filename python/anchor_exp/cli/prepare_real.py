"""Unified finished-dataset acquisition and real-workload importer."""

from __future__ import annotations

import argparse
import json
import pathlib

from anchor_exp.hf_real import (
    DEFAULT_CONFIG,
    DataPreparationError,
    check_prerequisites,
    prepare_real_datasets,
    resolve_configuration_paths,
    verify_real_collections,
)


def _datasets(args: argparse.Namespace) -> list[str]:
    if args.all:
        return ["cmab", "geolife", "coco"]
    selected = [name for name in ("cmab", "geolife", "coco") if getattr(args, name)]
    if not selected:
        raise ValueError("select --all or at least one of --cmab, --geolife, --coco")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cmab", action="store_true")
    parser.add_argument("--geolife", action="store_true")
    parser.add_argument("--coco", action="store_true")
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lock", type=pathlib.Path)
    parser.add_argument(
        "--endpoint", help="Hub endpoint override; HF_ENDPOINT is also honored"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    try:
        datasets = _datasets(args)
        config, _, config_path, lock_path = resolve_configuration_paths(
            args.config, args.lock
        )
        root = args.data_root.resolve()
        if args.check_only:
            details = check_prerequisites(
                root, datasets, config_path=config_path, lock_path=lock_path
            )
        elif args.verify_only:
            details = verify_real_collections(root, datasets, lock_path=lock_path)
        else:
            details = prepare_real_datasets(
                root,
                datasets,
                config_path=config_path,
                lock_path=lock_path,
                endpoint=args.endpoint,
                download_only=args.download_only,
            )
        print(
            json.dumps({"status": "OK", **details}, ensure_ascii=False, sort_keys=True)
        )
        return 0
    except (DataPreparationError, OSError, ValueError, KeyError) as error:
        print(
            json.dumps(
                {"status": "DATASET-CONSTRUCTION-FAILED", "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
