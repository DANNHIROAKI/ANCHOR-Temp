"""Aggregate raw ANCHOR JSONL runs without discarding failure records."""

from __future__ import annotations

import argparse
import json
import pathlib

from anchor_exp.aggregate import (
    aggregate_records,
    read_jsonl,
    write_aggregate_json,
    write_metrics_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metrics-csv", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    aggregate = aggregate_records(read_jsonl(args.inputs))
    write_aggregate_json(args.output, aggregate)
    if args.metrics_csv:
        write_metrics_csv(args.metrics_csv, aggregate["metrics"])
    print(json.dumps({"status": "OK", "records": aggregate["record_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
