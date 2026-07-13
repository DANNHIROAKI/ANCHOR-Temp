"""Render main and explanatory ANCHOR experiment figures."""

from __future__ import annotations

import argparse
import json
import pathlib

from anchor_exp.aggregate import read_jsonl
from anchor_exp.plotting import plot_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    paths = plot_results(read_jsonl(args.inputs), args.output_dir, formats=formats)
    print(json.dumps({"status": "OK", "figures": len(paths)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
