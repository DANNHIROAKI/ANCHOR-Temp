from __future__ import annotations

import argparse
import json

from anchor_exp.workload import read_workload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and inspect an ANCHOR workload")
    parser.add_argument("workload")
    parser.add_argument("--verify-file", action="store_true")
    args = parser.parse_args()
    item = read_workload(args.workload, verify_payload=True, verify_file=args.verify_file)
    print(
        json.dumps(
            {
                "path": str(item.path),
                "endpoint_type": item.endpoint_type,
                "dimension": item.dimension,
                "n_R": item.n_r,
                "n_S": item.n_s,
                "payload_sha256": item.payload_sha256,
                "file_sha256": item.file_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
