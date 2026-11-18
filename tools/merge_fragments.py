#!/usr/bin/env python3
import argparse
import gzip
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sorted JSONL/JSONL.GZ shard outputs")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = sorted(list(args.input.glob("*.jsonl")) + list(args.input.glob("*.jsonl.gz")))
    if not files:
        parser.error("no fragments found")
    opener_out = gzip.open if args.output.suffix == ".gz" else open
    with opener_out(args.output, "wb") as destination:
        for path in files:
            opener_in = gzip.open if path.suffix == ".gz" else open
            with opener_in(path, "rb") as source:
                shutil.copyfileobj(source, destination)
    print(f"Merged {len(files)} fragments into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
