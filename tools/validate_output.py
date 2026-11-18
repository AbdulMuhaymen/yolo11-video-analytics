#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

ALLOWED = {"car", "dog", "laptop", "cell_phone"}


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    seen = set()
    seconds = defaultdict(list)
    count = 0
    with open_text(args.path) as lines:
        for line_number, line in enumerate(lines, 1):
            record = json.loads(line)
            key = (str(record["video_id"]), record["second"])
            assert key not in seen, f"line {line_number}: duplicate key {key}"
            assert isinstance(record["second"], int) and record["second"] >= 0
            assert isinstance(record["people"], int) and record["people"] >= 0
            assert isinstance(record["objects"], list) and set(record["objects"]) <= ALLOWED
            seen.add(key)
            seconds[key[0]].append(key[1])
            count += 1
    for video_id, values in seconds.items():
        assert sorted(values) == list(range(max(values) + 1)), f"{video_id}: non-contiguous seconds"
    print(f"OK: {count} records across {len(seconds)} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
