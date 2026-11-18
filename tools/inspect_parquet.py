#!/usr/bin/env python3
"""Print a shard's schema and a size-safe preview of its first row."""
import argparse
from pathlib import Path


def describe(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{type(value).__name__}: {len(value):,} bytes>"
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    text = repr(value)
    return text if len(text) <= 300 else text[:297] + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(args.path)
    print(parquet.schema_arrow)
    print(f"row groups: {parquet.num_row_groups}; rows: {parquet.metadata.num_rows}")
    first = next(parquet.iter_batches(batch_size=1)).to_pylist()[0]
    print("first row:")
    for key, value in first.items():
        print(f"  {key}: {describe(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
