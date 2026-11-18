#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from video_indexer.decoder import decode_video
from video_indexer.inference import YoloDetector
from video_indexer.pipeline import index_frames

LOG = logging.getLogger("video-indexer")


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def extract_video(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return value["bytes"]
        if value.get("path"):
            return value["path"]
    return value


def iter_rows(path: Path, video_column: str, id_column: Optional[str]) -> Iterator[Tuple[str, Any]]:
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    columns = [video_column] + ([id_column] if id_column else [])
    for row_index, batch in enumerate(parquet.iter_batches(batch_size=1, columns=columns)):
        row = batch.to_pylist()[0]
        video_id = str(row[id_column]) if id_column else f"{path.stem}:{row_index}"
        yield video_id, extract_video(row[video_column])


def main() -> int:
    parser = argparse.ArgumentParser(description="Index target objects in Parquet-hosted MP4 videos")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="yolo11l.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-second", type=int, default=0,
                        help="0 uses every frame (exact); otherwise evenly samples N frames")
    parser.add_argument("--video-column", default="video")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    files = sorted(args.input.glob("*.parquet")) if args.input.is_dir() else [args.input]
    if not files:
        parser.error(f"no Parquet files found at {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completion_path = Path(str(args.output) + ".completed")
    completed = set()
    if args.resume and completion_path.exists():
        completed = set(completion_path.read_text(encoding="utf-8").splitlines())

    detector = YoloDetector(args.model, choose_device(args.device), args.batch_size)
    mode = "a" if args.resume else "w"
    errors = 0
    completion_mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8", buffering=1) as output, completion_path.open(
        completion_mode, encoding="utf-8", buffering=1
    ) as completion_file:
        for parquet_path in files:
            LOG.info("reading %s", parquet_path)
            try:
                rows = iter_rows(parquet_path, args.video_column, args.id_column or None)
                for video_id, source in rows:
                    if video_id in completed:
                        continue
                    try:
                        # Buffer one video's compact JSON records so a decoder
                        # failure never leaves a partially written video behind.
                        records = [
                            json.dumps({"video_id": video_id, **result.to_dict()}, separators=(",", ":"))
                            for result in index_frames(decode_video(source), detector, args.samples_per_second)
                        ]
                        output.write("\n".join(records) + ("\n" if records else ""))
                        output.flush()
                        completion_file.write(video_id + "\n")
                    except Exception:
                        errors += 1
                        LOG.exception("failed video %s", video_id)
            except Exception:
                errors += 1
                LOG.exception("failed shard %s", parquet_path)
    LOG.info("finished with %d error(s)", errors)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
