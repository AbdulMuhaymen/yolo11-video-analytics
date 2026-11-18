#!/usr/bin/env python3
"""Download model weights before an offline/timed run."""
from pathlib import Path


def main() -> None:
    from ultralytics import YOLO
    model = YOLO("yolo11l.pt")
    path = Path(model.ckpt_path)
    print(f"Model ready: {path.resolve()}")


if __name__ == "__main__":
    main()
