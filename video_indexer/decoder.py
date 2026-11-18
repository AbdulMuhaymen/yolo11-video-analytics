import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np


@dataclass
class TimedFrame:
    second: int
    timestamp: float
    image: np.ndarray


def _rotation(stream: object) -> int:
    metadata = getattr(stream, "metadata", {}) or {}
    try:
        value = int(float(metadata.get("rotate", 0))) % 360
    except (TypeError, ValueError):
        value = 0
    return value if value in (0, 90, 180, 270) else 0


def _rotate(image: np.ndarray, degrees_clockwise: int) -> np.ndarray:
    if degrees_clockwise == 90:
        return np.ascontiguousarray(np.rot90(image, k=3))
    if degrees_clockwise == 180:
        return np.ascontiguousarray(np.rot90(image, k=2))
    if degrees_clockwise == 270:
        return np.ascontiguousarray(np.rot90(image, k=1))
    return image


def decode_video(source: Union[bytes, bytearray, memoryview, str, Path]) -> Iterator[TimedFrame]:
    """Yield RGB frames, timestamped relative to the first decoded frame."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - exercised in integration
        raise RuntimeError("PyAV is required; run `python prepare.py`") from exc

    temporary: Optional[str] = None
    if isinstance(source, (bytes, bytearray, memoryview)):
        # BytesIO works for most codecs. A temporary file is a useful fallback for
        # demuxers that seek using a real file descriptor.
        input_value = io.BytesIO(bytes(source))
    else:
        input_value = str(source)

    try:
        try:
            container = av.open(input_value)
        except Exception:
            if not isinstance(source, (bytes, bytearray, memoryview)):
                raise
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
                handle.write(bytes(source))
                temporary = handle.name
            container = av.open(temporary)

        with container:
            stream = container.streams.video[0]
            rotation = _rotation(stream)
            first_ts: Optional[float] = None
            fallback_index = 0
            rate = float(stream.average_rate) if stream.average_rate else 30.0
            for frame in container.decode(stream):
                if frame.time is not None:
                    absolute_ts = float(frame.time)
                elif frame.pts is not None and stream.time_base is not None:
                    absolute_ts = float(frame.pts * stream.time_base)
                else:
                    absolute_ts = fallback_index / max(rate, 1.0)
                fallback_index += 1
                if first_ts is None:
                    first_ts = absolute_ts
                timestamp = max(0.0, absolute_ts - first_ts)
                yield TimedFrame(
                    second=int(timestamp),
                    timestamp=timestamp,
                    image=_rotate(frame.to_ndarray(format="rgb24"), rotation),
                )
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
