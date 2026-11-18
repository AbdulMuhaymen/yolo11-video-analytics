from typing import Iterable, Iterator, List, Sequence

from .aggregation import SecondResult, aggregate_second
from .decoder import TimedFrame


def sample_frames(frames: Sequence[TimedFrame], samples_per_second: int) -> List[TimedFrame]:
    """Choose frames spread across a second; 0 means exact/all-frame mode."""
    if samples_per_second <= 0 or len(frames) <= samples_per_second:
        return list(frames)
    count = len(frames)
    indices = [min(count - 1, int((i + 0.5) * count / samples_per_second))
               for i in range(samples_per_second)]
    return [frames[index] for index in sorted(set(indices))]


def index_frames(
    frames: Iterable[TimedFrame], detector: object, samples_per_second: int = 0
) -> Iterator[SecondResult]:
    # Keep only one second of decoded RGB frames in memory. A long source video
    # can otherwise consume many gigabytes before inference even starts.
    current_second = 0
    bucket: List[TimedFrame] = []
    for frame in frames:
        while frame.second > current_second:
            selected = sample_frames(bucket, samples_per_second)
            detections = detector.detect([item.image for item in selected]) if selected else []
            yield aggregate_second(current_second, detections)
            bucket = []
            current_second += 1
        # Broken timestamps occasionally move backwards; retain the frame in
        # the active bucket rather than emitting duplicate seconds.
        if frame.second <= current_second:
            bucket.append(frame)
    if bucket:
        selected = sample_frames(bucket, samples_per_second)
        detections = detector.detect([frame.image for frame in selected]) if selected else []
        yield aggregate_second(current_second, detections)
