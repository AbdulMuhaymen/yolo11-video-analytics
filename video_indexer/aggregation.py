from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping


TARGET_CLASSES = ("person", "car", "dog", "laptop", "cell_phone")
OBJECT_CLASSES = ("car", "dog", "laptop", "cell_phone")


@dataclass(frozen=True)
class SecondResult:
    second: int
    objects: List[str]
    people: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def aggregate_second(
    second: int, detections: Iterable[Mapping[str, int]]
) -> SecondResult:
    """Aggregate per-frame counts using the challenge's majority/median rules."""
    frames = list(detections)
    if not frames:
        return SecondResult(second=second, objects=[], people=0)

    threshold = (len(frames) + 1) // 2
    objects = [
        name
        for name in OBJECT_CLASSES
        if sum(int(frame.get(name, 0)) > 0 for frame in frames) >= threshold
    ]

    people_counts = sorted(max(0, int(frame.get("person", 0))) for frame in frames)
    middle = len(people_counts) // 2
    if len(people_counts) % 2:
        people = people_counts[middle]
    else:
        people = (people_counts[middle - 1] + people_counts[middle]) // 2
    return SecondResult(second=second, objects=objects, people=people)
