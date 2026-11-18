from typing import Dict, List, Sequence

import numpy as np


COCO_TARGETS = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell_phone"}


class YoloDetector:
    def __init__(self, model_path: str, device: str, batch_size: int = 8) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Ultralytics is required; run `python prepare.py`") from exc
        self.model = YOLO(model_path)
        self.device = device
        self.batch_size = batch_size

    def detect(self, images: Sequence[np.ndarray]) -> List[Dict[str, int]]:
        if not images:
            return []
        output: List[Dict[str, int]] = []
        for start in range(0, len(images), self.batch_size):
            results = self.model.predict(
                source=list(images[start : start + self.batch_size]),
                imgsz=640,
                conf=0.4,
                iou=0.7,
                device=self.device,
                verbose=False,
            )
            for result in results:
                counts: Dict[str, int] = {}
                classes = result.boxes.cls.detach().cpu().tolist()
                for class_id_value in classes:
                    name = COCO_TARGETS.get(int(class_id_value))
                    if name:
                        counts[name] = counts.get(name, 0) + 1
                output.append(counts)
        return output
