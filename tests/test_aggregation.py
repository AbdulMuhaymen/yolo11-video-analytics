import numpy as np

from video_indexer.aggregation import aggregate_second
from video_indexer.decoder import TimedFrame, _rotate
from video_indexer.pipeline import sample_frames
from video_indexer.pipeline import index_frames


def test_exact_half_counts_as_present():
    result = aggregate_second(0, [{"car": 1}, {"car": 1}, {}, {}])
    assert result.objects == ["car"]


def test_even_median_rounds_down():
    assert aggregate_second(0, [{"person": 1}, {"person": 4}]).people == 2


def test_odd_median_and_target_order():
    frames = [
        {"person": 7, "dog": 1, "cell_phone": 1},
        {"person": 1, "dog": 1, "cell_phone": 1},
        {"person": 3},
    ]
    result = aggregate_second(2, frames)
    assert result.people == 3
    assert result.objects == ["dog", "cell_phone"]


def test_empty_second():
    result = aggregate_second(4, [])
    assert result.people == 0 and result.objects == []


def test_sampling_spreads_across_second():
    frames = [TimedFrame(0, i / 10, np.zeros((1, 1, 3))) for i in range(10)]
    assert [frame.timestamp for frame in sample_frames(frames, 3)] == [0.1, 0.5, 0.8]


def test_clockwise_rotation():
    image = np.array([[1, 2], [3, 4]])
    assert _rotate(image, 90).tolist() == [[3, 1], [4, 2]]
    assert _rotate(image, 180).tolist() == [[4, 3], [2, 1]]
    assert _rotate(image, 270).tolist() == [[2, 4], [1, 3]]


def test_pipeline_emits_empty_timestamp_gaps():
    class Detector:
        def detect(self, images):
            return [{"dog": 1, "person": len(images)} for _ in images]

    image = np.zeros((1, 1, 3))
    results = list(index_frames([
        TimedFrame(0, 0.1, image),
        TimedFrame(2, 2.1, image),
    ], Detector()))
    assert [result.second for result in results] == [0, 1, 2]
    assert results[1].objects == [] and results[1].people == 0
