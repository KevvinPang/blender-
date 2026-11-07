import cv2
import numpy as np
import pytest

from video_input import read_frame, SequentialFrames


class Capture:
    def __init__(self, frame, angle, automatic=False, ok=True):
        self.frame, self.angle, self.automatic, self.ok = frame, angle, automatic, ok

    def get(self, key):
        return self.automatic if key == cv2.CAP_PROP_ORIENTATION_AUTO else self.angle

    def read(self):
        return self.ok, self.frame


@pytest.mark.parametrize('angle, turns', [(0, 0), (90, -1), (180, 2), (270, 1)])
def test_fallback_rotation(angle, turns):
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    ok, output = read_frame(Capture(image, angle))
    assert ok
    np.testing.assert_array_equal(output, np.rot90(image, turns))


def test_does_not_double_rotate():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    ok, output = read_frame(Capture(image, 270, automatic=True))
    assert ok and output is image
    assert read_frame(Capture(None, 270, ok=False)) == (False, None)


def test_sequential_sampling_does_not_seek():
    class Stream(Capture):
        def __init__(self):
            self.count = 0
            self.angle, self.automatic = 0, True
        def read(self):
            self.count += 1
            return (True, self.count-1) if self.count <= 8 else (False, None)
    stream = Stream()
    reader = SequentialFrames(stream)
    assert reader.read(0) == (True, 0)
    assert reader.read(3) == (True, 3)
    assert reader.read(3) == (True, 3)
    assert stream.count == 4
    assert reader.read(7) == (True, 7)
    assert reader.read(8) == (False, None)
    with pytest.raises(ValueError):
        reader.read(2)
