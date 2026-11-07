"""Decode video frames in display orientation, including phone rotation tags."""
import math

import cv2


def open_video(path):
    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return capture


def read_frame(capture):
    ok, frame = capture.read()
    if not ok or capture.get(cv2.CAP_PROP_ORIENTATION_AUTO):
        return ok, frame
    # Some backends expose the tag but do not implement automatic rotation.
    angle = capture.get(cv2.CAP_PROP_ORIENTATION_META)
    if math.isfinite(angle):
        rotation = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(round(angle) % 360)
        if rotation is not None:
            frame = cv2.rotate(frame, rotation)
    return ok, frame


class SequentialFrames:
    """Decode increasing frame indices without unreliable VFR/GOP seeking."""
    def __init__(self, capture):
        self.capture = capture
        self.index = -1
        self.frame = None

    def read(self, index):
        if index < 0 or index < self.index or int(index) != index:
            raise ValueError('Frame indices must be nonnegative and nondecreasing')
        while self.index < index:
            ok, frame = read_frame(self.capture)
            if not ok:
                return False, None
            self.index += 1
            self.frame = frame
        return self.frame is not None, self.frame
