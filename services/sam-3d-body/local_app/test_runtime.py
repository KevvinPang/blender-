from pathlib import Path

import cv2
import numpy as np
import pytest

from runtime import missing_weights, overlay_skeleton, plot_skeleton, read_video_frame, skeleton_edges


def test_metadata_and_preview():
    edges = skeleton_edges()
    assert edges and all(0 <= a < 70 and 0 <= b < 70 for a, b, _ in edges)
    points = np.column_stack([np.linspace(10, 90, 308), np.linspace(15, 85, 308)])
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    overlay = overlay_skeleton(image, points)
    assert overlay.shape == image.shape and overlay.max() > 0 and image.max() == 0
    points[0] = np.nan
    assert overlay_skeleton(image, points).shape == image.shape
    fig = plot_skeleton(np.zeros((308, 3)))
    assert len(fig.data) == len(edges) + 1


def test_weight_check_uses_required_files(tmp_path, monkeypatch):
    monkeypatch.setattr("runtime.weights_available", lambda: False)
    monkeypatch.setattr("runtime.MODEL_DIR", tmp_path)
    assert len(missing_weights()) == 3
    (tmp_path / "model.ckpt").touch()
    assert "model.ckpt" not in missing_weights()


def test_public_checkpoint_is_sufficient(monkeypatch):
    monkeypatch.setattr("runtime.weights_available", lambda: True)
    assert missing_weights() == []


def test_video_frame(tmp_path):
    path = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    assert writer.isOpened()
    try:
        for value in range(10):
            writer.write(np.full((64, 64, 3), value * 20, dtype=np.uint8))
    finally:
        writer.release()
    image, info = read_video_frame(path, 0.3)
    assert image.shape == (64, 64, 3) and info["frame"] == 3
    with pytest.raises(ValueError, match="past the video end"):
        read_video_frame(path, 2)
    with pytest.raises(ValueError, match="finite"):
        read_video_frame(path, float("nan"))


def test_missing_video():
    with pytest.raises(ValueError, match="existing video"):
        read_video_frame(Path("definitely-not-existing.mp4"), 0)
