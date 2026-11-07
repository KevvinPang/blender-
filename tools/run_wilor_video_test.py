"""Run WiLoR on sampled video frames and save a hand-pose inspection package."""

from __future__ import annotations
from mocap_paths import WILOR_ROOT

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Ultralytics 8.1.x predates PyTorch's weights_only default. The detector
# checkpoint is the official local file downloaded for this test, so retain
# the older trusted-checkpoint loading behavior for this process.
_torch_load = torch.load


def _trusted_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(WILOR_ROOT))
# This Windows run has no EGL runtime installed. Leaving the variable unset
# makes pyrender use its native pyglet context instead.
os.environ.pop("PYOPENGL_PLATFORM", None)

from ultralytics import YOLO  # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
from wilor.models import load_wilor  # noqa: E402
from wilor.utils import recursive_to  # noqa: E402
from wilor.utils.renderer import cam_crop_to_full  # noqa: E402


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
COLORS = [(255, 120, 40), (40, 210, 255)]


def project(points: np.ndarray, cam_t: np.ndarray, focal: float, size: np.ndarray) -> np.ndarray:
    points = points + cam_t
    z = np.maximum(points[:, 2:3], 1e-6)
    center = np.array([size[0] / 2.0, size[1] / 2.0], dtype=np.float32)
    result = points[:, :2] / z * focal
    return result + center


def draw_pose(image: np.ndarray, joints: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    h, w = image.shape[:2]
    for a, b in CONNECTIONS:
        pa, pb = joints[a], joints[b]
        if 0 <= pa[0] < w and 0 <= pa[1] < h and 0 <= pb[0] < w and 0 <= pb[1] < h:
            cv2.line(image, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), color, 2, cv2.LINE_AA)
    for index, point in enumerate(joints):
        x, y = np.round(point).astype(int)
        cv2.circle(image, (x, y), 4 if index == 0 else 3, color, -1, cv2.LINE_AA)
    x, y = np.round(joints[0]).astype(int)
    cv2.putText(image, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def infer_frame(model, detector, model_cfg, frame: np.ndarray, device: torch.device, confidence: float):
    # Both the YOLO ndarray API and ViTDetDataset expect OpenCV BGR images.
    detections = detector(frame, conf=confidence, verbose=False)[0]
    boxes, sides, scores = [], [], []
    for det in detections:
        boxes.append(det.boxes.xyxy[0].detach().cpu().numpy().tolist())
        sides.append(float(det.boxes.cls[0].detach().cpu().item()))
        scores.append(float(det.boxes.conf[0].detach().cpu().item()))
    if not boxes:
        return []

    dataset = ViTDetDataset(
        model_cfg,
        frame,
        np.asarray(boxes, dtype=np.float32),
        np.asarray(sides, dtype=np.float32),
        rescale_factor=2.0,
        fp16=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(4, len(boxes)), shuffle=False, num_workers=0)
    outputs = []
    with torch.inference_mode():
        for batch in loader:
            batch = recursive_to(batch, device)
            out = model(batch)
            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            image_size = batch["img_size"].float()
            focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * image_size.max()
            cam_t = cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                image_size,
                focal,
            ).detach().cpu().numpy()
            for index in range(len(batch["img"])):
                joints3d = out["pred_keypoints_3d"][index].detach().cpu().numpy()
                right = bool(batch["right"][index].detach().cpu().item())
                if not right:
                    joints3d[:, 0] *= -1
                joints2d = project(joints3d, cam_t[index], float(focal), image_size[index].detach().cpu().numpy())
                mano = {
                    name: value[index].detach().cpu().numpy().tolist()
                    for name, value in out["pred_mano_params"].items()
                }
                outputs.append({
                    "right": right,
                    "confidence": scores[len(outputs)],
                    "bbox": boxes[len(outputs)],
                    "joints3d": joints3d.tolist(),
                    "joints2d": joints2d.tolist(),
                    "cam_t": cam_t[index].tolist(),
                    "mano": mano,
                })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    output = Path(args.out).resolve()
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(WILOR_ROOT)

    model, model_cfg = load_wilor(
        checkpoint_path=str(WILOR_ROOT / "pretrained_models" / "wilor_final.ckpt"),
        cfg_path=str(WILOR_ROOT / "pretrained_models" / "model_config.yaml"),
    )
    model = model.to(device).eval()
    detector = YOLO(str(WILOR_ROOT / "pretrained_models" / "detector.pt")).to(device)

    capture = cv2.VideoCapture(str(Path(args.video)))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = list(range(0, total, max(1, args.stride)))[: args.max_frames]
    records = []
    for frame_index in selected:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        hands = infer_frame(model, detector, model_cfg, frame, device, args.confidence)
        annotated = frame.copy()
        for hand_index, hand in enumerate(hands):
            color = COLORS[hand_index % len(COLORS)]
            draw_pose(annotated, np.asarray(hand["joints2d"], dtype=np.float32), color,
                      f"{'R' if hand['right'] else 'L'} {hand['confidence']:.2f}")
        cv2.putText(annotated, f"frame {frame_index}/{total - 1} hands {len(hands)}", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(frame_dir / f"frame_{frame_index:06d}.jpg"), annotated)
        records.append({"frame": frame_index, "time": frame_index / fps, "hands": hands})

    capture.release()
    (output / "wilor_hand_results.json").write_text(
        json.dumps({"video": str(Path(args.video).resolve()), "fps": fps, "frames": records}, indent=2),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps({"device": str(device), "total_video_frames": total, "sampled_frames": len(records),
                    "detected_hands": sum(len(row["hands"]) for row in records)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "device": str(device), "sampled_frames": len(records),
                      "detected_hands": sum(len(row["hands"]) for row in records)}))


if __name__ == "__main__":
    main()
