"""Local, single-person evaluation wrapper; upstream model code is unchanged."""

import gc
import importlib
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "checkpoints" / "sam-3d-body-dinov3"
OUTPUT_DIR = ROOT / "local_output"
REPO_ID = "facebook/sam-3d-body-dinov3"
MODEL_URL = f"https://huggingface.co/{REPO_ID}"
REQUIRED_FILES = ("model.ckpt", "model_config.yaml", "assets/mhr_model.pt")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOMENTUM_ENABLED", "0")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
from public_backend import REPO_ID as PUBLIC_REPO_ID, WEIGHTS, weights_available
from video_input import open_video, read_frame
_estimator = None
_lock = threading.RLock()


def missing_weights():
    if weights_available():
        return []
    return [name for name in REQUIRED_FILES if not (MODEL_DIR / name).is_file()]


def diagnostics(check_imports=False):
    import torch

    result = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "missing_weights": missing_weights(),
        "model_directory": str(MODEL_DIR),
        "active_checkpoint": str(WEIGHTS) if weights_available() else str(MODEL_DIR / "model.ckpt"),
        "checkpoint_source": PUBLIC_REPO_ID if weights_available() else REPO_ID,
        "comfyui_required": False,
        "model_loaded": _estimator is not None,
        "mode": "single-person, full-image, default FOV; no temporal smoothing",
    }
    if _estimator is not None and hasattr(_estimator, "load_report"):
        result["load_report"] = _estimator.load_report
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        result["gpu_compute_capability"] = list(torch.cuda.get_device_capability(0))
    if check_imports:
        for name in ("torchvision", "cv2", "gradio", "plotly", "sam_3d_body"):
            try:
                importlib.import_module(name)
                result[f"import_{name}"] = "OK"
            except Exception as error:
                result[f"import_{name}"] = f"{type(error).__name__}: {error}"
        if torch.cuda.is_available():
            try:
                x = torch.randn(64, 64, device="cuda")
                result["cuda_matmul"] = bool(torch.isfinite(x @ x).all().item())
                torch.cuda.synchronize()
            except Exception as error:
                result["cuda_matmul"] = f"{type(error).__name__}: {error}"
    return result


def download_model():
    if weights_available():
        from public_backend import verify_weights
        verify_weights()
        return "Local public BF16 checkpoint verified; no download or login needed."
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(MODEL_DIR),
            allow_patterns=["model.ckpt", "model_config.yaml", "assets/*", "LICENSE*", "README.md"],
            max_workers=2,
        )
    except GatedRepoError:
        raise RuntimeError(
            f"Model access is required: {MODEL_URL}. Request access, then run the desktop model login shortcut."
        ) from None
    except HfHubHTTPError as error:
        code = getattr(getattr(error, "response", None), "status_code", None)
        raise RuntimeError(f"Hugging Face download failed (HTTP {code}). Check account access and network.") from None
    missing = missing_weights()
    if missing:
        raise RuntimeError(f"Download incomplete: {', '.join(missing)}")
    return "Model files downloaded."


def read_video_frame(path, seconds):
    import cv2
    import numpy as np

    if not path or not Path(path).is_file():
        raise ValueError("Select an existing video file.")
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Time must be a finite, nonnegative number.")
    cap = open_video(path)
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise ValueError("Cannot read video frame rate or duration.")
        index = int(round(seconds * fps))
        if index >= count:
            raise ValueError(f"Time is past the video end ({count / fps:.3f} seconds).")
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = read_frame(cap)
        if not ok:
            raise ValueError("Cannot decode the selected frame.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), {
            "source": str(Path(path).resolve()),
            "fps": float(fps),
            "frame": index,
            "seconds": index / fps,
            "duration_seconds": count / fps,
            "shape": list(np.shape(image)),
            "orientation_clockwise_degrees": cap.get(cv2.CAP_PROP_ORIENTATION_META),
        }
    finally:
        cap.release()


def skeleton_edges():
    from sam_3d_body.metadata.mhr70 import pose_info

    name_to_id = {item["name"]: key for key, item in pose_info["keypoint_info"].items()}
    return [
        (name_to_id[item["link"][0]], name_to_id[item["link"][1]], item["color"])
        for item in pose_info["skeleton_info"].values()
    ]


def overlay_skeleton(image, keypoints):
    import cv2
    import numpy as np

    result = np.asarray(image, dtype=np.uint8).copy()
    h, w = result.shape[:2]
    points = np.asarray(keypoints)
    visible = np.isfinite(points[:, :2]).all(axis=1)
    visible &= (points[:, 0] >= 0) & (points[:, 0] < w)
    visible &= (points[:, 1] >= 0) & (points[:, 1] < h)
    for a, b, color in skeleton_edges():
        if visible[a] and visible[b]:
            cv2.line(result, tuple(points[a, :2].astype(int)), tuple(points[b, :2].astype(int)), color, 2, cv2.LINE_AA)
    for point in points[:70][visible[:70]]:
        cv2.circle(result, tuple(point[:2].astype(int)), 3, (255, 240, 70), -1, cv2.LINE_AA)
    return result


def plot_skeleton(keypoints):
    import numpy as np
    import plotly.graph_objects as go
    from sam_3d_body.metadata.mhr70 import mhr_names

    points = np.asarray(keypoints)[:70].copy()
    points[:, 1:] *= -1
    traces = []
    for a, b, color in skeleton_edges():
        pair = points[[a, b]]
        traces.append(go.Scatter3d(x=pair[:, 0], y=pair[:, 1], z=pair[:, 2], mode="lines", line={"color": f"rgb{tuple(color)}", "width": 4}, hoverinfo="skip", showlegend=False))
    traces.append(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", text=mhr_names, hoverinfo="text", marker={"size": 3, "color": "#169b86"}, showlegend=False))
    fig = go.Figure(traces)
    fig.update_layout(scene={"aspectmode": "data", "yaxis": {"title": "Up"}, "zaxis": {"title": "Depth"}, "camera": {"up": {"x": 0, "y": 1, "z": 0}, "eye": {"x": 0.1, "y": 0.1, "z": 2}}}, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig


def get_estimator():
    global _estimator
    import torch

    if missing_weights():
        raise RuntimeError(f"Missing model weights. Request access at {MODEL_URL}, then run model login/download.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This evaluation uses the upstream CUDA inference path.")
    if _estimator is None:
        torch.backends.cuda.matmul.allow_tf32 = True
        if weights_available():
            from public_backend import PublicEstimator
            _estimator = PublicEstimator()
        else:
            from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
            model, config = load_sam_3d_body(
                str(MODEL_DIR / "model.ckpt"), device="cuda", mhr_path=str(MODEL_DIR / "assets" / "mhr_model.pt")
            )
            _estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=config)
    return _estimator


def unload_model():
    global _estimator
    import torch

    with _lock:
        _estimator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return diagnostics()


def infer_image(image, source=None):
    import cv2
    import numpy as np
    import torch

    if image is None:
        raise ValueError("Choose an image or extract a video frame first.")
    with _lock, torch.inference_mode():
        started = time.perf_counter()
        try:
            estimator = get_estimator()
            started = time.perf_counter()
            people = estimator.process_one_image(np.asarray(image, dtype=np.uint8), inference_type="full")
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            unload_model()
            raise RuntimeError("GPU memory is full. Close heavy Blender previews before retrying.") from None
        if not people:
            raise RuntimeError("No prediction returned.")
        person = people[0]
        elapsed = time.perf_counter() - started
        folder = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder.mkdir(parents=True, exist_ok=False)
        numeric = {key: value for key, value in person.items() if value is not None}
        np.savez_compressed(folder / "prediction.npz", **numeric)
        overlay = overlay_skeleton(image, person["pred_keypoints_2d"])
        # imencode/tofile supports Unicode paths on Windows.
        encoded, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not encoded:
            raise RuntimeError("Cannot encode preview image.")
        png.tofile(str(folder / "overlay.png"))
        report = {
            "source": source,
            "inference_seconds": elapsed,
            "checkpoint": PUBLIC_REPO_ID if hasattr(estimator, "load_report") else REPO_ID,
            "load_report": getattr(estimator, "load_report", None),
            "inference_type": "full",
            "automatic_detector": False,
            "fov": "upstream default; not calibrated",
            "temporal_smoothing": False,
            "blender_retargeted": False,
            "output": str(folder),
        }
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return overlay, plot_skeleton(person["pred_keypoints_3d"]), str(folder / "prediction.npz"), report
