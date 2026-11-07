"""Cancellable local video jobs; Blender retrieves a completed, timed skeleton take."""
import json
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import runtime
from video_input import open_video, SequentialFrames
from face_capture import run_face, run_alignment, FaceCancelled
from hand_capture import run_hands

PROTOCOL = "SAM3D_BLENDER_TAKE"
VERSION = 1
# Raw MHR: Y-up, Z-forward. Blender model: Z-up, -Y-forward.
RAW_TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
OUTPUT_TO_BLENDER = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
JOBS_DIR = runtime.OUTPUT_DIR / "video_jobs"
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam3d-video")
_guard = threading.RLock()
_jobs = {}
_cancels = {}


def sample_indices(count, fps, rate, start=0.0, end=0.0):
    values = (fps, rate, start, end)
    if not all(math.isfinite(v) for v in values) or fps <= 0 or not 1 <= rate <= 30 or start < 0:
        raise ValueError("Invalid frame rate or video interval.")
    duration = count / fps
    end = duration if end == 0 else min(end, duration)
    if start >= end or count < 1:
        raise ValueError("Start time must be before the end of the video.")
    first, last = int(math.ceil(start * fps)), min(count - 1, int(math.ceil(end * fps)) - 1)
    indices = np.unique(np.rint(np.arange(first, last + 0.01, fps / min(rate, fps))).astype(int))
    if not len(indices) or indices[-1] != last:
        indices = np.append(indices, last)
    return indices.tolist()


def _update(job_id, **values):
    with _guard:
        _jobs[job_id].update(values)
        status = dict(_jobs[job_id])
        path = JOBS_DIR / job_id / "status.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf8")
        temp.replace(path)


def get_job(job_id):
    if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("Invalid job ID.")
    with _guard:
        if job_id in _jobs:
            return dict(_jobs[job_id])
    path = JOBS_DIR / job_id / "status.json"
    if not path.is_file():
        raise KeyError("Job not found.")
    status = json.loads(path.read_text(encoding="utf8"))
    if status["state"] in ("queued", "running"):
        status.update(state="error", error="The service restarted before this job completed.")
    return status


def start_job(video, rate=10, start=0.0, end=0.0, face=False, face_rate=30, alignment=False, hands='OFF'):
    if hands not in ('OFF', 'L', 'R', 'BOTH'):
        raise ValueError('Invalid WiLoR hand selection')
    if not isinstance(alignment, bool):
        raise ValueError('Invalid screen alignment option')
    if not isinstance(face, bool) or not isinstance(face_rate, (int, float)) or not math.isfinite(face_rate) or not 1 <= face_rate <= 60:
        raise ValueError('Invalid facial capture options')
    path = Path(video).resolve()
    if not path.is_file() or path.suffix.lower() not in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"):
        raise ValueError("Select an existing local video file.")
    capture = open_video(path)
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open this video.")
        fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_indices(count, fps, float(rate), float(start), float(end))
        if len(indices) < 2:
            raise ValueError("Select an interval containing at least two video frames.")
    finally:
        capture.release()
    job_id = uuid.uuid4().hex
    folder = JOBS_DIR / job_id
    folder.mkdir(parents=True)
    with _guard:
        if any(row["state"] in ("queued", "running") for row in _jobs.values()):
            raise ValueError("Another video is processing. Wait or cancel it first.")
        _jobs[job_id] = {"id": job_id, "state": "queued", "video": str(path),
                         "fps": fps, "sample_fps": min(float(rate), fps),
                         "source_frame_count": count, "total": len(indices), "done": 0,
                         "elapsed_seconds": 0, "error": "", "take_path": "", "phase": "body"}
        _cancels[job_id] = threading.Event()
        _update(job_id)
        _pool.submit(_process, job_id, path, indices, fps, face, face_rate, alignment, hands)
        return dict(_jobs[job_id])


def cancel_job(job_id):
    status = get_job(job_id)
    with _guard:
        if job_id in _cancels:
            _cancels[job_id].set()
    return status


def _attach_face(job_id, payload, face_rate, clock):
    runtime.unload_model()
    rows, fps = payload['frames'], payload['fps']
    _update(job_id, phase='face', done=0, total=1)
    face = run_face(payload['video'], JOBS_DIR/job_id, face_rate,
                    rows[0]['source_frame']/fps, (rows[-1]['source_frame']+1)/fps,
                    _cancels[job_id], lambda **values: _update(job_id, **values,
                        elapsed_seconds=round(time.monotonic()-clock, 2)))
    payload['face'] = face
    payload['limitations'] = [v for v in payload.get('limitations', []) if v != 'no facial tracking']
    payload['limitations'].extend(face['limitations'])


def start_face_job(body_job, face_rate=30):
    if not isinstance(face_rate, (int, float)) or not math.isfinite(face_rate) or not 1 <= face_rate <= 60:
        raise ValueError('Invalid face sampling rate')
    source = get_job(body_job)
    if source['state'] != 'complete':
        raise ValueError('Finish body recognition before adding face capture')
    take = JOBS_DIR/body_job/'take.json'
    if not take.is_file():
        raise ValueError('Body cache not found')
    payload = json.loads(take.read_text(encoding='utf8'))
    if not Path(payload['video']).is_file():
        raise ValueError('The original video is needed for facial capture')
    job_id = uuid.uuid4().hex
    with _guard:
        if any(row['state'] in ('queued', 'running') for row in _jobs.values()):
            raise ValueError('Another video is processing. Wait or cancel it first.')
        (JOBS_DIR/job_id).mkdir(parents=True)
        _jobs[job_id] = dict(source, id=job_id, state='queued', done=0, total=1, phase='face',
                            elapsed_seconds=0, take_path='', error='', parent_job_id=body_job)
        _cancels[job_id] = threading.Event()
        _update(job_id)
        _pool.submit(_process_face, job_id, payload, face_rate)
        return dict(_jobs[job_id])


def start_hands_job(body_job, side='BOTH'):
    if side not in ('L', 'R', 'BOTH'):
        raise ValueError('Invalid WiLoR hand selection')
    source = get_job(body_job)
    if source['state'] != 'complete':
        raise ValueError('Finish body recognition first')
    payload = json.loads((JOBS_DIR/body_job/'take.json').read_text(encoding='utf8'))
    if not Path(payload['video']).is_file():
        raise ValueError('The original video is needed')
    job_id = uuid.uuid4().hex
    with _guard:
        if any(row['state'] in ('queued', 'running') for row in _jobs.values()):
            raise ValueError('Another video is processing. Wait or cancel it first.')
        (JOBS_DIR/job_id).mkdir(parents=True)
        _jobs[job_id] = dict(source, id=job_id, state='queued', done=0, total=len(payload['frames']),
                            phase='hands', elapsed_seconds=0, take_path='', error='', parent_job_id=body_job)
        _cancels[job_id] = threading.Event()
        _update(job_id)
        _pool.submit(_process_hands, job_id, payload, side)
        return dict(_jobs[job_id])


def _process_hands(job_id, payload, side):
    clock = time.monotonic()
    try:
        _update(job_id, state='running')
        runtime.unload_model()
        payload['job_id'] = job_id
        payload = run_hands(payload, JOBS_DIR/job_id, side, _cancels[job_id],
            lambda **values: _update(job_id, **values, elapsed_seconds=round(time.monotonic()-clock, 2)))
        take = JOBS_DIR/job_id/'take.json'
        take.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf8')
        count = len(payload['frames'])
        _update(job_id, state='complete', done=count, total=count, take_path=str(take),
                elapsed_seconds=round(time.monotonic()-clock, 2))
    except FaceCancelled:
        _update(job_id, state='cancelled')
    except Exception as error:
        import traceback
        traceback.print_exc()
        _update(job_id, state='error', error=str(error), elapsed_seconds=round(time.monotonic()-clock, 2))


def _process_face(job_id, payload, face_rate):
    clock = time.monotonic()
    try:
        _update(job_id, state='running')
        payload['job_id'] = job_id
        _attach_face(job_id, payload, face_rate, clock)
        take = JOBS_DIR/job_id/'take.json'
        take.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf8')
        count = len(payload['face']['frames'])
        _update(job_id, state='complete', done=count, total=count,
                take_path=str(take), elapsed_seconds=round(time.monotonic()-clock, 2))
    except FaceCancelled:
        _update(job_id, state='cancelled')
    except Exception as error:
        import traceback
        traceback.print_exc()
        _update(job_id, state='error', error=str(error), elapsed_seconds=round(time.monotonic()-clock, 2))


def _process(job_id, path, indices, fps, face=False, face_rate=30, alignment=False, hands='OFF'):
    folder = JOBS_DIR / job_id
    clock = time.monotonic()
    capture = open_video(path)
    reader = SequentialFrames(capture)
    frames = []
    try:
        _update(job_id, state="running")
        with runtime._lock:
            estimator = runtime.get_estimator()
            if not hasattr(estimator, "skeleton_reference"):
                raise RuntimeError("This bridge requires the deployed public BF16 backend.")
            reference = estimator.skeleton_reference()
            for index in indices:
                if _cancels[job_id].is_set():
                    _update(job_id, state="cancelled")
                    return
                ok, bgr = reader.read(index)
                if not ok:
                    raise RuntimeError(f"Cannot decode source frame {index}.")
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                person = estimator.process_one_image(image, inference_type="full")[0]
                positions = person["pred_joint_coords"] @ OUTPUT_TO_BLENDER.T
                rotations = RAW_TO_BLENDER @ person["pred_global_rots"] @ RAW_TO_BLENDER.T
                keypoints = person["pred_keypoints_3d"] @ OUTPUT_TO_BLENDER.T
                frames.append({"source_frame": index, "time": index / fps,
                               "image_points": person['pred_keypoints_2d'].tolist(),
                               "positions": positions.tolist(), "rotations": rotations.tolist(),
                               "keypoints": keypoints.tolist()})
                if len(frames) == 1:
                    overlay = runtime.overlay_skeleton(image, person["pred_keypoints_2d"])
                    ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    if ok:
                        png.tofile(str(folder / "first_frame.png"))
                _update(job_id, done=len(frames), elapsed_seconds=round(time.monotonic() - clock, 2))
            if _cancels[job_id].is_set():
                _update(job_id, state="cancelled")
                return
            payload = {"protocol": PROTOCOL, "version": VERSION, "job_id": job_id,
                       "video": str(path), "fps": fps, "sample_fps": get_job(job_id)["sample_fps"],
                       "coordinate_system": "Blender model space: X-right, Z-up, -Y-forward; meters",
                       "positions_include_camera_translation": False,
                       "video_orientation": {"metadata_clockwise_degrees": capture.get(cv2.CAP_PROP_ORIENTATION_META),
                                             "display_orientation_applied": True,
                                             "decoded_height_width": list(image.shape[:2])},
                       "reference": {"positions": (reference["positions_raw"] @ RAW_TO_BLENDER.T).tolist(),
                                     "rotations": (RAW_TO_BLENDER @ reference["rotations_raw"] @ RAW_TO_BLENDER.T).tolist(),
                                     "parents": reference["parents"].tolist()},
                       "load_report": estimator.load_report, "frames": frames,
                       "limitations": ["single-person", "no facial tracking", "nominal video FPS timing",
                                       "no temporal model", "no inferred camera or world movement"]}
        if hands != 'OFF':
            runtime.unload_model()
            _update(job_id, phase='hands', done=0, total=len(frames))
            payload = run_hands(payload, folder, hands, _cancels[job_id],
                lambda **values: _update(job_id, **values, elapsed_seconds=round(time.monotonic()-clock, 2)))
        if face:
            _attach_face(job_id, payload, face_rate, clock)
        if alignment:
            runtime.unload_model()
            _update(job_id, phase='alignment', done=0, total=len(frames))
            payload['screen'] = run_alignment(payload, folder, _cancels[job_id],
                lambda **values: _update(job_id, **values, elapsed_seconds=round(time.monotonic()-clock, 2)))
        take = folder / "take.json"
        take.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf8")
        count = len(payload['face']['frames']) if face else len(frames)
        _update(job_id, state="complete", done=count, total=count,
                take_path=str(take), elapsed_seconds=round(time.monotonic() - clock, 2))
    except FaceCancelled:
        _update(job_id, state='cancelled')
    except Exception as error:
        import traceback
        traceback.print_exc()
        _update(job_id, state="error", error=f"{type(error).__name__}: {error}", elapsed_seconds=round(time.monotonic() - clock, 2))
    finally:
        capture.release()
        # Leave VRAM available for Blender material preview after extraction.
        runtime.unload_model()


def mount_api(api):
    from fastapi import HTTPException, Request

    @api.post("/sam3d/jobs")
    async def create(request: Request):
        if request.headers.get("x-sam3d-client") != "blender":
            raise HTTPException(403, "Local Blender client header required.")
        try:
            data = await request.json()
            return start_job(data["video"], data.get("rate", 10), data.get("start", 0), data.get("end", 0),
                             data.get('face', False), data.get('face_rate', 30), data.get('alignment', False),
                             data.get('hands', 'OFF'))
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(400, str(error)) from None

    @api.get("/sam3d/jobs/{job_id}")
    def status(job_id: str):
        try:
            return get_job(job_id)
        except (ValueError, KeyError) as error:
            raise HTTPException(404, str(error)) from None

    @api.post('/sam3d/jobs/{job_id}/face')
    async def add_face(job_id: str, request: Request):
        if request.headers.get('x-sam3d-client') != 'blender':
            raise HTTPException(403, 'Local Blender client header required.')
        try:
            data = await request.json()
            return start_face_job(job_id, data.get('face_rate', 30))
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(400, str(error)) from None

    @api.post('/sam3d/jobs/{job_id}/hands')
    async def add_hands(job_id: str, request: Request):
        if request.headers.get('x-sam3d-client') != 'blender':
            raise HTTPException(403, 'Local Blender client header required.')
        try:
            data = await request.json()
            return start_hands_job(job_id, data.get('side', 'BOTH'))
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(400, str(error)) from None

    @api.post("/sam3d/jobs/{job_id}/cancel")
    def cancel(job_id: str, request: Request):
        if request.headers.get("x-sam3d-client") != "blender":
            raise HTTPException(403, "Local Blender client header required.")
        return cancel_job(job_id)
