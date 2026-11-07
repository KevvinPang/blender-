"""Local MediaPipe face extraction in its own compatible Python environment."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from video_input import open_video, SequentialFrames

MODEL = Path(__file__).parent / 'models/face_landmarker.task'
MODEL_SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'


def mouth_observation(landmarks, rgb):
    """Measure inner-lip separation in the image, independent of JawOpen score."""
    height, width = rgb.shape[:2]
    def point(index):
        p = landmarks[index]
        return (p.x*width, p.y*height)
    def distance(a, b):
        return math.dist(point(a), point(b))
    span = distance(61, 291)
    eye_span = distance(33, 263)
    if span < 4 or eye_span < 8:
        return None
    # Project the three inner-lip pairs onto the mouth's perpendicular axis;
    # using screen Y alone would underestimate opening when the head tilts.
    left, right = point(61), point(291)
    axis = (-(right[1]-left[1])/span, (right[0]-left[0])/span)
    gaps = []
    for upper, lower in ((82, 87), (13, 14), (312, 317)):
        u, l = point(upper), point(lower)
        gaps.append(abs((l[0]-u[0])*axis[0]+(l[1]-u[1])*axis[1]))
    indices = (61, 291, 13, 14, 82, 87, 312, 317)
    inside = all(0 <= landmarks[i].x <= 1 and 0 <= landmarks[i].y <= 1 for i in indices)
    aperture = sorted(gaps)[1]/span
    center = np.mean([point(13), point(14)], axis=0)
    across = np.array([(right[0]-left[0])/span, (right[1]-left[1])/span])
    xs, ys = np.meshgrid(np.linspace(-.65,.65,97), np.linspace(-.4,.4,65))
    coords = center+span*(xs[...,None]*across+ys[...,None]*np.array(axis))
    patch = cv2.remap(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY), coords[...,0].astype(np.float32),
                      coords[...,1].astype(np.float32), cv2.INTER_LINEAR)
    skin = float(np.percentile(patch[(abs(xs)>.48)&(abs(ys)>.2)],65))
    cavity = (patch < min(skin*.62, skin-24)) & (abs(xs)<.25) & (abs(ys)<min(.36,aperture*.6+.035))
    dark_gap = float(np.median(cavity[:,abs(xs[0])<.20].sum(axis=0))*.8/64)
    return {'aperture': aperture, 'width_to_eyes': span/eye_span,
            'dark_gap': dark_gap,
            'in_frame': inside, 'mouth_pixels': span,
            'points': {str(i): [landmarks[i].x, landmarks[i].y] for i in indices}}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf8')
    for attempt in range(20):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.01)


def extract(video, output, rate=30, start=0, end=0):
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError('Face model checksum does not match the verified download')
    if not all(math.isfinite(v) for v in (rate, start, end)) or not 1 <= rate <= 60 or start < 0:
        raise ValueError('Invalid facial capture interval or sampling rate')
    capture = open_video(video)
    reader = SequentialFrames(capture)
    clock = time.monotonic()
    frames = []
    output = Path(output)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or count < 2:
            raise ValueError('Cannot decode face input video')
        end = min(end or count/fps, count/fps)
        first, last = math.ceil(start*fps), min(count-1, math.ceil(end*fps)-1)
        if first >= last:
            raise ValueError('Face interval is too short')
        indices = sorted(set([round(first+i*fps/min(rate, fps))
                              for i in range(int((last-first)*min(rate, fps)/fps)+1)] + [last]))
        options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
            running_mode=RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
            min_face_detection_confidence=.55, min_face_presence_confidence=.55,
            min_tracking_confidence=.55)
        with FaceLandmarker.create_from_options(options) as model:
            previous_ms = -1
            for i, index in enumerate(indices):
                ok, bgr = reader.read(index)
                if not ok:
                    raise ValueError(f'Cannot decode face source frame {index}')
                # Keep a larger face than the 256-pixel detector input in close-ups.
                scale = min(1, 1280/max(bgr.shape[:2]))
                if scale < 1:
                    bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp = max(previous_ms+1, round(index/fps*1000))
                previous_ms = timestamp
                result = model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                active = bool(result.face_blendshapes and result.face_landmarks)
                weights = {c.category_name: float(max(0, min(1, c.score)))
                           for c in result.face_blendshapes[0]} if active else {}
                observation = mouth_observation(result.face_landmarks[0], rgb) if active else None
                frames.append({'source_frame': index, 'time': index/fps, 'active': active,
                               'blendshapes': weights, 'mouth': observation})
                write_json(output.with_name('face_progress.json'), {'done': i+1, 'total': len(indices)})
        active = sum(row['active'] for row in frames)
        if active == 0:
            raise ValueError('No face detected; previous Blender face animation is preserved')
        payload = {'protocol': 'ELLEN_FACE_TAKE', 'version': 1, 'model': 'MediaPipe Face Landmarker v1 float16',
                   'model_sha256': MODEL_SHA256, 'mouth_features_version': 1,
                   'video': str(Path(video).resolve()),
                   'fps': fps, 'sample_fps': min(rate, fps), 'frames': frames,
                   'detected_samples': active, 'total_samples': len(frames),
                   'elapsed_seconds': round(time.monotonic()-clock, 3),
                   'limitations': ['visual mouth shapes, not phoneme transcription',
                                   'detected face is not proof that an occluded mouth is reliable',
                                   'no inferred emotion labels or audio-based occlusion recovery']}
        write_json(output, payload)
        print(json.dumps({k: payload[k] for k in ('detected_samples', 'total_samples', 'elapsed_seconds')}), flush=True)
        return payload
    finally:
        capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rate', type=float, default=30)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float, default=0)
    args = parser.parse_args()
    extract(args.video, args.output, args.rate, args.start, args.end)
