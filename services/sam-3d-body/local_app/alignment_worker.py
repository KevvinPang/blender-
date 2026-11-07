"""Observe face-relative palms in the upright video, independently of SAM's pose."""
import argparse
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (FaceLandmarker, FaceLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions, RunningMode)

from video_input import open_video, SequentialFrames
from sam_arm_evidence import observe_forearm,PoseArmObserver


def extract(body, output):
    capture = open_video(body['video'])
    reader = SequentialFrames(capture)
    samples = []
    pose_samples=[]
    try:
        models = Path(__file__).parent/'models'
        face_options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(models/'face_landmarker.task')),
            running_mode=RunningMode.VIDEO, num_faces=1, min_face_detection_confidence=.6,
            output_facial_transformation_matrixes=True)
        hand_options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(models/'hand_landmarker.task')),
            running_mode=RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=.65,
            min_hand_presence_confidence=.65, min_tracking_confidence=.65)
        with FaceLandmarker.create_from_options(face_options) as face_model, \
             HandLandmarker.create_from_options(hand_options) as hands_model, PoseArmObserver() as pose_model:
            for index, row in enumerate(body['frames']):
                ok, bgr = reader.read(row['source_frame'])
                if not ok:
                    raise ValueError('Cannot decode alignment frame')
                h, w = bgr.shape[:2]
                aspect = w/h
                scale = min(1, 1280/max(h, w))
                if scale < 1:
                    bgr = cv2.resize(bgr, None, fx=scale, fy=scale)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = round(row['time']*1000)
                face = face_model.detect_for_video(image, timestamp)
                hands = hands_model.detect_for_video(image, timestamp)
                result = {'time': row['time'], 'hands': {}, 'face': None}
                if face.face_landmarks:
                    pts = face.face_landmarks[0]
                    a = np.mean([[pts[i].x*aspect, pts[i].y] for i in (33, 133)], axis=0)
                    b = np.mean([[pts[i].x*aspect, pts[i].y] for i in (263, 362)], axis=0)
                    width = float(np.linalg.norm(b-a))
                    if width > .015:
                        # Eye distance in xyz reduces the apparent shrink from head yaw.
                        a3 = np.mean([[pts[i].x*aspect, pts[i].y, pts[i].z*aspect]
                                      for i in (33, 133)], axis=0)
                        b3 = np.mean([[pts[i].x*aspect, pts[i].y, pts[i].z*aspect]
                                      for i in (263, 362)], axis=0)
                        result['face'] = {'center': ((a+b)*.5).tolist(), 'width': width,
                                          'scale': float(np.linalg.norm(b3-a3))}
                        result['face']['chin'] = [pts[152].x*aspect, pts[152].y]
                        result['face']['mouth'] = np.mean([[pts[i].x*aspect, pts[i].y]
                                                          for i in (13, 14)], axis=0).tolist()
                        result['face']['jaw'] = [[pts[i].x*aspect, pts[i].y]
                                                for i in (172, 136, 150, 149, 176, 148, 152,
                                                          377, 400, 378, 379, 365, 397)]
                        if face.facial_transformation_matrixes:
                            # Strip scale/shear: store a proper camera-space rotation.
                            matrix = np.asarray(face.facial_transformation_matrixes[0])[:3, :3]
                            u, _, vt = np.linalg.svd(matrix)
                            rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
                            if np.isfinite(rotation).all():
                                result['face']['rotation'] = rotation.tolist()
                # Anatomical labels from selfie detectors depend on mirroring. Match
                # observed wrists to SAM's anatomical screen wrists instead.
                candidates = []
                for hand, label in zip(hands.hand_landmarks, hands.handedness):
                    p = np.array([[v.x*aspect, v.y] for v in hand])
                    if not np.isfinite(p).all():
                        continue
                    inside = ((p[:, 0] >= 0) & (p[:, 0] <= aspect)
                              & (p[:, 1] >= 0) & (p[:, 1] <= 1))
                    # Do not reject the entire hand when fingers cross the frame edge.
                    # This is an in-frame mask, not per-joint detection confidence.
                    if inside.sum() < 4:
                        continue
                    candidates.append({'palm': p[[0, 5, 9, 13, 17]].mean(axis=0).tolist(),
                                       'wrist': p[0].tolist(), 'span': float(np.linalg.norm(p[5]-p[17])),
                                       'confidence': float(label[0].score),
                                       'handedness_score': float(label[0].score),
                                       'confidence_kind': 'handedness_classification',
                                       'localization_confidence': None,
                                       'points': p.tolist(), 'in_frame': inside.tolist()})
                points = row.get('image_points')
                if points:
                    pairs = []
                    for j, hand in enumerate(candidates):
                        for side, k in (('L', 62), ('R', 41)):
                            expected = np.array([points[k][0]/h, points[k][1]/h])
                            distance = float(np.linalg.norm(np.array(hand['wrist'])-expected))
                            pairs.append((distance, j, side))
                    used = set()
                    for distance, j, side in sorted(pairs):
                        if j in used or side in result['hands'] or distance > .18:
                            continue
                        result['hands'][side] = candidates[j]
                        result['hands'][side]['identity_distance_height'] = distance
                        used.add(j)
                result['arms']={}
                for side,hand in result['hands'].items():
                    observed=observe_forearm(bgr,hand)
                    if observed:result['arms'][side]=observed
                pose_samples.append(pose_model.observe(bgr,result['hands']))
                samples.append(result)
                Path(output).with_name('alignment_progress.json').write_text(
                    json.dumps({'done': index+1, 'total': len(body['frames'])}), encoding='utf8')
        payload = {'protocol': 'ELLEN_SCREEN_OBSERVATIONS', 'version': 1, 'features_version': 7,
                   'arm_evidence_version':2,'pose_arms':pose_samples,
                   'aspect': aspect,
                   'video': body['video'], 'fps': body['fps'], 'frames': samples,
                   'observation_semantics': {'confidence':'handedness classification, not localization accuracy',
                       'in_frame':'coordinate bounds only, not visibility',
                       'body_image_points':'projected SAM prediction, not independent 2D detection'},
                   'limitations': ['monocular depth is approximate', 'no invisible-hand reconstruction']}
        from sam_punch_evidence import extract as punch_evidence
        payload['punch_tracking'] = punch_evidence(body, payload)
        Path(output).write_text(json.dumps(payload, allow_nan=False), encoding='utf8')
        print(json.dumps({'samples': len(samples), 'visible_L': sum('L' in r['hands'] for r in samples),
                          'visible_R': sum('R' in r['hands'] for r in samples)}), flush=True)
    finally:
        capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--body', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    extract(json.loads(Path(args.body).read_text(encoding='utf8')), args.output)
