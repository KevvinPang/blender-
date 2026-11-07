"""WiLoR palm orientation and fingers, preserving wrist locations and body."""
from __future__ import annotations
from mocap_paths import LOCAL_APP

import argparse
import copy
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def mhr_indices(side):
    offset = 0 if side == 'R' else 36
    return {'wrist': 42+offset,
            **{name: list(range(first+offset, first+offset+4)) for name, first in
               [('thumb', 61), ('index', 56), ('middle', 52), ('ring', 48), ('pinky', 44)]}}


FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
CAMERA_TO_BLENDER = np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]])


def palm_basis(p):
    if not np.isfinite(p).all():
        raise ValueError('Invalid palm')
    forward = p[2]-p[0]
    across = p[1]-p[4]
    y = forward / max(np.linalg.norm(forward), 1e-9)
    x = across - np.dot(across, y)*y
    if np.linalg.norm(x) < 1e-7 or np.linalg.norm(forward) < 1e-7:
        raise ValueError('Degenerate palm')
    x /= np.linalg.norm(x)
    return np.stack((x, y, np.cross(x, y)), axis=1)


def palm_rotation(source, target):
    """Proper rotation from two independent palm axes; reject collapsed palms."""
    return palm_basis(target) @ palm_basis(source).T


def blend_rotation(a, b, weight):
    return Slerp([0, 1], Rotation.from_matrix(np.stack((a, b))))([weight]).as_matrix()[0]


def refine_palms(frames, records, side, confidence, fps):
    """WiLoR is in camera XYZ; never align its palm back to SAM's palm."""
    indices = mhr_indices(side)
    bases = [indices['wrist'], *[indices[f][0] for f in FINGERS[1:]]]
    hand_indices = sorted({k for f in FINGERS for k in indices[f]} | {indices['wrist']})
    lookup = {int(r['frame']): r for r in records}
    times = [r['time'] for r in frames]
    observed = []
    for row in frames:
        source_frame = int(row.get('source_frame', round(row['time']*fps)))
        record = lookup.get(source_frame, {})
        candidates = [h for h in record.get('hands', []) if h['confidence'] >= confidence]
        hand = match_hands(candidates, row, record.get('height', 1280)).get(side)
        try:
            joints = np.asarray(hand['joints3d']) @ CAMERA_TO_BLENDER.T if hand else None
            observed.append(palm_basis(joints[[0, 5, 9, 13, 17]]) if hand else None)
        except (ValueError, IndexError, np.linalg.LinAlgError):
            observed.append(None)
    # Reject isolated flips, while retaining turns supported by adjacent samples.
    raw = list(observed)
    for i in range(1, len(raw)-1):
        if any(raw[j] is None for j in (i-1, i, i+1)):
            continue
        angle = lambda a, b: (Rotation.from_matrix(a).inv()*Rotation.from_matrix(b)).magnitude()
        if (times[i+1]-times[i-1] < .15 and angle(raw[i-1], raw[i+1]) < .35
                and min(angle(raw[i], raw[j]) for j in (i-1, i+1)) > 1.7):
            observed[i] = None
    valid = [i for i, p in enumerate(observed) if p is not None]
    if len(valid) < 2:
        return {'detected_samples': len(valid), 'applied_samples': 0, 'short_gap_filled': 0}
    filled = list(observed)
    for a, b in zip(valid, valid[1:]):
        if b > a+1 and times[b]-times[a] <= .17:
            for i in range(a+1, b):
                filled[i] = blend_rotation(observed[a], observed[b], (times[i]-times[a])/(times[b]-times[a]))
    missing = [times[i] for i, p in enumerate(filled) if p is None]
    previous, last = None, None
    applied = []
    for i, (row, target) in enumerate(zip(frames, filled)):
        if target is None:
            previous = None
            continue
        if previous is not None and times[i]-last < .17:
            target = blend_rotation(previous, target, 1-np.exp(-(times[i]-last)/.035))
        previous, last = target, times[i]
        positions = np.asarray(row['positions'], dtype=float)
        current = palm_basis(positions[bases])
        weight = min(1., min(abs(times[i]-t) for t in missing)/.15) if missing else 1.
        delta = blend_rotation(np.eye(3), target @ current.T, weight)
        pivot = positions[indices['wrist']].copy()
        for k in hand_indices:
            if k != indices['wrist']:
                row['positions'][k] = (pivot+delta@(positions[k]-pivot)).tolist()
            row['rotations'][k] = (delta@np.asarray(row['rotations'][k])).tolist()
        applied.append(i)
    return {'detected_samples': len(valid), 'applied_samples': len(applied),
            'short_gap_filled': sum(a is None and b is not None for a, b in zip(observed, filled))}


def match_hands(predictions, row, height):
    """Use anatomical SAM screen wrists when available, not selfie handedness."""
    points = row.get('image_points')
    if points is None:
        return {side: max((p for p in predictions if p['right'] == (side == 'R')),
                         key=lambda p: p['confidence'], default=None) for side in ('L', 'R')}
    pairs = []
    for i, hand in enumerate(predictions):
        for side, k in (('L', 62), ('R', 41)):
            distance = np.linalg.norm(np.asarray(hand['joints2d'][0])-points[k][:2])/height
            if distance <= .18:
                pairs.append((distance, i, side))
    found, used = {}, set()
    for _, i, side in sorted(pairs):
        if i not in used and side not in found:
            found[side] = predictions[i]
            used.add(i)
    return found


def local_directions(hand, row, side):
    """Express finger directions in the current SAM wrist's coordinate frame."""
    indices = mhr_indices(side)
    joints = np.asarray(hand['joints3d'], dtype=float)
    positions = np.asarray(row['positions'])
    bases = [indices['wrist'], *[indices[f][0] for f in FINGERS[1:]]]
    rotation = palm_rotation(joints[[0, 5, 9, 13, 17]], positions[bases])
    wrist = np.asarray(row['rotations'][indices['wrist']])
    directions = []
    for start in (1, 5, 9, 13, 17):
        delta = np.diff(joints[start:start+4], axis=0)
        lengths = np.linalg.norm(delta, axis=1, keepdims=True)
        if not np.isfinite(delta).all() or np.min(lengths) < 1e-6:
            raise ValueError('Invalid finger geometry')
        directions.append((delta/lengths) @ rotation.T @ wrist)
    return np.asarray(directions)


def fill_short_gaps(values, times, limit=.31):
    """Interpolate short bounded gaps in wrist space, hold short edge gaps."""
    filled = list(values)
    valid = [i for i, v in enumerate(values) if v is not None]
    for a, b in zip(valid, valid[1:]):
        if b > a+1 and times[b]-times[a] <= limit:
            for i in range(a+1, b):
                t = (times[i]-times[a])/(times[b]-times[a])
                filled[i] = values[a]*(1-t)+values[b]*t
    if valid:
        for i in range(valid[0]):
            if times[valid[0]]-times[i] <= .15:
                filled[i] = values[valid[0]]
        for i in range(valid[-1]+1, len(values)):
            if times[i]-times[valid[-1]] <= .15:
                filled[i] = values[valid[-1]]
    return filled


def apply_predictions(take, records, side='BOTH', confidence=.65, temporal=True):
    if take.get('protocol') != 'SAM3D_BLENDER_TAKE':
        raise ValueError('Not a SAM3D Blender take')
    result = copy.deepcopy(take)
    if result.get('wilor_original'):
        for row, original in zip(result['frames'], result['wilor_original']):
            for prop in ('positions', 'rotations'):
                for k, value in original[prop].items():
                    row[prop][int(k)] = value
    result.pop('wilor_original', None)
    frames = result['frames']
    times = [r['time'] for r in frames]
    lookup = {int(r['frame']): r for r in records}
    sides = ('L', 'R') if side == 'BOTH' else (side,)
    all_indices = sorted({k for s in sides for f in FINGERS for k in mhr_indices(s)[f]}
                         | {mhr_indices(s)['wrist'] for s in sides})
    result['wilor_original'] = [
        {prop: {str(k): copy.deepcopy(row[prop][k]) for k in all_indices}
         for prop in ('positions', 'rotations')} for row in frames]
    report = {'model': 'WiLoR', 'side': side, 'detector_threshold': confidence,
              'confidence_is_pose_accuracy': False, 'hands': {},
              'transform': 'camera-space WiLoR palms / wrist-local fingers / unchanged bone lengths',
              'palm_orientation_version': 1, 'palms': {}}
    for s in sides:
        report['palms'][s] = refine_palms(frames, records, s, confidence, take['fps'])
        indices = mhr_indices(s)
        observations = []
        observed_hands = []
        palm_frames = []
        detected = []
        for i, row in enumerate(frames):
            source_frame = int(row.get('source_frame', round(row['time']*take['fps'])))
            record = lookup.get(source_frame, {})
            candidates = [p for p in record.get('hands', []) if p['confidence'] >= confidence]
            hand = match_hands(candidates, row, record.get('height', 1280)).get(s)
            try:
                value = local_directions(hand, row, s) if hand else None
            except (ValueError, np.linalg.LinAlgError):
                value = None
            observations.append(value)
            observed_hands.append(dict(hand, _image_height=record.get('height', 1280)) if value is not None else None)
            bases = [indices['wrist'], *[indices[f][0] for f in FINGERS[1:]]]
            palm_frames.append(np.asarray(row['rotations'][indices['wrist']]).T
                               @ palm_basis(np.asarray(row['positions'])[bases]) if value is not None else None)
            if value is not None:
                detected.append(i)
        from wilor_temporal import VERSION
        temporal_report = {'version': VERSION, 'enabled': False, 'accepted': [], 'skipped': []}
        if temporal:
            from wilor_temporal import filter_fingers
            source_frames = [int(r.get('source_frame', round(r['time']*take['fps']))) for r in frames]
            observations, temporal_report = filter_fingers(observations, palm_frames, observed_hands,
                times, source_frames, take.get('video'), image_observations=take.get('screen'), side=s)
        directions = fill_short_gaps(observations, times)
        applied = []
        missing_times = [times[j] for j, v in enumerate(directions) if v is None]
        for i, (row, local) in enumerate(zip(frames, directions)):
            if local is None:
                continue
            weight = min(1., min(abs(times[i]-t) for t in missing_times)/.2) if missing_times else 1.
            positions = np.asarray(row['positions'], dtype=float)
            wrist = np.asarray(row['rotations'][indices['wrist']], dtype=float)
            for f, name in enumerate(FINGERS):
                chain = indices[name]
                original = np.diff(positions[chain].copy(), axis=0)
                lengths = np.linalg.norm(original, axis=1)
                unit = original/np.maximum(lengths[:, None], 1e-9)
                world = local[f] @ wrist.T
                blended = unit*(1-weight)+world*weight
                blended /= np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1e-9)
                for n in range(3):
                    positions[chain[n+1]] = positions[chain[n]]+blended[n]*lengths[n]
                    # Carry the original SAM twist; the Blender solver changes
                    # only the swing to follow the WiLoR segment direction.
            for name in FINGERS:
                for k in indices[name][1:]:
                    row['positions'][k] = positions[k].tolist()
            applied.append(i)
        report['hands'][s] = {'detected_samples': len(detected), 'applied_samples': len(applied),
                              'missing_samples': [i for i in range(len(frames)) if i not in detected],
                              'short_gap_filled': len(applied)-len(detected),
                              'temporal_filter': temporal_report}
    if not any(h['detected_samples'] >= 2 for h in report['hands'].values()):
        # Missing optional fingers must not discard a valid body/face take.
        result['frames'] = copy.deepcopy(take['frames'])
        if take.get('wilor_original'):
            for row, original in zip(result['frames'], take['wilor_original']):
                for prop in ('positions', 'rotations'):
                    for k, value in original[prop].items():
                        row[prop][int(k)] = copy.deepcopy(value)
        report['fallback'] = 'SAM'
        report['warning'] = 'WiLoR 未充分识别所选手部，已保留 SAM 手指；身体与面部继续导入'
        for hand in report['hands'].values():
            hand['applied_samples'] = 0
            hand['short_gap_filled'] = 0
        for palm in report['palms'].values():
            palm['applied_samples'] = 0
    result['wilor_refinement'] = report
    return result


def refine_take(take, video, side, output, predictions='', progress_path='', temporal=True):
    source_video = Path(video or take['video']).resolve()
    if not predictions and not source_video.is_file():
        raise FileNotFoundError(source_video)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if predictions:
        cached = json.loads(Path(predictions).read_text(encoding='utf8'))
        if Path(cached['video']).resolve() != source_video:
            raise ValueError('WiLoR cache belongs to a different video')
        records = cached['frames']
    else:
        from run_wilor_video_test import (WILOR_ROOT, load_wilor, YOLO, infer_frame, torch)
        os.chdir(WILOR_ROOT)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, cfg = load_wilor(checkpoint_path=str(WILOR_ROOT/'pretrained_models/wilor_final.ckpt'),
                                cfg_path=str(WILOR_ROOT/'pretrained_models/model_config.yaml'))
        model = model.to(device).eval()
        detector = YOLO(str(WILOR_ROOT/'pretrained_models/detector.pt')).to(device)
        import sys
        worker_path = LOCAL_APP
        sys.path.insert(0, str(worker_path))
        from video_input import open_video, SequentialFrames
        cap = open_video(source_video)
        reader = SequentialFrames(cap)
        records = []
        try:
            if not cap.isOpened():
                raise RuntimeError('Cannot decode video')
            for i, row in enumerate(take['frames']):
                frame_index = int(row.get('source_frame', round(row['time']*take['fps'])))
                ok, frame = reader.read(frame_index)
                if not ok:
                    raise RuntimeError(f'Cannot decode frame {frame_index}')
                records.append({'frame': frame_index, 'height': frame.shape[0],
                                'hands': infer_frame(model, detector, cfg, frame, device, .25)})
                if progress_path:
                    p = Path(progress_path)
                    temp = p.with_suffix('.tmp')
                    temp.write_text(json.dumps({'done': i+1, 'total': len(take['frames'])}), encoding='utf8')
                    for attempt in range(20):
                        try:
                            temp.replace(p)
                            break
                        except PermissionError:
                            if attempt == 19:
                                raise
                            time.sleep(.01)
        finally:
            cap.release()
        output.with_name('wilor_predictions.json').write_text(
            json.dumps({'video': str(source_video), 'frames': records}, allow_nan=False), encoding='utf8')
    result = apply_predictions(take, records, side, temporal=temporal)
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf8')
    return result['wilor_refinement']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--take', required=True)
    parser.add_argument('--video', default='')
    parser.add_argument('--side', choices=('L', 'R', 'BOTH'), default='BOTH')
    parser.add_argument('--output', required=True)
    parser.add_argument('--predictions', default='')
    parser.add_argument('--progress', default='')
    parser.add_argument('--no-temporal-filter', action='store_true')
    args = parser.parse_args()
    take = json.loads(Path(args.take).read_text(encoding='utf8'))
    print(json.dumps(refine_take(take, args.video, args.side, args.output,
                                args.predictions, args.progress, not args.no_temporal_filter)), flush=True)
