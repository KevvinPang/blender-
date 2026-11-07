"""Opt-in bounded finger repair with SAM corroboration and source-image veto."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from enhance_sam_take_with_wilor import FINGERS, match_hands, mhr_indices, palm_basis
from wilor_temporal import VideoEvidence, arc, bends, segment_basis

VERSION = 1
BACKUP = '_finger_reliability_original'
REPORT = 'finger_reliability'


def restore_experiment(data):
    result = copy.deepcopy(data)
    for item in result.pop(BACKUP, []):
        for k, value in item['positions'].items():
            result['frames'][item['row']]['positions'][int(k)] = value
    result.pop(REPORT, None)
    return result


def observations(data, records, side):
    ids = mhr_indices(side)
    lookup = {r['frame']: r for r in records}
    directions, palms, hands, sam = [], [], [], []
    original = data.get('wilor_original', [])
    for i, row in enumerate(data['frames']):
        record = lookup.get(row['source_frame'], {})
        hand = match_hands(record.get('hands', []), row, record.get('height', 1280)).get(side)
        try:
            if hand is None:
                raise ValueError('Missing hand')
            positions = np.asarray(row['positions'], dtype=float)
            palm = palm_basis(positions[[ids['wrist'], *[ids[f][0] for f in FINGERS[1:]]]])
            d = np.asarray([np.diff(positions[ids[f]], axis=0) for f in FINGERS])
            lengths = np.linalg.norm(d, axis=-1, keepdims=True)
            if not np.isfinite(d).all() or np.min(lengths) < 1e-7:
                raise ValueError('Invalid geometry')
            source = original[i]['positions']
            s = np.asarray([np.diff([source[str(k)] for k in ids[f]], axis=0) for f in FINGERS])
            if not np.isfinite(s).all() or np.min(np.linalg.norm(s, axis=-1)) < 1e-7:
                raise ValueError('Invalid original SAM')
            directions.append(d/lengths)
            palms.append(palm)
            hands.append(hand)
            sam.append(bends(s))
        except (ValueError, KeyError, IndexError, np.linalg.LinAlgError):
            directions.append(None)
            palms.append(None)
            hands.append(None)
            sam.append(None)
    return directions, palms, hands, sam


def find_candidates(values, sam, times, max_seconds=1.2):
    """Find bounded pose disagreements, without assigning gesture identities."""
    current = [bends(v) if v is not None else None for v in values]
    result = []
    if len(times) < 7:
        return result
    for f in range(5):
        for state in ('straight', 'curled'):
            # The target is the observed pose on both sides, never a preset fist.
            wrong = lambda v: v is not None and (np.max(v[f]) <= 30 if state == 'straight'
                                                  else np.max(v[f]) >= 55)
            target = lambda v: v is not None and (np.max(v[f]) >= 40 if state == 'straight'
                                                   else np.max(v[f]) <= 30)
            supported = lambda v: v is not None and (np.max(v[f]) >= 55 if state == 'straight'
                                                       else np.max(v[f]) <= 30)
            i = 0
            while i < len(times):
                if not wrong(current[i]):
                    i += 1
                    continue
                start = i
                while i+1 < len(times) and wrong(current[i+1]) and 0 < times[i+1]-times[i] <= .12:
                    i += 1
                end = i
                i += 1
                a, b = start-1, end+1
                if a < 2 or b+2 >= len(times) or times[b]-times[a] > max_seconds:
                    continue
                if times[end]-times[start] < .12 or np.max(np.diff(times[a:b+1])) > .12:
                    continue
                if any(values[j] is None or sam[j] is None for j in range(a, b+1)):
                    continue
                if not all(target(current[j]) for j in [a-2, a-1, a, b, b+1, b+2]):
                    continue
                if max(np.ptp(np.asarray([current[j][f] for j in window]), axis=0).max()
                       for window in (range(a-2, a+1), range(b, b+3))) > 30:
                    continue
                if np.max(abs(current[a][f]-current[b][f])) > 35:
                    continue
                if any(np.max(abs(current[j][f]-sam[j][f])) > 35 for j in (a, b)):
                    continue
                support = sum(supported(sam[j]) for j in range(start, end+1))/(end-start+1)
                if support < .75:
                    continue
                result.append({'finger': f, 'left': a, 'right': b, 'start': start, 'end': end,
                               'sam_support_fraction': support})
    return sorted(result, key=lambda r: (r['left'], r['finger']))


def repair_take(data, records, side='L', enabled=False, evidence_factory=VideoEvidence):
    result = restore_experiment(data)
    if not enabled:
        return result
    frames = result['frames']
    times = np.asarray([r['time'] for r in frames], dtype=float)
    if len(times) == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid capture timestamps')
    report = {'version': VERSION, 'enabled': True, 'max_seconds': 1.2,
              'method': 'SAM-corroborated / bounded endpoints / source-image veto / distal-only',
              'accepted': [], 'skipped': [], 'changed_joint_samples': 0}
    result[REPORT] = report
    if not result.get('wilor_original'):
        report['warning'] = 'Original SAM hand evidence missing; experimental repair skipped'
        return result
    video = result.get('video')
    if not video or not Path(video).is_file():
        report['warning'] = 'Source video missing; experimental repair skipped'
        return result
    source_frames = [r['source_frame'] for r in frames]
    backup = {}
    for s in (('L', 'R') if side == 'BOTH' else (side,)):
        values, palms, hands, sam = observations(result, records, s)
        candidates = find_candidates(values, sam, times)
        if not candidates:
            continue
        evidence = evidence_factory(video, source_frames)
        try:
            for c in candidates:
                a, b, f = c['left'], c['right'], c['finger']
                details = {**c, 'side': s, 'finger': FINGERS[f],
                           'source_start': source_frames[c['start']], 'source_end': source_frames[c['end']]}
                try:
                    check = evidence.check(c, hands)
                    details.update(check)
                    if not check['accepted']:
                        report['skipped'].append(details)
                        continue
                    left, right = values[a][f]@palms[a], values[b][f]@palms[b]
                    la, rb = segment_basis(left[0]), segment_basis(right[0])
                    if la is None or rb is None:
                        raise ValueError('Degenerate endpoint basis')
                    left, right = left[1:]@la, right[1:]@rb
                    edits = []
                    ids = mhr_indices(s)[FINGERS[f]]
                    for j in range(c['start'], c['end']+1):
                        basis = segment_basis((values[j][f]@palms[j])[0])
                        if basis is None:
                            raise ValueError('Degenerate proximal basis')
                        t = (times[j]-times[a])/(times[b]-times[a])
                        distal = arc(left, right, t)@basis.T@palms[j].T
                        positions = np.asarray(frames[j]['positions'])[ids].copy()
                        lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                        positions[2] = positions[1]+distal[0]*lengths[1]
                        positions[3] = positions[2]+distal[1]*lengths[2]
                        edits.append((j, positions[2:].tolist()))
                    for j, positions in edits:
                        saved = backup.setdefault(j, {})
                        for k, value in zip(ids[2:], positions):
                            saved.setdefault(str(k), copy.deepcopy(frames[j]['positions'][k]))
                            frames[j]['positions'][k] = value
                            report['changed_joint_samples'] += 1
                    report['accepted'].append(details)
                except (ValueError, cv2.error, IndexError) as error:
                    details.update(accepted=False, reason=str(error))
                    report['skipped'].append(details)
        finally:
            evidence.close()
    if backup:
        result[BACKUP] = [{'row': j, 'positions': value} for j, value in sorted(backup.items())]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--take', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--side', choices=['L', 'R', 'BOTH'], default='L')
    args = parser.parse_args()
    source, predictions, output = map(lambda p: Path(p).resolve(), (args.take, args.predictions, args.output))
    if output in (source, predictions):
        raise ValueError('Experimental output must not overwrite input')
    data, raw = (json.loads(p.read_text(encoding='utf8')) for p in (source, predictions))
    if Path(data['video']).resolve() != Path(raw['video']).resolve():
        raise ValueError('WiLoR cache belongs to another video')
    result = repair_take(data, raw['frames'], args.side, enabled=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf8')
    print(json.dumps(result[REPORT], ensure_ascii=True))
