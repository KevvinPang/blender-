"""Image-verified, bounded thumb repair in a moving proximal-bone frame."""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import cv2


PALM = np.array([5, 9, 13, 17])


def similarity(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    x, y = source-a, target-b
    u, s, vt = np.linalg.svd(x.T@y)
    correction = np.diag([1., np.linalg.det(u@vt)])
    rotation = u@correction@vt
    scale = float((s*np.diag(correction)).sum()/max(1e-9, np.sum(x*x)))
    return rotation*scale, b-a@rotation*scale


def independent_points(video, source_frames, hands, cached, side, times):
    """Use matching cached image observations, or run the bundled hand detector."""
    points = [None]*len(hands)
    if cached and side and Path(cached.get('video', '')).resolve() == Path(video).resolve():
        rows = cached.get('frames', [])
        if len(rows) == len(hands) and all(abs(row.get('time', -100)-t) < 1e-5 for row,t in zip(rows,times)):
            for i, (row, hand) in enumerate(zip(rows, hands)):
                h = row.get('hands', {}).get(side)
                if hand is None or not h or h.get('confidence', 0) < .65:
                    continue
                p = np.asarray(h.get('points', []), dtype=float)
                mask = np.asarray(h.get('in_frame', [True]*21), dtype=bool)
                if p.shape == (21, 2) and mask.shape == (21,) and np.isfinite(p).all():
                    # WiLoR coordinates use decoded video pixels.
                    height = hand['_image_height']
                    points[i] = (p*height, mask)
            if any(p is not None for p in points):
                return points, 'cached-independent-hand-landmarks'
    # WiLoR's environment need not contain MediaPipe. The existing camera worker
    # runtime owns that dependency and the local model; no model download occurs.
    root = Path(__file__).resolve().parent
    python = root/'motion_venv/Scripts/python.exe'
    worker = root/'wilor_thumb_landmarks.py'
    if not python.is_file() or not worker.is_file():
        return points, 'independent-detector-unavailable'
    with tempfile.TemporaryDirectory(prefix='thumb_evidence_') as folder:
        request, response = Path(folder)/'input.json', Path(folder)/'output.json'
        request.write_text(json.dumps({'video':str(video), 'source_frames':source_frames,
            'hands':[{'palm':np.asarray(h['joints2d'])[PALM].tolist(), 'height':h['_image_height']}
                     if h is not None else None for h in hands]}), encoding='utf8')
        try:
            result = subprocess.run([str(python), str(worker), str(request), str(response)],
                capture_output=True, timeout=120, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (OSError, subprocess.TimeoutExpired):
            return points, 'independent-detector-unavailable'
        if result.returncode or not response.is_file():
            return points, 'independent-detector-failed'
        raw = json.loads(response.read_text(encoding='utf8'))
        for i, item in enumerate(raw[:len(points)]):
            if item:
                p, mask = np.asarray(item['points'], dtype=float), np.asarray(item['mask'], dtype=bool)
                if p.shape == (21, 2) and mask.shape == (21,) and np.isfinite(p).all():
                    points[i] = (p, mask)
    return points, 'video-independent-hand-landmarks'


def consistency(hands, independent):
    """Compare thumb shape after aligning independent, non-thumb palm anchors."""
    scores = [None]*len(hands)
    for i, (hand, observed) in enumerate(zip(hands, independent)):
        if hand is None or observed is None:
            continue
        target, mask = observed
        if not np.all(mask[[2, 3, 4]]) or np.count_nonzero(mask[PALM]) < 3:
            continue
        source = np.asarray(hand['joints2d'], dtype=float)
        anchors = PALM[mask[PALM]]
        span = float(np.linalg.norm(target[5]-target[17]))
        if span < 18 or not np.isfinite(target).all():
            continue
        transform, offset = similarity(source[anchors], target[anchors])
        scale = float(np.sqrt(np.linalg.det(transform)))
        if not .5 < scale < 2.:
            continue
        aligned = source@transform+offset
        palm_error = float(np.mean(np.linalg.norm(aligned[anchors]-target[anchors], axis=1))/span)
        if palm_error > .12:
            continue
        residual = np.linalg.norm(aligned[2:5]-target[2:5], axis=1)/span
        scores[i] = {'tip':float(residual[-1]), 'distal':float(np.mean(residual[1:])),
                     'palm':palm_error, 'span':span}
    return scores


def bounded_candidates(scores, observations, times):
    trusted = [i for i, s in enumerate(scores) if s is not None and s['tip'] < .35 and s['distal'] < .35]
    found = []
    for a, b in zip(trusted, trusted[1:]):
        if b-a < 2 or times[b]-times[a] > .45:
            continue
        if any(scores[i] is None or observations[i] is None for i in range(a, b+1)):
            continue
        if max(np.diff(times[a:b+1])) > .12:
            continue
        if not any(scores[i]['tip'] > .45 and scores[i]['distal'] > .30 for i in range(a+1, b)):
            continue
        found.append({'finger':0, 'left':a, 'right':b, 'start':a+1, 'end':b-1})
    return found


def independent_motion_check(candidate, independent, times):
    """A real articulation departing from the bounded path vetoes replacement."""
    a, b = candidate['left'], candidate['right']
    errors = []
    for i in range(a+1, b):
        current = independent[i][0]
        span = np.linalg.norm(current[5]-current[17])
        expected = []
        for j in (a, b):
            other = independent[j][0]
            matrix, offset = similarity(other[PALM], current[PALM])
            expected.append(other[2:5]@matrix+offset)
        weight = (times[i]-times[a])/(times[b]-times[a])
        estimate = expected[0]*(1-weight)+expected[1]*weight
        errors.append(float(np.max(np.linalg.norm(estimate-current[2:5], axis=1))/span))
    return max(errors, default=0.)


def aligned_photo_checks(candidate, independent, hands, evidence):
    """Verify each suspect against real pixels in the independent palm frame."""
    a,b=candidate['left'],candidate['right']
    images=evidence.frames(a,b)
    predictions=[]
    for i in range(a,b+1):
        raw=np.asarray(hands[i]['joints2d'])
        target=independent[i][0]
        matrix,offset=similarity(raw[PALM],target[PALM])
        predictions.append(raw@matrix+offset)
    predictions=np.asarray(predictions)
    predictions[0]=independent[a][0]
    span=float(np.linalg.norm(predictions[0,5]-predictions[0,17]))
    return {i:evidence.appearance_check(dict(candidate,start=i,end=i),images,predictions,[2,3,4],span)
            for i in range(candidate['start'],candidate['end']+1)}


def repair_thumb(observations, palm_frames, hands, times, source_frames, video,
                 cached=None, side=None, evidence_factory=None):
    from wilor_temporal import VideoEvidence, segment_basis, arc, unit
    report = {'version':1, 'max_gap_seconds':.45, 'accepted':[], 'skipped':[], 'image_verified':True}
    output = [v.copy() if v is not None else None for v in observations]
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        report['reason'] = 'invalid_sample_times'
        return output, report
    # Do not load an independent model for a clip without appreciable thumb changes.
    changes = []
    for i in range(1, len(times)):
        if observations[i-1] is not None and observations[i] is not None and times[i]-times[i-1] <= .12:
            a = observations[i-1][0]@palm_frames[i-1]
            b = observations[i][0]@palm_frames[i]
            changes.append(np.degrees(np.arccos(np.clip(np.sum(a*b, axis=1), -1, 1))).max())
    if not changes or max(changes) < 25 or not video or not Path(video).is_file():
        report['reason'] = 'no_large_thumb_change_or_source_unavailable'
        return output, report
    independent, method = independent_points(video, source_frames, hands, cached, side, times)
    report['landmark_source'] = method
    scores = consistency(hands, independent)
    report['independent_samples'] = sum(s is not None for s in scores)
    found = bounded_candidates(scores, observations, times)
    if not found:
        return output, report
    evidence = (evidence_factory or VideoEvidence)(video, source_frames)
    try:
        for c in found:
            a, b = c['left'], c['right']
            detail = {**c, 'finger':'thumb', 'source_start':int(source_frames[c['start']]),
                      'source_end':int(source_frames[c['end']]), 'start_seconds':float(times[c['start']]),
                      'end_seconds':float(times[c['end']])}
            try:
                motion = independent_motion_check(c, independent, times)
                detail['independent_motion_error_palm_spans'] = motion
                check = evidence.check(c, hands)
                detail.update(check)
                accepted_indices=list(range(a+1,b)) if check['accepted'] else []
                if not accepted_indices:
                    per_frame=aligned_photo_checks(c,independent,hands,evidence)
                    accepted_indices=[i for i,check_i in per_frame.items() if check_i['accepted']]
                    detail['per_frame_image_checks']={str(i):v for i,v in per_frame.items()}
                # Landmarks can also jump. Only strong, stable source texture may
                # overrule their motion veto; visible image changes still reject.
                if motion > .25:
                    strong=(check['accepted'] and check.get('palm_tracks',0)>=30
                            and check.get('contradictory_fraction',0)>=.8
                            and check.get('median_image_difference',100)<5)
                    if not strong:
                        accepted_indices=[]
                        detail['reason']='independent_image_supports_articulation_or_is_ambiguous'
                    else:
                        detail['landmark_motion_veto']='overruled_by_stable_source_pixels'
                if not accepted_indices:
                    detail['accepted']=False
                    report['skipped'].append(detail)
                    continue
                left, right = observations[a][0]@palm_frames[a], observations[b][0]@palm_frames[b]
                la, rb = segment_basis(left[0]), segment_basis(right[0])
                if la is None or rb is None:
                    raise ValueError('Degenerate thumb proximal basis')
                left, right = left[1:]@la, right[1:]@rb
                edits = []
                for i in accepted_indices:
                    current = observations[i][0]@palm_frames[i]
                    basis = segment_basis(current[0])
                    if basis is None:
                        raise ValueError('Degenerate thumb proximal basis')
                    weight = (times[i]-times[a])/(times[b]-times[a])
                    distal = arc(left, right, weight)@basis.T@palm_frames[i].T
                    edits.append((i, unit(distal)))
                for i, distal in edits:
                    output[i][0,1:] = distal
                detail['accepted']=True
                detail['repaired_samples']=accepted_indices
                if not check['accepted']:
                    detail['reason']='independent_palm_aligned_source_pixels_disagree'
                report['accepted'].append(detail)
            except (ValueError, IndexError, np.linalg.LinAlgError, cv2.error) as error:
                detail.update(accepted=False, reason=str(error))
                report['skipped'].append(detail)
    finally:
        evidence.close()
    return output, report
