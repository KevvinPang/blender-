"""Conservative offline finger outlier rejection with video-flow verification."""
from mocap_paths import LOCAL_APP
from pathlib import Path

import cv2
import numpy as np

VERSION = 2
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-10)


def bends(directions):
    d = unit(directions)
    return np.degrees(np.arccos(np.clip(np.sum(d[..., :-1, :] * d[..., 1:, :], axis=-1), -1, 1)))


def segment_basis(proximal):
    y = unit(proximal)
    x = np.array([1., 0, 0]) - y[0]*y
    if np.linalg.norm(x) < .15:
        return None
    x = unit(x)
    return np.stack((x, y, np.cross(x, y)), axis=1)


def arc(a, b, t):
    """Interpolate unit directions without reducing their length."""
    angle = np.arccos(np.clip(np.sum(a*b, axis=-1, keepdims=True), -1, 1))
    if np.any(angle > np.radians(150)):
        raise ValueError('Ambiguous opposite finger directions')
    sine = np.sin(angle)
    curved = (np.sin((1-t)*angle)*a + np.sin(t*angle)*b) / np.maximum(sine, 1e-10)
    return unit(np.where(angle < 1e-5, a*(1-t)+b*t, curved))


def candidates(observations, times, max_seconds=.55):
    """Find bounded excursions; no correction across missing data or long gaps."""
    if len(times) < 7:
        return []
    dt = np.diff(times)
    if not np.isfinite(times).all() or np.any(dt <= 0):
        return []
    step = float(np.median(dt))
    support = max(3, int(np.ceil(.10/step)))
    values = np.full((len(times), 5, 2), np.nan)
    for i, obs in enumerate(observations):
        if obs is not None:
            values[i] = bends(obs)
    stable = lambda v: np.isfinite(v).all() and np.max(np.ptp(v, axis=0)) <= 14
    found = []
    for f in range(5):
        used_until = -1
        for a in range(support-1, len(times)-support-1):
            if a <= used_until or not stable(values[a-support+1:a+1, f]):
                continue
            for b in range(a+2, len(times)-support+1):
                if times[b]-times[a] > max_seconds+1e-7:
                    break
                if np.max(dt[a:b+support-1]) > min(.12, step*1.8):
                    break
                window = values[a:b+support, f]
                if not np.isfinite(window).all():
                    break
                if not stable(values[b:b+support, f]):
                    continue
                if np.max(np.abs(values[a,f]-values[b,f])) > 18:
                    continue
                t = (np.asarray(times[a:b+1])-times[a])/(times[b]-times[a])
                expected = values[a,f]*(1-t[:,None])+values[b,f]*t[:,None]
                error = np.max(np.abs(values[a:b+1,f]-expected), axis=1)
                if np.max(error) < 45:
                    continue
                wrong = np.flatnonzero(error > 22)
                if not len(wrong) or wrong[0] == 0 or wrong[-1] == b-a:
                    continue
                found.append({'finger':f, 'left':a, 'right':b,
                              'start':a+int(wrong[0]), 'end':a+int(wrong[-1]),
                              'max_bend_error_degrees':float(np.max(error))})
                used_until = b
                break
    return sorted(found, key=lambda r: (r['left'],r['finger']))


class VideoEvidence:
    """Use independent image motion, not WiLoR detector confidence, as a veto."""
    def __init__(self, video, source_frames):
        import sys
        worker = LOCAL_APP
        if str(worker) not in sys.path:
            sys.path.insert(0,str(worker))
        from video_input import open_video, SequentialFrames
        self.capture = open_video(video)
        self.reader = SequentialFrames(self.capture)
        self.source_frames = source_frames
        self.cache = {}

    def close(self):
        self.capture.release()

    def frames(self, a, b):
        # Candidates are processed in temporal order; discard old decoded frames.
        self.cache = {i:v for i,v in self.cache.items() if i >= a}
        for i in range(a,b+1):
            if i not in self.cache:
                ok, frame = self.reader.read(self.source_frames[i])
                if not ok:
                    raise ValueError('Video decode failed')
                self.cache[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return [self.cache[i] for i in range(a,b+1)]

    @staticmethod
    def track(images, points):
        tracks = [np.asarray(points,dtype=np.float32).reshape(-1,1,2)]
        alive = np.ones(len(points),dtype=bool)
        for before, after in zip(images,images[1:]):
            p, status, err = cv2.calcOpticalFlowPyrLK(before,after,tracks[-1],None,
                winSize=(21,21),maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
            if p is None:
                return None,None
            back, status_back, _ = cv2.calcOpticalFlowPyrLK(after,before,p,None,
                winSize=(21,21),maxLevel=3)
            if back is None:
                return None,None
            alive &= status[:,0].astype(bool) & status_back[:,0].astype(bool)
            alive &= np.linalg.norm(back[:,0]-tracks[-1][:,0],axis=1) < 1.8
            alive &= err[:,0] < 25
            h,w = after.shape
            alive &= (p[:,0,0] >= 12) & (p[:,0,0] < w-12) & (p[:,0,1] >= 12) & (p[:,0,1] < h-12)
            tracks.append(p)
        return np.asarray(tracks)[:,:,0,:],alive

    def check(self, candidate, hands):
        a,b,f = candidate['left'],candidate['right'],candidate['finger']
        first = 1+4*f
        ids = [first+1,first+2,first+3]
        p = np.asarray([h['joints2d'] for h in hands[a:b+1]])
        images = self.frames(a,b)
        height,width = images[0].shape
        # Folded, hidden finger joints are not automatically reliable features.
        ends = p[[0,-1]][:,ids]
        if not np.isfinite(p).all() or np.any(ends < 12) or np.any(ends[...,0] >= width-12) or np.any(ends[...,1] >= height-12):
            return {'accepted':False,'reason':'finger_at_edge_or_invalid'}
        forward, good_f = self.track(images,p[0,ids])
        backward, good_b = self.track(images[::-1],p[-1,ids])
        if forward is None or backward is None:
            return {'accepted':False,'reason':'unreliable_image_tracking'}
        backward = backward[::-1]
        span = float(np.median(np.linalg.norm(p[:,5]-p[:,17],axis=1)))
        tolerance = max(5., span*.10)
        good = good_f & good_b
        good &= np.linalg.norm(forward[-1]-p[-1,ids],axis=1) < tolerance
        good &= np.linalg.norm(backward[0]-p[0,ids],axis=1) < tolerance
        good &= np.max(np.linalg.norm(forward-backward,axis=2),axis=0) < tolerance*1.5
        if good.sum() < 2:
            return self.appearance_check(candidate,images,p,ids,span)
        tracked = (forward+backward)*.5
        distance = np.median(np.linalg.norm(p[:,ids][:,good]-tracked[:,good],axis=2),axis=1)
        middle = distance[candidate['start']-a:candidate['end']-a+1]
        contradictory = float(np.mean(middle > max(12.,span*.15)))
        return {'accepted':contradictory >= .5,'reason':'image_motion_disagrees' if contradictory >= .5 else 'image_supports_motion_or_is_ambiguous',
                'reliable_points':int(good.sum()),'contradictory_fraction':contradictory,
                'median_image_residual_pixels':float(np.median(middle))}

    def appearance_check(self, candidate, images, points, ids, span):
        """Occluded joints need stable surrounding texture, not invented tracks."""
        p = points[0]
        h,w = images[0].shape
        mask = np.zeros((h,w),np.uint8)
        anchors = p[[0,1,2,5,9,13,17]]
        hull = cv2.convexHull(anchors.astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(mask,hull,255)
        features = cv2.goodFeaturesToTrack(images[0],maxCorners=160,qualityLevel=.015,
                                          minDistance=5,mask=mask,blockSize=5)
        if features is None or len(features) < 12:
            return {'accepted':False,'reason':'insufficient_palm_texture'}
        tracks, good = self.track(images,features[:,0])
        if tracks is None or good.sum() < 12:
            return {'accepted':False,'reason':'unreliable_palm_tracks'}
        accepted, residuals, differences = [],[],[]
        start,end = candidate['start']-candidate['left'],candidate['end']-candidate['left']
        for i in range(start,end+1):
            # Pixel tolerance must follow hand size in close-up footage.
            transform,inliers = cv2.estimateAffinePartial2D(tracks[0,good],tracks[i,good],
                method=cv2.RANSAC,ransacReprojThreshold=max(2.,span*.025),maxIters=1000)
            if transform is None or inliers.mean() < .65:
                continue
            project = lambda v: np.asarray(v)@transform[:,:2].T+transform[:,2]
            if np.median(np.linalg.norm(project(p[[0,5,9,13,17]])-points[i,[0,5,9,13,17]],axis=1)) > max(8.,span*.18):
                continue
            expected = project(p[ids])
            residual = float(np.median(np.linalg.norm(expected-points[i,ids],axis=1)))
            if residual < max(12.,span*.15):
                continue
            # A real extra finger should alter the image in its destination corridor.
            region = np.zeros((h,w),np.uint8)
            thickness = max(9,round(span*.13))
            for chain in (expected,points[i,ids]):
                for x,y in zip(chain,chain[1:]):
                    cv2.line(region,tuple(x.astype(int)),tuple(y.astype(int)),255,thickness)
            warped = cv2.warpAffine(images[0],transform,(w,h))
            valid = cv2.warpAffine(np.ones((h,w),np.uint8),transform,(w,h)) > 0
            region = (region > 0) & valid
            if region.sum() < 80:
                continue
            before = cv2.GaussianBlur(warped,(5,5),0).astype(float)[region]
            after = cv2.GaussianBlur(images[i],(5,5),0).astype(float)[region]
            difference = np.abs(after-before-np.median(after-before))
            unchanged = float(np.median(difference)) < 12 and np.percentile(difference,80) < 30
            accepted.append(unchanged)
            residuals.append(residual)
            differences.append(float(np.median(difference)))
        fraction = float(sum(accepted)/(end-start+1))
        return {'accepted':fraction >= .55,'reason':'stable_hand_image_disagrees' if fraction >= .55 else 'image_changed_or_tracking_ambiguous',
                'palm_tracks':int(good.sum()),'contradictory_fraction':fraction,
                'median_image_residual_pixels':float(np.median(residuals)) if residuals else None,
                'median_image_difference':float(np.median(differences)) if differences else None}


def filter_legacy_fingers(observations, palm_frames, hands, times, source_frames, video):
    """Repair distal articulation only, retaining the current proximal direction."""
    report = {'version':VERSION,'enabled':True,'max_seconds':.55,
              'image_verified':True,'accepted':[],'skipped':[]}
    found = candidates(observations,times)
    output = [v.copy() if v is not None else None for v in observations]
    if not found:
        return output,report
    if not video or not Path(video).is_file():
        report['warning'] = 'Source video missing; ambiguous finger poses were preserved'
        return output,report
    evidence = VideoEvidence(video,source_frames)
    try:
        for c in found:
            a,b,f = c['left'],c['right'],c['finger']
            details = {**c,'finger':FINGERS[f],'source_start':int(source_frames[c['start']]),
                       'source_end':int(source_frames[c['end']]),'start_seconds':float(times[c['start']]),
                       'end_seconds':float(times[c['end']])}
            try:
                check = evidence.check(c,hands)
                details.update(check)
                if not check['accepted']:
                    report['skipped'].append(details)
                    continue
                # Work in the palm frame, then relative to the first finger bone.
                left = observations[a][f] @ palm_frames[a]
                right = observations[b][f] @ palm_frames[b]
                la,rb = segment_basis(left[0]),segment_basis(right[0])
                if la is None or rb is None:
                    raise ValueError('Degenerate proximal basis')
                left,right = left[1:]@la,right[1:]@rb
                edits=[]
                for i in range(c['start'],c['end']+1):
                    current = observations[i][f]@palm_frames[i]
                    basis = segment_basis(current[0])
                    if basis is None:
                        raise ValueError('Degenerate proximal basis')
                    t=(times[i]-times[a])/(times[b]-times[a])
                    distal=arc(left,right,t)@basis.T@palm_frames[i].T
                    edits.append((i,distal))
                for i,distal in edits:
                    output[i][f,1:]=distal
                report['accepted'].append(details)
            except (ValueError,cv2.error,IndexError) as error:
                details.update(accepted=False,reason=str(error))
                report['skipped'].append(details)
    finally:
        evidence.close()
    return output,report


def filter_fingers(observations, palm_frames, hands, times, source_frames, video,
                   image_observations=None, side=None):
    output, report = filter_legacy_fingers(observations, palm_frames, hands, times, source_frames, video)
    from wilor_thumb import repair_thumb
    output, thumb = repair_thumb(output, palm_frames, hands, times, source_frames, video,
                                 cached=image_observations, side=side)
    report['thumb_consistency'] = thumb
    report['accepted'].extend(thumb['accepted'])
    report['skipped'].extend(thumb['skipped'])
    return output, report
