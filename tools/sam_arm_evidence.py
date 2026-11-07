"""Conservative visible-forearm image evidence; no invisible elbow labels."""
import cv2
import numpy as np


class PoseArmObserver:
    """Independent image pose, matched to an observed hand rather than SAM xyz.

    An off-image elbow supplies only a direction hypothesis, never an observed
    elbow position. Hand agreement and pose visibility gate every such cue.
    """
    def __init__(self):
        import mediapipe as mp
        import mediapipe.python.solution_base as base
        from pathlib import Path
        import shutil, tempfile
        # The legacy graph loader on Windows cannot open a Unicode path.
        # Its bundled models are copied locally; no model download is needed.
        package=Path(mp.__file__).parent
        if not str(package).isascii():
            runtime=Path(tempfile.gettempdir())/'ellen_pose_runtime_v1'
            modules=runtime/'mediapipe/modules'
            if not (modules/'pose_landmark/pose_landmark_full.tflite').is_file():
                shutil.copytree(package/'modules',modules,dirs_exist_ok=True)
            self._base=base;self._file=base.__file__
            base.__file__=str(runtime/'mediapipe/python/solution_base.py')
        else:self._base=None
        try:
            self.pose=mp.solutions.pose.Pose(static_image_mode=False,model_complexity=1,
                smooth_landmarks=False,min_detection_confidence=.6,min_tracking_confidence=.6)
        finally:
            if self._base:self._base.__file__=self._file

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def observe(self,bgr,hands):
        h,w=bgr.shape[:2];aspect=w/h
        result=self.pose.process(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:return {}
        pts=result.pose_landmarks.landmark;pairs=[]
        for j,ids in enumerate(((11,13,15),(12,14,16))):
            joints=[pts[k] for k in ids]
            p=np.array([[v.x*aspect,v.y] for v in joints])
            visibility=np.array([v.visibility for v in joints])
            if not np.isfinite(p).all() or min(visibility)<.7:continue
            v=p[1]-p[2];length=np.linalg.norm(v)
            if length<.055:continue  # A foreshortened forearm has no stable 2D axis.
            for side,hand in hands.items():
                if len(hand.get('points',[]))!=21:continue
                distance=float(np.linalg.norm(p[2]-hand['points'][0]))
                if distance>.065:continue
                confidence=float(min(visibility)*np.clip(1-distance/.10,0,1))
                if confidence<.5:continue
                inside=(p[:,0]>=0)&(p[:,0]<=aspect)&(p[:,1]>=0)&(p[:,1]<=1)
                pairs.append((distance,j,side,{'direction':(v/length).tolist(),
                    'confidence':confidence,'visible_length':float(min(length,.35)),
                    'source':'independent-video-pose','points':p.tolist(),
                    'visibility':visibility.tolist(),'in_frame':inside.tolist(),
                    'elbow_observed':bool(inside[1]),'wrist_match_height':distance}))
        found={};used=set()
        for _,j,side,row in sorted(pairs,key=lambda p:p[0]):
            if side not in found and j not in used:found[side]=row;used.add(j)
        return found


def observe_forearm(bgr, hand):
    if not hand or len(hand.get('points', [])) != 21:
        return None
    height, width = bgr.shape[:2]
    points = np.asarray(hand['points'], dtype=float)*height
    wrist = points[0]
    away = wrist-points[[5,9,13,17]].mean(0)
    length = np.linalg.norm(away)
    if length < .015*height or not (0 <= wrist[0] < width and 0 <= wrist[1] < height):
        return None
    away /= length
    scale = min(1.,480/height)
    image = cv2.resize(bgr,None,fx=scale,fy=scale)
    h,w = image.shape[:2]; p=points*scale; origin=wrist*scale
    palm = np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(palm,cv2.convexHull(p[[0,5,9,13,17]].astype(np.int32)),1)
    palm=cv2.erode(palm,np.ones((3,3),np.uint8))
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(float)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    selected=lab[(palm>0)&(hsv[:,:,1]>20)]
    if len(selected)<15:return None
    color=np.median(selected,axis=0)
    yy,xx=np.mgrid[:h,:w];delta=np.stack((xx-origin[0],yy-origin[1]),axis=-1)
    distance=np.linalg.norm(delta,axis=2); along=delta@away
    chroma=np.linalg.norm(lab[:,:,1:]-color[1:],axis=2)
    mask=((chroma<19)&(abs(lab[:,:,0]-color[0])<85)&(hsv[:,:,1]>22)
          &(distance>.018*h)&(distance<.35*h)&(along>.012*h)
          &(along>distance*.25)).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    choices=[]
    for k in range(1,count):
        y,x=np.nonzero(labels==k)
        if len(x)<35:continue
        cloud=np.c_[x,y].astype(float);d=cloud-origin
        nearest=np.linalg.norm(d,axis=1).min()
        if nearest>.075*h:continue
        # Robust axis from several visible cross-sections, with its origin at
        # the observed wrist. It is a direction cue, not a made-up elbow point.
        cov=d.T@d/len(d);values,vectors=np.linalg.eigh(cov)
        direction=vectors[:,-1]
        if direction@away<0:direction=-direction
        extent=float(np.percentile(d@direction,90))
        ratio=values[-1]/max(1.,values[0])
        if extent<.065*h or ratio<3 or direction@away<.35:continue
        confidence=float(np.clip((ratio-2)/8,0,1)*np.clip(extent/(.12*h),0,1))
        choices.append((confidence*extent/(1+nearest),direction,extent,confidence))
    if not choices:return None
    _,direction,extent,confidence=max(choices,key=lambda r:r[0])
    return {'direction':direction.tolist(),'visible_length':extent/h,
            'confidence':confidence,'source':'visible-forearm-color-axis',
            'wrist':(wrist/height).tolist(),'elbow_observed':False}


def enrich(body, screen):
    capture=cv2.VideoCapture(body['video']);previous_index=-1
    assert capture.isOpened(),body['video']
    found=0
    try:
        for i,(sample,row) in enumerate(zip(body['frames'],screen['frames'])):
            index=int(sample['source_frame'])
            if index!=previous_index+1:capture.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,bgr=capture.read();previous_index=index
            if not ok:raise ValueError('Cannot decode forearm evidence frame')
            row['arms']={}
            for side,hand in row.get('hands',{}).items():
                evidence=observe_forearm(bgr,hand)
                if evidence:row['arms'][side]=evidence;found+=1
            if i%100==0:print('ARM_EVIDENCE',i+1,len(body['frames']),flush=True)
    finally:capture.release()
    screen['arm_evidence_version']=1
    return found
