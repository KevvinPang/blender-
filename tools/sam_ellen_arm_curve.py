"""Continuous acceleration fitting inside the whole-chain safety corridor."""
import math
import numpy as np
from mathutils import Quaternion


def fit_curve(times,rows,rotations,flexes,upper,lower):
    """Refine graph knots without quantizing the final trajectory to that graph.

    Observed fast motion is weighted more strongly. Protected poses are exact.
    Every accepted update includes the interior of each moving-torso edge.
    """
    from sam_ellen_temporal import quaternions,trajectory
    from sam_ellen_arm_trajectory import _rotate,_angle
    from sam_ellen_body_clearance import chains_safe
    times=np.asarray(times);q=np.array(rotations,copy=True);f=np.array(flexes,copy=True)
    locked=np.array([r.get('locked',False) for r in rows]);visible=np.array([r['visible'] for r in rows])
    max_flex=np.radians([180-r.get('minimum_elbow',35.) for r in rows])
    bodies=[r['body'] for r in rows];a=np.array([r['a'] for r in rows]);env=np.array([r['env'] for r in rows])
    dt=np.diff(times);weights=np.where(visible,12.,1.);weights[locked]=1e9
    def geometry(q,f):
        u=_rotate(q,[0,1,0]);v=u*np.cos(f)[:,None]+_rotate(q,[0,0,1])*np.sin(f)[:,None]
        e=a+u*upper;return e,e+v*lower
    def safe(q,f):
        if np.any((f[~locked]<math.radians(15)-1e-5)|(f[~locked]>max_flex[~locked]+1e-5)):return False
        # chains_safe has one socket; batch moving sockets explicitly.
        e,w=geometry(q,f)
        for i in np.flatnonzero(~locked):
            if not chains_safe(a[i],e[i:i+1],w[i:i+1],env[i],upper,lower,margin=.0005)[0]:return False
        for i in range(len(q)-1):
            if locked[i] and locked[i+1]:continue
            rate=780 if visible[i] or visible[i+1] else 480
            if _angle(q[i:i+1],q[i+1:i+2])[0,0]>math.radians(rate*dt[i])+1e-5:return False
            if abs(f[i+1]-f[i])>math.radians(660*dt[i])+1e-5:return False
            lo=_rotate(q[i],[0,1,0])*np.cos(f[i])+_rotate(q[i],[0,0,1])*np.sin(f[i])
            ln=_rotate(q[i+1],[0,1,0])*np.cos(f[i+1])+_rotate(q[i+1],[0,0,1])*np.sin(f[i+1])
            if np.arccos(np.clip(lo@ln,-1,1))>math.radians(rate*dt[i])+1e-5:return False
            for t in (.25,.5,.75):
                last=q[i+1].copy()
                if q[i]@last<0:last=-last
                qi=q[i]*(1-t)+last*t;qi/=np.linalg.norm(qi)
                fi=f[i]*(1-t)+f[i+1]*t;ai=a[i]*(1-t)+a[i+1]*t
                u=_rotate(qi,[0,1,0]);v=u*np.cos(fi)+_rotate(qi,[0,0,1])*np.sin(fi)
                ei=ai+u*upper;wi=ei+v*lower
                if not chains_safe(ai,ei[None,:],wi[None,:],env[i]*(1-t)+env[i+1]*t,upper,lower,margin=.0005)[0]:return False
        return True
    # Fit relative to the moving chest, so normal torso motion is not damped.
    local=[b.inverted()@Quaternion(x) for b,x in zip(bodies,q)]
    fitted=quaternions(times,local,.105,weights)
    target=np.array([tuple(b@x) for b,x in zip(bodies,fitted)])
    target[locked]=q[locked]
    tf=np.clip(trajectory(times,f,.105,weights),math.radians(15),max_flex);tf[locked]=f[locked]
    target[np.sum(q*target,axis=1)<0]*=-1
    accepted=0.
    for gain in (1.,.8,.6,.4,.25,.125,.0625):
        candidate=q*(1-gain)+target*gain;candidate/=np.linalg.norm(candidate,axis=1)[:,None]
        cf=f*(1-gain)+tf*gain
        if safe(candidate,cf):q,f=candidate,cf;accepted=gain;break
    return q,f,{'model':'continuous-acceleration-fit-v1','gain':accepted,
        'observed_weight':12.,'time_scale_seconds':.105}


def smooth_keys(action,bone,locked):
    """C1 quaternion components with monotone Hermite tangents, arms only.

    Local bounds prevent component overshoot. Evaluated skeletal collision is
    still verified after commit, since a bounded component is not a safe limb.
    Authored/protected edges retain their exact original linear interpolation.
    """
    path=bone.path_from_id('rotation_quaternion');locked=np.asarray(locked,bool)
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    if curve.data_path!=path:continue
                    keys=curve.keyframe_points
                    if len(keys)!=len(locked) or len(keys)<3:continue
                    x=np.array([k.co.x for k in keys]);y=np.array([k.co.y for k in keys])
                    h=np.diff(x);slopes=np.diff(y)/h
                    tangent=np.zeros(len(keys));tangent[0]=slopes[0];tangent[-1]=slopes[-1]
                    for i in range(1,len(keys)-1):
                        if slopes[i-1]*slopes[i]>0:
                            tangent[i]=np.sign(slopes[i])*min(abs((slopes[i-1]+slopes[i])*.5),3*min(abs(slopes[i-1]),abs(slopes[i])))
                    for i,key in enumerate(keys):
                        key.handle_left_type=key.handle_right_type='FREE'
                        left=h[max(0,i-1)]/3;right=h[min(len(h)-1,i)]/3
                        key.handle_left=(x[i]-left,y[i]-tangent[i]*left)
                        key.handle_right=(x[i]+right,y[i]+tangent[i]*right)
                        if i+1<len(keys) and not (locked[i] or locked[i+1]):key.interpolation='BEZIER'
                    curve.update()
