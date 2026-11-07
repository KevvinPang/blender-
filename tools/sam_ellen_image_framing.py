"""Fit face skin extent and upper-body framing in a shared perspective camera."""
import numpy as np
from mathutils import Vector


def fit_framing(scene,target_rig,scratch,rig,channels,data,options,camera,cameras,report,geometry):
    from sam_ellen_screen import project,unproject
    from sam_ellen_image_geometry import posed_geometry
    from sam_ellen_temporal import trajectory
    from sam_ellen_sequence import timeline_frame,curves
    from sam_ellen_camera_fit import camera_source
    source=camera_source(camera)
    # Preserve authored/animated optics. A focal search is only appropriate for
    # the otherwise static reference camera of a replacement capture.
    adjustable=not data.get('_sam_timeline') and not source.animation_data and not source.data.animation_data
    times=np.array([r['time'] for r in data['frames']]);contexts=[]
    for i,row in enumerate(data['frames']):
        face=data['screen']['frames'][i].get('face')
        if not face or 'jaw' not in face or 'chin' not in face:contexts.append(None);continue
        for n,samples in channels.items():rig.pose.bones[n].rotation_quaternion=samples[i][1]
        scratch.view_layers[0].update()
        eye=(rig.pose.bones['Eye_L'].head+rig.pose.bones['Eye_R'].head)*.5
        chin,left,right=posed_geometry(rig,geometry)
        points=[eye,chin,left,right,*[rig.pose.bones['Arm_'+s].head.copy() for s in ('L','R')]]
        target=np.array(face['center']);target[0]*=cameras[i][2]/data['screen']['aspect']
        jaw=np.array(face['jaw']);jaw[:,0]*=cameras[i][2]/data['screen']['aspect']
        fc=np.array(face['chin']);fc[0]*=cameras[i][2]/data['screen']['aspect']
        if options.get('mirror'):target[0]=cameras[i][2]-target[0];fc[0]=cameras[i][2]-fc[0]
        target_width=np.ptp(jaw[:,0]);target_height=np.linalg.norm(fc-target)
        shoulder=None
        if row.get('image_points'):
            a=np.array(row['image_points'])[[5,6],:2]/data['video_orientation']['decoded_height_width'][0]
            if ((a[:,0]>=0)&(a[:,0]<=data['screen']['aspect'])&(a[:,1]>=0)&(a[:,1]<=1)).all():
                a[:,0]*=cameras[i][2]/data['screen']['aspect'];shoulder=np.linalg.norm(a[0]-a[1])
        contexts.append((points,target,target_width,target_height,shoulder))
        if i%100==0:yield i+1,len(times)*2
    def fit(i,factor):
        points,target,width,height,shoulder=contexts[i];old=cameras[i]
        projection=old[1].copy();projection[0][0]*=factor;projection[1][1]*=factor
        cam=(old[0],projection,*old[2:]);eye=points[0]
        old_depth=-(cam[0]@eye).z
        def image(depth):
            wanted=unproject(Vector(target),float(depth),cam)
            offset=eye-wanted;inverse=cam[0].inverted();inverse.translation+=offset
            updated=(inverse.inverted(),*cam[1:])
            p=np.array([project(v,updated) for v in points])
            lengths=[np.linalg.norm(p[2]-p[3]),np.linalg.norm(p[0]-p[1])]
            targets=[width,height];weights=[.55,.10]
            if shoulder is not None:lengths.append(np.linalg.norm(p[4]-p[5]));targets.append(shoulder);weights.append(.35)
            error=np.log(np.maximum(1e-6,lengths)/np.maximum(1e-6,targets))
            return float(np.sum(np.array(weights)*error**2)),offset,updated,p
        # Search in distance, never distort head/limb scale to match a pixel.
        grid=np.geomspace(max(.13,old_depth*.4*factor),max(.5,old_depth*3*factor),23)
        costs=[image(z)[0] for z in grid];k=int(np.argmin(costs))
        lo,hi=grid[max(0,k-1)],grid[min(len(grid)-1,k+1)]
        for _ in range(11):
            a=lo+(hi-lo)*.382;b=lo+(hi-lo)*.618
            if image(a)[0]<image(b)[0]:hi=b
            else:lo=a
        return image((lo+hi)*.5)
    valid=np.array([i for i,c in enumerate(contexts) if c is not None])
    if not len(valid):return
    sample=valid[np.linspace(0,len(valid)-1,min(24,len(valid))).astype(int)]
    # A telephoto fit can hide proportion differences but makes the observed
    # near-camera hands unreachable with the character's actual arm lengths.
    factors=[1.] if not adjustable else [f/camera.data.lens for f in sorted(set([camera.data.lens,28.,50.]))]
    reaches={s:(rig.data.bones['Arm_'+s].head_local-rig.data.bones['Elbow_'+s].head_local).length+
                  (rig.data.bones['Elbow_'+s].head_local-rig.data.bones['Wrist_'+s].head_local).length for s in ('L','R')}
    palms={s:(rig.data.bones['IndexFinger1_'+s].head_local-rig.data.bones['LittleFinger1_'+s].head_local).length for s in ('L','R')}
    close=[]
    for i in valid:
        for s,h in data['screen']['frames'][i].get('hands',{}).items():
            if s in reaches and h.get('span',0)>.10:close.append((h['span'],int(i),s))
    close=sorted(close,reverse=True)[::max(1,len(close)//16)][:16]
    def focal_cost(f):
        framing=float(np.median([fit(int(i),f)[0] for i in sample]));reach_errors=[]
        for span,i,s in close:
            _,_,cam,_=fit(i,f)
            shoulder=contexts[i][0][4 if s=='L' else 5]
            shoulder_depth=-(cam[0]@shoulder).z
            estimated=abs(cam[1][1][1])*.5*palms[s]/span
            reach_errors.append(max(0.,shoulder_depth-estimated-reaches[s]*.95)/reaches[s])
        return framing+(4*float(np.mean(np.array(reach_errors)**2)) if reach_errors else 0.)
    trials=[(focal_cost(f),f) for f in factors]
    cost,factor=min(trials)
    # Prefer the existing optics when the multi-frame evidence is inconclusive.
    initial=next(c for c,f in trials if abs(f-1)<1e-6)
    if initial-cost<.002:factor=1.
    offsets=np.zeros((len(times),3));weights=np.zeros(len(times));details={}
    for i in valid:
        score,offset,cam,p=fit(int(i),factor)
        relative=times[i]-times[0];weight=float(np.clip((relative-options.get('start',0))/.15,0,1)*options['strength'])
        if options.get('end',0)>0:weight*=float(np.clip((options['end']-relative)/.15,0,1))
        offsets[i]=np.array(offset)*weight;weights[i]=weight;details[int(i)]=score
    for j in range(3):offsets[:,j]=np.interp(times,times[valid],offsets[valid,j])
    offsets=trajectory(times,offsets,.07,np.maximum(.05,weights),robust=.025)
    from sam_ellen_framing_stability import stabilize_punch_framing
    offsets,punch_stable=stabilize_punch_framing(data,offsets)
    camera.data.lens*=factor
    frame,subframe=scene.frame_current,scene.frame_subframe
    try:
        for i,old in enumerate(list(cameras)):
            projection=old[1].copy();projection[0][0]*=factor;projection[1][1]*=factor
            inverse=old[0].inverted();inverse.translation+=Vector(offsets[i])
            cameras[i]=(inverse.inverted(),projection,*old[2:])
            f=timeline_frame(data,data['frames'][i]['time']);scene.frame_set(int(f),subframe=f-int(f))
            world=target_rig.evaluated_get(__import__('bpy').context.evaluated_depsgraph_get()).matrix_world.copy()
            camera.matrix_world=world@inverse
            for prop in ('location','rotation_quaternion','scale'):camera.keyframe_insert(prop,frame=f)
            if contexts[i] is not None:
                points,target,width,height,shoulder=contexts[i];p=np.array([project(v,cameras[i]) for v in points])
                report.append({'frame':float(f),'model':'surface-and-shoulder-framing-v1','lens_mm':camera.data.lens,
                    'face_width_ratio':float(np.linalg.norm(p[2]-p[3])/width),
                    'chin_height_ratio':float(np.linalg.norm(p[0]-p[1])/height),
                    'shoulder_width_ratio':float(np.linalg.norm(p[4]-p[5])/shoulder) if shoulder else None,
                    'center_error_px':float(np.linalg.norm(p[0]-target)*options.get('height',1920)),
                    'punch_camera_stability':float(punch_stable[i]),
                    'focal_source':'multi-frame artistic projection fit; not recovered phone intrinsics'})
            if i%100==0:yield len(times)+i+1,len(times)*2
        for c in curves(camera.animation_data.action,camera.animation_data.action_slot):
            for k in c.keyframe_points:k.interpolation='LINEAR'
        camera['sam_framing_model']='surface-and-shoulder-framing-v1'
    finally:scene.frame_set(frame,subframe=subframe)
