"""One final owner for image evidence, torso clearance and continuous arms.

Choose complete fixed-length chains over time. A safe key pose is insufficient:
the edges between poses must also be safe. Fingers are never written here.
"""
import math
import numpy as np
from mathutils import Quaternion, Vector


def _rotate(q, v):
    q=np.asarray(q);v=np.broadcast_to(np.asarray(v),q.shape[:-1]+(3,))
    t=2*np.cross(q[...,1:],v)
    return v+q[...,:1]*t+np.cross(q[...,1:],t)


def _quats(elbows, wrists, a):
    from sam_ellen_arm_chain import frame_rotation
    u=elbows-np.asarray(a);v=wrists-elbows
    return np.array([tuple(frame_rotation(x,np.cross(x,y))) for x,y in zip(u,v)])


def _angle(first,last):
    return 2*np.arccos(np.clip(np.abs(first@last.T),0,1))


def _circles(a, goals, upper, lower, phases):
    goals=np.asarray(goals);axis=goals-a;d=np.linalg.norm(axis,axis=1)
    good=(d>.025)&(d<upper+lower-1e-5)
    goals=goals[good];d=d[good];axis=axis[good]/d[:,None]
    seed=np.tile([0.,1.,0.],(len(axis),1));seed[abs(axis[:,1])>.9]=[1.,0.,0.]
    x=np.cross(axis,seed);x/=np.maximum(1e-8,np.linalg.norm(x,axis=1))[:,None]
    y=np.cross(axis,x);along=(d*d+upper*upper-lower*lower)/(2*d)
    radius=np.sqrt(np.maximum(0,upper*upper-along*along))
    bends=x[:,None,:]*np.cos(phases)[None,:,None]+y[:,None,:]*np.sin(phases)[None,:,None]
    es=a+axis[:,None,:]*along[:,None,None]+bends*radius[:,None,None]
    ws=np.broadcast_to(goals[:,None,:],es.shape)
    return es.reshape(-1,3),ws.reshape(-1,3)


def _pose_cost(es,ws,row,goal,camera,data,side,measured,mirror,upper,lower,index):
    from sam_ellen_screen import project
    from sam_ellen_screen_fit import projected
    reach=upper+lower;aa=row['a']
    cp=np.c_[ws,np.ones(len(ws))]@np.asarray(camera[0]).T
    ep=np.c_[es,np.ones(len(es))]@np.asarray(camera[0]).T
    wp=projected(cp[:,:3],camera);elp=projected(ep[:,:3],camera)
    target=np.array(project(Vector(goal),camera))
    cost=np.sum((ws-goal)**2,axis=1)/reach**2
    if row['visible']:
        cost+=180*np.sum((wp-target)**2,axis=1)
        reference=max(.06,-(camera[0]@Vector(goal)).z)
        cost+=5*np.log(np.maximum(.025,-cp[:,2])/reference)**2
        pose=data['screen'].get('pose_arms',[])
        observation=pose[index].get(side) if len(pose)==len(data['frames']) else None
        if measured and observation:
            expected=np.asarray(observation['points'],float).copy();expected[:,0]*=camera[2]/data['screen']['aspect']
            if mirror:expected[:,0]=camera[2]-expected[:,0]
            direction=expected[1]-expected[2];direction/=max(1e-8,np.linalg.norm(direction))
            actual=elp-wp;actual/=np.maximum(1e-8,np.linalg.norm(actual,axis=1))[:,None]
            cost+=16*np.arccos(np.clip(actual@direction,-1,1))**2
            cost+=30*np.sum((elp-expected[1])**2,axis=1)
    else:cost+=18*np.sum((ws-goal)**2,axis=1)/reach**2
    down=np.array(row['body']@Vector((0,0,-1)))
    raised=np.arccos(np.clip((es-aa)@down/upper,-1,1))
    if not measured:cost+=.45*raised**2
    cost+=.06*np.sum((es-row['e'])**2,axis=1)/reach**2
    return cost


def _path(sets, times, rows, upper, lower):
    """Dynamic programming on anatomically valid poses AND collision-free edges."""
    from sam_ellen_body_clearance import chains_safe
    costs=sets[0]['cost'].copy();parents=[]
    for i in range(1,len(sets)):
        old,new=sets[i-1],sets[i];dt=max(1e-6,times[i]-times[i-1])
        turn=_angle(old['q'],new['q'])
        flex=np.abs(old['flex'][:,None]-new['flex'][None,:])
        # Limit both links; an unchanged wrist must not conceal a hinge flip.
        lo=_rotate(old['q'],[0,1,0])*np.cos(old['flex'])[:,None]+_rotate(old['q'],[0,0,1])*np.sin(old['flex'])[:,None]
        ln=_rotate(new['q'],[0,1,0])*np.cos(new['flex'])[:,None]+_rotate(new['q'],[0,0,1])*np.sin(new['flex'])[:,None]
        lower_turn=np.arccos(np.clip(lo@ln.T,-1,1))
        rate=780. if rows[i]['visible'] or rows[i-1]['visible'] else 480.
        valid=(turn<=math.radians(rate*dt))&(lower_turn<=math.radians(rate*dt))&(flex<=math.radians(660*dt))
        unchanged=rows[i].get('locked',False) and rows[i-1].get('locked',False)
        if unchanged:valid[:]=True
        # Check the moving torso and arm at multiple interior points. This
        # precedes path selection; no independent projection may follow it.
        oi,ni=np.nonzero(valid)
        if len(oi) and not unchanged:
            q0=old['q'][oi];q1=new['q'][ni].copy();q1[np.sum(q0*q1,axis=1)<0]*=-1
            for t in (.25,.5,.75):
                q=q0*(1-t)+q1*t;q/=np.linalg.norm(q,axis=1)[:,None]
                f=old['flex'][oi]*(1-t)+new['flex'][ni]*t
                a=rows[i-1]['a']*(1-t)+rows[i]['a']*t
                e=a+_rotate(q,[0,1,0])*upper
                w=e+(_rotate(q,[0,1,0])*np.cos(f)[:,None]+_rotate(q,[0,0,1])*np.sin(f)[:,None])*lower
                env=rows[i-1]['env']*(1-t)+rows[i]['env']*t
                safe=chains_safe(a,e,w,env,upper,lower,margin=.0015)
                valid[oi[~safe],ni[~safe]]=False
        transition=.24*(turn/max(.08,dt*5))**2+.12*(flex/max(.08,dt*5))**2
        transition+=.4*np.sum((old['w'][:,None,:]-new['w'][None,:,:])**2,axis=2)/(upper+lower)**2
        score=costs[:,None]+transition
        score[~valid]=np.inf
        parent=np.argmin(score,axis=0);costs=new['cost']+score[parent,np.arange(len(parent))]
        if not np.isfinite(costs).any():
            print('ARM_PATH_DISCONNECTED',i+1,'minimum_turn',float(turn.min()),'edges',int(valid.sum()),
                  'prior_reachable',int(np.isfinite(score).any(1).sum()),'tail_turn',float(turn[-1,-1]),flush=True)
            raise ValueError('手臂安全轨迹无法连续连接（样本 %d），已保留原动作；请检查接触预设或观测数据' %(i+1))
        parents.append(parent)
    index=int(np.argmin(costs));result=[index]
    for parent in reversed(parents):index=int(parent[index]);result.append(index)
    return list(reversed(result))


def _bridge_bank(sets, rows, visible, locked):
    """Include exact protected endpoints in the shared transition corridor.

    Protected poses have only one graph node. Sampling only editable observed
    frames omitted their hinge planes, leaving otherwise safe gestures without
    a legal entry/exit edge. Carry those endpoints into neighboring editable
    frames; the normal pose, speed and swept-collision checks still apply.
    """
    from sam_ellen_arm_chain import frame_rotation
    visible,locked=np.asarray(visible,bool),np.asarray(locked,bool)
    observed=np.flatnonzero(visible&~locked)
    selected=observed[np.unique(np.linspace(0,len(observed)-1,min(8,len(observed))).astype(int))] if len(observed) else []
    boundaries=np.flatnonzero(locked&(np.r_[True,~locked[:-1]]|np.r_[~locked[1:],True]))
    bank=[]
    for j in np.unique(np.r_[selected,boundaries]).astype(int):
        best=int(np.argmin(sets[j]['cost']))
        goalq=rows[j]['body'].inverted()@Quaternion(sets[j]['q'][best])
        goalf=float(sets[j]['flex'][best])
        ie,iw=rows[j]['idle_e'],rows[j]['idle_w'];aa=rows[j]['a']
        idleq=rows[j]['body'].inverted()@frame_rotation(Vector(ie-aa),Vector(np.cross(ie-aa,iw-ie)))
        idlef=float(Vector(ie-aa).angle(Vector(iw-ie)))
        for t in np.linspace(0,1,13):bank.append((idleq.slerp(goalq,float(t)),idlef*(1-t)+goalf*t))
    return bank


def solve_trajectory(scratch,rig,channels,data,cameras,options,protected=None,
                     allowed=None,sides=('L','R'),report=None):
    from sam_ellen_arm_visibility import observed_frames,neutral_pose,screen_hand,hand_landmark_mask
    from sam_ellen_arm_chain import frame_rotation,anatomical_rotations,bounded_wrist_rotation
    from sam_ellen_body_clearance import torso_geometry,torso_envelope,chains_safe,arm_clearance,observed_elbows
    from sam_ellen_arm_policy import PREFIXES,wrist_limit,supported_elbow_minimum
    from sam_ellen_screen import project,unproject
    from sam_ellen_screen_fit import hand_points
    from sam_ellen_motion_continuity import translation_goal
    from sam_ellen_retarget import set_global_rotation
    times=np.array([r['time'] for r in data['frames']]);n=len(times)
    if n<2 or not cameras:return
    original={name:list(values) for name,values in channels.items()}
    geometry=torso_geometry(rig);view=scratch.view_layers[0]
    allowed=np.ones(n,bool) if allowed is None else np.asarray(allowed,bool)
    chest=rig.pose.bones.get('UpperBody2') or rig.pose.bones.get('UpperBody1')
    def load(i):
        for name,values in original.items():rig.pose.bones[name].rotation_quaternion=values[i][1]
        view.update()
    for side in sides:
        if not all(p+side in channels for p in PREFIXES):continue
        visible=observed_frames(data,side)
        locked=(np.asarray(protected[side],bool) if protected and side in protected else np.zeros(n,bool))|~allowed
        if locked.all():continue
        measured=observed_elbows(data,side)
        a,e,w=[rig.pose.bones[p+'_'+side] for p in ('Arm','Elbow','Wrist')]
        rows=[];goals=[]
        for i in range(n):
            load(i);aa,ee,ww=[b.head.copy() for b in (a,e,w)]
            body=chest.matrix.to_quaternion()@chest.bone.matrix_local.to_quaternion().inverted()
            palm=w.matrix.to_quaternion().copy();roll=rig.pose.bones['HandTwist_'+side].rotation_quaternion.copy()
            offsets=np.array([palm.inverted()@(p-ww) for p in hand_points(rig,side)])
            h=screen_hand(data['screen'],i,side);goal=ww.copy()
            if visible[i] and not locked[i] and h:
                obs={'points':np.array(h['points']),'mask':hand_landmark_mask(h)}
                goal=translation_goal(offsets,palm,ww,obs,cameras[i],data['screen']['aspect'],options.get('mirror',False))
            goals.append(np.array(goal))
            idle=neutral_pose(rig,side,view)
            for name,q in idle.items():
                # The shoulder and fingers belong to the source take.
                if name.startswith(PREFIXES):rig.pose.bones[name].rotation_quaternion=q
            view.update();ie,iw=e.head.copy(),w.head.copy()
            env=torso_envelope(rig,geometry)
            # Find a nearby natural pose outside this character's shoulder
            # socket, rather than exempting half its upper arm from collision.
            natural=[]
            for spread in (.24,.38,.55,.75):
                for forward in (.06,.20,.38,.60):
                    se=aa+body@Vector(((1 if side=='L' else -1)*spread,-forward,-1)).normalized()*(ee-aa).length
                    sw=se+body@Vector(((1 if side=='L' else -1)*spread*.65,-(forward+.60),-1)).normalized()*(ww-ee).length
                    natural.append((se,sw))
            ns=chains_safe(np.array(aa),np.array([x[0] for x in natural]),np.array([x[1] for x in natural]),env,(ee-aa).length,(ww-ee).length,margin=.003)
            feasible=[natural[k] for k in np.flatnonzero(ns)]
            if feasible:ie,iw=min(feasible,key=lambda p:(p[0]-ie).length_squared+(p[1]-iw).length_squared)
            rows.append({'a':np.array(aa),'e':np.array(ee),'w':np.array(ww),'idle_e':np.array(ie),'idle_w':np.array(iw),
                'body':body,'palm':palm,'roll':roll,'visible':bool(visible[i]),'locked':bool(locked[i]),'env':env,
                'minimum_elbow':supported_elbow_minimum(data,side,i),
                'natural':[(np.array(x),np.array(y)) for x,y in feasible]})
            if i%100==0:yield side,i,n*3
        upper=np.linalg.norm(rows[0]['e']-rows[0]['a']);lower=np.linalg.norm(rows[0]['w']-rows[0]['e']);reach=upper+lower
        valid=np.flatnonzero(visible|locked)
        # An observed downward departure carries on towards that image edge.
        # Long losses then converge to a lowered arm without holding at the face.
        for i in range(n):
            if visible[i] or locked[i]:continue
            left=valid[valid<i];right=valid[valid>i]
            if len(left) and len(right) and times[right[0]]-times[left[-1]]<=.24:
                j,k=int(left[-1]),int(right[0]);t=(times[i]-times[j])/(times[k]-times[j]);goals[i]=goals[j]*(1-t)+goals[k]*t
            elif len(left):
                j=int(left[-1]);dt=times[i]-times[j];gain=float(np.clip(dt/.26,0,1));gain=gain*gain*(3-2*gain)
                # Move relative to the torso, not the old world-space socket.
                ref=goals[j]+rows[i]['a']-rows[j]['a']
                if j>0 and visible[j-1] and dt<.20:
                    velocity=(goals[j]-goals[j-1])/max(1e-6,times[j]-times[j-1])
                    velocity*=min(1.,1.8/max(1e-8,np.linalg.norm(velocity)))
                    ref=ref+velocity*min(dt,.12)*(1-gain)
                goals[i]=ref*(1-gain)+rows[i]['idle_w']*gain
            elif len(right) and times[right[0]]-times[i]<.30:
                j=int(right[0]);t=float(np.clip(1-(times[j]-times[i])/.30,0,1));t=t*t*(3-2*t)
                goals[i]=rows[i]['idle_w']*(1-t)+(goals[j]+rows[i]['a']-rows[j]['a'])*t
            else:goals[i]=rows[i]['idle_w'].copy()
        sets=[]
        for i,row in enumerate(rows):
            aa=row['a'];camera=cameras[i]
            if locked[i]:
                es=row['e'][None,:];ws=row['w'][None,:]
            else:
                goal=goals[i];xy=project(Vector(goal),camera);depth=max(.06,-(camera[0]@Vector(goal)).z)
                shoulder=-(camera[0]@Vector(aa)).z
                zs=np.unique(np.r_[depth,np.linspace(max(.06,shoulder-reach),shoulder+reach,17)])
                gs=np.array([unproject(xy,float(z),camera) for z in zs])
                # Natural poses remain graph candidates. They are expensive
                # when observed, but guarantee a conservative loss fallback.
                bridge=[row['idle_w']*(1-t)+goal*t for t in (.2,.4,.6,.8)]
                gs=np.vstack((gs,row['idle_w'],row['w'],bridge))
                es,ws=_circles(aa,gs,upper,lower,np.arange(64)*math.tau/64)
                es=np.vstack((es,row['e'],row['idle_e']));ws=np.vstack((ws,row['w'],row['idle_w']))
                if row['natural']:
                    es=np.vstack((es,[p[0] for p in row['natural']]))
                    ws=np.vstack((ws,[p[1] for p in row['natural']]))
            u=es-aa;v=ws-es
            flex=np.arccos(np.clip(np.sum(u*v,axis=1)/(upper*lower),-1,1))
            safe=chains_safe(aa,es,ws,row['env'],upper,lower,margin=.003)
            safe&=(flex>=math.radians(15)-1e-5)&(flex<=math.radians(180-row['minimum_elbow'])+1e-5)
            if locked[i]:safe[:]=True
            es,ws,flex=es[safe],ws[safe],flex[safe]
            if not len(es):raise ValueError('模型肩窝附近没有安全手臂解；已保留原动作（%s，第 %d 个样本）'%(side,i+1))
            q=_quats(es,ws,aa)
            cost=_pose_cost(es,ws,row,goals[i],camera,data,side,bool(measured[i]),options.get('mirror',False),upper,lower,i)
            # Retain diverse depths/phases, plus the safe idle, so a local
            # image minimum cannot remove the only continuous route.
            order=np.argsort(cost);keep=[]
            for k in order:
                if not keep or np.min(np.linalg.norm(es[keep]-es[k],axis=1)+np.linalg.norm(ws[keep]-ws[k],axis=1))>reach*.025:
                    keep.append(int(k))
                if len(keep)>=28:break
            idle_id=int(np.argmin(np.sum((es-row['idle_e'])**2+(ws-row['idle_w'])**2,axis=1)))
            if idle_id not in keep:keep.append(idle_id)
            if not locked[i]:
                for t in (.2,.4,.6,.8,1.):
                    target_w=row['idle_w']*(1-t)+goals[i]*t
                    distance=np.sum((ws-target_w)**2,axis=1)/reach**2
                    index=int(np.argmin(distance*100+cost*.025))
                    if index not in keep:keep.append(index)
                for ne,nw in row['natural']:
                    index=int(np.argmin(np.sum((es-ne)**2+(ws-nw)**2,axis=1)))
                    if index not in keep:keep.append(index)
            sets.append({'e':es[keep],'w':ws[keep],'q':q[keep],'flex':flex[keep],'cost':cost[keep]})
            if i%50==0:yield side,n+i,n*3
        # Shared torso-relative bridge poses make the graph connected across
        # abrupt detection starts. Pruning each frame independently otherwise
        # retains only idle and target clusters, with no legal route between.
        bank=_bridge_bank(sets,rows,visible,locked)
        for i,row in enumerate(rows):
            if locked[i] or not bank:continue
            q=np.array([tuple(row['body']@x[0]) for x in bank]);flex=np.array([x[1] for x in bank])
            es=row['a']+_rotate(q,[0,1,0])*upper
            ws=es+(_rotate(q,[0,1,0])*np.cos(flex)[:,None]+_rotate(q,[0,0,1])*np.sin(flex)[:,None])*lower
            safe=chains_safe(row['a'],es,ws,row['env'],upper,lower,margin=.003)
            safe&=(flex>=math.radians(15)-1e-5)&(flex<=math.radians(180-row['minimum_elbow'])+1e-5)
            es,ws,q,flex=es[safe],ws[safe],q[safe],flex[safe]
            if not len(es):continue
            costs=_pose_cost(es,ws,row,goals[i],cameras[i],data,side,bool(measured[i]),options.get('mirror',False),upper,lower,i)
            extra={'e':es,'w':ws,'q':q,'flex':flex,'cost':costs}
            for key,values in extra.items():sets[i][key]=np.concatenate((sets[i][key],values))
        path=_path(sets,times,rows,upper,lower)
        from sam_ellen_arm_curve import fit_curve
        fitted_q,fitted_f,curve_report=fit_curve(times,rows,
            np.array([s['q'][k] for s,k in zip(sets,path)]),
            np.array([s['flex'][k] for s,k in zip(sets,path)]),upper,lower)
        if report is not None:report.append(dict(curve_report,side=side))
        changes=[];hinge={}
        for i,k in enumerate(path):
            if locked[i]:hinge={};continue
            load(i);row=rows[i];aa=Vector(row['a']);q=Quaternion(fitted_q[i]);flex=float(fitted_f[i])
            u=q@Vector((0,1,0));v=u*math.cos(flex)+(q@Vector((0,0,1)))*math.sin(flex)
            ee=aa+u*upper;ww=ee+v*lower
            palm=bounded_wrist_rotation(row['palm'],ww-ee,wrist_limit(180-math.degrees(flex)))
            anatomical_rotations(a,e,w,aa,ee,ww,palm,set_global_rotation,view,hinge)
            held=w.matrix.to_quaternion().copy()
            rig.pose.bones['HandTwist_'+side].rotation_quaternion=row['roll'];view.update();set_global_rotation(w,held);view.update()
            for prefix in PREFIXES:
                name=prefix+side;channels[name][i]=channels[name][i][0],tuple(rig.pose.bones[name].rotation_quaternion)
            changes.append({'frame':i+1,'observed':bool(visible[i]),'elbow_measured':bool(measured[i]),
                'clearance_m':arm_clearance(a.head,e.head,w.head,row['env']),
                'wrist_shift_m':float(np.linalg.norm(np.array(w.head)-row['w']))})
            if i%80==0:yield side,2*n+i,n*3
        if report is not None:report.append({'model':'coupled-image-clearance-trajectory-v1','side':side,'changes':changes,
            'frames':n,'protected_frames':(np.flatnonzero(locked)+1).tolist(),
            'constraints':'fixed lengths / observed elbow / whole upper arm / speed / quarter-frame collision edges'})
