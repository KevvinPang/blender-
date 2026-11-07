"""Final temporal fitting of complete arm hinges, after image and safety passes."""
import math

import numpy as np
from mathutils import Quaternion, Vector


def phone_weights(scene, data, side):
    from sam_ellen_phone_ranges import ranges
    from sam_ellen_sequence import timeline_frame
    settings = getattr(scene, 'sam_ellen', None)
    times = np.asarray([r['time'] for r in data['frames']], float)
    weights = np.zeros(len(times))
    if not settings or not settings.phone_enabled or settings.phone_side != side:
        return weights
    for first, last in ranges(settings):
        frames = np.asarray([timeline_frame(data, t) for t in times])
        # Ramp the extra damping inside the range, never change its visibility.
        distance = np.minimum(frames-first, last-frames)/data['fps']
        w = np.clip(distance/.18, 0., 1.)
        weights = np.maximum(weights, w*w*(3-2*w))
    return weights


def stabilize_arms(scene, scratch, rig, channels, data, protected=None,
                   allowed=None, sides=('L', 'R'), report=None):
    """Filter torso-relative hinge, flexion and palm together, preserving lengths.

    A hand's classification score is not localization confidence. Partial crop
    support gets less weight; genuine low-frequency motion gets more. Explicit
    contact/fist intervals and their boundary poses remain exact anchors.
    """
    from sam_ellen_arm_visibility import screen_hand, hand_landmark_mask
    from sam_ellen_arm_chain import (frame_rotation, anatomical_rotations,
        torso_front_plane, bounded_wrist_rotation)
    from sam_ellen_body_clearance import torso_geometry, torso_envelope, arm_clearance, choose_chain
    from sam_ellen_arm_policy import wrist_limit, PREFIXES
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_temporal import quaternions, trajectory

    times = np.asarray([r['time'] for r in data['frames']], float)
    count = len(times)
    if count < 3:
        return
    allowed = np.ones(count, bool) if allowed is None else np.asarray(allowed, bool)
    geometry = torso_geometry(rig)
    original = {name: list(rows) for name, rows in channels.items()}
    view = scratch.view_layers[0]
    chest = rig.pose.bones.get('UpperBody2') or rig.pose.bones.get('UpperBody1')
    if chest is None:
        return

    def load(i):
        for name, rows in original.items():
            rig.pose.bones[name].rotation_quaternion = rows[i][1]
        view.update()

    for side in sides:
        if not all(p+side in channels for p in PREFIXES):
            continue
        a, e, w = [rig.pose.bones[p+'_'+side] for p in ('Arm', 'Elbow', 'Wrist')]
        controller = rig.pose.bones['HandTwist_'+side]
        anchors = (np.asarray(protected.get(side, np.zeros(count, bool)), bool)
                   if protected else np.zeros(count, bool)) | ~allowed
        amount = np.ones(count)
        if anchors.any():
            distance = np.min(abs(times[:, None]-times[anchors][None, :]), axis=1)
            amount = np.clip(distance/.20, 0., 1.)
            amount = amount*amount*(3-2*amount)
        frames, flexes, palms, rolls, positions = [], [], [], [], []
        support = np.zeros(count)
        for i in range(count):
            load(i)
            body = chest.matrix.to_quaternion() @ chest.bone.matrix_local.to_quaternion().inverted()
            u, v = e.head-a.head, w.head-e.head
            frames.append(body.inverted() @ frame_rotation(u, u.cross(v)))
            flexes.append(u.angle(v))
            palms.append(body.inverted() @ w.matrix.to_quaternion())
            q = controller.rotation_quaternion.copy()
            if q.w < 0:
                q.negate()
            rolls.append(q.to_swing_twist('Y')[1])
            positions.append(body.inverted() @ (w.head-a.head))
            hand = screen_hand(data.get('screen', {}), i, side)
            mask = hand_landmark_mask(hand)
            support[i] = float(np.mean(mask)) if hand else 0.
            if i % 80 == 0:
                yield side, i, count*2
        coarse = trajectory(times, np.asarray(positions), .14)
        speed = np.linalg.norm(np.gradient(coarse, times, axis=0), axis=1)
        # Fit the complete motion first: frame-wise IK corrections must not be
        # allowed to masquerade as fast intentional movement in this weighting.
        weights = .65 + 1.35*support + 6*np.clip(speed/.65, 0., 1.)**2
        grip = phone_weights(scene, data, side)
        filtered_frames = quaternions(times, frames, .065, weights)
        filtered_flexes = trajectory(times, flexes, .065, weights)
        filtered_palms = quaternions(times, palms, .055, weights)
        # This is a bounded MMD controller, not a freely spinning orientation.
        # Its fractional helper bones make roll+2*pi a different skin pose.
        # Unwrapping then clamping would break protected interval boundaries.
        filtered_rolls = trajectory(times, rolls, .075, weights)
        if grip.any():
            phone_frames = quaternions(times, frames, .095, weights)
            phone_flexes = trajectory(times, flexes, .095, weights)
            phone_palms = quaternions(times, palms, .085, weights)
            phone_rolls = trajectory(times, rolls, .10, weights)
            for i, gain in enumerate(grip):
                filtered_frames[i] = filtered_frames[i].slerp(phone_frames[i], float(gain))
                filtered_flexes[i] = (1-gain)*filtered_flexes[i] + gain*phone_flexes[i]
                filtered_palms[i] = filtered_palms[i].slerp(phone_palms[i], float(gain))
                filtered_rolls[i] = (1-gain)*filtered_rolls[i] + gain*phone_rolls[i]
        previous = None
        hinge = {}
        changes = []
        for i in range(count):
            if amount[i] <= 0:
                previous = None
                hinge = {}
                continue
            load(i)
            body = chest.matrix.to_quaternion() @ chest.bone.matrix_local.to_quaternion().inverted()
            frame = body @ frames[i].slerp(filtered_frames[i], float(amount[i]))
            flex = float((1-amount[i])*flexes[i] + amount[i]*filtered_flexes[i])
            # Preserve the original endpoint-safe elbow range; avoid a straight
            # singularity even if a filtered flexion overshoots at a boundary.
            flex = float(np.clip(flex, math.radians(15), math.radians(145)))
            hand = body @ palms[i].slerp(filtered_palms[i], float(amount[i]))
            aa = a.head.copy()
            upper, lower = (e.head-aa).length, (w.head-e.head).length
            old_wrist = w.head.copy()
            u = frame @ Vector((0, 1, 0))
            v = Quaternion(frame @ Vector((1, 0, 0)), flex) @ u
            ee, ww = aa + u*upper, aa + u*upper + v*lower
            envelope = torso_envelope(rig, geometry)
            margin = arm_clearance(aa, ee, ww, envelope)
            if margin < -.001:
                ee, ww, margin = choose_chain(aa, ee, ww, envelope,
                    torso_front_plane(rig), 35., side, previous)
            elbow = math.degrees((aa-ee).angle(ww-ee))
            hand = bounded_wrist_rotation(hand, ww-ee, wrist_limit(elbow))
            anatomical_rotations(a, e, w, aa, ee, ww, hand, set_global_rotation, view, hinge)
            palm = w.matrix.to_quaternion().copy()
            roll = (1-amount[i])*rolls[i] + amount[i]*filtered_rolls[i]
            controller.rotation_quaternion = Quaternion((0, 1, 0),
                float(np.clip(roll, -math.radians(170), math.radians(170))))
            view.update()
            set_global_rotation(w, palm)
            view.update()
            for prefix in PREFIXES:
                name = prefix+side
                channels[name][i] = channels[name][i][0], tuple(rig.pose.bones[name].rotation_quaternion)
            previous = (e.head.copy(), w.head.copy())
            changes.append({'frame': i+1, 'wrist_shift_m': (w.head-old_wrist).length,
                            'clearance_m': margin, 'phone_weight': float(grip[i])})
            if i % 80 == 0:
                yield side, count+i, count*2
        if report is not None:
            report.append({'model': 'final-hinge-residual-stability-v1', 'side': side,
                'frames': count, 'protected_frames': (np.flatnonzero(anchors)+1).tolist(),
                'changes': changes, 'time_filter': 'centered / torso-relative / crop and motion weighted'})
