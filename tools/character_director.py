"""Keyboard locomotion and camera takes for the project's MMD character scenes."""
bl_info = {
    'name': '角色操控与运镜 / Character Director', 'author': 'Local Tools',
    'version': (0, 7, 0), 'blender': (5, 2, 0), 'category': 'Animation',
    'location': '3D View > Sidebar > 角色操控',
}

import json
import math
import time
import bisect
import importlib
import struct
import unicodedata
from functools import lru_cache
from pathlib import Path
from datetime import datetime

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from mathutils import Euler, Matrix, Quaternion, Vector

SESSION = None


def physics_module():
    import sys
    if 'ellen_physics' not in sys.modules:
        import importlib.util
        spec = importlib.util.spec_from_file_location('ellen_physics', Path(__file__).with_name('ellen_physics.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules['ellen_physics'] = module
        spec.loader.exec_module(module)
    return sys.modules['ellen_physics']


def active_session():
    """A finished take must never keep the entry buttons locked."""
    global SESSION
    if SESSION is not None and SESSION.closed:
        SESSION = None
    return SESSION


TAU = 2 * math.pi
PRESETS = {
    'WALK': ('自然走 · 交替摆臂', 1.0, .95, .065, .48, .025),
    'JOG': ('小跑 · 高抬腿屈肘', 2.1, 1.18, .20, 1.05, .025),
    'TINY': ('小碎步 · 短步快频', .60, .30, .03, .14, .012),
    'SWAY': ('摇摆走 · 侧摆重心', .65, .75, .08, .32, .16),
    'MMD_WALK04': ('MMD 走路 04 · 普通步行 B', 1., 1.0752, 0., 0., 0.),
    'MMD_RUN02': ('MMD 奔跑 02 · 女孩子跑', 1475/605, 1.31093, 0., 0., 0.),
    'MMD_WALK01': ('MMD 走路 01 · 女孩子走', 591/605, 1.0506667, 0., 0., 0.),
    'MMD_RUN01': ('MMD 奔跑 01 · 标准跑步', 2000/605, 2.1333333, 0., 0., 0.),
}
# Stance fraction, vertical bounce, forward lean, elbow bend, foot width, body roll.
STYLES = {
    'WALK': (.62, .022, .025, .12, .01, .025),
    'JOG': (.40, .065, .20, 1.10, .035, .05),
    'TINY': (.70, .008, .03, .28, -.012, .015),
    'SWAY': (.65, .030, .00, .55, .06, .20),
    'MMD_WALK04': (0., 0., 0., 0., 0., 0.),
    'MMD_RUN02': (0., 0., 0., 0., 0., 0.),
    'MMD_WALK01': (0., 0., 0., 0., 0., 0.),
    'MMD_RUN01': (0., 0., 0., 0., 0., 0.),
}
MMD_LOOPS = {'MMD_WALK04': ('susuki_walk04.vmd', 40, 40),
             'MMD_RUN02': ('susuki_run02.vmd', 20, 20),
             'MMD_WALK01': ('susuki_walk01.vmd', 40, 40),
             'MMD_RUN01': ('susuki_run01.vmd', 24, 24)}


@lru_cache(maxsize=4)
def read_motion(filename):
    """Read only bone keyframes from the bundled, unmodified VMD assets."""
    path = Path(__file__).resolve().with_name('character_director_assets')/filename
    if not path.is_file():
        raise ValueError('缺少配套 MMD 动作文件：' + str(path))
    data = path.read_bytes()
    if not data.startswith(b'Vocaloid Motion Data 0002'):
        raise ValueError('不支持的 VMD 动作文件：' + filename)
    count = struct.unpack_from('<I', data, 50)[0]
    if 54+count*111 > len(data):
        raise ValueError('VMD 骨骼数据不完整：' + filename)
    channels = {}
    for i in range(count):
        name, frame, *values = struct.unpack_from('<15sI7f64s', data, 54+i*111)
        name = unicodedata.normalize('NFKC', name.split(b'\0')[0].decode('cp932', errors='replace'))
        channels.setdefault(name, []).append((frame, Vector(values[:3]),
            Quaternion((values[6], *values[3:6])).normalized(), values[7]))
    return {name: (sorted(rows, key=lambda row: row[0]), sorted(row[0] for row in rows))
            for name, rows in channels.items()}


def vmd_bezier(interpolation, channel, x):
    x1, y1, x2, y2 = (interpolation[channel+i]/127 for i in (0, 4, 8, 12))
    if abs(x1-y1)+abs(x2-y2) < 1e-6:
        return x
    def cubic(t, a, b):
        return 3*(1-t)**2*t*a+3*(1-t)*t*t*b+t**3
    lo, hi = 0., 1.
    for _ in range(16):
        t = (lo+hi)*.5
        if cubic(t, x1, x2) < x:
            lo = t
        else:
            hi = t
    return cubic((lo+hi)*.5, y1, y2)


def motion_sample(channel, frame):
    rows, frames = channel
    i = bisect.bisect_right(frames, frame)
    if i == 0 or i == len(rows):
        row = rows[0] if i == 0 else rows[-1]
        return row[1].copy(), row[2].copy()
    a, b = rows[i-1], rows[i]
    x = (frame-a[0])/(b[0]-a[0])
    location = Vector([a[1][j]+(b[1][j]-a[1][j])*vmd_bezier(b[3], j, x) for j in range(3)])
    return location, a[2].slerp(b[2], vmd_bezier(b[3], 3, x))


class MMDLoop:
    """Convert one stable cycle to this rig, removing common forward travel."""
    def __init__(self, rig, bones, preset, height):
        filename, start, period = MMD_LOOPS[preset]
        raw = read_motion(filename)
        module = next((a.module for a in bpy.context.preferences.addons if a.module.endswith('mmd_tools')), None)
        if module is None:
            raise ValueError('请先启用项目自带的 MMD Tools')
        converter = importlib.import_module(module+'.core.vmd.importer').BoneConverter
        mapping = {unicodedata.normalize('NFKC', getattr(getattr(b, 'mmd_bone', None), 'name_j', '') or b.name): b
                   for b in rig.pose.bones}
        for role, (jp, _) in SEMANTICS.items():
            if role in bones:
                mapping[unicodedata.normalize('NFKC', jp)] = bones[role]
        allowed = set(b.name for b in bones.values())
        allowed.discard(bones['root'].name)
        for b in rig.pose.bones:
            jp = getattr(getattr(b, 'mmd_bone', None), 'name_j', '')
            if any(word in jp for word in ('親指', '人指', '人差指', '中指', '薬指', '小指', '腕捩', '手捩')):
                allowed.add(b.name)
        self.samples, self.idle = {}, {}
        self.count = period*2
        scale = .08*height/1.49
        origin = motion_sample(raw['センター'], start)[0].z
        drift = (motion_sample(raw['センター'], start+period)[0].z-origin)/period
        translated = {bones[k].name for k in ('center', 'ikL', 'ikR')}
        for jp, channel in raw.items():
            bone = mapping.get(jp)
            if bone is None or bone.name not in allowed:
                continue
            convert = converter(bone, scale)
            samples = []
            for i in range(self.count+1):
                frame = start+i*.5
                loc, rot = motion_sample(channel, frame)
                if bone.name in translated:
                    loc.z -= origin+drift*(frame-start)
                q = convert.convert_rotation((rot.x, rot.y, rot.z, rot.w))
                samples.append((convert.convert_location(loc), q))
            # Force an exact closed cycle; stable source cycles already match to rounding.
            samples[-1] = (samples[0][0].copy(), samples[0][1].copy())
            self.samples[bone.name] = samples
            idle_q = Quaternion()
            if any(word in jp for word in ('肩', '腕', 'ひじ', '手首', '指', '捩')):
                idle_q = samples[0][1].copy()
                for i, (_, q) in enumerate(samples[1:-1], 2):
                    idle_q = idle_q.slerp(q, 1/i)
            idle_loc = local_translation(bone, (0, 0, -.025*height/1.49)) if bone == bones['center'] else Vector()
            self.idle[bone.name] = (idle_loc, idle_q)

    def sample(self, name, phase):
        at = (phase % 1)*self.count
        i = int(at)
        a, b = self.samples[name][i:i+2]
        return a[0].lerp(b[0], at-i), a[1].slerp(b[1], at-i)


def previous_take_end(scene, rig):
    p = scene.character_director
    owners = ([scene.camera] if p.record_target == 'CAMERA' else [rig, p.last_root])
    for owner in owners:
        action = owner.animation_data.action if owner and owner.animation_data else None
        if action and action.get('character_director_take'):
            data = json.loads(action['character_director_take'])
            return int(data.get('end_frame') or data['events'][-1][0])
    if p.last_end > 0:
        return p.last_end
    raise ValueError('没有上一段操控录制；可选择“从当前帧接录”')
SEMANTICS = {
    'root': ('全ての親', 'ParentNode'), 'center': ('センター', 'Center'),
    'groove': ('グルーブ', 'Groove'), 'waist': ('腰', 'Waist'),
    'lower': ('下半身', 'LowerBody'), 'upper': ('上半身', 'UpperBody'),
    'upper2': ('上半身2', 'UpperBody2'), 'head': ('頭', 'Head'),
    'neck': ('首', 'Neck'),
}
for side, jp in (('L', '左'), ('R', '右')):
    for role, jpname, english in (
        ('leg', '足', 'Leg'), ('knee', 'ひざ', 'Knee'), ('ankle', '足首', 'Ankle'),
        ('ik', '足ＩＫ', 'LegIK'), ('ikparent', '足IK親', 'LegIKParent'),
        ('toe', 'つま先ＩＫ', 'ToeTipIK'), ('arm', '腕', 'Arm'),
        ('elbow', 'ひじ', 'Elbow'), ('wrist', '手首', 'Wrist'), ('shoulder', '肩', 'Shoulder')):
        SEMANTICS[role + side] = (jp + jpname, english + '_' + side)


def curves(action, slot=None):
    return [fc for layer in action.layers for strip in layer.strips
            if strip.type == 'KEYFRAME' for bag in strip.channelbags
            if slot is None or bag.slot_handle == slot.handle for fc in bag.fcurves] if action else []


def resolve_rig(scene):
    p = scene.character_director
    if p.rig and p.rig.name in scene.objects:
        return p.rig
    active = bpy.context.view_layer.objects.active
    if active and active.type == 'ARMATURE':
        return active
    for attr in ('sam_ellen', 'xr_ellen'):
        candidate = getattr(getattr(scene, attr, None), 'rig', None)
        if candidate and candidate.name in scene.objects:
            return candidate
    return None


def bone_binding(rig):
    if not rig or rig.type != 'ARMATURE':
        raise ValueError('请选择人物骨架，或先打开项目的角色场景')
    by_jp = {getattr(getattr(b, 'mmd_bone', None), 'name_j', ''): b for b in rig.pose.bones}
    result = {}
    for role, (jp, en) in SEMANTICS.items():
        b = by_jp.get(jp) or by_jp.get(jp.replace('ＩＫ', 'IK')) or rig.pose.bones.get(en) or rig.pose.bones.get(jp)
        if b:
            result[role] = b
    required = ('root', 'center', 'head', 'legL', 'legR', 'kneeL', 'kneeR', 'ikL', 'ikR')
    missing = [SEMANTICS[k][0] for k in required if k not in result]
    if missing:
        raise ValueError('此试用版需要 MMD 双腿 IK 骨架，缺少：' + '、'.join(missing))
    for side in ('L', 'R'):
        if not any(c.type == 'IK' and not c.mute and c.influence > .5
                   and c.target == rig for c in result['knee' + side].constraints):
            raise ValueError('请先启用人物的双腿 MMD IK：' + result['knee' + side].name)
    if abs(rig.matrix_world.to_3x3().determinant()) < 1e-8:
        raise ValueError('人物缩放无效，无法建立移动层')
    return result


def smooth(current, target, dt, seconds):
    return current + (target - current) * (1 - math.exp(-dt / max(seconds, .001)))


def angle_delta(a, b):
    return (b - a + math.pi) % TAU - math.pi


def sampled_basis(owner, source, frame):
    loc, rot, scale = owner.matrix_basis.decompose()
    values = {}
    for prop, default in (('location', loc), ('scale', scale),
                          ('rotation_quaternion', rot), ('rotation_euler', owner.rotation_euler)):
        path = owner.path_from_id(prop) if isinstance(owner, bpy.types.PoseBone) else prop
        if any((path, i) in source for i in range(len(default))):
            values[prop] = [source[path, i].evaluate(frame) if (path, i) in source else default[i]
                            for i in range(len(default))]
    if owner.rotation_mode == 'QUATERNION':
        rot = Quaternion(values.get('rotation_quaternion', rot)).normalized()
    elif 'rotation_euler' in values:
        rot = Euler(values['rotation_euler'], owner.rotation_mode).to_quaternion()
    return Matrix.LocRotScale(Vector(values.get('location', loc)), rot, Vector(values.get('scale', scale)))


def local_translation(bone, offset):
    return bone.bone.matrix_local.to_3x3().inverted() @ Vector(offset)


def local_rotation(bone, rotation):
    rest = bone.bone.matrix_local.to_quaternion()
    return rest.inverted() @ rotation @ rest


def foot_cycle(phase, stance=.6):
    """Constant ground speed during stance; smooth swing with foot clearance."""
    p = phase % 1
    if p < stance:
        return stance*.5 - p, 0.
    u = (p - stance) / (1-stance)
    return -stance*.5 + stance * (u*u*(3-2*u)), math.sin(math.pi*u) ** 1.2


def write_samples(owner, source, slot, samples, label, trim_from=None):
    """Copy source, changing only sampled properties inside the recorded interval."""
    action = source.copy() if source else bpy.data.actions.new(label)
    action.name = label
    action.use_fake_user = True
    ad = owner.animation_data_create()
    ad.action = action
    if slot:
        ad.action_slot = next(s for s in action.slots if s.identifier == slot)
    preserved = []
    if trim_from is not None:
        # A replacement tail must not resume old motion when the take ends.
        for fc in curves(action, ad.action_slot):
            if fc.modifiers:
                raise ValueError('接录的动作含曲线修改器，请先烘焙：' + fc.data_path)
            for key in fc.keyframe_points:
                if key.co.x < trim_from:
                    preserved.append((fc, key.co.x, key.handle_left.copy(), key.handle_right.copy(),
                                      key.handle_left_type, key.handle_right_type))
            for key in list(fc.keyframe_points)[::-1]:
                if key.co.x >= trim_from:
                    fc.keyframe_points.remove(key, fast=True)
            fc.update()
    previous = {}
    channels = {}
    for frame, values in samples:
        for path, value in values.items():
            value = tuple(value)
            if path.endswith('rotation_quaternion'):
                q = Quaternion(value).normalized()
                if path in previous and q.dot(previous[path]) < 0:
                    q.negate()
                previous[path] = q.copy()
                value = tuple(q)
            for i, v in enumerate(value):
                channels.setdefault((path, i), []).append((frame, v))
    for (path, index), series in channels.items():
        fc = action.fcurve_ensure_for_datablock(owner, path, index=index)
        if fc.modifiers or fc.mute:
            raise ValueError('受控曲线含修改器或静音，请先烘焙：' + path)
        a, b = series[0][0], series[-1][0]
        # Retain earlier/later curve values and all original handles outside the take.
        boundary = ([(a-1, fc.evaluate(a-1)), (b+1, fc.evaluate(b+1))]
                    if trim_from is None and len(fc.keyframe_points) else [])
        for k in list(fc.keyframe_points)[::-1]:
            if a <= k.co.x <= b:
                fc.keyframe_points.remove(k, fast=True)
        existing = {round(k.co.x, 6) for k in fc.keyframe_points}
        for f, v in boundary:
            if f not in existing:
                fc.keyframe_points.insert(f, v, options={'FAST'}).interpolation = 'LINEAR'
        for f, v in series:
            fc.keyframe_points.insert(f, v, options={'FAST'}).interpolation = 'LINEAR'
        fc.update()
    # Automatic handles may otherwise change when the old tail is removed.
    for fc, frame, left, right, left_type, right_type in preserved:
        key = next(k for k in fc.keyframe_points if abs(k.co.x-frame) < 1e-5)
        key.handle_left_type, key.handle_right_type = left_type, right_type
        key.handle_left, key.handle_right = left, right
    return action


def animation_state(obj):
    ad = obj.animation_data
    return (ad.action if ad else None,
            ad.action_slot.identifier if ad and ad.action_slot else None,
            [(t, t.mute) for t in ad.nla_tracks] if ad else [])


def restore_animation(obj, saved):
    action, slot, tracks = saved
    ad = obj.animation_data_create()
    ad.action = action
    if action and slot:
        ad.action_slot = next(s for s in action.slots if s.identifier == slot)
    for t, mute in tracks:
        t.mute = mute


def bind_camera_interval(scene, camera, start, end):
    """Existing camera markers must not override a committed camera retake."""
    markers = sorted((m for m in scene.timeline_markers if m.camera), key=lambda m: m.frame)
    if not markers:
        return
    after = scene.camera
    for marker in markers:
        if marker.frame <= end+1:
            after = marker.camera
    for marker in markers:
        if start <= marker.frame <= end:
            marker.camera = camera
    if not any(m.frame == start for m in markers):
        scene.timeline_markers.new('运镜开始 · '+camera.name, frame=start).camera = camera
    if not any(m.frame == end+1 for m in markers):
        scene.timeline_markers.new('运镜结束 · 恢复原机位', frame=end+1).camera = after


def secondary_solver(session):
    return physics_module().Runtime(session)


class DirectorSession:
    def __init__(self, scene, recording=False):
        self.scene, self.p = scene, scene.character_director
        self.rig = resolve_rig(scene)
        self.bones = bone_binding(self.rig)
        self.recording = recording
        self.body_enabled = self.p.record_target != 'CAMERA'
        self.camera_enabled = self.p.record_target != 'BODY'
        self.mode = self.p.start_mode if self.body_enabled else 'CURRENT'
        self.keep_upper = self.mode != 'NEW' and self.p.keep_upper
        self.frame = scene.frame_current
        self.start = (previous_take_end(scene, self.rig) if self.mode == 'APPEND' else
                      self.frame if self.mode == 'CURRENT' and self.body_enabled else
                      self.p.start_frame if recording or self.mode == 'NEW' or not self.body_enabled else self.frame)
        self.end = (self.start+self.p.duration_frames-1 if self.body_enabled and self.mode != 'NEW'
                    else self.p.end_frame)
        self.fps = scene.render.fps / scene.render.fps_base
        self.old_range = (scene.frame_start, scene.frame_end)
        self.old_camera = scene.camera
        self.camera_anim = animation_state(self.old_camera) if self.old_camera else (None, None, [])
        self.opening_camera = None
        self.old_physics = getattr(scene, 'wiggle_enable', None)
        self.secondary = None
        self.rig_anim = animation_state(self.rig)
        self.saved_pose = {b.name: (b.matrix_basis.copy(), b.rotation_mode) for b in self.rig.pose.bones}
        self.created_root = self.created_camera = False
        self.root = self.camera = None
        self.samples, self.camera_samples, self.root_samples = [], [], []
        self.events = []
        self.keys = set()
        self.t = self.distance = self.phase = self.weight = self.yaw = 0.
        self.velocity = Vector((0, 0, 0))
        self.offset = Vector((0, 0, 0))
        self.accumulator = 0.
        self.camera_yaw = self.camera_pitch = 0.
        self.camera_mode = self.p.camera_mode
        self.last_camera_mode = None
        self.preset_values = list(PRESETS[self.p.preset][1:])
        self.style_values = list(STYLES[self.p.preset])
        self.last_preset = self.p.preset
        self.gait_transition = 1.
        self.gait_opening = {}
        self.look_yaw = self.look_pitch = 0.
        self.motion_direction = Vector((0, -1, 0))
        self.closed = False
        self.audio_device = None
        self.audio_handles = {}
        self.control_camera = self.p.record_target == 'CAMERA'
        self.paused = False
        self.channels = set()
        self.original_names = {a.as_pointer() for a in bpy.data.actions}
        self.original_matrices = {}
        self.original_props = {}
        self.room_matrix = bpy.data.objects.get('Ellen Room Placement')
        self.room_start = self.room_matrix.matrix_world.copy() if self.room_matrix else None
        self._validate()
        physics_module()._busy.add(scene.as_pointer())
        try:
            self._prepare()
        except Exception:
            self.close(False)
            raise

    def _validate(self):
        if self.recording and self.end <= self.start:
            raise ValueError('结束帧必须大于开始帧')
        if getattr(getattr(self.scene, 'sam_ellen', None), 'busy', False):
            raise ValueError('请等待动捕生成完成后再操控')
        import sys
        xr = sys.modules.get('xr_ellen_bridge')
        sam = sys.modules.get('sam_ellen_bridge')
        if (xr and getattr(xr, '_runtime', None)) or (sam and getattr(sam, '_work', None)):
            raise ValueError('请先停止实时动捕或当前动捕任务')
        ad = self.rig.animation_data
        if self.body_enabled and ad and (ad.use_tweak_mode or any(not t.mute for t in ad.nla_tracks)):
            raise ValueError('请先退出 NLA 编辑，并将启用的 NLA 合成为独立动作')
        self.model = self.rig.parent if self.rig.parent and self.rig.parent.get('character_director_root') is None else self.rig
        if self.model.get('character_director_root'):
            self.model = self.rig
        # Imported project characters have one MMD model Empty above the armature.
        if self.rig.parent and self.rig.parent.type == 'EMPTY' and not self.rig.parent.get('character_director_root'):
            self.model = self.rig.parent
        self.previous_parent = self.model.parent
        self.previous_parent_inverse = self.model.matrix_parent_inverse.copy()
        self.previous_model_basis = self.model.matrix_basis.copy()
        if self.body_enabled and self.model.constraints:
            raise ValueError('模型根带约束，试用版暂不支持；请选择项目原始角色场景')

    def _prepare(self):
        if self.old_physics is not None:
            self.scene.wiggle_enable = False
        self.scene.frame_set(self.start)
        if self.body_enabled:
            existing = self.model.parent
            if existing and existing.get('character_director_root') == self.rig:
                self.root = existing
            else:
                self.root = bpy.data.objects.new('人物移动 · ' + self.rig.name, None)
                self.scene.collection.objects.link(self.root)
                self.root.empty_display_type = 'ARROWS'
                self.root.empty_display_size = .22
                self.root['character_director_root'] = self.rig
                self.created_root = True
                world = self.model.matrix_world.copy()
                self.root.parent = self.previous_parent
                self.root.matrix_world = world
                self.model.parent = self.root
                self.model.matrix_parent_inverse = Matrix.Identity(4)
                self.model.matrix_world = world
            self.root_anim = animation_state(self.root)
            self.root_basis = self.root.matrix_basis.copy()
            if not self.root.get('character_director_origin'):
                # Upgrade 0.1 takes using their first recorded placement.
                source = { (fc.data_path, fc.array_index): fc
                           for fc in curves(self.root_anim[0], self.root.animation_data.action_slot
                                            if self.root.animation_data else None) }
                first = self.root_anim[0].frame_range[0] if self.root_anim[0] else self.start
                origin = sampled_basis(self.root, source, first)
                self.root['character_director_origin'] = [v for row in origin for v in row]
            if self.mode == 'NEW':
                values = self.root['character_director_origin']
                self.root.matrix_basis = Matrix([values[i:i+4] for i in range(0,16,4)])
                self.scene.view_layers[0].update()
            self.root_start = self.root.matrix_world.copy()
            self.root.rotation_mode = 'QUATERNION'
            # Root curves are applied manually during preview and restored on cancellation.
            self.root.animation_data_create().action = None
        self.scene.view_layers[0].update()
        self.rig_start = self.rig.matrix_world.copy()
        self.rig_relative = self.root_start.inverted() @ self.rig_start if self.body_enabled else None
        self.current_rig_world = self.rig_start.copy()
        self.scale = sum(self.rig_start.to_scale()) / 3
        self.height = max(.2, self.bones['head'].bone.head_local.z)
        self.forward = Vector((0, -1, 0))
        forward = self.rig_start.to_quaternion() @ self.forward
        self.base_heading = math.atan2(forward.x, -forward.y)
        self.last_frame_time = time.monotonic()
        self._controlled_bones()
        self.start_pose = {b.name: b.matrix_basis.copy() for b in self.rig.pose.bones}
        if self.body_enabled:
            self.rig.animation_data_create().action = None
            if self.mode == 'NEW':
                for bone in self.rig.pose.bones:
                    bone.matrix_basis = Matrix.Identity(4)
            # Keep a frozen opening pose. Old unowned head/finger curves must
            # not keep playing while a new body take is being recorded.
            self.frozen_pose = {b.name: b.matrix_basis.copy() for b in self.rig.pose.bones}
        head = self.bones['head']
        self.head_chain = list(reversed(head.parent_recursive)) + [head]
        if not self.body_enabled:
            self.evaluate_character()
        # Every take needs a viewing camera, including body-only recordings.
        # camera_enabled controls baking/retention, never the operator's view.
        self.camera = bpy.data.objects.new('拍摄相机 · ' + datetime.now().strftime('%H%M%S'),
            self.old_camera.data.copy() if self.old_camera else bpy.data.cameras.new('角色拍摄镜头'))
        self.scene.collection.objects.link(self.camera)
        self.camera['character_director_camera'] = True
        self.created_camera = True
        self.camera.rotation_mode = 'QUATERNION'
        self.camera.data.animation_data_clear()
        self.camera.data.lens = self.p.lens
        self.camera.data.shift_x = self.camera.data.shift_y = 0.
        self.camera.data.clip_start = .08*self.scale
        self.camera.data.dof.use_dof = False
        if self.old_camera:
            self.camera.matrix_world = self.old_camera.matrix_world.copy()
            direction = self.camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
            self.free_yaw = math.atan2(direction.x, -direction.y)
            self.free_pitch = math.asin(max(-1, min(1, direction.z)))
        else:
            self.free_yaw, self.free_pitch = self.base_heading, 0.
            self.camera.location = self.head_position()+Vector((0, 2*self.scale, .2*self.scale))
        self.scene.camera = self.camera
        physical = physics_module().migrate(self.scene)
        if self.body_enabled and physical.enabled and physical.during_control:
            self.secondary = secondary_solver(self)
            for bone in self.secondary.bones:
                if bone not in self.recorded_bones:
                    self.recorded_bones.append(bone)
        self.apply_pose(0)
        self.update_camera(0)
        if self.body_enabled and self.mode != 'NEW' and self.old_camera:
            self.camera.matrix_world = self.old_camera.matrix_world.copy()
            self.opening_camera = self.camera.matrix_world.copy()
        else:
            self.opening_camera = None
        self.scene.view_layers[0].update()
        if self.secondary:
            self.secondary.step(0)
            self.scene.view_layers[0].update()
        if self.recording:
            self.capture(self.start)
        self.p.running = True

    def _controlled_bones(self):
        roles = ['root', 'center', 'groove', 'waist', 'lower']
        for side in ('L', 'R'):
            roles += [x+side for x in ('leg', 'knee', 'ankle', 'ik', 'ikparent', 'toe')]
        if not self.keep_upper:
            roles += ['upper', 'upper2']
            for side in ('L', 'R'):
                roles += [x+side for x in ('shoulder', 'arm', 'elbow', 'wrist')]
        self.controlled = [self.bones[k] for k in roles if k in self.bones]
        # Supplemental MMD arm twist bones must not retain a previous captured twist.
        if not self.keep_upper:
            for b in self.rig.pose.bones:
                jp = getattr(getattr(b, 'mmd_bone', None), 'name_j', '')
                if any(x in jp for x in ('腕捩', '手捩')) and b not in self.controlled:
                    self.controlled.append(b)
        # Capture look channels even when switching into first person mid-take.
        self.look_bones = [self.bones[k] for k in ('neck', 'head') if k in self.bones]
        self.recorded_bones = self.controlled + [b for b in self.look_bones if b not in self.controlled]
        self.mmd_loops = ({key: MMDLoop(self.rig, self.bones, key, self.height) for key in MMD_LOOPS}
                          if self.body_enabled else {})
        lower = {self.bones[k].name for k in roles if k in self.bones}
        for loop in self.mmd_loops.values():
            for name in loop.samples:
                if self.keep_upper and name not in lower:
                    continue
                bone = self.rig.pose.bones[name]
                if bone not in self.controlled:
                    self.controlled.append(bone)
                if bone not in self.recorded_bones:
                    self.recorded_bones.append(bone)

    def switch_camera(self, mode):
        if mode in {'FIRST', 'FACE'}:
            self.camera_yaw = self.yaw
            self.camera_pitch = 0.
        elif mode == 'FREE' and self.camera:
            direction = self.camera.rotation_quaternion @ Vector((0, 0, -1))
            self.free_yaw = math.atan2(direction.x, -direction.y)
            self.free_pitch = math.asin(max(-1, min(1, direction.z)))
        self.p.camera_mode = mode

    def mouse(self, dx, dy):
        sensitivity = self.p.mouse_sensitivity * .002
        if self.p.invert_y:
            dy = -dy
        if self.p.camera_mode == 'FREE':
            self.free_yaw -= dx * sensitivity
            self.free_pitch = max(-1.45, min(1.45, self.free_pitch + dy*sensitivity))
        else:
            self.camera_yaw -= dx*sensitivity
            limit = 1.25 if self.p.camera_mode == 'FIRST' else 1.
            self.camera_pitch = max(-limit, min(limit, self.camera_pitch + dy*sensitivity))

    def evaluate_character(self):
        """Use Blender's evaluated animation, including NLA, constraints and parents."""
        evaluated = self.rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
        self.current_rig_world = evaluated.matrix_world.copy()
        self.evaluated_pose = {b.name: b.matrix.copy() for b in evaluated.pose.bones}

    def adjust_framing(self, dt):
        if not self.body_enabled and self.p.camera_mode in {'FOLLOW', 'FACE'}:
            p = self.p
            speed = dt*p.fly_speed*(2 if self.keys & {'LEFT_SHIFT', 'RIGHT_SHIFT'} else 1)
            field = 'follow_distance' if p.camera_mode == 'FOLLOW' else 'face_distance'
            setattr(p, field, getattr(p, field)+speed*(int('S' in self.keys)-int('W' in self.keys)))
            p.shoulder += speed*(int('D' in self.keys)-int('A' in self.keys))*(-1 if p.camera_mode == 'FOLLOW' else 1)
            p.camera_height += speed*(int('E' in self.keys)-int('Q' in self.keys))

    def step(self, dt, inputs=None, evaluate=True):
        if inputs is not None:
            self.keys = set(inputs)
        if self.room_matrix and (self.room_matrix.matrix_world.translation-self.room_start.translation).length > 1e-5:
            raise ValueError('录制期间房间已切换，本次操作已取消；请重新进入操控')
        self.t += dt
        frame = self.start + round(self.t*self.fps)
        if evaluate or not self.body_enabled:
            self.scene.frame_set(frame)
            # Timeline camera markers may select a different render camera.
            self.scene.camera = self.camera
        target_preset = PRESETS[self.p.preset][1:]
        self.preset_values = [smooth(a, b, dt, .22) for a, b in zip(self.preset_values, target_preset)]
        self.style_values = [smooth(a, b, dt, .22) for a, b in zip(self.style_values, STYLES[self.p.preset])]
        speed_mul, cycle, lift, arm, sway = self.preset_values
        if self.body_enabled:
            axes = Vector((int('D' in self.keys)-int('A' in self.keys),
                           int('W' in self.keys)-int('S' in self.keys), 0))
            if self.control_camera:
                axes.zero()
            if axes.length > 1:
                axes.normalize()
            # Camera heading is independent from the character's facing direction.
            heading = self.free_yaw if self.p.camera_mode == 'FREE' else self.base_heading+self.camera_yaw
            forward = Vector((math.sin(heading), -math.cos(heading), 0))
            # FACE keeps the character facing the front camera: W advances
            # toward it; strafing/backward movement must not turn the back to it.
            right = forward.cross(Vector((0, 0, 1)))
            desired = (right*axes.x + forward*axes.y) * self.p.speed * speed_mul
            if 'LEFT_SHIFT' in self.keys or 'RIGHT_SHIFT' in self.keys:
                desired *= 1.65
            self.velocity = smooth(self.velocity, desired, dt, .12)
            delta = self.velocity*dt
            self.offset += delta
            actual = self.velocity.length
            if self.p.camera_mode in {'FIRST', 'FACE'} and not self.control_camera:
                difference = angle_delta(self.yaw, self.camera_yaw)
                limit = self.p.head_turn_limit
                # Moving aligns the torso with the view; at rest the head has
                # a comfortable free-look range before the torso catches up.
                turn = difference
                if self.p.camera_mode == 'FIRST' and axes.length <= .01:
                    turn = math.copysign(max(0., abs(difference)-limit), difference)
                self.yaw += turn*(1-math.exp(-dt/self.p.body_turn_time))
            elif actual > .025 and self.p.camera_mode not in {'FIRST', 'FACE'}:
                desired_yaw = math.atan2(self.velocity.x, -self.velocity.y)-self.base_heading
                self.yaw += angle_delta(self.yaw, desired_yaw)*(1-math.exp(-dt/.12))
            if actual > .025:
                world_to_body = Quaternion((0, 0, 1), -(self.base_heading+self.yaw))
                self.motion_direction = world_to_body @ self.velocity.normalized()
            self.weight = smooth(self.weight, min(1., actual/.18), dt, .15)
            self.distance += delta.length
            body_scale = self.height/1.49 if self.p.preset in MMD_LOOPS else 1.
            self.phase += delta.length / max(.1, cycle*self.scale*body_scale*self.p.stride) * self.p.pace
            self.root.matrix_world = Matrix.LocRotScale(
                self.root_start.translation+self.offset,
                Quaternion((0, 0, 1), self.yaw) @ self.root_start.to_quaternion(),
                self.root_start.to_scale())
            self.current_rig_world = self.root.matrix_world @ self.rig_relative
            self.apply_pose(dt)
        else:
            self.evaluate_character()
            self.adjust_framing(dt)
        if self.secondary:
            # Resolve leg IK before the cloth collision pass. The legacy
            # frame handler cannot see these newly generated control poses.
            self.scene.view_layers[0].update()
            self.secondary.step(dt)
        self.update_camera(dt)
        if self.recording:
            self.capture(frame)
        if evaluate:
            self.scene.view_layers[0].update()

    def apply_pose(self, dt):
        if not self.body_enabled:
            return
        if self.last_preset != self.p.preset:
            self.gait_opening = {b.name: b.matrix_basis.copy() for b in self.controlled}
            self.gait_transition = 0.
            self.last_preset = self.p.preset
        self.gait_transition = min(1., self.gait_transition+dt/.22)
        for b in self.rig.pose.bones:
            b.matrix_basis = self.frozen_pose[b.name]
        for b in self.controlled:
            b.rotation_mode = 'QUATERNION'
            b.matrix_basis = Matrix.Identity(4)
        if self.p.preset in self.mmd_loops:
            self.apply_mmd_pose()
            self.finish_pose(dt)
            return
        _, cycle, lift, arm, sway = self.preset_values
        hscale = self.height/1.49
        w = self.weight
        amp = self.p.amplitude
        phase = self.phase+.3
        stance, bounce, lean, elbow, width, roll = self.style_values
        bob = (1-math.cos(phase*TAU*2))*.5*bounce*hscale*w*amp
        lateral = math.sin(phase*TAU)*sway*hscale*w*amp
        # Slight knee bend keeps the IK chain away from a straight-leg singularity.
        self.bones['center'].location = local_translation(self.bones['center'], (lateral, 0, -.025*hscale+bob))
        if 'lower' in self.bones:
            b = self.bones['lower']
            b.rotation_quaternion = local_rotation(b, Quaternion((0, 1, 0), math.sin(phase*TAU)*roll*w*amp))
        for side, off in (('L', 0), ('R', .5)):
            stride, clearance = foot_cycle(phase+off, stance)
            b = self.bones['ik'+side]
            direction = self.motion_direction if self.p.camera_mode in {'FIRST', 'FACE'} else self.forward
            offset = direction*(stride*cycle*self.p.stride*hscale*w)
            offset.z = clearance*lift*hscale*w*amp
            offset.x += (1 if side == 'L' else -1)*width*hscale*w*amp
            b.location = local_translation(b, offset)
        if not self.keep_upper:
            for side, sign in (('L', 1), ('R', -1)):
                b = self.bones.get('arm'+side)
                if b:
                    swing = math.sin(phase*TAU)*arm*sign*w*amp
                    desired = Vector((.17*sign, swing, -1)).normalized()
                    rest_dir = (b.bone.tail_local-b.bone.head_local).normalized()
                    b.rotation_quaternion = local_rotation(b, rest_dir.rotation_difference(desired))
                    forearm = self.bones.get('elbow'+side)
                    if forearm:
                        # Bend about the arm's rest-space lateral hinge. The
                        # upper arm first rotates down from the MMD T pose.
                        axis = rest_dir.cross(Vector((0, -1, 0))).normalized()
                        forearm.rotation_quaternion = local_rotation(forearm, Quaternion(axis, elbow*w*amp))
            if 'upper' in self.bones:
                b = self.bones['upper']
                b.rotation_quaternion = local_rotation(b, Euler((-lean*w*amp,
                    -math.sin(phase*TAU)*roll*.6*w*amp, math.sin(phase*TAU)*.04*w*amp)).to_quaternion())
        self.finish_pose(dt)

    def apply_mmd_pose(self):
        loop = self.mmd_loops[self.p.preset]
        for b in self.controlled:
            if b.name not in loop.samples:
                continue
            loc, rot = loop.sample(b.name, self.phase)
            idle_loc, idle_rot = loop.idle[b.name]
            if b in (self.bones['ikL'], self.bones['ikR']):
                world = b.bone.matrix_local.to_3x3() @ loc
                forward = world.y*self.p.stride
                if self.p.camera_mode in {'FIRST', 'FACE'}:
                    world.x -= self.motion_direction.x*forward
                    world.y = -self.motion_direction.y*forward
                else:
                    world.y = forward
                loc = local_translation(b, world)
            b.location = idle_loc.lerp(loc, self.weight*self.p.amplitude)
            b.rotation_quaternion = idle_rot.slerp(rot, self.weight*self.p.amplitude)

    def finish_pose(self, dt):
        blend = min(1., self.t/.25) if self.mode != 'NEW' else 1.
        blend = blend*blend*(3-2*blend)
        for b in self.controlled:
            result = b.matrix_basis.copy()
            if blend < 1:
                loc, rot, scale = self.start_pose[b.name].decompose()
                loc2, rot2, scale2 = result.decompose()
                result = Matrix.LocRotScale(loc.lerp(loc2, blend), rot.slerp(rot2, blend), scale.lerp(scale2, blend))
            if self.gait_transition < 1 and b.name in self.gait_opening:
                loc, rot, scale = self.gait_opening[b.name].decompose()
                loc2, rot2, scale2 = result.decompose()
                t = self.gait_transition
                t = t*t*(3-2*t)
                result = Matrix.LocRotScale(loc.lerp(loc2, t), rot.slerp(rot2, t), scale.lerp(scale2, t))
            b.rotation_mode = self.saved_pose[b.name][1]
            b.matrix_basis = result
        self.apply_look(dt)

    def bone_pose_matrix(self, target):
        if not self.body_enabled:
            return self.evaluated_pose[target.name].copy()
        matrix = None
        frame = self.start+round(self.t*self.fps)
        for bone in list(reversed(target.parent_recursive)) + [target]:
            basis = bone.matrix_basis.copy()
            kwargs = {'parent_matrix': matrix, 'parent_matrix_local': bone.parent.bone.matrix_local} if bone.parent else {}
            matrix = bone.bone.convert_local_to_pose(basis, bone.bone.matrix_local, **kwargs)
        return matrix

    def apply_look(self, dt):
        if self.p.camera_mode != 'FIRST':
            return
        target_yaw = max(-1.3, min(1.3, angle_delta(self.yaw, self.camera_yaw)))
        self.look_yaw = smooth(self.look_yaw, target_yaw, dt, .045) if dt else target_yaw
        self.look_pitch = smooth(self.look_pitch, self.camera_pitch, dt, .045) if dt else self.camera_pitch
        # Rotate neck gently, then solve the head orientation against its actual
        # posed parent. This compensates gait lean and arbitrary MMD bone axes.
        neck = self.bones.get('neck')
        if neck:
            rotation = Quaternion((0,0,1), self.look_yaw*.3) @ Quaternion((1,0,0), -self.look_pitch*.3)
            neck.matrix_basis = self.frozen_pose[neck.name] @ local_rotation(neck, rotation).to_matrix().to_4x4()
        head = self.bones['head']
        yaw = self.base_heading+self.yaw+self.look_yaw
        world = Quaternion((0,0,1), yaw) @ Quaternion((1,0,0), -self.look_pitch)
        desired = self.current_rig_world.to_quaternion().inverted() @ world @ head.bone.matrix_local.to_quaternion()
        posed = self.bone_pose_matrix(head)
        loc, _, scale = posed.decompose()
        kwargs = {'parent_matrix': self.bone_pose_matrix(head.parent), 'parent_matrix_local': head.parent.bone.matrix_local} if head.parent else {}
        head.matrix_basis = head.bone.convert_local_to_pose(Matrix.LocRotScale(loc, desired, scale), head.bone.matrix_local, invert=True, **kwargs)
        if self.mode != 'NEW' and self.t < .25:
            blend = min(1., self.t/.25)
            blend = blend*blend*(3-2*blend)
            for bone in self.look_bones:
                loc, rot, scale = self.start_pose[bone.name].decompose()
                loc2, rot2, scale2 = bone.matrix_basis.decompose()
                bone.matrix_basis = Matrix.LocRotScale(loc.lerp(loc2, blend), rot.slerp(rot2, blend), scale.lerp(scale2, blend))

    def head_position(self):
        """Evaluate the head's FK chain without deforming all scene meshes per sample."""
        return self.current_rig_world @ self.bone_pose_matrix(self.bones['head']).translation

    def update_camera(self, dt):
        if self.camera is None:
            return
        p, camera = self.p, self.camera
        mode = p.camera_mode
        camera.data.lens = p.lens
        if mode == 'FIRST':
            head = self.bones['head']
            deform = self.current_rig_world @ self.bone_pose_matrix(head) @ head.bone.matrix_local.inverted()
            eye = head.bone.head_local + Vector((0, -p.eye_forward, p.eye_height))
            camera.location = deform @ eye
            if self.body_enabled:
                heading = self.base_heading+self.camera_yaw
                direction = Vector((math.sin(heading)*math.cos(self.camera_pitch),
                    -math.cos(heading)*math.cos(self.camera_pitch), math.sin(self.camera_pitch)))
            else:
                direction = deform.to_quaternion() @ Vector((math.sin(self.camera_yaw)*math.cos(self.camera_pitch),
                    -math.cos(self.camera_yaw)*math.cos(self.camera_pitch), math.sin(self.camera_pitch)))
            camera.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
        elif mode == 'FREE':
            # Explicit view direction is more robust than assuming Blender camera Euler axes.
            direction = Vector((math.sin(self.free_yaw)*math.cos(self.free_pitch),
                                -math.cos(self.free_yaw)*math.cos(self.free_pitch), math.sin(self.free_pitch)))
            camera.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
            if self.control_camera and dt:
                right = Vector((-math.cos(self.free_yaw), -math.sin(self.free_yaw), 0))
                velocity = direction*(int('W' in self.keys)-int('S' in self.keys))
                velocity += right*(int('D' in self.keys)-int('A' in self.keys))
                velocity.z += int('E' in self.keys)-int('Q' in self.keys)
                if velocity.length > 1:
                    velocity.normalize()
                camera.location += velocity * dt*p.fly_speed*(2 if self.keys & {'LEFT_SHIFT', 'RIGHT_SHIFT'} else 1)
        else:
            head = self.head_position()
            head += Vector((0, 0, .07*self.scale))
            heading = self.base_heading+self.camera_yaw
            if not self.body_enabled and p.camera_follow_heading:
                forward = self.current_rig_world.to_quaternion() @ self.forward
                heading = math.atan2(forward.x, -forward.y)+self.camera_yaw
            front = Vector((math.sin(heading), -math.cos(heading), 0))
            right = Vector((math.cos(heading), math.sin(heading), 0))
            if mode == 'FACE':
                target = head + Vector((0, 0, p.camera_height))
                orbit = front*p.face_distance + right*p.shoulder
            else:
                target = self.current_rig_world.translation + Vector((0, 0, self.height*self.scale*.55+p.camera_height))
                orbit = -front*p.follow_distance + right*p.shoulder
            pitch = Quaternion(right, self.camera_pitch * (-1 if mode == 'FOLLOW' else 1))
            orbit = pitch @ orbit
            if mode == 'FACE' and p.follow_head_rotation:
                neutral = self.bones['head'].bone.matrix_local.to_quaternion()
                pose = self.bone_pose_matrix(self.bones['head']).to_quaternion()
                world = self.current_rig_world.to_quaternion()
                orbit = world @ (pose @ neutral.inverted()) @ world.inverted() @ orbit
            desired = target+orbit
            if dt and self.last_camera_mode == mode:
                camera.location = smooth(camera.location, desired, dt, p.camera_smoothing)
            else:
                camera.location = desired
            camera.rotation_quaternion = (target-camera.location).to_track_quat('-Z', 'Y')
        if self.opening_camera is not None and self.t < .35:
            blend = max(0., self.t/.35)
            blend = blend*blend*(3-2*blend)
            loc, rot, scale = self.opening_camera.decompose()
            loc += self.offset
            camera.location = loc.lerp(camera.location, blend)
            camera.rotation_quaternion = rot.slerp(camera.rotation_quaternion, blend)
        self.last_camera_mode = mode

    def capture(self, frame):
        if self.samples and self.samples[-1][0] == frame:
            return
        if not self.body_enabled and self.camera_samples and self.camera_samples[-1][0] == frame:
            self.camera_samples.pop()
            self.events.pop()
        if self.body_enabled:
            values = {}
            bones = self.rig.pose.bones if not self.samples else self.recorded_bones
            for b in bones:
                rotation = ('rotation_quaternion' if b.rotation_mode == 'QUATERNION' else
                            'rotation_axis_angle' if b.rotation_mode == 'AXIS_ANGLE' else 'rotation_euler')
                for prop in ('location', rotation, 'scale'):
                    values[b.path_from_id(prop)] = tuple(getattr(b, prop))
            if not self.samples:
                rotation = 'rotation_quaternion' if self.rig.rotation_mode == 'QUATERNION' else 'rotation_euler'
                for prop in ('location', rotation, 'scale'):
                    values[prop] = tuple(getattr(self.rig, prop))
            self.samples.append((frame, values))
            self.root_samples.append((frame, {'location': tuple(self.root.location),
                'rotation_quaternion': tuple(self.root.rotation_quaternion), 'scale': tuple(self.root.scale)}))
        if self.camera_enabled:
            self.camera_samples.append((frame, {'location': tuple(self.camera.location),
                'rotation_quaternion': tuple(self.camera.rotation_quaternion)}))
        self.events.append((frame, self.p.preset, sorted(self.keys), self.p.camera_mode))

    def sync_audio(self):
        """Monitor existing unmuted sound strips without modifying the sequencer."""
        if not self.p.play_audio or not self.scene.sequence_editor:
            return
        try:
            import aud
            if self.audio_device is None:
                self.audio_device = aud.Device()
            frame = self.start+self.t*self.fps
            editor = self.scene.sequence_editor
            strips = getattr(editor, 'strips_all', editor.strips)
            active = set()
            for strip in strips:
                if strip.type != 'SOUND' or strip.mute or not strip.sound:
                    continue
                if not strip.frame_final_start <= frame < strip.frame_final_end:
                    continue
                channel = editor.channels[strip.channel-1] if len(editor.channels) >= strip.channel else None
                if channel and channel.mute:
                    continue
                active.add(strip.name)
                position = max(0., (frame-strip.frame_start)/self.fps + getattr(strip, 'sound_offset', 0.))
                handle = self.audio_handles.get(strip.name)
                if handle is None or not handle.status:
                    sound = strip.sound.factory
                    if sound is None:
                        sound = aud.Sound(bpy.path.abspath(strip.sound.filepath))
                    handle = self.audio_device.play(sound)
                    self.audio_handles[strip.name] = handle
                    handle.position = position
                elif abs(handle.position-position) > .18:
                    handle.position = position
                handle.volume = strip.volume*self.scene.audio_volume
            for name in list(self.audio_handles):
                if name not in active:
                    self.audio_handles.pop(name).stop()
        except Exception:
            # Unavailable audio device/files must not discard a body/camera take.
            self.p.status = '正在操控；声音监听不可用，原声音轨道已保留'

    def close(self, commit=False):
        if self.closed:
            return
        self.closed = True
        for handle in self.audio_handles.values():
            handle.stop()
        self.audio_handles.clear()
        created_actions = []
        success = False
        try:
            if commit and self.recording:
                if max(len(self.samples), len(self.camera_samples)) < 2:
                    raise ValueError('录制不足两帧，已恢复原场景')
                label = '操控拍摄 ' + datetime.now().strftime('%m%d_%H%M%S')
                if self.body_enabled:
                    fresh = self.mode == 'NEW'
                    body = write_samples(self.rig, None if fresh else self.rig_anim[0],
                        None if fresh else self.rig_anim[1], self.samples, label+' · 身体',
                        trim_from=None if fresh else self.start)
                    created_actions.append(body)
                    root = write_samples(self.root, None if fresh else self.root_anim[0],
                        None if fresh else self.root_anim[1], self.root_samples, label+' · 移动',
                        trim_from=None if fresh else self.start)
                    created_actions.append(root)
                    self.p.last_body = body
                    self.p.last_root = self.root
                if self.camera_enabled:
                    continuation = self.body_enabled and self.mode != 'NEW'
                    camera = write_samples(self.camera, self.camera_anim[0] if continuation else None,
                        self.camera_anim[1] if continuation else None, self.camera_samples,
                        label+' · 镜头', trim_from=self.start if continuation else None)
                    created_actions.append(camera)
                    self.p.last_camera = self.camera
                end = max(x[-1][0] for x in (self.samples, self.camera_samples) if x)
                metadata = {'version': 7, 'rig': self.rig.name, 'start': self.start, 'end_frame': end,
                    'start_mode': self.mode,
                    'fps': self.fps, 'target': self.p.record_target, 'keep_upper': self.p.keep_upper,
                    'speed': self.p.speed, 'stride': self.p.stride, 'pace': self.p.pace,
                    'amplitude': self.p.amplitude, 'events': self.events,
                    'secondary_motion': bool(self.secondary and self.secondary.bones),
                    'motion_author': 'susuki', 'motion_files': {k: v[0] for k, v in MMD_LOOPS.items()}}
                for a in created_actions:
                    a['character_director_take'] = json.dumps(metadata, ensure_ascii=False)
                if self.body_enabled:
                    self.p.last_end = end
                    if self.secondary:
                        body['ellen_physics_bake'] = json.dumps({'start': self.start, 'end': end,
                            'profile': physics_module().fingerprint(self.scene)})
                if not self.body_enabled:
                    # The temporary camera is active now; use the original render
                    # camera as the fallback outside the recorded interval.
                    self.scene.camera = self.old_camera
                    bind_camera_interval(self.scene, self.camera, self.start, end)
                self.scene.frame_start = self.start if self.body_enabled and self.mode == 'NEW' else min(self.old_range[0], self.start)
                self.scene.frame_end = end if self.body_enabled else max(self.old_range[1], end)
                self.p.status = (f'已录制运镜 {end-self.start+1} 帧；人物动作保留，可再次录制新镜头'
                    if not self.body_enabled else f'已录制 {end-self.start+1} 帧；时间轴可播放，Ctrl+F12 渲染')
                success = True
            else:
                self.p.status = '已退出试走并恢复场景' if not self.recording else '已取消录制并恢复场景'
        finally:
            if not success:
                if self.body_enabled:
                    restore_animation(self.rig, self.rig_anim)
                    for b in self.rig.pose.bones:
                        matrix, mode = self.saved_pose[b.name]
                        b.rotation_mode, b.matrix_basis = mode, matrix
                    if self.root:
                        if self.created_root:
                            self.model.parent = self.previous_parent
                            self.model.matrix_parent_inverse = self.previous_parent_inverse
                            self.model.matrix_basis = self.previous_model_basis
                            bpy.data.objects.remove(self.root, do_unlink=True)
                        else:
                            restore_animation(self.root, self.root_anim)
                            self.root.matrix_basis = self.root_basis
                for a in list(bpy.data.actions):
                    if a.as_pointer() not in self.original_names:
                        bpy.data.actions.remove(a)
                self.scene.frame_start, self.scene.frame_end = self.old_range
            # Body-only takes use this camera for control, then restore the
            # user's render camera without adding or replacing camera keys.
            if not success or not self.camera_enabled:
                self.scene.camera = self.old_camera
                if self.created_camera and self.camera:
                    data = self.camera.data
                    bpy.data.objects.remove(self.camera, do_unlink=True)
                    if not data.users:
                        bpy.data.cameras.remove(data)
            self.scene.frame_set(self.start if success else self.frame)
            self.scene.camera = self.camera if success and self.camera_enabled else self.old_camera
            if self.secondary:
                self.secondary.close(success)
            if self.old_physics is not None:
                self.scene.wiggle_enable = self.old_physics
            self.p.running = False
            physics_module()._busy.discard(self.scene.as_pointer())


class SecondaryBake:
    """Replace only the secondary curves of an existing animated take."""
    def __init__(self, scene, rig=None):
        from types import SimpleNamespace
        self.scene = scene
        self.p = getattr(scene, 'character_director', None) or SimpleNamespace(last_body=None, status='', running=False)
        self.rig = rig or physics_module().rig_for(scene)
        self.bones = physics_module().bone_binding(self.rig)
        ad = self.rig.animation_data
        if not ad or not ad.action:
            raise ValueError('请先录制人物动作，再更新裙摆与尾巴')
        if ad.use_tweak_mode or any(not t.mute for t in ad.nla_tracks):
            raise ValueError('请先将 NLA 合成为独立动作')
        self.rig_anim = animation_state(self.rig)
        self.saved_pose = {b.name: (b.matrix_basis.copy(), b.rotation_mode) for b in self.rig.pose.bones}
        self.old_physics = getattr(scene, 'wiggle_enable', None)
        self.frame = scene.frame_current
        self.start, self.end = scene.frame_start, scene.frame_end
        self.at, self.samples, self.closed = self.start, [], False
        self.mode = 'NEW'
        self.secondary = None
        self.owns_busy = scene.as_pointer() not in physics_module()._busy
        physics_module()._busy.add(scene.as_pointer())
        try:
            if self.old_physics is not None:
                scene.wiggle_enable = False
            scene.frame_set(self.start)
            scene.view_layers[0].update()
            self.current_rig_world = self.rig.matrix_world.copy()
            self.scale = sum(self.current_rig_world.to_scale())/3
            self.secondary = secondary_solver(self)
            if not self.secondary.bones:
                raise ValueError('当前模型未找到启用的人物物理骨骼')
            self.p.running = True
        except Exception:
            self.close(False)
            raise

    def step(self):
        self.scene.frame_set(self.at)
        self.current_rig_world = self.rig.matrix_world.copy()
        self.secondary.step(0 if self.at == self.start else self.scene.render.fps_base/self.scene.render.fps)
        values = {}
        for b in self.secondary.bones:
            rotation = ('rotation_quaternion' if b.rotation_mode == 'QUATERNION' else
                        'rotation_axis_angle' if b.rotation_mode == 'AXIS_ANGLE' else 'rotation_euler')
            for prop in ('location', rotation, 'scale'):
                values[b.path_from_id(prop)] = tuple(getattr(b, prop))
        self.samples.append((self.at, values))
        self.at += 1
        return self.at > self.end

    def close(self, commit=False):
        if self.closed:
            return
        self.closed = True
        success, created = False, None
        original = {a.as_pointer() for a in bpy.data.actions}
        try:
            if commit:
                created = write_samples(self.rig, self.rig_anim[0], self.rig_anim[1], self.samples,
                self.rig_anim[0].name+' · 人物物理')
                created['character_director_secondary'] = json.dumps({'start': self.start, 'end': self.end})
                created['ellen_physics_bake'] = json.dumps({'start': self.start, 'end': self.end,
                    'profile': physics_module().fingerprint(self.scene)})
                self.p.last_body = created
                success = True
                self.p.status = f'已更新第 {self.start}—{self.end} 帧的人物物理'
            else:
                self.p.status = '已取消物理更新并恢复原动作'
        finally:
            if not success:
                restore_animation(self.rig, self.rig_anim)
                for a in list(bpy.data.actions):
                    if a.as_pointer() not in original:
                        bpy.data.actions.remove(a)
            for b in self.rig.pose.bones:
                b.matrix_basis, b.rotation_mode = self.saved_pose[b.name]
            if self.secondary:
                self.secondary.close(success)
            self.scene.frame_set(self.frame)
            if self.old_physics is not None:
                self.scene.wiggle_enable = self.old_physics
            self.p.running = False
            if self.owns_busy:
                physics_module()._busy.discard(self.scene.as_pointer())


class CDSettings(bpy.types.PropertyGroup):
    rig: PointerProperty(type=bpy.types.Object, name='人物骨架', poll=lambda self, obj: obj.type == 'ARMATURE')
    preset: EnumProperty(name='循环动作', items=[(k, v[0], '') for k, v in PRESETS.items()], default='WALK')
    speed: FloatProperty(name='移动速度 (米/秒)', description='自然走基准米/秒，其他步态按预设调整', default=1.0, min=.05, max=5.)
    stride: FloatProperty(name='步幅', default=1., min=.3, max=2.)
    pace: FloatProperty(name='步频倍率', description='1 为移动距离匹配；偏离 1 会产生有意的滑步风格', default=1., min=.3, max=2.)
    amplitude: FloatProperty(name='动作幅度', default=1., min=0., max=2.)
    keep_upper: BoolProperty(name='保留开头的上身姿势', description='接录时固定开头的上身姿势，只生成腿部和重心', default=False)
    # Compatibility aliases for older scripts; all UI and solvers use the
    # same scene profile rather than keeping a second copy in Director.
    secondary_motion: BoolProperty(get=lambda self: self.id_data.ellen_physics.during_control,
        set=lambda self, value: setattr(self.id_data.ellen_physics, 'during_control', value))
    skirt_clearance: FloatProperty(get=lambda self: self.id_data.ellen_physics.skirt_clearance,
        set=lambda self, value: setattr(self.id_data.ellen_physics, 'skirt_clearance', value))
    skirt_softness: FloatProperty(get=lambda self: self.id_data.ellen_physics.skirt_softness,
        set=lambda self, value: setattr(self.id_data.ellen_physics, 'skirt_softness', value))
    tail_inertia: FloatProperty(get=lambda self: self.id_data.ellen_physics.tail_inertia,
        set=lambda self, value: setattr(self.id_data.ellen_physics, 'tail_inertia', value))
    camera_mode: EnumProperty(name='相机模式', items=[('FIRST', '第一人称 · 眼位操控', '鼠标扭头，大角度带动身体转身'), ('FOLLOW', '外部跟拍 · 背后', ''), ('FACE', '外部跟拍 · 正面', '进入试走或录制后从正面拍脸；鼠标带动人物转向，W 向人物前方走'), ('FREE', '自由相机', '')], default='FIRST')
    head_turn_limit: FloatProperty(name='扭头后带动转身', description='站立时超过此角度，身体开始跟随；行走时自动朝向视线', default=math.radians(35), min=.05, max=1.2, subtype='ANGLE')
    body_turn_time: FloatProperty(name='身体转向平滑', default=.18, min=.05, max=.8, subtype='TIME')
    eye_height: FloatProperty(name='眼位高度（头骨以上）', default=.075, min=-.1, max=.3, subtype='DISTANCE')
    eye_forward: FloatProperty(name='眼位前移', description='按角色骨架尺寸调整；镜头近裁剪避免看到面部内部', default=.09, min=0., max=.3, subtype='DISTANCE')
    follow_distance: FloatProperty(name='跟随距离', default=2.8, min=.3, max=30, subtype='DISTANCE')
    face_distance: FloatProperty(name='贴脸距离', default=.65, min=.22, max=5, subtype='DISTANCE')
    camera_height: FloatProperty(name='构图高度偏移', default=0., min=-2, max=3, subtype='DISTANCE')
    shoulder: FloatProperty(name='左右偏移', default=.0, min=-3, max=3, subtype='DISTANCE')
    camera_smoothing: FloatProperty(name='跟随平滑', default=.10, min=.01, max=1., subtype='TIME')
    follow_head_rotation: BoolProperty(name='贴脸时随头转动', default=False)
    camera_follow_heading: BoolProperty(name='跟随人物转身', description='二次运镜时随原动画的身体朝向转动；关闭后保持世界方向，仍跟随位移', default=True)
    lens: FloatProperty(name='焦距 (mm)', default=35., min=12, max=150)
    fly_speed: FloatProperty(name='自由相机速度', default=2., min=.1, max=20.)
    mouse_sensitivity: FloatProperty(name='鼠标灵敏度', default=1., min=.1, max=4.)
    invert_y: BoolProperty(name='反转鼠标上下', default=False)
    record_target: EnumProperty(name='录制内容', items=[('BOTH', '人物 + 镜头', ''), ('BODY', '仅人物', ''), ('CAMERA', '仅镜头（播放原动作）', '')], default='BOTH')
    start_frame: IntProperty(name='开始帧', default=1, min=0)
    start_mode: EnumProperty(name='录制起点', items=[('NEW', '清空身体动作，重新录制', '从设定开始帧清空重录；原动作数据保留，Esc 可恢复'),
        ('APPEND', '接上一段结尾', '自动从上次录制末帧接录'),
        ('CURRENT', '从当前帧接录', '保留当前帧之前的动作，覆盖之后的部分')], default='NEW')
    duration_frames: IntProperty(name='本段帧数', default=450, min=2, max=100000)
    last_end: IntProperty(default=0)
    end_frame: IntProperty(name='结束帧', default=450, min=1)
    countdown: FloatProperty(name='录制倒计时', default=2., min=0., max=10.)
    play_audio: BoolProperty(name='监听时间轴声音', description='播放已有声音轨道，不改动声音文件；复杂变速声音建议录后核对', default=True)
    running: BoolProperty(default=False, options={'SKIP_SAVE'})
    status: StringProperty(default='选择人物后进入试走；首次支持 MMD 双腿 IK 角色')
    last_body: PointerProperty(type=bpy.types.Action)
    last_root: PointerProperty(type=bpy.types.Object)
    last_camera: PointerProperty(type=bpy.types.Object)


class CD_OT_Bind(bpy.types.Operator):
    bl_idname = 'character_director.bind'
    bl_label = '检查人物绑定'
    def execute(self, context):
        try:
            rig = resolve_rig(context.scene)
            bones = bone_binding(rig)
            p = context.scene.character_director
            p.rig = rig
            p.status = f'已绑定 {rig.name}，识别 {len(bones)} 个控制骨骼'
            self.report({'INFO'}, p.status)
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


class CD_OT_GaitPreview(bpy.types.Operator):
    bl_idname = 'character_director.gait_preview'
    bl_label = '预览所选动作（艾莲）'
    bl_description = '播放艾莲动作预览；实际操控可点“进入试走”，WASD 移动，Esc 恢复'

    def execute(self, context):
        from pathlib import Path
        folder = Path(__file__).resolve().parents[1]/'output/CharacterDirector'
        name = {'MMD_WALK04': '艾莲_MMD走路04_0.4.mp4', 'MMD_RUN02': '艾莲_MMD奔跑02_0.4.mp4',
                'MMD_WALK01': '艾莲_MMD走路01_0.4.1.mp4', 'MMD_RUN01': '艾莲_MMD奔跑01_0.4.1.mp4'}.get(context.scene.character_director.preset,
            '艾莲_四种步态对比_0.2.mp4')
        path = folder/name
        if not path.is_file():
            self.report({'INFO'}, '当前安装未附带艾莲预览视频，请打开动作素材预览链接')
            return {'CANCELLED'}
        bpy.ops.wm.path_open(filepath=str(path))
        return {'FINISHED'}


class CD_OT_CameraPass(bpy.types.Operator):
    bl_idname = 'character_director.camera_pass'
    bl_label = '为已有人物动画录运镜'
    bl_description = '切到仅镜头，采用当前时间轴范围；然后点预演或录制，人物原动作与移动路径保持原样'

    @classmethod
    def poll(cls, context):
        return active_session() is None

    def execute(self, context):
        scene, p = context.scene, context.scene.character_director
        p.record_target = 'CAMERA'
        p.start_frame, p.end_frame = scene.frame_start, scene.frame_end
        if p.camera_mode == 'FIRST':
            p.camera_mode = 'FOLLOW'
        p.status = '二次运镜：确认机位和帧范围，再点“预演运镜”或“录制运镜”'
        return {'FINISHED'}


class CD_OT_Control(bpy.types.Operator):
    bl_idname = 'character_director.control'
    bl_label = '进入角色操控'
    bl_description = '人物模式 WASD 行走；仅镜头模式播放原动画，WASD/QE 调机位，Space 暂停构图；Enter 保留，Esc 恢复'
    bl_options = {'REGISTER', 'UNDO'}
    recording: BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == 'VIEW_3D' and active_session() is None

    def invoke(self, context, event):
        global SESSION
        # With use_local_camera=False, SpaceView3D.camera resolves to scene.camera.
        # Snapshot it BEFORE _prepare installs a temporary scene camera; that
        # temporary datablock is removed when a rehearsal is cancelled.
        self.area = context.area
        self.space = context.space_data
        self.region3d = self.space.region_3d
        self.old_view = self.region3d.view_perspective
        self.old_space_camera = self.space.camera
        self.old_local_camera = self.space.use_local_camera
        self.wm, self.window = context.window_manager, context.window
        self.timer = None
        self.session = None
        try:
            if context.screen.is_animation_playing:
                bpy.ops.screen.animation_cancel(restore_frame=False)
            self.session = DirectorSession(context.scene, self.recording)
            SESSION = self.session
            CD_OT_Control.bind_view(self)
            self.origin = self.last_tick = time.monotonic()
            self.ready_at = self.origin+(self.session.p.countdown if self.recording else 0)
            self.timer = self.wm.event_timer_add(1/60, window=self.window)
            self.last_mouse = None
            # Infinite mouse wrapping avoids hitting the edge of the desktop.
            self.window.cursor_modal_set('CROSSHAIR')
            self.wm.modal_handler_add(self)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            if self.session:
                self.finish(context, False)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def bind_view(self):
        """Always look through the control camera, even when only baking body."""
        if self.session.camera:
            self.space.use_local_camera = True
            self.space.camera = self.session.camera
            self.region3d.view_perspective = 'CAMERA'

    def restore_view(self, commit):
        camera = self.session.camera
        try:
            retained = (commit and self.recording and self.session.camera_enabled
                        and camera and camera.name in self.session.scene.objects)
        except ReferenceError:
            retained = False
        if retained:
            self.space.camera = camera
            self.space.use_local_camera = False
            self.region3d.view_perspective = 'CAMERA'
        else:
            try:
                old = self.old_space_camera
                if old is not None:
                    old.name  # The user may have removed it before cancellation.
            except ReferenceError:
                old = self.session.scene.camera
            self.space.camera = old
            self.space.use_local_camera = self.old_local_camera
            self.region3d.view_perspective = self.old_view

    def finish(self, context, commit):
        global SESSION
        # Release ownership even if restoring a deleted viewport/camera fails.
        # A late modal callback from an older take must not clear a newer one.
        if SESSION is self.session:
            SESSION = None
        succeeded = commit
        try:
            self.session.close(commit)
        except Exception as e:
            succeeded = False
            self.report({'ERROR'}, str(e))
        finally:
            cleanup = [lambda: self.wm.event_timer_remove(self.timer) if self.timer else None,
                       self.window.cursor_modal_restore,
                       lambda: self.area.header_text_set(None),
                       lambda: self.restore_view(succeeded), self.area.tag_redraw,
                       lambda: setattr(self.session.p, 'running', False)]
            for operation in cleanup:
                try:
                    operation()
                except (ReferenceError, RuntimeError, TypeError) as e:
                    print('Character Director UI cleanup:', str(e))
            self.timer = None
        return {'FINISHED'} if succeeded else {'CANCELLED'}

    def modal(self, context, event):
        s = self.session
        if SESSION is not s or s.closed:
            return self.finish(context, False)
        try:
            if event.type == 'ESC':
                return self.finish(context, False)
            if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                return self.finish(context, self.recording)
            if event.type == 'WINDOW_DEACTIVATE':
                return self.finish(context, False)
            if event.type in {'W', 'A', 'S', 'D', 'Q', 'E', 'LEFT_SHIFT', 'RIGHT_SHIFT'}:
                if event.value == 'PRESS':
                    s.keys.add(event.type)
                elif event.value == 'RELEASE':
                    s.keys.discard(event.type)
            if event.value == 'PRESS' and not event.is_repeat:
                if event.type == 'SPACE' and not s.body_enabled:
                    s.paused = not s.paused
                    s.accumulator = 0.
                    for handle in s.audio_handles.values():
                        handle.stop()
                    s.audio_handles.clear()
                    self.last_tick = time.monotonic()
                numbers = {'ONE': 'WALK', 'TWO': 'JOG', 'THREE': 'TINY', 'FOUR': 'SWAY',
                           'FIVE': 'MMD_WALK04', 'SIX': 'MMD_RUN02',
                           'SEVEN': 'MMD_WALK01', 'EIGHT': 'MMD_RUN01'}
                if event.type in numbers and s.body_enabled:
                    s.p.preset = numbers[event.type]
                if event.type == 'F' and s.camera:
                    modes = ['FIRST', 'FOLLOW', 'FACE', 'FREE']
                    s.switch_camera(modes[(modes.index(s.p.camera_mode)+1)%len(modes)])
                    CD_OT_Control.bind_view(self)
                if event.type == 'V' and s.camera:
                    s.switch_camera('FOLLOW' if s.p.camera_mode == 'FIRST' else 'FIRST')
                    CD_OT_Control.bind_view(self)
                if event.type == 'TAB' and s.p.record_target == 'BOTH':
                    s.control_camera = not s.control_camera
                if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and s.p.camera_mode != 'FIRST':
                    factor = .9 if event.type == 'WHEELUPMOUSE' else 1.1
                    field = {'FOLLOW': 'follow_distance', 'FACE': 'face_distance', 'FREE': 'fly_speed'}[s.p.camera_mode]
                    setattr(s.p, field, getattr(s.p, field)*factor)
            if event.type == 'MOUSEMOVE':
                point = (event.mouse_x, event.mouse_y)
                if self.last_mouse:
                    dx, dy = point[0]-self.last_mouse[0], point[1]-self.last_mouse[1]
                    if abs(dx) < 200 and abs(dy) < 200:
                        s.mouse(dx, dy)
                self.last_mouse = point
                # Warp only inside the active viewport and ignore the warp delta.
                area = self.area
                if point[0] < area.x+15 or point[0] > area.x+area.width-15 or point[1] < area.y+15 or point[1] > area.y+area.height-15:
                    center = (area.x+area.width//2, area.y+area.height//2)
                    self.window.cursor_warp(*center)
                    self.last_mouse = center
            # Blender 5.2 Event has no public timer handle. Elapsed monotonic
            # time keeps extra timers from advancing the simulation twice.
            if event.type == 'TIMER':
                now = time.monotonic()
                elapsed = now-self.last_tick
                self.last_tick = now
                if now < self.ready_at:
                    if not s.body_enabled:
                        s.adjust_framing(min(elapsed, .1))
                        s.update_camera(min(elapsed, .1))
                        if self.recording:
                            s.capture(s.start)
                        self.area.tag_redraw()
                    self.area.header_text_set(f'准备录制 {self.ready_at-now:.1f} 秒 · Esc 取消')
                    return {'RUNNING_MODAL'}
                if s.paused:
                    s.adjust_framing(min(elapsed, .1))
                    s.update_camera(min(elapsed, .1))
                    if self.recording:
                        s.capture(s.start+round(s.t*s.fps))
                    self.area.header_text_set('暂停构图 | WASD / Q E 调机位 · 鼠标看向 · Space 继续 · Enter 完成 · Esc 取消')
                    self.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                if elapsed > 1.5:
                    self.report({'WARNING'}, '预览发生长时间卡顿，录制已结束；可降低视图复杂度再录')
                    return self.finish(context, self.recording)
                s.accumulator += elapsed
                # Deterministic sampling at scene fps; bounded backlog avoids freezing the editor.
                steps = min(int(s.accumulator*s.fps+1e-8), 60)
                for i in range(steps):
                    s.accumulator -= 1/s.fps
                    # Simulate/record every frame; deform meshes only for the last visible frame.
                    s.step(1/s.fps, evaluate=i == steps-1)
                    if not s.body_enabled and s.start+round(s.t*s.fps) >= s.end:
                        return self.finish(context, self.recording)
                    if self.recording and s.start+round(s.t*s.fps) >= s.end:
                        return self.finish(context, True)
                if s.accumulator > .5:
                    self.report({'WARNING'}, '场景无法实时预览，已结束当前录制；请降低视图复杂度')
                    return self.finish(context, self.recording)
                if steps:
                    s.sync_audio()
                target = '相机' if s.control_camera else '人物'
                state = '录制' if self.recording else '试走'
                if not s.body_enabled:
                    self.area.header_text_set(f'二次运镜 {s.start+round(s.t*s.fps)}/{s.end} 帧 | 人物播放原动画 | WASD / QE 调机位 · 鼠标运镜 · F 机位 · Space 暂停 · Enter 完成 · Esc 取消')
                else:
                    self.area.header_text_set(f'{state} {s.t:.1f}s | 控制：{target} | {PRESETS[s.p.preset][0]} | WASD 移动 · 鼠标运镜 · F 机位 · Tab 切控制 · Enter 完成 · Esc 取消')
                self.area.tag_redraw()
            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return self.finish(context, False)


class CD_OT_SecondaryBake(bpy.types.Operator):
    bl_idname = 'character_director.secondary_bake'
    bl_label = '仅更新人物物理关键帧'
    bl_description = '按时间轴范围补算头发、衣带、裙摆和尾巴，保留身体与镜头；Esc 取消'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return active_session() is None

    def invoke(self, context, event):
        global SESSION
        try:
            self.session = SecondaryBake(context.scene)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        SESSION = self.session
        self.wm = context.window_manager
        self.timer = self.wm.event_timer_add(.02, window=context.window)
        self.wm.progress_begin(self.session.start, self.session.end+1)
        self.wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self, commit):
        global SESSION
        if SESSION is self.session:
            SESSION = None
        try:
            self.session.close(commit)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            commit = False
        finally:
            self.wm.event_timer_remove(self.timer)
            self.wm.progress_end()
        return {'FINISHED'} if commit else {'CANCELLED'}

    def modal(self, context, event):
        if self.session.closed or event.type == 'ESC':
            return self.finish(False)
        if event.type == 'TIMER':
            try:
                until = time.monotonic()+.05
                while time.monotonic() < until:
                    if self.session.step():
                        return self.finish(True)
                self.wm.progress_update(self.session.at)
                self.session.p.status = f'正在更新人物物理：{self.session.at-self.session.start}/{self.session.end-self.session.start+1} 帧 · Esc 取消'
                if context.area:
                    context.area.tag_redraw()
            except Exception as e:
                self.report({'ERROR'}, str(e))
                return self.finish(False)
        return {'RUNNING_MODAL'}


class CD_PT_Main(bpy.types.Panel):
    bl_idname = 'CD_PT_character_director'
    bl_label = '角色操控与运镜 · 试用版 0.7'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '角色操控'
    bl_ui_units_x = 24
    def draw(self, context):
        p = context.scene.character_director
        layout = self.layout
        settings = layout.column()
        settings.enabled = not p.running
        settings.prop(p, 'rig')
        settings.operator('character_director.bind', icon='ARMATURE_DATA')
        box = settings.box()
        box.label(text='先录人物，再单独录运镜', icon='CAMERA_DATA')
        box.operator('character_director.camera_pass', icon='CAMERA_DATA')
        box.prop(p, 'record_target')
        if p.record_target == 'CAMERA':
            box.label(text='人物播放原动画，只保存新相机关键帧')
            box.label(text='每次生成独立镜头，原相机保留')
        box = settings.box()
        box.enabled = p.record_target != 'CAMERA'
        box.label(text='人物与循环动作', icon='OUTLINER_OB_ARMATURE')
        box.prop(p, 'preset')
        box.operator('character_director.gait_preview', icon='PLAY')
        if p.preset in MMD_LOOPS:
            box.label(text='susuki 原动作 · 已接入 WASD 操控')
        box.prop(p, 'speed')
        box.prop(p, 'stride')
        box.prop(p, 'pace')
        box.prop(p, 'amplitude')
        row = box.row()
        row.enabled = p.start_mode != 'NEW'
        row.prop(p, 'keep_upper')
        physics_module().draw(settings, context.scene)
        box = settings.box()
        box.label(text='拍摄相机', icon='CAMERA_DATA')
        box.prop(p, 'camera_mode')
        if p.camera_mode == 'FIRST':
            box.prop(p, 'head_turn_limit')
            box.prop(p, 'body_turn_time')
            box.prop(p, 'eye_height')
            box.prop(p, 'eye_forward')
            box.label(text='鼠标扭头 / 转身；WASD 前后横移')
            if p.record_target == 'CAMERA':
                box.label(text='仅镜头：随原动作头部拍摄，不修改人物')
        elif p.camera_mode == 'FOLLOW':
            box.prop(p, 'follow_distance')
        elif p.camera_mode == 'FACE':
            box.prop(p, 'face_distance')
            box.prop(p, 'follow_head_rotation')
            box.label(text='鼠标绕拍人物；W 拉近 / S 拉远' if p.record_target == 'CAMERA' else '鼠标带动人物转向；W 向前 / S 后退')
            if p.record_target == 'BOTH':
                box.label(text='Tab 切到相机可独立绕拍人物')
        else:
            box.prop(p, 'fly_speed')
        if p.camera_mode in {'FOLLOW', 'FACE'}:
            if p.record_target == 'CAMERA':
                box.prop(p, 'camera_follow_heading')
                box.label(text='W/S 距离 · A/D 横移 · Q/E 高度')
            box.prop(p, 'camera_height')
            box.prop(p, 'shoulder')
            box.prop(p, 'camera_smoothing')
        box.prop(p, 'lens')
        box.prop(p, 'mouse_sensitivity')
        box.prop(p, 'invert_y')
        box = settings.box()
        box.label(text='录制为可编辑关键帧', icon='REC')
        if p.record_target == 'BODY':
            box.label(text='仍从所选机位操控，仅保存人物关键帧')
        if p.record_target != 'CAMERA':
            box.prop(p, 'start_mode')
        if p.record_target == 'CAMERA' or p.start_mode == 'NEW':
            row = box.row(align=True)
            row.prop(p, 'start_frame')
            row.prop(p, 'end_frame')
        else:
            box.prop(p, 'duration_frames')
            if p.start_mode == 'APPEND':
                box.label(text=f'上一段结束：第 {p.last_end} 帧' if p.last_end else '将自动读取上一段录制的结束帧')
            else:
                box.label(text=f'从播放头第 {context.scene.frame_current} 帧接录')
        box.prop(p, 'countdown')
        box.prop(p, 'play_audio')
        row = box.row(align=True)
        row.operator('character_director.control', text='预演运镜' if p.record_target == 'CAMERA' else '从所选机位试走', icon='PLAY').recording = False
        row.operator('character_director.control', text='录制运镜' if p.record_target == 'CAMERA' else '开始录制', icon='REC').recording = True
        if p.record_target == 'CAMERA':
            box.label(text='Space 暂停构图 / 继续；Esc 恢复原镜头')
        layout.label(text=p.status, icon='INFO')
        helpbox = layout.box()
        for line in ('WASD 人物移动 · 鼠标运镜', '1 自然走 / 2 小跑 / 3 碎步 / 4 摇摆',
                     '5 MMD 走路04 / 6 MMD 奔跑02',
                     '7 MMD 走路01 / 8 MMD 奔跑01',
                     'V 第一人称/外部 · F 切全部机位', '滚轮调外部距离 · Shift 加速', 'Tab 切人物/相机 · 自由相机 Q/E 升降',
                     'Enter 结束录制 · Esc 取消并恢复', '试走退出后恢复；录制 Enter 后保留',
                     '平地试用；暂不检测墙壁和家具', '物理与 SAM 共用；可仅在渲染时启用'):
            helpbox.label(text=line)


@persistent
def before_load(_):
    global SESSION
    if SESSION:
        SESSION.close(False)
        SESSION = None


@persistent
def before_save(_):
    if SESSION:
        # Never serialize temporary preview hierarchy or unsaved partial poses.
        before_load(None)


CLASSES = (CDSettings, CD_OT_Bind, CD_OT_GaitPreview, CD_OT_CameraPass, CD_OT_Control, CD_OT_SecondaryBake, CD_PT_Main)


def draw_launcher(self, context):
    self.layout.popover(panel='CD_PT_character_director', text='角色操控', icon='ARMATURE_DATA')


def register():
    physics_module().register('director')
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.character_director = PointerProperty(type=CDSettings)
    bpy.app.handlers.load_pre.append(before_load)
    bpy.app.handlers.save_pre.append(before_save)
    bpy.types.VIEW3D_MT_editor_menus.append(draw_launcher)


def unregister():
    before_load(None)
    bpy.types.VIEW3D_MT_editor_menus.remove(draw_launcher)
    for handlers, callback in ((bpy.app.handlers.load_pre, before_load), (bpy.app.handlers.save_pre, before_save)):
        if callback in handlers:
            handlers.remove(callback)
    del bpy.types.Scene.character_director
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    physics_module().unregister('director')
