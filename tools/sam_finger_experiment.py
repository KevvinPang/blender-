"""Separate opt-in Blender panel; no hooks into production motion import."""
bl_info = {'name': 'SAM Finger Reliability Experiment', 'author': 'Local Tools',
           'version': (0, 1, 0), 'blender': (5, 2, 0), 'category': 'Animation'}

import json
from pathlib import Path
import subprocess
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, PointerProperty, StringProperty

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT/'tools/finger_reliability_experiment.py').is_file():
    ROOT = Path.home()/'Desktop/小小布朗尼'
_job = None


class FingerExperimentSettings(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='启用实验性检查（不修改动作）', default=False)
    take_path: StringProperty(name='待检查动作', subtype='FILE_PATH',
                              description='留空使用 SAM 动捕面板当前动作的数据')
    side: EnumProperty(name='检查手部', items=[('L', '人物左手', ''), ('R', '人物右手', ''),
                                              ('BOTH', '双手', '')], default='L')
    status: StringProperty(default='尚未运行')
    report_path: StringProperty(subtype='FILE_PATH')


def poll_job():
    global _job
    if not _job:
        return None
    job = _job
    if job['process'].poll() is None:
        return .3
    job['log'].close()
    scene = job['scene']
    _job = None
    if scene.name not in bpy.data.scenes:
        return None
    settings = scene.sam_finger_experiment
    if job['process'].returncode == 0:
        report = json.loads((job['output']/'report.json').read_text(encoding='utf8'))
        settings.report_path = str(job['output']/'report.html')
        settings.status = f"完成：{len(report['intervals'])} 个待复查片段，未改动动作"
    else:
        settings.status = '检查失败，请查看输出目录的 experiment.log'
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
    return None


class SAM_OT_finger_audit(bpy.types.Operator):
    bl_idname = 'sam_finger_experiment.audit'
    bl_label = '检查当前动作'
    bl_description = '独立离线检查，生成对照报告；不修正或替换动作'

    def execute(self, context):
        global _job
        s = context.scene.sam_finger_experiment
        if _job or not s.enabled:
            return {'CANCELLED'}
        bridge = getattr(context.scene, 'sam_ellen', None)
        raw_path = s.take_path or getattr(bridge, 'take_path', '')
        if not raw_path:
            self.report({'ERROR'}, '请先导入动作，或选择待检查动作的 take.json')
            return {'CANCELLED'}
        path = Path(bpy.path.abspath(raw_path))
        required = [path, path.with_name('hands_body.json'), path.with_name('wilor_predictions.json'),
                    path.with_name('alignment.json')]
        if not all(p.is_file() for p in required):
            self.report({'ERROR'}, '缺少原始 SAM、WiLoR 或二维缓存，请选择完整任务的 take.json')
            return {'CANCELLED'}
        output = ROOT/'ConversionWork/FingerReliabilityExperiment'/f'{time.time_ns()}'
        output.mkdir(parents=True)
        log = (output/'experiment.log').open('w', encoding='utf8')
        try:
            process = subprocess.Popen([str(ROOT/'tools/motion_venv/Scripts/python.exe'),
                str(ROOT/'tools/finger_reliability_experiment.py'), '--take', str(path),
                '--output', str(output), '--side', s.side], stdout=log, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as error:
            log.close()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        s.status = '正在检查，不影响动画播放……'
        s.report_path = ''
        _job = {'scene': context.scene, 'process': process, 'log': log, 'output': output}
        bpy.app.timers.register(poll_job, first_interval=.3)
        return {'FINISHED'}


class SAM_OT_finger_report(bpy.types.Operator):
    bl_idname = 'sam_finger_experiment.report'
    bl_label = '打开对照报告'

    def execute(self, context):
        path = Path(context.scene.sam_finger_experiment.report_path)
        if not path.is_file():
            self.report({'ERROR'}, '请先运行检查')
            return {'CANCELLED'}
        bpy.ops.wm.url_open(url=path.resolve().as_uri())
        return {'FINISHED'}


class SAM_PT_finger_experiment(bpy.types.Panel):
    bl_label = '手指可靠性实验'
    bl_idname = 'SAM_PT_finger_experiment'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SAM 动捕'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.sam_finger_experiment
        self.layout.prop(s, 'enabled')
        column = self.layout.column()
        column.enabled = s.enabled and not bool(_job)
        column.prop(s, 'take_path')
        column.prop(s, 'side')
        column.operator('sam_finger_experiment.audit', icon='VIEWZOOM')
        self.layout.label(text=s.status)
        if s.report_path:
            self.layout.operator('sam_finger_experiment.report', icon='FILE_FOLDER')


CLASSES = (FingerExperimentSettings, SAM_OT_finger_audit, SAM_OT_finger_report, SAM_PT_finger_experiment)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.sam_finger_experiment = PointerProperty(type=FingerExperimentSettings)


def unregister():
    global _job
    if _job:
        _job['process'].terminate()
        _job['process'].wait(timeout=5)
        _job['log'].close()
        _job = None
    if bpy.app.timers.is_registered(poll_job):
        bpy.app.timers.unregister(poll_job)
    del bpy.types.Scene.sam_finger_experiment
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
