"""Frame-based exceptions to camera eye contact; drawing is strictly read-only."""
import math

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty


def merged_ranges(ranges):
    ordered = []
    for start, end in ranges:
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end)):
            raise ValueError('不看镜头的帧范围必须是有限数值')
        if end < start:
            raise ValueError('不看镜头的结束帧不能早于开始帧')
        ordered.append((start, end))
    result = []
    for start, end in sorted(ordered):
        if result and start <= result[-1][1] + 1:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def contact_weight(frame, ranges, fade):
    """One outside exceptions; relax and return within their inclusive bounds."""
    for start, end in ranges:
        # Timestamp conversion can land a few ulps past an integer boundary.
        if start - 1e-6 <= frame <= end + 1e-6:
            frame = max(start, min(end, frame))
            if fade <= 0:
                return 0.
            span = min(fade, (end - start) / 2 + 1)
            t = min(1., (frame - start + 1) / span, (end - frame + 1) / span)
            return 1 - t*t*(3 - 2*t)
    return 1.


def range_options(settings):
    return merged_ranges((r.start_frame, r.end_frame) for r in settings.gaze_away_ranges if r.enabled)


class SAMGazeRange(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='启用', default=True)
    start_frame: IntProperty(name='开始帧', default=1, min=-1048574, max=1048574)
    end_frame: IntProperty(name='结束帧', default=61, min=-1048574, max=1048574,
                           description='包含结束帧；范围内解除镜头注视，恢复原始眼睛姿态')


class SAM_UL_gaze_ranges(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, 'enabled', text='')
        row.label(text=f'{item.start_frame}–{item.end_frame} 帧')


class SAMGazeRangeEdit(bpy.types.Operator):
    bl_idname = 'sam_ellen.gaze_range'
    bl_label = '编辑不看镜头时间段'
    bl_options = {'REGISTER', 'UNDO'}
    operation: EnumProperty(items=[(v, v, '') for v in ('ADD', 'REMOVE', 'START', 'END', 'JUMP')])

    def execute(self, context):
        import sam_ellen_bridge as bridge
        if bridge._work:
            self.report({'ERROR'}, '请等待当前动捕任务完成')
            return {'CANCELLED'}
        scene = context.scene
        s = scene.sam_ellen
        if self.operation == 'ADD':
            item = s.gaze_away_ranges.add()
            item.start_frame = scene.frame_current
            item.end_frame = scene.frame_current + max(1, round(scene.render.fps / scene.render.fps_base))
            s.gaze_away_index = len(s.gaze_away_ranges) - 1
        else:
            if not 0 <= s.gaze_away_index < len(s.gaze_away_ranges):
                self.report({'ERROR'}, '请先添加或选中一个时间段')
                return {'CANCELLED'}
            item = s.gaze_away_ranges[s.gaze_away_index]
            if self.operation == 'REMOVE':
                s.gaze_away_ranges.remove(s.gaze_away_index)
                s.gaze_away_index = max(0, min(s.gaze_away_index, len(s.gaze_away_ranges) - 1))
            elif self.operation == 'START':
                item.start_frame = scene.frame_current
            elif self.operation == 'END':
                item.end_frame = scene.frame_current
            elif self.operation == 'JUMP':
                scene.frame_set(item.start_frame)
        return {'FINISHED'}


def draw_ranges(layout, settings):
    box = layout.box()
    box.label(text='不看镜头的时间段（帧）')
    if not any(r.enabled for r in settings.gaze_away_ranges):
        box.label(text='未设置：全程看向镜头', icon='INFO')
    row = box.row()
    row.template_list('SAM_UL_gaze_ranges', '', settings, 'gaze_away_ranges', settings, 'gaze_away_index', rows=2)
    buttons = row.column(align=True)
    buttons.operator('sam_ellen.gaze_range', text='', icon='ADD').operation = 'ADD'
    buttons.operator('sam_ellen.gaze_range', text='', icon='REMOVE').operation = 'REMOVE'
    if 0 <= settings.gaze_away_index < len(settings.gaze_away_ranges):
        item = settings.gaze_away_ranges[settings.gaze_away_index]
        row = box.row(align=True)
        row.prop(item, 'start_frame')
        row.operator('sam_ellen.gaze_range', text='', icon='TIME').operation = 'START'
        row = box.row(align=True)
        row.prop(item, 'end_frame')
        row.operator('sam_ellen.gaze_range', text='', icon='TIME').operation = 'END'
        box.operator('sam_ellen.gaze_range', text='跳到开始帧', icon='REW').operation = 'JUMP'
    box.prop(settings, 'gaze_away_fade')
    box.label(text='范围内恢复原眼姿，随头部转动')
    try:
        range_options(settings)
    except ValueError as error:
        box.label(text=str(error), icon='ERROR')


CLASSES = (SAMGazeRange, SAM_UL_gaze_ranges, SAMGazeRangeEdit)
