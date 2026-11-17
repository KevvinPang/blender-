"""Keep room lights included in the studio's viewport preview."""
import bpy


def configure_viewports():
    changed = []
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            if space.local_view:
                region = next(r for r in area.regions if r.type == 'WINDOW')
                with bpy.context.temp_override(screen=screen, area=area, region=region):
                    result = bpy.ops.view3d.localview(frame_selected=False)
                if 'FINISHED' not in result or space.local_view:
                    raise RuntimeError('Could not leave local view: ' + screen.name)
                changed.append(screen.name)
            space.use_local_collections = False
            space.shading.use_scene_lights = True
            space.shading.use_scene_world = True
            space.shading.use_scene_lights_render = True
            space.shading.use_scene_world_render = True
            space.shading.type = 'RENDERED'
            area.tag_redraw()
    return changed
