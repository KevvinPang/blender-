"""Run explicitly inside Blender after installing and enabling MMD Tools."""
import bpy

for module in ('wiggle_2','xr_ellen_bridge','sam_ellen_bridge','ellen_expression_timeline','motion_tweak'):
    bpy.ops.preferences.addon_enable(module=module)
bpy.ops.wm.save_userpref()
print('Mocap, expression timeline and independent motion tweak plugins enabled')
