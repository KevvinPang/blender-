"""Factory-startup registration check; does not change scenes or preferences."""
import importlib
import json
import sys
from pathlib import Path
import bpy

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tools'))
registered=[]
try:
    for name in ('wiggle_2','xr_ellen_bridge','sam_ellen_bridge','ellen_expression_timeline','motion_tweak'):
        mod=importlib.import_module(name)
        mod.register();registered.append(mod)
    assert hasattr(bpy.types.Scene,'sam_ellen')
    assert hasattr(bpy.types.Scene,'motion_tweak')
    assert bpy.context.scene.motion_tweak.edit_mode=='NEAR'
    print('MOCAP_REPOSITORY_BLENDER_PASS',json.dumps({m.__name__:m.bl_info['version'] for m in registered}))
finally:
    for mod in reversed(registered):mod.unregister()
