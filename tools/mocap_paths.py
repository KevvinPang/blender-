"""Repository paths shared by Blender and isolated inference workers.

Only deployment paths differ from the 2026-11-07 production snapshot. Capture,
retargeting, expression and motion-solving algorithms are preserved.
"""
import json
import os
from pathlib import Path


def _root():
    if os.environ.get('MOCAP_ROOT'):
        return Path(os.environ['MOCAP_ROOT']).expanduser().resolve()
    config=Path(__file__).with_name('mocap_runtime.json')
    if config.is_file():
        return Path(json.loads(config.read_text(encoding='utf8'))['repository']).resolve()
    return Path(__file__).resolve().parents[1]


ROOT=_root()
TOOLS=ROOT/'tools'


def _path(variable,default,legacy=None):
    configured=os.environ.get(variable)
    if configured:return Path(configured).expanduser().resolve()
    if default.exists() or legacy is None or not legacy.exists():return default
    return legacy


def python_in(folder):
    return folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


SAM_ROOT=_path('SAM3D_ROOT',ROOT/'services/sam-3d-body',Path.home()/'Desktop/sam-3d-body-main')
LOCAL_APP=SAM_ROOT/'local_app'
MOTION_PYTHON=_path('MOCAP_MOTION_PYTHON',python_in(ROOT/'.venvs/motion'),
                     python_in(Path.home()/'Desktop/小小布朗尼/tools/motion_venv'))
WILOR_ROOT=_path('WILOR_ROOT',ROOT/'externals/WiLoR',Path.home()/'Downloads/WiLoR-main')
WILOR_PYTHON=_path('MOCAP_WILOR_PYTHON',python_in(WILOR_ROOT/'.venv'))
SAM2_ROOT=_path('SAM2_ROOT',ROOT/'externals/sam2')
SAM2_CHECKPOINT=_path('SAM2_CHECKPOINT',ROOT/'models/sam2/sam2.1_hiera_tiny.pt')
SAM3D_WEIGHTS=_path('SAM3D_WEIGHTS',ROOT/'models/sam3d/sam_3d_body_dinov3_bf16.safetensors',
                      Path.home()/'Desktop/ssam/sam_3d_body_dinov3_bf16.safetensors')
