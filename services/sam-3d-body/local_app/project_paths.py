"""Expose repository deployment settings when app.py is run directly."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools'))
from mocap_paths import LOCAL_APP,MOTION_PYTHON,SAM3D_WEIGHTS,TOOLS,WILOR_PYTHON,WILOR_ROOT
