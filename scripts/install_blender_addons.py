"""Copy standalone plugins and shared helpers into an explicit Blender directory.

python scripts/install_blender_addons.py --addons-dir PATH
Restart Blender after installation. This never opens or saves a .blend file.
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def install(destination):
    destination=Path(destination).expanduser().resolve()
    destination.mkdir(parents=True,exist_ok=True)
    sources=sorted((ROOT/'tools').glob('*.py'))+[ROOT/'tools/sam_ellen_phone_grips.json']
    existing=[p for p in sources if (destination/p.name).is_file()]
    backup=destination/'mocap_install_backups'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if existing or (destination/'mocap_runtime.json').exists():
        backup.mkdir(parents=True)
        for p in existing:shutil.copy2(destination/p.name,backup/p.name)
        if (destination/'mocap_runtime.json').exists():
            shutil.copy2(destination/'mocap_runtime.json',backup/'mocap_runtime.json')
    for p in sources:shutil.copy2(p,destination/p.name)
    (destination/'mocap_runtime.json').write_text(json.dumps({'repository':str(ROOT)},ensure_ascii=False,indent=2),encoding='utf8')
    return {'copied':len(sources),'addons':str(destination),'repository':str(ROOT),'restart_required':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--addons-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(install(args.addons_dir),ensure_ascii=False,indent=2))
