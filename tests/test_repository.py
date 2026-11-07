import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


class RepositoryTests(unittest.TestCase):
    def test_production_snapshot_and_module_closure(self):
        validator=module('validate_sources',ROOT/'scripts/validate_sources.py')
        report=validator.validate()
        self.assertTrue(report['passed'])
        self.assertGreater(report['snapshot_files_verified'],200)

    def test_install_keeps_backup_and_points_to_checkout(self):
        installer=module('install_addons',ROOT/'scripts/install_blender_addons.py')
        with tempfile.TemporaryDirectory() as folder:
            addon=Path(folder)
            previous=b'previous local addon\n'
            (addon/'motion_tweak.py').write_bytes(previous)
            result=installer.install(addon)
            self.assertTrue(result['restart_required'])
            self.assertEqual((addon/'motion_tweak.py').read_bytes(),(ROOT/'tools/motion_tweak.py').read_bytes())
            backups=list((addon/'mocap_install_backups').glob('*/motion_tweak.py'))
            self.assertEqual(len(backups),1)
            self.assertEqual(backups[0].read_bytes(),previous)
            settings=json.loads((addon/'mocap_runtime.json').read_text(encoding='utf8'))
            self.assertEqual(Path(settings['repository']),ROOT)
            # Resolve deployment paths from installed files, outside the checkout.
            env=dict(os.environ)
            for key in ('MOCAP_ROOT','SAM3D_ROOT'):env.pop(key,None)
            result=subprocess.check_output([sys.executable,'-c',
                'import sys;sys.path.insert(0,sys.argv[1]);import mocap_paths as p;print(p.ROOT)',folder],env=env,text=True)
            self.assertEqual(Path(result.strip()),ROOT)

    def test_environment_overrides_are_shared_by_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'custom_worker.exe'
            env=dict(os.environ,MOCAP_ROOT=str(ROOT),MOCAP_MOTION_PYTHON=str(target))
            code='import sys;sys.path.insert(0,sys.argv[1]);import project_paths as p;print(p.MOTION_PYTHON)'
            result=subprocess.check_output([sys.executable,'-c',code,str(ROOT/'services/sam-3d-body/local_app')],env=env,text=True)
            self.assertEqual(Path(result.strip()),target)


if __name__=='__main__':unittest.main()
