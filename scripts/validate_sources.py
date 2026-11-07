"""Offline source validation; does not download weights or start inference."""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def validate():
    manifest=json.loads((ROOT/'docs/source_snapshot.json').read_text(encoding='utf8'))
    changed=set(manifest['deployment_only_changes'])
    for row in manifest['files']:
        data=(ROOT/row['path']).read_bytes()
        actual=hashlib.sha256(data).hexdigest()
        assert actual==row['repository_sha256'],row['path']
        if row['path'] not in changed:assert actual==row['source_sha256'],row['path']
    count=0
    for folder in ('tools','services','scripts','tests','externals/sam2'):
        for path in (ROOT/folder).rglob('*.py'):
            if '__pycache__' in path.parts:continue
            text=path.read_text(encoding='utf-8-sig')
            compile(text,str(path),'exec');count+=1
            assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,})',text),path
    modules={p.stem for p in (ROOT/'tools').glob('*.py')}
    for path in (ROOT/'tools').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            for name in names:
                base=name.split('.')[0]
                if base.startswith(('sam_ellen','ellen_expression','wilor_')):
                    assert base in modules,(path.name,base)
    for name in ('sam_ellen_bridge','ellen_expression_timeline','motion_tweak'):
        tree=ast.parse((ROOT/'tools'/f'{name}.py').read_text(encoding='utf8'))
        info=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bl_info' for t in n.targets))
        assert info['version']
    return {'passed':True,'python_files_compiled':count,'snapshot_files_verified':len(manifest['files']),'deployment_path_changes':len(changed)}


if __name__=='__main__':
    print(json.dumps(validate(),indent=2))
