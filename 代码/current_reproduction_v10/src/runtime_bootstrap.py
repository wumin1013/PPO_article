"""Import a byte-preserved runtime snapshot with absolute paths rebound in memory.

Physical algorithms and constants are unchanged. Original-source SHA checks continue
to see original bytes. Setup never writes to the original research directories.
"""
from pathlib import Path
import ast
import importlib.abc
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]/'runtime'
WORK = ROOT/'PPO_article_work'
BASE = WORK/'论文项目/joint_policy_20260915_v1'
RUNS = ROOT/'PPO_article_joint_runs/natural_closed_resume'
SEARCH = [RUNS/'compact_dense_research', BASE, RUNS]

class _Paths(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            normalized = node.value.replace('\\', '/')
            for old, new in [('D:/PPO_article_joint_runs', ROOT/'PPO_article_joint_runs'),
                             ('D:/PPO_article_work', WORK)]:
                if normalized == old or normalized.startswith(old+'/'):
                    return ast.copy_location(ast.Constant(new.as_posix()+normalized[len(old):]), node)
        return node

class _Loader(importlib.abc.Loader):
    def __init__(self, path): self.path = path
    def create_module(self, spec): return None
    def exec_module(self, module):
        module.__file__ = str(self.path)
        tree = ast.parse(self.path.read_text(encoding='utf-8-sig'), filename=str(self.path))
        tree = ast.fix_missing_locations(_Paths().visit(tree))
        exec(compile(tree, str(self.path), 'exec'), module.__dict__)

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if '.' in fullname: return None
        for directory in SEARCH:
            candidate = directory/(fullname+'.py')
            if candidate.is_file():
                return importlib.util.spec_from_loader(fullname, _Loader(candidate), origin=str(candidate))
        return None

def setup():
    if not (ROOT/'source_manifest.json').is_file():
        raise FileNotFoundError('Build the runtime snapshot with engineering/build_runtime_snapshot.py first')
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    for directory in [WORK/'PPO_project', *SEARCH]:
        if str(directory) not in sys.path: sys.path.insert(0, str(directory))
    return ROOT
