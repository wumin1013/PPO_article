"""Restore exact archived experiment bytes into a new, isolated replay directory.

No source-hash normalization, working-tree rollback, training or evaluation is
performed. The archived runner computes its ordinary hash and checks its own
unchanged protocol. Evaluation can later run explicitly inside the replay root.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path('论文项目/revision_20260908/preview_direct_explore')
DEFAULT_TARGET = ROOT / '论文项目/revision_20260908/frozen_replay_explore'
DATA = Path('PPO_project/data/preview_v2')
HISTORICAL_CONFIG = Path('论文项目/generated/configs/full_method_snapshot.yaml')


def sha(content):
    return hashlib.sha256(content).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def source_digest(files):
    return sha(json.dumps(files, sort_keys=True).encode())


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or path.is_absolute() or ':' in name
            or any(part in ('..', '.git', '.agents', '.codex') for part in path.parts)):
        raise ValueError(f'Unsafe relative archive/copy path: {name}')
    return Path(*path.parts)


def archived_files(content, expected):
    """Require exactly the registered files and verify their original raw bytes."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        if len(names) != len(set(name.casefold() for name in names)) or set(names) != set(expected):
            raise ValueError('Source archive members differ from registered source_files')
        result = {}
        for name in names:
            safe_relative(name)
            if ((archive.getinfo(name).external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Source archive may not contain symbolic links')
            value = archive.read(name)
            if sha(value) != expected[name]:
                raise ValueError(f'Archived source bytes do not match provenance: {name}')
            result[name] = value
    return result


def new_target(root, target):
    root, target = Path(root).resolve(), Path(target).resolve()
    revision = root / '论文项目/revision_20260908'
    if not target.is_relative_to(revision) or target == revision or target.is_relative_to(root / STUDY):
        raise ValueError('Replay target must be a new separate directory inside 论文项目/revision_20260908')
    if target.exists():
        raise FileExistsError(f'Preserve existing replay target; choose a new directory: {target}')
    return target


def copy_plan(root, selected_run=None):
    root = Path(root).resolve()
    contents, inputs, roles = {}, {}, {}

    def read(relative, expected=None, role='frozen metadata'):
        name = Path(relative).as_posix()
        value = (root / safe_relative(name)).read_bytes()
        digest = sha(value)
        if expected is not None and digest != expected:
            raise ValueError(f'Input hash mismatch: {name}')
        contents[name], inputs[name], roles[name] = value, digest, role
        return value

    def json_file(relative, expected=None, role='frozen metadata'):
        return json.loads(read(relative, expected, role).decode('utf-8'))

    protocol = json_file(STUDY / 'protocol.json')
    freeze = json_file(STUDY / 'freeze.json')
    json_file(STUDY / 'selection.json', freeze['selection_sha256'])
    if (inputs[(STUDY/'protocol.json').as_posix()] != freeze['protocol_sha256']
            or protocol['source_sha256'] != freeze['source_sha256']
            or protocol['manifest_sha256'] != freeze['path_manifest_sha256']):
        raise ValueError('Frozen source/protocol/path metadata disagree')
    expected_runs = {f'formal/{method}_seed{seed}' for method in protocol['formal_variants']
                     for seed in protocol['formal_seeds']}
    if set(freeze['models']) != expected_runs or set(freeze['configs']) != expected_runs:
        raise ValueError('Freeze lacks the exact complete formal model/config matrix')
    if selected_run is not None and selected_run not in expected_runs:
        raise ValueError('Requested single run is not a frozen formal model')
    run_keys = sorted(expected_runs if selected_run is None else [selected_run])
    runtime_files, runtime_bytes = None, None
    for key in run_keys:
        run = STUDY / key
        provenance = json_file(run / 'provenance.json')
        training = json_file(run / 'training_summary.json')
        read(run / 'checkpoint.pt', freeze['models'][key], 'frozen checkpoint')
        read(run / 'config.yaml', freeze['configs'][key], 'frozen training config')
        source_files = provenance['source_files']
        if (provenance['source_sha256'] != freeze['source_sha256']
                or source_digest(source_files) != freeze['source_sha256']
                or provenance['config_sha256'] != freeze['configs'][key]
                or provenance['path_manifest_sha256'] != freeze['path_manifest_sha256']
                or training['checkpoint_sha256'] != freeze['models'][key]
                or training['steps'] != protocol['formal_steps']):
            raise ValueError(f'Frozen training/source provenance mismatch: {key}')
        restored = archived_files(read(run/'source.zip', role='original source archive'), source_files)
        if runtime_files is not None and (runtime_files != source_files or runtime_bytes != restored):
            raise ValueError('Frozen models do not share identical registered source bytes')
        runtime_files, runtime_bytes = source_files, restored
        for filename in ('launcher_source.py','launcher_provenance.json','launcher_invocations.jsonl'):
            if (root/run/filename).exists():
                read(run/filename, role='original CLI-only launcher provenance')
    for name, value in runtime_bytes.items():
        if name in contents:
            raise ValueError('Unexpected archive/metadata destination collision')
        contents[name], roles[name] = value, 'exact original source.zip member; encoding preserved'
    manifest = json_file(DATA/'manifest.json', freeze['path_manifest_sha256'], 'frozen path manifest')
    for record in manifest['paths']:
        relative = safe_relative(record['reference_file'])
        read(DATA/relative, record['reference_sha256'], 'frozen reference points')
        sidecar = DATA/'paths'/f"{record['id']}.json"
        if (root/sidecar).exists():
            read(sidecar, role='original path generation metadata')
    registration = json_file(STUDY/'regression_reference_provenance.json')
    if (registration['protocol_sha256'] != freeze['protocol_sha256']
            or registration['source_sha256'] != freeze['source_sha256']):
        raise ValueError('Historical reference registration belongs to another study')
    read(HISTORICAL_CONFIG, registration['historical_config_sha256'], 'historical configuration used by frozen config builder')
    for item in registration['references']:
        read(STUDY/safe_relative(item['file']), item['sha256'], 'registered historical reference points')
    if (root/STUDY/'protocol_errata.json').exists():
        errata = json_file(STUDY/'protocol_errata.json')
        if errata['frozen_protocol_sha256'] != freeze['protocol_sha256']:
            raise ValueError('Protocol errata belongs to another frozen protocol')
    # This existing CLI bridge is outside the frozen source map. Its actual
    # bytes are copied and separately declared, never included in old hashes.
    for name in ('preview_direct_explore_resume.py', 'preview_direct_explore_resume_batch.py'):
        read(Path('PPO_project/tools')/name, role='separately hashed CLI-only launcher; not frozen numerical source')
    return contents, dict(input_files=inputs, file_roles=roles, runtime_source_files=runtime_files,
        runtime_source_sha256=freeze['source_sha256'], copied_model_runs=run_keys,
        complete_frozen_matrix_runs=sorted(expected_runs), protocol_sha256=freeze['protocol_sha256'])


def prepare(target, selected_run=None, command=None, root=ROOT):
    root = Path(root).resolve()
    target = new_target(root, target)
    contents, meta = copy_plan(root, selected_run)
    target.mkdir(parents=True, exist_ok=False)
    for name, value in contents.items():
        path = target / safe_relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    (target/'replay_inputs').mkdir()
    helper = Path(__file__).read_bytes()
    (target/'replay_inputs/prepare_frozen_replay.py').write_bytes(helper)
    meta.update(schema_version='exact-frozen-replay-v1', prepared_utc=datetime.now(timezone.utc).isoformat(),
        original_workspace=str(root), replay_root=str(target),
        preparation_command=command, preparation_tool_sha256=sha(helper),
        copied_files={name:sha(value) for name,value in contents.items()},
        original_results_copied=False, original_paper_copied=False,
        working_tree_source_rollback=False, source_hash_normalization=False,
        notes='Old source bytes including original BOM are preserved only in this archival replay. Outputs from explicit future evaluation belong to this separate directory. Original manifest HEAD is retained in each provenance; git HEAD reported by a new run reflects its repository context.')
    (target/'replay_manifest.json').write_bytes(encoded(meta))
    command_lines = [
        "Import-Module 'D:/Anaconda/shell/condabin/Conda.psm1'",
        'conda activate PPO', "$env:PYTHONIOENCODING='utf-8'",
        f"Set-Location -LiteralPath '{str(target).replace(chr(39), chr(39)*2)}'",
        'python PPO_project/tools/preview_direct_explore_resume.py evaluate --run 论文项目/revision_20260908/preview_direct_explore/formal/direct_seed11 --split validation',
    ]
    (target/'REPLAY.md').write_text('# 冻结源码独立重放\n\n本目录没有论文正文或PDF，也未复制旧评价结果。原始数值源码逐字节来自 source.zip；旧编码只作为实验归档保留。当前工作区不被回滚，源码哈希不被替换或归一化。\n\n准备及复制命令、所有输入/输出哈希见 replay_manifest.json；校验结果见 replay_protocol_check.json。\n\n需要显式重放验证时执行：\n\n```powershell\n'+'\n'.join(command_lines)+'\n```\n\n该命令只重放已训练策略，产生此目录内新的3条验证路径结果。运行前检查 copied_model_runs 包含 formal/direct_seed11。若用其他选定模型准备目录，请替换 --run。训练未由准备工具执行；本目录不具备原长训练的随机数/优化器历史之外新增状态。\n',encoding='utf-8')
    return verify(target)


def verify(target):
    target = Path(target).resolve()
    meta = json.loads((target/'replay_manifest.json').read_text(encoding='utf-8'))
    for name, expected in meta['copied_files'].items():
        if sha((target/safe_relative(name)).read_bytes()) != expected:
            raise ValueError(f'Replay bytes changed: {name}')
    if sha((target/'replay_inputs/prepare_frozen_replay.py').read_bytes()) != meta['preparation_tool_sha256']:
        raise ValueError('Archived preparation helper changed')
    actual = {name:sha((target/safe_relative(name)).read_bytes()) for name in meta['runtime_source_files']}
    if actual != meta['runtime_source_files'] or source_digest(actual) != meta['runtime_source_sha256']:
        raise ValueError('Restored numerical runtime differs from frozen source')
    return dict(replay_root=str(target), files_verified=len(meta['copied_files']),
        numerical_source_files=len(actual), model_count=len(meta['copied_model_runs']),
        runtime_source_sha256=meta['runtime_source_sha256'], source_bytes_verified=True)


def check_protocol(target):
    target = Path(target).resolve()
    verified = verify(target)
    code = '''import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'PPO_project/tools'))
import preview_direct_explore as candidate
study=candidate.install()
study.activate_backend()
protocol=study.check_protocol()
source=study.source_state()
print(json.dumps(dict(check_protocol_passed=True,source_sha256=source['source_sha256'],
    git_head_context=source['head'],runtime_root=str(study.ROOT),
    imported_main_file=__import__('main').__file__,protocol_version=protocol['version'])))
'''
    command = [sys.executable, '-I', '-c', code]
    process = subprocess.run(command, cwd=target, capture_output=True, text=True, encoding='utf-8',
        env={**os.environ, 'PYTHONIOENCODING':'utf-8'}, check=False)
    result = dict(verified, executed_command=command, returncode=process.returncode,
        stdout=process.stdout, stderr=process.stderr, training_executed=False, evaluation_executed=False)
    if process.returncode == 0:
        payload = json.loads(process.stdout)
        if payload['source_sha256'] != verified['runtime_source_sha256'] or Path(payload['runtime_root']) != target:
            raise ValueError('Archived runner imported another runtime root or source')
        result.update(payload)
    (target/'replay_protocol_check.json').write_bytes(encoded(result))
    if process.returncode:
        raise RuntimeError(f"Archived check_protocol failed; see {target/'replay_protocol_check.json'}")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','verify','check-protocol'])
    parser.add_argument('--target',type=Path,default=DEFAULT_TARGET)
    parser.add_argument('--single-run',help='Optional registered formal/<method>_seed<n>; default copies all 20 frozen final models')
    args=parser.parse_args()
    if args.command=='prepare':
        value=prepare(args.target,args.single_run,command=['python',*sys.argv])
    elif args.single_run:
        parser.error('--single-run only applies to prepare')
    else:
        value=(check_protocol if args.command=='check-protocol' else verify)(args.target)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
