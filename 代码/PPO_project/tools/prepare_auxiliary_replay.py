"""Add archived auxiliary models and runners to an existing exact replay.

Existing files are verified and never overwritten. Numerical source comes
only from archived source.zip members and executed runner snapshots. No
training or policy evaluation is started by this preparation/check utility.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import prepare_frozen_replay as primary

ROOT = primary.ROOT
STUDY = primary.STUDY
V2 = Path('论文项目/revision_20260908/preview_v2')
SEEDS = (11, 22, 33, 44, 55)


def add_bytes(contents, name, value):
    name = primary.safe_relative(str(name).replace('\\', '/')).as_posix()
    if name in contents and contents[name] != value:
        raise ValueError(f'Conflicting archived bytes for one destination: {name}')
    contents[name] = value


def plan(root):
    root = Path(root).resolve()
    contents, inputs = {}, {}

    def read(relative, expected=None, destination=None):
        relative = primary.safe_relative(Path(relative).as_posix())
        data = (root / relative).read_bytes()
        digest = primary.sha(data)
        if expected is not None and digest != expected:
            raise ValueError(f'Auxiliary archive input hash mismatch: {relative}')
        inputs[relative.as_posix()] = digest
        add_bytes(contents, relative if destination is None else destination, data)
        return data

    def json_file(relative, expected=None):
        return json.loads(read(relative, expected).decode('utf-8'))

    final_freeze = json.loads((root / STUDY / 'freeze.json').read_text(encoding='utf-8'))
    # The archived common config builder briefly reads V1 training IDs before
    # V2 replaces them. Preserve this transitive data dependency explicitly.
    v1_data = Path('PPO_project/data/revision_20260908')
    v1_manifest = json_file(v1_data / 'manifest.json')
    for record in v1_manifest['paths']:
        read(v1_data / primary.safe_relative(record['reference_file']), record['reference_sha256'])
    model_keys = []
    groups = [(V2, {f'finetune/{method}_seed{seed}' for method in ('fixed', 'residual', 'absolute') for seed in SEEDS}),
              (V2 / 'nnc_auxiliary', {f'runs/nnc_seed{seed}' for seed in SEEDS})]
    for base, expected_keys in groups:
        frozen = json_file(base / 'freeze.json')
        if set(frozen['models']) != expected_keys or set(frozen['configs']) != expected_keys:
            raise ValueError('Auxiliary freeze lacks its full registered model/config roster')
        if frozen['path_manifest_sha256'] != final_freeze['path_manifest_sha256']:
            raise ValueError('Auxiliary and final references differ')
        for optional in ('protocol.json', 'selection.json'):
            if (root / base / optional).exists():
                read(base / optional)
        for key in sorted(expected_keys):
            run = base / key
            provenance = json_file(run / 'provenance.json')
            training = json_file(run / 'training_summary.json')
            read(run / 'checkpoint.pt', frozen['models'][key])
            read(run / 'config.yaml', frozen['configs'][key])
            if (provenance['source_sha256'] != frozen['source_sha256']
                    or primary.source_digest(provenance['source_files']) != frozen['source_sha256']
                    or provenance['config_sha256'] != frozen['configs'][key]
                    or provenance['path_manifest_sha256'] != frozen['path_manifest_sha256']
                    or training['checkpoint_sha256'] != frozen['models'][key]):
                raise ValueError(f'Auxiliary model provenance differs from freeze: {run}')
            archived = primary.archived_files(read(run / 'source.zip'), provenance['source_files'])
            for name, value in archived.items():
                add_bytes(contents, name, value)
            model_keys.append(run.as_posix())
    # Runners are recovered from the bytes actually used by the completed
    # matrix, not from potentially edited current working-tree scripts.
    transfer = STUDY / 'unified_baselines/final-matrix-v3/evaluate_test_nominal/absolute_seed11'
    transfer_protocol = json.loads((root / transfer / 'protocol.json').read_text(encoding='utf-8'))
    transfer_name = 'PPO_project/tools/preview_unified_baselines.py'
    read(transfer / 'evaluation_adapter.py', transfer_protocol['evaluation_adapter_hashes'][transfer_name], transfer_name)
    traditional = STUDY / 'two_step/test_nominal'
    traditional_protocol = json.loads((root / traditional / 'protocol.json').read_text(encoding='utf-8'))
    read(traditional / 'preview_two_step_eval_snapshot.py', traditional_protocol['script_sha256'],
         'PPO_project/tools/preview_two_step_eval.py')
    schedule = STUDY / 'unified_baselines/final-matrix-v3/evaluate_regression_nominal/scheduling'
    shard_provenance = json.loads((root / schedule / 'shard0_provenance.json').read_text(encoding='utf-8'))
    read(schedule / 'shard0_launcher.py', shard_provenance['launcher_sha256'], 'PPO_project/tools/preview_auxiliary_shards.py')
    for relative in (transfer / 'protocol.json', traditional / 'protocol.json', schedule / 'shard0_provenance.json'):
        read(relative, destination=Path('replay_inputs/auxiliary_evidence') / relative.name.replace('protocol.json',
             'traditional_protocol.json' if relative.parent == traditional else 'transfer_protocol.json'))
    return contents, dict(input_files=inputs, old_model_runs=model_keys,
                          source_policy='Only source.zip members and executed runner snapshots; no live numerical source copied')


def additive_copy(target, contents):
    """Preflight every destination, then add missing bytes with exclusive writes."""
    target = Path(target).resolve()
    missing, reused = [], []
    for name, data in contents.items():
        path = target / primary.safe_relative(name)
        if not path.resolve().is_relative_to(target):
            raise ValueError(f'Destination leaves replay root: {name}')
        if path.exists():
            if not path.is_file() or path.read_bytes() != data:
                raise FileExistsError(f'Preserve differing replay file; no files overwritten: {name}')
            reused.append(name)
        else:
            missing.append(name)
    for name in missing:
        path = target / primary.safe_relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(contents[name])
    return missing, reused


def verify(target):
    target = Path(target).resolve()
    base = primary.verify(target)
    meta = json.loads((target / 'auxiliary_replay_manifest.json').read_text(encoding='utf-8'))
    verified_files = dict(meta['verified_files'])
    for extension in sorted(target.glob('auxiliary_replay_extension_*.json')):
        extended = json.loads(extension.read_text(encoding='utf-8'))
        for name, expected in extended['verified_files'].items():
            if name in verified_files and verified_files[name] != expected:
                raise ValueError('Auxiliary extension conflicts with preserved replay bytes')
            verified_files[name] = expected
    for name, expected in verified_files.items():
        if primary.sha((target / primary.safe_relative(name)).read_bytes()) != expected:
            raise ValueError(f'Auxiliary replay bytes changed: {name}')
    return dict(base, auxiliary_model_count=len(meta['old_model_runs']),
                auxiliary_files_verified=len(verified_files),
                existing_files_overwritten=False, source_hash_normalization=False)


def prepare(target, root=ROOT):
    root, target = Path(root).resolve(), Path(target).resolve()
    if (not target.is_relative_to(root / '论文项目/revision_20260908')
            or target == root / '论文项目/revision_20260908'
            or target.is_relative_to(root / STUDY)):
        raise ValueError('Supplement an existing separate archival replay inside revision_20260908')
    primary.verify(target)
    manifest = target / 'auxiliary_replay_manifest.json'
    helper = Path(__file__).read_bytes()
    if manifest.exists():
        verify(target)
        manifest = target / f'auxiliary_replay_extension_{primary.sha(helper)[:12]}.json'
        if manifest.exists():
            return verify(target)
    contents, meta = plan(root)
    add_bytes(contents, f'replay_inputs/prepare_auxiliary_replay_{primary.sha(helper)[:12]}.py', helper)
    added, reused = additive_copy(target, contents)
    meta.update(schema_version='additive-auxiliary-replay-v1', prepared_utc=datetime.now(timezone.utc).isoformat(),
                preparation_command=['python', *sys.argv], preparation_tool_sha256=primary.sha(helper),
                original_workspace=str(root), replay_root=str(target),
                verified_files={name: primary.sha(value) for name, value in contents.items()},
                added_files=added, reused_identical_files=reused,
                original_evaluation_results_copied=False, existing_files_overwritten=False,
                training_executed=False, policy_evaluation_executed=False)
    with manifest.open('xb') as handle:
        handle.write(primary.encoded(meta))
    guide = '''# 辅助冻结方法隔离重放

本补充仅加入旧 V2 全 15 个和 NNC 全 5 个冻结模型与来源，以及实际执行过的适配器、传统调度入口和互斥分片入口。原 20 个最终模型和 36 个数值源文件未被覆盖。旧 source.zip 的 BOM 字节只保留于此隔离归档。没有复制原始评价结果；今后显式重放的结果写入本隔离目录。

准备前后的输入/输出哈希、新增及已存在相同文件清单见 auxiliary_replay_manifest.json；普通原门禁实查见 auxiliary_replay_protocol_check.json。

先激活 conda PPO，再进入本目录运行：

```powershell
python PPO_project/tools/preview_unified_baselines.py --study direct-explore evaluate --split test --regime nominal
python PPO_project/tools/preview_unified_baselines.py --study direct-explore evaluate --split test --regime tight
python PPO_project/tools/preview_unified_baselines.py --study direct-explore evaluate --split test --regime low_jerk
python PPO_project/tools/preview_unified_baselines.py --study direct-explore evaluate --split regression --regime nominal
python PPO_project/tools/preview_two_step_eval.py --study direct-explore --split test --regime nominal
python PPO_project/tools/preview_two_step_eval.py --study direct-explore --split test --regime tight
python PPO_project/tools/preview_two_step_eval.py --study direct-explore --split test --regime low_jerk
python PPO_project/tools/preview_two_step_eval.py --study direct-explore --split regression --regime nominal
```

历史旧策略回归可用 preview_auxiliary_shards.py 的 --shard 0/1/2 --shards 3 分三个互不重叠的模型组运行；完成后原串行命令只校验缓存并合并。所有并发组必须使用相同的 --shards 3。验证/测试/回归数值输出仅属于本次显式重放，不能混入原论文的既有结果。
'''
    if not (target / 'AUXILIARY_REPLAY.md').exists():
        with (target / 'AUXILIARY_REPLAY.md').open('x', encoding='utf-8') as handle:
            handle.write(guide)
    return verify(target)


def check_protocol(target):
    target = Path(target).resolve()
    result = verify(target)
    code = '''import json, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'PPO_project/tools'))
import preview_unified_baselines as runner
candidate=runner.final_study('direct-explore')
frozen=runner.validate_final(candidate)
items=runner.baseline_runs()
assert len(items)==10
for item in items:
    cfg=runner.yaml.safe_load((item['run']/'config.yaml').read_text(encoding='utf-8'))
    for regime in ['nominal','tight','low_jerk']:
        effective=runner.effective_config(cfg,candidate,regime)
        assert effective['environment']['interpolation_period']==0.001
for split in ['validation','test','regression']:
    assert len(runner.reference_specs(split))==3
import preview_two_step_eval as traditional
study=traditional.load_study('direct-explore')
traditional.require_test_freeze(study,'direct-explore')
assert Path(runner.ROOT)==Path.cwd()
assert Path(__import__('main').__file__).resolve().is_relative_to(Path.cwd())
print(json.dumps(dict(final_model_count=len(frozen['models']),selected_auxiliary_models=len(items),
    source_sha256=candidate.source_state()['source_sha256'],runtime_root=str(runner.ROOT),
    imported_main_file=__import__('main').__file__,ordinary_legacy_and_final_freeze_checks_passed=True,
    effective_config_regimes_checked=['nominal','tight','low_jerk'],policy_evaluation_executed=False)))
'''
    command = [sys.executable, '-X', 'utf8', '-I', '-c', code]
    process = subprocess.run(command, cwd=target, capture_output=True, text=True, encoding='utf-8',
                             env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, check=False)
    result.update(executed_command=command, returncode=process.returncode, stdout=process.stdout,
                  stderr=process.stderr, policy_evaluation_executed=False,
                  check_tool_sha256=primary.sha(Path(__file__).read_bytes()),
                  checked_utc=datetime.now(timezone.utc).isoformat())
    if process.returncode == 0:
        result.update(json.loads(process.stdout))
    attempt = 1
    while (target / f'auxiliary_replay_protocol_check_attempt{attempt}.json').exists():
        attempt += 1
    report = target / f'auxiliary_replay_protocol_check_attempt{attempt}.json'
    result['check_report'] = str(report)
    with report.open('xb') as handle:
        handle.write(primary.encoded(result))
    if process.returncode:
        raise RuntimeError(f'Ordinary archived auxiliary checks failed; see {report.name}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'check-protocol'])
    parser.add_argument('--target', type=Path, default=primary.DEFAULT_TARGET)
    args = parser.parse_args()
    function = {'prepare': prepare, 'verify': verify, 'check-protocol': check_protocol}[args.command]
    print(json.dumps(function(args.target), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
