"""Complete only the missing matched fixed3 target-tracking development chain.

The corrected adaptive1024 chain and both uncorrected groups are immutable
caches. The original FullBatchTrainer, numerical config, feed wrapper and v2
brake are reused verbatim. This file changes scheduling and audit records only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

import torch
import yaml

import preview_restart_closed_loop as audit
import preview_target_tracking_study as target
from preview_formal_holdout import validate_freeze

ROOT, BASE = audit.ROOT, audit.BASE
OUT = BASE/'target_tracking_matched_v1'
STEPS = (0, 256, 512, 1024)
TRAIN_IDS = ('mixed_lengths_train', 'dense_corners_train', 'line_curve_train')
BASE_CONFIG, BASE_SOURCE = target.configuration, target.source_state
full, core = target.full, target.full.core
METHOD_DIFFERENCES = frozenset(('variant', 'effective_variant', 'adaptive_preview',
    'active_action_dims', 'init_checkpoint', 'init_checkpoint_sha256'))


def configuration(candidate, method, seed):
    if candidate != 'preview_time' or method != 'fixed3' or seed not in (101, 901):
        raise ValueError('Only missing fixed3 seed101 and smoke901 may train')
    return BASE_CONFIG(candidate, method, seed)


def common_config(cfg):
    result = copy.deepcopy(cfg)
    for key in METHOD_DIFFERENCES:
        result['revision'].pop(key, None)
    return result


def source_state():
    original = BASE_SOURCE()
    files = dict(original['source_files'])
    for name in ('PPO_project/tools/preview_target_tracking_matched.py',
                 'PPO_project/tests/test_preview_target_tracking_matched.py',
                 'PPO_project/tools/preview_restart_closed_loop.py',
                 'PPO_project/tools/preview_target_tracking_transfer.py',
                 'PPO_project/tools/preview_formal_holdout.py'):
        files[name] = audit.sha(ROOT/name)
    return dict(source_files=files,
        source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        head=original.get('head'))


def snapshot_path(root, method, step):
    return root/f'development/preview_time_{method}_seed101/checkpoints/step{step}'


def preflight():
    frozen = validate_freeze(BASE/'formal_v1/freeze.json')
    if not audit.read(audit.OUT/'summary.json')['gate']['eligible_for_separate_training_screen']:
        raise ValueError('Closed-loop gate must authorize the separate training screen')
    original = target.prepare()
    expected_adaptive = BASE_CONFIG('preview_time', 'adaptive', 101)
    expected_fixed = configuration('preview_time', 'fixed3', 101)
    if common_config(expected_adaptive) != common_config(expected_fixed):
        raise ValueError('Matched fixed/adaptive numeric configuration differs beyond explicit policy/initializer fields')
    if tuple(expected_fixed['revision']['training_path_ids']) != TRAIN_IDS:
        raise ValueError('Training whitelist changed')
    ledger = {}
    cache = {}
    config_evidence = []
    groups = [('corrected_adaptive', target.OUT, 'adaptive', BASE_CONFIG,
               'target_tracking_1024_report_v1'),
              ('original_fixed3', target.BASE_OUT, 'fixed3', target.BASE_CONFIG,
               'full_batch_4096_report_v1'),
              ('original_adaptive', target.BASE_OUT, 'adaptive', target.BASE_CONFIG,
               'full_batch_4096_report_v1')]
    for label, root, method, config_function, report in groups:
        nodes = []
        for step in STEPS:
            parent = snapshot_path(root, method, step)
            provenance = audit.verify_parent(parent, ledger)
            cfg = yaml.safe_load((parent/'config.yaml').read_text(encoding='utf-8'))
            if cfg != config_function('preview_time', method, 101):
                raise ValueError(f'Historical numerical config mismatch: {label}/step{step}')
            if label == 'corrected_adaptive' and provenance['source_files'] != original['source']['source_files']:
                raise ValueError('Historical adaptive numerical dependencies differ from registered target tracking')
            rows = audit.cached_rows(parent/'validation', ledger,
                archive_manifest=BASE/f'reports/{report}/manifest.json')
            nodes.append(dict(step=step, checkpoint=parent.relative_to(ROOT).as_posix(),
                checkpoint_sha256=audit.sha(parent/'checkpoint.pt'), rows=rows,
                source_sha256=provenance['source_sha256'], cache_reused=True))
            config_evidence.append(dict(group=label, step=step, exact_historical_config_match=True,
                configuration_sha256=audit.sha(parent/'config.yaml'),
                original_training_source_sha256=provenance['source_sha256']))
        cache[label] = nodes
    for relative in ('PPO_project/data/preview_event_v1/manifest.json',
                     '论文项目/revision_20260910_continued/formal_v1/freeze.json',
                     '论文项目/revision_20260910_continued/restart_closed_loop_v1/summary.json',
                     '论文项目/revision_20260910_continued/target_tracking_v1/protocol.json'):
        ledger[relative] = audit.sha(ROOT/relative)
    for path_id in (*TRAIN_IDS, *audit.PATH_IDS):
        path = core.old.DATA/'paths'/(path_id+'.csv')
        ledger[path.relative_to(ROOT).as_posix()] = audit.sha(path)
    source = source_state()
    ledger.update(source['source_files'])
    for cfg in (expected_adaptive, expected_fixed):
        name = cfg['revision']['init_checkpoint']
        ledger[name] = cfg['revision']['init_checkpoint_sha256']
        if audit.sha(ROOT/name) != ledger[name]:
            raise ValueError('Original sensor initializer changed')
    original_files = original['source']['source_files']
    numeric_match = {name: dict(historical_sha256=digest, new_sha256=source['source_files'][name],
        identical=digest == source['source_files'][name]) for name, digest in original_files.items()}
    if not all(row['identical'] for row in numeric_match.values()):
        raise ValueError('A frozen numerical dependency changed')
    return dict(version='target-tracking-matched-v1', source=source, cache=cache, input_sha256=ledger,
        frozen_gate=dict(model_count=len(frozen['models']), passed=True, test_geometry_opened=False),
        exact_config_evidence=config_evidence, numerical_dependency_evidence=numeric_match,
        added_audit_sources=sorted(set(source['source_files'])-set(original_files)),
        matched_config_difference_fields=sorted(METHOD_DIFFERENCES),
        original_target_source_sha256=original['source']['source_sha256'],
        new_training=dict(method='fixed3', seed=101, additional_decisions=1024, smoke_seed=901,
            smoke_decisions=32, sensor_pretraining_new_decisions=0, prior='Original retained fixed3 sensor4096',
            adam='Fresh at step0; original FullBatchTrainer restores same Adam at later milestones'),
        cached_corrected_adaptive=dict(seed=101, additional_decisions=1024,
            newly_executed_decisions=0, source='Historical target_tracking_v1, not a new independent run'),
        selection=dict(milestones=list(STEPS), step0_eligible_for_every_group=True,
            rule='Max full successes; compare total time only for3/3; otherwise earliest step',
            scope='New matched screening selection over uniformly capped existing nodes; historical selection reports unchanged'),
        budget_cap=1024, training_path_ids=list(TRAIN_IDS), validation_path_ids=list(audit.PATH_IDS),
        original_state_dimension=26, reward_changed=False, original_angular_KCM=True,
        terminal_brake='Original v2', test_access=False, adopted=False)


def prepare():
    protocol = audit.read(OUT/'protocol.json')
    audit.verify_inputs(protocol)
    if protocol['source'] != source_state():
        raise ValueError('Registered matched runner/source changed')
    return protocol


@contextmanager
def installed():
    names = ('OUT', 'configuration', 'source_state', 'prepare')
    core_names = (*names, 'MacroEnvironment', 'make_agent')
    saved = {name: getattr(full, name) for name in names}
    core_saved = {name: getattr(core, name) for name in core_names}
    with audit.corrected_execution():
        try:
            full.OUT, full.configuration, full.source_state, full.prepare = OUT, configuration, source_state, prepare
            core.OUT, core.configuration, core.source_state, core.prepare = OUT, configuration, source_state, prepare
            core.MacroEnvironment, core.make_agent = full.local.LocalCornerEnvironment, full.local.make_agent
            yield
        finally:
            for name, value in saved.items():
                setattr(full, name, value)
            for name, value in core_saved.items():
                setattr(core, name, value)


def validate_request(step, smoke=False):
    if (smoke and step != 32) or (not smoke and step not in STEPS):
        raise ValueError('Only smoke32 or fixed3 nodes0/256/512/1024; no2048 or new adaptive chain')


def worker(step, smoke=False):
    validate_request(step, smoke)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with installed():
        snapshot = full.train('fixed3', 901 if smoke else 101, step,
                              'smoke' if smoke else 'development')
        if smoke:
            first_hash = audit.sha(snapshot/'checkpoint.pt')
            restored = full.train('fixed3', 901, 32, 'smoke')
            if audit.sha(restored/'checkpoint.pt') != first_hash:
                raise ValueError('Smoke restore changed completed snapshot')
            audit.write(OUT/'smoke_verification.json', dict(decisions=32,
                restoration_checked=True, checkpoint_sha256=first_hash,
                optimizer_updates=audit.read(snapshot/'training_summary.json')['optimizer_updates']))
        else:
            if step == 0:
                checkpoint = torch.load(snapshot/'checkpoint.pt', map_location='cpu', weights_only=False)
                if checkpoint['actor_optimizer']['state'] or checkpoint['critic_optimizer']['state']:
                    raise ValueError('Matched step0 requires fresh Adam')
                audit.write(OUT/'step0_optimizer_check.json', dict(actor_adam_state_entries=0,
                    critic_adam_state_entries=0, initialized_from_original_fixed3_sensor=True))
            if not (snapshot/'validation/summary.json').exists():
                core.evaluate(snapshot, snapshot/'validation')
            audit.require_paths(audit.read(snapshot/'validation/summary.json')['rows'])
    prepare()


def choose(nodes):
    if len(nodes) != 4 or {n['step'] for n in nodes} != set(STEPS):
        raise ValueError('Every group must have exactly all four budget-capped milestones')
    for node in nodes:
        audit.require_paths(node['rows'])
    def rank(node):
        count = sum(r['success'] for r in node['rows'])
        return (-count, sum(r['elapsed_time_s'] for r in node['rows']) if count == 3 else 0., node['step'])
    selected = min(nodes, key=rank)
    return dict(selected_step=selected['step'], checkpoint=selected['checkpoint'],
        checkpoint_sha256=selected['checkpoint_sha256'],
        success_count=sum(r['success'] for r in selected['rows']),
        complete_total_time_s=sum(r['elapsed_time_s'] for r in selected['rows']) if all(r['success'] for r in selected['rows']) else None,
        rows=selected['rows'], available_budget=1024, all_nodes=nodes,
        cache_reused=selected.get('cache_reused', False))


def compare(before, after):
    baseline = [dict(r, model_label='paired') for r in before['rows']]
    changed = [dict(r, model_label='paired') for r in after['rows']]
    gate = audit.screening_gate(baseline, changed)
    gate['comparison_kind'] = 'Selected single-seed complete methods at the same1024 cap; different actors, no isolated learned-D causal claim'
    return gate


def report(protocol, invocations):
    nodes = []
    for step in STEPS:
        path = snapshot_path(OUT, 'fixed3', step)
        summary = audit.read(path/'training_summary.json')
        rows = audit.read(path/'validation/summary.json')['rows']
        audit.require_paths(rows)
        if summary['decisions'] != step or summary['checkpoint_sha256'] != audit.sha(path/'checkpoint.pt'):
            raise ValueError('New fixed checkpoint mismatch')
        for row in rows:
            if row['trace_sha256'] != audit.sha(path/'validation'/(row['path_id']+'.npz')):
                raise ValueError('New validation trace changed')
        nodes.append(dict(step=step, checkpoint=path.relative_to(ROOT).as_posix(),
            checkpoint_sha256=summary['checkpoint_sha256'], rows=rows, cache_reused=False,
            training_summary=summary))
    groups = {name: choose(values) for name, values in protocol['cache'].items()}
    groups['corrected_fixed3'] = choose(nodes)
    comparisons = dict(
        corrected_adaptive_vs_corrected_fixed3=compare(groups['corrected_fixed3'], groups['corrected_adaptive']),
        corrected_fixed3_vs_original_fixed3=compare(groups['original_fixed3'], groups['corrected_fixed3']),
        corrected_adaptive_vs_original_adaptive=compare(groups['original_adaptive'], groups['corrected_adaptive']))
    final = nodes[-1]['training_summary']
    costs = dict(new_development_decisions=1024, new_smoke_decisions=32,
        cached_corrected_adaptive_decisions=1024, cached_corrected_adaptive_new_decisions=0,
        new_sensor_pretraining_decisions=0, original_prior_pretraining_decisions=4096,
        fixed_training_wall_s=final['training_wall_s'], fixed_restore_wall_s=final['environment_replay_wall_s'],
        fixed_optimizer_updates=final['optimizer_updates'],
        new_validation_count=12, new_validation_wall_s=sum(r['evaluation_wall_s'] for n in nodes for r in n['rows']),
        invocation_wall_s=sum(r['wall_s'] for r in invocations),
        timing_scope='Final cumulative training/replay totals plus per-path validation; invocation wall contains these and is not added again')
    adaptive_gate = comparisons['corrected_adaptive_vs_corrected_fixed3']
    execution_no_regressions = all(not comparisons[k]['original_success_regressions'] for k in
        ('corrected_fixed3_vs_original_fixed3', 'corrected_adaptive_vs_original_adaptive'))
    all_selected_successful = all(groups[k]['success_count'] == 3 for k in ('corrected_fixed3', 'corrected_adaptive'))
    continue_gate = dict(adopted=False, all_corrected_selected_paths_successful=all_selected_successful,
        public_execution_has_no_success_regression=execution_no_regressions,
        eligible_for_larger_matched_study=bool(all_selected_successful and execution_no_regressions and
            adaptive_gate['eligible_for_separate_training_screen']),
        rule='Both corrected methods3/3, no success regression versus their original selected counterparts, and corrected adaptive improves success/time versus corrected fixed3',
        no_automatic_budget_extension=True, no_independent_repeatability_claim=True)
    audit.verify_inputs(protocol)
    gate = validate_freeze(BASE/'formal_v1/freeze.json')
    result = dict(groups=groups, comparisons=comparisons, continuation_gate=continue_gate, costs=costs,
        final_frozen_gate_passed=len(gate['models']) == 10, original_input_sha256_unchanged=True,
        test_access=False, new_training_methods=['fixed3'], historical_adaptive_reused=True)
    audit.write(OUT/'summary.json', result)
    lines = ['# 公共目标追踪的匹配短训', '',
        '只新训练fixed3 seed101追加1024宏决策及smoke901的32决策；历史corrected adaptive的1024决策与12条验证全部复用缓存，未重启。原fixed/adaptive至1024缓存统一参与四节点选择，step0对所有组同等可选。', '',
        '|方法|选中step|完整成功|三路径完整总时间/s|来源|', '|---|---:|---:|---:|---|']
    for name, selected in groups.items():
        total = selected['complete_total_time_s']
        lines.append(f'|{name}|{selected["selected_step"]}|{selected["success_count"]}/3|{total if total is not None else "不适用"}|{"旧缓存" if selected["cache_reused"] else "本次新训练"}|')
    lines += ['', '阶段门禁：'+json.dumps(continue_gate, ensure_ascii=False), '',
        '完整成功同时要求有序覆盖、几何允差、位置反算六约束与完整停稳。失败记录时间不当作有效完成时间。此结果是单种子开发筛选，不是前瞻独立作用或未见测试证据；目标追踪仍未标记采用。', '',
        '新wrapper只安排缺失训练/验证与记录。protocol.json逐项保存旧数值依赖SHA完全匹配及12份历史配置一致证据。原26维状态、PPO、reward、64batch、对应sensor先验、目标追踪标量函数、角KCM及v2制动不变。', '',
        '成本：'+json.dumps(costs, ensure_ascii=False)]
    (OUT/'结果说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def run():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Keep completed/partial matched run; use worker only for registered unfinished nodes')
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = preflight()
    audit.write(OUT/'protocol.json', protocol)
    with zipfile.ZipFile(OUT/'registered_source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['source']['source_files']:
            archive.write(ROOT/name, name)
    records = []
    for step, smoke in ((32, True), *((step, False) for step in STEPS)):
        label = 'smoke32' if smoke else f'step{step}'
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--step', str(step)]
        if smoke:
            cmd.append('--smoke')
        logdir = OUT/'invocations'
        logdir.mkdir(exist_ok=True)
        record = dict(argv=cmd, command=subprocess.list2cmdline(cmd), cwd=str(ROOT),
            started_utc=datetime.now(timezone.utc).isoformat(), parent_pid=os.getpid(), status='starting')
        started = time.perf_counter()
        with (logdir/(label+'.stdout.log')).open('x', encoding='utf-8') as stdout, \
                (logdir/(label+'.stderr.log')).open('x', encoding='utf-8') as stderr:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            record.update(pid=proc.pid, status='running')
            audit.write(logdir/(label+'.json'), record)
            code = proc.wait()
        record.update(exit_code=code, wall_s=time.perf_counter()-started,
            finished_utc=datetime.now(timezone.utc).isoformat(), status='complete' if code == 0 else 'failed')
        audit.write(logdir/(label+'.json'), record)
        records.append(record)
        print(f'{label}: exit={code}, wall={record["wall_s"]:.3f}s', flush=True)
        if code:
            raise RuntimeError(f'{label} failed; all logs/partial outputs preserved, later stages not started')
    result = report(protocol, records)
    print(json.dumps({k: dict(step=v['selected_step'], success=v['success_count'], total_time_s=v['complete_total_time_s'])
        for k, v in result['groups'].items()}, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(result['continuation_gate'], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'worker'))
    parser.add_argument('--step', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.step, args.smoke)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT/'coordinator_invocation.json'
    if log.exists():
        raise FileExistsError('Do not duplicate existing coordinator')
    record = dict(argv=[sys.executable, *sys.argv], pid=os.getpid(), cwd=str(ROOT),
        started_utc=datetime.now(timezone.utc).isoformat(), status='running')
    audit.write(log, record)
    started = time.perf_counter()
    code = 0
    try:
        run()
    except BaseException:
        code = 1
        record['error'] = traceback.format_exc()
        raise
    finally:
        record.update(exit_code=code, status='complete' if code == 0 else 'failed',
            wall_s=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat())
        audit.write(log, record)
        files = {p.relative_to(OUT).as_posix(): audit.sha(p) for p in OUT.rglob('*')
                 if p.is_file() and p.name != 'delivery_manifest.json'}
        audit.write(OUT/'delivery_manifest.json', dict(files=files, completed=code == 0))


if __name__ == '__main__':
    main()
