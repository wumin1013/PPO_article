"""Bounded common-execution fixed-distance grid; only missing1.2/4mm chains.

The completed corrected adaptive/fixed3 chains and D intervention are reused.
All numerical training/control code is inherited unchanged; this module only
registers, schedules and verifies the missing independent continuation chains.
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

import preview_target_tracking_matched as matched

audit, target = matched.audit, matched.target
full, core, ROOT, BASE = matched.full, matched.core, matched.ROOT, matched.BASE
OUT = BASE/'target_tracking_grid_v1'
METHODS = ('fixed1p2', 'fixed4')
FIXED_DISTANCES = {'fixed1p2': 1.2, 'fixed3': 3., 'fixed4': 4.}
STEPS = matched.STEPS
ALLOWED_DIFFERENCES = frozenset(('revision.variant', 'revision.effective_variant',
    'revision.fixed_distance_mm', 'revision.initial_preview_mm', 'reward_weights.lookahead.distance'))


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        output = {}
        for key in set(a) | set(b):
            output.update(differences(a.get(key), b.get(key), f'{prefix}.{key}' if prefix else key))
        return output
    return {} if a == b else {prefix: [a, b]}


def configuration(candidate, method, seed):
    if candidate != 'preview_time' or method not in METHODS or seed not in (101, 901):
        raise ValueError('Only missing fixed1p2/fixed4 seed101 or smoke901 may train')
    cfg = matched.BASE_CONFIG(candidate, method, seed)
    base = matched.BASE_CONFIG(candidate, 'fixed3', seed)
    change = differences(base, cfg)
    if set(change) != ALLOWED_DIFFERENCES:
        raise ValueError('Fixed-grid configuration differs beyond the registered distance/method fields')
    return cfg


def source_state():
    original = matched.source_state()
    files = dict(original['source_files'])
    for name in ('PPO_project/tools/preview_target_tracking_grid.py',
                 'PPO_project/tests/test_preview_target_tracking_grid.py'):
        files[name] = audit.sha(ROOT/name)
    return dict(source_files=files, source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        head=original.get('head'))


def validate_cached_binding(parent, rows, step, method):
    audit.require_paths(rows)
    binding = audit.read(parent/'validation/binding.json')
    expected = dict(checkpoint_sha256=audit.sha(parent/'checkpoint.pt'), config_sha256=audit.sha(parent/'config.yaml'),
        source_sha256=audit.read(parent/'provenance.json')['source_sha256'])
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError('Validation binding does not identify its verified parent')
    if any(any(row.get(key) != value for key, value in expected.items()) or row.get('seed') != 101
           or row.get('method') != method or row.get('training_decisions') != step for row in rows):
        raise ValueError('Cached row method/seed/node/source/config/model identity mismatch')
    training = audit.read(parent/'training_summary.json')
    if (training.get('seed') != 101 or training.get('method') != method or training.get('decisions') != step
            or training.get('optimized_decisions') != step or training.get('pending_buffer_decisions') != 0
            or training.get('checkpoint_sha256') != expected['checkpoint_sha256']):
        raise ValueError('Every candidate must have its completed and optimized registered budget, including1024')


def preflight():
    inherited = matched.preflight()
    ledger = dict(inherited['input_sha256'])
    delivery = matched.OUT/'final_delivery_manifest_v3.json'
    audit.verify_files(matched.OUT, audit.read(delivery)['files'], ledger)
    ledger[delivery.relative_to(ROOT).as_posix()] = audit.sha(delivery)
    mechanism_path = matched.OUT/'intervention_adaptive_step0_v1/decision.json'
    mechanism = audit.read(mechanism_path)
    if not mechanism['same_actor_D_screen_passed'] or mechanism['adopted']:
        raise ValueError('Require the successful development D screen, without adoption')
    prior_report = audit.read(matched.OUT/'reports/v2/report.json')
    if not prior_report['continuation_gate']['eligible_for_larger_matched_study']:
        raise ValueError('Matched fixed3/adaptive gate did not pass')
    cached = {'adaptive': inherited['cache']['corrected_adaptive']}
    fixed = []
    for step in STEPS:
        parent = matched.snapshot_path(matched.OUT, 'fixed3', step)
        provenance = audit.verify_parent(parent, ledger)
        cfg = yaml.safe_load((parent/'config.yaml').read_text(encoding='utf-8'))
        if cfg != matched.BASE_CONFIG('preview_time', 'fixed3', 101):
            raise ValueError('Cached matched fixed3 config differs from original target numerical configuration')
        rows = audit.cached_rows(parent/'validation', ledger)
        validate_cached_binding(parent, rows, step, 'fixed3')
        fixed.append(dict(step=step, checkpoint=parent.relative_to(ROOT).as_posix(),
            checkpoint_sha256=audit.sha(parent/'checkpoint.pt'), rows=rows, cache_reused=True,
            source_sha256=provenance['source_sha256']))
    cached['fixed3'] = fixed
    for node in cached['adaptive']:
        validate_cached_binding(ROOT/node['checkpoint'], node['rows'], node['step'], 'adaptive')
    expected_fixed = matched.BASE_CONFIG('preview_time', 'fixed3', 101)
    configs = {method: configuration('preview_time', method, 101) for method in METHODS}
    source = source_state()
    numerical = inherited['numerical_dependency_evidence']
    checks = {}
    for name, prior in numerical.items():
        checks[name] = dict(historical_sha256=prior['historical_sha256'],
            current_sha256=source['source_files'][name], identical=prior['historical_sha256'] == source['source_files'][name])
    if not all(row['identical'] for row in checks.values()):
        raise ValueError('Original target-tracking numerical dependency changed')
    ledger.update(source['source_files'])
    original_off_adaptive = matched.choose(inherited['cache']['original_adaptive'])
    if abs(original_off_adaptive['complete_total_time_s']-13.931) > 1e-9:
        raise ValueError('Original off-adaptive comparator does not reproduce13.931s')
    return dict(version='target-tracking-grid-v1', source=source, input_sha256=ledger,
        inherited_frozen_gate=inherited['frozen_gate'], numerical_dependency_evidence=checks,
        added_grid_sources=sorted(set(source['source_files'])-set(inherited['source']['source_files'])),
        cached=cached, cached_original_off_adaptive=original_off_adaptive,
        cached_D_mechanism=mechanism, new_config_differences={m: differences(expected_fixed, cfg) for m, cfg in configs.items()},
        initializers={m: dict(file=cfg['revision']['init_checkpoint'], sha256=cfg['revision']['init_checkpoint_sha256']) for m,cfg in configs.items()},
        new_training=dict(methods=list(METHODS), seed=101, per_method_additional_decisions=1024,
            per_method_smoke_seed=901, per_method_smoke_decisions=32, total_development_decisions=2048,
            total_smoke_decisions=64, new_sensor_pretraining_decisions=0, fresh_adam_each_chain=True),
        cached_training=dict(adaptive_additional_decisions=1024, fixed3_additional_decisions=1024,
            newly_executed_adaptive_or_fixed3_decisions=0, D_intervention_new_rollouts=0),
        training_path_ids=list(matched.TRAIN_IDS), validation_path_ids=list(audit.PATH_IDS),
        uniform_fixed_selection='All1.2/3/4mm x0/256/512/1024 nodes: maximize full successes; time only for3/3; ties earliest step then shorter distance. One distance/model for all three paths.',
        equal_node_selection=True, step0_eligible_for_all=True, budget_cap=1024,
        larger_budget_gate='Selected adaptive3/3 and faster than uniform best complete fixed and original off-adaptive13.931s; previous same-actor D gate must pass',
        continuation_checkpoint='FullBatchTrainer retains actor, critic, both Adam states, RNG, replay actions and buffer at1024; later continuation must reuse it without prefix retraining under a separately registered budget',
        baseline_full_test_count=827, test_scope='Two new grid files after827-test full suite; only targeted tests rerun',
        adopted=False, test_access=False, no_automatic_4096=True)


def prepare():
    protocol = audit.read(OUT/'protocol.json')
    audit.verify_inputs(protocol)
    if protocol['source'] != source_state():
        raise ValueError('Frozen grid source changed')
    return protocol


@contextmanager
def installed():
    names = ('OUT', 'configuration', 'source_state', 'prepare')
    saved = {name: getattr(full,name) for name in names}
    core_saved = {name: getattr(core,name) for name in (*names,'MacroEnvironment','make_agent')}
    with audit.corrected_execution():
        try:
            full.OUT, full.configuration, full.source_state, full.prepare = OUT, configuration, source_state, prepare
            core.OUT, core.configuration, core.source_state, core.prepare = OUT, configuration, source_state, prepare
            core.MacroEnvironment, core.make_agent = full.local.LocalCornerEnvironment, full.local.make_agent
            yield
        finally:
            for name,value in saved.items(): setattr(full,name,value)
            for name,value in core_saved.items(): setattr(core,name,value)


def worker(method, step, smoke=False):
    if method not in METHODS:
        raise ValueError('No repeated adaptive or fixed3 training')
    matched.validate_request(step, smoke)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with installed():
        snapshot = full.train(method,901 if smoke else 101,step,'smoke' if smoke else 'development')
        if smoke:
            digest = audit.sha(snapshot/'checkpoint.pt')
            restored = full.train(method,901,32,'smoke')
            if audit.sha(restored/'checkpoint.pt') != digest:
                raise ValueError('Smoke restoration altered the completed checkpoint')
            audit.write(OUT/f'{method}_smoke_check.json',dict(decisions=32,restore_checked=True,checkpoint_sha256=digest))
        else:
            if step == 0:
                payload = torch.load(snapshot/'checkpoint.pt',map_location='cpu',weights_only=False)
                if payload['actor_optimizer']['state'] or payload['critic_optimizer']['state']:
                    raise ValueError('Both optimizers must start with fresh Adam states')
                audit.write(OUT/f'{method}_step0_adam_check.json',dict(actor_entries=0,critic_entries=0))
            if not (snapshot/'validation/summary.json').exists():
                core.evaluate(snapshot,snapshot/'validation')
            validate_cached_binding(snapshot,audit.read(snapshot/'validation/summary.json')['rows'],step,method)
    prepare()


def select_uniform_fixed(groups):
    if set(groups) != set(FIXED_DISTANCES):
        raise ValueError('Every fixed distance1.2/3/4mm must be available')
    selected = {name: matched.choose(nodes) for name,nodes in groups.items()}
    def rank(item):
        name, result = item
        return (-result['success_count'], result['complete_total_time_s'] if result['success_count']==3 else 0.,
                result['selected_step'], FIXED_DISTANCES[name])
    method, result = min(selected.items(),key=rank)
    return dict(method=method,distance_mm=FIXED_DISTANCES[method],**result), selected


def eligibility(adaptive, fixed, original_time, D_passed):
    complete = adaptive['success_count']==3 and fixed['success_count']==3
    beats_fixed = bool(complete and adaptive['complete_total_time_s']<fixed['complete_total_time_s']-1e-9)
    beats_original = bool(adaptive['success_count']==3 and adaptive['complete_total_time_s']<original_time-1e-9)
    return dict(eligible4096=bool(complete and beats_fixed and beats_original and D_passed),
        adaptive_and_best_fixed_complete=complete, adaptive_beats_uniform_fixed=beats_fixed,
        adaptive_beats_original_off_adaptive=beats_original, same_actor_D_gate_passed=bool(D_passed),
        adopted=False, executed4096=False, automatically_extend_budget=False,
        limitation='Single-seed development screen only; eligibility is permission to consider a separately registered continuation, not stable superiority')


def report(protocol, invocations):
    groups={'fixed3':protocol['cached']['fixed3']}
    costs=[]
    for method in METHODS:
        nodes=[]
        for step in STEPS:
            snapshot=matched.snapshot_path(OUT,method,step)
            ledger={}
            provenance=audit.verify_parent(snapshot,ledger)
            rows=audit.cached_rows(snapshot/'validation',ledger)
            validate_cached_binding(snapshot,rows,step,method)
            summary=audit.read(snapshot/'training_summary.json')
            nodes.append(dict(step=step,checkpoint=snapshot.relative_to(ROOT).as_posix(),
                checkpoint_sha256=audit.sha(snapshot/'checkpoint.pt'),rows=rows,cache_reused=False,
                source_sha256=provenance['source_sha256'],training_summary=summary))
        groups[method]=nodes
        summary=nodes[-1]['training_summary']
        costs.append(dict(method=method,new_decisions=1024,new_smoke_decisions=32,
            training_wall_s=summary['training_wall_s'],restore_wall_s=summary['environment_replay_wall_s'],
            optimizer_updates=summary['optimizer_updates'],validation_count=12,
            validation_wall_s=sum(r['evaluation_wall_s'] for node in nodes for r in node['rows'])))
    fixed,fixed_selected=select_uniform_fixed(groups)
    adaptive=matched.choose(protocol['cached']['adaptive'])
    original=protocol['cached_original_off_adaptive']['complete_total_time_s']
    gate=eligibility(adaptive,fixed,original,protocol['cached_D_mechanism']['same_actor_D_screen_passed'])
    audit.verify_inputs(protocol)
    frozen=matched.validate_freeze(BASE/'formal_v1/freeze.json')
    result=dict(fixed_groups=fixed_selected,uniform_selected_fixed=fixed,selected_adaptive=adaptive,
        original_off_adaptive_time_s=original,gate=gate,
        adaptive_gain_vs_selected_fixed_pct=(1-adaptive['complete_total_time_s']/fixed['complete_total_time_s'])*100
            if adaptive['success_count']==3 and fixed['success_count']==3 else None,
        adaptive_gain_vs_original_off_pct=(1-adaptive['complete_total_time_s']/original)*100
            if adaptive['success_count']==3 else None,
        costs=dict(per_new_method=costs,new_development_decisions=2048,new_smoke_decisions=64,
            new_validation_count=24,new_cached_method_training_decisions=0,new_D_intervention_rollouts=0,
            cached_continuations=dict(adaptive_historical_decisions=1024,fixed3_historical_decisions=1024,
                newly_executed_decisions=0),
            sensor_prior_training=dict(shared_fixed3_prior_unique_decisions=4096,
                adaptive_prior_unique_decisions=4096,total_unique_historical_decisions=8192,
                attributed_prior_decisions_per_method=4096,new_pretraining_decisions=0,
                sharing='All three fixed distances reuse the same fixed3 sensor prior; do not count its copies as independent training'),
            invocation_wall_s=sum(r['wall_s'] for r in invocations),
            timing_scope='Final cumulative training/replay totals and individual validation times; invocation wall includes these and is not added again'),
        original_inputs_unchanged=True,final_frozen_gate_passed=len(frozen['models'])==10,test_access=False)
    audit.write(OUT/'summary.json',result)
    lines=['# 公共目标追踪：完整短中长固定网格筛选','','只新增fixed1p2/fixed4各1024宏决策＋各smoke32；adaptive/fixed3的既有四节点和D干预均复用，不重跑。所有方法都允许step0，以完整成功数优先，再按全3路径完成总时选择统一模型。','','|方法|选中step|完整成功|完整总时间/s|来源|','|---|---:|---:|---:|---|']
    for method,chosen in {**fixed_selected,'adaptive':adaptive}.items():
        lines.append(f'|{method}|{chosen["selected_step"]}|{chosen["success_count"]}/3|{chosen["complete_total_time_s"]}|{"缓存" if chosen["cache_reused"] else "本次新训练"}|')
    lines+=['',f'统一固定选择：{fixed["method"]}、step{fixed["selected_step"]}。',
        '阶段门禁：'+json.dumps(gate,ensure_ascii=False),'',
        '不按路径挑固定距离，失败时间不作有效完成时间。原26维状态、奖励、网络、64batch、固定sensor先验及freshAdam、原目标追踪标量执行、角KCM与v2制动保持；逐文件/配置匹配证据见protocol.json。',
        '该筛选是827项全套之后新增的两个grid代码文件及专项测试，不宣称重跑全套。没有新预训练、没有重复1024前缀、没有自动进入4096，未作采用或稳定优势结论。',
        '后续如批准4096，须新登记并利用现有FullBatchTrainer完整checkpoint继续，保留source/config/Adam/RNG/未完成回合，不从头重训前缀。']
    (OUT/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def run():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Preserve registered partial/completed grid; do not restart coordinator')
    protocol=preflight();audit.write(OUT/'protocol.json',protocol)
    with zipfile.ZipFile(OUT/'registered_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['source']['source_files']:archive.write(ROOT/name,name)
    invocations=[]
    for method in METHODS:
        for step,smoke in ((32,True),*((step,False) for step in STEPS)):
            label=method+('_smoke32' if smoke else f'_step{step}')
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--method',method,'--step',str(step)]
            if smoke:cmd.append('--smoke')
            logdir=OUT/'invocations';logdir.mkdir(exist_ok=True)
            record=dict(argv=cmd,command=subprocess.list2cmdline(cmd),cwd=str(ROOT),
                started_utc=datetime.now(timezone.utc).isoformat(),parent_pid=os.getpid(),status='starting')
            started=time.perf_counter()
            with (logdir/(label+'.stdout.log')).open('x',encoding='utf-8') as stdout,(logdir/(label+'.stderr.log')).open('x',encoding='utf-8') as stderr:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                record.update(pid=proc.pid,status='running');audit.write(logdir/(label+'.json'),record)
                code=proc.wait()
            record.update(exit_code=code,status='complete' if code==0 else 'failed',wall_s=time.perf_counter()-started,
                finished_utc=datetime.now(timezone.utc).isoformat());audit.write(logdir/(label+'.json'),record)
            invocations.append(record);print(f'{label}: exit={code}, wall={record["wall_s"]:.3f}s',flush=True)
            if code:raise RuntimeError(f'{label} failed; logs/partial outputs retained, later nodes not started')
    result=report(protocol,invocations)
    print(json.dumps(dict(best_fixed=result['uniform_selected_fixed']['method'],
        fixed_time_s=result['uniform_selected_fixed']['complete_total_time_s'],
        adaptive_time_s=result['selected_adaptive']['complete_total_time_s'],gate=result['gate']),ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','worker'));parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--step',type=int);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.command=='worker':
        worker(args.method,args.step,args.smoke);return
    OUT.mkdir(parents=True,exist_ok=True);log=OUT/'coordinator_invocation.json'
    if log.exists():raise FileExistsError('Do not duplicate coordinator')
    record=dict(argv=[sys.executable,*sys.argv],actual_interpreter_flag_B=bool(sys.dont_write_bytecode),
        pid=os.getpid(),cwd=str(ROOT),started_utc=datetime.now(timezone.utc).isoformat(),status='running')
    audit.write(log,record);started=time.perf_counter();code=0
    try:run()
    except BaseException:
        code=1;record['error']=traceback.format_exc();raise
    finally:
        record.update(exit_code=code,status='complete' if code==0 else 'failed',wall_s=time.perf_counter()-started,
            finished_utc=datetime.now(timezone.utc).isoformat());audit.write(log,record)
        files={p.relative_to(OUT).as_posix():audit.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
        audit.write(OUT/'delivery_manifest.json',dict(files=files,completed=code==0))


if __name__=='__main__':main()
