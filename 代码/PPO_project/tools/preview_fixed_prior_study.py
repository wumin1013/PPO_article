"""One development change: adaptive full-batch PPO starts from the fixed prior.

All numerical settings and training/evaluation implementations are the frozen
full_batch original. Shared seed101 pretraining and reused fixed controls are
explicitly accounted for; this is not an independent formal repetition.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

import preview_full_batch_study as full
from src.environment import cnc_env, kinematics

ROOT=full.core.old.ROOT
BASE_OUT=ROOT/'论文项目/revision_20260910_continued/full_batch_v1'
OUT=ROOT/'论文项目/revision_20260910_continued/fixed_prior_v1'
PRIOR=full.local.positive.PRIOR/'preview_time_fixed3_seed101'
MILESTONES=(0,256,512,1024)
BASE_CONFIG,BASE_SOURCE,BASE_INSTALL,BASE_TRAIN=(full.configuration,full.source_state,full.install,full.train)
old=full.core.old
_active=False


def configuration(candidate,method,seed):
    if candidate!='preview_time' or method!='adaptive' or seed not in (101,901):
        raise ValueError('One adaptive development seed101; smoke seed901 only')
    cfg=BASE_CONFIG(candidate,method,seed)
    cfg['revision'].update(study='adaptive-from-fixed-sensor-prior-v1',
        init_checkpoint=(PRIOR/'checkpoint.pt').relative_to(ROOT).as_posix(),
        init_checkpoint_sha256=old.sha(PRIOR/'checkpoint.pt'),
        initialization='Original sensor fixed3_seed101 actor/critic; fresh Adam as in full_batch; same adaptive action dimensions',
        candidate_change='Only initializer identity: sensor adaptive_seed101 -> sensor fixed3_seed101',
        validation_milestones=list(MILESTONES),zero_step_role='Eligible for both adaptive candidate and reused fixed control',
        registered_additional_budget=1024)
    return cfg


def source_state():
    value=BASE_SOURCE();path=Path(__file__)
    value['source_files'][path.relative_to(ROOT).as_posix()]=old.sha(path)
    value['source_sha256']=hashlib.sha256(json.dumps(value['source_files'],sort_keys=True).encode()).hexdigest()
    return value


def checked_snapshot(snapshot):
    snapshot=Path(snapshot)
    manifest=old.read(snapshot/'snapshot_manifest.json')
    for name,digest in manifest['files'].items():
        if old.sha(snapshot/name)!=digest: raise ValueError(f'Frozen snapshot artifact changed: {name}')
    return old.read(snapshot/'training_summary.json')


def prepare():
    old.check_protocol()
    parent=old.read(BASE_OUT/'protocol.json')
    if parent['source']!=BASE_SOURCE(): raise ValueError('Original frozen full_batch source changed')
    summary=old.read(PRIOR/'training_summary.json')
    prior_sha=old.sha(PRIOR/'checkpoint.pt')
    if summary['decisions']!=4096 or summary['checkpoint_sha256']!=prior_sha or parent['initializers']['fixed3']['checkpoint_sha256']!=prior_sha:
        raise ValueError('Preserve the original complete fixed3_seed101 sensor prior')
    controls=[]
    for step in MILESTONES:
        snapshot=BASE_OUT/f'development/preview_time_fixed3_seed101/checkpoints/step{step}'
        checked_snapshot(snapshot)
        controls.append(dict(step=step,checkpoint=snapshot.relative_to(ROOT).as_posix(),
            checkpoint_sha256=old.sha(snapshot/'checkpoint.pt'),
            training_summary_sha256=old.sha(snapshot/'training_summary.json'),
            snapshot_manifest_sha256=old.sha(snapshot/'snapshot_manifest.json'),
            validation_sha256=old.sha(snapshot/'validation/summary.json')))
    value=dict(version='adaptive-from-fixed-sensor-prior-v1',source=source_state(),
        parent_protocol_file=(BASE_OUT/'protocol.json').relative_to(ROOT).as_posix(),
        parent_protocol_sha256=old.sha(BASE_OUT/'protocol.json'),
        methods=['adaptive'],seed=101,smoke_seed=901,smoke_decisions=16,
        main_change='Only source identity of loaded actor/critic: adaptive prior replaced by fixed3 prior',
        numerical_unchanged='26D local-corner state, reward, PositiveFeedPPO mapping/loss/network/hyperparameters, full64 update schedule,1ms KCM,20ms decisions and original v2 brake',
        active_action_dims=[1,2],adaptive_distance_is_policy_action=True,
        initialization='Fixed3 seed101 pretraining shared with fixed control; fresh Adam on both original fixed and new adaptive continuation',
        prior=dict(checkpoint=(PRIOR/'checkpoint.pt').relative_to(ROOT).as_posix(),checkpoint_sha256=prior_sha,
            config_sha256=old.sha(PRIOR/'config.yaml'),training_summary_sha256=old.sha(PRIOR/'training_summary.json'),
            source_provenance_sha256=old.sha(PRIOR/'provenance.json'),
            macro_decisions=4096,training_wall_s=summary['training_wall_s'],physics_steps=summary['physics_steps']),
        pretraining_macro_decisions=4096,additional_budget=1024,validation_milestones=list(MILESTONES),
        selection_rule='Full success count across all3 validation paths; sum time only if3/3; earliest checkpoint breaks remaining ties; step0 eligible for both',
        fixed_controls=controls,
        fixed_control_cost='Reuse original fixed3 corresponding0/256/512/1024 checkpoints, no new fixed training; archive all original cost. Original full4096 campaign is known development data, comparison pool limited to <=1024',
        independent_training_units='One new adaptive continuation and one reused fixed chain share one pretraining model; not two independent pretrains and not5seed evidence',
        baseline_results_previously_known=True,
        manifest_sha256=old.sha(old.DATA/'manifest.json'),
        validation_path_ids=[p['id'] for p in old.manifest()['paths'] if p['split']=='validation'],
        test_access=False,machine_execution_authorized=False)
    path=OUT/'protocol.json'
    if path.exists():
        previous=old.read(path);value['registered_utc']=previous['registered_utc']
        if previous!=value: raise ValueError('Preserve registered fixed-prior development protocol')
    else:
        value['registered_utc']=datetime.now(timezone.utc).isoformat();old.write(path,value)
    return value


@contextmanager
def installed():
    global _active
    if _active: raise RuntimeError('Do not nest fixed-prior installation')
    if full.OUT!=BASE_OUT or cnc_env.apply_kinematic_constraints is not kinematics.apply_kinematic_constraints or full.core.plan_terminal_brake.__module__!='terminal_braking':
        raise RuntimeError('Use a fresh process without another study, feed correction or v3 brake')
    names=('OUT','configuration','source_state','prepare')
    saved_full={name:getattr(full,name) for name in names}
    saved_core={name:getattr(full.core,name) for name in (*names,'MacroEnvironment','make_agent')}
    _active=True
    try:
        full.OUT,full.configuration,full.source_state,full.prepare=OUT,configuration,source_state,prepare
        BASE_INSTALL()
        yield full
    finally:
        for name,value in saved_full.items():setattr(full,name,value)
        for name,value in saved_core.items():setattr(full.core,name,value)
        _active=False


def train(decisions,stage='development'):
    if decisions not in (MILESTONES if stage=='development' else (16,)) or stage not in ('development','smoke'):
        raise ValueError('Registered development0/256/512/1024 or smoke16 only; never exceed1024')
    seed=101 if stage=='development' else 901
    with installed():
        run=OUT/stage/f'preview_time_adaptive_seed{seed}'
        # Keep a failed temporary save before the frozen trainer resumes from
        # its authoritative checkpoint. No numerical recovery logic changes.
        if (run/'checkpoint.tmp').exists():
            recovery=run/'recovery';recovery.mkdir(exist_ok=True)
            target=recovery/f'{old.sha(run/"checkpoint.tmp")}_checkpoint.tmp'
            if not target.exists():target.write_bytes((run/'checkpoint.tmp').read_bytes())
        return BASE_TRAIN('adaptive',seed,decisions,stage)


def evaluate(decisions,stage='development'):
    if decisions not in (MILESTONES if stage=='development' else (16,)):
        raise ValueError('Unregistered evaluation milestone')
    seed=101 if stage=='development' else 901
    snapshot=OUT/stage/f'preview_time_adaptive_seed{seed}/checkpoints/step{decisions}'
    checked_snapshot(snapshot)
    with installed():
        protocol=prepare();path=snapshot/'validation/summary.json'
        rows=old.read(path)['rows'] if path.exists() else full.core.evaluate(snapshot)
        if len(rows)!=3 or {r['path_id'] for r in rows}!=set(protocol['validation_path_ids']):
            raise ValueError('Every milestone requires all three validation paths')
        for row in rows:
            if (row['checkpoint_sha256']!=old.sha(snapshot/'checkpoint.pt') or
                row['source_sha256']!=protocol['source']['source_sha256'] or
                row['config_sha256']!=old.sha(snapshot/'config.yaml') or
                row['trace_sha256']!=old.sha(path.parent/f'{row["path_id"]}.npz')):
                raise ValueError('Validation model/source/config/trace mismatch')
        files={p.name:old.sha(p) for p in path.parent.iterdir() if p.is_file() and p.name!='delivery_manifest.json'}
        manifest=path.parent/'delivery_manifest.json'
        if manifest.exists():
            if old.read(manifest)!=dict(files=files):raise ValueError('Preserve validation output')
        else:old.write(manifest,dict(files=files))
        return rows


def selection_key(row):
    return (-row['successes'],row['complete_time_s'] if row['successes']==3 else math.inf,row['step'])


def report():
    protocol=prepare();run=OUT/'development/preview_time_adaptive_seed101'
    if old.read(run/'training_summary.json')['decisions']!=1024:
        raise ValueError('Complete the1024 continuation budget before comparing selected models')
    groups={};all_rows={};costs={}
    for method in ('adaptive','fixed3'):
        records=[];all_rows[method]=[]
        for step in MILESTONES:
            snapshot=(run if method=='adaptive' else BASE_OUT/'development/preview_time_fixed3_seed101')/'checkpoints'/f'step{step}'
            training=checked_snapshot(snapshot)
            rows=evaluate(step) if method=='adaptive' else old.read(snapshot/'validation/summary.json')['rows']
            if len(rows)!=3 or {r['path_id'] for r in rows}!=set(protocol['validation_path_ids']):raise ValueError('Incomplete comparison pool')
            if any(row['checkpoint_sha256']!=old.sha(snapshot/'checkpoint.pt') for row in rows):raise ValueError('Comparison model mismatch')
            success=sum(row['success'] for row in rows);all_rows[method].extend(rows)
            records.append(dict(method=method,step=step,successes=success,count=3,
                complete_time_s=sum(r['elapsed_time_s'] for r in rows) if success==3 else None,
                checkpoint=snapshot.relative_to(ROOT).as_posix(),checkpoint_sha256=old.sha(snapshot/'checkpoint.pt'),
                training_summary_sha256=old.sha(snapshot/'training_summary.json'),
                validation_sha256=old.sha(snapshot/'validation/summary.json')))
            if step==1024:costs[method]=dict(training=training,
                validation_wall_s=sum(r['evaluation_wall_s'] for r in all_rows[method]),
                reused_training=method=='fixed3',new_training_decisions=0 if method=='fixed3' else 1024)
        groups[method]=dict(selected=min(records,key=selection_key),all_checkpoints=records)
    adaptive,fixed=groups['adaptive']['selected'],groups['fixed3']['selected']
    comparable=adaptive['successes']==fixed['successes']==3
    value=dict(protocol_sha256=old.sha(OUT/'protocol.json'),groups=groups,rows=all_rows,costs=costs,
        additional_available_budget=1024,shared_pretraining=protocol['prior'],
        common_complete_validation=comparable,
        adaptive_time_reduction_percent=(fixed['complete_time_s']-adaptive['complete_time_s'])/fixed['complete_time_s']*100 if comparable else None,
        interpretation='Development-only shared-prior initialization ablation; preserve all12 adaptive and12 reused fixed validation outcomes; no newholdout or independent5seed conclusion')
    path=OUT/'summary.json'
    if path.exists():
        if old.read(path)!=value:raise ValueError('Preserve completed fixed-prior comparison')
    else:old.write(path,value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','smoke','chain','report'):commands.add_parser(name)
    for name in ('train','evaluate'):
        p=commands.add_parser(name);p.add_argument('--decisions',type=int,choices=MILESTONES,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    record=dict(argv=sys.argv,started_utc=datetime.now(timezone.utc).isoformat(),status='running',
        runner_sha256=old.sha(Path(__file__)),torch_threads=1)
    path=OUT/'invocations'/f'{time.time_ns()}_{args.command}.json';old.write(path,record)
    started=time.perf_counter()
    try:
        if args.command=='prepare':result=prepare()
        elif args.command=='train':result=str(train(args.decisions))
        elif args.command=='evaluate':result=evaluate(args.decisions)
        elif args.command=='report':result=report()
        elif args.command=='smoke':train(16,'smoke');result=evaluate(16,'smoke')
        else:
            for step in MILESTONES:train(step);evaluate(step)
            result=report()
        record['status']='complete'
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except BaseException as error:
        record.update(status='failed_or_interrupted',error=repr(error));raise
    finally:
        record.update(wall_s=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        old.write(path,record)


if __name__=='__main__':main()
