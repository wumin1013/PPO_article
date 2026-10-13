"""Bounded development study: learn D while the feed policy stays frozen."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

import preview_fixed_prior_study as prior_study
from preview_only_ppo import PreviewOnlyPPO,frozen_actor_digest

full=prior_study.full;old=full.core.old;ROOT=old.ROOT
BASE_OUT=prior_study.BASE_OUT
OUT=ROOT/'论文项目/revision_20260910_continued/preview_only_v1'
INITIAL=prior_study.OUT/'development/preview_time_adaptive_seed101/checkpoints/step0'
FIXED=BASE_OUT/'development/preview_time_fixed3_seed101/checkpoints/step0'
MILESTONES=(0,256,512,1024)
BASE_TRAIN,BASE_INSTALL,BASE_TRAINER=full.train,full.install,full.FullBatchTrainer
_active=False


def configuration(candidate,method,seed):
    cfg=prior_study.configuration(candidate,method,seed)
    cfg['revision'].update(study='frozen-feed-preview-only-v1',
        init_checkpoint=(INITIAL/'checkpoint.pt').relative_to(ROOT).as_posix(),
        init_checkpoint_sha256=old.sha(INITIAL/'checkpoint.pt'),active_action_dims=[2],
        candidate_change='Freeze actor trunk and feed/angular heads; deterministic current-state feed, sample/train only D',
        initialization='Same actor/critic values as fixed_prior_v1 step0; new preview-head Adam and ordinary critic Adam',
        likelihood='Only raw D Gaussian, action mask[0,0,1]; no feed sampling, density or entropy',
        frozen_feed_scope='Freeze the state-to-feed function, not its time sequence; D can still change sensed state and hence deterministic feed output')
    return cfg


def make_agent(cfg,env):
    keys=('hidden_dim','actor_lr','critic_lr','lmbda','epochs','eps','gamma','ent_coef')
    path=ROOT/cfg['revision']['init_checkpoint']
    if old.sha(path)!=cfg['revision']['init_checkpoint_sha256']:raise ValueError('Registered initializer changed')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    agent=PreviewOnlyPPO(state_dim=env.observation_dim,device=torch.device('cpu'),
        active_action_dims=cfg['revision']['active_action_dims'],
        fixed_preview=(cfg['revision']['fixed_distance_mm']-.8)/3.8,
        initial_actor_state=saved['actor'],initial_critic_state=saved['critic'],
        **{key:cfg['ppo'][key] for key in keys})
    return agent


def source_state():
    value=prior_study.source_state()
    for name in ('preview_only_ppo.py','preview_only_study.py'):
        path=Path(__file__).with_name(name)
        value['source_files'][path.relative_to(ROOT).as_posix()]=old.sha(path)
    value['source_sha256']=hashlib.sha256(json.dumps(value['source_files'],sort_keys=True).encode()).hexdigest()
    return value


def _model_state_digest(state):
    digest=hashlib.sha256()
    for name,tensor in state.items():
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checked_fixed_rows(expected_paths):
    """Bind the reused exact3mm results to their original source/model/trace."""
    prior_study.checked_snapshot(FIXED)
    provenance=old.read(FIXED/'provenance.json')
    rows=old.read(FIXED/'validation/summary.json')['rows']
    if len(rows)!=3 or {r['path_id'] for r in rows}!=set(expected_paths):
        raise ValueError('Exact3mm control needs all3 original validation paths')
    for row in rows:
        if row['checkpoint_sha256']!=old.sha(FIXED/'checkpoint.pt') or row['source_sha256']!=provenance['source_sha256'] or row['config_sha256']!=old.sha(FIXED/'config.yaml') or row['trace_sha256']!=old.sha(FIXED/'validation'/f'{row["path_id"]}.npz'):
            raise ValueError('Exact3mm validation source/model/config/trace binding changed')
    return rows


def prepare():
    parent=prior_study.prepare()
    checked_fixed_rows(parent['validation_path_ids'])
    prior_study.checked_snapshot(INITIAL);prior_study.checked_snapshot(FIXED)
    initial=torch.load(INITIAL/'checkpoint.pt',map_location='cpu',weights_only=False)
    fixed=torch.load(FIXED/'checkpoint.pt',map_location='cpu',weights_only=False)
    digests={name:_model_state_digest(initial[name]) for name in ('actor','critic')}
    if any(digests[name]!=_model_state_digest(fixed[name]) for name in digests):
        raise ValueError('Frozen fixed D=3 control must have the exact same actor and critic initializer')
    value=dict(version='frozen-feed-preview-only-v1',source=source_state(),
        parent_protocol_sha256=old.sha(prior_study.OUT/'protocol.json'),
        initial_checkpoint=INITIAL.relative_to(ROOT).as_posix(),initial_checkpoint_sha256=old.sha(INITIAL/'checkpoint.pt'),
        initial_model_state_digests=digests,methods=['adaptive'],seed=101,smoke_seed=901,smoke_decisions=16,
        main_change='Only learning/sampling scope: frozen actor trunk/feed, deterministic feed mean, D-only PPO mask',
        numerical_unchanged='Actor architecture/initial values, critic optimizer and training, original state/reward/D mapping/KCM/v2,64-decision batching and1ms/20ms periods',
        additional_budget=1024,validation_milestones=list(MILESTONES),
        selection_rule=parent['selection_rule'],
        same_mapping_control='Own step0 all3 validation paths; same frozen feed parameters and float32 D mapping',
        exact_3mm_control=dict(checkpoint=FIXED.relative_to(ROOT).as_posix(),
            checkpoint_sha256=old.sha(FIXED/'checkpoint.pt'),validation_sha256=old.sha(FIXED/'validation/summary.json')),
        baseline_role='Same feed policy with fixed3mm, zero new baseline training; never call it a new independent trained fixed method',
        shared_pretraining=parent['prior'],new_pretraining_decisions=0,
        baseline_results_previously_known=True,test_access=False,machine_execution_authorized=False,
        feed_function_coupling='Frozen function receives current state; effects of D-mediated state changes are included, but no feed parameters or feed exploration are trained',
        manifest_sha256=old.sha(old.DATA/'manifest.json'),validation_path_ids=parent['validation_path_ids'])
    path=OUT/'protocol.json'
    if path.exists():
        previous=old.read(path);value['registered_utc']=previous['registered_utc']
        if previous!=value:raise ValueError('Preserve frozen preview-only protocol')
    else:
        value['registered_utc']=datetime.now(timezone.utc).isoformat();old.write(path,value)
    return value


def _trainer(cfg,run,source_hash):
    return BASE_TRAINER(cfg,run,env_factory=full.local.LocalCornerEnvironment,agent_factory=make_agent,source_hash=source_hash)


@contextmanager
def installed():
    global _active
    if _active:raise RuntimeError('Do not nest preview-only installation')
    if full.OUT!=BASE_OUT or prior_study.cnc_env.apply_kinematic_constraints is not prior_study.kinematics.apply_kinematic_constraints or full.core.plan_terminal_brake.__module__!='terminal_braking':
        raise RuntimeError('Use a clean process with original KCM and v2 brake')
    names=('OUT','configuration','source_state','prepare','FullBatchTrainer')
    previous={name:getattr(full,name) for name in names}
    core_names=('OUT','configuration','source_state','prepare','MacroEnvironment','make_agent')
    core_previous={name:getattr(full.core,name) for name in core_names}
    _active=True
    try:
        full.OUT,full.configuration,full.source_state,full.prepare=OUT,configuration,source_state,prepare
        full.FullBatchTrainer=_trainer
        BASE_INSTALL();full.core.make_agent=make_agent
        yield
    finally:
        for name,value in previous.items():setattr(full,name,value)
        for name,value in core_previous.items():setattr(full.core,name,value)
        _active=False


def train(decisions,stage='development'):
    if stage not in ('development','smoke') or decisions not in (MILESTONES if stage=='development' else (16,)):
        raise ValueError('Only0/256/512/1024 development or16 smoke; cap1024')
    seed=101 if stage=='development' else 901
    with installed():return BASE_TRAIN('adaptive',seed,decisions,stage)


def evaluate(decisions,stage='development'):
    if stage not in ('development','smoke') or decisions not in (MILESTONES if stage=='development' else (16,)):
        raise ValueError('Only registered validation nodes')
    seed=101 if stage=='development' else 901
    snapshot=OUT/stage/f'preview_time_adaptive_seed{seed}/checkpoints/step{decisions}'
    prior_study.checked_snapshot(snapshot)
    with installed():
        protocol=prepare();path=snapshot/'validation/summary.json'
        rows=old.read(path)['rows'] if path.exists() else full.core.evaluate(snapshot)
        if len(rows)!=3 or {r['path_id'] for r in rows}!=set(protocol['validation_path_ids']):raise ValueError('All3 validation paths required')
        for row in rows:
            if row['checkpoint_sha256']!=old.sha(snapshot/'checkpoint.pt') or row['source_sha256']!=protocol['source']['source_sha256'] or row['config_sha256']!=old.sha(snapshot/'config.yaml') or row['trace_sha256']!=old.sha(path.parent/f'{row["path_id"]}.npz'):
                raise ValueError('Saved validation binding changed')
        files={p.name:old.sha(p) for p in path.parent.iterdir() if p.is_file() and p.name!='delivery_manifest.json'}
        manifest=path.parent/'delivery_manifest.json'
        if manifest.exists():
            if old.read(manifest)!=dict(files=files):raise ValueError('Validation artifact changed')
        else:old.write(manifest,dict(files=files))
        return rows


def report():
    protocol=prepare();run=OUT/'development/preview_time_adaptive_seed101'
    if old.read(run/'training_summary.json')['decisions']!=1024:raise ValueError('Finish1024 budget before selection')
    records=[];all_rows=[]
    for step in MILESTONES:
        rows=evaluate(step);all_rows.extend(rows)
        success=sum(row['success'] for row in rows)
        records.append(dict(step=step,successes=success,count=3,
            complete_time_s=sum(row['elapsed_time_s'] for row in rows) if success==3 else None,
            checkpoint=(run/f'checkpoints/step{step}').relative_to(ROOT).as_posix(),
            checkpoint_sha256=old.sha(run/f'checkpoints/step{step}/checkpoint.pt'),
            validation_sha256=old.sha(run/f'checkpoints/step{step}/validation/summary.json')))
    selected=min(records,key=prior_study.selection_key)
    fixed_rows=checked_fixed_rows(protocol['validation_path_ids'])
    fixed_success=sum(row['success'] for row in fixed_rows)
    fixed=dict(successes=fixed_success,count=3,complete_time_s=sum(row['elapsed_time_s'] for row in fixed_rows) if fixed_success==3 else None)
    comparisons={}
    for name,control in (('same_mapping_step0',records[0]),('exact_fixed3',fixed)):
        common=selected['successes']==control['successes']==3
        comparisons[name]=dict(all3_common_complete=common,
            time_reduction_percent=(control['complete_time_s']-selected['complete_time_s'])/control['complete_time_s']*100 if common else None)
    result=dict(protocol_sha256=old.sha(OUT/'protocol.json'),selected=selected,all_checkpoints=records,
        adaptive_rows=all_rows,exact_fixed3_rows=fixed_rows,comparisons=comparisons,
        training=old.read(run/'training_summary.json'),validation_wall_s=sum(row['evaluation_wall_s'] for row in all_rows),
        shared_pretraining=protocol['shared_pretraining'],new_pretraining_decisions=0,new_fixed_training_decisions=0,
        interpretation='D-only learning development diagnostic with frozen feed function; single seed, no holdout/independent fixed-training claim')
    path=OUT/'summary.json'
    if path.exists():
        if old.read(path)!=result:raise ValueError('Preserve completed preview-only report')
    else:old.write(path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','smoke','chain','report'):sub.add_parser(name)
    for name in ('train','evaluate'):
        p=sub.add_parser(name);p.add_argument('--decisions',type=int,choices=MILESTONES,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    started=time.perf_counter();record=dict(argv=sys.argv,started_utc=datetime.now(timezone.utc).isoformat(),
        status='running',runner_sha256=old.sha(Path(__file__)),torch_threads=1)
    path=OUT/'invocations'/f'{time.time_ns()}_{args.command}.json';old.write(path,record)
    try:
        if args.command=='prepare':result=prepare()
        elif args.command=='train':result=str(train(args.decisions))
        elif args.command=='evaluate':result=evaluate(args.decisions)
        elif args.command=='report':result=report()
        elif args.command=='smoke':train(16,'smoke');result=evaluate(16,'smoke')
        else:
            for step in MILESTONES:train(step);evaluate(step)
            result=report()
        record['status']='complete';print(json.dumps(result,ensure_ascii=False,indent=2))
    except BaseException as error:
        record.update(status='failed_or_interrupted',error=repr(error));raise
    finally:
        record.update(wall_s=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat());old.write(path,record)


if __name__=='__main__':main()
