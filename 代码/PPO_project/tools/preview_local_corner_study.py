"""Observation ablation and bounded continuation with validation checkpoints.

From the positive-feed candidate, change only five corner coordinates to the
nearest sign-consistent turn group. Motion guidance, reward and constraints
remain unchanged. All distances/angles come from geometry, never a rule that
sets the learned preview action. No test path evaluation is exposed here.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

import preview_macro_study as core
import preview_sensor_study as sensor
import preview_positive_feed_study as positive
from preview_positive_feed_ppo import PositiveFeedPPO
from nearest_turn_state import NearestTurnGeometry

OUT=core.old.ROOT/'论文项目/revision_20260910_continued/local_corner_v1'
METHODS=('fixed1p2','fixed3','fixed4','adaptive')
MILESTONES=(4096,8192,16384)


class LocalCornerEnvironment(positive.PositiveFeedEnvironment):
    def __init__(self,*args,**kwargs):
        self.local_phase=False
        self.turn_geometry=None
        super().__init__(*args,**kwargs)

    def _sense(self):
        super()._sense()
        raw=self.inner.raw
        if self.turn_geometry is None:
            self.turn_geometry=NearestTurnGeometry(self.reference,
                one_sided_tolerance_mm=self.cfg['environment']['epsilon']/2,
                kappa_threshold=float(getattr(raw,'_p8_kappa_th',0.)))
        s=float(raw._project_onto_progress_path(raw.current_position,update=False)[2])
        turn=self.turn_geometry.at(s)
        distance,angle,sign=turn['dist_to_turn'],turn['turn_angle'],turn['turn_sign']
        if distance<=6. and abs(angle)>=math.radians(10.):
            self.local_phase=True
        elif distance>=7. or abs(angle)<=math.radians(8.):
            self.local_phase=False
        self.state[7]=min(distance/6.,10.)/10.
        self.state[8]=np.clip(angle/math.pi,-1.,1.)
        self.state[9]=sign
        self.state[10]=float(self.local_phase)
        self.state[11]=sign*self.state[1]
        self.trace[-1].update(local_turn_distance_mm=distance if math.isfinite(distance) else -1.,
            local_turn_angle_rad=angle,local_turn_sign=float(sign),local_corner_phase=float(self.local_phase),
            local_turn_start_vertex=float(turn['start_vertex'] or -1),
            local_turn_end_vertex=float(turn['end_vertex'] or -1))


def configuration(candidate,method,seed):
    if candidate!='preview_time' or method not in METHODS or seed not in (101,901):
        raise ValueError('Development seeds101/smoke901 only')
    cfg=sensor.configuration(candidate,method,seed)
    prior_method='adaptive' if method=='adaptive' else 'fixed3'
    prior=positive.PRIOR/f'preview_time_{prior_method}_seed101'
    cfg['revision'].update(study='nearest-consistent-corner-observation-v1',
        candidate_change='Only corner coordinates7:12: nearest sign-consistent group, first-vertex distance',
        prior_sampling_change='Positive-feed Gaussian logit as registered in positive_feed_v1',
        unchanged_controller='No change to reference direction, control D, speed target, reward or geometric/kinematic audit',
        initialization='Same retained sensor actor/critic as positive_feed_v1; fresh Adam states',
        init_checkpoint=(prior/'checkpoint.pt').relative_to(core.old.ROOT).as_posix(),
        init_checkpoint_sha256=core.old.sha(prior/'checkpoint.pt'),
        pretraining_macro_decisions=4096,validation_milestones=list(MILESTONES),
        corner_observation=dict(coordinates=list(range(7,12)),distance_anchor='first significant vertex',
            grouping='Consecutive significant same-sign turns; stop at reversal or nonsignificant vertex',
            angle='signed sum without wrapping; normalize/clip at observation boundary',
            hysteresis='6mm/10deg enter,7mm/8deg exit; local observation memory only'),
        likelihood='Gaussian feed logits in PPO, sigmoid only at execution; latent entropy',
        feed_logit_std='min(2, physical std/(sigmoid(raw_mean)*(1-sigmoid(raw_mean))))')
    return cfg


def make_agent(cfg,env):
    return positive.make_agent(cfg,env)


def source_state():
    state=positive.source_state()
    for name in ('nearest_turn_state.py','preview_local_corner_study.py'):
        path=Path(__file__).with_name(name)
        state['source_files'][path.relative_to(core.old.ROOT).as_posix()]=core.old.sha(path)
    state['source_sha256']=hashlib.sha256(json.dumps(state['source_files'],sort_keys=True).encode()).hexdigest()
    return state


def prepare():
    core.old.check_protocol()
    priors={}
    for method in ('fixed3','adaptive'):
        run=positive.PRIOR/f'preview_time_{method}_seed101'
        summary=core.old.read(run/'training_summary.json')
        assert summary['decisions']==4096 and summary['checkpoint_sha256']==core.old.sha(run/'checkpoint.pt')
        priors[method]=dict(checkpoint_sha256=summary['checkpoint_sha256'],
            training_summary_sha256=core.old.sha(run/'training_summary.json'))
    value=dict(version='nearest-consistent-corner-observation-v1',source=source_state(),
        manifest_sha256=core.old.sha(core.old.DATA/'manifest.json'),priors=priors,methods=list(METHODS),
        seed=101,pretraining_decisions=4096,additional_validation_milestones=list(MILESTONES),
        max_additional_decisions=MILESTONES[-1],
        change='Five local-corner observation entries only relative to positive_feed_v1; same prior initialization',
        initialization_selection='Earlier sensor checkpoints retained; positive_feed_v1 final checkpoints worsened validation and are not used for this ablation',
        fixed_grid_mm=[1.2,3.,4.],
        checkpoint_selection='All three validation paths per saved checkpoint; rank full success count then sum of complete times; final method comparison uses the same available training budget',
        fixed_selection='One distance selected on pooled validation, never per test path',
        training_failures='Local recovery/diagnosis logs only, not the paper performance endpoint',
        final_evaluation='New geometry holdout only after configuration and model selection freeze; no test interface in development',
        test_access=False,machine_execution_authorized=False)
    target=OUT/'protocol.json'
    if target.exists() and core.old.read(target)!=value:
        raise ValueError('Keep registered development source/protocol')
    if not target.exists():core.old.write(target,value)
    return value


def install():
    core.OUT=OUT;core.source_state=source_state;core.configuration=configuration
    core.MacroEnvironment=LocalCornerEnvironment;core.make_agent=make_agent;core.prepare=prepare
    return core


def train(method,seed,decisions,stage):
    if stage=='development' and (seed!=101 or decisions not in MILESTONES):
        raise ValueError('Use registered development milestone and seed101')
    if stage=='smoke' and (seed!=901 or not 1<=decisions<=64):
        raise ValueError('Use smoke seed901 and1..64 decisions')
    run=core.train('preview_time',method,seed,decisions,stage)
    checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    if checkpoint['decisions']!=decisions:
        raise ValueError('Requested checkpoint is older than current continuation; use its saved milestone directory')
    snapshot=run/'checkpoints'/f'step{decisions}'
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in ('checkpoint.pt','config.yaml','provenance.json','source.zip','training_summary.json'):
        target=snapshot/name
        if target.exists():
            if core.old.sha(target)!=core.old.sha(run/name):
                raise ValueError(f'Preserve saved checkpoint artifact: {target}')
        else:shutil.copyfile(run/name,target)
    return snapshot


def select(budget):
    if budget not in MILESTONES:raise ValueError('Unregistered budget')
    prepare();records=[]
    for method in METHODS:
        run=OUT/'development'/f'preview_time_{method}_seed101'
        if core.old.read(run/'training_summary.json')['decisions']<budget:
            raise ValueError('Every method requires the same training budget before selection')
        candidates=[]
        for step in MILESTONES:
            if step>budget:continue
            checkpoint=run/'checkpoints'/f'step{step}'
            report=core.old.read(checkpoint/'validation/summary.json')
            rows=report['rows']
            expected={p['id'] for p in core.old.manifest()['paths'] if p['split']=='validation'}
            assert len(rows)==len(expected) and {r['path_id'] for r in rows}==expected
            success=sum(r['success'] for r in rows)
            total=sum(r['elapsed_time_s'] for r in rows if r['success'])
            candidates.append(dict(method=method,step=step,successes=success,count=len(rows),
                common_complete_time_s=total if success==len(rows) else None,
                checkpoint=checkpoint.relative_to(core.old.ROOT).as_posix(),
                checkpoint_sha256=core.old.sha(checkpoint/'checkpoint.pt'),
                validation_sha256=core.old.sha(checkpoint/'validation/summary.json')))
        key=lambda item:(-item['successes'],item['common_complete_time_s'] if item['common_complete_time_s'] is not None else math.inf,item['step'])
        records.append(dict(selected=min(candidates,key=key),all_checkpoints=candidates))
    fixed=[r['selected'] for r in records if r['selected']['method']!='adaptive']
    fixed_best=min(fixed,key=lambda r:(-r['successes'],r['common_complete_time_s'] or math.inf))
    result=dict(budget=budget,records=records,uniform_fixed=fixed_best,
                rule='Full success then complete validation time; no test access',data_role='development selection')
    target=OUT/f'selection_budget{budget}.json'
    if target.exists():raise FileExistsError('Preserve completed selection')
    core.old.write(target,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    p=sub.add_parser('train');p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,default=101);p.add_argument('--decisions',type=int,default=4096)
    p.add_argument('--stage',choices=('smoke','development'),default='development')
    p=sub.add_parser('evaluate');p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path)
    p=sub.add_parser('select');p.add_argument('--budget',type=int,required=True)
    args=parser.parse_args();install()
    if args.command=='prepare':prepare()
    elif args.command=='train':print(train(args.method,args.seed,args.decisions,args.stage))
    elif args.command=='evaluate':core.evaluate(args.run,args.output)
    else:select(args.budget)


if __name__=='__main__':main()
