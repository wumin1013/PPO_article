"""Frozen execution entry points and complete, failure-inclusive matrices."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import sys
from pathlib import Path
import time
import numpy as np
import torch
from common import ROOT, load_path, manifest, sha, write_json, utc
from evaluate import rollout
from execution_settings import configure_dt


def verify_freeze():
    freeze=json.loads((ROOT/'audit'/'test_freeze.json').read_text('utf-8'))
    for file,digest in freeze['sha256'].items():
        if sha(ROOT/file)!=digest:raise ValueError('Frozen input changed: '+file)
    return freeze


class LegacyActor:
    def __init__(self,method):
        folder=ROOT/'baseline_records'/'supplementary_reproduction'
        sys.path.insert(0,str(folder))
        from frozen_policy import FrozenPolicy
        self.policy=FrozenPolicy(folder/'models'/('main.pt' if method=='legacy_adaptive' else 'two.pt'))
        self.d=self.policy.preview_mm
    def act(self,x,deterministic=True):return self.policy.mean(x).numpy(),0.,0.


def legacy(case,method,dt):
    configure_dt(dt);torch.set_num_threads(1)
    path=ROOT/'baseline_records'/'supplementary_reproduction'/'cases'/case/'reference.csv'
    reference=np.loadtxt(path,delimiter=',',skiprows=1);actor=LegacyActor(method)
    out=ROOT/'results'/'legacy_replay'/(method+'_'+case+'_'+str(dt)+'.npz')
    return rollout(actor,reference,out,fixed_d=actor.d,dt=dt,wall_s=1680.,runtime_version='v1')


def frozen_matrix(method,seed,dt,split,tag='fullrange'):
    from model_io import load_model
    configure_dt(dt);torch.set_num_threads(1)
    if method.startswith('legacy'):
        actor=LegacyActor(method);d=actor.d;key=method
    else:
        model=ROOT/'results'/'paired'/tag/f'seed_{seed}'/method/'final.pt'
        actor,payload=load_model(model);d=payload['d'] if method=='fixed' else None;key=f'{method}_{seed}' if tag=='main' else f'{tag}_{method}_{seed}'
    if split=='test':
        verify_freeze()
    for row in manifest()['paths']:
        if row['split']!=split:continue
        out=ROOT/'results'/('unseen' if split=='test' else 'validation')/(key+'_'+row['id']+'_'+str(dt)+'.npz')
        if out.with_suffix('.json').exists():continue
        rollout(actor,load_path(row['id']),out,fixed_d=d,dt=dt)


def baseline(split,dt,tolerance=1.25,v=100.,a=2000.,j=62500.,method='SCMO'):
    from baselines.motion_overlap import plan
    from metrics import metrics
    if split=='test':verify_freeze()
    limits=dict(v=v,a=a,j=j)
    version='v3' if method=='S2' else 'v2'
    key=f'{method.lower()}_{version}_t{tolerance}_v{v}_a{a}_j{j}'
    records=[]
    for row in manifest()['paths']:
        if row['split']!=split:continue
        file=ROOT/'results'/'deterministic'/(key+'_'+row['id']+'_'+str(dt)+'.npz')
        file.parent.mkdir(exist_ok=True,parents=True)
        if file.with_suffix('.json').exists():continue
        reference=load_path(row['id']);start=time.perf_counter()
        try:
            if method=='S2':
                from baselines.legacy_s2 import plan as s2_plan
                p,meta=s2_plan(reference,dt=dt,tolerance=tolerance,limits=(v,a,j))
                trajectory={'positions':p,'meta':meta}
            else:trajectory=plan(reference,tolerance=tolerance,limits=limits,dt=dt)
            p=trajectory['positions']
            np.savez_compressed(file,reference=reference,positions=p,time=np.arange(len(p))*dt,dt_s=dt)
            result=metrics(reference,p,dt,tolerance,(v,a,j))
            result['planner']=trajectory.get('metrics',trajectory.get('meta'))
            result['trace_sha256']=sha(file)
        except Exception as exc:result=dict(success=False,complete_time_s=None,reason=type(exc).__name__+': '+str(exc),trace_sha256=sha(file) if file.exists() else None)
        result.update(method=method,path=row['id'],wall_s=time.perf_counter()-start)
        write_json(file.with_suffix('.json'),result);print('BASELINE',row['id'],result['success'],result['complete_time_s'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['legacy','matrix','baseline']);p.add_argument('--case',default='dense')
    p.add_argument('--method',default='legacy_adaptive');p.add_argument('--seed',type=int,default=101);p.add_argument('--dt',type=float,default=.001)
    p.add_argument('--split',default='test');p.add_argument('--tolerance',type=float,default=1.25);p.add_argument('--v',type=float,default=100.);p.add_argument('--a',type=float,default=2000.);p.add_argument('--j',type=float,default=62500.)
    p.add_argument('--tag',default='fullrange')
    args=p.parse_args()
    if args.mode=='legacy':legacy(args.case,args.method,args.dt)
    elif args.mode=='matrix':frozen_matrix(args.method,args.seed,args.dt,args.split,args.tag)
    else:baseline(args.split,args.dt,args.tolerance,args.v,args.a,args.j,'S2' if args.method=='S2' else 'SCMO')
