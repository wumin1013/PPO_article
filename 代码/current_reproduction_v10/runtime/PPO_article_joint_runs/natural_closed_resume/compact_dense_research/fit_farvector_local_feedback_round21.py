"""Fit previously valid local controls with neighborhood feedback,without PPO."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,write,sha,verify_bound_inputs

STAGE=ROOT/'direct14_farvector_local_feedback_fit_round21'
MIGRATION=ROOT/'direct14_farvector_feedback_fit_round19'
LOCAL_SOURCE=ROOT/'farvector_saved_data/local_optimized.npz'
TARGETS=ROOT/'direct14_joint_response_probe_round14/offline_target_feasibility.json'
TEACHER_LOCAL=ROOT/'direct14_previewcue_ppo_seed922302_round13/checkpoint_0024.pt'
TEACHER_RETAIN=ROOT/'direct14_guided_batchreset_ppo_seed922302_round17/checkpoint_0024.pt'


def initial_path():
    r=json.loads((MIGRATION/'result.json').read_text(encoding='utf-8'))
    return Path(r['selected_initialization']['checkpoint'])


def worker():
    import numpy as np
    import torch
    from compact_fine_policy import FineCompactPolicy
    from compact_farvector_runtime import FarVectorRuntime
    from compact_research_eval import CompleteChecks
    from offline_feedback_fit import fit
    started=time.perf_counter()
    deadline=started+1640.
    torch.set_num_threads(1)
    torch.manual_seed(922302)
    checks=CompleteChecks(STAGE,deadline,runtime_class=FarVectorRuntime)
    teachers={}
    for name,path in [('local',TEACHER_LOCAL),('retained',TEACHER_RETAIN)]:
        p=FineCompactPolicy(feed_gain=1.)
        p.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['policy'])
        teachers[name]=p.eval().requires_grad_(False)
    targets=json.loads(TARGETS.read_text(encoding='utf-8'))['selected']
    gains=torch.tensor([[t['feed_gain'],t['preview_gain']] for t in targets]).repeat_interleave(48,dim=0)
    with np.load(LOCAL_SOURCE) as data:
        local=torch.tensor(data['observations'],dtype=torch.float32)
        physical=data['actions']
    groups=[dict(name='local',x=local,teacher='local',gains=gains,fraction=.375,batch_size=192)]
    for case in ['dense','original']:
        with np.load(ROOT/'farvector_round17_data'/(case+'.npz')) as data:
            value=data['observations']
        if case=='original':
            straight=np.flatnonzero((abs(value[:,6])<.008)&(abs(value[:,13])<.005))
            ids=np.unique(np.r_[np.arange(100),np.arange(len(value)-100,len(value)),
                straight[np.linspace(0,len(straight)-1,768).astype(int)]])
            value=value[ids]
        x=torch.tensor(value,dtype=torch.float32)
        groups.append(dict(name=case,x=x,teacher='retained',gains=torch.ones((len(x),2)),
            fraction=.5 if case=='dense' else .125,batch_size=256 if case=='dense' else 64))

    def labels(value,group,ids):
        old=value.clone()
        old[:,5]=torch.sqrt(torch.clamp(1.-old[:,6].square(),min=1e-8))
        raw=teachers[group['teacher']].actor(old)
        gain=group['gains'][ids]
        feed=raw[:,1].double()
        feed=(feed+(gain[:,0].double()-1.)*torch.relu(feed-.6)).to(raw.dtype)
        preview=((6.4+5.6*raw[:,2].double())*gain[:,1].double()-6.4)/5.6
        return torch.stack([raw[:,0],feed,preview.to(raw.dtype)],dim=-1)

    arrays={}
    for group in groups:
        with torch.no_grad():
            target=labels(group['x'],group,torch.arange(len(group['x'])))
        arrays[group['name']+'_observations']=group['x'].numpy()
        arrays[group['name']+'_targets']=target.numpy()
    expected=physical/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])
    error=float(abs(np.clip(arrays['local_targets'],-1,1)-np.clip(expected,-1,1)).max())
    assert error<3e-5,error
    np.savez_compressed(STAGE/'supervision.npz',**arrays)
    write(STAGE/'dataset.json',dict(groups=[dict(name=g['name'],samples=len(g['x']),fraction=g['fraction']) for g in groups],
        local_target_reconstruction_error=error,local_source_sha256=sha(LOCAL_SOURCE),
        local_targets_sha256=sha(TARGETS),local_teacher_sha256=sha(TEACHER_LOCAL),retained_teacher_sha256=sha(TEACHER_RETAIN),
        scope='Local controls individually passed their earlier probes; they are not a full policy. Synthetic neighbors carry feedback labels only,not physical validity labels. No case/corner identity or gain input reaches the student.',
        new_physics_ticks_for_preparation=0,new_ppo_updates=0))
    initial=initial_path()
    policy=FineCompactPolicy(feed_gain=1.)
    policy.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['policy'])
    evaluations,iterations=fit(policy,groups,labels,STAGE,checks,seed=922302,
        deadline=deadline,initial_path=initial)
    eligible=[e for e in evaluations if e['eligible_for_refinement']]
    best=min(eligible,key=lambda e:(-e['results']['original']['corner_comparisons']['historical_main']['not_worse'],
        e['results']['original']['complete_time_s'])) if eligible else None
    write(STAGE/'result.json',dict(status='offline_fit_completed' if iterations==30000 else 'wall_budget_completed',
        offline_iterations=iterations,new_ppo_updates=0,inherited_ppo_updates=88,
        initial_checkpoint=str(initial),initial_checkpoint_sha256=sha(initial),evaluations=evaluations,
        selected_for_refinement=best,adopted=False,retained_checkpoint=str(TEACHER_RETAIN),
        wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))
    print('ROUND_FINISHED',iterations,'eligible',len(eligible),flush=True)


def main():
    if '--worker' in sys.argv:
        worker()
        return
    previous=ROOT/'direct14_farvector_corner_ppo_seed922302_round20'
    assert json.loads((previous/'receipt.json').read_text(encoding='utf-8'))['exit_code']==0
    assert json.loads((previous/'result.json').read_text(encoding='utf-8'))['selected_for_refinement'] is None,'First verify the eligible PPO candidate'
    manifest=verify_bound_inputs()
    STAGE.mkdir(exist_ok=False)
    (STAGE/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/n for n in ['offline_feedback_fit.py','compact_farvector_runtime.py',
        'compact_research_eval.py','compact_fine_policy.py','compact_previewcue_runtime.py','direct_compact_runtime.py','compact_control.py']]
    hashes={str(p):sha(p) for p in sources}
    for p in sources:
        shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Offline feedback fit of locally optimized controls,not PPO',source_sha256=hashes,bound_manifest=manifest,
        wall_budget_s=1800,worker_deadline_s=1640,child_timeout_s=1740,offline_iterations=30000,
        checkpoints=[8000,16000,30000],new_ppo_updates=0,initial_checkpoint=str(initial_path()),
        initial_checkpoint_sha256=sha(initial_path()),
        evidence_gap='Successful encoder migration alone did not establish a better joint candidate. Local response probes already show joint path/deficit improvements but old static fitting lost closed-loop success. Fit their feedback functions with new geometry and local neighborhood supervision.',
        procedure='Same14x64x64x3actor,no runtime teacher or memory. Original local/dense/straight-and-endpoint fractions3/8,1/2,1/8. Alternate witnessed and synthetic-neighbor minibatches. Interior action MSE; for clipped boundary targets use one-sided loss with0.02latent saturation margin. This is only a training loss,not a new execution rule.',
        adoption='Intermediate complete-time/corner screen; same candidate must then pass fine resolution,all34and credible historical time advantage. No automatic final adoption.',
        stop='30000offline iterations,3scheduled full checks,or wall deadline. Preserve failures; do not count these iterations as PPO updates.',
        goal_complete=False,method_frozen=False))
    started=time.perf_counter()
    code=None
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1740)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:
            status='timeout'
    verify_bound_inputs()
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:
        print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-3000:])


if __name__=='__main__':
    main()
