"""Train the same14input actor on validated five-phase local feedback labels."""
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

STAGE=ROOT/'direct14_phase_feedback_fit_round24'
PROBES=ROOT/'direct14_local_decision_phase_probe_round23'
INITIAL_STAGE=ROOT/'direct14_farvector_local_feedback_fit_round21'
INITIAL=INITIAL_STAGE/'fit_08000.pt'
LOCAL_TEACHER=ROOT/'direct14_previewcue_ppo_seed922302_round13/checkpoint_0024.pt'


def screen(results):
    if not all(r['success'] for r in results.values()):
        return False
    r=results['original']
    c=r['corner_comparisons']
    return bool(r['complete_time_s']<=444.236+1e-9 and
        results['dense']['complete_time_s']<=15.06985+1e-9 and
        c['historical_main']['not_worse']>=30 and c['S2']['not_worse']==34 and
        c['historical_strong_fixed']['not_worse']>=33 and
        c['historical_main']['total_change_mm']<=-4.42375693882674+1e-7 and
        (c['historical_main']['not_worse']>=31 or r['complete_time_s']<444.216-1e-9))


def worker():
    import numpy as np
    import torch
    from compact_fine_policy import FineCompactPolicy
    from compact_farvector_runtime import FarVectorRuntime
    from compact_research_eval import CompleteChecks
    from offline_feedback_fit import fit
    torch.set_num_threads(1)
    torch.manual_seed(922302)
    started=time.perf_counter()
    deadline=started+1640.
    checks=CompleteChecks(STAGE,deadline,runtime_class=FarVectorRuntime)
    selected=json.loads((PROBES/'result.json').read_text(encoding='utf-8'))['selected']
    assert len(selected)==34 and all('response' in r for r in selected)
    teachers={}
    for name,path in [('old_local',LOCAL_TEACHER),('new_retained',INITIAL)]:
        teacher=FineCompactPolicy(feed_gain=1.)
        teacher.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['policy'])
        teachers[name]=teacher.eval().requires_grad_(False)
    xs,old_xs,physical,gains,source_hashes=[],[],[],[],{}
    for selection in selected:
        assert len(selection['traces'])==5
        for name in selection['traces']:
            path=PROBES/name
            source_hashes[str(path)]=sha(path)
            with np.load(path) as d:
                x=d['observations'];old=d['original_observations'];a=d['actions']
            assert len(x)==48 and x.shape==(48,14)
            xs.append(x);old_xs.append(old);physical.append(a)
            gains.extend([[selection['feed_gain'],selection['preview_gain'],selection['feed_scale']]]*len(x))
    local=torch.tensor(np.vstack(xs),dtype=torch.float32)
    old_local=np.vstack(old_xs)
    sign=torch.tensor(np.where(old_local[:,5]>=0,1.,-1.),dtype=torch.float32)
    groups=[dict(name='local',x=local,teacher='old_local',gains=torch.tensor(gains,dtype=torch.float64),
        cos_sign=sign,fraction=.375,batch_size=192)]
    for case in ['dense','original']:
        trace=INITIAL_STAGE/f'eval_8000_{case}.npz'
        source_hashes[str(trace)]=sha(trace)
        with np.load(trace) as d:
            x=d['observations']
        if case=='original':
            straight=np.flatnonzero((abs(x[:,6])<.008)&(abs(x[:,13])<.005))
            ids=np.unique(np.r_[np.arange(100),np.arange(len(x)-100,len(x)),
                straight[np.linspace(0,len(straight)-1,768).astype(int)]])
            x=x[ids]
        groups.append(dict(name=case,x=torch.tensor(x,dtype=torch.float32),teacher='new_retained',
            fraction=.5 if case=='dense' else .125,batch_size=256 if case=='dense' else 64))

    def labels(x,group,ids):
        if group['teacher']=='new_retained':
            return teachers['new_retained'].actor(x)
        old=x.clone()
        old[:,5]=group['cos_sign'][ids]*torch.sqrt((1.-old[:,6].square()).clamp_min(1e-8))
        raw=teachers['old_local'].actor(old)
        gain=group['gains'][ids]
        feed=raw[:,1].double()
        feed=(feed+(gain[:,0]-1.)*torch.relu(feed-.6)).to(raw.dtype)
        feed=((feed.double()+1.)*gain[:,2]-1.).to(raw.dtype)
        preview=((6.4+5.6*raw[:,2].double())*gain[:,1]-6.4)/5.6
        return torch.stack([raw[:,0],feed,preview.to(raw.dtype)],dim=-1)

    arrays={}
    for group in groups:
        with torch.no_grad():
            target=labels(group['x'],group,torch.arange(len(group['x'])))
        arrays[group['name']+'_observations']=group['x'].numpy()
        arrays[group['name']+'_targets']=target.numpy()
    expected=np.vstack(physical)/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])
    error=float(abs(np.clip(arrays['local_targets'],-1,1)-np.clip(expected,-1,1)).max())
    assert error<3e-5,error
    np.savez_compressed(STAGE/'supervision.npz',**arrays)
    write(STAGE/'dataset.json',dict(groups=[dict(name=g['name'],samples=len(g['x']),fraction=g['fraction']) for g in groups],
        local_target_reconstruction_error=error,local_negative_cosine_samples=int((old_local[:,5]<0).sum()),
        source_sha256=source_hashes,selected_local_responses=selected,
        procedure='Old-encoding response teacher on validated local phase data. Current round21new-encoding teacher on its own complete dense states and original straight/end anchors. Neighbor labels are synthetic feedback queries,not valid physical trajectories.',
        new_physics_ticks_for_preparation=0,new_ppo_updates=0))
    print('DATA_READY',[(g['name'],len(g['x'])) for g in groups],'label_error',error,flush=True)
    policy=FineCompactPolicy(feed_gain=1.)
    policy.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=False)['policy'])
    evaluations,iterations=fit(policy,groups,labels,STAGE,checks,seed=922302,
        deadline=deadline,initial_path=INITIAL,screen=screen)
    eligible=[e for e in evaluations if e['eligible_for_refinement']]
    best=min(eligible,key=lambda r:(-r['results']['original']['corner_comparisons']['historical_main']['not_worse'],
        r['results']['original']['complete_time_s'])) if eligible else None
    write(STAGE/'result.json',dict(status='offline_fit_completed' if iterations==30000 else 'wall_budget_completed',
        offline_iterations=iterations,new_ppo_updates=0,inherited_ppo_updates=88,
        initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),evaluations=evaluations,
        selected_for_refinement=best,adopted=False,retained_checkpoint=str(INITIAL),
        wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))
    print('ROUND_FINISHED',iterations,'eligible',len(eligible),flush=True)


def main():
    if '--worker' in sys.argv:
        worker()
        return
    assert json.loads((PROBES/'receipt.json').read_text(encoding='utf-8'))['exit_code']==0
    assert json.loads((PROBES/'result.json').read_text(encoding='utf-8'))['all_phase_valid_corners']==34
    verified=json.loads((ROOT/'farvector_round21_verified_status.json').read_text(encoding='utf-8'))
    assert verified['all_four_full_successes'] and sha(INITIAL)==verified['checkpoint_sha256']
    manifest=verify_bound_inputs()
    STAGE.mkdir(exist_ok=False)
    (STAGE/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/n for n in ['offline_feedback_fit.py','compact_farvector_runtime.py',
        'compact_research_eval.py','compact_fine_policy.py','compact_previewcue_runtime.py','direct_compact_runtime.py','compact_control.py']]
    hashes={str(p):sha(p) for p in sources}
    for p in sources:
        shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Offline phase-robust feedback fitting,not PPO',source_sha256=hashes,bound_manifest=manifest,
        wall_budget_s=1800,worker_deadline_s=1640,child_timeout_s=1740,offline_iterations=30000,
        checkpoints=[8000,16000,30000],new_ppo_updates=0,initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),
        probe_result=str(PROBES/'result.json'),probe_result_sha256=sha(PROBES/'result.json'),
        evidence_gap='The current14input candidate passes both cases and resolutions but misses original corners6/10/20/23at1ms and the100ms historical time margin. Local response loss changes with decision phase;fit the physically verified multi-phase coverage instead of repeatedly fitting one phase.',
        procedure='Same14x64x64x3network,no added inference rule or memory. Use all5phase traces for each selected local response;3/8local,1/2current successful dense feedback,1/8current original anchors. Existing neighborhood augmentation and clipped-action-equivalence loss remain. Teacher encodings are separately bound and verified against stored physical actions.',
        adoption='Both full successes,original<=444.236,dense<=15.06985;historical/S2/Fixed counts>=30/34/33;historical total deficit change<=-4.42375694mm;either>=31historical corners or>20ms original time gain. Then independent0.5ms checks;final joint criterion unchanged.',
        stop='30000iterations,three complete checks,or registered deadline. Preserve failures and retain round21if no qualified candidate.',
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
