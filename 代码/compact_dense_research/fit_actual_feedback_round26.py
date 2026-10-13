"""Fit actual-arrival local targets while retaining both dense resolutions."""
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

STAGE=ROOT/'direct14_actual_feedback_fit_round26'
PROBES=ROOT/'direct14_actual_feedback_probe_round25'
NOMINAL=ROOT/'direct14_phase_feedback_fit_round24'
VERIFIED=ROOT/'direct14_farvector_local_feedback_fit_round21'
INITIAL=NOMINAL/'fit_30000.pt'


def screen(results):
    if not all(r['success'] for r in results.values()):return False
    o=results['original'];c=o['corner_comparisons']
    return bool(o['complete_time_s']<=444.236+1e-9 and
        results['dense']['complete_time_s']<=15.06985+1e-9 and
        results['dense_half']['complete_time_s']<=15.1069+1e-9 and
        c['historical_main']['not_worse']>=32 and c['S2']['not_worse']==34 and
        c['historical_strong_fixed']['not_worse']==34 and
        c['historical_main']['total_change_mm']<=-4.42375694+1e-7)


def worker():
    import numpy as np
    import torch
    from compact_fine_policy import FineCompactPolicy
    from compact_farvector_runtime import FarVectorRuntime
    from compact_research_eval import CompleteChecks
    from offline_feedback_fit import fit
    from natural_case_metrics import ordered_projection
    torch.set_num_threads(1);torch.manual_seed(922302)
    started=time.perf_counter();deadline=started+1640.

    class Checks(CompleteChecks):
        def run(self,policy,checkpoint,step,**kwargs):
            record=super().run(policy,checkpoint,step,**kwargs)
            nominal=STAGE/f'eval_{step:04d}_dense.json'
            if record['results']['dense']['success'] and time.perf_counter()<deadline-35.:
                name=f'{STAGE.name}_dense_half_{step}'
                output=STAGE/f'dense_half_{step}_launch.txt'
                with output.open('w',encoding='utf-8') as stream:
                    process=subprocess.run([sys.executable,'-B',str(ROOT/'refine_farvector_candidate.py'),
                        '--name',name,'--case','dense','--checkpoint',str(checkpoint),
                        '--nominal',str(nominal)],stdout=stream,stderr=subprocess.STDOUT,
                        timeout=min(310.,max(1.,deadline-time.perf_counter())))
                assert process.returncode==0
                result=json.loads((ROOT/name/'result.json').read_text(encoding='utf-8'))
                record['results']['dense_half']=result
                print('DENSE_HALF_CHECK',step,result['success'],result['complete_time_s'],flush=True)
            else:
                record['results']['dense_half']=dict(success=False,complete_time_s=None,
                    reason='Not launched after nominal failure or deadline')
            return record

    checks=Checks(STAGE,deadline,runtime_class=FarVectorRuntime)
    teachers={}
    for name,path in [('nominal',INITIAL),('verified',VERIFIED/'fit_08000.pt')]:
        teacher=FineCompactPolicy(feed_gain=1.)
        teacher.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['policy'])
        teachers[name]=teacher.eval().requires_grad_(False)
    selected=json.loads((PROBES/'result.json').read_text(encoding='utf-8'))['selected']
    assert len(selected)==2 and all('response' in s for s in selected)
    groups=[];source_hashes={};xs=[];gains=[];actions=[]
    for item in selected:
        for name in item['traces']:
            path=PROBES/name;source_hashes[str(path)]=sha(path)
            with np.load(path) as data:
                x=data['observations'].copy();actions.append(data['actions'].copy())
            xs.append(x);gains.extend([item['response']]*len(x))
    groups.append(dict(name='critical',x=torch.tensor(np.vstack(xs),dtype=torch.float32),
        teacher='nominal',gains=torch.tensor(gains,dtype=torch.float64),fraction=.25,batch_size=128))
    local=[]
    for path in sorted(PROBES.glob('baseline_*.npz')):
        if any(ident in path.name for ident in ['corner_02','corner_06','corner_07']):continue
        source_hashes[str(path)]=sha(path)
        with np.load(path) as data:local.append(data['observations'].copy())
    groups.append(dict(name='original',x=torch.tensor(np.vstack(local)),teacher='nominal',fraction=.1875,batch_size=96))
    dense=[]
    for path in [VERIFIED/'eval_8000_dense.npz',ROOT/'direct14_farvector_local_feedback_fit_round21_dense_half_ms/trace.npz']:
        source_hashes[str(path)]=sha(path)
        with np.load(path) as data:dense.append(data['observations'].copy())
    failed=ROOT/'direct14_phase_feedback_round24_dense_half_ms/trace.npz'
    source_hashes[str(failed)]=sha(failed)
    with np.load(failed) as data:
        failed_states=data['observations'][-35:].copy()
    dense.append(np.tile(failed_states,(5,1)))
    groups.append(dict(name='dense',x=torch.tensor(np.vstack(dense)),teacher='verified',fraction=.375,batch_size=192))
    corner7=[]
    geom=checks.geometry['original'];s7=next(c['reference_s_mm'] for c in geom['sharp_corners'] if c['id']=='corner_07')
    for path,stride in [(VERIFIED/'eval_8000_original.npz',20),
                        (ROOT/'direct14_farvector_local_feedback_fit_round21_original_half_ms/trace.npz',40)]:
        source_hashes[str(path)]=sha(path)
        with np.load(path) as data:
            rows=data['rows'];x=data['observations'];ref=data['reference']
        positions=np.vstack([ref[0],rows[(np.arange(1,len(x))*stride)-1,:2]])
        progress=ordered_projection(ref,positions)
        ids=np.flatnonzero(abs(progress-s7)<45.)
        corner7.append(x[ids])
    groups.append(dict(name='corner7',x=torch.tensor(np.vstack(corner7)),teacher='verified',fraction=.0625,batch_size=32))
    path=NOMINAL/'eval_30000_original.npz';source_hashes[str(path)]=sha(path)
    with np.load(path) as data:x=data['observations'].copy()
    straight=np.flatnonzero((abs(x[:,6])<.008)&(abs(x[:,13])<.005))
    ids=np.unique(np.r_[np.arange(100),np.arange(len(x)-100,len(x)),straight[np.linspace(0,len(straight)-1,768).astype(int)]])
    groups.append(dict(name='anchors',x=torch.tensor(x[ids]),teacher='nominal',fraction=.125,batch_size=64))

    def labels(x,group,ids):
        raw=teachers[group['teacher']].actor(x)
        if 'gains' not in group:return raw
        gain=group['gains'][ids]
        steer=raw[:,0].double()*gain[:,2]
        feed=(raw[:,1].double()+1.)*gain[:,0]-1.
        preview=((6.4+5.6*raw[:,2].double())*gain[:,1]-6.4)/5.6
        return torch.stack([steer,feed,preview],dim=-1).to(raw.dtype)

    arrays={}
    for group in groups:
        with torch.no_grad():target=labels(group['x'],group,torch.arange(len(group['x'])))
        arrays[group['name']+'_observations']=group['x'].numpy()
        arrays[group['name']+'_targets']=target.numpy()
    expected=np.vstack(actions)/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])
    error=float(np.max(np.abs(np.clip(arrays['critical_targets'],-1,1)-np.clip(expected,-1,1))))
    assert error<3e-5,error
    np.savez_compressed(STAGE/'supervision.npz',**arrays)
    write(STAGE/'dataset.json',dict(groups=[dict(name=g['name'],samples=len(g['x']),fraction=g['fraction']) for g in groups],
        critical_action_reconstruction_error=error,source_sha256=source_hashes,
        selected_local_responses=selected,new_physics_ticks_for_preparation=0,new_ppo_updates=0,
        label_scope='Local response search plus incumbent feedback preservation;failed-state and neighborhood feedback queries are synthetic supervision,not valid physical trajectories.'))
    policy=FineCompactPolicy(feed_gain=1.)
    policy.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=False)['policy'])
    evaluations,iterations=fit(policy,groups,labels,STAGE,checks,seed=922302,deadline=deadline,
        initial_path=INITIAL,checkpoint_steps=(8000,16000,30000),screen=screen)
    eligible=[e for e in evaluations if e['eligible_for_refinement']]
    best=min(eligible,key=lambda r:(-r['results']['original']['corner_comparisons']['historical_main']['not_worse'],
        r['results']['original']['complete_time_s'])) if eligible else None
    write(STAGE/'result.json',dict(status='offline_fit_completed' if iterations==30000 else 'wall_budget_completed',
        evaluations=evaluations,selected_for_original_half_refinement=best,offline_iterations=iterations,
        new_ppo_updates=0,inherited_ppo_updates=88,initial_checkpoint=str(INITIAL),
        adopted=False,retained_checkpoint=str(VERIFIED/'fit_08000.pt'),wall_s=time.perf_counter()-started,
        goal_complete=False,method_frozen=False))
    print('ROUND_FINISHED',iterations,'eligible',len(eligible),flush=True)


def main():
    if '--worker' in sys.argv:
        worker();return
    verify_bound_inputs()
    assert json.loads((PROBES/'receipt.json').read_text(encoding='utf-8'))['exit_code']==0
    STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/n for n in ['offline_feedback_fit.py','compact_research_eval.py',
        'compact_farvector_runtime.py','compact_fine_policy.py','refine_farvector_candidate.py']]
    hashes={str(p):sha(p) for p in sources}
    for p in sources:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Offline feedback fitting;zero new PPO updates',wall_budget_s=1800,child_timeout_s=1740,
        worker_deadline_s=1640,source_sha256=hashes,initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),
        evidence_gap='Round24 nominal progress fails dense at0.5ms and misses original corners02/06;corner07 also regresses at0.5ms.',
        procedure='Fit physically checked actual-arrival response labels;retain round21dense states at both resolutions and its corner07feedback. Current round24model supplies other original feedback and end anchors. Same14x64x64x3actor,no added execution rule or memory.',
        adoption='Both nominal tasks and independent dense0.5ms pass;original<=444.236;>=32/34/34historical/S2/Fixed corners,maintained total improvement. Then independent original0.5ms audit;joint final gate unchanged.',
        stop='30000iterations,three declared checkpoints,or wall deadline;retain round21unless complete paired verification supports replacement.',
        probe_result_sha256=sha(PROBES/'result.json'),goal_complete=False,method_frozen=False))
    started=time.perf_counter();code=None
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1740)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status='timeout'
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-4000:])


if __name__=='__main__':main()
