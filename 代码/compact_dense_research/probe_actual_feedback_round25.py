"""Bounded local response search from the current actor's own arrival states."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,write,sha,verify_bound_inputs

STAGE=ROOT/'direct14_actual_feedback_probe_round25'
SOURCE=ROOT/'direct14_phase_feedback_fit_round24'
INITIAL=SOURCE/'fit_30000.pt'
CRITICAL=['corner_02','corner_06']
RESPONSES=list(itertools.product([1.,1.03,1.06,1.10],[.90,1.,1.10,1.20,1.30],[.8,1.,1.2]))
PHASES=[0,4,8,12,16]


def worker():
    import numpy as np
    import torch
    import joint_runtime
    from compact_fine_policy import FineCompactPolicy
    from compact_farvector_runtime import FarVectorRuntime
    from saved_local_contexts import from_complete_trace
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import ordered_projection
    started=time.perf_counter();deadline=started+1500.
    torch.set_num_threads(1)
    folder=BASE/'paper_evidence_review/natural_closed_case'
    geometry=json.loads((folder/'geometry.json').read_text(encoding='utf-8'))
    corners={c['id']:c for c in geometry['sharp_corners']}
    ref,original,contexts,audit=from_complete_trace(SOURCE/'eval_30000_original.npz',
        SOURCE/'eval_30000_original_gates.json',geometry)
    torch.save(contexts,STAGE/'all34_contexts.pt')
    protocol=json.loads((ROOT/'joint_lowdim_acceptance_20260922.json').read_text(encoding='utf-8'))
    old={k:json.loads(Path(v['path']).read_text(encoding='utf-8')) for k,v in protocol['original']['comparators'].items()}
    budgets={c['id']:min(m['corners'][i]['speed_deficit_mm'] for m in old.values()) for i,c in enumerate(geometry['sharp_corners'])}
    policy=FineCompactPolicy(feed_gain=1.)
    policy.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=False)['policy'])
    policy.eval()
    matrix=[];ticks=0

    def simulate(context,response,phase,label):
        nonlocal ticks
        if time.perf_counter()>deadline-15.:
            raise TimeoutError('Registered local-search deadline')
        feed,preview,steer=response
        env=FarVectorRuntime(ref);env.restore(context['state'])
        if phase:
            held=env.held.copy()
            action=held/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])
            previous=joint_runtime.MACRO_TICKS
            try:
                joint_runtime.MACRO_TICKS=phase
                _,_,_,info=env.step(action)
                ticks+=len(info['rows'])
            finally:
                joint_runtime.MACRO_TICKS=previous
            assert len(info['rows'])==phase and not env.done
            warm=info['rows']
        else:
            warm=np.empty((0,13))
        start=env.env.state();chunks=[];xs=[];actions=[]
        for _ in range(64):
            x=env.observation();z=policy.act(x,True)[0].copy()
            z[0]*=steer
            z[1]=(float(z[1])+1.)*feed-1.
            z[2]=((6.4+5.6*float(z[2]))*preview-6.4)/5.6
            _,_,_,info=env.step(z)
            chunks.append(info['rows']);xs.append(x);actions.append(info['held_action'])
            if env.done:break
        rows=np.vstack(chunks);ticks+=len(rows)
        raw_a=np.diff(np.vstack([start['velocity'],rows[:,2:4]]),axis=0)/.001
        raw_j=np.diff(np.vstack([start['acceleration'],raw_a]),axis=0)/.001
        peaks=[float(np.linalg.norm(rows[:,2:4],axis=1).max()),float(np.linalg.norm(raw_a,axis=1).max()),float(np.linalg.norm(raw_j,axis=1).max())]
        physical=bool(np.all(np.asarray(peaks)<=[100.000001,2000.001,62500.02]))
        ga=analytic_command_audit(start['position'],rows,geometry)
        p=ordered_projection(ref,rows[:,:2]);corner=corners[context['id']]
        lo=corner['reference_s_mm']-corner['window_before_mm'];hi=corner['reference_s_mm']+corner['window_after_mm']
        complete=bool(p.min()<lo and p.max()>hi)
        mask=(p>=lo)&(p<=hi);speed=np.linalg.norm(rows[:,2:4],axis=1)
        crossing=np.interp([lo,hi],np.maximum.accumulate(p),np.arange(1,len(rows)+1)*.001)
        name=f'{label}_{context["id"]}_p{phase:02d}'
        np.savez_compressed(STAGE/(name+'.npz'),rows=rows,observations=np.asarray(xs),
            actions=np.asarray(actions),warmup_rows=warm,start_position=start['position'],
            initial_velocity=start['velocity'],initial_acceleration=start['acceleration'])
        item=dict(name=name,id=context['id'],response=list(response),phase_ms=phase,
            valid=bool(not env.done and complete and physical and ga['analytic_corridor_pass']),
            reason=env.reason,window_complete=complete,peaks=peaks,
            deficit_mm=float(np.maximum(100-speed[mask],0).sum()*.001),budget_mm=budgets[context['id']],
            window_time_s=float(crossing[1]-crossing[0]) if complete else None,
            maximum_distance_mm=ga['maximum_evaluated_distance_mm'])
        if label=='baseline':
            expected=original[context['tick']:context['tick']+len(rows)]
            # Jerk magnifies last-bit heading reconstruction; compare physical
            # position and velocity separately from finite-difference quantities.
            item['replay_max_xyv_error']=float(np.max(np.abs(rows[:,:4]-expected[:,:4])))
            item['replay_max_all_error']=float(np.max(np.abs(rows-expected)))
            assert item['valid'] and item['replay_max_xyv_error']<2e-5,item
        matrix.append(item)
        return item

    for context in contexts:
        simulate(context,(1.,1.,1.),0,'baseline')
    audit.update(contexts_sha256=sha(STAGE/'all34_contexts.pt'),
        maximum_local_replay_xyv_error=max(r['replay_max_xyv_error'] for r in matrix),
        all34_unmodified_local_checks_pass=True)
    write(STAGE/'context_reconstruction_audit.json',audit)
    print('ACTUAL_CONTEXTS_CHECKED',len(contexts),'wall',time.perf_counter()-started,flush=True)
    for ident in CRITICAL:
        context=next(c for c in contexts if c['id']==ident)
        for i,response in enumerate(RESPONSES):
            simulate(context,response,0,f'r{i:03d}')
        write(STAGE/'matrix.json',matrix)
        print('SEARCHED',ident,'records',len(matrix),'wall',time.perf_counter()-started,flush=True)
    selected=[]
    for ident in CRITICAL:
        context=next(c for c in contexts if c['id']==ident)
        nominal=[r for r in matrix if r['id']==ident and r['phase_ms']==0 and r['valid'] and r['name'].startswith('r')]
        nominal.sort(key=lambda r:(r['deficit_mm']>r['budget_mm']+1e-7,max(r['deficit_mm']-r['budget_mm']-1e-7,0.),r['window_time_s']))
        options=[]
        for candidate in nominal[:8]:
            records=[candidate]
            label=candidate['name'].split('_')[0]
            for phase in PHASES[1:]:
                records.append(simulate(context,candidate['response'],phase,label))
            if all(r['valid'] for r in records):
                excess=max(r['deficit_mm']-r['budget_mm'] for r in records)
                options.append(dict(id=ident,response=candidate['response'],worst_excess_mm=excess,
                    all_phase_budget_pass=excess<=1e-7,mean_window_time_s=float(np.mean([r['window_time_s'] for r in records])),
                    traces=[r['name']+'.npz' for r in records]))
        options.sort(key=lambda r:(not r['all_phase_budget_pass'],max(r['worst_excess_mm']-1e-7,0.),r['mean_window_time_s']))
        selected.append(options[0] if options else dict(id=ident,no_valid_response=True))
        write(STAGE/'matrix.json',matrix)
        print('SELECTED',selected[-1],flush=True)
    write(STAGE/'result.json',dict(selected=selected,new_physics_ticks=ticks,
        wall_s=time.perf_counter()-started,records=len(matrix),new_ppo_updates=0,new_fit_updates=0,
        source_checkpoint=str(INITIAL),source_checkpoint_sha256=sha(INITIAL),
        goal_complete=False,method_frozen=False,
        scope='Local feedback targets only;different corner responses are not one whole-task policy.'))


def main():
    if '--worker' in sys.argv:
        worker();return
    verify_bound_inputs()
    STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/n for n in ['saved_local_contexts.py','compact_farvector_runtime.py',
        'compact_fine_policy.py','compact_control.py']]
    hashes={str(p):sha(p) for p in sources}
    for p in sources:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Local physical response search,not PPO or full-task acceptance',
        wall_budget_s=1800,worker_deadline_s=1500,child_timeout_s=1740,
        source_sha256=hashes,checkpoint=str(INITIAL),checkpoint_sha256=sha(INITIAL),
        responses=RESPONSES,critical_corners=CRITICAL,phases_ms=PHASES,
        evidence_gap='Round24 fit30000 nominally misses only historical corners02/06;prior local sources have different arrival states.',
        procedure='Reconstruct current actor arrival states from its bound complete trace;validate all34 by unmodified local replay. Search60 three-action response gains on each of2remaining corners;check best8each over five local phases with unchanged20ms ordinary decisions.',
        adoption='Only physically valid complete-window local responses with better deficit budgets become training material. Keep all negative results. No candidate or baseline change follows from local probes alone.',
        stop='Finite registered matrix or wall deadline;no sealed data;limits,all34windows,and reference remain fixed.',
        new_ppo_updates=0,goal_complete=False,method_frozen=False))
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
