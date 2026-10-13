"""Local response robustness over five decision phases; never a full-task claim."""
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
from evaluate_historical_dense import ROOT,BASE,sha,write,verify_bound_inputs

STAGE=ROOT/'direct14_local_decision_phase_probe_round23'
CONTEXTS=ROOT/'direct14_previewcue_ppo_seed922302_round13/all34_contexts.pt'
TEACHER=ROOT/'direct14_previewcue_ppo_seed922302_round13/checkpoint_0024.pt'
PHASES_MS=[0,4,8,12,16]
RESPONSES=[(1.,1.,1.),(1.25,1.,1.),(1.5,1.,1.),(1.,1.05,1.),
    (1.25,1.05,1.),(1.25,1.10,1.),(1.25,1.15,1.),(1.5,1.05,1.),(1.,1.10,1.),
    (1.25,1.20,1.),(1.25,1.15,1.015),(1.25,1.15,1.03),
    (1.25,1.20,1.015),(1.25,1.20,1.03)]
CRITICAL={'corner_06','corner_10','corner_20','corner_23'}


def worker():
    import numpy as np
    import torch
    import joint_runtime
    from compact_joint_response import JointResponsePolicy
    from compact_farvector_runtime import FarVectorRuntime
    from compact_previewcue_runtime import PreviewCueRuntime
    from reconstruct_farvector_observations import set_physical
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import ordered_projection
    torch.set_num_threads(1)
    started=time.perf_counter()
    deadline=started+1650.
    folder=BASE/'paper_evidence_review/natural_closed_case'
    ref=np.loadtxt(folder/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((folder/'geometry.json').read_text(encoding='utf-8'))
    corners={c['id']:c for c in geometry['sharp_corners']}
    contexts=torch.load(CONTEXTS,map_location='cpu',weights_only=False)
    saved=torch.load(TEACHER,map_location='cpu',weights_only=False)
    protocol=json.loads((ROOT/'joint_lowdim_acceptance_20260922.json').read_text(encoding='utf-8'))
    old={k:json.loads(Path(v['path']).read_text(encoding='utf-8')) for k,v in protocol['original']['comparators'].items()}
    budgets={c['id']:min(m['corners'][i]['speed_deficit_mm'] for m in old.values()) for i,c in enumerate(geometry['sharp_corners'])}
    matrix=[]
    new_ticks=0
    restored_ticks=0
    phase_audits=[]

    def physical(initial,rows):
        raw_a=np.diff(np.vstack([initial['velocity'],rows[:,2:4]]),axis=0)/.001
        raw_j=np.diff(np.vstack([initial['acceleration'],raw_a]),axis=0)/.001
        peaks=[float(np.linalg.norm(rows[:,2:4],axis=1).max()),float(np.linalg.norm(raw_a,axis=1).max()),float(np.linalg.norm(raw_j,axis=1).max())]
        return dict(peaks=peaks,passed=bool(np.all(np.asarray(peaks)<=[100.000001,2000.001,62500.02])))

    for response_index,(feed_gain,preview_gain,feed_scale) in enumerate(RESPONSES):
        if time.perf_counter()>deadline-15.:
            break
        teacher=JointResponsePolicy(feed_gain=feed_gain,preview_gain=preview_gain)
        teacher.load_state_dict(saved['policy'])
        teacher.eval()
        for context in contexts:
            if response_index>=9 and context['id'] not in CRITICAL:
                continue
            for phase in PHASES_MS:
                if time.perf_counter()>deadline-15.:
                    break
                env=FarVectorRuntime(ref)
                env.restore(context['state'])
                assert np.array_equal(PreviewCueRuntime.observation(env),context['observation'])
                warm_rows=np.empty((0,13))
                if phase:
                    # Generate a local reset by physically holding the previous
                    # command for a further4/8/12/16ms. This is initialization,
                    # not an altered action period in a complete task.
                    before=env.env.state()
                    held=env.env.held.copy()
                    raw=held/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])
                    old_macro=joint_runtime.MACRO_TICKS
                    try:
                        joint_runtime.MACRO_TICKS=phase
                        _,_,_,info=env.step(raw)
                        warm_rows=info['rows']
                    finally:
                        joint_runtime.MACRO_TICKS=old_macro
                    assert len(warm_rows)==phase and not env.done
                    pa=physical(before,warm_rows)
                    ga=analytic_command_audit(before['position'],warm_rows,geometry)
                    assert pa['passed'] and ga['analytic_corridor_pass']
                    new_ticks+=len(warm_rows)
                    if response_index==0:
                        phase_audits.append(dict(id=context['id'],phase_ms=phase,physical=pa,
                            maximum_distance_mm=ga['maximum_evaluated_distance_mm']))
                initial=env.env.state()
                start_position=env.position.copy()
                observations=[]
                old_observations=[]
                actions=[]
                raw_actions=[]
                chunks=[]
                source=None
                if phase==0 and response_index<9:
                    source=ROOT/'direct14_joint_response_probe_round14'/f'response_{response_index:02d}'/(context['id']+'.npz')
                    with np.load(source) as data:
                        rows=data['rows']
                        recorded=data['actions']
                    for i,action in enumerate(recorded):
                        if i:
                            for row in rows[(i-1)*20:i*20]:
                                env.env.tracker.update(row[:2])
                            set_physical(env.env,rows[i*20-1])
                        observations.append(env.observation())
                        old_observations.append(PreviewCueRuntime.observation(env))
                        actions.append(action)
                    restored_ticks+=len(rows)
                    old_result=json.loads((source.parent/'local_result.json').read_text(encoding='utf-8'))['corners']
                    source_record=next(v for v in old_result if v['id']==context['id'])
                    local_done=not source_record['local_valid']
                    reason=source_record['reason']
                else:
                    for _ in range(48):
                        x=env.observation()
                        old_x=PreviewCueRuntime.observation(env)
                        z=teacher.act(old_x,True)[0]
                        if feed_scale!=1.:
                            z=z.copy()
                            z[1]=float((float(z[1])+1.)*feed_scale-1.)
                        _,_,_,info=env.step(z)
                        observations.append(x)
                        old_observations.append(old_x)
                        actions.append(info['held_action'])
                        raw_actions.append(z)
                        chunks.append(info['rows'])
                        if env.done:
                            break
                    rows=np.vstack(chunks)
                    new_ticks+=len(rows)
                    local_done=env.done
                    reason=env.reason
                p=ordered_projection(ref,rows[:,:2])
                c=corners[context['id']]
                lo=c['reference_s_mm']-c['window_before_mm']
                hi=c['reference_s_mm']+c['window_after_mm']
                mask=(p>=lo)&(p<=hi)
                complete=bool(p.min()<lo and p.max()>hi)
                speed=np.linalg.norm(rows[:,2:4],axis=1)
                ga=analytic_command_audit(start_position,rows,geometry)
                pa=physical(initial,rows)
                valid=bool(not local_done and complete and ga['analytic_corridor_pass'] and pa['passed'])
                crossing=np.interp([lo,hi],np.maximum.accumulate(p),np.arange(1,len(rows)+1)*.001)
                name=f'r{response_index:02d}_{context["id"]}_p{phase:02d}'
                np.savez_compressed(STAGE/(name+'.npz'),rows=rows,observations=np.asarray(observations),
                    original_observations=np.asarray(old_observations),actions=np.asarray(actions),
                    raw_actions=np.asarray(raw_actions),initial_velocity=initial['velocity'],
                    initial_acceleration=initial['acceleration'],start_position=start_position,
                    warmup_rows=warm_rows,decision_times_local_s=np.arange(len(actions))*.02)
                item=dict(name=name,id=context['id'],response=response_index,phase_ms=phase,
                    feed_gain=feed_gain,preview_gain=preview_gain,feed_scale=feed_scale,local_valid=valid,
                    reason=reason,window_complete=complete,deficit_mm=float(np.maximum(100-speed[mask],0).sum()*.001),
                    budget_mm=budgets[context['id']],window_time_s=float(crossing[1]-crossing[0]) if complete else None,
                    maximum_distance_mm=ga['maximum_evaluated_distance_mm'],physical=pa,
                    reused_source=str(source) if source else None,reused_source_sha256=sha(source) if source else None)
                matrix.append(item)
            write(STAGE/'matrix.json',matrix)
        print('RESPONSE_FINISHED',response_index,'records',len(matrix),'new_ticks',new_ticks,
              'wall',time.perf_counter()-started,flush=True)
    selections=[]
    for context in contexts:
        candidates=[]
        for response_index in range(len(RESPONSES)):
            records=[r for r in matrix if r['id']==context['id'] and r['response']==response_index]
            if len(records)!=len(PHASES_MS) or not all(r['local_valid'] for r in records):
                continue
            worst=max(r['deficit_mm']-r['budget_mm'] for r in records)
            candidates.append(dict(id=context['id'],response=response_index,
                feed_gain=records[0]['feed_gain'],preview_gain=records[0]['preview_gain'],feed_scale=records[0]['feed_scale'],
                phases=len(records),worst_deficit_excess_mm=worst,all_phase_budget_pass=worst<=1e-7,
                mean_window_time_s=float(np.mean([r['window_time_s'] for r in records])),
                traces=[r['name']+'.npz' for r in records]))
        candidates.sort(key=lambda r:(not r['all_phase_budget_pass'],max(r['worst_deficit_excess_mm']-1e-7,0),r['mean_window_time_s']))
        selections.append(candidates[0] if candidates else dict(id=context['id'],no_all_phase_valid_response=True))
    write(STAGE/'phase_initialization_audit.json',dict(cases=phase_audits,
        scope='Locally generated physical reset states by previous-command holding; not a complete-task execution or modified complete-task action period'))
    write(STAGE/'result.json',dict(status='finite_probe_completed' if len(matrix)==1630 else 'wall_budget_completed',
        records=len(matrix),expected_records=1630,selected=selections,
        all_phase_valid_corners=sum('response' in s for s in selections),
        all_phase_budget_corners=sum(s.get('all_phase_budget_pass',False) for s in selections),
        new_physics_ticks=new_ticks,reused_physics_ticks=restored_ticks,new_ppo_updates=0,new_fit_updates=0,
        wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False,
        scope='Local phase robustness and training material selection only. Different local teachers are not one executable policy or full-task result.'))
    print('ROUND_FINISHED',len(matrix),'valid',sum('response' in s for s in selections),flush=True)


def main():
    if '--worker' in sys.argv:
        worker()
        return
    verify_bound_inputs()
    STAGE.mkdir(exist_ok=False)
    (STAGE/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/name for name in ['compact_joint_response.py','compact_fine_policy.py',
        'compact_farvector_runtime.py','compact_previewcue_runtime.py','direct_compact_runtime.py',
        'compact_control.py','reconstruct_farvector_observations.py']]+[BASE/'natural_case_metrics.py',BASE/'natural_case_audit.py']
    hashes={str(p):sha(p) for p in sources}
    for p in sources:
        shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        wall_budget_s=1800,worker_deadline_s=1650,child_timeout_s=1740,source_sha256=hashes,
        phases_ms=PHASES_MS,responses=RESPONSES,extra_responses_only_on=sorted(CRITICAL),
        teacher=str(TEACHER),teacher_sha256=sha(TEACHER),contexts=str(CONTEXTS),contexts_sha256=sha(CONTEXTS),
        evidence_gap='Round21fit8000 succeeds but misses4historical corner budgets and34ms of the minimum historical time margin. Its126degree corner visits states poorly covered by the one-phase local training traces. Evaluate whether response teachers remain valid and efficient across a20ms decision cycle.',
        procedure='Physically create0/4/8/12/16ms local reset phases by holding the previously witnessed command,then use ordinary20ms actions for48decisions. All34 corners and9prior responses;5additional preview/feed-scale responses only on4remaining corners. Reconstruct existing phase0traces without rerunning their physics. Check actual vector V/A/J,analytic corridor and complete registered window.',
        adoption='Training material only: choose one response per corner valid across all5phases,prioritizing all-phase deficit budgets then mean window time. Never combine these local results into claimed whole-path performance.',
        stop='1630registered local records or wall deadline;keep invalid responses and missing data. No baseline/task/physical-limit changes and no sealed data.',
        new_ppo_updates=0,new_fit_updates=0,goal_complete=False,method_frozen=False))
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
