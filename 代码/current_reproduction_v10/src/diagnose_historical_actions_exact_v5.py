"""Finite, registered action-branch diagnosis from exact historical failure states."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'

import argparse
import copy
import json
import math
import pickle
import time
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from replay_strong_checkpoint_v4 import HistoricalActor, MODEL, EXPECTED_MODEL
from runtime_physical_time_v5 import install, VERSION

OUT=ROOT/'results/historical_action_diagnosis_exact_v5'
CASES=['final_dense_fold','final_chamfer_chain','validation_zigzag']
OFFSETS=[-.04,-.02,0.,.02,.04]
FACTORS=[.8,1.]
DT=.001


def read(path):
    return json.loads(path.read_text('utf-8'))


def register():
    file=OUT/'registration.json'
    if not file.exists():
        basefile=ROOT/'results/historical_strong_v5/registration.json'
        base=read(basefile)
        cases=[x for x in base['cases'] if x['id'] in CASES]
        write_json(file,dict(created_utc=utc(),scope='DEVELOPMENT ONLY; local action diagnosis, not complete-program performance',
            cases=cases,model_sha256=EXPECTED_MODEL,model_updated=False,runtime_version=VERSION,dt_s=DT,
            amendment='Repeat the same 180 prescribed branches with float64 physical decode exactly matching DirectCompactRuntime; earlier float32 diagnostic retained in historical_action_diagnosis_v5',
            superseded_diagnostic_registration_sha256=sha(ROOT/'results/historical_action_diagnosis_v5/registration.json'),
            states_per_case=6,state_rule='The last six actor-decision start states, including the final failing decision, from the registered unchanged historical-policy replay',
            steering_offsets_radian=OFFSETS,physical_feed_factors=FACTORS,preview='Preserve original physical and normalized preview',
            action_rule='Add physical steering offset then divide by pi/2; multiply decoded feed then map by feed/50-1; clip only to existing normalized decoder domain',
            branches_per_state=10,expected_total_branches=180,
            execution_rule='Restore exact runtime.state in a fresh common FarVectorRuntime, set the same max_ticks, execute one real common 20 ms decision (retain earlier failure). No simplified proxy integration.',
            control='Reproduce archived observations, normalized actions and rows exactly; offset zero/feed factor one must reproduce each original 20 ms or failed prefix.',
            acceptance='Local branch only: full 20 ms, every command interval inside ordered corridor, vector constraints from finite differences with actual initial velocity/acceleration. No complete-program or final-rest success claim.',
            interpretation='A surviving 20 ms branch demonstrates local feasible action room only. Failure of all ten tested branches does not prove inevitable failure or absence of a different braking maneuver.',
            retention='All 180 branch traces, failed prefixes, exact restored states and original replay are retained; no runtime, gate, tolerance, weights or completion changes.',
            parent_registration_sha256=sha(basefile),
            source_sha256={name:sha(ROOT/'src'/name) for name in list(base['source_sha256'])+['diagnose_historical_actions_exact_v5.py']},
            failed_trace_sha256={case:sha(ROOT/'results/historical_strong_v5'/f'{case}_dt1000us.npz') for case in CASES}))
    r=read(file)
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Historical model changed')
    for name,expected in r['source_sha256'].items():
        if sha(ROOT/'src'/name)!=expected:raise ValueError('Registered source changed: '+name)
    for case,expected in r['failed_trace_sha256'].items():
        if sha(ROOT/'results/historical_strong_v5'/f'{case}_dt1000us.npz')!=expected:raise ValueError('Registered failed trace changed')
    return r


def environment(reference):
    from compact_farvector_runtime import FarVectorRuntime
    env=FarVectorRuntime(reference,potential_speed=100.)
    env.env.max_ticks=int(round((5.+np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()/5.)/DT))
    return env


def step_retained(env,action):
    error=None;info=None;rows=np.empty((0,13));started=time.perf_counter()
    try:
        _,_,_,info=env.step(action)
        rows=info['rows']
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        frame=exc.__traceback__
        while frame is not None:
            if frame.tb_frame.f_code.co_name=='step' and frame.tb_frame.f_code.co_filename.endswith('joint_runtime.py'):
                candidate=frame.tb_frame.f_locals.get('rows',[])
                if len(candidate):rows=np.asarray(candidate)
            frame=frame.tb_next
    return rows,info,error,time.perf_counter()-started


def decode(action):
    import joint_runtime as joint
    # Match DirectCompactRuntime.step's float64 conversion before inverse tanh.
    action=np.asarray(action,dtype=float)
    latent=np.arctanh(np.clip(action,-1.+1e-10,1.-1e-10))
    return joint.JointRuntime.decode_action(latent,None)


def local_audit(env,state,rows,error):
    p0=state['physical']['position'];v0=state['physical']['velocity'];a0=state['physical']['acceleration']
    positions=np.vstack([p0,rows[:,:2]])
    extra=False
    if not np.array_equal(env.position,positions[-1]):
        positions=np.vstack([positions,env.position]);extra=True
    v=np.diff(positions,axis=0)/DT
    a=np.diff(np.vstack([v0,v]),axis=0)/DT
    j=np.diff(np.vstack([a0,a]),axis=0)/DT
    peaks=[float(np.linalg.norm(x,axis=1).max(initial=0.)) for x in [v,a,j]]
    ratios=np.asarray(peaks)/[100.,2000.,62500.]
    vector=bool(np.all(ratios<=1.+2e-5))
    # A separately restored tracker consumes each entire saved command interval.
    from geometry_spec import GateTracker
    tracker=GateTracker(env.corridor,env.corridor.reference[0])
    for key,value in state['physical']['tracker'].items():
        setattr(tracker,key,copy.deepcopy(value))
    for p in positions[1:]:tracker.update(p)
    local_geometry=bool(tracker.inside)
    start_ticks=state['physical']['ticks'];executed=env.ticks-start_ticks
    survived=bool(error is None and executed==round(.02/DT) and not env.done and vector and local_geometry)
    return dict(local_20ms_feasible=survived,executed_ticks=executed,elapsed_branch_s=executed*DT,
        kinematic_pass=vector,vector_peaks=peaks,vector_limit_ratios=ratios.tolist(),
        every_command_interval_inside=local_geometry,
        gate_frontier_start=state['physical']['tracker']['frontier'],gate_frontier_end=tracker.frontier,
        error=error,executor_done=bool(env.done),executor_reason=env.reason,
        appended_state_position_after_exception=extra,
        endpoint_position_mm=env.position.tolist(),endpoint_velocity_mm_s=env.velocity.tolist(),
        endpoint_acceleration_mm_s2=env.acceleration.tolist(),
        complete_program_success=False,completion_note='Local branch diagnostic does not test full traversal or final rest'),positions


def run(case):
    registration=register();row=next(x for x in registration['cases'] if x['id']==case)
    caseout=OUT/case
    if (caseout/'summary.json').exists():raise FileExistsError(caseout/'summary.json')
    caseout.mkdir(parents=True,exist_ok=True)
    reference=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
    install(dt=DT,tolerance=row['evaluation_tolerance_mm'])
    archived=ROOT/'results/historical_strong_v5'/f'{case}_dt1000us.npz'
    with np.load(archived) as data:
        archived_data={key:data[key].copy() for key in data.files}
    count=len(archived_data['actions']);indices=list(range(max(0,count-6),count))
    if len(indices)!=6:raise ValueError('Insufficient six observed decisions')
    env=environment(reference);actor=HistoricalActor();states=[];allrows=[];offset=0
    for i in range(count):
        observation=env.observation();action=actor.act(observation,True)[0]
        if not np.array_equal(observation,archived_data['observations'][i]):raise AssertionError('Observation replay mismatch')
        if not np.array_equal(action,archived_data['actions'][i]):raise AssertionError('Historical action replay mismatch')
        state=copy.deepcopy(env.state())
        rows,info,error,wall=step_retained(env,action)
        expected=archived_data['rows'][offset:offset+len(rows)]
        if not np.array_equal(rows,expected):raise AssertionError('Original row replay mismatch')
        if i in indices:
            states.append(dict(decision_index=i,state=state,observation=observation.copy(),
                action=action.copy(),physical_action=decode(action),original_rows=rows.copy(),original_error=error))
        allrows.append(rows);offset+=len(rows)
        if env.done and i!=count-1:raise AssertionError('Original replay terminated early')
    if offset!=len(archived_data['rows']):raise AssertionError('Original prefix length mismatch')
    with (caseout/'exact_states.pkl').open('wb') as stream:pickle.dump(states,stream,protocol=5)
    np.savez_compressed(caseout/'original_exact_replay.npz',reference=reference,rows=np.vstack(allrows),
        observations=archived_data['observations'],actions=archived_data['actions'])
    records=[]
    for item in states:
        i=item['decision_index'];state=item['state'];physical=item['physical_action']
        for angle in OFFSETS:
            for factor in FACTORS:
                action=item['action'].astype(float).copy()
                action[0]=(physical[0]+angle)/(math.pi/2.)
                action[1]=physical[1]*factor/50.-1.
                action=np.clip(action,-1.+1e-10,1.-1e-10)
                # Preserve the original exact command for the control branch.
                if angle==0. and factor==1.:action=item['action'].copy()
                branch=environment(reference);branch.restore(state)
                if not np.array_equal(branch.observation(),item['observation']):raise AssertionError('Restored observation mismatch')
                rows,info,error,wall=step_retained(branch,action)
                audit,positions=local_audit(branch,state,rows,error)
                control=(angle==0. and factor==1.)
                equal=bool(np.array_equal(rows,item['original_rows']) and error==item['original_error']) if control else None
                if control and not equal:raise AssertionError('Original action branch mismatch')
                stem=f'decision{i:03d}_steer{angle:+.2f}_feed{factor:.1f}'
                file=caseout/(stem+'.npz')
                if file.exists():raise FileExistsError(file)
                np.savez_compressed(file,reference=reference,positions=positions,rows=rows,
                    initial_position=state['physical']['position'],initial_velocity=state['physical']['velocity'],
                    initial_acceleration=state['physical']['acceleration'],observation=item['observation'],
                    original_action=item['action'],branch_action=action,
                    original_physical_action=physical,branch_physical_action=decode(action))
                audit.update(path_id=case,decision_index=i,start_time_s=state['physical']['ticks']*DT,
                    original_physical_action=physical.tolist(),branch_physical_action=decode(action).tolist(),
                    requested_steering_offset_radian=angle,requested_physical_feed_factor=factor,
                    original_control=control,exact_control_equal=equal,trace_sha256=sha(file),trace=str(file),
                    branch_wall_s=wall,registration_sha256=sha(OUT/'registration.json'))
                write_json(file.with_suffix('.json'),audit);records.append(audit)
    summaries=[]
    for item in states:
        subset=[r for r in records if r['decision_index']==item['decision_index']]
        ctrl=next(r for r in subset if r['original_control'])
        summaries.append(dict(decision_index=item['decision_index'],start_time_s=ctrl['start_time_s'],
            original_physical_action=ctrl['original_physical_action'],original_20ms_feasible=ctrl['local_20ms_feasible'],
            feasible_branch_count=sum(r['local_20ms_feasible'] for r in subset),
            feasible_actions=[dict(steering_offset_radian=r['requested_steering_offset_radian'],
                                  physical_feed_factor=r['requested_physical_feed_factor']) for r in subset if r['local_20ms_feasible']],
            interpretation='Local safe action margin among tested commands' if any(r['local_20ms_feasible'] for r in subset) else
                'No tested command survives 20 ms from this state; no proof of globally unavoidable collision or impossible braking'))
    summary=dict(created_utc=utc(),path_id=case,scope=registration['scope'],expected_branches=60,actual_branches=len(records),
        exact_original_replay=True,all_six_controls_exact=all(r['exact_control_equal'] for r in records if r['original_control']),
        state_file_sha256=sha(caseout/'exact_states.pkl'),model_sha256=EXPECTED_MODEL,states=summaries,
        all_records=records,registration_sha256=sha(OUT/'registration.json'))
    write_json(caseout/'summary.json',summary)
    print(json.dumps(dict(case=case,branches=len(records),states=summaries),indent=2),flush=True)


def summarize():
    records=[read(OUT/case/'summary.json') for case in CASES]
    data=dict(created_utc=utc(),scope='Local diagnostic only; no full-program successes',
        total_branches=sum(r['actual_branches'] for r in records),cases=records,
        interpretation='Only the prescribed ten actions per state were tested. Local survival cannot establish downstream completion; no surviving branch does not establish dynamic inevitability or impossible braking.')
    write_json(ROOT/'audit/historical_action_diagnosis_exact_v5.json',data)
    lines=['# Historical actor local action diagnosis','',data['scope'],data['interpretation'],'',
           '| Program | Decision start (s) | Original survives 20 ms | Feasible branches / 10 |',
           '|---|---:|---|---:|']
    for row in records:
        for state in row['states']:
            lines.append(f"| {row['path_id']} | {state['start_time_s']:.3f} | {state['original_20ms_feasible']} | {state['feasible_branch_count']} |")
    lines.extend(['','All original replays and six zero-offset/unit-feed controls per program matched the archived rows exactly.',
        'Offsets are physical radians [-0.04,-0.02,0,0.02,0.04]; feed factors are [0.8,1]. Preview is unchanged. Every branch uses the common v5 executor and the actual initial position, velocity, acceleration and gate state.',
        '',f"Retained {data['total_branches']} branch traces, including all failures and the 18 restored states.",''])
    (ROOT/'audit/historical_action_diagnosis_exact_v5.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(total_branches=data['total_branches'],audit=str(ROOT/'audit/historical_action_diagnosis_exact_v5.md'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.case:run(args.case)
    elif args.all:
        for case in CASES:run(case)
        summarize()
    elif args.summarize:summarize()
    else:parser.error('Choose case, all or summarize')
