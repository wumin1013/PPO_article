"""Registered physical-state teacher collection; no student optimizer exists.

Every source interval is witnessed. Both action labels share the same physical
state. All branch outcomes and command prefixes remain in the output archive.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import importlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from frozen_policy import FrozenPolicy

MODEL=ROOT/'baseline_records/supplementary_reproduction/models/main.pt'
MODEL_HASH='c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764'
TRACE_ROOT=ROOT/'baseline_records/supplementary_reproduction/records/J-NNC'
DT=.001
NOISE_SEED=2026101105
NOISE_STD=np.array([.002,.01,.005])
SOURCE_KEYS=('position','velocity','acceleration','heading','held','ticks','decisions',
             'frontier','pending_negative','segment_count','start_error','inside','rest_ticks')


def wrap(x):
    return (x+np.pi)%(2*np.pi)-np.pi


def source_case(case):
    trace_path=TRACE_ROOT/f'{case}_1ms.npz'
    with np.load(trace_path) as data:
        trace={key:data[key].copy() for key in data.files}
    geometry_path=ROOT/'baseline_records/supplementary_reproduction/cases'/case/'geometry.json'
    geometry=json.loads(geometry_path.read_text('utf-8'))
    chord=max([primitive.get('chord_error_bound_mm',0.) for primitive in geometry['primitives']]+[0.])
    assert float(trace['dt_s'])==DT
    ticks=np.rint(trace['decision_times_s']/DT).astype(np.int64)
    assert np.max(abs(ticks*DT-trace['decision_times_s']))<1e-10
    assert ticks[0]==0 and np.all(np.diff(ticks)==20)
    count=len(ticks)
    assert count=={'original':22204,'dense':737}[case]
    indices=np.unique(np.rint(np.linspace(0,count-1,4096)).astype(np.int64)) if case=='original' else np.arange(count)
    assert len(indices)=={'original':4096,'dense':737}[case]
    return trace,ticks,indices,1.25-chord,dict(case=case,trace=str(trace_path.relative_to(ROOT)),
        trace_sha256=sha(trace_path),geometry_sha256=sha(geometry_path),source_decisions=count,
        selected_decisions=len(indices),nominal_tolerance_mm=1.25,chord_error_bound_mm=chord,
        evaluation_tolerance_mm=1.25-chord,selection='Rounded equally spaced decision indices including both ends' if case=='original' else 'All decisions')


def environment(reference,tolerance,installer):
    installer(dt=DT,tolerance=tolerance)
    from compact_farvector_runtime import FarVectorRuntime
    env=FarVectorRuntime(reference,potential_speed=100.)
    env.env.max_ticks=round((5.+np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()/5.)/DT)
    return env


def restore_source(env,state):
    """Restore acceptance state; the immutable source prefix event ledger is separate."""
    core=env.env
    for key in ('position','velocity','acceleration','held'):
        setattr(core,key,np.asarray(state[key],float).copy())
    for key in ('heading','ticks','decisions','rest_ticks'):
        setattr(core,key,copy.deepcopy(state[key]))
    core.done=False;core.success=False;core.reason=None;core.brake=None;core.brake_index=0
    core.pending_pulse=np.zeros(3);core._executed=None
    tracker=core.tracker
    tracker.position=core.position.copy()
    for key in ('frontier','pending_negative','segment_count','start_error','inside'):
        setattr(tracker,key,copy.deepcopy(state[key]))
    tracker.events=[]
    if 'update' in vars(tracker):delattr(tracker,'update')
    for key in core.experimental_projector.counts:core.experimental_projector.counts[key]=0


def capture_state(env):
    core=env.env
    state={key:copy.deepcopy(getattr(core,key)) for key in ('position','velocity','acceleration','heading','held','ticks','decisions','rest_ticks')}
    state.update({key:copy.deepcopy(getattr(core.tracker,key)) for key in ('frontier','pending_negative','segment_count','start_error','inside')})
    return state


def guide_angle(core,steering,feed,preview):
    old=core.held.copy()
    try:
        core.held=np.array([steering,feed,preview])
        delta=core.guidance_target()-core.position
        return math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>1e-12 else core.heading
    finally:
        core.held=old


def labels(env,teacher):
    env.env._executed=None
    obs=env.observation().copy()
    assert obs.shape==(14,) and np.isfinite(obs).all()
    raw=teacher.mean(obs).numpy().astype(float)
    bounded=np.clip(raw,-1.+1e-10,1.-1e-10)
    theta=bounded[0]*np.pi/2;feed=50.*(1.+bounded[1]);preview=6.4+5.6*bounded[2]
    beta=guide_angle(env.env,theta,feed,preview)
    fixed_beta=guide_angle(env.env,theta,feed,3.)
    fixed_theta=wrap(beta+theta-fixed_beta)
    raw_two=np.array([fixed_theta/(np.pi/2),raw[1]])
    bounded_two=np.clip(raw_two,-1.+1e-10,1.-1e-10)
    error=float(wrap(fixed_beta+bounded_two[0]*np.pi/2-beta-theta))
    clipped=bool(abs(raw_two[0])>1.-1e-10)
    if not clipped:assert abs(error)<1e-10
    return dict(observations=obs,targets_three=np.clip(raw,-1,1),targets_three_raw=raw,
        targets_two=np.clip(raw_two,-1,1),targets_two_raw=raw_two,
        teacher_physical_action=np.array([theta,feed,preview]),teacher_guide_angle=beta,
        fixed_guide_angle=fixed_beta,fixed_theta_unclipped=fixed_theta,
        fixed_steering_clipped=clipped,fixed_direction_error_rad=error)


def pack_records(path,records):
    if not records:
        np.savez_compressed(path,record_count=np.array(0))
        return
    keys=records[0].keys()
    assert all(item.keys()==keys for item in records)
    np.savez_compressed(path,**{key:np.asarray([item[key] for item in records]) for key in keys})


def reconstruct_case(case,out,installer,teacher):
    started=time.perf_counter();cpu=time.process_time()
    trace,ticks,indices,tolerance,source=source_case(case)
    env=environment(trace['reference'],tolerance,installer)
    from geometry_spec import GateTracker
    witness=GateTracker(env.corridor,trace['reference'][0])
    states=[];records=[];cursor=0
    rows=trace['rows']
    for decision in indices:
        tick=int(ticks[decision])
        while cursor<tick:
            witness.update(rows[cursor,:2]);cursor+=1
        if not witness.inside:raise ValueError(f'Source prefix outside registered corridor: {case}:{tick}')
        before=rows[tick-1] if tick else None
        state=dict(position=trace['reference'][0].copy() if tick==0 else before[:2].copy(),
            velocity=np.zeros(2) if tick==0 else before[2:4].copy(),
            acceleration=np.zeros(2) if tick==0 else before[4:6].copy(),
            heading=math.atan2(*env.corridor.tangents[0][::-1]) if tick==0 else math.atan2(before[9],before[8]),
            held=env.held.copy() if decision==0 else trace['actions'][decision-1].copy(),
            ticks=tick,decisions=int(decision),rest_ticks=0,
            frontier=witness.frontier,pending_negative=witness.pending_negative,
            segment_count=witness.segment_count,start_error=witness.start_error,inside=witness.inside)
        if tick:
            assert np.linalg.norm(before[8:10])>1e-12 and before[12]<.5
            assert np.max(abs(state['position']-witness.position))==0.
        restore_source(env,state)
        labelled=labels(env,teacher)
        labelled.update({key:state[key] for key in SOURCE_KEYS})
        labelled.update(case=case,source_decision=int(decision),source_tick=tick,
            archived_observations=trace['observations'][decision].copy(),source_gate_event_count=len(witness.events))
        states.append(state);records.append(labelled)
        if len(states)%512==0:print('SOURCE_RECONSTRUCTION',case,len(states),tick,flush=True)
    while cursor<len(rows):
        witness.update(rows[cursor,:2]);cursor+=1
    geometry=witness.snapshot(endpoint_tolerance=tolerance)
    if not geometry['geometry_complete']:raise ValueError(f'Source full traversal does not pass: {case}')
    write_json(out/f'{case}_source_gate_events.json',dict(source=source,geometry=geometry,events=witness.events))
    pack_records(out/f'{case}_source_states.npz',records)
    source.update(reconstruction_wall_s=time.perf_counter()-started,reconstruction_cpu_s=time.process_time()-cpu,
        full_command_intervals=len(rows),geometry=geometry,
        original_observation_max_difference=float(max(np.max(abs(r['observations']-r['archived_observations'])) for r in records)),
        fixed_labels_clipped=sum(r['fixed_steering_clipped'] for r in records),
        maximum_unclipped_direction_error_rad=max(abs(r['fixed_direction_error_rad']) for r in records if not r['fixed_steering_clipped']))
    return env,states,records,source


class BranchArchive:
    def __init__(self,out):
        self.out=out;self.batch=0;self.records=[];self.rows=[];self.positions=[];self.total=0
        self.log=(out/'branch_records.jsonl').open('x',encoding='utf-8')

    def append(self,record,rows,positions):
        record['trace_batch']=self.batch
        record['row_offset']=sum(len(x) for x in self.rows)
        record['position_offset']=sum(len(x) for x in self.positions)
        record['row_count']=len(rows);record['position_count']=len(positions)
        self.records.append(record);self.rows.append(rows);self.positions.append(positions)
        self.total+=1
        if len(self.records)>=128:self.flush()

    def flush(self):
        if not self.records:return
        path=self.out/f'branch_trace_{self.batch:04d}.npz'
        np.savez_compressed(path,rows=np.vstack(self.rows),positions=np.vstack(self.positions),
            branch_ids=np.asarray([r['branch_id'] for r in self.records]),
            row_offsets=np.asarray([r['row_offset'] for r in self.records]),
            row_counts=np.asarray([r['row_count'] for r in self.records]),
            position_offsets=np.asarray([r['position_offset'] for r in self.records]),
            position_counts=np.asarray([r['position_count'] for r in self.records]))
        digest=sha(path)
        for record in self.records:
            record['trace_file']=path.name;record['trace_sha256']=digest
            self.log.write(json.dumps(record,allow_nan=False)+'\n')
        self.log.flush();self.batch+=1;self.records=[];self.rows=[];self.positions=[]

    def close(self):
        self.flush();self.log.close()


def branch(env,state,action):
    restore_source(env,state)
    positions=[env.position.copy()]
    tracker=env.tracker
    def record(position):
        positions.append(np.asarray(position).copy())
        return type(tracker).update(tracker,position)
    tracker.update=record
    rows=np.empty((0,13));error=None;info={}
    try:
        _,_,_,info=env.step(action)
        rows=info['rows']
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        frame=exc.__traceback__
        while frame is not None:
            if frame.tb_frame.f_code.co_name=='step' and frame.tb_frame.f_code.co_filename.endswith('joint_runtime.py'):
                found=frame.tb_frame.f_locals.get('rows',[])
                if len(found):rows=np.asarray(found)
            frame=frame.tb_next
    positions=np.asarray(positions)
    assert len(positions)-1==len(rows), 'Speculative or lost branch samples'
    assert env.ticks-state['ticks']==len(rows)
    if len(rows):
        velocity=np.diff(positions,axis=0)/DT
        acceleration=np.diff(np.vstack([state['velocity'],velocity]),axis=0)/DT
        jerk=np.diff(np.vstack([state['acceleration'],acceleration]),axis=0)/DT
        peaks=[float(np.linalg.norm(x,axis=1).max()) for x in (velocity,acceleration,jerk)]
    else:peaks=[0.,0.,0.]
    return rows,positions,dict(error=error,executor_reason=env.reason,done=bool(env.done),success=bool(env.success),
        finite_command_vector_peaks=peaks,kinematic_pass=bool(np.all(np.asarray(peaks)/[100.,2000.,62500.]<=1.00002)),
        corridor_inside=bool(env.tracker.inside),frontier_after=env.tracker.frontier,
        eligible_nonterminal_state=bool(error is None and not env.done and env.tracker.inside),
        terminal_tail=bool(info.get('terminal_tail',False)),gates_crossed=env.tracker.frontier-state['frontier'])


def main(args):
    torch.set_num_threads(1)
    if sha(MODEL)!=MODEL_HASH:raise ValueError('Historical teacher hash mismatch')
    installer_module=importlib.import_module(args.runtime_module)
    installer=installer_module.install
    setup=installer(dt=DT,tolerance=1.25)
    out=ROOT/'results/physical_teacher_v5'/args.run
    out.mkdir(parents=True,exist_ok=False)
    started,cpu=time.perf_counter(),time.process_time()
    source_names=['physical_teacher_v5.py','frozen_policy.py','runtime_bootstrap.py','runtime_configuration.py',
        'corridor_v3.py','execution_settings.py','runtime_no_target_override_v4b.py','replay_no_target_override_v4b.py',
        'terminal_completion_v4d.py','runtime_resolution_fixed_v4e.py',args.runtime_module+'.py']
    sources={name:sha(ROOT/'src'/name) for name in source_names}
    registration=dict(created_utc=utc(),version='physical_teacher_v5_1',scope='Development dataset preparation only; no training',
        runtime_module=args.runtime_module,executor_setup=setup,dt_s=DT,model_sha256=MODEL_HASH,
        source_sha256=sources,runtime_snapshot_manifest_sha256=sha(ROOT/'runtime/source_manifest.json'),
        cases=[source_case(case)[-1] for case in ('original','dense')],base_states=4833,
        selected_original=4096,selected_dense=737,source_selection='Uniform whole-program decision indexing; no failure-local weighting',
        source_prefix_state='Every original command interval is replayed through current unchanged GateTracker; source event history stored once and indexed by each base state',
        observation_rule='Recompute current encoder from physical state; preserve archived values separately; do not assert observation equality across runtime versions',
        fixed_preview_mm=3.,fixed_conversion='wrap(beta(teacher preview)+teacher steering-beta(3mm)); record clipping and direction error; same physical state for both arms',
        branch_decisions=9666,branch_action='Final teacher raw normalized action plus registered antithetic Gaussian vector; common runtime decoder performs bounds',
        noise_seed=NOISE_SEED,noise_std=NOISE_STD.tolist(),noise_truncation_standard_deviations=4.,
        branch_duration_s=.02,terminal_tail='Retain the actual common terminal tail if initiated',
        branch_wall_limit_s=900.,budget_scope='One collection loop including pilot branch subset; source reconstruction and labeling preparation costs recorded separately',
        pilot_global_base_indices=[0,2048,4095,4096,4464,4832],
        branch_order='Pilot indices first, then ascending remaining indices, positive then negative sign; each planned branch at most once',
        failures='All failure, terminal, exception and incomplete-budget records retained; only nonterminal in-corridor states are marked eligible for future action supervision',
        no_training=True,no_final_test_data=True,no_source_or_model_change=True)
    write_json(out/'registration.json',registration)
    teacher=FrozenPolicy(MODEL)
    environments={};all_states=[];base_records=[];source_receipts=[];branch_labels=[];counts={}
    for case in ('original','dense'):
        env,states,records,receipt=reconstruct_case(case,out,installer,teacher)
        environments[case]=env;all_states.extend(states);base_records.extend(records);source_receipts.append(receipt)
    assert len(base_records)==4833
    pack_records(out/'base_teacher_records.npz',base_records)
    pilot_ids=registration['pilot_global_base_indices']
    validation=[]
    import compact_control as compact
    import runtime_configuration as runtime_configuration
    for index in pilot_ids:
        record=base_records[index];env=environments[record['case']]
        restore_source(env,all_states[index]);again=labels(env,teacher)
        error=float(np.max(abs(again['observations']-record['observations'])))
        assert error==0.
        current_transitions=compact.active_gate_transitions
        try:
            compact.active_gate_transitions=runtime_configuration._ORIGINAL
            legacy_obs=env.observation().copy()
        finally:
            compact.active_gate_transitions=current_transitions
        legacy_error=float(np.max(abs(legacy_obs-record['archived_observations'])))
        assert legacy_error<2e-6, 'Reconstructed state does not reproduce archived historical encoder'
        assert all(np.isfinite(again[k]).all() for k in ('targets_three','targets_two','targets_two_raw'))
        validation.append(dict(base_id=index,case=record['case'],source_decision=record['source_decision'],
            source_tick=record['source_tick'],restored_observation_error=error,
            legacy_encoder_vs_archived_observation_max_error=legacy_error,
            fixed_clipped=again['fixed_steering_clipped'],direction_error_rad=again['fixed_direction_error_rad']))
    write_json(out/'small_sample_validation.json',dict(status='passed',states=validation,
        all_base_labels_unclipped_direction_pass=all(r['fixed_steering_clipped'] or abs(r['fixed_direction_error_rad'])<1e-10 for r in base_records),
        source_receipts=source_receipts,source_preparation_wall_s=time.perf_counter()-started))
    print('SMALL_SAMPLE_STATE_AND_DIRECTION_VALIDATION_PASSED',flush=True)
    if args.prepare_only:
        write_json(out/'receipt.json',dict(status='prepared_only',base_states=len(base_records),no_training=True,
            total_wall_s=time.perf_counter()-started,total_cpu_s=time.process_time()-cpu))
        return
    noise=np.clip(np.random.default_rng(NOISE_SEED).standard_normal((4833,3)),-4,4)*NOISE_STD
    np.save(out/'registered_positive_noise.npy',noise)
    order=pilot_ids+[i for i in range(4833) if i not in set(pilot_ids)]
    archive=BranchArchive(out)
    collection_start,collection_cpu=time.perf_counter(),time.process_time()
    error=None;completed=0;physics_ticks=0;pilot_checks=[]
    try:
        for base_index in order:
            record=base_records[base_index];state=all_states[base_index];env=environments[record['case']]
            for sign in (1,-1):
                if time.perf_counter()-collection_start>900.:
                    raise TimeoutError('Registered 900-second branch collection budget')
                action=record['targets_three_raw']+sign*noise[base_index]
                branch_start=time.perf_counter()
                rows,positions,outcome=branch(env,state,action)
                outcome.update(branch_id=2*base_index+(0 if sign==1 else 1),execution_index=completed,
                    base_id=base_index,case=record['case'],source_decision=record['source_decision'],
                    source_tick=record['source_tick'],sign=sign,action_raw=action.tolist(),
                    perturbation=(sign*noise[base_index]).tolist(),wall_s=time.perf_counter()-branch_start)
                if len(rows):
                    labelled=labels(env,teacher)
                    labelled.update(capture_state(env))
                    labelled.update(branch_id=outcome['branch_id'],base_id=base_index,case=record['case'],
                        source_decision=record['source_decision'],sign=sign,
                        eligible_nonterminal_state=outcome['eligible_nonterminal_state'],terminal=outcome['done'])
                    branch_labels.append(labelled)
                outcome['label_record_index']=len(branch_labels)-1 if len(rows) else -1
                archive.append(outcome,rows,positions)
                status='exception' if outcome['error'] else ('terminal_success' if outcome['success'] else ('terminal_failure' if outcome['done'] else 'active'))
                counts[status]=counts.get(status,0)+1
                completed+=1;physics_ticks+=len(rows)
                if completed<=12:
                    assert outcome['kinematic_pass'], 'Pilot physical command audit failed'
                    pilot_checks.append({k:outcome[k] for k in ('branch_id','case','error','done','success','kinematic_pass','corridor_inside','row_count')})
                    if completed==12:
                        write_json(out/'pilot_branch_validation.json',dict(status='passed',branches=pilot_checks,
                            these_branches_are_included_in_total_budget=True))
                        print('PILOT_BRANCH_VALIDATION_PASSED',flush=True)
                if completed%128==0:
                    print('PHYSICAL_TEACHER_BRANCHES',completed,physics_ticks,counts,round(time.perf_counter()-collection_start,3),flush=True)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:
        archive.close()
        pack_records(out/'branch_teacher_records.npz',branch_labels)
    unchanged=sha(MODEL)==MODEL_HASH and all(sha(ROOT/'src'/name)==digest for name,digest in sources.items())
    result=dict(status='completed' if error is None and completed==9666 else 'incomplete',error=error,
        base_states=len(base_records),planned_branches=9666,completed_branches=completed,
        branch_label_records=len(branch_labels),physics_ticks=physics_ticks,outcome_counts=counts,
        eligible_branch_states=sum(r['eligible_nonterminal_state'] for r in branch_labels),
        base_fixed_clipped=sum(r['fixed_steering_clipped'] for r in base_records),
        branch_fixed_clipped=sum(r['fixed_steering_clipped'] for r in branch_labels),
        maximum_unclipped_direction_error_rad=max(abs(r['fixed_direction_error_rad']) for r in base_records+branch_labels if not r['fixed_steering_clipped']),
        source_receipts=source_receipts,preparation_wall_s=collection_start-started,
        collection_wall_s=time.perf_counter()-collection_start,collection_cpu_s=time.process_time()-collection_cpu,
        total_wall_s=time.perf_counter()-started,total_cpu_s=time.process_time()-cpu,
        model_and_registered_sources_unchanged=unchanged,no_training=True,no_new_final_test_data=True,
        registration_sha256=sha(out/'registration.json'),
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='receipt.json'},
        interpretation='Witnessed development states and paired instantaneous direction labels; no new policy reliability, complete-path success or independent generalization claim')
    write_json(out/'receipt.json',result)
    print(json.dumps(result),flush=True)
    if not unchanged:raise ValueError('Registered dependency changed during collection')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-module',default='runtime_physical_time_v5')
    parser.add_argument('--run',default='collection_001')
    parser.add_argument('--prepare-only',action='store_true')
    main(parser.parse_args())
