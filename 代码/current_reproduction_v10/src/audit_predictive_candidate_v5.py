"""Independent saved-trace, branch selection and actual recorder audit."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import copy
import json
import pickle
import time
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from frozen_policy import FrozenPolicy
from replay_strong_checkpoint_v4 import MODEL
from runtime_physical_time_v5 import install
from metrics import metrics
from predictive_action_candidate_v5 import first_action,change_cost,OFFSETS,FACTORS

OUT=ROOT/'results/predictive_action_candidate_v5'


def read(path):
    return json.loads(path.read_text('utf-8'))


def jsonl(path):
    return [json.loads(line) for line in path.read_text('utf-8').splitlines()]


def audit_case(row,reg,teacher):
    case=row['id'];base=OUT/case;pred=base/'predictions'
    reference=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
    assert sha(ROOT/row['reference'])==row['reference_sha256']
    install(dt=.001,tolerance=row['evaluation_tolerance_mm'])
    from geometry_spec import Corridor,GateTracker
    corridor=Corridor(reference,epsilon=row['evaluation_tolerance_mm'],closed=np.linalg.norm(reference[0]-reference[-1])<1e-9)
    actual=read(base/'actual.json');summary=read(base/'prediction_summary.json')
    with np.load(base/'actual.npz') as data:
        actual_data={key:data[key].copy() for key in data.files}
    assert sha(base/'actual.npz')==actual['trace_sha256']
    rows=actual_data['rows'];positions=actual_data['positions']
    assert len(positions)==len(rows)+1 and np.array_equal(rows[:,:2],positions[1:])
    independent=metrics(reference,positions,.001,row['evaluation_tolerance_mm'],
        declared_success=actual['success'],reason=actual['executor_reason'])
    assert independent['success']==actual['success']
    assert independent['complete_time_s']==actual['complete_time_s']
    assert independent['vector_peaks']==actual['vector_peaks']
    decisions=jsonl(pred/'decisions.jsonl');branches=jsonl(pred/'branch_records.jsonl')
    states=[]
    with (pred/'decision_states.pkl').open('rb') as stream:
        while True:
            try:states.append(pickle.load(stream))
            except EOFError:break
    assert len(decisions)==len(states)==actual['actor_calls']==summary['decisions']
    assert len(branches)==summary['branches']
    arrays={};branch_by_id={};branch_rows={};total_ticks=0;max_duration=0.;source_counts={}
    for expected_id,b in enumerate(branches):
        assert b['branch_id']==expected_id
        state=states[b['decision_id']]
        file=pred/b['trace_file']
        if b['trace_file'] not in arrays:
            assert sha(file)==b['trace_sha256']
            with np.load(file) as data:
                arrays[b['trace_file']]={key:data[key].copy() for key in data.files}
        trace=arrays[b['trace_file']]
        one=trace['rows'][b['row_offset']:b['row_offset']+b['row_count']]
        pp=trace['positions'][b['position_offset']:b['position_offset']+b['position_count']]
        assert len(pp)==len(one)+1 and np.array_equal(pp[1:],one[:,:2])
        assert np.array_equal(pp[0],state['core']['position'])
        v=np.diff(pp,axis=0)/.001
        a=np.diff(np.vstack([state['core']['velocity'],v]),axis=0)/.001
        j=np.diff(np.vstack([state['core']['acceleration'],a]),axis=0)/.001
        peaks=[float(np.linalg.norm(x,axis=1).max(initial=0.)) for x in (v,a,j)]
        assert peaks==b['vector_peaks']
        vector=bool(np.all(np.asarray(peaks)/[100,2000,62500]<=1.00002))
        assert vector==b['vector_pass']
        tracker=GateTracker(corridor,reference[0])
        assert 'update' not in state['tracker'] and 'max_ticks' in state['core'] and 'fixed_d' in state['core']
        for key,value in state['tracker'].items():setattr(tracker,key,copy.deepcopy(value))
        for point in pp[1:]:tracker.update(point)
        assert tracker.inside==b['every_interval_inside'] and tracker.frontier==b['final_gate_frontier']
        assert b['accounted'] and b['independent_tracker_matches']
        raw=np.asarray(decisions[b['decision_id']]['raw_model_action'])
        if b['kind']=='original_model_prediction':
            expected=raw
        else:
            assert b['steering_offset_rad']==OFFSETS[b['candidate_order']//3]
            assert b['feed_factor']==FACTORS[b['candidate_order']%3]
            expected=first_action(raw,b['steering_offset_rad'],b['feed_factor'])
        assert np.array_equal(expected,np.asarray(b['first_action']))
        assert change_cost(raw,expected)==b['action_change_cost']
        if b['predicted_actions']:
            assert np.array_equal(expected,np.asarray(b['predicted_actions'][0]))
        for obs,action in zip(b['predicted_observations'][1:],b['predicted_actions'][1:]):
            assert teacher.mean(np.asarray(obs)).numpy().tolist()==action
        duration=len(one)*.001
        assert duration==b['elapsed_prediction_s']
        if b['feasible']:
            assert b['error'] is None and vector and tracker.inside
            if b['terminal_absorbing_rest']:
                assert b['success'] and b['done'] and tracker.snapshot(endpoint_tolerance=corridor.epsilon)['geometry_complete']
                rest=(np.linalg.norm(v,axis=1)<=.001)&(np.linalg.norm(a,axis=1)<=.01)
                nonrest=np.flatnonzero(~rest)
                final_rest=len(rest)-(int(nonrest[-1])+1 if len(nonrest) else 0)
                # A terminal branch can start while already resting; actual
                # saved initial rest counter supplies the remaining prefix.
                if not len(nonrest):final_rest+=state['core']['rest_ticks']
                assert final_rest*.001>=.02-1e-10
            else:
                assert duration==.12 and len(b['predicted_actions'])==6 and not b['done']
        segment=min(tracker.frontier,len(corridor.lengths)-1)
        distance=float((pp[-1]-corridor.starts[segment])@corridor.tangents[segment])
        progress=float(corridor.arcs[segment]+np.clip(distance,0,corridor.lengths[segment]))
        assert progress==b['final_ordered_progress_mm']
        total_ticks+=len(one);max_duration=max(max_duration,duration)
        branch_by_id[b['branch_id']]=b;branch_rows[b['branch_id']]=one
    submitted=0
    for index,(d,state) in enumerate(zip(decisions,states)):
        assert d['decision_id']==index and d['real_state_and_recorder_unchanged']
        tick=d['actual_start_tick']
        assert tick==state['core']['ticks'] and np.array_equal(positions[tick],state['core']['position'])
        if tick:
            assert np.array_equal(rows[tick-1,2:4],state['core']['velocity'])
            assert np.array_equal(rows[tick-1,4:6],state['core']['acceleration'])
        assert d['max_ticks']==state['core']['max_ticks'] and d['fixed_d']==state['core']['fixed_d']
        group=[branch_by_id[b] for b in d['branch_ids']]
        first=group[0]
        assert first['kind']=='original_model_prediction'
        raw=teacher.mean(actual_data['observations'][index]).numpy().tolist()
        assert raw==d['raw_model_action']
        if d['source']=='raw_model':
            assert len(group)==1 and first['feasible'] and d['submitted_action']==raw
            chosen=first
        elif d['source']=='predictive_correction':
            assert len(group)==16 and not first['feasible']
            valid=[b for b in group[1:] if b['feasible']]
            chosen=min(valid,key=lambda b:(-b['final_ordered_progress_mm'],b['action_change_cost'],b['candidate_order']))
            assert d['submitted_action']==chosen['first_action']
        elif d['source']=='no_feasible_candidate':
            assert len(group)==16 and not any(b['feasible'] for b in group) and d['submitted_action']==raw
            chosen=first
        else:
            assert d['source']=='aborted' and d['error'] and d['submitted_action'] is None
            continue
        assert chosen['branch_id']==d['selected_branch_id']
        assert np.array_equal(actual_data['actions'][submitted],d['submitted_action'])
        stop=states[index+1]['core']['ticks'] if index+1<len(states) else len(rows)
        actual_segment=rows[tick:stop]
        assert np.array_equal(actual_segment,branch_rows[chosen['branch_id']][:len(actual_segment)])
        submitted+=1;source_counts[d['source']]=source_counts.get(d['source'],0)+1
    assert submitted==len(actual_data['actions'])==actual['submitted_actions']
    assert total_ticks==summary['counts']['branch_ticks']
    assert all(source_counts.get(k,0)==summary['counts'][k] for k in ('raw_model','predictive_correction','no_feasible_candidate'))
    assert len(decisions)==summary['counts']['isolation_checks']
    return dict(case=case,status='passed',actual_success=actual['success'],
        complete_time_s=actual['complete_time_s'],elapsed_prefix_s=actual['elapsed_prefix_s'],
        actual_intervals=len(rows),branches=len(branches),prediction_intervals=total_ticks,
        maximum_prediction_duration_s=max_duration,model_calls=sum(b['model_calls'] for b in branches),
        failed_branches=sum(not b['feasible'] for b in branches),sources=source_counts,
        all_actual_actions_equal_selected_branch_prefix=True,
        all_later_branch_actions_from_original_model=True,
        every_branch_interval_independently_witnessed=True,
        all_decisions_real_state_recorder_unchanged=True,
        actual_trace_sha256=actual['trace_sha256'],
        file_hashes={p.relative_to(base).as_posix():sha(p) for p in base.rglob('*') if p.is_file()})


def main():
    torch.set_num_threads(1)
    started=time.perf_counter();reg=read(OUT/'registration.json')
    assert all(sha(ROOT/'src'/k)==v for k,v in reg['source_sha256'].items())
    assert sha(MODEL)==reg['model_sha256']
    teacher=FrozenPolicy(MODEL)
    records=[audit_case(row,reg,teacher) for row in reg['cases']]
    report=dict(status='passed',created_utc=utc(),scope='Independent saved evidence audit of development-only predictive candidate',
        registration_sha256=sha(OUT/'registration.json'),audit_source_sha256=sha(__file__),
        unchanged_registered_sources_and_model=True,cases=records,
        branches=sum(r['branches'] for r in records),prediction_intervals=sum(r['prediction_intervals'] for r in records),
        total_wall_s=time.perf_counter()-started)
    write_json(OUT/'independent_audit.json',report)
    text=['# Predictive action candidate v5: retained development diagnosis','',
        'This is a separately versioned execution-policy candidate around unchanged c16d45 weights. It does not replace the historical main method or Figure 8, and it is not five-seed or independent final-test evidence.','',
        '| Program | Complete | Completion time (s) | Failed-prefix time (s) | Corrections | No feasible candidate | Branches |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        done=f"{r['complete_time_s']:.3f}" if r['actual_success'] else '--'
        prefix='--' if r['actual_success'] else f"{r['elapsed_prefix_s']:.3f}"
        text.append(f"| {r['case']} | {r['actual_success']} | {done} | {prefix} | {r['sources'].get('predictive_correction',0)} | {r['sources'].get('no_feasible_candidate',0)} | {r['branches']} |")
    text.extend(['',f"All {report['branches']} saved prediction branches and {report['prediction_intervals']} command intervals passed archive, selection, independent geometry and finite-difference re-audit. This audit validates the recorded branch classifications; failed branches remain failures.",
        'Every submitted action produced an actual first-decision trajectory exactly matching its selected speculative prefix. Every subsequent speculative action matches the unchanged model on that branch observation. All source-state and recorder isolation checks passed.',
        'The registered 120 ms / six-decision horizon retains a complete common terminal tail when triggered. Successful early completion implies absorbing rest. The full-program geometric, vector and final-rest acceptance contract is unchanged.',
        'Fold retained six explicit no-solution decisions and ultimately violated the corridor at 0.330 s; it has no completion time. The two successful development programs do not establish general reliability. No expansion, original-program run, shared runtime edit, main-result replacement or five-seed claim is made.',
        '',r'Reproduce in this work directory with D:\Anaconda\envs\PPO\python.exe src\audit_predictive_candidate_v5.py',''])
    (ROOT/'audit/predictive_action_candidate_v5.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
