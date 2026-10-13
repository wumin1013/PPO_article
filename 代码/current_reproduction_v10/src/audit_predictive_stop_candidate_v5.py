"""Independent audit of the fixed v5_2 stopping-admissibility experiment."""
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
import audit_predictive_candidate_v5 as base_audit

OUT=ROOT/'results/predictive_stop_candidate_v5_2'


def read(path):return json.loads(path.read_text('utf-8'))
def lines(path):return [json.loads(x) for x in path.read_text('utf-8').splitlines()]


def stop_audit(row):
    directory=OUT/row['id']/'predictions';stops=directory/'stopping_extensions'
    branches=lines(directory/'branch_records.jsonl');stop_records=lines(stops/'branch_records.jsonl')
    by_id={r['branch_id']:r for r in stop_records};assert len(by_id)==len(stop_records)
    states=[]
    with (directory/'decision_states.pkl').open('rb') as stream:
        while True:
            try:states.append(pickle.load(stream))
            except EOFError:break
    install(dt=.001,tolerance=row['evaluation_tolerance_mm'])
    import joint_runtime as joint
    from geometry_spec import Corridor,GateTracker
    reference=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
    corridor=Corridor(reference,epsilon=row['evaluation_tolerance_mm'],closed=np.linalg.norm(reference[0]-reference[-1])<1e-9)
    files={};ticks=0;feasible=0;all_stop_wall=0.;all_stop_cpu=0.
    def arrays(base,record):
        path=base/record['trace_file']
        if str(path) not in files:
            assert sha(path)==record['trace_sha256']
            with np.load(path) as data:files[str(path)]={k:data[k].copy() for k in data.files}
        data=files[str(path)]
        rr=data['rows'][record['row_offset']:record['row_offset']+record['row_count']]
        pp=data['positions'][record['position_offset']:record['position_offset']+record['position_count']]
        return rr,pp
    for b in branches:
        if not b['six_decision_horizon_feasible']:
            assert not b['feasible'] and b['stopping_check']=='not_attempted' and b['branch_id'] not in by_id
            continue
        if b['terminal_absorbing_rest']:
            assert b['feasible'] and b['stopping_check']=='complete_terminal' and b['branch_id'] not in by_id
            continue
        assert b['stopping_check']=='assessed' and b['branch_id'] in by_id
        stop=by_id[b['branch_id']]
        assert stop['decision_id']==b['decision_id'] and b['feasible']==stop['feasible']
        assert b['stopping_extension']['feasible']==stop['feasible']
        initial=states[b['decision_id']]
        horizon,hp=arrays(directory,b);rows,pp=arrays(stops,stop)
        assert len(pp)==len(rows)+1 and np.array_equal(pp[1:],rows[:,:2])
        assert np.array_equal(hp[-1],pp[0]) and np.array_equal(horizon[-1,2:4],stop['initial_velocity_mm_s'])
        assert np.array_equal(horizon[-1,4:6],stop['initial_acceleration_mm_s2'])
        tracker=GateTracker(corridor,reference[0])
        for k,v in initial['tracker'].items():setattr(tracker,k,copy.deepcopy(v))
        for point in hp[1:]:tracker.update(point)
        assert tracker.frontier==stop['initial_frontier']
        first_bad=None
        for index,point in enumerate(pp[1:],1):
            tracker.update(point)
            if first_bad is None and not tracker.inside:first_bad=index
        assert first_bad==stop['first_outside_tick'] and tracker.inside==stop['every_interval_inside']
        assert tracker.frontier==stop['final_frontier']
        gate=corridor.gates[tracker.frontier] if tracker.frontier<len(corridor.gates) else None
        signed=None if gate is None else float((pp[-1]-gate.point)@gate.normal)
        forward=bool(signed is None or signed<=1e-9)
        assert signed==stop['next_unvisited_gate_signed_mm'] and forward==stop['next_unvisited_gate_not_behind_stop']
        v0=np.asarray(stop['initial_velocity_mm_s']);a0=np.asarray(stop['initial_acceleration_mm_s2'])
        v=np.diff(pp,axis=0)/.001
        a=np.diff(np.vstack([v0,v]),axis=0)/.001
        j=np.diff(np.vstack([a0,a]),axis=0)/.001
        peaks=[float(np.linalg.norm(z,axis=1).max(initial=0.)) for z in (v,a,j)]
        assert peaks==stop['vector_peaks']
        vector=bool(np.all(np.asarray(peaks)/[100.,2000.,62500.]<=1.00002))
        rest=(np.linalg.norm(v,axis=1)<=.001)&(np.linalg.norm(a,axis=1)<=.01)
        nonrest=np.flatnonzero(~rest);rest_count=len(rest)-(int(nonrest[-1])+1 if len(nonrest) else 0)
        assert rest_count*.001==stop['terminal_rest_s']
        expected=bool(stop['error'] is None and len(rows)>0 and tracker.inside and forward and vector and rest_count*.001>=.02-1e-10)
        assert expected==stop['feasible']
        # Independently regenerate the original fixed stopping family.
        reproduced_error=None
        try:
            plan=joint.terminal.plan_planar_terminal_brake(pp[0],v0,a0,dt=.001,limits=joint.terminal.LIMITS)
        except Exception as exc:
            reproduced_error=type(exc).__name__+': '+str(exc)
        assert reproduced_error==stop['error']
        if reproduced_error is None:
            cv=plan['columns'];command=np.c_[cv['velocity_x_mm_s'],cv['velocity_y_mm_s']]
            assert np.array_equal(command[1:],rows[:,2:4])
            assert plan['braking_time_s']==stop['braking_time_s'] and plan['total_tail_time_s']==stop['total_tail_time_s']
            p=pp[0].copy();rebuilt=[p.copy()]
            for velocity in command[1:]:p=p+.001*velocity;rebuilt.append(p.copy())
            assert np.array_equal(np.asarray(rebuilt),pp)
        else:
            assert len(rows)==0 and len(pp)==1 and not stop['feasible']
        ticks+=len(rows);feasible+=int(expected)
        all_stop_wall+=stop['assessment_wall_s'];all_stop_cpu+=stop['assessment_cpu_s']
    summary=read(OUT/row['id']/'prediction_summary.json')
    assert summary['stopping_counts']==dict(assessments=len(stop_records),feasible=feasible,
        failed=len(stop_records)-feasible,command_intervals=ticks)
    assert abs(summary['stopping_cost']['wall_s']-all_stop_wall)<1e-10
    assert abs(summary['stopping_cost']['cpu_s']-all_stop_cpu)<1e-10
    return dict(case=row['id'],status='passed',stopping_assessments=len(stop_records),
        feasible_stops=feasible,failed_stops=len(stop_records)-feasible,stopping_command_intervals=ticks,
        all_saved_stops_match_original_cubic_family=True,all_intervals_vectors_gates_and_rest_reaudited=True)


def main():
    torch.set_num_threads(1);started=time.perf_counter();reg=read(OUT/'registration.json')
    assert sha(MODEL)==reg['model_sha256']
    assert all(sha(ROOT/'src'/k)==v for k,v in reg['source_sha256'].items())
    base_audit.OUT=OUT
    teacher=FrozenPolicy(MODEL)
    base=[base_audit.audit_case(row,reg,teacher) for row in reg['cases']]
    stopping=[stop_audit(row) for row in reg['cases']]
    report=dict(status='passed',created_utc=utc(),registration_sha256=sha(OUT/'registration.json'),
        audit_source_sha256=sha(__file__),unchanged_registered_sources_and_model=True,
        feedback_and_selection_audit=base,stopping_extension_audit=stopping,
        branches=sum(r['branches'] for r in base),prediction_intervals=sum(r['prediction_intervals'] for r in base),
        stopping_assessments=sum(r['stopping_assessments'] for r in stopping),
        stopping_intervals=sum(r['stopping_command_intervals'] for r in stopping),
        total_wall_s=time.perf_counter()-started)
    write_json(OUT/'independent_audit.json',report)
    summary=read(OUT/'summary.json');old=read(ROOT/'results/predictive_action_candidate_v5/summary.json')
    text=['# One fixed stopping-extension candidate: development results','',
        'The sole change from v5_1 adds an endpoint stop-feasibility predicate to the unchanged six-decision prediction and 15 first-action candidates. The original model, true executor, tolerance, gates, limits, completion rule and no-solution raw fallback are unchanged.','',
        '| Program | v5_1 complete/time | v5_2 complete/time | v5_2 failed prefix | Corrections | No solution |',
        '|---|---|---|---:|---:|---:|']
    for first,second in zip(old['results'],summary['results']):
        a=f"yes / {first['complete_time_s']:.3f} s" if first['success'] else 'no'
        b=f"yes / {second['complete_time_s']:.3f} s" if second['success'] else 'no'
        prefix='--' if second['success'] else f"{second['elapsed_prefix_s']:.3f} s"
        counts=second['prediction']['counts']
        text.append(f"| {second['path']} | {a} | {b} | {prefix} | {counts['predictive_correction']} | {counts['no_feasible_candidate']} |")
    text.extend(['',
        'A six-step horizon can be collision-free while ending outside the tested cubic stopping set. On the retained fold v5_1 trajectory this first occurred for the prediction accepted at 0.040 s. The first actual no-solution decision was at 0.220 s, while a common cubic stop remained feasible through 0.280 s. At 0.300 s the tested brake left the corridor near ordered gate 4, vertex [24,4]. Neither an empty finite candidate set nor failure of this one cubic family establishes physical inevitability.',
        '',f"Audited {report['branches']} feedback predictions ({report['prediction_intervals']} intervals) and {report['stopping_assessments']} stopping extensions ({report['stopping_intervals']} intervals), retaining all outcomes and costs.",
        'This extra predicate is not a recursive safety controller: the specified raw fallback can enter states without feasible candidates. It is one retained development experiment, not a replacement of the historical main model, v5_1, Figure 8, or five-seed evidence. No result-dependent grid expansion or repeated tuning was performed.',''])
    (ROOT/'audit/predictive_stop_candidate_v5_2.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('feedback_and_selection_audit','stopping_extension_audit')}),flush=True)


if __name__=='__main__':main()
