"""Retain physical stopping diagnostics for every observed fold decision."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
import pickle
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from runtime_physical_time_v5 import install
from predictive_action_candidate_v5 import new_environment,restore,prediction
from braking_reachability_v5 import stopping_extension
from frozen_policy import FrozenPolicy
from replay_strong_checkpoint_v4 import MODEL


def main():
    torch.set_num_threads(1)
    out=ROOT/'results/predictive_action_candidate_v5/fold_stopping_diagnosis'
    out.mkdir(parents=True,exist_ok=False)
    pred=out.parent/'final_dense_fold/predictions'
    decisions=[json.loads(s) for s in (pred/'decisions.jsonl').read_text('utf-8').splitlines()]
    branches=[json.loads(s) for s in (pred/'branch_records.jsonl').read_text('utf-8').splitlines()]
    states=[]
    with (pred/'decision_states.pkl').open('rb') as stream:
        while True:
            try:states.append(pickle.load(stream))
            except EOFError:break
    reference=np.loadtxt(out.parent/'inputs/final_dense_fold.csv',delimiter=',',skiprows=1)
    install(dt=.001,tolerance=1.25)
    env=new_environment(reference);teacher=FrozenPolicy(MODEL);records=[]
    for decision,state in zip(decisions,states):
        index=decision['decision_id'];restore(env,state)
        gate=env.corridor.gates[env.tracker.frontier]
        rows,positions,stop=stopping_extension(env)
        file=out/f'actual_state_{index:03d}_stop.npz'
        np.savez_compressed(file,rows=rows,positions=positions)
        record=dict(decision_id=index,actual_time_s=state['core']['ticks']*.001,
            source=decision['source'],actual_speed_mm_s=float(np.linalg.norm(env.velocity)),
            actual_position_mm=env.position.tolist(),frontier=env.tracker.frontier,
            next_gate_vertex=gate.vertex_index,next_gate_point_mm=gate.point.tolist(),
            actual_stop=stop,actual_stop_trace_sha256=sha(file))
        selected=branches[decision['selected_branch_id']]
        record['selected_horizon_feasible']=selected['feasible']
        if selected['feasible']:
            first=np.asarray(selected['first_action'])
            horizon_rows,horizon_points,outcome=prediction(env,state,first,teacher)
            assert outcome['feasible'] and outcome['elapsed_prediction_s']==.12
            old=np.load(pred/selected['trace_file'])
            expected=old['rows'][selected['row_offset']:selected['row_offset']+selected['row_count']]
            assert np.array_equal(expected,horizon_rows)
            rows,positions,stop=stopping_extension(env)
            file=out/f'selected_horizon_{index:03d}_stop.npz'
            np.savez_compressed(file,rows=rows,positions=positions,
                horizon_rows=horizon_rows,horizon_positions=horizon_points)
            record.update(horizon_end_time_s=env.ticks*.001,horizon_end_speed_mm_s=float(np.linalg.norm(env.velocity)),
                horizon_end_stop=stop,horizon_stop_trace_sha256=sha(file))
        records.append(record)
    first_horizon_bad=next(r for r in records if r.get('horizon_end_stop',{}).get('feasible') is False)
    first_grid_bad=next(r for r in records if r['source']=='no_feasible_candidate')
    first_actual_bad=next(r for r in records if not r['actual_stop']['feasible'])
    report=dict(created_utc=utc(),scope='Development-only physical diagnosis of fixed retained v5_1 fold',
        first_accepted_horizon_without_safe_cubic_stop=first_horizon_bad['actual_time_s'],
        first_no_feasible_registered_grid=first_grid_bad['actual_time_s'],
        first_actual_state_without_safe_cubic_stop=first_actual_bad['actual_time_s'],
        records=records,source_sha256={name:sha(ROOT/'src'/name) for name in
            ('diagnose_predictive_fold_v5.py','braking_reachability_v5.py','predictive_action_candidate_v5.py')},
        evidence_sha256={name:sha(pred/name) for name in ('decision_states.pkl','decisions.jsonl','branch_records.jsonl')},
        caution='No registered 15-action solution is not physical inevitability. Failed stopping extension establishes only failure of the common fixed cubic family.',
        proposed_one_change='Retain six feedback decisions and all 15 first-action candidates; additionally require the prediction endpoint to admit the unchanged common vector cubic stop inside the corridor, without overtaking the next unvisited gate, with vector bounds and final rest. Preserve original progress ranking and explicit raw-action no-solution fallback. No shared runtime change or grid expansion.')
    write_json(out/'diagnosis.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','source_sha256','evidence_sha256')}),flush=True)


if __name__=='__main__':main()
