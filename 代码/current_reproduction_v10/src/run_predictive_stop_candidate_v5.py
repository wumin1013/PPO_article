"""One fixed v5_2 change on the same three complete development programs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import json
import pickle
import time
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from runtime_physical_time_v5 import install
from evaluate import rollout
from frozen_policy import FrozenPolicy
from replay_strong_checkpoint_v4 import MODEL,EXPECTED_MODEL
from predictive_action_candidate_v5 import new_environment,snapshot,state_bytes,retained_step
from predictive_stop_candidate_v5 import VERSION,PredictiveStopCandidate
from braking_reachability_v5 import stopping_extension

OUT=ROOT/'results/predictive_stop_candidate_v5_2'
PRIOR=ROOT/'results/predictive_action_candidate_v5'


def register():
    file=OUT/'registration.json'
    if not file.exists():
        prior=json.loads((PRIOR/'registration.json').read_text('utf-8'))
        reg=copy.deepcopy(prior)
        reg.update(created_utc=utc(),version=VERSION,parent_candidate_registration_sha256=sha(PRIOR/'registration.json'),
            fold_diagnosis_sha256=sha(PRIOR/'fold_stopping_diagnosis/diagnosis.json'),
            sole_algorithm_change='Every feasible nonterminal 120ms feedback prediction must also admit the unchanged common bounded vector cubic stop. Require all stop intervals inside the same corridor, vector bounds, 20ms rest, and no next unvisited gate behind the stopping point. Stop is speculative only; actual commands remain unchanged common executor actions.',
            stopping_cost='Record every attempted stopping profile, including failures, cost and full intervals separately from model-feedback predictions',
            no_recursive_guarantee='Finite action set may have no feasible candidate; original raw fallback remains explicit and may fail.',
            parameter_rule='One preregistered extra predicate, no grid expansion, no horizon retuning, no per-case settings, no outcome-selected repeat',
            main_binding='Preserve original main and v5_1 files and evidence unchanged')
        names=['braking_reachability_v5.py','predictive_stop_candidate_v5.py','run_predictive_stop_candidate_v5.py']
        reg['source_sha256'].update({n:sha(ROOT/'src'/n) for n in names})
        write_json(file,reg)
    reg=json.loads(file.read_text('utf-8'))
    assert sha(MODEL)==EXPECTED_MODEL==reg['model_sha256']
    assert all(sha(ROOT/'src'/k)==v for k,v in reg['source_sha256'].items())
    return reg


def audit(reg):
    out=OUT/'unit_audit';out.mkdir(parents=True,exist_ok=False)
    install(dt=.001,tolerance=1.25)
    reference=np.loadtxt(ROOT/'data/train_box.csv',delimiter=',',skiprows=1)
    env=new_environment(reference);env.env.max_ticks=97531
    recorded=[];update=env.tracker.update
    def record(point):
        recorded.append(np.asarray(point).copy());return update(point)
    env.tracker.update=record
    saved=state_bytes(env);raw=FrozenPolicy(MODEL).mean(env.observation()).numpy()
    stop_rows,stop_positions,stop=stopping_extension(env)
    assert stop['feasible'] and stop['stopping_length_mm']==0 and saved==state_bytes(env) and recorded==[]
    p=PredictiveStopCandidate(MODEL,out/'normal_action')
    action=p.action_for(env);normal=p.close()
    assert np.array_equal(action,raw) and action.dtype==raw.dtype and saved==state_bytes(env) and recorded==[]
    expected=new_environment(reference,snapshot(env))
    expected_rows,error=retained_step(expected,raw)
    actual_rows,error2=retained_step(env,action)
    assert error is None and error2 is None and np.array_equal(expected_rows,actual_rows)
    assert len(recorded)==len(actual_rows)
    result=dict(status='passed',created_utc=utc(),zero_velocity_stop_feasible=True,
        no_real_state_or_recorder_mutation=True,raw_action_bitwise_equal=True,
        actual_action_rows_bitwise_equal=True,normal=normal,registered_source_sha256=reg['source_sha256'])
    write_json(out/'report.json',result)
    print('STOP_CANDIDATE_UNIT_AUDIT_PASSED',flush=True)


def run(case,reg):
    unit=json.loads((OUT/'unit_audit/report.json').read_text('utf-8'))
    assert unit['status']=='passed' and unit['registered_source_sha256']==reg['source_sha256']
    row=next(r for r in reg['cases'] if r['id']==case)
    assert sha(ROOT/row['reference'])==row['reference_sha256']
    output=OUT/case;output.mkdir(parents=True,exist_ok=False)
    reference=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
    setup=install(dt=.001,tolerance=row['evaluation_tolerance_mm'])
    started=time.perf_counter();policy=PredictiveStopCandidate(MODEL,output/'predictions',deadline=started+300.)
    try:
        result=rollout(policy,reference,output/'actual.npz',dt=.001,wall_s=300.,
            tolerance=row['evaluation_tolerance_mm'],runtime_version='v3')
    finally:
        summary=policy.close();write_json(output/'prediction_summary.json',summary)
    result.update(path=case,scope=reg['scope'],method=VERSION,model_sha256=EXPECTED_MODEL,
        model_updated=False,executor_setup=setup,runtime_version=setup['version'],
        action_source='Original c16d45 model with separately versioned six-decision selection and speculative endpoint stop admissibility; common true executor unchanged',
        reference_sha256=row['reference_sha256'],nominal_tolerance_mm=row['nominal_tolerance_mm'],
        arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],registration_sha256=sha(OUT/'registration.json'),
        prediction=summary,full_run_wall_s=time.perf_counter()-started)
    write_json(output/'actual.json',result)
    print('STOP_CANDIDATE_RUN',case,result['success'],result.get('complete_time_s'),result.get('elapsed_prefix_s'),result['reason'],summary['counts'],flush=True)


def summarize(reg):
    results=[json.loads((OUT/case/'actual.json').read_text('utf-8')) for case in reg['run_order']]
    result=dict(created_utc=utc(),status='all_three_registered_runs_recorded',version=VERSION,
        scope=reg['scope'],successes=sum(r['success'] for r in results),
        model_sha256=EXPECTED_MODEL,results=[dict(path=r['path'],success=r['success'],
            complete_time_s=r.get('complete_time_s'),elapsed_prefix_s=r.get('elapsed_prefix_s'),
            reason=r['reason'],error=r['error'],actual_calls=r['actor_calls'],prediction=r['prediction']) for r in results],
        interpretation='One fixed development-only stopping-admissibility candidate; no replacement of historical main or v5_1, no formal final-test or five-seed evidence')
    write_json(OUT/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',action='store_true');parser.add_argument('--all',action='store_true')
    parser.add_argument('--summarize',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    reg=register()
    if args.audit:audit(reg)
    elif args.all:
        for case in reg['run_order']:run(case,reg)
        summarize(reg)
    elif args.summarize:summarize(reg)
    else:parser.error('Choose audit, all, or summarize')
