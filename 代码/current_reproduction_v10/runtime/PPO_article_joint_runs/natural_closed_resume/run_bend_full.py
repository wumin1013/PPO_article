"""Full-lap check of a conditionally fitted bend residual with passive context capture."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,pickle,time,shutil,argparse
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
from candidate_policy_factory import candidate_policy
from turn_aware_projector import environment_factory
from run_fitted_turn_full import CapturePolicy
from cache_corner_contexts import memory_state,restore_memory
from paper_complete_evaluation import evaluate
from natural_case_metrics import metrics
from natural_case_audit import analytic_command_audit


def main():
    ap=argparse.ArgumentParser();ap.add_argument('fit_folder');ap.add_argument('--output');args=ap.parse_args()
    fit=Path(args.fit_folder);selection=json.loads((fit/'selected_policy.json').read_text(encoding='utf-8'))
    checks=json.loads((fit/'all34_checks.json').read_text(encoding='utf-8'))
    checked=next(c['result'] for c in checks if c['seed']==selection['seed'] and c['result']['passed'] and c['result']['spec']==selection['bend_spec'])
    assert checked['passed'] and len(checked['records'])==34
    for r in checked['records']:assert sha(fit/(checked['label']+'_'+r['corner']+'.npz'))==r['trace_sha256']
    assert sha(Path(selection['model']))==selection['model_sha256'];torch.set_num_threads(1)
    stage=Path(args.output) if args.output else fit.with_name(fit.name+'_full');stage.mkdir(exist_ok=False)
    files=[Path(__file__),OUT/'candidate_policy_factory.py',OUT/'bend_conditioned_policy.py',OUT/'coordinated_cruise_policy.py',OUT/'turn_aware_projector.py',OUT/'run_fitted_turn_full.py',BASE/'paper_complete_evaluation.py']
    (stage/'source').mkdir()
    for p in files:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='full_lap_bend_residual_validation',wall_budget_s=1800,execution_deadline_s=1710,
        ppo_updates=0,fit_selection_sha256=sha(fit/'selected_policy.json'),source_sha256={str(p):sha(p) for p in files},
        acceptance='Original complete ordered gate coverage,1.25mm analytic Euclidean corridor, rawXY100/2000/62500,20ms rest, all34 registered windows. Nearest-polyline projection reversals are diagnostics, not substitutes for witnessed ordered gate crossings.',
        stop='Stop on physical failure or deadline; preserve prefix. No freeze before all34 dominance and two-resolution complete comparison.'))
    write_json(stage/'selected_policy.json',selection)
    case=BASE/'paper_evidence_review/natural_closed_case';ref=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    g=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
    # CapturePolicy's compatibility field records the inner cruise residual.
    capture_selection=dict(selection,cruise_spec=selection['bend_spec']['cruise_base'])
    policy=CapturePolicy(candidate_policy(metadata,selection),capture_selection,g,ref,stage)
    env=environment_factory(ref)
    if selection['fixed_d'] is not None:env.fixed_d=selection['fixed_d']
    start=time.perf_counter();result=evaluate(policy,env,stage/'full_trace.npz',start+1710)
    if result['success']:
        rows=np.load(stage/'full_trace.npz')['rows'];result['analytic']=analytic_command_audit(ref[0],rows,g)
        m=metrics(ref,rows,g);write_json(stage/'full_metrics.json',m)
        result['success']=bool(result['analytic']['analytic_corridor_pass'] and m['all34_windows_complete'])
        result['projection_backward_steps_diagnostic']=m['projection_raw_backward_steps']
        if not result['success']:result['complete_time_s']=None
    write_json(stage/'ordered_gate_events.json',env.tracker.events)
    error=0.
    for c in policy.contexts:
        c['source_bend_spec']=selection['bend_spec'];c['source_trace_sha256']=sha(stage/'full_trace.npz');c['selection_sha256']=sha(stage/'selected_policy.json')
        restored=candidate_policy(metadata,selection);replay=environment_factory(ref)
        if selection['fixed_d'] is not None:replay.fixed_d=selection['fixed_d']
        restored.bind(replay);replay.restore(c['state']);restore_memory(restored,c['memory'])
        actual=replay.decode_action(restored.act(c['observation'],True)[0],replay.fixed_d)
        error=max(error,float(np.max(abs(actual-c['expected_first_action']))))
    with (stage/'causal_contexts.pkl').open('wb') as f:pickle.dump(policy.contexts,f,protocol=5)
    write_json(stage/'causal_capture_verification.json',dict(contexts=len(policy.contexts),maximum_decoded_action_error=error,verified=error<1e-7))
    if 'TimeoutError' in str(result.get('error')):
        with (stage/'exact_checkpoint.pkl').open('wb') as f:pickle.dump(dict(state=env.state(),memory=memory_state(policy),selection=selection,contexts=policy.contexts),f,protocol=5)
    result.update(goal_complete=False,causal_context_count=len(policy.contexts),causal_capture_verified=error<1e-7,
        experimental_projector_counts=env.experimental_projector.counts)
    write_json(stage/'full_result.json',result)
    write_json(stage/'receipt.json',dict(goal_complete=False,status='bend_full_validation_finished',success=result['success'],
        complete_time_s=result['complete_time_s'],wall_s=time.perf_counter()-start,ppo_updates=0))
    print('BEND_FULL_RESULT',result['success'],result['complete_time_s'],result.get('reason'),result.get('error'),time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
