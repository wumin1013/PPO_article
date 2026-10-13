"""Exact causal replay and continuation of a wall-budget-limited full-lap prefix."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,argparse,shutil
import numpy as np
import torch
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from paper_complete_evaluation import evaluate
from natural_case_metrics import metrics
from natural_case_audit import analytic_command_audit
from study_io import sha,write_json
from arc_cruise_policy import build_policy
from resume_resolution_candidate import replay


def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('--stage-name',default='exact_continuation',choices=['exact_continuation','checkpoint_continuation']);args=ap.parse_args();candidate=Path(args.candidate)
    old=json.loads((candidate/'full_result.json').read_text(encoding='utf-8'))
    assert not old['success'] and 'TimeoutError' in str(old.get('error')), 'Only budget-limited prefixes can resume'
    assert old['geometry']['corridor_inside'] and old['audit']['kinematic_pass']
    selection=json.loads((candidate/'selected_policy.json').read_text(encoding='utf-8'))
    parent_reg=json.loads((candidate/'registration.json').read_text(encoding='utf-8'))
    stage=candidate/args.stage_name;stage.mkdir(exist_ok=False)
    start=time.perf_counter();torch.set_num_threads(1)
    prefix=dict(np.load(candidate/'full_trace.npz'));n=len(prefix['actions'])
    assert len(prefix['rows'])==20*n and np.all(prefix['rows'][:,12]==0.)
    model=Path(selection['model']);assert sha(model)==selection['model_sha256']
    files=[Path(__file__),OUT/'resume_resolution_candidate.py',OUT/'cache_corner_contexts.py',OUT/'arc_cruise_policy.py',OUT/'cruise_residual_policy.py',OUT/'turn_feedback_policy.py',BASE/'paper_complete_evaluation.py']
    write_json(stage/'registration.json',dict(arm=parent_reg['arm'],kind='exact_causal_prefix_continuation',wall_budget_s=1800,
        total_deadline_s=1720,replay_deadline_s=1180,ppo_updates=0,selection_sha256=sha(candidate/'selected_policy.json'),
        prefix_sha256=sha(candidate/'full_trace.npz'),prefix_physics_ticks=len(prefix['rows']),prefix_actor_calls=n,
        adoption='Exact saved-action and observation identity required before any resumed physics. Final full raw trajectory includes original prefix, remaining ordered gates, braking and stationary tail.',
        independent_repeat=False,code_sha256={str(p):sha(p) for p in files}))
    shutil.copy2(candidate/'selected_policy.json',stage/'selected_policy.json');(stage/'source').mkdir()
    for path in files:shutil.copy2(path,stage/'source'/path.name)
    policy=build_policy(load_credit_policy(torch.load(model,map_location='cpu',weights_only=False)),selection['spec'])
    if parent_reg.get('probe'):
        from evaluate_retained import ActionProbe
        policy=ActionProbe(policy,parent_reg['probe'])
    env=environment_factory(prefix['reference'])
    if parent_reg['arm']=='fixed':env.fixed_d=selection['spec']['fixed_d']
    fingerprint=dict(trace_sha256=sha(candidate/'full_trace.npz'),model_sha256=sha(model),spec=selection['spec'],dt_s=.001,
        probe=parent_reg.get('probe'),helpers={n:sha(OUT/n) for n in ('arc_cruise_policy.py','cruise_residual_policy.py','turn_feedback_policy.py')})
    try:identity=replay(policy,env,prefix,.001,start+1180,candidate/'verified_replay_cache',fingerprint)
    except Exception as exc:
        write_json(stage/'receipt.json',dict(status='replay_not_verified_no_new_motion',error=f'{type(exc).__name__}: {exc}',wall_s=time.perf_counter()-start,goal_complete=False));raise
    identity.update(wall_s=time.perf_counter()-start,new_physics_ticks=0)
    write_json(stage/'replay_identity.json',identity)
    result=evaluate(policy,env,stage/'full_trace.npz',start+1720,prefix=prefix,bind_policy=False)
    case=BASE/'paper_evidence_review/natural_closed_case';geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    if result['success']:
        rows=np.load(stage/'full_trace.npz')['rows'];result['analytic']=analytic_command_audit(prefix['reference'][0],rows,geometry)
        m=metrics(prefix['reference'],rows,geometry);write_json(stage/'full_metrics.json',m)
        result['success']=bool(result['analytic']['analytic_corridor_pass'] and m['all34_windows_complete'] and m['projection_raw_backward_steps']==0)
        if not result['success']:result['complete_time_s']=None
    result.update(spec=selection['spec'],prefix_sha256=sha(candidate/'full_trace.npz'),exact_continuation=True,
        new_physics_ticks=result['physics_ticks']-len(prefix['rows']))
    write_json(stage/'full_result.json',result)
    write_json(stage/'receipt.json',dict(status='complete_exact_continuation',success=result['success'],complete_time_s=result['complete_time_s'],
        total_wall_s=time.perf_counter()-start,goal_complete=False,ppo_updates=0,independent_repeat=False,
        result_sha256=sha(stage/'full_result.json')))
    print('CONTINUATION_COMPLETE',result['success'],result['complete_time_s'],time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
