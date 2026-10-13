"""Check retained Fixed strength against the newly observed full-lap failure state."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,pickle,time,copy,shutil
import numpy as np
import torch
OUT=Path(__file__).resolve().parent;BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
import probe_all34_feed_boundary as local
from turn_aware_projector import environment_factory
from bend_conditioned_policy import BendConditionedPolicy

def main():
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/'fixed_actual_failure_recovery';stage.mkdir(exist_ok=False)
    failed=root/'fixed_bend_physical_fit_saturation_feed_grid_full'
    selection=json.loads((failed/'selected_policy.json').read_text(encoding='utf-8'))
    with (failed/'causal_contexts.pkl').open('rb') as f:actual=pickle.load(f)[-1]
    with (root/'fixed_fit_seed922202_full/causal_contexts.pkl').open('rb') as f:contexts=pickle.load(f)
    source=[Path(__file__),OUT/'bend_conditioned_policy.py',OUT/'coordinated_cruise_policy.py',OUT/'turn_aware_projector.py'];(stage/'source').mkdir()
    for p in source:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='fixed_actual_failure_context_recovery',wall_budget_s=600,ppo_updates=0,
        feed_residual_scales=[1.,.9,.8,.65,.5,0.],source_sha256={str(p):sha(p) for p in source},
        failed_trace_sha256=sha(failed/'full_trace.npz'),failed_cache_sha256=sha(failed/'causal_contexts.pkl'),
        objective='Retain fastest feasible residual according to all34 whole-local deficit, after checking the new actual failure context.',
        adopt='Complete-lap validation still required; no global feed scaling, unchanged baseline model and fixed distance.',
        stop='First local failure stops a proposal; all work ends before590s. Preserve stronger unsuccessful proposals.'))
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
    def factory(ref):
        env=environment_factory(ref);env.fixed_d=selection['fixed_d'];return env
    local.environment_factory=factory;start=time.perf_counter();trials=[];checks=[]
    for scale in [1.,.9,.8,.65,.5,0.]:
        if time.perf_counter()>start+515:break
        spec=copy.deepcopy(selection['bend_spec']);spec['bend_weights'][1]=(np.array(spec['bend_weights'][1])*scale).tolist();label='scale'+str(scale)
        actual_result=local.rollout(metadata,actual,spec,stage/(label+'_actual_failure.npz'),start+590,BendConditionedPolicy)
        records=[]
        if actual_result['context_pass']:
            for c in contexts:
                if time.perf_counter()>start+575:break
                r=local.rollout(metadata,c,spec,stage/(label+'_'+c['corner']['id']+'.npz'),start+590,BendConditionedPolicy);records.append(r)
                if not r['context_pass']:break
        passed=bool(actual_result['context_pass'] and len(records)==34 and all(r['context_pass'] and r['complete_window'] for r in records))
        cost=sum(r['total_local_speed_deficit_mm'] for r in records) if passed else None
        trial=dict(label=label,passed=passed,spec=spec,fixed_d=selection['fixed_d'],cost=cost,records=records,actual_failure_context=actual_result)
        trials.append(trial);checks.append(dict(seed=selection['seed'],result=trial));write_json(stage/'all34_checks.json',checks)
        print('FIXED_ACTUAL_RECOVERY',scale,passed,cost,'actual_distance',actual_result['maximum_local_distance_mm'],'wall',time.perf_counter()-start,flush=True)
    eligible=[r for r in trials if r['passed']]
    if eligible:
        best=min(eligible,key=lambda r:r['cost']);write_json(stage/'selected_policy.json',dict(selection,bend_spec=best['spec'],goal_complete=False,full_lap_not_tested=True))
    write_json(stage/'receipt.json',dict(goal_complete=False,wall_s=time.perf_counter()-start,eligible=len(eligible),ppo_updates=0))

if __name__=='__main__':main()
