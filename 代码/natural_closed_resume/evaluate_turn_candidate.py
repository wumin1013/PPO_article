"""All34 conditional screen followed by the untouched complete natural lap."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json,pickle,shutil,argparse
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
import probe_all34_feed_boundary as local
from turn_aware_projector import environment_factory
from coordinated_cruise_policy import CoordinatedCruisePolicy
from arc_cruise_policy import build_policy
from deep_joint_credit_policy import load_credit_policy
from cache_corner_contexts import memory_state
from paper_complete_evaluation import evaluate
from natural_case_metrics import metrics
from natural_case_audit import analytic_command_audit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['joint','fixed'],default='joint');ap.add_argument('--stage',required=True)
    ap.add_argument('--high',type=float,default=None);args=ap.parse_args()
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/args.stage;stage.mkdir(exist_ok=False)
    grid=json.loads((root/'cruise_transition_grid/results.json').read_text(encoding='utf-8'))
    eligible=[r for r in grid if r['passed'] and (args.high is None or abs(r['spec']['high']-args.high)<1e-12)]
    selected=min(eligible,key=lambda r:sum(q['total_local_speed_deficit_mm'] for q in r['records']))
    spec=selected['spec'];cache=root/('selected_causal_contexts' if args.arm=='joint' else 'fixed_causal_contexts')/'contexts.pkl'
    with cache.open('rb') as f:contexts=pickle.load(f)
    source=[Path(__file__),OUT/'turn_aware_projector.py',OUT/'coordinated_cruise_policy.py',OUT/'probe_all34_feed_boundary.py',BASE/'paper_complete_evaluation.py']
    (stage/'source').mkdir()
    for p in source:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='screened_full_lap_executor_candidate',arm=args.arm,wall_budget_s=1800,
        screen_deadline_s=300,full_deadline_s=1710,ppo_updates=0,independent_training_seeds=0,
        spec=spec,source_cache_sha256=sha(cache),source_sha256={str(p):sha(p) for p in source},
        selection=('Lowest whole-local deficit among feasible predeclared feed-support grid entries.' if args.high is None else
                   'Explicit predeclared-grid alternative after a broader context screen failed: high='+str(args.high)),
        adoption='Not adopted by this run. Strong Fixed needs same executor and training opportunity; 0.5ms, full34 dominance and mechanism remain mandatory.',
        stop='No full lap unless all34 local contexts pass. Abort on physical violations. Preserve incomplete prefix and exact state when wall budget expires.'))
    start=time.perf_counter();metadata=torch.load(contexts[0]['model'],map_location='cpu',weights_only=False);screen=[]
    for c in contexts:
        def factory(reference):
            env=environment_factory(reference)
            if c.get('fixed_d') is not None:env.fixed_d=c['fixed_d']
            return env
        local.environment_factory=factory
        r=local.rollout(metadata,c,spec,stage/('screen_'+c['corner']['id']+'.npz'),start+300,CoordinatedCruisePolicy)
        screen.append(r);print('CANDIDATE_SCREEN',args.arm,r['corner'],r['context_pass'],r['speed_deficit_mm'],r['maximum_local_distance_mm'],flush=True)
        write_json(stage/'screen.json',screen)
        if r['raw_xy_vector_peaks'] and np.any(np.asarray(r['raw_xy_vector_peaks'])>np.array([100.,2000.,62500.])*1.00001):raise ValueError('Actual physical bound violation')
    if len(screen)!=34 or not all(r['context_pass'] for r in screen):
        write_json(stage/'receipt.json',dict(goal_complete=False,status='all34_screen_failed_no_full_lap',wall_s=time.perf_counter()-start));return
    selection=dict(model=contexts[0]['model'],model_sha256=contexts[0]['model_sha256'],source_spec=contexts[0]['source_spec'],
                   cruise_spec=spec,fixed_d=contexts[0].get('fixed_d'),executor='turn_aware_projector',goal_complete=False)
    write_json(stage/'selected_policy.json',selection)
    reference=contexts[0]['reference'];env=environment_factory(reference)
    if selection['fixed_d'] is not None:env.fixed_d=selection['fixed_d']
    policy=CoordinatedCruisePolicy(build_policy(load_credit_policy(metadata),selection['source_spec']),spec)
    result=evaluate(policy,env,stage/'full_trace.npz',start+1710)
    if result['success']:
        rows=np.load(stage/'full_trace.npz')['rows']
        geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
        result['analytic']=analytic_command_audit(reference[0],rows,geometry)
        m=metrics(reference,rows,geometry);write_json(stage/'full_metrics.json',m)
        result['success']=bool(result['analytic']['analytic_corridor_pass'] and m['all34_windows_complete'] and m['projection_raw_backward_steps']==0)
        if not result['success']:result['complete_time_s']=None
    elif 'TimeoutError' in str(result.get('error')):
        with (stage/'exact_checkpoint.pkl').open('wb') as f:pickle.dump(dict(state=env.state(),memory=memory_state(policy),selection=selection,
            source_sha256={str(p):sha(p) for p in source},trace_sha256=sha(stage/'full_trace.npz')),f,protocol=5)
    result['experimental_projector_counts']=env.experimental_projector.counts
    result['goal_complete']=False;write_json(stage/'full_result.json',result)
    write_json(stage/'receipt.json',dict(goal_complete=False,status='full_candidate_evaluation_finished',
        success=result['success'],complete_time_s=result['complete_time_s'],wall_s=time.perf_counter()-start,ppo_updates=0))
    print('TURN_FULL_RESULT',args.arm,result['success'],result['complete_time_s'],result.get('reason'),result.get('error'),time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
