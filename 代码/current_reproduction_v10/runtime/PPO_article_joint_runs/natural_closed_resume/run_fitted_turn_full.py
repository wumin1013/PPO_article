"""Validate an eligible conditional-fit seed on the complete lap, capture causal states."""
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
from turn_aware_projector import environment_factory
from coordinated_cruise_policy import CoordinatedCruisePolicy
from arc_cruise_policy import build_policy
from deep_joint_credit_policy import load_credit_policy
from cache_corner_contexts import memory_state,restore_memory
from paper_complete_evaluation import evaluate
from natural_case_metrics import metrics
from natural_case_audit import analytic_command_audit

class CapturePolicy:
    """Passive pre-action snapshots, avoiding a second full-history replay."""
    def __init__(self,base,selection,geometry,reference,folder):
        self.base=base;self.selection=selection;self.geometry=geometry;self.reference=reference;self.folder=folder
        self.contexts=[];self.next_corner=0
    def bind(self,env):self.env=env;return self.base.bind(env)
    def act(self,observation,deterministic=True):
        progress,_=self.env.progress();captured=[]
        while self.next_corner<len(self.geometry['sharp_corners']):
            c=self.geometry['sharp_corners'][self.next_corner]
            if progress<c['reference_s_mm']-42.:break
            record=dict(label='actual_'+c['id'],source_name=self.folder.name,source_folder=str(self.folder),
                source_spec=self.selection['source_spec'],source_cruise_spec=self.selection['cruise_spec'],
                fixed_d=self.env.fixed_d,model=self.selection['model'],model_sha256=self.selection['model_sha256'],
                state=self.env.state(),memory=memory_state(self.base),observation=np.asarray(observation).copy(),
                corner=c,stop_s_mm=c['reference_s_mm']+50.,reference=self.reference.copy(),
                actual_start_s_mm=float(progress),capture_kind='passive_pre_action_snapshot_at_42mm_threshold')
            self.contexts.append(record);captured.append(record);self.next_corner+=1
        result=self.base.act(observation,deterministic)
        for record in captured:record['expected_first_action']=self.env.decode_action(result[0],self.env.fixed_d)
        return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['joint','fixed'],required=True);ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';fit=root/(args.arm+'_paired_turn_cruise_fit')
    checks=json.loads((fit/'all34_checks.json').read_text(encoding='utf-8'))
    checked=next(c for c in checks if c['seed']==args.seed)['result']
    assert checked['passed'] and len(checked['records'])==34
    for record in checked['records']:
        file=fit/(checked['label']+'_'+record['corner']+'.npz');assert sha(file)==record['trace_sha256']
    # Selected seed identity comes from its own stored all34 result, not a new search.
    inherited=json.loads((fit/'selected_policy.json').read_text(encoding='utf-8'))
    selection=dict(model=inherited['model'],model_sha256=inherited['model_sha256'],source_spec=inherited['source_spec'],
                   cruise_spec=checked['spec'],fixed_d=checked['fixed_d'],seed=args.seed,source_all34_label=checked['label'],goal_complete=False)
    assert sha(Path(selection['model']))==selection['model_sha256']
    stage=root/(args.arm+'_fit_seed'+str(args.seed)+'_full');stage.mkdir(exist_ok=False)
    source=[Path(__file__),OUT/'turn_aware_projector.py',OUT/'coordinated_cruise_policy.py',OUT/'arc_cruise_policy.py',BASE/'paper_complete_evaluation.py']
    (stage/'source').mkdir()
    for p in source:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='full_lap_conditional_seed_validation',arm=args.arm,seed=args.seed,
        wall_budget_s=1800,execution_deadline_s=1710,ppo_updates=0,independent_full_pipeline_repeat=False,
        source_sha256={str(p):sha(p) for p in source},source_all34_checks_sha256=sha(fit/'all34_checks.json'),
        acceptance='Full coverage, raw vector bounds, analytic corridor, complete brake and rest; complete time and all34 against strongest baselines; no freeze before half-ms validation.',
        capture='Pre-action dynamic state and all causal memories at about40-42mm before each corner, without extra policy calls or new motion.'))
    write_json(stage/'selected_policy.json',selection)
    case=BASE/'paper_evidence_review/natural_closed_case';reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
    base=CoordinatedCruisePolicy(build_policy(load_credit_policy(metadata),selection['source_spec']),selection['cruise_spec'])
    policy=CapturePolicy(base,selection,geometry,reference,stage);env=environment_factory(reference)
    if selection['fixed_d'] is not None:env.fixed_d=selection['fixed_d']
    start=time.perf_counter();result=evaluate(policy,env,stage/'full_trace.npz',start+1710)
    if result['success']:
        rows=np.load(stage/'full_trace.npz')['rows'];result['analytic']=analytic_command_audit(reference[0],rows,geometry)
        m=metrics(reference,rows,geometry);write_json(stage/'full_metrics.json',m)
        result['success']=bool(result['analytic']['analytic_corridor_pass'] and m['all34_windows_complete'] and m['projection_raw_backward_steps']==0)
        if not result['success']:result['complete_time_s']=None
    trace_hash=sha(stage/'full_trace.npz');selection_hash=sha(stage/'selected_policy.json');cache_error=0.
    for c in policy.contexts:
        c['source_trace_sha256']=trace_hash;c['selection_sha256']=selection_hash
        restored=CoordinatedCruisePolicy(build_policy(load_credit_policy(metadata),selection['source_spec']),selection['cruise_spec'])
        replay_env=environment_factory(reference)
        if selection['fixed_d'] is not None:replay_env.fixed_d=selection['fixed_d']
        restored.bind(replay_env);replay_env.restore(c['state']);restore_memory(restored,c['memory'])
        actual=replay_env.decode_action(restored.act(c['observation'],True)[0],replay_env.fixed_d)
        cache_error=max(cache_error,float(np.max(abs(actual-c['expected_first_action']))))
    write_json(stage/'causal_capture_verification.json',dict(contexts=len(policy.contexts),maximum_decoded_action_error=cache_error,
        verified=cache_error<1e-7,physics_ticks=0,extra_policy_calls=len(policy.contexts)))
    with (stage/'causal_contexts.pkl').open('wb') as f:pickle.dump(policy.contexts,f,protocol=5)
    if 'TimeoutError' in str(result.get('error')):
        with (stage/'exact_checkpoint.pkl').open('wb') as f:pickle.dump(dict(state=env.state(),memory=memory_state(policy),selection=selection,
            contexts=policy.contexts,source_sha256={str(p):sha(p) for p in source},trace_sha256=sha(stage/'full_trace.npz')),f,protocol=5)
    result.update(experimental_projector_counts=env.experimental_projector.counts,goal_complete=False,causal_context_count=len(policy.contexts),causal_capture_verified=cache_error<1e-7)
    write_json(stage/'full_result.json',result)
    write_json(stage/'receipt.json',dict(goal_complete=False,status='conditional_seed_full_validation_finished',success=result['success'],
        complete_time_s=result['complete_time_s'],wall_s=time.perf_counter()-start,ppo_updates=0,contexts_sha256=sha(stage/'causal_contexts.pkl')))
    print('FITTED_TURN_FULL_RESULT',args.arm,args.seed,result['success'],result['complete_time_s'],result.get('reason'),result.get('error'),time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
