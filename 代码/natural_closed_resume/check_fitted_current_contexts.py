"""Check fitted policies against the newly observed complete-history contexts."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,pickle,time,shutil
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
from turn_aware_projector import environment_factory
from coordinated_cruise_policy import CoordinatedCruisePolicy
import probe_all34_feed_boundary as local

def main():
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/'fitted_current_context_check';stage.mkdir(exist_ok=False)
    source=[Path(__file__),OUT/'turn_aware_projector.py',OUT/'coordinated_cruise_policy.py',OUT/'probe_all34_feed_boundary.py']
    (stage/'source').mkdir()
    for p in source:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='new_actual_history_conditional_validation',wall_budget_s=600,ppo_updates=0,
        purpose='Check the selected fitted policy on causal states from the new complete executor trajectories, including changed arrival phases.',
        stop='Continue collecting geometric failures; stop before590s. No complete-time claims from context rollouts.',source_sha256={str(p):sha(p) for p in source}))
    start=time.perf_counter();records=[]
    for arm in ('joint','fixed'):
        with (root/('current_turn_'+arm+'_contexts')/'contexts.pkl').open('rb') as f:contexts=pickle.load(f)
        selection=json.loads((root/(arm+'_paired_turn_cruise_fit')/'selected_policy.json').read_text(encoding='utf-8'))
        metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
        assert selection['model_sha256']==contexts[0]['model_sha256']
        for c in contexts:
            if time.perf_counter()>start+570:break
            def factory(reference):
                env=environment_factory(reference)
                if selection['fixed_d'] is not None:env.fixed_d=selection['fixed_d']
                return env
            local.environment_factory=factory
            result=local.rollout(metadata,c,selection['cruise_spec'],stage/(arm+'_'+c['corner']['id']+'.npz'),start+590,CoordinatedCruisePolicy)
            result['arm']=arm;records.append(result);write_json(stage/'results.json',records)
            print('CURRENT_CONTEXT',arm,result['corner'],result['context_pass'],result['speed_deficit_mm'],result['maximum_local_distance_mm'],flush=True)
    write_json(stage/'receipt.json',dict(goal_complete=False,status='current_history_context_check_complete',records=len(records),
        failures=[(r['arm'],r['corner']) for r in records if not r['context_pass']],wall_s=time.perf_counter()-start,ppo_updates=0))

if __name__=='__main__':main()
