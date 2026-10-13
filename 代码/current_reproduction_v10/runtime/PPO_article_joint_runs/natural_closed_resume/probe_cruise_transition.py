"""Predeclared grid on the failed feed-support transition, no full adoption."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,pickle,shutil
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
import probe_all34_feed_boundary as local
from turn_aware_projector import environment_factory
from coordinated_cruise_policy import CoordinatedCruisePolicy

def main():
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/'cruise_transition_grid';stage.mkdir(exist_ok=False)
    with (root/'selected_causal_contexts/contexts.pkl').open('rb') as f:all_contexts=pickle.load(f)
    contexts=[all_contexts[i] for i in (30,4,31,0,22,2,6,28)]
    (stage/'source').mkdir()
    for p in (Path(__file__),OUT/'coordinated_cruise_policy.py',OUT/'turn_aware_projector.py',OUT/'probe_all34_feed_boundary.py'):
        shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='conditional_feed_support_grid',ppo_updates=0,wall_budget_s=600,
        high_grid=[1.,1.025,1.05,1.075,1.10],low=.85,amplitude=20.,
        contexts=[c['label'] for c in contexts],objective='First all selected contexts feasible, then registered corner deficits and whole-local deficit.',
        stop='Stop each candidate at first failed context. No adoption from local grid.',script_sha256=sha(Path(__file__))))
    local.environment_factory=environment_factory;metadata=torch.load(contexts[0]['model'],map_location='cpu',weights_only=False)
    start=time.perf_counter();records=[]
    for high in (1.,1.025,1.05,1.075,1.10):
        spec=dict(amplitude=20.,low=.85,high=high,feedback_delta=[0.,0.,0.],global_fraction=1.);rr=[]
        for c in contexts:
            if time.perf_counter()>start+570:break
            r=local.rollout(metadata,c,spec,stage/(str(high)+'_'+c['corner']['id']+'.npz'),start+590,CoordinatedCruisePolicy);rr.append(r)
            print('TRANSITION_GRID',high,r['corner'],r['context_pass'],r['speed_deficit_mm'],r['maximum_local_distance_mm'],flush=True)
            if not r['context_pass']:break
        records.append(dict(spec=spec,passed=len(rr)==len(contexts) and all(r['context_pass'] for r in rr),records=rr))
        write_json(stage/'results.json',records)
    write_json(stage/'receipt.json',dict(goal_complete=False,wall_s=time.perf_counter()-start,ppo_updates=0,
        passed_candidates=sum(r['passed'] for r in records),status='diagnostic_grid_complete'))

if __name__=='__main__':main()
