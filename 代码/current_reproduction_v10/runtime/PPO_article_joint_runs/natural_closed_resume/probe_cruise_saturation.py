"""Matched bounded probes of near-cruise request saturation; no adoption."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,pickle,time,copy
import numpy as np
import torch
OUT=Path(__file__).resolve().parent;BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
import probe_all34_feed_boundary as local
from turn_aware_projector import environment_factory
from coordinated_cruise_policy import CoordinatedCruisePolicy

def main():
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/'cruise_saturation_probe';stage.mkdir(exist_ok=False)
    source=Path(__file__);(stage/'source.py').write_bytes(source.read_bytes())
    write_json(stage/'registration.json',dict(kind='diagnostic_matched_cruise_saturation',wall_budget_s=300,ppo_updates=0,
        arms=['joint','fixed'],low_values=[.85,.90,.93,.945],corners=[23,21,26,6,28,30],
        adopt='No adoption. Need all34 and complete-lap validation.',stop='Stop each probe on physical violation; preserve all outcomes; wall limit290s.',source_sha256=sha(source)))
    start=time.perf_counter();records=[]
    for arm in ['joint','fixed']:
        parent=root/(arm+'_fit_seed922202_full');selection=json.loads((parent/'selected_policy.json').read_text(encoding='utf-8'))
        with (parent/'causal_contexts.pkl').open('rb') as f:contexts=pickle.load(f)
        metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
        def factory(ref):
            env=environment_factory(ref)
            if selection['fixed_d'] is not None:env.fixed_d=selection['fixed_d']
            return env
        local.environment_factory=factory
        for low in [.85,.90,.93,.945]:
            spec=copy.deepcopy(selection['cruise_spec']);spec['low']=low
            for i in [23,21,26,6,28,30]:
                if time.perf_counter()>start+280:break
                r=local.rollout(metadata,contexts[i],spec,stage/(arm+'_low'+str(low)+'_corner_'+str(i)+'.npz'),start+290,CoordinatedCruisePolicy)
                r.update(arm=arm,low=low);records.append(r)
                write_json(stage/'results.json',records)
                print('SATURATION',arm,low,i,r['context_pass'],r['speed_deficit_mm'],r['maximum_local_distance_mm'],flush=True)
    write_json(stage/'receipt.json',dict(goal_complete=False,records=len(records),wall_s=time.perf_counter()-start,ppo_updates=0))

if __name__=='__main__':main()
