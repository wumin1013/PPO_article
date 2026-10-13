"""Closed-loop diagnosis of three global inputs; no retraining claim."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json,pickle,shutil
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from probe_all34_feed_boundary import rollout
from study_io import write_json,sha

class MaskProbe:
    def __init__(self,base,mode):self.base=base;self.mode=mode
    def bind(self,env):return self.base.bind(env)
    def act(self,observation,deterministic=True):
        x=np.asarray(observation).copy();x[[16,17,19]]=0.
        return self.base.act(x,deterministic)

def main():
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/'global_input_mask_probe';stage.mkdir(exist_ok=False)
    cache=root/'selected_causal_contexts/contexts.pkl'
    with cache.open('rb') as f:contexts=pickle.load(f)
    shutil.copy2(Path(__file__),stage/'source.py');shutil.copy2(OUT/'probe_all34_feed_boundary.py',stage/'rollout_source.py')
    write_json(stage/'registration.json',dict(kind='conditional_closed_loop_input_mask_diagnostic',wall_budget_s=600,ppo_updates=0,
        masked_indices=[16,17,19],mask_values=[0.,0.,0.],contexts=34,
        preserved='All local physical/geometric observations, all parent parameters, original causal memory at each intervention start, unchanged physical executor and limits.',
        interpretation='Not a newly trained 69-dimensional actor and not a full-lap validation. Intervention begins 40mm before each corner.',
        adoption='Use only to identify where global observation dependence affects local movement.',
        stop='End each failed context immediately, stop before 590seconds, retain all failures.',
        source_cache_sha256=sha(cache),source_sha256=sha(Path(__file__)),rollout_sha256=sha(OUT/'probe_all34_feed_boundary.py')))
    start=time.perf_counter();metadata=torch.load(contexts[0]['model'],map_location='cpu',weights_only=False);records=[]
    baseline=json.loads((root/'existing_evidence_audit.json').read_text(encoding='utf-8'))['records']
    for c,b in zip(contexts,baseline):
        if time.perf_counter()>start+570:break
        r=rollout(metadata,c,'global_triplet_zero',stage/(c['corner']['id']+'.npz'),start+590,MaskProbe)
        r['original_speed_deficit_mm']=b['joint'];r['change_mm']=None if r['speed_deficit_mm'] is None else r['speed_deficit_mm']-b['joint']
        records.append(r);write_json(stage/'results.json',dict(records=records,goal_complete=False,wall_s=time.perf_counter()-start))
        print('MASK_CONTEXT',r['corner'],r['context_pass'],'deficit',r['speed_deficit_mm'],'change',r['change_mm'],'distance',r['maximum_local_distance_mm'],flush=True)
    write_json(stage/'receipt.json',dict(goal_complete=False,status='diagnostic_complete',contexts=len(records),
        passed=sum(r['context_pass'] for r in records),improved_complete=sum(r['context_pass'] and r['change_mm']<0 for r in records),
        ppo_updates=0,wall_s=time.perf_counter()-start))

if __name__=='__main__':main()
