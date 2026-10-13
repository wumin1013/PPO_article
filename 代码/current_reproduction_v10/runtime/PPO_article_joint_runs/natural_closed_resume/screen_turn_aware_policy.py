"""All-corner conditional screening under an explicitly versioned common executor."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json,pickle,argparse,shutil
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
import probe_all34_feed_boundary as local
from turn_aware_projector import environment_factory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['joint','fixed'],default='joint')
    ap.add_argument('--mode',default='shallow_max_feed');args=ap.parse_args()
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/(args.arm+'_turn_aware_all34_'+args.mode);stage.mkdir(exist_ok=False)
    cache=root/('selected_causal_contexts' if args.arm=='joint' else 'fixed_causal_contexts')/'contexts.pkl'
    with cache.open('rb') as f:contexts=pickle.load(f)
    source=[Path(__file__),OUT/'turn_aware_projector.py',OUT/'probe_all34_feed_boundary.py']
    (stage/'source').mkdir()
    for p in source:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='all34_conditional_executor_and_feed_screen',wall_budget_s=600,ppo_updates=0,
        arm=args.arm,mode=args.mode,source_cache_sha256=sha(cache),source_sha256={str(p):sha(p) for p in source},
        adoption='Screen only. Full original lap and two execution resolutions remain mandatory; no independent training claim.',
        stop='End failed context and preserve it. Any actual V/A/J violation stops the whole round. Stop by 590 seconds.',
        safety_checks=str(root/'turn_aware_execution_cruise_precision_probe/safety_checks.json')))
    metadata=torch.load(contexts[0]['model'],map_location='cpu',weights_only=False);start=time.perf_counter();records=[]
    originals=json.loads((root/'existing_evidence_audit.json').read_text(encoding='utf-8'))['records']
    for c,b in zip(contexts,originals):
        if time.perf_counter()>start+570:break
        def factory(reference):
            env=environment_factory(reference)
            if c.get('fixed_d') is not None:env.fixed_d=c['fixed_d']
            return env
        local.environment_factory=factory
        r=local.rollout(metadata,c,args.mode,stage/(c['corner']['id']+'.npz'),start+590)
        r.update(original_joint_deficit=b['joint'],historical_s2_deficit=b['s2'],historical_fixed_deficit=b['fixed'])
        records.append(r)
        print('TURN_ALL34',args.arm,r['corner'],r['context_pass'],'deficit',r['speed_deficit_mm'],'limits',min(b['s2'],b['fixed']),'distance',r['maximum_local_distance_mm'],flush=True)
        write_json(stage/'results.json',dict(records=records,goal_complete=False,wall_s=time.perf_counter()-start))
        if r['raw_xy_vector_peaks'] and np.any(np.asarray(r['raw_xy_vector_peaks'])>np.array([100.,2000.,62500.])*1.00001):
            raise ValueError('Actual physical bound violation')
    write_json(stage/'receipt.json',dict(goal_complete=False,status='all34_conditional_screen_complete',
        count=len(records),passed=sum(r['context_pass'] for r in records),wall_s=time.perf_counter()-start,
        against_historical_baselines_only=True,ppo_updates=0))

if __name__=='__main__':main()
