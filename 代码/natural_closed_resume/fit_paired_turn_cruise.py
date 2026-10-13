"""Matched conditional searches for Joint/Fixed under the experimental executor."""
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

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['joint','fixed'],required=True);args=ap.parse_args()
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/(args.arm+'_paired_turn_cruise_fit');stage.mkdir(exist_ok=False)
    cache=root/('selected_causal_contexts' if args.arm=='joint' else 'fixed_causal_contexts')/'contexts.pkl'
    with cache.open('rb') as f:all_contexts=pickle.load(f)
    training_ids=[4,31,20,10,14,23,26,30,2,6,28,0,22];contexts=[all_contexts[i] for i in training_ids]
    seeds=[922201,922202,922203];source=[Path(__file__),OUT/'coordinated_cruise_policy.py',OUT/'turn_aware_projector.py',OUT/'probe_all34_feed_boundary.py']
    (stage/'source').mkdir()
    for p in source:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='paired_conditional_closed_loop_parameter_fit',arm=args.arm,
        wall_budget_s=1800,per_seed_wall_budget_s=300,proposals_per_seed=7,seeds=seeds,ppo_updates=0,
        independent_full_pipeline_training=False,shared_inherited_model=True,
        training_contexts=[c['label'] for c in contexts],fixed_distance_grid=[.8,1.5,3.,4.6,6.,8.,12.] if args.arm=='fixed' else [],
        parameters=['steering_error_feedback_delta','feed_error_feedback_delta','preview_error_feedback_delta' if args.arm=='joint' else 'global_fixed_preview_mm','cruise_support_high','global_input_multiplier'],
        objective='Lexicographic complete local feasibility, then sum whole-local speed deficit; same objective for both arms. No incomplete trajectory receives speed gain.',
        selection='Retain best feasible result per search seed; check all34 actual contexts for each. No full-lap or method adoption from this search.',
        stop='End candidate at first failed context. Equal7 proposals and300s per seed. End all checks by1780s; preserve all outcomes.',
        source_cache_sha256=sha(cache),source_sha256={str(p):sha(p) for p in source}))
    metadata=torch.load(contexts[0]['model'],map_location='cpu',weights_only=False);start=time.perf_counter();trials=[];retained=[]
    def run(vector,label,deadline,selected_contexts=contexts):
        spec=dict(amplitude=20.,low=.85,high=float(vector[3]),global_fraction=float(vector[4]),
                  feedback_delta=[float(vector[0]),float(vector[1]),float(vector[2]) if args.arm=='joint' else 0.])
        distance=float(vector[2]) if args.arm=='fixed' else None;records=[]
        for c in selected_contexts:
            if time.perf_counter()>deadline-2:break
            def factory(reference):
                env=environment_factory(reference)
                if args.arm=='fixed':env.fixed_d=distance
                return env
            local.environment_factory=factory
            r=local.rollout(metadata,c,spec,stage/(label+'_'+c['corner']['id']+'.npz'),deadline,CoordinatedCruisePolicy);records.append(r)
            if not r['context_pass']:break
        passed=len(records)==len(selected_contexts) and all(r['context_pass'] for r in records)
        cost=sum(r['total_local_speed_deficit_mm'] for r in records) if passed else None
        key=[0 if passed else 1,cost if passed else len(selected_contexts)-len(records)]
        rec=dict(label=label,vector=np.asarray(vector).tolist(),spec=spec,fixed_d=distance,passed=passed,cost=cost,key=key,records=records)
        trials.append(rec);write_json(stage/'progress.json',dict(trials=trials,retained=retained,wall_s=time.perf_counter()-start,goal_complete=False))
        print('PAIRED_FIT',args.arm,label,passed,'cost',cost,'contexts',len(records),'parameters',rec['vector'],'wall',time.perf_counter()-start,flush=True)
        return rec
    initial=np.array([0.,0.,4.6 if args.arm=='fixed' else 0.,1.,1.])
    prior=run(initial,'inherited_candidate',start+240)
    if args.arm=='fixed':
        for distance in (.8,1.5,3.,4.6,6.,8.,12.):
            if distance==4.6:continue
            x=initial.copy();x[2]=distance;r=run(x,'fixed_distance_'+str(distance),start+300)
            if r['key']<prior['key']:prior=r
    for seed in seeds:
        seed_start=time.perf_counter();deadline=min(start+1350,seed_start+300);rng=np.random.default_rng(seed);best=prior
        for step in range(7):
            if time.perf_counter()>deadline-35:break
            center=np.asarray(best['vector']);scale=np.array([.015,.09,.20 if args.arm=='fixed' else .10,.02,.035])
            if step==0:
                vector=center+np.array([-.01,.10,0. if args.arm=='fixed' else -.12,0.,-.025])+rng.normal(0,scale*.2)
            else:vector=center+rng.normal(0,scale)
            low=np.array([-.12,-.15,.8 if args.arm=='fixed' else -.6,.96,.5])
            high=np.array([.06,.65,12. if args.arm=='fixed' else .15,1.10,1.])
            vector=np.clip(vector,low,high)
            r=run(vector,f'seed{seed}_step{step}',deadline)
            if r['key']<best['key']:best=r
        retained.append(dict(seed=seed,selected=best));write_json(stage/'retained.json',retained)
    checked=[]
    for item in sorted(retained,key=lambda x:x['selected']['key']):
        if time.perf_counter()>start+1660:break
        if not item['selected']['passed']:continue
        r=run(np.asarray(item['selected']['vector']),f'seed{item["seed"]}_all34',start+1770,all_contexts)
        checked.append(dict(seed=item['seed'],result=r));write_json(stage/'all34_checks.json',checked)
    eligible=[r for r in checked if r['result']['passed']]
    chosen=min(eligible,key=lambda r:r['result']['cost']) if eligible else None
    if chosen:
        write_json(stage/'selected_policy.json',dict(model=contexts[0]['model'],model_sha256=contexts[0]['model_sha256'],
            source_spec=contexts[0]['source_spec'],cruise_spec=chosen['result']['spec'],fixed_d=chosen['result']['fixed_d'],
            selected_from=chosen['result']['label'],goal_complete=False,full_lap_not_tested=True))
    write_json(stage/'receipt.json',dict(goal_complete=False,status='conditional_paired_fit_complete',arm=args.arm,ppo_updates=0,
        independent_full_pipeline_seeds=0,conditional_seeds=len(retained),eligible_all34=len(eligible),wall_s=time.perf_counter()-start))
    print('PAIRED_FIT_COMPLETE',args.arm,len(eligible),time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
