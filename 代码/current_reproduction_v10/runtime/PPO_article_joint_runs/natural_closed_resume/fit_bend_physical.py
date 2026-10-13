"""Bounded conditional closed-loop fitting; all outcomes retained, no PPO claims."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,pickle,time,shutil,argparse
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
from natural_case_metrics import ordered_projection
import probe_all34_feed_boundary as local
from turn_aware_projector import environment_factory
from bend_conditioned_policy import BendConditionedPolicy


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['joint','fixed'],required=True)
    ap.add_argument('--budget',type=float,default=1200);args=ap.parse_args()
    torch.set_num_threads(1);root=OUT/'all34_dominance_research';stage=root/(args.arm+'_bend_physical_fit');stage.mkdir(exist_ok=False)
    parent=root/(args.arm+'_fit_seed922202_full');cache=parent/'causal_contexts.pkl'
    with cache.open('rb') as f:all_contexts=pickle.load(f)
    selection=json.loads((parent/'selected_policy.json').read_text(encoding='utf-8'))
    audit=json.loads((root/'fitted_turn_pair_audit.json').read_text(encoding='utf-8'))
    targets={r['id']:min(r['s2'],r['fixed_successful_prior'],r['fixed_unaccepted_probe']) for r in audit['all34']}
    train_ids=[23,29,4,20,28,16,10,18,31,30,26,14,2,6]
    contexts=[all_contexts[i] for i in train_ids]
    source=[Path(__file__),OUT/'bend_conditioned_policy.py',OUT/'coordinated_cruise_policy.py',OUT/'turn_aware_projector.py',OUT/'probe_all34_feed_boundary.py']
    (stage/'source').mkdir()
    for p in source:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='bend_conditioned_closed_loop_parameter_optimization',arm=args.arm,
        wall_budget_s=args.budget,ppo_updates=0,seeds=[922211,922212,922213],independent_ppo_seeds=0,
        inherited_policy=str(parent/'selected_policy.json'),source_cache_sha256=sha(cache),
        source_sha256={str(p):sha(p) for p in source},training_corner_ids=train_ids,
        feature='Three smooth bend bases and observed cross-track feedback, plus anticipatory feed support; no corner IDs, route position, or time tables in the added policy.',
        targets=targets,target_scope='Envelope includes Fixed seed922202, independently accepted by witnessed ordered gate crossings in complete_contract_audit.json; plot projection reversals are diagnostic only.',
        objective='Feasibility first, then total positive per-corner deficit gap plus0.05 times whole-local total deficit. Same objective and proposal opportunities per arm.',
        checks='All34 actual contexts with ordered gates and physical limits; nearest projection reversals diagnostic only. Full laps and half-ms required separately.',
        stop='First failed context stops proposal. Stop fitting at75 percent wall budget; preserve trials. Final checks stop before wall deadline.'))
    start=time.perf_counter();deadline=start+args.budget-10
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False);trials=[];retained=[]
    def run(weights,distance,label,selected,until):
        spec=dict(cruise_base=selection['cruise_spec'],bend_weights=np.asarray(weights).reshape(3,3).tolist())
        def factory(reference):
            env=environment_factory(reference)
            if args.arm=='fixed':env.fixed_d=float(distance)
            return env
        local.environment_factory=factory;records=[]
        for c in selected:
            if time.perf_counter()>until-3:break
            file=stage/(label+'_'+c['corner']['id']+'.npz')
            r=local.rollout(metadata,c,spec,file,until,BendConditionedPolicy)
            rows=np.load(file)['rows']
            p=ordered_projection(c['reference'],rows[:,:2]) if len(rows) else np.array([])
            r['projection_backward_steps']=int((np.diff(p)<-1e-7).sum())
            r['strict_context_pass']=bool(r['context_pass'] and r['complete_window'])
            records.append(r)
            if not r['strict_context_pass']:break
        passed=len(records)==len(selected) and all(r['strict_context_pass'] for r in records)
        positive=sum(max(r['speed_deficit_mm']-targets[r['corner']],0.) for r in records) if passed else None
        total=sum(r['total_local_speed_deficit_mm'] for r in records) if passed else None
        cost=positive+.05*total if passed else None
        key=[0,cost] if passed else [1,len(selected)-len(records)]
        rec=dict(label=label,passed=passed,key=key,cost=cost,positive_gap_sum_mm=positive,total_deficit_mm=total,
            spec=spec,fixed_d=distance,records=records)
        trials.append(rec);write_json(stage/'progress.json',dict(trials=trials,retained=retained,wall_s=time.perf_counter()-start,goal_complete=False))
        print('BEND_FIT',args.arm,label,passed,'cost',cost,'gap',positive,'contexts',len(records),'wall',time.perf_counter()-start,flush=True)
        return rec
    initial=np.zeros((3,3));distance=selection['fixed_d']
    prior=run(initial,distance,'zero_residual',contexts,min(deadline,start+180))
    # Deterministic sensitivity-informed first steps are shared by every arm.
    warm=[]
    w=initial.copy();w[1]=[.3,.3,.15];warm.append(w)
    w=initial.copy();w[1]=[.6,.6,.3];w[2]=[.15,.15,.1] if args.arm=='joint' else 0.;warm.append(w)
    w=initial.copy();w[1]=[.5,.5,.2];w[0]=[-.04,-.04,-.02];w[2]=[-.15,-.15,-.1] if args.arm=='joint' else 0.;warm.append(w)
    for k,w in enumerate(warm):
        r=run(w,distance,'warm'+str(k),contexts,min(deadline,start+args.budget*.25))
        if r['key']<prior['key']:prior=r
    for seed in (922211,922212,922213):
        until=min(start+args.budget*.75,time.perf_counter()+args.budget*.17);rng=np.random.default_rng(seed);best=prior
        for step in range(8):
            if time.perf_counter()>until-30:break
            center=np.asarray(best['spec']['bend_weights']);scale=np.array([.02,.18,.13])[:,None]
            w=center+rng.normal(size=(3,3))*scale
            w=np.clip(w,np.array([-.20,-.4,-.7])[:,None],np.array([.15,1.8,.7])[:,None])
            d=best['fixed_d']
            if args.arm=='fixed':w[2]=0.;d=float(np.clip(d+rng.normal(0,.12),.8,12.))
            r=run(w,d,f'seed{seed}_step{step}',contexts,until)
            if r['key']<best['key']:best=r
        retained.append(dict(seed=seed,selected=best));write_json(stage/'retained.json',retained)
    checked=[]
    for item in sorted(retained,key=lambda r:r['selected']['key']):
        if time.perf_counter()>deadline-65:break
        if not item['selected']['passed']:continue
        s=item['selected'];r=run(s['spec']['bend_weights'],s['fixed_d'],f'seed{item["seed"]}_all34',all_contexts,deadline)
        checked.append(dict(seed=item['seed'],result=r));write_json(stage/'all34_checks.json',checked)
    eligible=[r for r in checked if r['result']['passed']]
    if eligible:
        chosen=min(eligible,key=lambda x:x['result']['key']);r=chosen['result']
        write_json(stage/'selected_policy.json',dict(model=selection['model'],model_sha256=selection['model_sha256'],
            source_spec=selection['source_spec'],bend_spec=r['spec'],fixed_d=r['fixed_d'],seed=chosen['seed'],
            goal_complete=False,full_lap_not_tested=True))
    write_json(stage/'receipt.json',dict(goal_complete=False,status='conditional_fit_finished',wall_s=time.perf_counter()-start,
        ppo_updates=0,retained=len(retained),all34_checked=len(checked),eligible=len(eligible)))
    print('BEND_FIT_FINISHED',args.arm,len(eligible),time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
